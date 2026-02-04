"""
LangGraph ReAct Agent for Claim Verification.

This module implements a ReAct (Reasoning + Acting) agent using LangGraph
for verifying claims through web search and content analysis.
"""
import re
import json
import time
import datetime

from tqdm import tqdm
from typing import List, Dict, Any, Optional, Literal, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END

from .state import AgentState, ChecklistItem, VerificationResult
from jade.tools import web_search, web_scraper, analyze_content
from jade.tools.web_scraper import set_scraper_llm, get_scraper_tool
from jade.tools.web_search import get_search_tool
from jade.tools.analyzer import set_analyzer_llm
from jade.llm import BaseLLMClient, create_llm_client
from jade.utils.logger import AgentLogger, create_logger


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are an expert fact-checker. Verify claims by gathering and analyzing evidence from web sources.

## Current Date: {current_date}

## Claim to Verify
{claim}

{source_info}

## Tools

1. **web_search** - Search the web
```
<tool_call>
{{"name": "web_search", "arguments": {{"queries": ["query1", "query2"]}}}}
</tool_call>
```

2. **web_scraper** - Visit URLs to extract information
```
<tool_call>
{{"name": "web_scraper", "arguments": {{"urls": ["url"], "goal": "what to find"}}}}
</tool_call>
```

3. **analyze_content** - Final analysis (use ONLY after gathering evidence)
```
<tool_call>
{{"name": "analyze_content", "arguments": {{"claim": "claim", "evidence": "all evidence", "source": "url"}}}}
</tool_call>
```

## Workflow
1. Search for relevant information
2. Visit URLs to extract details
3. Call analyze_content for final verdict

## Important
- Use exact `<tool_call>` format
- If source URL provided, visit it first
- Gather evidence before concluding

Begin by searching for information.
"""


class VerificationAgent:
    """LangGraph-based ReAct agent for claim verification."""
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        max_iterations: int = 15,
        verbose: bool = True,
        log_dir: str = "logs",
        enable_logging: bool = True,
        session_id: Optional[str] = None,
        concurrency: int = 1
    ):
        """
        Initialize the verification agent.
        
        Args:
            llm_client: LLM client for generating responses
            max_iterations: Maximum iterations per verification
            verbose: Enable console output
            log_dir: Directory for log files
            enable_logging: Enable file logging
            session_id: Optional session ID for consistent log/output naming
            concurrency: Number of parallel verification tasks (default: 1)
        """
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.session_id = session_id
        self.concurrency = concurrency
        
        # Initialize logger with session_id
        self.logger = create_logger(log_dir=log_dir, enabled=enable_logging, session_id=session_id)
        
        # Configure tools
        set_scraper_llm(llm_client)
        set_analyzer_llm(llm_client)
        
        self.tools = [web_search, web_scraper, analyze_content]
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent", self._should_continue,
            {"continue": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")
        return workflow.compile()
    
    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """Determine whether to continue or terminate."""
        # If already have analysis result, we're done
        if state.get("analysis_result"):
            self._log("🛑 Terminating: analysis complete", "analysis_complete")
            return "end"
        
        if state.get("should_terminate"):
            self._log("🛑 Terminating: completion flag set", "terminate_flag")
            return "end"
        
        if state.get("error_message"):
            self._log(f"🛑 Terminating: {state.get('error_message')}", "error")
            return "end"
        
        pending = state.get("pending_tool_calls", [])
        if pending:
            self._log(f"▶️ Continuing: {len(pending)} tool(s)", "continue")
            return "continue"
        
        # Check if we're at max iterations - if so, force analysis
        if state.get("iteration_count", 0) >= state.get("max_iterations", 15):
            self._log("⚠️ Max iterations - will force analysis", "max_iterations_force_analysis")
            return "continue"  # Continue to force analysis
        
        self._log("🛑 Terminating: no pending actions", "no_actions")
        return "end"
    
    def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        """Agent reasoning node."""
        iteration = state.get("iteration_count", 0) + 1
        
        self._log(f"\n{'─'*60}")
        self._log(f"🤖 Agent Iteration {iteration}")
        self._log(f"{'─'*60}")
        
        # Start iteration logging
        self.logger.start_iteration(iteration)
        
        # Check if we need to force analyze_content call
        max_iters = state.get("max_iterations", 15)
        if iteration > max_iters and not state.get("analysis_result"):
            self._log("⚠️ Max iterations exceeded - forcing analyze_content call")
            forced_call = self._create_forced_analysis_call(state)
            if forced_call:
                # Don't end iteration here - let tool_node end it after execution
                return {
                    "iteration_count": iteration,
                    "pending_tool_calls": [forced_call]
                }
            else:
                self.logger.end_iteration("max_iterations_exceeded")
                return {
                    "iteration_count": iteration,
                    "should_terminate": True,
                    "error_message": "Maximum iterations reached"
                }
        
        # Check if we already have an analysis result from tools
        analysis_result = state.get("analysis_result")
        if analysis_result:
            self._log("📊 Using analysis result from analyze_content tool")
            self.logger.end_iteration("using_analysis_result")
            return {
                "iteration_count": iteration,
                "current_result": analysis_result,
                "should_terminate": True,
                "pending_tool_calls": []
            }
        
        # Build context
        current_date = datetime.date.today().strftime("%Y-%m-%d")
        current_item = state.get("current_item")
        
        source_info = ""
        if current_item and current_item.source:
            source_info = f"📎 Source: {current_item.source}\n⚠️ Visit this URL first."
        
        system_msg = SYSTEM_PROMPT.format(
            current_date=current_date,
            claim=current_item.description if current_item else "No claim",
            source_info=source_info
        )
        
        messages = [{"role": "system", "content": system_msg}]
        
        for msg in state.get("messages", []):
            if isinstance(msg, dict):
                messages.append(msg)
            elif isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                messages.append({"role": "user", "content": f"Tool result: {msg.content}"})
        
        if len(messages) == 1:
            messages.append({
                "role": "user",
                "content": f"Verify: {current_item.description if current_item else 'No claim'}"
            })
        
        # Call LLM
        try:
            self._log("📡 Calling LLM...")
            response = self.llm_client.chat_completion(messages)
            
            self._log(f"📝 Response ({len(response)} chars):")
            self._log(f"   {response[:400]}..." if len(response) > 400 else f"   {response}")
            
            # Parse tool calls
            tool_calls = self._parse_tool_calls(response)
            
            # Log LLM call
            self.logger.log_llm_call(
                messages=messages,
                response=response,
                parsed_tool_calls=tool_calls
            )
            
            if tool_calls:
                self._log(f"🔧 Found {len(tool_calls)} tool call(s)")
                for tc in tool_calls:
                    self._log(f"   • {tc['name']}")
            
            # If no tool calls, try to parse verdict or force analysis
            if not tool_calls:
                # First try to parse a final verdict from response
                final_result = self._try_parse_final_verdict(response, state)
                if final_result:
                    self._log("📊 Parsed final verdict from response")
                    self.logger.end_iteration("final_verdict_parsed")
                    return {
                        "messages": [{"role": "assistant", "content": response}],
                        "iteration_count": iteration,
                        "current_result": final_result,
                        "should_terminate": True,
                        "pending_tool_calls": []
                    }
                
                # No verdict parsed - force analyze_content if we have evidence
                has_evidence = (
                    state.get("search_results") or 
                    state.get("scraped_content")
                )
                if has_evidence and not state.get("analysis_result"):
                    self._log("⚠️ No tool calls - forcing analyze_content")
                    forced_call = self._create_forced_analysis_call(state)
                    if forced_call:
                        # Don't end iteration here - let tool_node end it after execution
                        return {
                            "messages": [{"role": "assistant", "content": response}],
                            "iteration_count": iteration,
                            "pending_tool_calls": [forced_call]
                        }
                
                # No tool calls and no forced analysis - end iteration here
                self.logger.end_iteration("no_tools")
            
            # If we have tool_calls, don't end iteration here - let tool_node end it after execution
            # This ensures tool calls are logged before iteration ends
            
            return {
                "messages": [{"role": "assistant", "content": response}],
                "iteration_count": iteration,
                "pending_tool_calls": tool_calls
            }
            
        except Exception as e:
            self._log(f"❌ Error: {e}")
            self.logger.log_llm_call(messages=messages, response="", error=str(e))
            self.logger.end_iteration("error")
            return {
                "iteration_count": iteration,
                "should_terminate": True,
                "error_message": f"LLM error: {str(e)}"
            }
    
    def _tool_node(self, state: AgentState) -> Dict[str, Any]:
        """Tool execution node."""
        self._log(f"\n{'─'*60}")
        self._log("🔧 Executing Tools")
        self._log(f"{'─'*60}")
        
        tool_calls = state.get("pending_tool_calls", [])
        if not tool_calls:
            # No tools to execute - end iteration if it exists
            if self.logger.current_iteration:
                self.logger.end_iteration("no_tools")
            return {"messages": [], "pending_tool_calls": []}
        
        results = []
        search_results = list(state.get("search_results", []))
        scraped_content = list(state.get("scraped_content", []))
        analysis_result = None
        current_item = state.get("current_item")
        
        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("args", {})
            
            self._log(f"\n▶️ {name}")
            self._log(f"   Args: {json.dumps(args, ensure_ascii=False)[:150]}...")
            
            start_time = time.time()
            
            try:
                result = None
                raw_result = None
                
                for tool in self.tools:
                    if tool.name == name:
                        result = tool.invoke(args)
                        break
                
                if result is None:
                    result = f"Error: Unknown tool '{name}'"
                
                # Get raw result for web_search and web_scraper
                if name == "web_search":
                    try:
                        search_tool = get_search_tool()
                        queries = args.get("queries", [])
                        raw_result = search_tool.get_raw_results(queries)
                    except Exception as e:
                        self._log(f"   ⚠️ Failed to get raw search results: {e}")
                elif name == "web_scraper":
                    try:
                        scraper_tool = get_scraper_tool()
                        urls = args.get("urls", [])
                        raw_result = scraper_tool.get_raw_content(urls)
                    except Exception as e:
                        self._log(f"   ⚠️ Failed to get raw scraper content: {e}")
                
                result_str = str(result)
                duration_ms = int((time.time() - start_time) * 1000)
                
                self._log(f"   ✅ Result: {len(result_str)} chars ({duration_ms}ms)")
                
                # Log tool call with raw result
                self.logger.log_tool_call(
                    tool_name=name,
                    args=args,
                    result=result_str,
                    success=True,
                    duration_ms=duration_ms,
                    raw_result=raw_result
                )
                
                # Handle analyze_content result specially - extract the final verdict
                if name == "analyze_content":
                    analysis_result = self._parse_analysis_result(result_str, current_item, state)
                    if analysis_result:
                        self._log(f"   📊 Extracted verdict: {analysis_result.verification_status} ({analysis_result.confidence}%)")
                
                if name == "web_search":
                    search_results.append(result_str)
                elif name == "web_scraper":
                    scraped_content.append(result_str)
                
                results.append({
                    "role": "user",
                    "content": f"<tool_response>\n📦 Tool: {name}\n\n{result_str}\n</tool_response>\n\nAnalyze the result. If enough evidence, use analyze_content for final verdict."
                })
                
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                self._log(f"   ❌ Error: {e}")
                
                self.logger.log_tool_call(
                    tool_name=name,
                    args=args,
                    result="",
                    success=False,
                    error=str(e),
                    duration_ms=duration_ms
                )
                
                results.append({
                    "role": "user",
                    "content": f"<tool_response>\n📦 Tool: {name}\n❌ Error: {e}\n</tool_response>"
                })
        
        return_state = {
            "messages": results,
            "search_results": search_results,
            "scraped_content": scraped_content,
            "pending_tool_calls": []
        }
        
        # If we got an analysis result, add it to state
        if analysis_result:
            return_state["analysis_result"] = analysis_result
        
        # End iteration after all tools are executed and logged
        # This ensures tool calls are logged before iteration ends
        self.logger.end_iteration("continue")
        
        return return_state
    
    def _parse_analysis_result(
        self, 
        result_str: str, 
        item: Optional[ChecklistItem],
        state: AgentState
    ) -> Optional[VerificationResult]:
        """
        Parse the result from analyze_content tool into a VerificationResult.
        
        Format: 
        {
            "conclusion": "yes/no",
            "confidence": 0-100,
            "reason": {"summary": "...", "supporting": [...], "contradicting": [...]},
            "reference_urls": {"supporting": [...], "contradicting": [...]}
        }
        """
        from .state import ReasonDetail, ReferenceUrls
        
        try:
            data = json.loads(result_str)
            
            if "conclusion" not in data:
                return None
            
            # Parse conclusion
            conclusion_val = str(data.get("conclusion", "")).lower().strip()
            conclusion = "yes" if conclusion_val in ["yes", "true", "verified"] else "no"
            
            # Parse confidence (0-100)
            confidence_raw = data.get("confidence", 50)
            if isinstance(confidence_raw, (int, float)):
                if confidence_raw <= 1:
                    confidence = int(confidence_raw * 100)
                else:
                    confidence = int(confidence_raw)
            else:
                confidence = 50
            confidence = max(0, min(100, confidence))
            
            # Parse reason (handle both string and dict formats)
            reason_data = data.get("reason", {})
            if isinstance(reason_data, str):
                reason = ReasonDetail(
                    summary=reason_data,
                    supporting=[],
                    contradicting=[]
                )
            elif isinstance(reason_data, dict):
                reason = ReasonDetail(
                    summary=reason_data.get("summary", ""),
                    supporting=reason_data.get("supporting", []),
                    contradicting=reason_data.get("contradicting", [])
                )
            else:
                reason = ReasonDetail(summary="Analysis completed")
            
            # Parse reference_urls (handle both list and dict formats)
            refs_data = data.get("reference_urls", {})
            if isinstance(refs_data, list):
                reference_urls = ReferenceUrls(supporting=refs_data, contradicting=[])
            elif isinstance(refs_data, dict):
                reference_urls = ReferenceUrls(
                    supporting=refs_data.get("supporting", []),
                    contradicting=refs_data.get("contradicting", [])
                )
            else:
                reference_urls = ReferenceUrls()
            
            # Add source URL if provided
            if item and item.source:
                if item.source not in reference_urls.supporting:
                    reference_urls.supporting.insert(0, item.source)
            
            # Determine legacy verification status
            if conclusion == "yes":
                status = "VERIFIED" if confidence >= 70 else "PARTIALLY_VERIFIED"
            else:
                if confidence >= 70:
                    status = "REFUTED"
                elif confidence >= 40:
                    status = "PARTIALLY_VERIFIED"
                else:
                    status = "INSUFFICIENT_EVIDENCE"
            
            return VerificationResult(
                item_id=item.id if item else 0,
                description=item.description if item else "",
                conclusion=conclusion,
                confidence=confidence,
                reason=reason,
                reference_urls=reference_urls,
                verification_status=status,
                source_url=item.source if item else None
            )
            
        except json.JSONDecodeError:
            # Try to extract JSON from the string
            try:
                extracted = self._extract_json(result_str)
                if extracted:
                    return self._parse_analysis_result(extracted, item, state)
            except:
                pass
        except Exception as e:
            self._log(f"   ⚠️ Failed to parse analysis result: {e}")
        
        return None
    
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Extract tool calls from response with robust parsing."""
        tool_calls = []
        
        # Normalize the response - handle incomplete closing tags
        # e.g., </tool_call without > or </tool_call >
        normalized = response
        normalized = re.sub(r'</tool_call\s*>?', '</tool_call>', normalized)
        
        if "<tool_call>" not in normalized:
            return tool_calls
        
        try:
            parts = normalized.split("<tool_call>")
            for i, part in enumerate(parts[1:], 1):
                # Find the end - either </tool_call> or end of string
                if "</tool_call>" in part:
                    raw = part.split("</tool_call>")[0].strip()
                else:
                    # No closing tag - try to extract JSON anyway
                    raw = part.strip()
                
                # Clean up markdown code blocks
                for prefix in ["```json", "```"]:
                    if raw.startswith(prefix):
                        raw = raw[len(prefix):]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()
                
                # Skip empty content
                if not raw:
                    continue
                
                # Try to parse JSON
                try:
                    data = json.loads(raw)
                    tool_calls.append({
                        "id": f"call_{i}",
                        "name": data.get("name", ""),
                        "args": data.get("arguments", {})
                    })
                except json.JSONDecodeError:
                    # Try to extract JSON with brace matching
                    extracted = self._extract_json(raw)
                    if extracted:
                        try:
                            data = json.loads(extracted)
                            tool_calls.append({
                                "id": f"call_{i}",
                                "name": data.get("name", ""),
                                "args": data.get("arguments", {})
                            })
                            self._log("   ✅ Fixed JSON parsing")
                        except:
                            self._log(f"   ⚠️ Could not parse tool call JSON")
        except Exception as e:
            self._log(f"⚠️ Parse error: {e}")
        
        return tool_calls
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract valid JSON by matching braces."""
        start = text.find('{')
        if start == -1:
            return None
        
        depth = 0
        in_string = False
        escape = False
        
        for i, c in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            if c == '\\' and in_string:
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        return None
    
    def _create_forced_analysis_call(self, state: AgentState) -> Optional[Dict[str, Any]]:
        """
        Create a forced analyze_content tool call using gathered evidence.
        Called when max iterations reached without analysis.
        """
        current_item = state.get("current_item")
        if not current_item:
            return None
        
        # Gather all evidence from search results and scraped content
        evidence_parts = []
        
        search_results = state.get("search_results", [])
        for i, sr in enumerate(search_results[:3], 1):
            # Truncate each search result
            truncated = sr[:3000] if len(sr) > 3000 else sr
            evidence_parts.append(f"[Search Result {i}]\n{truncated}")
        
        scraped_content = state.get("scraped_content", [])
        for i, sc in enumerate(scraped_content[:3], 1):
            truncated = sc[:3000] if len(sc) > 3000 else sc
            evidence_parts.append(f"[Scraped Content {i}]\n{truncated}")
        
        if not evidence_parts:
            evidence_parts.append("No evidence was gathered during the verification process.")
        
        evidence = "\n\n---\n\n".join(evidence_parts)
        
        self._log(f"📊 Creating forced analysis with {len(evidence_parts)} evidence items")
        
        return {
            "id": "forced_analysis",
            "name": "analyze_content",
            "args": {
                "claim": current_item.description,
                "evidence": evidence,
                "source": current_item.source or ""
            }
        }
    
    def _try_parse_final_verdict(self, response: str, state: AgentState) -> Optional[VerificationResult]:
        """
        Try to parse a final verdict from LLM response.
        """
        item = state.get("current_item")
        response_lower = response.lower()
        
        # Method 1: Try JSON parsing
        if "verification_status" in response_lower:
            try:
                extracted = self._extract_json(response)
                if extracted:
                    data = json.loads(extracted)
                    if "verification_status" in data:
                        return self._build_result_from_json(data, item, state)
            except:
                pass
        
        # Method 2: Parse text verdict format
        verdict_patterns = [
            (r'verdict[:\s]+(\w+)', 1),
            (r'verification[:\s]+(\w+)', 1),
            (r'status[:\s]+(\w+)', 1),
        ]
        
        for pattern, group in verdict_patterns:
            match = re.search(pattern, response_lower)
            if match:
                status_word = match.group(group).upper()
                status_map = {
                    "VERIFIED": "VERIFIED",
                    "TRUE": "VERIFIED",
                    "ACCURATE": "VERIFIED",
                    "CONFIRMED": "VERIFIED",
                    "REFUTED": "REFUTED",
                    "FALSE": "REFUTED",
                    "INACCURATE": "REFUTED",
                    "PARTIAL": "PARTIALLY_VERIFIED",
                    "PARTIALLY": "PARTIALLY_VERIFIED",
                    "INCONCLUSIVE": "INSUFFICIENT_EVIDENCE",
                    "INSUFFICIENT": "INSUFFICIENT_EVIDENCE",
                }
                
                if status_word in status_map:
                    evidence = self._extract_evidence_from_text(response)
                    sources = self._extract_urls(response)
                    
                    # Add item source if available
                    if item and item.source and item.source not in sources:
                        sources.insert(0, item.source)
                    
                    return VerificationResult(
                        item_id=item.id if item else 0,
                        description=item.description if item else "",
                        verification_status=status_map[status_word],
                        confidence=0.75 if status_word in ["VERIFIED", "REFUTED", "TRUE", "FALSE"] else 0.5,
                        conclusion=self._extract_conclusion_from_text(response, status_map[status_word]),
                        supporting_evidence=evidence.get("supporting", []),
                        contradicting_evidence=evidence.get("contradicting", []),
                        reasoning=response[:3000],
                        recommendations=[],
                        sources_consulted=sources,
                        source_url=item.source if item else None
                    )
        
        return None
    
    def _build_result_from_json(self, data: Dict, item: ChecklistItem, state: AgentState) -> VerificationResult:
        """Build VerificationResult from parsed JSON data."""
        status = data.get("verification_status") or "INSUFFICIENT_EVIDENCE"
        confidence = float(data.get("confidence") or 0.5)
        
        sources = self._extract_urls(str(data))
        if item and item.source and item.source not in sources:
            sources.insert(0, item.source)
        
        conclusion = data.get("conclusion") or self._generate_conclusion(status, confidence)
        
        return VerificationResult(
            item_id=item.id if item else 0,
            description=item.description if item else "",
            verification_status=status,
            confidence=confidence,
            conclusion=conclusion,
            supporting_evidence=data.get("supporting_evidence") or [],
            contradicting_evidence=data.get("contradicting_evidence") or [],
            reasoning=data.get("reasoning") or "",
            recommendations=data.get("recommendations") or [],
            sources_consulted=sources,
            source_url=item.source if item else None
        )
    
    def _extract_evidence_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extract evidence points from text response."""
        evidence = {"supporting": [], "contradicting": []}
        
        lines = text.split('\n')
        current_type = "supporting"
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_lower = line.lower()
            if 'supporting' in line_lower or 'evidence for' in line_lower or 'why:' in line_lower:
                current_type = 'supporting'
                continue
            elif 'contradicting' in line_lower or 'evidence against' in line_lower:
                current_type = 'contradicting'
                continue
            
            if line.startswith(('-', '•', '*')) or re.match(r'^\d+\.', line):
                item = re.sub(r'^[-•*\d.]+\s*', '', line).strip()
                if item and len(item) > 15:
                    evidence[current_type].append(item[:500])
        
        return evidence
    
    def _extract_conclusion_from_text(self, text: str, status: str) -> str:
        """Extract or generate conclusion from text."""
        # Look for explicit conclusion patterns
        patterns = [
            r'conclusion[:\s]+([^.]+\.)',
            r'summary[:\s]+([^.]+\.)',
            r'therefore[,\s]+([^.]+\.)',
            r'the claim (?:is|appears to be) ([^.]+\.)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                conclusion = match.group(1).strip()
                if len(conclusion) > 20:
                    return conclusion.capitalize()
        
        return self._generate_conclusion(status, 0.7)
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        urls = re.findall(r'https?://[^\s\)\]\"\'>]+', text)
        # Clean up URLs
        cleaned = []
        for url in urls:
            url = url.rstrip('.,;:')
            if url not in cleaned:
                cleaned.append(url)
        return cleaned[:10]
    
    def _generate_conclusion(self, status: str, confidence: float) -> str:
        """Generate conclusion based on status."""
        conclusions = {
            "VERIFIED": f"The claim has been VERIFIED as accurate based on the evidence (confidence: {confidence}%).",
            "REFUTED": f"The claim has been REFUTED - evidence contradicts it (confidence: {confidence}%).",
            "PARTIALLY_VERIFIED": f"The claim is PARTIALLY VERIFIED - some aspects are accurate (confidence: {confidence}%).",
            "INSUFFICIENT_EVIDENCE": "INCONCLUSIVE - insufficient evidence to verify or refute the claim.",
        }
        return conclusions.get(status, "Unable to determine claim accuracy.")
    
    def _log(self, message: str, log_type: str = "info"):
        """Print log message if verbose."""
        if self.verbose:
            print(message)
    
    def verify_item(self, item: ChecklistItem) -> VerificationResult:
        """Verify a single checklist item."""
        self._log(f"\n{'═'*70}")
        self._log(f"📋 VERIFYING ITEM #{item.id}")
        self._log(f"{'═'*70}")
        self._log(f"Claim: {item.description[:100]}...")
        if item.source:
            self._log(f"Source: {item.source}")
        
        # Start item logging
        self.logger.start_item(item.id, item.description, item.source)
        
        initial_state: AgentState = {
            "current_item": item,
            "messages": [],
            "search_results": [],
            "scraped_content": [],
            "current_result": None,
            "all_results": [],
            "iteration_count": 0,
            "max_iterations": self.max_iterations,
            "should_terminate": False,
            "error_message": None,
            "pending_tool_calls": [],
            "analysis_result": None  # New field to store analyze_content result
        }
        
        try:
            final_state = self.graph.invoke(initial_state)
            
            # Priority order for getting result:
            # 1. Direct analysis_result from analyze_content tool
            # 2. current_result parsed from LLM response
            # 3. Fallback result
            
            result = (
                final_state.get("analysis_result") or 
                final_state.get("current_result")
            )
            
            if not result:
                from .state import ReasonDetail, ReferenceUrls
                
                # Build fallback result with collected evidence info
                evidence_count = len(final_state.get("search_results", [])) + len(final_state.get("scraped_content", []))
                
                # Collect any sources we found
                sources = []
                for sr in final_state.get("search_results", []):
                    urls = re.findall(r'https?://[^\s\)\]]+', sr)
                    sources.extend(urls[:3])
                if item.source:
                    sources.insert(0, item.source)
                sources = list(dict.fromkeys(sources))[:10]
                
                summary = final_state.get("error_message") or f"Verification incomplete. Gathered {evidence_count} evidence items but could not reach a conclusion."
                
                result = VerificationResult(
                    item_id=item.id,
                    description=item.description,
                    conclusion="no",
                    confidence=0,
                    reason=ReasonDetail(summary=summary),
                    reference_urls=ReferenceUrls(supporting=sources),
                    verification_status="INSUFFICIENT_EVIDENCE",
                    source_url=item.source
                )
            
            # End item logging
            self.logger.end_item({
                "conclusion": result.conclusion,
                "confidence": result.confidence,
                "reason": result.reason.summary if hasattr(result.reason, 'summary') else str(result.reason)
            })
            
            self._log(result.to_report())
            return result
            
        except Exception as e:
            from .state import ReasonDetail, ReferenceUrls
            
            self._log(f"\n❌ Verification failed: {e}")
            
            result = VerificationResult(
                item_id=item.id,
                description=item.description,
                conclusion="no",
                confidence=0,
                reason=ReasonDetail(summary=f"Verification failed: {str(e)}"),
                reference_urls=ReferenceUrls(supporting=[item.source] if item.source else []),
                verification_status="ERROR",
                source_url=item.source
            )
            
            self.logger.end_item({"conclusion": "no", "error": str(e)})
            return result
    
    def verify_checklist(self, checklist: List[ChecklistItem]) -> List[VerificationResult]:
        """Verify all items in a checklist (supports parallel execution)."""
        
        self._log(f"\n{'=' * 70}")
        self._log(f"  STARTING VERIFICATION")
        self._log(f"  {len(checklist)} items to verify (concurrency: {self.concurrency})")
        self._log(f"{'=' * 70}")
        
        indexed_items = list(enumerate(checklist))
        results: List[Tuple[int, VerificationResult]] = []
        
        if self.concurrency > 1 and len(checklist) > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                futures = {executor.submit(self.verify_item, item): idx for idx, item in indexed_items}
                with tqdm(total=len(checklist), desc="Verifying items", unit="item") as pbar:
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            result = future.result()
                            results.append((idx, result))
                        except Exception as e:
                            self._log(f"❌ Item {idx} failed: {e}")
                            # Create error result
                            from .state import ReasonDetail, ReferenceUrls
                            item = checklist[idx]
                            results.append((idx, VerificationResult(
                                item_id=item.id,
                                description=item.description,
                                conclusion="no",
                                confidence=0,
                                reason=ReasonDetail(summary=f"Verification failed: {e}"),
                                reference_urls=ReferenceUrls(),
                                verification_status="ERROR",
                                source_url=item.source
                            )))
                        finally:
                            pbar.update(1)
        else:
            # Sequential execution
            with tqdm(total=len(checklist), desc="Verifying items", unit="item") as pbar:
                for idx, item in indexed_items:
                    self._log(f"\n[{idx+1}/{len(checklist)}] Processing...")
                    result = self.verify_item(item)
                    results.append((idx, result))
                    pbar.update(1)
        
        # Sort by original index and extract results
        results.sort(key=lambda x: x[0])
        final_results = [r for _, r in results]
        
        # Finalize logging
        self.logger.finalize()
        self._log(f"\n📁 Log saved to: {self.logger.get_log_path()}")
        
        return final_results


def create_agent(
    client_type: str = "custom",
    model_name: Optional[str] = None,
    max_iterations: int = 15,
    verbose: bool = True,
    log_dir: str = "logs",
    enable_logging: bool = True,
    concurrency: int = 1,
    **kwargs
) -> VerificationAgent:
    """
    Create a configured verification agent.
    """
    client = create_llm_client(
        client_type=client_type,
        model_name=model_name,
        **kwargs
    )
    
    return VerificationAgent(
        client, max_iterations, verbose, log_dir, enable_logging, concurrency=concurrency
    )
