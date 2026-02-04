"""
GenAI-based Verification Agent.

This module implements a verification agent using Google GenAI's built-in
GoogleSearch and UrlContext tools for web search and scraping capabilities.

Key features:
- Leverages GenAI's native GoogleSearch tool for real-time web search
- Uses UrlContext tool for direct URL content extraction
- Simple, clean implementation without complex ReAct loops
- Compatible with the existing ChecklistItem and VerificationResult structures

Example:
    >>> from jade.verification_agent import create_genai_agent, ChecklistItem
    >>> agent = create_genai_agent()
    >>> item = ChecklistItem(id=1, description="Verify this claim...")
    >>> result = agent.verify_item(item)
"""

import json
import time
import datetime
import threading
from typing import List, Dict, Any, Optional, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, UrlContext, GoogleSearch

from .state import ChecklistItem, VerificationResult, ReasonDetail, ReferenceUrls
from jade.utils.logger import create_logger


# =============================================================================
# System Prompts
# =============================================================================

VERIFICATION_PROMPT = """You are an expert fact-checker. Your task is to verify the following claim using web search and URL context tools.

## Current Date: {current_date}

## Claim to Verify
{claim}

{source_info}

## Instructions
1. If a source URL is provided, analyze it first using URL context
2. Use web search to find additional evidence if needed
3. Analyze all gathered evidence carefully
4. Provide your final verdict in the EXACT JSON format below

## IMPORTANT: Handling Equivalent Terms and Partial Matches
When verifying claims, recognize that many terms have **equivalent expressions**. Do NOT require the report/listing to repeat every alias verbatim if the meaning is clearly the same.

1. **Parenthetical equivalence** like "18/8 (304)" or "X (Y equivalent)" means these are **equivalent terms**. Finding EITHER term satisfies the claim.
   - Example: "18/8 stainless steel" = "304 stainless steel" = "SUS304" (same material, different naming conventions)
   - If claim says "18/8 (304) stainless steel" and evidence shows only "18/8 stainless steel", this is still a MATCH.

2. **Slash notations** like "USB-C/Type-C" are interchangeable terms.

3. **Common industry equivalents** you should recognize:
   - 18/8 stainless steel = 304 stainless steel = SUS304
   - Wi‑Fi 6 = 802.11ax
   - USB 3.0 = USB 3.1 Gen 1 = USB 3.2 Gen 1
   - 4K = 2160p = UHD

4. **Focus on semantic meaning**: if the evidence confirms the essential claim using an equivalent term, conclude "yes".

## Response Format (MUST be valid JSON)
```json
{{
    "conclusion": "yes" or "no",
    "confidence": 0-100,
    "reason": {{
        "summary": "Brief summary of your findings",
        "supporting": ["Evidence point 1", "Evidence point 2"],
        "contradicting": ["Contradicting evidence if any"]
    }},
    "reference_urls": {{
        "supporting": ["url1", "url2"],
        "contradicting": ["url3"]
    }}
}}
```

## Rules
- "yes" means the claim is accurate/verified
- "no" means the claim is inaccurate, outdated, or cannot be verified
- Confidence should reflect how certain you are (0-100)
- Include ALL relevant URLs you found in reference_urls
- Be thorough but concise in your reasoning
"""


class GenAIVerificationAgent:
    """
    Verification agent powered by Google GenAI's built-in tools.
    
    Uses GoogleSearch and UrlContext for evidence gathering,
    providing a simpler alternative to the ReAct-based agent.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: str = "v1",
        headers: Optional[Dict[str, str]] = None,
        verbose: bool = True,
        log_dir: str = "logs",
        enable_logging: bool = True,
        session_id: Optional[str] = None,
        concurrency: int = 1,
        use_search: bool = True,
        use_url_context: bool = True,
        temperature: float = 0.3,
        max_retries: int = 3,
    ):
        """
        Initialize the GenAI verification agent.
        
        Args:
            model_name: GenAI model name (default: gemini-2.5-flash)
            api_key: API key for GenAI
            base_url: Base URL for GenAI API
            api_version: API version
            headers: Custom headers
            verbose: Enable console output
            log_dir: Directory for log files
            enable_logging: Enable file logging
            session_id: Optional session ID for consistent naming
            concurrency: Number of parallel verification tasks
            use_search: Enable GoogleSearch tool
            use_url_context: Enable UrlContext tool
            temperature: Sampling temperature
            max_retries: Maximum retry attempts
        """
        import os
        
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GENAI_API_KEY")
        self.base_url = base_url or os.environ.get("GENAI_BASE_URL")
        self.api_version = api_version
        self.headers = headers or {}
        self.verbose = verbose
        self.concurrency = concurrency
        self.use_search = use_search
        self.use_url_context = use_url_context
        self.temperature = temperature
        self.max_retries = max_retries
        
        # Initialize logger
        self.logger = create_logger(
            log_dir=log_dir, 
            enabled=enable_logging, 
            session_id=session_id
        )
        
        # Thread-local storage for GenAI clients (each thread gets its own client)
        # This is necessary because google-genai's httpx client is not thread-safe
        self._thread_local = threading.local()
    
    @property
    def client(self) -> genai.Client:
        """
        Get or create GenAI client for current thread.
        
        Each thread gets its own client instance to avoid 
        'client has been closed' errors in concurrent execution.
        """
        if not hasattr(self._thread_local, 'client') or self._thread_local.client is None:
            http_options = types.HttpOptions(
                api_version=self.api_version,
                base_url=self.base_url,
                headers=self.headers,
            ) if self.base_url else None
            
            self._thread_local.client = genai.Client(
                api_key=self.api_key,
                http_options=http_options,
            )
        return self._thread_local.client
    
    def _build_tools(self, item: ChecklistItem) -> List[Tool]:
        """Build tool list based on configuration and item."""
        tools = []
        
        # Add URL context tool if enabled and source URL exists
        if self.use_url_context:
            tools.append(Tool(url_context=UrlContext))
        
        # Add search tool if enabled
        if self.use_search:
            tools.append(Tool(google_search=GoogleSearch()))
        
        return tools
    
    def _build_prompt(self, item: ChecklistItem) -> str:
        """Build verification prompt for the item."""
        current_date = datetime.date.today().strftime("%Y-%m-%d")
        
        source_info = ""
        if item.source:
            source_info = f"## Source URL to verify\n{item.source}\n\n⚠️ IMPORTANT: Analyze this URL first using URL context."
        
        return VERIFICATION_PROMPT.format(
            current_date=current_date,
            claim=item.description,
            source_info=source_info,
        )
    
    def _parse_response(
        self, 
        response_text: str, 
        item: ChecklistItem
    ) -> VerificationResult:
        """Parse GenAI response into VerificationResult."""
        try:
            # Try to extract JSON from response
            json_str = self._extract_json(response_text)
            if json_str:
                data = json.loads(json_str)
                return self._build_result_from_json(data, item)
        except (json.JSONDecodeError, Exception) as e:
            self._log(f"⚠️ Failed to parse JSON response: {e}")
        
        # Fallback: try to parse from text
        return self._parse_text_response(response_text, item)
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON object from text."""
        # Try to find JSON in code blocks
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # Try to find standalone JSON
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
    
    def _build_result_from_json(
        self, 
        data: Dict[str, Any], 
        item: ChecklistItem
    ) -> VerificationResult:
        """Build VerificationResult from parsed JSON."""
        # Parse conclusion
        conclusion_val = str(data.get("conclusion", "no")).lower().strip()
        conclusion: Literal["yes", "no"] = "yes" if conclusion_val in ["yes", "true", "verified"] else "no"
        
        # Parse confidence
        confidence_raw = data.get("confidence", 50)
        if isinstance(confidence_raw, (int, float)):
            confidence = int(confidence_raw) if confidence_raw > 1 else int(confidence_raw * 100)
        else:
            confidence = 50
        confidence = max(0, min(100, confidence))
        
        # Parse reason
        reason_data = data.get("reason", {})
        if isinstance(reason_data, str):
            reason = ReasonDetail(summary=reason_data)
        elif isinstance(reason_data, dict):
            reason = ReasonDetail(
                summary=reason_data.get("summary", ""),
                supporting=reason_data.get("supporting", []),
                contradicting=reason_data.get("contradicting", []),
            )
        else:
            reason = ReasonDetail(summary="Analysis completed")
        
        # Parse reference URLs
        refs_data = data.get("reference_urls", {})
        if isinstance(refs_data, list):
            reference_urls = ReferenceUrls(supporting=refs_data)
        elif isinstance(refs_data, dict):
            reference_urls = ReferenceUrls(
                supporting=refs_data.get("supporting", []),
                contradicting=refs_data.get("contradicting", []),
            )
        else:
            reference_urls = ReferenceUrls()
        
        # Add source URL if not already present
        if item.source and item.source not in reference_urls.supporting:
            reference_urls.supporting.insert(0, item.source)
        
        # Determine verification status
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
            item_id=item.id,
            description=item.description,
            conclusion=conclusion,
            confidence=confidence,
            reason=reason,
            reference_urls=reference_urls,
            verification_status=status,
            source_url=item.source,
        )
    
    def _parse_text_response(
        self, 
        text: str, 
        item: ChecklistItem
    ) -> VerificationResult:
        """Parse verification result from plain text response."""
        text_lower = text.lower()
        
        # Determine conclusion from keywords
        verified_keywords = ["verified", "accurate", "confirmed", "true", "correct"]
        refuted_keywords = ["refuted", "inaccurate", "false", "incorrect", "not verified"]
        
        conclusion: Literal["yes", "no"] = "no"
        confidence = 50
        
        for kw in verified_keywords:
            if kw in text_lower:
                conclusion = "yes"
                confidence = 70
                break
        
        for kw in refuted_keywords:
            if kw in text_lower:
                conclusion = "no"
                confidence = 70
                break
        
        # Extract URLs
        import re
        urls = re.findall(r'https?://[^\s\)\]\"\'>]+', text)
        urls = list(dict.fromkeys(urls))[:10]  # Deduplicate and limit
        
        # Add source URL
        if item.source and item.source not in urls:
            urls.insert(0, item.source)
        
        status = "VERIFIED" if conclusion == "yes" else "INSUFFICIENT_EVIDENCE"
        
        return VerificationResult(
            item_id=item.id,
            description=item.description,
            conclusion=conclusion,
            confidence=confidence,
            reason=ReasonDetail(summary=text[:500]),
            reference_urls=ReferenceUrls(supporting=urls),
            verification_status=status,
            source_url=item.source,
        )
    
    def _log(self, message: str):
        """Print log message if verbose."""
        if self.verbose:
            print(message)
    
    def verify_item(self, item: ChecklistItem) -> VerificationResult:
        """
        Verify a single checklist item.
        
        Args:
            item: ChecklistItem to verify
            
        Returns:
            VerificationResult with conclusion and evidence
        """
        self._log(f"\n{'═'*70}")
        self._log(f"📋 VERIFYING ITEM #{item.id}")
        self._log(f"{'═'*70}")
        self._log(f"Claim: {item.description[:100]}...")
        if item.source:
            self._log(f"Source: {item.source}")
        
        # Start item logging
        self.logger.start_item(item.id, item.description, item.source)
        self.logger.start_iteration(1)
        
        start_time = time.time()
        
        try:
            # Build tools and prompt
            tools = self._build_tools(item)
            prompt = self._build_prompt(item)
            
            # Build config
            config = GenerateContentConfig(
                tools=tools,
                temperature=self.temperature,
                response_modalities=["TEXT"],
            )
            
            # Call GenAI with retry (includes response parsing)
            response_text = ""
            response = None
            last_error = None
            
            for attempt in range(self.max_retries):
                try:
                    self._log(f"📡 Calling GenAI (attempt {attempt + 1}/{self.max_retries})...")
                    
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=config,
                    )
                    
                    # Extract response text (inside retry loop to handle parse errors)
                    response_text = ""
                    if hasattr(response, 'text') and response.text:
                        response_text = response.text
                    elif hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content') and candidate.content:
                            parts = candidate.content.parts
                            if parts:  # Check if parts is not None
                                for part in parts:
                                    if hasattr(part, 'text') and part.text:
                                        response_text += part.text
                    
                    # Validate we got actual content
                    if not response_text.strip():
                        raise ValueError("Empty response received from GenAI")
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    last_error = e
                    self._log(f"⚠️ Attempt {attempt + 1} failed: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
            
            if not response_text.strip():
                raise Exception(f"All {self.max_retries} retries failed: {last_error}")
            
            duration_ms = int((time.time() - start_time) * 1000)
            self._log(f"✅ Response received ({duration_ms}ms, {len(response_text)} chars)")
            
            # Log the call
            self.logger.log_llm_call(
                messages=[{"role": "user", "content": prompt}],
                response=response_text,
            )
            
            # Extract grounding metadata if available
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                    gm = candidate.grounding_metadata
                    if hasattr(gm, 'web_search_queries'):
                        self._log(f"🔍 Search queries: {gm.web_search_queries}")
            
            # Parse response
            result = self._parse_response(response_text, item)
            
            # Log completion
            self.logger.end_iteration("complete")
            self.logger.end_item({
                "conclusion": result.conclusion,
                "confidence": result.confidence,
                "reason": result.reason.summary,
            })
            
            self._log(result.to_report())
            return result
            
        except Exception as e:
            self._log(f"❌ Verification failed: {e}")
            
            self.logger.end_iteration("error")
            self.logger.end_item({"conclusion": "no", "error": str(e)})
            
            return VerificationResult(
                item_id=item.id,
                description=item.description,
                conclusion="no",
                confidence=0,
                reason=ReasonDetail(summary=f"Verification failed: {str(e)}"),
                reference_urls=ReferenceUrls(
                    supporting=[item.source] if item.source else []
                ),
                verification_status="ERROR",
                source_url=item.source,
            )
    
    def verify_checklist(
        self, 
        checklist: List[ChecklistItem]
    ) -> List[VerificationResult]:
        """
        Verify all items in a checklist.
        
        Supports parallel execution based on concurrency setting.
        
        Args:
            checklist: List of ChecklistItem to verify
            
        Returns:
            List of VerificationResult
        """
        self._log(f"\n{'=' * 70}")
        self._log(f"  STARTING GENAI VERIFICATION")
        self._log(f"  {len(checklist)} items to verify (concurrency: {self.concurrency})")
        self._log(f"  Tools: search={self.use_search}, url_context={self.use_url_context}")
        self._log(f"{'=' * 70}")
        
        results: List[tuple] = []
        
        if self.concurrency > 1 and len(checklist) > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                futures = {
                    executor.submit(self.verify_item, item): idx 
                    for idx, item in enumerate(checklist)
                }
                with tqdm(total=len(checklist), desc="Verifying", unit="item") as pbar:
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            result = future.result()
                            results.append((idx, result))
                        except Exception as e:
                            self._log(f"❌ Item {idx} failed: {e}")
                            item = checklist[idx]
                            results.append((idx, VerificationResult(
                                item_id=item.id,
                                description=item.description,
                                conclusion="no",
                                confidence=0,
                                reason=ReasonDetail(summary=f"Verification failed: {e}"),
                                reference_urls=ReferenceUrls(),
                                verification_status="ERROR",
                                source_url=item.source,
                            )))
                        finally:
                            pbar.update(1)
        else:
            # Sequential execution
            with tqdm(total=len(checklist), desc="Verifying", unit="item") as pbar:
                for idx, item in enumerate(checklist):
                    result = self.verify_item(item)
                    results.append((idx, result))
                    pbar.update(1)
        
        # Sort by original index
        results.sort(key=lambda x: x[0])
        final_results = [r for _, r in results]
        
        # Finalize logging
        self.logger.finalize()
        self._log(f"\n📁 Log saved to: {self.logger.get_log_path()}")
        
        return final_results


def create_genai_agent(
    model_name: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_version: str = "v1",
    headers: Optional[Dict[str, str]] = None,
    verbose: bool = True,
    log_dir: str = "logs",
    enable_logging: bool = True,
    concurrency: int = 1,
    use_search: bool = True,
    use_url_context: bool = True,
    **kwargs
) -> GenAIVerificationAgent:
    """
    Factory function to create a GenAI verification agent.
    
    Args:
        model_name: GenAI model name
        api_key: API key for GenAI
        base_url: Base URL for GenAI API
        api_version: API version
        headers: Custom headers
        verbose: Enable console output
        log_dir: Directory for log files
        enable_logging: Enable file logging
        concurrency: Number of parallel verification tasks
        use_search: Enable GoogleSearch tool
        use_url_context: Enable UrlContext tool
        **kwargs: Additional configuration
        
    Returns:
        Configured GenAIVerificationAgent instance
    """
    return GenAIVerificationAgent(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        api_version=api_version,
        headers=headers,
        verbose=verbose,
        log_dir=log_dir,
        enable_logging=enable_logging,
        concurrency=concurrency,
        use_search=use_search,
        use_url_context=use_url_context,
        **kwargs
    )
