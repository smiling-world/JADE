"""
Content Analysis Tool for Claim Verification.

This module provides an LLM-powered tool for analyzing gathered evidence
and making verification determinations with structured reasoning.
"""

import json
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class AnalyzerTool:
    """
    Tool for analyzing evidence and verifying claims.
    
    Uses an LLM to evaluate gathered evidence against a claim and
    produce a structured verification result with reasoning.
    """
    
    ANALYSIS_PROMPT = """You are an expert fact-checker. Based on the evidence provided, determine whether the claim is true or false.

## Claim to Verify
{claim}

## Source Information
{source}

## Gathered Evidence
{evidence}

## Analysis Instructions

Carefully evaluate the evidence, separating supporting and contradicting points.

### IMPORTANT: Handling Equivalent Terms and Partial Matches

When verifying claims, recognize that many terms have **equivalent expressions**:

1. **Parenthetical notations** like "18/8 (304)" or "Type A (Class 1)" mean these are **equivalent terms**. Finding EITHER term satisfies the claim.
   - Example: "18/8 stainless steel" = "304 stainless steel" (same material, different naming conventions)
   - If claim says "18/8 (304) stainless steel" and evidence shows "18/8 stainless steel", this is a MATCH.

2. **Slash notations** like "USB-C/Type-C" mean these are interchangeable terms.

3. **Common industry equivalents** you should recognize:
   - 18/8 stainless steel = 304 stainless steel = SUS304
   - 18/10 stainless steel = 316 stainless steel
   - Wi-Fi 6 = 802.11ax
   - USB 3.0 = USB 3.1 Gen 1 = USB 3.2 Gen 1
   - 4K = 2160p = UHD

4. **Focus on semantic meaning**: If the evidence confirms the essential claim using an equivalent term, conclude "yes".

## Required Output Format

Provide your analysis as a JSON object with this EXACT structure:

{{
    "conclusion": "yes" or "no",
    "confidence": 0-100,
    "reason": {{
        "summary": "One sentence overall conclusion",
        "supporting": ["Evidence point 1 that supports the claim", "Evidence point 2..."],
        "contradicting": ["Evidence point 1 that contradicts the claim", "Evidence point 2..."]
    }},
    "reference_urls": {{
        "supporting": ["url1 that supports", "url2..."],
        "contradicting": ["url1 that contradicts", "url2..."]
    }}
}}

## Guidelines

- **conclusion**: "yes" if evidence supports the claim (including via equivalent terms), "no" if evidence contradicts or is insufficient
- **confidence**: 0-100 (80+ strong, 50-80 partial, below 50 weak)
- **reason.summary**: Brief overall conclusion
- **reason.supporting**: List specific evidence points that support the claim
- **reason.contradicting**: List specific evidence points that contradict the claim
- **reference_urls.supporting**: URLs of sources that support
- **reference_urls.contradicting**: URLs of sources that contradict

Provide ONLY the JSON object, no additional text.
"""
    
    def __init__(self, llm_client=None):
        """
        Initialize the analyzer tool.
        
        Args:
            llm_client: LLM client for generating analysis
        """
        self.llm_client = llm_client
    
    def set_llm_client(self, llm_client):
        """Set the LLM client for analysis."""
        self.llm_client = llm_client
    
    def analyze(self, claim: str, evidence: str, source: Optional[str] = None) -> str:
        """
        Analyze evidence to verify a claim.
        
        Args:
            claim: The claim to verify
            evidence: Gathered evidence from searches and scraping
            source: Optional source URL or reference
            
        Returns:
            JSON string with verification result
        """
        if not self.llm_client:
            return json.dumps({
                "conclusion": "no",
                "confidence": 0,
                "reason": {
                    "summary": "Analysis unavailable - LLM client not configured",
                    "supporting": [],
                    "contradicting": []
                },
                "reference_urls": {
                    "supporting": [],
                    "contradicting": []
                }
            }, indent=2, ensure_ascii=False)
        
        prompt = self.ANALYSIS_PROMPT.format(
            claim=claim,
            source=source or "No specific source provided",
            evidence=evidence[:50000] if len(evidence) > 50000 else evidence
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.llm_client.chat_completion(messages)
            
            # Clean response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # Validate JSON
            try:
                result = json.loads(response)
                
                # Normalize conclusion
                if "conclusion" not in result or result["conclusion"] is None:
                    result["conclusion"] = "no"
                else:
                    conc = str(result["conclusion"]).lower().strip()
                    result["conclusion"] = "yes" if conc in ["yes", "true", "verified"] else "no"
                
                # Normalize confidence
                if "confidence" not in result or result["confidence"] is None:
                    result["confidence"] = 50
                else:
                    conf = result["confidence"]
                    if isinstance(conf, float) and conf <= 1:
                        conf = int(conf * 100)
                    result["confidence"] = max(0, min(100, int(conf)))
                
                # Normalize reason (handle both old string and new dict format)
                reason = result.get("reason", {})
                if isinstance(reason, str):
                    reason = {
                        "summary": reason,
                        "supporting": [],
                        "contradicting": []
                    }
                elif not isinstance(reason, dict):
                    reason = {"summary": "Analysis completed", "supporting": [], "contradicting": []}
                else:
                    reason = {
                        "summary": reason.get("summary", ""),
                        "supporting": reason.get("supporting", []),
                        "contradicting": reason.get("contradicting", [])
                    }
                
                # Normalize reference_urls (handle both old list and new dict format)
                refs = result.get("reference_urls", {})
                if isinstance(refs, list):
                    refs = {"supporting": refs, "contradicting": []}
                elif not isinstance(refs, dict):
                    refs = {"supporting": [], "contradicting": []}
                else:
                    refs = {
                        "supporting": refs.get("supporting", []),
                        "contradicting": refs.get("contradicting", [])
                    }
                
                clean_result = {
                    "conclusion": result["conclusion"],
                    "confidence": result["confidence"],
                    "reason": reason,
                    "reference_urls": refs
                }
                
                return json.dumps(clean_result, indent=2, ensure_ascii=False)
                
            except json.JSONDecodeError:
                left = response.find('{')
                right = response.rfind('}')
                if left != -1 and right != -1 and left < right:
                    try:
                        result = json.loads(response[left:right+1])
                        return json.dumps(result, indent=2, ensure_ascii=False)
                    except:
                        pass
                
                return json.dumps({
                    "conclusion": "no",
                    "confidence": 30,
                    "reason": {
                        "summary": f"Output format invalid: {response[:300]}",
                        "supporting": [],
                        "contradicting": []
                    },
                    "reference_urls": {"supporting": [], "contradicting": []}
                }, indent=2, ensure_ascii=False)
                
        except Exception as e:
            return json.dumps({
                "conclusion": "no",
                "confidence": 0,
                "reason": {
                    "summary": f"Analysis failed: {str(e)}",
                    "supporting": [],
                    "contradicting": []
                },
                "reference_urls": {"supporting": [], "contradicting": []}
            }, indent=2, ensure_ascii=False)


# Global instance
_analyzer_tool = None


def get_analyzer_tool() -> AnalyzerTool:
    """Get or create the global analyzer tool instance."""
    global _analyzer_tool
    if _analyzer_tool is None:
        _analyzer_tool = AnalyzerTool()
    return _analyzer_tool


def set_analyzer_llm(llm_client):
    """Set the LLM client for the global analyzer tool."""
    get_analyzer_tool().set_llm_client(llm_client)


class AnalyzerInput(BaseModel):
    """Input schema for the analyzer tool."""
    claim: str = Field(description="The claim or statement to verify")
    evidence: str = Field(description="All gathered evidence from searches and page visits")
    source: Optional[str] = Field(default=None, description="Optional source URL for the claim")


@tool("analyze_content", args_schema=AnalyzerInput)
def analyze_content(claim: str, evidence: str, source: Optional[str] = None) -> str:
    """
    Analyze gathered evidence to verify a claim and provide a final verdict.
    
    Use this tool AFTER gathering sufficient evidence from web_search and web_scraper.
    It will analyze all evidence and return a structured verification result.
    
    Args:
        claim: The claim to verify
        evidence: All evidence gathered from searches and page visits
        source: Optional source URL for the claim
        
    Returns:
        JSON with verification_status, confidence, reasoning, and evidence lists
    """
    return get_analyzer_tool().analyze(claim, evidence, source)
