"""
Agent state definitions for LangGraph-based verification agent.

This module defines the state structures used throughout the agent's
execution lifecycle, including checklist items, verification results,
and the main agent state.
"""

from typing import List, Dict, Any, Optional, Literal, Annotated, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class ChecklistItem(BaseModel):
    """
    Represents a single item in the verification checklist.
    """
    id: int
    description: str
    source: Optional[str] = None


class ReasonDetail(BaseModel):
    """Structured reason with supporting and contradicting evidence."""
    summary: str = ""
    supporting: List[str] = Field(default_factory=list)
    contradicting: List[str] = Field(default_factory=list)


class ReferenceUrls(BaseModel):
    """Structured URLs with supporting and contradicting sources."""
    supporting: List[str] = Field(default_factory=list)
    contradicting: List[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    """
    Contains the complete result of verifying a checklist item.
    
    Output format:
    {
        "conclusion": "yes/no",
        "confidence": 0-100,
        "reason": {
            "summary": "...",
            "supporting": ["..."],
            "contradicting": ["..."]
        },
        "reference_urls": {
            "supporting": ["..."],
            "contradicting": ["..."]
        }
    }
    """
    item_id: int
    description: str
    
    # Core fields
    conclusion: Literal["yes", "no"] = "no"
    confidence: int = Field(ge=0, le=100, default=0)
    reason: ReasonDetail = Field(default_factory=ReasonDetail)
    reference_urls: ReferenceUrls = Field(default_factory=ReferenceUrls)
    
    # Legacy fields for backwards compatibility
    verification_status: Literal[
        "VERIFIED", "REFUTED", "PARTIALLY_VERIFIED", 
        "INSUFFICIENT_EVIDENCE", "ERROR"
    ] = "INSUFFICIENT_EVIDENCE"
    source_url: Optional[str] = None
    
    def get_conclusion_emoji(self) -> str:
        """Get emoji for conclusion."""
        return "✅" if self.conclusion == "yes" else "❌"
    
    def to_json(self) -> Dict[str, Any]:
        """Convert to the simple JSON format."""
        return {
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "reason": {
                "summary": self.reason.summary,
                "supporting": self.reason.supporting,
                "contradicting": self.reason.contradicting
            },
            "reference_urls": {
                "supporting": self.reference_urls.supporting,
                "contradicting": self.reference_urls.contradicting
            }
        }
    
    def to_report(self) -> str:
        """Generate a formatted report string."""
        lines = []
        
        # Header - simple yes/no
        emoji = self.get_conclusion_emoji()
        conclusion_text = "YES - Claim Verified" if self.conclusion == "yes" else "NO - Claim Not Verified"
        
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"  {emoji} {conclusion_text}")
        lines.append("=" * 70)
        
        # Confidence bar
        conf_bar_len = self.confidence // 5
        conf_bar = "▓" * conf_bar_len + "░" * (20 - conf_bar_len)
        lines.append(f"  Confidence: {self.confidence}%  [{conf_bar}]")
        
        # Claim
        lines.append("")
        lines.append(f"  📋 Claim (ID: {self.item_id})")
        desc = self.description
        while desc:
            chunk = desc[:65]
            desc = desc[65:]
            lines.append(f"     {chunk}")
        
        # Summary
        if self.reason.summary:
            lines.append("")
            lines.append("  💡 Summary")
            summary = self.reason.summary
            while summary:
                chunk = summary[:65]
                summary = summary[65:]
                lines.append(f"     {chunk}")
        
        # Supporting Evidence
        if self.reason.supporting:
            lines.append("")
            lines.append(f"  ✅ Supporting Evidence ({len(self.reason.supporting)})")
            for i, ev in enumerate(self.reason.supporting[:5], 1):
                ev_text = ev[:60] + "..." if len(ev) > 60 else ev
                lines.append(f"     {i}. {ev_text}")
        
        # Contradicting Evidence
        if self.reason.contradicting:
            lines.append("")
            lines.append(f"  ❌ Contradicting Evidence ({len(self.reason.contradicting)})")
            for i, ev in enumerate(self.reason.contradicting[:5], 1):
                ev_text = ev[:60] + "..." if len(ev) > 60 else ev
                lines.append(f"     {i}. {ev_text}")
        
        # Supporting URLs
        if self.reference_urls.supporting:
            lines.append("")
            lines.append(f"  🔗 Supporting Sources ({len(self.reference_urls.supporting)})")
            for url in self.reference_urls.supporting[:3]:
                url_text = url[:63] if len(url) <= 63 else url[:60] + "..."
                lines.append(f"     • {url_text}")
        
        # Contradicting URLs
        if self.reference_urls.contradicting:
            lines.append("")
            lines.append(f"  🔗 Contradicting Sources ({len(self.reference_urls.contradicting)})")
            for url in self.reference_urls.contradicting[:3]:
                url_text = url[:63] if len(url) <= 63 else url[:60] + "..."
                lines.append(f"     • {url_text}")
        
        lines.append("")
        lines.append("=" * 70)
        
        return '\n'.join(lines)


class AgentState(TypedDict, total=False):
    """
    State maintained throughout the agent's execution.
    """
    current_item: Optional[ChecklistItem]
    messages: Annotated[list, add_messages]
    search_results: List[str]
    scraped_content: List[str]
    current_result: Optional[VerificationResult]
    all_results: List[VerificationResult]
    iteration_count: int
    max_iterations: int
    should_terminate: bool
    error_message: Optional[str]
    pending_tool_calls: List[Dict[str, Any]]
    analysis_result: Optional[VerificationResult]
