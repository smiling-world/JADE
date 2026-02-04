"""
Checklist Generation Prompts.

Two prompt variants:
1. WITH_SKILL: Uses expert rubric hints for domain-specific evaluation
2. NO_SKILL: Generic evaluation without domain knowledge (for ablation)

Design principles:
- Flat structure: All prompts in one file
- Atomic criteria: Each criterion asks ONE thing
- Expert hints: Inject domain knowledge concisely
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any


# =============================================================================
# Prompt Templates
# =============================================================================

QUERY_CHECKLIST_WITH_SKILL = """
# TASK
Generate a checklist to evaluate if an AI response adequately answers the user query.
Each criterion must be an atomic Yes/No question (ask ONE thing only).

# QUERY
{query}

# CORE DELIVERABLE (L1 Gate)
{deliverable_check}

# EXPERT CHECKPOINTS
{expert_hints}

# RULES
1. **L1 Gate First**: item_id=0 must check if core deliverable exists
   - Product/supplier queries → check for links (URLs, ASINs)
   - Data queries → check for specific data/numbers
   - Analysis queries → check for conclusions with reasoning

2. **Atomic Questions**: Each criterion = ONE check
   - ❌ BAD: "Does it provide links AND analyze trends?"
   - ✅ GOOD: "Does it provide product links?"
   - ✅ GOOD: "Does it analyze trends?"

3. **Quantity**: Scale with query complexity
   - Simple query (few requirements): 4-6 items
   - Complex query (many L2/L3 checkpoints): 8-15 items
   - Cover each expert checkpoint; Cover the user's requirements; Skip redundant or trivial checks

4. **Critical Flaw**: Only for ACTIVE violations (recommending wrong things)
   - ✅ "Does it recommend items outside the specified scope?"
   - ❌ "Does it fail to provide X?" (covered by positive check)

5. **Independent Items** (depends_on: null): Always include these at the end
   - **Graceful Degradation**: If core request cannot be fully met, does the response acknowledge limitations and provide alternatives?
   - **Risk Awareness**: For recommendation/decision queries, does the response mention potential risks or uncertainties?

# OUTPUT FORMAT
Each item has: item_id, tier, depends_on, category, description, weight, source_skill

Weights: 15 (L1 core), 10 (L2/L3), 5 (general), -15 (critical flaw)

```json
[
  {{"item_id": 0, "tier": "L1", "depends_on": null, "category": "Core Deliverable",
    "description": "Does the response provide [SPECIFIC DELIVERABLE]?", "weight": 15, "source_skill": "L1"}},
  {{"item_id": 1, "tier": "L2", "depends_on": 0, "category": "[Checkpoint Name]",
    "description": "[Atomic question based on expert hint]?", "weight": 10, "source_skill": "L2"}},
  {{"item_id": 2, "tier": "general", "depends_on": 0, "category": "Analysis Depth",
    "description": "[Check if abstract requirements are analyzed in depth]", "weight": 5, "source_skill": "general"}},
  {{"item_id": N-1, "tier": "independent", "depends_on": null, "category": "Graceful Degradation",
    "description": "If core deliverable is not fully met, does the response acknowledge limitations and provide alternatives or partial solutions?", "weight": 5, "source_skill": "independent"}},
  {{"item_id": N, "tier": "independent", "depends_on": null, "category": "Risk Awareness",
    "description": "For recommendations or decisions, does the response mention potential risks, uncertainties, or verification steps?", "weight": 5, "source_skill": "independent"}}
]
```
"""

QUERY_CHECKLIST_NO_SKILL = """
# TASK
Generate a checklist to evaluate if an AI response adequately answers the user query.
Each criterion must be a Yes/No question.

# QUERY
{query}

# RULES
1. Generate 5-15 criteria to check if the response answers the query
2. Each criterion should be a simple Yes/No question
3. Consider basic dimensions:
   - Relevance: Does it address what was asked?
   - Helpfulness: Does it provide useful information?
   - Completeness: Does it cover the main points?
   - Clarity: Is it easy to understand?

# OUTPUT FORMAT
```json
[
  {{"item_id": 0, "type": "criterion", "category": "General",
    "description": "Does the response [check something]?", "weight": 5, "source_skill": "general"}}
]
```
"""

REPORT_CHECKLIST = """
# TASK
Generate a checklist to verify factual claims and reasoning quality in the AI response.

# QUERY
{query}

# RESPONSE TO EVALUATE
{report_content}

# ITEM TYPES

1. **Evidence** (type: "evidence"): Verifiable facts
   - Factual claims (entity existence, specs, certifications)
   - Quantitative claims (numbers, dates, prices)
   - Source validity (URLs cited in response)

2. **Reasoning** (type: "reasoning"): Judgment quality
   - Is the conclusion supported by stated evidence?
   - Are key assumptions stated?
   - Is the reasoning logically valid?

# RULES
1. Each description must be SELF-CONTAINED (understandable without the response)
   - ❌ BAD: "Verify the Enaiter example's price claim"
   - ✅ GOOD: "Verify that Enaiter silicone products are priced at $7.50-$9.80/piece"

2. Focus on HIGH-IMPACT claims that affect user decisions

3. Quantity: 4-10 items based on response complexity

# OUTPUT FORMAT
```json
[
  {{"item_id": 0, "type": "evidence", "category": "Factual Claim",
    "description": "Verify that [SPECIFIC CLAIM with full context].", "weight": 5}},
  {{"item_id": 1, "type": "evidence", "category": "Source Validity",
    "source": "https://example.com/page",
    "description": "Verify this URL is accessible and shows [expected content].", "weight": 5}},
  {{"item_id": 2, "type": "reasoning", "category": "Evidence Support",
    "description": "Is the claim that [X is better than Y] supported by the comparison data?", "weight": 10,
    "depends_on": [0]}}
]
```
"""


# =============================================================================
# Configuration
# =============================================================================

class PromptVariant(Enum):
    """Experiment variants."""
    BASELINE = "baseline"       # No skill, no report checklist
    WITH_SKILL = "with_skill"   # With skill, no report checklist
    WITH_REPORT = "with_report" # No skill, with report checklist
    FULL = "full"               # With skill + report checklist


@dataclass
class PromptConfig:
    """Simple configuration."""
    use_skill: bool = True
    use_report_specific: bool = True
    
    @classmethod
    def from_variant(cls, variant: PromptVariant) -> "PromptConfig":
        mapping = {
            PromptVariant.BASELINE:    (False, False),
            PromptVariant.WITH_SKILL:  (True, False),
            PromptVariant.WITH_REPORT: (False, True),
            PromptVariant.FULL:        (True, True),
        }
        use_skill, use_report = mapping[variant]
        return cls(use_skill=use_skill, use_report_specific=use_report)


# =============================================================================
# Prompt Builder (Flat, No Dependencies)
# =============================================================================

class PromptBuilder:
    """Simple prompt builder - all logic in one place."""
    
    def __init__(self, config: Optional[PromptConfig] = None):
        self.config = config or PromptConfig()
    
    def get_query_prompt(self) -> str:
        """Get query checklist prompt template."""
        if self.config.use_skill:
            return QUERY_CHECKLIST_WITH_SKILL
        return QUERY_CHECKLIST_NO_SKILL
    
    def get_report_prompt(self) -> str:
        """Get report checklist prompt template."""
        if not self.config.use_report_specific:
            return ""
        return REPORT_CHECKLIST
    
    def format_query_prompt(
        self,
        query: str,
        deliverable_check: str = "",
        expert_hints: str = "",
    ) -> str:
        """Format query prompt with actual values."""
        template = self.get_query_prompt()
        
        if self.config.use_skill:
            return template.format(
                query=query,
                deliverable_check=deliverable_check or "Check if the core request is fulfilled.",
                expert_hints=expert_hints or "• Use query context to determine key checkpoints.",
            )
        return template.format(query=query)
    
    def format_report_prompt(self, query: str, report_content: str) -> str:
        """Format report prompt with actual values."""
        template = self.get_report_prompt()
        if not template:
            return ""
        return template.format(query=query, report_content=report_content)


# =============================================================================
# Convenience Functions
# =============================================================================

def get_prompts(
    variant: Optional[PromptVariant] = None,
    use_skill: Optional[bool] = None,
    use_report_specific: Optional[bool] = None,
) -> Dict[str, Any]:
    """Get prompt templates.
    
    Args:
        variant: Predefined variant (takes precedence)
        use_skill: Enable skill-based prompts
        use_report_specific: Enable report checklist
    
    Returns:
        Dictionary with query_checklist, report_checklist, and config.
    """
    if variant is not None:
        config = PromptConfig.from_variant(variant)
    else:
        config = PromptConfig(
            use_skill=use_skill if use_skill is not None else True,
            use_report_specific=use_report_specific if use_report_specific is not None else True,
        )
    
    builder = PromptBuilder(config)
    
    return {
        "query_checklist": builder.get_query_prompt(),
        "report_checklist": builder.get_report_prompt(),
        "config": {
            "use_skill": config.use_skill,
            "use_report_specific": config.use_report_specific,
        },
    }


def create_builder(variant: PromptVariant = PromptVariant.FULL) -> PromptBuilder:
    """Create a prompt builder."""
    return PromptBuilder(PromptConfig.from_variant(variant))
