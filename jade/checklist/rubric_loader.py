"""
Rubric Loader for Checklist Generation.

Extracts essential information from rubric YAML files:
- L1 deliverable check (what must exist)
- Expert hints (key checkpoints from L2/L3)
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class CompactRubric:
    """Minimal rubric for checklist generation."""
    
    # L1 core deliverable
    deliverable_name: str = ""
    deliverable_must_have: str = ""
    
    # Expert hints (combined from L1/L2/L3)
    hints: List[str] = None
    
    # Source labels (for source_skill field)
    l1_intent: str = ""
    l2_labels: List[str] = None
    l3_labels: List[str] = None
    
    def __post_init__(self):
        self.hints = self.hints or []
        self.l2_labels = self.l2_labels or []
        self.l3_labels = self.l3_labels or []
    
    def format_deliverable_check(self) -> str:
        """Format L1 deliverable for prompt injection."""
        if not self.deliverable_name:
            return "Check if the response provides what the user asked for."
        
        parts = [f"The response must provide: {self.deliverable_name}"]
        if self.deliverable_must_have:
            parts.append(f"Required: {self.deliverable_must_have}")
        return "\n".join(parts)
    
    def format_expert_hints(self) -> str:
        """Format expert hints for prompt injection."""
        if not self.hints:
            return "• Use query context to determine evaluation criteria."
        return "\n".join(f"• {hint}" for hint in self.hints)


class CompactRubricLoader:
    """
    Loads rubrics and extracts only essential information.
    
    Directory structure:
        rubric_dir/
        ├── L1_intent/
        │   └── product_discovery.yaml
        ├── L2_information_need/
        │   └── trending_analysis.yaml
        └── L3_constraints/
            └── certification_required.yaml
    """
    
    def __init__(self, rubric_dir: str = "rubrics/bizbench"):
        self.rubric_dir = Path(rubric_dir)
        self._cache: Dict[str, Dict] = {}
    
    def _load_yaml(self, path: Path) -> Optional[Dict]:
        """Load YAML file with caching."""
        key = str(path)
        if key in self._cache:
            return self._cache[key]
        
        if not path.exists():
            return None
        
        try:
            with path.open('r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                self._cache[key] = data
                return data
        except Exception:
            return None
    
    def load_compact(self, labels: Dict[str, Any]) -> CompactRubric:
        """
        Load and compose a compact rubric from labels.
        
        Args:
            labels: {
                "L1_primary_intent": "product_discovery",
                "L2_information_need": ["trending_analysis", "platform_data"],
                "L3_constraints": ["certification_required"]
            }
        
        Returns:
            CompactRubric with deliverable check and expert hints.
        """
        l1_intent = labels.get("L1_primary_intent", "")
        l2_needs = labels.get("L2_information_need", [])
        l3_constraints = labels.get("L3_constraints", [])
        
        # Normalize to lists
        if isinstance(l2_needs, str):
            l2_needs = [l2_needs] if l2_needs else []
        if isinstance(l3_constraints, str):
            l3_constraints = [l3_constraints] if l3_constraints else []
        
        # Initialize result
        result = CompactRubric(
            l1_intent=l1_intent,
            l2_labels=l2_needs,
            l3_labels=l3_constraints,
        )
        
        hints = []
        
        # Load L1 intent
        if l1_intent:
            l1_data = self._load_yaml(
                self.rubric_dir / "L1_intent" / f"{l1_intent}.yaml"
            )
            if l1_data:
                # Extract deliverable
                pd = l1_data.get("primary_deliverable", {})
                result.deliverable_name = pd.get("name", "")
                result.deliverable_must_have = pd.get("must_have", "")
                
                # Extract hints from L1
                l1_hints = l1_data.get("hints", [])
                if l1_hints:
                    hints.extend(l1_hints)
        
        # Load L2 information needs
        for need in l2_needs:
            l2_data = self._load_yaml(
                self.rubric_dir / "L2_information_need" / f"{need}.yaml"
            )
            if l2_data:
                l2_hints = l2_data.get("hints", [])
                if l2_hints:
                    hints.extend(l2_hints)
        
        # Load L3 constraints
        for constraint in l3_constraints:
            l3_data = self._load_yaml(
                self.rubric_dir / "L3_constraints" / f"{constraint}.yaml"
            )
            if l3_data:
                l3_hints = l3_data.get("hints", [])
                if l3_hints:
                    hints.extend(l3_hints)
        
        # Deduplicate and limit hints
        seen = set()
        unique_hints = []
        for h in hints:
            if h not in seen:
                seen.add(h)
                unique_hints.append(h)
        
        result.hints = unique_hints[:8]  # Max 8 hints
        
        return result


# =============================================================================
# Convenience Function
# =============================================================================

def load_rubric_compact(
    labels: Dict[str, Any],
    rubric_dir: str = "rubrics/bizbench",
) -> CompactRubric:
    """Load compact rubric from labels.
    
    Example:
        >>> labels = {"L1_primary_intent": "product_discovery", "L2_information_need": ["trending_analysis"]}
        >>> rubric = load_rubric_compact(labels)
        >>> print(rubric.format_deliverable_check())
        >>> print(rubric.format_expert_hints())
    """
    loader = CompactRubricLoader(rubric_dir)
    return loader.load_compact(labels)
