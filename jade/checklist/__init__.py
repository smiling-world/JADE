"""
Checklist module for verification checklist generation.

Features:
- Compact prompts for atomic checklist criteria
- Expert rubric hints injection for domain knowledge
- Support for ablation experiments (with/without skill)

Quick Start:
    from jade.checklist import ChecklistGenerator, create_generator
    
    generator = create_generator(
        client_type="openai",
        model_name="gpt-4",
        use_skill=True,
    )
    checklist = generator.generate_query_checklist(
        query="Find trending products on Amazon",
        labels={"L1_primary_intent": "product_discovery"}
    )

Ablation Experiments:
    from jade.checklist import PromptVariant, create_generator
    
    # Variants: BASELINE, WITH_SKILL, WITH_REPORT, FULL
    generator = create_generator(variant=PromptVariant.BASELINE)
"""

from .prompts import (
    PromptBuilder,
    PromptConfig,
    PromptVariant,
    get_prompts,
    create_builder,
    QUERY_CHECKLIST_WITH_SKILL,
    QUERY_CHECKLIST_NO_SKILL,
    REPORT_CHECKLIST,
)

from .rubric_loader import (
    CompactRubric,
    CompactRubricLoader,
    load_rubric_compact,
)

from .generator import (
    ChecklistGenerator,
    GenerationResult,
    create_generator,
)

from .multilabel_loader import (
    extract_labels_from_item,
    infer_labels_from_query,
)

__all__ = [
    # Prompts
    "PromptBuilder",
    "PromptConfig",
    "PromptVariant",
    "get_prompts",
    "create_builder",
    "QUERY_CHECKLIST_WITH_SKILL",
    "QUERY_CHECKLIST_NO_SKILL",
    "REPORT_CHECKLIST",
    
    # Rubric Loader
    "CompactRubric",
    "CompactRubricLoader",
    "load_rubric_compact",
    
    # Generator
    "ChecklistGenerator",
    "GenerationResult",
    "create_generator",
    
    # Label Utils
    "extract_labels_from_item",
    "infer_labels_from_query",
]
