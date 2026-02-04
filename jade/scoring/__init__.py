"""
Scoring module for LLM-as-a-Judge evaluation.

Three-Dimensional Scoring Framework:
- D1: Reasoning Score (推理质量)
- D2: Evidence Score (事实准确性)
- D3: Credibility Score (来源可信度)
"""

from .score_generator import ScoreGenerator, ScoringResult
from .analysis_generator import AnalysisGenerator
from .statistics import EvalStatistics, calc_stats
from .conciseness import (
    ConcisenessScorer,
    ConcisenessResult,
    DensityMethod,
    calculate_knowledge_density,
)
from .source_quality import (
    SourceQualityScorer,
    SourceQualityResult,
    SourceEvaluation,
    SourceTier,
    evaluate_source_quality,
    get_source_tier,
    adjust_confidence_by_source_quality,
    generate_source_quality_report,
)

__all__ = [
    # Score Generator
    "ScoreGenerator",
    "ScoringResult",
    # Analysis Generator
    "AnalysisGenerator",
    # Statistics
    "EvalStatistics",
    "calc_stats",
    # Conciseness
    "ConcisenessScorer",
    "ConcisenessResult",
    "DensityMethod",
    "calculate_knowledge_density",
    # Source Quality
    "SourceQualityScorer",
    "SourceQualityResult",
    "SourceEvaluation",
    "SourceTier",
    "evaluate_source_quality",
    "get_source_tier",
    "adjust_confidence_by_source_quality",
    "generate_source_quality_report",
]
