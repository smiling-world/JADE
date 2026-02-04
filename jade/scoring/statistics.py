"""
Statistics calculation for evaluation results.

This module extracts all statistics calculation logic from the pipelines
into a clean, reusable class.

Supports:
- Three-dimensional scoring (reasoning, evidence, credibility)
- Tier breakdown (L1/L2/L3/general) for dependency-aware scoring
- Skill composition statistics
"""

from typing import Dict, List, Any


def calc_stats(scores: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of scores."""
    if not scores:
        return {}
    return {
        "mean": sum(scores) / len(scores),
        "min": min(scores),
        "max": max(scores),
        "median": sorted(scores)[len(scores) // 2]
    }


class EvalStatistics:
    """
    Centralized statistics calculator for evaluation results.
    
    Eliminates duplicate code between EvalPipeline and GenAIEvalPipeline.
    """
    
    def __init__(self, all_scores: List[Dict[str, Any]], config: Dict[str, Any] = None):
        """
        Initialize with evaluation scores.
        
        Args:
            all_scores: List of item score dictionaries
            config: Optional config dict with keys like 'score_fusion_mode', 'conciseness_method'
        """
        self.all_scores = all_scores
        self.config = config or {}
    
    def calculate(self) -> Dict[str, Any]:
        """Calculate all statistics and return as a dictionary."""
        if not self.all_scores:
            return {}
        
        return {
            "total_items": len(self.all_scores),
            "score_fusion_mode": self.config.get("score_fusion_mode", "weighted"),
            **self._core_scores(),
            **self._dimension_stats(),
            **self._track_stats(),
            **self._tier_stats(),
            **self._dependency_stats(),
            **self._skill_stats(),
            **self._distribution(),
            **self._conciseness_stats(),
            **self._credibility_stats(),
        }
    
    def _core_scores(self) -> Dict[str, Any]:
        """Calculate core three-dimensional scores."""
        final = [s.get("final_score", 0.0) or 0.0 for s in self.all_scores]
        reasoning = [s.get("reasoning_score", 0.0) for s in self.all_scores]
        evidence = [s.get("evidence_score", 0.0) for s in self.all_scores]
        credibility = [s.get("credibility_score", 0.0) for s in self.all_scores]
        
        fusion_mode = self.config.get("score_fusion_mode", "weighted")
        
        return {
            "reasoning_score": calc_stats(reasoning),
            "evidence_score": calc_stats(evidence),
            "credibility_score": calc_stats(credibility),
            "final_score": calc_stats(final) if fusion_mode != "independent" else None,
        }
    
    def _dimension_stats(self) -> Dict[str, Any]:
        """Aggregate scores by dimension."""
        dims = {}
        for s in self.all_scores:
            for dim, data in s.get("dimension_scores", {}).items():
                if dim not in dims:
                    dims[dim] = {"scores": [], "reasoning": [], "evidence": [], "count": 0}
                dims[dim]["scores"].append(data.get("score_rate", 0.0))
                dims[dim]["reasoning"].append(data.get("reasoning_score_rate", 0.0))
                dims[dim]["evidence"].append(data.get("evidence_score_rate", 0.0))
                dims[dim]["count"] += 1
        
        result = {}
        for dim, data in dims.items():
            n = len(data["scores"])
            result[dim] = {
                "total_items": data["count"],
                "score_rate_avg": sum(data["scores"]) / n if n else 0,
                "reasoning_rate_avg": sum(data["reasoning"]) / n if n else 0,
                "evidence_rate_avg": sum(data["evidence"]) / n if n else 0,
            }
        
        return {"dimension_statistics": result}
    
    def _track_stats(self) -> Dict[str, Any]:
        """Aggregate scores by track/intent."""
        buckets: Dict[str, List] = {}
        for s in self.all_scores:
            track = s.get("track", "unlabeled")
            buckets.setdefault(track, []).append(s)
        
        result = {}
        for track, items in buckets.items():
            result[track] = {
                "total_items": len(items),
                "final_score": calc_stats([x.get("final_score", 0) or 0 for x in items]),
                "reasoning_score": calc_stats([x.get("reasoning_score", 0) for x in items]),
                "evidence_score": calc_stats([x.get("evidence_score", 0) for x in items]),
                "credibility_score": calc_stats([x.get("credibility_score", 0) for x in items]),
            }
        
        return {"track_statistics": result}
    
    def _tier_stats(self) -> Dict[str, Any]:
        """
        Aggregate scores by tier (L1/L2/L3/general).
        
        This provides insight into how different tiers of checklist items
        contribute to the overall score.
        """
        tier_agg = {
            "L1": {"scores": [], "counts": {"total": 0, "evaluated": 0, "blocked": 0}},
            "L2": {"scores": [], "counts": {"total": 0, "evaluated": 0, "blocked": 0}},
            "L3": {"scores": [], "counts": {"total": 0, "evaluated": 0, "blocked": 0}},
            "general": {"scores": [], "counts": {"total": 0, "evaluated": 0, "blocked": 0}},
        }
        
        for s in self.all_scores:
            tier_breakdown = s.get("tier_breakdown", {})
            for tier, data in tier_breakdown.items():
                if tier in tier_agg and isinstance(data, dict):
                    avg_score = data.get("average_score", 0.0)
                    if avg_score > 0:
                        tier_agg[tier]["scores"].append(avg_score)
                    tier_agg[tier]["counts"]["total"] += data.get("count", 0)
                    tier_agg[tier]["counts"]["evaluated"] += data.get("evaluated", 0)
                    tier_agg[tier]["counts"]["blocked"] += data.get("blocked", 0)
        
        result = {}
        for tier, data in tier_agg.items():
            if data["scores"] or data["counts"]["total"] > 0:
                result[tier] = {
                    "average_score": calc_stats(data["scores"]) if data["scores"] else {},
                    "item_counts": data["counts"],
                    "evaluation_rate": (
                        round(data["counts"]["evaluated"] / data["counts"]["total"], 4)
                        if data["counts"]["total"] > 0 else 0.0
                    ),
                }
        
        return {"tier_statistics": result} if result else {}
    
    def _dependency_stats(self) -> Dict[str, Any]:
        """
        Aggregate dependency-related statistics.
        
        This tracks how often L1 Primary Deliverable passes/fails,
        and how many items are blocked due to dependency failures.
        """
        l1_passed_count = 0
        l1_failed_count = 0
        total_blocked = 0
        total_evaluated = 0
        total_items = 0
        
        for s in self.all_scores:
            dep_stats = s.get("dependency_stats", {})
            if dep_stats:
                if dep_stats.get("l1_passed", True):
                    l1_passed_count += 1
                else:
                    l1_failed_count += 1
                total_blocked += dep_stats.get("blocked", 0)
                total_evaluated += dep_stats.get("evaluated", 0)
                total_items += dep_stats.get("total_items", 0)
        
        total_queries = l1_passed_count + l1_failed_count
        if total_queries == 0:
            return {}
        
        return {
            "dependency_statistics": {
                "total_queries": total_queries,
                "l1_pass_count": l1_passed_count,
                "l1_fail_count": l1_failed_count,
                "l1_pass_rate": round(l1_passed_count / total_queries, 4),
                "total_checklist_items": total_items,
                "total_evaluated": total_evaluated,
                "total_blocked": total_blocked,
                "blocked_rate": round(total_blocked / total_items, 4) if total_items > 0 else 0.0,
            }
        }
    
    def _skill_stats(self) -> Dict[str, Any]:
        """
        Aggregate scores by source_skill for detailed skill-level analysis.
        
        This provides insight into which L1/L2/L3 skills contribute most
        to score variations.
        """
        skill_agg: Dict[str, List[float]] = {}
        
        for s in self.all_scores:
            for detail in s.get("reasoning_details", []):
                source_skill = detail.get("original_item", {}).get("source_skill", "unknown")
                if source_skill == "unknown":
                    source_skill = detail.get("source_skill", "unknown")
                
                # Only count evaluated items (not blocked)
                if detail.get("dependency_status", "EVALUATED") == "EVALUATED":
                    if detail.get("is_applicable", True):
                        score = detail.get("score", 0.0)
                        skill_agg.setdefault(source_skill, []).append(score)
        
        if not skill_agg:
            return {}
        
        result = {}
        for skill, scores in skill_agg.items():
            result[skill] = {
                "count": len(scores),
                "average": round(sum(scores) / len(scores), 4) if scores else 0.0,
                "min": min(scores) if scores else 0.0,
                "max": max(scores) if scores else 0.0,
            }
        
        # Sort by skill type (L1, L2, L3, general)
        sorted_result = {}
        for prefix in ["L1:", "L2:", "L3:", "general"]:
            for skill, data in sorted(result.items()):
                if skill.startswith(prefix) or (prefix == "general" and skill == "general"):
                    sorted_result[skill] = data
        
        # Add any remaining skills
        for skill, data in result.items():
            if skill not in sorted_result:
                sorted_result[skill] = data
        
        return {"skill_statistics": sorted_result}
    
    def _distribution(self) -> Dict[str, Any]:
        """Calculate score distribution."""
        fusion_mode = self.config.get("score_fusion_mode", "weighted")
        if fusion_mode == "independent":
            return {"score_distribution": None}
        
        finals = [s.get("final_score", 0) or 0 for s in self.all_scores]
        return {
            "score_distribution": {
                "excellent": sum(1 for x in finals if x >= 0.9),
                "good": sum(1 for x in finals if 0.7 <= x < 0.9),
                "fair": sum(1 for x in finals if 0.5 <= x < 0.7),
                "poor": sum(1 for x in finals if x < 0.5),
            }
        }
    
    def _conciseness_stats(self) -> Dict[str, Any]:
        """Calculate conciseness (knowledge density) statistics."""
        density = []
        tokens = []
        for s in self.all_scores:
            c = s.get("conciseness", {})
            if c:
                density.append(c.get("density_score", 0.0))
                tokens.append(c.get("token_count", 0))
        
        if not density:
            return {}
        
        return {
            "conciseness": {
                "method": self.config.get("conciseness_method", "log"),
                "alpha": self.config.get("conciseness_alpha"),
                "density_score": calc_stats(density),
                "token_count": calc_stats(tokens),
            }
        }
    
    def _credibility_stats(self) -> Dict[str, Any]:
        """Calculate source credibility statistics."""
        summaries = []
        for s in self.all_scores:
            sc = s.get("source_credibility_summary", {})
            if sc and sc.get("items_with_credibility", 0) > 0:
                summaries.append(sc)
        
        if not summaries:
            return {}
        
        tier_agg = {"T1": 0, "T2": 0, "T3": 0, "T4": 0}
        grade_agg = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        
        for sc in summaries:
            for tier, cnt in sc.get("tier_distribution", {}).items():
                tier_agg[tier] = tier_agg.get(tier, 0) + cnt
            for grade, cnt in sc.get("grade_distribution", {}).items():
                grade_agg[grade] = grade_agg.get(grade, 0) + cnt
        
        return {
            "source_credibility": {
                "enabled": True,
                "weight": self.config.get("credibility_weight", 0.2),
                "average_credibility_score": calc_stats(
                    [cs.get("average_credibility_score", 0) for cs in summaries]
                ),
                "cross_verified_ratio": calc_stats(
                    [cs.get("cross_verified_ratio", 0) for cs in summaries]
                ),
                "tier_distribution": tier_agg,
                "grade_distribution": grade_agg,
            }
        }

