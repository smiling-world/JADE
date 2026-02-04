"""
Score Generator for LLM-as-a-Judge Evaluation.

Three-Dimensional Scoring Framework:
- D1: Reasoning Score (推理质量)
- D2: Evidence Score (事实准确性)
- D3: Credibility Score (来源可信度)

Dependency-Aware Scoring:
- L1 Primary Deliverable is evaluated FIRST
- If L1 fails, dependent items (L2/L3/general) are BLOCKED
- BLOCKED items are not counted in the score calculation
- This ensures we don't evaluate attributes of a deliverable that doesn't exist
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from urllib.parse import urlparse
from pathlib import Path

from jade.llm import BaseLLMClient
from .base_prompt import REASONING_SCORE_PROMPT
from .source_quality import SourceQualityScorer

# Tier score mapping
TIER_SCORES = {"T1": 1.0, "T2": 0.75, "T3": 0.5, "T4": 0.25}
GRADE_THRESHOLDS = [(0.9, "A"), (0.75, "B"), (0.5, "C"), (0.25, "D"), (0, "F")]


def _get_grade(score: float) -> str:
    """Convert score to letter grade."""
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"


def _extract_domain(url: str) -> Optional[str]:
    """Extract root domain from URL (e.g., 'example.com')."""
    try:
        parts = urlparse(url).netloc.lower().split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else None
    except:
        return None


def _normalize_url_for_dedup(url: str) -> str:
    """
    Normalize URL for deduplication by removing text fragments.
    
    Chrome's text fragments (#:~:text=...) point to the same page but with different
    highlighted text. These should be treated as the same URL for deduplication.
    
    Examples:
        - "https://example.com/page#:~:text=hello,world" -> "https://example.com/page"
        - "https://example.com/page#section" -> "https://example.com/page#section" (regular anchor preserved)
        - "https://example.com/page?q=1#:~:text=foo" -> "https://example.com/page?q=1"
    """
    if not url:
        return url
    
    # Remove #:~:text fragment (Chrome text fragment)
    text_fragment_idx = url.find("#:~:text")
    if text_fragment_idx != -1:
        return url[:text_fragment_idx]
    
    return url


@dataclass
class ScoringResult:
    """Result of scoring a single checklist item."""
    item_id: int
    item_type: str  # "reasoning" or "evidence"
    score: float
    weight: float
    weighted_score: float
    analysis: str
    dimension: str = ""
    criterion: str = ""
    target_score: Optional[str] = None
    verification_result: Optional[Dict] = None
    is_applicable: bool = True
    source_credibility: Optional[Dict] = None
    source: Optional[Any] = None  # Original source URL(s) from checklist item
    checklist_source: str = ""  # "query" or "report" to indicate origin
    original_item: Optional[Dict] = None  # Original checklist item for debugging
    # Dependency-aware scoring fields
    tier: str = ""  # "L1", "L2", "L3", or "general"
    depends_on: Optional[int] = None  # item_id this depends on (null for root)
    dependency_status: str = "EVALUATED"  # "EVALUATED", "BLOCKED", "SKIPPED"


class ScoreGenerator:
    """Generator for scoring checklist items against reports."""
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        max_workers: int = 4,
        verbose: bool = True,
        source_quality_scorer: Optional[SourceQualityScorer] = None,
    ):
        self.llm_client = llm_client
        self.max_workers = max_workers
        self.verbose = verbose
        self.source_quality_scorer = source_quality_scorer or SourceQualityScorer()
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    # =========================================================================
    # Reasoning Scoring
    # =========================================================================
    
    def _parse_reasoning_response(self, response: str) -> Tuple[float, str, str, bool]:
        """Parse LLM response. Returns: (score, analysis, target_score, is_applicable)"""
        try:
            match = re.search(r'\{.*?"whether_meet_the_criterion"\s*:\s*".*?".*?\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                analysis = data.get("analysis", "")
                meet = data.get("whether_meet_the_criterion", "NO").upper()
                
                if any(x in meet for x in ["N_A", "N/A", "NOT_APPLICABLE", "NA"]):
                    return 0.0, analysis, "N_A", False
                
                score = {"YES": 1.0, "PARTIAL": 0.5}.get(meet.split()[0], 0.0)
                return score, analysis, meet, True
            return 0.0, "Failed to parse response", None, True
        except Exception as e:
            return 0.0, f"Parse error: {e}", None, True
    
    def score_single_reasoning_item(
        self, item: Dict, query: str, report: str,
        checklist_source: str = "", original_item: Optional[Dict] = None,
        dependency_status: str = "EVALUATED",
    ) -> ScoringResult:
        """Score a single reasoning item with dependency awareness.
        
        Args:
            item: Checklist item to score
            query: User query
            report: AI response to evaluate
            checklist_source: "query" or "report"
            original_item: Original checklist item for debugging
            dependency_status: "EVALUATED", "BLOCKED", or "SKIPPED"
        """
        item_id = item.get("item_id", 0)
        # Support both old format (criterion/principle) and new unified format (description/category)
        criterion = item.get("criterion") or item.get("description", "")
        dimension = item.get("principle") or item.get("category", "")
        weight = float(item.get("weight", 1.0))
        tier = item.get("tier", "")
        depends_on = item.get("depends_on")
        
        # If item is BLOCKED, skip LLM call and return blocked result
        if dependency_status == "BLOCKED":
            return ScoringResult(
                item_id=item_id, item_type="reasoning", score=0.0, weight=weight,
                weighted_score=0.0,
                analysis="BLOCKED: Dependency (L1 Primary Deliverable) not satisfied",
                dimension=dimension, criterion=criterion,
                target_score=None, is_applicable=False,
                checklist_source=checklist_source,
                original_item=original_item or item,
                tier=tier, depends_on=depends_on, dependency_status="BLOCKED",
            )
        
        prompt = REASONING_SCORE_PROMPT.format(query=query, report=report, criterion=criterion)
        
        try:
            response = self.llm_client.chat_completion([{"role": "user", "content": prompt}])
            score, analysis, target_score, is_applicable = self._parse_reasoning_response(response)
        except Exception as e:
            score, analysis, target_score, is_applicable = 0.0, f"Error: {e}", None, True
        
        return ScoringResult(
            item_id=item_id, item_type="reasoning", score=score, weight=weight,
            weighted_score=score * weight if is_applicable else 0.0,
            analysis=analysis, dimension=dimension, criterion=criterion,
            target_score=target_score, is_applicable=is_applicable,
            checklist_source=checklist_source,
            original_item=original_item or item,
            tier=tier, depends_on=depends_on, dependency_status=dependency_status,
        )
    
    def _evaluate_l1_first(
        self,
        checklist_items: List[Dict],
        query: str,
        report: str,
        checklist_source: str = "",
    ) -> Tuple[List[ScoringResult], bool]:
        """
        Evaluate L1 Primary Deliverable items first.
        
        Returns:
            Tuple of (l1_results, l1_passed)
            - l1_results: Scoring results for L1 items
            - l1_passed: Whether L1 Primary Deliverable is satisfied
        """
        l1_items = [item for item in checklist_items if item.get("tier") == "L1"]
        
        if not l1_items:
            # No L1 items - assume passed (backward compatibility)
            return [], True
        
        self._log(f"  Evaluating {len(l1_items)} L1 Primary Deliverable items first...")
        
        # Score L1 items
        l1_results = []
        for item in l1_items:
            result = self.score_single_reasoning_item(
                item, query, report, checklist_source, item, dependency_status="EVALUATED"
            )
            l1_results.append(result)
        
        # Determine if L1 passed:
        # - All positive L1 items (weight > 0) should score >= 0.5
        # - No critical flaw L1 items (weight < 0) should trigger (score == 1.0 means flaw exists)
        l1_passed = True
        for result in l1_results:
            if result.is_applicable:
                if result.weight > 0:
                    # Positive criterion - should pass (score >= 0.5)
                    if result.score < 0.5:
                        l1_passed = False
                        self._log(f"  ❌ L1 item {result.item_id} failed: {result.criterion[:50]}...")
                        break
                else:
                    # Critical flaw criterion - should NOT trigger (score should be 0)
                    if result.score > 0.5:
                        l1_passed = False
                        self._log(f"  ❌ L1 critical flaw triggered: {result.criterion[:50]}...")
                        break
        
        if l1_passed:
            self._log(f"  ✅ L1 Primary Deliverable satisfied")
        else:
            self._log(f"  ⚠️ L1 Primary Deliverable NOT satisfied - dependent items will be BLOCKED")
        
        return l1_results, l1_passed
    
    def score_reasoning_items(
        self, checklist_items: List[Dict], query: str, report: str,
        parallel: bool = True, checklist_source: str = "",
        original_items: Optional[List[Dict]] = None,
        enable_dependency_blocking: bool = True,
    ) -> List[ScoringResult]:
        """Score multiple reasoning items with dependency awareness.
        
        Args:
            checklist_items: List of checklist items to score
            query: User query
            report: AI response to evaluate
            parallel: Enable parallel execution
            checklist_source: "query" or "report"
            original_items: Original items for debugging
            enable_dependency_blocking: If True, BLOCK dependent items when L1 fails
        
        Returns:
            List of ScoringResult with dependency_status indicating evaluation state
        """
        if not checklist_items:
            return []
        
        self._log(f"Scoring {len(checklist_items)} reasoning items...")
        
        # Use original_items if provided, otherwise use checklist_items as original
        orig_items = original_items or checklist_items
        
        # Check if any items have tier/depends_on (new dependency model)
        has_dependency_model = any(
            item.get("tier") or item.get("depends_on") is not None
            for item in checklist_items
        )
        
        if has_dependency_model and enable_dependency_blocking:
            # NEW: Dependency-aware scoring
            # Step 1: Evaluate L1 items first
            l1_results, l1_passed = self._evaluate_l1_first(
                checklist_items, query, report, checklist_source
            )
            l1_item_ids = {r.item_id for r in l1_results}
            
            # Step 2: Determine which items to evaluate vs block
            non_l1_items = [item for item in checklist_items if item.get("item_id") not in l1_item_ids]
            
            if l1_passed:
                # L1 passed - evaluate all dependent items normally
                if parallel and self.max_workers > 1 and len(non_l1_items) > 1:
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        futures = {
                            executor.submit(
                                self.score_single_reasoning_item, item, query, report,
                                checklist_source, item, "EVALUATED"
                            ): item
                            for item in non_l1_items
                        }
                        non_l1_results = [f.result() for f in tqdm(as_completed(futures), total=len(futures), desc="Scoring (L2/L3)")]
                else:
                    non_l1_results = [
                        self.score_single_reasoning_item(item, query, report, checklist_source, item, "EVALUATED")
                        for item in tqdm(non_l1_items, desc="Scoring (L2/L3)")
                    ]
            else:
                # L1 failed - BLOCK all dependent items
                non_l1_results = []
                for item in non_l1_items:
                    depends_on = item.get("depends_on")
                    # Items that depend on L1 (item_id 0) or have any dependency are blocked
                    if depends_on is not None:
                        result = self.score_single_reasoning_item(
                            item, query, report, checklist_source, item, "BLOCKED"
                        )
                    else:
                        # Items with no dependency are still evaluated
                        result = self.score_single_reasoning_item(
                            item, query, report, checklist_source, item, "EVALUATED"
                        )
                    non_l1_results.append(result)
            
            # Combine and sort
            all_results = l1_results + non_l1_results
            all_results.sort(key=lambda x: x.item_id)
            return all_results
        
        else:
            # Legacy behavior - no dependency model
            if parallel and self.max_workers > 1 and len(checklist_items) > 1:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {
                        executor.submit(
                            self.score_single_reasoning_item, item, query, report,
                            checklist_source, orig_items[i] if i < len(orig_items) else item
                        ): (i, item)
                        for i, item in enumerate(checklist_items)
                    }
                    results = [f.result() for f in tqdm(as_completed(futures), total=len(futures), desc="Scoring")]
                results.sort(key=lambda x: x.item_id)
                return results
            
            return [
                self.score_single_reasoning_item(item, query, report, checklist_source, orig_items[i] if i < len(orig_items) else item)
                for i, item in enumerate(tqdm(checklist_items, desc="Scoring"))
            ]
    
    # =========================================================================
    # Evidence Scoring
    # =========================================================================
    
    def _evaluate_source_credibility(self, item: Dict) -> Optional[Dict[str, Any]]:
        """
        Evaluate source credibility for an evidence item's claimed source.
        Returns None if no claimed_source (report didn't cite a URL for this claim).
        
        Handles both single URL strings and lists of URLs.
        For lists, uses the first URL for credibility scoring.
        """
        claimed_source = item.get("source")
        if not claimed_source:
            return None
        
        # Handle case where source is a list of URLs
        # Use the first URL for credibility evaluation
        if isinstance(claimed_source, list):
            if not claimed_source:  # Empty list
                return None
            claimed_source = claimed_source[0]  # Use first URL
        
        # Ensure claimed_source is a string
        if not isinstance(claimed_source, str):
            return None
        
        eval_result = self.source_quality_scorer._classify_source(claimed_source)
        tier = eval_result.tier.value
        score = TIER_SCORES.get(tier, 0.25)
        
        return {
            "item_id": item.get("item_id", 0),
            "claimed_source": claimed_source,
            "claimed_source_tier": tier,
            "claimed_source_score": score,
            "credibility_score": score,
            "credibility_grade": _get_grade(score),
        }
    
    def score_evidence_items(
        self,
        checklist_items: List[Dict],
        verification_results: List[Dict],
        query: str = "",
        enable_source_credibility: bool = True,
        checklist_source: str = "report",
    ) -> List[ScoringResult]:
        """Score evidence items from verification results."""
        if not checklist_items:
            return []
        
        self._log(f"Scoring {len(checklist_items)} evidence items...")
        verification_map = {r.get("item_id"): r for r in verification_results}
        
        results = []
        for item in checklist_items:
            item_id = item.get("item_id", 0)
            weight = float(item.get("weight", 1.0))
            verification = verification_map.get(item_id)
            
            # Calculate score
            if verification:
                conclusion = verification.get("conclusion", "no")
                confidence = verification.get("confidence", 0)
                score = confidence / 100.0 if conclusion == "yes" else (100 - confidence) / 100.0 * 0.3
                analysis = verification.get("reason", {}).get("summary", f"Verification: {conclusion}")
            else:
                score, analysis = 0.0, "No verification result"
            
            # Check N/A
            is_applicable = True
            if verification:
                conc = str(verification.get("conclusion", "")).upper()
                if any(x in conc for x in ["N_A", "N/A", "NA"]):
                    is_applicable, score = False, 0.0
            
            # Source credibility - collect sources from multiple places
            source_cred = None
            if enable_source_credibility and is_applicable:
                sources = []
                
                # 1. Check checklist item's source field (for Source Validity items)
                if item_source := item.get("source"):
                    if isinstance(item_source, list):
                        sources.extend(item_source)
                    elif isinstance(item_source, str):
                        sources.append(item_source)
                
                # 2. Get sources from verification result (reference_urls.supporting)
                if verification:
                    ref_urls = verification.get("reference_urls", {})
                    for url in (ref_urls.get("supporting", []) if isinstance(ref_urls, dict) else []):
                        if url not in sources:
                            sources.append(url)
                    
                    # Also check source_url field
                    if (source_url := verification.get("source_url")) and source_url not in sources:
                        sources.append(source_url)
                
                if sources:
                    item_with_source = {**item, "source": sources}
                    source_cred = self._evaluate_source_credibility(item_with_source)
            
            results.append(ScoringResult(
                item_id=item_id, item_type="evidence", score=score, weight=weight,
                weighted_score=score * weight if is_applicable else 0.0,
                analysis=analysis, dimension=item.get("category", ""),
                criterion=item.get("description", ""),
                target_score=verification.get("conclusion") if verification else None,
                verification_result=verification, is_applicable=is_applicable,
                source_credibility=source_cred,
                source=item.get("source"),  # Preserve original source URL(s)
                checklist_source=checklist_source,
                original_item=item,  # Preserve original checklist item for debugging
            ))
        
        return results
    
    # =========================================================================
    # Weighted Average Calculation
    # =========================================================================
    
    def _summarize_credibility(self, evidence_scores: List[ScoringResult]) -> Dict[str, Any]:
        """Summarize source credibility at report level with before/after dedup statistics."""
        empty_result = {
            # Before dedup (raw)
            "total_source_count": 0,
            "all_sources": [],
            "tier_distribution_before": {"T1": 0, "T2": 0, "T3": 0, "T4": 0},
            "avg_score_before": 0.0,
            # After URL dedup
            "unique_url_count": 0,
            "unique_sources": [],
            "tier_distribution": {"T1": 0, "T2": 0, "T3": 0, "T4": 0},
            "average_credibility_score": 0.0,
            # After domain dedup
            "unique_domain_count": 0,
            "unique_domains": [],
            # Grades and diversity
            "grade_distribution": {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0},
            "diversity_assessment": "no_sources",
            "items_with_source": 0,
            "items_without_source": 0,
        }
        
        if not evidence_scores:
            return empty_result
        
        # Collect all claimed sources (before dedup)
        all_sources, grade_dist = [], {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        tier_dist_before = {"T1": 0, "T2": 0, "T3": 0, "T4": 0}
        
        for e in evidence_scores:
            if sc := e.source_credibility:
                src_entry = {
                    "url": sc["claimed_source"],
                    "tier": sc["claimed_source_tier"],
                    "score": sc["claimed_source_score"],
                    "item_id": sc["item_id"],
                    "grade": sc.get("credibility_grade", "F"),
                }
                all_sources.append(src_entry)
                tier_dist_before[sc["claimed_source_tier"]] += 1
                grade_dist[sc.get("credibility_grade", "F")] += 1
        
        items_with = len(all_sources)
        items_without = len(evidence_scores) - items_with
        
        if not all_sources:
            return {**empty_result, "items_without_source": items_without}
        
        avg_before = sum(s["score"] for s in all_sources) / len(all_sources)
        
        # URL-level dedup (normalize URLs by removing #:~:text fragments before comparison)
        seen_urls, unique_sources, tier_dist = set(), [], {"T1": 0, "T2": 0, "T3": 0, "T4": 0}
        for src in all_sources:
            if url := src["url"]:
                normalized_url = _normalize_url_for_dedup(url)
                if normalized_url not in seen_urls:
                    seen_urls.add(normalized_url)
                    unique_sources.append(src)
                    tier_dist[src["tier"]] += 1
        
        avg_after = sum(s["score"] for s in unique_sources) / len(unique_sources) if unique_sources else 0.0
        
        # Domain-level dedup
        domain_map = {}  # domain -> best source
        for src in unique_sources:
            if domain := _extract_domain(src["url"]):
                if domain not in domain_map or src["score"] > domain_map[domain]["score"]:
                    domain_map[domain] = src
        
        unique_domains = list(domain_map.keys())
        diversity = "no_sources" if not unique_domains else ("single_source" if len(unique_domains) == 1 else "diverse")
        
        return {
            # Before dedup (raw)
            "total_source_count": len(all_sources),
            "all_sources": all_sources,
            "tier_distribution_before": tier_dist_before,
            "avg_score_before": round(avg_before, 4),
            # After URL dedup
            "unique_url_count": len(unique_sources),
            "unique_sources": unique_sources,
            "tier_distribution": tier_dist,
            "average_credibility_score": round(avg_after, 4),
            # After domain dedup
            "unique_domain_count": len(unique_domains),
            "unique_domains": unique_domains,
            # Grades and diversity
            "grade_distribution": grade_dist,
            "diversity_assessment": diversity,
            "items_with_source": items_with,
            "items_without_source": items_without,
        }
    
    def _calculate_tier_breakdown(
        self,
        reasoning_scores: List[ScoringResult],
    ) -> Dict[str, Any]:
        """
        Calculate score breakdown by tier (L1/L2/L3/general).
        
        Returns:
            Dictionary with per-tier statistics
        """
        tier_stats = {
            "L1": {"count": 0, "evaluated": 0, "blocked": 0, "score_sum": 0.0, "weight_sum": 0.0},
            "L2": {"count": 0, "evaluated": 0, "blocked": 0, "score_sum": 0.0, "weight_sum": 0.0},
            "L3": {"count": 0, "evaluated": 0, "blocked": 0, "score_sum": 0.0, "weight_sum": 0.0},
            "general": {"count": 0, "evaluated": 0, "blocked": 0, "score_sum": 0.0, "weight_sum": 0.0},
        }
        
        for r in reasoning_scores:
            tier = r.tier if r.tier in tier_stats else "general"
            tier_stats[tier]["count"] += 1
            
            if r.dependency_status == "BLOCKED":
                tier_stats[tier]["blocked"] += 1
            elif r.is_applicable:
                tier_stats[tier]["evaluated"] += 1
                # For positive weights, add score * weight
                # For negative weights (critical flaws), add (1-score) * |weight| (inverted)
                if r.weight >= 0:
                    tier_stats[tier]["score_sum"] += r.score * r.weight
                else:
                    # Critical flaw: score=1 means flaw exists (bad), score=0 means no flaw (good)
                    tier_stats[tier]["score_sum"] += (1 - r.score) * abs(r.weight)
                tier_stats[tier]["weight_sum"] += abs(r.weight)
        
        # Calculate averages
        result = {}
        for tier, stats in tier_stats.items():
            if stats["count"] > 0:
                avg_score = stats["score_sum"] / stats["weight_sum"] if stats["weight_sum"] > 0 else 0.0
                result[tier] = {
                    "count": stats["count"],
                    "evaluated": stats["evaluated"],
                    "blocked": stats["blocked"],
                    "average_score": round(avg_score, 4),
                    "pass_rate": round(stats["evaluated"] / stats["count"], 4) if stats["count"] > 0 else 0.0,
                }
        
        # Add L1 specific info
        if "L1" in result and result["L1"]["count"] > 0:
            result["l1_passed"] = result["L1"]["average_score"] >= 0.5 and result["L1"]["blocked"] == 0
        else:
            result["l1_passed"] = True  # No L1 items = assume passed (backward compatibility)
        
        return result
    
    def calculate_weighted_average(
        self,
        reasoning_scores: List[ScoringResult],
        evidence_scores: List[ScoringResult],
        reasoning_weight: float = 0.4,
        evidence_weight: float = 0.4,
        credibility_weight: float = 0.2,
        confidence_threshold: float = 0.7,
        score_fusion_mode: str = "multiplicative",
    ) -> Dict[str, Any]:
        """
        Calculate three-dimensional scores with optional fusion and tier breakdown.
        
        Args:
            reasoning_scores: List of reasoning scoring results
            evidence_scores: List of evidence scoring results
            reasoning_weight: Weight for reasoning dimension (legacy, used only in "weighted" mode)
            evidence_weight: Weight for evidence dimension (legacy, used only in "weighted" mode)
            credibility_weight: Weight for credibility dimension (legacy, used only in "weighted" mode)
            confidence_threshold: Minimum confidence for evidence to be considered valid
            score_fusion_mode: Fusion strategy:
                - "multiplicative": final = reasoning × evidence × credibility (default, recommended)
                - "two_stage": final = reasoning × evidence (for ablation without credibility)
                - "weighted": legacy weighted average (for backward compatibility)
                - "independent": no fusion, returns None for final_score
        
        Returns:
            Dictionary with scores, tier_breakdown, and dependency statistics
        """
        # Filter applicable items (exclude BLOCKED items from scoring)
        applicable_reasoning = [
            r for r in reasoning_scores 
            if r.is_applicable and r.dependency_status != "BLOCKED"
        ]
        valid_evidence = [
            e for e in evidence_scores
            if e.is_applicable and e.verification_result
            and (e.verification_result.get("confidence", 0) / 100.0) >= confidence_threshold
        ]
        
        # Weighted average helper
        def calc_avg(items: List[ScoringResult]) -> float:
            if not items:
                return 0.0
            contrib = sum(r.score * r.weight if r.weight >= 0 else (1 - r.score) * abs(r.weight) for r in items)
            total = sum(abs(r.weight) for r in items)
            return contrib / total if total > 0 else 0.0
        
        reasoning_avg = calc_avg(applicable_reasoning)
        evidence_avg = calc_avg(valid_evidence)
        
        # D3: Credibility (only items with claimed_source contribute)
        cred_scores = [e.source_credibility["claimed_source_score"] for e in valid_evidence if e.source_credibility]
        credibility_avg = sum(cred_scores) / len(cred_scores) if cred_scores else 0.0
        
        # Fusion
        # Note: For multiplicative modes, we use max(score, 0.01) to avoid zero products
        # while still heavily penalizing very low scores.
        # IMPORTANT: If a dimension has NO data (count=0), skip it (use 1.0) to avoid
        # unfairly penalizing cases where that dimension wasn't evaluated.
        has_reasoning = len(applicable_reasoning) > 0
        has_evidence = len(evidence_scores) > 0  # Check original list, not filtered
        has_credibility = len(cred_scores) > 0
        
        if score_fusion_mode == "independent":
            final_score = None
        elif score_fusion_mode == "multiplicative":
            # Multiplicative: final = reasoning × evidence × credibility
            # Skip dimensions that have no data (use 1.0)
            r = max(reasoning_avg, 0.01) if has_reasoning else 1.0
            e = max(evidence_avg, 0.01) if has_evidence else 1.0
            c = max(credibility_avg, 0.01) if has_credibility else 1.0
            final_score = r * e * c
        elif score_fusion_mode == "two_stage":
            # Two-stage: reasoning × evidence only (for ablation)
            # If evidence is 0 (no valid evidence scores), use reasoning only
            if evidence_avg == 0.0 or len(valid_evidence) == 0:
                final_score = reasoning_avg
            else:
                # Skip dimensions that have no data (use 1.0)
                r = max(reasoning_avg, 0.01) if has_reasoning else 1.0
                e = max(evidence_avg, 0.01) if has_evidence else 1.0
                final_score = r * e
        else:  # weighted (legacy)
            total = reasoning_weight + evidence_weight + credibility_weight
            final_score = (reasoning_avg * reasoning_weight + evidence_avg * evidence_weight + credibility_avg * credibility_weight) / total if total > 0 else 0.0
        
        # Dimension breakdown
        dims = {}
        for items, key in [(applicable_reasoning, "r"), (valid_evidence, "e")]:
            for item in items:
                d = item.dimension or "General"
                dims.setdefault(d, {"r_c": 0, "r_w": 0, "e_c": 0, "e_w": 0})
                contrib = item.score * item.weight if item.weight >= 0 else (1 - item.score) * abs(item.weight)
                dims[d][f"{key}_c"] += contrib
                dims[d][f"{key}_w"] += abs(item.weight)
        
        dimension_scores = {}
        for d, data in dims.items():
            r_rate = data["r_c"] / data["r_w"] if data["r_w"] else 0
            e_rate = data["e_c"] / data["e_w"] if data["e_w"] else 0
            total_w = data["r_w"] + data["e_w"]
            dimension_scores[d] = {
                "score_rate": (data["r_c"] + data["e_c"]) / total_w if total_w else 0,
                "reasoning_score_rate": r_rate, "evidence_score_rate": e_rate,
            }
        
        # Calculate tier breakdown for dependency-aware scoring
        tier_breakdown = self._calculate_tier_breakdown(reasoning_scores)
        
        # Calculate dependency statistics
        total_items = len(reasoning_scores)
        blocked_items = sum(1 for r in reasoning_scores if r.dependency_status == "BLOCKED")
        evaluated_items = sum(1 for r in reasoning_scores if r.dependency_status == "EVALUATED" and r.is_applicable)
        
        result = {
            "reasoning_score": reasoning_avg, "evidence_score": evidence_avg,
            "credibility_score": credibility_avg, "final_score": final_score,
            "score_fusion_mode": score_fusion_mode,
            "weights": {"reasoning": reasoning_weight, "evidence": evidence_weight, "credibility": credibility_weight},
            "dimension_scores": dimension_scores,
            "source_credibility_summary": self._summarize_credibility(valid_evidence),
            # Tier breakdown for dependency-aware scoring
            "tier_breakdown": tier_breakdown,
            "dependency_stats": {
                "total_items": total_items,
                "evaluated": evaluated_items,
                "blocked": blocked_items,
                "l1_passed": tier_breakdown.get("l1_passed", True),
            },
            "metadata": {
                "reasoning_count": len(reasoning_scores), "reasoning_applicable": len(applicable_reasoning),
                "evidence_count": len(evidence_scores), "valid_evidence_count": len(valid_evidence),
                "blocked_count": blocked_items,
            },
            "reasoning_details": [
                {
                    "item_id": r.item_id,
                    "type": r.item_type,
                    "tier": r.tier,
                    "depends_on": r.depends_on,
                    "dependency_status": r.dependency_status,
                    "checklist_source": r.checklist_source,
                    "category": r.dimension,
                    "description": r.criterion,
                    "weight": r.weight,
                    "score": r.score,
                    "analysis": r.analysis,
                    "is_applicable": r.is_applicable,
                    "original_item": r.original_item,
                }
                for r in reasoning_scores
            ],
            "evidence_details": [
                {
                    "item_id": e.item_id,
                    "type": e.item_type,
                    "checklist_source": e.checklist_source,
                    "category": e.dimension,
                    "source": e.source,
                    "description": e.criterion,
                    "weight": e.weight,
                    "score": e.score,
                    "analysis": e.analysis,
                    "is_applicable": e.is_applicable,
                    "source_credibility": e.source_credibility,
                    "original_item": e.original_item,
                }
                for e in evidence_scores
            ],
        }
        
        return result