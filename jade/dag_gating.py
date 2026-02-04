"""
Evidence -> Reasoning DAG gating utilities.

This module is intentionally dependency-free (no third-party imports),
so it can be imported in lightweight tests without requiring the full stack.
"""

from typing import Any, Dict, List


def apply_evidence_gating(
    *,
    report_reasoning_checklist: List[Dict[str, Any]],
    report_reasoning_scores: List[Any],
    evidence_scores: List[Any],
    confidence_threshold: float,
) -> List[Any]:
    """
    Apply evidence->reasoning DAG gating:
    If any depended evidence item FAILS with confidence >= threshold, the reasoning item is forced to FAIL.

    Expected inputs:
    - report_reasoning_checklist: list of dicts. Each may have:
        - item_id: int
        - depends_on: list[int]  (evidence item_ids)
    - report_reasoning_scores: list of objects with attributes:
        - item_id: int
        - weight: float
        - score: float
        - target_score: str|None
        - analysis: str
        - weighted_score: float
    - evidence_scores: list of objects with attributes:
        - item_id: int
        - verification_result: dict with keys {conclusion, confidence}
    """
    if not report_reasoning_checklist or not report_reasoning_scores:
        return report_reasoning_scores

    # Evidence status map (only high-confidence results)
    evidence_status: Dict[int, str] = {}
    evidence_meta: Dict[int, Dict[str, Any]] = {}

    for e in evidence_scores:
        vr = getattr(e, "verification_result", None)
        if not vr:
            continue
        conf = float(vr.get("confidence", 0)) / 100.0
        if conf < float(confidence_threshold):
            continue
        concl = str(vr.get("conclusion", "")).lower()
        if concl in ("n_a", "na", "n/a"):
            continue
        if concl == "yes":
            evidence_status[getattr(e, "item_id")] = "pass"
        elif concl == "no":
            evidence_status[getattr(e, "item_id")] = "fail"
        evidence_meta[getattr(e, "item_id")] = {"confidence": vr.get("confidence", 0), "conclusion": vr.get("conclusion")}

    # Build dependency map from checklist
    depends_map: Dict[int, List[int]] = {}
    for it in report_reasoning_checklist:
        rid = it.get("item_id")
        deps = it.get("depends_on", []) or []
        if isinstance(rid, int) and isinstance(deps, list):
            depends_map[rid] = [d for d in deps if isinstance(d, int)]

    for r in report_reasoning_scores:
        rid = getattr(r, "item_id", None)
        deps = depends_map.get(rid, [])
        if not deps:
            continue
        failed = [d for d in deps if evidence_status.get(d) == "fail"]
        if not failed:
            continue

        # For positive-weight criteria: fail => NO/0
        # For negative-weight (penalty) criteria: fail => YES/1
        weight = float(getattr(r, "weight", 0.0))
        forced_score = 0.0 if weight >= 0 else 1.0
        forced_target = "NO" if weight >= 0 else "YES"

        detail = ", ".join(
            f"{d}({evidence_meta.get(d, {}).get('conclusion','?')}/{evidence_meta.get(d, {}).get('confidence','?')}%)"
            for d in failed
        )
        original_analysis = getattr(r, "analysis", "")
        setattr(
            r,
            "analysis",
            (
                f"[GATED BY EVIDENCE] Depended evidence failed: {detail}. "
                f"Therefore this reasoning item is forced to {forced_target}.\n"
                f"Original analysis: {original_analysis}"
            ),
        )
        setattr(r, "score", forced_score)
        setattr(r, "target_score", forced_target)
        setattr(r, "weighted_score", forced_score * weight)

    return report_reasoning_scores


