"""
Checklist Utilities for Processing and Merging Checklists.

This module provides utility functions for:
- Loading checklist data from files
- Filtering checklists by type (reasoning/evidence)
- Merging query-specific and report-specific checklists
- Converting between different checklist formats
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

from jade.verification_agent import ChecklistItem


_EQUIVALENCE_HINT_RE = re.compile(
    r"\((?P<inner>[^)]*?\b(?:equivalent|aka|a\.k\.a\.|also known as|i\.e\.|ie)\b[^)]*?)\)",
    flags=re.IGNORECASE,
)


def normalize_claim_for_verification(description: str) -> str:
    """
    Normalize an evidence claim description to reduce false negatives caused by
    alias/equivalence parentheticals.

    Why:
    - Some checklist items embed *equivalence hints* like "(304 equivalent)".
      Verification agents can mistakenly require the parenthetical text to appear
      verbatim in the source. We rewrite these into a "semantic match" instruction.

    Safety:
    - Only triggers when parentheses contain explicit equivalence markers like
      "equivalent", "aka", "also known as", "i.e.".
    - Leaves other parentheticals (units, clarifications) untouched.
    """
    if not description:
        return description

    matches = list(_EQUIVALENCE_HINT_RE.finditer(description))
    if not matches:
        return description

    # Remove only the equivalence-marked parentheticals from the claim text.
    normalized = _EQUIVALENCE_HINT_RE.sub("", description)
    normalized = re.sub(r"\s{2,}", " ", normalized).strip()

    # Append a short instruction to treat equivalence as an alias match.
    # Include the extracted inner content to aid the verifier.
    hints = "; ".join(m.group("inner").strip() for m in matches)
    normalized += (
        " NOTE: The removed parenthetical(s) were equivalence/alias hints "
        f"({hints}). Treat them as synonyms—do not require the source to repeat the "
        "parenthetical verbatim; confirming either term's meaning is sufficient."
    )
    return normalized


def load_input_data(input_path: str) -> Dict[str, Any]:
    """
    Load input data from JSON file.
    
    Expected format:
    - Single object: {"id": <int>, "query": "<text>", "report": "<text>"}
    - List of objects: [{"id": <int>, ...}, ...]
    - Dict mapping: {"0": {"query": "...", "report": "..."}, ...}
    
    Args:
        input_path: Path to input JSON file
        
    Returns:
        Dictionary mapping item IDs to data objects
        
    Raises:
        ValueError: If input format is invalid
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Normalize to dict format
    if isinstance(data, list):
        return {str(item.get("id", idx)): item for idx, item in enumerate(data)}
    elif isinstance(data, dict):
        # Check if it's already in the right format
        if all(isinstance(v, dict) for v in data.values()):
            return data
        # Otherwise treat it as a single item
        if "id" in data or "query" in data:
            item_id = str(data.get("id", 0))
            return {item_id: data}
    
    raise ValueError(f"Invalid input format: expected list or dict, got {type(data)}")


def load_checklist_from_file(
    checklist_path: Path,
    extract_key: str = "checklist"
) -> List[Dict[str, Any]]:
    """
    Load checklist from JSON file.
    
    Handles both wrapped format {"checklist": [...]} and direct list format.
    
    Args:
        checklist_path: Path to checklist JSON file
        extract_key: Key to extract if data is wrapped (default: "checklist")
        
    Returns:
        List of checklist items
    """
    if not checklist_path.exists():
        return []
    
    with open(checklist_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both wrapped format and direct list format
    if isinstance(data, dict) and extract_key in data:
        return data[extract_key]
    elif isinstance(data, list):
        return data
    else:
        return []


def filter_checklist_by_type(
    checklist: List[Dict[str, Any]],
    item_type: str
) -> List[Dict[str, Any]]:
    """
    Filter checklist items by type.
    
    Args:
        checklist: List of checklist items
        item_type: Type to filter by ("reasoning" or "evidence")
        
    Returns:
        Filtered list of checklist items
    """
    return [item for item in checklist if item.get("type") == item_type]


def convert_to_checklist_items(
    checklist: List[Dict[str, Any]]
) -> List[ChecklistItem]:
    """
    Convert checklist dictionaries to ChecklistItem objects.
    
    Args:
        checklist: List of checklist dictionaries
        
    Returns:
        List of ChecklistItem objects
    """
    items = []
    for item in checklist:
        # Handle source field - convert list to comma-separated string if needed
        source = item.get("source")
        if isinstance(source, list):
            source = ", ".join(str(s) for s in source) if source else None
        elif source is not None:
            source = str(source)
            
        items.append(ChecklistItem(
            id=item.get("item_id", 0),
            description=normalize_claim_for_verification(item.get("description", "")),
            source=source
        ))
    return items


def build_criterion_from_description(item: Dict[str, Any]) -> str:
    """
    Build criterion field from description, appending source link if available.
    
    Converts report-specific checklist item description to query-specific format.
    
    Args:
        item: Report-specific checklist item with 'description' and optional 'source'
        
    Returns:
        Criterion string (description + source link if available)
    """
    description = item.get("description", "")
    source = item.get("source")
    
    if source:
        return f"{description} [Source: {source}]"
    else:
        return description


def merge_and_save_reasoning_checklist(
    item_id: int,
    query_checklist: List[Dict[str, Any]],
    report_checklist: List[Dict[str, Any]],
    output_dir: Path
) -> List[Dict[str, Any]]:
    """
    Merge query-specific and reasoning-type report-specific checklists.
    
    Converts report-specific format to match query-specific format and saves the result.
    
    Args:
        item_id: Item ID for file naming
        query_checklist: Query-specific checklist (format: {principle, criterion, weight})
        report_checklist: Report-specific checklist (format: {item_id, type, description, ...})
        output_dir: Output directory path
        
    Returns:
        Merged checklist in query-specific format
    """
    merged_items = []
    
    # Add all query-specific items
    for item in query_checklist:
        merged_items.append(item.copy())
    
    # Convert and add report-specific reasoning items (unified format)
    report_reasoning = filter_checklist_by_type(report_checklist, "reasoning")
    for item in report_reasoning:
        converted_item = {
            "item_id": item.get("item_id", len(merged_items)),
            "type": "reasoning",
            "category": item.get("category", "General"),
            "description": build_criterion_from_description(item),
            "weight": item.get("weight", 1.0)
        }
        merged_items.append(converted_item)
    
    # Save merged checklist
    merged_file = (
        output_dir / "checklists" / "merged_reasoning_checklist" /
        f"{item_id}_merged_reasoning.json"
    )
    merged_file.parent.mkdir(parents=True, exist_ok=True)
    
    merged_data = {
        "id": item_id,
        "checklist": merged_items
    }
    
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    return merged_items


def load_merged_reasoning_checklist(
    item_id: int,
    output_dir: Path
) -> List[Dict[str, Any]]:
    """
    Load merged reasoning checklist from saved file.
    
    Args:
        item_id: Item ID
        output_dir: Output directory path
        
    Returns:
        List of checklist items in query-specific format
    """
    merged_file = (
        output_dir / "checklists" / "merged_reasoning_checklist" /
        f"{item_id}_merged_reasoning.json"
    )
    
    if not merged_file.exists():
        return []
    
    return load_checklist_from_file(merged_file)

