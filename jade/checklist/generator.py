"""
Checklist Generator - Simplified Version.

Features:
- Compact prompts for atomic checklist generation
- Expert rubric hints injection
- Flat structure, minimal abstraction
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import filelock
from tqdm import tqdm

from jade.llm import BaseLLMClient, create_llm_client
from .prompts import PromptBuilder, PromptConfig, PromptVariant
from .rubric_loader import CompactRubricLoader
from .multilabel_loader import extract_labels_from_item, infer_labels_from_query


@dataclass
class GenerationResult:
    """Result of a checklist generation task."""
    item_id: int
    task_type: str  # "query" or "report"
    success: bool
    checklist: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class ChecklistGenerator:
    """
    Checklist generator for query and report evaluation.
    
    Example:
        >>> from jade.llm import create_llm_client
        >>> client = create_llm_client("openai", model_name="gpt-4")
        >>> generator = ChecklistGenerator(client, output_root="./output")
        >>> 
        >>> checklist = generator.generate_query_checklist(
        ...     query="Find trending products on Amazon",
        ...     labels={"L1_primary_intent": "product_discovery"}
        ... )
    """
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        output_root: str = "./output",
        max_workers: int = 4,
        verbose: bool = True,
        multilabel_rubric_dir: str = "rubrics/bizbench",
        use_skill: bool = True,
        use_report_specific: bool = True,
        prompt_config: Optional[PromptConfig] = None,  # For backward compatibility
    ):
        self.llm_client = llm_client
        self.output_root = Path(output_root)
        self.max_workers = max_workers
        self.verbose = verbose
        self._locks: Dict[str, filelock.FileLock] = {}
        
        # Config (prompt_config overrides individual flags if provided)
        if prompt_config:
            self.use_skill = prompt_config.use_skill
            self.use_report_specific = prompt_config.use_report_specific
        else:
            self.use_skill = use_skill
            self.use_report_specific = use_report_specific
        
        # Prompt builder
        config = PromptConfig(use_skill=self.use_skill, use_report_specific=self.use_report_specific)
        self.prompt_builder = PromptBuilder(config)
        
        # Rubric loader (only if using skill)
        self.rubric_loader = None
        if self.use_skill:
            try:
                self.rubric_loader = CompactRubricLoader(multilabel_rubric_dir)
                self._log(f"✅ Rubric loader: {multilabel_rubric_dir}")
            except Exception as e:
                self._log(f"⚠️ Rubric loader failed: {e}")
        
        self.output_root.mkdir(parents=True, exist_ok=True)
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    # =========================================================================
    # JSON Parsing
    # =========================================================================
    
    def _parse_json(self, response: str) -> List[Dict[str, Any]]:
        """Extract JSON array from LLM response."""
        # Try code blocks
        for pattern in [r'```json\s*([\s\S]*?)\s*```', r'```\s*([\s\S]*?)\s*```']:
            match = re.search(pattern, response)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
        
        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try finding array
        start = response.find('[')
        if start != -1:
            depth = 0
            for i, c in enumerate(response[start:], start):
                depth += (c == '[') - (c == ']')
                if depth == 0:
                    try:
                        return json.loads(response[start:i + 1])
                    except json.JSONDecodeError:
                        break
        
        raise ValueError(f"Failed to parse JSON: {response[:200]}...")
    
    # =========================================================================
    # Core Generation
    # =========================================================================
    
    def generate_query_checklist(
        self,
        query: str,
        labels: Optional[Dict[str, Any]] = None,
        multilabel: Optional[Dict[str, Any]] = None,  # Alias for backward compat
        return_prompt: bool = False,
    ) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], str]:
        """Generate query-specific checklist.
        
        Args:
            query: User query
            labels: Multi-label classification
            multilabel: Alias for labels (backward compatibility)
            return_prompt: If True, also return the filled prompt
        
        Returns:
            Checklist items, or (checklist, prompt) if return_prompt=True
        """
        # Support both parameter names
        labels = labels or multilabel
        
        # Build prompt
        if self.use_skill and self.rubric_loader and labels:
            rubric = self.rubric_loader.load_compact(labels)
            prompt = self.prompt_builder.format_query_prompt(
                query=query,
                deliverable_check=rubric.format_deliverable_check(),
                expert_hints=rubric.format_expert_hints(),
            )
        else:
            prompt = self.prompt_builder.format_query_prompt(query=query)
        
        # Call LLM
        response = self.llm_client.chat_completion([{"role": "user", "content": prompt}])
        if not response or not response.strip():
            raise ValueError("Empty LLM response")
        
        # Parse checklist
        checklist = self._parse_json(response)
        
        # Normalize: ensure required fields
        for i, item in enumerate(checklist):
            if "item_id" not in item:
                item["item_id"] = i
            if "tier" not in item:
                item["tier"] = "general"
            if "depends_on" not in item:
                item["depends_on"] = 0 if i > 0 else None
            if "type" not in item:
                item["type"] = "criterion"
        
        if return_prompt:
            return checklist, prompt
        return checklist
    
    def generate_report_checklist(
        self,
        query: str,
        report: str,
        return_prompt: bool = False,
        return_completion_assessment: bool = False,  # Ignored, for backward compat
    ) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], str]:
        """Generate report-specific checklist.
        
        Args:
            query: User query
            report: AI response to evaluate
            return_prompt: If True, also return the filled prompt
        
        Returns:
            Checklist items, or (checklist, prompt) if return_prompt=True
        """
        if not self.use_report_specific:
            raise ValueError("Report checklist is disabled")
        
        prompt = self.prompt_builder.format_report_prompt(query=query, report_content=report)
        
        response = self.llm_client.chat_completion([{"role": "user", "content": prompt}])
        if not response or not response.strip():
            raise ValueError("Empty LLM response")
        
        checklist = self._parse_json(response)
        
        # Normalize
        for i, item in enumerate(checklist):
            if "item_id" not in item:
                item["item_id"] = i
        
        if return_prompt:
            return checklist, prompt
        return checklist
    
    # =========================================================================
    # File I/O
    # =========================================================================
    
    def _query_path(self, item_id: int) -> Path:
        path = self.output_root / "query_specific_checklist" / "checklists"
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{item_id}_query_specific.json"
    
    def _report_path(self, item_id: int) -> Path:
        path = self.output_root / "report_specific_checklist" / "report_report_specific_checklist"
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{item_id}_report_specific.json"
    
    def _summary_path(self, task_type: str) -> Path:
        folder = self.output_root / f"{task_type}_specific_checklist"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{task_type}_summary.jsonl"
    
    def _save_checklist(
        self,
        item_id: int,
        task_type: str,
        query: str,
        report: Optional[str],
        labels: Optional[Dict],
        checklist: List[Dict],
        prompt: Optional[str] = None,
    ):
        """Save checklist to file."""
        json_path = self._query_path(item_id) if task_type == "query" else self._report_path(item_id)
        
        payload = {
            "id": item_id,
            "query": query,
            "report": report,
            "multilabel": labels,  # Keep 'multilabel' key for backward compat
            "checklist": checklist,
        }
        if prompt:
            payload["generation_prompt"] = prompt
        
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        
        # Append to JSONL summary (thread-safe)
        jsonl_path = self._summary_path(task_type)
        lock_path = jsonl_path.parent / f".{jsonl_path.name}.lock"
        
        if str(lock_path) not in self._locks:
            self._locks[str(lock_path)] = filelock.FileLock(lock_path)
        
        with self._locks[str(lock_path)]:
            with jsonl_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps({"item_id": item_id, **payload}, ensure_ascii=False) + '\n')
    
    def _get_existing_ids(self, task_type: str) -> set:
        """Get already processed IDs."""
        jsonl_path = self._summary_path(task_type)
        if not jsonl_path.exists():
            return set()
        
        ids = set()
        with jsonl_path.open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        ids.add(json.loads(line).get("item_id"))
                    except json.JSONDecodeError:
                        continue
        return ids
    
    # =========================================================================
    # Batch Processing
    # =========================================================================
    
    def _process_item(
        self,
        item_id: int,
        task_type: str,
        query: str,
        report: Optional[str],
        labels: Dict[str, Any],
    ) -> GenerationResult:
        """Process single item."""
        try:
            if task_type == "query":
                checklist, prompt = self.generate_query_checklist(query, labels, return_prompt=True)
            else:
                if not self.use_report_specific:
                    return GenerationResult(item_id, task_type, True, checklist=[])
                checklist, prompt = self.generate_report_checklist(query, report, return_prompt=True)
            
            self._save_checklist(item_id, task_type, query, report, labels, checklist, prompt)
            return GenerationResult(item_id, task_type, True, checklist)
        
        except Exception as e:
            error = str(e)[:200]
            self._log(f"❌ {task_type}_{item_id}: {error}")
            return GenerationResult(item_id, task_type, False, error=error)
    
    def generate_from_data(
        self,
        data: Dict[str, Any],
        resume: bool = True,
        item_ids: Optional[List[int]] = None,
    ) -> Dict[str, List[GenerationResult]]:
        """Generate checklists from dataset.
        
        Args:
            data: {id: {query, report, L1_primary_intent, ...}}
            resume: Skip already processed items
            item_ids: Only process these IDs
        
        Returns:
            {"query": [results], "report": [results]}
        """
        self._log(f"\n📂 Processing {len(data)} items")
        
        existing_query = self._get_existing_ids("query") if resume else set()
        existing_report = self._get_existing_ids("report") if resume else set()
        
        tasks = []
        
        for key, item in data.items():
            item_id = item.get("id") or int(key)
            
            if item_ids and item_id not in item_ids:
                continue
            
            query = item.get("query", "")
            if not query:
                continue
            
            labels = extract_labels_from_item(item)
            if not labels.get("L1_primary_intent"):
                labels = infer_labels_from_query(query)
            
            if item_id not in existing_query:
                tasks.append(("query", item_id, query, None, labels))
            
            report = item.get("report", "")
            if report and item_id not in existing_report:
                tasks.append(("report", item_id, query, report, labels))
        
        self._log(f"📋 {len(tasks)} tasks to execute")
        
        if not tasks:
            self._log("✅ All done.")
            return {"query": [], "report": []}
        
        results = {"query": [], "report": []}
        
        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._process_item, t[1], t[0], t[2], t[3], t[4]): t
                    for t in tasks
                }
                
                with tqdm(total=len(futures), desc="Generating") as pbar:
                    for future in as_completed(futures):
                        task_type = futures[future][0]
                        try:
                            result = future.result()
                            results[task_type].append(result)
                        except Exception as e:
                            self._log(f"❌ {e}")
                        finally:
                            pbar.update(1)
        else:
            for t in tqdm(tasks, desc="Generating"):
                result = self._process_item(t[1], t[0], t[2], t[3], t[4])
                results[t[0]].append(result)
        
        q_ok = sum(1 for r in results["query"] if r.success)
        r_ok = sum(1 for r in results["report"] if r.success)
        self._log(f"\n📊 Query: {q_ok}/{len(results['query'])}, Report: {r_ok}/{len(results['report'])}")
        
        return results
    
    def get_prompt_metadata(self) -> Dict[str, Any]:
        """Get prompt configuration for logging."""
        return {
            "config": {
                "use_skill": self.use_skill,
                "use_report_specific": self.use_report_specific,
            },
            "prompts": {
                "query_checklist_prompt": self.prompt_builder.get_query_prompt(),
                "report_checklist_prompt": self.prompt_builder.get_report_prompt(),
            },
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_generator(
    client_type: str = "custom",
    model_name: Optional[str] = None,
    output_root: str = "./output",
    max_workers: int = 4,
    verbose: bool = True,
    multilabel_rubric_dir: str = "rubrics/bizbench",
    variant: Optional[PromptVariant] = None,
    use_skill: Optional[bool] = None,
    use_report_specific: Optional[bool] = None,
    **kwargs,
) -> ChecklistGenerator:
    """Create a checklist generator.
    
    Args:
        client_type: LLM client type
        model_name: Model name
        output_root: Output directory
        max_workers: Parallel workers
        verbose: Enable logging
        multilabel_rubric_dir: Rubric directory
        variant: Prompt variant (overrides use_skill/use_report_specific)
        use_skill: Enable skill-based prompts
        use_report_specific: Enable report checklist
        **kwargs: Additional LLM client args
    
    Returns:
        ChecklistGenerator instance
    """
    if variant:
        config = PromptConfig.from_variant(variant)
        use_skill = config.use_skill
        use_report_specific = config.use_report_specific
    else:
        use_skill = use_skill if use_skill is not None else True
        use_report_specific = use_report_specific if use_report_specific is not None else True
    
    client = create_llm_client(client_type=client_type, model_name=model_name, **kwargs)
    
    return ChecklistGenerator(
        llm_client=client,
        output_root=output_root,
        max_workers=max_workers,
        verbose=verbose,
        multilabel_rubric_dir=multilabel_rubric_dir,
        use_skill=use_skill,
        use_report_specific=use_report_specific,
    )
