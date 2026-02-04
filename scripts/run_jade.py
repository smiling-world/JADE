#!/usr/bin/env python3
"""
Batch Evaluation Script for Experiment 1.

This script runs evaluation pipeline on all models in data/input
and generates a comparison report.

Features:
- Resume mode: Automatically skips models that already have completed evaluations
- Parallel execution: Run multiple models concurrently using --max-workers
- Config-based settings: All parameters can be set in config file (batch section)
- CLI override: Command line arguments override config file values
- Failed case filtering: Automatically identifies and excludes scoring failures
  from statistics (empty responses, all-zero scores, invalid evaluations)

Concurrency Levels (three-layer architecture):
1. Outer level (batch.max_workers): Number of models evaluated in parallel
   - Controlled by config or --max-workers argument (default: 1, sequential)
2. Pipeline level (evaluation.max_workers): Internal parallelism for checklist/scoring
   - Controlled by config file: evaluation.max_workers (default: 4)
   - Used for checklist generation, scoring, analysis generation
3. Agent level (agent.verify_concurrency): Evidence verification parallelism
   - Controlled by config file: agent.verify_concurrency (default: 5)
   - Used for parallel evidence verification within GenAI agent

Note: max_workers and verify_concurrency are NOT multiplied - they're used
in different stages. Max concurrent = batch_workers × max(pipeline_workers, agent_concurrency)
Example: 3 models × max(4 workers, 5 verifications) = 15 concurrent operations
⚠️  Be careful with high concurrency to avoid API rate limits!

Configuration File (configs/bizbench_eval.yaml):
    batch:
      models: "model1,model2"           # Comma-separated model names
      models_file: "models.txt"         # File with model names (one per line)
      pattern: "*_with_tool"            # Glob pattern to match models
      exclude: "model3,model4"          # Comma-separated models to exclude
      exclude_pattern: "*_no_tool"      # Glob pattern to exclude models
      max_workers: 1                    # Parallel model evaluations
      max_items: null                   # Max items per model (null = all)
      item_ids: null                    # Specific item IDs list (null = all)
      resume: true                      # Skip completed models
      dry_run: false                    # Dry run mode

Usage (CLI arguments override config):
    python scripts/run_jade.py
    python scripts/run_jade.py --config configs/bizbench_eval.yaml
    python scripts/run_jade.py --models "claude-opus-4.5_with_tool,gpt-5.2_with_tool"
    python scripts/run_jade.py --models-file models_to_run.txt
    python scripts/run_jade.py --pattern "*_with_tool"
    python scripts/run_jade.py --exclude "gpt-4o_no_tool"
    python scripts/run_jade.py --max-workers 3
    python scripts/run_jade.py --max-items 5
    python scripts/run_jade.py --item-ids "1,2,3,5,10"
    python scripts/run_jade.py --item-ids-file item_ids.txt
    python scripts/run_jade.py --no-resume
    python scripts/run_jade.py --dry-run

Note: --item-ids takes precedence over --max-items if both are specified.
"""

import argparse
import fnmatch
import json
import os
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

import yaml

# Add parent directory to path to import eval_pipeline_genai
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from scripts.eval_pipeline_genai import GenAIEvalPipeline


# =============================================================================
# Constants
# =============================================================================

DEFAULT_CONFIG_PATH = "configs/bizbench_eval.yaml"
DEFAULT_INPUT_DIR = "data/input"
DEFAULT_OUTPUT_DIR = "output/bizbench_eval_batch"
DEFAULT_METADATA_DIR = "data/metadata"


# =============================================================================
# Helper Functions
# =============================================================================

def load_yaml_file(path: Path) -> Dict[str, Any]:
    """Load YAML file safely."""
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def load_json_file(path: Path) -> Dict[str, Any]:
    """Load JSON file safely."""
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_json_file(path: Path, data: Dict[str, Any]) -> None:
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def is_failed_case(score_data: Dict[str, Any]) -> tuple[bool, str]:
    """Check if a score item represents a failed evaluation case.
    
    Failed cases are identified by:
    1. All scores (final, reasoning, evidence, credibility) are zero
    2. Most reasoning items are marked as not applicable (is_applicable=false)
    3. Analysis text indicates target_report is None
    
    Args:
        score_data: Score data dictionary from individual score file
        
    Returns:
        Tuple of (is_failed, reason)
    """
    final_score = score_data.get('final_score', 0)
    reasoning_score = score_data.get('reasoning_score', 0)
    evidence_score = score_data.get('evidence_score', 0)
    credibility_score = score_data.get('credibility_score', 0)
    
    # Check 1: All scores are zero
    if final_score == 0 and reasoning_score == 0 and evidence_score == 0 and credibility_score == 0:
        return True, "all_scores_zero"
    
    # Check 2: Reasoning score is 0 and most items are not applicable
    if reasoning_score == 0:
        details = score_data.get('reasoning_details', [])
        if details:
            applicable_count = sum(1 for d in details if d.get('is_applicable', True))
            total_count = len(details)
            
            # If only 0-1 items are applicable out of many (>5), it's likely a failed case
            if applicable_count <= 1 and total_count > 5:
                return True, "mostly_not_applicable"
            
            # Check 3: Analysis text indicates None report
            for detail in details:
                analysis = detail.get('analysis', '').lower()
                if "target_report is 'none'" in analysis or "target_report is none" in analysis:
                    return True, "none_report_detected"
    
    return False, ""


def calculate_filtered_statistics(scores_dir: Path) -> Dict[str, Any]:
    """Calculate statistics after filtering out failed cases.
    
    Args:
        scores_dir: Path to the scores directory containing *_scores.json files
        
    Returns:
        Dictionary with filtered statistics and failure info
    """
    if not scores_dir.exists():
        return {}
    
    final_scores = []
    reasoning_scores = []
    evidence_scores = []
    credibility_scores = []
    
    failed_cases = []
    total_items = 0
    
    for score_file in scores_dir.glob("*_scores.json"):
        try:
            with open(score_file, 'r', encoding='utf-8') as f:
                score_data = json.load(f)
            
            total_items += 1
            item_id = score_file.stem.replace('_scores', '')
            
            # Check if this is a failed case
            is_failed, reason = is_failed_case(score_data)
            
            if is_failed:
                failed_cases.append({
                    'item_id': item_id,
                    'reason': reason,
                    'file': score_file.name
                })
                continue
            
            # Collect valid scores
            final_scores.append(score_data.get('final_score', 0))
            reasoning_scores.append(score_data.get('reasoning_score', 0))
            evidence_scores.append(score_data.get('evidence_score', 0))
            credibility_scores.append(score_data.get('credibility_score', 0))
            
        except (json.JSONDecodeError, KeyError) as e:
            failed_cases.append({
                'item_id': score_file.stem.replace('_scores', ''),
                'reason': f'parse_error: {str(e)}',
                'file': score_file.name
            })
    
    # Calculate statistics
    def calc_stats(scores: List[float]) -> Dict[str, float]:
        if not scores:
            return {'mean': 0, 'min': 0, 'max': 0, 'count': 0}
        return {
            'mean': sum(scores) / len(scores),
            'min': min(scores),
            'max': max(scores),
            'count': len(scores)
        }
    
    return {
        'filtered_statistics': {
            'final_score': calc_stats(final_scores),
            'reasoning_score': calc_stats(reasoning_scores),
            'evidence_score': calc_stats(evidence_scores),
            'credibility_score': calc_stats(credibility_scores),
        },
        'filtering_info': {
            'total_items': total_items,
            'valid_items': len(final_scores),
            'failed_items': len(failed_cases),
            'failure_rate': len(failed_cases) / total_items if total_items > 0 else 0,
        },
        'failed_cases': failed_cases
    }


def save_yaml_file(path: Path, data: Dict[str, Any]) -> None:
    """Save data to YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def load_models_from_file(file_path: Path) -> List[str]:
    """Load model names from a text file (one per line, supports comments)."""
    models = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                models.append(line)
    return models


def load_item_ids_from_file(file_path: Path) -> List[int]:
    """Load item IDs from a text file (one per line, supports comments)."""
    item_ids = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    item_ids.append(int(line))
                except ValueError:
                    continue  # Skip invalid lines
    return item_ids


def parse_item_ids(value: Optional[str]) -> Optional[List[int]]:
    """Parse comma-separated item IDs string into list of integers."""
    if not value:
        return None
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError:
        raise ValueError(f"Invalid item IDs format: {value}. Expected comma-separated integers.")


def parse_comma_separated(value: Optional[str]) -> List[str]:
    """Parse comma-separated string into list."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


# =============================================================================
# BatchEvaluator Class
# =============================================================================

class BatchEvaluator:
    """Batch evaluation orchestrator for input data."""
    
    def __init__(
        self,
        project_root: Optional[Path] = None,
        config_template: str = DEFAULT_CONFIG_PATH,
        dry_run: bool = False,
        resume: bool = True,
        max_workers: int = 1,
        max_items: Optional[int] = None,
        item_ids: Optional[List[int]] = None,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        metadata_dir: Optional[str] = None
    ):
        """Initialize batch evaluator.
        
        Args:
            project_root: Root directory of the project (auto-detect if None)
            config_template: Path to config template file
            dry_run: If True, only print what would be done
            resume: If True, skip models that already have completed evaluations
            max_workers: Maximum number of parallel model evaluations
            max_items: Maximum number of items to process per model (None = all)
            input_dir: Input directory for model files
            output_dir: Output directory for results
            metadata_dir: Metadata directory
        """
        # Set project root
        self.root = Path(project_root) if project_root else Path(__file__).parent.parent
        
        # Store settings
        self.config_template = config_template
        self.dry_run = dry_run
        self.resume = resume
        self.max_workers = max_workers
        self.max_items = max_items
        self.item_ids = item_ids
        
        # Load batch config and resolve paths
        batch_config = self._load_batch_config()
        self.input_dir = self.root / (input_dir or batch_config.get('input_dir') or DEFAULT_INPUT_DIR)
        self.output_base = self.root / (output_dir or batch_config.get('output_dir') or DEFAULT_OUTPUT_DIR)
        self.metadata_dir = self.root / (metadata_dir or batch_config.get('metadata_dir') or DEFAULT_METADATA_DIR)
        
        # Thread lock for thread-safe printing
        self._print_lock = threading.Lock()
        
        # Load metadata
        self._models_meta = load_json_file(self.metadata_dir / "models.json")
        self._experiments_meta = load_json_file(self.metadata_dir / "experiments.json")
    
    # -------------------------------------------------------------------------
    # Configuration Loading
    # -------------------------------------------------------------------------
    
    def _load_batch_config(self) -> Dict[str, Any]:
        """Load batch configuration from config file."""
        config = load_yaml_file(self.root / self.config_template)
        return config.get('batch', {})
    
    def _load_full_config(self) -> Dict[str, Any]:
        """Load full configuration from config file."""
        return load_yaml_file(self.root / self.config_template)
    
    # -------------------------------------------------------------------------
    # Thread-safe Printing
    # -------------------------------------------------------------------------
    
    def _print(self, *args, **kwargs) -> None:
        """Thread-safe print."""
        with self._print_lock:
            print(*args, **kwargs)
    
    def _print_header(self, model_name: str, input_file: Optional[Path] = None) -> None:
        """Print evaluation header."""
        self._print(f"\n{'=' * 80}")
        self._print(f"📊 Evaluating: {model_name}")
        if input_file:
            self._print(f"📁 Input: {input_file}")
        self._print(f"{'=' * 80}")
    
    # -------------------------------------------------------------------------
    # Model Discovery
    # -------------------------------------------------------------------------
    
    def get_available_models(self) -> List[str]:
        """Get list of available model input files."""
        if not self.input_dir.exists():
            return []
        return sorted(f.stem for f in self.input_dir.glob("*.json"))
    
    def _resolve_input_file(self, model_name: str) -> Optional[Path]:
        """Resolve input file for a model name (supports exact and fuzzy match)."""
        # Try exact match first
        input_file = self.input_dir / f"{model_name}.json"
        if input_file.exists():
            return input_file
        
        # Try fuzzy match
        matched_files = list(self.input_dir.glob(f"*{model_name}.json"))
        if len(matched_files) == 1:
            self._print(f"💡 Matched '{model_name}' to '{matched_files[0].name}'")
            return matched_files[0]
        
        return None
    
    # -------------------------------------------------------------------------
    # Output Directory Management
    # -------------------------------------------------------------------------
    
    def _find_existing_output_dir(self, model_name: str) -> Optional[Path]:
        """Find existing output directory for a model.
        
        Uses multiple matching strategies:
        1. Exact name match: {model_name}_*
        2. Input file match: Check final_scores.json metadata for matching input_file
        
        Returns the most recent completed output directory, or None.
        """
        if not self.output_base.exists():
            return None
        
        # Strategy 1: Exact name match
        pattern = f"{model_name}_*"
        matching_dirs = sorted(
            (d for d in self.output_base.glob(pattern) if d.is_dir()),
            key=lambda x: x.name,
            reverse=True
        )
        
        for output_dir in matching_dirs:
            if self._is_evaluation_complete(output_dir):
                return output_dir
        
        # Strategy 2: Match by input file path in metadata
        input_file = self._resolve_input_file(model_name)
        if not input_file:
            return None
        
        input_file_abs = input_file.resolve()
        all_dirs = sorted(
            (d for d in self.output_base.iterdir() if d.is_dir()),
            key=lambda x: x.name,
            reverse=True
        )
        
        for output_dir in all_dirs:
            stored_input = self._get_stored_input_file(output_dir)
            if stored_input and stored_input.resolve() == input_file_abs:
                return output_dir
        
        return None
    
    def _get_stored_input_file(self, output_dir: Path) -> Optional[Path]:
        """Get the input file path stored in final_scores.json metadata."""
        final_scores = output_dir / "final_scores.json"
        if not final_scores.exists():
            return None
        
        try:
            data = load_json_file(final_scores)
            stored_path = data.get("metadata", {}).get("input_file", "")
            if not stored_path:
                return None
            
            path = Path(stored_path)
            if not path.is_absolute():
                path = (self.root / stored_path).resolve()
            
            return path if path.exists() else None
        except Exception:
            return None
    
    def _is_evaluation_complete(self, output_dir: Path) -> bool:
        """Check if evaluation is complete by looking for final_scores.json."""
        return (output_dir / "final_scores.json").exists()
    
    # -------------------------------------------------------------------------
    # Temporary Config Creation
    # -------------------------------------------------------------------------
    
    def _create_temp_config(self, input_file: Path, output_dir: Path, model_name: str) -> Path:
        """Create a temporary config file with updated paths."""
        config = self._load_full_config()
        
        # Update evaluation settings
        config['evaluation']['input_path'] = str(input_file)
        config['evaluation']['output_dir'] = str(output_dir)
        config['evaluation']['make_new_folder'] = False
        
        # Disable verbose in parallel mode
        is_parallel = self.max_workers > 1
        config['evaluation']['verbose'] = not is_parallel
        config['agent']['verbose'] = not is_parallel
        
        # Set item_ids if specified (takes precedence over max_items)
        if self.item_ids is not None:
            config['evaluation']['item_ids'] = self.item_ids
        elif self.max_items and self.max_items > 0:
            item_ids = self._get_limited_item_ids(input_file)
            if item_ids:
                config['evaluation']['item_ids'] = item_ids
        
        # Generate unique temp config path
        safe_name = model_name.replace("/", "_").replace(":", "_")
        temp_config = self.output_base / f"temp_config_{safe_name}_{uuid.uuid4().hex[:8]}.yaml"
        save_yaml_file(temp_config, config)
        
        return temp_config
    
    def _get_limited_item_ids(self, input_file: Path) -> Optional[List[int]]:
        """Get limited item IDs from input file."""
        input_data = load_json_file(input_file)
        if not input_data:
            return None
        
        # Handle both list and dict formats
        if isinstance(input_data, list):
            all_ids = sorted(int(item.get('id', i)) for i, item in enumerate(input_data))
        else:
            all_ids = sorted(int(k) for k in input_data.keys())
        
        return all_ids[:self.max_items]
    
    # -------------------------------------------------------------------------
    # Single Model Evaluation
    # -------------------------------------------------------------------------
    
    def run_evaluation(self, model_name: str) -> Dict[str, Any]:
        """Run evaluation for a single model."""
        # Resolve input file
        input_file = self._resolve_input_file(model_name)
        
        # Handle input file not found
        if not input_file or not input_file.exists():
            matched = list(self.input_dir.glob(f"*{model_name}.json"))
            if len(matched) > 1:
                error_msg = f"Ambiguous model name '{model_name}'. Matches: {[f.stem for f in matched]}"
            else:
                error_msg = f"Input file not found: {self.input_dir / f'{model_name}.json'}"
            
            self._print_header(model_name)
            self._print(f"❌ {error_msg}")
            return {"model": model_name, "status": "error", "error": error_msg}
        
        # Check resume mode
        if self.resume:
            existing = self._find_existing_output_dir(model_name)
            if existing and self._is_evaluation_complete(existing):
                return self._handle_skipped_model(model_name, input_file, existing)
        
        # Print header
        self._print_header(model_name, input_file)
        
        # Handle dry run
        if self.dry_run:
            self._print("🔍 [DRY RUN] Would execute evaluation")
            return {"model": model_name, "status": "dry_run", "input_file": str(input_file)}
        
        # Run actual evaluation
        return self._execute_evaluation(model_name, input_file)
    
    def _handle_skipped_model(
        self,
        model_name: str,
        input_file: Path,
        existing_output: Path
    ) -> Dict[str, Any]:
        """Handle a model that should be skipped (already completed)."""
        self._print_header(model_name, input_file)
        self._print(f"⏭️  SKIP - Already completed: {existing_output.name}")
        
        # Load existing statistics
        final_scores = load_json_file(existing_output / "final_scores.json")
        statistics = final_scores.get("statistics")
        
        # Calculate filtered statistics
        scores_dir = existing_output / "scores"
        filtered_result = calculate_filtered_statistics(scores_dir)
        filtered_stats = filtered_result.get('filtered_statistics')
        filtering_info = filtered_result.get('filtering_info', {})
        
        if filtering_info.get('failed_items', 0) > 0:
            self._print(f"   📊 Filtered: {filtering_info['failed_items']} failed cases excluded")
        
        return {
            "model": model_name,
            "status": "skipped",
            "output_dir": str(existing_output),
            "statistics": statistics,
            "statistics_filtered": filtered_stats,
            "filtering_info": filtering_info,
            "error": None
        }
    
    def _execute_evaluation(self, model_name: str, input_file: Path) -> Dict[str, Any]:
        """Execute actual evaluation for a model."""
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.output_base / f"{model_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temp config
        temp_config = self._create_temp_config(input_file, output_dir, model_name)
        
        start_time = datetime.now()
        try:
            pipeline = GenAIEvalPipeline(config_path=str(temp_config))
            statistics = pipeline.run()
            duration = (datetime.now() - start_time).total_seconds()
            
            # Calculate filtered statistics
            scores_dir = output_dir / "scores"
            filtered_result = calculate_filtered_statistics(scores_dir)
            filtered_stats = filtered_result.get('filtered_statistics')
            filtering_info = filtered_result.get('filtering_info', {})
            
            self._print(f"✅ Completed: {model_name} ({duration:.1f}s)")
            if filtering_info.get('failed_items', 0) > 0:
                self._print(f"   📊 Filtered: {filtering_info['failed_items']} failed cases excluded")
            
            return {
                "model": model_name,
                "status": "success",
                "duration_seconds": duration,
                "output_dir": str(output_dir),
                "statistics": statistics,
                "statistics_filtered": filtered_stats,
                "filtering_info": filtering_info,
                "error": None
            }
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            self._print(f"❌ Evaluation failed for {model_name}")
            self._print(f"Error: {error_msg}")
            
            return {
                "model": model_name,
                "status": "error",
                "duration_seconds": duration,
                "output_dir": str(output_dir),
                "error": error_msg
            }
        finally:
            # Clean up temp config
            if temp_config.exists():
                temp_config.unlink()
    
    # -------------------------------------------------------------------------
    # Batch Evaluation
    # -------------------------------------------------------------------------
    
    def run_batch(self, model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run batch evaluation on specified models."""
        if model_names is None:
            model_names = self.get_available_models()
        
        if not model_names:
            self._print("❌ No models found to evaluate")
            return {"results": [], "summary": {}}
        
        # Print batch info
        self._print_batch_info(model_names)
        
        # Run evaluations
        results = self._run_evaluations(model_names)
        
        # Sort results by model name
        results.sort(key=lambda x: x.get("model", ""))
        
        # Generate and save results
        summary = self._generate_summary(results)
        batch_result = self._create_batch_result(model_names, results, summary)
        
        if not self.dry_run:
            self._save_batch_result(batch_result)
        
        return batch_result
    
    def _print_batch_info(self, model_names: List[str]) -> None:
        """Print batch evaluation info."""
        config = self._load_full_config()
        pipeline_workers = config.get('evaluation', {}).get('max_workers', 4)
        agent_concurrency = config.get('agent', {}).get('verify_concurrency', 5)
        
        self._print(f"\n🚀 Starting Batch Evaluation")
        self._print(f"📊 Models to evaluate: {len(model_names)}")
        self._print(f"📂 Output directory: {self.output_base}")
        self._print(f"🔄 Resume mode: {'ON' if self.resume else 'OFF'}")
        
        if self.max_items:
            self._print(f"📝 Max items per model: {self.max_items}")
        
        if self.item_ids:
            ids_str = ','.join(map(str, self.item_ids[:10]))
            if len(self.item_ids) > 10:
                ids_str += f", ... ({len(self.item_ids)} total)"
            self._print(f"🎯 Specific item IDs: {ids_str}")
        
        self._print(f"\n⚙️  Concurrency Settings:")
        self._print(f"   • Outer level (batch): {self.max_workers} parallel models")
        self._print(f"   • Pipeline level: {pipeline_workers} workers (checklist gen, scoring)")
        self._print(f"   • Agent level: {agent_concurrency} concurrent verifications")
        
        if self.max_workers > 1:
            max_per_model = max(pipeline_workers, agent_concurrency)
            theoretical_max = self.max_workers * max_per_model
            self._print(f"\n⚠️  Note: Max concurrent operations per stage:")
            self._print(f"   • Checklist/Scoring: {self.max_workers * pipeline_workers}")
            self._print(f"   • Evidence Verification: {self.max_workers * agent_concurrency}")
            self._print(f"   • Overall max: {theoretical_max} (may cause API rate limits)")
            self._print(f"   Consider reducing --max-workers if you encounter issues.")
            self._print(f"\n💡 Pipeline verbose output disabled in parallel mode")
        
        if self.dry_run:
            self._print("🔍 DRY RUN MODE - No actual evaluation will be performed\n")
    
    def _run_evaluations(self, model_names: List[str]) -> List[Dict[str, Any]]:
        """Run evaluations for all models (parallel or sequential)."""
        if self.max_workers > 1:
            return self._run_parallel(model_names)
        return self._run_sequential(model_names)
    
    def _run_sequential(self, model_names: List[str]) -> List[Dict[str, Any]]:
        """Run evaluations sequentially."""
        return [self.run_evaluation(model) for model in model_names]
    
    def _run_parallel(self, model_names: List[str]) -> List[Dict[str, Any]]:
        """Run evaluations in parallel."""
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_model = {
                executor.submit(self.run_evaluation, model): model
                for model in model_names
            }
            
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    self._print(f"❌ Exception for {model}: {e}")
                    results.append({"model": model, "status": "error", "error": str(e)})
        
        return results
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from batch results."""
        summary = {
            "total": len(results),
            "success": sum(1 for r in results if r["status"] == "success"),
            "skipped": sum(1 for r in results if r["status"] == "skipped"),
            "error": sum(1 for r in results if r["status"] == "error"),
            "dry_run": sum(1 for r in results if r["status"] == "dry_run"),
        }
        
        # Duration statistics
        durations = [r.get("duration_seconds", 0) for r in results if r["status"] == "success"]
        if durations:
            summary["avg_duration_seconds"] = sum(durations) / len(durations)
            summary["total_duration_seconds"] = sum(durations)
        
        # Count analysis reports
        analysis_count = sum(
            1 for r in results
            if r["status"] == "success" and "output_dir" in r
            and os.path.exists(os.path.join(r["output_dir"], "analysis"))
            and any(f.endswith("_analysis.txt") for f in os.listdir(os.path.join(r["output_dir"], "analysis")))
        )
        if analysis_count > 0:
            summary["with_analysis"] = analysis_count
        
        # Aggregate filtering statistics
        total_items = 0
        valid_items = 0
        failed_items = 0
        
        for r in results:
            if r["status"] in ("success", "skipped"):
                info = r.get("filtering_info", {})
                total_items += info.get("total_items", 0)
                valid_items += info.get("valid_items", 0)
                failed_items += info.get("failed_items", 0)
        
        if total_items > 0:
            summary["filtering"] = {
                "total_items": total_items,
                "valid_items": valid_items,
                "failed_items": failed_items,
                "failure_rate": failed_items / total_items
            }
        
        return summary
    
    def _create_batch_result(
        self,
        model_names: List[str],
        results: List[Dict[str, Any]],
        summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create batch result dictionary."""
        return {
            "timestamp": datetime.now().isoformat(),
            "config_template": self.config_template,
            "dry_run": self.dry_run,
            "resume": self.resume,
            "max_workers": self.max_workers,
            "max_items": self.max_items,
            "total_models": len(model_names),
            "results": results,
            "summary": summary
        }
    
    def _save_batch_result(self, batch_result: Dict[str, Any]) -> None:
        """Save batch result to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_file = self.output_base / f"batch_results_{timestamp}.json"
        save_json_file(batch_file, batch_result)
        self._print(f"\n📄 Batch results saved to: {batch_file}")
    
    def print_summary(self, batch_result: Dict[str, Any]) -> None:
        """Print batch evaluation summary."""
        summary = batch_result["summary"]
        
        self._print(f"\n{'=' * 80}")
        self._print(f"📊 Batch Evaluation Summary")
        self._print(f"{'=' * 80}")
        self._print(f"Total models:     {summary['total']}")
        self._print(f"✅ Success:       {summary['success']}")
        
        if summary.get('skipped', 0) > 0:
            self._print(f"⏭️  Skipped:       {summary['skipped']}")
        
        self._print(f"❌ Error:         {summary['error']}")
        
        if "avg_duration_seconds" in summary:
            self._print(f"\n⏱️  Average duration: {summary['avg_duration_seconds']:.1f}s")
            self._print(f"⏱️  Total duration:   {summary['total_duration_seconds']:.1f}s")
        
        if "with_analysis" in summary:
            self._print(f"\n📄 Analysis reports: {summary['with_analysis']} models")
        
        # Print filtering statistics
        if "filtering" in summary:
            filt = summary["filtering"]
            self._print(f"\n🔍 Failed Case Filtering:")
            self._print(f"   Total items:   {filt['total_items']}")
            self._print(f"   Valid items:   {filt['valid_items']}")
            self._print(f"   Failed items:  {filt['failed_items']} ({filt['failure_rate']*100:.1f}%)")
        
        # List failed models
        failed = [r for r in batch_result["results"] if r["status"] == "error"]
        if failed:
            self._print(f"\n❌ Failed models:")
            for r in failed:
                self._print(f"   - {r['model']}")
                if error := r.get("error"):
                    error_msg = error[:200] + "..." if len(error) > 200 else error
                    self._print(f"     Error: {error_msg}")
        
        self._print(f"{'=' * 80}\n")


# =============================================================================
# CLI Argument Parsing
# =============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Batch evaluation for Experiment 1 models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_jade.py
    python scripts/run_jade.py --models "claude-opus-4.5_with_tool,gpt-5.2_with_tool"
    python scripts/run_jade.py --models-file models_to_run.txt
    python scripts/run_jade.py --pattern "*_with_tool"
    python scripts/run_jade.py --exclude "gpt-4o_no_tool"
    python scripts/run_jade.py --max-workers 3
    python scripts/run_jade.py --max-items 5
    python scripts/run_jade.py --item-ids "1,2,3,5,10"
    python scripts/run_jade.py --item-ids-file item_ids.txt
    python scripts/run_jade.py --no-resume
    python scripts/run_jade.py --dry-run

Note: CLI arguments override values in config file (batch section).
Note: --item-ids takes precedence over --max-items if both are specified.
"""
    )
    
    # Model selection arguments
    model_group = parser.add_argument_group("Model Selection")
    model_group.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated list of model names to evaluate"
    )
    model_group.add_argument(
        "--models-file", type=str, default=None,
        help="Path to file containing model names, one per line"
    )
    model_group.add_argument(
        "--pattern", type=str, default=None,
        help="Glob pattern to match model names, e.g., '*_with_tool'"
    )
    model_group.add_argument(
        "--exclude", type=str, default=None,
        help="Comma-separated list of model names to exclude"
    )
    model_group.add_argument(
        "--exclude-pattern", type=str, default=None,
        help="Glob pattern to exclude model names, e.g., '*_no_tool'"
    )
    model_group.add_argument(
        "--list", action="store_true",
        help="List available models and exit"
    )
    
    # Path arguments
    path_group = parser.add_argument_group("Paths")
    path_group.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG_PATH,
        help=f"Path to config template file (default: {DEFAULT_CONFIG_PATH})"
    )
    path_group.add_argument(
        "--root", type=Path, default=None,
        help="Project root directory (default: auto-detect)"
    )
    path_group.add_argument(
        "--input-dir", type=str, default=None,
        help="Input directory for model files"
    )
    path_group.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results"
    )
    path_group.add_argument(
        "--metadata-dir", type=str, default=None,
        help="Metadata directory"
    )
    
    # Execution arguments
    exec_group = parser.add_argument_group("Execution")
    exec_group.add_argument(
        "--max-workers", type=int, default=None,
        help="Maximum number of parallel model evaluations"
    )
    exec_group.add_argument(
        "--max-items", type=int, default=None,
        help="Maximum number of items to process per model (takes first N items)"
    )
    exec_group.add_argument(
        "--item-ids", type=str, default=None,
        help="Comma-separated list of item IDs to process, e.g., '1,2,3,5,10'"
    )
    exec_group.add_argument(
        "--item-ids-file", type=str, default=None,
        help="Path to file containing item IDs (one per line, supports comments)"
    )
    exec_group.add_argument(
        "--resume", action="store_true", default=None,
        help="Enable resume mode, skip completed models"
    )
    exec_group.add_argument(
        "--no-resume", action="store_true", default=None,
        help="Disable resume mode, re-run all models"
    )
    exec_group.add_argument(
        "--dry-run", action="store_true", default=None,
        help="Print what would be done without running"
    )
    
    return parser


def load_batch_config(config_path: str) -> Dict[str, Any]:
    """Load batch configuration from config file."""
    project_root = Path(__file__).parent.parent
    config = load_yaml_file(project_root / config_path)
    return config.get('batch', {})


def merge_config_with_args(args: argparse.Namespace, batch_config: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """Merge config file values with CLI arguments (CLI takes precedence)."""
    
    def get_value(cli_val, config_key, default=None):
        """Get value with CLI override."""
        return cli_val if cli_val is not None else batch_config.get(config_key, default)
    
    # Determine resume mode
    if args.no_resume:
        resume = False
    elif args.resume:
        resume = True
    else:
        resume = batch_config.get('resume', True)
    
    # Parse item_ids
    item_ids = None
    if args.item_ids:
        item_ids = parse_item_ids(args.item_ids)
    elif args.item_ids_file:
        ids_file = Path(args.item_ids_file)
        if not ids_file.is_absolute():
            ids_file = project_root / ids_file
        if not ids_file.exists():
            raise FileNotFoundError(f"Item IDs file not found: {ids_file}")
        item_ids = load_item_ids_from_file(ids_file)
        if not item_ids:
            raise ValueError(f"No valid item IDs found in file: {ids_file}")
    elif batch_config.get('item_ids'):
        # Support item_ids from config file (can be list or comma-separated string)
        config_item_ids = batch_config['item_ids']
        if isinstance(config_item_ids, list):
            item_ids = config_item_ids
        elif isinstance(config_item_ids, str):
            item_ids = parse_item_ids(config_item_ids)
    
    return {
        # Model selection
        'models': get_value(args.models, 'models'),
        'models_file': get_value(args.models_file, 'models_file'),
        'pattern': get_value(args.pattern, 'pattern'),
        'exclude': get_value(args.exclude, 'exclude'),
        'exclude_pattern': get_value(args.exclude_pattern, 'exclude_pattern'),
        # Execution
        'max_workers': get_value(args.max_workers, 'max_workers', 1),
        'max_items': get_value(args.max_items, 'max_items'),
        'item_ids': item_ids,
        'resume': resume,
        'dry_run': get_value(args.dry_run, 'dry_run', False),
        # Paths
        'input_dir': get_value(args.input_dir, 'input_dir'),
        'output_dir': get_value(args.output_dir, 'output_dir'),
        'metadata_dir': get_value(args.metadata_dir, 'metadata_dir'),
    }


def resolve_model_list(
    settings: Dict[str, Any],
    all_models: List[str],
    project_root: Path
) -> tuple[Optional[List[str]], bool]:
    """Resolve model list from settings.
    
    Returns:
        Tuple of (model_list, has_filter)
    """
    models = settings['models']
    models_file = settings['models_file']
    pattern = settings['pattern']
    exclude = settings['exclude']
    exclude_pattern = settings['exclude_pattern']
    
    model_names = None
    has_filter = False
    
    # Priority 1: Explicit model list
    if models:
        model_names = parse_comma_separated(models)
        has_filter = True
    
    # Priority 2: Models file
    elif models_file:
        mf_path = Path(models_file)
        if not mf_path.is_absolute():
            mf_path = project_root / mf_path
        
        if not mf_path.exists():
            raise FileNotFoundError(f"Models file not found: {mf_path}")
        
        model_names = load_models_from_file(mf_path)
        if not model_names:
            raise ValueError(f"No models found in file: {mf_path}")
        
        print(f"📄 Loaded {len(model_names)} models from {mf_path}")
        has_filter = True
    
    # Priority 3: Pattern matching
    elif pattern:
        model_names = [m for m in all_models if fnmatch.fnmatch(m, pattern)]
        if not model_names:
            available = all_models[:5]
            suffix = '...' if len(all_models) > 5 else ''
            raise ValueError(f"No models match pattern: {pattern}\n   Available: {available}{suffix}")
        
        print(f"🔍 Pattern '{pattern}' matched {len(model_names)} models")
        has_filter = True
    
    # Apply exclusions
    working_models = model_names if model_names is not None else all_models.copy()
    
    if exclude:
        exclude_list = parse_comma_separated(exclude)
        original_count = len(working_models)
        working_models = [m for m in working_models if m not in exclude_list]
        excluded = original_count - len(working_models)
        if excluded > 0:
            print(f"🚫 Excluded {excluded} models via exclude list")
        has_filter = True
    
    if exclude_pattern:
        original_count = len(working_models)
        working_models = [m for m in working_models if not fnmatch.fnmatch(m, exclude_pattern)]
        excluded = original_count - len(working_models)
        if excluded > 0:
            print(f"🚫 Excluded {excluded} models via exclude pattern '{exclude_pattern}'")
        has_filter = True
    
    return (working_models if has_filter else None), has_filter


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Determine project root
        project_root = args.root if args.root else Path(__file__).parent.parent
        
        # Load and merge config
        batch_config = load_batch_config(args.config)
        settings = merge_config_with_args(args, batch_config, project_root)
        
        # Create evaluator
        evaluator = BatchEvaluator(
            project_root=args.root,
            config_template=args.config,
            dry_run=settings['dry_run'],
            resume=settings['resume'],
            max_workers=settings['max_workers'],
            max_items=settings['max_items'],
            item_ids=settings['item_ids'],
            input_dir=settings['input_dir'],
            output_dir=settings['output_dir'],
            metadata_dir=settings['metadata_dir']
        )
        
        # Get available models
        all_models = evaluator.get_available_models()
        
        # Handle --list
        if args.list:
            print(f"Available models ({len(all_models)}):")
            for model in all_models:
                print(f"  - {model}")
            return 0
        
        # Resolve model list
        final_models, _ = resolve_model_list(settings, all_models, evaluator.root)
        
        # Run batch evaluation
        batch_result = evaluator.run_batch(final_models)
        evaluator.print_summary(batch_result)
        
        # Return error code if any failures
        return 1 if batch_result["summary"].get("error", 0) > 0 else 0
    
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
