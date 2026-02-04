"""
Base Pipeline for Report Evaluation.

This module provides a clean, abstract base class that eliminates
duplication between EvalPipeline and GenAIEvalPipeline.

Architecture:
    BasePipeline (abstract)
        ├── load_data()           - Common
        ├── generate_checklists() - Common  
        ├── process_item()        - Abstract (override for different agents)
        ├── calculate_statistics()- Common
        └── save_results()        - Common
"""

import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from jade.config import Config, load_config
from jade.checklist import ChecklistGenerator, PromptConfig
from jade.scoring import ScoreGenerator, AnalysisGenerator, ConcisenessScorer
from jade.scoring.statistics import EvalStatistics
from jade.llm import create_llm_client_from_config
from jade.dag_gating import apply_evidence_gating
from jade.utils import (
    load_input_data,
    load_checklist_from_file,
    filter_checklist_by_type,
    convert_to_checklist_items,
)


class BasePipeline(ABC):
    """
    Abstract base class for evaluation pipelines.
    
    Provides common infrastructure for:
    - Configuration loading
    - Data loading and filtering
    - Checklist generation
    - Score calculation
    - Results persistence
    
    Subclasses only need to implement:
    - _create_verification_agent(): Return agent for evidence verification
    - _get_agent_metadata(): Return agent-specific config for output
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize pipeline from configuration file."""
        self.config = self._load_config(config_path)
        self._init_from_config()
        self._init_components()
        
        self.input_data: Dict[str, Any] = {}
        self.all_scores: List[Dict[str, Any]] = []
        
        self._log(f"✅ {self.__class__.__name__} initialized")
    
    # =========================================================================
    # Configuration & Initialization
    # =========================================================================
    
    def _load_config(self, config_path: Optional[str]) -> Config:
        """Load configuration from file."""
        return load_config(config_path)
    
    def _init_from_config(self):
        """Extract config values into instance attributes."""
        cfg = self.config.evaluation
        
        # Core settings
        self.input_path = cfg.input_path
        if not self.input_path:
            raise ValueError("evaluation.input_path is required")
        
        self.max_workers = cfg.max_workers
        self.item_ids = cfg.item_ids
        self.verbose = cfg.verbose
        self.enable_logging = cfg.enable_logging
        self.confidence_threshold = cfg.confidence_threshold
        
        # Scoring weights
        self.reasoning_weight = cfg.reasoning_weight
        self.evidence_weight = cfg.evidence_weight
        self.credibility_weight = getattr(cfg, "credibility_weight", 0.2)
        self.score_fusion_mode = getattr(cfg, "score_fusion_mode", "weighted")
        
        # Multi-label rubrics
        self.multilabel_rubric_dir = getattr(cfg, "multilabel_rubric_dir", "rubrics/bizbench")
        
        # Conciseness
        self.enable_conciseness = getattr(cfg, "enable_conciseness", True)
        self.conciseness_method = getattr(cfg, "conciseness_method", "log")
        self.conciseness_alpha = getattr(cfg, "conciseness_alpha", 0.5)
        
        # Source credibility
        self.enable_source_credibility = getattr(cfg, "enable_source_credibility", True)
        
        # Analysis report generation
        self.enable_analysis = getattr(cfg, "enable_analysis", False)
        self.analysis_language = getattr(cfg, "analysis_language", "EN")
        self.regenerate_analysis = getattr(cfg, "regenerate_analysis", False)
        
        # Checklist prompt settings (ablation experiment)
        self.use_skill = self.config.checklist.use_skill
        self.use_report_specific = self.config.checklist.use_report_specific
        
        # Output directory
        self.output_dir = self._init_output_dir(
            cfg.output_dir, cfg.make_new_folder, cfg.clear_folder
        )
    
    def _init_output_dir(self, base_dir: str, make_new: bool, clear: bool) -> Path:
        """Create and configure output directory."""
        path = Path(base_dir)
        if make_new:
            path = path / datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if clear:
            for sub in ["checklists", "scores", "logs"]:
                shutil.rmtree(path / sub, ignore_errors=True)
        
        for sub in ["checklists", "scores", "logs"]:
            (path / sub).mkdir(parents=True, exist_ok=True)
        
        self._log(f"📁 Output: {path}")
        return path
    
    def _init_components(self):
        """Initialize shared components."""
        self.llm_client = create_llm_client_from_config(self.config)
        
        # Create prompt config for ablation experiments
        prompt_config = PromptConfig(
            use_skill=self.use_skill,
            use_report_specific=self.use_report_specific,
        )
        self._log(f"📝 Prompt config: use_skill={self.use_skill}, use_report_specific={self.use_report_specific}")
        
        self.checklist_generator = ChecklistGenerator(
            llm_client=self.llm_client,
            output_root=str(self.output_dir / "checklists"),
            max_workers=self.max_workers,
            verbose=self.verbose,
            multilabel_rubric_dir=self.multilabel_rubric_dir,
            prompt_config=prompt_config,
        )
        
        self.score_generator = ScoreGenerator(
            llm_client=self.llm_client,
            max_workers=self.max_workers,
            verbose=False,
        )
        
        self.analysis_generator = AnalysisGenerator(
            llm_client=self.llm_client,
            max_workers=self.max_workers,
            verbose=self.verbose,
        ) if self.enable_analysis else None
        
        self.conciseness_scorer = ConcisenessScorer() if self.enable_conciseness else None
    
    def _log(self, msg: str):
        """Print if verbose enabled."""
        if self.verbose:
            print(msg)
    
    # =========================================================================
    # Abstract Methods (override in subclasses)
    # =========================================================================
    
    @abstractmethod
    def _verify_evidence(
        self,
        evidence_checklist: List[Dict[str, Any]],
        item_id: int,
        query: str = "",
    ) -> List[Any]:
        """
        Verify evidence items and return scores.
        
        Override in subclass to use different verification agents.
        """
        pass
    
    @abstractmethod
    def _get_agent_metadata(self) -> Dict[str, Any]:
        """Return agent-specific metadata for output."""
        pass
    
    # =========================================================================
    # Pipeline Steps
    # =========================================================================
    
    def load_data(self):
        """Load and filter input data."""
        self._log(f"\n📂 Loading: {self.input_path}")
        
        self.input_data = load_input_data(self.input_path)
        self._log(f"   Found {len(self.input_data)} items")
        
        if self.item_ids:
            self.input_data = {
                k: v for k, v in self.input_data.items()
                if int(k) in self.item_ids
            }
            self._log(f"   Filtered to {len(self.input_data)} items")
        
        if not self.input_data:
            raise ValueError("No items to process")
    
    def generate_checklists(self):
        """Generate query and report checklists."""
        self._log("\n📝 STEP 1: GENERATING CHECKLISTS")
        self.checklist_generator.generate_from_data(self.input_data, resume=True)
        
        # Save prompt metadata
        self._save_prompt_metadata()
        
        self._log("✅ Checklists generated")
    
    def _save_prompt_metadata(self):
        """Save prompt configuration and templates to output."""
        try:
            prompt_metadata = self.checklist_generator.get_prompt_metadata()
            
            # Save to checklists directory
            metadata_path = self.output_dir / "checklists" / "prompt_metadata.json"
            metadata_path.write_text(
                json.dumps(prompt_metadata, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            self._log(f"   💾 Prompt metadata saved to: {metadata_path}")
        except Exception as e:
            self._log(f"   ⚠️ Failed to save prompt metadata: {e}")
    
    def process_items(self, resume: bool = True):
        """Score reasoning and verify evidence for all items."""
        self._log("\n🚀 STEP 2: PROCESSING ITEMS")
        
        # Get already-processed items for resume
        processed_ids = set()
        if resume:
            scores_dir = self.output_dir / "scores"
            if scores_dir.exists():
                for f in scores_dir.glob("*_scores.json"):
                    try:
                        processed_ids.add(int(f.stem.split("_")[0]))
                    except (ValueError, IndexError):
                        pass
            if processed_ids:
                self._log(f"   📌 Resume: {len(processed_ids)} items already processed")
        
        for item_id_str, item_data in self.input_data.items():
            item_id = int(item_id_str)
            
            # Skip if already processed
            if resume and item_id in processed_ids:
                self._log(f"\n{'─' * 50}\nItem #{item_id} [SKIP - already done]")
                # Load existing scores for statistics
                scores_file = self.output_dir / "scores" / f"{item_id}_scores.json"
                if scores_file.exists():
                    import json
                    self.all_scores.append(json.loads(scores_file.read_text(encoding="utf-8")))
                continue
            
            query = item_data.get("query", "")
            report = item_data.get("report", "")
            
            self._log(f"\n{'─' * 50}\nItem #{item_id}")
            
            # Load checklists
            query_cl, report_cl = self._load_checklists(item_id)
            evidence_cl = filter_checklist_by_type(report_cl, "evidence")
            reasoning_cl = filter_checklist_by_type(report_cl, "reasoning")
            
            # Score query-specific items
            query_scores = self._score_items(query_cl, query, report, prefix="Query")
            
            # Verify evidence
            evidence_scores = self._verify_evidence(evidence_cl, item_id, query)
            
            # Score report reasoning with DAG gating
            report_scores = self._score_items(reasoning_cl, query, report, prefix="Report")
            report_scores = apply_evidence_gating(
                report_reasoning_checklist=reasoning_cl,
                report_reasoning_scores=report_scores,
                evidence_scores=evidence_scores,
                confidence_threshold=float(self.confidence_threshold),
            )
            
            # Calculate combined scores
            reasoning_scores = query_scores + report_scores
            item_scores = self._calculate_item_scores(
                item_id, query, report, reasoning_scores, evidence_scores
            )
            
            # Add metadata
            item_scores["track"] = self._resolve_track(item_data)
            for key in ["task_archetype", "dataset_id"]:
                if key in item_data:
                    item_scores[key] = item_data[key]
            
            # Save
            self._save_item_scores(item_id, item_scores)
            self.all_scores.append(item_scores)
            
            self._log(f"✅ Final: {item_scores['final_score']:.3f}")
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate aggregate statistics."""
        self._log("\n📊 STEP 3: STATISTICS")
        
        stats_config = {
            "score_fusion_mode": self.score_fusion_mode,
            "conciseness_method": self.conciseness_method,
            "conciseness_alpha": self.conciseness_alpha,
            "credibility_weight": self.credibility_weight,
        }
        
        return EvalStatistics(self.all_scores, stats_config).calculate()
    
    def save_results(self, statistics: Dict[str, Any]):
        """Save final results to JSON."""
        # Get prompt config summary (without full prompts to keep file size small)
        prompt_config_summary = {}
        try:
            prompt_metadata = self.checklist_generator.get_prompt_metadata()
            prompt_config_summary = {
                "checklist_prompt_config": prompt_metadata.get("config", {}),
                "prompt_lengths": prompt_metadata.get("prompt_lengths", {}),
            }
        except Exception:
            pass
        
        output = {
            "metadata": {
                "input_file": self.input_path,
                "evaluation_date": datetime.now().isoformat(),
                "total_items": len(self.all_scores),
                **self._get_agent_metadata(),
                **prompt_config_summary,
            },
            "statistics": statistics,
            "scores": self.all_scores,
        }
        
        path = self.output_dir / "final_scores.json"
        path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        self._log(f"\n💾 Saved: {path}")
    
    def print_summary(self, stats: Dict[str, Any]):
        """Print human-readable summary."""
        if not self.verbose:
            return
        
        print(f"\n{'=' * 60}")
        print("  📊 EVALUATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Items: {stats.get('total_items', 0)}")
        print(f"Mode: {stats.get('score_fusion_mode', 'weighted')}")
        
        for dim, name in [("reasoning_score", "D1 Reasoning"), 
                          ("evidence_score", "D2 Evidence"),
                          ("credibility_score", "D3 Credibility")]:
            s = stats.get(dim, {})
            if s:
                print(f"{name}: {s.get('mean', 0):.3f} [{s.get('min', 0):.3f}-{s.get('max', 0):.3f}]")
        
        final = stats.get("final_score", {})
        if final:
            print(f"Final: {final.get('mean', 0):.3f}")
        
        print("=" * 60)
    
    def run(self, resume: bool = True) -> Dict[str, Any]:
        """
        Execute the complete pipeline.
        
        Args:
            resume: If True, skip already-processed items (default: True)
        """
        try:
            self._print_banner()
            self.load_data()
            self.generate_checklists()  # Checklist generation has its own resume
            self.process_items(resume=resume)  # Scoring with resume
            statistics = self.calculate_statistics()
            self.save_results(statistics)
            
            # Generate analysis reports if enabled
            if self.enable_analysis:
                self.generate_analysis_reports()
            
            self.print_summary(statistics)
            return statistics
        except Exception as e:
            self._log(f"\n❌ Pipeline failed: {e}")
            raise
    
    def _print_banner(self):
        """Print startup banner."""
        if not self.verbose:
            return
        print(f"\n{'=' * 60}")
        print(f"  📊 {self.__class__.__name__}")
        print(f"{'=' * 60}")
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _load_checklists(self, item_id: int) -> tuple:
        """
        Load query and report checklists for an item.
        
        Returns:
            Tuple of (query_checklist, report_checklist)
        """
        base = self.output_dir / "checklists"
        
        query_path = base / "query_specific_checklist" / "checklists" / f"{item_id}_query_specific.json"
        report_path = base / "report_specific_checklist" / "report_report_specific_checklist" / f"{item_id}_report_specific.json"
        
        query_cl = load_checklist_from_file(query_path)
        report_cl = load_checklist_from_file(report_path)
        
        return query_cl, report_cl
    
    def _score_items(self, checklist: List[Dict], query: str, report: str, prefix: str = "") -> List[Any]:
        """Score reasoning items."""
        if not checklist:
            return []
        
        self._log(f"   📝 Scoring {len(checklist)} {prefix.lower()} items...")
        
        # Determine checklist_source based on prefix
        checklist_source = "query" if prefix.lower() == "query" else "report"
        
        items = [
            {
                "item_id": item.get("item_id", i),
                "criterion": item.get("criterion") or item.get("description", ""),
                "principle": f"{prefix}:{item.get('principle', item.get('category', 'General'))}",
                "weight": float(item.get("weight", 1.0)),
            }
            for i, item in enumerate(checklist)
        ]
        
        return self.score_generator.score_reasoning_items(
            items, query, report, parallel=True,
            checklist_source=checklist_source,
            original_items=checklist,  # Pass original checklist items for debugging
        )
    
    def _calculate_item_scores(
        self, item_id: int, query: str, report: str,
        reasoning_scores: List, evidence_scores: List,
    ) -> Dict[str, Any]:
        """Calculate weighted scores for a single item."""
        # When report-specific checklist is disabled, don't include evidence in final score
        # Redistribute weight to reasoning only (credibility stays 0 without evidence)
        if self.use_report_specific:
            evidence_weight = self.evidence_weight
            credibility_weight = self.credibility_weight
        else:
            evidence_weight = 0.0
            credibility_weight = 0.0
        
        scores = self.score_generator.calculate_weighted_average(
            reasoning_scores=reasoning_scores,
            evidence_scores=evidence_scores,
            reasoning_weight=self.reasoning_weight,
            evidence_weight=evidence_weight,
            credibility_weight=credibility_weight,
            confidence_threshold=self.confidence_threshold,
            score_fusion_mode=self.score_fusion_mode,
        )
        
        scores["item_id"] = item_id
        scores["query"] = query
        
        # Conciseness
        if self.enable_conciseness and self.conciseness_scorer:
            result = self.conciseness_scorer.calculate_knowledge_density(
                response=report,
                total_score=scores["final_score"],
                method=self.conciseness_method,
                alpha=self.conciseness_alpha if self.conciseness_method == "power" else None,
            )
            scores["conciseness"] = result.to_dict()
        
        return scores
    
    def _save_item_scores(self, item_id: int, scores: Dict[str, Any]):
        """Save individual item scores."""
        path = self.output_dir / "scores" / f"{item_id}_scores.json"
        path.write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
    
    def _resolve_track(self, item_data: Dict[str, Any]) -> str:
        """Resolve track label from item data."""
        for key in ["L1_primary_intent", "track", "task_archetype"]:
            if item_data.get(key):
                return str(item_data[key])
        return "unlabeled"
    
    def generate_analysis_reports(self):
        """Generate analysis reports for all evaluated items."""
        if not self.analysis_generator:
            self._log("⚠️  Analysis generator not initialized")
            return
        
        if not self.all_scores:
            self._log("⚠️  No scores to generate analysis from")
            return
        
        self._log("\n📄 STEP 4: GENERATING ANALYSIS REPORTS")
        
        # Prepare data for analysis generation
        analysis_data = []
        for score_item in self.all_scores:
            item_id = score_item.get("item_id")
            query = score_item.get("query", "")
            
            # Extract evaluation results (remove raw data to keep prompt size manageable)
            eval_results = {
                "item_id": item_id,
                "final_score": score_item.get("final_score"),
                "reasoning_score": score_item.get("reasoning_score"),
                "evidence_score": score_item.get("evidence_score"),
                "credibility_score": score_item.get("credibility_score"),
                "dimension_scores": score_item.get("dimension_scores", {}),
                "reasoning_details": score_item.get("reasoning_details", []),
                "evidence_details": [
                    {
                        "item_id": e.get("item_id"),
                        "criterion": e.get("criterion"),
                        "dimension": e.get("dimension"),
                        "score": e.get("score"),
                        "weight": e.get("weight"),
                        "analysis": e.get("analysis"),
                    }
                    for e in score_item.get("evidence_details", [])
                ],
                "source_credibility_summary": score_item.get("source_credibility_summary", {}),
                "metadata": score_item.get("metadata", {}),
            }
            
            analysis_data.append({
                "item_id": str(item_id),
                "query": query,
                "evaluation_results": eval_results,
            })
        
        # Generate reports using AnalysisGenerator
        stats = self.analysis_generator.generate_analysis_reports(
            scores_data=analysis_data,
            output_dir=self.output_dir,
            language=self.analysis_language,
            regenerate=self.regenerate_analysis,
            parallel=True,
        )
        
        self._log(f"✅ Analysis generation complete: {stats['success']}/{stats['total_items']} successful")

