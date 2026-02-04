"""
Analysis Report Generator for Evaluation Results.

Generates comprehensive analysis reports for evaluated items,
including overall assessment, strengths, weaknesses, and recommendations.
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from jade.llm import BaseLLMClient
from jade.prompts import ANALYSIS_REPORT_PROMPT_EN, ANALYSIS_REPORT_PROMPT_ZH


class AnalysisGenerator:
    """Generator for analysis reports based on evaluation results."""
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        max_workers: int = 4,
        verbose: bool = True,
    ):
        """
        Initialize analysis generator.
        
        Args:
            llm_client: LLM client for generating analysis
            max_workers: Number of parallel workers
            verbose: Whether to print progress messages
        """
        self.llm_client = llm_client
        self.max_workers = max_workers
        self.verbose = verbose
    
    def _log(self, msg: str):
        """Print message if verbose enabled."""
        if self.verbose:
            print(msg)
    
    def _parse_analysis_response(self, response: str) -> Tuple[str, Optional[str]]:
        """
        Parse analysis report response.
        
        Returns:
            Tuple of (analysis_content, verdict)
        """
        analysis_content = ""
        verdict = None
        
        # Try to match <case_review>
        pattern_case_review = r'<case_review>(.*?)</case_review>'
        match = re.search(pattern_case_review, response, re.DOTALL)
        if match:
            analysis_content = match.group(1).strip()
        else:
            # Fallback: try <report> tag for backward compatibility
            pattern_report = r'<report>(.*?)</report>'
            match = re.search(pattern_report, response, re.DOTALL)
            if match:
                analysis_content = match.group(1).strip()
        
        # Extract <verdict>
        pattern_verdict = r'<verdict>(.*?)</verdict>'
        match_verdict = re.search(pattern_verdict, response, re.DOTALL)
        if match_verdict:
            verdict = match_verdict.group(1).strip()
        
        # Fallback to full response if no tags found
        if not analysis_content:
            analysis_content = response.strip() if response.strip() else "Analysis generation failed."
        
        return analysis_content, verdict
    
    def _generate_single_analysis(
        self,
        item_id: str,
        query: str,
        evaluation_results: Dict[str, Any],
        language: str = "EN",
    ) -> Dict[str, Any]:
        """
        Generate analysis report for a single item.
        
        Args:
            item_id: Unique identifier for the item
            query: The original query
            evaluation_results: Dictionary containing evaluation scores and details
            language: "EN" or "ZH" for English or Chinese
        
        Returns:
            Dictionary with analysis_report, verdict, and success status
        """
        # Select prompt based on language
        prompt_template = ANALYSIS_REPORT_PROMPT_ZH if language == "ZH" else ANALYSIS_REPORT_PROMPT_EN
        
        # Format evaluation results for prompt
        results_json = json.dumps(evaluation_results, indent=2, ensure_ascii=False)
        
        prompt = prompt_template.format(
            query=query,
            evaluation_results=results_json,
        )
        
        # Call LLM with retry logic
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat_completion([{"role": "user", "content": prompt}])
                analysis_content, verdict = self._parse_analysis_response(response)
                
                return {
                    "item_id": item_id,
                    "analysis_report": analysis_content,
                    "verdict": verdict,
                    "success": True,
                }
            except Exception as e:
                if attempt < max_retries - 1:
                    self._log(f"Analysis generation attempt {attempt + 1} failed: {e}, retrying...")
                else:
                    self._log(f"Analysis generation failed after {max_retries} attempts: {e}")
                    return {
                        "item_id": item_id,
                        "analysis_report": f"Analysis generation failed: {str(e)}",
                        "verdict": None,
                        "success": False,
                    }
        
        return {
            "item_id": item_id,
            "analysis_report": "Analysis generation failed.",
            "verdict": None,
            "success": False,
        }
    
    def generate_analysis_reports(
        self,
        scores_data: List[Dict[str, Any]],
        output_dir: Path,
        language: str = "EN",
        regenerate: bool = False,
        parallel: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate analysis reports for evaluation results.
        
        Args:
            scores_data: List of score dictionaries with keys: item_id, query, evaluation_results
            output_dir: Directory to save analysis reports (will save to {output_dir}/analysis/)
            language: "EN" or "ZH" for English or Chinese
            regenerate: If True, regenerate existing reports; if False, skip valid existing reports
            parallel: Whether to use parallel processing
        
        Returns:
            Dictionary with generation statistics:
            - total_items: Total number of items
            - generated: Number of reports generated in this run
            - skipped: Number of existing valid reports skipped
            - success: Number of successfully generated reports
            - failed: Number of failed generations
        """
        self._log(f"\n{'=' * 60}")
        self._log(f"  Generating Analysis Reports ({language})")
        self._log(f"{'=' * 60}")
        
        analysis_dir = output_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter items that need generation
        tasks = []
        for item_data in scores_data:
            item_id = str(item_data.get("item_id", "unknown"))
            analysis_path = analysis_dir / f"{item_id}_analysis.txt"
            
            # Check if we should skip this item
            skip = False
            if not regenerate and analysis_path.exists():
                try:
                    content = analysis_path.read_text(encoding="utf-8").strip()
                    # Valid if not empty and not a failure message
                    if content and "generation failed" not in content.lower() and len(content) > 10:
                        skip = True
                except Exception:
                    pass
            
            if not skip:
                tasks.append(item_data)
        
        if not tasks:
            self._log(f"✓ All {len(scores_data)} analysis reports are valid, skipping generation")
            return {
                "total_items": len(scores_data),
                "generated": 0,
                "skipped": len(scores_data),
                "success": len(scores_data),
                "failed": 0,
            }
        
        self._log(f"Found {len(tasks)} items to generate (out of {len(scores_data)} total)")
        self._log(f"Generating analysis reports...")
        
        # Generate reports (parallel or sequential)
        results = []
        if parallel and self.max_workers > 1 and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._generate_single_analysis,
                        item_data.get("item_id", "unknown"),
                        item_data.get("query", ""),
                        item_data.get("evaluation_results", {}),
                        language,
                    ): item_data
                    for item_data in tasks
                }
                
                with tqdm(total=len(futures), desc="Generating analysis") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
        else:
            for item_data in tqdm(tasks, desc="Generating analysis"):
                result = self._generate_single_analysis(
                    item_data.get("item_id", "unknown"),
                    item_data.get("query", ""),
                    item_data.get("evaluation_results", {}),
                    language,
                )
                results.append(result)
        
        # Save results to files
        success_count = 0
        failed_count = 0
        for result in results:
            item_id = result["item_id"]
            analysis_path = analysis_dir / f"{item_id}_analysis.txt"
            verdict_path = analysis_dir / f"{item_id}_verdict.json"
            
            try:
                # Save analysis text
                analysis_path.write_text(result["analysis_report"], encoding="utf-8")
                
                # Save verdict JSON if available
                if result.get("verdict"):
                    verdict_data = {
                        "item_id": item_id,
                        "verdict": result["verdict"],
                    }
                    verdict_path.write_text(json.dumps(verdict_data, ensure_ascii=False, indent=2), encoding="utf-8")
                
                if result.get("success", False):
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                self._log(f"Failed to save analysis for item {item_id}: {e}")
                failed_count += 1
        
        self._log(f"\n{'=' * 60}")
        self._log(f"✓ Successfully generated {success_count} analysis reports")
        if failed_count > 0:
            self._log(f"✗ Failed to generate {failed_count} analysis reports")
        self._log(f"Analysis reports saved to: {analysis_dir}")
        self._log(f"{'=' * 60}\n")
        
        return {
            "total_items": len(scores_data),
            "generated": len(tasks),
            "skipped": len(scores_data) - len(tasks),
            "success": success_count,
            "failed": failed_count,
        }

