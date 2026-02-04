#!/usr/bin/env python3
"""
GenAI Evaluation Pipeline for Report Verification.

This pipeline uses Google GenAI's native tools:
- GoogleSearch for web search
- UrlContext for URL content extraction

Key advantages:
- No external API keys needed (no SerpApi, no Jina)
- Higher concurrency support
- Simplified configuration

Usage:
    python eval_pipeline_genai.py
    python eval_pipeline_genai.py --config config.genai.yaml
"""

import argparse
from datetime import datetime
from typing import Dict, List, Any

from jade.pipeline.base import BasePipeline
from jade.verification_agent import GenAIVerificationAgent, create_genai_agent
from jade.utils import convert_to_checklist_items


class GenAIEvalPipeline(BasePipeline):
    """
    Evaluation pipeline using GenAI's native search and URL tools.
    
    Example:
        >>> pipeline = GenAIEvalPipeline(config_path="config.genai.yaml")
        >>> results = pipeline.run()
    """
    
    def __init__(self, config_path: str = None):
        super().__init__(config_path)
        
        # GenAI-specific settings
        agent_cfg = self.config.agent
        self.genai_concurrency = agent_cfg.verify_concurrency
        self.genai_use_search = agent_cfg.genai_use_search
        self.genai_use_url_context = agent_cfg.genai_use_url_context
        
        self._log(f"   Concurrency: {self.genai_concurrency}")
        self._log(f"   Tools: search={self.genai_use_search}, url={self.genai_use_url_context}")
    
    def _create_agent(self, session_id: str) -> GenAIVerificationAgent:
        """Create GenAI verification agent."""
        genai_cfg = self.config.genai
        
        return create_genai_agent(
            model_name=genai_cfg.model_name,
            api_key=genai_cfg.api_key,
            base_url=genai_cfg.base_url,
            api_version=genai_cfg.api_version,
            headers=genai_cfg.custom_headers,
            temperature=genai_cfg.temperature,
            max_retries=genai_cfg.max_retries,
            verbose=self.verbose,
            log_dir=str(self.output_dir / "logs"),
            enable_logging=self.enable_logging,
            session_id=session_id,
            concurrency=self.genai_concurrency,
            use_search=self.genai_use_search,
            use_url_context=self.genai_use_url_context,
        )
    
    def _verify_evidence(
        self,
        evidence_checklist: List[Dict[str, Any]],
        item_id: int,
        query: str = "",
    ) -> List[Any]:
        """Verify evidence using GenAI native tools."""
        if not evidence_checklist:
            return []
        
        self._log(f"   🔍 Verifying {len(evidence_checklist)} evidence items (GenAI)...")
        
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{item_id}"
        agent = self._create_agent(session_id)
        
        items = convert_to_checklist_items(evidence_checklist)
        results = agent.verify_checklist(items)
        
        verification_dicts = [
            {
                "item_id": r.item_id,
                "conclusion": r.conclusion,
                "confidence": r.confidence,
                "reason": {
                    "summary": r.reason.summary if hasattr(r.reason, 'summary') else str(r.reason),
                    "supporting": getattr(r.reason, 'supporting', []),
                    "contradicting": getattr(r.reason, 'contradicting', []),
                },
                "reference_urls": {
                    "supporting": getattr(r.reference_urls, 'supporting', []),
                    "contradicting": getattr(r.reference_urls, 'contradicting', []),
                }
            }
            for r in results
        ]
        
        return self.score_generator.score_evidence_items(
            checklist_items=evidence_checklist,
            verification_results=verification_dicts,
            enable_source_credibility=self.enable_source_credibility,
        )
    
    def _get_agent_metadata(self) -> Dict[str, Any]:
        """Return pipeline configuration for output."""
        genai_cfg = self.config.genai
        
        return {
            "agent_type": "genai",
            "config": {
                "genai_model": genai_cfg.model_name,
                "llm_model": self.config.llm.model_name,
                "client_type": self.config.llm.client_type,
                "concurrency": self.genai_concurrency,
                "use_search": self.genai_use_search,
                "use_url_context": self.genai_use_url_context,
                "multilabel_rubric_dir": self.multilabel_rubric_dir,
                "scoring": {
                    "fusion_mode": self.score_fusion_mode,
                    "weights": {
                        "reasoning": self.reasoning_weight,
                        "evidence": self.evidence_weight,
                        "credibility": self.credibility_weight,
                    },
                    "confidence_threshold": self.confidence_threshold,
                },
                "conciseness": {
                    "enabled": self.enable_conciseness,
                    "method": self.conciseness_method,
                    "alpha": self.conciseness_alpha,
                },
            }
        }
    
    def _print_banner(self):
        """Print GenAI-specific banner."""
        if not self.verbose:
            return
        print(f"\n{'=' * 60}")
        print("  📊 GenAI Evaluation Pipeline")
        print("  Native Google Search + UrlContext")
        print(f"{'=' * 60}")
        print(f"🤖 GenAI: {self.config.genai.model_name}")
        print(f"🔧 LLM: {self.config.llm.model_name}")
        print(f"⚡ Concurrency: {self.genai_concurrency}")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate reports using GenAI native tools.")
    parser.add_argument("-c", "--config", help="Configuration YAML file (default: config.genai.yaml)")
    args = parser.parse_args()
    
    try:
        GenAIEvalPipeline(config_path=args.config).run()
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

