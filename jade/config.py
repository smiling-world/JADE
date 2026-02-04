"""
Configuration Management for the Jade Agent.

This module provides configuration classes and utilities for managing
agent settings through YAML files or environment variables.

Example:
    >>> config = load_config("config.yaml")
    >>> config = Config.from_env()
"""

import json
import os
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

import yaml

# Pattern for env var substitution in YAML: ${VAR_NAME} or ${VAR_NAME:-default}
_ENV_REF_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}$")


def _expand_env_in_value(value: Any) -> Any:
    """Recursively expand ${VAR_NAME} or ${VAR_NAME:-default} in strings. Dict/list passed through."""
    if isinstance(value, str):
        m = _ENV_REF_PATTERN.fullmatch(value.strip())
        if m:
            var_name, default = m.group(1), (m.group(2) or "")
            return os.environ.get(var_name, default)
        return value
    if isinstance(value, dict):
        return {k: _expand_env_in_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_in_value(v) for v in value]
    return value


@dataclass
class LLMConfig:
    """
    LLM client configuration.
    
    Supports four client types:
    - openai: Standard OpenAI API
    - azure: Azure OpenAI Service
    - custom: Custom Azure-compatible endpoints
    - genai: Google GenAI API
    """
    client_type: str = "openai"  # openai, azure, custom, genai
    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    max_retries: int = 3
    
    # OpenAI specific (env: OPENAI_API_KEY, OPENAI_BASE_URL)
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    
    # Azure specific (env: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT)
    azure_api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_api_version: str = "2024-02-15-preview"
    
    # Custom client specific (env: CUSTOM_API_KEY, CUSTOM_ENDPOINT)
    custom_api_key: Optional[str] = None
    custom_endpoint: Optional[str] = None
    custom_api_version: str = "2024-02-15-preview"
    custom_headers: Dict[str, str] = field(default_factory=dict)
    supports_temperature: bool = True
    supports_max_tokens: bool = True
    
    # GenAI client specific (env: GENAI_API_KEY, GENAI_BASE_URL)
    genai_api_key: Optional[str] = None
    genai_base_url: Optional[str] = None
    genai_api_version: str = "v1"
    genai_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class ToolsConfig:
    """
    Tools configuration.
    
    API keys should be set via environment variables:
    - SERPAPI_KEY: For SerpApi web search
    - JINA_API_KEY: For web scraping
    """
    # SerpApi search tool (env: SERPAPI_KEY)
    serpapi_key: Optional[str] = None
    
    # Scraper tool (env: JINA_API_KEY)
    jina_api_key: Optional[str] = None
    max_content_length: int = 150000
    scraper_timeout: int = 60
    scraper_max_concurrency: int = 1  # Concurrent scraping requests
    scraper_cache_dir: Optional[str] = None  # Cache directory for scraped URLs


@dataclass
class AgentConfig:
    """Agent behavior configuration."""
    # Agent type: "react" (LangGraph ReAct) or "genai" (GenAI native tools)
    agent_type: str = "react"
    max_iterations: int = 15
    verbose: bool = True
    log_dir: str = "logs"
    output_dir: str = "output"
    enable_logging: bool = True
    verify_concurrency: int = 1  # Concurrent verification tasks
    
    # GenAI agent specific options
    genai_use_search: bool = True  # Enable GoogleSearch tool
    genai_use_url_context: bool = True  # Enable UrlContext tool


@dataclass
class GenAIConfig:
    """
    GenAI agent specific configuration.
    
    This is separate from LLMConfig because GenAI agent uses
    the native google-genai SDK directly with built-in tools.
    """
    # Model name (gemini-2.5-flash, gemini-3-flash-preview, etc.)
    model_name: str = "gemini-2.5-flash"
    
    # API credentials (env: GENAI_API_KEY, GENAI_BASE_URL)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: str = "v1"
    # Custom headers (e.g. from env GENAI_CUSTOM_HEADERS as JSON, or dict in YAML)
    custom_headers: Dict[str, str] = field(default_factory=dict)
    
    # Generation parameters
    temperature: float = 0.3
    max_retries: int = 3


@dataclass
class EvaluationConfig:
    """Evaluation pipeline configuration."""
    input_path: Optional[str] = None
    output_dir: str = "output"
    max_workers: int = 4
    max_iterations: int = 15
    item_ids: Optional[List[int]] = None
    verbose: bool = True
    enable_logging: bool = True
    make_new_folder: bool = False
    clear_folder: bool = False
    confidence_threshold: float = 0.7  # Raised from 0.5 for stricter filtering
    # Multi-label skill composition rubric directory
    # Contains hierarchical rubrics: L1_intent/, L2_information_need/, L3_constraints/
    multilabel_rubric_dir: str = "rubrics/bizbench"
    
    # Conciseness (knowledge density) settings
    # Enable knowledge density calculation (score per token)
    enable_conciseness: bool = True
    # Method: "linear" (1/x), "log" (1/log(x)), "power" (1/x^alpha)
    conciseness_method: str = "log"
    # Alpha for power method (0.3-0.7 recommended, lower = less sensitive)
    conciseness_alpha: float = 0.5
    
    # ==========================================================================
    # Three-Dimensional Scoring Framework
    # ==========================================================================
    # D1: Reasoning Score (推理质量)
    # D2: Evidence Score (事实准确性)
    # D3: Credibility Score (来源可信度)
    
    # Enable source credibility as independent dimension (D3)
    enable_source_credibility: bool = True
    
    # Final score fusion mode:
    # - "independent": Report D1, D2, D3 separately (no fusion)
    # - "weighted": Final = w1*D1 + w2*D2 + w3*D3
    # - "two_stage": Final = w1*D1 + w2*D2, D3 reported separately
    score_fusion_mode: str = "weighted"
    
    # Weights for fusion (only used when score_fusion_mode != "independent")
    reasoning_weight: float = 0.4   # D1 weight
    evidence_weight: float = 0.4    # D2 weight  
    credibility_weight: float = 0.2 # D3 weight (only for "weighted" mode)
    
    # ==========================================================================
    # Analysis Report Generation
    # ==========================================================================
    # Enable automatic generation of analysis reports for each evaluated item
    enable_analysis: bool = False
    # Language for analysis reports: "EN" (English) or "ZH" (Chinese)
    analysis_language: str = "EN"
    # Regenerate existing valid analysis reports (false = skip existing)
    regenerate_analysis: bool = False


@dataclass
class ChecklistConfig:
    """
    Checklist generation configuration for ablation experiments.
    
    Controls which prompt variants to use:
    - FULL: use_skill=True, use_report_specific=True (default)
    - WITH_SKILL: use_skill=True, use_report_specific=False
    - WITH_REPORT: use_skill=False, use_report_specific=True
    - BASELINE: use_skill=False, use_report_specific=False
    """
    # Enable multi-label skill composition in query checklist
    use_skill: bool = True
    # Enable report-specific checklist generation  
    use_report_specific: bool = True


@dataclass
class Config:
    """
    Main configuration container.
    
    Can be loaded from:
    - YAML file: Config.from_yaml("config.yaml")
    - Environment variables: Config.from_env()
    - Dictionary: Config.from_dict(data)
    
    Configuration structure:
    - llm: LLM settings for ReAct Agent's reasoning
    - genai: GenAI-specific settings for GenAI Agent
    - tools: External tool settings (SerpApi, Jina) for ReAct Agent
    - agent: Agent behavior settings
    - checklist: Checklist prompt settings for ablation experiments
    """
    llm: LLMConfig = field(default_factory=LLMConfig)
    genai: GenAIConfig = field(default_factory=GenAIConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    checklist: ChecklistConfig = field(default_factory=ChecklistConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file.
        String values in the form ${VAR_NAME} or ${VAR_NAME:-default} are replaced
        with the corresponding environment variable (sensitive data should use this).
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        data = _expand_env_in_value(data)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        llm_data = dict(data.get("llm", {}))
        genai_data = dict(data.get("genai", {}))
        tools_data = data.get("tools", {})
        agent_data = data.get("agent", {})
        eval_data = data.get("evaluation", {})
        checklist_data = data.get("checklist", {})
        
        def _parse_custom_headers(v: Any) -> Dict[str, str]:
            if isinstance(v, dict):
                return v
            if isinstance(v, str) and v.strip().startswith("{"):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    return {}
            return {}
        
        # LLM: custom_headers may be JSON string from env (e.g. ${LLM_CUSTOM_HEADERS})
        if llm_data and "custom_headers" in llm_data:
            llm_data["custom_headers"] = _parse_custom_headers(llm_data["custom_headers"])
        
        # GenAI: normalize custom_headers (accept "headers" as legacy; allow JSON string from env)
        if genai_data:
            ch = genai_data.pop("headers", None)
            if ch is not None and "custom_headers" not in genai_data:
                genai_data["custom_headers"] = ch
            genai_data["custom_headers"] = _parse_custom_headers(genai_data.get("custom_headers"))
        
        return cls(
            llm=LLMConfig(**llm_data) if llm_data else LLMConfig(),
            genai=GenAIConfig(**genai_data) if genai_data else GenAIConfig(),
            tools=ToolsConfig(**tools_data) if tools_data else ToolsConfig(),
            agent=AgentConfig(**agent_data) if agent_data else AgentConfig(),
            evaluation=EvaluationConfig(**eval_data) if eval_data else EvaluationConfig(),
            checklist=ChecklistConfig(**checklist_data) if checklist_data else ChecklistConfig()
        )
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        return cls(
            llm=LLMConfig(
                client_type=os.environ.get("LLM_CLIENT_TYPE", "openai"),
                model_name=os.environ.get("MODEL_NAME", "gpt-4o"),
                temperature=float(os.environ.get("LLM_TEMPERATURE", "0.7")),
                max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "4096")),
                openai_api_key=os.environ.get("OPENAI_API_KEY"),
                openai_base_url=os.environ.get("OPENAI_BASE_URL"),
                azure_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                custom_api_key=os.environ.get("CUSTOM_API_KEY"),
                custom_endpoint=os.environ.get("CUSTOM_ENDPOINT"),
            ),
            genai=GenAIConfig(
                model_name=os.environ.get("GENAI_MODEL_NAME", "gemini-2.5-flash"),
                api_key=os.environ.get("GENAI_API_KEY"),
                base_url=os.environ.get("GENAI_BASE_URL"),
                api_version=os.environ.get("GENAI_API_VERSION", "v1"),
                temperature=float(os.environ.get("GENAI_TEMPERATURE", "0.3")),
            ),
            tools=ToolsConfig(
                serpapi_key=os.environ.get("SERPAPI_KEY"),
                jina_api_key=os.environ.get("JINA_API_KEY"),
            ),
            agent=AgentConfig(
                agent_type=os.environ.get("AGENT_TYPE", "react"),
                max_iterations=int(os.environ.get("AGENT_MAX_ITERATIONS", "15")),
                verbose=os.environ.get("AGENT_VERBOSE", "true").lower() == "true",
                genai_use_search=os.environ.get("GENAI_USE_SEARCH", "true").lower() == "true",
                genai_use_url_context=os.environ.get("GENAI_USE_URL_CONTEXT", "true").lower() == "true",
            )
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excludes sensitive data)."""
        return {
            "llm": {
                "client_type": self.llm.client_type,
                "model_name": self.llm.model_name,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
            },
            "tools": {
                "max_content_length": self.tools.max_content_length,
                "scraper_timeout": self.tools.scraper_timeout,
            },
            "agent": {
                "max_iterations": self.agent.max_iterations,
                "verbose": self.agent.verbose,
            },
            "evaluation": {
                "input_path": self.evaluation.input_path,
                "output_dir": self.evaluation.output_dir,
                "reasoning_weight": self.evaluation.reasoning_weight,
                "evidence_weight": self.evaluation.evidence_weight,
                "confidence_threshold": self.evaluation.confidence_threshold,
                # Conciseness settings
                "enable_conciseness": self.evaluation.enable_conciseness,
                "conciseness_method": self.evaluation.conciseness_method,
                "conciseness_alpha": self.evaluation.conciseness_alpha,
                # Source credibility settings
                "enable_source_credibility": self.evaluation.enable_source_credibility,
                "credibility_weight": self.evaluation.credibility_weight,
                # Analysis report settings
                "enable_analysis": self.evaluation.enable_analysis,
                "analysis_language": self.evaluation.analysis_language,
                "regenerate_analysis": self.evaluation.regenerate_analysis,
            },
            "checklist": {
                "use_skill": self.checklist.use_skill,
                "use_report_specific": self.checklist.use_report_specific,
            }
        }


def load_config(
    config_path: Optional[str] = None,
    use_env: bool = True
) -> Config:
    """
    Load configuration from file or environment.
    
    Priority: config_path > environment variables > defaults
    
    Args:
        config_path: Path to YAML config file
        use_env: Whether to fall back to environment variables
        
    Returns:
        Loaded configuration
    """
    if config_path and os.path.exists(config_path):
        return Config.from_yaml(config_path)
    elif use_env:
        return Config.from_env()
    else:
        return Config()
