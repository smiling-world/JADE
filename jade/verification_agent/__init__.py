"""
Verification Agent module for claim verification.

This module provides verification agent implementations:
- ReAct Agent: LangGraph-based agent with custom tools
- GenAI Agent: Native GenAI tools (GoogleSearch + UrlContext)

Use create_verification_agent() to create an agent based on configuration.

Example:
    >>> from jade.verification_agent import create_verification_agent, ChecklistItem
    >>> 
    >>> # Create agent based on config (default: react)
    >>> agent = create_verification_agent(agent_type="genai")
    >>> 
    >>> # Verify a claim
    >>> item = ChecklistItem(id=1, description="Verify this...")
    >>> result = agent.verify_item(item)
"""

from typing import Optional, Union

from .state import AgentState, ChecklistItem, VerificationResult, ReasonDetail, ReferenceUrls
from .react_agent import VerificationAgent, create_agent
from .genai_agent import GenAIVerificationAgent, create_genai_agent


def create_verification_agent(
    agent_type: str = "react",
    # Common options
    verbose: bool = True,
    log_dir: str = "logs",
    enable_logging: bool = True,
    concurrency: int = 1,
    # ReAct agent options
    client_type: str = "custom",
    model_name: Optional[str] = None,
    max_iterations: int = 15,
    # GenAI agent options
    genai_model_name: str = "gemini-2.5-flash",
    genai_api_key: Optional[str] = None,
    genai_base_url: Optional[str] = None,
    genai_api_version: str = "v1",
    genai_headers: Optional[dict] = None,
    use_search: bool = True,
    use_url_context: bool = True,
    **kwargs
) -> Union[VerificationAgent, GenAIVerificationAgent]:
    """
    Factory function to create a verification agent based on type.
    
    Args:
        agent_type: Agent type - "react" or "genai"
        verbose: Enable console output
        log_dir: Directory for log files
        enable_logging: Enable file logging
        concurrency: Number of parallel verification tasks
        
        # ReAct agent options
        client_type: LLM client type for ReAct agent
        model_name: Model name for ReAct agent
        max_iterations: Maximum iterations for ReAct agent
        
        # GenAI agent options
        genai_model_name: Model name for GenAI agent
        genai_api_key: API key for GenAI
        genai_base_url: Base URL for GenAI API
        genai_api_version: API version for GenAI
        genai_headers: Custom headers for GenAI
        use_search: Enable GoogleSearch tool (GenAI only)
        use_url_context: Enable UrlContext tool (GenAI only)
        
        **kwargs: Additional configuration
        
    Returns:
        Configured verification agent instance
    """
    if agent_type.lower() == "genai":
        return create_genai_agent(
            model_name=genai_model_name,
            api_key=genai_api_key,
            base_url=genai_base_url,
            api_version=genai_api_version,
            headers=genai_headers,
            verbose=verbose,
            log_dir=log_dir,
            enable_logging=enable_logging,
            concurrency=concurrency,
            use_search=use_search,
            use_url_context=use_url_context,
            **kwargs
        )
    else:
        # Default to ReAct agent
        return create_agent(
            client_type=client_type,
            model_name=model_name,
            max_iterations=max_iterations,
            verbose=verbose,
            log_dir=log_dir,
            enable_logging=enable_logging,
            concurrency=concurrency,
            **kwargs
        )


__all__ = [
    # State classes
    "AgentState",
    "ChecklistItem", 
    "VerificationResult",
    "ReasonDetail",
    "ReferenceUrls",
    # ReAct agent
    "VerificationAgent",
    "create_agent",
    # GenAI agent
    "GenAIVerificationAgent",
    "create_genai_agent",
    # Unified factory
    "create_verification_agent",
]
