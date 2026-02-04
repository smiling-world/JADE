"""
LLM Client Factory.

Centralized factory for creating LLM clients based on configuration.
"""

from typing import Optional, Dict, Any

from jade.llm.base import BaseLLMClient
from jade.llm.openai_client import OpenAIClient
from jade.llm.azure_client import AzureOpenAIClient
from jade.llm.custom_client import CustomLLMClient
from jade.llm.genai_client import GenAIClient


def create_llm_client(
    client_type: str,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    max_retries: int = 3,
    # OpenAI params
    openai_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    # Azure params
    azure_api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    azure_api_version: str = "2024-02-15-preview",
    # Custom params
    custom_api_key: Optional[str] = None,
    custom_endpoint: Optional[str] = None,
    custom_api_version: str = "2024-02-15-preview",
    custom_headers: Optional[Dict[str, str]] = None,
    supports_temperature: bool = True,
    supports_max_tokens: bool = True,
    # GenAI params
    genai_api_key: Optional[str] = None,
    genai_base_url: Optional[str] = None,
    genai_api_version: str = "v1",
    genai_headers: Optional[Dict[str, str]] = None,
    # Legacy param names (for backwards compatibility)
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> BaseLLMClient:
    """
    Create an LLM client based on configuration.
    
    Args:
        client_type: Type of client ("openai", "azure", "custom", "genai")
        model_name: Model name/deployment
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        max_retries: Maximum retries on error
        
        # Client-specific params use prefixed names (e.g., genai_api_key)
        # Legacy params (api_key, base_url) also supported for backwards compatibility
        
    Returns:
        Configured LLM client instance
    """
    if client_type == "openai":
        return OpenAIClient(
            model_name=model_name or "gpt-4o",
            api_key=openai_api_key or api_key,
            base_url=openai_base_url or base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
    
    elif client_type == "azure":
        return AzureOpenAIClient(
            model_name=model_name or "gpt-4o",
            api_key=azure_api_key or api_key,
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
    
    elif client_type == "custom":
        return CustomLLMClient(
            model_name=model_name,
            api_key=custom_api_key or api_key,
            azure_endpoint=custom_endpoint,
            api_version=custom_api_version,
            default_headers=custom_headers,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            supports_temperature=supports_temperature,
            supports_max_tokens=supports_max_tokens,
        )
    
    elif client_type == "genai":
        return GenAIClient(
            model_name=model_name or "gemini-2.5-flash",
            api_key=genai_api_key or api_key,
            base_url=genai_base_url or base_url,
            api_version=genai_api_version,
            headers=genai_headers,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
    
    else:
        raise ValueError(f"Unknown client type: {client_type}")


def create_llm_client_from_config(config) -> BaseLLMClient:
    """
    Create an LLM client from a Config object.
    
    Args:
        config: Config object with llm settings
        
    Returns:
        Configured LLM client instance
    """
    llm = config.llm
    return create_llm_client(
        client_type=llm.client_type,
        model_name=llm.model_name,
        temperature=llm.temperature,
        max_tokens=llm.max_tokens,
        max_retries=llm.max_retries,
        # OpenAI
        openai_api_key=llm.openai_api_key,
        openai_base_url=llm.openai_base_url,
        # Azure
        azure_api_key=llm.azure_api_key,
        azure_endpoint=llm.azure_endpoint,
        azure_api_version=llm.azure_api_version,
        # Custom
        custom_api_key=llm.custom_api_key,
        custom_endpoint=llm.custom_endpoint,
        custom_api_version=llm.custom_api_version,
        custom_headers=llm.custom_headers,
        supports_temperature=llm.supports_temperature,
        supports_max_tokens=llm.supports_max_tokens,
        # GenAI
        genai_api_key=llm.genai_api_key,
        genai_base_url=llm.genai_base_url,
        genai_api_version=llm.genai_api_version,
        genai_headers=llm.genai_headers,
    )
