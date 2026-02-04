"""
Custom LLM Client Implementation.

This module provides a flexible LLM client that can work with custom
Azure OpenAI endpoints or other compatible API endpoints.

Example:
    >>> client = CustomLLMClient(
    ...     api_key=os.environ["CUSTOM_API_KEY"],
    ...     azure_endpoint=os.environ["CUSTOM_ENDPOINT"],
    ...     model_name="gpt-4"
    ... )
    >>> response = client.chat_completion([{"role": "user", "content": "Hello"}])
"""

import os
import time
import random
from typing import List, Dict, Optional

import openai
from openai import APIError, APIConnectionError, APITimeoutError

from jade.llm.base import BaseLLMClient


class CustomLLMClient(BaseLLMClient):
    """
    Custom LLM client for Azure OpenAI compatible endpoints.
    
    This client supports custom Azure endpoints that may have specific
    header requirements or parameter limitations. It can be configured
    to work with various API endpoints.
    
    Attributes:
        api_key: API key for authentication
        azure_endpoint: Base URL for the API endpoint
        api_version: API version string
        default_headers: Custom headers to include in requests
        supports_temperature: Whether the endpoint supports temperature
        supports_max_tokens: Whether the endpoint supports max_tokens
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
        default_headers: Optional[Dict[str, str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 3,
        supports_temperature: bool = True,
        supports_max_tokens: bool = True,
        **kwargs
    ):
        """
        Initialize custom LLM client.
        
        All configuration can be provided via environment variables:
        - CUSTOM_MODEL_NAME: Model name/deployment
        - CUSTOM_API_KEY: API key
        - CUSTOM_ENDPOINT: API endpoint URL
        - CUSTOM_API_VERSION: API version
        
        Args:
            model_name: Model name/deployment (env: CUSTOM_MODEL_NAME)
            api_key: API key (env: CUSTOM_API_KEY)
            azure_endpoint: API endpoint URL (env: CUSTOM_ENDPOINT)
            api_version: API version
            default_headers: Custom headers for requests
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            max_retries: Number of retry attempts
            supports_temperature: Whether endpoint supports temperature
            supports_max_tokens: Whether endpoint supports max_tokens
            **kwargs: Additional configuration
            
        Raises:
            ValueError: If required configuration is missing
        """
        # Get configuration from parameters or environment
        model = model_name or os.environ.get("CUSTOM_MODEL_NAME", "gpt-4")
        super().__init__(model, temperature, max_tokens, **kwargs)
        print(os.environ)
        self.api_key = api_key or os.environ.get("CUSTOM_API_KEY")
        self.azure_endpoint = azure_endpoint or os.environ.get("CUSTOM_ENDPOINT")
        self.api_version = api_version or os.environ.get("CUSTOM_API_VERSION", "2024-02-15-preview")
        self.default_headers = default_headers or {}
        self.max_retries = max_retries
        self.supports_temperature = supports_temperature
        self.supports_max_tokens = supports_max_tokens
        self._client = None
        
        # Validate required configuration
        if not self.api_key:
            raise ValueError(
                "API key is required. Set CUSTOM_API_KEY environment variable "
                "or pass api_key parameter."
            )
        if not self.azure_endpoint:
            raise ValueError(
                "API endpoint is required. Set CUSTOM_ENDPOINT environment variable "
                "or pass azure_endpoint parameter."
            )
    
    def get_client(self) -> openai.AzureOpenAI:
        """
        Get or create the Azure OpenAI client instance.
        
        Returns:
            Configured AzureOpenAI client
        """
        if self._client is None:
            self._client = openai.AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version,
                default_headers=self.default_headers if self.default_headers else None,
            )
        return self._client
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Generate a chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (if supported)
            max_tokens: Maximum tokens (if supported)
            stop: Stop sequences
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        client = self.get_client()
        
        # Build request parameters
        request_params = {
            "messages": messages,
            "model": self.model_name,
        }
        
        # Conditionally add parameters based on endpoint support
        if self.supports_temperature:
            request_params["temperature"] = temperature or self.temperature
            
        if self.supports_max_tokens:
            request_params["max_tokens"] = max_tokens or self.max_tokens
            
        if stop:
            request_params["stop"] = stop
        
        request_params.update(kwargs)
        
        # Retry logic with exponential backoff
        base_sleep_time = 1
        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(**request_params)
                content = response.choices[0].message.content
                
                if content and content.strip():
                    return content.strip()
                    
            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(f"API error (attempt {attempt + 1}/{self.max_retries}): {e}")
            except Exception as e:
                print(f"Unexpected error (attempt {attempt + 1}/{self.max_retries}): {e}")
            
            if attempt < self.max_retries - 1:
                sleep_time = min(base_sleep_time * (2 ** attempt) + random.uniform(0, 1), 30)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
        
        return "Error: Failed to get response after all retries."


def create_custom_client(
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    headers: Optional[Dict] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> CustomLLMClient:
    """
    Factory function to create a CustomLLMClient.
    
    Args:
        endpoint: API endpoint URL
        api_key: API key
        headers: Custom headers
        model_name: Model name
        **kwargs: Additional configuration
        
    Returns:
        Configured CustomLLMClient instance
    """
    return CustomLLMClient(
        model_name=model_name,
        api_key=api_key,
        azure_endpoint=endpoint,
        default_headers=headers,
        **kwargs
    )
