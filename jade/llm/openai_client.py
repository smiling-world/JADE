"""OpenAI LLM client implementation."""

import os
import time
import random
from typing import List, Dict, Optional, Any

from openai import OpenAI, APIError, APIConnectionError, APITimeoutError

from jade.llm.base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """OpenAI API client implementation."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize OpenAI client.
        
        Args:
            model_name: Model name (e.g., 'gpt-4o', 'gpt-4-turbo')
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Custom API base URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            max_retries: Number of retry attempts
            **kwargs: Additional configuration
        """
        super().__init__(model_name, temperature, max_tokens, **kwargs)
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.max_retries = max_retries
        self._client = None
    
    def get_client(self) -> OpenAI:
        """Get or create OpenAI client instance."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=120.0,
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
        Generate chat completion using OpenAI API.
        
        Args:
            messages: List of message dicts
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            stop: Stop sequences
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        client = self.get_client()
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        base_sleep_time = 1
        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                    stop=stop,
                    **kwargs
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()
                    
            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(f"OpenAI API error (attempt {attempt + 1}/{self.max_retries}): {e}")
            except Exception as e:
                print(f"Unexpected error (attempt {attempt + 1}/{self.max_retries}): {e}")
            
            if attempt < self.max_retries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
        
        return "Error: Failed to get response from OpenAI API after all retries."

