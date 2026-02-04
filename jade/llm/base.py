"""Base LLM client interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ):
        """
        Initialize base LLM client.
        
        Args:
            model_name: Name of the model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional configuration options
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.config = kwargs
    
    @abstractmethod
    def get_client(self) -> Any:
        """
        Get the underlying client instance.
        
        Returns:
            Client instance (OpenAI, Azure, etc.)
        """
        pass
    
    @abstractmethod
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
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            stop: Stop sequences
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        pass
    
    def __call__(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Allow calling the client directly."""
        return self.chat_completion(messages, **kwargs)

