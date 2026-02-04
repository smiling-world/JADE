"""
Conciseness Scoring Module.

This module provides functionality to evaluate the knowledge density of responses
by calculating the score per token ratio. This helps assess how efficiently
information is conveyed in the response.

Three density calculation methods are available:
1. Linear (1/x): Most sensitive to length differences
2. Logarithmic (1/log(x)): Moderate sensitivity, good balance
3. Power (1/x^α): Configurable sensitivity via alpha parameter

Example:
    >>> from jade.scoring.conciseness import ConcisenessScorer, ConcisenessResult
    >>> scorer = ConcisenessScorer()
    >>> result = scorer.calculate_knowledge_density(
    ...     response="This is the response text...",
    ...     total_score=0.85,
    ...     method="log"  # Use logarithmic scaling
    ... )
    >>> print(f"Knowledge density: {result.density_score:.4f}")
    >>> print(f"Token count: {result.token_count}")
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Literal

import tiktoken


class DensityMethod(str, Enum):
    """Methods for calculating knowledge density."""
    LINEAR = "linear"      # score / tokens (most sensitive)
    LOG = "log"            # score / log(tokens) (moderate)
    POWER = "power"        # score / tokens^alpha (configurable)


@dataclass
class ConcisenessResult:
    """Result of conciseness evaluation."""
    
    token_count: int  # Total number of tokens in the response
    total_score: float  # The original total score (0.0 to 1.0)
    density_score: float  # The calculated knowledge density
    method: str  # Method used for calculation
    encoding_name: str  # Name of the tokenizer encoding used
    
    # Raw metrics for reference
    score_per_token: float = 0.0  # total_score / token_count (linear)
    score_per_1k_tokens: float = 0.0  # Score per 1000 tokens
    
    # Method-specific parameters
    method_params: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "token_count": self.token_count,
            "total_score": self.total_score,
            "density_score": self.density_score,
            "method": self.method,
            "method_params": self.method_params,
            "score_per_token": self.score_per_token,
            "score_per_1k_tokens": self.score_per_1k_tokens,
            "encoding_name": self.encoding_name
        }


class ConcisenessScorer:
    """
    Scorer for evaluating response conciseness via knowledge density.
    
    Knowledge density represents how much "value" each token contributes.
    Higher knowledge density indicates a more concise and efficient response.
    
    Three calculation methods are available:
    - "linear": score / tokens (most sensitive to length)
    - "log": score / log(tokens + 1) (moderate sensitivity, recommended)
    - "power": score / tokens^alpha (configurable sensitivity)
    
    Example:
        >>> scorer = ConcisenessScorer()
        >>> # Compare different methods
        >>> result_linear = scorer.calculate_knowledge_density("...", 0.9, method="linear")
        >>> result_log = scorer.calculate_knowledge_density("...", 0.9, method="log")
        >>> result_power = scorer.calculate_knowledge_density("...", 0.9, method="power", alpha=0.5)
    """
    
    # Default alpha for power method (0.5 = square root, less sensitive than linear)
    DEFAULT_ALPHA = 0.5
    
    # Default base for logarithm (e ≈ 2.718, natural log)
    DEFAULT_LOG_BASE = math.e
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize the conciseness scorer.
        
        Args:
            encoding_name: Name of the tiktoken encoding to use.
                Common options:
                - "cl100k_base": Used by GPT-4, GPT-3.5-turbo, text-embedding-ada-002
                - "p50k_base": Used by Codex models
                - "r50k_base": Used by GPT-3 models (davinci, curie, etc.)
                Default is "cl100k_base" for modern models.
        """
        self.encoding_name = encoding_name
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            # Fallback to cl100k_base if specified encoding not found
            print(f"Warning: Could not load encoding '{encoding_name}': {e}")
            print("Falling back to 'cl100k_base'")
            self.encoding_name = "cl100k_base"
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text: The text to tokenize
            
        Returns:
            Number of tokens in the text
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def _calculate_density_linear(self, score: float, tokens: int) -> float:
        """Linear method: score / tokens"""
        return score / tokens if tokens > 0 else 0.0
    
    def _calculate_density_log(
        self, 
        score: float, 
        tokens: int, 
        log_base: float = None
    ) -> float:
        """
        Logarithmic method: score / log(tokens + 1)
        
        Using log(tokens + 1) to handle tokens=0 and make the curve smoother.
        The +1 ensures log is always positive for tokens >= 0.
        """
        if log_base is None:
            log_base = self.DEFAULT_LOG_BASE
        
        if tokens <= 0:
            return 0.0
        
        # log_base(tokens + 1) = ln(tokens + 1) / ln(base)
        log_value = math.log(tokens + 1) / math.log(log_base)
        return score / log_value if log_value > 0 else 0.0
    
    def _calculate_density_power(
        self, 
        score: float, 
        tokens: int, 
        alpha: float = None
    ) -> float:
        """
        Power method: score / tokens^alpha
        
        alpha < 1: Less sensitive than linear (e.g., 0.5 = square root)
        alpha = 1: Same as linear
        alpha > 1: More sensitive than linear
        """
        if alpha is None:
            alpha = self.DEFAULT_ALPHA
        
        if tokens <= 0:
            return 0.0
        
        return score / (tokens ** alpha)
    
    def calculate_knowledge_density(
        self,
        response: str,
        total_score: float,
        method: Literal["linear", "log", "power"] = "log",
        alpha: float = None,
        log_base: float = None,
        min_tokens: int = 1
    ) -> ConcisenessResult:
        """
        Calculate knowledge density of a response.
        
        Args:
            response: The response text to evaluate
            total_score: The total score achieved (typically 0.0 to 1.0)
            method: Calculation method
                - "linear": score / tokens (most sensitive)
                - "log": score / log(tokens + 1) (recommended, moderate sensitivity)
                - "power": score / tokens^alpha (configurable)
            alpha: Power exponent for "power" method (default 0.5)
                - 0.5: Square root, moderate sensitivity
                - 0.3: Cube root-ish, low sensitivity
                - 1.0: Same as linear
            log_base: Base for logarithm in "log" method (default e ≈ 2.718)
            min_tokens: Minimum token count to prevent extreme values
            
        Returns:
            ConcisenessResult with token count and density metrics
            
        Example sensitivity comparison (score=0.9):
            tokens=100 vs tokens=1000:
            - linear: 10x difference
            - log (base e): ~1.5x difference  
            - power (α=0.5): ~3.2x difference
        """
        token_count = self.count_tokens(response)
        effective_tokens = max(token_count, min_tokens)
        
        # Always calculate linear metrics for reference
        score_per_token = total_score / effective_tokens
        score_per_1k_tokens = score_per_token * 1000
        
        # Calculate density based on method
        method_params = {}
        
        if method == "linear":
            density_score = self._calculate_density_linear(total_score, effective_tokens)
            
        elif method == "log":
            actual_log_base = log_base if log_base is not None else self.DEFAULT_LOG_BASE
            density_score = self._calculate_density_log(
                total_score, effective_tokens, actual_log_base
            )
            method_params["log_base"] = actual_log_base
            
        elif method == "power":
            actual_alpha = alpha if alpha is not None else self.DEFAULT_ALPHA
            density_score = self._calculate_density_power(
                total_score, effective_tokens, actual_alpha
            )
            method_params["alpha"] = actual_alpha
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'linear', 'log', or 'power'.")
        
        return ConcisenessResult(
            token_count=token_count,
            total_score=total_score,
            density_score=density_score,
            method=method,
            method_params=method_params,
            score_per_token=score_per_token,
            score_per_1k_tokens=score_per_1k_tokens,
            encoding_name=self.encoding_name
        )
    
    def compare_responses(
        self,
        responses: list[tuple[str, float]],
        labels: Optional[list[str]] = None,
        method: Literal["linear", "log", "power"] = "log",
        alpha: float = None,
        log_base: float = None
    ) -> dict:
        """
        Compare knowledge density across multiple responses.
        
        Useful for comparing different model outputs or response strategies.
        
        Args:
            responses: List of (response_text, total_score) tuples
            labels: Optional labels for each response (e.g., model names)
            method: Calculation method ("linear", "log", or "power")
            alpha: Power exponent for "power" method
            log_base: Base for logarithm in "log" method
            
        Returns:
            Dictionary with comparison results including rankings
        """
        if labels is None:
            labels = [f"Response_{i}" for i in range(len(responses))]
        
        results = []
        for i, (response, score) in enumerate(responses):
            density_result = self.calculate_knowledge_density(
                response, score, method=method, alpha=alpha, log_base=log_base
            )
            results.append({
                "label": labels[i],
                "result": density_result,
                **density_result.to_dict()
            })
        
        # Sort by knowledge density (higher is better)
        ranked = sorted(results, key=lambda x: x["density_score"], reverse=True)
        for rank, item in enumerate(ranked, 1):
            item["rank"] = rank
        
        return {
            "results": results,
            "ranked_results": ranked,
            "best": ranked[0] if ranked else None,
            "method": method,
            "comparison_summary": {
                "total_responses": len(responses),
                "best_label": ranked[0]["label"] if ranked else None,
                "best_density_score": ranked[0]["density_score"] if ranked else None,
                "worst_label": ranked[-1]["label"] if ranked else None,
                "worst_density_score": ranked[-1]["density_score"] if ranked else None,
            }
        }
    
    def analyze_sensitivity(
        self,
        response: str,
        total_score: float
    ) -> dict:
        """
        Analyze how different methods affect the density score.
        
        Useful for understanding and choosing the right method.
        
        Args:
            response: The response text to evaluate
            total_score: The total score achieved
            
        Returns:
            Dictionary with density scores for all methods
        """
        token_count = self.count_tokens(response)
        
        linear = self.calculate_knowledge_density(response, total_score, method="linear")
        log_result = self.calculate_knowledge_density(response, total_score, method="log")
        power_03 = self.calculate_knowledge_density(response, total_score, method="power", alpha=0.3)
        power_05 = self.calculate_knowledge_density(response, total_score, method="power", alpha=0.5)
        power_07 = self.calculate_knowledge_density(response, total_score, method="power", alpha=0.7)
        
        return {
            "token_count": token_count,
            "total_score": total_score,
            "methods": {
                "linear": linear.density_score,
                "log": log_result.density_score,
                "power_0.3": power_03.density_score,
                "power_0.5": power_05.density_score,
                "power_0.7": power_07.density_score,
            },
            "recommendation": "log" if token_count > 50 else "power_0.5"
        }


def calculate_knowledge_density(
    response: str,
    total_score: float,
    method: Literal["linear", "log", "power"] = "log",
    alpha: float = None,
    log_base: float = None,
    encoding_name: str = "cl100k_base"
) -> ConcisenessResult:
    """
    Convenience function to calculate knowledge density without instantiating a scorer.
    
    Args:
        response: The response text to evaluate
        total_score: The total score achieved (typically 0.0 to 1.0)
        method: Calculation method ("linear", "log", or "power")
        alpha: Power exponent for "power" method (default 0.5)
        log_base: Base for logarithm in "log" method (default e)
        encoding_name: Name of the tiktoken encoding to use
        
    Returns:
        ConcisenessResult with token count and density metrics
        
    Example:
        >>> from jade.scoring.conciseness import calculate_knowledge_density
        >>> result = calculate_knowledge_density("Response text here...", 0.85, method="log")
        >>> print(f"Density score: {result.density_score:.4f}")
    """
    scorer = ConcisenessScorer(encoding_name=encoding_name)
    return scorer.calculate_knowledge_density(
        response, total_score, method=method, alpha=alpha, log_base=log_base
    )

