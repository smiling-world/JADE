"""Tool implementations for the jade agent."""

from jade.tools.web_search import web_search, WebSearchTool, set_search_api_key
from jade.tools.web_scraper import (
    web_scraper, WebScraperTool, set_scraper_llm, set_scraper_jina_key,
    set_scraper_concurrency, set_scraper_cache_dir
)
from jade.tools.analyzer import analyze_content, AnalyzerTool, set_analyzer_llm

__all__ = [
    # Tool functions
    "web_search",
    "web_scraper", 
    "analyze_content",
    # Tool classes
    "WebSearchTool",
    "WebScraperTool",
    "AnalyzerTool",
    # Configuration setters
    "set_search_api_key",
    "set_scraper_llm",
    "set_scraper_jina_key",
    "set_scraper_concurrency",
    "set_scraper_cache_dir",
    "set_analyzer_llm",
]

