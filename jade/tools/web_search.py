"""Web search tool implementation using SerpApi."""

import os
import json
import http.client
from typing import List, Union, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class WebSearchTool:
    """Web search tool using SerpApi (Google Search)."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize web search tool.
        
        Args:
            api_key: SerpApi key (defaults to SERPAPI_KEY env var)
        """
        self.api_key = api_key or os.environ.get("SERPAPI_KEY")
        if not self.api_key:
            raise ValueError("SerpApi key is required. Set SERPAPI_KEY environment variable or pass api_key parameter.")
        self._last_raw_results = {}  # Store raw results for logging
    
    @staticmethod
    def _contains_chinese(text: str) -> bool:
        """Check if text contains Chinese characters."""
        return any('\u4E00' <= char <= '\u9FFF' for char in text)
    
    def search_single(self, query: str, num_results: int = 10) -> str:
        """
        Perform a single Google search using SerpApi.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            
        Returns:
            Formatted search results string
        """
        conn = http.client.HTTPSConnection("google.serper.dev")
        
        # Determine locale based on query language
        if self._contains_chinese(query):
            payload = json.dumps({
                "q": query,
                "location": "China",
                "gl": "cn",
                "hl": "zh-cn",
                "num": num_results
            })
        else:
            payload = json.dumps({
                "q": query,
                "location": "United States",
                "gl": "us",
                "hl": "en",
                "num": num_results
            })
        
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        # Retry logic
        for attempt in range(5):
            try:
                conn.request("POST", "/search", payload, headers)
                res = conn.getresponse()
                break
            except Exception as e:
                print(f"Search attempt {attempt + 1} failed: {e}")
                if attempt == 4:
                    return f"Google search timeout for query: '{query}'. Please try again."
                continue
        
        data = res.read()
        results = json.loads(data.decode("utf-8"))
        
        # Store raw results for logging
        self._last_raw_results[query] = results
        
        try:
            if "organic" not in results:
                # Check for error
                if "error" in results:
                    return f"Search error for '{query}': {results.get('error', 'Unknown error')}"
                return f"No results found for query: '{query}'. Try a less specific query."
            
            web_snippets = []
            idx = 0
            
            for page in results.get("organic", [])[:num_results]:
                idx += 1
                
                date_published = ""
                if "date" in page:
                    date_published = f"\nDate published: {page['date']}"
                
                source = ""
                if "source" in page:
                    source = f"\nSource: {page['source']}"
                
                snippet = ""
                if "snippet" in page:
                    snippet = f"\n{page['snippet']}"
                
                title = page.get('title', 'No title')
                link = page.get('link', '')
                
                entry = f"{idx}. [{title}]({link}){date_published}{source}{snippet}"
                entry = entry.replace("Your browser can't play this video.", "")
                web_snippets.append(entry)
            
            # Include additional information if available
            additional_info = []
            
            # Answer box
            if "answerBox" in results:
                answer_box = results["answerBox"]
                if "answer" in answer_box:
                    additional_info.append(f"**Quick Answer**: {answer_box['answer']}")
                elif "snippet" in answer_box:
                    additional_info.append(f"**Featured Snippet**: {answer_box['snippet']}")
            
            # Knowledge graph
            if "knowledgeGraph" in results:
                kg = results["knowledgeGraph"]
                if "description" in kg:
                    additional_info.append(f"**Knowledge Graph**: {kg.get('title', '')}: {kg['description']}")
            
            # Build final content
            content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n"
            
            if additional_info:
                content += "## Quick Information\n" + "\n".join(additional_info) + "\n\n"
            
            content += "## Web Results\n" + "\n\n".join(web_snippets)
            
            return content
            
        except Exception as e:
            return f"Error processing search results for '{query}': {str(e)}"
    
    def search(self, queries: Union[str, List[str]]) -> str:
        """
        Perform web searches for one or more queries.
        
        Args:
            queries: Single query string or list of queries
            
        Returns:
            Combined search results
        """
        if isinstance(queries, str):
            return self.search_single(queries)
        
        results = []
        for query in queries:
            results.append(self.search_single(query))
        
        return "\n\n=======\n\n".join(results)
    
    def get_raw_results(self, queries: Union[str, List[str]]) -> Union[dict, List[dict]]:
        """
        Get raw search results for logging purposes.
        
        Args:
            queries: Single query string or list of queries
            
        Returns:
            Raw results dict or list of raw results dicts
        """
        if isinstance(queries, str):
            return self._last_raw_results.get(queries, {})
        
        return [self._last_raw_results.get(query, {}) for query in queries]


# Global tool instance
_search_tool = None


def get_search_tool() -> WebSearchTool:
    """Get or create the global search tool instance."""
    global _search_tool
    if _search_tool is None:
        _search_tool = WebSearchTool()
    return _search_tool


def set_search_api_key(api_key: str):
    """Set the API key for the global search tool."""
    global _search_tool
    _search_tool = WebSearchTool(api_key=api_key)


class SearchInput(BaseModel):
    """Input schema for web search tool."""
    queries: List[str] = Field(
        description="List of search queries to execute. Include multiple complementary queries for comprehensive results."
    )


@tool("web_search", args_schema=SearchInput)
def web_search(queries: List[str]) -> str:
    """
    Perform Google web searches and return top results.
    
    Use this tool to search the web for information. You can provide multiple
    queries to get comprehensive results from different search angles.
    
    Args:
        queries: List of search query strings
        
    Returns:
        Formatted search results with titles, links, and snippets
    """
    search_tool = get_search_tool()
    return search_tool.search(queries)
