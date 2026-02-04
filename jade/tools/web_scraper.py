"""Web scraper tool implementation using Jina Reader API."""

import os
import json
import time
import hashlib
from typing import List, Union, Optional, Dict
from urllib.parse import urlparse, urlunparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import tiktoken
from langchain_core.tools import tool
from pydantic import BaseModel, Field


def normalize_url(url: str) -> str:
    """
    Normalize URL by removing fragment (e.g., #:~:text=...) to identify identical pages.
    
    Args:
        url: Original URL
        
    Returns:
        Normalized URL without fragment
    """
    parsed = urlparse(url)
    # Remove fragment part (everything after #)
    normalized = urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        parsed.query,
        ''  # Remove fragment
    ))
    return normalized


def url_cache_key(url: str) -> str:
    """Generate a cache key for a normalized URL."""
    normalized = normalize_url(url)
    return hashlib.md5(normalized.encode()).hexdigest()


class URLCache:
    """Simple in-memory cache for scraped URL raw content.
    
    Note: This cache stores RAW webpage content (from Jina), not processed results.
    This allows re-processing with different goals while reusing fetched content.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self._memory_cache: Dict[str, str] = {}  # Stores raw content
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def _file_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json") if self.cache_dir else None
    
    def get(self, url: str) -> Optional[str]:
        """Get cached raw content for URL."""
        key = url_cache_key(url)
        # Check memory first
        if key in self._memory_cache:
            return self._memory_cache[key]
        return None
    
    def set(self, url: str, raw_content: str):
        """Cache raw content for URL."""
        key = url_cache_key(url)
        self._memory_cache[key] = raw_content

    def has(self, url: str) -> bool:
        """Check if URL is cached."""
        return self.get(url) is not None


def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    """
    Truncate text to a maximum number of tokens.
    
    Args:
        text: Input text
        max_tokens: Maximum number of tokens
        
    Returns:
        Truncated text
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    except Exception:
        # Fallback to character-based truncation
        char_limit = max_tokens * 4  # Approximate 4 chars per token
        return text[:char_limit] if len(text) > char_limit else text


class WebScraperTool:
    """Web scraper tool using Jina Reader API."""
    
    # Prompt for extracting information from webpage
    EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rationale**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" fields**
"""
    
    def __init__(
        self, 
        jina_api_key: Optional[str] = None,
        llm_client = None,
        max_content_length: int = 150000,
        timeout: int = 60,
        max_concurrency: int = 1,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize web scraper tool.
        
        Args:
            jina_api_key: Jina Reader API key
            llm_client: LLM client for content summarization
            max_content_length: Maximum content length to process
            timeout: Request timeout in seconds
            max_concurrency: Maximum concurrent scraping tasks
            cache_dir: Directory for persistent URL cache (None for memory-only)
        """
        self.jina_api_key = jina_api_key or os.environ.get("JINA_API_KEY") or os.environ.get("JINA_API_KEYS")
        self.llm_client = llm_client
        self.max_content_length = max_content_length
        self.timeout = timeout
        self.max_concurrency = max_concurrency
        self._last_raw_content = {}  # Store raw content for logging
        self._cache = URLCache(cache_dir)
    
    def set_llm_client(self, llm_client):
        """Set the LLM client for content summarization."""
        self.llm_client = llm_client
    
    def _fetch_with_jina(self, url: str, max_retries: int = 3) -> str:
        """
        Fetch webpage content using Jina Reader API.
        
        Args:
            url: URL to fetch
            max_retries: Maximum retry attempts
            
        Returns:
            Webpage content or error message
        """
        headers = {
            "X-Engine": "direct",
            "X-Timeout": "30"
        }
        if self.jina_api_key:
            headers["Authorization"] = f"Bearer {self.jina_api_key}"
        
        jina_url = f"https://r.jina.ai/{url}"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    jina_url,
                    headers=headers,
                    timeout=self.timeout
                )
                if response.status_code == 200:
                    raw_content = response.text
                    # Store raw content for logging
                    self._last_raw_content[url] = raw_content
                    return raw_content
                else:
                    print(f"Jina fetch failed (status {response.status_code}): {response.text[:200]}")
                    
            except Exception as e:
                print(f"Jina fetch attempt {attempt + 1} failed: {e}")
                time.sleep(0.5)
        
        return "[scraper] Failed to fetch webpage content."
    
    def _summarize_content(self, content: str, goal: str, max_retries: int = 3) -> dict:
        """
        Summarize webpage content using LLM.
        
        Args:
            content: Webpage content
            goal: User's goal for visiting the page
            max_retries: Maximum retry attempts
            
        Returns:
            Dict with 'rational', 'evidence', 'summary' keys
        """
        if not self.llm_client:
            return {
                "rational": "LLM client not configured",
                "evidence": content[:5000],
                "summary": "Content summarization unavailable - no LLM client configured."
            }
        
        # Truncate content to token limit
        content = truncate_to_tokens(content, max_tokens=95000)
        
        prompt = self.EXTRACTOR_PROMPT.format(
            webpage_content=content,
            goal=goal
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat_completion(messages)
                
                # Clean up response
                response = response.replace("```json", "").replace("```", "").strip()
                
                # Try to parse JSON
                try:
                    result = json.loads(response)
                    return result
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    left = response.find('{')
                    right = response.rfind('}')
                    if left != -1 and right != -1 and left < right:
                        try:
                            result = json.loads(response[left:right+1])
                            return result
                        except json.JSONDecodeError:
                            pass
                    
            except Exception as e:
                print(f"Summarization attempt {attempt + 1} failed: {e}")
        
        return {
            "rational": "Failed to process content",
            "evidence": content[:3000],
            "summary": "Content summarization failed after multiple attempts."
        }
    
    def scrape_single(self, url: str, goal: str, use_cache: bool = True) -> str:
        """
        Scrape a single webpage and extract relevant information.
        
        Args:
            url: URL to scrape
            goal: Goal/purpose of visiting the page
            use_cache: Whether to use cached raw content
            
        Returns:
            Formatted extraction result
        """
        raw_content = None
        cache_hit = False
        cache_key = url_cache_key(url)
        
        # Check cache for raw content first (using normalized URL)
        if use_cache:
            raw_content = self._cache.get(url)
            if raw_content:
                cache_hit = True
                normalized = normalize_url(url)
                print(f"[Cache hit] {url} -> {normalized}")
        
        # Fetch content if not cached
        if raw_content is None:
            raw_content = self._fetch_with_jina(url)
            # Cache the raw content (not the processed result)
            if use_cache and not raw_content.startswith("[scraper]"):
                self._cache.set(url, raw_content)
        
        # Log scrape details if cache_dir is set
        if self._cache.cache_dir:
            log_path = os.path.join(self._cache.cache_dir, "scrape_log.jsonl")
            with open(log_path, "a") as f:
                f.write(json.dumps({"url": url, "cache_key": cache_key, "cache_hit": cache_hit, "goal": goal, "raw_content": raw_content}) + "\n")
        
        # Always update _last_raw_content for logging (even for cache hits)
        self._last_raw_content[url] = raw_content
        
        if raw_content.startswith("[scraper]"):
            result = f"""The useful information in {url} for user goal "{goal}" as follows:

Evidence in page:
The provided webpage content could not be accessed. Please check the URL or try again later.

Summary:
The webpage content could not be processed, and therefore, no information is available.
"""
            return result
        
        # Summarize content with current goal (always re-process, even from cache)
        result = self._summarize_content(raw_content, goal)
        
        output = f"""The useful information in {url} for user goal "{goal}" as follows:

Rationale:
{result.get('rational', 'N/A')}

Evidence in page:
{result.get('evidence', 'N/A')}

Summary:
{result.get('summary', 'N/A')}
"""
        return output
    
    def scrape(self, urls: Union[str, List[str]], goal: str) -> str:
        """
        Scrape one or more webpages with caching and optional concurrency.
        
        Args:
            urls: Single URL or list of URLs
            goal: Goal/purpose of visiting the pages
            
        Returns:
            Combined scraping results
        """
        if isinstance(urls, str):
            return self.scrape_single(urls, goal)
        
        # Deduplicate URLs by normalized form
        seen_normalized = {}
        unique_urls = []
        url_mapping = {}  # original -> first original with same normalized
        
        for url in urls:
            normalized = normalize_url(url)
            if normalized not in seen_normalized:
                seen_normalized[normalized] = url
                unique_urls.append(url)
            url_mapping[url] = seen_normalized[normalized]
        
        if len(unique_urls) < len(urls):
            print(f"[Dedup] {len(urls)} URLs -> {len(unique_urls)} unique pages")
        
        results_map = {}
        start_time = time.time()
        
        def scrape_with_timeout(url: str) -> tuple:
            if time.time() - start_time > 900:
                return url, f"[Timeout] Skipped {url} due to time limit."
            try:
                return url, self.scrape_single(url, goal)
            except Exception as e:
                return url, f"Error scraping {url}: {str(e)}"
        
        # Concurrent or sequential scraping
        if self.max_concurrency > 1 and len(unique_urls) > 1:
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
                futures = {executor.submit(scrape_with_timeout, url): url for url in unique_urls}
                for future in as_completed(futures):
                    url, result = future.result()
                    results_map[normalize_url(url)] = result
        else:
            for url in unique_urls:
                _, result = scrape_with_timeout(url)
                results_map[normalize_url(url)] = result
        
        # Build results in original order, reusing results for duplicate URLs
        results = []
        for url in urls:
            normalized = normalize_url(url)
            results.append(results_map[normalized])
        
        return "\n\n=======\n\n".join(results)
    
    def get_raw_content(self, urls: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Get raw scraped content for logging purposes.
        
        Args:
            urls: Single URL or list of URLs
            
        Returns:
            Raw content string or list of raw content strings
        """
        if isinstance(urls, str):
            return self._last_raw_content.get(urls, "")
        
        return [self._last_raw_content.get(url, "") for url in urls]


# Global tool instance
_scraper_tool = None


def get_scraper_tool() -> WebScraperTool:
    """Get or create the global scraper tool instance."""
    global _scraper_tool
    if _scraper_tool is None:
        _scraper_tool = WebScraperTool()
    return _scraper_tool


def set_scraper_llm(llm_client):
    """Set the LLM client for the global scraper tool."""
    tool = get_scraper_tool()
    tool.set_llm_client(llm_client)


def set_scraper_jina_key(jina_api_key: str):
    """Set the Jina API key for the global scraper tool."""
    global _scraper_tool
    if _scraper_tool is None:
        _scraper_tool = WebScraperTool(jina_api_key=jina_api_key)
    else:
        _scraper_tool.jina_api_key = jina_api_key


def set_scraper_concurrency(max_concurrency: int):
    """Set the maximum concurrency for scraping."""
    tool = get_scraper_tool()
    tool.max_concurrency = max_concurrency


def set_scraper_cache_dir(cache_dir: str):
    """Set the cache directory for persistent URL caching."""
    tool = get_scraper_tool()
    tool._cache = URLCache(cache_dir)


class ScraperInput(BaseModel):
    """Input schema for web scraper tool."""
    urls: List[str] = Field(
        description="List of URLs to visit and extract information from."
    )
    goal: str = Field(
        description="The specific information goal for visiting the webpage(s). Be specific about what information you're looking for."
    )


@tool("web_scraper", args_schema=ScraperInput)
def web_scraper(urls: List[str], goal: str) -> str:
    """
    Visit webpage(s) and extract relevant information based on a goal.
    
    Use this tool to visit web pages and extract specific information.
    The tool will fetch the page content, analyze it based on your goal,
    and return relevant evidence and a summary.
    
    Args:
        urls: List of URLs to visit
        goal: What specific information you're looking for
        
    Returns:
        Extracted information with evidence and summary for each URL
    """
    scraper_tool = get_scraper_tool()
    return scraper_tool.scrape(urls, goal)

