"""
GenAI LLM Client Implementation.

This module provides a flexible LLM client that works with Google GenAI API
or compatible endpoints (e.g., custom GenAI proxies).

Example:
    >>> client = GenAIClient(
    ...     api_key=os.environ["GENAI_API_KEY"],
    ...     base_url=os.environ["GENAI_BASE_URL"],
    ...     model_name="gemini-2.5-flash"
    ... )
    >>> response = client.chat_completion([{"role": "user", "content": "Hello"}])
"""

import os
import time
import random
from typing import List, Dict, Optional

from google import genai
from google.genai import types

from jade.llm.base import BaseLLMClient


class GenAIClient(BaseLLMClient):
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "v1",
        base_url: Optional[str] = None,
        headers: Optional[Dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 3,
        **kwargs
    ):
        # Get configuration from parameters or environment
        model = model_name or os.environ.get("CUSTOM_MODEL_NAME", "gemini-2.5-flash")
        super().__init__(model, temperature, max_tokens, **kwargs)
        
        self.api_key = api_key or os.environ.get("GENAI_API_KEY")
        self.api_version = api_version or os.environ.get("GENAI_API_VERSION", "v1")
        self.base_url = base_url or os.environ.get("GENAI_BASE_URL")
        self.headers = headers or {}
        self.max_retries = max_retries
        self._client = None
        

    def get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(
                    api_version=self.api_version,
                    base_url=self.base_url,
                    headers=self.headers,
                ),
            )
        return self._client
    
    @staticmethod
    def _convert_messages(messages: List[Dict[str, str]]) -> tuple:
        """
        Convert OpenAI-style messages to GenAI contents format.
        
        Returns:
            Tuple of (contents, system_instruction)
        """
        system_instruction: Optional[str] = None
        contents: List[types.Content] = []
        
        for m in messages:
            role = m.get('role', '')
            content = m.get('content', '')
            
            # Handle system messages
            if role == 'system':
                system_instruction = content
                continue
            
            # Map roles: assistant -> model, user -> user
            mapped_role = 'model' if role == 'assistant' else 'user'
            
            if content:
                parts = [types.Part.from_text(text=content)]
                contents.append(types.Content(role=mapped_role, parts=parts))
        
        return contents, system_instruction
    
    @staticmethod
    def _convert_tools(tools: List[Dict]) -> Optional[List[types.Tool]]:
        """
        Convert OpenAI-style tools to GenAI format.
        
        Args:
            tools: List of tool definitions in OpenAI format
                   [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]
        
        Returns:
            List of GenAI Tool objects
        """
        if not tools:
            return None
        
        function_declarations = []
        for tool in tools:
            if isinstance(tool, dict):
                fn = tool.get('function') if tool.get('type') == 'function' else tool
                if fn:
                    function_declarations.append(
                        types.FunctionDeclaration(
                            name=fn.get('name'),
                            description=fn.get('description', ''),
                            parameters=fn.get('parameters')
                        )
                    )
        
        return [types.Tool(function_declarations=function_declarations)] if function_declarations else None
    
    @staticmethod
    def _extract_tools_from_messages(messages: List[Dict[str, str]]) -> tuple:
        """
        Extract tool definitions from system message content and return cleaned system instruction.
        
        Parses tool definitions like:
        1. **tool_name** - description
        ```
        <tool_call>
        {"name": "tool_name", "arguments": {...}}
        </tool_call>
        ```
        
        Returns:
            Tuple of (tools_list, cleaned_system_instruction)
        """
        import re
        import json
        
        tools = []
        cleaned_system = None
        
        for m in messages:
            if m.get('role') == 'system':
                content = m.get('content', '')
                
                # Pattern to match tool definitions with nested JSON
                # N. **tool_name** - description
                # ```
                # <tool_call>
                # {...json...}
                # </tool_call>
                # ```
                pattern = r'\d+\.\s+\*\*(\w+)\*\*\s*-\s*([^\n]+)\n```\n<tool_call>\n(\{.*?\})\n</tool_call>\n```'
                
                matches = re.findall(pattern, content, re.DOTALL)
                
                for match in matches:
                    tool_name = match[0]
                    description = match[1].strip()
                    try:
                        # Parse the example to get parameter structure
                        example = json.loads(match[2])
                        arguments = example.get('arguments', {})
                        
                        # Build parameters schema from example
                        properties = {}
                        required = []
                        for key, value in arguments.items():
                            if isinstance(value, list):
                                properties[key] = {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": f"Parameter: {key}"
                                }
                            else:
                                properties[key] = {
                                    "type": "string",
                                    "description": f"Parameter: {key}"
                                }
                            required.append(key)
                        
                        tools.append({
                            'name': tool_name,
                            'description': description,
                            'parameters': {
                                "type": "object",
                                "properties": properties,
                                "required": required
                            }
                        })
                    except json.JSONDecodeError:
                        continue
                
                # Remove the ## Tools section from system instruction
                # Match from "## Tools" to the next "##" section or end
                tools_section_pattern = r'\n## Tools\n.*?(?=\n## |\Z)'
                cleaned_system = re.sub(tools_section_pattern, '', content, flags=re.DOTALL).strip()
        
        return tools, cleaned_system
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> str:
        """
        Generate a chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (if supported)
            max_tokens: Maximum tokens (if supported)
            stop: Stop sequences
            tools: List of tool definitions in OpenAI format
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        import json
        
        client = self.get_client()
        
        # Convert messages to GenAI format
        contents, system_instruction = self._convert_messages(messages)

        # Extract tools from messages if not explicitly provided
        extracted_tools = []
        cleaned_system = system_instruction
        if not tools:
            extracted_tools, cleaned_system = self._extract_tools_from_messages(messages)
            if cleaned_system:
                system_instruction = cleaned_system
            tools = extracted_tools
        
        # Convert tools to GenAI format
        genai_tools = self._convert_tools(tools) if tools else None
        
        # Build tool config
        tool_config = None
        if not genai_tools:
            # Explicitly disable function calling if no tools found
            tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode='NONE')
            )
        
        # Build generation config
        actual_temperature = temperature if temperature is not None else self.temperature
        actual_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=actual_temperature,
            max_output_tokens=actual_max_tokens,
            tools=genai_tools,
            tool_config=tool_config,
        )
        # Retry logic with exponential backoff
        base_sleep_time = 1
        for attempt in range(self.max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                
                # Extract content from response
                content = ''
                tool_calls = []
                
                if getattr(response, 'text', None):
                    content = response.text
                elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    for p in response.candidates[0].content.parts:
                        if getattr(p, 'text', None):
                            content += p.text
                        # Handle function calls
                        if getattr(p, 'function_call', None):
                            fc = p.function_call
                            tool_calls.append({
                                'name': getattr(fc, 'name', ''),
                                'arguments': getattr(fc, 'args', {}) or {}
                            })
                
                # If there are tool calls, format them as text response
                if tool_calls:
                    for tc in tool_calls:
                        tool_call_str = f'<tool_call>\n{{"name": "{tc["name"]}", "arguments": {json.dumps(tc["arguments"], ensure_ascii=False)}}}\n</tool_call>'
                        content += tool_call_str
                
                if content and content.strip():
                    return content.strip()
                    
            except Exception as e:
                print(f"GenAI API error (attempt {attempt + 1}/{self.max_retries}): {e}")
            
            if attempt < self.max_retries - 1:
                sleep_time = min(base_sleep_time * (2 ** attempt) + random.uniform(0, 1), 30)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
        
        return "Error: Failed to get response after all retries."


def create_genai_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    headers: Optional[Dict] = None,
    model_name: Optional[str] = None,
    api_version: str = "v1",
    **kwargs
) -> GenAIClient:
    """
    Factory function to create a GenAIClient.
    
    Args:
        base_url: API base URL
        api_key: API key
        headers: Custom headers
        model_name: Model name
        api_version: API version (default: v1)
        **kwargs: Additional configuration
        
    Returns:
        Configured GenAIClient instance
    """
    return GenAIClient(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        headers=headers,
        api_version=api_version,
        **kwargs
    )
