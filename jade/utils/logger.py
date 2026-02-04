"""
Logging utility for the verification agent.

Provides structured logging of LLM calls, tool executions,
and agent decisions for debugging and analysis.
"""

import json
import os
import datetime
import threading
from typing import Dict, Any, List, Optional, Union
from pathlib import Path


class AgentLogger:
    """
    Structured logger for agent execution traces.
    
    Logs all LLM calls, tool executions, and decisions to a JSON file
    for debugging and analysis purposes.
    
    Attributes:
        log_dir: Directory for log files
        session_id: Unique identifier for this logging session
        log_file: Path to the current log file
        entries: In-memory log entries
    """
    
    def __init__(
        self, 
        log_dir: str = "logs",
        session_id: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize the agent logger.
        
        Args:
            log_dir: Directory to store log files
            session_id: Optional custom session ID
            enabled: Whether logging is enabled
        """
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Generate session ID
        self.session_id = session_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"agent_{self.session_id}.json"
        
        # Thread safety
        self._lock = threading.Lock()
        self._thread_local = threading.local()
        
        # Initialize log structure
        self.log_data = {
            "session_id": self.session_id,
            "start_time": datetime.datetime.now().isoformat(),
            "end_time": None,
            "items": []
        }
    
    @property
    def current_item(self):
        """Thread-local current item."""
        return getattr(self._thread_local, 'current_item', None)
    
    @current_item.setter
    def current_item(self, value):
        self._thread_local.current_item = value
    
    @property
    def current_iteration(self):
        """Thread-local current iteration."""
        return getattr(self._thread_local, 'current_iteration', None)
    
    @current_iteration.setter
    def current_iteration(self, value):
        self._thread_local.current_iteration = value
    
    def start_item(self, item_id: int, description: str, source: Optional[str] = None):
        """Start logging a new verification item."""
        if not self.enabled:
            return
            
        self.current_item = {
            "item_id": item_id,
            "description": description,
            "source": source,
            "start_time": datetime.datetime.now().isoformat(),
            "end_time": None,
            "iterations": [],
            "final_result": None
        }
        self.current_iteration = None
    
    def end_item(self, result: Optional[Dict] = None):
        """End logging for the current item (thread-safe)."""
        if not self.enabled or not self.current_item:
            return
            
        self.current_item["end_time"] = datetime.datetime.now().isoformat()
        self.current_item["final_result"] = result
        
        # Thread-safe append to shared log_data
        with self._lock:
            self.log_data["items"].append(self.current_item)
            self._save_locked()
        
        self.current_item = None
    
    def start_iteration(self, iteration_num: int):
        """Start a new agent iteration."""
        if not self.enabled or not self.current_item:
            return
            
        self.current_iteration = {
            "iteration": iteration_num,
            "timestamp": datetime.datetime.now().isoformat(),
            "llm_call": None,
            "tool_calls": [],
            "decision": None
        }
    
    def end_iteration(self, decision: str):
        """End the current iteration."""
        if not self.enabled or not self.current_iteration:
            return
            
        self.current_iteration["decision"] = decision
        self.current_item["iterations"].append(self.current_iteration)
        self.current_iteration = None
    
    def log_llm_call(
        self,
        messages: List[Dict],
        response: str,
        parsed_tool_calls: Optional[List[Dict]] = None,
        error: Optional[str] = None
    ):
        """
        Log an LLM call and its response.
        
        Args:
            messages: Input messages sent to LLM
            response: Raw response from LLM
            parsed_tool_calls: Tool calls parsed from response
            error: Error message if call failed
        """
        if not self.enabled or not self.current_iteration:
            return
        
        # Truncate long messages for storage
        truncated_messages = []
        for msg in messages:
            truncated = {
                "role": msg.get("role"),
                "content": msg.get("content", "")[:5000] + ("..." if len(msg.get("content", "")) > 5000 else "")
            }
            truncated_messages.append(truncated)
        
        self.current_iteration["llm_call"] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "input_messages_count": len(messages),
            "input_messages": truncated_messages,
            "response": response[:10000] + ("..." if len(response) > 10000 else ""),
            "response_length": len(response),
            "parsed_tool_calls": parsed_tool_calls,
            "error": error
        }
    
    def log_tool_call(
        self,
        tool_name: str,
        args: Dict,
        result: str,
        success: bool = True,
        error: Optional[str] = None,
        duration_ms: Optional[int] = None,
        raw_result: Optional[Union[str, dict, list]] = None
    ):
        """
        Log a tool execution.
        
        Args:
            tool_name: Name of the tool called
            args: Arguments passed to the tool
            result: Result returned by the tool
            success: Whether the tool call succeeded
            error: Error message if failed
            duration_ms: Execution time in milliseconds
            raw_result: Raw result from the tool (for web_search and web_scraper)
        """
        if not self.enabled or not self.current_iteration:
            return
        
        tool_call_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "tool_name": tool_name,
            "arguments": args,
            "result": result[:10000] + ("..." if len(result) > 10000 else ""),
            "result_length": len(result),
            "success": success,
            "error": error,
            "duration_ms": duration_ms
        }
        
        # Add raw result if provided
        if raw_result is not None:
            if isinstance(raw_result, (dict, list)):
                # For JSON data, store as JSON string (truncated if too long)
                raw_json = json.dumps(raw_result, ensure_ascii=False, indent=2)
                tool_call_entry["raw_result"] = raw_json[:50000] + ("..." if len(raw_json) > 50000 else "")
                tool_call_entry["raw_result_length"] = len(raw_json)
            else:
                # For string data, store directly (truncated if too long)
                raw_str = str(raw_result)
                tool_call_entry["raw_result"] = raw_str[:50000] + ("..." if len(raw_str) > 50000 else "")
                tool_call_entry["raw_result_length"] = len(raw_str)
        
        self.current_iteration["tool_calls"].append(tool_call_entry)
    
    def log_decision(self, decision: str, reason: str):
        """Log an agent decision."""
        if not self.enabled or not self.current_iteration:
            return
        
        self.current_iteration["decision"] = {
            "action": decision,
            "reason": reason,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def finalize(self):
        """Finalize and save the log (thread-safe)."""
        if not self.enabled:
            return
        
        with self._lock:
            self.log_data["end_time"] = datetime.datetime.now().isoformat()
            self._save_locked()
    
    def _save_locked(self):
        """Save log data to file (must be called with lock held)."""
        if not self.enabled:
            return
            
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.log_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Failed to save log: {e}")
    
    def _save(self):
        """Save log data to file (thread-safe)."""
        if not self.enabled:
            return
        
        with self._lock:
            self._save_locked()
    
    def get_log_path(self) -> str:
        """Get the path to the current log file."""
        return str(self.log_file)


# Global logger instance
_logger: Optional[AgentLogger] = None


def get_logger() -> AgentLogger:
    """Get or create the global logger instance."""
    global _logger
    if _logger is None:
        _logger = AgentLogger()
    return _logger


def set_logger(logger: AgentLogger):
    """Set the global logger instance."""
    global _logger
    _logger = logger


def create_logger(
    log_dir: str = "logs", 
    enabled: bool = True,
    session_id: Optional[str] = None
) -> AgentLogger:
    """Create and set a new global logger."""
    global _logger
    _logger = AgentLogger(log_dir=log_dir, enabled=enabled, session_id=session_id)
    return _logger

