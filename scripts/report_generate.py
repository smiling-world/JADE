#!/usr/bin/env python3
"""
Batch evaluation script for running queries against different models.

Usage:
    # Using command line arguments:
    python scripts/report_generate.py --model deepseek/deepseek-v3.2 --threads 4 --dataset core
    
    # Using config file:
    python scripts/report_generate.py --config configs/my_eval.yaml
    
    # Config file + override specific args:
    python scripts/report_generate.py --config configs/my_eval.yaml --model openai/gpt-4.1 -n 5

Arguments:
    --config: Path to YAML config file (command line args override config file values)
    --model: Model name (e.g., deepseek/deepseek-v3.2, openai/gpt-4.1)
    --threads: Number of concurrent threads (default: 4)
    --dataset: 'core' for 30 queries or 'full' for 150 queries (default: core)
    --output-dir: Output directory for results (saves directly, auto-resumes)
    --use-optimized: Use optimized_query instead of original query (default: False)
    --limit / -n: Limit the number of NEW queries to run (skips completed ones)
    --stream: Enable streaming output mode (default: False)
    --timeout: Request timeout in seconds (default: 180)
    --verbose / -v: Print streaming output in real-time (best with --threads 1)
    --prefix: Prefix to prepend to every query (e.g., instructions to answer directly)
"""

import os
import json
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# File write lock for thread-safe JSONL appending
jsonl_lock = Lock()


def load_config_file(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required to use config files. "
            "Install it with: pip install pyyaml"
        )
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config or {}


def merge_args_with_config(args: argparse.Namespace, file_config: dict) -> argparse.Namespace:
    """Merge command line args with config file. CLI args take precedence."""
    # Map config file keys to argparse attribute names
    key_mapping = {
        "model": "model",
        "threads": "threads",
        "dataset": "dataset",
        "output_dir": "output_dir",
        "use_optimized": "use_optimized",
        "limit": "limit",
        "stream": "stream",
        "timeout": "timeout",
        "verbose": "verbose",
        "prefix": "prefix",
    }
    
    # Default values to check if user explicitly set them
    defaults = {
        "model": "deepseek/deepseek-v3.2",
        "threads": 4,
        "dataset": "core",
        "output_dir": "output/report_generate",
        "use_optimized": False,
        "limit": None,
        "stream": False,
        "timeout": 180.0,
        "verbose": False,
        "prefix": "",
    }
    
    for config_key, arg_key in key_mapping.items():
        if config_key in file_config:
            current_value = getattr(args, arg_key)
            default_value = defaults.get(arg_key)
            
            # Only use config value if CLI arg is at default
            # (meaning user didn't explicitly set it)
            if current_value == default_value:
                setattr(args, arg_key, file_config[config_key])
    
    return args


def load_completed_ids(jsonl_path: Path) -> set[int]:
    """Load IDs of already completed queries from JSONL file."""
    completed_ids = set()
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        if record.get("success", False):
                            completed_ids.add(record["id"])
                    except json.JSONDecodeError:
                        continue
    return completed_ids


def load_dataset(dataset_type: str) -> list[dict]:
    """Load dataset based on type (core or full)."""
    data_dir = Path(__file__).parent.parent
    
    if dataset_type == "core":
        file_path = data_dir / "bizbench_core.json"
    else:
        file_path = data_dir / "bizbench.json"
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_api_config(model: str, config: dict = None) -> dict:
    """Get API configuration based on model name or config.
    
    Returns:
        dict with keys: base_url, api_key_env, tool_type
    """
    config = config or {}
    
    # Check if explicitly configured
    if "base_url" in config:
        return {
            "base_url": config["base_url"],
            "api_key_env": config.get("api_key_env", "OPENROUTER_API_KEY"),
            "tool_type": config.get("tool_type", "web_search_preview"),
        }
    
    # Auto-detect based on model name
    model_lower = model.lower()
    if "doubao" in model_lower:
        return {
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "api_key_env": "ARK_API_KEY",
            "tool_type": "web_search",
        }
    else:
        # Default to OpenRouter
        return {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "tool_type": "web_search_preview",
        }


def create_client(model: str, timeout: float = 180.0, config: dict = None) -> OpenAI:
    """Create OpenAI client configured for the specified model.
    
    Args:
        model: Model name (used to determine API endpoint)
        timeout: Request timeout in seconds
        config: Optional configuration dict with base_url, api_key_env, etc.
    """
    api_config = get_api_config(model, config)
    api_key = os.environ.get(api_config["api_key_env"], "")
    
    if not api_key:
        raise ValueError(
            f"API key not found in environment variable: {api_config['api_key_env']}. "
            f"Please set it before running the script."
        )
    
    return OpenAI(
        base_url=api_config["base_url"],
        api_key=api_key,
        timeout=timeout,
    )


def run_single_query(
    client: OpenAI,
    item: dict,
    model: str,
    use_optimized: bool = False,
    stream: bool = False,
    verbose: bool = False,
    prefix: str = "",
    prefix_map: dict = None,
    tool_type: str = "web_search_preview",
) -> dict:
    """Run a single query and return the result.
    
    Args:
        client: OpenAI client instance
        item: Query item from dataset
        model: Model name to use
        use_optimized: Whether to use optimized_query field
        stream: Whether to use streaming mode
        verbose: Whether to print streaming output in real-time
        prefix: Prefix to prepend to query (e.g., instructions to not ask questions)
        prefix_map: Dictionary mapping language codes to prefixes (e.g., {"zh": "...", "en": "..."})
        tool_type: Type of tool to use (e.g., "web_search_preview" or "web_search")
    """
    query_text = item.get("optimized_query", item["query"]) if use_optimized else item["query"]
    
    # Select prefix based on language if prefix_map is provided
    selected_prefix = ""
    if prefix_map:
        language = item.get("language", "en")  # Default to "en" if language not specified
        selected_prefix = prefix_map.get(language, prefix_map.get("en", ""))
    elif prefix:
        selected_prefix = prefix
    
    # Add prefix if provided
    if selected_prefix:
        query_text = f"{selected_prefix}\n\n{query_text}"
    
    start_time = datetime.now()
    
    try:
        if stream:
            # Streaming mode
            full_text_chunks = []
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Query ID {item['id']}:")
                print(f"{'='*60}")
            
            with client.responses.stream(
                model=model,
                input=query_text,
                tools=[{"type": tool_type}],
            ) as response_stream:
                for event in response_stream:
                    if event.type == "response.output_text.delta":
                        full_text_chunks.append(event.delta)
                        if verbose:
                            print(event.delta, end="", flush=True)
            
            if verbose:
                print(f"\n{'='*60}\n")
            
            response_text = "".join(full_text_chunks)
        else:
            # Non-streaming mode
            response = client.responses.create(
                model=model,
                input=query_text,
                tools=[{"type": tool_type}],
            )
            response_text = response.output_text
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return {
            "success": True,
            "id": item["id"],
            "query": item["query"],
            "optimized_query": item.get("optimized_query", ""),
            "used_query": query_text,
            "response": response_text,
            "model": model,
            "duration_seconds": duration,
            "timestamp": end_time.isoformat(),
            "L1_primary_intent": item.get("L1_primary_intent", ""),
            "L2_information_need": item.get("L2_information_need", []),
            "L3_constraints": item.get("L3_constraints", []),
            "error": None,
        }
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return {
            "success": False,
            "id": item["id"],
            "query": item["query"],
            "optimized_query": item.get("optimized_query", ""),
            "used_query": query_text,
            "response": None,
            "model": model,
            "duration_seconds": duration,
            "timestamp": end_time.isoformat(),
            "L1_primary_intent": item.get("L1_primary_intent", ""),
            "L2_information_need": item.get("L2_information_need", []),
            "L3_constraints": item.get("L3_constraints", []),
            "error": str(e),
        }


def save_result_jsonl(result: dict, jsonl_path: Path, config: dict):
    """Append result to JSONL file (thread-safe)."""
    record = {
        "config": config,
        **result,
    }
    
    with jsonl_lock:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_result_md(result: dict, md_dir: Path, config: dict):
    """Save response as markdown file."""
    md_path = md_dir / f"{result['id']}.md"
    
    content = f"""# Query {result['id']}

## Configuration
- **Model**: {config['model']}
- **Dataset**: {config['dataset']}
- **Use Optimized Query**: {config['use_optimized']}
- **Stream Mode**: {config.get('stream', False)}
- **Timestamp**: {result['timestamp']}
- **Duration**: {result['duration_seconds']:.2f}s
- **Success**: {result['success']}

## Metadata
- **L1 Primary Intent**: {result['L1_primary_intent']}
- **L2 Information Need**: {', '.join(result['L2_information_need']) if result['L2_information_need'] else 'None'}
- **L3 Constraints**: {', '.join(result['L3_constraints']) if result['L3_constraints'] else 'None'}

## Query
```
{result['used_query']}
```

## Response
{result['response'] if result['response'] else f"**Error**: {result['error']}"}
"""
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)


def process_item(
    item: dict,
    model: str,
    use_optimized: bool,
    jsonl_path: Path,
    md_dir: Path,
    config: dict,
    stream: bool = False,
    timeout: float = 180.0,
    verbose: bool = False,
    prefix: str = "",
    prefix_map: dict = None,
) -> dict:
    """Process a single item: run query and save results."""
    api_config = get_api_config(model, config)
    client = create_client(model, timeout=timeout, config=config)
    result = run_single_query(
        client, item, model, use_optimized, 
        stream=stream, verbose=verbose, prefix=prefix, prefix_map=prefix_map,
        tool_type=api_config["tool_type"]
    )
    
    # Save results
    save_result_jsonl(result, jsonl_path, config)
    save_result_md(result, md_dir, config)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation script for running queries against different models."
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML config file (CLI args override config file values)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek/deepseek-v3.2",
        help="Model name (e.g., deepseek/deepseek-v3.2, openai/gpt-4.1)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of concurrent threads (default: 4)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["core", "full"],
        default="core",
        help="'core' for 30 queries or 'full' for 150 queries (default: core)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/report",
        help="Output directory for results (default: output/report)",
    )
    parser.add_argument(
        "--use-optimized",
        action="store_true",
        help="Use optimized_query instead of original query",
    )
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        help="Limit the number of queries to run (useful for testing)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming output mode",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Request timeout in seconds (default: 180)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print streaming output in real-time (best with --threads 1)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix to prepend to every query (e.g., instructions to answer directly)",
    )
    
    args = parser.parse_args()
    
    # Load config file if specified
    file_config = {}
    if args.config:
        print(f"Loading config from: {args.config}")
        file_config = load_config_file(args.config)
        args = merge_args_with_config(args, file_config)
    
    # Setup output directory (directly use output_dir, no timestamped subdirectory)
    run_dir = Path(args.output_dir)
    md_dir = run_dir / "md"
    
    run_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)
    
    jsonl_path = run_dir / "results.jsonl"
    config_path = run_dir / "config.json"
    
    # Always check for completed IDs (auto-resume)
    completed_ids = load_completed_ids(jsonl_path)
    if completed_ids:
        print(f"Found {len(completed_ids)} completed queries, will skip them")
    
    # Get API configuration for the model
    api_config = get_api_config(args.model, file_config)
    
    # Configuration object for logging
    config = {
        "model": args.model,
        "threads": args.threads,
        "dataset": args.dataset,
        "use_optimized": args.use_optimized,
        "stream": args.stream,
        "timeout": args.timeout,
        "verbose": args.verbose,
        "prefix": args.prefix,
        "base_url": api_config["base_url"],
        "api_key_env": api_config["api_key_env"],
        "tool_type": api_config["tool_type"],
    }
    
    # Merge any API config from file_config if present
    if args.config and file_config:
        if "base_url" in file_config:
            config["base_url"] = file_config["base_url"]
        if "api_key_env" in file_config:
            config["api_key_env"] = file_config["api_key_env"]
        if "tool_type" in file_config:
            config["tool_type"] = file_config["tool_type"]
    
    # Save/update config to file
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, ensure_ascii=False, indent=2, fp=f)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset)
    total_in_dataset = len(data)
    print(f"Loaded {total_in_dataset} queries")
    
    # Filter out completed queries (for resume mode)
    if completed_ids:
        data = [item for item in data if item["id"] not in completed_ids]
        print(f"Skipping {len(completed_ids)} completed queries, {len(data)} remaining")
    
    # Apply limit if specified
    if args.limit is not None and args.limit > 0:
        data = data[:args.limit]
        print(f"Limiting to {len(data)} queries (--limit {args.limit})")
    
    print(f"\n{'='*60}")
    print(f"Run Configuration:")
    print(f"  Model: {config['model']}")
    print(f"  API Base URL: {config.get('base_url', 'N/A')}")
    print(f"  API Key Env: {config.get('api_key_env', 'N/A')}")
    print(f"  Tool Type: {config.get('tool_type', 'N/A')}")
    print(f"  Threads: {args.threads}")
    print(f"  Dataset: {args.dataset} ({total_in_dataset} total)")
    print(f"  Queries to run: {len(data)}")
    if completed_ids:
        print(f"  Already completed: {len(completed_ids)}")
    if args.limit:
        print(f"  Limit: {args.limit}")
    print(f"  Use Optimized Query: {args.use_optimized}")
    print(f"  Stream Mode: {config['stream']}")
    print(f"  Verbose: {config.get('verbose', False)}")
    print(f"  Timeout: {config['timeout']}s")
    prefix_config = config.get("prefix", "")
    if prefix_config:
        if isinstance(prefix_config, dict):
            lang_list = ", ".join(prefix_config.keys())
            print(f"  Prefix (multi-language): {lang_list}")
        else:
            print(f"  Prefix: {prefix_config[:50]}{'...' if len(prefix_config) > 50 else ''}")
    print(f"  Output Directory: {run_dir}")
    print(f"{'='*60}\n")
    
    # Early exit if no queries to run
    if len(data) == 0:
        print("No queries to run. All queries may have been completed already.")
        return
    
    # Run batch processing
    success_count = 0
    error_count = 0
    total_duration = 0.0
    
    # Handle prefix configuration (can be string or dict)
    prefix_config = config.get("prefix", "")
    prefix_map = prefix_config if isinstance(prefix_config, dict) else None
    prefix_str = prefix_config if isinstance(prefix_config, str) else ""
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(
                process_item,
                item,
                config["model"],
                args.use_optimized,
                jsonl_path,
                md_dir,
                config,
                stream=config["stream"],
                timeout=config["timeout"],
                verbose=config.get("verbose", False),
                prefix=prefix_str,
                prefix_map=prefix_map,
            ): item
            for item in data
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            item = futures[future]
            try:
                result = future.result()
                total_duration += result["duration_seconds"]
                
                if result["success"]:
                    success_count += 1
                    status = "✓"
                else:
                    error_count += 1
                    status = "✗"
                
                print(
                    f"[{i}/{len(data)}] {status} ID {result['id']:3d} | "
                    f"{result['duration_seconds']:.2f}s | "
                    f"{result['L1_primary_intent']}"
                )
                
            except Exception as e:
                error_count += 1
                print(f"[{i}/{len(data)}] ✗ ID {item['id']:3d} | Error: {e}")
    
    # Summary
    queries_run = success_count + error_count
    print(f"\n{'='*60}")
    print(f"Run Complete!")
    print(f"  Queries Run: {queries_run}")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total Duration: {total_duration:.2f}s")
    if queries_run > 0:
        print(f"  Avg Duration: {total_duration/queries_run:.2f}s per query")
    if completed_ids:
        print(f"  Previously Completed: {len(completed_ids)}")
    print(f"  Total Completed: {len(completed_ids) + success_count}")
    print(f"\nResults saved to:")
    print(f"  JSONL: {jsonl_path}")
    print(f"  Markdown: {md_dir}")
    print(f"  Config: {run_dir / 'config.json'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

