# Experiment 1: Multi-Model Evaluation Data

## Overview
This directory contains organized evaluation data from testing 30 benchmark queries across multiple models with different experimental conditions.

## Data Organization

```
exp1/
├── README.md                          # This file
├── input/                             # Input files ready for eval_pipeline_genai.py
│   ├── gpt_web_search.json           # GPT with web search
│   ├── shopping_research.json        # Shopping Research model
│   ├── claude-opus-4.5_with_tool.json
│   ├── claude-opus-4.5_no_tool.json
│   ├── claude-sonnet-4.5_with_tool.json
│   ├── claude-sonnet-4.5_no_tool.json
│   ├── deepseek-v3.2_with_tool.json
│   ├── deepseek-v3.2_no_tool.json
│   ├── doubao-seed-1-6-250615_with_tool.json
│   ├── doubao-seed-1-6-250615_no_tool.json
│   ├── gemini3-pro-preview_with_tool.json
│   ├── gemini3-pro-preview_no_tool.json
│   ├── gpt-4.1_with_tool.json
│   ├── gpt-4.1_no_tool.json
│   ├── gpt-5.2_with_tool.json
│   ├── gpt-5.2_no_tool.json
│   ├── llama-4-maverick_with_tool.json
│   ├── llama-4-maverick_no_tool.json
│   ├── qwen3-235b-a22b-2507_with_tool.json
│   ├── qwen3-235b-a22b-2507_no_tool.json
│   ├── qwen3-max_with_tool.json
│   └── qwen3-max_no_tool.json
├── metadata/                          # Experiment metadata
│   ├── models.json                   # Model configurations
│   └── experiments.json              # Experiment conditions
└── source/                           # Original data references
    ├── json_format.md                # Links to original JSON files
    └── cache_format.md               # Links to original cache directories

```

## Input File Format

Each input file in `input/` follows the format required by `eval_pipeline_genai.py`:

```json
[
  {
    "id": 1,
    "query": "Original user query text",
    "report": "Model's generated response",
    "L1_primary_intent": "product_discovery",
    "L2_information_need": ["trending_analysis", "platform_data"],
    "L3_constraints": [],
    "model": "model_name",
    "experiment": "with_tool|no_tool",
    "timestamp": "2026-01-06T16:54:44.345367",
    "duration_seconds": 16.93
  }
]
```

## Metadata Format

### models.json
Contains model configurations and capabilities:

```json
{
  "gpt_web_search": {
    "display_name": "GPT Web Search",
    "provider": "OpenAI",
    "capabilities": ["web_search"],
    "source_type": "json"
  },
  "claude-opus-4.5": {
    "display_name": "Claude Opus 4.5",
    "provider": "Anthropic",
    "capabilities": ["tool_use", "web_search"],
    "source_type": "cache"
  }
}
```

### experiments.json
Documents experimental conditions:

```json
{
  "with_tool": {
    "description": "Model with tool/web search capabilities enabled",
    "date": "2026-01-05",
    "config": {
      "use_optimized": true,
      "stream": true,
      "timeout": 600.0
    }
  },
  "no_tool": {
    "description": "Model without tool/web search (baseline)",
    "date": "2026-01-05",
    "config": {
      "use_optimized": true,
      "stream": true,
      "timeout": 600.0
    }
  }
}
```
