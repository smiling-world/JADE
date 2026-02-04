# Cache Format Sources

Original cache directories containing model responses:

## With Tool

**Directory**: `cache/0105`

**Models** (10):
- `anthropic_claude-opus-4.5/`
- `anthropic_claude-sonnet-4.5/`
- `deepseek_deepseek-v3.2/`
- `doubao-seed-1-6-250615/`
- `gemini3-pro-preview/`
- `meta-llama_llama-4-maverick/`
- `openai_gpt-4.1/`
- `openai_gpt-5.2/`
- `qwen_qwen3-235b-a22b-2507/`
- `qwen_qwen3-max/`

## No Tool

**Directory**: `cache/0105_notool`

**Models** (10):
- `anthropic_claude-opus-4.5/`
- `anthropic_claude-sonnet-4.5/`
- `deepseek_deepseek-v3.2/`
- `doubao-seed-1-6-250615/`
- `gemini3-pro-preview/`
- `meta-llama_llama-4-maverick/`
- `openai_gpt-4.1/`
- `openai_gpt-5.2/`
- `qwen_qwen3-235b-a22b-2507/`
- `qwen_qwen3-max/`

## Structure

```
model_name/
├── config.json    # Model configuration
├── results.jsonl  # Results in JSONL format
└── md/            # Markdown reports for each query
    ├── 1.md
    ├── 2.md
    └── ...
```

## JSONL Format

Each line in `results.jsonl` is a JSON object:

```json
{
  "id": 1,
  "query": "...",
  "optimized_query": "...",
  "response": "...",
  "model": "...",
  "duration_seconds": 16.93,
  "timestamp": "2026-01-06T16:54:44.345367",
  "L1_primary_intent": "...",
  "L2_information_need": [...],
  "L3_constraints": [...],
  "success": true,
  "error": null
}
```
