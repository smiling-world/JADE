# JSON Format Sources

Original JSON files containing model responses:

- [✓] **gpt_web_search**: `bizbench30_with_reports_gpt_web_search.json`
- [✓] **shopping_research**: `bizbench30_with_reports_shopping_research(1).json`

## Format

```json
[
  {
    "id": 1,
    "query": "...",
    "optimized_query": "...",
    "report": "...",
    "L1_primary_intent": "...",
    "L2_information_need": [...],
    "L3_constraints": [...],
    "timestamp": "...",
    "model": "..."
  }
]
```
