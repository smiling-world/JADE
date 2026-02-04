import argparse
import html
import json
import re
import os
from pathlib import Path
from typing import Any, Dict, List


def load_checklist(path: Path) -> tuple[List[Dict[str, Any]], str, str]:
    """
    Load checklist from JSON file.
    Returns: (checklist_items, query, report)
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle two possible formats:
    # 1. Direct array: [{"item_id": 0, ...}, ...]
    # 2. Object with metadata: {"query": "...", "report": "...", "checklist": [...]}
    if isinstance(data, list):
        return data, "", ""
    elif isinstance(data, dict):
        checklist_items = data.get("checklist", [])
        query = data.get("query", "")
        report = data.get("report", "")
        return checklist_items, query, report
    else:
        return [], "", ""


def load_log(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_rows(
    checklist: List[Dict[str, Any]], log_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    items = {item["item_id"]: item for item in log_data.get("items", [])}

    rows: List[Dict[str, Any]] = []
    for v in checklist:
        vid = v.get("item_id")
        item = items.get(vid)
        if item is None:
            conclusion = "missing"
            confidence: Any = ""
            reason = "No matching item_id in log."
        else:
            fr = item.get("final_result") or {}
            conclusion = fr.get("conclusion", "unknown")
            confidence = fr.get("confidence", "")
            reason = fr.get("reason", "")

        rows.append(
            {
                "item_id": vid,
                "type": v.get("type", ""),
                "category": v.get("category", ""),
                "source": v.get("source", ""),
                "description": v.get("description", ""),
                "weight": v.get("weight", ""),
                "conclusion": conclusion,
                "confidence": confidence,
                "reason": reason,
            }
        )
    return rows


def render_html(
    rows: List[Dict[str, Any]], 
    title: str, 
    query: str = "", 
    report: str = ""
) -> str:
    summary = {
        "total": len(rows),
        "yes": sum(1 for r in rows if r["conclusion"] == "yes"),
        "no": sum(1 for r in rows if r["conclusion"] == "no"),
        "na": sum(1 for r in rows if r["conclusion"] in ("n_a", "N_A", "na", "NA", "n/a", "N/A")),
        "missing": sum(1 for r in rows if r["conclusion"] == "missing"),
    }

    html_rows: List[str] = []
    for r in rows:
        concl = r["conclusion"]
        concl_lower = str(concl).lower().replace("/", "_")
        if concl_lower == "yes":
            bg = "#e6ffed"
        elif concl_lower == "no":
            bg = "#ffecec"
        elif concl_lower in ("n_a", "na"):
            bg = "#fff8e6"  # Light yellow/amber for N/A
        elif concl_lower == "missing":
            bg = "#f5f5f5"
        else:
            bg = "#fdfdfd"

        # Format source as link if it's a URL
        source_cell = r.get('source') or ''
        if source_cell and (source_cell.startswith('http://') or source_cell.startswith('https://')):
            source_html = f'<a href="{html.escape(source_cell)}" target="_blank">{html.escape(source_cell)}</a>'
        else:
            source_html = html.escape(str(source_cell))

        html_rows.append(
            f"""
      <tr style='background:{bg}'>
        <td>{r['item_id']}</td>
        <td>{html.escape(str(r['type']))}</td>
        <td>{html.escape(str(r['category']))}</td>
        <td>{source_html}</td>
        <td>{html.escape(str(r['description']))}</td>
        <td>{r['weight']}</td>
        <td class='conclusion-{concl}'>{html.escape(str(r['conclusion']))}</td>
        <td>{html.escape(str(r['confidence']))}</td>
        <td>{html.escape(str(r['reason']))}</td>
      </tr>
    """
        )

    # Build query and report sections
    query_section = ""
    if query:
        query_section = f"""
  <div class='content-section'>
    <h2>📋 User Query</h2>
    <div class='content-box query-box'>
      {html.escape(query)}
    </div>
  </div>
"""

    report_section = ""
    if report:
        # Escape report content and store in data attribute for client-side markdown rendering
        report_escaped = html.escape(report)
        report_section = f"""
  <div class='content-section'>
    <h2>Report</h2>
    <div class='content-box report-box' id='report-content' data-markdown='{report_escaped}'>
      {report_escaped}
    </div>
  </div>
"""

    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='UTF-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1.0'>
  <title>{html.escape(title)}</title>
  <script src='https://cdn.jsdelivr.net/npm/marked/marked.min.js'></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ 
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; 
      margin: 0;
      padding: 24px;
      background: #f5f5f5;
      line-height: 1.6;
    }}
    .container {{
      max-width: 1400px;
      margin: 0 auto;
      background: white;
      padding: 32px;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    h1 {{ 
      font-size: 28px; 
      margin-bottom: 8px; 
      color: #333;
      border-bottom: 3px solid #4a90e2;
      padding-bottom: 12px;
    }}
    .summary {{ 
      margin-bottom: 24px; 
      padding: 12px 16px;
      background: #f8f9fa;
      border-radius: 6px;
      font-size: 14px;
      color: #555;
    }}
    .summary strong {{
      color: #333;
      margin-right: 4px;
    }}
    .content-section {{
      margin-bottom: 32px;
    }}
    .content-section h2 {{
      font-size: 20px;
      margin-bottom: 12px;
      color: #333;
      font-weight: 600;
    }}
    .content-box {{
      padding: 16px 20px;
      border-radius: 6px;
      border: 1px solid #e1e4e8;
      background: #fafbfc;
      white-space: pre-wrap;
      word-wrap: break-word;
      font-size: 14px;
      line-height: 1.7;
      max-height: 500px;
      overflow-y: auto;
    }}
    .query-box {{
      background: #e8f4f8;
      border-color: #b3d9e6;
      color: #1a365d;
    }}
    .report-box {{
      background: #f8f9fa;
      border-color: #d1d5db;
      color: #374151;
      white-space: normal;
    }}
    .report-box a {{
      color: #2563eb;
      text-decoration: none;
    }}
    .report-box a:hover {{
      text-decoration: underline;
    }}
    .report-box h1, .report-box h2, .report-box h3, .report-box h4, .report-box h5, .report-box h6 {{
      margin-top: 1em;
      margin-bottom: 0.5em;
      font-weight: 600;
      color: #1a1a1a;
    }}
    .report-box h1 {{ font-size: 1.5em; }}
    .report-box h2 {{ font-size: 1.3em; }}
    .report-box h3 {{ font-size: 1.1em; }}
    .report-box p {{
      margin: 0.5em 0;
    }}
    .report-box ul, .report-box ol {{
      margin: 0.5em 0;
      padding-left: 2em;
    }}
    .report-box li {{
      margin: 0.25em 0;
    }}
    .report-box code {{
      background: #f1f3f5;
      padding: 0.2em 0.4em;
      border-radius: 3px;
      font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
      font-size: 0.9em;
    }}
    .report-box pre {{
      background: #f1f3f5;
      padding: 1em;
      border-radius: 4px;
      overflow-x: auto;
      margin: 0.5em 0;
    }}
    .report-box pre code {{
      background: transparent;
      padding: 0;
    }}
    .report-box blockquote {{
      border-left: 4px solid #d1d5db;
      padding-left: 1em;
      margin: 0.5em 0;
      color: #6b7280;
    }}
    .report-box table {{
      border-collapse: collapse;
      margin: 0.5em 0;
      width: 100%;
    }}
    .report-box table th, .report-box table td {{
      border: 1px solid #d1d5db;
      padding: 0.5em;
    }}
    .report-box table th {{
      background: #e5e7eb;
      font-weight: 600;
    }}
    table {{ 
      border-collapse: collapse; 
      width: 100%; 
      font-size: 13px;
      margin-top: 24px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    th, td {{ 
      border: 1px solid #ddd; 
      padding: 10px 12px; 
      vertical-align: top; 
      text-align: left;
    }}
    th {{ 
      background: #4a90e2; 
      color: white;
      position: sticky; 
      top: 0; 
      z-index: 10;
      font-weight: 600;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    td {{
      background: white;
    }}
    tr:nth-child(even) td {{
      background: #fafbfc;
    }}
    tbody tr:hover td {{
      background: #f0f7ff;
    }}
    table a {{
      color: #2563eb;
      text-decoration: none;
      word-break: break-all;
    }}
    table a:hover {{
      text-decoration: underline;
    }}
    .conclusion-yes {{
      color: #28a745;
      font-weight: 600;
    }}
    .conclusion-no {{
      color: #dc3545;
      font-weight: 600;
    }}
    .conclusion-n_a, .conclusion-na {{
      color: #d4a017;
      font-weight: 600;
      font-style: italic;
    }}
    .conclusion-missing {{
      color: #6c757d;
      font-style: italic;
    }}
  </style>
</head>
<body>
  <div class='container'>
    <h1>{html.escape(title)}</h1>
    <div class='summary'>
      <strong>Total:</strong> {summary['total']} &nbsp; | &nbsp;
      <strong>Yes:</strong> <span style='color: #28a745;'>{summary['yes']}</span> &nbsp; | &nbsp;
      <strong>No:</strong> <span style='color: #dc3545;'>{summary['no']}</span> &nbsp; | &nbsp;
      <strong>N/A:</strong> <span style='color: #d4a017;'>{summary['na']}</span> &nbsp; | &nbsp;
      <strong>Missing:</strong> <span style='color: #6c757d;'>{summary['missing']}</span>
    </div>
{query_section}{report_section}
    <div class='content-section'>
      <h2>✅ Verification Results</h2>
      <table>
        <thead>
          <tr>
            <th>Item ID</th>
            <th>Type</th>
            <th>Category</th>
            <th>Source</th>
            <th>Description</th>
            <th>Weight</th>
            <th>Conclusion</th>
            <th>Confidence</th>
            <th>Reason</th>
          </tr>
        </thead>
        <tbody>
          {''.join(html_rows)}
        </tbody>
      </table>
    </div>
  </div>
  <script>
    // Render markdown content
    const reportEl = document.getElementById('report-content');
    if (reportEl && typeof marked !== 'undefined') {{
      const markdownText = reportEl.getAttribute('data-markdown');
      if (markdownText) {{
        reportEl.innerHTML = marked.parse(markdownText);
      }}
    }}
  </script>
</body>
</html>
"""


def main() -> None:
    '''
    Usage:
        python report_verification_html.py --checklist output/report_specific_checklist/shopping_research/5_report_specific.json --log log.json
    '''

    parser = argparse.ArgumentParser(
        description=(
            "Match checklist item_id with agent log item_id and "
            "generate an HTML visualization of verification results."
        )
    )
    parser.add_argument(
        "--checklist",
        required=True,
        help="Path to checklist JSON (with item_id).",
    )
    parser.add_argument(
        "--log",
        required=True,
        help="Path to agent log JSON (with items[].item_id and final_result).",
    )
    parser.add_argument(
        "--output",
        required=False,
        help="Output directory path. HTML filename will be auto-generated from checklist filename.",
    )
    parser.add_argument(
        "--title",
        default="Verification Results",
        help="Title used in the HTML page.",
    )

    args = parser.parse_args()

    checklist_path = Path(args.checklist)
    log_path = Path(args.log)
    output_dir = Path(args.output) if args.output else checklist_path.parent / 'vis'
    os.makedirs(output_dir, exist_ok=True)

    # Generate HTML filename from checklist filename
    checklist_stem = checklist_path.stem  # filename without extension
    html_filename = f"{checklist_stem}_visual.html"
    output_path = output_dir / html_filename

    checklist_items, query, report = load_checklist(checklist_path)
    log_data = load_log(log_path)
    rows = build_rows(checklist_items, log_data)
    html_content = render_html(rows, args.title, query, report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    print(f"Wrote HTML to: {output_path}")


if __name__ == "__main__":
    main()


