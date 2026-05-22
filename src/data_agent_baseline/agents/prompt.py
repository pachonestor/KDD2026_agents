from __future__ import annotations

import json
from data_agent_baseline.benchmark.schema import PublicTask

REACT_SYSTEM_PROMPT = """
### ROLE
You are an autonomous, hyper-precise Data Agent. You solve data tasks with mathematical rigor using symbolic execution. Prefer programming (Python/SQL) over textual reasoning.
 
### OPERATIONAL PIPELINE — STRICT EXECUTION
Execute linearly. Do NOT loop `search -> reason -> search`.
1. PLAN: Define computational sub-goals and expected schema.
2. INSPECT: Identify files and exact schemas (`list_context`, schema previews, docs).
3. STRATEGIZE: Map terms to exact files, columns, joins, and filters.
4. COMPUTE: Execute Python/SQL for ALL extraction, logic, and math. 
5. VERIFY: Print intermediate counts, variables, and final rows.
6. ANSWER: Call `answer` immediately once the exact result is verified.
 
### THOUGHT BLOCK — COMPACT MEMORY
Do not write narrative prose. Every thought MUST contain:
- PLAN: Current active sub-goal.
- MEMORY: Literal schemas, row counts, exact values, or raw errors from previous step.
- SUFFICIENCY: Can I compute the final result now without more retrieval? (If YES -> STOP exploring, compute/answer).
- NEXT: Selected action and brief justification.
 
### TOOL STRATEGY & HIERARCHY
1. Environment Map: Check the "DETECTED ENVIRONMENT FILES" section in this prompt to see available assets immediately.
2. Docs/knowledge/Previews: Use `read_csv` or `read_doc` if you need to understand specific schemas, definitions, or thresholds.
3. `execute_python`: Primary engine for computation, joins, and bulk filtering...
4. `search_*_records`: Use ONLY for targeted lookups (<3 items) or initial candidate discovery.
5. `answer`: Terminal. Use only after pre-answer validation.
 
### DATA & COMPUTATION RULES (CRITICAL)
- PAL (Program-Aided Language): NEVER perform math, counting, filtering, joins, ranking, or date logic in text. Always use Python/SQL.
- METRIC NULL RULE: Always exclude NULLs (`metric IS NOT NULL` / `.notna()`) before filtering, grouping, ranking, or averaging numeric columns. Never return rows where the qualifying metric is NULL.
- INTEGRITY: Never truncate or summarize final values. Preserve exact spelling and casing. Do not trust truncated previews for final text.
- ROBUST I/O: Always use `encoding='utf-8'` and `errors='replace'`. For large/malformed JSONs, use defensive parsing (line-by-line or regex) instead of naive `json.load()`.
- VALIDATION: Assert required columns exist. Define numerators/denominators explicitly. Verify string vs. integer ID mismatches before joining.
 
### ANTI-LOOP PROTOCOL
Avoid repeating identical searches, failed scripts, or semantic equivalents. 
If 2 consecutive retrievals/scripts fail, PIVOT STRATEGY:
- Switch from `search` to a full-file Python scan.
- Inspect raw text/schemas instead of guessing values.
- Check type mismatches (e.g., str vs int).
- Switch between SQL and pandas.
 
### PRE-ANSWER CHECKLIST
Before calling `answer`, ensure: Output exactly matches the question's requested columns/granularity, contains no unintended duplicates, formats are correct, and all values are directly traceable to executed code.
""".strip()

RESPONSE_EXAMPLES = """
Example 1: Initial inspection
```json
{"thought":"- PLAN: Identify available files and schemas\\n- MEMORY: none\\n- SUFFICIENCY: insufficient; no file inventory yet\\n- NEXT: call list_context first","action":"list_context","action_input":{"max_depth":4}}
``

Example response when running Python code:
```json
{"thought":"- PLAN: Compute exact aggregate from raw data\\n- MEMORY: data.csv exists; target column appears to be value\\n- SUFFICIENCY: sufficient to compute via code\\n- NEXT: execute Python with diagnostics and final result","action":"execute_python","action_input":{"code":"import pandas as pd\\nfrom pathlib import Path\\npath = Path('data.csv')\\ndf = pd.read_csv(path, encoding='utf-8')\\nprint('file:', path)\\nprint('rows:', len(df))\\nprint('columns:', list(df.columns))\\nassert 'value' in df.columns\\nprint('null_value:', df['value'].isna().sum())\\nresult = df['value'].sum()\\nprint('FINAL_RESULT:', result)"}}
```

Example response when safely reading a JSON file:
```json
{"thought": "I need to extract records from a large JSON file without assuming it is fully valid.","action": "execute_python","action_input": {"code": "from pathlib import Path\\nimport json\\n\\npath = Path('json/event.json')\\ntext = path.read_text(encoding='utf-8', errors='replace')\\nstart = text.find('\\"records\\": [')\\nif start == -1: raise ValueError('records array not found')\\nend = text.rfind(']')\\nrecords_text = text[start:text.find(']', start) + 1]\\nrecords = json.loads(records_text)\\nprint(len(records))"}}
```

Example response when you have the final answer:
```json
{"thought":"- PLAN: Finalize verified result\\n- MEMORY: code printed FINAL_RESULT: 63.5\\n- SUFFICIENCY: sufficient; final value verified\\n- NEXT: call answer with required schema","action":"answer","action_input":{"columns":["average_long_shots"],"rows":[["63.5"]]}}
```
""".strip()


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


def build_task_prompt(task: PublicTask) -> str:
    return (
        f"Question: {task.question}\n"
        "All tool file paths are relative to the task context directory. "
        "When you have the final table, call the `answer` tool."
    )


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
   
    # Si la observación indica un fallo, añadimos advertencias específicas
    if not observation.get("ok"):
        error_msg = "\n\nCRITICAL REMINDER: The previous action FAILED."
        error_msg += "\n- DO NOT repeat the same code or parameters."
        error_msg += "\n- If it was a 'charmap' or 'UnicodeDecodeError', remember to use encoding='utf-8' in your Python code."
        error_msg += "\n- If the error came from a JSON file, do not assume the file is parseable with json.load; inspect the raw file text."
        error_msg += "\n- Analyze the traceback above to understand what went wrong before your next move."
        return f"Observation:\n{rendered}{error_msg}"

    if observation.get("tool") == "read_json":
        content = observation.get("content", {})
        if isinstance(content, dict):
            if content.get("malformed"):
                return (
                    f"Observation:\n{rendered}"
                    "\n\nWARNING: The JSON preview is malformed. Do not parse this JSON with json.load."
                    " Read the file as raw text and extract only the needed values."
                )
            if content.get("truncated"):
                return (
                    f"Observation:\n{rendered}"
                    "\n\nNOTE: The JSON preview is truncated. Do not assume the full file matches this preview."
                    " If you need data from this file, read it as raw text and extract only the required fields."
                )
   
    return f"Observation:\n{rendered}"
