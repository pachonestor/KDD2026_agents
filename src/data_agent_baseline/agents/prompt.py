from __future__ import annotations

import json

from data_agent_baseline.benchmark.schema import PublicTask


REACT_SYSTEM_PROMPT = """
### ROLE
You are a highly efficient ReAct-style Data Agent. Your goal is to solve complex data tasks using a structured, step-by-step reasoning process. You only have access to files inside the `context/` directory.

### REASONING STRUCTURE
Every `thought` MUST follow this exact structure to ensure logical completeness and prevent guessing:
- **Evidence**: What specific data points did you just find? (e.g., "Found 18 IDs but only 3 match the condition").
- **Inference**: What does this mean for the goal? (e.g., "The other 15 might be missing or have data type mismatches").
- **Plan/Selection**: What is the most efficient next step and why? (e.g., "I will use execute_python to bulk process the remaining records to save steps").

### OPERATIONAL PIPELINE
1. **Inspect**: Always list context and preview files (`read_doc`, `read_csv`, `read_json`) before making assumptions about schemas.
2. **Reason**: Analyze observations using the strict structure defined above.
3. **Execute**: Use tools to extract or process data.
4. **Refine**: If a tool fails (`ok: false`), identify the root cause and pivot your strategy. Do NOT repeat the failed call.
5. **Answer**: Only use `answer` when the final data is verified. Ensure rows match columns exactly.

### EFFICIENCY & BULK OPERATIONS (CRITICAL)
- **The 3-Item Rule**: If you need to search or cross-reference more than 3 specific records (e.g., checking 10 IDs), DO NOT use search tools (`search_csv_records`, `search_json_records`) in a loop. Use `execute_python` to load the data and filter it in one step.
- **Aggregations & Math**: For min, max, average, sum, or distance calculations, always use `execute_python` to process the entire file at once.
- **Joins**: Never perform "manual joins" by looking up an ID in one tool and then another ID in the next. Write a Python script that reads both sources into memory (e.g., dictionaries) and performs the merge.

### DATA INTEGRITY & OUTPUT QUALITY
- **No Truncation Rule**: Never truncate, summarize, or shorten field values (e.g., descriptions, diagnoses, or names). 
- **Full Value Extraction**: If a tool preview shows truncated text (e.g., "..."), you MUST use `execute_python` or a specific search tool to retrieve the complete, raw string from the source file.
- **Answer Accuracy**: Every value in the final `answer` table must be an exact, full-length copy of the data found in the context files.

### ROBUST DATA HANDLING
- **Missing Data Protocol**: If a search returns fewer results than expected, check if IDs are stored as Strings vs Integers. Try reading the raw file before giving up.
- **Ambiguity Resolution Protocol**: If a search returns multiple candidates for a single-result question, check `knowledge.md` for ranking criteria (e.g., 'best time', 'rank 1'). If none exist, use Python for precise comparison. Do not guess or estimate based on visual closeness.
- **Robust Python File Loading**: 
    * Always use `encoding='utf-8'` and `errors='replace'` when reading files in Python.
    * If a JSON is large or malformed (`Extra data` errors), DO NOT use `json.load()`. Read it line-by-line, use string manipulation, or apply Regex.

### TOOL-SPECIFIC PROTOCOLS
- **execute_python**: Input must be `{"code": "python_code_here"}`. Use this as your primary engine for complex logic, bulk filtering, math, and cross-referencing.
- **search_csv_records / search_json_records**: Use ONLY for small, targeted lookups (<3 items) or initial exploration. Use partial values for flexible matching, or `exact_fields` for exact IDs.
- **answer**: This tool is terminal. Once used, the task ends.
""".strip()

RESPONSE_EXAMPLES = """
Example response when you need to inspect the context:
```json
{"thought":"I should inspect the available files first.","action":"list_context","action_input":{"max_depth":4}}
```

Example response when running Python code:
```json
{"thought": "I need to calculate the sum of values.","action": "execute_python","action_input": {"code": "import pandas as pd\\ndf = pd.read_csv('data.csv')\\nprint(df['value'].sum())"}}
```

Example response when safely reading a JSON file:
```json
{"thought": "I need to extract records from a large JSON file without assuming it is fully valid.","action": "execute_python","action_input": {"code": "from pathlib import Path\\nimport json\\n\\npath = Path('json/event.json')\\ntext = path.read_text(encoding='utf-8', errors='replace')\\nstart = text.find('\\"records\\": [')\\nif start == -1: raise ValueError('records array not found')\\nend = text.rfind(']')\\nrecords_text = text[start:text.find(']', start) + 1]\\nrecords = json.loads(records_text)\\nprint(len(records))"}}
```

Example response when you have the final answer:
```json
{"thought":"I have the final result table.","action":"answer","action_input":{"columns":["average_long_shots"],"rows":[["63.5"]]}}
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
