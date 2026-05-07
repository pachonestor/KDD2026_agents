from __future__ import annotations

import json

from data_agent_baseline.benchmark.schema import PublicTask


REACT_SYSTEM_PROMPT = """
You are a ReAct-style data agent.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

Rules:
1. Use tools to inspect the available context before answering.
2. Base your answer only on information you can observe through the provided tools.
3. The task is complete only when you call the `answer` tool.
4. The `answer` tool must receive a table with `columns` and `rows`.
5. Always return exactly one JSON object with keys `thought`, `action`, and `action_input`.
6. Always wrap that JSON object in exactly one fenced code block that starts with ```json and ends with ```.
7. Do not output any text before or after the fenced JSON block.
8. Ensure the JSON is syntactically valid, complete, and properly formatted. Avoid truncation or incomplete responses.
9. Keep the entire response concise to prevent truncation.
10. If a tool execution fails (ok: false), do not repeat the exact same action_input.
   Analyze the error message, identify the root cause, and try a corrected or alternative approach.
11. `read_json` previews may be truncated or the file may be malformed. Do not assume the JSON file can be safely parsed with `json.load`.
   If a JSON file is large, invalid, or truncated, read it as raw text and search for the relevant keys.
12. If a Python JSON parser fails with `Extra data`, do not retry the same parsing strategy.
   Parse the file as raw text instead and extract only the relevant JSON segment or values.
Output discipline:
- execute_python always requires action_input as {"code": "your_python_code_here"} — never pass code as a plain string outside the "code" key.
- Escape special characters in strings correctly (e.g., use \\n for newlines, \\\" for quotes).
- execute_python must always use encoding="utf-8" when writing files, and decoding with encoding="utf-8" when reading files.
- Do not use `json.load` directly on task JSON files unless you have confirmed the file is complete and valid.
  If the JSON is large or the preview is truncated, read the file as raw text and extract only the fields you need.

Keep reasoning concise and grounded in the observed data.
""".strip()

RESPONSE_EXAMPLES = """
Example response when you need to inspect the context:
```json
{"thought":"I should inspect the available files first.","action":"list_context","action_input":{"max_depth":4}}
```

Example response when running Python code:
```json
{
  "thought": "I need to calculate the sum of values.",
  "action": "execute_python",
  "action_input": {
    "code": "import pandas as pd\\ndf = pd.read_csv('data.csv')\\nprint(df['value'].sum())"
  }
}
```

Example response when safely reading a JSON file:
```json
{
  "thought": "I need to extract records from a large JSON file without assuming it is fully valid.",
  "action": "execute_python",
  "action_input": {
    "code": "from pathlib import Path\\nimport json\\n\\npath = Path('json/event.json')\\ntext = path.read_text(encoding='utf-8', errors='replace')\\nstart = text.find('\\"records\\": [')\\nif start == -1: raise ValueError('records array not found')\\nend = text.rfind(']')\\nrecords_text = text[start:text.find(']', start) + 1]\\nrecords = json.loads(records_text)\\nprint(len(records))"
  }
}
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
