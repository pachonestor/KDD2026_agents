from __future__ import annotations
 
import csv
from importlib.resources import path
import json
from pathlib import Path
 
from data_agent_baseline.benchmark.schema import PublicTask
 
 
def resolve_context_path(task: PublicTask, relative_path: str) -> Path:
    candidate = (task.context_dir / relative_path).resolve()
    context_root = task.context_dir.resolve()
    if context_root not in candidate.parents and candidate != context_root:
        raise ValueError(f"Path escapes context dir: {relative_path}")
    if not candidate.exists():
        raise FileNotFoundError(f"Missing context asset: {relative_path}")
    return candidate
 
 
def list_context_tree(task: PublicTask, *, max_depth: int = 4) -> dict[str, object]:
    entries: list[dict[str, object]] = []
 
    def walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        for child in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name)):
            rel_path = child.relative_to(task.context_dir).as_posix()
            entries.append(
                {
                    "path": rel_path,
                    "kind": "dir" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
            if child.is_dir():
                walk(child, depth + 1)
 
    walk(task.context_dir, 1)
    return {
        "root": str(task.context_dir),
        "entries": entries,
    }
 
 
def read_csv_preview(task: PublicTask, relative_path: str, *, max_rows: int = 20) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
   
    # Intentamos UTF-8 primero, si falla usamos latin-1
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
    except UnicodeDecodeError:
        with path.open(newline="", encoding="latin-1") as handle:
            rows = list(csv.reader(handle))
 
    if not rows:
        return {"path": relative_path, "columns": [], "rows": [], "row_count": 0}
 
    header = rows[0]
    data_rows = rows[1:]
    return {
        "path": relative_path,
        "columns": header,
        "rows": data_rows[:max_rows],
        "row_count": len(data_rows),
    }
 
 
def read_json_preview(task: PublicTask, relative_path: str, *, max_chars: int = 4000) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
   
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Los archivos JSON deberían ser UTF-8, pero si fallan, latin-1 los leerá sin error
        content = path.read_text(encoding="latin-1")
       
    try:
        payload = json.loads(content)
        preview = json.dumps(payload, ensure_ascii=False, indent=2)
        return {
            "path": relative_path,
            "preview": preview[:max_chars],
            "truncated": len(preview) > max_chars,
            "malformed": False,
        }
    except json.JSONDecodeError as exc:
        return {
            "path": relative_path,
            "preview": content[:max_chars],
            "truncated": len(content) > max_chars,
            "malformed": True,
            "error": str(exc),
        }
 
 
def read_doc_preview(task: PublicTask, relative_path: str, *, max_chars: int = 4000) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
   
    try:
        # Intentamos leer limpio
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Si falla, usamos latin-1 para preservar caracteres como la 'ä' de Räikkönen
        text = path.read_text(encoding="latin-1", errors="replace")
       
    return {
        "path": relative_path,
        "preview": text[:max_chars],
        "truncated": len(text) > max_chars,
    }