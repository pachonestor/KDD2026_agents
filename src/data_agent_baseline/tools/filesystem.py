from __future__ import annotations
 
import csv
from importlib.resources import path
import json
from pathlib import Path
import re
 
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

def search_csv_records(
    csv_path: str,
    search_criteria: dict[str, str],
    exact_fields: list[str] | None = None,
    limit: int = 50,
    contains: bool = True,
    case_sensitive: bool = False,
):
    """
    Busca registros en un CSV usando criterios específicos por campo.

    Parámetros:
    - search_criteria:
        diccionario con pares campo->valor a buscar
        Ejemplo: {"driverId": "123", "surname": "Räik"}
    - exact_fields:
        lista de campos que deben coincidir exactamente
        Los demás campos usarán búsqueda parcial (si contains=True)
    - limit:
        número máximo de resultados a retornar
    - contains:
        si True, búsqueda parcial en campos no-exactos (substring)
        si False, búsqueda exacta en campos no-exactos
    - case_sensitive:
        si False (default), la búsqueda ignora mayúsculas/minúsculas

    Ejemplo:
        search_criteria={"driverId": "123", "surname": "Räikkönen"}
        exact_fields=["driverId"]  # driverId debe coincidir exacto, surname parcial
    """

    exact_fields = set(exact_fields or [])
    matches = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Todos los criterios deben cumplirse para que sea un match
            all_match = True

            for field, query in search_criteria.items():
                if field not in row:
                    all_match = False
                    break

                value_str = str(row.get(field, ""))
                query_str = str(query)

                left = value_str if case_sensitive else value_str.lower()
                right = query_str if case_sensitive else query_str.lower()

                # match exacto
                if field in exact_fields:
                    ok = left == right
                # búsqueda parcial
                else:
                    ok = right in left if contains else left == right

                if not ok:
                    all_match = False
                    break

            if all_match:
                matches.append(dict(row))

            if len(matches) >= limit:
                break

    return {
        "search_criteria": search_criteria,
        "exact_fields": list(exact_fields),
        "match_count": len(matches),
        "matches": matches,
    }    

def search_json_records(
    json_path: str,
    search_criteria: dict[str, str],
    exact_fields: list[str] | None = None,
    limit: int = 50,
    contains: bool = True,
    case_sensitive: bool = False,
):
    """
    Busca registros en un JSON usando criterios específicos por campo.

    Compatible con:
    - listas de objetos
    - {"records": [...]}

    Parámetros:
    - search_criteria:
        diccionario con pares campo->valor a buscar
        Ejemplo: {"driverId": "123", "surname": "Räik"}
    - exact_fields:
        lista de campos que deben coincidir exactamente
        Los demás campos usarán búsqueda parcial (si contains=True)
    - limit:
        número máximo de resultados a retornar
    - contains:
        si True, búsqueda parcial en campos no-exactos
        si False, búsqueda exacta en campos no-exactos
    - case_sensitive:
        si False (default), la búsqueda ignora mayúsculas/minúsculas

    Ejemplo:
        search_criteria={"driverId": "123", "surname": "Räikkönen"}
        exact_fields=["driverId"]
    """

    exact_fields = set(exact_fields or [])

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # soporta {"records": [...]}
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break

    if not isinstance(data, list):
        raise ValueError("JSON does not contain a list of records")

    matches = []

    for record in data:
        if not isinstance(record, dict):
            continue

        # Todos los criterios deben cumplirse
        all_match = True

        for field, query in search_criteria.items():
            if field not in record:
                all_match = False
                break

            value_str = str(record.get(field, ""))
            query_str = str(query)

            left = value_str if case_sensitive else value_str.lower()
            right = query_str if case_sensitive else query_str.lower()

            # match exacto
            if field in exact_fields:
                ok = left == right
            # búsqueda parcial
            else:
                ok = right in left if contains else left == right

            if not ok:
                all_match = False
                break

        if all_match:
            matches.append(record)

        if len(matches) >= limit:
            break

    return {
        "search_criteria": search_criteria,
        "exact_fields": list(exact_fields),
        "match_count": len(matches),
        "matches": matches,
    }

def search_markdown_text(
    md_path: str,
    queries: list[str] | None = None,
    regex_patterns: list[str] | None = None,
    mode: str = "OR",
    neighborhood: int = 20,
    return_mode: str = "paragraph",
    include_headings: bool = True,
    case_sensitive: bool = False,
    limit: int = 20,
):
    """
    Busca contenido dentro de archivos markdown.

    Parámetros:
    - queries:
        lista de búsquedas literales
    - regex_patterns:
        lista de regex
    - mode:
        "OR" o "AND"
    - neighborhood:
        nº de palabras antes/después del match
        solo usado en return_mode="window"
    - return_mode:
        "paragraph" -> devuelve párrafo completo
        "window" -> devuelve contexto local
    - include_headings:
        añade la jerarquía markdown
    """

    queries = queries or []
    regex_patterns = regex_patterns or []

    if not queries and not regex_patterns:
        raise ValueError("At least one query or regex_pattern must be provided")

    if mode not in {"OR", "AND"}:
        raise ValueError("mode must be OR or AND")

    if return_mode not in {"paragraph", "window"}:
        raise ValueError("return_mode must be paragraph or window")

    flags = 0 if case_sensitive else re.IGNORECASE

    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    paragraphs = re.split(r"\n\s*\n", text)

    matches = []

    current_headings: list[str] = []

    for paragraph_index, paragraph in enumerate(paragraphs):

        stripped = paragraph.strip()

        if not stripped:
            continue

        # Detectar headings markdown
        heading_match = re.match(r"^(#+)\s+(.*)", stripped)

        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()

            current_headings = current_headings[: level - 1]
            current_headings.append(heading_text)

        paragraph_matches = []

        # búsquedas literales
        for q in queries:
            pattern = re.escape(q)
            found = list(re.finditer(pattern, paragraph, flags))

            if found:
                paragraph_matches.extend(found)

        # búsquedas regex
        for regex in regex_patterns:
            found = list(re.finditer(regex, paragraph, flags))

            if found:
                paragraph_matches.extend(found)

        # lógica AND / OR
        checks = []

        for q in queries:
            checks.append(
                bool(re.search(re.escape(q), paragraph, flags))
            )

        for regex in regex_patterns:
            checks.append(
                bool(re.search(regex, paragraph, flags))
            )

        if not checks:
            continue

        if mode == "AND" and not all(checks):
            continue

        if mode == "OR" and not any(checks):
            continue

        # devolver párrafo completo
        if return_mode == "paragraph":

            result = {
                "paragraph_index": paragraph_index,
                "match_count": len(paragraph_matches),
                "content": stripped,
            }

            if include_headings:
                result["heading_path"] = list(current_headings)

            matches.append(result)

        # devolver ventanas de contexto
        else:

            words = paragraph.split()

            for match in paragraph_matches:

                matched_text = match.group(0)

                char_pos = match.start()

                cumulative = 0
                word_index = 0

                for i, word in enumerate(words):
                    cumulative += len(word) + 1

                    if cumulative >= char_pos:
                        word_index = i
                        break

                start = max(0, word_index - neighborhood)
                end = min(len(words), word_index + neighborhood + 1)

                snippet = " ".join(words[start:end])

                result = {
                    "paragraph_index": paragraph_index,
                    "matched_text": matched_text,
                    "content": snippet,
                }

                if include_headings:
                    result["heading_path"] = list(current_headings)

                matches.append(result)

                if len(matches) >= limit:
                    break

        if len(matches) >= limit:
            break

    return {
        "queries": queries,
        "regex_patterns": regex_patterns,
        "mode": mode,
        "return_mode": return_mode,
        "neighborhood": neighborhood,
        "match_count": len(matches),
        "matches": matches[:limit],
    }
