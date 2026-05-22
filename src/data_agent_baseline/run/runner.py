from __future__ import annotations

import csv
import json
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from time import monotonic, perf_counter
from typing import Any

from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AppConfig
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry

FALLBACK_PREDICTION_COLUMNS = ["no_answer"]


@dataclass(frozen=True, slots=True)
class TaskRunArtifacts:
    task_id: str
    task_output_dir: Path
    prediction_csv_path: Path | None
    trace_path: Path | None  # Opcional para soportar flat_output
    succeeded: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": str(self.prediction_csv_path) if self.prediction_csv_path else None,
            "trace_path": str(self.trace_path) if self.trace_path else None,
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }


def create_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_run_id(run_id: str | None = None) -> str:
    timestamp = create_run_id()
    if run_id is None:
        return timestamp

    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty.")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must be a single directory name, not a path.")
    return f"{normalized}_{timestamp}"


def create_run_output_dir(output_root: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    effective_run_id = resolve_run_id(run_id)
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return effective_run_id, run_output_dir


def resolve_run_output_directory(config: AppConfig) -> tuple[str, Path]:
    """Decide si usar la estructura de subcarpetas o la plana (flat_output) mediante la configuración."""
    if config.run.flat_output:
        root = config.run.output_dir
        root.mkdir(parents=True, exist_ok=True)
        return "eval_flat", root
    return create_run_output_dir(config.run.output_dir, run_id=config.run.run_id)


def build_model_adapter(config: AppConfig):
    return OpenAIModelAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        temperature=config.agent.temperature,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)


def _failure_run_result_payload(task_id: str, failure_reason: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "answer": None,
        "steps": [],
        "failure_reason": failure_reason,
        "succeeded": False,
    }


def _run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
) -> dict[str, Any]:
    public_dataset = DABenchPublicDataset(config.dataset.root_path)
    task = public_dataset.get_task(task_id)

    agent = ReActAgent(
        model=model or build_model_adapter(config),
        tools=tools or create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=config.agent.max_steps),
    )
    run_result = agent.run(task)
    return run_result.to_dict()


def _run_single_task_in_subprocess(task_id: str, config: AppConfig, queue: multiprocessing.Queue[Any]) -> None:
    try:
        queue.put(
            {
                "ok": True,
                "run_result": _run_single_task_core(task_id=task_id, config=config),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        queue.put(
            {
                "ok": False,
                "error": str(exc),
            }
        )


def _cleanup_child_process(process: multiprocessing.Process, timeout: float = 1.0) -> None:
    process.join(timeout=timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=timeout)
    if process.is_alive():
        process.kill()
        process.join()


def _run_single_task_with_timeout(*, task_id: str, config: AppConfig) -> dict[str, Any]:
    timeout_seconds = config.run.task_timeout_seconds
    if timeout_seconds <= 0:
        return _run_single_task_core(task_id=task_id, config=config)

    queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_single_task_in_subprocess,
        args=(task_id, config, queue),
    )
    process.start()

    deadline = monotonic() + timeout_seconds

    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            break

        try:
            result = queue.get(timeout=min(0.2, remaining))
            if process.is_alive():
                _cleanup_child_process(process)
            if result.get("ok"):
                return dict(result["run_result"])
            return _failure_run_result_payload(task_id, f"Task failed with uncaught error: {result['error']}")
        except Empty:
            if not process.is_alive():
                break
            continue

    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join()
        return _failure_run_result_payload(task_id, f"Task timed out after {timeout_seconds} seconds.")

    try:
        result = queue.get_nowait()
    except Empty:
        exit_code = process.exitcode
        if exit_code not in (None, 0):
            return _failure_run_result_payload(
                task_id,
                f"Task exited unexpectedly with exit code {exit_code}.",
            )
        return _failure_run_result_payload(task_id, "Task exited without returning a result.")

    if result.get("ok"):
        if process.is_alive():
            _cleanup_child_process(process)
        return dict(result["run_result"])
    if process.is_alive():
        _cleanup_child_process(process)
    return _failure_run_result_payload(task_id, f"Task failed with uncaught error: {result['error']}")


def _last_answer_candidate(run_result: dict[str, Any]) -> dict[str, Any] | None:
    """Busca en los pasos intermedios para rescatar tablas generadas si falla la respuesta final."""
    steps = run_result.get("steps")
    if not isinstance(steps, list):
        return None
    for step in reversed(steps):
        if not isinstance(step, dict) or step.get("action") != "answer":
            continue
        action_input = step.get("action_input")
        if not isinstance(action_input, dict):
            continue
        columns = action_input.get("columns")
        rows = action_input.get("rows")
        if not isinstance(columns, list) or not columns:
            continue
        if not isinstance(rows, list) or not all(isinstance(row, list) for row in rows):
            continue
        return {"columns": [str(column) for column in columns], "rows": [list(row) for row in rows]}
    return None


def _write_task_outputs(task_id: str, run_output_dir: Path, run_result: dict[str, Any], flat_output: bool) -> TaskRunArtifacts:
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Soporte flat_output
    trace_path: Path | None = None
    if not flat_output:
        trace_path = task_output_dir / "trace.json"
        _write_json(trace_path, run_result)

    prediction_csv_path = task_output_dir / "prediction.csv"
    
    answer = run_result.get("answer")
    candidate = _last_answer_candidate(run_result)
    
    if isinstance(answer, dict) and answer.get("columns"):
        _write_csv(
            prediction_csv_path,
            list(answer.get("columns", [])),
            [list(row) for row in answer.get("rows", [])],
        )
    elif candidate is not None:
        _write_csv(
            prediction_csv_path,
            candidate["columns"],
            candidate["rows"],
        )
    else:
        # FALLBACK: Escribir CSV vacío con la columna definida
        _write_csv(
            prediction_csv_path,
            FALLBACK_PREDICTION_COLUMNS,
            [] 
        )

    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_csv_path,
        trace_path=trace_path,
        succeeded=bool(run_result.get("succeeded")),
        failure_reason=run_result.get("failure_reason"),
    )


def run_single_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    model=None,
    tools: ToolRegistry | None = None,
) -> TaskRunArtifacts:
    
    dataset = DABenchPublicDataset(config.dataset.root_path)
    task = dataset.get_task(task_id)

    # OMISIÓN EXTREMA controlada por el config
    if config.run.skip_extreme and task.difficulty.lower() == "extreme":
        run_result = {
            "task_id": task_id,
            "answer": None,
            "steps": [],
            "failure_reason": "Skipped extreme task.",
            "succeeded": False,
        }
        return _write_task_outputs(task_id, run_output_dir, run_result, config.run.flat_output)

    started_at = perf_counter()
    if model is None and tools is None:
        run_result = _run_single_task_with_timeout(task_id=task_id, config=config)
    else:
        run_result = _run_single_task_core(task_id=task_id, config=config, model=model, tools=tools)
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    
    return _write_task_outputs(task_id, run_output_dir, run_result, config.run.flat_output)


def run_benchmark(
    *,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    limit: int | None = None,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    
    # Resolvemos el directorio pasándole directamente la configuración
    effective_run_id, run_output_dir = resolve_run_output_directory(config)

    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks()
    if limit is not None:
        tasks = tasks[:limit]

    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if model is not None or tools is not None:
        effective_workers = 1

    task_ids = [task.task_id for task in tasks]

    task_artifacts: list[TaskRunArtifacts]
    if effective_workers == 1:
        shared_model = model or build_model_adapter(config)
        shared_tools = tools or create_default_tool_registry()
        task_artifacts = []
        for task_id in task_ids:
            artifact = run_single_task(
                task_id=task_id,
                config=config,
                run_output_dir=run_output_dir,
                model=shared_model,
                tools=shared_tools,
            )
            task_artifacts.append(artifact)
            if progress_callback is not None:
                progress_callback(artifact)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_index = {
                executor.submit(
                    run_single_task,
                    task_id=task_id,
                    config=config,
                    run_output_dir=run_output_dir,
                ): index
                for index, task_id in enumerate(task_ids)
            }
            indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(task_ids)
            for future in as_completed(future_to_index):
                artifact = future.result()
                indexed_artifacts[future_to_index[future]] = artifact
                if progress_callback is not None:
                    progress_callback(artifact)
            task_artifacts = [artifact for artifact in indexed_artifacts if artifact is not None]

    # Soporte flat_output: No escribir summary.json masivo si está en layout plano
    if not config.run.flat_output:
        summary_path = run_output_dir / "summary.json"
        _write_json(
            summary_path,
            {
                "run_id": effective_run_id,
                "task_count": len(task_artifacts),
                "succeeded_task_count": sum(1 for artifact in task_artifacts if artifact.succeeded),
                "max_workers": effective_workers,
                "tasks": [artifact.to_dict() for artifact in task_artifacts],
            },
        )
    return run_output_dir, task_artifacts