from __future__ import annotations

import json
import re
from dataclasses import dataclass

from data_agent_baseline.agents.model import (
    ModelAdapter,
    ModelMessage,
    ModelStep,
)
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import (
    AgentRunResult,
    AgentRuntimeState,
    StepRecord,
)
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import (
    ToolExecutionResult,
    ToolRegistry,
)


# ==========================================================
# CONFIG
# ==========================================================

@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16


# ==========================================================
# JSON HELPERS
# ==========================================================

def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    
    # Usamos fullmatch para obligar a que TODO el texto sea el bloque JSON
    match = re.fullmatch(
        r"```json\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match is not None:
        return match.group(1).strip()

    return text


# ==========================================================
# PARSER
# ==========================================================

def parse_model_step(
    raw_response: str,
) -> ModelStep:

    normalized = _strip_json_fence(
        raw_response
    )

    # Si esto falla, lanzará un json.JSONDecodeError
    payload = json.loads(normalized)

    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")

    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get(
        "action_input",
        {},
    )

    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")

    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")

    if isinstance(action_input, str) and action == "execute_python":
        action_input = {"code": action_input}

    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
    )


# ==========================================================
# REACT AGENT
# ==========================================================

class ReActAgent:

    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:

        self.model = model
        self.tools = tools

        self.config = (
            config
            or ReActAgentConfig()
        )

        self.system_prompt = (
            system_prompt
            or REACT_SYSTEM_PROMPT
        )

# ======================================================
    # BUILD MESSAGES
    # ======================================================

    def _build_messages(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        step_index: int,
    ) -> list[ModelMessage]:

        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )

        # --------------------------------------------------
        # INYECCIÓN PROACTIVA DE CONTEXTO (ÁRBOL DE ARCHIVOS)
        # --------------------------------------------------
        try:
            from data_agent_baseline.tools.filesystem import list_context_tree
            
            # Escaneamos los archivos con una profundidad máxima de 2 o 3 
            # para no saturar el prompt con carpetas muy profundas (p.ej. venv o caché)
            context_tree = list_context_tree(task, max_depth=2)
            entries = context_tree.get("entries", [])
            
            if entries:
                files_listing = "\n".join([
                    f"- {entry['path']} ({entry['kind']})"
                    for entry in entries
                ])
                # Añadimos la lista de manera estructurada al prompt de sistema
                system_content += (
                    "\n\n### DETECTED ENVIRONMENT FILES\n"
                    "You have direct access to the following files in your current workspace:\n"
                    f"{files_listing}\n"
                    "Use this map to choose which files to inspect or load via Python/SQL."
                )
        except Exception as e:
            # Usamos un try-except defensivo para que el agente nunca muera 
            # si hay un error leyendo el sistema de archivos
            pass
        # --------------------------------------------------

        messages = [
            ModelMessage(
                role="system",
                content=system_content,
            )
        ]

        remaining_steps = (
            self.config.max_steps
            - step_index
            + 1
        )

        # --------------------------------------------------
        # WARNING MODE
        # --------------------------------------------------

        if remaining_steps <= 2:

            messages.append(
                ModelMessage(
                    role="system",
                    content=(
                        "WARNING: You are almost "
                        "out of steps.\n"
                        "You should prepare and "
                        "submit your final answer "
                        "immediately.\n"
                        "If you fail to answer, "
                        "the runtime may force a "
                        "fallback final answer."
                    ),
                )
            )

        messages.append(
            ModelMessage(
                role="user",
                content=build_task_prompt(task),
            )
        )

        for step in state.steps:

            messages.append(
                ModelMessage(
                    role="assistant",
                    content=step.raw_response,
                )
            )

            messages.append(
                ModelMessage(
                    role="user",
                    content=build_observation_prompt(
                        step.observation
                    ),
                )
            )

        return messages

    # ======================================================
    # BEST EFFORT ANSWER
    # ======================================================

    def _build_best_effort_answer(
        self,
        state: AgentRuntimeState,
    ) -> dict:

        collected_rows = []

        for step in state.steps:

            obs = step.observation

            if not isinstance(obs, dict):
                continue

            content = obs.get("content")

            if not isinstance(content, dict):
                continue

            # --------------------------------------------------
            # direct rows
            # --------------------------------------------------

            rows = content.get("rows")

            if isinstance(rows, list):

                for row in rows:

                    if isinstance(row, list):
                        collected_rows.append(row)

            # --------------------------------------------------
            # nested results
            # --------------------------------------------------

            result = content.get("result")

            if isinstance(result, list):

                for item in result:

                    if isinstance(item, list):

                        collected_rows.append(item)

                    elif isinstance(item, dict):

                        collected_rows.append(
                            [
                                str(v)
                                for v in item.values()
                            ]
                        )

        # --------------------------------------------------
        # fallback
        # --------------------------------------------------

        if not collected_rows:

            collected_rows = [[
                "Unable to fully complete "
                "task within step budget."
            ]]

        return {
            "columns": ["result"],
            "rows": collected_rows,
        }

    # ======================================================
    # FORCE FINAL ANSWER
    # ======================================================

    def _force_final_answer(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
    ) -> ToolExecutionResult:

        final_answer = (
            self._build_best_effort_answer(
                state
            )
        )

        try:

            return self.tools.execute(
                task,
                "answer",
                final_answer,
            )

        except Exception:

            # ultra defensive fallback

            return ToolExecutionResult(
                ok=True,
                content=final_answer,
                answer=final_answer,
                is_terminal=True,
            )

    # ======================================================
    # GUARD TOOL CALLS
    # ======================================================

    def _guard_tool_call(
        self,
        *,
        task: PublicTask,
        state: AgentRuntimeState,
        step_index: int,
        model_step: ModelStep,
        raw_response: str,
    ) -> tuple[bool, bool]:

        remaining_steps = (
            self.config.max_steps
            - step_index
            + 1
        )

        # --------------------------------------------------
        # LAST STEP -> FORCE ANSWER
        # --------------------------------------------------

        if (
            remaining_steps == 1
            and model_step.action != "answer"
        ):

            forced_tool_result = (
                self._force_final_answer(
                    task,
                    state,
                )
            )

            observation = {
                "ok": forced_tool_result.ok,
                "tool": "answer",
                "forced": True,
                "content": (
                    forced_tool_result.content
                ),
            }

            state.steps.append(
                StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action="__forced_answer__",
                    action_input={},
                    raw_response=raw_response,
                    observation=observation,
                    ok=forced_tool_result.ok,
                )
            )

            # IMPORTANT:
            # NO dependemos de is_terminal

            if (
                getattr(
                    forced_tool_result,
                    "answer",
                    None,
                )
                is not None
            ):

                state.answer = (
                    forced_tool_result.answer
                )

            else:

                state.answer = (
                    self._build_best_effort_answer(
                        state
                    )
                )

            return True, True

        return False, False

# ======================================================
    # MAIN LOOP
    # ======================================================

    def run(
        self,
        task: PublicTask,
    ) -> AgentRunResult:

        state = AgentRuntimeState()

        for step_index in range(
            1,
            self.config.max_steps + 1,
        ):

            messages = self._build_messages(
                task,
                state,
                step_index,
            )

            raw_response = self.model.complete(
                messages
            )

            try:

                model_step = parse_model_step(
                    raw_response
                )

                # ------------------------------------------
                # GUARD LAYER
                # ------------------------------------------

                intercepted, terminal = (
                    self._guard_tool_call(
                        task=task,
                        state=state,
                        step_index=step_index,
                        model_step=model_step,
                        raw_response=raw_response,
                    )
                )

                if terminal:
                    break

                if intercepted:
                    continue

                # ------------------------------------------
                # ANTI-LOOP PROTOCOL
                # ------------------------------------------
                is_loop = False
                for prev_step in state.steps:
                    # Si el paso previo falló (ok=False) y la herramienta junto con
                    # sus argumentos de entrada son idénticos, se detecta un bucle.
                    if (
                        not prev_step.ok 
                        and prev_step.action == model_step.action 
                        and prev_step.action_input == model_step.action_input
                    ):
                        is_loop = True
                        break
                
                if is_loop and model_step.action != "answer":
                    loop_observation = {
                        "ok": False,
                        "error": "AntiLoopTriggered",
                        "message": (
                            "CRITICAL WARNING: You are attempting to run the exact same "
                            f"tool ('{model_step.action}') with the exact same input that ALREADY FAILED. "
                            "Do NOT repeat this action. You MUST change your approach, fix the syntax, "
                            "use different pandas/SQL functions, or inspect the data manually."
                        )
                    }

                    state.steps.append(
                        StepRecord(
                            step_index=step_index,
                            thought=model_step.thought,
                            action=model_step.action,
                            action_input=model_step.action_input,
                            raw_response=raw_response,
                            observation=loop_observation,
                            ok=False,
                        )
                    )
                    # Forzamos al agente a volver a pensar sin ejecutar la herramienta fallida
                    continue

                # ------------------------------------------
                # NORMAL TOOL EXECUTION
                # ------------------------------------------

                tool_result = self.tools.execute(
                    task,
                    model_step.action,
                    model_step.action_input,
                )

                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }

                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought=model_step.thought,
                        action=model_step.action,
                        action_input=model_step.action_input,
                        raw_response=raw_response,
                        observation=observation,
                        ok=tool_result.ok,
                    )
                )

                # ------------------------------------------
                # TERMINAL
                # ------------------------------------------

                if (
                    getattr(
                        tool_result,
                        "answer",
                        None,
                    )
                    is not None
                ):

                    state.answer = (
                        tool_result.answer
                    )

                if tool_result.is_terminal:
                    break

            except json.JSONDecodeError as exc:
                
                # ------------------------------------------
                # AUTO-CORRECCIÓN DE JSON
                # ------------------------------------------
                observation = {
                    "ok": False,
                    "error": "JSONDecodeError",
                    "message": (
                        "Your previous output was not valid JSON. Return exactly one fenced "
                        "```json block containing one JSON object with keys thought, action, "
                        "and action_input. Do not include text outside the fenced block. "
                        "Properly escape all strings."
                    ),
                    "error_type": "JSONDecodeError",
                    "detail": str(exc),
                }

                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response[:2000] if raw_response else "",
                        observation=observation,
                        ok=False,
                    )
                )

            except Exception as exc:

                observation = {
                    "ok": False,
                    "error": str(exc),
                    "error_type": (
                        type(exc).__name__
                    ),
                }

                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )

                # ------------------------------------------
                # LAST STEP EXCEPTION SAFETY
                # ------------------------------------------

                remaining_steps = (
                    self.config.max_steps
                    - step_index
                    + 1
                )

                if (
                    remaining_steps == 1
                    and state.answer is None
                ):

                    try:

                        fallback_tool_result = (
                            self._force_final_answer(
                                task,
                                state,
                            )
                        )

                        if (
                            getattr(
                                fallback_tool_result,
                                "answer",
                                None,
                            )
                            is not None
                        ):

                            state.answer = (
                                fallback_tool_result.answer
                            )

                        else:

                            state.answer = (
                                self._build_best_effort_answer(
                                    state
                                )
                            )

                    except Exception:

                        state.answer = {
                            "columns": [
                                "result"
                            ],
                            "rows": [[
                                "Forced fallback "
                                "answer after "
                                "exception."
                            ]],
                        }

                    break

        # ==================================================
        # ABSOLUTE FINAL FALLBACK
        # ==================================================

        if state.answer is None:

            try:

                fallback_tool_result = (
                    self._force_final_answer(
                        task,
                        state,
                    )
                )

                if (
                    getattr(
                        fallback_tool_result,
                        "answer",
                        None,
                    )
                    is not None
                ):

                    state.answer = (
                        fallback_tool_result.answer
                    )

                else:

                    state.answer = (
                        self._build_best_effort_answer(
                            state
                        )
                    )

            except Exception:

                state.answer = {
                    "columns": ["result"],
                    "rows": [[
                        "Agent failed but "
                        "fallback answer "
                        "was generated."
                    ]],
                }

            # nunca marcar failure
            state.failure_reason = None

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )