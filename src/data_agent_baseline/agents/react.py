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

    fence_match = re.search(
        r"```json\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if fence_match is not None:
        return fence_match.group(1).strip()

    generic_fence_match = re.search(
        r"```\s*(.*?)\s*```",
        text,
        flags=re.DOTALL,
    )

    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()

    return text


def _extract_first_json_object(text: str) -> str:

    start = text.find("{")

    if start == -1:
        raise ValueError(
            "Model response must contain a JSON object."
        )

    depth = 0
    in_string = False
    escape = False

    for index, char in enumerate(
        text[start:],
        start=start,
    ):

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1

        elif char == "}":

            depth -= 1

            if depth == 0:
                return text[start : index + 1]

    raise ValueError(
        "Model response JSON object is incomplete."
    )


def _load_single_json_object(
    text: str,
) -> dict[str, object]:

    try:

        payload = json.loads(text)

    except json.JSONDecodeError as e:

        if (
            "Extra data" in str(e)
            or "Expecting value" in str(e)
        ):

            try:

                candidate = (
                    _extract_first_json_object(text)
                )

                payload = json.loads(candidate)

            except (
                ValueError,
                json.JSONDecodeError,
            ):
                raise e

        elif (
            "Expecting ',' delimiter" in str(e)
            or "Expecting ':' delimiter" in str(e)
        ):

            fixed_text = text.rstrip() + "}"

            try:
                payload = json.loads(fixed_text)

            except json.JSONDecodeError:
                raise e

        else:
            raise e

    if not isinstance(payload, dict):

        raise ValueError(
            "Model response must be a JSON object."
        )

    return payload


# ==========================================================
# PARSER
# ==========================================================

def parse_model_step(
    raw_response: str,
) -> ModelStep:

    normalized = _strip_json_fence(
        raw_response
    )

    payload = _load_single_json_object(
        normalized
    )

    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get(
        "action_input",
        {},
    )

    if not isinstance(thought, str):

        raise ValueError(
            "thought must be a string."
        )

    if (
        not isinstance(action, str)
        or not action
    ):

        raise ValueError(
            "action must be a non-empty string."
        )

    if (
        isinstance(action_input, str)
        and action == "execute_python"
    ):

        action_input = {
            "code": action_input
        }

    if not isinstance(action_input, dict):

        raise ValueError(
            "action_input must be a JSON object."
        )

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
