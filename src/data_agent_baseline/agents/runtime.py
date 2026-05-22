from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from data_agent_baseline.benchmark.schema import AnswerTable


@dataclass(frozen=True, slots=True)
class StepRecord:
    step_index: int
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str
    observation: dict[str, Any]
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentRuntimeState:
    steps: list[StepRecord] = field(default_factory=list)
    answer: AnswerTable | None = None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    task_id: str
    answer: AnswerTable | None
    steps: list[StepRecord]
    failure_reason: str | None

    @property
    def succeeded(self) -> bool:
        return self.answer is not None and self.failure_reason is None

    def to_dict(self) -> dict[str, Any]:
        if self.answer is None:
            answer_payload = None
        elif hasattr(self.answer, "to_dict"):
            answer_payload = self.answer.to_dict()
        elif isinstance(self.answer, dict):
            answer_payload = self.answer
        else:
            try:
                answer_payload = asdict(self.answer)
            except Exception:
                answer_payload = str(self.answer)

        return {
            "task_id": self.task_id,
            "answer": answer_payload,
            "steps": [step.to_dict() for step in self.steps],
            "failure_reason": self.failure_reason,
            "succeeded": self.succeeded,
        }
