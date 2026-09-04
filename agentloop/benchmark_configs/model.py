"""Declarative configuration for the two interactive Repo benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Literal

from ..errors import AgentLoopError
from ..finish_contract import FinishContract

from evaluation_core.contracts import (
    GROUNDING_PROFILES,
    RepoPredictionContract,
    load_prediction_schema,
)
from evaluation_core.messages import (
    PromptVariant,
    build_repo_initial_user_prompt,
    load_task_prompt,
)


RepoBenchmark = Literal["specgap", "silentswap"]
ReasoningEffort = Literal["none", "light", "medium", "high"]


@dataclass(frozen=True, slots=True)
class PredictionRequestConfig:
    """Exact provider fields frozen for one benchmark's prediction model."""

    thinking: Literal["disabled"] | None
    reasoning_effort: ReasoningEffort | None

    def __post_init__(self) -> None:
        if self.thinking not in {None, "disabled"}:
            raise AgentLoopError("prediction thinking must be disabled or omitted")
        if self.reasoning_effort not in {None, "none", "light", "medium", "high"}:
            raise AgentLoopError("unsupported prediction reasoning_effort")

    def public_dict(self) -> dict[str, Any]:
        return {
            "thinking": self.thinking if self.thinking is not None else "omitted",
            "reasoning_effort": (
                self.reasoning_effort
                if self.reasoning_effort is not None
                else "omitted"
            ),
        }


@dataclass(frozen=True, slots=True)
class BoundModule7:
    """Schema-bound values injected into one shared AgentLoop run."""

    prediction_schema: dict[str, Any]
    initial_user_prompt: str
    max_rounds: int
    finish_contract: FinishContract


@dataclass(frozen=True, slots=True)
class RepoBenchmarkConfig:
    """Frozen differences that are owned by one Repo benchmark."""

    benchmark: RepoBenchmark
    max_rounds: int
    task_document_filename: str
    directory_strategy: str
    required_directory_sections: tuple[str, ...]
    prediction_schema_filename: str
    grounding_profile: str
    prediction_request: PredictionRequestConfig

    def __post_init__(self) -> None:
        if self.benchmark not in {"specgap", "silentswap"}:
            raise AgentLoopError(f"unsupported Repo benchmark: {self.benchmark!r}")
        if type(self.max_rounds) is not int or self.max_rounds <= 0:
            raise AgentLoopError("benchmark max_rounds must be a positive integer")
        filenames = {
            "task_document_filename": self.task_document_filename,
            "prediction_schema_filename": self.prediction_schema_filename,
        }
        for name, value in filenames.items():
            if (
                not isinstance(value, str)
                or not value
                or PurePath(value).name != value
            ):
                raise AgentLoopError(f"benchmark {name} must be a plain filename")
        text_fields = {
            "directory_strategy": self.directory_strategy,
            "grounding_profile": self.grounding_profile,
        }
        for name, value in text_fields.items():
            if not isinstance(value, str) or not value.strip():
                raise AgentLoopError(f"benchmark {name} must be non-empty")
        if not isinstance(self.prediction_request, PredictionRequestConfig):
            raise AgentLoopError(
                "benchmark prediction_request must be PredictionRequestConfig"
            )
        if (
            not self.required_directory_sections
            or len(self.required_directory_sections)
            != len(set(self.required_directory_sections))
            or any(
                not isinstance(item, str) or not item.strip()
                for item in self.required_directory_sections
            )
        ):
            raise AgentLoopError(
                "required_directory_sections must be unique non-empty labels"
            )
        if self.grounding_profile != GROUNDING_PROFILES[self.benchmark]:
            raise AgentLoopError(
                f"grounding profile does not belong to {self.benchmark}"
            )

    def validate_for(self, benchmark: str) -> None:
        if benchmark != self.benchmark:
            raise AgentLoopError(
                f"benchmark config {self.benchmark!r} cannot run {benchmark!r}"
            )

    def validate_task_document(self, filename: str) -> None:
        if PurePath(filename).name != self.task_document_filename:
            raise AgentLoopError(
                f"{self.benchmark} requires task document "
                f"{self.task_document_filename!r}, got {filename!r}"
            )

    def validate_graph_directory(self, initial_index: str) -> None:
        if not isinstance(initial_index, str) or not initial_index.strip():
            raise AgentLoopError("graph directory must be non-empty")
        for section in self.required_directory_sections:
            if f"[{section}]" not in initial_index:
                raise AgentLoopError(
                    f"{self.benchmark} directory lacks required section: {section}"
                )

    def bind_module7(
        self,
        schema_root: str | Path,
        *,
        input_id: str,
        task_document_name: str,
        task_document: str,
        initial_index: str,
        prompt_variant: PromptVariant = "BEG",
    ) -> BoundModule7:
        """Bind an overrideable schema and construct one frozen run input."""

        self.validate_task_document(task_document_name)
        schema = load_prediction_schema(
            schema_root,
            self.prediction_schema_filename,
        )
        contract = RepoPredictionContract(
            self.benchmark,
            schema,
            grounding_profile=self.grounding_profile,
        )
        prompt = build_repo_initial_user_prompt(
            input_id=input_id,
            benchmark=self.benchmark,
            task_prompt=load_task_prompt(prompt_variant, self.benchmark),
            task_document=task_document,
            initial_index=initial_index,
        )
        return BoundModule7(
            prediction_schema=schema,
            initial_user_prompt=prompt,
            max_rounds=self.max_rounds,
            finish_contract=contract,
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "max_rounds": self.max_rounds,
            "task_document_filename": self.task_document_filename,
            "directory_strategy": self.directory_strategy,
            "required_directory_sections": list(
                self.required_directory_sections
            ),
            "task_prompts": {
                "raw": f"prompts/baseline/{self.benchmark}.txt",
                "graph": f"prompts/BEG/{self.benchmark}.txt",
            },
            "prediction_schema_filename": self.prediction_schema_filename,
            "grounding_profile": self.grounding_profile,
            "prediction_request": self.prediction_request.public_dict(),
        }


def repo_benchmark_config(benchmark: str) -> RepoBenchmarkConfig:
    """Load one explicit Repo benchmark config; FeedbackTrace is absent."""

    if benchmark == "specgap":
        from .specgap import CONFIG

        return CONFIG
    if benchmark == "silentswap":
        from .silentswap import CONFIG

        return CONFIG
    raise AgentLoopError(
        "FeedbackTrace is one-shot and has no interactive AgentLoop config"
    )
