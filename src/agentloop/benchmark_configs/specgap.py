"""SpecGap-owned inputs injected into the shared Repo AgentLoop."""

from evaluation_core.contracts import GROUNDING_PROFILES

from .model import RepoBenchmarkConfig


CONFIG = RepoBenchmarkConfig(
    benchmark="specgap",
    max_rounds=8,
    task_document_filename="3_document_after.md",
    directory_strategy="specgap_ranked_file_directory_v4",
    required_directory_sections=(
        "FILES LINKED TO THE TASK DOCUMENT",
        "ADDITIONAL FILES WITH OBSERVABLE BEHAVIOR",
    ),
    prediction_schema_filename="specgap_prediction.schema.json",
    grounding_profile=GROUNDING_PROFILES["specgap"],
)
