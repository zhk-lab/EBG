"""SilentSwap-owned inputs injected into the shared Repo AgentLoop."""

from evaluation_core.contracts import GROUNDING_PROFILES

from .model import RepoBenchmarkConfig


CONFIG = RepoBenchmarkConfig(
    benchmark="silentswap",
    max_rounds=6,
    task_document_filename="original_document.md",
    directory_strategy="silentswap_ranked_file_directory_v4",
    required_directory_sections=(
        "FILES LINKED TO THE ORIGINAL DOCUMENT",
    ),
    prediction_schema_filename="silentswap_prediction.schema.json",
    grounding_profile=GROUNDING_PROFILES["silentswap"],
)
