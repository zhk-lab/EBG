"""Load evaluation model profiles from .env without guessing model capabilities."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSettings:
    model: str
    base_url: str
    api_key_env: str
    request_options: dict[str, Any]


def add_model_arguments(parser: argparse.ArgumentParser, *, judge: bool = False) -> None:
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--model-profile", help=".env profile prefix, e.g. GPT_LUNA or QWEN")
    parser.add_argument("--judge-model" if judge else "--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument(
        "--reasoning-effort", help="Provider value such as none/low/high; omit removes it"
    )
    parser.add_argument("--thinking-type", help="Provider thinking.type value; omit removes it")
    parser.add_argument("--enable-thinking", choices=("true", "false", "omit"))
    parser.add_argument(
        "--temperature", help="Sampling temperature; omit uses the provider default"
    )
    parser.add_argument(
        "--top-p", help="Nucleus sampling probability; omit uses the provider default"
    )
    parser.add_argument(
        "--show-config", action="store_true",
        help="Print resolved settings without API calls",
    )


def resolve_model_settings(args: argparse.Namespace, *, judge: bool = False) -> ModelSettings:
    """CLI overrides role settings, which override the selected profile.

    GPT_LUNA_MODEL_NAME can share GPT_BASE_URL, GPT_API_KEY and GPT_REASONING_EFFORT.
    Missing request options stay absent; no model-name-based defaults are applied.
    """
    load_dotenv(getattr(args, "env_file", PROJECT_ROOT / ".env"), override=False)
    role = "JUDGE" if judge else "BEG"
    profile = (
        getattr(args, "model_profile", None)
        or os.environ.get(f"{role}_MODEL_PROFILE", "")
    ).upper()
    prefixes = [profile] if profile else []
    if "_" in profile:
        prefixes.append(profile.rsplit("_", 1)[0])

    def setting(suffix: str, cli: str | None = None) -> str | None:
        override = getattr(args, cli, None) if cli else None
        if override is not None:
            return str(override)
        for prefix in [role, *prefixes]:
            value = os.environ.get(f"{prefix}_{suffix}")
            if value is not None:
                return value.strip()
        return None

    model = (
        setting("MODEL", "judge_model" if judge else "model")
        or setting("MODEL_NAME") or ""
    )
    base_url = setting("BASE_URL", "base_url") or (
        os.environ.get("BEG_API_BASE_URL", "") if not judge else ""
    )
    key_name = setting("API_KEY_ENV", "api_key_env")
    if not key_name:
        key_name = next(
            (f"{p}_API_KEY" for p in [role, *prefixes] if f"{p}_API_KEY" in os.environ),
            f"{role}_API_KEY",
        )

    options: dict[str, Any] = {}
    effort = setting("REASONING_EFFORT", "reasoning_effort")
    if effort and effort != "omit":
        options["reasoning_effort"] = effort
    thinking_type = setting("THINKING_TYPE", "thinking_type")
    if thinking_type and thinking_type != "omit":
        options["thinking"] = {"type": thinking_type}
    enabled = setting("ENABLE_THINKING", "enable_thinking")
    if enabled and enabled != "omit":
        if enabled.lower() not in {"true", "false"}:
            raise ValueError("ENABLE_THINKING must be true, false or omit")
        options["enable_thinking"] = enabled.lower() == "true"
    # Some profiles declare mandatory thinking as a capability, not an API field.
    # Preserve their reasoning effort and let the provider enforce that capability.
    mode = setting("THINKING_MODE")
    if mode not in {None, "", "required", "omit"}:
        raise ValueError(
            "THINKING_MODE supports required or omit; use THINKING_TYPE for an API switch"
        )
    if mode == "required" and (
        thinking_type == "disabled"
        or options.get("enable_thinking") is False
        or effort == "none"
    ):
        raise ValueError("THINKING_MODE=required conflicts with disabled thinking")
    for suffix, flag in (("TEMPERATURE", "temperature"), ("TOP_P", "top_p")):
        value = setting(suffix, flag)
        if value and value != "omit":
            number = float(value)
            upper = 2 if flag == "temperature" else 1
            if not math.isfinite(number) or not 0 <= number <= upper:
                raise ValueError(f"{suffix} must be between 0 and {upper}, or omit")
            options[flag] = number
    if not model or not base_url.startswith(("http://", "https://")):
        raise ValueError(f"Set {role}_MODEL_PROFILE or provide a model and HTTP(S) base URL")
    return ModelSettings(model, base_url.rstrip("/"), key_name, options)


def apply_model_settings(args: argparse.Namespace, *, judge: bool = False) -> ModelSettings:
    settings = resolve_model_settings(args, judge=judge)
    setattr(args, "judge_model" if judge else "model", settings.model)
    args.base_url = settings.base_url
    args.api_key_env = settings.api_key_env
    args.request_options = settings.request_options
    return settings


def public_settings(settings: ModelSettings) -> dict[str, Any]:
    return {
        "model": settings.model,
        "base_url": settings.base_url,
        "api_key_env": settings.api_key_env,
        "request_options": settings.request_options,
    }
