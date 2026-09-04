"""Shared deterministic AgentLoop infrastructure."""

from .config import DEFAULT_CONFIG, AgentLoopConfig
from .finish_contract import FinishContract
from .provider import ModelCompletion, OpenAICompatibleJsonClient
from .raw_backend import RawBackend
from .runner import AgentLoop, RunOutcome, prepare_initial_request
from .storage import RunStore

__all__ = [
    "AgentLoop",
    "AgentLoopConfig",
    "DEFAULT_CONFIG",
    "FinishContract",
    "GraphBackend",
    "ModelCompletion",
    "OpenAICompatibleJsonClient",
    "RawBackend",
    "RunOutcome",
    "RunStore",
    "prepare_initial_request",
]


def __getattr__(name: str):
    if name == "GraphBackend":
        from .graph_backend import GraphBackend

        return GraphBackend
    raise AttributeError(name)
