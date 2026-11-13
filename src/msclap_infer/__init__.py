"""Public inference-only MS-CLAP 2023 API."""

from .__about__ import __version__
from .checkpoints import Asset, Checkpoints, default_checkpoints
from .results import EmbeddingSemantics, result_semantics
from .session import MSClapSession, embed_audio, embed_text

__all__ = ["Asset", "Checkpoints", "EmbeddingSemantics", "MSClapSession", "__version__", "default_checkpoints", "embed_audio", "embed_text", "result_semantics"]
