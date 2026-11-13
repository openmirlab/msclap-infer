"""Package-owned machine-readable embedding semantics."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingSemantics:
    version: str = "msclap-2023-embedding-v1"
    dimension: int = 1024
    audio_sample_rate: int = 44100
    audio_samples: int = 308700
    audio_channels: int = 1
    input_duration_seconds: float = 7.0
    normalization: str = "L2 unit norm"
    score: str = "normalized embeddings: cosine similarity; similarity(): learned-scale logits, not probabilities"
    audio_scope: str = "provided 7-second window only"
    text_scope: str = "provided text prompt only"


def result_semantics() -> EmbeddingSemantics:
    return EmbeddingSemantics()
