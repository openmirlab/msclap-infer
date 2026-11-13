"""Independent MS-CLAP 2023 inference lifecycle and exact-window encoders.

Reads: verified checkpoint resolver, extracted author model, pinned GPT-2
config/tokenizer, result semantics. Tactus or another caller owns windowing.
"""

from __future__ import annotations

import gc
import re
from pathlib import Path

import numpy as np

from .checkpoints import Checkpoints, default_checkpoints, resolve_asset
from .results import result_semantics

_ALLOWED_LEGACY_KEYS = {
    f"caption_encoder.base.h.{layer}.attn.{field}"
    for layer in range(12)
    for field in ("bias", "masked_bias")
}


def _resolve_device(value: str) -> str:
    import torch

    if value == "mps":
        raise ValueError("MS-CLAP inference does not support device='mps'")
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cpu":
        return value
    if value == "cuda" or re.fullmatch(r"cuda:[0-9]+", value):
        target = torch.device(value)
        if not torch.cuda.is_available() or target.index is not None and target.index >= torch.cuda.device_count():
            raise ValueError(f"Requested CUDA device is unavailable: {value}")
        return value
    raise ValueError(f"Unsupported device: {value}")


class MSClapSession:
    """Resident encoder; one session is not safe for simultaneous inference calls."""

    def __init__(
        self,
        *,
        device: str = "auto",
        cache_dir: str | Path | None = None,
        checkpoints: Checkpoints | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.device = device
        self.cache_dir = cache_dir
        self.checkpoints = checkpoints or default_checkpoints()
        self.local_files_only = local_files_only
        self._model = None
        self._tokenizer = None
        self._active_device = None
        self._status = "unloaded"

    @property
    def status(self) -> str:
        return self._status

    def cache_info(self) -> dict:
        result = {}
        for name, asset in (("clap", self.checkpoints.clap), ("gpt2", self.checkpoints.gpt2)):
            try:
                path = resolve_asset(asset, cache_dir=self.cache_dir, local_files_only=True)
            except (OSError, ValueError, FileNotFoundError):
                path = None
            result[name] = {"cached": path is not None, "path": str(path) if path else None,
                            "revision": asset.revision}
        return result

    def load(self) -> MSClapSession:
        if self._status == "closed":
            raise RuntimeError("MS-CLAP session is closed")
        if self._model is not None:
            return self
        device = _resolve_device(self.device)
        weight_dir = resolve_asset(
            self.checkpoints.clap, cache_dir=self.cache_dir, local_files_only=self.local_files_only
        )
        text_dir = resolve_asset(
            self.checkpoints.gpt2, cache_dir=self.cache_dir, local_files_only=self.local_files_only
        )
        import torch
        from transformers import AutoConfig, AutoTokenizer

        from ._impl.clap import CLAP

        config = AutoConfig.from_pretrained(str(text_dir), local_files_only=True)
        model = CLAP(
            audioenc_name="HTSAT", sample_rate=44100, window_size=1024, hop_size=320,
            mel_bins=64, fmin=50, fmax=8000, classes_num=527, out_emb=768,
            text_model="gpt2", transformer_embed_dim=768, d_proj=1024,
            text_config=config,
        )
        weight_path = weight_dir / "CLAP_weights_2023.pth"
        try:
            state = torch.load(weight_path, map_location="cpu", weights_only=True)["model"]
        except Exception as exc:
            raise RuntimeError(f"Unable to read verified CLAP checkpoint: {weight_path}") from exc
        incompat = model.load_state_dict(state, strict=False)
        missing, unexpected = set(incompat.missing_keys), set(incompat.unexpected_keys)
        if missing or unexpected != _ALLOWED_LEGACY_KEYS:
            raise RuntimeError(
                f"CLAP checkpoint key mismatch: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        tokenizer = AutoTokenizer.from_pretrained(str(text_dir), local_files_only=True)
        tokenizer.add_special_tokens({"pad_token": "!"})
        self._model = model.to(device).eval()
        self._tokenizer = tokenizer
        self._active_device = device
        self._status = "ready"
        return self

    def embed_audio(self, audio: np.ndarray, sample_rate: int, *, normalize: bool = True) -> np.ndarray:
        """Embed a mono 7-second 44.1 kHz waveform or batch of such waveforms."""
        self._require_ready()
        if sample_rate != 44100:
            raise ValueError("MS-CLAP 2023 requires 44100 Hz; resample before calling")
        array = np.asarray(audio)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != 308700:
            raise ValueError("Expected mono [308700] or batch [B, 308700] (7 seconds)")
        if array.shape[0] < 1 or not np.issubdtype(array.dtype, np.number):
            raise ValueError("Audio batch must contain numeric samples")
        array = np.asarray(array, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("Audio contains NaN or infinity")
        import torch
        import torch.nn.functional as F

        with torch.inference_mode():
            tensor = torch.from_numpy(np.ascontiguousarray(array)).to(self._active_device)
            result = self._model.audio_encoder(tensor)[0]
            if normalize:
                result = F.normalize(result, dim=-1)
        return result.detach().cpu().numpy().copy()

    def embed_text(self, descriptions: list[str] | tuple[str, ...], *, normalize: bool = True) -> np.ndarray:
        """Embed English prompts with the author's exact 2023 GPT-2 tokenization."""
        self._require_ready()
        if not isinstance(descriptions, (list, tuple)) or not descriptions or any(
            not isinstance(t, str) or not t.strip() for t in descriptions
        ):
            raise ValueError("descriptions must contain nonempty strings")
        import torch
        import torch.nn.functional as F

        batches = []
        for description in descriptions:
            tokens = self._tokenizer.encode_plus(
                text=description + " <|endoftext|>", add_special_tokens=True,
                max_length=77, padding="max_length", return_tensors="pt",
            )
            if tokens["input_ids"].shape[-1] != 77:
                raise ValueError("Text exceeds the 77-token MS-CLAP 2023 limit")
            batches.append({key: value.reshape(-1) for key, value in tokens.items()
                            if key in ("input_ids", "attention_mask")})
        batch = {key: torch.stack([item[key] for item in batches]).to(self._active_device)
                 for key in ("input_ids", "attention_mask")}
        with torch.inference_mode():
            result = self._model.caption_encoder(batch)
            if normalize:
                result = F.normalize(result, dim=-1)
        return result.detach().cpu().numpy().copy()

    def similarity(self, audio_embeddings: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
        """Author-compatible scaled audio-by-text logits, not probabilities."""
        self._require_ready()
        audio_array = np.asarray(audio_embeddings, dtype=np.float32)
        text_array = np.asarray(text_embeddings, dtype=np.float32)
        if (audio_array.ndim != 2 or text_array.ndim != 2
                or audio_array.shape != (audio_array.shape[0], 1024)
                or text_array.shape != (text_array.shape[0], 1024)
                or not audio_array.shape[0] or not text_array.shape[0]):
            raise ValueError("Expected audio [A, 1024] and text [T, 1024] embeddings")
        if not np.isfinite(audio_array).all() or not np.isfinite(text_array).all():
            raise ValueError("Embeddings contain NaN or infinity")
        if np.any(np.linalg.norm(audio_array, axis=1) == 0) or np.any(np.linalg.norm(text_array, axis=1) == 0):
            raise ValueError("Embeddings must be nonzero")
        import torch

        with torch.inference_mode():
            audio = torch.from_numpy(audio_array).to(self._active_device)
            text = torch.from_numpy(text_array).to(self._active_device)
            audio = audio / torch.norm(audio, dim=-1, keepdim=True)
            text = text / torch.norm(text, dim=-1, keepdim=True)
            result = (self._model.logit_scale.exp() * text @ audio.T).T
        return result.detach().cpu().numpy().copy()

    def infer(self, *, audio: np.ndarray | None = None, sample_rate: int | None = None,
              descriptions: list[str] | tuple[str, ...] | None = None) -> np.ndarray:
        """Ready-only dispatcher for the two package-owned embedding tasks."""
        if (audio is None) == (descriptions is None):
            raise ValueError("Provide exactly one of audio or descriptions")
        if audio is not None:
            if sample_rate is None:
                raise ValueError("sample_rate is required for bare audio arrays")
            return self.embed_audio(audio, sample_rate)
        return self.embed_text(descriptions)

    def semantics(self):
        return result_semantics()

    def _require_ready(self) -> None:
        if self._status != "ready":
            raise RuntimeError("Call load() before inference")

    def release(self) -> None:
        self._model = None
        self._tokenizer = None
        self._active_device = None
        gc.collect()
        if self._status != "closed":
            self._status = "unloaded"

    def close(self) -> None:
        self.release()
        self._status = "closed"

    def __enter__(self) -> MSClapSession:  # noqa: PYI034 - Python 3.10 lacks typing.Self
        return self.load()

    def __exit__(self, *_exc: object) -> None:
        self.close()


def embed_audio(audio: np.ndarray, sample_rate: int, **session_options: object) -> np.ndarray:
    with MSClapSession(**session_options) as session:
        return session.embed_audio(audio, sample_rate)


def embed_text(descriptions: list[str] | tuple[str, ...], **session_options: object) -> np.ndarray:
    with MSClapSession(**session_options) as session:
        return session.embed_text(descriptions)
