"""One resolver for loading and cache inspection of both pinned assets.

Reads: package checkpoint TOML, Hugging Face hub, SHA-256. The CLAP pickle is
deserialized only after its author SHA-256 has been checked.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


@dataclass(frozen=True)
class Asset:
    repo_id: str
    revision: str
    hashes: dict[str, str]


@dataclass(frozen=True)
class Checkpoints:
    clap: Asset
    gpt2: Asset


def default_checkpoints() -> Checkpoints:
    data = tomllib.loads(files("msclap_infer").joinpath("config/checkpoints.toml").read_text())
    clap = data["clap_2023"]
    gpt2 = data["gpt2"]
    return Checkpoints(
        clap=Asset(clap["repo_id"], clap["revision"], {clap["filename"]: clap["sha256"]}),
        gpt2=Asset(gpt2["repo_id"], gpt2["revision"], dict(gpt2["files"])),
    )


def resolve_asset(
    asset: Asset, *, cache_dir: str | Path | None = None, local_files_only: bool = False
) -> Path:
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            repo_id=asset.repo_id,
            revision=asset.revision,
            allow_patterns=list(asset.hashes),
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
        )
    )
    for name, expected in asset.hashes.items():
        file_path = path / name
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing {name} in {path}")
        hasher = hashlib.sha256()
        with file_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch for {file_path}")
    return path
