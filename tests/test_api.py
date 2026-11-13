"""Fast checks for package metadata, boundaries and lifecycle."""

import numpy as np
import pytest

from msclap_infer import MSClapSession, __version__, default_checkpoints, result_semantics
from msclap_infer.checkpoints import Asset, resolve_asset
from msclap_infer.session import _resolve_device


def test_public_import_and_config():
    assert __version__ == "0.1.0"
    assert result_semantics().dimension == 1024
    checkpoints = default_checkpoints()
    assert checkpoints.clap.repo_id == "microsoft/msclap"
    assert len(checkpoints.clap.hashes["CLAP_weights_2023.pth"]) == 64
    assert set(checkpoints.gpt2.hashes) == {
        "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"
    }


def test_session_lifecycle_and_guards():
    session = MSClapSession(device="cpu", local_files_only=True)
    assert session.status == "unloaded"
    with pytest.raises(RuntimeError, match="load"):
        session.embed_audio(np.zeros(308_700, dtype=np.float32), 44_100)
    with pytest.raises(ValueError, match="exactly one"):
        session.infer()
    session.release()
    assert session.status == "unloaded"
    session.close()
    assert session.status == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        session.load()


@pytest.mark.parametrize("device", ["mps", "cpu:0", "cuda:abc"])
def test_invalid_device_is_rejected(device):
    with pytest.raises(ValueError):
        _resolve_device(device)


def test_asset_sha256_guard_and_cache_resolver(tmp_path, monkeypatch):
    payload = tmp_path / "weight.bin"
    payload.write_bytes(b"author checkpoint bytes")
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **_kwargs: str(tmp_path))
    asset = Asset("example/model", "pinned", {"weight.bin": "0" * 64})
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolve_asset(asset, local_files_only=True)
    import hashlib

    valid = Asset(asset.repo_id, asset.revision, {"weight.bin": hashlib.sha256(payload.read_bytes()).hexdigest()})
    assert resolve_asset(valid, local_files_only=True) == tmp_path
