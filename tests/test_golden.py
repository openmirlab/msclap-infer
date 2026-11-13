"""End-to-end parity against captured msclap 1.3.4 author outputs."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import transformers

from msclap_infer import MSClapSession

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.model
def test_author_cpu_audio_text_and_similarity():
    metadata = json.loads((FIXTURES / "upstream_cpu.json").read_text())
    golden = np.load(FIXTURES / "upstream_cpu.npz")
    audio = np.stack([
        sf.read(FIXTURES / name, dtype="float32")[0] for name in metadata["audio"]
    ])
    assert audio.shape == (2, 308_700)
    assert np.any(audio[0] == 0) and np.any(audio[0] != 0)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(metadata["threads"])
    try:
        with MSClapSession(device="cpu") as session:
            raw_audio = session.embed_audio(audio, 44_100, normalize=False)
            raw_text = session.embed_text(metadata["phrases"], normalize=False)
            scores = session.similarity(raw_audio, raw_text)
            normalized_audio = session.embed_audio(audio[0], 44_100)
            normalized_text = session.embed_text(metadata["phrases"])
            assert session.status == "ready"
            assert session.load() is session
            assert session.cache_info()["clap"]["cached"]
            assert session.infer(audio=audio[0], sample_rate=44_100).shape == (1, 1024)
            assert session.infer(descriptions=metadata["phrases"]).shape == (2, 1024)
            assert np.isclose(np.linalg.norm(normalized_audio[0]), 1.0, atol=1e-6)
            assert np.allclose(np.linalg.norm(normalized_text, axis=1), 1.0, atol=1e-6)
            with pytest.raises(ValueError, match="44100"):
                session.embed_audio(audio[0], 16_000)
            with pytest.raises(ValueError, match="308700"):
                session.embed_audio(audio[0, :-1], 44_100)
            with pytest.raises(ValueError, match="NaN"):
                session.embed_audio(np.full(308_700, np.nan, dtype=np.float32), 44_100)
            with pytest.raises(ValueError, match="nonempty"):
                session.embed_text([""])
        assert session.status == "closed"
    finally:
        torch.set_num_threads(previous_threads)

    exact_env = (
        torch.__version__ == metadata["torch"]
        and transformers.__version__ == metadata["transformers"]
        and np.__version__ == metadata["numpy"]
    )
    for name, actual in (("audio", raw_audio), ("text", raw_text), ("score", scores)):
        expected = golden[name]
        assert actual.shape == expected.shape
        assert np.max(np.abs(expected)) > 0.1
        if exact_env:
            assert np.array_equal(actual, expected), name
        else:
            # Runtime kernels may choose different floating-point reduction paths.
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
