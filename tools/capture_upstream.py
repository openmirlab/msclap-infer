"""Capture an independent fixture with the official msclap package.

Run only in an isolated environment containing official msclap==1.3.4.
This tool is source-only and is absent from the installed wheel.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from msclap import CLAP

EXPECTED_CHECKPOINT_SHA256 = "2cef4016d47d00eb28d153d522f397222057f95000e9bad6b9583c631284a1e6"
PROMPTS = ["A low bass tone plays.", "A high electronic tone plays."]
AUDIO_FILES = ["tone_silence.wav", "second_tone.wav"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if digest != EXPECTED_CHECKPOINT_SHA256:
        parser.error("checkpoint SHA-256 differs from the pinned author checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    source_dir = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    torch.set_num_threads(4)
    model = CLAP(model_fp=args.checkpoint, version="2023", use_cuda=False)
    with torch.inference_mode():
        audio = model.get_audio_embeddings([str(source_dir / name) for name in AUDIO_FILES], resample=False)
        text = model.get_text_embeddings(PROMPTS)
        score = model.compute_similarity(audio, text)
    arrays = {name: value.detach().cpu().numpy() for name, value in
              (("audio", audio), ("text", text), ("score", score))}
    np.savez(args.output / "upstream_cpu.npz", **arrays)
    metadata = {
        "source": "microsoft/CLAP msclap 1.3.4",
        "checkpoint_revision": "c47d441165daa21986ead0850660917636a81775",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "numpy": np.__version__, "device": "cpu", "threads": torch.get_num_threads(),
        "phrases": PROMPTS, "audio": AUDIO_FILES,
        "shapes": {name: list(value.shape) for name, value in arrays.items()},
    }
    (args.output / "upstream_cpu.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
