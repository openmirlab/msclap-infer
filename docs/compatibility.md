# Compatibility and parity

## Recorded baseline

`tests/fixtures/upstream_cpu.npz` was recorded on 2026-11-13 using official
`msclap==1.3.4` against the pinned Microsoft 2023 checkpoint, before the
inference-only extraction. The fixture includes two seven-second, 44.1 kHz
mono clips with genuine digital silence and nonzero tones; the source WAVs
and two text prompts are committed with it. `upstream_cpu.json` records
Torch `2.7.0+cu126` in CPU mode, Transformers `4.57.6`, NumPy `2.5.3`, four
Torch threads, and the pinned checkpoint revision.

On the same host and runtime, this package's raw audio embeddings, text
embeddings and learned-scale similarity logits are **bit-identical** to the
official outputs. On a different kernel/runtime, the golden test accepts
`rtol=atol=2e-5`; this is a numerical portability limit, not an accuracy claim.
The fixture is small enough to commit; checkpoint weights are not.

To regenerate the baseline in a separate environment with official
`msclap==1.3.4`, run:

```bash
python tools/capture_upstream.py \
  --checkpoint /path/to/CLAP_weights_2023.pth \
  --output /tmp/msclap-upstream-regeneration
```

Compare that output to the committed fixture before replacing any fixture in
Git. The script verifies the author checkpoint SHA-256 and invokes the
official file, text and similarity APIs. It is source-only, outside the wheel.

An additional end-to-end A/B used two real Rytho sample previews, converted
to mono 44.1 kHz and cropped to exact seven-second windows. In the same
recorded CPU runtime, both clips' raw audio embeddings, two text embeddings,
and the full similarity matrix matched the official wrapper **bit-for-bit**.
No Rytho audio is included in this repository. As a sensitivity check, making
the mel matrix contiguous in memory broke the fixture equality (maximum
audio embedding difference `1.91e-6`), so the exact fixture catches that
numeric regression.

## Dependency choices

The official wheel constrains NumPy to `<2` and librosa to `<0.11`. Those
metadata limits conflict with the target environment. Here, the checkpoint's
own frozen STFT kernels and mel matrix replace the `torchlibrosa` runtime
dependency, with matrix stride preserved to retain exact CPU outputs.
The text encoder constructs pinned GPT-2 architecture from config and loads
all weights from the Microsoft CLAP checkpoint; no second GPT-2 model weight
download occurs.

Transformers is temporarily bounded at `<5` because major-version behavior
of GPT-2/tokenizer is unverified. Review the ceiling when Transformers 5 has
an actual release and run the golden suite there before removing it. The
minimum `4.57` is the tested modern runtime. NumPy 2 support is required.

This package currently verifies CPU parity. CUDA loading is implemented with
an explicit device path but does not claim bit-exact parity across GPU kernels.
