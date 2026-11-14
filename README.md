# msclap-infer

Embed seven-second audio windows and text with Microsoft CLAP 2023.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

## Why this exists

The [original Microsoft CLAP package](https://github.com/microsoft/CLAP) provides the 2023 audio–text model, but its published dependency constraints conflict with current NumPy/Transformers installations and its file wrapper randomly crops long audio. This standalone package retains the author's inference model, accepts exact NumPy windows, verifies pinned files, and leaves long-audio slicing to the caller. OpenMIRLab maintains this packaging; the model and research are Microsoft's.

## Acknowledgments

- Original research: Benjamin Elizalde, Soham Deshmukh, and Huaming Wang ([CLAP 2023 paper](https://arxiv.org/abs/2309.05767)).
- Original code and model: [Microsoft CLAP](https://github.com/microsoft/CLAP).
- Checkpoint host: [Hugging Face, microsoft/msclap](https://huggingface.co/microsoft/msclap). OpenMIRLab did not train these weights.

## Citation

For the 2023 checkpoint, cite the [authors' CLAP 2023 paper](https://arxiv.org/abs/2309.05767):

```bibtex
@misc{CLAP2023,
  title={Natural Language Supervision for General-Purpose Audio Representations},
  author={Benjamin Elizalde and Soham Deshmukh and Huaming Wang},
  year={2023},
  eprint={2309.05767},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2309.05767}
}
```

## Features

- Direct NumPy audio and English text embedding into a shared 1024-dimensional space.
- Learned-scale similarity logits, explicit CPU/CUDA device selection, and reusable sessions.
- Verified author checkpoint and GPT-2 config/tokenizer, with manual offline cache setup.

## Scope

- Audio and text embeddings: 1024 dimensions each.
- Learned-scale audio/text similarity logits for retrieval and zero-shot ranking.
- Explicit `cpu`, `cuda`, `cuda:N`, and `auto` device selection.
- No training, evaluation framework, 2022 model, or captioning model.
- No checkpoint weights in the package.

## Install

Python 3.10 or later:

```bash
pip install 'msclap-infer @ git+https://github.com/openmirlab/msclap-infer.git'
# For the WAV-reading CLI, include the optional wav extra:
pip install 'msclap-infer[wav] @ git+https://github.com/openmirlab/msclap-infer.git'
```

There is no PyPI release yet. A local checkout can be installed with `pip install -e .`.

## Quick start

The caller supplies **mono, 44.1 kHz, exactly seven-second** windows as float
NumPy arrays. For longer recordings, the caller owns resampling, windowing and
aggregation. This makes every embedding's time span explicit and avoids the
original file wrapper's random crop of long inputs.

```python
import numpy as np
from msclap_infer import MSClapSession

t = np.arange(308_700, dtype=np.float32) / 44_100
audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
prompts = ["A steady electronic tone plays.", "A drum kit plays."]
with MSClapSession(device="cpu") as model:
    audio_vectors = model.embed_audio(audio, sample_rate=44_100)
    text_vectors = model.embed_text(prompts)
    scores = model.similarity(audio_vectors, text_vectors)
    meaning = model.semantics()
```

## Usage and output

`audio_vectors` has shape `[1, 1024]`, `text_vectors` shape `[2, 1024]`,
and `scores` shape `[1, 2]`. Embeddings are L2 normalized by default.
`similarity()` returns the author's learned-scale logits, **not probabilities**.
Plain cosine similarity of normalized vectors yields the same ranking.
Passing `normalize=False` returns the author's raw embeddings. Empty text,
nonfinite audio and wrong window length raise an error. Text inputs should be
English, as in the original model; prompts longer than 77 GPT-2 tokens are
rejected.

`MSClapSession` also provides `load()`, ready-only `infer(...)`, `release()`,
`close()`, `status`, `cache_info()` and context-manager support. The one-shot
`embed_audio()` and `embed_text()` functions use the same session implementation.
The command `msclap-infer path/to/seven-second.wav --device cpu` returns JSON;
`msclap-infer --text "A piano plays."` embeds text.

## Checkpoints and offline use

On first `load()`, the package downloads the author's 2023 checkpoint and the
GPT-2 **config/tokenizer only** through the Hugging Face cache. All six files
are SHA-256 verified against `src/msclap_infer/config/checkpoints.toml`.
The CLAP checkpoint itself contains the GPT-2 model parameters. The package
does not download a separate GPT-2 weight file.

The default cache is Hugging Face's `~/.cache/huggingface/hub`. Pass
`cache_dir="/some/folder"` to use another directory. For manual download or
offline deployment, pre-populate that cache with the pinned revisions:

```bash
hf download microsoft/msclap CLAP_weights_2023.pth \
  --revision c47d441165daa21986ead0850660917636a81775 --cache-dir /some/folder
hf download gpt2 config.json tokenizer.json tokenizer_config.json vocab.json merges.txt \
  --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e --cache-dir /some/folder
```

Then use `MSClapSession(cache_dir="/some/folder", local_files_only=True)`.
The exact source URLs, hashes and licenses live in the checkpoint TOML.
`Checkpoints` and `Asset` allow a caller to substitute a verified repository
revision without altering package code.

The [Microsoft checkpoint model card](https://huggingface.co/microsoft/msclap)
declares MS-PL. The adapted source code is MIT licensed; the downloaded weights
have separate terms. See [NOTICE](NOTICE).

## What this project will never bundle

The package never includes checkpoint weights, training loops, evaluation datasets/frameworks, the 2022 model, or the captioning model.

## Development

GitHub CI runs offline tests (`pytest -m 'not model'`) and lint on Python
3.10–3.15. A separate job builds the wheel from its sdist and imports that
wheel in a clean Python 3.12 environment. The real-checkpoint golden remains
an explicit additional verification step because it downloads model assets.

```bash
pip install '.[dev]'
pytest -q
ruff check src tests
python -m build
```

The golden test downloads the pinned checkpoint if it is not cached. It
compares audio embeddings, text embeddings and similarity against an upstream
`msclap==1.3.4` CPU fixture with genuine silence. Exact equality is asserted
only under the fixture's recorded Torch/Transformers/NumPy environment; other
environments use a documented numerical tolerance. See
[compatibility notes](docs/compatibility.md).

## License

The adapted package source is MIT licensed. The downloaded checkpoint is separately marked MS-PL by its [model card](https://huggingface.co/microsoft/msclap); that license governs its use. See [LICENSE](LICENSE), [NOTICE](NOTICE), and the pinned [checkpoint manifest](src/msclap_infer/config/checkpoints.toml).

## Support

Report package bugs in [openmirlab/msclap-infer issues](https://github.com/openmirlab/msclap-infer/issues). Research and upstream model questions belong with [Microsoft CLAP](https://github.com/microsoft/CLAP).
