# msclap-infer maintainer guide

Standalone, inference-only MS-CLAP 2023 audio/text encoder. The 2023 model and research are by Benjamin Elizalde, Soham Deshmukh, and Huaming Wang; source is `microsoft/CLAP` and checkpoint host is `microsoft/msclap` on Hugging Face. Public import:
`msclap_infer`; PyPI/CLI name: `msclap-infer`. Version lives only in
`src/msclap_infer/__about__.py`. No training, evaluation, 2022 model,
captioning model, or bundled model weights belong here.

The independent session in `session.py` owns device choice, verified loading,
inference, release and cache inspection. `checkpoints.py` is the single
resolver for both loading and cache inspection; pinned hashes and provenance
live in `config/checkpoints.toml`. `_impl/` contains only the author model
pieces needed for inference and our pure-Torch replacement for the frozen
torchlibrosa frontend. Preserve the frontend mel matrix's noncontiguous
stride: making it contiguous changes CPU embedding bits.

External callers supply exact mono seven-second 44.1 kHz windows. The package
returns normalized 1024D embeddings by default and author-scale similarity
logits. Result interpretation is declared in `results.py`; score is not a
probability. Full-length audio processing belongs to callers.

Checkpoint weights are from `microsoft/msclap` under the model card's MS-PL;
source code is MIT. GPT-2 config/tokenizer are pinned separately under MIT.
Only `checkpoints.toml` should change when updating asset revisions; changes
must include new independent SHA-256 checks and parity evidence.

Verification from the repo root:

```bash
pip install '.[dev]'
pytest -q
ruff check src tests
python -m build
```

`pytest -q` runs the real checkpoint golden test, downloading the pinned
asset if absent. For quick code-only checks use `pytest -q -m 'not model'`.
The golden fixture was captured from official `msclap==1.3.4` before
extraction; its environment and regeneration steps are in
`docs/compatibility.md`. A release must additionally build wheel-from-sdist,
install that wheel in a clean environment, import the package and touch
`MSClapSession` and `default_checkpoints()`.

The repository is public at `openmirlab/msclap-infer`; no PyPI package has
been published. Install source with `pip install 'msclap-infer @ git+https://github.com/openmirlab/msclap-infer.git'`, adding `[wav]` for the WAV CLI. Load-bearing source files carry title/rationale/`Reads:` navigation headers; keep those accurate. Before publishing a package or release, obtain user sign-off per the openmirlab constitution.
