# Changelog

## Unreleased

- Gate pushes and pull requests with Python 3.10–3.15 offline tests, lint, and a clean wheel-from-sdist import check; keep real-checkpoint parity separate.
- Reorganized the README around upstream credit, citation, public GitHub installation, usage, weights, development, license, and support.

## 0.1.0 — 2026-11-13

- Extracted the MS-CLAP 2023 audio and text encoders into a standalone inference package.
- Added pinned, SHA-256-verified model assets; independent session and CLI.
- Removed the original package's NumPy/librosa dependency limits and its
  torchlibrosa runtime dependency, retaining exact CPU output against the
  recorded upstream fixture in the reference environment.
