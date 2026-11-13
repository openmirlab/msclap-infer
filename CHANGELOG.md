# Changelog

## 0.1.0 — 2026-11-13

- Extracted the MS-CLAP 2023 audio and text encoders into a standalone inference package.
- Added pinned, SHA-256-verified model assets; independent session and CLI.
- Removed the original package's NumPy/librosa dependency limits and its
  torchlibrosa runtime dependency, retaining exact CPU output against the
  recorded upstream fixture in the reference environment.
