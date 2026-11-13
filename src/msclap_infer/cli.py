"""Small inference-only JSON CLI using optional WAV file reading."""

import argparse
import json

from .session import MSClapSession


def main() -> None:
    parser = argparse.ArgumentParser(prog="msclap-infer")
    parser.add_argument("audio", nargs="?")
    parser.add_argument("--text", action="append")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if bool(args.audio) == bool(args.text):
        parser.error("provide one WAV path or one or more --text prompts")
    with MSClapSession(device=args.device, cache_dir=args.cache_dir,
                       local_files_only=args.local_files_only) as session:
        if args.audio:
            try:
                import soundfile as sf
            except ImportError as exc:
                raise ImportError("WAV CLI requires msclap-infer[wav]") from exc
            audio, sample_rate = sf.read(args.audio, dtype="float32")
            result = session.embed_audio(audio, sample_rate)
        else:
            result = session.embed_text(args.text)
    print(json.dumps({"embeddings": result.tolist(), "semantics": session.semantics().__dict__}))


if __name__ == "__main__":
    main()
