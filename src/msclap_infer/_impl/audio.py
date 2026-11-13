"""The 2023 checkpoint uses only the local HTSAT audio tower."""

from .htsat import HTSATWrapper


def get_audio_encoder(name: str):
    if name != "HTSAT":
        raise ValueError(f"Unsupported MS-CLAP 2023 audio tower: {name}")
    return HTSATWrapper
