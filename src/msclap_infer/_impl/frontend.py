"""Pure-Torch runtime for the frozen torchlibrosa frontend parameters.

The pinned MS-CLAP checkpoint contains the real/imag STFT Conv1d kernels and
mel matrix. We instantiate their exact parameter layout, then load all values
from the author checkpoint; no librosa or torchlibrosa initializer is needed.
Reads: author torchlibrosa STFT/Spectrogram/LogmelFilterBank forward operations.
"""

import torch
import torch.nn.functional as F
from torch import nn


class _STFT(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, center: bool, pad_mode: str):
        super().__init__()
        self.n_fft = n_fft
        self.center = center
        self.pad_mode = pad_mode
        self.conv_real = nn.Conv1d(1, n_fft // 2 + 1, n_fft, stride=hop_length, bias=False)
        self.conv_imag = nn.Conv1d(1, n_fft // 2 + 1, n_fft, stride=hop_length, bias=False)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, samples):
        x = samples[:, None, :]
        if self.center:
            x = F.pad(x, (self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)
        real = self.conv_real(x)[:, None, :, :].transpose(2, 3)
        imag = self.conv_imag(x)[:, None, :, :].transpose(2, 3)
        return real, imag


class Spectrogram(nn.Module):
    def __init__(
        self, n_fft=1024, hop_length=320, win_length=None, window="hann",
        center=True, pad_mode="reflect", freeze_parameters=True, power=2.0,
    ):
        super().__init__()
        if win_length not in (None, n_fft) or window != "hann" or not freeze_parameters:
            raise ValueError("MS-CLAP 2023 requires its fixed frozen Hann STFT")
        self.stft = _STFT(n_fft, hop_length, center, pad_mode)
        self.power = power

    def forward(self, samples):
        real, imag = self.stft(samples)
        spectrogram = real ** 2 + imag ** 2
        return spectrogram if self.power == 2.0 else spectrogram ** (self.power / 2.0)


class LogmelFilterBank(nn.Module):
    def __init__(
        self, sr=32000, n_fft=1024, n_mels=64, fmin=50, fmax=14000,
        is_log=True, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True,
    ):
        super().__init__()
        if not (is_log and ref == 1.0 and top_db is None and freeze_parameters):
            raise ValueError("MS-CLAP 2023 requires its fixed logmel configuration")
        # torchlibrosa builds this from a transposed mel matrix. Preserve its
        # non-contiguous stride: BLAS chooses a different numerical path for a
        # contiguous matrix even when checkpoint values are bit-identical.
        self.melW = nn.Parameter(torch.empty(n_mels, n_fft // 2 + 1).T, requires_grad=False)
        self.amin = amin

    def forward(self, spectrogram):
        mel = torch.matmul(spectrogram, self.melW)
        return 10.0 * torch.log10(torch.clamp(mel, min=self.amin))
