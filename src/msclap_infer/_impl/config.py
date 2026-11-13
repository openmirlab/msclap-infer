"""Only the HTSAT inference constants read by the author 2023 model.

These deliberately preserve the author's HTSAT module constants, including its
32 kHz mel-filter setting even though the wrapper expects 44.1 kHz waveforms.
The checkpoint is the numerical source of truth for the frontend parameters.
"""

enable_tscam = True
htsat_attn_heatmap = False
sample_rate = 32000
window_size = 1024
hop_size = 320
mel_bins = 64
fmin = 50
fmax = 14000
