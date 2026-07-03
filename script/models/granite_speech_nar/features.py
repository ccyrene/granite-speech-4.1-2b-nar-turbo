"""Log-mel feature extraction for Granite Speech NAR (plain torch, no torchaudio).

The mel filterbank + Hann window are precomputed once (from torchaudio) and saved
to ``mel_filters.pt``; at runtime we compute the spectrogram with ``torch.stft`` and
apply the filterbank with a matmul. This reproduces
``torchaudio.transforms.MelSpectrogram`` bit-for-bit (it calls the same
``torch.stft`` under the hood), then applies the reference whisper-style log
compression and 2-frame stacking.

Regenerate ``mel_filters.pt`` with ``script/extract_mel.py`` if the STFT params change.
"""
from __future__ import annotations

import os

import torch
from torch import nn

DEFAULT_MEL_PT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mel_filters.pt")


class MelSpectrogram(nn.Module):
    """Pure-torch equivalent of ``torchaudio.transforms.MelSpectrogram``.

    Uses a precomputed window + mel filterbank loaded from a ``.pt`` payload, so it
    carries no torchaudio dependency.
    """

    def __init__(self, payload: dict | None = None, pt_path: str = DEFAULT_MEL_PT):
        super().__init__()
        if payload is None:
            payload = torch.load(pt_path, map_location="cpu", weights_only=True)
        self.register_buffer("window", payload["window"].float(), persistent=False)
        self.register_buffer("fb", payload["fb"].float(), persistent=False)  # (n_freqs, n_mels)
        self.n_fft = int(payload["n_fft"])
        self.hop_length = int(payload["hop_length"])
        self.win_length = int(payload["win_length"])
        self.pad = int(payload.get("pad", 0))
        self.power = float(payload.get("power", 2.0))
        self.center = bool(payload.get("center", True))
        self.pad_mode = str(payload.get("pad_mode", "reflect"))
        self.onesided = bool(payload.get("onesided", True))
        self.normalized = bool(payload.get("normalized", False))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # The mel front-end MUST run in float32 to stay bit-identical to the
        # reference: window/fb/waveform are forced to float here so that an
        # outer ``.to(dtype=bf16)`` on this module cannot change feature numerics
        # (device movement remains fine).
        waveform = waveform.float()
        window = self.window.float()
        fb = self.fb.float()
        if self.pad > 0:
            waveform = torch.nn.functional.pad(waveform, (self.pad, self.pad), "constant")
        # Spectrogram (matches torchaudio F.spectrogram with these defaults)
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True,
        )
        spec = spec.abs().pow(self.power)  # power spectrogram (B, n_freqs, n_frames)
        # MelScale: (spec^T @ fb)^T  -> (B, n_mels, n_frames)
        mel = torch.matmul(spec.transpose(-1, -2), fb).transpose(-1, -2)
        return mel


class MelFeatureExtractor(nn.Module):
    """Extracts stacked 80-band log-mel features (160-dim, half frame rate)."""

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        sampling_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        mel_pt_path: str = DEFAULT_MEL_PT,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mel_filters = MelSpectrogram(pt_path=mel_pt_path)

    @torch.no_grad()
    def _extract_features(self, raw_audio: torch.Tensor) -> torch.Tensor:
        mel_filters = self.mel_filters.to(raw_audio.device)
        B, T = raw_audio.shape
        l = 2 * (T // (2 * self.hop_length))
        mel = mel_filters(raw_audio.float())[..., :l]
        logmel = mel.transpose(-1, -2).clamp_min_(1e-10).log10_()
        mx = logmel.amax(dim=(-2, -1), keepdim=True)
        logmel = torch.maximum(logmel, mx - 8.0).div_(4).add_(1)
        return logmel.reshape(B, -1, 2 * self.n_mels)

    def _pad_pinned_h2d(self, seqs: list[torch.Tensor], device) -> torch.Tensor:
        """P2.5: pad CPU waveforms into a reused pinned ping-pong buffer and issue ONE
        ``non_blocking`` H2D copy (replaces per-sample blocking ``.to(device)`` copies).
        Bit-identical: pad value 0.0, same (B, T_max) layout as ``pad_sequence``.

        Two buffers alternate; a recorded CUDA event guards against overwriting a buffer
        whose DMA is still in flight (matters when prep runs ahead on a side stream)."""
        B = len(seqs)
        T = max(int(s.shape[-1]) for s in seqs)
        if not hasattr(self, "_pin_bufs"):
            self._pin_bufs = [None, None]
            self._pin_evts = [None, None]
            self._pin_i = 0
        i = self._pin_i
        self._pin_i ^= 1
        buf = self._pin_bufs[i]
        if buf is None or buf.shape[0] < B or buf.shape[1] < T:
            rows = max(B, buf.shape[0] if buf is not None else 0)
            cols = max(T, buf.shape[1] if buf is not None else 0)
            buf = torch.empty(rows, cols, dtype=torch.float32, pin_memory=True)
            self._pin_bufs[i] = buf
            self._pin_evts[i] = None
        evt = self._pin_evts[i]
        if evt is not None:
            evt.synchronize()                      # previous DMA from this buffer must be done
        for r, s in enumerate(seqs):
            n = int(s.shape[-1])
            row = buf[r]
            row[:n].copy_(s if s.dtype == torch.float32 else s.float())
            if n < T:
                row[n:T].zero_()
        out = buf[:B, :T].to(device, non_blocking=True)
        e = torch.cuda.Event()
        e.record()                                 # on the caller's current stream
        self._pin_evts[i] = e
        return out

    @torch.no_grad()
    def __call__(self, audios, device: str | None = None) -> dict:
        if isinstance(audios, torch.Tensor):
            if audios.ndim == 1:
                audios = [audios]
            elif audios.ndim == 2:
                audios = [audios[i] for i in range(audios.shape[0])]
            else:
                raise ValueError(f"Expected 1-D or 2-D tensor, got {audios.ndim}-D")

        raw_lengths = [a.shape[-1] for a in audios]
        encoder_frame_counts = [l // (2 * self.hop_length) for l in raw_lengths]

        # Move the waveforms to the target device BEFORE padding. `pad_sequence` on CPU dominates
        # the whole front-end on a fast GPU / contended host (~150ms vs ~0.03ms on-GPU for one
        # utterance). Default CUDA path (P2.5): one pinned batched non_blocking H2D; fallback
        # (GRANITE_PINNED_H2D=0 / CUDA inputs / CPU device): per-sample move + on-device pad.
        # Bit-identical either way: padding/stft/matmul are device-agnostic in value.
        seqs = [a.squeeze(0) if a.ndim > 1 else a for a in audios]
        use_pinned = (
            device is not None and str(device).startswith("cuda") and torch.cuda.is_available()
            and os.environ.get("GRANITE_PINNED_H2D", "1") != "0"
            and all(not s.is_cuda for s in seqs)
        )
        if use_pinned:
            raw_audio = self._pad_pinned_h2d(seqs, device)
        else:
            if device is not None:
                seqs = [a.to(device) for a in seqs]
            raw_audio = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)

        input_features = self._extract_features(raw_audio)

        max_enc_frames = input_features.shape[1]
        if device is not None:
            # P2.4e: build the mask directly on-device from the host lengths (no CPU mask + move)
            x_sizes = torch.as_tensor(encoder_frame_counts, dtype=torch.long, device=device)
            attention_mask = (torch.arange(max_enc_frames, device=device).unsqueeze(0)
                              < x_sizes.unsqueeze(1))
            input_features = input_features.to(device)   # no-op when already there
        else:
            x_sizes = torch.tensor(encoder_frame_counts, dtype=torch.long)
            attention_mask = torch.arange(max_enc_frames).unsqueeze(0) < x_sizes.unsqueeze(1)

        # encoder_lengths: host ints for the P2.3 host-length consolidation (transcribe* kwargs)
        return {"input_features": input_features, "attention_mask": attention_mask,
                "encoder_lengths": encoder_frame_counts}
