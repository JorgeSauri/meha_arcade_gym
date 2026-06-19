from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


AUDIO_SAMPLE_RATE = 16_000
AUDIO_FEATURE_DIM = 16
MEL_BINS = 32
MEL_FRAMES = 24


@dataclass
class AudioEvent:
    kind: str
    sample_rate: int
    waveform: np.ndarray
    audio_features: np.ndarray
    mel_spectrogram: np.ndarray

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "sample_rate": int(self.sample_rate),
            "audio_features": self.audio_features.astype(float).tolist(),
            "mel_spectrogram": self.mel_spectrogram.astype(float).tolist(),
            "mel_bins": int(self.mel_spectrogram.shape[0]),
            "mel_frames": int(self.mel_spectrogram.shape[1]),
        }


def empty_audio_event(kind: str = "silence") -> AudioEvent:
    return AudioEvent(
        kind=kind,
        sample_rate=AUDIO_SAMPLE_RATE,
        waveform=np.zeros((1,), dtype=np.float32),
        audio_features=np.zeros((AUDIO_FEATURE_DIM,), dtype=np.float32),
        mel_spectrogram=np.zeros((MEL_BINS, MEL_FRAMES), dtype=np.float32),
    )


def _adsr(length: int, attack: float = 0.08, release: float = 0.35) -> np.ndarray:
    if length <= 1:
        return np.ones((max(1, length),), dtype=np.float32)
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    env = np.ones_like(t)
    attack = max(0.01, min(0.45, float(attack)))
    release = max(0.05, min(0.8, float(release)))
    env[t < attack] = t[t < attack] / attack
    rel_start = 1.0 - release
    tail = t > rel_start
    env[tail] = np.maximum(0.0, (1.0 - t[tail]) / release)
    return env.astype(np.float32, copy=False)


def _event_waveform(kind: str, *, intensity: float = 1.0, duration: float = 0.18) -> np.ndarray:
    kind = str(kind or "silence")
    intensity = float(np.clip(intensity, 0.0, 1.5))
    if kind == "silence" or intensity <= 1e-5:
        return np.zeros((max(1, int(AUDIO_SAMPLE_RATE * duration)),), dtype=np.float32)
    n = max(64, int(AUDIO_SAMPLE_RATE * float(duration)))
    t = np.arange(n, dtype=np.float32) / float(AUDIO_SAMPLE_RATE)
    env = _adsr(n)

    freq_map = {
        "step": (310.0, 0.0),
        "slide": (235.0, 175.0),
        "push": (185.0, 245.0),
        "score": (880.0, 1320.0),
        "bonus": (1040.0, 1560.0),
        "penalty": (190.0, 95.0),
        "wrong": (155.0, 82.0),
        "collision": (75.0, 145.0),
        "launch": (260.0, 740.0),
        "hit": (120.0, 260.0),
        "jump": (420.0, 680.0),
        "laser": (720.0, 1180.0),
        "bird": (520.0, 360.0),
        "fall_good": (760.0, 980.0),
        "fall_neutral": (340.0, 410.0),
        "fall_bad": (165.0, 105.0),
        "danger": (95.0, 58.0),
        "win": (523.25, 783.99),
    }
    f0, f1 = freq_map.get(kind, (330.0, 0.0))
    if kind in {"launch", "jump", "laser", "fall_good"}:
        freqs = np.linspace(f0, f1, n, dtype=np.float32)
        phase = 2.0 * np.pi * np.cumsum(freqs) / float(AUDIO_SAMPLE_RATE)
        wave = np.sin(phase)
    elif kind == "win":
        thirds = np.array_split(np.arange(n), 3)
        wave = np.zeros((n,), dtype=np.float32)
        notes = [523.25, 659.25, 783.99]
        for idx, seg in enumerate(thirds):
            if seg.size:
                wave[seg] = np.sin(2.0 * np.pi * notes[idx] * t[seg])
    elif kind in {"collision", "wrong", "fall_bad", "danger"}:
        noise = np.sin(2.0 * np.pi * f0 * t) + 0.35 * np.sign(np.sin(2.0 * np.pi * f1 * t))
        wave = noise.astype(np.float32)
    elif f1 > 0.0:
        wave = 0.65 * np.sin(2.0 * np.pi * f0 * t) + 0.35 * np.sin(2.0 * np.pi * f1 * t)
    else:
        wave = np.sin(2.0 * np.pi * f0 * t)
    base_gain = 0.72 if kind in {"score", "bonus", "win", "wrong", "penalty", "collision", "fall_bad", "danger"} else 0.58
    wave = (base_gain * intensity * env * wave).astype(np.float32)
    peak = float(np.max(np.abs(wave)))
    if peak > 1.0:
        wave = wave / peak
    return wave.astype(np.float32, copy=False)


def audio_features_from_waveform(waveform: np.ndarray, dim: int = AUDIO_FEATURE_DIM) -> np.ndarray:
    dim = max(1, int(dim))
    audio = np.nan_to_num(np.asarray(waveform, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if audio.size <= 1:
        return np.zeros((dim,), dtype=np.float32)
    feats = np.zeros((dim,), dtype=np.float32)
    for i, chunk in enumerate(np.array_split(audio, dim)):
        if chunk.size:
            feats[i] = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))
    mx = float(np.max(feats))
    if mx > 1e-8:
        feats = feats / mx
    return feats.astype(np.float32, copy=False)


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)


def _mel_filterbank(n_fft: int, sample_rate: int, n_mels: int) -> np.ndarray:
    low_mel = float(_hz_to_mel(np.asarray([80.0]))[0])
    high_mel = float(_hz_to_mel(np.asarray([sample_rate / 2.0]))[0])
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / float(sample_rate)).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = int(bins[m - 1]), int(bins[m]), int(bins[m + 1])
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for k in range(left, min(center, fb.shape[1])):
            fb[m - 1, k] = (k - left) / max(1, center - left)
        for k in range(center, min(right, fb.shape[1])):
            fb[m - 1, k] = (right - k) / max(1, right - center)
    return fb


def mel_spectrogram_from_waveform(
    waveform: np.ndarray,
    *,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    n_mels: int = MEL_BINS,
    frames: int = MEL_FRAMES,
) -> np.ndarray:
    audio = np.nan_to_num(np.asarray(waveform, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if audio.size <= 1 or float(np.max(np.abs(audio))) <= 1e-8:
        return np.zeros((int(n_mels), int(frames)), dtype=np.float32)
    n_fft = 512
    win = np.hanning(n_fft).astype(np.float32)
    if audio.size < n_fft:
        audio = np.pad(audio, (0, n_fft - audio.size))
    starts = np.linspace(0, max(0, audio.size - n_fft), max(1, int(frames))).round().astype(int)
    spec = np.zeros((n_fft // 2 + 1, starts.size), dtype=np.float32)
    for i, start in enumerate(starts):
        chunk = audio[start : start + n_fft]
        if chunk.size < n_fft:
            chunk = np.pad(chunk, (0, n_fft - chunk.size))
        fft = np.fft.rfft(chunk * win)
        spec[:, i] = np.square(np.abs(fft)).astype(np.float32)
    fb = _mel_filterbank(n_fft, int(sample_rate), int(n_mels))
    mel = fb @ spec
    mel = np.log1p(mel)
    mx = float(np.max(mel))
    if mx > 1e-8:
        mel = mel / mx
    return mel.astype(np.float32, copy=False)


def make_audio_event(kind: str, *, intensity: float = 1.0, duration: float = 0.18) -> AudioEvent:
    waveform = _event_waveform(kind, intensity=intensity, duration=duration)
    return AudioEvent(
        kind=str(kind or "silence"),
        sample_rate=AUDIO_SAMPLE_RATE,
        waveform=waveform,
        audio_features=audio_features_from_waveform(waveform),
        mel_spectrogram=mel_spectrogram_from_waveform(waveform),
    )
