"""Deterministic chunk descriptors and physical activity guards, no critic."""
from pathlib import Path
import wave
import numpy as np

CHUNK = 4096
RATE = 48000
NAMES = ['log_mid_rms', 'log_side_rms'] + [f'log_band_fraction_{i}' for i in range(8)] + [
    'band_flux', 'stereo_correlation', 'level_balance']


def read_wav(path):
    with wave.open(str(path), 'rb') as source:
        if source.getnchannels() != 2 or source.getframerate() != RATE or source.getsampwidth() != 2:
            raise ValueError('expected original 48-kHz stereo PCM16 analysis WAV')
        return np.frombuffer(source.readframes(source.getnframes()), dtype='<i2').reshape(-1, 2).astype(np.float64) / 32768


def write_wav(path, audio):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) >= 1:
        raise ValueError('refuse nonfinite/railed export')
    with wave.open(str(path), 'wb') as output:
        output.setparams((2, 2, RATE, 0, 'NONE', 'not compressed'))
        output.writeframes(np.round(audio * 32767).astype('<i2').tobytes())


def describe(audio):
    if not np.isfinite(audio).all() or len(audio) % CHUNK:
        raise ValueError('finite whole audio chunks required')
    chunks = audio.reshape(-1, CHUNK, 2)
    left, right = chunks[:, :, 0], chunks[:, :, 1]
    mid, side = (left + right) / 2, (left - right) / 2
    rms = lambda x: np.sqrt(np.mean(x * x, axis=1))
    mr, sr = rms(mid), rms(side)
    spectrum = np.abs(np.fft.rfft(mid * np.hanning(CHUNK), axis=1)) ** 2
    spectrum[:, :2] = 0
    frequency = np.fft.rfftfreq(CHUNK, 1 / RATE)
    limits = np.geomspace(20, 20000, 9)
    bands = np.column_stack([spectrum[:, (frequency >= lo) & (frequency < hi)].sum(1)
                             for lo, hi in zip(limits[:-1], limits[1:])])
    bands /= np.maximum(bands.sum(1, keepdims=True), 1e-18)
    flux = np.zeros(len(chunks))
    flux[1:] = np.sqrt(np.mean(np.diff(bands, axis=0) ** 2, axis=1))
    correlation = np.mean(left * right, axis=1) / np.maximum(rms(left) * rms(right), 1e-12)
    balance = (rms(left) - rms(right)) / np.maximum(rms(left) + rms(right), 1e-12)
    y = np.column_stack((np.log(np.maximum(mr, 1e-6)), np.log(np.maximum(sr, 1e-6)),
                         np.log(np.maximum(bands, 1e-9)), flux, correlation, balance))
    top = np.partition(spectrum, -3, axis=1)[:, -3:].sum(1) / np.maximum(spectrum.sum(1), 1e-18)
    report = {'frames': len(audio), 'chunks': len(chunks), 'seconds': len(audio) / RATE,
              'rms': float(np.sqrt(np.mean(audio * audio))), 'peak': float(np.abs(audio).max()),
              'quiet_chunk_fraction': float(np.mean(mr < 1e-4)),
              'top_three_fft_bins_median_power_fraction': float(np.median(top)),
              'clipping_sample_fraction': float(np.mean(np.abs(audio) >= 0.999)),
              'descriptor_names': NAMES,
              'eligible': bool(np.median(mr) >= 1e-4 and np.median(top) < 0.95 and
                               np.mean(np.abs(audio) >= 0.999) == 0)}
    return y, report


def phase_surrogate(audio, seed):
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(audio, axis=0)
    angles = rng.uniform(-np.pi, np.pi, len(spectrum))
    angles[0] = 0
    if len(audio) % 2 == 0:
        angles[-1] = 0
    return np.fft.irfft(spectrum * np.exp(1j * angles[:, None]), n=len(audio), axis=0)
