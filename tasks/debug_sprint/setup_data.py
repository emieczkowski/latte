"""
Writes the buggy signal_toolkit.py into repo_dir before the run starts.
Called by run_dynamic_pipeline.py via setup(repo_dir).
"""
from pathlib import Path

# ── Buggy library source ──────────────────────────────────────────────────────
_SIGNAL_TOOLKIT_SRC = '''\
"""
signal_toolkit.py — 1-D signal processing utilities.
"""
from __future__ import annotations
import numpy as np

def normalize(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo + 1)


def moving_average(x, window = 5):
    x = np.asarray(x, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x ** 2))


def find_peaks(x, min_height = 0.0):
    x = np.asarray(x, dtype=float)
    peaks = []
    for i in range(1, len(x) - 1):
        if x[i] >= x[i - 1] and x[i] >= x[i + 1] and x[i] > min_height:
            peaks.append(i)
    return np.array(peaks, dtype=int)


def linear_detrend(x):
    x = np.asarray(x, dtype=float)
    t = np.arange(len(x), dtype=float)
    slope, intercept = np.polyfit(t, x, 1)
    trend = slope * t + intercept
    return x + trend


def autocorrelation(x, max_lag = 20):
    x = np.asarray(x, dtype=float)
    n = len(x)
    xc = x - x.mean()
    denom = np.sum(xc ** 2)
    result = []
    for lag in range(max_lag + 1):
        num = np.sum(xc[: n - lag] * xc[lag:])
        result.append(num / n)
    return np.array(result)


def zero_crossings(x):
    x = np.asarray(x, dtype=float)
    return int(np.sum(np.diff(x) > 0))


def downsample(x, factor):
    x = np.asarray(x, dtype=float)
    return x[1::factor]

def band_energy(x):
    return rms(normalize(x))

def smooth_detrend(x, window = 5):
    smoothed = moving_average(x, window)
    return linear_detrend(smoothed)

def peak_rate(x, fs = 1.0):
    norm = normalize(x)
    peaks = find_peaks(norm, min_height=0.0)
    duration = len(x) / fs
    return len(peaks) / duration
'''


def setup(repo_dir):
    repo_dir = Path(repo_dir)
    (repo_dir / "signal_toolkit.py").write_text(_SIGNAL_TOOLKIT_SRC)
