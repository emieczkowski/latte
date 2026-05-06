"""
Public test suite for signal_toolkit.py — copied into the repo and available
to agents via <run_tests />.

Each test class targets one function.  Failure messages show expected vs actual
values so agents can diagnose the bug without guessing.
"""
import numpy as np
import pytest
from signal_toolkit import (
    normalize, moving_average, rms, find_peaks,
    linear_detrend, autocorrelation, zero_crossings, downsample,
    band_energy, smooth_detrend, peak_rate,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def assert_close(got, expected, atol=1e-6, label=""):
    got = np.asarray(got, dtype=float)
    expected = np.asarray(expected, dtype=float)
    diff = np.max(np.abs(got - expected))
    assert diff <= atol, (
        f"{label}\n  expected: {expected}\n  got:      {got}\n  max diff: {diff:.6g}"
    )


# ── normalize ─────────────────────────────────────────────────────────────────

class TestNormalize:
    def test_basic_range(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        out = normalize(x)
        assert_close(out, [0.0, 0.25, 0.5, 0.75, 1.0], label="normalize basic range")

    def test_min_is_zero(self):
        x = np.array([2.0, 5.0, 8.0])
        out = normalize(x)
        assert out[0] == pytest.approx(0.0), f"min should map to 0.0, got {out[0]}"

    def test_max_is_one(self):
        x = np.array([2.0, 5.0, 8.0])
        out = normalize(x)
        assert out[-1] == pytest.approx(1.0), f"max should map to 1.0, got {out[-1]}"

    def test_constant_array(self):
        x = np.array([3.0, 3.0, 3.0])
        out = normalize(x)
        assert_close(out, [0.0, 0.0, 0.0], label="normalize constant array")


# ── moving_average ────────────────────────────────────────────────────────────

class TestMovingAverage:
    def test_output_length(self):
        x = np.arange(10, dtype=float)
        out = moving_average(x, window=3)
        assert len(out) == len(x), (
            f"moving_average must return same length as input. "
            f"Got {len(out)}, expected {len(x)}."
        )

    def test_flat_signal(self):
        x = np.ones(8, dtype=float) * 5.0
        out = moving_average(x, window=3)
        assert_close(out[2:-2], np.ones(4) * 5.0, label="moving_average on flat signal (interior)")

    def test_interior_values(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = moving_average(x, window=3)
        # Interior point at index 2: mean of [1,2,3] = 2.0; [2,3,4]=3.0; [3,4,5]=4.0
        assert_close(out[1:4], [2.0, 3.0, 4.0], label="moving_average interior points")


# ── rms ───────────────────────────────────────────────────────────────────────

class TestRms:
    def test_pythagorean(self):
        # rms([3, 4]) = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
        x = np.array([3.0, 4.0])
        got = rms(x)
        expected = np.sqrt(12.5)
        assert got == pytest.approx(expected, rel=1e-5), (
            f"rms([3,4]) expected {expected:.6f}, got {got:.6f}"
        )

    def test_unit_sine(self):
        t = np.linspace(0, 2 * np.pi, 10000)
        x = np.sin(t)
        got = rms(x)
        assert got == pytest.approx(1.0 / np.sqrt(2), rel=1e-3), (
            f"rms(sin) expected ~0.7071, got {got:.6f}"
        )

    def test_constant(self):
        x = np.full(50, 3.0)
        assert rms(x) == pytest.approx(3.0), f"rms of constant 3 should be 3, got {rms(x)}"


# ── find_peaks ────────────────────────────────────────────────────────────────

class TestFindPeaks:
    def test_simple_peak(self):
        x = np.array([0.0, 1.0, 0.0])
        peaks = find_peaks(x)
        assert list(peaks) == [1], f"Expected peak at index 1, got {peaks}"

    def test_no_plateau_peaks(self):
        # Plateau [1, 1] in the middle — no strict local max
        x = np.array([0.0, 1.0, 1.0, 0.0])
        peaks = find_peaks(x)
        assert len(peaks) == 0, (
            f"Plateau should produce no strict local maxima, got peaks at {peaks}"
        )

    def test_two_peaks(self):
        x = np.array([0.0, 2.0, 0.0, 3.0, 0.0])
        peaks = find_peaks(x)
        assert set(peaks) == {1, 3}, f"Expected peaks at {{1, 3}}, got {set(peaks)}"

    def test_min_height_filter(self):
        x = np.array([0.0, 0.5, 0.0, 2.0, 0.0])
        peaks = find_peaks(x, min_height=1.0)
        assert list(peaks) == [3], f"Only peak above min_height=1.0 should be index 3, got {peaks}"


# ── linear_detrend ────────────────────────────────────────────────────────────

class TestLinearDetrend:
    def test_pure_trend_removed(self):
        # A perfectly linear signal detrended should give zeros
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = linear_detrend(x)
        assert_close(out, np.zeros(5), atol=1e-10, label="linear_detrend of pure trend")

    def test_constant_unchanged(self):
        # A constant signal has zero slope; detrending should leave it (nearly) unchanged
        x = np.full(5, 7.0)
        out = linear_detrend(x)
        assert_close(out, np.zeros(5), atol=1e-10, label="linear_detrend of constant")

    def test_detrend_removes_slope(self):
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(100) * 0.1
        x = noise + np.linspace(0, 20.0, 100)   # strong linear trend dominates
        out = linear_detrend(x)
        t = np.arange(100, dtype=float)
        # After detrending, Pearson r with time index should be near zero
        r = float(np.corrcoef(t, out)[0, 1])
        assert abs(r) < 0.05, (
            f"Detrending should remove the linear slope (|r| < 0.05 with t), got r={r:.4f}"
        )


# ── autocorrelation ───────────────────────────────────────────────────────────

class TestAutocorrelation:
    def test_lag_zero_is_one(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        ac = autocorrelation(x, max_lag=10)
        assert ac[0] == pytest.approx(1.0, rel=1e-9), (
            f"autocorrelation at lag 0 must be 1.0, got {ac[0]:.6f}"
        )

    def test_output_length(self):
        x = np.arange(50, dtype=float)
        ac = autocorrelation(x, max_lag=5)
        assert len(ac) == 6, f"Expected length 6 (lags 0..5), got {len(ac)}"

    def test_white_noise_decays(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(2000)
        ac = autocorrelation(x, max_lag=5)
        for lag in range(1, 6):
            assert abs(ac[lag]) < 0.1, (
                f"White noise autocorrelation at lag {lag} should be near 0, got {ac[lag]:.4f}"
            )


# ── zero_crossings ────────────────────────────────────────────────────────────

class TestZeroCrossings:
    def test_alternating_sign(self):
        # [1, -1, 1, -1] has 3 sign changes
        x = np.array([1.0, -1.0, 1.0, -1.0])
        got = zero_crossings(x)
        assert got == 3, f"Expected 3 zero crossings, got {got}"

    def test_no_crossings(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert zero_crossings(x) == 0, f"All-positive signal should have 0 crossings, got {zero_crossings(x)}"

    def test_single_crossing(self):
        x = np.array([2.0, 1.0, -1.0, -2.0])
        assert zero_crossings(x) == 1, f"Expected 1 crossing, got {zero_crossings(x)}"

    def test_sine_wave(self):
        # sin over one full period crosses zero twice
        t = np.linspace(0, 2 * np.pi, 1001)
        x = np.sin(t)
        got = zero_crossings(x)
        # Expect exactly 2 (start-to-negative at π, negative-to-start at 2π)
        assert got == 2, f"sin over one period should have 2 zero crossings, got {got}"


# ── downsample ────────────────────────────────────────────────────────────────

class TestDownsample:
    def test_factor_two(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        out = downsample(x, factor=2)
        assert_close(out, [0.0, 2.0, 4.0], label="downsample by 2")

    def test_factor_three(self):
        x = np.arange(9, dtype=float)
        out = downsample(x, factor=3)
        assert_close(out, [0.0, 3.0, 6.0], label="downsample by 3")

    def test_first_sample_preserved(self):
        x = np.array([10.0, 20.0, 30.0, 40.0])
        out = downsample(x, factor=2)
        assert out[0] == pytest.approx(10.0), (
            f"First sample after downsample should be x[0]=10.0, got {out[0]}"
        )


# ── composite: band_energy ────────────────────────────────────────────────────

class TestBandEnergy:
    def test_normalized_sine(self):
        # Normalized sine lives in [0,1]. Its RMS should be 1/sqrt(2) ≈ 0.707
        # minus the DC offset from shifting [−1,1] to [0,1]: E[norm] = 0.5,
        # E[norm²] = E[(sin+1)²/4] = (1/4)(E[sin²] + 2E[sin] + 1) = (1/4)(0.5 + 0 + 1) = 3/8
        # rms = sqrt(3/8) ≈ 0.6124
        t = np.linspace(0, 2 * np.pi, 10000)
        x = np.sin(t)
        got = band_energy(x)
        assert got == pytest.approx(np.sqrt(3 / 8), rel=1e-2), (
            f"band_energy(sin) expected ~{np.sqrt(3/8):.4f}, got {got:.4f}"
        )

    def test_constant_signal(self):
        x = np.full(100, 5.0)
        # normalize([5]*100) = [0]*100, rms([0]*100) = 0
        assert band_energy(x) == pytest.approx(0.0), (
            f"band_energy of constant signal should be 0, got {band_energy(x)}"
        )


# ── composite: smooth_detrend ─────────────────────────────────────────────────

class TestSmoothDetrend:
    def test_output_length_matches_interior(self):
        # smooth_detrend: moving_average (same-length) → linear_detrend (same-length)
        # output must equal input length
        x = np.random.default_rng(0).standard_normal(20)
        out = smooth_detrend(x, window=3)
        assert len(out) == len(x), (
            f"smooth_detrend output length {len(out)} != input length {len(x)}"
        )

    def test_slope_removed(self):
        # After smooth-detrend of a strongly-trended signal, the linear
        # correlation with the time index should be near zero.
        rng = np.random.default_rng(1)
        noise = rng.standard_normal(80) * 0.1
        x = noise + np.linspace(0, 30.0, 80)
        out = smooth_detrend(x, window=5)
        t = np.arange(len(out), dtype=float)
        r = float(np.corrcoef(t, out)[0, 1])
        assert abs(r) < 0.1, (
            f"smooth_detrend should remove the linear trend (|r| < 0.1 with t), got r={r:.4f}"
        )


# ── composite: peak_rate ──────────────────────────────────────────────────────

class TestPeakRate:
    def test_known_peaks(self):
        # Two clear peaks in 10 samples at fs=1 → rate = 2/10 = 0.2
        x = np.array([0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        got = peak_rate(x, fs=1.0)
        assert got == pytest.approx(0.2, rel=0.01), (
            f"Expected peak_rate=0.2 (2 peaks / 10 s), got {got:.4f}"
        )

    def test_no_peaks(self):
        x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        assert peak_rate(x, fs=1.0) == pytest.approx(0.0), (
            f"Flat signal should have peak_rate=0, got {peak_rate(x, fs=1.0)}"
        )
