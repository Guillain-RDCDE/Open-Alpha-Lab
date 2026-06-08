"""The cycle theorist's machinery — bandpass extraction, instantaneous phase, turning
points, and forward projection of the next high/low.

Study 03 measured the VIX as a *trigger*; here we treat it the way a **cycles trader**
does: extract the component at a chosen period, read off where in the swing we are, and
**project** the next turn forward on the assumption the period is fixed. None of this
asks whether the cycle is real — that is :mod:`spectral`'s job. This module builds the
strongest possible version of the theorist's *forecast*, so that when :mod:`backtest`
scores it out-of-sample, we are scoring the real method at full strength, not a strawman.

Pure NumPy: the Hilbert transform (analytic signal → instantaneous phase) and the FFT
bandpass are both done by hand so the package imports with no SciPy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bandpass(x, period: float, width_frac: float = 0.5) -> pd.Series:
    """Isolate the component of ``x`` near ``period`` with a brick-wall FFT band filter.

    Keeps frequencies in ``(1/period)·(1 ± width_frac)`` and zeroes the rest, then inverts.
    A wide ``width_frac`` is deliberate: a *fixed* cycle would survive a narrow window, but a
    real-world quasi-cycle smears across nearby frequencies, so the theorist's own
    reconstruction needs the band. The mean is removed first and restored as zero — the
    output is the oscillation around the local level.
    """
    s = pd.Series(x, dtype=float).dropna()
    v = s.to_numpy()
    v = v - v.mean()
    n = v.size
    spec = np.fft.rfft(v)
    freqs = np.fft.rfftfreq(n)
    f0 = 1.0 / period
    lo, hi = f0 * (1 - width_frac), f0 * (1 + width_frac)
    mask = (freqs >= lo) & (freqs <= hi)
    filtered = np.fft.irfft(spec * mask, n=n)
    return pd.Series(filtered, index=s.index, name=f"band_{period:g}")


def analytic_signal(v: np.ndarray) -> np.ndarray:
    """Hilbert analytic signal of a real array, by hand via the FFT (no SciPy)."""
    n = v.size
    spec = np.fft.fft(v)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(spec * h)


def instantaneous_phase(x, period: float, width_frac: float = 0.5) -> pd.Series:
    """Instantaneous phase (radians) of the band-limited cycle — 'where in the swing are we'.

    Bandpass to ``period``, take the analytic signal, return its angle. Phase 0 ≈ a cycle
    *high* of the extracted component, ±π ≈ a *low*. This is the read a theorist uses to say
    "we're early in the up-leg".
    """
    band = bandpass(x, period, width_frac)
    z = analytic_signal(band.to_numpy())
    return pd.Series(np.angle(z), index=band.index, name="phase")


def turning_points(x, period: float, width_frac: float = 0.5) -> dict:
    """Indices of cycle **highs** and **lows** = local extrema of the band-limited component.

    A minimum separation of half a period suppresses jitter (two 'lows' a week apart aren't
    two cycle lows). Returns ``{highs: DatetimeIndex, lows: DatetimeIndex}`` — the dated turns
    the theorist would have drawn on the chart.
    """
    band = bandpass(x, period, width_frac)
    v = band.to_numpy()
    min_sep = max(int(period * 0.5), 1)

    def _extrema(sign: int) -> list[int]:
        sv = sign * v
        cand = [i for i in range(1, len(sv) - 1) if sv[i] >= sv[i - 1] and sv[i] >= sv[i + 1]]
        kept: list[int] = []
        for i in cand:
            if kept and i - kept[-1] < min_sep:
                if sv[i] > sv[kept[-1]]:
                    kept[-1] = i
            else:
                kept.append(i)
        return kept

    highs = _extrema(+1)
    lows = _extrema(-1)
    return {"highs": band.index[highs], "lows": band.index[lows]}


def fit_sinusoid(x, period: float) -> dict:
    """Least-squares amplitude & phase of a *fixed-period* cosine fitted to ``x``.

    Solves ``x_t ≈ c0 + A·cos(2π t / P) + B·sin(2π t / P)`` by OLS — the causal building
    block the backtest uses, since it depends only on data already seen. Returns
    ``{c0, A, B, amplitude, phase, period}`` with ``amplitude = √(A²+B²)`` and the value at
    step ``t`` recoverable as ``c0 + amplitude·cos(2π t / P − phase)``.
    """
    s = pd.Series(x, dtype=float).dropna()
    v = s.to_numpy()
    t = np.arange(v.size, dtype=float)
    w = 2.0 * np.pi / period
    design = np.column_stack([np.ones_like(t), np.cos(w * t), np.sin(w * t)])
    coef, *_ = np.linalg.lstsq(design, v, rcond=None)
    c0, a, b = coef
    return {
        "c0": float(c0),
        "A": float(a),
        "B": float(b),
        "amplitude": float(np.hypot(a, b)),
        "phase": float(np.arctan2(b, a)),
        "period": float(period),
        "n": int(v.size),
    }


def project(fit: dict, steps_ahead: np.ndarray | int) -> np.ndarray:
    """Project the fitted cosine forward ``steps_ahead`` steps past the fit window's end.

    ``steps_ahead`` is the offset from the *last* fitted sample (1 = next session). Returns
    the model's cycle value(s) — the forecast a fixed-period theorist commits to.
    """
    k = np.atleast_1d(np.asarray(steps_ahead, dtype=float))
    t = (fit["n"] - 1) + k
    w = 2.0 * np.pi / fit["period"]
    return fit["c0"] + fit["A"] * np.cos(w * t) + fit["B"] * np.sin(w * t)


def next_turn(fit: dict, kind: str = "low", max_ahead: int = 400) -> int:
    """Steps ahead to the next projected cycle ``low`` (or ``high``) — the dated call.

    Walks the projected cosine forward and returns the first step where it turns. This is
    the literal "I expect the VIX 40-day low into early July" forecast, reduced to a number
    of sessions. Returns ``-1`` if no turn is found within ``max_ahead``.
    """
    k = np.arange(1, max_ahead + 1)
    vals = project(fit, k)
    sign = -1.0 if kind == "low" else 1.0
    sv = sign * vals
    for i in range(1, len(sv) - 1):
        if sv[i] <= sv[i - 1] and sv[i] <= sv[i + 1]:
            return int(k[i])
    return -1
