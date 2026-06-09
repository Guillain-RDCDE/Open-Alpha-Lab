"""Worked extension (beat 7) — make σ *actually* adaptive, and see if that buys anything.

The headline study's deflation is that AdaptiveRSI's σ-zones are a **static** monotone relabel: for
a fixed length a σ landmark is one constant RSI level, so it moves zero trades versus that constant.
The framework gestures at Cardwell-style **regimes** but never implements any vol-dependence — its
zones are a function of length alone. This module builds the version the pitch implies but omits: a
**vol-adaptive** band whose oversold/overbought σ landmark *widens with realized volatility*, so the
RSI entry level becomes **time-varying** instead of a single constant.

That one change is the only modification that can possibly add signal, because a time-varying
threshold is *not* an order-preserving relabel — it can enter on a bar a fixed constant would skip,
and skip one a constant would take. So the falsifiable question for beat 7 is sharp:

    Does letting the threshold breathe with realized vol beat the **best fixed constant** — net of
    costs, and after paying for the extra knob (the vol-sensitivity ``gamma``)?

Two baked-in checks pin the machine before the real run decides it:

    * **Constant-vol tape** → the vol ratio sits at ~1, so the adaptive threshold barely moves
      (small dispersion) and collapses toward the fixed band: with no regime to read, adaptivity
      adds ~nothing. (:func:`synthetic_regime_prices` with equal vols.)
    * **Regime-vol tape** → the threshold genuinely varies in time (large dispersion): now it is
      *not* a relabel — it moves trades. Whether those trades are *better* is the empirical part,
      left to :mod:`examples.extension` on real SPY/QQQ.

House-standard *worked complement* in the study's own beat 7 — not a new study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import signals
from .decompose import _strategy_stats
from .rsi import sigma_to_rsi, wilder_rsi


# --------------------------------------------------------------------------- #
# Realized-vol regime measure
# --------------------------------------------------------------------------- #

def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling realized volatility — the standard deviation of log returns over ``window`` bars."""
    ret = np.log(close).diff()
    return ret.rolling(window, min_periods=max(5, window // 2)).std().rename("rv")


def vol_ratio(close: pd.Series, window: int = 20, ref: int = 252) -> pd.Series:
    """Realized vol divided by its own slow rolling median — a regime indicator centred on ~1.

    Above 1 = a higher-than-usual vol regime; below 1 = a calmer one. The slow ``ref`` median is the
    "what's normal lately" baseline, so the ratio is unit-free and comparable across assets.
    """
    rv = realized_vol(close, window)
    base = rv.rolling(ref, min_periods=ref // 4).median()
    return (rv / base).rename("vol_ratio")


# --------------------------------------------------------------------------- #
# The vol-adaptive band
# --------------------------------------------------------------------------- #

def vol_adaptive_lower_rsi(close: pd.Series, length: int, base_sigma: float = -1.0,
                           gamma: float = 0.5, window: int = 20) -> pd.Series:
    """A **time-varying** oversold RSI entry level that widens with realized volatility.

    The oversold σ landmark is scaled each bar by the vol regime::

        sigma_t   = base_sigma * (1 + gamma * (vol_ratio_t - 1))
        lower_t   = sigma_to_rsi(sigma_t, length)

    With ``base_sigma < 0`` and ``gamma > 0``, a high-vol bar pushes the landmark *more* negative
    (a deeper, harder-to-hit oversold) and a calm bar pulls it toward 50 — exactly the regime
    behaviour the framework describes but never builds. When ``gamma = 0`` (or vol is constant) this
    collapses to the static constant band ``sigma_to_rsi(base_sigma, length)``.
    """
    vr = vol_ratio(close, window).fillna(1.0)
    sigma_t = base_sigma * (1.0 + gamma * (vr - 1.0))
    return sigma_to_rsi(sigma_t, length).rename("lower_rsi_t")


def vol_adaptive_positions(close: pd.Series, length: int, base_sigma: float = -1.0,
                           gamma: float = 0.5, window: int = 20,
                           exit_level: float = 50.0) -> pd.Series:
    """Long/flat mean reversion against the **time-varying** oversold level.

    Enter long when ``RSI_t`` dips below the vol-adaptive ``lower_rsi_t``; exit above ``exit_level``
    (the 50 equilibrium). Same exit and one-bar shift as the static strategies, so the *only*
    difference from :func:`signals.rsi_band_positions` is that the entry threshold breathes with
    vol — which is what makes it more than a relabel.
    """
    rsi = wilder_rsi(close, length)
    lower_t = vol_adaptive_lower_rsi(close, length, base_sigma, gamma, window)
    r = rsi.to_numpy()
    lt = lower_t.to_numpy()
    pos = np.zeros(len(r))
    holding = False
    for i in range(len(r)):
        if np.isnan(r[i]) or np.isnan(lt[i]):
            holding = False
        elif not holding and r[i] < lt[i]:
            holding = True
        elif holding and r[i] > exit_level:
            holding = False
        pos[i] = 1.0 if holding else 0.0
    return pd.Series(pos, index=rsi.index, name="pos").shift(1).fillna(0.0)


def threshold_dispersion(close: pd.Series, length: int, base_sigma: float = -1.0,
                         gamma: float = 0.5, window: int = 20) -> float:
    """Std of the time-varying entry level (in RSI points) — 0 for a static band, > 0 once it moves.

    The one-number diagnostic that separates a genuine regime rule from a relabel: a static σ-zone
    has dispersion exactly 0; a vol-adaptive band has dispersion that grows with how much the vol
    regime actually swings.
    """
    return float(vol_adaptive_lower_rsi(close, length, base_sigma, gamma, window).std())


# --------------------------------------------------------------------------- #
# The horse race: vol-adaptive vs the best fixed constant
# --------------------------------------------------------------------------- #

def _best_fixed(close: pd.Series, rsi: pd.Series, grid, cost_bps: float) -> dict:
    best = None
    for lower in grid:
        pos = signals.rsi_band_positions(rsi, float(lower), 50.0)
        s = _strategy_stats(close, pos, cost_bps)
        if best is None or s["sharpe"] > best["sharpe"]:
            best = {**s, "lower": float(lower)}
    return best


def adaptive_vs_fixed(close: pd.Series, length: int = 2, base_sigma: float = -1.0,
                      gamma: float = 0.5, window: int = 20, cost_bps: float = 1.0,
                      grid=None) -> dict:
    """Does the vol-adaptive band beat the best fixed constant for one length?

    Runs both net of ``cost_bps``: the in-sample best fixed lower band over ``grid`` (the same
    generous, overfit control the headline horse race uses) and the vol-adaptive band. Returns both
    stat blocks, the Sharpe ``increment`` (adaptive − best-fixed), and the ``threshold_dispersion``
    — so a positive increment can be read against whether the threshold actually moved enough to be
    more than a relabel. The increment is **not** sign-constrained (unlike the static σ-band, which
    *was* one of the grid's constants): a time-varying rule genuinely can win or lose here.
    """
    if grid is None:
        grid = np.arange(5.0, 45.0, 1.0)
    rsi = wilder_rsi(close, length)
    fixed = _best_fixed(close, rsi, grid, cost_bps)
    adaptive = _strategy_stats(close, vol_adaptive_positions(close, length, base_sigma, gamma, window),
                               cost_bps)
    return {
        "length": int(length),
        "base_sigma": float(base_sigma),
        "gamma": float(gamma),
        "fixed_best": fixed,
        "adaptive": adaptive,
        "increment_sharpe": adaptive["sharpe"] - fixed["sharpe"],
        "threshold_dispersion": threshold_dispersion(close, length, base_sigma, gamma, window),
    }


# --------------------------------------------------------------------------- #
# Synthetic tape with a vol *regime* — for the baked-in checks and the offline demo
# --------------------------------------------------------------------------- #

def synthetic_regime_prices(n_bars: int = 2520, kappa: float = 0.06, sigma_lo: float = 0.007,
                            sigma_hi: float = 0.020, regime_len: int = 60, seed: int = 0):
    """An OU mean-reverting close whose shock vol switches between two regimes every ~``regime_len``.

    Same baked-in mean reversion as :func:`sigma_sleight.data.synthetic_prices`, but the per-bar
    shock vol alternates between ``sigma_lo`` and ``sigma_hi`` in blocks — so realized vol has a
    genuine regime structure for the vol-adaptive band to read. Set ``sigma_lo == sigma_hi`` for the
    constant-vol control (no regime, adaptivity should collapse). Deterministic given ``seed``;
    returns a price ``pd.Series`` on a business-day index.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2016-01-04", periods=n_bars, name="date")

    # A slow two-state vol path: blocks of length ~regime_len alternate lo/hi.
    block = (np.arange(n_bars) // regime_len) % 2
    sig = np.where(block == 0, sigma_lo, sigma_hi)

    x = np.empty(n_bars)
    x[0] = np.log(100.0)
    mu = np.log(100.0)
    for t in range(1, n_bars):
        mu += (sigma_lo * 0.15) * rng.standard_normal()
        x[t] = x[t - 1] + kappa * (mu - x[t - 1]) + sig[t] * rng.standard_normal()
    return pd.Series(np.exp(x), index=idx, name="close")
