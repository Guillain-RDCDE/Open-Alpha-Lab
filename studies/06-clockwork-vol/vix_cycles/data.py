"""Data access for the cycles study — the **gauge** to clock, and a synthetic series
with a **baked-in fixed-period cycle** so the whole detector runs offline before it ever
touches a real series.

Two ways in, the desk's usual split:

    * :func:`synthetic_cycle` — fully offline. A known sinusoid (or a pair of them) at an
      *exact* period, buried in AR(1) **red noise** of a chosen strength. This is the
      ground truth the spectral machine *should* recover: the detector must light up at
      the injected period and the out-of-sample projector must time its turns. It is the
      analogue of Study 05's synthetic twins — the place where the method, if it is any
      good, *works*.
    * :func:`vix_series` / :func:`spx_series` — read the cached parquets (the ``^VIX`` and
      ``^GSPC`` closes Study 03 already cached). Cache-only by construction: no parquet,
      no network stall — just a clear error.

The signal we analyse is **log-VIX**. The VIX is a level, not a price (no splits, no
dividends — adjustment is meaningless, see Study 03), but it is *multiplicatively*
heteroskedastic: a 5-point move off 15 and off 45 are not the same event. Cycle theory is
a statement about *proportional* swings ("higher volatility into late June"), so logs are
the honest scale — they also make the additive sinusoid-plus-red-noise model the synthetic
uses a faithful null, not a strawman. Stated as a decision, not buried: house rule.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Study 03 already cached ^VIX (since 1990) and ^GSPC; we reuse those parquets rather than
# re-downloading. A study is self-contained at the *results* level (docs/ is committed and
# fingerprinted) — the parquets are git-ignored and local either way.
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE = os.path.join(_HERE, "..", "03-fear-gauge", "_cache")

VIX_FILE = "_VIX__raw.parquet"        # ^VIX, daily close, since 1990
SPX_FILE = "_GSPC__split_only.parquet"  # S&P 500 spot, for the stock-cycle expression


# --------------------------------------------------------------------------- #
# Real series — read the cached parquets
# --------------------------------------------------------------------------- #

def _load_close(filename: str, cache_dir: str, asof: str | None) -> pd.Series:
    path = os.path.join(cache_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cached series not found: {path}. This study reuses Study 03's _cache; run "
            f"Study 03's fetch (or copy the parquet) so the close is available offline."
        )
    df = pd.read_parquet(path).sort_index()
    if asof is not None:
        df = df[df.index <= pd.Timestamp(asof)]
    s = df["Close"].dropna()
    s.index.name = "Date"
    return s


def vix_series(cache_dir: str = DEFAULT_CACHE, asof: str | None = None) -> pd.Series:
    """The gauge: ``^VIX`` daily close as a clean Series, indexed by date."""
    return _load_close(VIX_FILE, cache_dir, asof).rename("vix")


def spx_series(cache_dir: str = DEFAULT_CACHE, asof: str | None = None) -> pd.Series:
    """The S&P 500 spot close — the tradeable expression of a 'stocks bottom on the cycle' call."""
    return _load_close(SPX_FILE, cache_dir, asof).rename("spx")


def log_signal(level: pd.Series) -> pd.Series:
    """Log of a level series — the honest scale for *proportional* cyclicity (see module doc)."""
    return np.log(level.astype(float)).rename(f"log_{level.name}")


# --------------------------------------------------------------------------- #
# Synthetic series — offline, with a known fixed cycle to recover
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class InjectedCycle:
    """Ground truth for the synthetic series — the cycle the detector should find."""
    period: float          # in sessions
    amplitude: float       # in log units (the signal lives in log-VIX space)
    phase: float           # radians at t=0


def synthetic_cycle(
    n_days: int = 4000,
    cycles: tuple[InjectedCycle, ...] = (
        InjectedCycle(period=80.0, amplitude=0.18, phase=0.0),
        InjectedCycle(period=40.0, amplitude=0.10, phase=1.1),
    ),
    rho: float = 0.94,
    noise_vol: float = 0.06,
    mean_level: float = np.log(18.0),
    seed: int = 0,
) -> tuple[pd.Series, tuple[InjectedCycle, ...]]:
    """A toy log-VIX with **exact** fixed cycles buried in AR(1) red noise.

    The series is ``y_t = mean + Σ a·cos(2π t / P + φ) + n_t`` where ``n_t`` is an AR(1)
    red-noise process with lag-1 autocorrelation ``rho`` (the kind of slow, wandering
    background a financial level always has — and exactly the null that fakes "cycles" to
    the naked eye). Because the cycles here are *real and stationary by construction*, the
    spectral detector **should** clear the red-noise envelope at 80 and 40 sessions, and
    the out-of-sample projector **should** time the turns. That is the whole point of the
    offline harness: it proves the machinery, so a *null* result on the real VIX is a
    statement about the market, not a bug in the code.

    Returns ``(series, injected)`` — a date-indexed log-level Series and the ground-truth
    cycle list. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_days, dtype=float)

    signal = np.zeros(n_days)
    for c in cycles:
        signal += c.amplitude * np.cos(2.0 * np.pi * t / c.period + c.phase)

    # AR(1) red noise with stationary variance noise_vol**2.
    eps = rng.standard_normal(n_days) * noise_vol * np.sqrt(1.0 - rho**2)
    noise = np.empty(n_days)
    noise[0] = rng.standard_normal() * noise_vol
    for i in range(1, n_days):
        noise[i] = rho * noise[i - 1] + eps[i]

    idx = pd.bdate_range("2005-01-03", periods=n_days)
    y = pd.Series(mean_level + signal + noise, index=idx, name="log_synth")
    y.index.name = "Date"
    return y, cycles


def red_noise_like(x: pd.Series, rho: float | None = None, seed: int = 0) -> pd.Series:
    """One AR(1) surrogate matching ``x``'s length, variance and (optionally fitted) ``rho``.

    The pure null: same red-noise character, **no** embedded cycle. Used in tests to confirm
    the detector does *not* manufacture a significant peak out of structureless noise.
    """
    from .spectral import ar1_rho  # local import to avoid a cycle at module load

    xv = np.asarray(x, float)
    xv = xv[np.isfinite(xv)]
    n = xv.size
    if rho is None:
        rho = ar1_rho(xv)
    sd = xv.std(ddof=1)
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n) * sd * np.sqrt(max(1.0 - rho**2, 1e-9))
    y = np.empty(n)
    y[0] = rng.standard_normal() * sd
    for i in range(1, n):
        y[i] = rho * y[i - 1] + eps[i]
    return pd.Series(y + xv.mean(), index=x.index[:n], name="red_noise")
