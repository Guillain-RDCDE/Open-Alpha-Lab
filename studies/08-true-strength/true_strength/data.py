"""Data access for the True-Strength study — per-name OHLCV frames, and an offline
synthetic universe with **known momentum structure** so the whole machine runs before it
ever touches a real series.

Like the Coiled-Spring study, this rule is **single-name**: every oscillator is computed
inside one ticker's own price history. We need Close (the oscillators), Open (an honest
next-bar fill), and Volume (the capacity beat). We reuse Study 04's ~180-name liquid US
cache rather than re-downloading.

Two ways in, the desk's usual split:

    * :func:`load_universe` — read the cached ``TICKER__split_only.parquet`` files. Cache-
      only by construction: a name with no parquet is skipped, never a silent network stall.
    * :func:`synthetic_universe` — fully offline. A handful of **trending / cyclical** names
      (where any honest momentum oscillator *must* light up) among random-walk noise names.
      On the trending names the three oscillators should move together — which is exactly
      what the collinearity machinery has to recover, with no network.

**The adjustment-mode decision (house rule).** We run on the cached **split_only** series
(the only mode cached for this universe). The oscillators are functions of *daily changes*;
an ex-dividend gap injects one small spurious down-move on the ex-date. On these large,
low-yield names it is second-order and affects all three oscillators identically (so it
cannot bias a *comparison* between them). A total_return rerun is the obvious robustness PR.
Stated as a decision, not buried.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_MODE = "split_only"

# Reuse Study 04's cache (same parquets Studies 04/05/07 read) rather than re-fetching.
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE = os.path.join(_HERE, "..", "04-social-oracle", "_cache")

OHLC_COLS = ["Open", "High", "Low", "Close"]


# --------------------------------------------------------------------------- #
# Real universe — read the cached parquets
# --------------------------------------------------------------------------- #

def _cache_path(ticker: str, cache_dir: str, mode: str) -> str:
    safe = ticker.replace("^", "_").replace("/", "_")
    return os.path.join(cache_dir, f"{safe}__{mode}.parquet")


def available_tickers(cache_dir: str = DEFAULT_CACHE, mode: str = DEFAULT_MODE) -> list[str]:
    """Every ticker with a cached parquet for ``mode`` (the universe we *can* price)."""
    if not os.path.isdir(cache_dir):
        return []
    suffix = f"__{mode}.parquet"
    return sorted(f[: -len(suffix)] for f in os.listdir(cache_dir) if f.endswith(suffix))


def clean_frame(frame: pd.DataFrame, clip_daily: float = 1.0) -> pd.DataFrame:
    """Winsorize daily close returns to ``±clip_daily`` and rebuild a consistent bar.

    A single bad print (an un-adjusted split, a mis-denominated ADR) is a physically
    impossible daily "return" that would inject a fake momentum spike into every oscillator
    at once. We clip daily *close* returns at ±100% (same decision as Studies 04/05/07),
    rebuild the close from the clipped returns, and scale Open/High/Low by the same per-bar
    factor so each bar stays internally consistent. Volume is left untouched.
    """
    out = frame.copy()
    close = out["Close"].astype(float)
    first = close.first_valid_index()
    if first is None:
        return out
    valid = close.loc[first:]
    r = valid.pct_change()
    r_clip = r.clip(-clip_daily, clip_daily).fillna(0.0)
    rebuilt = float(valid.iloc[0]) * (1.0 + r_clip).cumprod()
    scale = (rebuilt / valid).reindex(out.index).fillna(1.0)
    for c in OHLC_COLS:
        if c in out.columns:
            out[c] = out[c].astype(float) * scale
    return out


def load_universe(
    tickers: list[str] | None = None,
    cache_dir: str = DEFAULT_CACHE,
    mode: str = DEFAULT_MODE,
    min_rows: int = 750,
    asof: str | None = None,
    clip_daily: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """Read cached OHLCV frames for ``tickers`` (default: all cached names), winsorized.

    Cache-only: a ticker with no parquet, or fewer than ``min_rows`` sessions on or before
    ``asof``, is dropped — too short for the slow EMAs to warm up plus a meaningful sample.
    Returns ``{ticker: frame}`` with Open/High/Low/Close (and Volume where present).
    """
    if tickers is None:
        tickers = available_tickers(cache_dir, mode)
    out: dict[str, pd.DataFrame] = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir, mode)
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path).sort_index()
        if asof is not None:
            df = df[df.index <= pd.Timestamp(asof)]
        if len(df) >= min_rows and "Close" in df.columns:
            out[tk] = clean_frame(df, clip_daily=clip_daily)
    return out


def close_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One wide Close panel (columns = tickers) — the object the fingerprint is taken on."""
    return pd.DataFrame({t: f["Close"] for t, f in frames.items()}).sort_index()


def dollar_volume(frame: pd.DataFrame) -> pd.Series:
    """Daily dollar volume (Close × Volume) for one name — for the capacity beat."""
    if "Volume" not in frame.columns:
        return pd.Series(index=frame.index, dtype=float)
    return (frame["Close"] * frame["Volume"]).rename("dollar_volume")


# --------------------------------------------------------------------------- #
# Synthetic universe — offline, with known momentum structure
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TrendName:
    """Ground truth: a synthetic name built to *trend* (oscillators must agree on it)."""
    ticker: str
    kind: str            # "trend" or "cycle"


def _bars_from_close(close: np.ndarray, rng: np.random.Generator, vol: np.ndarray) -> pd.DataFrame:
    prev = np.concatenate([[close[0]], close[:-1]])
    open_ = prev * (1 + 0.003 * rng.standard_normal(close.size))
    high = np.maximum(open_, close) * (1 + np.abs(0.006 * rng.standard_normal(close.size)))
    low = np.minimum(open_, close) * (1 - np.abs(0.006 * rng.standard_normal(close.size)))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol})


def synthetic_universe(
    n_trend: int = 4,
    n_cycle: int = 4,
    n_noise: int = 12,
    n_days: int = 1400,
    seed: int = 0,
) -> tuple[dict[str, pd.DataFrame], list[TrendName]]:
    """A toy market with momentum structure the oscillators *must* recover, plus noise.

    Three kinds of name:
      * ``TRND`` — persistent drift with regime flips every ~250 sessions. Any honest
        momentum oscillator must be positive through an up-leg and negative through a down-
        leg, so the three must be highly **collinear** here (the machinery test).
      * ``CYCL`` — a slow sine-wave drift (a "cycle"), same story at a different frequency.
      * ``NOIS`` — pure random walks: oscillators wander, with no shared structure to find.

    Returns ``(frames, truth)`` — per-name OHLCV frames (Volume included) and the list of
    structured names. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n_days)
    frames: dict[str, pd.DataFrame] = {}
    truth: list[TrendName] = []

    for i in range(n_trend):
        name = f"TRND{i:02d}"
        # Regime-switching drift: flip sign every ~250 bars, small daily noise on top.
        seg = 200 + int(rng.integers(0, 120))
        drift = np.empty(n_days)
        sign, t = 1.0, 0
        while t < n_days:
            length = min(seg, n_days - t)
            drift[t : t + length] = sign * 0.0010
            sign *= -1.0
            t += length
        logr = drift + 0.011 * rng.standard_normal(n_days)
        logp = np.cumsum(logr) + rng.uniform(2.5, 3.5)
        vol = rng.integers(2_000_000, 6_000_000, n_days).astype(float)
        frames[name] = _bars_from_close(np.exp(logp), rng, vol).set_index(idx)
        truth.append(TrendName(name, "trend"))

    for i in range(n_cycle):
        name = f"CYCL{i:02d}"
        period = rng.uniform(180, 320)
        phase = rng.uniform(0, 2 * np.pi)
        t = np.arange(n_days)
        cyc = 0.0014 * np.sin(2 * np.pi * t / period + phase)
        logr = cyc + 0.010 * rng.standard_normal(n_days)
        logp = np.cumsum(logr) + rng.uniform(2.5, 3.5)
        vol = rng.integers(2_000_000, 6_000_000, n_days).astype(float)
        frames[name] = _bars_from_close(np.exp(logp), rng, vol).set_index(idx)
        truth.append(TrendName(name, "cycle"))

    for j in range(n_noise):
        logp = np.cumsum(0.0002 * rng.standard_normal(n_days) + 0.018 * rng.standard_normal(n_days)) + rng.uniform(2.5, 4.0)
        vol = rng.integers(1_000_000, 8_000_000, n_days).astype(float)
        frames[f"NOIS{j:02d}"] = _bars_from_close(np.exp(logp), rng, vol).set_index(idx)

    return frames, truth
