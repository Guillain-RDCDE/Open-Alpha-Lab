"""Data access for the Coiled-Spring study — per-name OHLCV frames, and a synthetic
universe with **planted springboards** so the whole machine runs offline before it ever
touches a real series.

Unlike the pairs study (which lives in the cross-section and needs a wide *panel*), this
rule is **single-name**: every signal is detected inside one ticker's own OHLCV history,
so the unit of work is the per-ticker frame. Volume rides along because the breakout rule
is volume-gated (2x the trailing month) and the capacity beat needs dollar volume.

Two ways in, the desk's usual split:

    * :func:`load_universe` — read the cached ``TICKER__split_only.parquet`` files (the
      same parquets Studies 04-05 wrote; we reuse Study 04's ~180-name liquid US cache
      rather than re-downloading). Cache-only by construction: a name with no parquet is
      simply skipped, never a silent network stall.
    * :func:`synthetic_universe` — fully offline. A handful of **true springboards** (names
      that genuinely coil on a rising EMA and then run) hidden among random-walk noise
      names, so the detector *should* find the planted breakouts and the backtest *should*
      harvest the planted runs — the signature the tests assert on, with no network.

**The adjustment-mode decision (house rule).** We run on the cached **split_only** series
(the only mode cached for this universe). For a long-only momentum breakout held days-to-
weeks, an ex-dividend gap drops the close on the ex-date — on these large, low-yield names
over short holds it is second-order, and if anything it nicks a *long*'s return slightly
(a payout leaves the price), so it works mildly *against* the rule rather than flattering
it. A total_return rerun is the obvious robustness PR. Stated as a decision, not buried.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# split_only is the desk's cached default for this universe (see module docstring).
DEFAULT_MODE = "split_only"

# Reuse Study 04's cache (~180 liquid US names with deep history) rather than re-fetching.
# A study is self-contained at the *results* level (docs/ is committed, fingerprinted);
# parquets are git-ignored and local either way.
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


def load_universe(
    tickers: list[str] | None = None,
    cache_dir: str = DEFAULT_CACHE,
    mode: str = DEFAULT_MODE,
    min_rows: int = 750,
    asof: str | None = None,
    clip_daily: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """Read cached OHLCV frames for ``tickers`` (default: all cached names), winsorized.

    Cache-only: a ticker with no parquet, or with fewer than ``min_rows`` sessions on or
    before ``asof``, is dropped — too short for a 20-EMA warm-up plus a meaningful sample.
    Each frame is passed through :func:`clean_frame` to kill impossible prints. Returns
    ``{ticker: frame}`` with Open/High/Low/Close (and Volume where present).
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


def clean_frame(frame: pd.DataFrame, clip_daily: float = 1.0) -> pd.DataFrame:
    """Winsorize a name's daily close returns to ``±clip_daily`` and rebuild a consistent bar.

    Real price data is filthy: a single bad print (a split not back-adjusted, an ADR
    mis-denominated) shows up as a physically-impossible daily "return". Fed into a
    breakout rule, such a spike fakes a colossal "breakout" that vanishes next day when the
    print corrects — manufacturing phantom signals and phantom P&L. We winsorize daily
    *close* returns at ±100% (same stated decision as Studies 04-05: it removes only the
    *impossible*; a genuine +100% close survives), rebuild the close from the clipped
    returns, then **scale Open/High/Low by the same per-bar correction factor** so the bar
    stays internally consistent (wicks keep their shape relative to the close). Volume is
    left untouched. On a liquid, low-yield universe this is a targeted scrub of a handful of
    glitches, not a reshaping of the data.
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
    # Per-bar scale factor (1.0 where the bar was untouched), reindexed to the full frame.
    scale = (rebuilt / valid).reindex(out.index).fillna(1.0)
    for c in OHLC_COLS:
        if c in out.columns:
            out[c] = out[c].astype(float) * scale
    return out


def dollar_volume(frame: pd.DataFrame) -> pd.Series:
    """Daily dollar volume (Close x Volume) for one name — for the capacity beat."""
    if "Volume" not in frame.columns:
        return pd.Series(index=frame.index, dtype=float)
    return (frame["Close"] * frame["Volume"]).rename("dollar_volume")


# --------------------------------------------------------------------------- #
# Synthetic universe — offline, with true springboards to recover
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TrueSpring:
    """Ground truth for the synthetic universe — a planted breakout the detector should find."""
    ticker: str
    breakout_index: int    # the bar where the pivot is broken (approximate)


def _ema(close: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(close)
    out[0] = close[0]
    for i in range(1, close.size):
        out[i] = alpha * close[i] + (1.0 - alpha) * out[i - 1]
    return out


def _bars_from_close(close: np.ndarray, rng: np.random.Generator, vol: np.ndarray) -> pd.DataFrame:
    prev = np.concatenate([[close[0]], close[:-1]])
    open_ = prev * (1 + 0.003 * rng.standard_normal(close.size))
    high = np.maximum(open_, close) * (1 + np.abs(0.006 * rng.standard_normal(close.size)))
    low = np.minimum(open_, close) * (1 - np.abs(0.006 * rng.standard_normal(close.size)))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol})


def _spring_cycle(rng: np.random.Generator, noise: float = 0.003):
    """One deterministic springboard cycle as a list of per-bar log-returns.

    Shape (the structure the rule bets on), bounded so it resets near where it started:
      decline below the EMA  ->  rally to a pivot high  ->  shallow pullback that *holds*
      above the (lagging) EMA  ->  a short coil  ->  the breakout bar  ->  a ~10-bar run  ->
      a give-back that returns price toward the cycle's start (so levels don't run away).
    Returns ``(steps, breakout_offset)`` where ``breakout_offset`` is the bar index of the
    breakout within the cycle (the planted ground-truth signal).
    """
    def seg(n, mu):
        return [mu + noise * rng.standard_normal() for _ in range(n)]
    steps = []
    steps += seg(14, -0.011)        # 0..13  decline ~ -15% (clearly below EMA)
    steps += seg(8, +0.013)         # 14..21 rally ~ +11% -> pivot high at the last bar
    steps += seg(4, -0.006)         # 22..25 pullback ~ -2.4% (stays above the lagging EMA)
    steps += seg(5, +0.002)         # 26..30 coil just under the pivot
    breakout_offset = len(steps)    # 31 the breakout bar
    steps += [0.045]                # one clean bar clearing the pivot high
    steps += seg(9, +0.020)         # 32..40 the run ~ +20%
    steps += seg(8, -0.012)         # 41..48 give-back, reset toward the start
    return steps, breakout_offset


def synthetic_universe(
    n_springs: int = 6,
    n_noise: int = 18,
    n_days: int = 1400,
    seed: int = 0,
) -> tuple[dict[str, pd.DataFrame], list[TrueSpring]]:
    """A toy market with ``n_springs`` **true springboards** hidden among ``n_noise`` walks.

    Each springboard name repeats :func:`_spring_cycle`: price sags below its EMA, rallies to
    a pivot high, pulls back *without losing the EMA*, coils, then breaks the pivot on a
    **volume surge** and runs — exactly the structure the rule bets on. The detector *should*
    fire on the breakouts and the backtest *should* harvest the runs. The noise names are
    pure random walks: tempting breakouts by chance, but with no planted follow-through.

    Returns ``(frames, true_springs)`` — per-name OHLCV frames (Volume included for the
    capacity beat) and the ground-truth breakout list. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n_days)
    frames: dict[str, pd.DataFrame] = {}
    truth: list[TrueSpring] = []

    for i in range(n_springs):
        name = f"SPRG{i:02d}"
        base_vol = int(rng.integers(2_000_000, 6_000_000))
        vol = rng.integers(int(0.6 * base_vol), int(1.4 * base_vol), n_days).astype(float)
        logp = np.empty(n_days)
        logp[0] = rng.uniform(2.5, 3.5)
        t = 1
        # a short calm lead-in so the first cycle has EMA warm-up behind it
        for _ in range(30):
            if t >= n_days:
                break
            logp[t] = logp[t - 1] + 0.0003 + 0.006 * rng.standard_normal()
            t += 1
        while t < n_days - 60:
            steps, b_off = _spring_cycle(rng)
            for k, s in enumerate(steps):
                if t >= n_days:
                    break
                logp[t] = logp[t - 1] + s
                if k == b_off:
                    vol[t] = base_vol * rng.uniform(3.0, 5.0)
                    truth.append(TrueSpring(name, t))
                t += 1
        # Fill any tail the cycle loop didn't reach with calm drift (np.empty leaves garbage
        # there otherwise -> exp() overflow). Keeps every bar finite and well-scaled.
        while t < n_days:
            logp[t] = logp[t - 1] + 0.0003 + 0.006 * rng.standard_normal()
            t += 1
        frames[name] = _bars_from_close(np.exp(logp[:n_days]), rng, vol[:n_days]).set_index(idx)

    for j in range(n_noise):
        d = 0.0002 * rng.standard_normal()
        logp = np.cumsum(d + 0.020 * rng.standard_normal(n_days)) + rng.uniform(2.5, 4.0)
        vol = rng.integers(1_000_000, 8_000_000, n_days).astype(float)
        frames[f"NOIS{j:02d}"] = _bars_from_close(np.exp(logp), rng, vol).set_index(idx)

    return frames, truth
