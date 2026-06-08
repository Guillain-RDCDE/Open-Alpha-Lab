"""Data access for the pairs study — a *panel* of aligned close prices, and a
synthetic universe with **baked-in cointegrated twins** so the whole machine runs
offline before it ever touches a real series.

Two objects, kept straight:

    * the **frames** — per-ticker daily OHLC(V), exactly the parquet the rest of the
      desk caches (Studies 02–04 share the format). Volume rides along because the
      beat-6 capacity question needs dollar-volume.
    * the **panel** — a single wide ``DataFrame`` of close prices, dates × tickers,
      aligned on the union calendar. Pairs trading lives in the *cross-section*: every
      pair is a column-pair of this frame, so the formation/trading machinery never
      re-aligns anything.

Two ways in, the desk's usual split:

    * :func:`synthetic_universe` — fully offline. A handful of **true twins** (two
      names sharing a common random-walk factor plus a mean-reverting spread) hidden
      among independent noise names, so the selector *should* find the twins and the
      trader *should* harvest the reversion — the signature the tests assert on, with
      no network.
    * :func:`load_universe` + :func:`close_panel` — read the cached parquets (the same
      ``TICKER__split_only.parquet`` the other studies write) into frames and a panel.
      Cache-only by construction: a name with no parquet is simply skipped, never a
      silent network stall.

The adjustment-mode decision is the desk's standing house rule (see ``quantlab/data.py``).
The *cleanest* choice for pairs trading is **total_return** — a long held against a
short for weeks should compare relative *performance*, dividends included. We run on the
cached **split_only** series (the only mode the desk has cached for this universe), and
name the consequence rather than hide it: an ex-dividend gap drops one leg's price
permanently, so it reads as a **non-reverting** divergence — a small noise source that,
if anything, *works against* the strategy (a spread that gaps on a payout and never
closes is a losing or stopped trade, never a free one). On these large, low-yield names
over ~1-month holds it is second-order; a total_return rerun is the obvious robustness
PR. Stated as a decision, not buried — house rule.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# split_only is the desk's cached default for this universe. total_return is marginally
# cleaner for a multi-week long/short (see module docstring) but isn't cached here; the
# ex-dividend gap is a named, second-order noise source that works *against* the rule.
DEFAULT_MODE = "split_only"

# The 04 study already cached ~180 liquid US names with deep history; we reuse that
# cache rather than re-downloading. A study is self-contained at the *results* level
# (docs/ is committed, fingerprinted) — parquets are git-ignored and local either way.
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE = os.path.join(_HERE, "..", "04-social-oracle", "_cache")

OHLC_COLS = ["Open", "High", "Low", "Close"]


# --------------------------------------------------------------------------- #
# Real universe — read the cached parquets, align into a panel
# --------------------------------------------------------------------------- #

def _cache_path(ticker: str, cache_dir: str, mode: str) -> str:
    safe = ticker.replace("^", "_").replace("/", "_")
    return os.path.join(cache_dir, f"{safe}__{mode}.parquet")


def available_tickers(cache_dir: str = DEFAULT_CACHE, mode: str = "split_only") -> list[str]:
    """Every ticker with a cached parquet for ``mode`` (the universe we *can* price)."""
    if not os.path.isdir(cache_dir):
        return []
    suffix = f"__{mode}.parquet"
    return sorted(
        f[: -len(suffix)] for f in os.listdir(cache_dir) if f.endswith(suffix)
    )


def load_universe(
    tickers: list[str] | None = None,
    cache_dir: str = DEFAULT_CACHE,
    mode: str = "split_only",
    min_rows: int = 750,
    asof: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Read cached OHLC(V) frames for ``tickers`` (default: all cached names).

    Cache-only: a ticker with no parquet, or with fewer than ``min_rows`` sessions
    on or before ``asof``, is dropped — too short to span a 12-month formation plus a
    trading window. Returns ``{ticker: frame}`` with a ``Close`` column at minimum.
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
            out[tk] = df
    return out


def close_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide close-price panel (dates × tickers), union calendar, NaN before listing.

    No forward-fill: a NaN means "not trading yet / halted", and the formation step
    only ever pairs names with a fully-populated overlapping window, so a name's
    pre-IPO blanks can never leak into a spread.
    """
    cols = {t: f["Close"] for t, f in frames.items()}
    panel = pd.DataFrame(cols).sort_index()
    panel.index.name = "Date"
    return panel


def clean_panel(panel: pd.DataFrame, clip_daily: float = 1.0) -> pd.DataFrame:
    """Winsorize each name's daily returns to ``±clip_daily`` and rebuild prices from them.

    Real price data is filthy: a single bad print (a foreign ADR mis-denominated, a split
    not back-adjusted) shows up as a physically-impossible daily "return" — in this cache,
    **BMW reads +6,192,999** on one 2015 session, **ARK +2,166** in 2007. Fed into a pairs
    spread, such a spike fakes a colossal "divergence" that "reconverges" next day when the
    print corrects, manufacturing phantom profit (the well-known reason naive pairs
    backtests on raw data explode). We winsorize daily returns at ±100% — the same stated
    decision as Study 04: it removes only the *impossible* (a genuine +100% close is
    preserved) — then rebuild a self-consistent price panel so SSD, the spread, and the
    backtest all see the same cleaned series. A liquid, low-yield universe makes this a
    targeted scrub of a handful of glitches, not a reshaping of the data.
    """
    out = {}
    for col in panel.columns:
        s = panel[col]
        first = s.first_valid_index()
        if first is None:
            continue
        valid = s.loc[first:]
        r = valid.pct_change().clip(-clip_daily, clip_daily).fillna(0.0)
        rebuilt = float(valid.iloc[0]) * (1.0 + r).cumprod()
        out[col] = rebuilt.reindex(panel.index)
    cleaned = pd.DataFrame(out).sort_index()
    cleaned.index.name = "Date"
    return cleaned


def dollar_volume_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide panel of daily dollar volume (Close × Volume) — for the capacity beat."""
    cols = {}
    for t, f in frames.items():
        if "Volume" in f.columns:
            cols[t] = f["Close"] * f["Volume"]
    return pd.DataFrame(cols).sort_index()


# --------------------------------------------------------------------------- #
# Synthetic universe — offline, with true cointegrated twins to recover
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TruePair:
    """Ground truth for the synthetic universe — the twins the selector should find."""
    a: str
    b: str


def synthetic_universe(
    n_pairs: int = 8,
    n_noise: int = 24,
    n_days: int = 1500,
    spread_vol: float = 0.009,
    spread_phi: float = 0.96,
    factor_vol: float = 0.013,
    noise_vol: float = 0.020,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[TruePair]]:
    """A toy market with ``n_pairs`` **true twins** hidden among ``n_noise`` singles.

    Each twin pair shares a common random-walk factor ``f`` (the thing that makes them
    *look alike* — same sector, same beta), and their log-price difference is a
    **mean-reverting** OU-like process ``s`` (AR(1) coefficient ``spread_phi`` < 1):

        log A = f + s/2 ,  log B = f - s/2 ,  s_t = phi·s_{t-1} + eps

    So the spread always pulls back toward zero — exactly the structure pairs trading
    bets on. The noise names are independent random walks: tempting by chance, but with
    no reversion to harvest. The selector *should* rank the twins first; the trader
    *should* make money on them and roughly nothing on spurious noise pairs.

    Returns ``(panel, frames, true_pairs)`` — a close panel, per-name OHLCV frames
    (Volume included for capacity), and the ground-truth twin list. Deterministic.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n_days)
    frames: dict[str, pd.DataFrame] = {}
    closes: dict[str, np.ndarray] = {}
    true_pairs: list[TruePair] = []

    def _make_frame(name: str, logp: np.ndarray) -> None:
        close = np.exp(logp)
        prev = np.concatenate([[close[0]], close[:-1]])
        open_ = prev * (1 + 0.002 * rng.standard_normal(n_days))
        high = np.maximum(open_, close) * (1 + np.abs(0.004 * rng.standard_normal(n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(0.004 * rng.standard_normal(n_days)))
        vol = rng.integers(1_000_000, 8_000_000, n_days).astype(float)
        frames[name] = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
            index=idx,
        )
        closes[name] = close

    for i in range(n_pairs):
        drift = 0.0002 * rng.standard_normal()
        f = np.cumsum(drift + factor_vol * rng.standard_normal(n_days)) + rng.uniform(2.5, 4.0)
        s = np.empty(n_days)
        s[0] = spread_vol * rng.standard_normal()
        eps = spread_vol * rng.standard_normal(n_days)
        for t in range(1, n_days):
            s[t] = spread_phi * s[t - 1] + eps[t]
        a, b = f"TW{i:02d}A", f"TW{i:02d}B"
        _make_frame(a, f + s / 2.0)
        _make_frame(b, f - s / 2.0)
        true_pairs.append(TruePair(a, b))

    for j in range(n_noise):
        drift = 0.0002 * rng.standard_normal()
        logp = np.cumsum(drift + noise_vol * rng.standard_normal(n_days)) + rng.uniform(2.5, 4.0)
        _make_frame(f"NS{j:02d}", logp)

    return close_panel(frames), frames, true_pairs


def market_return(panel: pd.DataFrame) -> pd.Series:
    """Equal-weight daily return across the panel — a stand-in 'tape' for the beta beat.

    Pairs trading is meant to be dollar-neutral, so the interesting question is whether
    its returns have *any* market beta. With no external index handy offline, the
    equal-weight cross-section is the honest local benchmark.
    """
    rets = panel.pct_change()
    return rets.mean(axis=1).rename("r_mkt")
