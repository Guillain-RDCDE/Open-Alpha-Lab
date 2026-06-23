"""Data layer for Study 396 — Reshoring-Basket (US industrials + Mexico + automation vs benchmark).

Two sources, both offline-friendly:

* **Real tape.** Daily adjusted closes for the reshoring sleeve — ``XLI`` (US industrials),
  ``EWW`` (iShares MSCI Mexico) and an automation/robotics name (``ROK`` Rockwell Automation,
  long-listed) — plus two benchmarks, ``SPY`` (US market) and ``ACWI`` (global, from 2008)
  (yfinance, no key), cached under ``_cache/reshoring_prices.csv`` (a wide CSV indexed by
  date, one column per ticker). From these we build the **reshoring basket**: the equal-weight
  daily return of the three sleeves, and measure its return *in excess of* a benchmark.

* **Synthetic.** A deterministic, fixed-seed generator that produces a controllable basket
  return series as ``beta * benchmark + planted alpha + idiosyncratic noise`` — the planted
  daily alpha is the knob. It is the positive control: with the alpha knob at zero the engine
  must NOT manufacture a significant edge; with a large planted alpha it must light up.

Pure numpy + pandas + stdlib for the offline path. ``fetch_prices`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "reshoring_prices.csv")

# The reshoring / nearshoring basket: an equal-weight sleeve of the trade's three pillars.
# These are explicit, named, long-listed tickers — a transparent PROXY for "the reshoring
# trade", not a fabrication. XLI (US Industrial Select Sector, 1998-) carries US factory
# beta; EWW (iShares MSCI Mexico, 1996-) is the canonical nearshoring beneficiary; ROK
# (Rockwell Automation, long-listed) stands in for the factory-automation / robotics theme.
RESHORING_NAMES = ["XLI", "EWW", "ROK"]   # equal-weight reshoring basket
BENCH_US = "SPY"                          # US market benchmark (long history)
BENCH_GLOBAL = "ACWI"                     # global benchmark (iShares MSCI ACWI, from 2008)

# The reshoring narrative didn't really exist before the trade war; its loudest era starts
# with the 2018 US-China tariffs and the 2020 COVID supply-chain shock. We use this as a
# pre-announced, NOT-snooped sub-period split (the believers' "it's a post-2018 story").
NARRATIVE_START = "2018-01-01"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_prices(start: str = "1999-01-01", end: str | None = None,
                 path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download the reshoring sleeve + benchmarks via yfinance and cache a wide CSV.

    Network-only; used once to build ``_cache/reshoring_prices.csv``. Never imported by the
    offline notebook cells.
    """
    import yfinance as yf

    tickers = RESHORING_NAMES + [BENCH_US, BENCH_GLOBAL]
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached wide adjusted-close frame (index = date, columns = tickers)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def reshoring_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Build the daily-return frame the strategy consumes.

    Returns a frame indexed by date with columns:
      * ``mkt``     — SPY daily simple return (the US benchmark)
      * ``acwi``    — ACWI daily simple return (global benchmark; NaN before 2008)
      * ``basket``  — equal-weight daily return of the three reshoring sleeves
      * ``excess``  — ``basket - mkt`` (reshoring return in excess of the US benchmark)

    Days where fewer than two basket names have a return are dropped (early gaps). The frame
    is trimmed to where both the basket and SPY are available.
    """
    rets = prices.pct_change()
    names = [c for c in RESHORING_NAMES if c in rets.columns]
    basket = rets[names].mean(axis=1, skipna=True)
    enough = rets[names].notna().sum(axis=1) >= 2
    basket = basket.where(enough)
    mkt = rets[BENCH_US] if BENCH_US in rets.columns else pd.Series(index=rets.index, dtype=float)
    acwi = rets[BENCH_GLOBAL] if BENCH_GLOBAL in rets.columns else pd.Series(np.nan, index=rets.index)
    out = pd.DataFrame({"mkt": mkt, "acwi": acwi, "basket": basket,
                        "excess": basket - mkt})
    out = out.dropna(subset=["mkt", "basket"])
    return out


def load_real(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Convenience: cached prices -> reshoring daily-return frame in one call."""
    return reshoring_frame(load_prices(path))


def narrative_start() -> pd.Timestamp:
    """The pre-announced (NOT snooped) post-2018 reshoring-narrative split date."""
    return pd.Timestamp(NARRATIVE_START)


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_reshoring(n_days: int = 6_000, alpha_daily: float = 0.0,
                        seed: int = 396, mkt_mu: float = 0.0003,
                        mkt_sig: float = 0.011, beta: float = 1.05,
                        idio_sig: float = 0.009) -> pd.DataFrame:
    """Deterministic reshoring frame with a PLANTED daily alpha knob.

    Builds a market return series and a reshoring basket as
    ``beta * mkt + alpha_daily + idiosyncratic noise`` — so ``excess`` has expected value
    ``alpha_daily + (beta - 1) * mkt_mu`` by construction. ``alpha_daily`` is the planted
    edge the inference must recover.

    With ``alpha_daily = 0`` the basket's *alpha* (the beta-adjusted excess) is zero by
    construction, and the engine must NOT manufacture a significant alpha however the noise
    falls. With a large planted alpha (e.g. ``0.0004`` ≈ +10%/yr) the alpha t must light up.
    The raw ``excess`` is deliberately non-zero in expectation (the basket is high-beta), to
    demonstrate that *raw* excess ≠ alpha — the whole point of the beta adjustment.
    """
    rng = np.random.default_rng(seed)
    mkt = rng.normal(mkt_mu, mkt_sig, size=n_days)
    idio = rng.normal(0.0, idio_sig, size=n_days)
    basket = beta * mkt + alpha_daily + idio
    excess = basket - mkt
    # period_range keeps a decorative date label without OutOfBounds risk for long series
    idx = pd.period_range("1999-01-01", periods=n_days, freq="D").to_timestamp()
    # mirror the real frame's columns (acwi present, equal to mkt here as a stand-in)
    return pd.DataFrame({"mkt": mkt, "acwi": mkt, "basket": basket, "excess": excess},
                        index=idx)
