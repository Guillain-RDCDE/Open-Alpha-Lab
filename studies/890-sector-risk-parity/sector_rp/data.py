"""Data layer for Study 890 — Sector Risk-Parity.

The claim under test: cap-weight concentrates the S&P 500 into a handful of mega-cap
sectors (today ~a third of SPY is Information Technology). If you instead **equal-RISK-
weight** the GICS sectors — inverse-volatility, or full equal-risk-contribution (ERC) —
you diversify *within* equities the way All-Weather does across asset classes, and the
question is whether that lifts the **excess-of-cash Sharpe** and cuts the drawdown versus
cap-weight SPY, net of quarterly rebalancing costs. Diversification, not forecasting.

Two sources, both offline-friendly once cached:

* **Real tape.** Daily auto-adjusted (total-return) closes for the eleven SPDR **Select
  Sector** ETFs — XLB (materials), XLE (energy), XLF (financials), XLI (industrials),
  XLK (technology), XLP (staples), XLU (utilities), XLV (health care), XLY (discretionary),
  XLRE (real estate) and XLC (communication services) — plus **SPY** (the cap-weight
  benchmark) and **BIL** (1-3m T-bill ETF, the cash / risk-free leg every excess-of-cash
  Sharpe is measured against). Everything from yfinance (public, no key), cached under this
  study's OWN ``_cache/`` as parquet, and every computation runs cache-first / OFFLINE.

  **Short history — named on the Signal axis.** Nine of the sectors date from Dec 1998, but
  **XLRE launched 2015-10** and **XLC 2018-06**, so the *joint eleven-sector* window only
  starts at XLC's inception (2018-06). We therefore report TWO real panels: an **11-sector**
  headline (2018-06 → as-of, ~8y — short, tech-bull-dominated) and a longer **9-sector**
  robustness panel (2007-06 → as-of, ~19y, bounded by BIL's 2007-05 inception) that reaches
  back through 2008. The short-history caveat travels with every eleven-sector number.

* **Synthetic world.** A deterministic, fixed-seed daily multi-asset generator with a
  TUNABLE ``vol_spread``: every asset carries the SAME per-period Sharpe but a DIFFERENT
  volatility (dispersion set by ``vol_spread``), lightly correlated through a common factor.
  ``vol_spread = 0`` makes all vols equal, so inverse-vol weighting collapses to equal
  weight and the risk-parity Sharpe advantage over equal-weight is **mechanically zero**
  (the null). ``vol_spread > 0`` plants genuine vol dispersion that inverse-vol risk-
  weighting can exploit to lift the portfolio Sharpe. This is the machinery proof for
  "is the diversification real?" — never cited for the real-tape stamp.

Pure numpy + pandas for the offline path; ``fetch()`` (network) runs once to build the
cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
PRICES_CACHE = os.path.join(CACHE_DIR, "sector_prices.parquet")

# Nine original Select-Sector SPDRs (Dec-1998) + the two young ones.
SECTORS_9 = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
SECTORS_11 = SECTORS_9 + ["XLRE", "XLC"]         # XLRE 2015-10, XLC 2018-06
BENCH = "SPY"                                     # cap-weight benchmark
CASH = "BIL"                                      # cash leg + risk-free proxy (2007-05)
TICKERS = SECTORS_11 + [BENCH, CASH]

AS_OF = "2026-06-30"        # last complete calendar month at publication
LOOKBACK = 63               # trailing trading days (~1 quarter) for the vol / cov estimate

__all__ = [
    "SECTORS_9", "SECTORS_11", "BENCH", "CASH", "TICKERS", "AS_OF", "LOOKBACK",
    "CACHE_DIR", "PRICES_CACHE",
    "fetch", "have_real", "load_prices", "daily_panel", "synthetic_world",
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "1998-12-01", path: str = PRICES_CACHE, retries: int = 4) -> None:
    """Download the 13 ETF closes once (auto_adjust=True → total-return) and cache them.

    Network-only; never called from the offline / notebook path. Retries up to
    ``retries`` times on a transient Yahoo failure.
    """
    import yfinance as yf

    px = None
    for _ in range(retries):
        try:
            raw = yf.download(TICKERS, start=start, auto_adjust=True, progress=False)
            px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
            if px is not None and len(px) > 0:
                break
        except Exception:
            time.sleep(2.0)
    if px is None or len(px) == 0:
        raise RuntimeError("yfinance returned no data for the sector panel")
    px = px[[t for t in TICKERS if t in px.columns]].copy()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    px = px.dropna(how="all")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    px.to_parquet(path)


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = PRICES_CACHE) -> pd.DataFrame:
    """Cached daily total-return closes for all 13 tickers — OFFLINE, no yfinance import."""
    px = pd.read_parquet(path)
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    return px.sort_index()


def daily_panel(
    sectors: list[str] | None = None,
    asof: str | None = AS_OF,
    path: str = PRICES_CACHE,
) -> dict[str, pd.DataFrame | pd.Series]:
    """The cache-first daily panel every real-tape number is computed from.

    ``sectors`` selects the sleeve list (default :data:`SECTORS_11`). Returns a dict:

    * ``sector_ret`` — daily simple total returns of the chosen sectors (joint window: the
      first date on which **all** chosen sectors AND ``SPY``/``BIL`` have a price, so the
      eleven-sector window starts at XLC's 2018-06 inception, the nine-sector at BIL's
      2007-05);
    * ``bench_ret``  — daily SPY total return over the same index (the cap-weight benchmark);
    * ``cash_ret``   — daily BIL total return over the same index (the risk-free / cash leg);
    * ``prices``     — the aligned price levels (for provenance / fingerprint).

    The sample is sliced to ``asof`` so it never creeps as new sessions arrive.
    """
    if sectors is None:
        sectors = SECTORS_11
    cols = sectors + [BENCH, CASH]
    px = load_prices(path)[cols]
    if asof is not None:
        px = px[px.index <= pd.Timestamp(asof)]
    px = px.dropna(how="any")                      # joint window: every leg present
    ret = px.pct_change().dropna(how="any")
    return {
        "sector_ret": ret[sectors],
        "bench_ret": ret[BENCH],
        "cash_ret": ret[CASH],
        "prices": px,
    }


# --------------------------------------------------------------------------- #
# Synthetic world — planted vol dispersion (the machinery control)
# --------------------------------------------------------------------------- #
def synthetic_world(
    n_days: int = 4500,
    vol_spread: float = 0.02,
    seed: int = 890,
    n_assets: int = 8,
    base_vol: float = 0.004,
    rho: float = 0.1,
    sharpe_d: float = 0.035,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Deterministic daily multi-asset world with a TUNABLE planted vol dispersion.

    Each asset ``k`` has the SAME per-period Sharpe ``sharpe_d`` but volatility
    ``base_vol + vol_spread * k`` (so ``vol_spread`` sets how dispersed the vols are), lightly
    correlated (``rho``) through a shared common factor. The **cap-weight benchmark** here
    weights each asset ∝ its variance (``vol²``) — a deliberate caricature of how cap-weight
    concentrates the index into the biggest, highest-vol names (today's mega-cap tech). Because
    every asset earns the *same* Sharpe, the ONLY thing that can move a book's Sharpe is how
    well it diversifies risk:

    * ``vol_spread = 0`` — all vols equal, so both inverse-vol weights AND the ∝vol² cap-weights
      collapse to equal weight; the risk-parity Sharpe advantage over cap-weight is
      **mechanically zero** (the null).
    * ``vol_spread > 0`` — genuine dispersion; the cap-weight benchmark piles risk onto the
      highest-vol asset (poor diversification) while inverse-vol equalises the risk budget, so
      the risk-parity book earns a positive excess-of-cash **Sharpe advantage** (the planted
      positive control).

    Returns the same dict shape as :func:`daily_panel` (``sector_ret`` = the risky assets,
    ``bench_ret`` = the concentrated ∝vol² cap-weight reference used only for the machinery
    check, ``cash_ret`` = a flat cash leg). Business-day index, span well below the pandas
    ns-Timestamp horizon (OOB-safe).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n_days, name="date")
    cols = [f"A{k}" for k in range(n_assets)]
    vols = base_vol + vol_spread * np.arange(n_assets)     # 0, +spread, +2·spread, ...
    common = rng.standard_normal(n_days)                   # a light common factor
    data = {}
    for k, c in enumerate(cols):
        idio = rng.standard_normal(n_days)
        shock = rho * common + np.sqrt(1.0 - rho ** 2) * idio
        data[c] = sharpe_d * vols[k] + vols[k] * shock
    sector_ret = pd.DataFrame(data, index=idx)
    capw = vols ** 2
    capw = capw / capw.sum()                               # ∝ variance → concentrated cap-weight
    bench_ret = (sector_ret * capw).sum(axis=1)
    cash_ret = pd.Series(0.02 / 252.0, index=idx, name=CASH)
    return {
        "sector_ret": sector_ret,
        "bench_ret": bench_ret,
        "cash_ret": cash_ret,
        "prices": (1.0 + sector_ret).cumprod() * 100.0,
    }
