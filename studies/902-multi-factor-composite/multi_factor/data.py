"""Data layer for Study 902 — Multi-Factor Composite.

Two sources, both offline once cached:

* **Real tape (yfinance, no key).** Daily **total-return** (auto-adjusted) closes for the
  five single-factor iShares ETFs that make up the sleeve — **VLUE** (value, 2013-04),
  **QUAL** (quality, 2013-07), **MTUM** (momentum, 2013-04), **USMV** (min-vol, 2011-10),
  **SIZE** (size, 2013-04) — plus the **SPY** benchmark and **BIL** (1-3-month T-bill ETF)
  as the tradable cash leg. Cached wide under the study's own ``_cache/`` as parquet.

* **Synthetic world.** A deterministic, seeded monthly generator (market, a set of sleeve
  members that each load on the market plus an idiosyncratic factor, a benchmark and cash)
  with a TUNABLE planted per-annum blend edge. ``edge=0`` is the null: the composite's
  Sharpe-advantage HAC *t* must NOT light up. Machinery proof only — never cited for a stamp.

Pure numpy + pandas + stdlib on the offline path; ``fetch`` (network) runs once to build the
cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
TAPE_CACHE = os.path.join(CACHE_DIR, "mfc_prices.parquet")

# The five single-factor iShares ETFs blended into the equal-weight sleeve.
SLEEVE = ["VLUE", "QUAL", "MTUM", "USMV", "SIZE"]
BENCH = "SPY"
CASH = "BIL"  # SPDR 1-3 month T-bill ETF — the tradable risk-free leg.

TICKERS = SLEEVE + [BENCH, CASH]

# Frozen as-of: monthly stats never include a partial month (June 2026 = last complete).
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2011-10-01", end: str | None = None,
          path: str = TAPE_CACHE, retries: int = 4) -> pd.DataFrame:
    """Download the tape once and cache it as parquet (network-only path).

    All columns are auto-adjusted closes (TOTAL-RETURN, dividends reinvested, net of each
    fund's expense ratio). BIL's total return is our cash/risk-free leg.
    """
    import yfinance as yf

    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True,
                              progress=False)["Close"]
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            time.sleep(2.0)
    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance returned no data for the sleeve tape.")
    raw = raw.dropna(how="all").sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_parquet(path)
    return raw


def have_real(path: str = TAPE_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = TAPE_CACHE) -> pd.DataFrame:
    """Cached wide frame of daily total-return closes (offline)."""
    return pd.read_parquet(path).sort_index()


# --------------------------------------------------------------------------- #
# Monthly building blocks
# --------------------------------------------------------------------------- #
def monthly_total_returns(prices: pd.DataFrame, asof: str = AS_OF) -> pd.DataFrame:
    """Month-end-to-month-end simple returns from a total-return price frame.

    Sliced to the frozen ``asof`` (the last complete calendar month) so a stamped run
    never contains a partial month and never drifts as new sessions arrive.
    """
    m = prices.resample("ME").last()
    ret = m.pct_change()
    return ret[ret.index <= pd.Timestamp(asof)]


# --------------------------------------------------------------------------- #
# Synthetic world (machinery proof only)
# --------------------------------------------------------------------------- #
def synthetic_world(n_months: int = 168, n_sleeve: int = 5, edge_ann: float = 0.0,
                    beta: float = 1.0, seed: int = 902) -> pd.DataFrame:
    """Deterministic monthly world: market, ``n_sleeve`` members, benchmark, cash.

    * market excess: 6%/yr mean, 15% vol. cash: 2%/yr, tiny vol.
    * each sleeve member excess = ``edge_ann``/12 + ``beta``·market + factor_i + idio,
      where ``factor_i`` is a member-specific zero-mean style factor (6%/yr vol) that
      diversifies away in the equal-weight blend, and idio is 4%/yr-vol noise.
    * benchmark excess = market (the SPY analogue).

    ``edge_ann=0`` is the null: the equal-weight blend carries the SAME risk-adjusted
    return as the benchmark, so the Sharpe-advantage HAC *t* must stay quiet. A positive
    ``edge_ann`` plants a genuine per-annum blend advantage the detector must recover.
    Index built via ``period_range`` kept small (far below the ns-Timestamp horizon).
    """
    rng = np.random.default_rng(seed)
    pidx = pd.period_range("2012-01", periods=n_months, freq="M")
    idx = pidx.to_timestamp(how="end").normalize()

    mkt = 0.06 / 12.0 + (0.15 / np.sqrt(12.0)) * rng.standard_normal(n_months)
    cash = np.full(n_months, 0.02 / 12.0) + (0.001 / np.sqrt(12.0)) * rng.standard_normal(n_months)

    cols = {"MKT_ex": mkt, "cash": cash, "SPY": cash + mkt}
    for i in range(n_sleeve):
        fac = (0.06 / np.sqrt(12.0)) * rng.standard_normal(n_months)
        idio = (0.04 / np.sqrt(12.0)) * rng.standard_normal(n_months)
        member_ex = edge_ann / 12.0 + beta * mkt + fac + idio
        cols[f"F{i+1}"] = cash + member_ex
    return pd.DataFrame(cols, index=idx)
