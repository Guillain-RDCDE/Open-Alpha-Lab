"""Data layer for Study 664 — ESG Premium.

Two ingredients:

* **Real tape.** Daily adjusted (total-return) closes for the two flagship "ESG" large-cap
  US equity ETFs and their plain-vanilla benchmarks, plus two style-factor ETFs used purely
  to *decompose* any ESG-vs-benchmark gap into a growth/value tilt and a quality tilt, and a
  3-month T-bill proxy for the risk-free rate used in every excess-of-cash Sharpe. All from
  yfinance (no key), cached as CSV under the study's own ``_cache/``.

    - ESGU  iShares ESG Aware MSCI USA ETF        (the "ESG-aware" large-cap fund)
    - SUSA  iShares MSCI USA ESG Select ETF        (the older, more selective ESG fund)
    - SPY   SPDR S&P 500 ETF Trust                 (plain-vanilla benchmark for ESGU)
    - IVV   iShares Core S&P 500 ETF               (plain-vanilla benchmark for SUSA)
    - IVW   iShares S&P 500 Growth ETF             (growth leg of the growth-value spread)
    - IVE   iShares S&P 500 Value ETF              (value leg of the growth-value spread)
    - QUAL  iShares MSCI USA Quality Factor ETF    (the quality-tilt factor)
    - ^IRX  13-week T-bill discount yield          (risk-free proxy for excess-of-cash Sharpe)

* **Fund facts, hardcoded.** Inception dates and prospectus expense ratios for every ticker
  above — plain facts, no network, no fitting. Used only to caption the tracking-difference
  tables (the expense-ratio gap is a *documented*, not fitted, cost).

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build the
cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

TICKERS = ["ESGU", "SUSA", "SPY", "IVV", "IVW", "IVE", "QUAL", "^IRX"]

AS_OF = "2026-06-30"          # last complete calendar month at publication (2026-07-10)
FETCH_START = "2004-01-01"    # well before SUSA's 2005 inception
FETCH_END = "2026-07-01"

# --------------------------------------------------------------------------- #
# Fund facts — hardcoded, no network. Source: fund prospectuses / issuer fact sheets
# (ishares.com, ssga.com), figures stable over the study period.
# --------------------------------------------------------------------------- #
FUND_FACTS = {
    "ESGU": {"name": "iShares ESG Aware MSCI USA ETF", "inception": "2016-12-01",
              "expense_ratio": 0.0015, "role": "ESG fund (broad, screens + tilts)"},
    "SUSA": {"name": "iShares MSCI USA ESG Select ETF", "inception": "2005-01-24",
              "expense_ratio": 0.0025, "role": "ESG fund (older, more selective)"},
    "SPY":  {"name": "SPDR S&P 500 ETF Trust", "inception": "1993-01-22",
              "expense_ratio": 0.000945, "role": "plain-vanilla benchmark (for ESGU)"},
    "IVV":  {"name": "iShares Core S&P 500 ETF", "inception": "2000-05-15",
              "expense_ratio": 0.0003, "role": "plain-vanilla benchmark (for SUSA)"},
    "IVW":  {"name": "iShares S&P 500 Growth ETF", "inception": "2000-05-22",
              "expense_ratio": 0.0018, "role": "growth leg of the growth-value spread"},
    "IVE":  {"name": "iShares S&P 500 Value ETF", "inception": "2000-05-22",
              "expense_ratio": 0.0018, "role": "value leg of the growth-value spread"},
    "QUAL": {"name": "iShares MSCI USA Quality Factor ETF", "inception": "2013-07-16",
              "expense_ratio": 0.0015, "role": "quality-tilt factor"},
}

# The two ESG-fund-vs-benchmark pairs under test.
PAIRS = {"ESGU": "SPY", "SUSA": "IVV"}

CACHE_FILES = {t: os.path.join(CACHE_DIR, f"esgprem_{t.strip('^').lower()}.csv")
               for t in TICKERS}


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = FETCH_START, end: str = FETCH_END) -> None:
    """Download adjusted (total-return) closes for every ticker; cache each as its own CSV.

    ``auto_adjust=True`` so ETF prices already include reinvested dividends (fair total-return
    comparison for the ESG-vs-benchmark race). ^IRX is a discount-yield index, not a price —
    cached as-is (percent points).
    """
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in TICKERS:
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df[["Close"]].dropna().to_csv(CACHE_FILES[t])


def have_real() -> bool:
    return all(os.path.exists(p) for p in CACHE_FILES.values())


def load_real(asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    """Cached {ticker: frame} dict, each sliced to [inception, asof]."""
    out = {}
    for t in TICKERS:
        df = pd.read_csv(CACHE_FILES[t], index_col=0, parse_dates=True).sort_index()
        out[t] = df.loc[df.index <= asof].copy()
    return out


# --------------------------------------------------------------------------- #
# Synthetic world — a paired (fund, benchmark) return world with a TUNABLE planted premium
# --------------------------------------------------------------------------- #
def synthetic_world(premium_bps: float = 0.0, seed: int = 664,
                     n_days: int = 2500, corr: float = 0.97,
                     mu_bench: float = 0.00035, sig_bench: float = 0.0095,
                     ) -> pd.DataFrame:
    """Deterministic correlated daily-return world for a "fund" and its "benchmark".

    The benchmark is i.i.d. Normal(mu_bench, sig_bench); the fund's daily return is a
    correlated (``corr``) copy plus a TUNABLE constant ``premium_bps`` (in basis points/day)
    on top — the planted "ESG premium". ``premium_bps = 0`` is the null world: the fund and
    benchmark are statistically identical in expectation, and the active-return NW t-test
    must NOT reach significance. Business-day index, ~2,500 sessions (~10 years) — far below
    the 250-year pandas ns-timestamp trap.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=n_days)
    z1 = rng.normal(0.0, 1.0, n_days)
    z2 = rng.normal(0.0, 1.0, n_days)
    bench_shock = z1
    fund_shock = corr * z1 + np.sqrt(1 - corr ** 2) * z2
    bench_ret = mu_bench + sig_bench * bench_shock
    fund_ret = mu_bench + premium_bps / 1e4 + sig_bench * fund_shock
    return pd.DataFrame({"fund_ret": fund_ret, "bench_ret": bench_ret}, index=idx)
