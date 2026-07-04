"""Data layer for Study 601 — Factor ETFs, Live Test.

Two sources, both offline-friendly once cached:

* **Real tape (yfinance, no key).** Daily **total-return** (auto-adjusted) closes for the
  four iShares factor ETFs under audit (USMV 2011-10, MTUM/VLUE 2013-04, QUAL 2013-07),
  the SPY benchmark, the style-proxy pairs (IWD/IWF = Russell 1000 Value/Growth for the
  value spread, SPHQ = the independent-provider S&P 500 Quality product for the quality
  spread), the nine original SPDR sector ETFs (the raw material for a 12-1 sector-momentum
  winners-minus-losers proxy) and ^IRX (13-week T-bill discount yield, the risk-free leg).
  Cached wide under the study's own ``_cache/``.

* **Synthetic world.** A deterministic, seeded joint monthly generator (market excess,
  style-spread factor, fund excess) with TUNABLE planted parameters — beta, style loading
  and annual alpha. ``beta=1, loading=0, alpha=0`` is the null: the CAPM-alpha NW *t* and
  the style-loading NW *t* must NOT light up on it. It is the machinery proof only — never
  cited in support of a stamp.

Pure numpy + pandas + stdlib on the offline path; ``fetch_tape`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
TAPE_CACHE = os.path.join(CACHE_DIR, "fel_prices.csv")

# The four products under audit (iShares factor ETFs, all 0.15%/yr expense ratio) + bench.
FUNDS = ["USMV", "MTUM", "VLUE", "QUAL"]
BENCH = "SPY"

# Style-proxy raw material (all long-listed, all pre-date the funds):
#   IWD - IWF  : Russell 1000 Value minus Growth (the value spread), live since 2000-05.
#   SPHQ       : Invesco S&P 500 Quality — an INDEPENDENT-provider quality product
#                (its quality spread vs SPY is the quality proxy; note SPHQ switched to the
#                S&P 500 Quality Index in 2016 — flagged in results).
#   9 SPDRs    : the original 1998 sector suite, raw material for the 12-1 sector-momentum
#                winners-minus-losers spread.
STYLE = ["IWD", "IWF", "SPHQ"]
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
RF_TICKER = "^IRX"

TICKERS = [BENCH] + FUNDS + STYLE + SECTORS + [RF_TICKER]

# Frozen as-of: monthly stats never include a partial month (June 2026 = last complete).
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_tape(start: str = "1998-12-01", end: str | None = None,
               path: str = TAPE_CACHE, retries: int = 3) -> pd.DataFrame:
    """Download the full tape once and cache it (network-only path).

    Funds/benchmark/style/sector columns are auto-adjusted closes (TOTAL-RETURN, i.e. net
    of each fund's expense ratio, dividends reinvested); ^IRX is a 13-week T-bill
    discount-yield LEVEL in percent (not a price).
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
    raw = raw.dropna(how="all").sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = TAPE_CACHE) -> bool:
    return os.path.exists(path)


def load_tape(path: str = TAPE_CACHE) -> pd.DataFrame:
    """Cached wide frame: total-return closes + ^IRX yield level (%)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


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


def monthly_rf(irx: pd.Series, asof: str = AS_OF) -> pd.Series:
    """Monthly risk-free return from the ^IRX 13-week discount-yield level.

    rf for month t = the yield level observed at the END of month t-1 (annual %, /100/12)
    — known in advance, no look-ahead.
    """
    y = irx.dropna().resample("ME").last() / 100.0
    rf = (y / 12.0).shift(1)
    return rf[rf.index <= pd.Timestamp(asof)]


def sector_momentum_spread(sector_monthly: pd.DataFrame, top_n: int = 3,
                           skip: int = 1, window: int = 11) -> pd.Series:
    """12-1 sector-momentum winners-minus-losers monthly spread (the momentum proxy).

    Formation for month t = cumulative return over months t-12 … t-2 (``window`` = 11
    months, skipping the most recent ``skip`` = 1 — the classic 12-1). The ranking is
    therefore fully known at the END of month t-1 and applied to month t: exactly ONE
    month of execution lag, no look-ahead. Long the top ``top_n`` sectors equal-weight,
    short the bottom ``top_n`` — the month-t spread return is the factor realization used
    as a contemporaneous regressor / month-sign splitter for MTUM.
    """
    form = (1.0 + sector_monthly).rolling(window).apply(np.prod, raw=True).shift(skip + 1)
    out = {}
    for t in sector_monthly.index:
        f = form.loc[t].dropna()
        if len(f) < 2 * top_n:
            continue
        f = f.sort_values()
        losers, winners = f.index[:top_n], f.index[-top_n:]
        r = sector_monthly.loc[t]
        out[t] = float(r[winners].mean() - r[losers].mean())
    return pd.Series(out, name="wml").sort_index()


def value_spread(monthly: pd.DataFrame) -> pd.Series:
    """Russell 1000 Value minus Growth monthly return spread (the value proxy)."""
    return (monthly["IWD"] - monthly["IWF"]).rename("vmg").dropna()


def quality_spread(monthly: pd.DataFrame) -> pd.Series:
    """SPHQ minus SPY monthly return spread — an independent-provider quality proxy."""
    return (monthly["SPHQ"] - monthly["SPY"]).rename("qmb").dropna()


# --------------------------------------------------------------------------- #
# Synthetic world (machinery proof only)
# --------------------------------------------------------------------------- #
def synthetic_world(n_months: int = 168, beta: float = 1.0, loading: float = 0.0,
                    alpha_ann: float = 0.0, seed: int = 601) -> pd.DataFrame:
    """Deterministic joint monthly world: market excess, style spread, fund excess.

    * market: 6%/yr excess mean, 15% vol; spread: zero-mean, 8% vol, independent of the
      market (a clean style factor).
    * fund excess = alpha_ann/12 + beta x market + loading x spread + idiosyncratic noise
      (2%/yr-vol) — the three TUNABLE planted parameters.

    ``beta=1, loading=0, alpha_ann=0`` is the null: the CAPM-alpha NW t, the style-loading
    NW t and the beta-below-one t must all stay quiet. Index built via ``period_range``
    (spans stay far below the 250-year ns-Timestamp trap).
    """
    rng = np.random.default_rng(seed)
    pidx = pd.period_range("2012-01", periods=n_months, freq="M")
    idx = pidx.to_timestamp(how="end").normalize()

    mkt = 0.06 / 12.0 + (0.15 / np.sqrt(12.0)) * rng.standard_normal(n_months)
    spr = (0.08 / np.sqrt(12.0)) * rng.standard_normal(n_months)
    eps = (0.02 / np.sqrt(12.0)) * rng.standard_normal(n_months)
    fund = alpha_ann / 12.0 + beta * mkt + loading * spr + eps

    return pd.DataFrame({"mkt": mkt, "spread": spr, "fund": fund}, index=idx)
