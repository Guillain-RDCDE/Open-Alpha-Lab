"""Data layer for Study 900 — Quality-Income.

The claim under test: **chasing dividend YIELD selects value traps; screening dividends
for QUALITY selects durable payers — so a quality-dividend sleeve should beat a raw
high-yield sleeve on risk-adjusted terms.** We test it as a live-product race between:

* **Quality-dividend sleeve** — SCHD (Schwab US Dividend Equity, a quality + growth +
  yield screen, inception 2011-10) and NOBL (ProShares S&P 500 Dividend Aristocrats,
  25+ straight years of raised dividends, inception 2013-10). Equal weight, rebalanced
  monthly.
* **Raw high-yield sleeve** — SPHD (Invesco S&P 500 High Dividend Low Volatility, the
  top-yield-then-low-vol screen, inception 2012-10) and VYM (Vanguard High Dividend
  Yield, a broad above-median-yield cut, inception 2006-11). Equal weight, monthly.
* **SPY** — the plain-cap-weight benchmark, and **BIL** (SPDR 1-3 Month T-Bill) as the
  cash leg: every Sharpe here is **excess of cash** (a sleeve's monthly return minus
  BIL's monthly total return, which *is* the realized cash return — no ^IRX modelling).

Two sources, both offline-friendly once cached.

* **Real tape (yfinance, no key).** Daily **total-return** (``auto_adjust=True``,
  dividends reinvested) closes for the six tickers, cached wide under this study's OWN
  ``_cache/`` as parquet. ``fetch`` (network) runs once with retries; ``load_prices``
  reads the parquet directly, OFFLINE.

* **Synthetic world — the positive control.** A deterministic, seeded joint monthly
  generator (market excess + a quality sleeve + a yield sleeve, all excess-of-cash)
  with a TUNABLE knob ``edge`` that plants a quality-over-yield mean/Sharpe advantage.
  ``edge = 0`` is the null: the two sleeves have the same mean, and the HAC *t* on their
  difference must NOT light up. It is the machinery proof only — never cited for a stamp.

Pure numpy + pandas + stdlib on the offline path.

**Survivorship / short-history — named on the Signal axis.** All four sleeve ETFs are
young (NOBL, the binding constraint, only lists 2013-10), and they are *surviving*
flagship products; the common-window race therefore lives inside a single ~2013-2026
mostly-bull regime with a couple of drawdowns. The caveat travels with every number.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
TAPE_CACHE = os.path.join(CACHE_DIR, "qi_prices.parquet")

# The two sleeves under audit + benchmark + cash leg.
QUALITY = ["SCHD", "NOBL"]          # quality-dividend sleeve
YIELD = ["SPHD", "VYM"]             # raw high-yield sleeve
BENCH = "SPY"
CASH = "BIL"                        # 1-3m T-bill ETF -> its monthly TR is the cash return

TICKERS = QUALITY + YIELD + [BENCH, CASH]

START = "2006-01-01"                # VYM lists 2006-11; earliest possible
AS_OF = "2026-06-30"               # last complete calendar month at publication

__all__ = [
    "QUALITY", "YIELD", "BENCH", "CASH", "TICKERS", "START", "AS_OF", "CACHE_DIR",
    "fetch", "have_real", "load_prices", "monthly_total_returns", "synthetic_world",
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str | None = None,
          path: str = TAPE_CACHE, retries: int = 4) -> pd.DataFrame:
    """Download the six-ticker total-return tape once and cache it (network path).

    All columns are auto-adjusted closes (TOTAL-RETURN, dividends reinvested, net of
    each fund's expense ratio). Retries up to ``retries`` times on a transient failure.
    """
    import yfinance as yf

    raw = None
    for _ in range(retries):
        try:
            dl = yf.download(TICKERS, start=start, end=end, auto_adjust=True,
                             progress=False)
            raw = dl["Close"] if isinstance(dl.columns, pd.MultiIndex) else dl
            if raw is not None and len(raw) > 0 and raw.notna().any().any():
                break
        except Exception:
            time.sleep(2.0)
    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance returned no data for the Quality-Income tape")
    raw = raw[[c for c in TICKERS if c in raw.columns]].dropna(how="all").sort_index()
    os.makedirs(CACHE_DIR, exist_ok=True)
    raw.to_parquet(path)
    return raw


def have_real(path: str = TAPE_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = TAPE_CACHE) -> pd.DataFrame:
    """Cached wide total-return price frame (columns = tickers). OFFLINE, no yfinance."""
    return pd.read_parquet(path).sort_index()


def monthly_total_returns(prices: pd.DataFrame, asof: str = AS_OF) -> pd.DataFrame:
    """Month-end-to-month-end simple total returns, sliced to the frozen ``asof``.

    The partial current month is dropped so a stamped run never drifts as new sessions
    arrive.
    """
    m = prices.resample("ME").last()
    ret = m.pct_change()
    return ret[ret.index <= pd.Timestamp(asof)]


# --------------------------------------------------------------------------- #
# Synthetic world — planted quality-over-yield edge (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(n_months: int = 150, edge: float = 0.0, seed: int = 900,
                    mkt_mean_ann: float = 0.06, mkt_vol_ann: float = 0.15,
                    beta_q: float = 0.85, beta_y: float = 0.90,
                    idio_vol_ann: float = 0.06) -> pd.DataFrame:
    """Deterministic joint monthly world: market excess + quality sleeve + yield sleeve.

    All three columns are **excess of cash**. Both sleeves load on a common market
    factor; the quality sleeve carries a planted monthly alpha ``edge/12`` and a touch
    less idiosyncratic noise (a cleaner ride), so ``edge > 0`` plants BOTH a mean
    advantage and a Sharpe advantage of quality over yield:

        mkt      ~ N(mkt_mean_ann/12, (mkt_vol_ann/sqrt12)^2)
        quality  = edge/12 + beta_q * mkt + eps_q      (eps_q vol slightly lower)
        yield    =           beta_y * mkt + eps_y

    ``edge = 0`` is the null: quality and yield have the same mean, and the HAC *t* on
    their monthly difference must stay quiet. Index is a ``PeriodIndex`` kept as periods
    (never ``.to_timestamp`` on a large monthly span — stays clear of the ns horizon).
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2013-11", periods=n_months, freq="M")

    mkt = mkt_mean_ann / 12.0 + (mkt_vol_ann / np.sqrt(12.0)) * rng.standard_normal(n_months)
    eps_q = (0.80 * idio_vol_ann / np.sqrt(12.0)) * rng.standard_normal(n_months)
    eps_y = (idio_vol_ann / np.sqrt(12.0)) * rng.standard_normal(n_months)
    quality = edge / 12.0 + beta_q * mkt + eps_q
    yld = beta_y * mkt + eps_y

    return pd.DataFrame({"mkt": mkt, "quality": quality, "yield": yld}, index=idx)
