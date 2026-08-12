"""Data layer for Study 904 — Shareholder-Yield + Quality.

The claim under test: **buybacks add value only when they are REAL net share
reductions bought at reasonable valuations — not dilution theatre.** A raw buyback /
shareholder-yield screen scoops up every serial repurchaser, including firms whose
"buybacks" merely mop up option grants (net share count flat) or which repurchase at
rich prices. Overlay a QUALITY screen (high ROE, low accruals, stable earnings) and you
are supposed to keep the *funded, value-accretive* buyers and drop the theatre. So a
**quality-screened shareholder-yield** sleeve should beat both a raw buyback sleeve and
the plain market on risk-adjusted (excess-of-cash) terms, net of costs.

We test it as a live-product race between:

* **Quality-screened shareholder yield (QSY)** — equal weight **PKW** (Invesco Buyback
  Achievers: firms that cut net shares ≥5% over the trailing year) **+ QUAL** (iShares
  MSCI USA Quality Factor: high ROE, stable earnings, low leverage). The QUAL overlay is
  the "real buybacks, not dilution theatre" screen. Equal weight, rebalanced monthly.
* **Raw buyback (RAW)** — **PKW** alone: the unscreened shareholder-yield vehicle.
* **SPY** — the plain cap-weight market benchmark, and **BIL** (SPDR 1-3 Month T-Bill)
  as the cash leg: every Sharpe here is **excess of cash** (a sleeve's monthly return
  minus BIL's monthly total return, which *is* the realized cash return — no ^IRX
  modelling).
* **SPYD** — SPDR Portfolio S&P 500 High Dividend ETF (a raw *dividend*-yield screen),
  carried as a shorter-window shareholder-yield-context leg. **BUYB** (a standalone
  buyback ETF) lists only 2026-05 and is **too young to test** — named, not raced.

Two sources, both offline-friendly once cached.

* **Real tape (yfinance, no key).** Daily **total-return** (``auto_adjust=True``,
  dividends reinvested) closes for the six tickers, cached wide under this study's OWN
  ``_cache/`` as parquet. ``fetch`` (network) runs once with retries; ``load_prices``
  reads the parquet directly, OFFLINE.

* **Synthetic world — the positive control.** A deterministic, seeded joint monthly
  generator (market excess + a quality-screened sleeve + a raw sleeve, all
  excess-of-cash) with a TUNABLE knob ``edge`` that plants a quality-over-raw
  mean/Sharpe advantage. ``edge = 0`` is the null: both sleeves share the same mean and
  the HAC *t* on their difference must NOT light up. Machinery proof only — never a stamp.

Pure numpy + pandas + stdlib on the offline path.

**Survivorship / short-history — named on the Signal axis.** The QUAL overlay lists only
2013-07, so the quality-screened race lives inside a single ~2013-2026 mostly-bull regime
(one COVID crash, one 2022 bear). These are *surviving* flagship products. The caveat
travels with every number.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
TAPE_CACHE = os.path.join(CACHE_DIR, "syq_prices.parquet")

# The sleeves under audit + benchmark + cash leg + context legs.
QSY = ["PKW", "QUAL"]      # quality-screened shareholder yield (the "real buybacks" blend)
RAW = ["PKW"]              # raw buyback / shareholder yield (unscreened)
BENCH = "SPY"              # plain cap-weight market
CASH = "BIL"               # 1-3m T-bill ETF -> its monthly TR is the realized cash return
DIV = "SPYD"               # raw high-dividend-yield context leg (lists 2015-10)
YOUNG = "BUYB"             # standalone buyback ETF, lists 2026-05 -> too young to test

TICKERS = ["PKW", "QUAL", "SPYD", "SPY", "BIL", "BUYB"]

START = "2006-01-01"       # PKW lists 2006-12; SPY earlier — earliest possible
AS_OF = "2026-06-30"       # last complete calendar month at publication

__all__ = [
    "QSY", "RAW", "BENCH", "CASH", "DIV", "YOUNG", "TICKERS", "START", "AS_OF",
    "CACHE_DIR", "fetch", "have_real", "load_prices", "monthly_total_returns",
    "synthetic_world",
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str | None = None,
          path: str = TAPE_CACHE, retries: int = 4) -> pd.DataFrame:
    """Download the total-return tape once and cache it (network path).

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
        raise RuntimeError("yfinance returned no data for the Shareholder-Yield+Quality tape")
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
# Synthetic world — planted quality-over-raw edge (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(n_months: int = 150, edge: float = 0.0, seed: int = 904,
                    mkt_mean_ann: float = 0.07, mkt_vol_ann: float = 0.15,
                    beta_qsy: float = 0.95, beta_raw: float = 1.00,
                    idio_vol_ann: float = 0.05) -> pd.DataFrame:
    """Deterministic joint monthly world: market excess + QSY sleeve + raw sleeve.

    All three columns are **excess of cash**. Both sleeves load on a common market
    factor; the quality-screened sleeve carries a planted monthly alpha ``edge/12`` and
    a touch less idiosyncratic noise (the quality overlay's cleaner ride), so
    ``edge > 0`` plants BOTH a mean advantage and a Sharpe advantage of QSY over raw:

        mkt   ~ N(mkt_mean_ann/12, (mkt_vol_ann/sqrt12)^2)
        qsy   = edge/12 + beta_qsy * mkt + eps_q      (eps_q vol slightly lower)
        raw   =           beta_raw * mkt + eps_r

    ``edge = 0`` is the null: both sleeves have the same mean, and the HAC *t* on their
    monthly difference must stay quiet. Index is a ``PeriodIndex`` kept as periods
    (never ``.to_timestamp`` on a large monthly span — stays clear of the ns horizon).
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2013-08", periods=n_months, freq="M")

    mkt = mkt_mean_ann / 12.0 + (mkt_vol_ann / np.sqrt(12.0)) * rng.standard_normal(n_months)
    eps_q = (0.80 * idio_vol_ann / np.sqrt(12.0)) * rng.standard_normal(n_months)
    eps_r = (idio_vol_ann / np.sqrt(12.0)) * rng.standard_normal(n_months)
    qsy = edge / 12.0 + beta_qsy * mkt + eps_q
    raw = beta_raw * mkt + eps_r

    return pd.DataFrame({"mkt": mkt, "qsy": qsy, "raw": raw}, index=idx)
