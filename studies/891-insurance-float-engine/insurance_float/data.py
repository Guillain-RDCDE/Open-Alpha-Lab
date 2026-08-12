"""Data layer for Study 891 — Insurance Float Engine.

The claim under test: a broad P&C-insurer basket is a **structural** money-machine — the
"float" (premiums held before claims) compounds like near-zero-cost leverage, so the basket
should deliver a genuine **risk-adjusted edge over the market** *attributable to the float
structure*, not merely to riding financial-sector beta.

Two ingredients, both offline-friendly once cached.

* **Real tape — five liquid ETFs.** Daily **total-return** (``auto_adjust=True``) closes,
  pulled once with yfinance and cached under this study's OWN ``_cache/`` as parquet:

    - **KIE** — SPDR S&P Insurance ETF (**equal-weight** the S&P insurance sub-industry;
      inception 2005-11). The headline insurer basket.
    - **IAK** — iShares U.S. Insurance ETF (cap-weight; inception 2006-05). The second
      wrapper — if the edge is structural it should show in both.
    - **SPY** — the market benchmark.
    - **KBE** — SPDR S&P Bank ETF (equal-weight banks; inception 2005-11). The **control**:
      banks are the *other* financial sub-sector, levered on a spread rather than on float.
      Decomposing insurer excess against a bank-vs-market spread separates a genuine float
      premium from plain financial-sector beta.
    - **BIL** — SPDR 1-3 Month T-Bill ETF (inception 2007-05). The **cash leg** — every
      Sharpe here is measured *excess-of-cash* (both legs minus BIL), so a rising short rate
      can't flatter a thin edge. BIL's ~0.14 % ER makes it a *tradable* cash proxy (a hair
      under the true bill), which is the honest, conservative choice.

  The common window with all five (BIL binds) is **2007-06 → AS_OF**. ``auto_adjust=True``
  gives total-return prices (dividends reinvested, net of each fund's expense ratio).

* **Synthetic world — the positive control.** A deterministic, seeded monthly generator
  (``synthetic_world``) with a TUNABLE knob ``edge_ann``: insurer excess =
  ``edge_ann/12 + beta·market + load·(bank − market) + idiosyncratic``. ``edge_ann = 0`` is
  the null (insurer is pure market + financial-sector beta, no float premium — the race and
  the CAPM alpha must NOT fire); ``edge_ann > 0`` plants a real risk-adjusted edge the
  machinery must recover. The synthetic proves the estimators are unbiased — it is **never**
  cited in support of a stamp.

The offline path is pure numpy + pandas + stdlib. ``fetch`` (network) runs once to build the
cache and is never imported by the notebooks' offline cells; ``load_prices`` reads the cached
parquet directly (no yfinance import). Monthly-index synthetics use ``period_range`` kept well
below the pandas ns-timestamp horizon (no overflow on the Python-3.10 CI).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "insfloat_prices.parquet")

# Insurer baskets, market, bank control, cash leg.
INSURERS = ["KIE", "IAK"]
MARKET = "SPY"
BANK = "KBE"
CASH = "BIL"
TICKERS = INSURERS + [MARKET, BANK, CASH]

START = "2005-01-01"       # download start (KIE/KBE list 2005-11; BIL 2007-05 binds)
AS_OF = "2026-06-30"       # last complete calendar month at build time; drop partial month

__all__ = [
    "INSURERS", "MARKET", "BANK", "CASH", "TICKERS", "START", "AS_OF",
    "CACHE_DIR", "PRICES_CACHE",
    "fetch", "have_real", "load_prices", "monthly_returns", "synthetic_world",
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str | None = None,
          path: str = PRICES_CACHE, retries: int = 4) -> pd.DataFrame:
    """Download total-return closes for the five ETFs and cache them (network, run once)."""
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
        raise RuntimeError("yfinance returned no data for the insurance-float tape.")
    raw = raw.dropna(how="all").sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_parquet(path)
    return raw


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = PRICES_CACHE) -> pd.DataFrame:
    """Cached wide daily total-return close frame. OFFLINE — reads parquet, no yfinance."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No cache at {path}. Run insurance_float.data.fetch() once (needs network)."
        )
    return pd.read_parquet(path).sort_index()


def monthly_returns(prices: pd.DataFrame, asof: str = AS_OF) -> pd.DataFrame:
    """Month-end-to-month-end simple TOTAL returns, common sample where all five exist.

    Sliced to the frozen ``asof`` (the last complete calendar month) so a stamped run never
    contains a partial month and never drifts as new sessions arrive. Rows are dropped until
    every column (BIL binds the start) has a return.
    """
    px = prices[prices.index <= pd.Timestamp(asof)]
    m = px.resample("ME").last()
    ret = m.pct_change()
    ret = ret[ret.index <= pd.Timestamp(asof)]
    cols = [c for c in TICKERS if c in ret.columns]
    return ret.dropna(subset=cols)[cols]


# --------------------------------------------------------------------------- #
# Synthetic world (positive control + null) — never market evidence
# --------------------------------------------------------------------------- #
def synthetic_world(n_months: int = 240, edge_ann: float = 0.0, seed: int = 891,
                    mkt_mean_ann: float = 0.06, mkt_vol_ann: float = 0.15,
                    rf_ann: float = 0.02, beta_ins: float = 1.05, load_bank: float = 0.35,
                    beta_bank: float = 1.40, idio_ins_ann: float = 0.06,
                    idio_bank_ann: float = 0.12) -> pd.DataFrame:
    """Deterministic monthly world with a PLANTED insurer edge (the positive control).

    Construction (all per-month), with ``rf`` a flat cash return::

        mkt_ex  ~ N(mkt_mean_ann/12, mkt_vol_ann/sqrt12)
        bank_ex = beta_bank·mkt_ex + idio_bank
        ins_ex  = edge_ann/12 + beta_ins·mkt_ex + load_bank·(bank_ex − mkt_ex) + idio_ins
        <ticker>_total = rf + <ticker>_ex

    So the insurer carries market beta *and* a financial-sector (bank-minus-market) loading —
    exactly the confound the real study must strip — plus, only when ``edge_ann > 0``, a
    genuine risk-adjusted edge. ``edge_ann = 0`` is the null: insurer excess is pure
    market + financial beta, and both the excess-vs-excess Sharpe race and the CAPM/two-factor
    alpha must stay quiet. Returns a frame with columns ``KIE, IAK, SPY, KBE, BIL`` (IAK is a
    noisier twin of KIE) so the real analysis code runs unchanged.

    Monthly index via ``period_range`` (240 months from 2006 ≈ 2026 — far below the
    ns-Timestamp horizon; kept as timestamps for a natural calendar-year table).
    """
    rng = np.random.default_rng(seed)
    pidx = pd.period_range("2006-01", periods=n_months, freq="M")
    idx = pidx.to_timestamp(how="end").normalize()

    rf = np.full(n_months, rf_ann / 12.0)
    mkt_ex = rng.normal(mkt_mean_ann / 12.0, mkt_vol_ann / np.sqrt(12.0), n_months)
    bank_ex = beta_bank * mkt_ex + rng.normal(0.0, idio_bank_ann / np.sqrt(12.0), n_months)

    def _ins(extra_idio: float) -> np.ndarray:
        eps = rng.normal(0.0, (idio_ins_ann + extra_idio) / np.sqrt(12.0), n_months)
        return edge_ann / 12.0 + beta_ins * mkt_ex + load_bank * (bank_ex - mkt_ex) + eps

    kie_ex = _ins(0.0)
    iak_ex = _ins(0.01)  # a slightly noisier second wrapper
    return pd.DataFrame(
        {
            "KIE": rf + kie_ex,
            "IAK": rf + iak_ex,
            "SPY": rf + mkt_ex,
            "KBE": rf + bank_ex,
            "BIL": rf,
        },
        index=idx,
    )
