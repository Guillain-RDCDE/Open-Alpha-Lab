"""Data layer for Study 910 — Managed-Distribution CEF.

The claim under test: **closed-end funds (CEFs) that trade at a persistent discount and pay a
big "managed distribution" hand the buyer both the discount pull *and* the payout** — a
structural double-carry the passive index can't offer. The sceptic's counter (the mREIT lesson
of [611-mreit-carry](../611-mreit-carry/)): the fat distribution is often **return-of-capital**,
and the leverage inside the wrapper is financed at short rates — so **NAV erosion and leverage
cost can quietly eat the whole payout**, leaving a levered-beta clone of the asset class with a
seductive yield sticker.

We test this on liquid, buyable tape (yfinance, ``auto_adjust=True`` **total-return** closes, so
every distribution is reinvested and the number we hold is the honest economic return, whatever
label the fund put on the cash):

  * **PCEF** — Invesco CEF Income Composite ETF, a *CEF-of-CEFs* holding ~130 income CEFs
    (inception 2010-02): the diversified basket you can buy in one click.
  * A **hand basket** of four large single-name managed-distribution CEFs, equal-weight,
    monthly-rebalanced:
      - **PDI** — PIMCO Dynamic Income Fund (leveraged multi-sector bond CEF, ~13 % dist. yield);
      - **UTF** — Cohen & Steers Infrastructure Fund (leveraged infra equity CEF);
      - **BST** — BlackRock Science & Technology Trust (option-overwrite tech CEF);
      - **RQI** — Cohen & Steers Quality Income Realty Fund (leveraged real-estate CEF).
    The basket window starts at the youngest member (BST, 2014-10).
  * **SPY** — the broad risk benchmark (the asset class you'd otherwise hold).
  * **BIL** — SPDR 1-3 Month T-Bill ETF, the **cash / risk-free leg**: every Sharpe here is
    excess-of-cash, both legs minus BIL, so a leveraged fund cannot hide leverage in the number.

**What the tape can and cannot show — stated up front.** yfinance gives *market-price* total
return, not the fund's NAV, so we do **not** observe the discount series directly (that is
[367-closed-end-fund-discount](../367-closed-end-fund-discount/)'s job). What we *can* settle is
the buyer's bottom line: does holding these persistent-discount, big-distribution CEFs deliver a
**real excess-of-cash, risk-adjusted return vs the asset class**, or does it collapse to levered
beta once you net out cash and beta? That is the question a buyer actually cares about.

**Survivorship / short-history — named on the Signal axis.** These are the *flagship* CEFs that
gathered assets and survived; funds that blew up on their leverage are absent, so any positive
read is an **upper bound**. The four-name basket has only ~11.5 years of tape (BST-limited);
PCEF gives ~16 years for the diversified proxy. Both caveats travel with every number.

Synthetic world: a deterministic monthly generator with a TUNABLE planted excess carry
(``carry_annual``) on top of a market beta, plus a **return-of-capital leak** knob that drags the
*total* return below the headline distribution — the positive control (and, at
``carry_annual = 0``, the null that must NOT fire). Index built with ``period_range`` kept well
under the pandas ns-Timestamp horizon.

Cache-first: ``fetch`` (network, yfinance) runs once and writes ``_cache/mdc_prices.csv``;
everything else is offline.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "mdc_prices.csv")

# CEF-of-CEFs proxy, four single-name managed-distribution CEFs, broad benchmark, cash leg.
TICKERS = ["PCEF", "PDI", "UTF", "BST", "RQI", "SPY", "BIL"]

# The four single-name CEFs that make the equal-weight hand basket.
BASKET = ["PDI", "UTF", "BST", "RQI"]

AS_OF = "2026-06-30"  # last complete calendar month at build time

__all__ = [
    "TICKERS", "BASKET", "AS_OF", "CACHE_DIR", "PRICES_CACHE",
    "fetch", "have_real", "load_prices", "monthly_panel", "synthetic_world",
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2003-01-01", end: str | None = None,
          path: str = PRICES_CACHE, retries: int = 4) -> pd.DataFrame:
    """Download total-return closes (auto_adjust=True) for all tickers; cache (run once)."""
    import yfinance as yf

    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True,
                              progress=False)["Close"]
            if raw is not None and len(raw) > 0 and raw.notna().any().any():
                break
        except Exception:
            time.sleep(2.0)
    raw = raw.dropna(how="all").sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = PRICES_CACHE) -> pd.DataFrame:
    """Wide total-return close frame, cache-first (OFFLINE once cached)."""
    if not os.path.exists(path):
        return fetch(path=path)
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def monthly_panel(prices: pd.DataFrame, asof: str = AS_OF) -> pd.DataFrame:
    """Monthly simple total returns for every ticker, sliced to the as-of month-end.

    Month-end resample of the total-return closes, then ``pct_change``. The partial current
    month is dropped by the ``<= asof`` slice. Columns are the raw tickers; downstream code
    builds the equal-weight basket and the excess-of-cash legs.
    """
    px = prices[prices.index <= pd.Timestamp(asof)]
    m = px.resample("ME").last()
    m = m[m.index <= pd.Timestamp(asof)]
    out = pd.DataFrame(index=m.index)
    for t in TICKERS:
        if t in m.columns:
            out[t] = m[t].pct_change()
    return out


# --------------------------------------------------------------------------- #
# Synthetic world (positive control + null)
# --------------------------------------------------------------------------- #
def synthetic_world(n_months: int = 180, carry_annual: float = 0.03,
                    roc_leak_annual: float = 0.0, seed: int = 910,
                    beta: float = 1.1, mkt_vol: float = 0.042,
                    idio_vol: float = 0.020, cash_annual: float = 0.02) -> pd.DataFrame:
    """Deterministic monthly world with a PLANTED excess carry and a return-of-capital leak.

    A market factor drives everything; the "CEF" is a levered claim on it plus a structural
    carry and an idiosyncratic wobble, minus a return-of-capital leak that erodes the *total*
    return below what the headline distribution would suggest:

        mkt   ~ N(0.5%, mkt_vol)               (the asset class, monthly)
        cash  = cash_annual / 12               (the risk-free leg)
        cef   = cash + beta*(mkt - cash) + carry_annual/12 - roc_leak_annual/12 + idio

    So the **excess-of-cash** CEF return is ``beta*(mkt-cash) + (carry - roc_leak)/12 + idio``:
    a beta clone of the asset class PLUS a net structural carry. ``carry_annual = 0`` (and
    ``roc_leak = 0``) is the null — the CEF is pure levered beta and an excess-vs-excess alpha
    test must NOT fire. ``carry_annual > roc_leak`` plants a genuine net pickup; setting
    ``roc_leak >= carry`` models the mREIT trap where the fat payout is all return-of-capital.
    Monthly index via period_range (kept well under the 250-year ns-Timestamp cap).
    """
    rng = np.random.default_rng(seed)
    pidx = pd.period_range("2011-01", periods=n_months, freq="M")
    idx = pidx.to_timestamp(how="end").normalize()
    cash = np.full(n_months, cash_annual / 12.0)
    mkt = rng.normal(0.005, mkt_vol, n_months)
    idio = rng.normal(0.0, idio_vol, n_months)
    net_carry_m = (carry_annual - roc_leak_annual) / 12.0
    cef = cash + beta * (mkt - cash) + net_carry_m + idio
    return pd.DataFrame({"cef": cef, "mkt": mkt, "cash": cash}, index=idx)
