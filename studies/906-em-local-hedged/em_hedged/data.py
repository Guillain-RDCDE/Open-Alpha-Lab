"""Data layer for Study 906 — EM Local Bonds, FX-Hedged (a proxy).

The claim under test: **emerging-market LOCAL-currency government bonds pay a fat local
short rate, but the currency swings drown that carry — so if you STRIP the FX you are
left with a real, harvestable local-rate premium** over USD-denominated EM debt (EMB)
and over cash.

The honest problem: a clean, liquid *FX-hedged EM-local* ETF does not exist on US tape.
So we build a **PROXY hedge** and label it loudly as one. EM-local total return decomposes
(log-approx, per month) as

    EMLC_return  ~  local_bond_return (local rate + duration P&L)  +  EM_FX_return

When the broad dollar rallies, EM currencies almost always sell off together, so a **long
US-dollar-index overlay** (UUP — Invesco DB US Dollar Bullish, long USD vs the DXY basket
EUR/JPY/GBP/CAD/SEK/CHF) gains roughly when the EM-FX leg of EMLC loses. Adding a UUP
overlay therefore *approximately* offsets the EM-FX drag:

    EMLC_hedged_proxy  =  EMLC  −  b · UUP      (b < 0  ⇒  a LONG-UUP overlay)

where ``b`` is the variance-min hedge ratio from regressing EMLC on UUP. The residual is
the local-rate carry plus idiosyncratic bond P&L. **This is a proxy, not a hedge:** UUP
tracks the *developed-market* DXY basket, NOT the EMLC currency basket (BRL, MXN, IDR,
ZAR, THB, MYR, …). DXY-vs-EM-FX correlation is high but well under 1, so the overlay
strips only *part* of the FX variance — the residual still carries un-hedged EM-FX beta.
Every number that follows travels with that caveat.

Tickers (yfinance, ``auto_adjust=True`` total-return closes, this study's own ``_cache/``):

  * **EMLC** — VanEck J.P. Morgan EM Local Currency Bond (the headline local-EM tape, 2010-07+).
  * **LEMB** — iShares J.P. Morgan EM Local Currency Bond (independent local-EM confirm, 2011-10+).
  * **EBND** — SPDR Bloomberg EM Local Bond (second confirm, 2011-02+).
  * **EMB**  — iShares J.P. Morgan USD EM Bond (the USD-denominated EM sibling = the bench).
  * **UUP**  — Invesco DB US Dollar Bullish (the DXY-basket overlay = the proxy hedge).
  * **BIL**  — SPDR 1-3 Month T-Bill (the tradable cash / risk-free leg).

Synthetic world: a deterministic monthly generator with a TUNABLE planted local carry
(knob ``carry_annual``) plus a duration factor, an EM-FX leg, and a DXY leg negatively
correlated with EM-FX (so the UUP proxy can strip *part* of the FX). At ``carry_annual=0``
the hedged local series has no excess over cash beyond noise — the null the machinery must
not fire on. Monthly index built with ``period_range`` (span well under the 250-year
ns-Timestamp cap).

Cache-first: ``fetch`` (network, yfinance, retries) writes ``_cache/em_prices.parquet``;
``load_prices`` reads it OFFLINE.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
PRICES_CACHE = os.path.join(CACHE_DIR, "em_prices.parquet")

# One wide total-return close frame covers the whole study.
TICKERS = ["EMLC", "LEMB", "EBND", "EMB", "UUP", "BIL"]

# The three local-EM ETFs (hedged against the UUP overlay), the USD-EM bench, cash.
LOCAL = ["EMLC", "LEMB", "EBND"]
BENCH = "EMB"
OVERLAY = "UUP"
CASH = "BIL"

AS_OF = "2026-06-30"  # last complete calendar month at build time (drop the partial month)

__all__ = [
    "TICKERS", "LOCAL", "BENCH", "OVERLAY", "CASH", "AS_OF", "CACHE_DIR", "PRICES_CACHE",
    "fetch", "have_real", "load_prices", "synthetic_world",
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2010-01-01", end: str | None = "2026-07-01",
          path: str = PRICES_CACHE, retries: int = 4) -> pd.DataFrame:
    """Download total-return closes for the six tickers and cache them (network, run once).

    ``auto_adjust=True`` => distributions reinvested (total return). Guards yfinance
    flakiness with up to ``retries`` attempts. Writes a wide parquet (index = date,
    columns = tickers).
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
        raise RuntimeError("yfinance returned no data for the EM-local panel")
    prices = raw.dropna(how="all").sort_index()
    prices = prices[[t for t in TICKERS if t in prices.columns]]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prices.to_parquet(path)
    return prices


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = PRICES_CACHE, asof: str = AS_OF) -> pd.DataFrame:
    """Wide total-return close frame, cache-first, sliced to the as-of month-end (OFFLINE)."""
    if not os.path.exists(path):
        return fetch(path=path)
    px = pd.read_parquet(path).sort_index()
    return px[px.index <= pd.Timestamp(asof)]


# --------------------------------------------------------------------------- #
# Synthetic world (positive control + null)
# --------------------------------------------------------------------------- #
def synthetic_world(n_months: int = 190, carry_annual: float = 0.04, seed: int = 906,
                    dur_vol: float = 0.012, emfx_resid_vol: float = 0.015,
                    dxy_vol: float = 0.020, fx_dxy_beta: float = -1.05,
                    idio_vol: float = 0.004, credit_carry_annual: float = 0.02,
                    bill_annual: float = 0.02) -> pd.DataFrame:
    """Deterministic monthly world with a PLANTED EM local-rate carry.

    Columns (monthly simple TOTAL returns): ``EMLC`` (local-EM bond), ``EMB`` (USD-EM bond
    bench), ``UUP`` (DXY overlay fund), ``BIL`` (cash). Every risky fund earns the cash leg
    plus its risk premia, so the excess-of-cash of each is what the strategy actually
    harvests. Construction (``bill = bill_annual/12``)::

        dxy_t   ~ N(0, dxy_vol)                                      # dollar-futures leg (mean 0)
        emfx_t  = fx_dxy_beta * dxy_t + N(0, emfx_resid_vol)         # EM FX: DXY part + a residual the proxy can't strip
        dur_t   ~ N(0, dur_vol)                                      # shared duration factor
        EMLC_t  = bill + carry_annual/12 + dur_t + emfx_t + N(0, idio)          # cash + local-vs-US differential + dur + FX
        EMB_t   = bill + credit_carry_annual/12 + dur_t + 0.4*emfx_t + N(0, idio) # USD-EM: credit carry, less FX
        UUP_t   = bill + dxy_t                                       # a dollar-bull FUND: cash collateral + futures P&L
        BIL_t   = bill                                              # cash

    ``carry_annual`` is the planted **local-minus-US** rate differential — exactly what a
    clean FX hedge should leave behind. Regressing EMLC-excess on UUP-excess (= ``dxy``,
    mean 0) and taking the residual strips the DXY-explained FX and exposes ``carry_annual``
    (plus the ``emfx_resid`` the DXY proxy can NOT reach — the honest imperfection).
    ``carry_annual = 0`` is the null: the hedged local excess then has zero mean beyond
    noise, and the HAC / Sharpe machinery must NOT manufacture a premium.
    Decorative monthly index via ``period_range`` (span << 250-year ns-Timestamp cap).
    """
    rng = np.random.default_rng(seed)
    pidx = pd.period_range("2010-07", periods=n_months, freq="M")
    idx = pidx.to_timestamp(how="end").normalize()

    bill = bill_annual / 12.0
    dxy = rng.normal(0.0, dxy_vol, n_months)
    emfx = fx_dxy_beta * dxy + rng.normal(0.0, emfx_resid_vol, n_months)
    dur = rng.normal(0.0, dur_vol, n_months)

    emlc = bill + carry_annual / 12.0 + dur + emfx + rng.normal(0.0, idio_vol, n_months)
    emb = bill + credit_carry_annual / 12.0 + dur + 0.4 * emfx + rng.normal(0.0, idio_vol, n_months)
    uup = bill + dxy
    bil = np.full(n_months, bill)

    return pd.DataFrame({"EMLC": emlc, "EMB": emb, "UUP": uup, "BIL": bil}, index=idx)
