"""Data layer for Study 662 — EM-Local-Bonds.

Two ingredients, both offline-friendly once cached:

* **Real tape.** Daily total-return-adjusted closes (``auto_adjust=True`` — dividends folded
  into the price series) for six yfinance tickers (no key), cached as one wide CSV under the
  study's own ``_cache/``:

  - ``EBND`` — SPDR Bloomberg Emerging Markets Local Bond ETF (inception 2011-02-24), tracks
    the Bloomberg EM Local Currency Government Universal index.
  - ``LEMB`` — iShares J.P. Morgan EM Local Currency Bond ETF (inception 2011-10-20), tracks
    the J.P. Morgan GBI-EM Global Diversified index. A different index family from EBND —
    averaging the two is a cross-provider robustness check, not double-counting one benchmark.
  - ``EMB``  — iShares J.P. Morgan USD Emerging Markets Bond ETF (inception 2007-12-17), the
    USD-denominated EM sovereign benchmark the claim says local debt should beat.
  - ``AGG``  — iShares Core U.S. Aggregate Bond ETF (inception 2003-09-22), the "own the bond
    market and stay home" alternative.
  - ``UUP``  — Invesco DB US Dollar Index Bullish Fund (inception 2007-02-20), a tradable proxy
    for dollar-index (DXY) strength — the FX-drag regressor.
  - ``BIL``  — SPDR Bloomberg 1-3 Month T-Bill ETF (inception 2007-05-25), the cash proxy for
    excess-of-cash Sharpe.

  ``LEMB`` sets the common start: the earliest month at which *all six* series exist is
  **2011-11-30** (first full month after LEMB's 2011-10-20 inception).

* **Named crisis windows, hardcoded.** Three dollar-strength / EM-stress episodes the claim's
  believers point to as the moments local FX should "eat the carry": the 2013 taper tantrum,
  the 2015 EM-FX / commodity selloff, and the 2022 Fed-hiking dollar surge. Facts, no network —
  same convention as sibling studies' hardcoded calendars (e.g. 637's FOMC dates).

* **Synthetic world.** A deterministic, seeded pair of monthly bond-return series with a
  TUNABLE planted "extra yield" (``yield_pickup``, in monthly return) and an independent
  TUNABLE "FX drag" (``drag``) that ties the local leg to a common dollar-shock factor.
  ``yield_pickup=0`` is the null world — the two legs are statistically identical, and the
  Welch/NW machinery must NOT manufacture a difference from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build the
cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICE_CACHE = os.path.join(CACHE_DIR, "elb_prices.csv")

TICKERS = ["EBND", "LEMB", "EMB", "AGG", "UUP", "BIL"]

FETCH_START = "2003-01-01"       # AGG's own inception predates this; harmless wide net
AS_OF = "2026-06-30"             # last complete month at publication (2026-07-10)
COMMON_START = "2011-11-30"      # first full month all six tickers co-exist (LEMB inception)

# --------------------------------------------------------------------------- #
# Named crisis / dollar-strength windows, hardcoded (facts, no network).
# Chosen ex ante from the well-known EM-local-debt drawdown episodes the claim's own
# believers cite — not snooped from this study's results.
# --------------------------------------------------------------------------- #
CRISIS_WINDOWS = {
    "2013 taper tantrum":  ("2013-05-01", "2013-12-31"),   # Bernanke's 2013-05-22 testimony
    "2015 EM-FX selloff":  ("2015-08-01", "2015-12-31"),   # China deval (Aug) -> Fed liftoff (Dec)
    "2022 strong dollar":  ("2022-01-01", "2022-10-31"),   # Fed hiking cycle, DXY multi-decade high
}


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = FETCH_START, end: str = "2026-07-01") -> None:
    """Download total-return-adjusted daily closes for all six tickers; cache one wide CSV.

    ``auto_adjust=True`` folds dividends into the price series (total return), matching the
    desk's convention for buy-and-hold ETF comparisons. Network; runs once.
    """
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    cols = {}
    for t in TICKERS:
        d = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        cols[t] = d["Close"].dropna()
    px = pd.DataFrame(cols).dropna(how="all")
    px.to_csv(PRICE_CACHE)


def have_real() -> bool:
    return os.path.exists(PRICE_CACHE)


def load_real(start: str = COMMON_START, asof: str = AS_OF) -> pd.DataFrame:
    """Cached wide price panel (all six tickers), sliced to [start, asof]."""
    px = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True).sort_index()
    return px.loc[(px.index >= "2003-01-01") & (px.index <= asof)].copy()


# --------------------------------------------------------------------------- #
# Synthetic world — planted extra yield + independent FX drag (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(seed: int = 662, n_months: int = 176,
                     yield_pickup: float = 0.0, drag: float = 0.0,
                     mu_usd: float = 0.004, sigma_usd: float = 0.018,
                     sigma_idio: float = 0.010, sigma_fx: float = 0.028,
                     mu_fx: float = 0.0025,
                     ) -> tuple[pd.Series, pd.Series]:
    """Deterministic monthly (local, usd) bond-return pair with two independent knobs.

    ``usd`` is a plain AR(0) monthly bond-return series (mean ``mu_usd``, vol ``sigma_usd`` —
    the coarse shape of a USD EM-credit fund). ``local`` = ``usd`` + ``yield_pickup`` (the
    claimed extra carry, constant every month) − ``drag`` × a common monthly dollar factor
    (mean ``mu_fx``, matching the real UUP tape's own +0.28%/mo secular drift over this sample,
    plus noise ``sigma_fx``) + idiosyncratic noise.

    ``yield_pickup=0, drag=0`` is the null: the two legs are statistically identical and the
    Welch/NW split on the paired difference must NOT fire. A positive ``yield_pickup`` with
    ``drag=0`` is the planted positive control: a genuine, undragged extra-carry world the
    detector must recover. ``drag=1.0`` with ``mu_fx == yield_pickup`` is an illustrative
    "the secular dollar drift exactly eats the pickup" world (narrative only, never a
    certification requirement).

    Business-month span (176 months ~= 14.7 years) — far below the pandas ns-timestamp trap;
    ``pd.period_range`` used regardless as the house convention for monthly labels.
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2011-11", periods=n_months, freq="M").to_timestamp(how="end")

    dollar_factor = mu_fx + rng.normal(0.0, sigma_fx, n_months)  # common monthly dollar factor
    usd = mu_usd + rng.normal(0.0, sigma_usd, n_months)
    idio = rng.normal(0.0, sigma_idio, n_months)
    local = usd + yield_pickup - drag * dollar_factor + idio

    return pd.Series(local, index=idx, name="local"), pd.Series(usd, index=idx, name="usd")
