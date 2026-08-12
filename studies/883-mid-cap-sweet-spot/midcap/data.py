"""Data layer for Study 883 — Mid-Cap Sweet Spot.

The claim under test: **mid-caps are the "forgotten middle" — a better risk-adjusted
return than either large (SPY) or small (IWM)**. We test whether the mid-cap ETF
(**IJH** iShares Core S&P Mid-Cap 400, and **MDY** SPDR S&P MidCap 400 — same index,
longer tape) delivers a genuine **excess-of-cash Sharpe advantage over BOTH** SPY and
IWM, robust across eras, and whether it survives costs.

The cash leg is **BIL** (SPDR 1-3 Month T-Bill ETF) as the risk-free proxy, so every
Sharpe is an honest *excess-of-cash* Sharpe. BIL only lists from 2007-05, which fixes
the cash-anchored common sample; the pairwise return *difference* (mid − large) is
cash-independent, so the longer IJH (2000+) / MDY (1995+) tapes are still used for the
era-robustness cut.

Tickers (yfinance, ``auto_adjust=True`` → total-return closes):

  * **IJH** — iShares Core S&P Mid-Cap 400 ETF (2000-05+). Primary mid-cap leg.
  * **MDY** — SPDR S&P MidCap 400 ETF (1995-05+). Same index; the longest mid tape.
  * **SPY** — SPDR S&P 500 ETF (1993+). The large-cap leg.
  * **IWM** — iShares Russell 2000 ETF (2000-05+). The small-cap leg.
  * **BIL** — SPDR 1-3 Month T-Bill ETF (2007-05+). The cash / risk-free leg.

Synthetic world: a deterministic daily generator with a **tunable planted mid-cap
Sharpe edge** (knob ``edge``; null at 0). A common market factor drives large / mid /
small; ``edge`` lifts the mid leg's *excess-of-cash* mean so its Sharpe clears BOTH
neighbours — the positive control. At ``edge = 0`` no leg has an advantage (the null
the detector must NOT fire on). Daily index via ``bdate_range`` (n well under 10k).

Cache-first: ``fetch`` (network, yfinance) runs once and writes
``_cache/prices.parquet``; everything else is offline.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "prices.parquet")

TICKERS = ["IJH", "MDY", "SPY", "IWM", "BIL"]

# The five cash-anchored legs, and the human labels used everywhere downstream.
MID = "IJH"          # primary mid-cap leg (IJH; MDY is the longer-tape twin)
MID_LONG = "MDY"     # same S&P MidCap 400 index, tape back to 1995
LARGE = "SPY"
SMALL = "IWM"
CASH = "BIL"

AS_OF = "2026-06-30"  # last complete calendar month at build time; partial month dropped


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "1993-01-01", end: str | None = None,
          path: str = PRICES_CACHE, retries: int = 4) -> pd.DataFrame:
    """Download total-return closes for all tickers and cache as parquet (run once)."""
    import yfinance as yf

    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True,
                              progress=False)["Close"]
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            time.sleep(2.5)
    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance returned no data for the mid-cap tape")
    raw = raw[[t for t in TICKERS if t in raw.columns]].dropna(how="all").sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_parquet(path)
    return raw


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = PRICES_CACHE, asof: str = AS_OF) -> pd.DataFrame:
    """Wide total-return close frame (cache-first), sliced to the as-of date."""
    if not os.path.exists(path):
        fetch(path=path)
    px = pd.read_parquet(path).sort_index()
    return px[px.index <= pd.Timestamp(asof)]


# --------------------------------------------------------------------------- #
# Synthetic world (positive control + null)
# --------------------------------------------------------------------------- #
def synthetic_world(n_days: int = 3000, edge: float = 0.0, seed: int = 883,
                    cash_daily: float = 0.00018) -> pd.DataFrame:
    """Deterministic daily total-return price world with a PLANTED mid-cap Sharpe edge.

    A common market factor ``f`` drives the three equity legs (betas 0.90 / 1.00 / 1.20
    for large / mid / small) with i.i.d. idiosyncratic noise; ``cash`` grows at a flat
    ``cash_daily`` (the risk-free leg). The knob ``edge`` adds a per-day *excess* drift
    to the **mid** leg only, so its excess-of-cash Sharpe clears BOTH neighbours. At
    ``edge = 0`` every equity leg has a zero-mean excess (no advantage) — the null the
    detector must not fire on. Columns match the real frame's roles: ``mid, large,
    small, cash``. Decorative daily index via ``bdate_range`` (n well under the 10k cap).

    Returns a *price* frame (cumulative product of 1 + daily return), mirroring the real
    total-return closes so the strategy layer treats both identically.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n_days)
    f = rng.normal(0.0, 0.009, n_days)                 # market factor
    large = cash_daily + 0.90 * f + rng.normal(0.0, 0.004, n_days)
    mid = cash_daily + 1.00 * f + edge + rng.normal(0.0, 0.005, n_days)
    small = cash_daily + 1.20 * f + rng.normal(0.0, 0.007, n_days)
    cash = np.full(n_days, cash_daily)
    rets = pd.DataFrame({"mid": mid, "large": large, "small": small, "cash": cash},
                        index=idx)
    return (1.0 + rets).cumprod()
