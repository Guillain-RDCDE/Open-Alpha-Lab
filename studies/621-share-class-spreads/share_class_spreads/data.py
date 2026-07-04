"""Data layer for Study 621 — Share-Class Spreads.

Two sources, both offline-friendly once cached:

* **Real tape.** Daily **split-adjusted** closes (yfinance ``auto_adjust=True``) for two dual-
  class pairs:

    - **BRK-A / BRK-B** from the B-share issue (1996-05-09). One A share is convertible — one
      way, A→B only — into 1,500 B shares (30 before the 50:1 B split of 2010-01-21; on
      split-adjusted closes the economic parity is a constant **1,500** across the whole tape).
      Because any holder of an A share can mint 1,500 B and sell them, **B can never trade
      meaningfully above A/1500** — the bound the market is supposed to enforce daily. Nothing
      converts B→A, so B *can* trade at a discount.
    - **GOOG / GOOGL** from the 2014-04-03 class-C split. GOOGL (class A) carries one vote,
      GOOG (class C) none — and **no conversion bridge exists in either direction**, so the
      voting spread floats free. Both classes split 20:1 together (2022-07-18) and pay identical
      dividends (from 2024-06); adjusted closes keep the ratio clean.

  Cached wide under ``_cache/scs_prices.csv``. Neither Berkshire class ever paid a dividend, so
  adjusted = raw for BRK-A; the only BRK-B adjustment is the 2010 split.

* **Synthetic.** A deterministic two-class world with a shared fundamental value, class-specific
  microstructure noise, an AR(1) relative-value gap with a **TUNABLE planted mean discount**
  (knob ``mean_discount``) and a **one-way conversion bound** knob (``bound``): with the bound
  ON, any B premium above ``bound_eps`` is instantly arbitraged back to the cap; with it OFF
  (the null), the gap is symmetric and violations land on both sides ~half the time. The
  detector (violation asymmetry + HAC t on the mean discount) must recover the plant and must
  NOT manufacture a bound out of an unbounded null.

Pure numpy + pandas for the offline path. ``fetch_pairs`` (network) runs once to build the
cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "scs_prices.csv")

TICKERS = ["BRK-A", "BRK-B", "GOOG", "GOOGL"]

# Economic parity of the BRK pair on split-adjusted closes: 1 A == 1,500 B across the whole
# tape (the pre-2010 ratio was 30 B per A; the 50:1 B split of 2010-01-21 makes the adjusted
# ratio a constant 1,500). Source: Berkshire Hathaway "Comparative Rights and Relative Prices
# of Class A and B Stock" memo (berkshirehathaway.com/compab.pdf).
BRK_PARITY = 1500.0

# First day both Google classes trade (class-C distribution): 2014-04-03.
GOOG_START = "2014-04-03"
# First day BRK-B trades: 1996-05-09.
BRKB_START = "1996-05-09"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_pairs(start: str = "1996-05-01", end: str | None = None,
                path: str = PRICES_CACHE, retries: int = 3) -> pd.DataFrame:
    """Download split-adjusted daily closes for the two pairs and cache them wide.

    Network-only; used once to build the cache. ``auto_adjust=True`` handles the 2010 BRK-B
    50:1 split and the 2022 Google 20:1 split (both classes together) so the parity ratios are
    constant across the tape.
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
    prices = raw.dropna(how="all").sort_index()
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prices.to_csv(path)
    return prices


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_real(path: str = PRICES_CACHE) -> pd.DataFrame:
    """Wide adjusted-close frame (index = date, columns = the four tickers)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_pair(n_days: int = 5000, mean_discount: float = -0.004,
                   bound: bool = True, bound_eps: float = 0.0005,
                   gap_rho: float = 0.97, gap_sig: float = 0.0012,
                   noise_bps: float = 3.0, seed: int = 621,
                   parity: float = BRK_PARITY) -> pd.DataFrame:
    """Deterministic two-class tape with a PLANTED discount and an optional one-way bound.

    A shared fundamental ``V`` follows a GBM; class A trades at ``V`` and class B at
    ``(V/parity) * (1 + g_t)`` where the relative-value gap ``g_t`` is AR(1) around
    ``mean_discount`` (negative = B cheap). Each class also carries iid microstructure noise
    (``noise_bps``, close nonsynchrony). With ``bound=True`` the one-way A→B conversion is
    enforced: any gap above ``+bound_eps`` snaps back to the cap (arbitrageurs mint B). With
    ``bound=False`` and ``mean_discount=0`` the gap is a symmetric, unbounded null: upward and
    downward excursions must be ~equally frequent and the mean-discount t must stay quiet.

    Returns a frame with columns ``A`` and ``B``. The date span stays well under 250 years
    (business-day index from 2000) — the pandas ns-Timestamp CI trap does not bite.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n_days)

    ret = rng.normal(0.0004, 0.012, size=n_days)
    v = 100000.0 * np.exp(np.cumsum(ret))

    g = np.empty(n_days)
    g[0] = mean_discount
    eps = rng.normal(0.0, gap_sig, size=n_days)
    for t in range(1, n_days):
        g[t] = mean_discount + gap_rho * (g[t - 1] - mean_discount) + eps[t]
        if bound and g[t] > bound_eps:
            g[t] = bound_eps
    if bound and g[0] > bound_eps:
        g[0] = bound_eps

    na = rng.normal(0.0, noise_bps * 1e-4, size=n_days)
    nb = rng.normal(0.0, noise_bps * 1e-4, size=n_days)
    a = v * (1.0 + na)
    b = (v / parity) * (1.0 + g) * (1.0 + nb)
    return pd.DataFrame({"A": a, "B": b}, index=idx)
