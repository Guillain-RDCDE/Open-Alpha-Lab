"""Data layer for Study 901 — Profitable Small-Caps.

The claim under test (Asness, Frazzini, Israel, Moskowitz & Pedersen 2018, *"Size Matters,
If You Control Your Junk"*): the historical **size premium** is weak and unstable *until*
you control for quality — small caps are, on average, junkier (lower profitability, higher
beta), and once you hold quality fixed the size effect is large, stable and pervasive. The
tradable reading: a small-cap fund that keeps only the **profitable / high-quality** names
should earn a better **risk-adjusted** return than a plain small-cap index, and stand up
against large caps too.

We test it on liquid, live ETFs (yfinance, ``auto_adjust=True`` **total-return** closes):

* **CALF** — Pacer US Small Cap Cash Cows 100: the top-100 small caps by trailing
  free-cash-flow yield, FCF-weighted. A pure "profitable small-cap" expression (inception
  2017-06).
* **XSHQ** — Invesco S&P SmallCap Quality: S&P SmallCap 600 names screened/weighted on a
  quality composite (ROE, accruals, leverage) (inception 2017-04).
* **IWM** — iShares Russell 2000: plain small caps, the reference "junk-and-all" small-cap
  beta (inception 2000).
* **IJR** — iShares Core S&P Small-Cap 600: plain small caps, but the S&P 600 already
  imposes a **mild earnings screen** at index construction — a "half-cleaned" baseline
  (inception 2000).
* **SPY** — SPDR S&P 500: the large-cap yardstick (inception 1993).
* **BIL** — SPDR 1-3 Month T-Bill: the **cash leg**. Every Sharpe here is
  **excess-of-cash** (fund return minus BIL's total return); ``^IRX`` is a fallback proxy.

**Short-history caveat, named on the Signal axis.** CALF and XSHQ are young (2017+): the
profitable-small-cap read has ~9 years of live tape, one of which is the 2020 COVID
crash and 2022 bear. We slice every race to a **common window** so the comparison is
apples-to-apples, and we cut a pre-/post-2021 era to check era-robustness — but nine years
is nine years, and the stamp says so.

The synthetic control (``synthetic_world``) plants a **tunable quality Sharpe edge** on top
of a shared small-cap factor plus a junk drag, with a null at ``edge=0`` — it proves the
Sharpe-race machinery is unbiased, never that the real edge is real.

Cache-first: ``fetch`` (network, yfinance) runs once and writes ``_cache/psc_prices.csv``;
everything else is offline.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "psc_prices.csv")

# Profitable small caps · plain small caps · large cap · cash leg (+ ^IRX fallback).
QUALITY = ["CALF", "XSHQ"]          # profitable / high-quality small-cap ETFs
PLAIN = ["IWM", "IJR"]              # plain small-cap beta (IJR = mild S&P earnings screen)
LARGE = ["SPY"]                     # large-cap yardstick
CASH = ["BIL"]                      # 1-3 month T-bill total-return (the cash leg)
TICKERS = QUALITY + PLAIN + LARGE + CASH + ["^IRX"]

AS_OF = "2026-06-30"  # last complete calendar month at build time; drop the partial month.

# The era split: everything through 2020 (young ETFs' first ~3.5 yrs incl. COVID) vs 2021+.
ERA_SPLIT = "2021-01-01"

# Expense ratios (fund pages, 2026) — used in the tradability discussion.
EXPENSE_RATIOS = {"CALF": 0.59, "XSHQ": 0.29, "IWM": 0.19, "IJR": 0.06, "SPY": 0.09,
                  "BIL": 0.14}

__all__ = [
    "QUALITY", "PLAIN", "LARGE", "CASH", "TICKERS", "AS_OF", "ERA_SPLIT",
    "EXPENSE_RATIOS", "CACHE_DIR", "PRICES_CACHE",
    "fetch", "have_real", "load_prices", "daily_frame", "synthetic_world",
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2000-01-01", end: str | None = None,
          path: str = PRICES_CACHE, retries: int = 4) -> pd.DataFrame:
    """Download total-return closes for all tickers and cache them (network, run once)."""
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
    if raw is None:
        raise RuntimeError("yfinance returned no data after retries")
    raw = raw.dropna(how="all").sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = PRICES_CACHE) -> pd.DataFrame:
    """Wide total-return close frame (ETFs + ^IRX %), cache-first."""
    if not os.path.exists(path):
        return fetch(path=path)
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def daily_frame(prices: pd.DataFrame, asof: str = AS_OF) -> pd.DataFrame:
    """Daily **excess-of-cash** simple returns for every ETF, aligned on common dates.

    ``r_<ticker>`` is the fund's daily total return; ``rf`` is BIL's daily total return
    (the cash leg); ``x_<ticker> = r_<ticker> - rf`` is the excess-of-cash return that every
    Sharpe in :mod:`strategy` consumes. Sliced to ``asof`` (no partial current month).
    Rows are kept only where BIL and at least the plain/large legs are present; the young
    quality ETFs carry NaN before their inception and each race self-slices to its own
    common window.
    """
    px = prices[prices.index <= pd.Timestamp(asof)].sort_index()
    funds = QUALITY + PLAIN + LARGE + CASH
    rets = px[funds].pct_change()

    out = pd.DataFrame(index=px.index)
    for t in funds:
        out[f"r_{t}"] = rets[t]
    out["rf"] = rets["BIL"]
    # ^IRX is an annualized discount rate in %; a daily-compounded fallback cash rate.
    if "^IRX" in px.columns:
        out["rf_irx"] = (1.0 + px["^IRX"].clip(lower=0) / 100.0) ** (1.0 / 252.0) - 1.0
    for t in funds:
        out[f"x_{t}"] = out[f"r_{t}"] - out["rf"]
    # Drop the leading all-NaN rows (before SPY/BIL exist) but keep NaNs for young funds.
    out = out[out["rf"].notna() | out["r_SPY"].notna()]
    return out


# --------------------------------------------------------------------------- #
# Synthetic world — planted quality Sharpe edge (positive control + null)
# --------------------------------------------------------------------------- #
def synthetic_world(n_days: int = 2200, edge: float = 0.4, seed: int = 901,
                    mkt_vol: float = 0.011, small_vol: float = 0.007,
                    junk_vol: float = 0.004, rf_annual: float = 0.02) -> pd.DataFrame:
    """Deterministic daily world with a PLANTED quality edge (the positive control).

    A shared market factor ``m`` and a small-cap factor ``s`` drive everyone. Plain small
    caps additionally carry a **junk** factor ``j`` (extra volatility with *no* mean reward
    — the drag AFMP describe); the quality small-cap fund omits the junk factor and earns an
    extra daily drift ``q_bump = edge * 0.0009`` (the "cleaned" size premium):

        rf[t]     = rf_annual / 252                       (flat cash leg)
        large     = rf + m + idio
        plain     = rf + m + s + j + idio                 (junk drag: vol, NO mean reward)
        quality   = rf + m + s + q_bump + idio            (no junk, planted extra drift)

    The quality leg gets the same ``m + s`` systematic exposure as plain small caps, **drops**
    the zero-mean junk noise ``j``, and adds a mean bump. ``edge = 0`` is the null: quality
    and plain differ only by the (zero-mean) junk noise, so no return/Sharpe advantage should
    be detectable and a HAC t on the daily difference must stay below ~2. At the default
    ``edge = 0.4`` the planted quality-minus-plain daily difference is recovered at HAC
    t ≈ 3 (the machinery detects a true edge cleanly). Business-day index, span far under the
    ns-Timestamp cap (n_days < 10000).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-02", periods=n_days)   # < 10000 days: OOB-safe
    rf = np.full(n_days, rf_annual / 252.0)

    m = rng.normal(0.0003, mkt_vol, n_days)              # market factor (with drift)
    s = rng.normal(0.0001, small_vol, n_days)            # small-cap factor
    j = rng.normal(0.0, junk_vol, n_days)                # junk factor: vol, NO mean reward
    q_bump = edge * 0.0009                               # planted quality mean edge (daily)

    idio_l = rng.normal(0.0, 0.003, n_days)
    idio_p = rng.normal(0.0, 0.003, n_days)
    idio_q = rng.normal(0.0, 0.003, n_days)

    r_large = rf + m + idio_l
    r_plain = rf + m + s + j + idio_p                    # junky small caps
    r_quality = rf + m + s + q_bump + idio_q             # profitable small caps (no junk)

    return pd.DataFrame(
        {
            "r_quality": r_quality, "r_plain": r_plain, "r_large": r_large, "rf": rf,
            "x_quality": r_quality - rf, "x_plain": r_plain - rf, "x_large": r_large - rf,
        },
        index=idx,
    )
