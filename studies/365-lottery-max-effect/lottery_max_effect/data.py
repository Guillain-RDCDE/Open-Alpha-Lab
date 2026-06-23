"""Data layer for Study 365 — the Lottery / MAX effect.

Two sources, both offline-friendly:

* **Real tape.** Daily adjusted closes for a fixed, long-listed S&P-100-style basket of US
  large-caps (yfinance, no key), cached under ``_cache/basket_prices.csv`` (a wide CSV
  indexed by date, one column per ticker). From the daily returns we build, every
  month-end, each name's **MAX** = its single highest daily return over the *prior* calendar
  month. Sorting the cross-section by MAX and forming quintiles gives the
  Bali-Cakici-Whitelaw (2011) lottery portfolios. True MAX is a CRSP-universe object; this
  large-cap basket is an explicit, transparent **proxy** (survivorship-tilted, named on the
  Signal axis everywhere).

* **Synthetic.** A deterministic, fixed-seed generator that builds a monthly *panel* of
  returns with a controllable **lottery-preference edge** (``edge``): names that printed a
  large MAX last month are made to under-perform next month by a planted amount. ``edge=0``
  is the null — MAX carries no forward information, so a long-low / short-high MAX book is a
  coin; a large ``edge`` must light the harness up. This is the positive control and the
  null in one bottle. With ``edge=0`` the harness must NOT manufacture significance.

Pure numpy + pandas + stdlib for the offline path. ``fetch_basket`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "basket_prices.csv")

# A transparent, fixed S&P-100-style basket of large, long-listed US large-caps used as the
# cross-section for the MAX sort. Chosen for long price history and sector spread; this is a
# PROXY for the CRSP universe Bali-Cakici-Whitelaw used (thousands of names incl. small-caps,
# where the lottery effect is strongest). Survivorship is acknowledged on the Signal axis: a
# fixed surviving-large-cap basket is the *least* lottery-prone slice of the market.
BASKET = [
    "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "GE", "IBM",
    "CVX", "PFE", "MRK", "T", "VZ", "INTC", "CSCO", "HD", "MCD", "DIS",
    "BA", "CAT", "MMM", "HON", "UNH", "WFC", "C", "BAC", "ORCL", "PEP",
    "ABT", "TXN", "COST", "LOW", "AMGN", "GS", "USB", "AXP", "DE", "DUK",
    "NKE", "SBUX", "QCOM", "AMD", "MU", "ADBE", "CRM", "NFLX", "PYPL", "AMZN",
    "GOOGL", "META", "NVDA", "TSLA", "CMCSA", "PEP", "TMO", "DHR", "LIN", "ACN",
    "AVGO", "COP", "SLB", "EOG", "OXY", "MO", "PM", "CL", "GIS", "KMB",
    "SO", "D", "EXC", "AEP", "NEE", "F", "GM", "DAL", "UAL", "AAL",
    "EBAY", "GILD", "BIIB", "REGN", "VRTX", "BK", "STT", "SCHW", "MS", "BLK",
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_basket(start: str = "2005-01-01", end: str | None = None,
                 path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download the basket via yfinance and cache a wide adjusted-close CSV.

    Network-only; used once to build ``_cache/basket_prices.csv``. Never imported by the
    offline notebook cells. Keeps only names with a long, mostly-complete history.
    """
    import yfinance as yf

    tickers = sorted(set(BASKET))
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    # keep names present for >=50% of the window (long-listed survivors)
    keep = [c for c in raw.columns if raw[c].notna().mean() >= 0.50]
    out = raw[keep].copy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached wide adjusted-close frame (index = date, columns = tickers)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df


def build_panel(prices: pd.DataFrame) -> dict:
    """From daily prices, build the monthly MAX panel.

    Returns a dict with:
      * ``max``     — DataFrame (month-end index x ticker): each name's highest daily return
                      over that calendar month (the MAX signal, observed at month-end).
      * ``fwd_ret`` — DataFrame (month-end index x ticker): each name's *next* calendar-month
                      simple total return (what a portfolio formed at month-end earns).

    The two frames share the same month-end index; row ``t`` of ``max`` is the signal known at
    the close of month ``t`` and row ``t`` of ``fwd_ret`` is the return of month ``t+1`` — so
    pairing them is already lagged by construction (no look-ahead).
    """
    daily = prices.sort_index().pct_change()
    grp = daily.groupby(pd.Grouper(freq="ME"))
    # MAX = highest single-day return within the month (need >=15 obs to be meaningful)
    mx = grp.max()
    cnt = grp.count()
    mx = mx.where(cnt >= 15)
    # monthly total return = product of (1+daily) - 1
    mret = grp.apply(lambda d: (1.0 + d).prod() - 1.0)
    mret = mret.where(cnt >= 15)
    # forward return: month t's signal pairs with month t+1's return
    fwd = mret.shift(-1)
    return {"max": mx, "fwd_ret": fwd, "mret": mret}


def load_real(path: str = DEFAULT_CACHE) -> dict:
    """Convenience: cached prices -> monthly MAX panel in one call."""
    return build_panel(load_prices(path))


def fingerprint(prices: pd.DataFrame) -> str:
    """A short content fingerprint of the price frame, for the as-of stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_months: int = 240, n_names: int = 90, edge: float = 0.0,
                    seed: int = 365, mkt_vol: float = 0.045, idio_vol: float = 0.06,
                    mkt_drift: float = 0.006) -> dict:
    """Deterministic monthly panel with a *planted* lottery-preference edge.

    Each name loads on a common monthly market factor plus idiosyncratic noise. Every month we
    compute a stand-in **MAX** for each name (its idiosyncratic shock that month, the part that
    drives a one-day pop) and, if ``edge`` != 0, subtract ``edge`` × (cross-sectionally
    standardised MAX) from each name's *next*-month return — i.e. high-MAX names are penalised
    next month, exactly the lottery effect the sort should detect.

    * ``edge = 0``  → MAX is pure noise w.r.t. forward returns; the long-low / short-high MAX
      book is a coin. The harness must NOT print significance.
    * ``edge > 0``  → high-MAX names underperform by a known amount; the spread must turn
      positive and (for a large edge) clear t = 2. The positive control.

    A *decorative* month-end index is built with ``pd.period_range`` (never a huge
    ``date_range`` span) to stay far under the ns-Timestamp overflow wall.

    Returns the same ``{"max", "fwd_ret", "mret"}`` shape as :func:`build_panel`.
    """
    rng = np.random.default_rng(seed)
    n, k = n_months, n_names
    idx = pd.period_range("2005-01", periods=n, freq="M").to_timestamp(how="end").normalize()
    idx = pd.DatetimeIndex(idx, name="date")
    cols = [f"N{i:02d}" for i in range(k)]

    betas = rng.uniform(0.7, 1.3, size=k)
    mkt = rng.normal(mkt_drift, mkt_vol, size=n)
    idio = rng.normal(0.0, idio_vol, size=(n, k))

    # MAX stand-in: the per-name idiosyncratic shock magnitude (a name that popped this month).
    # We use the positive idiosyncratic shock as the "lottery" intensity.
    mx_raw = np.maximum(idio, 0.0) + 0.5 * idio_vol  # always positive, higher when name popped

    # base monthly returns (market + idio)
    base = mkt[:, None] * betas[None, :] + idio

    # plant the lottery penalty: next month, high-MAX (this month) names underperform.
    # standardise MAX cross-sectionally each month, then push next month's return down.
    z = (mx_raw - mx_raw.mean(axis=1, keepdims=True)) / (mx_raw.std(axis=1, keepdims=True) + 1e-9)
    penalty = np.zeros_like(base)
    penalty[1:, :] = edge * z[:-1, :]          # this month's MAX penalises next month's return
    ret = base - penalty

    mret = pd.DataFrame(ret, index=idx, columns=cols)
    mx = pd.DataFrame(mx_raw, index=idx, columns=cols)
    fwd = mret.shift(-1)
    return {"max": mx, "fwd_ret": fwd, "mret": mret}
