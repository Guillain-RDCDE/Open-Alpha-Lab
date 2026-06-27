"""Data layer for Study 515 — Earnings-Announcement Premium.

Two sources, both offline-friendly once cached:

* **Real tape.** Daily adjusted closes for a fixed basket of long-listed US large-caps
  (yfinance, no key) plus, for each name, its quarterly **earnings dates**
  (``Ticker.get_earnings_dates``). From these we tag, for every trading day, whether it falls
  inside an **announcement window** ``[D-1, D+1]`` (trading days) around any earnings print for
  that name. Prices are cached wide (``ep_prices.csv``); the long list of earnings dates is
  cached (``ep_dates.csv``).

* **Synthetic.** A deterministic, fixed-seed generator that produces a price panel with a
  controllable **planted announcement premium** (knob ``edge``): an extra mean return injected
  on the tagged announcement days. It is the positive control — with ``edge = 0`` the
  announcement-vs-rest gap must NOT manufacture significance; with a large planted ``edge`` it
  must light up.

Pure numpy + pandas + stdlib for the offline path. ``fetch_panel`` (network) is used only once
to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "ep_prices.csv")
DATES_CACHE = os.path.join(CACHE_DIR, "ep_dates.csv")

# A transparent, fixed basket of large, long-listed US large-caps with deep, clean earnings
# histories on yfinance. Chosen for long price history + sector spread. This is a *survivors*
# basket (all still trading in 2026) — survivorship is named on the Signal axis: a fixed
# surviving-names basket cannot capture firms that blew up, a mild upward tilt on every leg.
BASKET = [
    "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "IBM", "CVX",
    "PFE", "MRK", "INTC", "CSCO", "HD", "MCD", "DIS", "BA", "CAT", "MMM",
    "HON", "UNH", "ORCL", "PEP", "ABT", "TXN", "COST", "LOW", "AMGN", "GS",
]

WINDOW = 1   # announcement window half-width in trading days: tag [D-WINDOW, D+WINDOW]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = "2005-01-01", end: str | None = None,
                prices_path: str = PRICES_CACHE, dates_path: str = DATES_CACHE,
                limit: int = 90, retries: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download basket prices + per-name earnings dates and cache them.

    Network-only; used once to build the cache. Never imported by the offline notebook cells.
    Guards yfinance flakiness with simple retries. Writes a wide adjusted-close CSV and a long
    earnings-date CSV with one row per (ticker, date).
    """
    import yfinance as yf

    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(BASKET, start=start, end=end, auto_adjust=True,
                              progress=False)["Close"]
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            time.sleep(2.0)
    raw = raw.dropna(how="all")
    keep = [c for c in raw.columns if raw[c].notna().mean() >= 0.60]
    prices = raw[keep].copy()
    os.makedirs(os.path.dirname(prices_path), exist_ok=True)
    prices.to_csv(prices_path)

    rows = []
    for tk in keep:
        ed = None
        for _ in range(retries):
            try:
                ed = yf.Ticker(tk).get_earnings_dates(limit=limit)
                if ed is not None and len(ed) > 0:
                    break
            except Exception:
                time.sleep(1.5)
        if ed is None or len(ed) == 0:
            continue
        for ts, _r in ed.iterrows():
            d = pd.Timestamp(ts).tz_localize(None).normalize()
            rows.append({"ticker": tk, "date": d})
    dates = (pd.DataFrame(rows).dropna(subset=["date"])
             .drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"]))
    dates.to_csv(dates_path, index=False)
    return prices, dates


def have_real(prices_path: str = PRICES_CACHE, dates_path: str = DATES_CACHE) -> bool:
    return os.path.exists(prices_path) and os.path.exists(dates_path)


def load_prices(path: str = PRICES_CACHE) -> pd.DataFrame:
    """Wide adjusted-close frame (index = date, columns = tickers)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def load_dates(path: str = DATES_CACHE) -> pd.DataFrame:
    """Long earnings-date frame: columns ticker, date."""
    ed = pd.read_csv(path, parse_dates=["date"])
    return ed.sort_values(["ticker", "date"]).reset_index(drop=True)


def build_announce_flags(prices: pd.DataFrame, dates: pd.DataFrame,
                         window: int = WINDOW) -> dict:
    """Per-ticker boolean array: True on trading days inside any announcement window.

    For each earnings calendar date ``D`` we find the first trading position at/after ``D`` (the
    session that first prices the news) and tag ``[pos-window, pos+window]`` trading days. The
    tag is the same regardless of surprise sign — this is the *announcement premium*, not the
    drift. Returns ``{ticker: (idx, log_returns, is_announce)}`` where ``log_returns[i]`` is the
    daily return into day ``i`` and ``is_announce[i]`` flags day ``i`` as in-window.
    """
    out = {}
    for tk, grp in dates.groupby("ticker"):
        if tk not in prices.columns:
            continue
        px = prices[tk].dropna()
        if len(px) < 70:
            continue
        idx = px.index
        vals = px.values
        ret = np.empty(len(vals))
        ret[0] = np.nan
        ret[1:] = vals[1:] / vals[:-1] - 1.0
        flag = np.zeros(len(vals), dtype=bool)
        for d in grp["date"]:
            d = pd.Timestamp(d).normalize()
            pos = idx.searchsorted(d, side="left")
            if pos >= len(idx):
                continue
            lo = max(1, pos - window)
            hi = min(len(idx) - 1, pos + window)
            flag[lo:hi + 1] = True
        out[tk] = (idx, ret, flag)
    return out


def load_real() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: cached prices + earnings dates."""
    return load_prices(), load_dates()


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_names: int = 30, n_quarters: int = 70, edge: float = 0.0,
                    seed: int = 515, sig_daily: float = 0.014,
                    window: int = WINDOW) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic price panel + earnings dates with a PLANTED announcement premium.

    Each name has a daily random-walk price. Quarterly earnings dates (~63 trading days apart)
    each tag a ``[D-window, D+window]`` window. If ``edge`` != 0, EVERY tagged announcement day
    gets an *extra* mean daily return of ``edge`` (the planted premium — same sign for all
    names, the announcement-day level premium). With ``edge = 0`` the announcement days are
    statistically identical to the rest: the announcement-vs-rest gap must NOT reach
    significance however the noise falls.

    Returns (prices wide frame, earnings-date frame) in the same shape as ``load_real``.
    """
    rng = np.random.default_rng(seed)
    n_days = n_quarters * 63 + 200
    # decorative business-day index (only a label; positions carry the logic)
    idx = pd.bdate_range("2005-01-03", periods=n_days)
    names = [f"N{i:02d}" for i in range(n_names)]

    price_cols = {}
    date_rows = []
    for name in names:
        ret = rng.normal(0.0003, sig_daily, size=n_days)
        first = int(rng.integers(80, 140))
        q = first
        while q < n_days - 5:
            lo = max(1, q - window)
            hi = min(n_days - 1, q + window)
            if edge != 0.0:
                ret[lo:hi + 1] += edge
            date_rows.append({"ticker": name, "date": idx[q]})
            q += 63 + int(rng.integers(-4, 5))
        price_cols[name] = 100.0 * np.exp(np.cumsum(ret))

    prices = pd.DataFrame(price_cols, index=idx)
    dates = pd.DataFrame(date_rows).sort_values(["ticker", "date"]).reset_index(drop=True)
    return prices, dates
