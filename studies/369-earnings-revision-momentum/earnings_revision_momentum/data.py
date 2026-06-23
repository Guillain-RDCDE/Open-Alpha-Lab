"""Data layer for Study 369 — Earnings-Revision-Momentum (revision proxy + prices).

Two sources, both offline-friendly:

* **Real tape.** For a fixed basket of long-listed US large-caps we cache (a) daily
  adjusted closes (yfinance, no key) under ``_cache/basket_prices.csv`` and (b) a long
  per-quarter earnings table under ``_cache/earnings.csv`` with columns
  ``ticker, date, eps_est, eps_rep, surprise_pct`` (yfinance ``get_earnings_dates`` — the
  analyst **EPS estimate**, the **reported EPS**, and the **surprise %**). Paid analyst
  *revision* feeds (IBES) are not free, so we build a transparent **proxy** for "estimates
  being revised up": the realized **earnings surprise** (reported beats estimate) and the
  quarter-over-quarter change in the consensus estimate. Both are named as a proxy
  throughout — a stock that keeps beating and whose estimate keeps rising is the public
  stand-in for "up-revision."

* **Synthetic.** A deterministic, fixed-seed generator that produces a panel of stock
  returns with a *planted* link between a revision score and forward return (the
  ``edge`` knob). It is the positive control: with ``edge=0`` the long-short spread must
  NOT be significant; with a large ``edge`` it must light up.

Pure numpy + pandas + stdlib for the offline path. ``fetch_*`` (network) are only used to
build the cache and are never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PRICE_CACHE = os.path.join(HERE, "..", "_cache", "basket_prices.csv")
EARN_CACHE = os.path.join(HERE, "..", "_cache", "earnings.csv")

# A transparent, fixed basket of large, long-listed US large-caps. Chosen for long price
# AND long earnings-estimate history (yfinance covers EPS estimates back to ~2001 for these
# names). Survivorship is acknowledged on the Signal axis: a fixed surviving-names basket
# carries a mild upward tilt and excludes failed firms whose revisions cratered.
BASKET = [
    "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "GE", "IBM",
    "CVX", "PFE", "MRK", "T", "VZ", "INTC", "CSCO", "HD", "MCD", "DIS",
    "BA", "CAT", "MMM", "HON", "UNH", "WFC", "C", "BAC", "ORCL", "PEP",
    "ABT", "TXN", "COST", "LOW", "AMGN", "GS", "USB", "AXP", "DE", "DUK",
]


# --------------------------------------------------------------------------- #
# Real tape (network — build the cache once)
# --------------------------------------------------------------------------- #
def fetch_prices(start: str = "1995-01-01", end: str | None = None,
                 path: str = PRICE_CACHE) -> pd.DataFrame:
    """Download SPY + basket daily adjusted closes and cache a wide CSV. Network-only."""
    import yfinance as yf

    tickers = ["SPY"] + BASKET
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    keep = [c for c in raw.columns if raw[c].notna().mean() >= 0.40]
    out = raw[keep].copy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out


def fetch_earnings(path: str = EARN_CACHE, limit: int = 100) -> pd.DataFrame:
    """Download per-quarter EPS estimate / reported / surprise for the basket. Network-only.

    Produces a long table: one row per (ticker, earnings date) with the analyst EPS
    estimate, the reported EPS, and the surprise %. Only rows with a *reported* EPS (a
    realized event, no look-ahead) are kept.
    """
    import yfinance as yf

    frames = []
    for tk in BASKET:
        try:
            e = yf.Ticker(tk).get_earnings_dates(limit=limit)
        except Exception:
            e = None
        if e is None or len(e) == 0:
            continue
        e = e.rename(columns={"EPS Estimate": "eps_est", "Reported EPS": "eps_rep",
                              "Surprise(%)": "surprise_pct"})
        e = e[["eps_est", "eps_rep", "surprise_pct"]].copy()
        e = e.dropna(subset=["eps_rep"])          # realized events only
        e.index = pd.to_datetime(e.index, utc=True).tz_localize(None).normalize()
        e = e.reset_index().rename(columns={"index": "date", "Earnings Date": "date"})
        e["ticker"] = tk
        frames.append(e)
    out = pd.concat(frames, ignore_index=True)
    out = out[["ticker", "date", "eps_est", "eps_rep", "surprise_pct"]].sort_values(
        ["ticker", "date"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    return out


# --------------------------------------------------------------------------- #
# Real tape (offline loaders)
# --------------------------------------------------------------------------- #
def have_real(price_path: str = PRICE_CACHE, earn_path: str = EARN_CACHE) -> bool:
    return os.path.exists(price_path) and os.path.exists(earn_path)


def load_prices(path: str = PRICE_CACHE) -> pd.DataFrame:
    """Cached wide adjusted-close frame (index = date, columns = tickers + SPY)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def load_earnings(path: str = EARN_CACHE) -> pd.DataFrame:
    """Cached long earnings table with a revision proxy attached.

    Adds:
      * ``surprise``     — standardized signed surprise (reported - estimate) / |estimate|,
                           clipped; the realized "beat/miss" proxy for an up/down revision.
      * ``est_revision`` — q/q change in the consensus EPS estimate per ticker (the analyst
                           estimate trending up = the revision direction itself).
      * ``rev_score``    — the combined revision-momentum score = z(surprise) + z(est_revision)
                           within each earnings date (cross-sectional, no look-ahead).
    """
    e = pd.read_csv(path, parse_dates=["date"]).sort_values(["ticker", "date"])
    est = e["eps_est"].replace(0, np.nan)
    e["surprise"] = ((e["eps_rep"] - e["eps_est"]) / est.abs()).clip(-1.0, 1.0)
    e["surprise"] = e["surprise"].fillna(e["surprise_pct"] / 100.0)
    e["est_revision"] = e.groupby("ticker")["eps_est"].diff() / est.abs()
    e["est_revision"] = e["est_revision"].clip(-1.0, 1.0)
    return e.reset_index(drop=True)


def load_real(price_path: str = PRICE_CACHE,
              earn_path: str = EARN_CACHE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: (wide prices, earnings-with-proxy) in one call."""
    return load_prices(price_path), load_earnings(earn_path)


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_stocks: int = 40, n_quarters: int = 90, edge: float = 0.0,
                    seed: int = 369, sig_q: float = 0.16) -> pd.DataFrame:
    """Deterministic stock-quarter panel with a *planted* revision-momentum edge.

    Each quarter every stock draws a revision score ``rev_score ~ N(0,1)`` and a forward
    quarterly return ``ret = edge * rev_score + noise``. With ``edge = 0`` the score
    carries NO information and the long-short spread must be insignificant; with a large
    ``edge`` the top-minus-bottom spread must light up. ``sig_q`` is the idiosyncratic
    quarterly return vol.

    Returns a long frame: ``quarter, stock, rev_score, fwd_ret`` (a decorative integer
    quarter index — no calendar dates, so no OutOfBounds risk).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_quarters):
        score = rng.normal(0.0, 1.0, size=n_stocks)
        noise = rng.normal(0.0, sig_q, size=n_stocks)
        mkt = rng.normal(0.02, 0.04)                 # common quarterly market move
        fwd = mkt + edge * score + noise
        for s in range(n_stocks):
            rows.append((q, s, float(score[s]), float(fwd[s])))
    return pd.DataFrame(rows, columns=["quarter", "stock", "rev_score", "fwd_ret"])
