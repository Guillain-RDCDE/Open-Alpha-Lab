"""Data layer for Study 363 — Post-Earnings-Announcement Drift (PEAD).

Two sources, both offline-friendly:

* **Real tape.** Daily adjusted closes for a fixed basket of long-listed US large-caps
  (yfinance, no key) plus, for each name, its quarterly **earnings dates** with the reported
  EPS surprise (``Ticker.get_earnings_dates``). From these we build an **event table**: one
  row per (ticker, earnings date) carrying the *surprise proxy* — the **post-announcement
  gap**, i.e. the stock's one-day reaction return the session after the print — alongside the
  vendor's reported EPS ``Surprise(%)`` for cross-checking. Both are cached under ``_cache/``
  (``pead_prices.csv`` wide, ``pead_events.csv`` long).

* **Synthetic.** A deterministic, fixed-seed generator that produces a controllable set of
  earnings events whose post-event path has a **planted drift** proportional to the surprise
  (knob ``edge``). It is the positive control: with ``edge = 0`` the long-short drift must NOT
  manufacture significance; with a large planted ``edge`` it must light up.

Pure numpy + pandas + stdlib for the offline path. ``fetch_panel`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "pead_prices.csv")
EVENTS_CACHE = os.path.join(CACHE_DIR, "pead_events.csv")

# A transparent, fixed basket of large, long-listed US large-caps with deep, clean earnings
# histories on yfinance. Chosen for long price history + sector spread. This is a *survivors*
# basket (all still trading) — survivorship is named on the Signal axis: a fixed
# surviving-names basket cannot capture firms that blew up, a mild upward tilt on the long leg.
BASKET = [
    "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "IBM", "CVX",
    "PFE", "MRK", "INTC", "CSCO", "HD", "MCD", "DIS", "BA", "CAT", "MMM",
    "HON", "UNH", "ORCL", "PEP", "ABT", "TXN", "COST", "LOW", "AMGN", "GS",
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = "2005-01-01", end: str | None = None,
                prices_path: str = PRICES_CACHE, events_path: str = EVENTS_CACHE,
                limit: int = 80) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download basket prices + per-name earnings dates and cache them.

    Network-only; used once to build the cache. Never imported by the offline notebook cells.
    Writes a wide adjusted-close CSV and a long event CSV with one row per (ticker, date)
    carrying the reported EPS ``surprise_pct`` (vendor) — the post-announcement gap proxy is
    computed later from prices, so the cache stays source-faithful.
    """
    import yfinance as yf

    raw = yf.download(BASKET, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    keep = [c for c in raw.columns if raw[c].notna().mean() >= 0.60]
    prices = raw[keep].copy()
    os.makedirs(os.path.dirname(prices_path), exist_ok=True)
    prices.to_csv(prices_path)

    rows = []
    for tk in keep:
        try:
            ed = yf.Ticker(tk).get_earnings_dates(limit=limit)
        except Exception:
            continue
        if ed is None or len(ed) == 0:
            continue
        ed = ed.dropna(subset=["Reported EPS"])
        for ts, r in ed.iterrows():
            d = pd.Timestamp(ts).tz_localize(None).normalize()
            rows.append({"ticker": tk, "date": d,
                         "surprise_pct": float(r.get("Surprise(%)", np.nan))})
    events = pd.DataFrame(rows).dropna(subset=["date"]).sort_values(["ticker", "date"])
    events.to_csv(events_path, index=False)
    return prices, events


def have_real(prices_path: str = PRICES_CACHE, events_path: str = EVENTS_CACHE) -> bool:
    return os.path.exists(prices_path) and os.path.exists(events_path)


def load_prices(path: str = PRICES_CACHE) -> pd.DataFrame:
    """Wide adjusted-close frame (index = date, columns = tickers)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def load_events(path: str = EVENTS_CACHE) -> pd.DataFrame:
    """Long event frame: columns ticker, date, surprise_pct."""
    ev = pd.read_csv(path, parse_dates=["date"])
    return ev.sort_values(["ticker", "date"]).reset_index(drop=True)


def build_event_table(prices: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Attach the **post-announcement gap** (one-day reaction) to every earnings event.

    For an earnings print stamped on calendar day ``D``, the first *full* session that prices
    the news is the trading day at or after ``D`` whose close reflects the announcement. We use
    the announcement-day reaction = close[t1] / close[t0] - 1, where ``t0`` is the last trading
    day strictly before the print and ``t1`` is the first trading day on/after it. This gap is
    the **surprise proxy** the believer would have observed by the next session; the drift is
    measured strictly *after* it (see ``strategy.drift_returns``).

    Returns one row per usable event with: ticker, date, t1_idx (integer position of the
    reaction session in that ticker's price series), gap (reaction return), surprise_pct.
    """
    out = []
    for tk, grp in events.groupby("ticker"):
        if tk not in prices.columns:
            continue
        px = prices[tk].dropna()
        if len(px) < 70:
            continue
        idx = px.index
        vals = px.values
        for _, r in grp.iterrows():
            d = pd.Timestamp(r["date"]).normalize()
            # first trading day on/after the announcement date
            pos1 = idx.searchsorted(d, side="left")
            if pos1 <= 0 or pos1 >= len(idx):
                continue
            pos0 = pos1 - 1
            gap = vals[pos1] / vals[pos0] - 1.0
            out.append({"ticker": tk, "date": idx[pos1], "t1_idx": int(pos1),
                        "gap": float(gap), "surprise_pct": float(r["surprise_pct"])})
    et = pd.DataFrame(out)
    return et.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_real() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: cached prices + events -> (prices, event table with gap) in one call."""
    prices = load_prices()
    events = load_events()
    return prices, build_event_table(prices, events)


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_pead(n_names: int = 30, n_quarters: int = 60, edge: float = 0.0,
                   seed: int = 363, sig_daily: float = 0.014,
                   gap_sd: float = 0.045) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic price panel + earnings events with a PLANTED drift knob.

    Each name has a daily random-walk price. Quarterly earnings dates (~63 trading days apart)
    each get a **surprise** drawn N(0, ``gap_sd``); the announcement-day reaction equals that
    surprise. If ``edge`` != 0, the 60 sessions following each event get an *extra* daily drift
    of ``edge * sign(surprise) / 60`` — a post-earnings drift that continues the reaction, the
    exact PEAD pattern. With ``edge = 0`` the post-event path is pure noise: the long-short
    drift must NOT reach significance however the noise falls (the whole small-effect lesson on
    data where we know the truth).

    Returns (prices wide frame, event table) in the same shape as ``load_real``.
    """
    rng = np.random.default_rng(seed)
    n_days = n_quarters * 63 + 200
    # decorative business-day index (only a label; positions carry the logic)
    idx = pd.bdate_range("2005-01-03", periods=n_days)
    names = [f"N{i:02d}" for i in range(n_names)]

    price_cols = {}
    ev_rows = []
    H = 60
    for name in names:
        ret = rng.normal(0.0003, sig_daily, size=n_days)
        # quarterly events, jittered, leaving room for the 60-day drift window
        first = int(rng.integers(80, 140))
        q = first
        while q < n_days - H - 5:
            surprise = rng.normal(0.0, gap_sd)
            # announcement-day reaction = the surprise (one-day gap)
            ret[q] += surprise
            if edge != 0.0:
                ret[q + 1:q + 1 + H] += edge * np.sign(surprise) / H
            ev_rows.append({"ticker": name, "_pos": q, "gap": float(surprise),
                            "surprise_pct": float(surprise * 100.0)})
            q += 63 + int(rng.integers(-4, 5))
        price_cols[name] = 100.0 * np.exp(np.cumsum(ret))

    prices = pd.DataFrame(price_cols, index=idx)
    ev = pd.DataFrame(ev_rows)
    ev["date"] = [idx[p] for p in ev["_pos"]]
    ev["t1_idx"] = ev["_pos"].astype(int)
    ev = ev[["ticker", "date", "t1_idx", "gap", "surprise_pct"]]
    ev = ev.sort_values(["ticker", "date"]).reset_index(drop=True)
    return prices, ev
