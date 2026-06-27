"""Data layer for Study 534 — Revenue-Surprise-Drift (Jegadeesh & Livnat 2006).

Two sources, both offline-friendly once cached:

* **Real tape.**
  - Daily adjusted closes for a fixed basket of long-listed US large-caps (yfinance, no key).
  - Per name, the full history of **quarterly revenue** from EDGAR's XBRL ``companyconcept``
    API (``data.sec.gov``), restricted to the *frame*-tagged calendar quarters (clean,
    de-duplicated Q1..Q4 figures). For each quarter we keep its revenue, the period end, and
    the **filing date** of the 10-Q/10-K that disclosed it — the date the number became public.
  From these we build a per-name quarterly revenue series and compute the **standardized
  unexpected revenue (SUR)**: the seasonal random-walk surprise (revenue minus the same quarter
  a year ago) standardized by the trailing volatility of those seasonal differences — the
  revenue analogue of academic SUE (Jegadeesh-Livnat 2006).
  An **event** is one (ticker, filing date) row carrying the SUR; the drift is measured strictly
  *after* the filing is public. Everything is cached under ``_cache/``.

* **Synthetic.** A deterministic, fixed-seed generator that produces events whose post-event
  path carries a **planted drift** proportional to the surprise (knob ``edge``). It is the
  positive control: with ``edge = 0`` the long-short must NOT manufacture significance; with a
  large planted ``edge`` it must light up. No network.

Pure numpy + pandas + stdlib for the offline path. ``fetch_panel`` (network) is used once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "rev_prices.csv")
EVENTS_CACHE = os.path.join(CACHE_DIR, "rev_events.csv")

UA = "open-alpha-lab research guillain@poulpe.us"

# A transparent, fixed basket of large, long-listed US large-caps with deep, clean revenue
# histories on EDGAR. Chosen for long price history + sector spread. This is a *survivors*
# basket (all still trading) — survivorship is named on the Signal axis: a fixed
# surviving-names basket cannot capture firms that blew up, a mild upward tilt on the long leg.
BASKET = [
    "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "IBM", "CVX",
    "PFE", "MRK", "INTC", "CSCO", "HD", "MCD", "DIS", "BA", "CAT", "MMM",
    "HON", "UNH", "ORCL", "PEP", "ABT", "TXN", "COST", "LOW", "AMGN", "GS",
]

# SEC CIK (10-digit, zero-padded) for each basket name (resolved once from the SEC ticker map).
CIK = {
    "AAPL": "0000320193", "MSFT": "0000789019", "XOM": "0000034088", "JNJ": "0000200406",
    "PG": "0000080424", "KO": "0000021344", "JPM": "0000019617", "WMT": "0000104169",
    "IBM": "0000051143", "CVX": "0000093410", "PFE": "0000078003", "MRK": "0000310158",
    "INTC": "0000050863", "CSCO": "0000858877", "HD": "0000354950", "MCD": "0000063908",
    "DIS": "0001744489", "BA": "0000012927", "CAT": "0000018230", "MMM": "0000066740",
    "HON": "0000773840", "UNH": "0000731766", "ORCL": "0001341439", "PEP": "0000077476",
    "ABT": "0000001800", "TXN": "0000097476", "COST": "0000909832", "LOW": "0000060667",
    "AMGN": "0000318154", "GS": "0000886982",
}

# Revenue concepts in order of preference (firms tag revenue under different us-gaap concepts).
REV_CONCEPTS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)


# --------------------------------------------------------------------------- #
# EDGAR helpers (network; cache builders only)
# --------------------------------------------------------------------------- #
def _edgar_concept(cik: str, concept: str, retries: int = 3) -> dict | None:
    url = (f"https://data.sec.gov/api/xbrl/companyconcept/"
           f"CIK{cik}/us-gaap/{concept}.json")
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:
            time.sleep(0.6 * (k + 1))
    return None


def _quarterly_revenue(cik: str) -> pd.DataFrame:
    """Return frame-tagged calendar-quarter revenue for one CIK.

    Columns: end (period end), filed (filing/disclosure date), val (USD revenue). We restrict to
    *frame*-tagged quarters (``CYxxxxQn``), which EDGAR pre-de-duplicates to one clean value per
    calendar quarter, and keep the **filing date** so the drift can be measured strictly after
    the number is public (no look-ahead).
    """
    rows = []
    for concept in REV_CONCEPTS:
        d = _edgar_concept(cik, concept)
        if d is None:
            continue
        units = d.get("units", {}).get("USD", [])
        for u in units:
            fr = u.get("frame", "")
            # quarter frames look like CY2019Q1 (no I suffix => duration of one quarter)
            if not fr or "Q" not in fr or fr.endswith("I"):
                continue
            if u.get("form") not in ("10-Q", "10-K", "10-K/A", "10-Q/A"):
                continue
            rows.append({"frame": fr, "end": u.get("end"), "filed": u.get("filed"),
                         "val": float(u.get("val"))})
        if rows:
            break  # first concept that yields framed quarters wins
    if not rows:
        return pd.DataFrame(columns=["end", "filed", "val"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["frame"], keep="first")
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    return df.sort_values("end").reset_index(drop=True)[["end", "filed", "val"]]


# --------------------------------------------------------------------------- #
# SUR (standardized unexpected revenue)
# --------------------------------------------------------------------------- #
def compute_sur(rev: pd.DataFrame, min_history: int = 8) -> pd.DataFrame:
    """Standardized unexpected revenue per quarter (seasonal random walk + own-vol scaling).

    For a clean quarterly revenue series ``R_q`` the **seasonal** surprise is
    ``u_q = R_q - R_{q-4}`` (revenue this quarter minus the same quarter last year — the
    standard de-seasonalisation for sales). SUR standardizes it by the trailing volatility of
    those seasonal differences *known at q*:

        SUR_q = (R_q - R_{q-4}) / std({u_{q-1}, ..., u_{q-k}})   (expanding, lagged)

    This is the revenue analogue of academic SUE (Jegadeesh & Livnat 2006). We require at least
    ``min_history`` prior seasonal differences before emitting a SUR (so the scaling is stable).
    Returns the input frame with a ``sur`` column; rows without enough history are dropped.
    """
    df = rev.sort_values("end").reset_index(drop=True).copy()
    v = df["val"].values
    n = len(v)
    seas = np.full(n, np.nan)
    for i in range(4, n):
        seas[i] = v[i] - v[i - 4]
    sur = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(seas[i]):
            continue
        past = seas[max(0, i - 20):i]           # strictly prior seasonal diffs (lagged)
        past = past[~np.isnan(past)]
        if len(past) < min_history:
            continue
        sd = past.std(ddof=1)
        if sd > 0:
            sur[i] = seas[i] / sd
    df["sur"] = sur
    return df.dropna(subset=["sur"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = "2005-01-01", end: str | None = None,
                prices_path: str = PRICES_CACHE, events_path: str = EVENTS_CACHE
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download basket prices + EDGAR quarterly revenue, compute SUR, and cache.

    Network-only; used once to build the cache. Writes a wide adjusted-close CSV and a long
    event CSV with one row per (ticker, filing date) carrying the SUR. Never imported by the
    offline notebook cells.
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
        cik = CIK.get(tk)
        if cik is None:
            continue
        rev = _quarterly_revenue(cik)
        if len(rev) < 12:
            continue
        sur = compute_sur(rev)
        for _, r in sur.iterrows():
            rows.append({"ticker": tk, "filed": r["filed"], "period_end": r["end"],
                         "revenue": float(r["val"]), "sur": float(r["sur"])})
        time.sleep(0.15)
    events = pd.DataFrame(rows).dropna(subset=["filed", "sur"])
    events = events.sort_values(["ticker", "filed"]).reset_index(drop=True)
    events.to_csv(events_path, index=False)
    return prices, events


def have_real(prices_path: str = PRICES_CACHE, events_path: str = EVENTS_CACHE) -> bool:
    return os.path.exists(prices_path) and os.path.exists(events_path)


def load_prices(path: str = PRICES_CACHE) -> pd.DataFrame:
    """Wide adjusted-close frame (index = date, columns = tickers)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def load_events(path: str = EVENTS_CACHE) -> pd.DataFrame:
    """Long event frame: columns ticker, filed, period_end, revenue, sur."""
    ev = pd.read_csv(path, parse_dates=["filed", "period_end"])
    return ev.sort_values(["ticker", "filed"]).reset_index(drop=True)


def build_event_table(prices: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Attach the price-series position of the **first session after the filing** to each event.

    For a revenue figure disclosed (filed) on calendar day ``F``, the first trading day that can
    price the news is the session at or after ``F``. We anchor the event at that session
    (``t1_idx``); the strategy enters one day later (no look-ahead) and measures forward drift.

    Returns one row per usable event with: ticker, filed, t1_idx, sur (and revenue/period_end
    carried through for reference).
    """
    out = []
    for tk, grp in events.groupby("ticker"):
        if tk not in prices.columns:
            continue
        px = prices[tk].dropna()
        if len(px) < 70:
            continue
        idx = px.index
        for _, r in grp.iterrows():
            f = pd.Timestamp(r["filed"]).normalize()
            pos1 = idx.searchsorted(f, side="left")
            if pos1 <= 0 or pos1 >= len(idx):
                continue
            out.append({"ticker": tk, "filed": idx[pos1], "t1_idx": int(pos1),
                        "sur": float(r["sur"]),
                        "revenue": float(r.get("revenue", np.nan))})
    et = pd.DataFrame(out)
    return et.sort_values(["ticker", "filed"]).reset_index(drop=True)


def load_real() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: cached prices + events -> (prices, event table with t1_idx) in one call."""
    prices = load_prices()
    events = load_events()
    return prices, build_event_table(prices, events)


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_rev(n_names: int = 30, n_quarters: int = 40, edge: float = 0.0,
                  seed: int = 534, sig_daily: float = 0.014,
                  sur_sd: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic price panel + revenue-surprise events with a PLANTED drift knob.

    Each name has a daily random-walk price. Quarterly filing dates (~63 trading days apart)
    each get a **SUR** drawn N(0, ``sur_sd``). If ``edge`` != 0, the 60 sessions following each
    filing get an *extra* daily drift of ``edge * sign(SUR) / 60`` — a post-revenue drift that
    continues in the direction of the surprise, the exact Jegadeesh-Livnat pattern. With
    ``edge = 0`` the post-event path is pure noise: the long-short must NOT reach significance
    however the noise falls.

    Returns (prices wide frame, event table) in the same shape as ``load_real``.
    """
    rng = np.random.default_rng(seed)
    n_days = n_quarters * 63 + 200
    idx = pd.bdate_range("2009-01-02", periods=n_days)
    names = [f"N{i:02d}" for i in range(n_names)]

    price_cols = {}
    ev_rows = []
    H = 60
    for name in names:
        ret = rng.normal(0.0003, sig_daily, size=n_days)
        first = int(rng.integers(80, 140))
        q = first
        while q < n_days - H - 5:
            sur = rng.normal(0.0, sur_sd)
            if edge != 0.0:
                ret[q + 1:q + 1 + H] += edge * np.sign(sur) / H
            ev_rows.append({"ticker": name, "_pos": q, "sur": float(sur)})
            q += 63 + int(rng.integers(-4, 5))
        price_cols[name] = 100.0 * np.exp(np.cumsum(ret))

    prices = pd.DataFrame(price_cols, index=idx)
    ev = pd.DataFrame(ev_rows)
    ev["filed"] = [idx[p] for p in ev["_pos"]]
    ev["t1_idx"] = ev["_pos"].astype(int)
    ev["revenue"] = np.nan
    ev = ev[["ticker", "filed", "t1_idx", "sur", "revenue"]]
    return prices, ev.sort_values(["ticker", "filed"]).reset_index(drop=True)
