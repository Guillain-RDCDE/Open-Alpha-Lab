"""Data layer for Study 789 — SUE Earnings-Surprise Drift.

Two sources, both offline-friendly once cached:

* **Real tape.**
  - Daily adjusted closes for a fixed basket of long-listed US large-caps (yfinance, no key).
  - Per name, the full history of **quarterly diluted EPS** from EDGAR's XBRL
    ``companyconcept`` API (``data.sec.gov``), restricted to the *frame*-tagged calendar
    quarters (clean, de-duplicated Q1..Q4 figures). For each quarter we keep its EPS, the
    period end, and the **filing date** of the 10-Q/10-K that disclosed it — the date the
    number became public.
  From these we build a per-name quarterly EPS series and compute the **standardized
  unexpected earnings (SUE)**: the seasonal-random-walk surprise (EPS minus the same quarter
  a year ago) standardized by the rolling volatility of the last ~8 of those seasonal
  differences — the classic Foster-Olsen-Shevlin (1984) / Bernard-Thomas (1989) SUE.
  An **event** is one (ticker, filing date) row carrying the SUE; the drift is measured
  strictly *after* the filing is public. Everything is cached under this study's ``_cache/``.

  If the repo-level ``_cache/edgar_eps.parquet`` (from ``tools/fetch_altdata.py``) is present
  it is used as the EPS source; otherwise EPS is fetched per-CIK directly from EDGAR (the same
  frame-tagged quarterly diluted-EPS figures). Either way the offline study cache is the same.

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
PRICES_CACHE = os.path.join(CACHE_DIR, "sue_prices.csv")
EVENTS_CACHE = os.path.join(CACHE_DIR, "sue_events.csv")
# The desk-wide EDGAR EPS parquet (tools/fetch_altdata.py "EDGAR"), if it has been built.
REPO_EDGAR_EPS = os.path.abspath(os.path.join(HERE, "..", "..", "..", "_cache", "edgar_eps.parquet"))

AS_OF = "2026-06-30"        # last complete month at publication (partial current month dropped)

UA = "open-alpha-lab research guillain@poulpe.us"

# A transparent, fixed basket of large, long-listed US large-caps with deep, clean EPS
# histories on EDGAR (the same basket as the revenue sibling 534, for a like-for-like
# comparison). This is a *survivors* basket (all still trading) — survivorship is named on the
# Signal axis: a fixed surviving-names basket cannot include firms that blew up, a mild upward
# tilt on the long leg, and it can point *against* a drift as easily as for it.
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

# Diluted preferred; basic is the fallback for the rare firm that tags only basic EPS.
EPS_CONCEPTS = ("EarningsPerShareDiluted", "EarningsPerShareBasic")

MIN_HISTORY = 8   # require ~8 prior seasonal differences before emitting a SUE (the brief's ~8)
VOL_WINDOW = 8    # rolling std of the last ~8 seasonal-difference surprises


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


def _quarterly_eps(cik: str) -> pd.DataFrame:
    """Frame-tagged calendar-quarter diluted EPS for one CIK.

    Columns: end (period end), filed (filing/disclosure date), val (EPS). We restrict to
    *frame*-tagged quarters (``CYxxxxQn``), which EDGAR pre-de-duplicates to one clean value
    per calendar quarter, and keep the **filing date** so the drift can be measured strictly
    after the number is public (no look-ahead). Diluted EPS is preferred; a firm that tags only
    basic EPS falls back to it.
    """
    rows = []
    for concept in EPS_CONCEPTS:
        d = _edgar_concept(cik, concept)
        if d is None:
            continue
        units = d.get("units", {}).get("USD/shares", [])
        for u in units:
            fr = u.get("frame", "")
            # quarter frames look like CY2019Q1 (no I suffix => duration of one quarter)
            if not fr or "Q" not in fr or fr.endswith("I"):
                continue
            if u.get("form") not in ("10-Q", "10-K", "10-K/A", "10-Q/A"):
                continue
            if u.get("val") is None:
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


def _eps_from_repo_parquet() -> dict[str, pd.DataFrame] | None:
    """Per-ticker quarterly EPS frames from the desk-wide ``_cache/edgar_eps.parquet``.

    That parquet (tools/fetch_altdata.py) carries columns ticker, period_end, filed, eps for
    ~quarterly spans. Returns {ticker: DataFrame[end, filed, val]} or None if absent.
    """
    if not os.path.exists(REPO_EDGAR_EPS):
        return None
    raw = pd.read_parquet(REPO_EDGAR_EPS)
    raw = raw.rename(columns={"period_end": "end", "eps": "val"})
    raw["end"] = pd.to_datetime(raw["end"])
    raw["filed"] = pd.to_datetime(raw["filed"])
    out = {}
    for tk, g in raw.groupby("ticker"):
        g = g.drop_duplicates(subset=["end"], keep="last").sort_values("end")
        out[tk] = g[["end", "filed", "val"]].reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# SUE (standardized unexpected earnings)
# --------------------------------------------------------------------------- #
def compute_sue(eps: pd.DataFrame, min_history: int = MIN_HISTORY,
                vol_window: int = VOL_WINDOW) -> pd.DataFrame:
    """Standardized unexpected earnings per quarter (seasonal random walk + rolling-vol scaling).

    For a clean quarterly EPS series ``E_q`` the **seasonal** surprise is
    ``u_q = E_q - E_{q-4}`` (EPS this quarter minus the same quarter last year — the standard
    de-seasonalisation for earnings). SUE standardizes it by the rolling volatility of the last
    ``vol_window`` of those seasonal differences *known strictly before q*:

        SUE_q = (E_q - E_{q-4}) / std({u_{q-1}, ..., u_{q-vol_window}})

    This is the classic Foster-Olsen-Shevlin (1984) / Bernard-Thomas (1989) SUE. We require at
    least ``min_history`` prior seasonal differences before emitting a SUE (so the scaling is
    stable). Everything in the denominator is lagged — no look-ahead. Returns the input frame
    with a ``sue`` column; rows without enough history are dropped.
    """
    df = eps.sort_values("end").reset_index(drop=True).copy()
    v = df["val"].values
    n = len(v)
    seas = np.full(n, np.nan)
    for i in range(4, n):
        seas[i] = v[i] - v[i - 4]
    sue = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(seas[i]):
            continue
        past = seas[:i]                              # strictly prior seasonal diffs (lagged)
        past = past[~np.isnan(past)]
        if len(past) < min_history:
            continue
        window = past[-vol_window:]                  # the last ~vol_window surprises
        sd = window.std(ddof=1)
        if sd > 0:
            sue[i] = seas[i] / sd
    df["sue"] = sue
    return df.dropna(subset=["sue"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = "2005-01-01", end: str | None = None,
                prices_path: str = PRICES_CACHE, events_path: str = EVENTS_CACHE
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download basket prices + quarterly diluted EPS, compute SUE, and cache.

    Network-only; used once to build the cache. Writes a wide adjusted-close CSV and a long
    event CSV with one row per (ticker, filing date) carrying the SUE. EPS is sourced from the
    desk-wide ``_cache/edgar_eps.parquet`` if present, else fetched per-CIK from EDGAR. Never
    imported by the offline notebook cells.
    """
    import yfinance as yf

    raw = yf.download(BASKET, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    keep = [c for c in raw.columns if raw[c].notna().mean() >= 0.60]
    prices = raw[keep].copy()
    os.makedirs(os.path.dirname(prices_path), exist_ok=True)
    prices.to_csv(prices_path)

    repo_eps = _eps_from_repo_parquet()
    rows = []
    for tk in keep:
        if repo_eps is not None and tk in repo_eps:
            eps = repo_eps[tk]
        else:
            cik = CIK.get(tk)
            if cik is None:
                continue
            eps = _quarterly_eps(cik)
            time.sleep(0.15)
        if len(eps) < 12:
            continue
        sue = compute_sue(eps)
        for _, r in sue.iterrows():
            rows.append({"ticker": tk, "filed": r["filed"], "period_end": r["end"],
                         "eps": float(r["val"]), "sue": float(r["sue"])})
    events = pd.DataFrame(rows).dropna(subset=["filed", "sue"])
    events = events.sort_values(["ticker", "filed"]).reset_index(drop=True)
    events.to_csv(events_path, index=False)
    return prices, events


def have_real(prices_path: str = PRICES_CACHE, events_path: str = EVENTS_CACHE) -> bool:
    return os.path.exists(prices_path) and os.path.exists(events_path)


def load_prices(path: str = PRICES_CACHE) -> pd.DataFrame:
    """Wide adjusted-close frame (index = date, columns = tickers)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def load_events(path: str = EVENTS_CACHE) -> pd.DataFrame:
    """Long event frame: columns ticker, filed, period_end, eps, sue."""
    ev = pd.read_csv(path, parse_dates=["filed", "period_end"])
    return ev.sort_values(["ticker", "filed"]).reset_index(drop=True)


def build_event_table(prices: pd.DataFrame, events: pd.DataFrame,
                      asof: str | None = AS_OF) -> pd.DataFrame:
    """Attach the price-series position of the **first session after the filing** to each event.

    For an EPS figure disclosed (filed) on calendar day ``F``, the first trading day that can
    price the news is the session at or after ``F``. We anchor the event at that session
    (``t1_idx``); the strategy enters one day later (no look-ahead) and measures forward drift.
    Events filed after ``asof`` are dropped so the pinned run never creeps as the tape extends.

    Returns one row per usable event with: ticker, filed (the anchored session), t1_idx, sue
    (and eps/period_end carried through for reference).
    """
    out = []
    cut = None if asof is None else pd.Timestamp(asof)
    for tk, grp in events.groupby("ticker"):
        if tk not in prices.columns:
            continue
        px = prices[tk].dropna()
        if cut is not None:
            px = px[px.index <= cut]
        if len(px) < 70:
            continue
        idx = px.index
        for _, r in grp.iterrows():
            f = pd.Timestamp(r["filed"]).normalize()
            if cut is not None and f > cut:
                continue
            pos1 = idx.searchsorted(f, side="left")
            if pos1 <= 0 or pos1 >= len(idx):
                continue
            out.append({"ticker": tk, "filed": idx[pos1], "t1_idx": int(pos1),
                        "sue": float(r["sue"]),
                        "eps": float(r.get("eps", np.nan))})
    et = pd.DataFrame(out)
    if et.empty:
        return et
    return et.sort_values(["ticker", "filed"]).reset_index(drop=True)


def load_real(asof: str | None = AS_OF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: cached prices + events -> (prices, event table with t1_idx) in one call."""
    prices = load_prices()
    events = load_events()
    return prices, build_event_table(prices, events, asof=asof)


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_sue(n_names: int = 30, n_quarters: int = 44, edge: float = 0.0,
                  seed: int = 789, sig_daily: float = 0.014,
                  sue_sd: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic price panel + SUE events with a PLANTED drift knob.

    Each name has a daily random-walk price. Quarterly filing dates (~63 trading days apart)
    each get a **SUE** drawn N(0, ``sue_sd``). If ``edge`` != 0, the 63 sessions following each
    filing get an *extra* daily drift of ``edge * sign(SUE) / 63`` — a post-earnings drift that
    continues in the direction of the surprise, the exact Bernard-Thomas pattern. With
    ``edge = 0`` the post-event path is pure noise: the long-short must NOT reach significance
    however the noise falls (the whole small-effect lesson on data where we know the truth).

    Returns (prices wide frame, event table) in the same shape as ``load_real``.
    """
    rng = np.random.default_rng(seed)
    H = 63
    n_days = n_quarters * 63 + 200
    idx = pd.bdate_range("2009-01-02", periods=n_days)
    names = [f"N{i:02d}" for i in range(n_names)]

    price_cols = {}
    ev_rows = []
    for name in names:
        ret = rng.normal(0.0003, sig_daily, size=n_days)
        first = int(rng.integers(80, 140))
        q = first
        while q < n_days - H - 5:
            sue = rng.normal(0.0, sue_sd)
            if edge != 0.0:
                ret[q + 1:q + 1 + H] += edge * np.sign(sue) / H
            ev_rows.append({"ticker": name, "_pos": q, "sue": float(sue)})
            q += 63 + int(rng.integers(-4, 5))
        price_cols[name] = 100.0 * np.exp(np.cumsum(ret))

    prices = pd.DataFrame(price_cols, index=idx)
    ev = pd.DataFrame(ev_rows)
    ev["filed"] = [idx[p] for p in ev["_pos"]]
    ev["t1_idx"] = ev["_pos"].astype(int)
    ev["eps"] = np.nan
    ev = ev[["ticker", "filed", "t1_idx", "sue", "eps"]]
    return prices, ev.sort_values(["ticker", "filed"]).reset_index(drop=True)
