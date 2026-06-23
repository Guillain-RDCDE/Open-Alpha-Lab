"""Data layer for Study 362 — Piotroski F-Score (fundamentals panel + forward returns).

Two tapes, one shape — a **firm × fiscal-year panel** of the raw fundamentals the
F-score needs, plus a **firm × year forward-return panel** (the calendar year *after*
the fiscal year, a conservative reporting lag so the 10-K is never used before it is
filed):

* **Real tape.** Annual figures pulled once from **EDGAR companyfacts**
  (``data.sec.gov``, public, no key) for a fixed basket of large/mid-cap US firms, and
  daily adjusted closes from **yfinance** rolled up to calendar-year total returns.
  Cached under ``_cache/`` as ``edgar_fundamentals.parquet`` and ``yearly_returns.parquet``.
  The fetchers (``fetch_fundamentals`` / ``fetch_returns``) are network-only, called once
  to build the cache, and are **never imported by the offline notebook cells**.

* **Synthetic.** A deterministic, fixed-seed generator that builds a fundamentals panel
  whose firms carry a *latent quality* score; a ``planted_edge`` knob makes high-F-score
  firms out-earn low-F-score firms by a known annual amount next year. ``planted_edge = 0``
  is the null — the F-score must NOT manufacture a significant long-minus-short spread out
  of noise; a large planted edge MUST light up. This is the study's positive control.

**Survivorship caveat (named on the Signal axis).** The real basket is a fixed set of
firms that *survived* to 2026, pulled on current tickers. A surviving-names panel is
breadth-of-the-survivors and tilts results mildly *bullish*; real-tape numbers are upper
bounds, said so in ``docs/results.md`` and on every place the stamp appears.

Pure numpy + pandas + stdlib on the offline path.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
FUND_CACHE = os.path.join(CACHE_DIR, "edgar_fundamentals.parquet")
RET_CACHE = os.path.join(CACHE_DIR, "yearly_returns.parquet")

# A fixed basket of large/mid-cap US firms with long EDGAR + price history and a sector
# spread. This is a SURVIVING-names basket (named on the Signal axis): every firm lived to
# 2026, a mild upward tilt. CIKs are the EDGAR identifiers (zero-padded to 10 digits below).
BASKET = {
    "AAPL": 320193, "MSFT": 789019, "XOM": 34088, "JNJ": 200406, "PG": 80424,
    "KO": 21344, "JPM": 19617, "WMT": 104169, "IBM": 51143, "CVX": 93410,
    "PFE": 78003, "MRK": 310158, "INTC": 50863, "CSCO": 858877, "HD": 354950,
    "MCD": 63908, "DIS": 1744489, "BA": 12927, "CAT": 18230, "MMM": 66740,
    "HON": 773840, "UNH": 731766, "WFC": 72971, "ORCL": 1341439, "PEP": 77476,
    "ABT": 1800, "TXN": 97476, "COST": 909832, "LOW": 60667, "AMGN": 318154,
    "DE": 315189, "DUK": 1326160, "TGT": 27419, "EMR": 32604, "GD": 40533,
    "ADP": 8670, "LMT": 936468, "NKE": 320187, "MDT": 64670, "SO": 92122,
}

# The eight raw EDGAR concepts we need to assemble the 9 F-score points. (GrossProfit is
# computed from Revenues - CostOfRevenue when the GrossProfit tag is missing.)
CONCEPTS = (
    "NetIncomeLoss",
    "Assets",
    "NetCashProvidedByUsedInOperatingActivities",
    "LiabilitiesCurrent",
    "AssetsCurrent",
    "LongTermDebtNoncurrent",
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
    "GrossProfit",
    "CommonStockSharesOutstanding",
)

_UA = {"User-Agent": "Open-Alpha-Lab research (guillain@poulpe.us)"}


# --------------------------------------------------------------------------- #
# Real tape — EDGAR companyfacts + yfinance (network, run once to build cache)
# --------------------------------------------------------------------------- #
def _annual_series(facts: dict, concept: str) -> dict[int, float]:
    """Pull the annual (FY) value of one us-gaap concept from a companyfacts blob.

    Returns ``{fiscal_year: value}`` using full-year (``fp == 'FY'``) 10-K figures; for
    flow concepts we take the latest-filed FY value per fiscal year, for instant concepts
    the fiscal-year-end value. Missing concept -> empty dict.
    """
    try:
        units = facts["facts"]["us-gaap"][concept]["units"]
    except KeyError:
        return {}
    # pick the USD (or shares) unit list
    key = "USD" if "USD" in units else ("shares" if "shares" in units else None)
    if key is None:
        return {}
    rows = units[key]
    out: dict[int, tuple[str, float]] = {}  # fy -> (filed, val)
    for r in rows:
        if r.get("form") not in ("10-K", "10-K/A"):
            continue
        if r.get("fp") != "FY":
            continue
        fy = r.get("fy")
        if fy is None:
            continue
        filed = r.get("filed", "")
        val = r.get("val")
        if val is None:
            continue
        # keep the latest-filed figure for each fiscal year
        if fy not in out or filed > out[fy][0]:
            out[fy] = (filed, float(val))
    return {fy: v for fy, (f, v) in out.items()}


def fetch_fundamentals(path: str = FUND_CACHE, pause: float = 0.25) -> pd.DataFrame:
    """Download the annual fundamentals panel from EDGAR companyfacts and cache it.

    Network-only. Builds a long DataFrame with columns ``[ticker, year, concept, value]``
    (one row per ticker × fiscal-year × concept), cached as a parquet. Run once.
    """
    records = []
    for tkr, cik in BASKET.items():
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
        req = urllib.request.Request(url, headers=_UA)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                facts = json.load(resp)
        except Exception as e:  # pragma: no cover - network
            print(f"  ! {tkr}: {type(e).__name__} {e}")
            time.sleep(pause)
            continue
        for c in CONCEPTS:
            for fy, val in _annual_series(facts, c).items():
                records.append((tkr, int(fy), c, float(val)))
        print(f"  {tkr}: {sum(1 for r in records if r[0]==tkr)} concept-years")
        time.sleep(pause)
    df = pd.DataFrame(records, columns=["ticker", "year", "concept", "value"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)
    return df


def fetch_returns(path: str = RET_CACHE, start: str = "1998-01-01",
                  end: str | None = None) -> pd.DataFrame:
    """Download daily adjusted closes (yfinance) and roll up to calendar-year total returns.

    Network-only. Cached as a (year × ticker) DataFrame of calendar-year returns. Run once.
    """
    import yfinance as yf

    tickers = list(BASKET.keys())
    px = yf.download(tickers, start=start, end=end, auto_adjust=True,
                     progress=False)["Close"].sort_index()
    px = px.dropna(how="all")
    # calendar-year total return = last/first - 1 within each year
    yr = px.groupby(px.index.year).apply(lambda g: g.iloc[-1] / g.iloc[0] - 1.0)
    yr.index.name = "year"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    yr.to_parquet(path)
    return yr


def have_real(fund_path: str = FUND_CACHE, ret_path: str = RET_CACHE) -> bool:
    return os.path.exists(fund_path) and os.path.exists(ret_path)


def _pivot_concept(long_df: pd.DataFrame, concept: str) -> pd.DataFrame:
    sub = long_df[long_df["concept"] == concept]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="year", columns="ticker", values="value",
                           aggfunc="last").sort_index()


def load_real(fund_path: str = FUND_CACHE, ret_path: str = RET_CACHE) -> dict:
    """Load the cached real tape and assemble the panels the strategy consumes.

    Returns a dict of (year × ticker) frames keyed by concept name plus ``"GrossProfit"``
    (synthesised from Revenues - CostOfRevenue where the tag is absent) and ``"returns"``
    (calendar-year total returns).
    """
    long_df = pd.read_parquet(fund_path)
    panels = {c: _pivot_concept(long_df, c) for c in CONCEPTS}

    def _coalesce(*names):
        out = pd.DataFrame()
        for nm in names:
            f = panels.get(nm, pd.DataFrame())
            if f is not None and not f.empty:
                out = f if out.empty else out.combine_first(f)
        return out

    # canonical revenue / COGS from the primary tag + common alternates
    rev = _coalesce("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                    "SalesRevenueNet")
    cogs = _coalesce("CostOfRevenue", "CostOfGoodsAndServicesSold")
    panels["Revenues"] = rev
    panels["CostOfRevenue"] = cogs

    # synthesise gross profit when the explicit tag is sparse
    gp = panels.get("GrossProfit", pd.DataFrame())
    if not rev.empty and not cogs.empty:
        derived = rev.subtract(cogs)
        gp = gp.combine_first(derived) if not gp.empty else derived
    panels["GrossProfit"] = gp
    panels["returns"] = pd.read_parquet(ret_path).sort_index()
    return panels


# --------------------------------------------------------------------------- #
# Synthetic positive control — deterministic, no network
# --------------------------------------------------------------------------- #
def synthetic_panel(n_firms: int = 200, n_years: int = 18, planted_edge: float = 0.0,
                    base_ret: float = 0.10, ret_vol: float = 0.28,
                    seed: int = 362) -> dict:
    """Deterministic fundamentals panel with a *known* F-score → forward-return edge.

    Each firm-year draws a latent **quality** ``z`` (standard normal). Nine binary
    fundamental signals are generated so that high-quality firms tend to score the F-score
    points (a noisy monotone map from ``z`` to each point), reproducing a real F-score
    spread of roughly 0–9. Next calendar year's return is::

        base_ret + planted_edge * z(F_score)  +  noise

    where ``z(F_score)`` is the cross-sectional z-score of the F-score that year. With
    ``planted_edge = 0`` the F-score carries no information about returns and the
    long-minus-short spread must be indistinguishable from zero — the null in a bottle.
    A large ``planted_edge`` must drive the spread t-stat above 2.

    Returns a dict with ``"fscore"`` (year × firm F-score 0..9), ``"returns"``
    (year × firm next-year return) and ``"z"`` (latent quality, for diagnostics).
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2007, 2007 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    z = rng.standard_normal((n_years, n_firms))            # latent quality
    # nine binary points: each lights with probability rising in z (logistic), plus noise
    points = np.zeros((n_years, n_firms))
    for k in range(9):
        # slightly different sensitivity per point so the F-score isn't a step function
        slope = 0.7 + 0.15 * k / 8.0
        p = 1.0 / (1.0 + np.exp(-(slope * z + rng.normal(0, 0.4, z.shape))))
        points += (rng.random(z.shape) < p).astype(float)
    fscore = points  # integer 0..9 by construction

    # cross-sectional z-score of the F-score each year
    mu = fscore.mean(axis=1, keepdims=True)
    sd = fscore.std(axis=1, keepdims=True) + 1e-9
    fz = (fscore - mu) / sd

    fwd = base_ret + planted_edge * fz + ret_vol * rng.standard_normal((n_years, n_firms))

    # The F-score for fiscal year y predicts the return realised in calendar year y+1,
    # matching the real-tape reporting lag: ``strategy.long_short`` looks up ``fwd.loc[y+1]``.
    sig_idx = pd.Index(years, name="year")
    ret_idx = pd.Index(years + 1, name="year")
    return {
        "fscore": pd.DataFrame(fscore, index=sig_idx, columns=firms),
        "returns": pd.DataFrame(fwd, index=ret_idx, columns=firms),
        "z": pd.DataFrame(z, index=sig_idx, columns=firms),
        "planted_edge": planted_edge,
    }


# --------------------------------------------------------------------------- #
# Fingerprint helper
# --------------------------------------------------------------------------- #
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a DataFrame for the as-of stamp."""
    arr = np.ascontiguousarray(df.to_numpy(dtype=float, na_value=0.0))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
