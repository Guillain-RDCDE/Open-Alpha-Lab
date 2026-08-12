"""Data layer for Study 856 — Book-Tax Differences.

The claim under test (Hanlon 2005, *The Accounting Review*): a **large positive book-minus-tax
income difference** — book (pretax accounting) income sitting far above the taxable income
*implied* by the firm's tax expense — is a red flag. It marks earnings that are **less
persistent** (more of them will not recur) and, if the market is slow to see it, predicts
**lower future returns**. The tradeable read-through: rank firms on the book-tax gap, go **long
the low-gap ("clean") names and short the high-gap ("aggressive") names**.

The book-tax difference (BTD), grossed up through the statutory rate::

    implied_taxable_income = IncomeTaxExpenseBenefit / statutory_rate
    BTD                    = PretaxIncome − implied_taxable_income
    btd_assets             = BTD / Assets            (the scaled, comparable signal)

A **large positive** BTD means book income >> the income the tax bill implies — the classic
Hanlon flag. We scale by total assets so a $1bn gap at a small firm and at a mega-cap are
comparable. The statutory rate is time-varying: **35 %** for fiscal years ending through 2017 and
**21 %** from 2018 on (the Tax Cuts and Jobs Act, effective for tax years beginning after
2017-12-31). This is a deliberate simplification — see docs/references.md on the 2018 blended-rate
year and on why we gross up by the *statutory*, not the *effective*, rate.

Two sources, both offline-friendly once cached:

* **Real tape.**
  - Daily adjusted closes for a fixed basket of ~42 large, deep-history US filers (yfinance, no
    key).
  - Per name, the full history of three us-gaap concepts from EDGAR's XBRL ``companyconcept`` API
    (``data.sec.gov``): annual **pretax income**
    (``IncomeLossFromContinuingOperationsBeforeIncomeTaxes...`` with fallbacks), annual **income
    tax expense** (``IncomeTaxExpenseBenefit``) and total **Assets** (the scaling denominator).
    For each fiscal year we keep the period end and the **filing date** of the 10-K that disclosed
    it — the date the number became public.
  From these we build, per name, one **annual** event carrying ``btd_assets`` (the level),
  ``d_btd_assets`` (the year-on-year change) and the pretax return-on-assets ``roa`` plus next
  year's ``roa_next`` (for the earnings-persistence mechanism test). Each event is one (ticker,
  filing date) row; the forward return / persistence outcome is measured strictly *after* the
  10-K is public (no look-ahead).

* **Synthetic.** A deterministic, fixed-seed generator that produces a price + signal panel in
  which forward returns carry a **planted** component proportional to the (negated) book-tax gap
  (knob ``edge``). It is the positive control: with ``edge = 0`` the long-short must NOT
  manufacture significance; with a large planted ``edge`` the low-BTD-minus-high-BTD spread must
  light up. No network.

Honest about coverage: this is a **thin, uneven, annual** panel of **current survivors**. Book-
tax differences bite hardest in the broad cross-section (small, distressed, tax-shelter-heavy
names) — a 42-name large-cap survivor basket is the *conservative* place to look, and any red-flag
effect it shows is an upper bound. Deep-history names (IBM, KO, PG, XOM…) start ~2009 in XBRL; a
handful (GOOGL post-split class, MDT re-domicile, XOM/ExxonMobil) are shorter. That is a
first-class caveat, not a footnote.

Pure numpy + pandas + stdlib for the offline path. ``fetch_panel`` (network) is used once to build
the cache and is never imported by the notebooks' offline cells.
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
PRICES_CACHE = os.path.join(CACHE_DIR, "btd_prices.csv")
EVENTS_CACHE = os.path.join(CACHE_DIR, "btd_events.csv")

UA = "open-alpha-lab research guillain@poulpe.us"

AS_OF = "2026-06-30"        # last complete calendar month at publication
ERA_SPLIT = "2018-01-01"    # pre/post the TCJA statutory-rate cut (35% -> 21%)

# US federal statutory corporate tax rate by fiscal-year-end. 35% through 2017; 21% from the Tax
# Cuts and Jobs Act (effective for tax years beginning after 2017-12-31). The 2018 blended-rate
# nuance for off-calendar fiscal years is deliberately ignored — a documented simplification.
STATUTORY_PRE_TCJA = 0.35
STATUTORY_POST_TCJA = 0.21


def statutory_rate(end: pd.Timestamp) -> float:
    """Statutory corporate rate applicable to a fiscal year ending at ``end``."""
    return STATUTORY_PRE_TCJA if pd.Timestamp(end) < pd.Timestamp(ERA_SPLIT) else STATUTORY_POST_TCJA


# A transparent, fixed basket of large, deep-history US C-corporations that pay meaningful federal
# tax across a spread of sectors (tech, pharma, staples, energy, industrials, telecom). This is a
# *survivors* basket (all still trading). Survivorship is named on the Signal axis: it cannot
# include firms whose aggressive tax positions blew up and were delisted, so any red-flag return
# effect it shows is an UPPER BOUND — see docs/references.md.
BASKET = [
    "AAPL", "MSFT", "GOOGL", "ORCL", "IBM", "INTC", "CSCO", "TXN", "QCOM", "HPQ",  # tech
    "JNJ", "PFE", "MRK", "ABT", "AMGN", "BMY", "MDT",                              # health
    "PG", "KO", "PEP", "WMT", "COST", "MCD", "HD", "NKE", "SBUX",                  # staples/cons
    "XOM", "CVX", "COP",                                                           # energy
    "CAT", "GE", "BA", "HON", "MMM", "DE", "EMR", "UPS", "LMT", "RTX",             # industrials
    "T", "VZ", "DIS",                                                              # telecom/media
]

# SEC CIK (10-digit, zero-padded), resolved once from the SEC ticker map and frozen here so the
# offline path never needs the network. (XOM = classic ExxonMobil CIK with deep XBRL history, not
# the post-2024 reorganisation shell.)
CIK = {
    "AAPL": "0000320193", "MSFT": "0000789019", "GOOGL": "0001652044", "ORCL": "0001341439",
    "IBM": "0000051143", "INTC": "0000050863", "CSCO": "0000858877", "TXN": "0000097476",
    "QCOM": "0000804328", "HPQ": "0000047217", "JNJ": "0000200406", "PFE": "0000078003",
    "MRK": "0000310158", "ABT": "0000001800", "AMGN": "0000318154", "BMY": "0000014272",
    "MDT": "0001613103", "PG": "0000080424", "KO": "0000021344", "PEP": "0000077476",
    "WMT": "0000104169", "COST": "0000909832", "MCD": "0000063908", "HD": "0000354950",
    "NKE": "0000320187", "SBUX": "0000829224", "XOM": "0000034088", "CVX": "0000093410",
    "COP": "0001163165", "CAT": "0000018230", "GE": "0000040545", "BA": "0000012927",
    "HON": "0000773840", "MMM": "0000066740", "DE": "0000315189", "EMR": "0000032604",
    "UPS": "0001090727", "LMT": "0000936468", "RTX": "0000101829", "T": "0000732717",
    "VZ": "0000732712", "DIS": "0001744489",
}

# Pretax income concept changed name over the years; try the modern long name first, then the
# older variants. Take the concept with the longest per-name history.
PRETAX_CONCEPTS = (
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesAndExtraordinaryItems",
)
TAX_CONCEPTS = ("IncomeTaxExpenseBenefit",)
ASSET_CONCEPTS = ("Assets",)


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
            time.sleep(0.5 * (k + 1))
    return None


def _instant_series(cik: str, concepts: tuple[str, ...]) -> pd.DataFrame:
    """Best (longest-history) instant balance-sheet series for one CIK across ``concepts``.

    Instant (point-in-time) us-gaap concepts report one value AT a period end. We keep every
    10-K observation, de-duplicate on the period end (earliest filing that disclosed it — no
    restatement look-ahead), and pick the concept that yields the most distinct year ends.
    Columns: end, filed, val.
    """
    best = pd.DataFrame(columns=["end", "filed", "val"])
    for concept in concepts:
        d = _edgar_concept(cik, concept)
        if d is None:
            continue
        rows = []
        for u in d.get("units", {}).get("USD", []):
            if u.get("form") not in ("10-K", "10-K/A"):
                continue
            if not (u.get("end") and u.get("filed") and u.get("val") is not None):
                continue
            rows.append({"end": u["end"], "filed": u["filed"], "val": float(u["val"])})
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["end"] = pd.to_datetime(df["end"])
        df["filed"] = pd.to_datetime(df["filed"])
        df = (df.sort_values("filed").drop_duplicates(subset=["end"], keep="first")
                .sort_values("end").reset_index(drop=True))
        if len(df) > len(best):
            best = df
    return best


def _annual_flow_series(cik: str, concepts: tuple[str, ...]) -> pd.DataFrame:
    """Best (longest-history) *annual* flow series (pretax income, tax expense) for one CIK.

    Flow (duration) concepts report a value over a start..end span. We keep only ~one-year spans
    (330-400 days) from 10-K filings, de-duplicate on the period end (earliest filing wins).
    Columns: end, filed, val.
    """
    best = pd.DataFrame(columns=["end", "filed", "val"])
    for concept in concepts:
        d = _edgar_concept(cik, concept)
        if d is None:
            continue
        rows = []
        for u in d.get("units", {}).get("USD", []):
            if u.get("form") not in ("10-K", "10-K/A"):
                continue
            if not (u.get("end") and u.get("start") and u.get("filed")
                    and u.get("val") is not None):
                continue
            span = (pd.Timestamp(u["end"]) - pd.Timestamp(u["start"])).days
            if not (330 <= span <= 400):
                continue
            rows.append({"end": u["end"], "filed": u["filed"], "val": float(u["val"])})
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["end"] = pd.to_datetime(df["end"])
        df["filed"] = pd.to_datetime(df["filed"])
        df = (df.sort_values("filed").drop_duplicates(subset=["end"], keep="first")
                .sort_values("end").reset_index(drop=True))
        if len(df) > len(best):
            best = df
    return best


# --------------------------------------------------------------------------- #
# Signal construction — annual book-tax difference (point-in-time)
# --------------------------------------------------------------------------- #
def _asof_value(series: pd.DataFrame, end: pd.Timestamp, tol: int = 20) -> float | None:
    """Value in ``series`` whose end matches ``end`` within ``tol`` days (same fiscal year)."""
    if series.empty:
        return None
    gap = (series["end"] - end).abs().dt.days
    if gap.min() > tol:
        return None
    return float(series.loc[gap.idxmin(), "val"])


def _prior_year_value(series: pd.DataFrame, end: pd.Timestamp,
                      lo: int = 300, hi: int = 430) -> float | None:
    """Value in ``series`` whose end is ~1 year before ``end`` (gap in [lo, hi] days)."""
    if series.empty:
        return None
    gap = (end - series["end"]).dt.days
    m = (gap >= lo) & (gap <= hi)
    if not m.any():
        return None
    sub = series.loc[m]
    k = (sub["end"].map(lambda e: abs((end - e).days - 365))).idxmin()
    return float(series.loc[k, "val"])


def build_signal(pretax: pd.DataFrame, tax: pd.DataFrame,
                 assets: pd.DataFrame) -> pd.DataFrame:
    """One row per fiscal year carrying the book-tax-difference signals and persistence fields.

    For each pretax-income observation (end E, filed F, pretax P):
      * ``btd``        = P − Tax(E) / statutory_rate(E)          — the raw book-tax gap ($)
      * ``btd_assets`` = btd / Assets(E)                          — the scaled level signal
      * ``btd_neg``    = − btd_assets                             — the *ranking* signal (long top
                                                                    tercile = LOW gap = "clean")
      * ``d_btd_assets`` = btd_assets − btd_assets(E−1yr)         — the year-on-year change
      * ``d_btd_neg``  = − d_btd_assets
      * ``roa``        = P / Assets(E)                            — pretax return on assets (t)
      * ``roa_next``   = P(E+1yr) / Assets(E+1yr)                 — next year's ROA (persistence)
    Rows without a valid pretax, tax, positive assets, or non-positive statutory-implied taxable
    income base are handled gracefully; a row is dropped only if the level ``btd_assets`` cannot
    be formed.
    """
    rows = []
    p = pretax.sort_values("end").reset_index(drop=True)
    for _, r in p.iterrows():
        end, filed, pv = r["end"], r["filed"], r["val"]
        tv = _asof_value(tax, end)
        av = _asof_value(assets, end)
        if tv is None or av is None or av <= 0:
            continue
        rate = statutory_rate(end)
        implied_taxable = tv / rate
        btd = pv - implied_taxable
        btd_assets = btd / av
        roa = pv / av
        # year-on-year change in the scaled BTD
        p_prior = _prior_year_value(pretax, end)
        t_prior = _asof_value(tax, end - pd.Timedelta(days=365), tol=40)
        a_prior = _asof_value(assets, end - pd.Timedelta(days=365), tol=40)
        if (p_prior is not None and t_prior is not None and a_prior is not None
                and a_prior > 0):
            rate_p = statutory_rate(end - pd.Timedelta(days=365))
            btd_assets_prior = (p_prior - t_prior / rate_p) / a_prior
            d_btd_assets = btd_assets - btd_assets_prior
        else:
            d_btd_assets = np.nan
        # next-year ROA for the earnings-persistence mechanism
        p_next = _asof_value(pretax, end + pd.Timedelta(days=365), tol=40)
        a_next = _asof_value(assets, end + pd.Timedelta(days=365), tol=40)
        roa_next = (p_next / a_next) if (p_next is not None and a_next is not None
                                         and a_next > 0) else np.nan
        rows.append({"end": end, "filed": filed, "pretax": pv, "tax": tv, "assets": av,
                     "btd": btd, "btd_assets": btd_assets, "btd_neg": -btd_assets,
                     "d_btd_assets": d_btd_assets, "d_btd_neg": -d_btd_assets,
                     "roa": roa, "roa_next": roa_next})
    cols = ["end", "filed", "pretax", "tax", "assets", "btd", "btd_assets", "btd_neg",
            "d_btd_assets", "d_btd_neg", "roa", "roa_next"]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = "2007-01-01", end: str | None = None,
                prices_path: str = PRICES_CACHE, events_path: str = EVENTS_CACHE
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download basket prices + EDGAR pretax / tax / assets, build BTD signals, cache.

    Network-only; used once to build the cache. Writes a wide adjusted-close CSV and a long event
    CSV (one row per ticker × fiscal year). Never imported by the offline notebook cells.
    """
    import yfinance as yf

    raw = yf.download(BASKET, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    keep = [c for c in raw.columns if raw[c].notna().mean() >= 0.10]
    prices = raw[keep].copy()
    os.makedirs(os.path.dirname(prices_path), exist_ok=True)
    prices.to_csv(prices_path)

    rows = []
    for tk in BASKET:
        cik = CIK.get(tk)
        if cik is None or tk not in prices.columns:
            continue
        pretax = _annual_flow_series(cik, PRETAX_CONCEPTS)
        if len(pretax) < 4:
            continue
        tax = _annual_flow_series(cik, TAX_CONCEPTS)
        assets = _instant_series(cik, ASSET_CONCEPTS)
        sig = build_signal(pretax, tax, assets)
        for _, s in sig.iterrows():
            rows.append({"ticker": tk, **s.to_dict()})
        time.sleep(0.15)
    events = pd.DataFrame(rows).dropna(subset=["filed", "btd_assets"])
    events = events.sort_values(["ticker", "filed"]).reset_index(drop=True)
    events.to_csv(events_path, index=False)
    return prices, events


def have_real(prices_path: str = PRICES_CACHE, events_path: str = EVENTS_CACHE) -> bool:
    return os.path.exists(prices_path) and os.path.exists(events_path)


def load_prices(path: str = PRICES_CACHE) -> pd.DataFrame:
    """Wide adjusted-close frame (index = date, columns = tickers), sliced to AS_OF."""
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return px.loc[px.index <= pd.Timestamp(AS_OF)]


def load_events(path: str = EVENTS_CACHE) -> pd.DataFrame:
    """Long event frame: ticker, end, filed, pretax, tax, assets, btd, btd_assets, btd_neg,
    d_btd_assets, d_btd_neg, roa, roa_next (10-Ks filed on/before AS_OF)."""
    ev = pd.read_csv(path, parse_dates=["end", "filed"])
    ev = ev[ev["filed"] <= pd.Timestamp(AS_OF)]
    return ev.sort_values(["ticker", "filed"]).reset_index(drop=True)


def load_real() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached prices + events in one call."""
    return load_prices(), load_events()


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_names: int = 42, n_years: int = 16, edge: float = 0.0,
                    seed: int = 856, sig_daily: float = 0.018
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic price panel + annual book-tax-difference events with a PLANTED return knob.

    Each name has a daily random-walk price. Annual filing dates (~252 trading days apart) each
    get a ``btd_assets`` level drawn N(0, 0.05) (a book-tax gap as a fraction of assets). The
    ranking signal ``btd_neg = −btd_assets``. If ``edge`` != 0, the ~252 sessions of forward
    return following each filing get an *extra* daily drift of ``edge * btd_neg / 252`` — LOW-BTD
    ("clean") names drift up and HIGH-BTD ("aggressive") names drift down, exactly the Hanlon
    direction. With ``edge = 0`` the forward path is pure noise: the long-short must NOT reach
    significance however the noise falls.

    Returns (prices wide frame, event table) in the same shape as ``load_real``.
    """
    rng = np.random.default_rng(seed)
    H = 252
    n_days = n_years * H + 260
    idx = pd.bdate_range("2009-01-05", periods=n_days)
    names = [f"N{i:02d}" for i in range(n_names)]

    price_cols = {}
    ev_rows = []
    for name in names:
        ret = rng.normal(0.0003, sig_daily, size=n_days)
        first = int(rng.integers(120, 200))
        q = first
        prev_lvl = None
        while q < n_days - H - 5:
            lvl = float(rng.normal(0.0, 0.05))            # btd_assets level
            neg = -lvl
            if edge != 0.0:
                ret[q + 1:q + 1 + H] += edge * neg / H
            d_lvl = np.nan if prev_lvl is None else (lvl - prev_lvl)
            ev_rows.append({"ticker": name, "_pos": q, "btd_assets": lvl, "btd_neg": neg,
                            "d_btd_assets": d_lvl, "d_btd_neg": (-d_lvl if prev_lvl is not None
                                                                 else np.nan)})
            prev_lvl = lvl
            q += H + int(rng.integers(-8, 9))
        price_cols[name] = 100.0 * np.exp(np.cumsum(ret))

    prices = pd.DataFrame(price_cols, index=idx)
    ev = pd.DataFrame(ev_rows)
    ev["filed"] = [idx[p] for p in ev["_pos"]]
    ev["end"] = ev["filed"] - pd.Timedelta(days=60)
    ev["pretax"] = np.nan
    ev["tax"] = np.nan
    ev["assets"] = np.nan
    ev["btd"] = np.nan
    ev["roa"] = np.nan
    ev["roa_next"] = np.nan
    ev = ev[["ticker", "end", "filed", "pretax", "tax", "assets", "btd", "btd_assets",
             "btd_neg", "d_btd_assets", "d_btd_neg", "roa", "roa_next"]]
    return prices, ev.sort_values(["ticker", "filed"]).reset_index(drop=True)
