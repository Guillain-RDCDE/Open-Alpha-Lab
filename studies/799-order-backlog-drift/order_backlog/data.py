"""Data layer for Study 799 — Order-Backlog Drift (RPO).

The claim under test: **growth in a firm's order backlog leads its forward returns, and
the market underreacts to it.** The modern, machine-readable proxy for "backlog" is the
ASC-606 disclosure **Remaining Performance Obligations (RPO)** — the dollar value of
contracted revenue a company has *signed but not yet recognised*. A software firm that
signs a three-year enterprise deal today books the whole contract into RPO and bleeds it
into reported revenue over the next twelve quarters. If RPO is swelling faster than the
income statement, the argument goes, future revenue — and the stock — should follow.

Two sources, both offline-friendly once cached:

* **Real tape.**
  - Daily adjusted closes for a fixed basket of ~38 subscription/enterprise-software US
    names (yfinance, no key).
  - Per name, the full history of the us-gaap concept
    ``RevenueRemainingPerformanceObligation`` (total RPO, an *instant* balance-sheet
    figure) from EDGAR's XBRL ``companyconcept`` API (``data.sec.gov``). We also pull
    quarterly **Revenues** (to test the "leads sales" mechanism) and total **Assets** (a
    scaling denominator). For each figure we keep the period end and the **filing date**
    of the 10-Q/10-K that disclosed it — the date the number became public.
  From these we build, per name, the **year-over-year growth in RPO** (this quarter's RPO
  vs the same quarter a year ago) plus a **balance-sheet-scaled** variant (the YoY dollar
  change ÷ total assets). Each event is one (ticker, filing date) row carrying the signal;
  the forward return / sales lead is measured strictly *after* the filing is public.

* **Synthetic.** A deterministic, fixed-seed generator that produces a price + signal panel
  in which forward returns carry a **planted** component proportional to the signal (knob
  ``edge``). It is the positive control: with ``edge = 0`` the long-short must NOT
  manufacture significance; with a large planted ``edge`` it must light up. No network.

Honest about coverage: **RPO only exists post-ASC-606.** The disclosure requirement took
effect for fiscal years beginning after 2017-12-15, so essentially no name in this panel
reports RPO before 2018, and the cross-section is meaningfully wide only from ~2019
onward. That short, one-regime window is a first-class caveat of the study, not a
footnote — a ~7-year sample of a slow quarterly signal is thin by construction.

Pure numpy + pandas + stdlib for the offline path. ``fetch_panel`` (network) is used once
to build the cache and is never imported by the notebooks' offline cells.
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
PRICES_CACHE = os.path.join(CACHE_DIR, "rpo_prices.csv")
EVENTS_CACHE = os.path.join(CACHE_DIR, "rpo_events.csv")

UA = "open-alpha-lab research guillain@poulpe.us"

AS_OF = "2026-06-30"        # last complete calendar month at publication
ERA_SPLIT = "2022-01-01"    # early-ASC606 (2019-21) vs the mature RPO panel (2022+)

# A transparent, fixed basket of enterprise/subscription-software US names that disclose a
# total RemainingPerformanceObligation on EDGAR (>= ~8 distinct quarter ends each). This is
# a *survivors* basket (all still trading). Survivorship is named on the Signal axis: a
# fixed surviving-names basket cannot include software firms that were acquired or blew up.
# Direction of the bias is ambiguous for a long-top/short-bottom *growth* signal (both legs
# are survivors), but it can only be argued away, not ignored — see docs/references.md.
BASKET = [
    "MSFT", "ORCL", "ADBE", "CRM", "NOW", "ADSK", "SNOW", "DDOG", "ZS", "CRWD",
    "PANW", "TEAM", "HUBS", "DOCU", "TWLO", "OKTA", "NET", "MDB", "PTC", "TYL",
    "FTNT", "CDNS", "SNPS", "BILL", "PCTY", "PLTR", "DT", "S", "ESTC", "PATH",
    "GTLB", "U", "NCNO", "BOX", "FROG", "APPN", "AI", "ASAN",
]

# SEC CIK (10-digit, zero-padded), resolved once from the SEC ticker map and frozen here so
# the offline path never needs the network.
CIK = {
    "MSFT": "0000789019", "ORCL": "0001341439", "ADBE": "0000796343", "CRM": "0001108524",
    "NOW": "0001373715", "ADSK": "0000769397", "SNOW": "0001640147", "DDOG": "0001561550",
    "ZS": "0001713683", "CRWD": "0001535527", "PANW": "0001327567", "TEAM": "0001650372",
    "HUBS": "0001404655", "DOCU": "0001261333", "TWLO": "0001447669", "OKTA": "0001660134",
    "NET": "0001477333", "MDB": "0001441816", "PTC": "0000857005", "TYL": "0000860731",
    "FTNT": "0001262039", "CDNS": "0000813672", "SNPS": "0000883241", "BILL": "0001786352",
    "PCTY": "0001591698", "PLTR": "0001321655", "DT": "0001773383", "S": "0001583708",
    "ESTC": "0001707753", "PATH": "0001734722", "GTLB": "0001653482", "U": "0001810806",
    "NCNO": "0001902733", "BOX": "0001372612", "FROG": "0001800667", "APPN": "0001441683",
    "AI": "0001577526", "ASAN": "0001477720",
}

# RPO / backlog concept. RevenueRemainingPerformanceObligation is the standard ASC-606 tag
# for the total (current + non-current) remaining performance obligation. A couple of filers
# only tag current + non-current separately; we take the total when present and otherwise the
# concept with the longest history.
RPO_CONCEPTS = (
    "RevenueRemainingPerformanceObligation",
)
REV_CONCEPTS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)
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
    10-Q/10-K observation, de-duplicate on the period end (keeping the EARLIEST filing that
    disclosed it — first public disclosure, no restatement look-ahead), and pick the concept
    that yields the most distinct quarter ends. When several values share a period end (a
    restatement or a re-file), we keep the earliest-filed one. Columns: end, filed, val.
    """
    best = pd.DataFrame(columns=["end", "filed", "val"])
    for concept in concepts:
        d = _edgar_concept(cik, concept)
        if d is None:
            continue
        rows = []
        for u in d.get("units", {}).get("USD", []):
            if u.get("form") not in ("10-Q", "10-K", "10-K/A", "10-Q/A"):
                continue
            if not (u.get("end") and u.get("filed") and u.get("val") is not None):
                continue
            rows.append({"end": u["end"], "filed": u["filed"], "val": float(u["val"])})
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["end"] = pd.to_datetime(df["end"])
        df["filed"] = pd.to_datetime(df["filed"])
        # Some filers report the same period end multiple times (segment/re-file); keep the
        # earliest filing and, among ties, the largest value (the consolidated total).
        df = (df.sort_values(["end", "filed", "val"], ascending=[True, True, False])
                .drop_duplicates(subset=["end"], keep="first")
                .sort_values("end").reset_index(drop=True))
        if len(df) > len(best):
            best = df
    return best


def _quarterly_flow_series(cik: str, concepts: tuple[str, ...]) -> pd.DataFrame:
    """Best (longest-history) *quarterly* flow series (e.g. revenue) for one CIK.

    Flow (duration) concepts report a value over a start..end span. We keep only ~one-quarter
    spans (60-100 days), de-duplicate on the period end (earliest filing wins). Columns: end,
    filed, val.
    """
    best = pd.DataFrame(columns=["end", "filed", "val"])
    for concept in concepts:
        d = _edgar_concept(cik, concept)
        if d is None:
            continue
        rows = []
        for u in d.get("units", {}).get("USD", []):
            if u.get("form") not in ("10-Q", "10-K", "10-K/A", "10-Q/A"):
                continue
            if not (u.get("end") and u.get("start") and u.get("filed")
                    and u.get("val") is not None):
                continue
            span = (pd.Timestamp(u["end"]) - pd.Timestamp(u["start"])).days
            if not (60 <= span <= 100):
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
# Signal construction — YoY growth in RPO (point-in-time)
# --------------------------------------------------------------------------- #
def _yoy_match(series: pd.DataFrame, end: pd.Timestamp,
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


def _asof_value(series: pd.DataFrame, end: pd.Timestamp, tol: int = 20) -> float | None:
    """Value in ``series`` whose end matches ``end`` within ``tol`` days (same quarter)."""
    if series.empty:
        return None
    gap = (series["end"] - end).abs().dt.days
    if gap.min() > tol:
        return None
    return float(series.loc[gap.idxmin(), "val"])


def build_signal(rpo: pd.DataFrame, rev: pd.DataFrame,
                 assets: pd.DataFrame) -> pd.DataFrame:
    """One row per RPO quarter carrying the signals and forward-sales fields.

    For each RPO observation (end E, filed F, backlog B):
      * ``rpo_yoy`` = B / B(E-1yr) - 1          — the primary ranking signal
      * ``rpo_chg_assets`` = (B - B(E-1yr)) / Assets(E)  — the balance-sheet-scaled variant
      * ``rev_yoy``   = Revenue(E) / Revenue(E-1yr) - 1  — contemporaneous sales growth
      * ``next_rev_yoy`` = Revenue(E+1q) / Revenue(E+1q-1yr) - 1  — FUTURE sales growth (the
        "leads sales" outcome; measured one quarter ahead)
    Rows without a valid YoY RPO growth are dropped.
    """
    rows = []
    d = rpo.sort_values("end").reset_index(drop=True)
    for _, r in d.iterrows():
        end, filed, val = r["end"], r["filed"], r["val"]
        prior = _yoy_match(rpo, end)
        if prior is None or prior <= 0:
            continue
        yoy = val / prior - 1.0
        a = _asof_value(assets, end)
        chg_assets = ((val - prior) / a) if (a and a > 0) else np.nan
        rev_now = _asof_value(rev, end)
        rev_prior = _yoy_match(rev, end)
        rev_yoy = (rev_now / rev_prior - 1.0) if (rev_now and rev_prior and rev_prior > 0) else np.nan
        nxt = _asof_value(rev, end + pd.Timedelta(days=91), tol=25)
        nxt_prior = _yoy_match(rev, end + pd.Timedelta(days=91))
        next_rev_yoy = (nxt / nxt_prior - 1.0) if (nxt and nxt_prior and nxt_prior > 0) else np.nan
        rows.append({"end": end, "filed": filed, "rpo": val,
                     "rpo_yoy": yoy, "rpo_chg_assets": chg_assets,
                     "rev_yoy": rev_yoy, "next_rev_yoy": next_rev_yoy})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = "2017-01-01", end: str | None = None,
                prices_path: str = PRICES_CACHE, events_path: str = EVENTS_CACHE
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download basket prices + EDGAR RPO / revenue / assets, build signals, cache.

    Network-only; used once to build the cache. Writes a wide adjusted-close CSV and a long
    event CSV (one row per ticker × RPO quarter). Never imported by the offline notebook
    cells.
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
        rpo = _instant_series(cik, RPO_CONCEPTS)
        if len(rpo) < 8:
            continue
        rev = _quarterly_flow_series(cik, REV_CONCEPTS)
        assets = _instant_series(cik, ASSET_CONCEPTS)
        sig = build_signal(rpo, rev, assets)
        for _, s in sig.iterrows():
            rows.append({"ticker": tk, **s.to_dict()})
        time.sleep(0.12)
    events = pd.DataFrame(rows).dropna(subset=["filed", "rpo_yoy"])
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
    """Long event frame: ticker, end, filed, rpo, rpo_yoy, rpo_chg_assets, rev_yoy,
    next_rev_yoy (filings on/before AS_OF)."""
    ev = pd.read_csv(path, parse_dates=["end", "filed"])
    ev = ev[ev["filed"] <= pd.Timestamp(AS_OF)]
    return ev.sort_values(["ticker", "filed"]).reset_index(drop=True)


def load_real() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached prices + events in one call."""
    return load_prices(), load_events()


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_names: int = 30, n_quarters: int = 28, edge: float = 0.0,
                    seed: int = 799, sig_daily: float = 0.022
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic price panel + RPO-growth events with a PLANTED return knob.

    Each name has a daily random-walk price. Quarterly filing dates (~63 trading days apart)
    each get an RPO-YoY-growth signal drawn N(0, 0.25). If ``edge`` != 0, the ~63 sessions of
    forward return following each filing get an *extra* daily drift of ``edge * signal / 63``
    — high-RPO-growth names drift up, low ones drift down, exactly the claimed lead. With
    ``edge = 0`` the forward path is pure noise: the long-short must NOT reach significance
    however the noise falls.

    ``n_quarters = 28`` (~7 years) matches the real post-ASC-606 window. Returns (prices wide
    frame, event table) in the same shape as ``load_real``.
    """
    rng = np.random.default_rng(seed)
    H = 63
    n_days = n_quarters * H + 200
    idx = pd.bdate_range("2018-07-02", periods=n_days)
    names = [f"N{i:02d}" for i in range(n_names)]

    price_cols = {}
    ev_rows = []
    for name in names:
        ret = rng.normal(0.0004, sig_daily, size=n_days)
        first = int(rng.integers(80, 140))
        q = first
        while q < n_days - H - 5:
            g = rng.normal(0.0, 0.25)
            if edge != 0.0:
                ret[q + 1:q + 1 + H] += edge * g / H
            ev_rows.append({"ticker": name, "_pos": q, "rpo_yoy": float(g)})
            q += H + int(rng.integers(-4, 5))
        price_cols[name] = 100.0 * np.exp(np.cumsum(ret))

    prices = pd.DataFrame(price_cols, index=idx)
    ev = pd.DataFrame(ev_rows)
    ev["filed"] = [idx[p] for p in ev["_pos"]]
    ev["end"] = ev["filed"] - pd.Timedelta(days=40)
    ev["rpo"] = np.nan
    ev["rpo_chg_assets"] = ev["rpo_yoy"] * 0.1
    ev["rev_yoy"] = np.nan
    ev["next_rev_yoy"] = np.nan
    ev = ev[["ticker", "end", "filed", "rpo", "rpo_yoy",
             "rpo_chg_assets", "rev_yoy", "next_rev_yoy"]]
    return prices, ev.sort_values(["ticker", "filed"]).reset_index(drop=True)
