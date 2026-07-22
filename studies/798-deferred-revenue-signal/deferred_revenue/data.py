"""Data layer for Study 798 — Deferred-Revenue Signal.

The claim under test: **growth in deferred revenue / contract liabilities** (bookings a
subscription firm has billed but not yet recognised as revenue) **leads future sales and
forward returns.** Deferred revenue is a balance-sheet liability that mechanically front-runs
the income statement — you bill an annual SaaS contract today, park it in "contract
liabilities", and recognise it over the next four quarters. If that balance is swelling, the
argument goes, revenue is coming and the stock should follow.

Two sources, both offline-friendly once cached:

* **Real tape.**
  - Daily adjusted closes for a fixed basket of ~40 subscription/SaaS-type US names (yfinance,
    no key).
  - Per name, the full history of the **current deferred-revenue / contract-liability**
    balance from EDGAR's XBRL ``companyconcept`` API (``data.sec.gov``): the us-gaap concept
    ``DeferredRevenueCurrent`` (older filers) with a fallback to
    ``ContractWithCustomerLiabilityCurrent`` (the ASC-606 successor tag, post-2018). We also
    pull quarterly **Revenues** (to test the "leads future sales" claim) and total **Assets**
    (a scaling denominator). For each figure we keep the period end and the **filing date** of
    the 10-Q/10-K that disclosed it — the date the number became public.
  From these we build, per name, the **year-over-year growth in deferred revenue** (this
  quarter's balance vs the same quarter a year ago) plus a **balance-sheet-scaled** variant
  (the YoY dollar change divided by total assets). Each event is one (ticker, filing date) row
  carrying the signal; the forward return / sales lead is measured strictly *after* the filing
  is public (no look-ahead).

* **Synthetic.** A deterministic, fixed-seed generator that produces a price + signal panel in
  which forward returns carry a **planted** component proportional to the signal (knob
  ``edge``). It is the positive control: with ``edge = 0`` the long-short must NOT manufacture
  significance; with a large planted ``edge`` it must light up. No network.

Honest about coverage: this is a **thin, uneven** panel. The deep-history names (VRSN, ADBE,
ORCL, INTU…) start ~2009; the pure-play SaaS names (SNOW, DDOG, CRWD, NET…) only IPO'd
2018-2020, so the cross-section is small before ~2012 and only gets wide after ~2019. That is a
first-class caveat of the study, not a footnote.

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
PRICES_CACHE = os.path.join(CACHE_DIR, "dr_prices.csv")
EVENTS_CACHE = os.path.join(CACHE_DIR, "dr_events.csv")

UA = "open-alpha-lab research guillain@poulpe.us"

AS_OF = "2026-06-30"        # last complete calendar month at publication
ERA_SPLIT = "2019-01-01"    # pre/post the SaaS-panel-widens split (ASC-606 era)

# A transparent, fixed basket of subscription/SaaS-type US names that report a current
# deferred-revenue or contract-liability balance on EDGAR. This is a *survivors* basket (all
# still trading). Survivorship is named on the Signal axis: a fixed surviving-names basket
# cannot include subscription firms that were acquired or blew up. Direction of the bias is
# ambiguous for a long-top/short-bottom *growth* signal (both legs are survivors), but it can
# only be argued away, not ignored — see docs/references.md.
BASKET = [
    "MSFT", "ORCL", "ADBE", "CRM", "NOW", "INTU", "ADSK", "WDAY", "SNOW", "DDOG",
    "ZS", "CRWD", "PANW", "TEAM", "HUBS", "DOCU", "ZM", "TWLO", "OKTA", "NET",
    "MDB", "NFLX", "ADP", "PAYX", "VRSN", "PTC", "TYL", "MANH", "FICO", "AKAM",
    "FTNT", "CDNS", "SNPS", "EPAM", "BILL", "PCTY", "PAYC", "RNG", "BOX", "FIVN",
]

# SEC CIK (10-digit, zero-padded), resolved once from the SEC ticker map and frozen here so the
# offline path never needs the network.
CIK = {
    "MSFT": "0000789019", "ORCL": "0001341439", "ADBE": "0000796343", "CRM": "0001108524",
    "NOW": "0001373715", "INTU": "0000896878", "ADSK": "0000769397", "WDAY": "0001327811",
    "SNOW": "0001640147", "DDOG": "0001561550", "ZS": "0001713683", "CRWD": "0001535527",
    "PANW": "0001327567", "TEAM": "0001650372", "HUBS": "0001404655", "DOCU": "0001261333",
    "ZM": "0001585521", "TWLO": "0001447669", "OKTA": "0001660134", "NET": "0001477333",
    "MDB": "0001441816", "NFLX": "0001065280", "ADP": "0000008670", "PAYX": "0000723531",
    "VRSN": "0001014473", "PTC": "0000857005", "TYL": "0000860731", "MANH": "0001056696",
    "FICO": "0000814547", "AKAM": "0001086222", "FTNT": "0001262039", "CDNS": "0000813672",
    "SNPS": "0000883241", "EPAM": "0001352010", "BILL": "0001786352", "PCTY": "0001591698",
    "PAYC": "0001590955", "RNG": "0001384905", "BOX": "0001372612", "FIVN": "0001288847",
}

# Deferred-revenue / contract-liability concepts, in order of preference. Older filers tag the
# balance as DeferredRevenueCurrent; the ASC-606 (2018+) successor is
# ContractWithCustomerLiabilityCurrent. We take the concept with the longer history per name.
DEFREV_CONCEPTS = (
    "DeferredRevenueCurrent",
    "ContractWithCustomerLiabilityCurrent",
    "ContractWithCustomerLiabilityCurrentAndNoncurrent",
    "DeferredRevenueCurrentAndNoncurrent",
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
    that yields the most distinct quarter ends. Columns: end, filed, val.
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
        df = (df.sort_values("filed").drop_duplicates(subset=["end"], keep="first")
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
# Signal construction — YoY growth in deferred revenue (point-in-time)
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
    # closest to 365 days back
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


def build_signal(defrev: pd.DataFrame, rev: pd.DataFrame,
                 assets: pd.DataFrame) -> pd.DataFrame:
    """One row per deferred-revenue quarter carrying the signals and forward-sales fields.

    For each deferred-revenue observation (end E, filed F, balance D):
      * ``defrev_yoy`` = D / D(E-1yr) - 1   — the primary ranking signal
      * ``defrev_chg_assets`` = (D - D(E-1yr)) / Assets(E)  — the balance-sheet-scaled variant
      * ``rev_yoy``   = Revenue(E) / Revenue(E-1yr) - 1     — contemporaneous sales growth
      * ``next_rev_yoy`` = Revenue(E+1q) / Revenue(E+1q-1yr) - 1  — FUTURE sales growth (the
        "leads sales" outcome; measured one quarter ahead)
    Rows without a valid YoY deferred-revenue growth are dropped.
    """
    rows = []
    d = defrev.sort_values("end").reset_index(drop=True)
    for _, r in d.iterrows():
        end, filed, val = r["end"], r["filed"], r["val"]
        prior = _yoy_match(defrev, end)
        if prior is None or prior <= 0:
            continue
        yoy = val / prior - 1.0
        a = _asof_value(assets, end)
        chg_assets = ((val - prior) / a) if (a and a > 0) else np.nan
        rev_now = _asof_value(rev, end)
        rev_prior = _yoy_match(rev, end)
        rev_yoy = (rev_now / rev_prior - 1.0) if (rev_now and rev_prior and rev_prior > 0) else np.nan
        # future sales: next quarter end ~+91 days
        nxt = _asof_value(rev, end + pd.Timedelta(days=91), tol=25)
        nxt_prior = _yoy_match(rev, end + pd.Timedelta(days=91))
        next_rev_yoy = (nxt / nxt_prior - 1.0) if (nxt and nxt_prior and nxt_prior > 0) else np.nan
        rows.append({"end": end, "filed": filed, "defrev": val,
                     "defrev_yoy": yoy, "defrev_chg_assets": chg_assets,
                     "rev_yoy": rev_yoy, "next_rev_yoy": next_rev_yoy})
    cols = ["end", "filed", "defrev", "defrev_yoy", "defrev_chg_assets",
            "rev_yoy", "next_rev_yoy"]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = "2007-01-01", end: str | None = None,
                prices_path: str = PRICES_CACHE, events_path: str = EVENTS_CACHE
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download basket prices + EDGAR deferred revenue / revenue / assets, build signals, cache.

    Network-only; used once to build the cache. Writes a wide adjusted-close CSV and a long
    event CSV (one row per ticker × deferred-revenue quarter). Never imported by the offline
    notebook cells.
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
        defrev = _instant_series(cik, DEFREV_CONCEPTS)
        if len(defrev) < 8:
            continue
        rev = _quarterly_flow_series(cik, REV_CONCEPTS)
        assets = _instant_series(cik, ASSET_CONCEPTS)
        sig = build_signal(defrev, rev, assets)
        for _, s in sig.iterrows():
            rows.append({"ticker": tk, **s.to_dict()})
        time.sleep(0.12)
    events = pd.DataFrame(rows).dropna(subset=["filed", "defrev_yoy"])
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
    """Long event frame: ticker, end, filed, defrev, defrev_yoy, defrev_chg_assets,
    rev_yoy, next_rev_yoy (filings on/before AS_OF)."""
    ev = pd.read_csv(path, parse_dates=["end", "filed"])
    ev = ev[ev["filed"] <= pd.Timestamp(AS_OF)]
    return ev.sort_values(["ticker", "filed"]).reset_index(drop=True)


def load_real() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached prices + events in one call."""
    return load_prices(), load_events()


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_names: int = 30, n_quarters: int = 40, edge: float = 0.0,
                    seed: int = 798, sig_daily: float = 0.020
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic price panel + deferred-revenue-growth events with a PLANTED return knob.

    Each name has a daily random-walk price. Quarterly filing dates (~63 trading days apart)
    each get a deferred-revenue-growth signal drawn N(0, 0.25). If ``edge`` != 0, the ~63
    sessions of forward return following each filing get an *extra* daily drift of
    ``edge * signal / 63`` — high-deferred-growth names drift up, low ones drift down, exactly
    the claimed lead. With ``edge = 0`` the forward path is pure noise: the long-short must NOT
    reach significance however the noise falls.

    Returns (prices wide frame, event table) in the same shape as ``load_real``.
    """
    rng = np.random.default_rng(seed)
    H = 63
    n_days = n_quarters * H + 200
    idx = pd.bdate_range("2010-01-04", periods=n_days)
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
            ev_rows.append({"ticker": name, "_pos": q, "defrev_yoy": float(g)})
            q += H + int(rng.integers(-4, 5))
        price_cols[name] = 100.0 * np.exp(np.cumsum(ret))

    prices = pd.DataFrame(price_cols, index=idx)
    ev = pd.DataFrame(ev_rows)
    ev["filed"] = [idx[p] for p in ev["_pos"]]
    ev["end"] = ev["filed"] - pd.Timedelta(days=40)
    ev["defrev"] = np.nan
    ev["defrev_chg_assets"] = ev["defrev_yoy"] * 0.1
    ev["rev_yoy"] = np.nan
    ev["next_rev_yoy"] = np.nan
    ev = ev[["ticker", "end", "filed", "defrev", "defrev_yoy",
             "defrev_chg_assets", "rev_yoy", "next_rev_yoy"]]
    return prices, ev.sort_values(["ticker", "filed"]).reset_index(drop=True)
