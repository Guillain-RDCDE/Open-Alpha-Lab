"""Data layer for Study 854 — Cash Conversion Cycle.

The claim under test: **a shortening Cash Conversion Cycle is good, a bloating one is a
drag.** The CCC is how many days a firm's cash is tied up in operations between paying
suppliers and collecting from customers —

    CCC = DSO + DIO − DPO
        = AR/(Rev/365) + Inv/(COGS/365) − AP/(COGS/365)

with DSO = days sales outstanding (receivables), DIO = days inventory outstanding, DPO =
days payables outstanding. A firm that **shortens** its CCC frees working capital it can
redeploy and — the operations-finance literature argues — tends to out-earn; a firm whose
CCC is **rising** is bleeding cash into inventory and receivables faster than its payables
float can fund it. The prediction is directional — **falling CCC good, rising CCC bad** —
so the tradeable expression is **long the CCC-shorteners, short the CCC-bloaters**, sorted
on the year-over-year change in CCC.

Two sources, both offline-friendly once cached:

* **Real tape.**
  - Daily adjusted closes for a fixed basket of ~46 large US filers that actually carry
    inventory *and* trade payables (consumer, retail, industrials, healthcare-products,
    tech-hardware, materials — financials and pure-software excluded, they have no
    inventory so CCC is undefined) from yfinance (no key).
  - Per name, the full history of five us-gaap facts from EDGAR's XBRL ``companyconcept``
    API (``data.sec.gov``): the instant balances ``AccountsReceivableNetCurrent``,
    ``InventoryNet``, ``AccountsPayableCurrent`` and the quarterly flows ``Revenues`` and
    ``CostOfRevenue`` (COGS fallback ``CostOfGoodsAndServicesSold``). For every figure we
    keep the period end and the **filing date** of the 10-Q/10-K that disclosed it — the
    day the number went public.
  From these we build, per name, **CCC** (in days) and its **year-over-year change**
  (this quarter's CCC minus the same quarter a year ago) — the signal — plus a unit-free
  **percentage-change** robustness variant. Each event is one (ticker, filing date) row
  carrying the signal; forward returns are measured strictly *after* the filing is public
  (no look-ahead).

* **Synthetic.** A deterministic, fixed-seed generator producing a price + signal panel in
  which forward returns carry a **planted** component proportional to *minus* the CCC
  change (knob ``edge``): CCC-shortening names drift up, CCC-bloating names drift down —
  exactly the claim. It is the positive control: with ``edge = 0`` the long-short must NOT
  manufacture significance; with a large planted ``edge`` it must light up. No network.

Honest about coverage: this is a **thin, uneven** panel. The CCC needs *five* matched
facts per quarter (three balances + two flows), and each has its own tagging quirks — COGS
in particular is tagged as ``CostOfRevenue`` by some filers and ``CostOfGoodsAndServices
Sold`` by others, and ASC-606 shifted revenue tags in 2018 — so the quarters where all five
line up cleanly are fewer than the raw name count suggests. That is a first-class caveat of
the study, not a footnote.

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
PRICES_CACHE = os.path.join(CACHE_DIR, "ccc_prices.csv")
EVENTS_CACHE = os.path.join(CACHE_DIR, "ccc_events.csv")

UA = "OpenAlphaLab research contact@example.com"

AS_OF = "2026-06-30"        # last complete calendar month at publication
ERA_SPLIT = "2018-01-01"    # pre/post the ASC-606 revenue-tagging break

DAYS_Q = 365.0 / 4.0        # ≈91.25 — annualise one quarter of a flow to a daily run-rate

# A transparent, fixed basket of large US filers that carry genuine inventory AND trade
# payables (so all three CCC legs are defined) and report both a revenue and a COGS flow on
# EDGAR. Financials/REITs and pure-software names are excluded (they hold little or no
# inventory, so DIO — and thus CCC — is undefined or degenerate). This is a *survivors*
# basket (all still trading) — survivorship is named on the Signal axis (see
# docs/references.md): a fixed surviving-names roster cannot include firms that were acquired
# or delisted. For a long-shorteners/short-bloaters signal both legs are survivors, so the
# first-order tilt partly cancels, but it can only be argued away, not ignored.
BASKET = [
    # consumer staples / household / beverages
    "PG", "KO", "PEP", "CL", "KMB", "CLX", "CHD", "GIS", "K", "KHC", "MDLZ", "HSY",
    # consumer discretionary / retail / apparel
    "NKE", "WMT", "TGT", "COST", "HD", "LOW", "TJX", "VFC",
    # tech / hardware / semis (real inventory)
    "AAPL", "CSCO", "INTC", "HPQ", "TXN", "MU", "STX", "WDC",
    # industrials / machinery
    "CAT", "DE", "HON", "MMM", "EMR", "ITW", "PH", "DOV", "PCAR",
    # healthcare products / medtech / pharma with inventory
    "JNJ", "ABT", "MDT", "BDX", "BAX", "SYK",
    # materials / chemicals
    "DD", "PPG", "SHW",
]

# us-gaap concepts, in order of preference (longest per-name history wins).
AR_CONCEPTS = (
    "AccountsReceivableNetCurrent",
    "ReceivablesNetCurrent",
)
INV_CONCEPTS = (
    "InventoryNet",
)
AP_CONCEPTS = (
    "AccountsPayableCurrent",
    "AccountsPayableTradeCurrent",
)
REV_CONCEPTS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)
COGS_CONCEPTS = (
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
    "CostOfGoodsSold",
)


# --------------------------------------------------------------------------- #
# EDGAR helpers (network; cache builders only)
# --------------------------------------------------------------------------- #
def _http_json(url: str, retries: int = 3) -> dict | None:
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:
            time.sleep(0.5 * (k + 1))
    return None


def _ticker_cik_map() -> dict[str, str]:
    """Resolve ticker → 10-digit zero-padded CIK from the SEC company-ticker map (network)."""
    d = _http_json("https://www.sec.gov/files/company_tickers.json")
    out: dict[str, str] = {}
    if d is None:
        return out
    for _, row in d.items():
        tk = str(row.get("ticker", "")).upper()
        cik = row.get("cik_str")
        if tk and cik is not None and tk not in out:
            out[tk] = f"{int(cik):010d}"
    return out


def _edgar_concept(cik: str, concept: str) -> dict | None:
    url = (f"https://data.sec.gov/api/xbrl/companyconcept/"
           f"CIK{cik}/us-gaap/{concept}.json")
    return _http_json(url)


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
    """Best (longest-history) *quarterly* flow series (revenue / COGS) for one CIK.

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
# Signal construction — YoY change in CCC (point-in-time)
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


def _days(level: float | None, flow_q: float | None) -> float | None:
    """A working-capital *days* ratio: balance ÷ (annualised quarterly flow ÷ 365).

    With a quarter of the flow, the daily run-rate is flow_q / DAYS_Q, so
    days = level * DAYS_Q / flow_q. Any constant annualisation factor cancels in the
    same-fiscal-quarter YoY change, which also nets out seasonality.
    """
    if flow_q is None or flow_q <= 0 or level is None or level < 0:
        return None
    return level * DAYS_Q / flow_q


def _ccc(dso: float | None, dio: float | None, dpo: float | None) -> float | None:
    """CCC = DSO + DIO − DPO. Undefined if any leg is missing."""
    if dso is None or dio is None or dpo is None:
        return None
    return dso + dio - dpo


COLS = ["end", "filed", "ar", "inv", "ap", "rev", "cogs",
        "dso", "dio", "dpo", "ccc", "ccc_yoy_chg", "ccc_yoy_pct",
        "ccc_score", "ccc_pct_score", "gm", "next_gm_chg"]


def _ccc_at(ar_s, inv_s, ap_s, rev_s, cogs_s, end):
    """Assemble the CCC (and its three legs) at a given period ``end`` from the raw series.

    Returns (dso, dio, dpo, ccc, ar, inv, ap, rev, cogs) or Nones where a leg is missing.
    """
    ar = _asof_value(ar_s, end)
    inv = _asof_value(inv_s, end)
    ap = _asof_value(ap_s, end)
    rev = _asof_value(rev_s, end)
    cogs = _asof_value(cogs_s, end)
    dso = _days(ar, rev)
    dio = _days(inv, cogs)
    dpo = _days(ap, cogs)
    return dso, dio, dpo, _ccc(dso, dio, dpo), ar, inv, ap, rev, cogs


def build_signal(ar: pd.DataFrame, inv: pd.DataFrame, ap: pd.DataFrame,
                 rev: pd.DataFrame, cogs: pd.DataFrame) -> pd.DataFrame:
    """One row per quarter carrying the CCC signals and a forward-margin field.

    For each accounts-receivable observation (end E, filed F) we assemble the full CCC at E
    and at E−1yr:
      * ``ccc``          = DSO + DIO − DPO at E, in days
      * ``ccc_yoy_chg``  = CCC(E) − CCC(E−1yr)          — the change, in days (PRIMARY)
      * ``ccc_yoy_pct``  = CCC(E)/CCC(E−1yr) − 1        — unit-free change (robustness)
      * ``ccc_score``    = −ccc_yoy_chg                 — signed so HIGH score = FALLING CCC =
        the ATTRACTIVE (long) side per the claim; the whole 798-style machinery then goes long
        the top score / short the bottom with no special-casing, and a WRONG-SIGN result shows
        up as a *negative* long-short.
      * ``ccc_pct_score`` = −ccc_yoy_pct
      * ``gm``           = (Rev − COGS)/Rev at E        — contemporaneous gross margin
      * ``next_gm_chg``  = gm(E+1q) − gm(E)             — FUTURE gross-margin change (the
        "frees cash → out-earns" mechanism check: does CCC shortening precede rising margin?)
    Rows without a valid YoY CCC change are dropped.
    """
    rows = []
    a = ar.sort_values("end").reset_index(drop=True)
    for _, r in a.iterrows():
        end, filed = r["end"], r["filed"]
        dso, dio, dpo, ccc_now, ar_v, inv_v, ap_v, rev_v, cogs_v = _ccc_at(
            ar, inv, ap, rev, cogs, end)
        if ccc_now is None:
            continue
        p_end = end - pd.Timedelta(days=365)
        # prior-year CCC from ~1yr-back matches of every leg
        ar_p = _yoy_match(ar, end)
        inv_p = _yoy_match(inv, end)
        ap_p = _yoy_match(ap, end)
        rev_p = _yoy_match(rev, end)
        cogs_p = _yoy_match(cogs, end)
        dso_p = _days(ar_p, rev_p)
        dio_p = _days(inv_p, cogs_p)
        dpo_p = _days(ap_p, cogs_p)
        ccc_p = _ccc(dso_p, dio_p, dpo_p)
        if ccc_p is None or ccc_p == 0:
            continue
        chg = ccc_now - ccc_p
        pct = ccc_now / ccc_p - 1.0 if ccc_p > 0 else np.nan
        gm = (rev_v - cogs_v) / rev_v if (rev_v and cogs_v is not None and rev_v > 0) else np.nan
        # future gross margin one quarter ahead
        nxt_end = end + pd.Timedelta(days=91)
        _, _, _, _, _, _, _, rev_n, cogs_n = _ccc_at(ar, inv, ap, rev, cogs, nxt_end)
        gm_n = (rev_n - cogs_n) / rev_n if (rev_n and cogs_n is not None and rev_n > 0) else np.nan
        next_gm_chg = (gm_n - gm) if (np.isfinite(gm) and np.isfinite(gm_n)) else np.nan
        rows.append({"end": end, "filed": filed, "ar": ar_v, "inv": inv_v, "ap": ap_v,
                     "rev": rev_v, "cogs": cogs_v, "dso": dso, "dio": dio, "dpo": dpo,
                     "ccc": ccc_now, "ccc_yoy_chg": chg, "ccc_yoy_pct": pct,
                     "ccc_score": -chg, "ccc_pct_score": -pct if np.isfinite(pct) else np.nan,
                     "gm": gm, "next_gm_chg": next_gm_chg})
        _ = p_end  # documented anchor; matching is gap-based, not exact-date
    return pd.DataFrame(rows, columns=COLS)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = "2007-01-01", end: str | None = None,
                prices_path: str = PRICES_CACHE, events_path: str = EVENTS_CACHE
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download basket prices + EDGAR CCC components, build CCC signals, cache.

    Network-only; used once to build the cache. Writes a wide adjusted-close CSV and a long
    event CSV (one row per ticker × quarter). Never imported by the offline notebook cells.
    """
    import yfinance as yf

    cik_map = _ticker_cik_map()

    raw = yf.download(BASKET, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    keep = [c for c in raw.columns if raw[c].notna().mean() >= 0.10]
    prices = raw[keep].copy()
    os.makedirs(os.path.dirname(prices_path), exist_ok=True)
    prices.to_csv(prices_path)

    rows = []
    for tk in BASKET:
        cik = cik_map.get(tk)
        if cik is None or tk not in prices.columns:
            continue
        ar = _instant_series(cik, AR_CONCEPTS)
        inv = _instant_series(cik, INV_CONCEPTS)
        ap = _instant_series(cik, AP_CONCEPTS)
        rev = _quarterly_flow_series(cik, REV_CONCEPTS)
        cogs = _quarterly_flow_series(cik, COGS_CONCEPTS)
        if min(len(ar), len(inv), len(ap), len(rev), len(cogs)) < 8:
            continue
        sig = build_signal(ar, inv, ap, rev, cogs)
        for _, s in sig.iterrows():
            rows.append({"ticker": tk, **s.to_dict()})
        time.sleep(0.15)
    events = pd.DataFrame(rows).dropna(subset=["filed", "ccc_yoy_chg"])
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
    """Long event frame (filings on/before AS_OF)."""
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
                    seed: int = 854, sig_daily: float = 0.020
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic price panel + CCC-change events with a PLANTED return knob.

    Each name has a daily random-walk price. Quarterly filing dates (~63 trading days apart)
    each get a CCC-change signal drawn N(0, 12) days. If ``edge`` != 0, the ~63 sessions of
    forward return following each filing get an *extra* daily drift of
    ``−edge * chg / (12 * 63)`` — so a RISING (bloating) CCC drifts DOWN and a FALLING
    (shortening) CCC drifts UP, exactly the claim. With ``edge = 0`` the forward path is pure
    noise: the long-short must NOT reach significance however the noise falls.

    Returns (prices wide frame, event table) in the same shape as ``load_real``.
    """
    rng = np.random.default_rng(seed)
    H = 63
    n_days = n_quarters * H + 200
    idx = pd.bdate_range("2010-01-04", periods=n_days)   # n_days well under the 10k daily horizon
    names = [f"N{i:02d}" for i in range(n_names)]
    sig_days = 12.0                                      # sd of the CCC-change signal, in days

    price_cols = {}
    ev_rows = []
    for name in names:
        ret = rng.normal(0.0004, sig_daily, size=n_days)
        first = int(rng.integers(80, 140))
        q = first
        while q < n_days - H - 5:
            chg = rng.normal(0.0, sig_days)
            if edge != 0.0:
                ret[q + 1:q + 1 + H] += -edge * chg / (sig_days * H)
            ev_rows.append({"ticker": name, "_pos": q, "ccc_yoy_chg": float(chg)})
            q += H + int(rng.integers(-4, 5))
        price_cols[name] = 100.0 * np.exp(np.cumsum(ret))

    prices = pd.DataFrame(price_cols, index=idx)
    ev = pd.DataFrame(ev_rows)
    ev["filed"] = [idx[p] for p in ev["_pos"]]
    ev["end"] = ev["filed"] - pd.Timedelta(days=40)
    for c in ("ar", "inv", "ap", "rev", "cogs", "dso", "dio", "dpo"):
        ev[c] = np.nan
    ev["ccc"] = 60.0 + ev["ccc_yoy_chg"]
    ev["ccc_yoy_pct"] = ev["ccc_yoy_chg"] / 60.0
    ev["ccc_score"] = -ev["ccc_yoy_chg"]
    ev["ccc_pct_score"] = -ev["ccc_yoy_pct"]
    ev["gm"] = np.nan
    ev["next_gm_chg"] = np.nan
    ev = ev[["ticker", *COLS]]
    return prices, ev.sort_values(["ticker", "filed"]).reset_index(drop=True)
