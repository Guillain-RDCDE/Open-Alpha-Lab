"""Data layer for Study 858 — Margin ÷ Inventory Divergence.

The claim under test: the **Abarbanell–Bushee (1997/1998) fundamental-signals** intuition
that a firm whose **gross margin is rising while its inventory grows faster than sales** is
telling you two contradictory stories at once. Either the fat margin is unsustainable (it will
have to be discounted away to move the stock that is piling up), or the swollen inventory is
about to be written down — either way the accounting picture is *incoherent*, and the classic
result is that incoherence is a **negative** future signal. Turn it into one number:

    divergence = (Δ gross-margin%) − (inventory-growth − sales-growth)

so that a name whose margin is expanding *and* whose inventory is NOT outrunning sales scores
**high** (clean, coherent — the long), and a name whose margin is falling and/or whose
inventory is ballooning relative to sales scores **low** (contradictory — the short). Sort the
cross-section, go long the top, short the bottom.

Two sources, both offline-friendly once cached:

* **Real tape.**
  - Daily adjusted closes for a fixed basket of ~46 inventory-carrying US names (retailers,
    manufacturers, consumer-staples, hardware) from yfinance, no key.
  - Per name, from EDGAR's XBRL ``companyconcept`` API (``data.sec.gov``): quarterly
    **Revenues** and **CostOfRevenue** (with the usual tag fallbacks — the concept salad is
    real; older filers use ``SalesRevenueNet`` / ``CostOfGoodsSold``, ASC-606 filers use
    ``RevenueFromContractWithCustomerExcludingAssessedTax`` / ``CostOfGoodsAndServicesSold``)
    and the **InventoryNet** balance-sheet stock. For each figure we keep the period end and
    the **filing date** of the 10-Q/10-K that disclosed it — the day the number went public.
  From these, per period end E, we build the gross margin GM = (Rev−Cost)/Rev, its
  **year-over-year change** ΔGM = GM(E) − GM(E−1yr), the YoY **inventory growth** and **sales
  growth**, their gap, and finally the **divergence** signal above. Each event is one
  (ticker, filing date) row; the forward return is measured strictly *after* the filing (no
  look-ahead).

* **Synthetic.** A deterministic, fixed-seed generator producing a price + signal panel in
  which forward returns carry a **planted** component proportional to the signal (knob
  ``edge``). It is the positive control: with ``edge = 0`` the long-short must NOT manufacture
  significance; with a large planted ``edge`` it must light up. No network.

Honest about coverage: this is a **thin, uneven** panel. Some names carry deep quarterly
history back to ~2009; several report Revenues/CostOfRevenue only under one tag for part of
the sample, and the quarterly (60–100-day span) filter drops fiscal-Q4 figures disclosed only
in the annual 10-K, so the per-name series has gaps. That is a first-class caveat of the study,
not a footnote.

Pure numpy + pandas + stdlib for the offline path. ``fetch_panel`` (network) builds the cache
once and is never imported by the notebooks' offline cells.
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
PRICES_CACHE = os.path.join(CACHE_DIR, "mid_prices.csv")
EVENTS_CACHE = os.path.join(CACHE_DIR, "mid_events.csv")

UA = "OpenAlphaLab research contact@example.com"

AS_OF = "2026-06-30"        # last complete calendar month at publication
ERA_SPLIT = "2016-01-01"    # pre/post split (roughly halves the calendar sample)

# A transparent, fixed basket of inventory-carrying US names — retailers, manufacturers,
# consumer-staples, apparel, hardware/semis, autos — the firms for which an
# inventory-vs-sales / margin signal is even defined (a pure services or software name has no
# meaningful inventory). This is a *survivors* basket (all still trading). Survivorship is
# named on the Signal axis: a fixed surviving-names basket cannot include retailers/OEMs that
# were acquired or went bankrupt (exactly the names an inventory-glut short would want). The
# bias can be argued about but not ignored — see docs/references.md.
BASKET = [
    "WMT", "TGT", "COST", "HD", "LOW", "TJX", "ROST", "DG", "DLTR", "BBY",
    "KSS", "M", "KR", "NKE", "VFC", "RL", "PVH", "TPR", "HAS", "MAT",
    "CAT", "DE", "HON", "MMM", "EMR", "ETN", "PH", "DOV", "ITW", "HPQ",
    "DELL", "WDC", "STX", "F", "GM", "TSLA", "PG", "CL", "KMB", "GIS",
    "CPB", "HSY", "KHC", "INTC", "MU", "TXN",
]

# SEC CIK (10-digit, zero-padded), resolved once from the SEC ticker map and frozen here so the
# offline path never needs the network.
CIK = {
    "WMT": "0000104169", "TGT": "0000027419", "COST": "0000909832", "HD": "0000354950",
    "LOW": "0000060667", "TJX": "0000109198", "ROST": "0000745732", "DG": "0000029534",
    "DLTR": "0000935703", "BBY": "0000764478", "KSS": "0000885639", "M": "0000794367",
    "KR": "0000056873", "NKE": "0000320187", "VFC": "0000103379", "RL": "0001037038",
    "PVH": "0000078239", "TPR": "0001116132", "HAS": "0000046080", "MAT": "0000063276",
    "CAT": "0000018230", "DE": "0000315189", "HON": "0000773840", "MMM": "0000066740",
    "EMR": "0000032604", "ETN": "0001551182", "PH": "0000076334", "DOV": "0000029905",
    "ITW": "0000049826", "HPQ": "0000047217", "DELL": "0001571996", "WDC": "0000106040",
    "STX": "0001137789", "F": "0000037996", "GM": "0001467858", "TSLA": "0001318605",
    "PG": "0000080424", "CL": "0000021665", "KMB": "0000055785", "GIS": "0000040704",
    "CPB": "0000016732", "HSY": "0000047111", "KHC": "0001637459", "INTC": "0000050863",
    "MU": "0000723125", "TXN": "0000097476",
}

# Concept fallback ladders — take the concept with the longest per-name quarterly history.
REV_CONCEPTS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)
COST_CONCEPTS = (
    "CostOfGoodsAndServicesSold",
    "CostOfRevenue",
    "CostOfGoodsSold",
    "CostOfGoodsAndServicesSoldExcludingDepreciationDepletionAndAmortization",
)
INV_CONCEPTS = ("InventoryNet",)


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
    """Best (longest-history) *quarterly* flow series (revenue, cost) for one CIK.

    Flow (duration) concepts report a value over a start..end span. We keep only ~one-quarter
    spans (60-100 days), de-duplicate on the period end (earliest filing wins). This drops the
    fiscal-Q4 figure that appears only as a full-year span in the 10-K — an honest gap.
    Columns: end, filed, val.
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
# Signal construction — the margin/inventory divergence (point-in-time)
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


def _asof_filed(series: pd.DataFrame, end: pd.Timestamp, tol: int = 20):
    """Filing date attached to the value matched by :func:`_asof_value` (or None)."""
    if series.empty:
        return None
    gap = (series["end"] - end).abs().dt.days
    if gap.min() > tol:
        return None
    return series.loc[gap.idxmin(), "filed"]


def build_signal(rev: pd.DataFrame, cost: pd.DataFrame,
                 inv: pd.DataFrame) -> pd.DataFrame:
    """One row per quarter carrying the divergence signal and its components.

    For each revenue quarter (end E), matched to a same-quarter CostOfRevenue and InventoryNet:
      * ``gross_margin``  = (Rev(E) − Cost(E)) / Rev(E)
      * ``d_gross_margin`` = gross_margin(E) − gross_margin(E−1yr)      (ΔGM%, YoY)
      * ``inv_growth``    = Inv(E) / Inv(E−1yr) − 1
      * ``sales_growth``  = Rev(E) / Rev(E−1yr) − 1
      * ``inv_sales_gap`` = inv_growth − sales_growth                   (inventory outrunning sales)
      * ``divergence``    = d_gross_margin − inv_sales_gap              (THE signal; high = clean)
      * ``next_d_gross_margin`` = gross_margin(E+1yr) − gross_margin(E) (future margin change,
        the Abarbanell–Bushee mechanism check)
    The event is stamped with the LATEST of the revenue/cost/inventory filing dates (the day
    all three inputs were public). Rows missing any YoY leg are dropped.
    """
    # merged quarterly gross-margin frame
    gm_rows = []
    for _, r in rev.sort_values("end").iterrows():
        end = r["end"]
        c = _asof_value(cost, end)
        if c is None or r["val"] <= 0:
            continue
        gm = (r["val"] - c) / r["val"]
        cf = _asof_filed(cost, end)
        filed = max([d for d in (r["filed"], cf) if d is not None])
        gm_rows.append({"end": end, "filed": filed, "rev": r["val"], "gm": gm})
    gm = pd.DataFrame(gm_rows)
    cols = ["end", "filed", "gross_margin", "d_gross_margin", "inv_growth", "sales_growth",
            "inv_sales_gap", "divergence", "next_d_gross_margin"]
    if gm.empty:
        return pd.DataFrame(columns=cols)
    gm = gm.sort_values("end").reset_index(drop=True)

    rows = []
    for _, g in gm.iterrows():
        end = g["end"]
        gm_prior = _yoy_match(gm.rename(columns={"gm": "val"})[["end", "filed", "val"]], end)
        rev_prior = _yoy_match(rev, end)
        inv_now = _asof_value(inv, end)
        inv_prior = _yoy_match(inv, end)
        if gm_prior is None or rev_prior is None or rev_prior <= 0:
            continue
        if inv_now is None or inv_prior is None or inv_prior <= 0:
            continue
        d_gm = g["gm"] - gm_prior
        inv_growth = inv_now / inv_prior - 1.0
        sales_growth = g["rev"] / rev_prior - 1.0
        gap = inv_growth - sales_growth
        divergence = d_gm - gap
        # future margin change (one year AFTER E)
        gm_next = _asof_value(gm.rename(columns={"gm": "val"})[["end", "filed", "val"]],
                              end + pd.Timedelta(days=365), tol=30)
        next_d_gm = (gm_next - g["gm"]) if gm_next is not None else np.nan
        # inventory filing may lag the revenue filing; keep the latest known date
        inv_filed = _asof_filed(inv, end)
        filed = max([d for d in (g["filed"], inv_filed) if d is not None])
        rows.append({"end": end, "filed": filed, "gross_margin": g["gm"],
                     "d_gross_margin": d_gm, "inv_growth": inv_growth,
                     "sales_growth": sales_growth, "inv_sales_gap": gap,
                     "divergence": divergence, "next_d_gross_margin": next_d_gm})
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = "2007-01-01", end: str | None = None,
                prices_path: str = PRICES_CACHE, events_path: str = EVENTS_CACHE
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download basket prices + EDGAR revenue / cost / inventory, build signals, cache.

    Network-only; used once to build the cache. Writes a wide adjusted-close CSV and a long
    event CSV (one row per ticker × quarter). Never imported by the offline notebook cells.
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
        rev = _quarterly_flow_series(cik, REV_CONCEPTS)
        cost = _quarterly_flow_series(cik, COST_CONCEPTS)
        inv = _instant_series(cik, INV_CONCEPTS)
        if len(rev) < 8 or len(cost) < 8 or len(inv) < 8:
            continue
        sig = build_signal(rev, cost, inv)
        for _, s in sig.iterrows():
            rows.append({"ticker": tk, **s.to_dict()})
        time.sleep(0.12)
    events = pd.DataFrame(rows).dropna(subset=["filed", "divergence"])
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
    """Long event frame: ticker, end, filed, gross_margin, d_gross_margin, inv_growth,
    sales_growth, inv_sales_gap, divergence, next_d_gross_margin (filings on/before AS_OF)."""
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
                    seed: int = 858, sig_daily: float = 0.020
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic price panel + divergence events with a PLANTED return knob.

    Each name has a daily random-walk price. Quarterly filing dates (~63 trading days apart)
    each get a divergence signal drawn N(0, 0.15). If ``edge`` != 0, the ~63 sessions of
    forward return following each filing get an *extra* daily drift of ``edge * signal / 63`` —
    high-divergence (clean) names drift up, contradictory ones drift down, exactly the claimed
    effect. With ``edge = 0`` the forward path is pure noise: the long-short must NOT reach
    significance however the noise falls.

    Returns (prices wide frame, event table) in the same shape as ``load_real``. The synthetic
    time index is a plain ``bdate_range`` (n well under the Timestamp horizon).
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
            g = rng.normal(0.0, 0.15)
            if edge != 0.0:
                ret[q + 1:q + 1 + H] += edge * g / H
            ev_rows.append({"ticker": name, "_pos": q, "divergence": float(g)})
            q += H + int(rng.integers(-4, 5))
        price_cols[name] = 100.0 * np.exp(np.cumsum(ret))

    prices = pd.DataFrame(price_cols, index=idx)
    ev = pd.DataFrame(ev_rows)
    ev["filed"] = [idx[p] for p in ev["_pos"]]
    ev["end"] = ev["filed"] - pd.Timedelta(days=40)
    ev["gross_margin"] = np.nan
    ev["d_gross_margin"] = ev["divergence"] * 0.3
    ev["inv_growth"] = np.nan
    ev["sales_growth"] = np.nan
    ev["inv_sales_gap"] = -ev["divergence"] * 0.7
    ev["next_d_gross_margin"] = np.nan
    ev = ev[["ticker", "end", "filed", "gross_margin", "d_gross_margin", "inv_growth",
             "sales_growth", "inv_sales_gap", "divergence", "next_d_gross_margin"]]
    return prices, ev.sort_values(["ticker", "filed"]).reset_index(drop=True)
