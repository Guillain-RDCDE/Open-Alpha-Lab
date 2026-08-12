"""Data layer for Study 862 — Real Earnings Management (Roychowdhury 2006).

The claim under test: firms manage earnings by **real operating decisions**, not just accruals.
To hit a consensus number they (a) **cut discretionary expenses** — R&D and SG&A — and (b)
**overproduce** so fixed overhead spreads across more units and reported COGS falls. The
fingerprints are an **abnormally LOW discretionary expense** and an **abnormally HIGH production
cost**, both relative to a normal-operations benchmark. Roychowdhury (2006) defines the abnormal
piece as the residual from a cross-sectional model of the normal level:

* **Normal discretionary expense.**
  ``DISX_t / A_{t-1} = a0 + a1·(1/A_{t-1}) + a2·(Sales_{t-1}/A_{t-1}) + e``   with
  ``DISX = R&D + SG&A``. (Regressed on *lagged* sales to avoid the mechanical drop in expense
  when a firm boosts current sales.)
* **Normal production cost.**
  ``PROD_t / A_{t-1} = a0 + a1·(1/A_{t-1}) + a2·(Sales_t/A_{t-1}) + a3·(ΔSales_t/A_{t-1})
  + a4·(ΔSales_{t-1}/A_{t-1}) + e``   with ``PROD = COGS + ΔInventory``.

The abnormal pieces are the residuals; the aggregate REM proxy is
``REM = ab_PROD − ab_DISX`` (higher = more real management: overproduce **and** cut discretionary
spend). We then rank firms on REM at each 10-Q/10-K filing date and test the forward return.

Two sources, offline-friendly once cached:

* **Real tape.** Daily adjusted closes for a fixed basket of ~44 large US **manufacturers /
  hardware / pharma / industrials** (names with real inventory and production — Roychowdhury's
  setting) from yfinance, plus per-name quarterly **Revenues, CostOfRevenue, SG&A, R&D** (flows)
  and **InventoryNet, Assets** (instants) from EDGAR's XBRL ``companyconcept`` API. Each figure
  keeps its period end and the **filing date** of the 10-Q/10-K that disclosed it.

* **Synthetic.** A deterministic, fixed-seed generator producing a price + REM-signal panel whose
  forward returns carry a **planted** component proportional to the signal (knob ``edge``). It is
  the positive control: ``edge = 0`` must NOT manufacture significance; a large ``edge`` must.

Honest about coverage: this is a **thin, uneven** panel. XBRL quarterly flow tags are sparse for
the *fiscal fourth* quarter (firms disclose the full year in the 10-K, not Q4), and R&D is absent
for some consumer names, so the usable cross-section is smaller than the roster and gappy —
a first-class caveat, not a footnote. Two further caveats, stated up front: the normal-expense
**benchmark coefficients are estimated on the full pooled panel** (a mild in-sample look-ahead in
the *benchmark*, standard in the Roychowdhury literature which fits contemporaneous industry-year
cross-sections; the cross-sectional *rank* is what the sort trades), and with ~44 names we **pool
across industries** rather than fit Roychowdhury's per-industry-year regressions.

Pure numpy + pandas + scipy/statsmodels are available, but the offline path uses only numpy +
pandas. ``fetch_panel`` (network) builds the cache once and is never imported by the notebooks.
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
PRICES_CACHE = os.path.join(CACHE_DIR, "rem_prices.csv")
EVENTS_CACHE = os.path.join(CACHE_DIR, "rem_events.csv")

UA = "OpenAlphaLab research contact@example.com"

AS_OF = "2026-06-30"        # last complete calendar month at publication
ERA_SPLIT = "2015-01-01"    # pre/post split for the era robustness cut

# A transparent, fixed basket of large US manufacturers / hardware / pharma / industrials — the
# firms with real production and inventory that Roychowdhury's model was written for (pure-service
# names have no COGS/inventory and no production to manage). This is a *survivors* basket (all
# still trading). Survivorship is named on the Signal axis: a fixed surviving-names basket cannot
# include manufacturers that were acquired or failed. For a long-top/short-bottom REM sort both
# legs are survivors, so the first-order tilt partly cancels; the residual bias can be argued
# about but not ignored — see docs/references.md.
BASKET = [
    "AAPL", "INTC", "CSCO", "IBM", "TXN", "MU", "NVDA", "QCOM", "AMD", "HPQ",
    "WDC", "GLW", "KLAC", "LRCX", "AMAT", "ADI", "MCHP", "JNJ", "PFE", "MRK",
    "ABT", "MDT", "BMY", "LLY", "AMGN", "GILD", "MMM", "HON", "CAT", "DE",
    "EMR", "ITW", "PH", "GD", "LMT", "RTX", "KO", "PEP", "PG", "CL",
    "KMB", "NKE", "F", "GM",
]

# SEC CIK (10-digit, zero-padded), resolved once from the SEC ticker map and frozen so the offline
# path never needs the network.
CIK = {
    "AAPL": "0000320193", "INTC": "0000050863", "CSCO": "0000858877", "IBM": "0000051143",
    "TXN": "0000097476", "MU": "0000723125", "NVDA": "0001045810", "QCOM": "0000804328",
    "AMD": "0000002488", "HPQ": "0000047217", "WDC": "0000106040", "GLW": "0000024741",
    "KLAC": "0000319201", "LRCX": "0000707549", "AMAT": "0000006951", "ADI": "0000006281",
    "MCHP": "0000827054", "JNJ": "0000200406", "PFE": "0000078003", "MRK": "0000310158",
    "ABT": "0000001800", "MDT": "0001613103", "BMY": "0000014272", "LLY": "0000059478",
    "AMGN": "0000318154", "GILD": "0000882095", "MMM": "0000066740", "HON": "0000773840",
    "CAT": "0000018230", "DE": "0000315189", "EMR": "0000032604", "ITW": "0000049826",
    "PH": "0000076334", "GD": "0000040533", "LMT": "0000936468", "RTX": "0000101829",
    "KO": "0000021344", "PEP": "0000077476", "PG": "0000080424", "CL": "0000021665",
    "KMB": "0000055785", "NKE": "0000320187", "F": "0000037996", "GM": "0001467858",
}

# us-gaap concepts, in order of preference (take the concept with the longest per-name history).
REV_CONCEPTS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)
COGS_CONCEPTS = (
    "CostOfGoodsAndServicesSold",
    "CostOfRevenue",
    "CostOfGoodsSold",
)
SGA_CONCEPTS = (
    "SellingGeneralAndAdministrativeExpense",
    "GeneralAndAdministrativeExpense",
)
RND_CONCEPTS = ("ResearchAndDevelopmentExpense",)
INV_CONCEPTS = ("InventoryNet",)
ASSET_CONCEPTS = ("Assets",)

# The point-in-time signal columns carried on each event row.
SIGNAL_COLS = ("rem", "ab_prod", "ab_disx")


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
            time.sleep(0.4 * (k + 1))
    return None


def _instant_series(cik: str, concepts: tuple[str, ...]) -> pd.DataFrame:
    """Best (longest-history) instant balance-sheet series for one CIK across ``concepts``.

    Instant us-gaap concepts report one value AT a period end. Keep every 10-Q/10-K observation,
    de-duplicate on the period end (earliest filing kept — first public disclosure, no restatement
    look-ahead), pick the concept with the most distinct quarter ends. Columns: end, filed, val.
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
    """Best (longest-history) *quarterly* flow series for one CIK.

    Flow (duration) concepts report a value over a start..end span. Keep only ~one-quarter spans
    (60-100 days), de-duplicate on period end (earliest filing wins). Columns: end, filed, val.
    Note the fiscal-Q4 gap: 10-Ks usually disclose the full-year duration, not a Q4 quarter, so
    the fourth fiscal quarter is frequently absent — an honest source of panel gappiness.
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
# Per-firm quarterly panel construction (point-in-time)
# --------------------------------------------------------------------------- #
def _asof_value(series: pd.DataFrame, end: pd.Timestamp, tol: int = 20) -> float | None:
    """Value in ``series`` whose end matches ``end`` within ``tol`` days (same quarter)."""
    if series.empty:
        return None
    gap = (series["end"] - end).abs().dt.days
    if gap.min() > tol:
        return None
    return float(series.loc[gap.idxmin(), "val"])


def build_firm_quarters(rev: pd.DataFrame, cogs: pd.DataFrame, sga: pd.DataFrame,
                        rnd: pd.DataFrame, inv: pd.DataFrame, assets: pd.DataFrame
                        ) -> pd.DataFrame:
    """One row per fiscal quarter for a single firm, with the raw + one-lag inputs.

    The Revenues quarterly series is the spine (it carries the ``filed`` date). For each revenue
    period end E we look up the same-quarter COGS, SG&A, R&D (flows) and Inventory, Assets
    (instants), plus the prior quarter (E ≈ 91 days earlier) needed for ΔInventory, ΔSales and the
    lagged scaler A_{t-1}. Rows lacking the essentials (rev, cogs, sga, inv, prior inv, prior
    assets) are dropped. R&D missing → treated as 0 (DISX = SG&A alone) since several consumer
    names never report R&D. Columns: end, filed, rev, cogs, sga, rnd, inv, assets, and the
    ``*_lag`` versions plus rev_lag2 (for ΔSales_{t-1}).
    """
    if rev.empty:
        return pd.DataFrame()
    spine = rev.sort_values("end").reset_index(drop=True)
    rows = []
    for _, r in spine.iterrows():
        end, filed = r["end"], r["filed"]
        prev_end = end - pd.Timedelta(days=91)
        prev2_end = end - pd.Timedelta(days=182)
        rv = float(r["val"])
        cg = _asof_value(cogs, end)
        sg = _asof_value(sga, end)
        rd = _asof_value(rnd, end)
        iv = _asof_value(inv, end)
        iv_l = _asof_value(inv, prev_end, tol=25)
        as_l = _asof_value(assets, prev_end, tol=25)
        rv_l = _asof_value(rev, prev_end, tol=25)
        rv_l2 = _asof_value(rev, prev2_end, tol=25)
        if None in (cg, sg, iv, iv_l, as_l, rv_l) or as_l <= 0:
            continue
        rows.append({
            "end": end, "filed": filed, "rev": rv, "cogs": float(cg), "sga": float(sg),
            "rnd": (0.0 if rd is None else float(rd)), "inv": float(iv),
            "inv_lag": float(iv_l), "assets_lag": float(as_l),
            "rev_lag": float(rv_l), "rev_lag2": (np.nan if rv_l2 is None else float(rv_l2)),
        })
    cols = ["end", "filed", "rev", "cogs", "sga", "rnd", "inv", "inv_lag",
            "assets_lag", "rev_lag", "rev_lag2"]
    return pd.DataFrame(rows, columns=cols)


def _winsorize(x: np.ndarray, p: float = 0.01) -> np.ndarray:
    """Clip an array to its [p, 1-p] quantiles (NaN-safe)."""
    x = np.asarray(x, dtype=float)
    good = x[~np.isnan(x)]
    if good.size < 5:
        return x
    lo, hi = np.quantile(good, [p, 1.0 - p])
    return np.clip(x, lo, hi)


def scaled_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add the asset-scaled Roychowdhury regressors and the raw DISX / PROD levels.

    All levels are scaled by lagged assets A_{t-1}. DISX = R&D + SG&A; PROD = COGS + ΔInventory.
    Adds: disx_s, prod_s (dependent vars), inv_a (=1/A_{t-1}), salesL_a (Sales_{t-1}/A), sales_a
    (Sales_t/A), dsales_a (ΔSales_t/A), dsalesL_a (ΔSales_{t-1}/A), and gm (gross margin, for the
    mechanism axis). The scaled regressors are winsorised at 1/99% to keep the pooled OLS from
    being dragged by a handful of tiny-asset early quarters.
    """
    df = panel.copy()
    a = df["assets_lag"].to_numpy()
    disx = df["rnd"].to_numpy() + df["sga"].to_numpy()
    prod = df["cogs"].to_numpy() + (df["inv"].to_numpy() - df["inv_lag"].to_numpy())
    df["disx_s"] = disx / a
    df["prod_s"] = prod / a
    df["inv_a"] = 1.0 / a
    df["salesL_a"] = df["rev_lag"].to_numpy() / a
    df["sales_a"] = df["rev"].to_numpy() / a
    df["dsales_a"] = (df["rev"].to_numpy() - df["rev_lag"].to_numpy()) / a
    df["dsalesL_a"] = (df["rev_lag"].to_numpy() - df["rev_lag2"].to_numpy()) / a
    df["gm"] = (df["rev"].to_numpy() - df["cogs"].to_numpy()) / df["rev"].to_numpy()
    for c in ("disx_s", "prod_s", "inv_a", "salesL_a", "sales_a", "dsales_a", "dsalesL_a"):
        df[c] = _winsorize(df[c].to_numpy())
    return df


def _ols_resid(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Residuals of an OLS of y on X (X already includes an intercept column). NaN rows ignored
    for the *fit*; residuals returned for every row with finite y and X."""
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    resid = np.full(len(y), np.nan)
    resid[ok] = y[ok] - X[ok] @ beta
    return resid


def normal_model_residuals(panel: pd.DataFrame) -> pd.DataFrame:
    """Fit Roychowdhury's two normal-operations models on the *pooled* panel and attach the
    abnormal (residual) pieces and the REM aggregate.

    * ab_disx = residual of  disx_s ~ 1 + inv_a + salesL_a
    * ab_prod = residual of  prod_s ~ 1 + inv_a + sales_a + dsales_a + dsalesL_a
    * rem     = ab_prod − ab_disx   (higher ⇒ more real management: overproduce + cut discretionary)

    Cross-sectional / pooled OLS: the benchmark coefficients use the whole panel (a documented
    mild in-sample look-ahead in the *benchmark*; the sort trades the cross-sectional rank). Rows
    whose ab_prod or ab_disx is undefined (missing lag-2 sales, etc.) get NaN REM.
    """
    df = scaled_features(panel)
    n = len(df)
    ones = np.ones(n)
    Xd = np.column_stack([ones, df["inv_a"], df["salesL_a"]])
    Xp = np.column_stack([ones, df["inv_a"], df["sales_a"], df["dsales_a"], df["dsalesL_a"]])
    df["ab_disx"] = _ols_resid(df["disx_s"].to_numpy(), Xd)
    df["ab_prod"] = _ols_resid(df["prod_s"].to_numpy(), Xp)
    df["rem"] = df["ab_prod"] - df["ab_disx"]
    return df


def build_events(firm_panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack per-firm quarterly panels, fit the pooled normal models, and emit the event table.

    Returns one row per (ticker, fiscal quarter): ticker, end, filed, rem, ab_prod, ab_disx, gm,
    plus ``next_gm`` (the same firm's gross margin one quarter ahead — the mechanism-axis outcome).
    Rows without a finite REM are dropped. Everything is stamped with the **filing date** so the
    signal is strictly point-in-time.
    """
    frames = []
    for tk, p in firm_panels.items():
        if p is None or p.empty:
            continue
        q = p.copy()
        q.insert(0, "ticker", tk)
        frames.append(q)
    if not frames:
        return pd.DataFrame(columns=["ticker", "end", "filed", *SIGNAL_COLS, "gm", "next_gm"])
    stacked = pd.concat(frames, ignore_index=True)
    stacked = normal_model_residuals(stacked)
    # next-quarter gross margin per firm (the operating-reversal outcome), matched at ~+91 days
    stacked = stacked.sort_values(["ticker", "end"]).reset_index(drop=True)
    stacked["next_gm"] = np.nan
    for tk, g in stacked.groupby("ticker"):
        ends = g["end"].to_numpy()
        gm = g["gm"].to_numpy()
        idx = g.index.to_numpy()
        nxt = np.full(len(g), np.nan)
        for i in range(len(g)):
            gap = (ends - ends[i]).astype("timedelta64[D]").astype(float)
            cand = np.where((gap >= 70) & (gap <= 115))[0]
            if cand.size:
                nxt[i] = gm[cand[np.argmin(np.abs(gap[cand] - 91))]]
        stacked.loc[idx, "next_gm"] = nxt
    ev = stacked.dropna(subset=["rem"]).copy()
    cols = ["ticker", "end", "filed", "rem", "ab_prod", "ab_disx", "gm", "next_gm"]
    return ev[cols].sort_values(["ticker", "filed"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = "2007-01-01", end: str | None = None,
                prices_path: str = PRICES_CACHE, events_path: str = EVENTS_CACHE
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download basket prices + EDGAR fundamentals, build REM events, cache.

    Network-only; builds the cache once. Writes a wide adjusted-close CSV and a long event CSV
    (one row per ticker × fiscal quarter). Never imported by the offline notebook cells.
    """
    import yfinance as yf

    raw = yf.download(BASKET, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    keep = [c for c in raw.columns if raw[c].notna().mean() >= 0.10]
    prices = raw[keep].copy()
    os.makedirs(os.path.dirname(prices_path), exist_ok=True)
    prices.to_csv(prices_path)

    firm_panels = {}
    for tk in BASKET:
        cik = CIK.get(tk)
        if cik is None or tk not in prices.columns:
            continue
        rev = _quarterly_flow_series(cik, REV_CONCEPTS)
        cogs = _quarterly_flow_series(cik, COGS_CONCEPTS)
        sga = _quarterly_flow_series(cik, SGA_CONCEPTS)
        rnd = _quarterly_flow_series(cik, RND_CONCEPTS)
        inv = _instant_series(cik, INV_CONCEPTS)
        assets = _instant_series(cik, ASSET_CONCEPTS)
        fp = build_firm_quarters(rev, cogs, sga, rnd, inv, assets)
        if len(fp) >= 6:
            firm_panels[tk] = fp
        time.sleep(0.12)
    events = build_events(firm_panels)
    events = events.dropna(subset=["filed", "rem"]).reset_index(drop=True)
    events.to_csv(events_path, index=False)
    return prices, events


def have_real(prices_path: str = PRICES_CACHE, events_path: str = EVENTS_CACHE) -> bool:
    return os.path.exists(prices_path) and os.path.exists(events_path)


def load_prices(path: str = PRICES_CACHE) -> pd.DataFrame:
    """Wide adjusted-close frame (index = date, columns = tickers), sliced to AS_OF."""
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return px.loc[px.index <= pd.Timestamp(AS_OF)]


def load_events(path: str = EVENTS_CACHE) -> pd.DataFrame:
    """Long event frame: ticker, end, filed, rem, ab_prod, ab_disx, gm, next_gm (filings on/before
    AS_OF)."""
    ev = pd.read_csv(path, parse_dates=["end", "filed"])
    ev = ev[ev["filed"] <= pd.Timestamp(AS_OF)]
    return ev.sort_values(["ticker", "filed"]).reset_index(drop=True)


def load_real() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached prices + events in one call."""
    return load_prices(), load_events()


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_names: int = 30, n_quarters: int = 44, edge: float = 0.0,
                    seed: int = 862, sig_daily: float = 0.020
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic price panel + REM-signal events with a PLANTED forward-return knob.

    Each name has a daily random-walk price. Quarterly filing dates (~63 trading days apart) each
    get a REM signal drawn N(0, 1). If ``edge`` != 0, the ~63 sessions of forward return following
    each filing get an *extra* daily drift of ``edge · signal / 63`` — high-REM names drift up, low
    ones down (the sign is arbitrary for a machinery check; the real-tape sign is read separately).
    With ``edge = 0`` the forward path is pure noise: the long-short must NOT reach significance.

    Returns (prices wide frame, event table) in the same shape as ``load_real`` (ab_prod / ab_disx
    are split so that ab_prod − ab_disx = rem; gm / next_gm are NaN — the price sort is the
    control, the mechanism axis has its own inline synthetic in the tests).
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
            g = float(rng.normal(0.0, 1.0))
            if edge != 0.0:
                ret[q + 1:q + 1 + H] += edge * g / H
            ev_rows.append({"ticker": name, "_pos": q, "rem": g})
            q += H + int(rng.integers(-4, 5))
        price_cols[name] = 100.0 * np.exp(np.cumsum(ret))

    prices = pd.DataFrame(price_cols, index=idx)
    ev = pd.DataFrame(ev_rows)
    ev["filed"] = [idx[p] for p in ev["_pos"]]
    ev["end"] = ev["filed"] - pd.Timedelta(days=40)
    split = rng.normal(0.0, 0.5, size=len(ev))
    ev["ab_prod"] = ev["rem"].to_numpy() * 0.5 + split
    ev["ab_disx"] = ev["ab_prod"].to_numpy() - ev["rem"].to_numpy()
    ev["gm"] = np.nan
    ev["next_gm"] = np.nan
    ev = ev[["ticker", "end", "filed", "rem", "ab_prod", "ab_disx", "gm", "next_gm"]]
    return prices, ev.sort_values(["ticker", "filed"]).reset_index(drop=True)
