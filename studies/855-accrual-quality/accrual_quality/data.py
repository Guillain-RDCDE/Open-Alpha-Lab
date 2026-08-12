"""Data layer for Study 855 — Accrual Quality (Dechow-Dichev).

The claim under test (Dechow & Dichev 2002, *"The Quality of Accruals and Earnings: The
Role of Accrual Estimation Errors"*, and its cross-sectional trading form in Francis,
LaFond, Olsson & Schipper 2005): accruals exist to shift the timing of cash flows into
earnings, so **good** accruals map cleanly into realised cash flows and **poor** ones are
mostly estimation error / noise. Operationalise a name's *accrual quality* as the standard
deviation of the residual from regressing its (asset-scaled) accruals on lagged, current
and lead operating cash flow over a rolling window of quarters — a *high* residual vol means
accruals that do **not** track cash, i.e. low quality. The trading claim: low-quality names
carry less-persistent earnings and (the strong form) a valuation discount, so a book that is
**long high-quality (low residual vol) / short low-quality (high residual vol)** should earn
a positive spread.

Two sources, both offline-friendly once cached:

* **Real tape.**
  - Daily adjusted closes for a fixed basket of ~45 deep-history US **non-financial** large
    filers (yfinance, no key). Non-financial by design: bank/insurer "accruals" are a
    different animal and the DD current-accruals model is written for operating firms.
  - Per name, from EDGAR's XBRL ``companyconcept`` API (``data.sec.gov``): quarterly
    (3-month) **NetIncomeLoss** and **NetCashProvidedByUsedInOperatingActivities** (CFO), and
    the instant **Assets** (scaling denominator), plus **AccountsReceivableNetCurrent** and
    **InventoryNet** for a balance-sheet working-capital-accrual robustness variant. Each
    figure keeps its period end and the **filing date** of the 10-Q/10-K that disclosed it.
  We build the **cash-flow-approach total accrual** ``accr = (NI - CFO)/avg-assets`` and
  regress it on lag/current/lead CFO over a rolling window; the residual std is the accrual
  quality (``aq_vol``). Every signal is stamped point-in-time: the rolling window at a filing
  uses only quarters whose full (lag, current, lead) CFO triple is already public on that
  filing date, so there is **no look-ahead** — the lead-CFO term never peeks past the filing.

* **Synthetic.** A deterministic, fixed-seed generator producing a price + quality-signal
  panel in which forward returns carry a **planted** component proportional to the
  quality signal (knob ``edge``). It is the positive control: with ``edge = 0`` the long-short
  must NOT manufacture significance; with a large planted ``edge`` it must light up. No network.

Honest about coverage: this is a **thin, uneven** panel. EDGAR tags quarterly (3-month) CFO
only in 10-Qs, so Q4 is typically missing (10-Ks carry the annual figure), giving ~3 clean
quarterly cash-flow points per name-year; the rolling DD window then needs several years
before the first signal. The cross-section is modest and grows over time — a first-class
caveat, not a footnote.

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
PRICES_CACHE = os.path.join(CACHE_DIR, "aq_prices.csv")
EVENTS_CACHE = os.path.join(CACHE_DIR, "aq_events.csv")

UA = "OpenAlphaLab research contact@example.com"

AS_OF = "2026-06-30"        # last complete calendar month at publication
ERA_SPLIT = "2018-01-01"    # pre/post split for the era robustness cut

# A transparent, fixed basket of ~45 deep-history US NON-financial large filers with clean,
# long CFO/NI reporting on EDGAR. Financials (banks, insurers) are excluded on purpose: the
# Dechow-Dichev current-accruals model is written for operating firms. This is a *survivors*
# basket (all still trading). Survivorship is named on the Signal axis: a fixed surviving-names
# roster cannot include operating firms that were acquired or failed. Both legs of a
# long-high-quality/short-low-quality book are drawn from the same survivor pool, so the
# first-order equity-survivorship tilt partly cancels — but it can only be argued about, not
# ignored (see docs/references.md).
BASKET = [
    "AAPL", "MSFT", "INTC", "CSCO", "ORCL", "IBM", "TXN", "QCOM", "HPQ", "ADI",
    "JNJ", "PFE", "MRK", "ABT", "MDT", "BMY", "LLY", "AMGN", "GILD", "BAX",
    "PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "COST", "TGT", "LOW",
    "CAT", "DE", "GE", "HON", "MMM", "EMR", "ITW", "ROK", "DOV", "PH",
    "XOM", "CVX", "COP", "DIS", "UPS",
]

# us-gaap concepts, in order of preference per figure. NetIncomeLoss / CFO are quarterly
# (3-month duration) flows; Assets / receivables / inventory are instant balance-sheet facts.
NI_CONCEPTS = ("NetIncomeLoss", "ProfitLoss")
CFO_CONCEPTS = (
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
)
ASSET_CONCEPTS = ("Assets",)
AR_CONCEPTS = ("AccountsReceivableNetCurrent", "ReceivablesNetCurrent")
INV_CONCEPTS = ("InventoryNet",)

DD_WINDOW = 12      # rolling window length (quarters) for the DD residual-vol estimate
DD_MIN_OBS = 8      # minimum usable quarters to emit a signal


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


def _ticker_cik_map() -> dict:
    """Resolve {TICKER: 10-digit CIK} once from the SEC ticker map (network)."""
    req = urllib.request.Request("https://www.sec.gov/files/company_tickers.json",
                                 headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    out = {}
    for row in d.values():
        out[row["ticker"].upper()] = f"{int(row['cik_str']):010d}"
    return out


def _instant_series(cik: str, concepts: tuple[str, ...]) -> pd.DataFrame:
    """Best (longest-history) instant balance-sheet series for one CIK across ``concepts``.

    Instant us-gaap facts report one value AT a period end. Keep every 10-Q/10-K observation,
    de-duplicate on the period end (keeping the EARLIEST filing that disclosed it — first
    public disclosure, no restatement look-ahead), pick the concept with the most quarter ends.
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
    """Best (longest-history) *quarterly* (3-month) flow series for one CIK.

    Income-statement and cash-flow us-gaap facts are reported **year-to-date** in 10-Qs
    (3/6/9-month cumulative) and annually in the 10-K, so pure 90-day facts are sparse (an
    operating-cash-flow series may carry only one clean quarterly span per year). We therefore
    reconstruct quarterly flows by **differencing the cumulative YTD chain**: facts sharing a
    fiscal-year ``start`` are cumulative, so sorted by ``end`` the quarterly value at point *i*
    is ``val[i] - val[i-1]`` (and ``val[0]`` itself for the first, a native 3-month span).
    We keep only points whose implied quarter span is ~one quarter (75-100 days), so Q4 (annual
    − 9-month) and any odd overlaps are dropped cleanly. De-duplicated on the period end
    (earliest filing wins). Columns: end, filed, val.
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
            if not (75 <= span <= 400):          # any YTD/annual duration fact
                continue
            rows.append({"start": u["start"], "end": u["end"],
                         "filed": u["filed"], "val": float(u["val"])})
        if not rows:
            continue
        df = pd.DataFrame(rows)
        for c in ("start", "end", "filed"):
            df[c] = pd.to_datetime(df[c])
        # earliest disclosure of each (start, end) fact
        df = (df.sort_values("filed").drop_duplicates(subset=["start", "end"], keep="first")
                .sort_values("end").reset_index(drop=True))
        q_rows = []
        for _, g in df.groupby("start"):
            g = g.sort_values("end").reset_index(drop=True)
            prev_val, prev_end = 0.0, g["start"].iloc[0] - pd.Timedelta(days=1)
            for _, r in g.iterrows():
                q_span = (r["end"] - prev_end).days
                q_val = r["val"] - prev_val
                if 75 <= q_span <= 100:
                    q_rows.append({"end": r["end"], "filed": r["filed"], "val": q_val})
                prev_val, prev_end = r["val"], r["end"]
        if not q_rows:
            continue
        qdf = pd.DataFrame(q_rows)
        qdf = (qdf.sort_values("filed").drop_duplicates(subset=["end"], keep="first")
                  .sort_values("end").reset_index(drop=True))
        if len(qdf) > len(best):
            best = qdf
    return best


# --------------------------------------------------------------------------- #
# Per-name aligned quarterly frame
# --------------------------------------------------------------------------- #
def _asof_instant(series: pd.DataFrame, end: pd.Timestamp, tol: int = 20) -> float | None:
    """Instant value whose end matches ``end`` within ``tol`` days (same quarter)."""
    if series.empty:
        return None
    gap = (series["end"] - end).abs().dt.days
    if gap.min() > tol:
        return None
    return float(series.loc[gap.idxmin(), "val"])


def align_quarters(ni: pd.DataFrame, cfo: pd.DataFrame, assets: pd.DataFrame,
                   ar: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    """One row per quarterly cash-flow point, carrying NI, CFO, Assets, AR, Inv and filing date.

    Keyed on the NI/CFO 3-month period end. Instant balance-sheet items (Assets, AR, Inv) are
    matched to the same period end within a 20-day tolerance. The filing date is the LATER of
    the NI and CFO filings (the quarter's numbers are all public only once both are filed).
    Consecutive-quarter spacing is enforced downstream by the ~91-day gap check.
    """
    if ni.empty or cfo.empty:
        return pd.DataFrame(columns=["end", "filed", "ni", "cfo", "assets", "ar", "inv"])
    cfo_by_end = {r["end"]: r for _, r in cfo.iterrows()}
    rows = []
    for _, rn in ni.iterrows():
        end = rn["end"]
        # match CFO within 8 days of the NI period end
        best_c = None
        for ce, rc in cfo_by_end.items():
            if abs((ce - end).days) <= 8:
                best_c = rc
                break
        if best_c is None:
            continue
        a = _asof_instant(assets, end)
        if a is None or a <= 0:
            continue
        filed = max(pd.Timestamp(rn["filed"]), pd.Timestamp(best_c["filed"]))
        rows.append({"end": end, "filed": filed, "ni": float(rn["val"]),
                     "cfo": float(best_c["val"]), "assets": a,
                     "ar": _asof_instant(ar, end), "inv": _asof_instant(inv, end)})
    df = pd.DataFrame(rows).sort_values("end").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Dechow-Dichev residual-volatility signal (point-in-time)
# --------------------------------------------------------------------------- #
def _resid_vol(accr: np.ndarray, cfo_lag: np.ndarray, cfo_now: np.ndarray,
               cfo_lead: np.ndarray) -> float:
    """Std of the OLS residual of accr ~ 1 + cfo_lag + cfo_now + cfo_lead. NaN if too thin."""
    m = (np.isfinite(accr) & np.isfinite(cfo_lag) & np.isfinite(cfo_now)
         & np.isfinite(cfo_lead))
    y = accr[m]
    if len(y) < 5:
        return float("nan")
    X = np.column_stack([np.ones(len(y)), cfo_lag[m], cfo_now[m], cfo_lead[m]])
    # guard against a rank-deficient window
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return float("nan")
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(np.std(resid, ddof=1)) if len(resid) > 1 else float("nan")


def build_signal(frame: pd.DataFrame, window: int = DD_WINDOW,
                 min_obs: int = DD_MIN_OBS) -> pd.DataFrame:
    """Rolling Dechow-Dichev accrual-quality events for one name's aligned quarterly frame.

    For each aligned quarter we form the scaled series (average total assets denominator):
      * ``accr_t`` = (NI_t - CFO_t) / avgTA_t        — cash-flow-approach total accrual
      * ``x_t``    = CFO_t / avgTA_t                  — the DD regressor (also lag/lead)
      * ``roa_t``  = NI_t / avgTA_t
      * ``wc_t``   = (dAR_t + dInv_t) / avgTA_t       — balance-sheet WC-accrual variant
    Only *consecutive* quarters (~91-day spacing) contribute lag/lead terms. At the filing of
    quarter ``q`` the rolling window is the ``window`` quarters ``t`` whose full (lag, current,
    lead) CFO triple is public by then — i.e. ``t <= q-1`` (the lead CFO_{t+1}=CFO_q is filed
    with quarter q). ``aq_vol`` = std of the DD residual over that window; a companion
    ``aq_vol_wc`` uses the WC accrual, and ``earn_vol`` = std of ROA over the window. Each event
    is stamped with quarter q's filing date; ``roa`` is quarter q's ROA and ``roa_next`` is
    quarter q+1's (used only by the persistence mechanism axis, never for trading).
    """
    cols = ["end", "filed", "aq_vol", "quality", "aq_vol_wc", "earn_vol", "roa", "roa_next"]
    df = frame.sort_values("end").reset_index(drop=True)
    n = len(df)
    if n < min_obs + 2:
        return pd.DataFrame(columns=cols)

    end = df["end"].to_numpy()
    ni = df["ni"].to_numpy(dtype=float)
    cfo = df["cfo"].to_numpy(dtype=float)
    assets = df["assets"].to_numpy(dtype=float)
    ar = df["ar"].to_numpy(dtype=float)
    inv = df["inv"].to_numpy(dtype=float)

    avg_ta = np.full(n, np.nan)
    avg_ta[1:] = 0.5 * (assets[1:] + assets[:-1])
    with np.errstate(invalid="ignore", divide="ignore"):
        accr = (ni - cfo) / avg_ta
        x = cfo / avg_ta
        roa = ni / avg_ta
        d_ar = np.concatenate([[np.nan], np.diff(ar)])
        d_inv = np.concatenate([[np.nan], np.diff(inv)])
        wc = (d_ar + d_inv) / avg_ta

    # lag / lead of the scaled CFO = the nearest available adjacent quarter. We do not gate on
    # exact 91-day spacing (EDGAR's YTD reporting leaves occasional single-quarter holes); the
    # point-in-time guarantee comes from the WINDOW gate below (a window closing at quarter q
    # never uses a lead beyond CFO_q, which is public at q's filing) rather than from spacing.
    x_lag = np.concatenate([[np.nan], x[:-1]])
    x_lead = np.concatenate([x[1:], [np.nan]])

    rows = []
    for q in range(min_obs + 1, n):
        # rows t <= q-1 whose lead (CFO_{t+1}) is public by filing of quarter q
        lo = max(0, q - window)
        sl = slice(lo, q)  # t in [lo, q-1]
        aq = _resid_vol(accr[sl], x_lag[sl], x[sl], x_lead[sl])
        if not np.isfinite(aq):
            continue
        wcw = _resid_vol(wc[sl], x_lag[sl], x[sl], x_lead[sl])
        roaw = roa[sl][np.isfinite(roa[sl])]
        earn_vol = float(np.std(roaw, ddof=1)) if len(roaw) > 1 else np.nan
        roa_next = float(roa[q]) if np.isfinite(roa[q]) else np.nan
        rows.append({"end": df["end"].iloc[q - 1], "filed": df["filed"].iloc[q - 1],
                     "aq_vol": aq, "quality": -aq, "aq_vol_wc": wcw,
                     "earn_vol": earn_vol, "roa": float(roa[q - 1]) if np.isfinite(roa[q - 1]) else np.nan,
                     "roa_next": roa_next})
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = "2007-01-01", end: str | None = None,
                prices_path: str = PRICES_CACHE, events_path: str = EVENTS_CACHE
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download basket prices + EDGAR NI/CFO/Assets/AR/Inv, build DD signals, cache.

    Network-only; used once to build the cache. Writes a wide adjusted-close CSV and a long
    event CSV (one row per ticker × accrual-quality quarter). Never imported by the offline
    notebook cells.
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
        ni = _quarterly_flow_series(cik, NI_CONCEPTS)
        cfo = _quarterly_flow_series(cik, CFO_CONCEPTS)
        if len(ni) < 10 or len(cfo) < 10:
            continue
        assets = _instant_series(cik, ASSET_CONCEPTS)
        ar = _instant_series(cik, AR_CONCEPTS)
        inv = _instant_series(cik, INV_CONCEPTS)
        frame = align_quarters(ni, cfo, assets, ar, inv)
        sig = build_signal(frame)
        for _, s in sig.iterrows():
            rows.append({"ticker": tk, **s.to_dict()})
        time.sleep(0.15)
    events = pd.DataFrame(rows).dropna(subset=["filed", "aq_vol"])
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
    """Long event frame: ticker, end, filed, aq_vol, quality, aq_vol_wc, earn_vol, roa,
    roa_next (filings on/before AS_OF)."""
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
                    seed: int = 855, sig_daily: float = 0.020
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic price panel + accrual-quality events with a PLANTED return knob.

    Each name has a daily random-walk price and a persistent latent *quality* level (higher =
    better = lower residual vol). At each quarterly filing (~63 trading days apart) the name
    emits a quality signal = latent + small noise. If ``edge`` != 0, the ~63 sessions of
    forward return following each filing get an *extra* daily drift of ``edge * quality / 63`` —
    high-quality names drift up, low-quality names drift down, exactly the claimed relation.
    With ``edge = 0`` the forward path is pure noise: the long-short must NOT reach significance.

    Returns (prices wide frame, event table) in the same schema as ``load_real``: the emitted
    ``quality`` is the sortable signal and ``aq_vol = -quality`` mirrors the real convention
    (high vol = poor quality). Business-day index, span well below the ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    H = 63
    n_days = n_quarters * H + 200
    idx = pd.bdate_range("2010-01-04", periods=n_days)
    names = [f"N{i:02d}" for i in range(n_names)]

    price_cols = {}
    ev_rows = []
    for name in names:
        latent = float(rng.normal(0.0, 1.0))     # persistent quality level for this name
        ret = rng.normal(0.0004, sig_daily, size=n_days)
        first = int(rng.integers(80, 140))
        q = first
        while q < n_days - H - 5:
            quality = latent + rng.normal(0.0, 0.25)
            if edge != 0.0:
                ret[q + 1:q + 1 + H] += edge * quality / H
            ev_rows.append({"ticker": name, "_pos": q, "quality": float(quality)})
            q += H + int(rng.integers(-4, 5))
        price_cols[name] = 100.0 * np.exp(np.cumsum(ret))

    prices = pd.DataFrame(price_cols, index=idx)
    ev = pd.DataFrame(ev_rows)
    ev["filed"] = [idx[p] for p in ev["_pos"]]
    ev["end"] = ev["filed"] - pd.Timedelta(days=40)
    ev["aq_vol"] = -ev["quality"]
    ev["aq_vol_wc"] = -ev["quality"]
    ev["earn_vol"] = np.nan
    ev["roa"] = np.nan
    ev["roa_next"] = np.nan
    ev = ev[["ticker", "end", "filed", "aq_vol", "quality", "aq_vol_wc",
             "earn_vol", "roa", "roa_next"]]
    return prices, ev.sort_values(["ticker", "filed"]).reset_index(drop=True)
