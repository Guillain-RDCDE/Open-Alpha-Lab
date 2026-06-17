"""Data layer for Study 229 (Beneish M-score).

Two tapes, one contract (a year × ticker DataFrame of M-scores paired with a
next-year-return frame):

- ``synthetic_panel`` — a **deterministic, offline** generator. A tunable
  ``m_premium`` knob plants a known relationship between M-score and forward
  return; ``m_premium = 0`` is the null (M tells you nothing). High M-score =
  more likely manipulator; the *short-high-M / long-low-M* signal means a
  negative M → return relationship, so a negative m_premium plants that signal.
  The positive control uses a *negative* m_premium to test the strategy's
  short-high-M direction.

- ``fetch_panel`` — the real panel built from the desk's shared EDGAR caches
  plus two study-local caches fetched from EDGAR on first run:
  ``_cache/AccountsReceivableNetCurrent.parquet`` and
  ``_cache/PropertyPlantAndEquipmentNet.parquet`` (stored under the study
  directory, not the shared ``_cache/``).  Survivorship-bias is explicit: the
  panel covers *current* S&P 500 members projected backwards, so results are
  upper bounds on what a live strategy could have earned.

No look-ahead: the M-score for year *t* uses year-*t* and year-*(t-1)*
fundamentals (10-K accounting data). The signal is paired with the calendar
year-*t+1* return — a clean one-year lag.

**Beneish (1999) M-score components:**

    DSRI  = (AR_t / Sales_t) / (AR_{t-1} / Sales_{t-1})
    GMI   = ((Sales_{t-1} − GP_{t-1}) / Sales_{t-1}) /
             ((Sales_t − GP_t) / Sales_t)
    AQI   = [1 − (CA_t + PPE_t) / TA_t] /
             [1 − (CA_{t-1} + PPE_{t-1}) / TA_{t-1}]
    SGI   = Sales_t / Sales_{t-1}
    DEPI  = (DepPct_{t-1}) / (DepPct_t)   where DepPct = Dep/(Dep+PPE)
              (*Proxy*: we use gross change in net PPE minus capex; see below.)
    SGAI  = (SGA_t / Sales_t) / (SGA_{t-1} / Sales_{t-1})
              (*Proxy*: SGA ≈ GrossProfit − OperatingIncome)
    TATA  = (NI_t − CFO_t) / TA_t
    LVGI  = ((LTD_t + CL_t) / TA_t) / ((LTD_{t-1} + CL_{t-1}) / TA_{t-1})

**Published coefficients (Beneish 1999):**

    M = −4.84 + 0.920·DSRI + 0.528·GMI + 0.404·AQI + 0.892·SGI
             + 0.115·DEPI − 0.172·SGAI + 4.679·TATA − 0.327·LVGI

Threshold: M > −1.78 flags a likely manipulator.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
REPO_ROOT  = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
SHARED_CACHE = os.path.join(REPO_ROOT, "_cache")
LOCAL_CACHE  = os.path.join(STUDY_ROOT, "_cache")   # study-private yfinance cache

# Shared EDGAR concepts already in _cache/
_SHARED_CONCEPTS = [
    "Revenues",
    "GrossProfit",
    "Assets",
    "AssetsCurrent",
    "LiabilitiesCurrent",
    "LongTermDebtNoncurrent",
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "NetCashProvidedByUsedInOperatingActivities",
]

# Additional concepts fetched into LOCAL_CACHE on first run
_LOCAL_CONCEPTS = [
    "AccountsReceivableNetCurrent",
    "PropertyPlantAndEquipmentNet",
]

# Beneish (1999) published coefficients
M_INTERCEPT = -4.84
M_COEF = dict(
    DSRI=0.920,
    GMI=0.528,
    AQI=0.404,
    SGI=0.892,
    DEPI=0.115,
    SGAI=-0.172,
    TATA=4.679,
    LVGI=-0.327,
)

# Classification threshold: M > -1.78 → likely manipulator
M_THRESHOLD = -1.78


# ---------------------------------------------------------------------------
# Local EDGAR fetch (study-private cache)
# ---------------------------------------------------------------------------

_UA = {"User-Agent": "OpenAlphaLab research guillain@poulpe.us"}


def _http_get(url: str, retries: int = 3, delay: float = 0.5) -> bytes | None:
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            return urllib.request.urlopen(req, timeout=30).read()
        except Exception:
            time.sleep(delay)
    return None


def _fetch_edgar_concept(
    concept: str,
    tickers: list[str],
    t2cik: dict[str, str],
    start_year: int = 2006,
    end_year: int = 2024,
) -> pd.DataFrame:
    """Pull a single us-gaap concept from EDGAR for a list of tickers.

    Returns a DataFrame (year × ticker).  Only annual 10-K FY rows are kept.
    """
    store: dict[str, pd.Series] = {}
    for tk in tickers:
        cik = t2cik.get(tk) or t2cik.get(tk.replace("-", "."))
        if not cik:
            continue
        raw = _http_get(
            f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"
        )
        if not raw:
            time.sleep(0.4)
            continue
        try:
            d = json.loads(raw)
            rows = [
                (u["end"], u["val"])
                for u in d["units"]["USD"]
                if u.get("form") == "10-K" and u.get("fp") == "FY"
            ]
            if not rows:
                time.sleep(0.05)
                continue
            s = pd.Series(dict(rows))
            s.index = pd.to_datetime(s.index)
            s = (
                s[~s.index.duplicated(keep="last")]
                .sort_index()
                .groupby(lambda x: x.year)
                .last()
            )
            store[tk] = s
        except Exception:
            pass
        time.sleep(0.05)
    df = pd.DataFrame(store)
    return df.loc[
        (df.index >= start_year) & (df.index <= end_year)
    ] if not df.empty else df


def ensure_local_concepts(
    tickers: list[str],
    t2cik_path: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load or fetch the study-private EDGAR concepts into LOCAL_CACHE.

    Returns a dict mapping concept name → DataFrame.
    """
    os.makedirs(LOCAL_CACHE, exist_ok=True)
    needed = {
        c: os.path.join(LOCAL_CACHE, f"_edgar_{c}.parquet")
        for c in _LOCAL_CONCEPTS
    }
    missing = [c for c, p in needed.items() if not os.path.exists(p)]

    if missing:
        # Fetch the CIK map from EDGAR
        raw = _http_get("https://www.sec.gov/files/company_tickers.json")
        if raw is None:
            return {}
        t2cik = {
            v["ticker"]: str(v["cik_str"]).zfill(10)
            for v in json.loads(raw).values()
        }
        for c in missing:
            df = _fetch_edgar_concept(c, tickers, t2cik)
            if not df.empty:
                df.to_parquet(needed[c])
                print(f"  fetched {c}: {df.shape[1]} tickers")

    result: dict[str, pd.DataFrame] = {}
    for c, p in needed.items():
        if os.path.exists(p):
            result[c] = pd.read_parquet(p)
    return result


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------

def synthetic_panel(
    n_firms: int = 200,
    n_years: int = 16,
    m_premium: float = -0.05,   # negative → short-high-M earns positive returns
    base_ret: float = 0.10,
    ret_vol: float = 0.30,
    seed: int = 229,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """A synthetic firm panel with a known M-score → return relationship.

    Each firm-year draws an M-score from a realistic distribution (roughly
    normal, centred around −2.5).  The next-year return is:

        ret = base_ret + m_premium * (M − median_M) / IQR_M + noise

    When ``m_premium < 0``, high-M (manipulator) firms earn *less* the
    following year — the paper's prediction.  ``m_premium = 0`` is the null.

    Returns ``(m_scores, fwd_returns, truth)``.
    """
    rng = np.random.default_rng(seed)
    years = pd.Index(range(2008, 2008 + n_years), name="year")
    tickers = [f"F{j:03d}" for j in range(n_firms)]

    # M-scores: realistic distribution centred around -2.5, std ~ 1.5
    raw = rng.normal(loc=-2.5, scale=1.5, size=(n_years, n_firms))
    raw = np.clip(raw, -8.0, 4.0)
    m = pd.DataFrame(raw, index=years, columns=tickers)

    # Forward returns: loaded on M if m_premium != 0
    med = np.nanmedian(raw, axis=1, keepdims=True)
    iqr_val = (
        np.nanpercentile(raw, 75, axis=1, keepdims=True)
        - np.nanpercentile(raw, 25, axis=1, keepdims=True)
    )
    iqr_val = np.where(iqr_val > 0, iqr_val, 1.0)
    m_norm = (raw - med) / iqr_val

    noise = rng.normal(0.0, ret_vol, size=(n_years, n_firms))
    fwd_raw = base_ret + m_premium * m_norm + noise
    fwd = pd.DataFrame(fwd_raw, index=years, columns=tickers)

    truth = dict(
        n_firms=n_firms, n_years=n_years, m_premium=m_premium,
        base_ret=base_ret, ret_vol=ret_vol, seed=seed,
        has_premium=m_premium != 0.0,
    )
    return m, fwd, truth


# ---------------------------------------------------------------------------
# Real panel — EDGAR (shared + local) + yfinance returns
# ---------------------------------------------------------------------------

def fetch_panel(
    shared_cache: str = SHARED_CACHE,
    local_cache: str = LOCAL_CACHE,
    start_year: int = 2008,
    end_year: int = 2023,
    fetch_missing: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Beneish M-score panel + aligned next-year returns.

    Reads the desk's shared EDGAR concept caches and two study-private caches
    (AccountsReceivableNetCurrent, PropertyPlantAndEquipmentNet fetched on
    first run).

    **Survivorship bias is explicit and unavoidable**: the EDGAR caches contain
    only tickers that are *current* S&P 500 members, projected backwards.

    M-score for year *t* uses year-*t* and year-*(t-1)* 10-K data, paired
    with the year-*(t+1)* return (one-year lag).

    Returns
    -------
    m_scores : DataFrame, shape (n_years, n_tickers)
        Beneish M-score, year × ticker.
    fwd_ret : DataFrame, same shape
        Calendar-year *(t+1)* return, aligned to year-*t* signal index.
    """
    # ---- load shared EDGAR concept frames -----------------------------------
    shared_paths = {
        c: os.path.join(shared_cache, f"_edgar_{c}.parquet")
        for c in _SHARED_CONCEPTS
    }
    r_path = os.path.join(shared_cache, "_edgar_yrret.parquet")

    missing_shared = [c for c, p in shared_paths.items() if not os.path.exists(p)]
    if missing_shared or not os.path.exists(r_path):
        return pd.DataFrame(), pd.DataFrame()

    panels: dict[str, pd.DataFrame] = {
        c: pd.read_parquet(p) for c, p in shared_paths.items()
    }
    yr_ret = pd.read_parquet(r_path)

    # ---- load / fetch local EDGAR concepts ----------------------------------
    local_paths = {
        c: os.path.join(local_cache, f"_edgar_{c}.parquet")
        for c in _LOCAL_CONCEPTS
    }
    missing_local = [c for c, p in local_paths.items() if not os.path.exists(p)]

    if missing_local and fetch_missing:
        # Determine ticker set (use Assets as reference)
        ref_tickers = list(panels["Assets"].columns)
        ensure_local_concepts(ref_tickers)

    local_panels: dict[str, pd.DataFrame] = {}
    for c, p in local_paths.items():
        if os.path.exists(p):
            local_panels[c] = pd.read_parquet(p)

    if not local_panels:
        return pd.DataFrame(), pd.DataFrame()

    # ---- ticker universe: union of Assets, excluding those missing from
    #      the core 6 concepts needed by TATA, DSRI, SGI, AQI, DEPI, LVGI.
    #      GMI and SGAI (which need GrossProfit) are allowed to be NaN for
    #      firms that don't report GrossProfit (e.g., financial companies).
    all_panel = {**panels, **local_panels}

    # Core 9 concepts (excluding GrossProfit and OperatingIncomeLoss which
    # limit coverage to 74 tickers; their components are allowed to be NaN)
    _CORE = [
        "Assets", "AssetsCurrent", "LiabilitiesCurrent",
        "LongTermDebtNoncurrent", "Revenues", "NetIncomeLoss",
        "NetCashProvidedByUsedInOperatingActivities",
    ] + list(_LOCAL_CONCEPTS)

    common = set(panels["Assets"].columns)
    for c in _CORE:
        if c in all_panel:
            common &= set(all_panel[c].columns)
    common = sorted(common)
    if len(common) < 20:
        return pd.DataFrame(), pd.DataFrame()

    # ---- compute M-score components year by year ---------------------------
    years = list(range(start_year, end_year + 1))

    def _get(c: str, y: int) -> pd.Series:
        df = all_panel[c]
        if y in df.index:
            # Return values for the common tickers (NaN for missing tickers)
            return df.loc[y].reindex(common)
        return pd.Series(np.nan, index=common, dtype=float)

    rows_m: dict[int, pd.Series] = {}
    for yr in years:
        # Current-year values
        ta  = _get("Assets", yr)
        ca  = _get("AssetsCurrent", yr)
        cl  = _get("LiabilitiesCurrent", yr)
        ltd = _get("LongTermDebtNoncurrent", yr)
        rev = _get("Revenues", yr)
        gp  = _get("GrossProfit", yr)
        oi  = _get("OperatingIncomeLoss", yr)
        ni  = _get("NetIncomeLoss", yr)
        cfo = _get("NetCashProvidedByUsedInOperatingActivities", yr)
        ar  = _get("AccountsReceivableNetCurrent", yr)
        ppe = _get("PropertyPlantAndEquipmentNet", yr)

        # Prior-year values (for ratio indices)
        ta0  = _get("Assets", yr - 1)
        ca0  = _get("AssetsCurrent", yr - 1)
        cl0  = _get("LiabilitiesCurrent", yr - 1)
        ltd0 = _get("LongTermDebtNoncurrent", yr - 1)
        rev0 = _get("Revenues", yr - 1)
        gp0  = _get("GrossProfit", yr - 1)
        oi0  = _get("OperatingIncomeLoss", yr - 1)
        ar0  = _get("AccountsReceivableNetCurrent", yr - 1)
        ppe0 = _get("PropertyPlantAndEquipmentNet", yr - 1)

        valid = (ta > 0) & (ta0 > 0) & (rev > 0) & (rev0 > 0)

        # ---- DSRI: Days Sales Receivables Index ----------------------------
        dsr_t  = (ar  / rev.replace(0, np.nan)).where(valid)
        dsr_t0 = (ar0 / rev0.replace(0, np.nan)).where(valid)
        DSRI = (dsr_t / dsr_t0.replace(0, np.nan)).where(dsr_t0.notna() & (dsr_t0 > 0))
        # Fallback: if AR missing, use neutral DSRI = 1.0 (no change in
        # days-sales-receivables — conservative assumption)
        DSRI = DSRI.fillna(1.0).where(valid)

        # ---- GMI: Gross Margin Index ----------------------------------------
        # GrossProfit not reported by all firms (e.g., financial companies).
        # Fallback: GMI = 1.0 (neutral — no change in gross margin).
        gm_t  = (gp  / rev.replace(0, np.nan)).where(valid)
        gm_t0 = (gp0 / rev0.replace(0, np.nan)).where(valid)
        GMI_raw = (gm_t0 / gm_t.replace(0, np.nan)).where(gm_t.notna() & (gm_t > 0))
        GMI = GMI_raw.fillna(1.0).where(valid)

        # ---- AQI: Asset Quality Index -----------------------------------------
        aq_t  = (1 - (ca  + ppe)  / ta.replace(0, np.nan)).where(valid)
        aq_t0 = (1 - (ca0 + ppe0) / ta0.replace(0, np.nan)).where(valid)
        AQI = (aq_t / aq_t0.replace(0, np.nan)).where(aq_t0.notna() & (aq_t0.abs() > 1e-6))
        AQI = AQI.fillna(1.0).where(valid)

        # ---- SGI: Sales Growth Index ------------------------------------------
        SGI = (rev / rev0.replace(0, np.nan)).where(valid)

        # ---- DEPI: Depreciation Index -----------------------------------------
        # Proxy via net PPE intensity change; fallback = 1.0 (no change).
        ppe_pct_t  = (ppe  / ta.replace(0, np.nan)).where(valid)
        ppe_pct_t0 = (ppe0 / ta0.replace(0, np.nan)).where(valid)
        DEPI_raw = (ppe_pct_t0 / ppe_pct_t.replace(0, np.nan)).where(
            ppe_pct_t.notna() & (ppe_pct_t.abs() > 1e-6)
        )
        DEPI = DEPI_raw.fillna(1.0).where(valid)

        # ---- SGAI: SGA Expenses Index -----------------------------------------
        # SGA proxy = GrossProfit − OperatingIncome; needs GrossProfit.
        # Fallback: SGAI = 1.0 (neutral) when GrossProfit unavailable.
        sga  = (gp  - oi).clip(lower=0)
        sga0 = (gp0 - oi0).clip(lower=0)
        sgai_t  = (sga  / rev.replace(0, np.nan)).where(valid)
        sgai_t0 = (sga0 / rev0.replace(0, np.nan)).where(valid)
        SGAI_raw = (sgai_t / sgai_t0.replace(0, np.nan)).where(
            sgai_t0.notna() & (sgai_t0 > 0)
        )
        SGAI = SGAI_raw.fillna(1.0).where(valid)

        # ---- TATA: Total Accruals to Total Assets ----------------------------
        TATA = ((ni - cfo) / ta.replace(0, np.nan)).where(valid)
        # TATA can be NaN if NI or CFO missing — leave as NaN (these are
        # core concepts available for >400 tickers; rarely missing)

        # ---- LVGI: Leverage Index --------------------------------------------
        # LTD missing for some firms → treat LTD = 0 (most conservative for
        # an unlevered firm; avoids biasing the leverage index).
        ltd_fill  = ltd.fillna(0.0)
        ltd0_fill = ltd0.fillna(0.0)
        lev_t  = ((ltd_fill  + cl)  / ta.replace(0, np.nan)).where(valid)
        lev_t0 = ((ltd0_fill + cl0) / ta0.replace(0, np.nan)).where(valid)
        LVGI = (lev_t / lev_t0.replace(0, np.nan)).where(
            lev_t0.notna() & (lev_t0 > 0)
        )
        LVGI = LVGI.fillna(1.0).where(valid)

        # ---- M-score ----------------------------------------------------------
        # NaN only propagates from: TATA (NI/CFO missing), SGI (should be rare).
        m_yr = (
            M_INTERCEPT
            + M_COEF["DSRI"]  * DSRI
            + M_COEF["GMI"]   * GMI
            + M_COEF["AQI"]   * AQI
            + M_COEF["SGI"]   * SGI
            + M_COEF["DEPI"]  * DEPI
            + M_COEF["SGAI"]  * SGAI
            + M_COEF["TATA"]  * TATA
            + M_COEF["LVGI"]  * LVGI
        )
        rows_m[yr] = m_yr

    m_scores = pd.DataFrame(rows_m).T
    m_scores.index.name = "year"

    # ---- forward returns: year t+1 return, aligned to signal year t ----------
    fwd = yr_ret.reindex(index=[y + 1 for y in years], columns=common)
    fwd.index = years
    fwd.index.name = "year"

    return m_scores, fwd


def fingerprint(m: pd.DataFrame) -> str:
    """A short content fingerprint of the M-score panel."""
    arr = m.values.astype(float)
    flat = arr[np.isfinite(arr)]
    h = hashlib.sha1(np.ascontiguousarray(flat).tobytes())
    return h.hexdigest()[:12]
