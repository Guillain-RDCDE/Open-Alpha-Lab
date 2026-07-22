"""Data layer for Study 791 — Advertising-Brand-Capital.

The believers' version (Belo-Lin-Vitorino 2014; Chan-Lakonishok-Sougiannis 2001): the
advertising a firm expenses each year builds an intangible *brand-capital* stock that GAAP
never puts on the balance sheet, so the market under-prices it. Firms that advertise
heavily — high **AdvertisingExpense / Sales** — should therefore earn a forward return
premium. We sort a basket on advertising intensity, go long the heavy advertisers and short
the light ones, and measure the spread.

Two tapes, one shape (an annual **fundamentals panel** of advertising + revenue, and a
monthly **return panel** over the same names, plus the SPY benchmark):

* **Real tape.** A fixed basket of ~40 long-listed US consumer-facing large-caps that
  *actually file an advertising line* — the honest, load-bearing scope decision of this
  study. For each name we pull annual ``AdvertisingExpense`` (a handful of names file the
  broader ``MarketingAndAdvertisingExpense`` line instead) and revenue from **SEC EDGAR
  companyconcept** (public, no key). The intensity signal at each month uses only data
  *known* at that month (a one-year reporting lag on fundamentals):

      adv_sales = AdvertisingExpense(FY Y-1) / Revenue(FY Y-1)

  Monthly **total returns** come from yfinance. All cached as parquet under ``_cache/``;
  the offline core never touches the network unless ``fetch=True``.

  **Coverage is thin, and we say so.** The vast majority of US public firms *omit* the
  advertising line entirely (it stopped being a required disclosure after ASU 2014-09), and
  even inside this hand-picked consumer basket, some names (streaming, hotels, warehouse
  clubs) never report it and are excluded by construction. So the "universe" here is not the
  market — it is *the set of firms that advertise enough to disclose it*, a selected slice.
  The basket is chosen **by sector, not by return** (so it is not a winners' basket), but the
  residual **survivorship** bias — current-membership names projected back — is named on the
  Signal axis.

* **Synthetic.** A deterministic, fixed-seed generator with a single planted knob, ``edge``:
  the *true* per-year premium earned per unit of (cross-sectionally de-meaned) advertising
  intensity. At ``edge = 0`` the signal carries **no** information and any long-high/short-low
  spread is luck — the test must NOT manufacture significance. At ``edge > 0`` the heavy
  advertisers genuinely out-earn and the harness must recover it. Null and positive control
  in one.

No network import lives at module top; ``fetch_*`` is called once to build the cache.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

MONTHS_PER_YEAR = 12

# A transparent, fixed basket of long-listed US consumer-facing large-caps that FILE an
# advertising line, spanning the intensity spectrum (NOT chosen by realised return). CIKs
# pinned so the EDGAR pull is deterministic.
#   heavy advertisers : beverages, spirits, packaged food, household, restaurants, apparel
#   light advertisers : big-box / off-price retail, telco, autos
BASKET: dict[str, int] = {
    # --- staples / packaged goods (typically heavy advertisers) ---
    "KO": 21344, "PEP": 77476, "PG": 80424, "CL": 21665, "KMB": 55785,
    "GIS": 40704, "CPB": 16732, "HSY": 47111, "MDLZ": 1103982, "KHC": 1637459,
    "CLX": 21076, "CHD": 313927, "MKC": 63754, "CAG": 23217, "SJM": 91419,
    "STZ": 16918, "TAP": 24545, "KDP": 1418135, "PM": 1413329,
    # --- restaurants / apparel / discretionary brands ---
    "YUM": 1041061, "CMG": 1058090, "DPZ": 1286681, "SBUX": 829224, "NKE": 320187,
    "DECK": 910521, "HAS": 46080, "ULTA": 1403568, "LULU": 1397187,
    # --- media / telecom (mixed intensity) ---
    "DIS": 1744489, "CMCSA": 1166691, "VZ": 732712, "T": 732717, "TMUS": 1283699,
    "EXPE": 1324424, "EBAY": 1065088, "ETSY": 1370637, "MAR": 1048286,
    # --- retail / autos (typically light advertisers) ---
    "HD": 354950, "LOW": 60667, "TGT": 27419, "WMT": 104169, "TJX": 109198,
    "ROST": 745732, "BBY": 764478, "M": 794367, "F": 37996,
}

BENCH = "SPY"      # the market (S&P 500 total-return proxy)

_EDGAR_HEADERS = {"User-Agent": "OpenAlphaLab research guillain@poulpe.us"}
# The advertising line is filed under a few different us-gaap tags. AdvertisingExpense is the
# clean one; a handful of names (GIS, CHD, NKE, CMCSA, CMG, CAG, DECK, M) only file the
# broader MarketingAndAdvertisingExpense (marketing + advertising bundled), which slightly
# OVER-states pure advertising for those names — a documented concept-heterogeneity caveat.
_ADV_CONCEPTS = [
    "AdvertisingExpense",
    "AdvertisingCosts",
    "MarketingAndAdvertisingExpense",
    "AdvertisingExpenseAndPromotion",
]
_REV_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
]


# --------------------------------------------------------------------------- #
# Real tape — EDGAR companyconcept (advertising + revenue) and yfinance (return)
# --------------------------------------------------------------------------- #
def _companyconcept(cik: int, concept: str, tries: int = 3) -> dict | None:
    """One us-gaap concept for one CIK via EDGAR companyconcept (None if 404/absent)."""
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{concept}.json"
    last = None
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers=_EDGAR_HEADERS)
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:              # 404 = firm never filed this concept
            if e.code == 404:
                return None
            last = e
            time.sleep(1.0)
        except Exception as e:                           # pragma: no cover - transient network
            last = e
            time.sleep(1.0)
    if last is not None:                                 # pragma: no cover
        return None
    return None


def _annual_from_units(units: list[dict]) -> dict[int, float]:
    """Map fiscal-year-end year -> the LARGEST full-year USD value in that year.

    Keeps only 10-K, full-year (start->end span 330-400 days) facts, and for each fiscal year
    takes the **max** value. That max is the consolidated *total*: a companyconcept response
    can carry both a total fact and smaller **segment** facts for the same period (e.g. General
    Mills tags a ~$2 B segment under ``Revenues`` while total revenue lives on the ASC-606
    tag), and the total is always the largest, so max robustly recovers the whole-company number
    instead of a segment slice.
    """
    by_year: dict[int, float] = {}
    for u in units:
        if not str(u.get("form", "")).startswith("10-K"):
            continue
        e, s = u.get("end"), u.get("start")
        if not (e and s):
            continue
        ed, sd = dt.date.fromisoformat(e), dt.date.fromisoformat(s)
        if not (330 <= (ed - sd).days <= 400):
            continue
        v = float(u["val"])
        if ed.year not in by_year or v > by_year[ed.year]:
            by_year[ed.year] = v
    return by_year


def _first_concept_annual(cik: int, concepts: list[str], pause: float) -> tuple[dict[int, float], str | None]:
    """First concept in ``concepts`` that yields any annual data (max value per year)."""
    for c in concepts:
        j = _companyconcept(cik, c, tries=2)
        time.sleep(pause)
        if not j:
            continue
        annual = _annual_from_units(j.get("units", {}).get("USD", []))
        if annual:
            return annual, c
    return {}, None


def _merged_concept_annual(cik: int, concepts: list[str], pause: float) -> dict[int, float]:
    """Union across ALL concepts, per year taking the MAX value across tags.

    For total revenue this is the right merge: the several revenue tags
    (``Revenues``, ``RevenueFromContractWithCustomer…``, ``SalesRevenueNet``) all measure the
    same whole-company quantity, and any given year may only be covered by one of them (ASC 606
    split the pre-/post-2018 eras across tags). Taking the max per year picks the consolidated
    total and skips segment-scoped ``Revenues`` facts (which are strictly smaller)."""
    out: dict[int, float] = {}
    for c in concepts:
        j = _companyconcept(cik, c, tries=2)
        time.sleep(pause)
        if not j:
            continue
        for y, v in _annual_from_units(j.get("units", {}).get("USD", [])).items():
            if y not in out or v > out[y]:
                out[y] = v
    return out


def fetch_fundamentals(basket: dict[str, int] | None = None,
                       cache_dir: str = DEFAULT_CACHE, pause: float = 0.15) -> dict:
    """Build the annual fundamentals panels (rows = fiscal year, cols = ticker) from EDGAR.

    Network-only; used once. Writes ``adv.parquet`` (annual advertising expense) and
    ``rev.parquet`` (annual revenue), plus ``adv_concepts.json`` recording which us-gaap tag
    supplied each name's advertising (for the concept-heterogeneity caveat).
    """
    basket = basket or BASKET
    adv_cols, rev_cols, adv_concept = {}, {}, {}
    for tk, cik in basket.items():
        adv, adv_c = _first_concept_annual(cik, _ADV_CONCEPTS, pause)
        rev = _merged_concept_annual(cik, _REV_CONCEPTS, pause)
        if adv:
            adv_cols[tk] = pd.Series(adv)
            adv_concept[tk] = adv_c
        if rev:
            rev_cols[tk] = pd.Series(rev)
    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    for name, cols in [("adv", adv_cols), ("rev", rev_cols)]:
        panel = pd.DataFrame(cols).sort_index()
        panel.index.name = "fy"
        panel.to_parquet(os.path.join(cache_dir, f"{name}.parquet"))
        out[name] = panel
    with open(os.path.join(cache_dir, "adv_concepts.json"), "w") as f:
        json.dump(adv_concept, f, indent=0, sort_keys=True)
    return out


def fetch_prices(basket: dict[str, int] | None = None, start: str = "2010-01-01",
                 cache_dir: str = DEFAULT_CACHE) -> dict:
    """Monthly total returns (yfinance auto-adjusted close -> pct_change) for basket + SPY.

    Network-only; used once. Writes ``returns.parquet`` (monthly total return). Drops the
    in-progress final month (a stamped run never holds a partial bar)."""
    import yfinance as yf

    basket = basket or BASKET
    tickers = list(basket.keys()) + [BENCH]
    raw = None
    for attempt in range(4):
        try:
            raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
            if raw is not None and not raw.empty:
                break
        except Exception:                                # pragma: no cover - transient network
            time.sleep(2.0 * (attempt + 1))
    if raw is None or raw.empty:                         # pragma: no cover
        raise RuntimeError("yfinance returned no price data after retries")
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    monthly = raw.resample("ME").last()
    rets = monthly.pct_change().dropna(how="all")
    cutoff = pd.Timestamp.today().normalize().to_period("M").to_timestamp()
    rets = rets[rets.index < cutoff]
    os.makedirs(cache_dir, exist_ok=True)
    rets.to_parquet(os.path.join(cache_dir, "returns.parquet"))
    return {"returns": rets}


def have_real(cache_dir: str = DEFAULT_CACHE) -> bool:
    need = ["adv.parquet", "rev.parquet", "returns.parquet"]
    return all(os.path.exists(os.path.join(cache_dir, f)) for f in need)


def load_real(cache_dir: str = DEFAULT_CACHE,
              allow_survivorship_bias: bool = False) -> dict:
    """Load the cached real tape as a dict of panels.

    Keys: ``adv`` ``rev`` (annual, rows = fiscal year), ``returns`` (monthly, cols = ticker),
    ``spy`` (monthly SPY total return aligned to ``returns``), ``adv_concepts`` (tag map).

    **Survivorship — named on the Signal axis.** The basket is *current* membership projected
    back: consumer names that were acquired or delisted (and any that stopped filing the
    advertising line) are absent. Direction, reasoned in writing: both legs — the heavy
    advertisers (staples/spirits/restaurants) and the light ones (big-box retail/telco/autos)
    — are survivors, so the survivorship tilt is largely *common to both legs* of the
    long/short rather than a clean push on the spread; but a surviving consumer basket over
    2010-2026 also embeds the era's staples-vs-retail style rotation, which the caveat carries.
    Opt in with ``allow_survivorship_bias=True``."""
    if not allow_survivorship_bias:
        raise PermissionError(
            "load_real() returns a CURRENT-membership consumer basket projected back to 2010 "
            "(survivorship: acquired/delisted names and names that dropped the advertising "
            "disclosure are absent). Pass allow_survivorship_bias=True to opt in, and carry the "
            "caveat onto the Signal axis."
        )
    adv = pd.read_parquet(os.path.join(cache_dir, "adv.parquet"))
    rev = pd.read_parquet(os.path.join(cache_dir, "rev.parquet"))
    rets = pd.read_parquet(os.path.join(cache_dir, "returns.parquet"))
    concepts_path = os.path.join(cache_dir, "adv_concepts.json")
    adv_concepts = json.load(open(concepts_path)) if os.path.exists(concepts_path) else {}
    if BENCH in rets.columns:
        spy = rets[BENCH].copy()
        rets = rets[[c for c in rets.columns if c != BENCH]]
    else:                                                # pragma: no cover - cache always has SPY
        spy = pd.Series(0.0, index=rets.index, name=BENCH)
    spy.name = "spy"
    return {"adv": adv, "rev": rev, "returns": rets, "spy": spy, "adv_concepts": adv_concepts}


# --------------------------------------------------------------------------- #
# Signal construction — advertising intensity (Adv / Sales), with a reporting lag
# --------------------------------------------------------------------------- #
def known_annual(panel: pd.DataFrame, year: int, report_lag: int = 1) -> pd.Series:
    """The most-recent annual value *known* at the start of ``year`` (fiscal year <= year-lag)."""
    avail = panel[panel.index <= year - report_lag]
    if avail.empty:
        return pd.Series(dtype=float)
    return avail.ffill().iloc[-1]


def build_signal(real: dict, report_lag: int = 1) -> pd.DataFrame:
    """Build the monthly cross-sectional advertising-intensity panel (rows = month, cols = ticker).

    ``adv_sales`` = AdvertisingExpense(FY Y-1) / Revenue(FY Y-1). A name-month is NaN (and so
    goes unranked) whenever no advertising figure is *known* at that month — the honest
    consequence of thin coverage: a firm-year with no filed advertising line simply doesn't
    enter the sort, rather than being floored to zero (unlike R&D for a bank, a *consumer*
    firm that omits the line is missing data, not a genuine zero advertiser).
    """
    adv, rev = real["adv"], real["rev"]
    months = real["returns"].index
    cols = [c for c in real["returns"].columns]
    sig = pd.DataFrame(index=months, columns=cols, dtype=float)
    for ts in months:
        y = ts.year
        adv_k = known_annual(adv, y, report_lag)
        rev_k = known_annual(rev, y, report_lag)
        for c in cols:
            a = adv_k.get(c, np.nan) if c in adv_k.index else np.nan
            v = rev_k.get(c, np.nan) if c in rev_k.index else np.nan
            if pd.notna(a) and pd.notna(v) and v > 0 and a >= 0:
                sig.at[ts, c] = float(a) / float(v)
    return sig


def coverage_stats(real: dict) -> dict:
    """How thin is the advertising coverage? Name/year counts on the fundamentals panels."""
    adv, rev = real["adv"], real["rev"]
    n_names = len([c for c in real["returns"].columns])
    adv_names = adv.shape[1]
    adv_cells = int(adv.notna().to_numpy().sum())
    adv_possible = int(adv.shape[0] * adv.shape[1])
    concepts = real.get("adv_concepts", {})
    n_marketing = sum(1 for v in concepts.values() if v and v != "AdvertisingExpense")
    return {
        "n_basket": n_names,
        "n_adv_names": int(adv_names),
        "adv_fy_min": int(adv.index.min()), "adv_fy_max": int(adv.index.max()),
        "adv_cells": adv_cells, "adv_possible": adv_possible,
        "adv_fill_pct": 100.0 * adv_cells / adv_possible if adv_possible else float("nan"),
        "n_marketing_tag": int(n_marketing),
    }


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_years: int = 15, n_assets: int = 40, edge: float = 0.0,
                    market_vol: float = 0.045, idio_vol: float = 0.075,
                    seed: int = 791) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """A deterministic monthly panel whose names carry a *known* advertising-intensity premium.

    Each name has a **persistent** advertising-intensity score ``x_i`` (high for the first
    half of the field — the heavy advertisers — low for the second), drifting mildly month to
    month. The monthly return is::

        r_it = beta_i * mkt_t + (edge / 12) * sign(x_i - median_x) + eps_it

    so ``edge`` is the *true annual* long-heavy-minus-short-light premium: a heavy advertiser
    earns ``+edge/2`` per year and a light one ``-edge/2``, so the long-short spread's
    expected value is exactly ``edge``/yr. Market betas are uniform (= 1), so at ``edge = 0``
    the two legs have identical market exposure and the long-short carries **no spurious beta
    spread** — a clean null.

    Returns ``(signal, returns, bench, truth)``.
    """
    rng = np.random.default_rng(seed)
    n_months = n_years * MONTHS_PER_YEAR
    half = n_assets // 2

    base_x = np.concatenate([
        rng.uniform(0.06, 0.16, half),                # heavy advertisers (high Adv/Sales)
        rng.uniform(0.00, 0.03, n_assets - half),     # light advertisers
    ])
    sig = np.clip(base_x[None, :] + rng.normal(0.0, 0.005, size=(n_months, n_assets)), 0.0, 1.0)

    leg_sign = np.where(np.arange(n_assets) < half, +1.0, -1.0)
    premium = leg_sign * (edge / 2.0) / MONTHS_PER_YEAR
    betas = np.ones(n_assets)

    mkt = rng.normal(0.007, market_vol, n_months)
    eps = rng.normal(0.0, idio_vol, size=(n_months, n_assets))
    # Centre each name's idio series to EXACTLY zero time-mean so edge=0 is exactly flat.
    eps -= eps.mean(axis=0, keepdims=True)
    raw = mkt[:, None] * betas[None, :] + premium[None, :] + eps

    midx = pd.period_range("2010-01", periods=n_months, freq="M").to_timestamp()
    midx = pd.DatetimeIndex(midx, name="date")
    cols = [f"N{i:02d}" for i in range(n_assets)]
    returns = pd.DataFrame(raw, index=midx, columns=cols)
    signal = pd.DataFrame(sig, index=midx, columns=cols)
    bench = pd.Series(raw.mean(axis=1), index=midx, name="bench")

    truth = {
        "n_years": n_years, "n_assets": n_assets, "edge": edge,
        "market_vol": market_vol, "idio_vol": idio_vol, "seed": seed,
        "high_cols": cols[: n_assets // 2], "low_cols": cols[n_assets // 2:],
    }
    return signal, returns, bench, truth


def fingerprint(*panels: pd.DataFrame) -> str:
    """A short content fingerprint of the panels, for the as-of stamp."""
    h = hashlib.sha1()
    for p in panels:
        a = np.ascontiguousarray(np.nan_to_num(p.to_numpy(dtype=float), nan=-9.99).ravel())
        h.update(a.tobytes())
    return h.hexdigest()[:12]
