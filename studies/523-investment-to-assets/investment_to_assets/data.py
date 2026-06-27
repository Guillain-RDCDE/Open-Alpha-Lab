"""Data layer for Study 523 (Investment-To-Assets) — the Titman-Wei-Xie capex anomaly.

The angle here is deliberately the *capital-expenditure channel* of the investment
anomaly, not the broad total-asset-growth measure of [244 Asset-Growth]. Titman, Wei &
Xie (2004) sort firms on abnormal capital investment scaled by assets and show the
heavy investors subsequently underperform. The Fama-French CMA factor (Conservative
Minus Aggressive) packages the same idea at the factor level.

**Investment-to-Assets definition (Titman-Wei-Xie 2004, capex flavour)**::

    IA_t = CapEx_t / Total_Assets_{t-1}

High IA = the firm is ploughing a large fraction of its asset base back into property,
plant and equipment. The prediction is HIGH IA -> LOW future returns (over-investment;
the market over-extrapolates the expansion). We sort on IA and go long the low-IA
(disciplined) quintile, short the high-IA (aggressive) quintile.

Two tapes, one shape (a fiscal-year x ticker panel of IA signal + forward returns):

- ``synthetic_panel`` — a *deterministic, offline* generator. A ``premium`` knob plants
  the exact effect: high-IA firms underperform. ``premium = 0`` is the null. This is the
  study's positive control and null in one bottle.

- ``fetch_panel`` — the real tape. CapEx (``PaymentsToAcquirePropertyPlantAndEquipment``)
  and Total Assets (``Assets``) from **SEC EDGAR companyfacts** (data.sec.gov) for a fixed
  ~40-name large-cap survivor basket — a deep ~13-year fundamentals history, far longer
  than the ~5 years yfinance statements expose — plus daily adjusted prices from yfinance
  to build the forward return. Cache-first into this study's OWN ``_cache/`` so the
  test-suite and the reproducible core never need the network.

**Survivorship-bias caveat**: the basket is names *still trading and large-cap in 2026*.
Every firm survived. Heavy investors that over-invested and subsequently failed are
absent — so any real-tape result is an upper bound. Named explicitly on the Signal axis.

**Execution lag (one, documented)**: a fiscal year that ends in calendar year y is not
public until the 10-K lands (typically 1-3 months later). We are deliberately
conservative: the IA signal for fiscal year y drives the 12-month return measured over
calendar year y+1, entered at the close one trading day after the year-y+1 window opens.
No same-bar fills, no look-ahead.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))

_UA = {"User-Agent": "open-alpha-lab research guillain@poulpe.us"}
_EDGAR_TICKERS = "https://www.sec.gov/files/company_tickers.json"
_EDGAR_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
_CAPEX_TAG = "PaymentsToAcquirePropertyPlantAndEquipment"
_ASSETS_TAG = "Assets"

# A fixed ~40-name large-cap survivor basket, sector-spread so a single sector's capex
# cycle doesn't drive the sort. Names still trading and large-cap as of 2026.
BASKET = [
    # Tech / semis (capex-heavy in fabs/datacentres)
    "AAPL", "MSFT", "GOOGL", "NVDA", "AMD", "INTC", "TXN", "QCOM", "CSCO", "ORCL",
    # Consumer / retail
    "AMZN", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "COST",
    # Industrials / energy (capex-heavy capex cycles)
    "CAT", "DE", "HON", "GE", "UPS", "XOM", "CVX", "COP", "NEE", "UNP",
    # Health
    "JNJ", "PFE", "MRK", "ABT", "TMO", "UNH",
    # Financials / staples / telecom
    "JPM", "PG", "KO", "PEP", "VZ", "T",
]

TR_DAYS = 252


@dataclass(frozen=True)
class WorldTruth:
    """Records what was planted in the synthetic panel."""

    premium: float  # annualised alpha per unit of normalised IA rank (0 = null)
    # premium < 0 means high investment-to-assets -> lower returns (the paper's claim).

    @property
    def has_premium(self) -> bool:
        return self.premium != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 200,
    n_years: int = 18,
    premium: float = -0.06,
    base_ret: float = 0.10,
    ret_vol: float = 0.28,
    seed: int = 523,
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A firm x year panel with a known investment-to-assets effect — deterministic.

    Each firm-year draws an independent IA-like score. Next year's return is::

        base_ret + premium * z(IA_rank) + noise

    where ``z(.)`` normalises ranks cross-sectionally. With ``premium = 0`` the IA rank
    carries zero signal (the null). With ``premium < 0`` the high-IA firms underperform
    (the Titman-Wei-Xie prediction). Returns ``(ia_df, fwd_ret_df, truth)``.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2007, 2007 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    ia_scores = rng.standard_normal((n_years, n_firms))
    mu_n = ia_scores.mean(axis=1, keepdims=True)
    sd_n = ia_scores.std(axis=1, keepdims=True) + 1e-9
    z = (ia_scores - mu_n) / sd_n

    fwd = base_ret + premium * z + ret_vol * rng.standard_normal((n_years, n_firms))

    ia_df = pd.DataFrame(ia_scores, index=pd.Index(years, name="year"), columns=firms)
    fwd_ret_df = pd.DataFrame(fwd, index=pd.Index(years, name="year"), columns=firms)
    return ia_df, fwd_ret_df, WorldTruth(premium)


# ---------------------------------------------------------------------------
# Real tape — Yahoo Finance, cache-first into THIS study's _cache only
# ---------------------------------------------------------------------------
def _fund_cache() -> str:
    return os.path.join(LOCAL_CACHE, "ia_fundamentals.parquet")


def _price_cache() -> str:
    return os.path.join(LOCAL_CACHE, "ia_prices.parquet")


def _retry(fn, tries: int = 3, pause: float = 1.5):
    """Tiny retry wrapper around a flaky yfinance call."""
    last = None
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — yfinance raises a zoo of exceptions
            last = e
            time.sleep(pause * (k + 1))
    raise last  # type: ignore[misc]


def _http_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _ticker_cik_map() -> dict[str, int]:
    d = _retry(lambda: _http_json(_EDGAR_TICKERS))
    return {v["ticker"]: int(v["cik_str"]) for v in d.values()}


def _annual_flow(units: list[dict]) -> dict[int, float]:
    """Annual (~1-yr duration) values from 10-K FY rows, keyed by fiscal-year-end year."""
    out: dict[int, float] = {}
    for u in units:
        if u.get("form") != "10-K" or u.get("fp") != "FY":
            continue
        try:
            start = pd.Timestamp(u["start"])
            end = pd.Timestamp(u["end"])
        except (KeyError, ValueError):
            continue
        if (end - start).days < 330:  # keep full-year durations only
            continue
        out[end.year] = float(u["val"])  # latest filing wins
    return out


def _annual_instant(units: list[dict]) -> dict[int, float]:
    """Year-end instant values from 10-K FY rows, keyed by fiscal-year-end year."""
    out: dict[int, float] = {}
    for u in units:
        if u.get("form") != "10-K" or u.get("fp") != "FY":
            continue
        try:
            end = pd.Timestamp(u["end"])
        except (KeyError, ValueError):
            continue
        out[end.year] = float(u["val"])
    return out


def _pull_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """Long-form (ticker, year, capex, assets) frame from SEC EDGAR companyfacts.

    Pulls ``PaymentsToAcquirePropertyPlantAndEquipment`` (annual capex flow) and ``Assets``
    (year-end total assets) per name, keyed by fiscal-year-end year. ~13 years deep.
    """
    cik = _ticker_cik_map()
    rows: list[dict] = []
    for tk in tickers:
        c = cik.get(tk)
        if c is None:
            continue
        try:
            facts = _retry(lambda c=c: _http_json(_EDGAR_FACTS.format(cik=c)))
        except Exception:  # noqa: BLE001
            continue
        gaap = facts.get("facts", {}).get("us-gaap", {})
        if _CAPEX_TAG not in gaap or _ASSETS_TAG not in gaap:
            continue
        cap_units = gaap[_CAPEX_TAG]["units"].get("USD", [])
        ast_units = gaap[_ASSETS_TAG]["units"].get("USD", [])
        cap = _annual_flow(cap_units)
        ast = _annual_instant(ast_units)
        for yr in sorted(set(cap) | set(ast)):
            rows.append(dict(ticker=tk, year=int(yr),
                             capex=cap.get(yr, np.nan), assets=ast.get(yr, np.nan)))
        time.sleep(0.15)  # be polite to data.sec.gov
    return pd.DataFrame(rows)


def _pull_prices(tickers: list[str], start: str) -> pd.DataFrame:
    """Daily adjusted close panel (date x ticker)."""
    import yfinance as yf

    px = _retry(lambda: yf.download(tickers, start=start, auto_adjust=True,
                                    progress=False)["Close"])
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    return px.sort_index()


def fetch_panel(
    fetch: bool = False,
    start: str = "2007-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Investment-to-assets signal + aligned next-year returns, cache-first.

    Returns ``(ia, fwd)`` where ``ia.loc[y]`` is the IA = CapEx_y / Assets_{y-1} signal
    for fiscal year y (higher = heavier investor), and ``fwd.loc[y]`` is the firm's
    calendar-year y+1 total return measured from daily adjusted prices, entered one
    trading day after the y+1 window opens (the single documented execution lag).

    Cache-only by default (``fetch=False``): if the parquet caches are absent returns two
    empty frames so the synthetic demo and test-suite run offline. Network is touched only
    on an explicit ``fetch=True`` (then results are cached into this study's ``_cache/``).
    """
    fpath, ppath = _fund_cache(), _price_cache()
    if not fetch:
        if not (os.path.exists(fpath) and os.path.exists(ppath)):
            return pd.DataFrame(), pd.DataFrame()
        fund = pd.read_parquet(fpath)
        px = pd.read_parquet(ppath)
    else:
        os.makedirs(LOCAL_CACHE, exist_ok=True)
        fund = _pull_fundamentals(BASKET)
        px = _pull_prices(BASKET, start)
        fund.to_parquet(fpath)
        px.to_parquet(ppath)

    if fund.empty or px.empty:
        return pd.DataFrame(), pd.DataFrame()

    from .strategy import ia_signal, forward_year_returns

    ia = ia_signal(fund)
    fwd = forward_year_returns(px, ia.index.tolist(), tickers=list(ia.columns))
    fwd = fwd.reindex(index=ia.index, columns=ia.columns)
    return ia, fwd


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint for any DataFrame, for the as-of stamp."""
    arr = df.to_numpy(dtype=float, na_value=0.0)
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]
