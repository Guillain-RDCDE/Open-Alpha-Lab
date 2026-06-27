"""Data layer for Study 522 (Percent Operating Accruals) — Hafzalla-Lundholm-Van Winkle 2011.

Two tapes, one shape — a ``year x ticker`` panel of (percent-accruals signal, forward
1-year return):

- ``synthetic_panel`` — a *deterministic, offline* generator. A ``premium`` knob plants
  the exact effect HLVW (2011) claim: high percent-accrual firms underperform because their
  earnings are mostly non-cash. ``premium = 0`` is the null (the signal carries no
  information). It is the engine's positive control, never cited for a real-tape stamp.

- ``fetch_panel`` / ``load_panel`` — the **real tape**, self-pulled and self-cached to this
  study's own ``_cache/``:
    * fundamentals (Net Income, Operating Cash Flow) from **EDGAR** ``companyconcept``
      annual 10-K facts (data.sec.gov, no key) — ~19 years of history per name;
    * forward annual returns from **yfinance** daily adjusted closes (no key).

  Aligned so ``fwd_ret.loc[y]`` is calendar-year y+1's return — fundamentals from fiscal
  year y predict returns in the *following* full calendar year (a conservative reporting
  lag, no look-ahead).

**Percent operating accruals (HLVW 2011)**::

    operating accruals   = Net Income - Operating Cash Flow
    percent accruals     = (Net Income - Operating Cash Flow) / |Net Income|

    HIGH percent accruals => earnings mostly non-cash => LOW future returns
    LOW  percent accruals => earnings cash-backed      => HIGH future returns

This is distinct from **Study 231 (Sloan)** which scales the *same* operating accrual by
*average total assets*. HLVW's claim is that scaling by |earnings| sharpens the sort.

**Survivorship-bias caveat**: the basket is a fixed set of large-cap names *still trading*
in 2026. Firms that over-stated earnings, collapsed, and were delisted are absent —
precisely the high-accrual disasters the short leg would want. Every real-tape number is
an UPPER BOUND. This is named on the Signal axis and in the results.
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

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
FUND_CACHE = os.path.join(CACHE_DIR, "poa_fundamentals.csv")
PRICE_CACHE = os.path.join(CACHE_DIR, "poa_prices.csv")

_UA = {"User-Agent": "open-alpha-lab research guillain@poulpe.us"}

# Fixed ~40-name large-cap *survivor* basket: long-listed US large caps with deep, clean
# EDGAR + yfinance histories and a sector spread. Same flavour as Studies 231 / 238 / 363.
# This is a SURVIVORS basket (all still trading 2026) — survivorship is named on the Signal
# axis: a fixed surviving-names panel cannot capture firms that blew up.
BASKET = [
    "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "IBM", "CVX",
    "PFE", "MRK", "INTC", "CSCO", "HD", "MCD", "DIS", "BA", "CAT", "MMM",
    "HON", "UNH", "ORCL", "PEP", "ABT", "TXN", "COST", "LOW", "AMGN", "GS",
    "T", "VZ", "WFC", "C", "BAC", "GE", "F", "EMR", "DE", "DUK",
]

# EDGAR XBRL us-gaap concepts (annual 10-K facts).
NI_CONCEPT = "NetIncomeLoss"
CFO_CONCEPT = "NetCashProvidedByUsedInOperatingActivities"


@dataclass(frozen=True)
class WorldTruth:
    """Records what was planted in the synthetic panel."""
    premium: float  # annualised alpha per unit of normalised percent-accruals rank (0 = null)

    @property
    def has_premium(self) -> bool:
        return self.premium != 0.0


# --------------------------------------------------------------------------- #
# Synthetic panel — the deterministic offline positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_firms: int = 200,
    n_years: int = 17,
    premium: float = -0.05,
    base_ret: float = 0.10,
    ret_vol: float = 0.30,
    seed: int = 522,
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A firm x year panel with a known percent-accruals effect — deterministic given ``seed``.

    Each firm-year draws a standard-normal percent-accruals z-score. Next year's return is::

        base_ret + premium * z(acc_rank) + noise

    With ``premium = 0`` the signal carries zero information (the null). With ``premium < 0``
    high percent-accrual firms underperform (the HLVW prediction). Returns
    ``(acc_df, fwd_ret_df, truth)``; ``acc_df`` higher = more accrual / less cash backing.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2008, 2008 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    acc_scores = rng.standard_normal((n_years, n_firms))
    mu_a = acc_scores.mean(axis=1, keepdims=True)
    sd_a = acc_scores.std(axis=1, keepdims=True) + 1e-9
    z = (acc_scores - mu_a) / sd_a
    fwd = base_ret + premium * z + ret_vol * rng.standard_normal((n_years, n_firms))

    acc_df = pd.DataFrame(acc_scores, index=pd.Index(years, name="year"), columns=firms)
    fwd_ret_df = pd.DataFrame(fwd, index=pd.Index(years, name="year"), columns=firms)
    return acc_df, fwd_ret_df, WorldTruth(premium)


# --------------------------------------------------------------------------- #
# Real tape — EDGAR fundamentals + yfinance prices, self-cached
# --------------------------------------------------------------------------- #
def _ticker_cik_map() -> dict[str, str]:
    """ticker -> zero-padded 10-digit CIK, from SEC's company_tickers.json."""
    req = urllib.request.Request("https://www.sec.gov/files/company_tickers.json", headers=_UA)
    d = json.load(urllib.request.urlopen(req, timeout=30))
    return {v["ticker"]: str(v["cik_str"]).zfill(10) for v in d.values()}


def _annual_concept(cik: str, concept: str) -> dict[int, float]:
    """Annual 10-K values for one us-gaap concept keyed by fiscal-year-end calendar year.

    Picks the FY (full-year) 10-K fact for each fiscal year; for flow concepts (NI, CFO)
    that is the duration fact whose period spans ~one year and is reported on the 10-K.
    """
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"
    req = urllib.request.Request(url, headers=_UA)
    try:
        d = json.load(urllib.request.urlopen(req, timeout=30))
    except Exception:
        return {}
    facts = d.get("units", {}).get("USD", [])
    out: dict[int, float] = {}
    for f in facts:
        if f.get("form") != "10-K" or f.get("fp") != "FY":
            continue
        start, end = f.get("start"), f.get("end")
        if not start or not end:
            continue
        # Full-year duration only (~12 months)
        span_days = (pd.Timestamp(end) - pd.Timestamp(start)).days
        if not (330 <= span_days <= 400):
            continue
        yr = int(end[:4])
        # Keep the latest-filed value for that fiscal year
        out[yr] = float(f["val"])
    return out


def fetch_panel(
    basket: list[str] | None = None,
    price_start: str = "2008-01-01",
    fund_path: str = FUND_CACHE,
    price_path: str = PRICE_CACHE,
    pause: float = 0.12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull EDGAR fundamentals + yfinance prices for the basket and cache to ``_cache/``.

    Network-only; used once to build the cache. Never imported by the offline notebook cells.
    Writes a long fundamentals CSV (ticker, year, net_income, cfo) and a wide adjusted-close
    CSV (index = date, columns = tickers). Returns the same ``(acc, fwd_ret)`` as
    ``load_panel`` for convenience.
    """
    import yfinance as yf

    basket = basket or BASKET
    os.makedirs(CACHE_DIR, exist_ok=True)

    cik_map = _ticker_cik_map()
    rows = []
    for tk in basket:
        cik = cik_map.get(tk)
        if cik is None:
            continue
        ni = _annual_concept(cik, NI_CONCEPT)
        time.sleep(pause)
        cfo = _annual_concept(cik, CFO_CONCEPT)
        time.sleep(pause)
        for yr in sorted(set(ni) & set(cfo)):
            rows.append({"ticker": tk, "year": yr,
                         "net_income": ni[yr], "cfo": cfo[yr]})
    fund = pd.DataFrame(rows).sort_values(["ticker", "year"])
    fund.to_csv(fund_path, index=False)

    # Prices: daily adjusted closes -> later collapsed to annual returns.
    have = sorted(fund["ticker"].unique())
    raw = yf.download(have, start=price_start, auto_adjust=True, progress=False)["Close"]
    raw = raw.dropna(how="all")
    raw.to_csv(price_path)

    return load_panel(fund_path, price_path)


def have_real(fund_path: str = FUND_CACHE, price_path: str = PRICE_CACHE) -> bool:
    return os.path.exists(fund_path) and os.path.exists(price_path)


def _annual_returns_from_prices(price_path: str = PRICE_CACHE) -> pd.DataFrame:
    """Wide (year x ticker) calendar-year total return from daily adjusted closes.

    Drops any partial final year (the current, incomplete calendar year) so a stamped run
    never contains a fractional year — the desk forbids partial years in a stamped run.
    """
    px = pd.read_csv(price_path, index_col=0, parse_dates=True).sort_index()
    # Exclude the current incomplete calendar year (no partial years in a stamped run).
    last_date = px.index.max()
    if last_date.month != 12 or last_date.day < 28:
        px = px[px.index.year < last_date.year]
    # year-end last available close per ticker
    yr_close = px.resample("YE").last()
    yr_close.index = yr_close.index.year
    yr_close.index.name = "year"
    ret = yr_close.pct_change()
    return ret.dropna(how="all")


def load_panel(
    fund_path: str = FUND_CACHE,
    price_path: str = PRICE_CACHE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Percent-accruals signal + aligned next-year returns from the cached real tape.

    Returns ``(acc, fwd_ret)`` where ``acc.loc[y]`` is the percent-operating-accruals ratio
    for fiscal year y (higher = more accrual / less cash backing) and ``fwd_ret.loc[y]`` is
    calendar-year y+1's return (the next full year — a conservative reporting lag).

    If a cache file is absent returns two empty frames so offline tests/notebooks still run.
    """
    if not have_real(fund_path, price_path):
        return pd.DataFrame(), pd.DataFrame()

    from .strategy import percent_accruals_signal

    fund = pd.read_csv(fund_path)
    acc = percent_accruals_signal(fund)

    yr_ret = _annual_returns_from_prices(price_path)
    # align tickers
    common = [c for c in acc.columns if c in yr_ret.columns]
    acc = acc.reindex(columns=common)
    yr_ret = yr_ret.reindex(columns=common)

    # signal year y -> returns in year y+1
    fwd = yr_ret.reindex(acc.index + 1)
    fwd.index = acc.index
    fwd.index.name = "year"
    return acc, fwd


# --------------------------------------------------------------------------- #
# Fingerprint helper
# --------------------------------------------------------------------------- #
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint for any DataFrame, for the as-of stamp."""
    arr = df.to_numpy(dtype=float, na_value=0.0)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
