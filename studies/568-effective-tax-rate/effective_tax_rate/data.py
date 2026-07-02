"""Data layer for Study 568 (Effective-Tax-Rate) — the tax-rate return anomaly.

Two tapes, one shape — a ``year x ticker`` panel of (ETR signal, forward 1-year return):

- ``synthetic_panel`` — a *deterministic, offline* generator. A single ``premium`` knob plants
  the effect: ``premium != 0`` makes future return depend on the (z-scored) ETR level;
  ``premium = 0`` is the null (ETR carries no information). It is the engine's positive control,
  never cited for a real-tape stamp.

- ``fetch_panel`` / ``load_panel`` — the **real tape**, self-pulled and self-cached to this
  study's own ``_cache/``:
    * fundamentals (income-tax expense, pretax income) from **EDGAR** ``companyconcept`` annual
      10-K facts (data.sec.gov, no key) — ~15-18 years of history per name;
    * forward annual returns from **yfinance** daily adjusted closes (no key).

  Aligned so ``fwd_ret.loc[y]`` is calendar-year y+1's return — fundamentals from fiscal year y
  predict returns in the *following* full calendar year (a conservative reporting lag, no
  look-ahead: the 10-K is not public until well into y+1).

**Effective tax rate (ETR)**::

    ETR = income_tax_expense / pretax_income        (only when pretax_income > 0)

    LOW  ETR  => tax-efficient / aggressive shields  => ??? future returns  (sign is the question)
    HIGH ETR  => pays full freight                    => ??? future returns

We also compute the **change in ETR** (Δ ETR year-on-year) as a second signal, since some of the
literature argues *increases* in ETR (a shield reversing) predict weakness.

**Survivorship-bias caveat**: the basket is a fixed set of large-cap names *still trading* in
2026. Firms whose aggressive tax positions blew up and were delisted are absent. Every real-tape
number is therefore an UPPER BOUND on any red-flag effect and is named on the Signal axis. A
synthetic-only or survivor-only study can never earn `REAL` — that needs a robust ``t >= 2`` on a
survivorship-free tape.
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
FUND_CACHE = os.path.join(CACHE_DIR, "etr_fundamentals.csv")
PRICE_CACHE = os.path.join(CACHE_DIR, "etr_prices.csv")

_UA = {"User-Agent": "open-alpha-lab research guillain@poulpe.us"}

# Fixed ~40-name large-cap *survivor* basket: long-listed US large caps with deep, clean EDGAR +
# yfinance histories and a sector spread across tax regimes — tech (heavy offshore shields),
# pharma, staples, industrials, banks, energy, utilities. Same flavour as Studies 231/521/522.
# This is a SURVIVORS basket (all still trading 2026); survivorship is named on the Signal axis.
BASKET = [
    "AAPL", "MSFT", "GOOGL", "ORCL", "IBM", "INTC", "CSCO", "TXN", "QCOM",  # tech (offshore)
    "JNJ", "PFE", "MRK", "ABT", "AMGN", "BMY",                             # pharma
    "PG", "KO", "PEP", "WMT", "COST", "MCD",                               # staples/consumer
    "XOM", "CVX", "COP",                                                   # energy
    "JPM", "BAC", "WFC", "C", "GS",                                        # banks
    "CAT", "GE", "BA", "HON", "MMM", "DE", "EMR",                          # industrials
    "T", "VZ", "DIS",                                                      # telecom/media
    "DUK", "SO",                                                           # utilities
]
_seen: set[str] = set()
BASKET = [t for t in BASKET if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]

# EDGAR XBRL us-gaap concepts (annual 10-K facts).
TAX_CONCEPT = "IncomeTaxExpenseBenefit"
# Pretax income concept changed name over the years; try newer first, then the older long name.
PRETAX_CONCEPTS = [
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesAndExtraordinaryItems",
]


@dataclass(frozen=True)
class WorldTruth:
    """Records what was planted in the synthetic panel."""

    premium: float  # annualised alpha per unit of z-scored ETR level (0 = null)

    @property
    def has_premium(self) -> bool:
        return self.premium != 0.0


# --------------------------------------------------------------------------- #
# Synthetic panel — the deterministic offline positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_firms: int = 200,
    n_years: int = 16,
    premium: float = 0.05,
    base_ret: float = 0.10,
    ret_vol: float = 0.30,
    seed: int = 568,
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A firm x year panel with a known ETR-level effect — deterministic given ``seed``.

    Each firm-year draws a latent ETR level (a Beta so most firms cluster near a statutory rate
    with a low-ETR tail of aggressive avoiders). Next year's return is::

        base_ret + premium * z(etr) + noise

    With ``premium = 0`` the ETR carries zero information (the null). With ``premium < 0`` LOW-ETR
    firms *out*-earn (the quality / tax-avoidance-premium story: low tax -> high return, so return
    falls with ETR). With ``premium > 0`` HIGH-ETR firms out-earn (low ETR is a red flag). Returns
    ``(etr_df, fwd_ret_df, truth)`` — ``etr_df`` higher = pays more tax.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2009, 2009 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    # Latent ETR level per firm-year: centred ~21-35%, right/left spread, a low-ETR tail.
    etr = 0.10 + 0.30 * rng.beta(4.0, 3.0, size=(n_years, n_firms))
    mu_e = etr.mean(axis=1, keepdims=True)
    sd_e = etr.std(axis=1, keepdims=True) + 1e-9
    z = (etr - mu_e) / sd_e
    fwd = base_ret + premium * z + ret_vol * rng.standard_normal((n_years, n_firms))

    etr_df = pd.DataFrame(etr, index=pd.Index(years, name="year"), columns=firms)
    fwd_ret_df = pd.DataFrame(fwd, index=pd.Index(years, name="year"), columns=firms)
    return etr_df, fwd_ret_df, WorldTruth(premium)


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

    Picks the FY (full-year) 10-K fact for each fiscal year — the duration fact whose period
    spans ~one year and is reported on the 10-K. Keeps the latest-filed value for that year.
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
        span_days = (pd.Timestamp(end) - pd.Timestamp(start)).days
        if not (330 <= span_days <= 400):  # full-year duration only (~12 months)
            continue
        out[int(end[:4])] = float(f["val"])
    return out


def _pretax(cik: str) -> dict[int, float]:
    """Merge the pretax-income concept variants (name changed over the years) by fiscal year."""
    merged: dict[int, float] = {}
    for concept in PRETAX_CONCEPTS:
        vals = _annual_concept(cik, concept)
        for yr, v in vals.items():
            merged.setdefault(yr, v)  # prefer earlier (newer-named) concept
        time.sleep(0.05)
    return merged


def fetch_panel(
    basket: list[str] | None = None,
    price_start: str = "2008-01-01",
    fund_path: str = FUND_CACHE,
    price_path: str = PRICE_CACHE,
    pause: float = 0.12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull EDGAR fundamentals + yfinance prices for the basket and cache to ``_cache/``.

    Network-only; used once to build the cache. Never imported by the offline notebook cells.
    Writes a long fundamentals CSV (ticker, year, tax_expense, pretax_income) and a wide
    adjusted-close CSV. Returns the same ``(etr, fwd_ret)`` as ``load_panel`` for convenience.
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
        tax = _annual_concept(cik, TAX_CONCEPT)
        time.sleep(pause)
        pretax = _pretax(cik)
        time.sleep(pause)
        for yr in sorted(set(tax) & set(pretax)):
            rows.append({"ticker": tk, "year": yr,
                         "tax_expense": tax[yr], "pretax_income": pretax[yr]})
    fund = pd.DataFrame(rows).sort_values(["ticker", "year"])
    fund.to_csv(fund_path, index=False)

    have = sorted(fund["ticker"].unique())
    raw = yf.download(have, start=price_start, auto_adjust=True, progress=False)["Close"]
    raw = raw.dropna(how="all")
    raw.to_csv(price_path)

    return load_panel(fund_path, price_path)


def have_real(fund_path: str = FUND_CACHE, price_path: str = PRICE_CACHE) -> bool:
    return os.path.exists(fund_path) and os.path.exists(price_path)


def _annual_returns_from_prices(price_path: str = PRICE_CACHE) -> pd.DataFrame:
    """Wide (year x ticker) calendar-year total return from daily adjusted closes.

    Drops any partial final year (the current, incomplete calendar year) so a stamped run never
    contains a fractional year — the desk forbids partial years in a stamped run.
    """
    px = pd.read_csv(price_path, index_col=0, parse_dates=True).sort_index()
    last_date = px.index.max()
    if last_date.month != 12 or last_date.day < 28:
        px = px[px.index.year < last_date.year]
    yr_close = px.resample("YE").last()
    yr_close.index = yr_close.index.year
    yr_close.index.name = "year"
    ret = yr_close.pct_change()
    return ret.dropna(how="all")


def load_panel(
    fund_path: str = FUND_CACHE,
    price_path: str = PRICE_CACHE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ETR signal + aligned next-year returns from the cached real tape.

    Returns ``(etr, fwd_ret)`` where ``etr.loc[y]`` is the effective tax rate for fiscal year y
    (higher = pays more tax) and ``fwd_ret.loc[y]`` is calendar-year y+1's return (a conservative
    reporting lag). If a cache file is absent returns two empty frames so offline tests/notebooks
    still run.
    """
    if not have_real(fund_path, price_path):
        return pd.DataFrame(), pd.DataFrame()

    from .strategy import etr_signal

    fund = pd.read_csv(fund_path)
    etr = etr_signal(fund)

    yr_ret = _annual_returns_from_prices(price_path)
    common = [c for c in etr.columns if c in yr_ret.columns]
    etr = etr.reindex(columns=common)
    yr_ret = yr_ret.reindex(columns=common)

    fwd = yr_ret.reindex(etr.index + 1)
    fwd.index = etr.index
    fwd.index.name = "year"
    return etr, fwd


# --------------------------------------------------------------------------- #
# Fingerprint helper
# --------------------------------------------------------------------------- #
def fingerprint(df) -> str:
    """A short content fingerprint for any DataFrame/Series, for the as-of stamp."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    arr = df.to_numpy(dtype=float, na_value=0.0)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
