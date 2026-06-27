"""Data layer for Study 527 (Organizational-Capital).

Two tapes, one schema -- annual fundamentals (SG&A and total assets) plus a daily price
panel for the basket:

- ``synthetic_panel`` -- a *deterministic, offline* generator. A tunable ``oc_premium``
  knob controls how strongly high-organizational-capital firms outperform low-OC firms.
  ``oc_premium = 0`` is the null hypothesis. Tests/notebooks never touch the network for it.

- ``fetch_fundamentals`` -- annual SG&A and total assets from **EDGAR** (data.sec.gov XBRL
  companyfacts), cached in the study's own ``_cache/`` dir. EDGAR gives ~18-19 years of
  10-K history for large-caps -- enough to run a real perpetual inventory. Where a firm
  reports SG&A split into "Selling & Marketing" + "General & Administrative", we sum them.

- ``fetch_prices`` -- real daily adjusted-close prices from yfinance, cached to ``_cache/``.

No look-ahead: the org-capital ranking at the start of year Y uses only fundamentals whose
fiscal year *ended on or before* June of year Y (annual reports are public with a lag); the
portfolio is then held over year Y. Exactly one execution lag is applied in ``strategy``.

The basket is **survivorship-biased** -- it is current large-caps that still trade in 2026.
Firms that delisted (the natural high-risk tail) are absent; positive results are upper bounds.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

_UA = "open-alpha-lab research guillain@poulpe.us"

# Fixed large-cap survivor basket with EDGAR CIKs. ~40 names spanning sectors so the
# cross-section has real dispersion in SG&A intensity (consumer staples, retail and tech
# spend heavily on SG&A; capital-intensive energy/industrials less so).
# (ticker -> CIK). Tickers also used for yfinance price pulls.
BASKET: dict[str, int] = {
    "AAPL": 320193, "KO": 21344, "PG": 80424, "WMT": 104169, "JNJ": 200406,
    "PEP": 77476, "MCD": 63908, "NKE": 320187, "HD": 354950, "LOW": 60667,
    "TGT": 27419, "COST": 909832, "SBUX": 829224, "CL": 21665, "KMB": 55785,
    "GIS": 40704, "K": 55067, "CLX": 21076, "EL": 1001250, "MDLZ": 1103982,
    "DIS": 1744489, "CMCSA": 1166691, "VZ": 732712, "T": 732717, "ORCL": 1341439,
    "IBM": 51143, "CSCO": 858877, "INTC": 50863, "HPQ": 47217, "TXN": 97476,
    "CAT": 18230, "DE": 315189, "MMM": 66740, "HON": 773840, "GE": 40545,
    "EMR": 32604, "ITW": 49826, "DOV": 29905, "PH": 76334, "ROK": 1024478,
}
# CIKs verified against the SEC ticker map; the basket resolves to whatever EDGAR
# actually returns SG&A + Assets for (a few thin names drop out naturally).


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    oc_premium: float  # annual outperformance of high-OC over low-OC firms

    @property
    def has_premium(self) -> bool:
        return self.oc_premium != 0.0


# ---------------------------------------------------------------------------
# EDGAR fundamentals
# ---------------------------------------------------------------------------
def _edgar_facts(cik: int, retries: int = 3) -> dict | None:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(0.6 * (attempt + 1))
        except Exception:  # noqa: BLE001
            time.sleep(0.6 * (attempt + 1))
    return None


def _annual_duration(gaap: dict, concept: str) -> dict[int, float]:
    """Full fiscal-year values for a duration concept (e.g. SG&A), keyed by fiscal-year end year."""
    out: dict[int, float] = {}
    node = gaap.get(concept)
    if not node:
        return out
    for u in node["units"].get("USD", []):
        if u.get("form") != "10-K" or "start" not in u:
            continue
        try:
            s = _dt.date.fromisoformat(u["start"])
            e = _dt.date.fromisoformat(u["end"])
        except ValueError:
            continue
        if 350 <= (e - s).days <= 380:  # full year only
            out[e.year] = float(u["val"])  # last write wins (latest restatement)
    return out


def _annual_instant(gaap: dict, concept: str) -> dict[int, float]:
    """Year-end values for an instant concept (e.g. Total Assets)."""
    out: dict[int, float] = {}
    node = gaap.get(concept)
    if not node:
        return out
    for u in node["units"].get("USD", []):
        if u.get("form") != "10-K" or u.get("fp") != "FY":
            continue
        try:
            e = _dt.date.fromisoformat(u["end"])
        except ValueError:
            continue
        out[e.year] = float(u["val"])
    return out


def fetch_fundamentals(cache_dir: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Annual SG&A and total assets for the basket, from EDGAR.

    Cache-first: reads ``<cache_dir>/oc_fundamentals.parquet`` if present. On miss, pulls
    EDGAR companyfacts for each CIK, extracts annual SG&A (combined, or Selling&Marketing +
    G&A summed) and year-end total assets, and writes the parquet.

    Returns a long frame: columns ``ticker, fyear, sga, assets``. Empty frame if both cache
    and network fail (CI).
    """
    path = os.path.join(cache_dir, "oc_fundamentals.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)

    rows: list[dict] = []
    for ticker, cik in BASKET.items():
        facts = _edgar_facts(cik)
        time.sleep(0.25)  # be polite to data.sec.gov
        if not facts or "facts" not in facts or "us-gaap" not in facts["facts"]:
            continue
        gaap = facts["facts"]["us-gaap"]

        sga = _annual_duration(gaap, "SellingGeneralAndAdministrativeExpense")
        if not sga:
            # Fall back to the split components summed.
            mkt = _annual_duration(gaap, "SellingAndMarketingExpense")
            ga = _annual_duration(gaap, "GeneralAndAdministrativeExpense")
            years = set(mkt) & set(ga)
            sga = {y: mkt[y] + ga[y] for y in years}
            if not sga:  # last resort: G&A alone
                sga = _annual_duration(gaap, "GeneralAndAdministrativeExpense")

        assets = _annual_instant(gaap, "Assets")
        if not sga or not assets:
            continue

        for y in sorted(set(sga) & set(assets)):
            if sga[y] > 0 and assets[y] > 0:
                rows.append({"ticker": ticker, "fyear": y, "sga": sga[y], "assets": assets[y]})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(path)
    return df


# ---------------------------------------------------------------------------
# yfinance prices
# ---------------------------------------------------------------------------
def fetch_prices(
    tickers: list[str] | None = None,
    cache_dir: str = DEFAULT_CACHE,
    start: str = "2009-01-01",
    end: str = "2025-12-31",
    retries: int = 3,
) -> pd.DataFrame:
    """Daily adjusted-close prices for the basket.

    Cache-first: reads ``<cache_dir>/oc_prices.parquet`` if present, otherwise pulls from
    yfinance (with retries) and writes it. Returns ``pd.DataFrame()`` if both fail.
    """
    path = os.path.join(cache_dir, "oc_prices.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)

    tickers = tickers or list(BASKET.keys())
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            import yfinance as yf

            raw = yf.download(
                tickers, start=start, end=end,
                auto_adjust=True, progress=False, threads=True,
            )["Close"]
            if raw.empty:
                raise RuntimeError("empty yfinance frame")
            prices = raw.dropna(how="all")
            coverage = prices.notna().mean()
            prices = prices.loc[:, coverage >= 0.20]
            os.makedirs(cache_dir, exist_ok=True)
            prices.to_parquet(path)
            return prices
        except Exception as e:  # noqa: BLE001
            last_exc = e
            time.sleep(1.0 * (attempt + 1))
    _ = last_exc
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 40,
    n_years: int = 18,
    oc_premium: float = 0.05,
    seed: int = 527,
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A reproducible (fundamentals, monthly-returns) pair with a tunable OC premium.

    Each firm gets a persistent SG&A-intensity level (some firms structurally spend more on
    SG&A relative to assets). We build org-capital by perpetual inventory inside the panel,
    so the planted signal is exactly the variable the strategy sorts on. A firm's monthly
    return is::

        r = mkt + oc_premium * z(oc/assets) / 12 + idio

    where ``z`` is the cross-sectional standardised org-capital ratio. ``oc_premium = 0`` is
    the null. Returns ``(fundamentals_long, monthly_returns_wide, truth)``.

    Decorative monthly index uses ``pd.period_range`` (never ``date_range`` -- ns overflow).
    """
    rng = np.random.default_rng(seed)
    firms = [f"F{j:03d}" for j in range(n_firms)]
    base_year = 2008

    # Persistent SG&A intensity (fraction of assets spent on SG&A each year).
    intensity = rng.uniform(0.03, 0.40, size=n_firms)
    assets0 = rng.lognormal(mean=np.log(50e9), sigma=0.6, size=n_firms)
    asset_growth = rng.normal(0.05, 0.03, size=n_firms)

    # Vectorised fundamentals: assets compound by firm growth; SG&A = intensity * assets * noise.
    k_grid = np.arange(1, n_years + 1)[None, :]                       # (1 x years)
    assets_mat = assets0[:, None] * (1.0 + asset_growth)[:, None] ** k_grid  # (firms x years)
    sga_noise = 1.0 + rng.normal(0, 0.05, size=(n_firms, n_years))
    sga_mat = np.maximum(intensity[:, None] * assets_mat * sga_noise, 1.0)
    fyears = base_year + np.arange(n_years)
    fundamentals = pd.DataFrame({
        "ticker": np.repeat(firms, n_years),
        "fyear": np.tile(fyears, n_firms),
        "sga": sga_mat.ravel(),
        "assets": assets_mat.ravel(),
    })

    # Monthly returns: 12 months per fundamental year, indexed by decorative period_range.
    n_months = n_years * 12
    months = pd.period_range(f"{base_year + 1}-01", periods=n_months, freq="M")
    mkt = rng.normal(0.08 / 12, 0.16 / np.sqrt(12), size=n_months)

    # Org-capital ratio per firm-year (perpetual inventory) -> z-score -> alpha.
    oc = _perpetual_inventory_truth(fundamentals)
    # cross-sectional z of oc/assets each year
    ratio = oc.copy()
    z = ratio.groupby("fyear")["oc_assets"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-12))
    zmap = ratio.assign(z=z).set_index(["ticker", "fyear"])["z"]

    # Build a (n_months x n_firms) z matrix from the per-(firm, year) z scores, then vectorise.
    z_mat = np.zeros((n_months, n_firms))
    month_years = np.asarray(months.year)
    for j, f in enumerate(firms):
        col = np.array([zmap.get((f, int(y)), 0.0) for y in month_years])
        z_mat[:, j] = col
    idio = rng.normal(0, 0.30 / np.sqrt(12), size=(n_months, n_firms))
    rets = mkt[:, None] + oc_premium * z_mat / 12.0 + idio

    monthly = pd.DataFrame(rets, index=months, columns=firms)
    return fundamentals, monthly, WorldTruth(oc_premium)


def _perpetual_inventory_truth(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Helper for the synthetic generator: OC stock + oc/assets for the planted signal."""
    from .strategy import org_capital_stock  # local import to avoid cycle

    return org_capital_stock(fundamentals)


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame | pd.Series) -> str:
    """Short content fingerprint of a frame (for the as-of stamp in docs/results.md)."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    arr = np.ascontiguousarray(df.fillna(0.0).to_numpy(dtype="float64", na_value=0.0))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
