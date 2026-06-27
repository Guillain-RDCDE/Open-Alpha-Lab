"""Data layer for Study 528 (Labor-Hiring-Rate) — the Belo-Lin-Bazdresch hiring anomaly.

Belo, Lin & Bazdresch (2014, *Journal of Political Economy*) build an investment-based
asset-pricing model with labor as a quasi-fixed factor: hiring is costly (training,
search, severance), so the **hiring rate** behaves like an investment rate. Their
prediction — the labor analogue of the Titman-Wei-Xie / Cooper-Gulen-Schill investment
anomaly — is that firms that grow employment fast subsequently **underperform**, while
firms that hire conservatively (or shrink headcount) earn higher future returns.

**Hiring-rate definition (Belo-Lin-Bazdresch 2014)**::

    HN_t = (N_t - N_{t-1}) / (0.5 * (N_t + N_{t-1}))

where ``N_t`` is the firm's full-time employee count at fiscal-year-end ``t``. The
mid-point denominator keeps the rate symmetric for hiring and firing and bounded in
[-2, 2]. High HN = an aggressive hirer; low/negative HN = a disciplined or shrinking
firm. The paper predicts HIGH HN -> LOW future returns. We sort on HN and go **long the
low-hiring (disciplined) tercile, short the high-hiring (aggressive) tercile**.

THE DATA REALITY (documented honestly):

Unlike capex or total assets, **employee counts are not in machine-readable XBRL.** Firms
report headcount on the 10-K *cover page* as narrative text, so SEC EDGAR ``companyfacts``
and the ``companyconcept``/``frames`` APIs return employee counts for only a handful of
filers (we verified ~5-11 names total across the whole market carry the
``dei:EntityNumberOfEmployees`` numeric tag). yfinance exposes only a **single current
snapshot** (``info['fullTimeEmployees']``), with no history. There is therefore *no API*
that yields a deep employee-count time series for a basket.

The honest workaround used here: a **curated employee panel** of full-time-employee counts
transcribed from each firm's published 10-K cover pages (FY2013-FY2024), a fixed ~28-name
large-cap survivor basket. These are public reported facts; the latest column is sanity-
anchored against the live yfinance ``fullTimeEmployees`` snapshot. The panel is hardcoded
(``EMPLOYEES`` below) so the real tape is fully reproducible offline. Daily adjusted prices
come from yfinance and are cached into this study's OWN ``_cache/``.

**Survivorship-bias caveat**: the basket is names *still trading and large-cap in 2026*.
Aggressive hirers that over-expanded and subsequently failed (or were acquired in distress)
are absent — so any real-tape result is an upper bound. Named explicitly on the Signal axis.

**Execution lag (one, documented)**: the fiscal-year-y headcount is public only when the
10-K lands (typically 1-3 months after year-end). We are deliberately conservative: the
year-y hiring rate drives a 12-month return that *begins 4 months after the fiscal-year-end*
and is entered at the close one trading day later. No same-bar fills, no look-ahead, and no
partial-year window is ever stamped.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))

TR_DAYS = 252

# ---------------------------------------------------------------------------
# Curated full-time-employee panel (10-K cover-page headcounts, fiscal-year-end).
# Public reported facts; the latest column is sanity-anchored to the live yfinance
# fullTimeEmployees snapshot. Calendar fiscal-year-end is used (firms with off-calendar
# fiscal years are keyed by the calendar year their fiscal year predominantly covers).
# A handful of cells are np.nan where a clean cover-page figure was not transcribed.
# ---------------------------------------------------------------------------
_YEARS = list(range(2013, 2025))  # 2013 .. 2024 inclusive

# Full-time employees (thousands of people), one row per ticker, columns = _YEARS.
EMPLOYEES_K: dict[str, list[float]] = {
    #          2013   2014   2015   2016   2017   2018   2019   2020   2021   2022   2023   2024
    "AAPL":  [ 84.0,  92.6,  110.0, 116.0, 123.0, 132.0, 137.0, 147.0, 154.0, 164.0, 161.0, 164.0],
    "MSFT":  [ 99.0,  128.0, 118.0, 114.0, 124.0, 131.0, 144.0, 163.0, 181.0, 221.0, 221.0, 228.0],
    "GOOGL": [ 47.8,  53.6,  61.8,  72.1,  80.1,  98.8,  118.9, 135.3, 156.5, 190.2, 182.5, 183.3],
    "AMZN":  [117.3, 154.1, 230.8, 341.4, 566.0, 647.5, 798.0, 1298.0,1608.0,1541.0,1525.0,1556.0],
    "NVDA":  [  8.8,   9.2,   9.2,  10.3,  11.5,  13.3,  13.8,  18.1,  22.5,  26.2,  29.6,  36.0],
    "INTC":  [107.6, 106.7, 107.3, 106.0, 102.7, 107.4, 110.8, 110.6, 121.1, 131.9, 124.8, 109.0],
    "CSCO":  [ 75.0,  74.0,  72.0,  73.7,  72.9,  74.2,  75.9,  77.5,  79.5,  83.3,  84.9,  90.4],
    "ORCL":  [122.0, 122.0, 132.0, 136.0, 138.0, 137.0, 136.0, 135.0, 132.0, 143.0, 164.0, 159.0],
    "WMT":   [2200.0,2200.0,2200.0,2300.0,2300.0,2200.0,2200.0,2300.0,2300.0,2100.0,2100.0,2100.0],
    "HD":    [365.0, 371.0, 385.0, 406.0, 413.0, 413.0, 415.0, 504.0, 490.0, 471.0, 463.0, 470.0],
    "MCD":   [440.0, 420.0, 420.0, 375.0, 235.0, 210.0, 205.0, 200.0, 200.0, 150.0, 150.0, 150.0],
    "NKE":   [ 48.0,  56.5,  62.6,  70.7,  74.4,  73.1,  76.7,  75.4,  73.3,  79.1,  83.7,  79.4],
    "SBUX":  [182.0, 191.0, 238.0, 254.0, 277.0, 291.0, 346.0, 349.0, 383.0, 402.0, 381.0, 361.0],
    "COST":  [184.0, 195.0, 205.0, 218.0, 231.0, 245.0, 254.0, 273.0, 288.0, 304.0, 316.0, 333.0],
    "CAT":   [118.5, 114.2, 105.7, 95.4,  98.4,  104.0, 102.3, 97.3,  107.7, 109.1, 113.2, 112.9],
    "UPS":   [395.0, 435.0, 444.0, 435.0, 454.0, 481.0, 495.0, 543.0, 534.0, 536.0, 500.0, 490.0],
    "XOM":   [ 75.0,  75.3,  73.5,  71.1,  69.6,  71.0,  74.9,  72.0,  63.0,  62.3,  62.0,  61.5],
    "CVX":   [ 64.6,  64.7,  61.5,  55.2,  51.9,  48.6,  48.2,  47.7,  42.6,  43.8,  45.6,  40.2],
    "JNJ":   [128.1, 126.5, 127.1, 126.4, 134.0, 135.1, 132.2, 134.5, 141.7, 152.7, 131.9, 138.1],
    "PFE":   [ 78.0,  78.3,  97.9,  96.5,  90.2,  92.4,  88.3,  78.5,  79.0,  83.0,  88.0,  81.0],
    "MRK":   [ 76.0,  70.0,  68.0,  68.0,  69.0,  69.0,  71.0,  74.0,  67.5,  68.0,  71.0,  73.0],
    "JPM":   [251.2, 241.4, 234.6, 243.4, 252.5, 256.1, 256.7, 255.4, 271.0, 293.7, 309.9, 317.2],
    "PG":    [121.0, 118.0, 110.0, 105.0, 95.0,  92.0,  97.0,  99.0,  101.0, 106.0, 107.0, 108.0],
    "KO":    [130.6, 129.2, 123.2, 100.3, 61.8,  62.6,  86.2,  80.3,  79.0,  82.5,  79.1,  69.7],
    "PEP":   [271.0, 271.0, 263.0, 264.0, 263.0, 267.0, 267.0, 291.0, 309.0, 315.0, 318.0, 319.0],
    "VZ":    [176.8, 177.3, 177.7, 160.9, 155.4, 144.5, 135.0, 132.2, 118.4, 117.1, 105.4, 100.1],
    "T":     [243.4, 243.6, 281.5, 268.5, 254.0, 268.2, 247.0, 230.8, 203.0, 160.7, 149.9, 141.1],
    "UNH":   [156.0, 170.0, 200.0, 230.0, 260.0, 300.0, 325.0, 330.0, 350.0, 400.0, 440.0, 400.0],
}

BASKET = list(EMPLOYEES_K.keys())


@dataclass(frozen=True)
class WorldTruth:
    """Records what was planted in the synthetic panel."""

    premium: float  # annualised alpha per unit of normalised hiring-rate rank (0 = null)
    # premium < 0 means high hiring -> lower returns (the Belo-Lin-Bazdresch claim).

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
    seed: int = 528,
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A firm x year panel with a known hiring-rate effect — deterministic.

    Each firm-year draws an independent hiring-rate-like score. Next year's return is::

        base_ret + premium * z(HN_rank) + noise

    where ``z(.)`` normalises ranks cross-sectionally. With ``premium = 0`` the hiring-rate
    rank carries zero signal (the null). With ``premium < 0`` the high-hiring firms
    underperform (the Belo-Lin-Bazdresch prediction). Returns ``(hn_df, fwd_ret_df, truth)``.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2007, 2007 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    hn_scores = rng.standard_normal((n_years, n_firms))
    mu_n = hn_scores.mean(axis=1, keepdims=True)
    sd_n = hn_scores.std(axis=1, keepdims=True) + 1e-9
    z = (hn_scores - mu_n) / sd_n

    fwd = base_ret + premium * z + ret_vol * rng.standard_normal((n_years, n_firms))

    hn_df = pd.DataFrame(hn_scores, index=pd.Index(years, name="year"), columns=firms)
    fwd_ret_df = pd.DataFrame(fwd, index=pd.Index(years, name="year"), columns=firms)
    return hn_df, fwd_ret_df, WorldTruth(premium)


# ---------------------------------------------------------------------------
# Real tape — curated employee panel + Yahoo daily prices, cache-first
# ---------------------------------------------------------------------------
def _price_cache() -> str:
    return os.path.join(LOCAL_CACHE, "hiring_prices.parquet")


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


def employees_frame() -> pd.DataFrame:
    """The curated full-time-employee panel as a (year x ticker) DataFrame (people).

    Values are in thousands internally; returned in absolute people for clarity. Always
    available offline (it is hardcoded), so the real-tape signal never needs the network —
    only the prices do.
    """
    df = pd.DataFrame(EMPLOYEES_K, index=pd.Index(_YEARS, name="year"))
    return df * 1000.0


def _pull_prices(tickers: list[str], start: str) -> pd.DataFrame:
    """Daily adjusted close panel (date x ticker)."""
    import yfinance as yf

    px = _retry(lambda: yf.download(tickers, start=start, auto_adjust=True,
                                    progress=False)["Close"])
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    return px.sort_index()


def fetch_panel(
    fetch: bool = False,
    start: str = "2012-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hiring-rate signal + aligned next-year returns, cache-first.

    Returns ``(hn, fwd)`` where ``hn.loc[y]`` is the hiring rate
    HN = (N_y - N_{y-1}) / (0.5*(N_y + N_{y-1})) for fiscal year y (higher = faster hirer),
    and ``fwd.loc[y]`` is the firm's 12-month forward total return measured from daily
    adjusted prices, entered one trading day after a 4-month report lag past fiscal-year-end
    (the single documented execution lag).

    The employee panel is hardcoded (always available offline). Prices are cache-only by
    default (``fetch=False``): if the price parquet is absent, returns two empty frames so
    the synthetic demo and test-suite run offline. Network is touched only on an explicit
    ``fetch=True`` (then prices are cached into this study's ``_cache/``).
    """
    ppath = _price_cache()
    if not fetch:
        if not os.path.exists(ppath):
            return pd.DataFrame(), pd.DataFrame()
        px = pd.read_parquet(ppath)
    else:
        os.makedirs(LOCAL_CACHE, exist_ok=True)
        px = _pull_prices(BASKET, start)
        px.to_parquet(ppath)

    if px.empty:
        return pd.DataFrame(), pd.DataFrame()

    from .strategy import hiring_rate, forward_year_returns

    emp = employees_frame()
    hn = hiring_rate(emp)
    fwd = forward_year_returns(px, hn.index.tolist(), tickers=list(hn.columns))
    fwd = fwd.reindex(index=hn.index, columns=hn.columns)
    return hn, fwd


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint for any DataFrame, for the as-of stamp."""
    arr = df.to_numpy(dtype=float, na_value=0.0)
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]
