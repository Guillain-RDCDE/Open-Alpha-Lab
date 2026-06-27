"""Data layer for Study 529 (Inventory-Growth).

Two tapes, one shape — a (year x ticker) panel of an inventory-growth signal and the
aligned next-year stock return:

- ``synthetic_panel`` — a *deterministic, offline* generator. A ``premium`` knob plants
  the exact effect Belo & Lin (2012) / Thomas & Zhang (2002) claim: firms that grow
  inventory aggressively underperform. ``premium = 0`` is the null (the inventory-growth
  signal carries no information). This is the study's positive control and null in one
  bottle — a *faithful-engine* check only, never evidence for a real-tape stamp.

- ``fetch_panel`` — the real tape. Pulls each survivor name's **annual balance sheet**
  (Inventory, Total Assets) and **daily adjusted prices** from Yahoo Finance, caches
  both to THIS study's own ``_cache/`` (never the shared repo cache), then builds the
  inventory-growth signal and aligned forward-year returns.

**Inventory-Growth definition (Belo & Lin 2012; Thomas & Zhang 2002)**::

    INVG_t = (Inventory_t - Inventory_{t-1}) / Total_Assets_{t-1}

Scaling the inventory *change* by lagged total assets (rather than by lagged inventory)
follows Thomas & Zhang (2002): it keeps the signal comparable across firms of different
inventory intensity and avoids exploding ratios for low-inventory bases. High INVG =
aggressive inventory build. The literature predicts HIGH INVG -> LOW future returns.

**Data limitation (named, not hidden)**: Yahoo's ``balance_sheet`` only serves ~4-5
fiscal years of annual statements per name. That yields only ~3 usable inventory-growth
observations per firm and a short panel (~3 signal years). The panel is therefore THIN —
this is documented on the Signal axis and in the results, and is the honest reason the
verdict falls where it does.

**Survivorship bias (named, not hidden)**: the basket is inventory-heavy names *still
trading in 2026*. Firms that over-built inventory and subsequently failed (exactly the
names the anomaly is built around) are absent. Real-tape results are upper bounds.

**No look-ahead**: ``invg.loc[y]`` uses fiscal years y and y-1 from the annual report;
``fwd_ret.loc[y]`` is the stock return over the 12 months *starting one trading day after*
the fiscal-year-y report date plus a reporting lag (the lag lives in ``strategy.py``).
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

# ---------------------------------------------------------------------------
# Survivor basket — inventory-heavy large caps still trading in 2026.
# Retail / consumer goods / industrials / autos: names where inventory is a
# material, decision-relevant balance-sheet line (so INVG actually means something).
# ---------------------------------------------------------------------------
BASKET = [
    # Big-box / general retail
    "WMT", "TGT", "COST", "HD", "LOW", "DG", "DLTR", "BBY", "KSS", "M",
    # Apparel / specialty retail
    "NKE", "TJX", "ROST", "GPS", "URBN", "ANF", "DKS", "WSM", "RH", "FL",
    # Consumer staples / packaged goods (inventory-heavy)
    "PG", "KO", "PEP", "CL", "KMB", "GIS", "K", "HSY", "CLX", "CAG",
    # Industrials / autos / machinery (raw-material + finished-goods inventory)
    "CAT", "DE", "F", "GM", "WHR", "PH", "ETN", "EMR", "ROK", "MMM",
]

MONTHS_PER_YEAR = 12


@dataclass(frozen=True)
class WorldTruth:
    """Records what was planted in the synthetic panel."""

    premium: float  # annualised alpha per unit of normalised INVG z-rank (0 = null).
    # premium < 0 means high inventory-growth -> lower returns (the paper's claim).

    @property
    def has_premium(self) -> bool:
        return self.premium != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core (positive control + null)
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 200,
    n_years: int = 20,
    premium: float = -0.06,
    base_ret: float = 0.10,
    ret_vol: float = 0.28,
    seed: int = 529,
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A firm x year panel with a known inventory-growth effect — deterministic given ``seed``.

    Each firm-year draws an independent standard-normal INVG z-score. Next year's return is::

        base_ret + premium * z(INVG_rank) + noise

    where ``z(.)`` normalises ranks cross-sectionally to zero mean / unit variance.
    ``premium = 0`` -> the INVG rank carries zero signal (the null). ``premium < 0`` ->
    high-INVG firms underperform (the Belo-Lin / Thomas-Zhang prediction).

    Returns ``(invg_df, fwd_ret_df, truth)``. A decorative integer-year index is used
    (never a huge ``pd.date_range``) so there is no ns-Timestamp overflow on CI.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2005, 2005 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    invg_scores = rng.standard_normal((n_years, n_firms))

    mu_n = invg_scores.mean(axis=1, keepdims=True)
    sd_n = invg_scores.std(axis=1, keepdims=True) + 1e-9
    z = (invg_scores - mu_n) / sd_n

    fwd = base_ret + premium * z + ret_vol * rng.standard_normal((n_years, n_firms))

    invg_df = pd.DataFrame(
        invg_scores, index=pd.Index(years, name="year"), columns=firms
    )
    fwd_ret_df = pd.DataFrame(
        fwd, index=pd.Index(years, name="year"), columns=firms
    )
    return invg_df, fwd_ret_df, WorldTruth(premium)


# ---------------------------------------------------------------------------
# Real tape — Yahoo balance sheets + daily prices, cached to THIS study only
# ---------------------------------------------------------------------------
def _bs_cache() -> str:
    return os.path.join(LOCAL_CACHE, "inv_balance_sheets.parquet")


def _px_cache() -> str:
    return os.path.join(LOCAL_CACHE, "inv_prices.parquet")


def _retry(fn, tries: int = 3, pause: float = 1.0):
    """Run ``fn`` with a few retries — yfinance is flaky."""
    last = None
    for k in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — network flakiness, intentionally broad
            last = exc
            time.sleep(pause * (k + 1))
    raise last  # type: ignore[misc]


def fetch_balance_sheets(
    tickers: list[str] | None = None,
    cache_dir: str | None = None,
    fetch: bool = False,
) -> pd.DataFrame:
    """Annual Inventory & Total-Assets per name, long-form, cached to this study's ``_cache/``.

    Returns a long DataFrame with columns ``[ticker, fiscal_year, inventory, total_assets]``.
    Cache-only by default (``fetch=False``): reads the cached parquet if present, else an
    empty frame so offline tooling works. Network is touched only on ``fetch=True``.
    """
    tickers = tickers or BASKET
    cache_dir = cache_dir or LOCAL_CACHE
    path = _bs_cache()

    if not fetch and os.path.exists(path):
        return pd.read_parquet(path)
    if not fetch:
        return pd.DataFrame(columns=["ticker", "fiscal_year", "inventory", "total_assets"])

    import yfinance as yf  # lazy: network only

    rows: list[dict] = []
    for tk in tickers:
        def _pull(tk=tk):
            return yf.Ticker(tk).balance_sheet

        try:
            bs = _retry(_pull)
        except Exception:  # noqa: BLE001
            continue
        if bs is None or bs.empty:
            continue
        if "Inventory" not in bs.index or "Total Assets" not in bs.index:
            continue
        inv = bs.loc["Inventory"]
        ta = bs.loc["Total Assets"]
        for col in bs.columns:
            iv = inv.get(col, np.nan)
            tv = ta.get(col, np.nan)
            if pd.isna(iv) or pd.isna(tv):
                continue
            rows.append(
                {
                    "ticker": tk,
                    "fiscal_year": int(pd.Timestamp(col).year),
                    "inventory": float(iv),
                    "total_assets": float(tv),
                }
            )

    df = pd.DataFrame(rows, columns=["ticker", "fiscal_year", "inventory", "total_assets"])
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def fetch_prices(
    tickers: list[str] | None = None,
    cache_dir: str | None = None,
    fetch: bool = False,
    start: str = "2018-01-01",
) -> pd.DataFrame:
    """Daily adjusted close per name, cached to this study's ``_cache/``.

    Returns a (date x ticker) DataFrame of adjusted closes. Cache-only by default.
    """
    tickers = tickers or BASKET
    cache_dir = cache_dir or LOCAL_CACHE
    path = _px_cache()

    if not fetch and os.path.exists(path):
        px = pd.read_parquet(path)
        if px.index.tz is not None:
            px.index = px.index.tz_localize(None)
        return px
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf  # lazy

    def _dl():
        return yf.download(
            tickers, start=start, auto_adjust=True, progress=False
        )["Close"]

    px = _retry(_dl)
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    px = px.dropna(how="all")
    os.makedirs(cache_dir, exist_ok=True)
    px.to_parquet(path)
    return px


def _annual_returns_from_prices(px: pd.DataFrame) -> pd.DataFrame:
    """Calendar-year total returns (year x ticker) from a daily adjusted-close frame.

    Year y's return = (last close of year y / last close of year y-1) - 1. Only complete
    calendar years are kept (the current, partial year is dropped — house rule).
    """
    if px.empty:
        return pd.DataFrame()
    yearly_last = px.resample("YE").last()
    yearly_last.index = yearly_last.index.year
    ret = yearly_last.pct_change()
    ret.index.name = "year"

    # Drop the current (partial) year — no partial bar in a stamped run.
    this_year = pd.Timestamp.today().year
    ret = ret[ret.index < this_year]
    return ret.dropna(how="all")


def fetch_panel(
    cache_dir: str | None = None,
    fetch: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inventory-growth signal + aligned next-year returns from the cached real tape.

    Returns ``(invg, fwd_ret)`` where ``invg.loc[y]`` is the inventory-growth signal for
    fiscal year y (higher = more aggressive inventory build) and ``fwd_ret.loc[y]`` is the
    stock return over calendar year y+1 (a conservative reporting lag — the annual report
    is assumed not actionable until the following calendar year).

    Empty frames are returned if the caches are absent, so offline tooling works.
    """
    bs = fetch_balance_sheets(cache_dir=cache_dir, fetch=fetch)
    px = fetch_prices(cache_dir=cache_dir, fetch=fetch)
    if bs.empty or px.empty:
        return pd.DataFrame(), pd.DataFrame()

    from .strategy import invg_signal

    invg = invg_signal(bs)
    yr_ret = _annual_returns_from_prices(px)

    # Align: signal from fiscal year y -> calendar-year y+1 returns.
    fwd = yr_ret.reindex(invg.index + 1)
    fwd.index = invg.index
    fwd.index.name = "year"
    # Keep only signal columns/rows that have any forward return.
    fwd = fwd.reindex(columns=invg.columns)
    return invg, fwd


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint for any DataFrame, for the as-of stamp."""
    arr = df.to_numpy(dtype=float, na_value=0.0)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
