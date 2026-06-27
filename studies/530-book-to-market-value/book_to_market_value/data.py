"""Data layer for Study 530 (Book-To-Market-Value).

Two tapes, one shape -- a (fiscal-year x ticker) panel of book-to-market ratios paired
with the *next* calendar year's total return:

- ``synthetic_panel`` -- a *deterministic, offline* generator. A tunable ``hml_premium``
  knob controls how strongly high book-to-market (cheap/value) names beat low-B/M
  (expensive/growth) names. ``hml_premium = 0`` is the null: the value sort becomes a
  coin flip, so tests can assert the edge only appears when planted. Never touches the
  network.

- ``fetch_panel`` -- the real tape. For each name in a fixed large-cap survivor basket we
  pull yfinance ``balance_sheet`` (``Stockholders Equity`` = book value, ``Ordinary Shares
  Number`` = share count) and year-end adjusted prices. Book-to-market =
  book_equity / (shares x year-end price). The signal formed on fiscal-year-Y book equity
  is paired with the calendar-year-(Y+1) total return, so the trade is entered *after* the
  10-K is public (one full year of reporting lag -- the desk standard for annual
  cross-sectional work). Cached to this study's own ``_cache/`` ONLY.

**DATA LIMITATION (named everywhere):** yfinance serves only ~4-5 annual balance sheets per
ticker, so the real B/M panel is SHORT (a handful of cross-sectional years, ~2021-2024
signal -> 2022-2025 returns). The cross-section is wide (~40 names) but the time series is
thin; the HAC t-stat on so few annual hedge observations is low-powered by construction.

**Survivorship bias** is explicit: the basket is the names still trading in 2026 (current
large-caps projected backwards). Failed/delisted value traps -- the natural long candidates
for a deep-value strategy -- are absent. All positive magnitudes are upper-bound estimates.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

# Fixed large-cap survivor basket (~40 names, sector-diversified so the value sort has
# genuine cross-sectional dispersion: cheap financials/energy/industrials vs expensive
# tech/consumer). Names still trading in 2026 -> survivorship-biased (named explicitly).
UNIVERSE = [
    # Mega-cap growth / expensive (typically LOW B/M)
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "NFLX", "ADBE", "CRM", "ORCL",
    # Financials (typically HIGH B/M -- the value corner)
    "JPM", "BAC", "WFC", "C", "GS", "USB", "PNC", "MET", "PRU", "ALL",
    # Energy / materials / industrials (cyclical value)
    "XOM", "CVX", "COP", "SLB", "CAT", "DE", "GE", "HON", "MMM", "NUE",
    # Healthcare / staples / utilities (mixed)
    "JNJ", "PFE", "MRK", "KO", "PG", "WMT", "VZ", "T", "SO", "DUK",
]
_seen: set[str] = set()
UNIVERSE = [t for t in UNIVERSE if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    hml_premium: float  # extra annual return per unit of cross-sectional B/M z-score

    @property
    def has_premium(self) -> bool:
        return self.hml_premium != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 200,
    n_years: int = 24,
    hml_premium: float = 0.05,
    base_ret: float = 0.09,
    ret_vol: float = 0.28,
    seed: int = 530,
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A reproducible firm x year panel with a known book-to-market (value) effect.

    Each firm-year draws a raw B/M characteristic from a log-normal (positive, right-skewed
    -- the empirical shape of book-to-market). Within each year the characteristic is
    cross-sectionally z-scored. Next year's return is:

        ret_{t+1} = base_ret + hml_premium * z_bm + noise

    so high-B/M (value) names earn ``hml_premium`` more per z-score. Setting
    ``hml_premium = 0`` is the null -- the value quintile sort becomes a coin flip.

    Returns ``(bm_panel, fwd_ret, truth)`` -- both year-indexed (period_range to dodge the
    ns-overflow trap), columns = firm identifiers, values = decimal B/M and decimal returns.
    """
    rng = np.random.default_rng(seed)
    # period_range -> decorative monthly/annual index without ns overflow on CI pandas.
    years = pd.period_range("2000", periods=n_years, freq="Y")
    firms = [f"F{j:03d}" for j in range(n_firms)]

    raw_bm = rng.lognormal(mean=-0.7, sigma=0.5, size=(n_years, n_firms))  # ~0.2-1.5

    def _zscore(a: np.ndarray) -> np.ndarray:
        mu = a.mean(axis=1, keepdims=True)
        sigma = a.std(axis=1, keepdims=True) + 1e-9
        return (a - mu) / sigma

    z_bm = _zscore(raw_bm)
    noise = ret_vol * rng.standard_normal((n_years, n_firms))
    fwd = base_ret + hml_premium * z_bm + noise

    idx = pd.Index(years, name="year")
    bm_panel = pd.DataFrame(raw_bm, index=idx, columns=firms)
    fwd_panel = pd.DataFrame(fwd, index=idx, columns=firms)
    return bm_panel, fwd_panel, WorldTruth(hml_premium)


# ---------------------------------------------------------------------------
# Real panel -- yfinance balance sheet + prices, study-local cache
# ---------------------------------------------------------------------------
_BM_PATH = "bm_panel.parquet"
_FWD_PATH = "fwd_ret.parquet"

# yfinance balance-sheet row labels we need.
_EQUITY_ROW = "Stockholders Equity"
_SHARES_ROW = "Ordinary Shares Number"
_LAST_FWD_YEAR = 2025  # 2026 is a partial year -> excluded from any stamped run


def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    universe: list[str] | None = None,
    max_retries: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Book-to-market and next-year total return for the large-cap survivor basket.

    Returns ``(bm, fwd_ret)`` -- both fiscal-year x ticker DataFrames -- where the B/M
    signal of fiscal-year Y aligns to the calendar-year-(Y+1) total return (one-year
    reporting lag; the trade is entered after the 10-K is public).

    **Cache-first.** Reads ``<cache_dir>/bm_panel.parquet`` and ``fwd_ret.parquet`` if
    present. On cache miss with ``fetch=True`` it pulls yfinance balance sheets + prices
    (with retries) and writes the parquets. On cache miss with ``fetch=False`` (e.g. CI)
    it returns empty frames -- callers must handle gracefully.

    book-to-market = stockholders_equity_Y / (shares_Y x price at fiscal-year-end Y).
    Forward return = total return of the calendar year (Y+1) from yfinance auto-adjusted
    close (dividends reinvested). Survivorship-biased -> upper-bound magnitudes.
    """
    bm_p = os.path.join(cache_dir, _BM_PATH)
    fwd_p = os.path.join(cache_dir, _FWD_PATH)

    if os.path.exists(bm_p) and os.path.exists(fwd_p):
        bm = pd.read_parquet(bm_p)
        fwd = pd.read_parquet(fwd_p)
        return bm, fwd

    if not fetch:
        return pd.DataFrame(), pd.DataFrame()

    tickers = list(universe or UNIVERSE)
    book: dict[str, dict[int, float]] = {}   # ticker -> {fiscal_year: book_equity}
    shares: dict[str, dict[int, float]] = {}  # ticker -> {fiscal_year: shares}

    import yfinance as yf

    def _retry(fn):
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception:  # noqa: BLE001
                if attempt == max_retries - 1:
                    return None
                time.sleep(1.5 * (attempt + 1))
        return None

    # 1) Balance sheets: book equity + share count per fiscal year.
    for tk in tickers:
        yt = yf.Ticker(tk)
        bs = _retry(lambda yt=yt: yt.balance_sheet)
        if bs is None or bs.empty:
            continue
        for col in bs.columns:
            fy = pd.Timestamp(col).year
            eq = bs.loc[_EQUITY_ROW, col] if _EQUITY_ROW in bs.index else np.nan
            sh = bs.loc[_SHARES_ROW, col] if _SHARES_ROW in bs.index else np.nan
            if pd.notna(eq):
                book.setdefault(tk, {})[fy] = float(eq)
            if pd.notna(sh):
                shares.setdefault(tk, {})[fy] = float(sh)

    # 2) Prices: year-end adjusted close + calendar-year total returns.
    raw = _retry(
        lambda: yf.download(
            tickers, start="2018-01-01", end="2026-01-01", interval="1mo",
            auto_adjust=True, progress=False, threads=True,
        )
    )
    if raw is None or raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)

    yr_px = close.resample("YE").last()
    yr_px.index = yr_px.index.year
    yr_ret = yr_px.pct_change()  # calendar-year total return (adjusted -> dividends in)

    # 3) Assemble B/M and forward-return frames.
    bm_rows: dict[int, dict[str, float]] = {}
    for tk in tickers:
        if tk not in book or tk not in shares:
            continue
        for fy, eq in book[tk].items():
            sh = shares[tk].get(fy)
            if sh is None or sh <= 0:
                continue
            if fy not in yr_px.index or tk not in yr_px.columns:
                continue
            px = yr_px.loc[fy, tk]
            if pd.isna(px) or px <= 0:
                continue
            mktcap = sh * px
            if mktcap <= 0:
                continue
            bm_rows.setdefault(fy, {})[tk] = eq / mktcap

    bm = pd.DataFrame(bm_rows).T.sort_index()
    bm.index.name = "year"
    # Keep only signal years whose forward year (Y+1) is complete (<= 2025).
    bm = bm.loc[[y for y in bm.index if y + 1 <= _LAST_FWD_YEAR]]

    fwd_rows: dict[int, dict[str, float]] = {}
    for y in bm.index:
        fy = y + 1
        if fy in yr_ret.index:
            fwd_rows[y] = yr_ret.loc[fy].to_dict()
    fwd = pd.DataFrame(fwd_rows).T.sort_index()
    fwd.index.name = "year"
    fwd = fwd.reindex(columns=bm.columns)

    # Winsorise absurd B/M (data glitch from tiny denominators) -> NaN, not clip.
    bm = bm.where((bm > 0.0) & (bm < 5.0))

    os.makedirs(cache_dir, exist_ok=True)
    bm.to_parquet(bm_p)
    fwd.to_parquet(fwd_p)
    # Tiny manifest for provenance.
    with open(os.path.join(cache_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"n_tickers": int(bm.shape[1]), "years": [int(y) for y in bm.index],
             "source": "yfinance balance_sheet + monthly adjusted close"},
            f, indent=2,
        )
    return bm, fwd


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint (SHA-1 of flattened values) for the as-of stamp."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    arr = np.ascontiguousarray(df.fillna(0.0).to_numpy(dtype=float))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
