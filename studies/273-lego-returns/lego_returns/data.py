"""Data layer for Study 273 (Lego-Returns).

Three components, the first two fully offline:

- ``LEGO_INDEX`` -- a hardcoded annual secondary-market price index for retired
  LEGO sets, base 100 at year-end 1986, running through year-end 2024.  This is a
  *curated* index: the headline academic source is Dobrynskaya & Kishilova
  (2022), "LEGO: The Toy of Smart Investors" (Research in International Business
  and Finance, vol. 59), which estimates an average secondary-market return of
  roughly 8-11%/yr (nominal, price-only) on retired sets over 1987-2015, with a
  low correlation to equities.  We extend their era-level estimates to 2024 with
  conservative post-2015 numbers (the BrickLink/StockX collectibles boom cooled
  after 2021).  The series is deterministic and hardcoded so the core is fully
  offline and reproducible.  IT IS A STYLISED, CURATED INDEX, not a tick-by-tick
  reconstruction -- treated here as an *upper bound* on what a retail collector
  could actually realise (see the survivorship / selection caveat below).

- ``synthetic_annual`` -- a deterministic, offline annual-return generator with a
  planted "Lego premium" knob.  ``premium_bps = 0`` is the null (no edge over the
  market), so tests confirm the machinery is truthful before looking at real data.

- ``fetch_market_annual`` -- real S&P 500 calendar-year PRICE returns from the
  repo-level cache (``_cache/^GSPC_split_only.parquet``), cache-only by default;
  network (yfinance) only on an explicit ``fetch=True``.

SURVIVORSHIP / SELECTION BIAS (named on the Signal axis): a secondary-market
"index" of retired sets is built from sets that were actually traded and tracked.
Sets that flopped, were liquidated below RRP, or vanished from the resale market
are under-represented.  Published LEGO-as-investment returns are therefore upper
bounds; the realisable return for a typical buyer (paying RRP, storing for years,
eating eBay/BrickLink fees of ~12-15% one-way) is materially lower.  We label the
Lego series PRICE-ONLY (collectibles pay no dividend) and the market series
PRICE-ONLY too, so the comparison is apples-to-apples; the equity total-return
gap (dividends ~2%/yr) is discussed in the notebooks.

No look-ahead: the index is annual (year-end to year-end); the "buy retired sets,
hold, resell" strategy is compared to "buy the index" over identical calendar
years.  One year is the natural execution unit; resale frictions are charged
explicitly as a one-way cost on the Lego leg.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "_cache"))

# ---------------------------------------------------------------------------
# The hardcoded LEGO secondary-market price index -- the offline core
# ---------------------------------------------------------------------------
# Annual index level, base 100 at year-end 1986.  Each level is the curated
# secondary-market price of a representative basket of retired LEGO sets at that
# year-end.  Construction: era-level price-appreciation rates from Dobrynskaya &
# Kishilova (2022) for 1987-2015 (they report ~10-11%/yr over the full window,
# stronger in the 2000s, with high cross-set dispersion), extended to 2024 with
# conservative post-boom numbers.  Hardcoded as index LEVELS (not returns) so the
# annual return is recovered by pct_change and the fingerprint is content-stable.
#
# Sources:
#   - Dobrynskaya, V. & Kishilova, J. (2022). "LEGO: The Toy of Smart Investors."
#     Research in International Business and Finance, 59, 101539.
#   - BrickLink / BrickEconomy aggregate price trends (collectibles boom 2016-2021,
#     cool-down 2022-2024).
# As-of: 2026-06-17.

LEGO_INDEX: dict[int, float] = {
    1986: 100.0,
    1987: 108.0,
    1988: 117.0,
    1989: 128.0,
    1990: 140.0,
    1991: 151.0,
    1992: 163.0,
    1993: 178.0,
    1994: 192.0,
    1995: 205.0,
    1996: 221.0,
    1997: 240.0,
    1998: 256.0,
    1999: 278.0,
    2000: 305.0,
    2001: 332.0,
    2002: 360.0,
    2003: 398.0,
    2004: 437.0,
    2005: 482.0,
    2006: 528.0,
    2007: 583.0,
    2008: 640.0,
    2009: 712.0,
    2010: 792.0,
    2011: 871.0,
    2012: 962.0,
    2013: 1058.0,
    2014: 1153.0,
    2015: 1268.0,
    2016: 1370.0,
    2017: 1466.0,
    2018: 1540.0,
    2019: 1632.0,
    2020: 1779.0,
    2021: 2010.0,
    2022: 1969.0,
    2023: 2008.0,
    2024: 2068.0,
}


def lego_index() -> pd.DataFrame:
    """Return the hardcoded LEGO secondary-market price index as a tidy frame.

    Columns: ``year`` (int), ``lego_index`` (float, base 100 at 1986),
    ``lego_return`` (float, year-over-year price-only return; NaN for the first
    year).  Deterministic, offline, content-stable.
    """
    years = sorted(LEGO_INDEX)
    levels = [LEGO_INDEX[y] for y in years]
    df = pd.DataFrame({"year": years, "lego_index": levels})
    df["lego_return"] = df["lego_index"].pct_change()
    return df


# ---------------------------------------------------------------------------
# Synthetic annual-return generator -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_annual(
    n_years: int = 38,
    premium_bps: float = 0.0,
    market_mean: float = 0.085,
    market_vol: float = 0.16,
    lego_vol: float = 0.09,
    corr: float = 0.15,
    seed: int = 273,
) -> tuple[pd.DataFrame, dict]:
    """Reproducible paired annual returns (lego vs market) with a planted premium.

    Each year draws a market return ~ N(market_mean, market_vol^2) and a Lego
    return that shares correlation ``corr`` with the market but carries an extra
    drift of ``premium_bps`` basis points.  ``premium_bps = 0`` is the null: Lego
    has the same expected return as the market (no edge).  A positive
    ``premium_bps`` plants a genuine Lego outperformance the engine should detect.

    Returns ``(df, truth)`` with columns ``year``, ``lego_return``,
    ``market_return`` and a ``truth`` dict of the planted parameters.
    """
    rng = np.random.default_rng(seed)
    years = list(range(1987, 1987 + n_years))

    z_mkt = rng.standard_normal(n_years)
    z_ind = rng.standard_normal(n_years)
    z_lego = corr * z_mkt + np.sqrt(max(0.0, 1.0 - corr * corr)) * z_ind

    market_ret = market_mean + market_vol * z_mkt
    lego_ret = market_mean + premium_bps * 1e-4 + lego_vol * z_lego

    df = pd.DataFrame({
        "year": years,
        "lego_return": lego_ret,
        "market_return": market_ret,
    })
    truth = {
        "n_years": n_years,
        "premium_bps": premium_bps,
        "market_mean": market_mean,
        "market_vol": market_vol,
        "lego_vol": lego_vol,
        "corr": corr,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real market tape -- S&P 500 calendar-year price returns (cache-only default)
# ---------------------------------------------------------------------------
def fetch_market_annual(
    fetch: bool = False,
    cache_dir: str = REPO_CACHE,
    start_year: int = 1987,
    end_year: int = 2024,
) -> pd.DataFrame:
    """Load S&P 500 calendar-year PRICE returns and join with the LEGO index.

    Cache-only by default: reads ``_cache/^GSPC_split_only.parquet`` (daily OHLC,
    split-adjusted, PRICE-ONLY -- no dividends) staged in the repo-level cache and
    computes December-to-December calendar-year price returns.  Network (yfinance)
    is touched only on an explicit ``fetch=True``.

    Returns a DataFrame with columns:
      ``year``, ``market_return`` (S&P price-only), ``lego_index``,
      ``lego_return`` (price-only), ``lego_beats`` (bool: lego_return > market).

    Raises ``FileNotFoundError`` if the cache is absent and ``fetch=False``.
    """
    cache_path = os.path.join(cache_dir, "^GSPC_split_only.parquet")

    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"S&P 500 cache not found at {cache_path}. "
                "Call fetch_market_annual(fetch=True) once to populate it."
            )
        px = pd.read_parquet(cache_path)
    else:
        import yfinance as yf  # lazy import: network only on fetch=True

        raw = yf.download("^GSPC", start="1985-01-01", auto_adjust=False,
                          progress=False)
        if raw.empty:
            raise RuntimeError("yfinance returned no S&P 500 data")
        px = raw
        os.makedirs(cache_dir, exist_ok=True)
        px.to_parquet(cache_path)

    if not isinstance(px.index, pd.DatetimeIndex):
        px.index = pd.to_datetime(px.index)

    close = px["Close"]
    if isinstance(close, pd.DataFrame):  # MultiIndex columns from a live fetch
        close = close.iloc[:, 0]

    # Year-end (last trading day of each calendar year) close.
    yr_end = close.groupby(close.index.year).last()
    mkt_ret = yr_end.pct_change()
    mkt = pd.DataFrame({"year": mkt_ret.index.astype(int),
                        "market_return": mkt_ret.values}).dropna()

    lego = lego_index().dropna(subset=["lego_return"])

    merged = mkt.merge(lego, on="year", how="inner")
    merged = merged[(merged["year"] >= start_year) & (merged["year"] <= end_year)]
    merged = merged.sort_values("year").reset_index(drop=True)
    merged["lego_beats"] = merged["lego_return"] > merged["market_return"]
    return merged[["year", "market_return", "lego_index",
                   "lego_return", "lego_beats"]]


def fingerprint(df: pd.DataFrame, col: str = "market_return") -> str:
    """Short content fingerprint of a return column, for the as-of stamp."""
    vals = df[col].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]
