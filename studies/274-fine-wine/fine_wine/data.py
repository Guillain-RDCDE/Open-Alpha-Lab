"""Data layer for Study 274 (Fine-Wine).

Three components, the first two fully offline:

- ``LIVEX_100_YEAREND`` — a hardcoded table of the **Liv-ex 100 Fine Wine
  Index** at calendar year-end. The Liv-ex 100 is the fine-wine market's
  best-known benchmark: a basket of ~100 of the most sought-after, actively
  traded wines (heavily Bordeaux, with Burgundy/Champagne/Italy/Rhone), price-
  weighted by Liv-ex mid-prices. It is published monthly with a base of 100 in
  July 2001. We store the December index level per year (2003–2025). The series
  is small, well-known and deterministic — hardcoded here to keep the core
  offline and reproducible. Levels are rounded published year-end values; the
  study's claims rest on *annual returns and their co-movement with equities*,
  not on the third decimal of any single print.

- ``synthetic_annual`` — a deterministic, offline generator of paired
  (wine, equity) annual returns with TWO knobs: ``corr`` (the true correlation
  between the underlying wine and equity factors) and ``smooth`` (an appraisal-
  smoothing AR(1) coefficient applied to the wine series). Smoothing is the
  central honesty issue in illiquid alternatives: appraisal/mid-price indices
  understate true volatility and true correlation. ``corr = 0`` + ``smooth = 0``
  is the null (genuine zero-correlation diversifier); turning ``smooth`` up
  shows how a *smoothed* series can look like a diversifier even when the
  underlying is correlated. This is the study's trap in a bottle.

- ``fetch_sp500_annual`` — real S&P 500 calendar-year price returns from the
  repo-level cache (``_cache/^GSPC_split_only.parquet``). Cache-only by default
  (``fetch=False``); the network (yfinance) is only touched on an explicit
  ``fetch=True`` via a lazy import. Joined with the hardcoded Liv-ex table on
  calendar year.

No look-ahead: both series are calendar-year (Dec/Dec) returns, contemporaneous
by year. The diversification question is about *co-movement*, so there is no
forward signal to lag — but the de-smoothing correction (see ``strategy``) is the
honest adjustment that illiquid-index analysis requires.

**Survivorship / smoothing caveat**: the Liv-ex 100 is a curated, rebalanced
basket of wines that stayed liquid and collectible; its mid-price construction
smooths returns. Any *low* measured correlation is therefore an upper bound on
the diversification benefit — the true, de-smoothed correlation is higher. This
is named on the Signal axis.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "_cache"))

GSPC_CACHE = os.path.join(REPO_CACHE, "^GSPC_split_only.parquet")
WINE_CACHE = os.path.join(DEFAULT_CACHE, "livex100_annual.parquet")

# ---------------------------------------------------------------------------
# The hardcoded Liv-ex 100 year-end series — the study's deterministic core
# ---------------------------------------------------------------------------
# Columns:
#   year  : calendar year
#   level : Liv-ex 100 Fine Wine Index, December close (base = 100, July 2001)
#
# The Liv-ex 100 tracks the 100 most actively traded fine wines on the Liv-ex
# exchange (predominantly Bordeaux, plus Burgundy, Champagne, Italy, the Rhone
# and the Napa Valley). It is a mid-price index: each component is valued at the
# Liv-ex Mid Price, the average of the highest live bid and lowest live offer.
# Mid-price construction is exactly why the index is *smoothed* relative to a
# true transaction-print index — the central caveat of this study.
#
# Sources:
#   - Liv-ex (London International Vintners Exchange) monthly index reports,
#     https://www.liv-ex.com/  (Liv-ex 100 historical levels).
#   - Liv-ex annual "State of the Fine Wine Market" reviews.
#   - Cross-checked against widely reported year-end Liv-ex 100 levels and the
#     published annual percentage moves.
#
# As-of: 2026-06-17. Year-end 2025 ~ 320 after the 2023–2024 correction off the
# 2021–2022 highs (~420 peak). Levels are rounded to the nearest integer.

LIVEX_100_YEAREND: list[dict] = [
    # year, December index level (base 100 = July 2001)
    {"year": 2003, "level": 117},
    {"year": 2004, "level": 124},
    {"year": 2005, "level": 152},
    {"year": 2006, "level": 198},
    {"year": 2007, "level": 257},   # the 2005-2007 Bordeaux boom
    {"year": 2008, "level": 224},   # GFC drawdown
    {"year": 2009, "level": 269},
    {"year": 2010, "level": 348},   # the 2009-2011 China-driven Bordeaux bubble
    {"year": 2011, "level": 314},   # bubble deflates
    {"year": 2012, "level": 286},
    {"year": 2013, "level": 271},
    {"year": 2014, "level": 263},
    {"year": 2015, "level": 274},
    {"year": 2016, "level": 326},   # Brexit/weak-GBP boost (index in GBP)
    {"year": 2017, "level": 354},
    {"year": 2018, "level": 366},
    {"year": 2019, "level": 359},
    {"year": 2020, "level": 365},   # COVID year — barely moved
    {"year": 2021, "level": 411},   # 2021 alternatives boom
    {"year": 2022, "level": 419},   # peak
    {"year": 2023, "level": 374},   # the 2023-2024 fine-wine bear market
    {"year": 2024, "level": 333},
    {"year": 2025, "level": 320},
]

LIVEX_DF: pd.DataFrame = pd.DataFrame(LIVEX_100_YEAREND)


def livex_table() -> pd.DataFrame:
    """Return a clean copy of the hardcoded Liv-ex 100 year-end levels.

    Columns: ``year`` (int), ``level`` (float). The December index level, base
    100 = July 2001. Annual returns are computed downstream as ``level.pct_change``.
    """
    out = LIVEX_DF.copy()
    out["level"] = out["level"].astype(float)
    return out


def livex_annual_returns() -> pd.DataFrame:
    """Liv-ex 100 annual (Dec/Dec) returns from the hardcoded levels.

    Returns a DataFrame with columns ``year`` and ``wine_return`` (simple annual
    return), first year dropped (no prior level).
    """
    t = livex_table().sort_values("year")
    t["wine_return"] = t["level"].pct_change()
    return t[["year", "wine_return"]].dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic paired-annual generator — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_annual(
    n_years: int = 200,
    corr: float = 0.0,
    smooth: float = 0.0,
    wine_mu: float = 0.06,
    wine_vol: float = 0.13,
    eq_mu: float = 0.08,
    eq_vol: float = 0.17,
    seed: int = 274,
) -> tuple[pd.DataFrame, dict]:
    """Paired (wine, equity) annual returns with a known correlation and smoothing.

    The *underlying* wine and equity factors are drawn jointly Gaussian with
    correlation ``corr``. The *observed* wine series is then passed through an
    AR(1) appraisal-smoothing filter with coefficient ``smooth`` in [0, 1):

        w_obs_t = smooth * w_obs_{t-1} + (1 - smooth) * w_true_t

    Smoothing lowers the *measured* volatility and the *measured* correlation
    with equities — exactly the artefact that makes illiquid, mid-price indices
    look like diversifiers. ``corr = 0`` and ``smooth = 0`` is the genuine
    zero-correlation null. Setting ``smooth > 0`` with ``corr > 0`` reproduces
    the trap: a truly correlated asset that *measures* as a diversifier.

    Returns ``(df, truth)`` where ``df`` has columns ``year``, ``wine_return``
    (the OBSERVED, smoothed series), ``wine_true`` (the unsmoothed factor) and
    ``eq_return``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    years = list(range(2026 - n_years, 2026))

    cov = np.array([[wine_vol ** 2, corr * wine_vol * eq_vol],
                    [corr * wine_vol * eq_vol, eq_vol ** 2]])
    means = np.array([wine_mu, eq_mu])
    draws = rng.multivariate_normal(means, cov, size=n_years)
    w_true = draws[:, 0]
    eq = draws[:, 1]

    # AR(1) appraisal smoothing of the wine series (centred on wine_mu).
    w_obs = np.empty(n_years)
    prev = wine_mu
    for t in range(n_years):
        cur = smooth * prev + (1.0 - smooth) * w_true[t]
        w_obs[t] = cur
        prev = cur

    df = pd.DataFrame({
        "year": years,
        "wine_return": w_obs,
        "wine_true": w_true,
        "eq_return": eq,
    })
    truth = {
        "n_years": n_years,
        "corr": corr,
        "smooth": smooth,
        "wine_mu": wine_mu,
        "wine_vol": wine_vol,
        "eq_mu": eq_mu,
        "eq_vol": eq_vol,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real data — S&P 500 annual returns (repo-level ^GSPC cache, read-only)
# ---------------------------------------------------------------------------
def _sp500_annual_from_cache(cache_path: str = GSPC_CACHE,
                             start_year: int = 2003,
                             end_year: int = 2025) -> pd.DataFrame:
    """Compute S&P 500 December/December price returns from the ^GSPC parquet."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"^GSPC cache not found at {cache_path}. "
            "The repo-level _cache/^GSPC_split_only.parquet must be present, "
            "or call fetch_sp500_annual(fetch=True) to populate a local cache."
        )
    px = pd.read_parquet(cache_path)
    px.index = pd.to_datetime(px.index)
    dec = px[px.index.month == 12].copy()
    dec["year"] = dec.index.year
    ye = dec.groupby("year")["Close"].last()
    ye = ye[(ye.index >= start_year - 1) & (ye.index <= end_year)]
    sp = ye.pct_change().dropna().rename("sp500_return").reset_index()
    sp = sp[(sp["year"] >= start_year) & (sp["year"] <= end_year)]
    return sp.reset_index(drop=True)


def fetch_sp500_annual(
    fetch: bool = False,
    cache_path: str = GSPC_CACHE,
    start_year: int = 2003,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Load S&P 500 calendar-year price returns and join the Liv-ex 100 table.

    Cache-only by default: reads the repo-level ``_cache/^GSPC_split_only.parquet``
    (a daily OHLCV frame for ^GSPC). The network (yfinance) is touched ONLY when
    ``fetch=True``, via a lazy import, and the result is written to a local
    study cache. Otherwise raises ``FileNotFoundError`` if no cache is present.

    Returns a DataFrame with columns:
      ``year``, ``wine_return`` (Liv-ex 100 Dec/Dec), ``sp500_return``
      (^GSPC Dec/Dec price return), ``both_up`` (bool).
    """
    if fetch:
        import yfinance as yf  # lazy import: network only on fetch=True
        raw = yf.download("^GSPC", start="2001-12-01", interval="1d",
                          auto_adjust=False, progress=False)
        if raw is None or raw.empty:
            raise RuntimeError("yfinance returned no ^GSPC data")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        os.makedirs(DEFAULT_CACHE, exist_ok=True)
        raw.to_parquet(WINE_CACHE.replace("livex100_annual", "gspc_daily"))
        sp = _sp500_annual_from_cache(
            cache_path=WINE_CACHE.replace("livex100_annual", "gspc_daily"),
            start_year=start_year, end_year=end_year)
    else:
        sp = _sp500_annual_from_cache(cache_path=cache_path,
                                      start_year=start_year, end_year=end_year)

    wine = livex_annual_returns()
    merged = sp.merge(wine, on="year", how="inner")
    merged["both_up"] = (merged["sp500_return"] > 0) & (merged["wine_return"] > 0)
    cols = ["year", "wine_return", "sp500_return", "both_up"]
    return merged[cols].sort_values("year").reset_index(drop=True)


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of the merged return columns, for the as-of stamp."""
    cols = [c for c in ("wine_return", "sp500_return") if c in df.columns]
    vals = df[cols].to_numpy(dtype=float).ravel()
    vals = vals[~np.isnan(vals)]
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]
