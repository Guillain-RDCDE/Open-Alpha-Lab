"""Data layer for Study 265 (IPO-Volume).

The folklore: a flood of new issues is the bell at the top. When the IPO window
is wide open and hundreds of companies rush to list, the market is euphoric and
forward returns are poor; when the IPO window slams shut (almost no listings),
the market is fearful and forward returns are strong. This is the "hot-issue
market" of Ibbotson & Jaffe (1975) and Ritter (1991), and the IPO-volume leg of
Baker & Wurgler's (2006) investor-sentiment index.

Three components, all fully offline for the core:

- ``IPO_COUNTS`` -- a hardcoded annual table of US operating-company IPO counts
  (1980-2024), the canonical Jay Ritter / University of Florida series
  (IPOs excluding closed-end funds, REITs, ADRs, unit offers, SPACs, and
  penny-stock offerings under $5). This is a deterministic, well-known, small
  table -- hardcoded here to keep the core fully offline and reproducible.

- ``synthetic_annual`` -- a deterministic, offline generator with an optional
  planted "froth signal" knob. A ``beta = 0`` setting is the null (IPO volume
  carries no forward information). This is the study's null in a bottle, so the
  tests can confirm the regression machinery is truthful before looking at real
  data.

- ``fetch_sp500_annual`` -- real S&P 500 calendar-year price returns from the
  Shiller dataset (``_cache/shiller_sp500.parquet``, repo-level, read-only,
  already staged). Cache-only; no network call needed for the headline run.

- ``fetch_gspc_annual`` -- an optional yfinance ^GSPC annual fallback, cache-only
  by default; network only on explicit ``fetch=True`` (lazy import).

No look-ahead: the IPO count is *known by year-end* of year Y (it counts deals
that already priced that year). We pair the standardized froth signal at the end
of year Y with the S&P 500 return *earned during year Y+1* (one-year execution
lag). The signal is standardized using only an expanding (past-only) window so no
future IPO statistics leak into the score.

**Price-only, not total-return**: the Shiller SP500 column is the index price
level, so our S&P returns are price returns (dividends excluded). Labeled as such.
**No survivorship issue on the index side** -- the S&P 500 is a published index;
the honesty axis here is the tiny annual sample (n ~ 44 years).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "_cache"))

GSPC_CACHE = os.path.join(DEFAULT_CACHE, "ipo_gspc_annual.parquet")

# ---------------------------------------------------------------------------
# The hardcoded annual IPO-count table -- the study's deterministic offline core
# ---------------------------------------------------------------------------
# US operating-company IPO counts by year. This is the canonical Jay Ritter
# (University of Florida) series: IPOs of operating companies, excluding
# closed-end funds, REITs, ADRs of foreign companies, unit offers, SPACs/blank-
# check companies, partnerships, banks & S&Ls, and offerings priced below $5.
# The series is the standard one cited in the "hot-issue market" literature.
#
# Source: Jay R. Ritter, "Initial Public Offerings: Updated Statistics",
#   University of Florida (Warrington College of Business), IPO Data tables.
#   https://site.warrington.ufl.edu/ritter/ipo-data/
# As-of: 2026-06-17. Values are the widely circulated headline counts; small
# vintage revisions of +/-1 to 2 deals do not affect the standardized signal.
#
# The famous landmarks the folklore leans on:
#   1999-2000 : dot-com IPO flood (476, 380) -> 2000-2002 bear market
#   2008      : IPO window slams shut (31) -> strong 2009 rebound
#   2020-2021 : pandemic / SPAC-era flood (165, 311) -> 2022 drawdown
IPO_COUNTS: list[dict] = [
    {"year": 1980, "ipo_count": 71},
    {"year": 1981, "ipo_count": 192},
    {"year": 1982, "ipo_count": 77},
    {"year": 1983, "ipo_count": 451},
    {"year": 1984, "ipo_count": 179},
    {"year": 1985, "ipo_count": 187},
    {"year": 1986, "ipo_count": 393},
    {"year": 1987, "ipo_count": 285},
    {"year": 1988, "ipo_count": 105},
    {"year": 1989, "ipo_count": 116},
    {"year": 1990, "ipo_count": 110},
    {"year": 1991, "ipo_count": 286},
    {"year": 1992, "ipo_count": 412},
    {"year": 1993, "ipo_count": 510},
    {"year": 1994, "ipo_count": 402},
    {"year": 1995, "ipo_count": 461},
    {"year": 1996, "ipo_count": 677},
    {"year": 1997, "ipo_count": 474},
    {"year": 1998, "ipo_count": 281},
    {"year": 1999, "ipo_count": 476},
    {"year": 2000, "ipo_count": 380},
    {"year": 2001, "ipo_count": 79},
    {"year": 2002, "ipo_count": 66},
    {"year": 2003, "ipo_count": 63},
    {"year": 2004, "ipo_count": 173},
    {"year": 2005, "ipo_count": 159},
    {"year": 2006, "ipo_count": 157},
    {"year": 2007, "ipo_count": 159},
    {"year": 2008, "ipo_count": 21},
    {"year": 2009, "ipo_count": 41},
    {"year": 2010, "ipo_count": 91},
    {"year": 2011, "ipo_count": 81},
    {"year": 2012, "ipo_count": 93},
    {"year": 2013, "ipo_count": 158},
    {"year": 2014, "ipo_count": 206},
    {"year": 2015, "ipo_count": 118},
    {"year": 2016, "ipo_count": 75},
    {"year": 2017, "ipo_count": 107},
    {"year": 2018, "ipo_count": 134},
    {"year": 2019, "ipo_count": 112},
    {"year": 2020, "ipo_count": 165},
    {"year": 2021, "ipo_count": 311},
    {"year": 2022, "ipo_count": 71},
    {"year": 2023, "ipo_count": 54},
    {"year": 2024, "ipo_count": 75},
]

IPO_DF: pd.DataFrame = pd.DataFrame(IPO_COUNTS)


def ipo_table() -> pd.DataFrame:
    """Return a clean copy of the hardcoded annual IPO-count table.

    Columns: ``year`` (int), ``ipo_count`` (int).  The count is known at year-end
    of ``year`` (the deals already priced that calendar year), so it can be used
    to forecast the *following* year's market return without look-ahead.
    """
    return IPO_DF.copy()


# ---------------------------------------------------------------------------
# Synthetic annual generator -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_annual(
    n_years: int = 45,
    beta: float = -0.06,
    base_return: float = 0.08,
    vol_ann: float = 0.16,
    seed: int = 265,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible (IPO count, next-year return) panel with a planted froth signal.

    The IPO count is generated as an AR(1)-ish positive count series (so it looks
    like the real bursty series). The next-year return is::

        base_return + beta * z_ipo + noise

    where ``z_ipo`` is the *standardized* IPO count for the current year.
    ``beta < 0`` plants the froth signal (high IPO volume -> low forward return);
    ``beta = 0`` is the null -- IPO volume carries no forward information.

    Returns ``(df, truth)`` where ``df`` has columns ``year`` (int),
    ``ipo_count`` (int), ``fwd_return`` (the return earned in year+1), and
    ``truth`` records the planted parameters.

    No look-ahead: ``z_ipo`` is standardized with the full-sample mean/std here
    (this is the *generator*, not the estimator); the estimator in
    ``strategy.froth_signal`` uses an expanding past-only window.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(1980, 1980 + n_years)

    # Bursty positive count series: log-normal AR(1)
    log_c = np.zeros(n_years)
    log_c[0] = np.log(150.0)
    for t in range(1, n_years):
        log_c[t] = 0.6 * log_c[t - 1] + 0.4 * np.log(150.0) + rng.normal(0.0, 0.5)
    counts = np.clip(np.round(np.exp(log_c)), 10, None).astype(int)

    z = (counts - counts.mean()) / counts.std(ddof=0)
    noise = rng.normal(0.0, vol_ann, n_years)
    # fwd_return[t] is the return earned in year t+1, driven by this year's z.
    fwd = base_return + beta * z + noise

    df = pd.DataFrame(
        {
            "year": years.astype(int),
            "ipo_count": counts,
            "fwd_return": fwd,
        }
    )
    truth = {
        "n_years": n_years,
        "beta": beta,
        "base_return": base_return,
        "vol_ann": vol_ann,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real data -- Shiller S&P 500 annual price returns (repo-level cache, read-only)
# ---------------------------------------------------------------------------
def _shiller_annual_returns(
    cache_dir: str = REPO_CACHE,
    start_year: int = 1980,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Calendar-year S&P 500 *price* returns from the Shiller monthly dataset.

    Uses December year-end price levels: return(Y) = P_dec(Y) / P_dec(Y-1) - 1.
    Price-only (Shiller SP500 column is the index level; dividends excluded).

    Returns a DataFrame with columns ``year`` (int), ``sp500_return`` (float).
    Raises ``FileNotFoundError`` if the Shiller cache is absent.
    """
    shiller_path = os.path.join(cache_dir, "shiller_sp500.parquet")
    if not os.path.exists(shiller_path):
        raise FileNotFoundError(
            f"Shiller cache not found at {shiller_path}. "
            "The repo-level _cache/shiller_sp500.parquet must be present."
        )
    sh = pd.read_parquet(shiller_path)
    if "Date" in sh.columns:
        sh.index = pd.to_datetime(sh["Date"], errors="coerce")
        sh = sh.drop(columns=["Date"])
    elif not isinstance(sh.index, pd.DatetimeIndex):
        sh.index = pd.to_datetime(sh.index)

    dec = sh[sh.index.month == 12].copy()
    dec["year"] = dec.index.year
    dec = dec[dec["SP500"].gt(0) & dec["SP500"].notna()].copy()
    dec = dec.sort_values("year")
    dec["sp500_return"] = dec["SP500"].pct_change()
    dec = dec[["year", "sp500_return"]].dropna()
    dec = dec[(dec["year"] >= start_year) & (dec["year"] <= end_year)].copy()
    return dec.reset_index(drop=True)


def fetch_sp500_annual(
    cache_dir: str = REPO_CACHE,
    start_year: int = 1980,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Join the hardcoded IPO table with Shiller S&P 500 annual price returns.

    Builds the study's real-tape panel.  For each year ``Y`` we record:

    - ``ipo_count`` : IPOs priced in year Y (known at year-end Y)
    - ``sp500_return`` : the S&P 500 price return *during year Y* (contemporaneous)
    - ``fwd_return`` : the S&P 500 price return *during year Y+1* (the prediction
      target; one-year lag, no look-ahead)

    Rows without a valid ``fwd_return`` (the last year) are dropped.  Returns a
    DataFrame indexed 0..n-1 with the columns above plus ``year``.

    Raises ``FileNotFoundError`` if the Shiller cache is absent.
    """
    sp = _shiller_annual_returns(cache_dir, start_year, end_year)
    ipo = ipo_table()

    merged = ipo.merge(sp, on="year", how="inner").sort_values("year").reset_index(drop=True)
    # Forward (year+1) S&P return, indexed by the signal year Y.
    sp_lookup = sp.set_index("year")["sp500_return"]
    merged["fwd_return"] = merged["year"].apply(lambda y: sp_lookup.get(y + 1, np.nan))
    merged = merged.dropna(subset=["fwd_return"]).reset_index(drop=True)
    return merged[["year", "ipo_count", "sp500_return", "fwd_return"]]


def fetch_gspc_annual(
    fetch: bool = False,
    cache_path: str = GSPC_CACHE,
) -> pd.DataFrame:
    """Optional ^GSPC annual price-return panel via yfinance; cache-only by default.

    Network is touched only on an explicit ``fetch=True`` (then cached as a
    parquet under ``_cache/ipo_gspc_annual.parquet``).  Same column layout as
    ``fetch_sp500_annual`` (``year``, ``ipo_count``, ``sp500_return``,
    ``fwd_return``).  This is a cross-check on the Shiller real tape, not the
    primary source.

    Raises ``FileNotFoundError`` if the cache is absent and ``fetch=False``.
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached ^GSPC annual panel at {cache_path}. "
                "Call fetch_gspc_annual(fetch=True) once to populate the cache."
            )
        return pd.read_parquet(cache_path)

    import yfinance as yf  # lazy import: network only on fetch=True

    raw = yf.download(
        "^GSPC",
        start="1979-01-01",
        interval="1mo",
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no ^GSPC data")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    dec = close[close.index.month == 12]
    # Last December observation per year
    dec_year = pd.Series(dec.values, index=dec.index.year).groupby(level=0).last()
    sp = pd.DataFrame({"year": dec_year.index.astype(int), "sp500_return": dec_year.pct_change().values})
    sp = sp.dropna().reset_index(drop=True)

    ipo = ipo_table()
    merged = ipo.merge(sp, on="year", how="inner").sort_values("year").reset_index(drop=True)
    sp_lookup = sp.set_index("year")["sp500_return"]
    merged["fwd_return"] = merged["year"].apply(lambda y: sp_lookup.get(y + 1, np.nan))
    merged = merged.dropna(subset=["fwd_return"]).reset_index(drop=True)
    out = merged[["year", "ipo_count", "sp500_return", "fwd_return"]]

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    out.to_parquet(cache_path)
    print(f"Cached ^GSPC annual panel: {out.shape[0]} years -> {cache_path}")
    return out


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of the fwd_return column, for the as-of stamp."""
    col = "fwd_return" if "fwd_return" in df.columns else df.columns[-1]
    vals = df[col].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]
