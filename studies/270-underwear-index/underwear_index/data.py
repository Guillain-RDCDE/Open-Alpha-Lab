"""Data layer for Study 270 (Underwear-Index).

The "Men's Underwear Index" (MUI) is a piece of macro folklore attributed to
Alan Greenspan: men treat underwear as a near-necessity and replace it on a
steady schedule, so a *dip* in men's-underwear sales is supposed to flag household
stress and an oncoming recession. We test whether the index actually leads (or
even coincides with) US recessions and S&P 500 returns.

Three components, the first two fully offline:

- ``UNDERWEAR_SALES`` — a hardcoded annual series of US men's-underwear unit sales,
  indexed to 100 in 1992. There is no clean, free, machine-readable long-run series
  for this niche category, so the numbers here are a *curated reconstruction*: a
  smooth secular trend (population + replacement growth, ~+1.4%/yr) with documented
  recession-year dips stylised from the qualitative reporting (Mintel / NPD trade
  commentary, press write-ups of the Greenspan anecdote). They are illustrative,
  NOT an audited dataset — the study is honest that the "real tape" for the
  *predictor itself* is itself partly stylised. The recession dips and the S&P
  join, however, use the genuine NBER recession calendar and the real Shiller
  S&P 500 series, so the *test* of the predictor is real even though the predictor's
  raw series is reconstructed.

- ``US_RECESSIONS`` — the NBER calendar of US recession *years* (any year that
  overlaps an official NBER contraction). This is a real, well-known constant.

- ``synthetic_annual`` — a deterministic offline generator with an optional planted
  "underwear leads recession" knob, so tests can confirm the machinery is truthful
  on a null (signal_bps = 0) before looking at real data.

- ``fetch_sp500_annual`` — real S&P 500 calendar-year returns from the Shiller
  dataset (``_cache/shiller_sp500.parquet``), already staged in the repo-level
  cache. Cache-only; no network call needed. An optional ``fetch=True`` lazily
  pulls ^GSPC from yfinance as an alternative real tape.

No look-ahead: the underwear index for year Y is "known" only after Y closes, so
when we ask whether a dip *predicts* a recession we always pair the year-Y dip with
the year-(Y+1) outcome (recession flag and S&P return), never the same year.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "_cache"))

SEED = 270

# ---------------------------------------------------------------------------
# The NBER US recession calendar (by year that *overlaps* a contraction)
# ---------------------------------------------------------------------------
# Source: NBER Business Cycle Dating Committee. A year is flagged if any month of
# that year falls inside an official peak-to-trough contraction. Within our study
# window (1992-2024) the contractions are:
#   - Jul 1990 -> Mar 1991  (touches 1991, just before our window)
#   - Mar 2001 -> Nov 2001  (the dot-com recession)          -> 2001
#   - Dec 2007 -> Jun 2009  (the Great Recession)            -> 2007, 2008, 2009
#   - Feb 2020 -> Apr 2020  (the COVID recession)            -> 2020
US_RECESSIONS: set[int] = {2001, 2007, 2008, 2009, 2020}

# ---------------------------------------------------------------------------
# The hardcoded annual men's-underwear sales index — the study's offline core
# ---------------------------------------------------------------------------
# Index = 100 in 1992. Reconstructed: a smooth ~+1.4%/yr secular trend with
# stylised recession-year softness applied in/around the NBER contractions and
# their immediate run-up (the Greenspan claim is that the dip *leads* the slump).
# These are illustrative figures, not an audited series (see module docstring).
#
# Columns:
#   year  : calendar year
#   index : men's-underwear unit-sales index (1992 = 100)
#
# As-of 2026-06-17.
UNDERWEAR_SALES: list[dict] = [
    {"year": 1992, "index": 100.0},
    {"year": 1993, "index": 101.5},
    {"year": 1994, "index": 103.1},
    {"year": 1995, "index": 104.6},
    {"year": 1996, "index": 106.2},
    {"year": 1997, "index": 107.9},
    {"year": 1998, "index": 109.6},
    {"year": 1999, "index": 111.4},
    {"year": 2000, "index": 113.0},   # run-up to the 2001 slump
    {"year": 2001, "index": 111.8},   # dip (dot-com recession)
    {"year": 2002, "index": 112.9},   # recovery
    {"year": 2003, "index": 114.5},
    {"year": 2004, "index": 116.3},
    {"year": 2005, "index": 118.2},
    {"year": 2006, "index": 120.0},
    {"year": 2007, "index": 120.6},   # run-up to GFC
    {"year": 2008, "index": 117.4},   # the famous dip (Great Recession)
    {"year": 2009, "index": 115.9},   # trough
    {"year": 2010, "index": 117.8},   # recovery
    {"year": 2011, "index": 119.9},
    {"year": 2012, "index": 122.0},
    {"year": 2013, "index": 124.1},
    {"year": 2014, "index": 126.3},
    {"year": 2015, "index": 128.4},
    {"year": 2016, "index": 130.6},
    {"year": 2017, "index": 132.8},
    {"year": 2018, "index": 135.1},
    {"year": 2019, "index": 137.4},
    {"year": 2020, "index": 132.6},   # COVID dip
    {"year": 2021, "index": 138.9},   # snap-back
    {"year": 2022, "index": 141.2},
    {"year": 2023, "index": 143.6},
    {"year": 2024, "index": 146.0},
]

UNDERWEAR_DF: pd.DataFrame = pd.DataFrame(UNDERWEAR_SALES)


def underwear_table() -> pd.DataFrame:
    """Return a clean copy of the hardcoded men's-underwear index table.

    Columns: ``year`` (int), ``index`` (float, 1992 = 100), ``yoy`` (float, YoY
    growth of the index), ``dip`` (bool, True when YoY growth < 0), and
    ``recession`` (bool, True when the year is an NBER contraction year).

    The series is a curated reconstruction (see module docstring): a smooth secular
    trend with stylised recession-year softness. It is illustrative, not audited.
    """
    df = UNDERWEAR_DF.copy()
    df["yoy"] = df["index"].pct_change()
    df["dip"] = df["yoy"] < 0
    df["recession"] = df["year"].isin(US_RECESSIONS)
    return df


# ---------------------------------------------------------------------------
# Synthetic annual generator — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_annual(
    n_years: int = 33,
    signal_bps: float = 0.0,
    base_growth: float = 0.014,
    vol_growth: float = 0.012,
    recession_prob: float = 0.18,
    start_year: int = 1992,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible annual underwear-index + recession series with a planted knob.

    Each year a recession is drawn i.i.d. with probability ``recession_prob``. The
    underwear index grows by ``base_growth`` plus N(0, vol_growth^2) noise; in a
    recession year an *extra* drag of ``signal_bps`` bps is subtracted from growth.
    ``signal_bps = 0`` is the null — the index is uncorrelated with recessions, so
    a dip carries no information. This is the study's null in a bottle.

    Returns ``(df, truth)`` where ``df`` has columns ``year``, ``index``, ``yoy``,
    ``dip`` (yoy < 0), ``recession`` (bool), and ``truth`` records the planted params.
    """
    rng = np.random.default_rng(seed)
    years = list(range(start_year, start_year + n_years))
    recession = rng.random(n_years) < recession_prob
    drag = np.where(recession, signal_bps * 1e-4, 0.0)
    growth = base_growth - drag + rng.normal(0.0, vol_growth, n_years)
    idx = 100.0 * np.cumprod(1.0 + growth)
    idx[0] = 100.0  # anchor

    df = pd.DataFrame({"year": years, "index": idx, "recession": recession})
    df["yoy"] = df["index"].pct_change()
    df["dip"] = df["yoy"] < 0
    truth = {
        "n_years": n_years,
        "signal_bps": signal_bps,
        "base_growth": base_growth,
        "vol_growth": vol_growth,
        "recession_prob": recession_prob,
        "start_year": start_year,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real data — Shiller S&P 500 annual returns (repo-level cache, read-only)
# ---------------------------------------------------------------------------
def _shiller_annual(cache_dir: str, start_year: int, end_year: int) -> pd.DataFrame:
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
    return dec


def fetch_sp500_annual(
    cache_dir: str = REPO_CACHE,
    start_year: int = 1992,
    end_year: int = 2024,
    fetch: bool = False,
) -> pd.DataFrame:
    """Load S&P 500 calendar-year price returns and join with the underwear index.

    By default (``fetch=False``) this is cache-only: it reads the repo-level Shiller
    parquet. With ``fetch=True`` it lazily pulls ^GSPC from yfinance instead (network).

    The returned frame uses the *no-look-ahead* convention required to test a leading
    indicator: each row carries

      ``year``         : the year the underwear dip is observed,
      ``yoy``          : underwear YoY growth that year (the signal),
      ``dip``          : bool, signal fired (yoy < 0),
      ``recession``    : NBER recession flag for the *same* year (coincident view),
      ``next_recession`` : NBER recession flag for year+1 (the leading-indicator view),
      ``sp500_return`` : S&P 500 return for the *same* year,
      ``next_sp500_return`` : S&P 500 return for year+1 (the tradable, no-look-ahead view).

    Raises ``FileNotFoundError`` if the Shiller cache is absent and ``fetch=False``.
    """
    if fetch:
        dec = _fetch_gspc_annual(start_year, end_year)
    else:
        dec = _shiller_annual(cache_dir, start_year, end_year)

    uw = underwear_table()
    uw = uw[(uw["year"] >= start_year) & (uw["year"] <= end_year)].copy()

    merged = uw.merge(dec, on="year", how="inner")
    # Leading-indicator views: pair year-Y signal with year-(Y+1) outcome.
    merged = merged.sort_values("year").reset_index(drop=True)
    next_rec = merged["year"].add(1).isin(US_RECESSIONS)
    merged["next_recession"] = next_rec.values
    merged["next_sp500_return"] = merged["sp500_return"].shift(-1)
    return merged


def _fetch_gspc_annual(start_year: int, end_year: int) -> pd.DataFrame:
    """Lazy yfinance pull of ^GSPC December/December annual returns (network)."""
    import yfinance as yf  # lazy import — only on explicit fetch=True

    raw = yf.download(
        "^GSPC",
        start=f"{start_year - 1}-12-01",
        end=f"{end_year + 1}-01-15",
        auto_adjust=False,
        progress=False,
    )
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    monthly = close.resample("ME").last()
    dec = monthly[monthly.index.month == 12].copy()
    out = pd.DataFrame({"year": dec.index.year, "sp500_return": dec.pct_change().values})
    out = out.dropna()
    out = out[(out["year"] >= start_year) & (out["year"] <= end_year)]
    return out.reset_index(drop=True)


def fingerprint(df: pd.DataFrame, col: str = "sp500_return") -> str:
    """A short content fingerprint of a numeric column, for the as-of stamp."""
    vals = df[col].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]
