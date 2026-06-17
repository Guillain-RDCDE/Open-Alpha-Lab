"""Data layer for Study 275 (Whisky-Cask).

Three components, all offline for the core:

- ``WHISKY_INDEX`` — a hardcoded **annual rare-whisky / cask price index** (year-end
  level, rebased to 100 at 2008). The series is a desk-curated reconstruction of the
  publicly reported trajectory of two well-known appraisal-based indices: the
  Knight Frank *Luxury Investment Index — Rare Whisky* sub-index and the
  Rare Whisky 101 *Apex 1000* index. These indices are **appraisal/auction-hammer
  based**, published once per year, and are the numbers quoted in the marketing
  pitch for cask investing ("whisky beat every other luxury asset"). They are NOT
  a tradable price — they are a curated basket of the rarest *bottles*, used here
  as the most generous possible proxy for what a cask owner might mark to. We
  hardcode the series (small, well-known, published annually) to keep the core
  fully offline and reproducible. Source notes are inline below.

- ``synthetic_index`` — a deterministic offline generator of an annual alt-asset
  level series with a tunable *true* drift, vol, and an **autocorrelation /
  smoothing knob** that mimics how appraisal-based indices understate true
  volatility (the Geltner unsmoothing problem). ``smoothing = 0`` is the honest
  null; positive smoothing manufactures a flatteringly high Sharpe out of thin
  air. This is the study's null in a bottle.

- ``fetch_gspc_annual`` — real S&P 500 (^GSPC) calendar-year price returns from the
  repo-level cache (``_cache/^GSPC_split_only.parquet``). Cache-only by default
  (``fetch=False``); ``fetch=True`` lazily imports yfinance for a live pull. The
  whisky index is **price/appraisal-only** and the S&P proxy here is likewise
  **price-only** (no dividends) so the comparison is apples-to-apples on the
  Signal axis; the total-return gap is discussed in docs/results.md.

No look-ahead: each year-end index level is paired with the *same* calendar-year
S&P price return. The index is reported in arrears (a year-Y level is published in
early Y+1), which is itself a tradability problem flagged in the strategy layer.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "_cache"))

SEED = 275

# ---------------------------------------------------------------------------
# The hardcoded annual whisky-cask index — the study's deterministic core
# ---------------------------------------------------------------------------
# Columns:
#   year        : calendar year (year-end mark)
#   index_level : appraisal-based rare-whisky/cask index, rebased to 100.0 at 2008
#
# Provenance and shape.  The trajectory below reconstructs the *publicly reported*
# path of appraisal-based rare-whisky indices over 2008-2024:
#   - 2010-2018: the celebrated boom. Knight Frank reported rare whisky as the
#     best-performing luxury collectible of the 2010s (~+580% over the decade in
#     its sub-index); Rare Whisky 101's Apex 1000 roughly quintupled.
#   - 2019-2021: a plateau/peak — appraisal indices kept printing positive but
#     decelerating numbers even as the secondary auction market cooled.
#   - 2022-2024: the hangover. Knight Frank reported rare whisky FALLING in 2023
#     (about -9% that year) and broadly flat-to-down into 2024 as the cask-investment
#     bubble deflated and several cask-broker scandals hit confidence.
# The exact levels are a desk reconstruction (the underlying indices are proprietary
# and not redistributable); they are deliberately on the *generous* side so the
# verdict cannot be accused of cherry-picking a weak series. As-of 2026-06-17.
#
# Sources (publicly reported figures, not redistributed raw data):
#   - Knight Frank, "The Wealth Report" / Luxury Investment Index, annual editions.
#   - Rare Whisky 101, "Apex 1000" / "Icon 100" index commentary, annual.
#   - FCA & UK ASA warnings on cask-investment mis-selling (2023-2024).

WHISKY_INDEX: list[dict] = [
    {"year": 2008, "index_level": 100.0},   # base
    {"year": 2009, "index_level": 112.0},
    {"year": 2010, "index_level": 134.0},
    {"year": 2011, "index_level": 168.0},
    {"year": 2012, "index_level": 205.0},
    {"year": 2013, "index_level": 251.0},
    {"year": 2014, "index_level": 286.0},
    {"year": 2015, "index_level": 320.0},
    {"year": 2016, "index_level": 372.0},
    {"year": 2017, "index_level": 451.0},
    {"year": 2018, "index_level": 562.0},   # the celebrated peak-momentum year
    {"year": 2019, "index_level": 627.0},
    {"year": 2020, "index_level": 651.0},
    {"year": 2021, "index_level": 698.0},   # plateau
    {"year": 2022, "index_level": 712.0},
    {"year": 2023, "index_level": 648.0},   # the reported ~-9% hangover year
    {"year": 2024, "index_level": 628.0},   # broadly flat-to-down
]

WHISKY_DF: pd.DataFrame = pd.DataFrame(WHISKY_INDEX)


def whisky_index() -> pd.DataFrame:
    """Return a clean copy of the hardcoded annual whisky-cask index.

    Columns: ``year`` (int), ``index_level`` (float, rebased to 100 at 2008),
    plus a computed ``whisky_return`` (year-on-year simple return; NaN in the
    base year).
    """
    df = WHISKY_DF.copy()
    df["whisky_return"] = df["index_level"].pct_change()
    return df


# ---------------------------------------------------------------------------
# Synthetic alt-asset generator — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_index(
    n_years: int = 16,
    true_drift: float = 0.07,
    true_vol: float = 0.18,
    smoothing: float = 0.0,
    base_level: float = 100.0,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible annual alt-asset level series with a tunable *appraisal-smoothing* knob.

    The *true* annual log-return is drawn i.i.d. from N(true_drift, true_vol^2).
    The **reported** (appraised) return is an exponentially smoothed version:

        r_reported_t = (1 - smoothing) * r_true_t + smoothing * r_reported_{t-1}

    With ``smoothing = 0`` the reported series equals the true series (the honest
    null). With ``smoothing -> 1`` the reported series becomes a slow-moving,
    low-volatility line whose *measured* Sharpe is flatteringly high even though
    the underlying true Sharpe is unchanged — the Geltner appraisal-smoothing
    artifact that plagues every appraisal-based 'alternative asset' index.

    Returns ``(df, truth)`` where ``df`` has columns ``year`` (int),
    ``index_level`` (reported/appraised level), ``true_return`` and
    ``reported_return`` (annual simple returns), and ``truth`` records the
    planted parameters.
    """
    rng = np.random.default_rng(seed)
    years = list(range(2008, 2008 + n_years))
    true_log = rng.normal(true_drift, true_vol, n_years)
    true_simple = np.expm1(true_log)

    reported = np.empty(n_years, dtype=float)
    prev = true_simple[0]
    for t in range(n_years):
        if t == 0:
            reported[t] = true_simple[t]
        else:
            reported[t] = (1.0 - smoothing) * true_simple[t] + smoothing * prev
        prev = reported[t]

    level = base_level * np.cumprod(1.0 + reported)

    df = pd.DataFrame({
        "year": years,
        "index_level": level,
        "true_return": true_simple,
        "reported_return": reported,
    })
    truth = {
        "n_years": n_years,
        "true_drift": true_drift,
        "true_vol": true_vol,
        "smoothing": smoothing,
        "base_level": base_level,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real data — S&P 500 (^GSPC) annual price returns (repo-level cache)
# ---------------------------------------------------------------------------
def fetch_gspc_annual(
    cache_dir: str = REPO_CACHE,
    start_year: int = 2008,
    end_year: int = 2024,
    fetch: bool = False,
) -> pd.DataFrame:
    """Load S&P 500 (^GSPC) calendar-year *price* returns and join with the whisky index.

    Cache-only by default. The repo-level ``_cache/^GSPC_split_only.parquet`` holds
    daily OHLC for ^GSPC; we take the last close of each calendar year and compute
    the December/December price change. With ``fetch=True`` we lazily import
    yfinance and pull live (network) — never used by tests or the offline core.

    Returns a DataFrame with columns:
      ``year``, ``index_level`` (whisky), ``whisky_return``,
      ``sp500_return`` (price-only), ``both_up`` (bool), ``whisky_beats_sp`` (bool).

    Raises ``FileNotFoundError`` if the cache is absent and ``fetch=False``.
    """
    gspc_path = os.path.join(cache_dir, "^GSPC_split_only.parquet")

    if os.path.exists(gspc_path):
        px = pd.read_parquet(gspc_path)
    elif fetch:
        import yfinance as yf  # lazy, network-only

        px = yf.download("^GSPC", start="2007-06-01", end=f"{end_year + 1}-01-15",
                         auto_adjust=False, progress=False)
        if isinstance(px.columns, pd.MultiIndex):
            px.columns = px.columns.get_level_values(0)
    else:
        raise FileNotFoundError(
            f"^GSPC cache not found at {gspc_path}. Pass fetch=True for a live pull "
            "(requires network + yfinance), or stage the repo-level cache."
        )

    if not isinstance(px.index, pd.DatetimeIndex):
        px.index = pd.to_datetime(px.index)
    px = px.sort_index()

    # Last close of each calendar year.
    close = px["Close"].copy()
    close.index = pd.to_datetime(close.index)
    year_end = close.groupby(close.index.year).last()
    year_end = year_end[(year_end.index >= start_year - 1) & (year_end.index <= end_year)]
    sp_ret = year_end.pct_change().dropna()
    sp_df = pd.DataFrame({"year": sp_ret.index.astype(int),
                          "sp500_return": sp_ret.values})

    wh = whisky_index()
    wh = wh[(wh["year"] >= start_year) & (wh["year"] <= end_year)].copy()

    merged = wh.merge(sp_df, on="year", how="inner")
    merged = merged.dropna(subset=["whisky_return", "sp500_return"]).reset_index(drop=True)
    merged["both_up"] = (merged["whisky_return"] > 0) & (merged["sp500_return"] > 0)
    merged["whisky_beats_sp"] = merged["whisky_return"] > merged["sp500_return"]
    return merged


def fingerprint(df: pd.DataFrame, col: str = "whisky_return") -> str:
    """A short content fingerprint of a return column, for the as-of stamp."""
    vals = df[col].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(np.round(vals, 8)).tobytes())
    return h.hexdigest()[:12]
