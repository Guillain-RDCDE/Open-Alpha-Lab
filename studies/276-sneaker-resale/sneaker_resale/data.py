"""Data layer for Study 276 (Sneaker-Resale).

Three components, all fully offline for the core:

- ``SNEAKER_RESALE_INDEX`` — a hardcoded, curated **annual sneaker-resale price
  index** (level, base 100 in 2006). It tracks the broad secondary-market price of
  collectible / hyped sneakers (Jordan, Yeezy, Dunk, etc.) as proxied by the public
  reporting of the major resale marketplaces. The shape is well documented: a slow
  grind from the mid-2000s, an explosive boom once StockX (2016) and GOAT
  industrialised the secondary market, a 2020–2021 pandemic peak, and a sharp
  2022–2024 bust as resale premiums compressed and the marketplaces shed staff.
  This series is a *curated, smoothed appraisal index*, not a tradable price — that
  distinction is the heart of the study. Hardcoded so the core is fully offline and
  reproducible.

- ``synthetic_panel`` — a deterministic, offline annual-return generator with an
  optional planted "sneaker alpha" knob AND an optional AR(1) smoothing knob (to
  reproduce the stale-appraisal autocorrelation seen in the real index). A
  ``alpha_bps = 0`` / ``smooth_rho = 0`` setting is the null; tests confirm the
  machinery is truthful before looking at real data.

- ``fetch_gspc_annual`` — real ^GSPC (S&P 500 price index) calendar-year returns
  from the repo-level cache (``_cache/^GSPC_split_only.parquet``). Cache-only by
  default (``fetch=False``); a ``fetch=True`` path lazily imports yfinance to
  refresh the cache, and is never exercised in tests or CI.

No look-ahead: the sneaker index is an end-of-year level, and we match calendar-year
sneaker returns to *the same* calendar-year S&P return. The strategy layer applies a
one-year execution lag for the "trade the trend" variant so nothing is timed on
information that was not yet observable.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "_cache"))

SEED = 276

# ---------------------------------------------------------------------------
# The hardcoded sneaker-resale index — the study's deterministic offline core
# ---------------------------------------------------------------------------
# Annual end-of-year level of a broad sneaker-resale price index, base 100 in 2006.
#
# This is a *curated* index built to mirror the publicly reported shape of the
# secondary sneaker market. It is intentionally an annual, appraisal-style series:
# resale "prices" are aggregated, lagged, and reported infrequently, so the index is
# heavily smoothed relative to a continuously-traded asset. That smoothing is not a
# bug — it is the central illusion the study dissects.
#
# Narrative the levels encode (all widely documented in the sneaker-resale press):
#   - 2006-2014 : slow collector-driven grind, single-digit annual appreciation.
#   - 2016-2021 : the StockX / GOAT boom; the secondary market industrialises and
#                 prices/premiums explode (the index roughly 2.4x's 2016->2021).
#   - 2022-2024 : the bust. Resale premiums compress, hype cools, marketplaces cut
#                 staff and valuations; the index gives back a large chunk of the boom.
#
# Sources (for the SHAPE, not exact tick data — there is no clean public daily tape):
#   - StockX "Big Facts" / "Current Culture" annual market reports.
#   - GOAT Group and secondary-market press (Business of Fashion, Complex, Hypebeast).
#   - Cowen / Piper Sandler sell-side notes on the resale market size (~$2B -> ~$10B+).
#   - Academic notes on collectibles as an asset class (Dimson-Spaenjers lineage).
#
# As-of: 2026-06-17. This table is deterministic and hardcoded to keep the core offline.

SNEAKER_RESALE_INDEX: dict[int, float] = {
    2006: 100.0,
    2007: 104.0,
    2008: 101.0,
    2009: 106.0,
    2010: 112.0,
    2011: 118.0,
    2012: 126.0,
    2013: 137.0,
    2014: 150.0,
    2015: 168.0,
    2016: 196.0,
    2017: 233.0,
    2018: 280.0,
    2019: 330.0,
    2020: 392.0,
    2021: 470.0,  # pandemic-era peak
    2022: 405.0,  # bust begins
    2023: 360.0,
    2024: 338.0,
}


def sneaker_index() -> pd.DataFrame:
    """Return the hardcoded sneaker-resale index as a tidy DataFrame.

    Columns: ``year`` (int), ``level`` (float, base 100 in 2006),
    ``sneaker_return`` (float, year-over-year simple return; NaN for the first year).
    """
    s = pd.Series(SNEAKER_RESALE_INDEX, name="level").sort_index()
    df = s.to_frame()
    df["sneaker_return"] = df["level"].pct_change()
    df = df.reset_index().rename(columns={"index": "year"})
    return df


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core (with planted-effect knobs)
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_years: int = 18,
    alpha_bps: float = 0.0,
    smooth_rho: float = 0.0,
    base_return: float = 0.08,
    vol_ann: float = 0.20,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible annual panel of (sneaker, gspc) returns with planted knobs.

    The "true" (economic) sneaker return is drawn i.i.d. as
    ``base_return + alpha_bps*1e-4 + N(0, vol_ann^2)``. To mimic the real index's
    stale-appraisal behaviour, the *reported* sneaker return is an exponential
    moving blend ``r_obs_t = rho*r_obs_{t-1} + (1-rho)*r_true_t`` when
    ``smooth_rho > 0``. The S&P leg is a separate i.i.d. draw with no smoothing.

    ``alpha_bps = 0`` (no outperformance) and ``smooth_rho = 0`` (no smoothing) is
    the null — the sneaker leg neither beats stocks nor looks artificially calm.

    Returns ``(df, truth)`` where ``df`` has columns ``year`` (int),
    ``sneaker_return``, ``gspc_return`` and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    years = list(range(2007, 2007 + n_years))

    r_true = base_return + alpha_bps * 1e-4 + rng.normal(0.0, vol_ann, n_years)
    if smooth_rho > 0.0:
        r_obs = np.empty(n_years)
        prev = base_return + alpha_bps * 1e-4
        for i in range(n_years):
            prev = smooth_rho * prev + (1.0 - smooth_rho) * r_true[i]
            r_obs[i] = prev
    else:
        r_obs = r_true

    gspc = base_return + rng.normal(0.0, 0.18, n_years)

    df = pd.DataFrame({
        "year": years,
        "sneaker_return": r_obs,
        "gspc_return": gspc,
    })
    truth = {
        "n_years": n_years,
        "alpha_bps": alpha_bps,
        "smooth_rho": smooth_rho,
        "base_return": base_return,
        "vol_ann": vol_ann,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real data — ^GSPC annual returns (repo-level cache, read-only by default)
# ---------------------------------------------------------------------------
def fetch_gspc_annual(
    cache_dir: str = REPO_CACHE,
    start_year: int = 2007,
    end_year: int = 2024,
    fetch: bool = False,
) -> pd.DataFrame:
    """Load ^GSPC (S&P 500 price index) calendar-year returns, joined to the index.

    Cache-only by default. The parquet ``_cache/^GSPC_split_only.parquet`` ships with
    the repo and is read-only. We compute the December-to-December last-trading-day
    price return for each calendar year and join with the hardcoded sneaker index on
    ``year``.

    With ``fetch=True`` the function lazily imports yfinance to refresh the cache —
    this path is never used in tests or CI (offline & deterministic).

    Returns a DataFrame with columns:
      ``year``, ``sneaker_return``, ``gspc_return``, ``excess`` (sneaker - gspc),
      ``sneaker_up`` (bool), ``gspc_up`` (bool).

    Raises ``FileNotFoundError`` if the ^GSPC cache is absent and ``fetch=False``.
    """
    gspc_path = os.path.join(cache_dir, "^GSPC_split_only.parquet")

    if not os.path.exists(gspc_path):
        if not fetch:
            raise FileNotFoundError(
                f"^GSPC cache not found at {gspc_path}. "
                "Pass fetch=True to download (requires network + yfinance), "
                "or stage the repo-level _cache/^GSPC_split_only.parquet."
            )
        import yfinance as yf  # lazy: never imported in tests/CI

        raw = yf.download("^GSPC", start="2005-01-01", end=f"{end_year + 1}-01-01",
                          auto_adjust=False, progress=False)
        os.makedirs(cache_dir, exist_ok=True)
        raw[["Open", "High", "Low", "Close", "Volume"]].to_parquet(gspc_path)

    px = pd.read_parquet(gspc_path)
    if "Close" not in px.columns:
        raise ValueError("^GSPC cache missing a 'Close' column.")
    if not isinstance(px.index, pd.DatetimeIndex):
        px.index = pd.to_datetime(px.index)

    dec = px[px.index.month == 12].copy()
    dec["year"] = dec.index.year
    yr_end = dec.groupby("year")["Close"].last()
    gret = yr_end.pct_change()

    g = gret.reset_index()
    g.columns = ["year", "gspc_return"]
    g = g[(g["year"] >= start_year) & (g["year"] <= end_year)].dropna()

    sn = sneaker_index()[["year", "sneaker_return"]]
    sn = sn[(sn["year"] >= start_year) & (sn["year"] <= end_year)].dropna()

    merged = sn.merge(g, on="year", how="inner").sort_values("year").reset_index(drop=True)
    merged["excess"] = merged["sneaker_return"] - merged["gspc_return"]
    merged["sneaker_up"] = merged["sneaker_return"] > 0
    merged["gspc_up"] = merged["gspc_return"] > 0
    return merged


def fingerprint(df: pd.DataFrame, col: str = "gspc_return") -> str:
    """A short content fingerprint of a return column, for the as-of stamp."""
    vals = df[col].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]
