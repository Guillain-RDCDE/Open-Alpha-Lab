"""Data layer for Study 281 (El-Nino).

Two components, both fully offline for the deterministic core:

- ``ONI_NDJ`` -- a hardcoded table of NOAA's Oceanic Nino Index (ONI), the
  canonical 3-month-running-mean SST anomaly in the Nino-3.4 region that defines
  the El Nino-Southern Oscillation (ENSO) state. We hardcode the **NDJ**
  (November-December-January) season value for each ENSO year, because NDJ is the
  December-centered peak of the ENSO cycle and the cleanest single label for "what
  phase was the ocean in this winter". Each row carries the ENSO *year* (the year
  of the December in the NDJ triple), the ONI value, and the derived ``phase``:

      ONI >= +0.5  -> "El Nino"
      ONI <= -0.5  -> "La Nina"
      else         -> "Neutral"

  Source: NOAA Climate Prediction Center, Cold & Warm Episodes by Season
  (https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php).
  The table is small, stable, and well-known -- hardcoded here so the core is
  fully offline and reproducible.

- ``synthetic_panel`` -- a deterministic, offline annual-return generator with an
  optional planted "ENSO premium" knob. ``premium_bps = 0`` is the null (the ENSO
  phase carries no return information), so tests can confirm the machinery is
  truthful *before* anyone looks at the real tape.

- ``fetch_real`` -- real market returns joined to the ENSO phase, cache-only by
  default (``fetch=False``). Two tapes are available:
    * ``asset="equity"``  -> ^GSPC calendar-year price returns (Yahoo, 1950+),
    * ``asset="oil"``     -> USO (United States Oil Fund) calendar-year returns
                            (2006+), a tradable crude-oil proxy -- ENSO folklore
                            is loudest about commodities (crops, energy).
  Network access happens only on ``fetch=True`` via a lazy yfinance import.

No look-ahead: the ENSO state is *known by the end of the year* (NDJ peaks in
December). We deliberately use a one-year execution lag -- the phase observed in
the NDJ season ending in December of year Y is mapped to the *following*
calendar-year return (year Y+1). That is the only honest way to "trade" a winter
phase signal: you can act on it once the winter is over. We also report the
contemporaneous (same-year) join so the reader can see that the label-timing
choice is itself a degrees-of-freedom concern.
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
# The hardcoded NOAA ONI (NDJ season) table -- the deterministic offline core
# ---------------------------------------------------------------------------
# Columns:
#   enso_year : year of the December in the NDJ (Nov-Dec-Jan) triple. The NDJ
#               season "2023" spans Nov 2023, Dec 2023, Jan 2024.
#   oni_ndj   : NOAA ONI value (degC SST anomaly, Nino-3.4) for that NDJ season.
#   phase     : derived label, +/-0.5 threshold (filled by _classify below).
#
# Source: NOAA CPC ONI v5 table, "DJF...NDJ" seasonal columns, NDJ column.
# As-of: 2026-06-17. Values are the published 3-month running means; they are
# revised only at the second-decimal level across NOAA base-period updates, which
# does not move the +/-0.5 phase classification used here.
ONI_NDJ: list[dict] = [
    {"enso_year": 1950, "oni_ndj": -0.8},
    {"enso_year": 1951, "oni_ndj": 0.8},
    {"enso_year": 1952, "oni_ndj": 0.3},
    {"enso_year": 1953, "oni_ndj": 0.8},
    {"enso_year": 1954, "oni_ndj": -0.7},
    {"enso_year": 1955, "oni_ndj": -0.8},
    {"enso_year": 1956, "oni_ndj": -0.4},
    {"enso_year": 1957, "oni_ndj": 1.8},
    {"enso_year": 1958, "oni_ndj": 0.6},
    {"enso_year": 1959, "oni_ndj": -0.1},
    {"enso_year": 1960, "oni_ndj": -0.2},
    {"enso_year": 1961, "oni_ndj": -0.3},
    {"enso_year": 1962, "oni_ndj": -0.8},
    {"enso_year": 1963, "oni_ndj": 1.3},
    {"enso_year": 1964, "oni_ndj": -0.8},
    {"enso_year": 1965, "oni_ndj": 1.9},
    {"enso_year": 1966, "oni_ndj": -0.3},
    {"enso_year": 1967, "oni_ndj": -0.5},
    {"enso_year": 1968, "oni_ndj": 1.0},
    {"enso_year": 1969, "oni_ndj": -0.6},
    {"enso_year": 1970, "oni_ndj": -1.2},
    {"enso_year": 1971, "oni_ndj": -0.8},
    {"enso_year": 1972, "oni_ndj": 1.9},
    {"enso_year": 1973, "oni_ndj": -1.8},
    {"enso_year": 1974, "oni_ndj": -0.6},
    {"enso_year": 1975, "oni_ndj": -1.7},
    {"enso_year": 1976, "oni_ndj": 0.8},
    {"enso_year": 1977, "oni_ndj": 0.8},
    {"enso_year": 1978, "oni_ndj": -0.1},
    {"enso_year": 1979, "oni_ndj": 0.6},
    {"enso_year": 1980, "oni_ndj": -0.1},
    {"enso_year": 1981, "oni_ndj": -0.2},
    {"enso_year": 1982, "oni_ndj": 2.2},
    {"enso_year": 1983, "oni_ndj": -0.7},
    {"enso_year": 1984, "oni_ndj": -1.0},
    {"enso_year": 1985, "oni_ndj": -0.4},
    {"enso_year": 1986, "oni_ndj": 1.2},
    {"enso_year": 1987, "oni_ndj": 1.2},
    {"enso_year": 1988, "oni_ndj": -1.8},
    {"enso_year": 1989, "oni_ndj": 0.1},
    {"enso_year": 1990, "oni_ndj": 0.4},
    {"enso_year": 1991, "oni_ndj": 1.2},
    {"enso_year": 1992, "oni_ndj": 0.1},
    {"enso_year": 1993, "oni_ndj": 0.3},
    {"enso_year": 1994, "oni_ndj": 1.1},
    {"enso_year": 1995, "oni_ndj": -0.8},
    {"enso_year": 1996, "oni_ndj": -0.4},
    {"enso_year": 1997, "oni_ndj": 2.4},
    {"enso_year": 1998, "oni_ndj": -1.6},
    {"enso_year": 1999, "oni_ndj": -1.7},
    {"enso_year": 2000, "oni_ndj": -0.7},
    {"enso_year": 2001, "oni_ndj": -0.3},
    {"enso_year": 2002, "oni_ndj": 0.9},
    {"enso_year": 2003, "oni_ndj": 0.4},
    {"enso_year": 2004, "oni_ndj": 0.7},
    {"enso_year": 2005, "oni_ndj": -0.8},
    {"enso_year": 2006, "oni_ndj": 0.9},
    {"enso_year": 2007, "oni_ndj": -1.6},
    {"enso_year": 2008, "oni_ndj": -0.7},
    {"enso_year": 2009, "oni_ndj": 1.5},
    {"enso_year": 2010, "oni_ndj": -1.4},
    {"enso_year": 2011, "oni_ndj": -0.8},
    {"enso_year": 2012, "oni_ndj": -0.2},
    {"enso_year": 2013, "oni_ndj": -0.3},
    {"enso_year": 2014, "oni_ndj": 0.7},
    {"enso_year": 2015, "oni_ndj": 2.6},
    {"enso_year": 2016, "oni_ndj": -0.3},
    {"enso_year": 2017, "oni_ndj": -0.9},
    {"enso_year": 2018, "oni_ndj": 0.7},
    {"enso_year": 2019, "oni_ndj": 0.5},
    {"enso_year": 2020, "oni_ndj": -1.0},
    {"enso_year": 2021, "oni_ndj": -1.0},
    {"enso_year": 2022, "oni_ndj": -0.7},
    {"enso_year": 2023, "oni_ndj": 2.0},
    {"enso_year": 2024, "oni_ndj": -0.6},
]

PHASE_THRESHOLD = 0.5


def _classify(oni: float, thr: float = PHASE_THRESHOLD) -> str:
    if oni >= thr:
        return "El Nino"
    if oni <= -thr:
        return "La Nina"
    return "Neutral"


def oni_table(threshold: float = PHASE_THRESHOLD) -> pd.DataFrame:
    """Return a clean copy of the hardcoded NOAA ONI (NDJ) table.

    Columns: ``enso_year`` (int), ``oni_ndj`` (float), ``phase`` (str in
    {"El Nino", "La Nina", "Neutral"}), ``warm`` (bool: El Nino), ``cold`` (bool:
    La Nina).

    The phase is derived from ``oni_ndj`` with the standard +/-0.5 threshold; pass
    a different ``threshold`` to see how the (arbitrary) cutoff changes the labels.
    """
    df = pd.DataFrame(ONI_NDJ).copy()
    df["phase"] = df["oni_ndj"].apply(lambda v: _classify(v, threshold))
    df["warm"] = df["phase"] == "El Nino"
    df["cold"] = df["phase"] == "La Nina"
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic annual-return generator -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_years: int = 75,
    premium_bps: float = 0.0,
    base_return: float = 0.07,
    vol_ann: float = 0.17,
    seed: int = 281,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible annual-return series with an optional planted ENSO premium.

    Each year's ENSO phase is drawn to roughly match the historical mix
    (~35% El Nino, ~35% La Nina, ~30% Neutral). The annual return is::

        base_return + premium_bps*1e-4 * (phase == "El Nino")
                    - premium_bps*1e-4 * (phase == "La Nina") + noise

    so a positive ``premium_bps`` plants an El-Nino-bullish / La-Nina-bearish
    effect. ``premium_bps = 0`` is the null -- the phase is uncorrelated with the
    return. This is the study's null in a bottle.

    Returns ``(df, truth)`` where ``df`` has columns ``year`` (int), ``ret``
    (annual simple return), ``oni_ndj`` (float), ``phase`` (str), ``warm``,
    ``cold`` (bool).
    """
    rng = np.random.default_rng(seed)
    years = list(range(1950, 1950 + n_years))
    phases = rng.choice(["El Nino", "La Nina", "Neutral"], size=n_years,
                        p=[0.35, 0.35, 0.30])
    # A plausible ONI magnitude per phase (sign-correct), for display only.
    oni = np.where(phases == "El Nino", rng.uniform(0.5, 2.5, n_years),
                   np.where(phases == "La Nina", -rng.uniform(0.5, 2.0, n_years),
                            rng.uniform(-0.4, 0.4, n_years)))
    drift = (base_return
             + premium_bps * 1e-4 * (phases == "El Nino")
             - premium_bps * 1e-4 * (phases == "La Nina"))
    rets = drift + rng.normal(0.0, vol_ann, n_years)

    df = pd.DataFrame({
        "year": years,
        "ret": rets,
        "oni_ndj": np.round(oni, 2),
        "phase": phases,
        "warm": phases == "El Nino",
        "cold": phases == "La Nina",
    })
    truth = {
        "n_years": n_years,
        "premium_bps": premium_bps,
        "base_return": base_return,
        "vol_ann": vol_ann,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real data -- ^GSPC / USO calendar-year returns joined to the ENSO phase
# ---------------------------------------------------------------------------
_ASSET_TICKER = {"equity": "^GSPC", "oil": "USO"}
_ASSET_CACHE = {
    "equity": ["^GSPC_split_only.parquet"],
    "oil": ["USO_split_only.parquet"],
}


def _load_cached_prices(asset: str, cache_dir: str) -> pd.DataFrame | None:
    """Return the cached daily OHLC frame for ``asset`` or None if absent."""
    for name in _ASSET_CACHE[asset]:
        path = os.path.join(cache_dir, name)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if "Date" in df.columns:
                df.index = pd.to_datetime(df["Date"])
                df = df.drop(columns=["Date"])
            df.index = pd.to_datetime(df.index)
            return df
    return None


def _annual_returns_from_daily(px: pd.DataFrame) -> pd.DataFrame:
    """December-to-December calendar-year price returns from a daily OHLC frame."""
    close = px["Close"].dropna()
    yearly = close.resample("YE").last()
    rets = yearly.pct_change().dropna()
    out = pd.DataFrame({"year": rets.index.year, "asset_return": rets.values})
    return out.reset_index(drop=True)


def fetch_real(
    asset: str = "equity",
    lag: int = 1,
    fetch: bool = False,
    cache_dir: str = REPO_CACHE,
    threshold: float = PHASE_THRESHOLD,
    start_year: int = 1950,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Load real calendar-year returns joined to the ENSO phase (cache-only by default).

    Parameters
    ----------
    asset : "equity" (^GSPC) or "oil" (USO crude-oil proxy).
    lag : execution lag in years. ``lag=1`` (default, the honest convention) maps
        the NDJ phase ending in December of year Y to the year Y+1 return -- you can
        only act on a winter phase once the winter is over. ``lag=0`` is the
        contemporaneous join (shown for the degrees-of-freedom discussion).
    fetch : if False (default), read from the repo-level cache only and raise
        FileNotFoundError when absent -- never touches the network. If True, fall
        back to a lazy yfinance download when the cache is missing.
    cache_dir : where the cached parquet lives (repo-level ``_cache`` by default).

    Returns a DataFrame with columns:
      ``year`` (return year), ``asset_return``, ``enso_year``, ``oni_ndj``,
      ``phase``, ``warm``, ``cold``, ``mkt_up`` (asset_return > 0).
    """
    if asset not in _ASSET_TICKER:
        raise ValueError(f"asset must be one of {list(_ASSET_TICKER)}, got {asset!r}")

    px = _load_cached_prices(asset, cache_dir)
    if px is None:
        if not fetch:
            raise FileNotFoundError(
                f"No cached prices for asset={asset!r} in {cache_dir}. "
                "Pass fetch=True to download via yfinance (network)."
            )
        import yfinance as yf  # lazy import -- only on explicit fetch

        ticker = _ASSET_TICKER[asset]
        px = yf.download(ticker, start="1950-01-01", auto_adjust=True, progress=False)
        if isinstance(px.columns, pd.MultiIndex):
            px.columns = px.columns.get_level_values(0)
        px.index = pd.to_datetime(px.index)

    ann = _annual_returns_from_daily(px)

    oni = oni_table(threshold=threshold)
    # Apply the execution lag: NDJ phase of enso_year Y -> return year Y + lag.
    oni = oni.assign(year=oni["enso_year"] + lag)

    merged = ann.merge(oni, on="year", how="inner")
    merged = merged[(merged["year"] >= start_year) & (merged["year"] <= end_year)].copy()
    merged["mkt_up"] = merged["asset_return"] > 0
    return merged.reset_index(drop=True)


def fingerprint(df: pd.DataFrame, col: str = "asset_return") -> str:
    """A short content fingerprint of the return column, for the as-of stamp."""
    vals = df[col].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]
