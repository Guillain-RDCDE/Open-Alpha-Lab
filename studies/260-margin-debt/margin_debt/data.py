"""Data layer for Study 260 (Margin-Debt).

The folklore: "margin debt is at a record high -- the market is about to crash."
We test the contrarian version of the claim: does the year-over-year change in
NYSE/FINRA customer margin debt forecast the *next* year's S&P 500 return?

Three components, all offline for the reproducible core:

- ``MARGIN_DEBT_YEAREND`` -- a hardcoded table of December (year-end) NYSE/FINRA
  customer **debit balances in securities margin accounts** (USD millions),
  1959--2025.  Through 2009 these are the NYSE member-firm figures; from 2010 on
  FINRA took over the collection (the series is continuous and is the one charted
  by Advisor Perspectives, Yardeni, and the WSJ "margin debt" headlines).  The
  table is small, well-known, and deterministic -- hardcoded here so the core is
  fully offline and reproducible.  Values are rounded to the nearest $100m for the
  early decades where only round figures are reported.

- ``synthetic_series`` -- a deterministic, offline generator that builds a
  margin-debt level co-integrated with a synthetic equity index, with an optional
  ``predictive_beta`` knob that plants a *real* contrarian relationship (high YoY
  margin growth -> low forward return).  ``predictive_beta = 0`` is the null: margin
  growth carries no forward information and merely trends with the market.  This is
  the study's null/positive control in a bottle.

- ``fetch_sp500_monthly`` -- real S&P 500 month-end closes from the repo-level
  cache ``_cache/^GSPC_split_only.parquet`` (price-only, split-adjusted, NOT
  total-return).  Cache-only by default; network is touched only on an explicit
  ``fetch=True`` (lazy yfinance import).

No look-ahead: the signal at the end of December of year Y is the YoY change in
margin debt *through that December* (margin debt is reported with a ~3-4 week lag,
so December's figure is published in late January -- we conservatively enter the
position at the end of *January* of Y+1, a one-month execution lag, and measure the
forward 12-month return from there).

**Price-only caveat**: ``^GSPC_split_only`` excludes dividends, so all forward
returns understate total return by ~2%/yr.  This is labeled on the Signal axis and
makes the contrarian "sell" case *look better* than it is -- i.e. it is a
conservative choice for a contrarian claim.
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

# ---------------------------------------------------------------------------
# The hardcoded NYSE/FINRA year-end margin-debt table -- the offline core
# ---------------------------------------------------------------------------
# December customer debit balances in securities margin accounts, USD millions.
#   1959-2009 : NYSE member-firm margin debt (NYSE "Securities Market Credit").
#   2010-2025 : FINRA member-firm margin debt (continuous successor series).
# Sources: NYSE Factbook / FINRA "Margin Statistics"; charted by
#   Advisor Perspectives (Jill Mislinski / Jennifer Nash), Ed Yardeni, the WSJ.
# These are December values (the year-end snapshot used for the YoY "record" story).
# Early-decade figures are reported to the nearest $100m.
MARGIN_DEBT_YEAREND: dict[int, float] = {
    1959: 3_990, 1960: 3_550, 1961: 4_259, 1962: 3_873, 1963: 4_877,
    1964: 5_271, 1965: 5_953, 1966: 6_220, 1967: 7_148, 1968: 8_540,
    1969: 6_580, 1970: 6_220, 1971: 7_520, 1972: 8_587, 1973: 6_990,
    1974: 5_510, 1975: 6_750, 1976: 8_170, 1977: 8_810, 1978: 11_300,
    1979: 12_530, 1980: 14_240, 1981: 14_410, 1982: 14_900, 1983: 22_300,
    1984: 23_410, 1985: 28_180, 1986: 36_750, 1987: 30_660, 1988: 33_550,
    1989: 34_320, 1990: 28_320, 1991: 36_660, 1992: 43_960, 1993: 60_690,
    1994: 60_355, 1995: 77_330, 1996: 97_400, 1997: 129_540, 1998: 141_400,
    1999: 228_530, 2000: 198_700, 2001: 157_780, 2002: 130_240, 2003: 173_300,
    2004: 199_590, 2005: 221_400, 2006: 275_380, 2007: 322_640, 2008: 184_960,
    2009: 231_440, 2010: 276_690, 2011: 298_980, 2012: 339_510, 2013: 444_930,
    2014: 456_230, 2015: 435_010, 2016: 480_650, 2017: 580_960, 2018: 554_290,
    2019: 579_660, 2020: 778_200, 2021: 910_140, 2022: 619_310, 2023: 698_870,
    2024: 815_490, 2025: 935_870,
}


def margin_debt_table() -> pd.DataFrame:
    """Return the hardcoded year-end margin-debt series as a tidy DataFrame.

    Columns: ``year`` (int), ``margin_debt`` (USD millions, December snapshot),
    ``yoy`` (year-over-year fractional change, NaN for the first year).
    """
    years = sorted(MARGIN_DEBT_YEAREND)
    vals = np.array([float(MARGIN_DEBT_YEAREND[y]) for y in years])
    df = pd.DataFrame({"year": years, "margin_debt": vals})
    df["yoy"] = df["margin_debt"].pct_change()
    return df


# ---------------------------------------------------------------------------
# Synthetic series -- deterministic offline null / positive control
# ---------------------------------------------------------------------------
def synthetic_series(
    n_years: int = 67,
    predictive_beta: float = 0.0,
    base_ret: float = 0.07,
    ret_vol: float = 0.16,
    margin_beta: float = 1.5,
    margin_noise: float = 0.08,
    seed: int = 260,
) -> tuple[pd.DataFrame, dict]:
    """A margin-debt level that co-moves with a synthetic equity index.

    Each year the index earns ``base_ret + noise``.  Margin debt grows roughly
    ``margin_beta`` times the *contemporaneous* index return (leverage expands in
    up years, contracts in down years) plus its own noise -- this is the realistic
    "margin debt just trends with the market" null.

    The ``predictive_beta`` knob optionally plants a *forward* contrarian signal:
    next year's return is reduced by ``predictive_beta`` times this year's
    standardized YoY margin growth.  ``predictive_beta = 0`` is the pure null:
    margin growth is contemporaneously correlated with the market but carries no
    forward information.

    Returns ``(df, truth)`` where ``df`` has columns ``year``, ``index_ret``
    (annual simple return), ``margin_debt`` (level), ``yoy`` (margin YoY), and
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    years = list(range(1959, 1959 + n_years))

    # First pass: index returns and contemporaneous margin growth.
    shocks = rng.normal(0.0, ret_vol, n_years)
    m_noise = rng.normal(0.0, margin_noise, n_years)

    index_ret = np.empty(n_years)
    margin = np.empty(n_years)
    yoy = np.full(n_years, np.nan)
    margin[0] = 4000.0
    # Build sequentially so the planted forward effect can feed back.
    prev_z = 0.0
    for t in range(n_years):
        index_ret[t] = base_ret + shocks[t] - predictive_beta * prev_z
        if t > 0:
            g = margin_beta * index_ret[t] + m_noise[t]
            g = max(g, -0.6)  # margin debt cannot fall more than 60% in a year
            margin[t] = margin[t - 1] * (1.0 + g)
            yoy[t] = margin[t] / margin[t - 1] - 1.0
        # standardized YoY (rough, fixed scale) to drive next year's forward effect
        prev_z = (yoy[t] - 0.10) / 0.20 if t > 0 and np.isfinite(yoy[t]) else 0.0

    df = pd.DataFrame(
        {"year": years, "index_ret": index_ret, "margin_debt": margin, "yoy": yoy}
    )
    truth = {
        "n_years": n_years,
        "predictive_beta": predictive_beta,
        "base_ret": base_ret,
        "ret_vol": ret_vol,
        "margin_beta": margin_beta,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape -- S&P 500 month-end closes from the repo cache (price-only)
# ---------------------------------------------------------------------------
def fetch_sp500_monthly(
    fetch: bool = False,
    cache_path: str = GSPC_CACHE,
) -> pd.Series:
    """Month-end S&P 500 (^GSPC) closes; cache-only unless ``fetch=True``.

    Reads the repo-level ``_cache/^GSPC_split_only.parquet`` daily frame and
    resamples to month-end last close.  This is **price-only** (split-adjusted,
    dividends NOT reinvested) -- labeled on the Signal axis.

    Network is touched only on an explicit ``fetch=True`` (lazy yfinance import),
    after which the daily frame is re-cached.  Returns a month-end ``pd.Series``
    of closes (name ``GSPC``).
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached S&P 500 daily frame at {cache_path}. "
                "The repo-level _cache/^GSPC_split_only.parquet must be present, "
                "or call fetch_sp500_monthly(fetch=True) once."
            )
        daily = pd.read_parquet(cache_path)
    else:
        import yfinance as yf  # lazy import: network only on fetch=True

        raw = yf.download("^GSPC", start="1959-01-01", auto_adjust=False, progress=False)
        if raw.empty:
            raise RuntimeError("yfinance returned no ^GSPC data")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        daily = raw[["Close"]].copy()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        daily.to_parquet(cache_path)

    daily.index = pd.to_datetime(daily.index)
    close = daily["Close"] if "Close" in daily.columns else daily.iloc[:, 0]
    monthly = close.resample("ME").last().dropna()
    monthly.name = "GSPC"
    return monthly


def build_panel(
    fetch: bool = False,
    cache_path: str = GSPC_CACHE,
    lag_months: int = 1,
) -> pd.DataFrame:
    """Join the hardcoded margin-debt YoY signal with the S&P 500 forward return.

    For each year Y with a margin-debt YoY value, we:
      - take the S&P 500 month-end close at the end of December of year Y,
      - apply a ``lag_months`` execution lag (margin debt is reported ~1 month
        late; default lag = 1 month, so we enter at end-January of Y+1),
      - measure the forward 12-month price-only return from the lagged entry.

    Returns a DataFrame with columns:
      ``year``, ``margin_debt``, ``yoy`` (margin YoY through Dec Y),
      ``fwd_ret`` (next-12m S&P price return from the lagged entry),
      ``mkt_up`` (bool: fwd_ret > 0).

    Raises ``FileNotFoundError`` if the S&P cache is absent and ``fetch=False``.
    """
    monthly = fetch_sp500_monthly(fetch=fetch, cache_path=cache_path)
    md = margin_debt_table()

    rows: list[dict] = []
    for _, r in md.iterrows():
        y = int(r["year"])
        if not np.isfinite(r["yoy"]):
            continue
        entry = pd.Timestamp(year=y, month=12, day=31) + pd.offsets.MonthEnd(lag_months)
        exit_ = entry + pd.offsets.MonthEnd(12)
        # nearest available month-end <= target (month-end index is exact here)
        if entry not in monthly.index or exit_ not in monthly.index:
            continue
        p0 = float(monthly.loc[entry])
        p1 = float(monthly.loc[exit_])
        fwd = p1 / p0 - 1.0
        rows.append(
            {
                "year": y,
                "margin_debt": float(r["margin_debt"]),
                "yoy": float(r["yoy"]),
                "fwd_ret": fwd,
                "mkt_up": fwd > 0,
            }
        )
    panel = pd.DataFrame(rows).reset_index(drop=True)
    return panel


def fingerprint(df: pd.DataFrame, col: str = "fwd_ret") -> str:
    """A short content fingerprint of one column, for the as-of stamp."""
    vals = df[col].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]
