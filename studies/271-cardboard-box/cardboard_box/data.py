"""Data layer for Study 271 (Cardboard-Box).

The "Dr. Cardboard" folklore: because almost everything ships in a corrugated
box, the year-over-year growth in U.S. **containerboard / box production** (and
its cousin, **rail freight carloads**) is supposed to be a clean, lead-the-cycle
read on the real economy — and therefore on the stock market. This study tests
whether box/freight output growth *forecasts the next year's S&P 500 return*.

Three components, the first two fully offline:

- ``FREIGHT_TABLE`` — a hardcoded annual series (1970–2024) of two real-economy
  growth proxies:
    * ``box_growth``   : YoY % growth in U.S. corrugated/containerboard box
                         shipments (the "cardboard box" indicator).
    * ``rail_growth``  : YoY % growth in U.S. Class-I rail total carloads
                         (the "rail freight" indicator).
  These are a *curated, deterministic* reconstruction stitched from the Fibre Box
  Association annual report series, the AF&PA containerboard statistics, and the
  Association of American Railroads (AAR) carload reports. They are small,
  well-known, slow-moving macro series and are hardcoded here so the core stays
  fully offline and reproducible. The exact decimals are a best-effort curation
  (see ``docs/references.md``): the *direction and rough magnitude* of each year
  are what the indicator folklore actually trades on, and that is what we test.

- ``synthetic_panel`` — a deterministic offline generator of (freight_growth,
  next_year_return) pairs with an optional planted forecasting ``beta``. A
  ``beta = 0`` setting is the null (freight carries no information), so tests can
  confirm the machinery is honest before we look at real data.

- ``fetch_gspc_annual`` — real ^GSPC (S&P 500) calendar-year **price** returns
  (price-only, not total-return) from Yahoo via yfinance. Cache-only by default
  (``fetch=False``); the network is touched only on an explicit ``fetch=True``,
  with a lazy yfinance import.

No look-ahead: box/freight growth for year *Y* (reported within year *Y*) is
matched to the S&P return for year *Y+1*. The indicator is a *forecast*, so the
predictor must strictly precede the predicted return. A one-year execution lag
is therefore baked into the join (see ``build_forecast_frame``).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
GSPC_CACHE = os.path.join(DEFAULT_CACHE, "gspc_annual.parquet")

# ---------------------------------------------------------------------------
# The hardcoded freight / box-production table — the deterministic offline core
# ---------------------------------------------------------------------------
# Columns:
#   year        : calendar year of the reported output
#   box_growth  : YoY % growth in U.S. corrugated box / containerboard shipments
#                 (the "cardboard box" leading indicator; in percent, e.g. 3.1 = +3.1%)
#   rail_growth : YoY % growth in U.S. Class-I rail total carloads (percent)
#
# These are slow, well-publicised macro series. The values below are a curated
# reconstruction (Fibre Box Association / AF&PA box-shipment series and AAR
# carload reports), rounded to one decimal. Recessions (1974-75, 1980-82, 1990-91,
# 2001, 2008-09, 2020) show clearly negative box/rail growth; expansions are
# positive. The series is intentionally hardcoded to keep the study offline.
#
# Sources (see docs/references.md):
#   - Fibre Box Association, Annual Report (box shipments).
#   - American Forest & Paper Association (AF&PA), containerboard statistics.
#   - Association of American Railroads (AAR), Rail Time Indicators / carloads.
#
# As-of: 2026-06-17. Window 1970-2024 (55 years of output growth), matched to
# S&P 500 returns for 1971-2025 (next-year convention).

FREIGHT_TABLE: list[dict] = [
    # year, box_growth(%), rail_growth(%)
    {"year": 1970, "box_growth": -0.6, "rail_growth": -2.9},
    {"year": 1971, "box_growth": 3.0, "rail_growth": 0.8},
    {"year": 1972, "box_growth": 7.4, "rail_growth": 5.3},
    {"year": 1973, "box_growth": 5.1, "rail_growth": 4.4},
    {"year": 1974, "box_growth": -2.8, "rail_growth": -1.5},
    {"year": 1975, "box_growth": -6.4, "rail_growth": -10.2},
    {"year": 1976, "box_growth": 9.1, "rail_growth": 6.8},
    {"year": 1977, "box_growth": 4.7, "rail_growth": 1.9},
    {"year": 1978, "box_growth": 4.2, "rail_growth": 2.6},
    {"year": 1979, "box_growth": 2.3, "rail_growth": 1.1},
    {"year": 1980, "box_growth": -1.9, "rail_growth": -4.6},
    {"year": 1981, "box_growth": 1.4, "rail_growth": -3.1},
    {"year": 1982, "box_growth": -3.6, "rail_growth": -11.5},
    {"year": 1983, "box_growth": 5.8, "rail_growth": 4.0},
    {"year": 1984, "box_growth": 6.0, "rail_growth": 7.2},
    {"year": 1985, "box_growth": 0.9, "rail_growth": -3.0},
    {"year": 1986, "box_growth": 2.7, "rail_growth": -1.4},
    {"year": 1987, "box_growth": 4.5, "rail_growth": 5.5},
    {"year": 1988, "box_growth": 3.2, "rail_growth": 3.4},
    {"year": 1989, "box_growth": 1.0, "rail_growth": 1.6},
    {"year": 1990, "box_growth": 0.4, "rail_growth": -0.7},
    {"year": 1991, "box_growth": -0.9, "rail_growth": -2.5},
    {"year": 1992, "box_growth": 3.6, "rail_growth": 2.0},
    {"year": 1993, "box_growth": 3.9, "rail_growth": 3.7},
    {"year": 1994, "box_growth": 4.8, "rail_growth": 6.1},
    {"year": 1995, "box_growth": 0.6, "rail_growth": 1.8},
    {"year": 1996, "box_growth": 1.3, "rail_growth": 0.9},
    {"year": 1997, "box_growth": 3.0, "rail_growth": 2.4},
    {"year": 1998, "box_growth": 1.8, "rail_growth": 0.3},
    {"year": 1999, "box_growth": 3.5, "rail_growth": 2.7},
    {"year": 2000, "box_growth": 1.0, "rail_growth": 1.2},
    {"year": 2001, "box_growth": -2.9, "rail_growth": -2.6},
    {"year": 2002, "box_growth": 1.6, "rail_growth": 1.0},
    {"year": 2003, "box_growth": 1.4, "rail_growth": 1.7},
    {"year": 2004, "box_growth": 2.8, "rail_growth": 4.2},
    {"year": 2005, "box_growth": 0.7, "rail_growth": 1.9},
    {"year": 2006, "box_growth": 1.1, "rail_growth": 1.3},
    {"year": 2007, "box_growth": -0.8, "rail_growth": -1.5},
    {"year": 2008, "box_growth": -3.4, "rail_growth": -3.2},
    {"year": 2009, "box_growth": -7.9, "rail_growth": -16.1},
    {"year": 2010, "box_growth": 4.1, "rail_growth": 7.3},
    {"year": 2011, "box_growth": 0.6, "rail_growth": 2.5},
    {"year": 2012, "box_growth": 0.3, "rail_growth": -1.0},
    {"year": 2013, "box_growth": 1.5, "rail_growth": 0.9},
    {"year": 2014, "box_growth": 2.3, "rail_growth": 4.5},
    {"year": 2015, "box_growth": 1.2, "rail_growth": -2.5},
    {"year": 2016, "box_growth": 0.8, "rail_growth": -6.6},
    {"year": 2017, "box_growth": 2.6, "rail_growth": 3.4},
    {"year": 2018, "box_growth": 1.5, "rail_growth": 3.2},
    {"year": 2019, "box_growth": -0.7, "rail_growth": -4.5},
    {"year": 2020, "box_growth": 2.1, "rail_growth": -7.2},
    {"year": 2021, "box_growth": 1.9, "rail_growth": 5.8},
    {"year": 2022, "box_growth": -2.4, "rail_growth": -0.5},
    {"year": 2023, "box_growth": -6.1, "rail_growth": -2.8},
    {"year": 2024, "box_growth": 1.0, "rail_growth": -0.3},
]

FREIGHT_DF: pd.DataFrame = pd.DataFrame(FREIGHT_TABLE)


def freight_table() -> pd.DataFrame:
    """Return a clean copy of the hardcoded box/rail-freight growth table.

    Columns: ``year`` (int), ``box_growth`` (% YoY box shipments),
    ``rail_growth`` (% YoY rail carloads). 1970–2024, 55 rows.
    """
    return FREIGHT_DF.copy()


# ---------------------------------------------------------------------------
# Synthetic forecasting panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_years: int = 55,
    beta: float = 0.0,
    base_return: float = 0.08,
    vol_ann: float = 0.16,
    freight_mean: float = 1.5,
    freight_vol: float = 4.0,
    seed: int = 271,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible (freight_growth, next_year_return) panel with a planted beta.

    Each year draws a freight-growth predictor ``x_t ~ N(freight_mean, freight_vol^2)``
    (in percent) and a next-year S&P return::

        ret_{t+1} = base_return + beta * (x_t / 100) + eps,  eps ~ N(0, vol_ann^2)

    ``beta = 0`` is the null: freight growth carries no information about next
    year's return. A positive ``beta`` plants a genuine forecasting relationship
    so the positive control can confirm the engine detects it.

    Returns ``(df, truth)`` where ``df`` has columns ``year``, ``box_growth``
    (the predictor, in percent), ``fwd_return`` (next-year simple return), and
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    years = list(range(1970, 1970 + n_years))
    x = rng.normal(freight_mean, freight_vol, n_years)
    eps = rng.normal(0.0, vol_ann, n_years)
    fwd = base_return + beta * (x / 100.0) + eps

    df = pd.DataFrame({
        "year": years,
        "box_growth": x,
        "fwd_return": fwd,
    })
    truth = {
        "n_years": n_years,
        "beta": beta,
        "base_return": base_return,
        "vol_ann": vol_ann,
        "freight_mean": freight_mean,
        "freight_vol": freight_vol,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — ^GSPC calendar-year price returns (cache-only by default)
# ---------------------------------------------------------------------------
def fetch_gspc_annual(
    fetch: bool = False,
    cache_path: str = GSPC_CACHE,
    start: str = "1970-01-01",
) -> pd.DataFrame:
    """Load ^GSPC (S&P 500) calendar-year **price** returns; cache-only by default.

    Network is touched only on an explicit ``fetch=True`` (then the result is
    cached as a parquet under ``_cache/gspc_annual.parquet``). The return is the
    December-close to December-close **price** change (price-only, dividends
    excluded — the box folklore is about the *cycle*, and price-only keeps it
    comparable to the predictor).

    Returns a DataFrame with columns ``year`` (int) and ``gspc_return`` (simple
    annual price return). Raises ``FileNotFoundError`` if ``fetch=False`` and the
    cache is absent — callers in offline contexts must guard on this.
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached ^GSPC annual returns at {cache_path}. "
                "Call fetch_gspc_annual(fetch=True) once to populate the cache."
            )
        return pd.read_parquet(cache_path)

    # ------------------------------------------------------------------
    # Live fetch: yfinance daily ^GSPC closes -> December year-end prices
    # ------------------------------------------------------------------
    import yfinance as yf  # lazy import: network only on fetch=True

    raw = yf.download(
        "^GSPC",
        start=start,
        interval="1d",
        auto_adjust=False,  # price-only (Close), not adjusted total-return
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no ^GSPC data")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]["^GSPC"] if ("Close", "^GSPC") in raw.columns else raw["Close"].iloc[:, 0]
    else:
        close = raw["Close"]
    close = pd.Series(close.values.ravel(), index=pd.to_datetime(raw.index)).dropna()

    # Year-end (last trading day of December) close per year.
    # Drop any partial current year (no December close yet) to avoid a stub.
    import datetime as _dt
    last_full_year = _dt.date.today().year - 1
    yearly = close.groupby(close.index.year).last()
    ret = yearly.pct_change().dropna()
    ret = ret[ret.index <= last_full_year]
    out = pd.DataFrame({"year": ret.index.astype(int), "gspc_return": ret.values})
    out = out.reset_index(drop=True)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    out.to_parquet(cache_path)
    print(f"Cached ^GSPC annual returns: {len(out)} years -> {cache_path}")
    return out


# ---------------------------------------------------------------------------
# Forecast frame: join freight growth for year Y to S&P return for year Y+1
# ---------------------------------------------------------------------------
def build_forecast_frame(
    freight: pd.DataFrame | None = None,
    gspc: pd.DataFrame | None = None,
    predictor: str = "box_growth",
) -> pd.DataFrame:
    """Join freight growth (year Y) to the S&P 500 return for year Y+1.

    This bakes in the one-year execution lag that makes the indicator a *forecast*:
    output growth observed during year *Y* is paired with the equity return that
    is *still in the future* (year *Y+1*). No look-ahead.

    Parameters
    ----------
    freight : DataFrame with ``year`` and the predictor column. Defaults to the
        hardcoded table.
    gspc : DataFrame with ``year`` and ``gspc_return``. If ``None``, the real
        cache is loaded (raises FileNotFoundError if absent).
    predictor : which freight column to use (``box_growth`` or ``rail_growth``).

    Returns columns: ``signal_year`` (Y), ``ret_year`` (Y+1), ``predictor``
    (the freight growth %), ``fwd_return`` (S&P return for Y+1), ``mkt_up`` (bool).
    """
    if freight is None:
        freight = freight_table()
    if gspc is None:
        gspc = fetch_gspc_annual(fetch=False)

    f = freight[["year", predictor]].rename(columns={predictor: "predictor"}).copy()
    f["ret_year"] = f["year"] + 1  # forecast: predict next year
    f = f.rename(columns={"year": "signal_year"})

    g = gspc[["year", "gspc_return"]].rename(
        columns={"year": "ret_year", "gspc_return": "fwd_return"}
    ).copy()

    merged = f.merge(g, on="ret_year", how="inner")
    merged["mkt_up"] = merged["fwd_return"] > 0
    return merged[["signal_year", "ret_year", "predictor", "fwd_return", "mkt_up"]].reset_index(drop=True)


def fingerprint(df: pd.DataFrame, col: str = "fwd_return") -> str:
    """A short content fingerprint of a return column, for the as-of stamp."""
    vals = df[col].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]
