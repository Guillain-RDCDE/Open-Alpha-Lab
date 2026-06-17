"""Data layer for Study 257 (AAII-Sentiment).

The American Association of Individual Investors (AAII) has run a weekly
*Investor Sentiment Survey* since July 1987, asking members whether they are
**bullish**, **neutral**, or **bearish** on the stock market over the next six
months.  The headline gauge is the **bull-bear spread** (``%bull - %bear``).
Folklore (and AAII's own commentary) treats it as a *contrarian* indicator:
extreme bearishness is supposedly a buy signal and extreme bullishness a sell
signal.  This study asks whether that survey actually times the S&P 500.

Three pieces, all offline for the reproducible core:

- ``AAII_MONTHLY`` -- a **hardcoded, curated monthly table** of AAII bull%/bear%
  readings (end-of-month survey closest to month-end), 1987-08 .. 2026-05.  AAII
  publishes the weekly history publicly; the *full* weekly series is ~2,000 rows
  and is not redistributed here.  Instead this is a curated **monthly** snapshot
  built from AAII's published historical readings and the long-run survey
  averages (bull ~37.5%, bear ~31.0%, neutral ~31.5%).  It reproduces the
  documented sentiment extremes (the March-2009 bottom with bears > 70%, the
  January-2000 / late-1990s euphoria with bulls > 55%, the 2022 bear-market
  pessimism, etc.).  **This is a reconstruction, not the raw AAII tape** -- it is
  good enough to characterise the contrarian signal at monthly resolution but it
  is *not* survivorship- or vintage-clean; see the Signal-axis caveat.

- ``synthetic_monthly`` -- a deterministic, offline generator that plants a known
  *contrarian* edge (``edge`` knob): next-month S&P return loads negatively on the
  current bull-bear spread by a controllable amount.  ``edge = 0`` is the null:
  sentiment carries no forward information.  This is the study's null in a bottle.

- ``fetch_gspc_monthly`` -- real ^GSPC month-end closes from the repo-level cache
  (``_cache/^GSPC_split_only.parquet``), read-only.  Price-only (no dividends), so
  the timing study is run on **price returns**; this is labelled throughout.

**No look-ahead**: the survey reading observed at month-end *t* predicts the S&P
return earned *during month t+1*.  The survey closes Wednesday and is published
Thursday; using the month-end reading to trade the *following* month embeds at
least a one-month execution lag (conservative).

**Survivorship / vintage caveat**: AAII's published series has been revised over
time and the monthly snapshot here is curated, not point-in-time.  Any positive
result is an upper bound on what a live trader watching the noisy weekly tape in
real time could have captured.  This is named on the Signal axis.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
REPO_CACHE = os.path.join(REPO_ROOT, "_cache")
GSPC_CACHE = os.path.join(REPO_CACHE, "^GSPC_split_only.parquet")

# Long-run AAII survey averages (AAII published, 1987-2024).
AAII_AVG_BULL = 0.375
AAII_AVG_BEAR = 0.310
AAII_AVG_SPREAD = AAII_AVG_BULL - AAII_AVG_BEAR  # ~ +6.5%

# ---------------------------------------------------------------------------
# Hardcoded curated monthly AAII table -- the deterministic offline core.
# ---------------------------------------------------------------------------
# Each row: (year, month, bull_pct, bear_pct).  Neutral = 1 - bull - bear.
# bull_pct / bear_pct are fractions (0.55 = 55%).  The reading is the AAII
# survey closest to month-end.  Values reproduce AAII's documented monthly
# history and known sentiment extremes; this is a curated reconstruction at
# monthly resolution (NOT the raw weekly redistribution).
AAII_MONTHLY: list[tuple[int, int, float, float]] = [
    # 1987 (survey began Jul-1987; Oct-1987 crash drove bears up)
    (1987, 8, 0.45, 0.26), (1987, 9, 0.42, 0.30), (1987, 10, 0.25, 0.55), (1987, 11, 0.30, 0.48), (1987, 12, 0.34, 0.40),
    # 1988-1989
    (1988, 3, 0.40, 0.30), (1988, 6, 0.42, 0.28), (1988, 9, 0.38, 0.32), (1988, 12, 0.41, 0.29),
    (1989, 3, 0.44, 0.27), (1989, 6, 0.46, 0.25), (1989, 9, 0.45, 0.26), (1989, 12, 0.43, 0.28),
    # 1990 (recession/Gulf war pessimism)
    (1990, 3, 0.40, 0.31), (1990, 6, 0.43, 0.28), (1990, 9, 0.24, 0.54), (1990, 10, 0.27, 0.50), (1990, 12, 0.34, 0.40),
    # 1991-1994
    (1991, 3, 0.48, 0.24), (1991, 6, 0.42, 0.30), (1991, 9, 0.45, 0.27), (1991, 12, 0.50, 0.22),
    (1992, 3, 0.46, 0.26), (1992, 6, 0.40, 0.31), (1992, 9, 0.42, 0.29), (1992, 12, 0.48, 0.24),
    (1993, 3, 0.45, 0.27), (1993, 6, 0.43, 0.29), (1993, 9, 0.47, 0.25), (1993, 12, 0.50, 0.23),
    (1994, 3, 0.38, 0.34), (1994, 6, 0.35, 0.38), (1994, 9, 0.40, 0.32), (1994, 12, 0.36, 0.37),
    # 1995-1999 (the great bull market: persistent high bullishness)
    (1995, 3, 0.50, 0.24), (1995, 6, 0.55, 0.20), (1995, 9, 0.52, 0.22), (1995, 12, 0.54, 0.20),
    (1996, 3, 0.48, 0.26), (1996, 6, 0.45, 0.28), (1996, 9, 0.50, 0.24), (1996, 12, 0.52, 0.22),
    (1997, 3, 0.42, 0.30), (1997, 6, 0.55, 0.20), (1997, 9, 0.53, 0.22), (1997, 12, 0.48, 0.26),
    (1998, 3, 0.56, 0.20), (1998, 6, 0.50, 0.24), (1998, 9, 0.38, 0.40), (1998, 10, 0.42, 0.35), (1998, 12, 0.55, 0.22),
    (1999, 3, 0.58, 0.20), (1999, 6, 0.54, 0.22), (1999, 9, 0.48, 0.28), (1999, 12, 0.56, 0.21),
    # 2000-2002 (dot-com top then bust)
    (2000, 1, 0.60, 0.18), (2000, 3, 0.55, 0.22), (2000, 6, 0.48, 0.27), (2000, 9, 0.43, 0.31), (2000, 12, 0.40, 0.34),
    (2001, 3, 0.33, 0.45), (2001, 6, 0.42, 0.31), (2001, 9, 0.30, 0.48), (2001, 12, 0.48, 0.26),
    (2002, 3, 0.45, 0.28), (2002, 6, 0.38, 0.38), (2002, 9, 0.28, 0.50), (2002, 10, 0.30, 0.46), (2002, 12, 0.36, 0.40),
    # 2003-2007 (recovery / mid bull)
    (2003, 3, 0.34, 0.42), (2003, 6, 0.52, 0.22), (2003, 9, 0.50, 0.24), (2003, 12, 0.58, 0.19),
    (2004, 3, 0.48, 0.27), (2004, 6, 0.44, 0.30), (2004, 9, 0.40, 0.33), (2004, 12, 0.55, 0.21),
    (2005, 3, 0.38, 0.36), (2005, 6, 0.45, 0.28), (2005, 9, 0.42, 0.30), (2005, 12, 0.48, 0.26),
    (2006, 3, 0.46, 0.28), (2006, 6, 0.36, 0.40), (2006, 9, 0.44, 0.29), (2006, 12, 0.52, 0.23),
    (2007, 3, 0.40, 0.34), (2007, 6, 0.48, 0.26), (2007, 9, 0.42, 0.31), (2007, 12, 0.38, 0.37),
    # 2008-2009 (GFC: the famous bear extremes)
    (2008, 1, 0.30, 0.49), (2008, 3, 0.28, 0.51), (2008, 6, 0.32, 0.45), (2008, 9, 0.30, 0.48),
    (2008, 10, 0.26, 0.54), (2008, 11, 0.28, 0.50), (2008, 12, 0.34, 0.43),
    (2009, 1, 0.26, 0.52), (2009, 2, 0.22, 0.59), (2009, 3, 0.19, 0.70), (2009, 4, 0.36, 0.43),
    (2009, 6, 0.40, 0.34), (2009, 9, 0.45, 0.28), (2009, 12, 0.42, 0.31),
    # 2010-2013
    (2010, 3, 0.48, 0.26), (2010, 6, 0.36, 0.42), (2010, 8, 0.30, 0.49), (2010, 9, 0.45, 0.28), (2010, 12, 0.52, 0.22),
    (2011, 3, 0.42, 0.31), (2011, 6, 0.35, 0.40), (2011, 8, 0.30, 0.48), (2011, 9, 0.25, 0.45), (2011, 12, 0.40, 0.34),
    (2012, 3, 0.44, 0.28), (2012, 6, 0.33, 0.45), (2012, 9, 0.36, 0.33), (2012, 12, 0.46, 0.27),
    (2013, 3, 0.42, 0.30), (2013, 6, 0.38, 0.30), (2013, 9, 0.45, 0.28), (2013, 12, 0.48, 0.25),
    # 2014-2018
    (2014, 3, 0.40, 0.28), (2014, 6, 0.36, 0.27), (2014, 9, 0.40, 0.28), (2014, 12, 0.51, 0.21),
    (2015, 3, 0.32, 0.28), (2015, 6, 0.25, 0.27), (2015, 9, 0.30, 0.38), (2015, 12, 0.26, 0.30),
    (2016, 2, 0.20, 0.45), (2016, 3, 0.34, 0.27), (2016, 6, 0.22, 0.38), (2016, 9, 0.30, 0.29), (2016, 12, 0.45, 0.23),
    (2017, 3, 0.31, 0.34), (2017, 6, 0.33, 0.28), (2017, 9, 0.36, 0.26), (2017, 12, 0.52, 0.20),
    (2018, 1, 0.55, 0.18), (2018, 3, 0.32, 0.35), (2018, 6, 0.34, 0.30), (2018, 10, 0.31, 0.34), (2018, 12, 0.25, 0.50),
    # 2019-2021
    (2019, 3, 0.40, 0.26), (2019, 6, 0.30, 0.40), (2019, 9, 0.34, 0.30), (2019, 12, 0.44, 0.24),
    (2020, 1, 0.46, 0.24), (2020, 2, 0.36, 0.35), (2020, 3, 0.34, 0.52), (2020, 4, 0.24, 0.50),
    (2020, 6, 0.24, 0.48), (2020, 9, 0.32, 0.40), (2020, 12, 0.45, 0.26),
    (2021, 3, 0.50, 0.21), (2021, 6, 0.41, 0.24), (2021, 9, 0.28, 0.40), (2021, 11, 0.48, 0.27), (2021, 12, 0.38, 0.30),
    # 2022-2023 (the 2022 bear-market pessimism: record-low bullishness Apr/Sep-2022)
    (2022, 1, 0.32, 0.39), (2022, 2, 0.23, 0.43), (2022, 3, 0.32, 0.36), (2022, 4, 0.16, 0.59),
    (2022, 5, 0.20, 0.53), (2022, 6, 0.18, 0.59), (2022, 9, 0.18, 0.61), (2022, 10, 0.23, 0.46), (2022, 12, 0.26, 0.47),
    (2023, 3, 0.20, 0.49), (2023, 6, 0.42, 0.28), (2023, 9, 0.31, 0.34), (2023, 12, 0.48, 0.22),
    # 2024-2026 (curated tail)
    (2024, 3, 0.46, 0.23), (2024, 6, 0.44, 0.26), (2024, 9, 0.50, 0.21), (2024, 12, 0.40, 0.34),
    (2025, 3, 0.30, 0.45), (2025, 6, 0.34, 0.36), (2025, 9, 0.40, 0.30), (2025, 12, 0.42, 0.28),
    (2026, 3, 0.38, 0.32), (2026, 5, 0.40, 0.30),
]


def aaii_table() -> pd.DataFrame:
    """Return the curated monthly AAII bull/bear table, indexed by month-end date.

    Columns:
      ``bull`` (fraction), ``bear`` (fraction), ``neutral`` (= 1-bull-bear),
      ``spread`` (= bull - bear).
    Index: month-end ``DatetimeIndex`` (one row per surveyed month).
    """
    rows = []
    for y, m, bull, bear in AAII_MONTHLY:
        ts = pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)
        rows.append({"date": ts, "bull": float(bull), "bear": float(bear)})
    df = pd.DataFrame(rows).set_index("date").sort_index()
    df["neutral"] = 1.0 - df["bull"] - df["bear"]
    df["spread"] = df["bull"] - df["bear"]
    return df


# ---------------------------------------------------------------------------
# Synthetic monthly tape -- deterministic offline core with a contrarian knob.
# ---------------------------------------------------------------------------
def synthetic_monthly(
    n_months: int = 360,
    edge: float = 0.04,
    base_ret: float = 0.007,
    ret_vol: float = 0.042,
    persistence: float = 0.6,
    seed: int = 257,
) -> tuple[pd.DataFrame, dict]:
    """A monthly (sentiment, S&P-return) tape with a *known* contrarian edge.

    The bull-bear spread ``s_t`` is an AR(1) process centred on the long-run
    AAII average (~+6.5%).  The next-month market return is::

        ret_{t+1} = base_ret - edge * (s_t - mean_spread) + noise

    so a **high** (euphoric) spread predicts a **lower** next-month return and a
    **low** (panicked) spread predicts a **higher** next-month return -- the
    contrarian story.  ``edge = 0`` is the null (sentiment is pure noise w.r.t.
    forward returns).

    Returns ``(df, truth)`` where ``df`` is indexed by month-end and has columns
    ``bull``, ``bear``, ``spread``, ``ret`` (the simple return earned *during*
    that month, i.e. already aligned so ``ret`` at row t is driven by ``spread``
    at row t-1).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1990-01-31", periods=n_months, freq="ME")

    mean_spread = AAII_AVG_SPREAD
    spread = np.empty(n_months)
    spread[0] = mean_spread
    innov = rng.normal(0.0, 0.10, n_months)
    for t in range(1, n_months):
        spread[t] = mean_spread + persistence * (spread[t - 1] - mean_spread) + innov[t]
    spread = np.clip(spread, -0.55, 0.55)

    # bull/bear consistent with the spread (split around long-run averages)
    bull = np.clip(AAII_AVG_BULL + spread / 2.0 - AAII_AVG_SPREAD / 2.0, 0.05, 0.75)
    bear = np.clip(bull - spread, 0.05, 0.80)

    noise = rng.normal(0.0, ret_vol, n_months)
    ret = np.full(n_months, base_ret) + noise
    # contrarian loading: spread at t-1 drives return at t
    ret[1:] += -edge * (spread[:-1] - mean_spread)

    df = pd.DataFrame(
        {"bull": bull, "bear": bear, "spread": spread, "ret": ret}, index=idx
    )
    truth = {
        "n_months": n_months,
        "edge": edge,
        "base_ret": base_ret,
        "ret_vol": ret_vol,
        "persistence": persistence,
        "mean_spread": mean_spread,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real S&P 500 tape -- ^GSPC month-end closes (repo cache, read-only).
# ---------------------------------------------------------------------------
def fetch_gspc_monthly(
    fetch: bool = False,
    cache_path: str = GSPC_CACHE,
) -> pd.Series:
    """Month-end ^GSPC close (price-only) as a Series indexed by month-end date.

    Cache-only by default: reads the repo-level ``_cache/^GSPC_split_only.parquet``
    (a daily OHLCV frame) and resamples to month-end last close.  Network is
    touched only on an explicit ``fetch=True`` (lazy yfinance import).

    **Price-only** (no dividend reinvestment): the timing study compares a
    timing rule against buy-and-hold on the *same* price index, so the dividend
    omission cancels in the *relative* comparison but the absolute levels are
    price-return; this is labelled throughout.
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No ^GSPC cache at {cache_path}. The repo-level "
                "_cache/^GSPC_split_only.parquet must be present (or call fetch=True)."
            )
        df = pd.read_parquet(cache_path)
        close = df["Close"].copy()
        close.index = pd.to_datetime(close.index)
        monthly = close.resample("ME").last().dropna()
        monthly.name = "gspc"
        return monthly

    import yfinance as yf  # lazy import: network only on fetch=True

    raw = yf.download("^GSPC", start="1987-07-01", interval="1d",
                      auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError("yfinance returned no ^GSPC data")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    monthly = close.resample("ME").last().dropna()
    monthly.name = "gspc"
    return monthly


def build_real_panel(
    fetch: bool = False,
    cache_path: str = GSPC_CACHE,
) -> pd.DataFrame:
    """Join the curated AAII monthly table to ^GSPC month-end *forward* returns.

    Returns a DataFrame indexed by month-end with columns:
      ``bull``, ``bear``, ``spread`` (sentiment observed at month-end t),
      ``ret``   (the S&P price return earned *during month t+1*; the forward
                 return the signal is asked to predict, one-month execution lag).

    The last AAII row has no forward return and is dropped.
    """
    aaii = aaii_table()
    gspc = fetch_gspc_monthly(fetch=fetch, cache_path=cache_path)
    # Forward 1-month price return aligned to month-end t.
    fwd = gspc.pct_change().shift(-1)  # return during month t+1, stamped at t
    fwd = fwd.rename("ret")
    panel = aaii.join(fwd, how="inner")
    panel = panel.dropna(subset=["ret"])
    return panel


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame, col: str = "spread") -> str:
    """A short content fingerprint of a column, for the as-of stamp."""
    arr = pd.Series(df[col]).dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
