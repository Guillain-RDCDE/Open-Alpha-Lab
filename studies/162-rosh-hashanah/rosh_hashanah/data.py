"""Data layer for Study 162 (Rosh-Hashanah).

Two tapes:

- ``synthetic_annual`` — a *deterministic, offline* generator.  Produces annual S&P returns
  with a tunable ``holiday_drag_bp`` applied to the ~8-day window between Rosh Hashanah and
  Yom Kippur.  Setting ``holiday_drag_bp = 0`` gives the null (no seasonal); a negative value
  plants the "sell Rosh Hashanah" weakness.  This is the study's null in a bottle.
- ``fetch_gspc`` — the real S&P 500 daily index (``^GSPC``), cache-first, via yfinance.

The well-known Jewish High Holiday event table (Rosh Hashanah and Yom Kippur start dates,
1980–2026) is **hardcoded** as a module constant, making the core fully offline and
reproducible.  Dates are the **first calendar day** of each holiday (Rosh Hashanah is a
2-day holiday; Yom Kippur is 1 day).  We test the S&P window **from the close on the eve of
Rosh Hashanah to the close on Yom Kippur** — roughly 9 calendar days / 6–7 trading days.

Source for holiday dates: Hebrew Calendar conversion (see e.g. Hebcal.com; cross-checked
against Reingold & Dershowitz, "Calendrical Calculations", 4th ed., Cambridge University
Press 2018).  In the academic literature these dates are used by Hirsch (2020), Friesen &
Weller (2021), and a range of practitioner studies; see docs/references.md.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))

# ---------------------------------------------------------------------------
# High-Holiday event table, 1980–2026.
#
# Each row: (year, rosh_hashanah_date, yom_kippur_date)
# Rosh Hashanah = start of the Jewish New Year (1 Tishrei).
# Yom Kippur    = 10 Tishrei, always 9 days later (calendar days) than RH day-1.
#
# Source: Reingold & Dershowitz (2018) "Calendrical Calculations", and Hebcal.com.
# Cross-checked against Wikipedia's "Yom Kippur" date tables.
# ---------------------------------------------------------------------------
HOLIDAYS: list[tuple[int, str, str]] = [
    # year   Rosh Hashanah       Yom Kippur
    (1980, "1980-09-11", "1980-09-20"),
    (1981, "1981-09-29", "1981-10-08"),
    (1982, "1982-09-18", "1982-09-27"),
    (1983, "1983-09-08", "1983-09-17"),
    (1984, "1984-09-27", "1984-10-06"),
    (1985, "1985-09-16", "1985-09-25"),
    (1986, "1986-10-04", "1986-10-13"),
    (1987, "1987-09-24", "1987-10-03"),
    (1988, "1988-09-12", "1988-09-21"),
    (1989, "1989-09-30", "1989-10-09"),
    (1990, "1990-09-20", "1990-09-29"),
    (1991, "1991-09-09", "1991-09-18"),
    (1992, "1992-09-28", "1992-10-07"),
    (1993, "1993-09-16", "1993-09-25"),
    (1994, "1994-09-06", "1994-09-15"),
    (1995, "1995-09-25", "1995-10-04"),
    (1996, "1996-09-14", "1996-09-23"),
    (1997, "1997-10-02", "1997-10-11"),
    (1998, "1998-09-21", "1998-09-30"),
    (1999, "1999-09-11", "1999-09-20"),
    (2000, "2000-09-30", "2000-10-09"),
    (2001, "2001-09-18", "2001-09-27"),
    (2002, "2002-09-07", "2002-09-16"),
    (2003, "2003-09-27", "2003-10-06"),
    (2004, "2004-09-16", "2004-09-25"),
    (2005, "2005-10-04", "2005-10-13"),
    (2006, "2006-09-23", "2006-10-02"),
    (2007, "2007-09-13", "2007-09-22"),
    (2008, "2008-09-30", "2008-10-09"),
    (2009, "2009-09-19", "2009-09-28"),
    (2010, "2010-09-09", "2010-09-18"),
    (2011, "2011-09-29", "2011-10-08"),
    (2012, "2012-09-17", "2012-09-26"),
    (2013, "2013-09-05", "2013-09-14"),
    (2014, "2014-09-25", "2014-10-04"),
    (2015, "2015-09-14", "2015-09-23"),
    (2016, "2016-10-03", "2016-10-12"),
    (2017, "2017-09-21", "2017-09-30"),
    (2018, "2018-09-10", "2018-09-19"),
    (2019, "2019-09-30", "2019-10-09"),
    (2020, "2020-09-19", "2020-09-28"),
    (2021, "2021-09-07", "2021-09-16"),
    (2022, "2022-09-26", "2022-10-05"),
    (2023, "2023-09-16", "2023-09-25"),
    (2024, "2024-10-03", "2024-10-12"),
    (2025, "2025-09-23", "2025-10-02"),
    (2026, "2026-09-12", "2026-09-21"),
]

# Build convenient DataFrames for downstream use.
HOLIDAY_DF = pd.DataFrame(
    HOLIDAYS, columns=["year", "rosh_hashanah", "yom_kippur"]
).assign(
    rosh_hashanah=lambda d: pd.to_datetime(d["rosh_hashanah"]),
    yom_kippur=lambda d: pd.to_datetime(d["yom_kippur"]),
)


# ---------------------------------------------------------------------------
# Synthetic tape — deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_annual(
    n_years: int = 80,
    base_bp: float = 70.0,
    holiday_drag_bp: float = -50.0,
    vol_ann: float = 0.16,
    seed: int = 162,
) -> tuple[pd.DataFrame, dict]:
    """A yearly return series with an optional holiday-window drag.

    Each observation represents **one year**: annual return drawn i.i.d. normal with
    mean ``base_bp`` (bps/year) and standard deviation ``vol_ann``.  An extra
    ``holiday_drag_bp`` is planted in the "holiday window" — treated here as a fraction
    of the annual return assigned to the ~8-day window, scaled by 8/252.

    - ``holiday_drag_bp = 0``    → the null; window returns = unconditional slice.
    - ``holiday_drag_bp < 0``    → the "Sell Rosh Hashanah" weakness is planted.
    - ``holiday_drag_bp > 0``    → a holiday rally.

    Returns a DataFrame with columns ``window_ret``, ``rest_ret``, and ``annual_ret``
    (roughly: ``annual_ret ≈ window_ret + rest_ret``), plus a ``truth`` dict.
    """
    rng = np.random.default_rng(seed)
    n_window_days = 8
    n_rest_days = 252 - n_window_days

    # Base daily drift
    daily_base = base_bp * 1e-4 / 252.0
    daily_vol = vol_ann / np.sqrt(252.0)

    # Window mean (daily basis) with or without drag
    window_daily_mean = daily_base + holiday_drag_bp * 1e-4 / n_window_days
    rest_daily_mean = daily_base

    window_ret = (
        window_daily_mean * n_window_days
        + rng.standard_normal(n_years) * daily_vol * np.sqrt(n_window_days)
    )
    rest_ret = (
        rest_daily_mean * n_rest_days
        + rng.standard_normal(n_years) * daily_vol * np.sqrt(n_rest_days)
    )
    annual_ret = window_ret + rest_ret

    idx = pd.RangeIndex(1946, 1946 + n_years, name="year")
    df = pd.DataFrame(
        {"window_ret": window_ret, "rest_ret": rest_ret, "annual_ret": annual_ret},
        index=idx,
    )
    truth = {
        "n_years": n_years,
        "base_bp": base_bp,
        "holiday_drag_bp": holiday_drag_bp,
        "vol_ann": vol_ann,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — daily S&P 500, cache-first
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "rosh_hashanah_gspc_daily.parquet")


def fetch_gspc(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start_year: int = 1980,
) -> pd.DataFrame:
    """Daily price and return for the S&P 500 (^GSPC), cache-first.

    Returns a DataFrame with columns ``close`` and ``ret`` (simple daily return),
    indexed by date, from ``start_year`` onward.  Network is only touched with
    ``fetch=True``; otherwise raises ``FileNotFoundError`` if no cache exists.
    """
    path = _cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached S&P 500 data at {path}. "
                "Call fetch_gspc(fetch=True) once to populate the cache."
            )
        df = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only on explicit fetch

        raw = yf.download(
            "^GSPC", period="max", interval="1d", auto_adjust=True, progress=False
        )
        if raw.empty:
            raise RuntimeError("yfinance returned no data for ^GSPC")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        close = raw["Close"].dropna().squeeze()
        close.index = pd.DatetimeIndex(close.index).tz_localize(None)
        close.index.name = "date"
        ret = close.pct_change().dropna()
        df = pd.DataFrame({"close": close, "ret": ret})
        df.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(path)

    df = df[df.index.year >= start_year].copy()
    return df


# ---------------------------------------------------------------------------
# Holiday-window extraction
# ---------------------------------------------------------------------------
def holiday_windows(
    prices: pd.DataFrame,
    lookback_days: int = 1,
) -> pd.DataFrame:
    """Extract the S&P return in each Rosh Hashanah → Yom Kippur window.

    For each year in ``HOLIDAY_DF`` that overlaps with ``prices``, we measure the
    close-to-close return from the **trading day immediately before Rosh Hashanah**
    (the "eve" close) to the **close on or just after Yom Kippur**.

    - ``entry_date``  — last trading day strictly before Rosh Hashanah.
    - ``exit_date``   — first trading day on or after Yom Kippur.
    - ``window_ret``  — total S&P return over [entry_close, exit_close].
    - ``n_trading``   — number of trading days in the window.
    - ``rosh_hashanah``, ``yom_kippur`` — the calendar holiday dates.

    ``lookback_days`` is reserved for sensitivity testing (default 1 = one day before).
    Returns only years where both an entry and an exit could be located.
    """
    close = prices["close"].sort_index()
    trading_days = close.index

    rows = []
    for _, row in HOLIDAY_DF.iterrows():
        rh = row["rosh_hashanah"]
        yk = row["yom_kippur"]
        year = int(row["year"])

        # entry = last trading day strictly before Rosh Hashanah
        before_rh = trading_days[trading_days < rh]
        if len(before_rh) < lookback_days:
            continue
        entry_date = before_rh[-lookback_days]

        # exit = first trading day on or after Yom Kippur
        on_or_after_yk = trading_days[trading_days >= yk]
        if len(on_or_after_yk) == 0:
            continue
        exit_date = on_or_after_yk[0]

        entry_close = close.loc[entry_date]
        exit_close = close.loc[exit_date]
        window_ret = exit_close / entry_close - 1.0
        n_trading = int(((trading_days >= entry_date) & (trading_days <= exit_date)).sum()) - 1

        rows.append(
            {
                "year": year,
                "rosh_hashanah": rh,
                "yom_kippur": yk,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_close": float(entry_close),
                "exit_close": float(exit_close),
                "window_ret": float(window_ret),
                "n_trading": int(n_trading),
            }
        )

    return pd.DataFrame(rows).set_index("year")


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of the close column, for the as-of stamp."""
    col = "close" if "close" in df.columns else df.columns[0]
    h = hashlib.sha1(np.ascontiguousarray(df[col].to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]
