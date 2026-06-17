"""Data layer for Study 288 (Ramadan-Effect).

Three components, the core fully offline and deterministic:

- ``RAMADAN_WINDOWS`` -- a hardcoded table of the Gregorian start/end date of
  Ramadan for each Hijri year from 2000 through 2026. Ramadan is a *lunar*
  month (~29-30 days) that drifts ~11 days earlier each Gregorian year, so its
  start date is sighting-dependent and well documented. These dates are small,
  well-known, and hardcoded here to keep the core offline and reproducible.
  Source: Umm al-Qura calendar / standard Islamic-calendar conversion tables.

- ``synthetic_monthly`` -- a deterministic, offline monthly-return generator
  with an optional planted "Ramadan premium" knob. ``premium_bps = 0`` is the
  null (no effect), so tests confirm the machinery is truthful before any real
  data is touched. Each month is tagged ``ramadan`` if it overlaps a Ramadan
  window by a chosen fraction.

- ``fetch_proxy`` -- the real Yahoo! monthly close panel for a MENA-proxy ETF
  (``KSA`` -- iShares MSCI Saudi Arabia, the most liquid US-listed single-MENA
  ETF, history from 2015) and the S&P 500 (``^GSPC``), cache-only by default so
  the test-suite and the reproducible core never touch the network. Cache lives
  under ``_cache/ramadan_proxy.parquet`` (a date x series frame of monthly
  closes, rows = month-end dates).

The "Ramadan effect" (Bialkowski, Etebari & Wisniewski 2012) claims that equity
returns in Muslim-majority countries are higher and less volatile during Ramadan
-- attributed to optimism, social cohesion, and lower trading activity. We test
it on a MENA proxy and, as a placebo, on the S&P 500 (a non-Muslim-majority tape
where no Ramadan effect should exist).

**Survivorship/representativeness caveat**: ``KSA`` is a single US-listed ETF
tracking the MSCI Saudi Arabia index, with history only from 2015. It is a
coarse proxy for "MENA equities" and starts well after the literature's sample;
positive results are an upper bound on what a clean, broad, local-currency MENA
index over a longer window would show, and the ~11 Ramadan months available are
a punishingly small sample. Named on the Signal axis.

No look-ahead: a month is labelled ``ramadan`` purely from the calendar (the
Ramadan dates are known years in advance), and that label is matched to the
*same* month's realised return. There is no parameter fit on the return series.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
PROXY_CACHE = os.path.join(DEFAULT_CACHE, "ramadan_proxy.parquet")

# The MENA proxy ETF and the placebo index.
PROXY_TICKER = "KSA"    # iShares MSCI Saudi Arabia ETF (US-listed, from 2015)
PLACEBO_TICKER = "^GSPC"  # S&P 500 -- non-Muslim-majority placebo tape

# ---------------------------------------------------------------------------
# Hardcoded Ramadan windows -- the study's deterministic, offline core
# ---------------------------------------------------------------------------
# Columns:
#   hijri_year : the Hijri (Islamic) year whose Ramadan this is
#   start      : Gregorian first day of Ramadan (ISO 'YYYY-MM-DD')
#   end        : Gregorian last day of Ramadan (the day before Eid al-Fitr)
#
# Ramadan is the 9th month of the Hijri lunar calendar. Its Gregorian start
# drifts ~11 days earlier each year. Dates below follow the Umm al-Qura
# (Saudi) civil calendar, the standard reference for the Gulf markets that
# dominate the MENA proxy. Exact sighting-based observance can differ by a day,
# which is immaterial at monthly resolution.
#
# Source: Umm al-Qura calendar conversion; Time and Date AS Islamic calendar
# tables; cross-checked against published Ramadan start dates.
# As-of 2026-06-17.

RAMADAN_WINDOWS: list[dict] = [
    {"hijri_year": 1420, "start": "1999-12-09", "end": "2000-01-07"},
    {"hijri_year": 1421, "start": "2000-11-27", "end": "2000-12-26"},
    {"hijri_year": 1422, "start": "2001-11-16", "end": "2001-12-15"},
    {"hijri_year": 1423, "start": "2002-11-06", "end": "2002-12-04"},
    {"hijri_year": 1424, "start": "2003-10-27", "end": "2003-11-24"},
    {"hijri_year": 1425, "start": "2004-10-15", "end": "2004-11-13"},
    {"hijri_year": 1426, "start": "2005-10-04", "end": "2005-11-02"},
    {"hijri_year": 1427, "start": "2006-09-24", "end": "2006-10-22"},
    {"hijri_year": 1428, "start": "2007-09-13", "end": "2007-10-11"},
    {"hijri_year": 1429, "start": "2008-09-01", "end": "2008-09-29"},
    {"hijri_year": 1430, "start": "2009-08-22", "end": "2009-09-19"},
    {"hijri_year": 1431, "start": "2010-08-11", "end": "2010-09-09"},
    {"hijri_year": 1432, "start": "2011-08-01", "end": "2011-08-29"},
    {"hijri_year": 1433, "start": "2012-07-20", "end": "2012-08-18"},
    {"hijri_year": 1434, "start": "2013-07-09", "end": "2013-08-07"},
    {"hijri_year": 1435, "start": "2014-06-28", "end": "2014-07-27"},
    {"hijri_year": 1436, "start": "2015-06-18", "end": "2015-07-16"},
    {"hijri_year": 1437, "start": "2016-06-06", "end": "2016-07-05"},
    {"hijri_year": 1438, "start": "2017-05-27", "end": "2017-06-24"},
    {"hijri_year": 1439, "start": "2018-05-16", "end": "2018-06-14"},
    {"hijri_year": 1440, "start": "2019-05-06", "end": "2019-06-03"},
    {"hijri_year": 1441, "start": "2020-04-24", "end": "2020-05-23"},
    {"hijri_year": 1442, "start": "2021-04-13", "end": "2021-05-12"},
    {"hijri_year": 1443, "start": "2022-04-02", "end": "2022-05-01"},
    {"hijri_year": 1444, "start": "2023-03-23", "end": "2023-04-20"},
    {"hijri_year": 1445, "start": "2024-03-11", "end": "2024-04-09"},
    {"hijri_year": 1446, "start": "2025-03-01", "end": "2025-03-30"},
    {"hijri_year": 1447, "start": "2026-02-18", "end": "2026-03-19"},
]

RAMADAN_DF: pd.DataFrame = pd.DataFrame(RAMADAN_WINDOWS)


def ramadan_windows() -> pd.DataFrame:
    """Return a clean copy of the hardcoded Ramadan window table.

    Columns: ``hijri_year`` (int), ``start`` (Timestamp), ``end`` (Timestamp).
    """
    df = RAMADAN_DF.copy()
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    return df


def ramadan_overlap_fraction(month_end: pd.Timestamp) -> float:
    """Fraction of the calendar month ending at ``month_end`` that lies in Ramadan.

    A month is the period (month_start, month_end]. We compute the number of
    calendar days in that month that fall inside any Ramadan window, divided by
    the number of days in the month. Returns a float in [0, 1].
    """
    month_end = pd.Timestamp(month_end)
    month_start = month_end.replace(day=1)
    days_in_month = (month_end - month_start).days + 1
    overlap_days = 0
    for _, row in RAMADAN_DF.iterrows():
        r_start = pd.Timestamp(row["start"])
        r_end = pd.Timestamp(row["end"])
        lo = max(month_start, r_start)
        hi = min(month_end, r_end)
        if hi >= lo:
            overlap_days += (hi - lo).days + 1
    return min(1.0, overlap_days / days_in_month)


def label_months(month_ends, threshold: float = 0.5) -> pd.DataFrame:
    """Tag each month-end date as a Ramadan month or not.

    A month is ``ramadan = True`` if at least ``threshold`` (default 0.5) of its
    calendar days fall inside a Ramadan window. Returns a DataFrame indexed by
    the month-end dates with columns ``overlap`` (float) and ``ramadan`` (bool).
    """
    idx = pd.DatetimeIndex(month_ends)
    overlaps = np.array([ramadan_overlap_fraction(d) for d in idx])
    return pd.DataFrame(
        {"overlap": overlaps, "ramadan": overlaps >= threshold},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Synthetic monthly-return generator -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_monthly(
    n_months: int = 200,
    premium_bps: float = 0.0,
    base_ret: float = 0.006,
    vol_mo: float = 0.05,
    start: str = "2008-12-31",
    threshold: float = 0.5,
    seed: int = 288,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible monthly-return series with an optional planted Ramadan premium.

    Each month is drawn i.i.d. from N(base_ret, vol_mo^2). Months flagged as
    Ramadan (>= ``threshold`` overlap with a real hardcoded Ramadan window) get
    an extra drift of ``premium_bps`` bps. ``premium_bps = 0`` is the null -- the
    Ramadan label is uncorrelated with the return. The Ramadan flags come from
    the *real* hardcoded calendar, so the synthetic tape has the correct,
    irregular Ramadan-month spacing.

    Returns ``(df, truth)`` where ``df`` is indexed by month-end dates with
    columns ``ret`` (simple monthly return), ``overlap`` (float), ``ramadan``
    (bool); ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range(start=start, periods=n_months, freq="ME")
    lab = label_months(months, threshold=threshold)
    ramadan = lab["ramadan"].to_numpy()
    drift = np.where(ramadan, base_ret + premium_bps * 1e-4, base_ret)
    rets = drift + rng.normal(0.0, vol_mo, n_months)

    df = pd.DataFrame(
        {"ret": rets, "overlap": lab["overlap"].to_numpy(), "ramadan": ramadan},
        index=months,
    )
    truth = {
        "n_months": n_months,
        "premium_bps": premium_bps,
        "base_ret": base_ret,
        "vol_mo": vol_mo,
        "threshold": threshold,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape -- Yahoo monthly closes, cache-only by default
# ---------------------------------------------------------------------------
def fetch_proxy(
    fetch: bool = False,
    cache_path: str = PROXY_CACHE,
) -> pd.DataFrame:
    """Monthly close panel for the MENA proxy + S&P 500; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (lazy yfinance import),
    after which the result is cached as a parquet. Columns are ``MENA`` (the
    MENA proxy ETF, KSA) and ``SPX`` (the S&P 500 placebo); index is month-end
    dates.

    Raises ``FileNotFoundError`` if the cache is absent and ``fetch=False``.

    **Survivorship/representativeness**: ``KSA`` is a single US-listed ETF, a
    coarse stand-in for MENA equities, with history from 2015. Positive results
    are upper bounds.
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached proxy panel at {cache_path}. "
                "Call fetch_proxy(fetch=True) once to populate the cache."
            )
        return pd.read_parquet(cache_path)

    import yfinance as yf  # lazy import: network only on fetch=True

    raw = yf.download(
        [PROXY_TICKER, PLACEBO_TICKER],
        start="2008-01-01",
        interval="1mo",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no monthly data")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            closes = raw["Close"]
        else:
            closes = raw.xs("Close", axis=1, level=0)
    else:
        closes = raw

    closes = closes.rename(columns={PROXY_TICKER: "MENA", PLACEBO_TICKER: "SPX"})
    # Keep only the window where the proxy itself has data (drops pre-2015 NaN).
    if "MENA" in closes.columns:
        closes = closes.loc[closes["MENA"].first_valid_index():]
    closes.index = pd.to_datetime(closes.index) + pd.offsets.MonthEnd(0)
    closes = closes.sort_index().dropna(how="all")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    closes.to_parquet(cache_path)
    print(f"Cached proxy panel: {closes.shape[0]} months x {closes.shape[1]} series -> {cache_path}")
    return closes


def proxy_returns(
    prices: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Convert a cached price panel into a monthly-return frame with Ramadan labels.

    Input ``prices`` has columns ``MENA`` and ``SPX`` (month-end closes). Output
    is indexed by month-end dates with columns ``mena_ret``, ``spx_ret``,
    ``overlap`` (float), ``ramadan`` (bool).
    """
    rets = prices.pct_change().dropna(how="all")
    lab = label_months(rets.index, threshold=threshold)
    out = pd.DataFrame(index=rets.index)
    if "MENA" in rets.columns:
        out["mena_ret"] = rets["MENA"]
    if "SPX" in rets.columns:
        out["spx_ret"] = rets["SPX"]
    out["overlap"] = lab["overlap"].to_numpy()
    out["ramadan"] = lab["ramadan"].to_numpy()
    return out.dropna(how="all")


def fingerprint(df: pd.DataFrame, col: str = "mena_ret") -> str:
    """A short content fingerprint of a return column, for the as-of stamp."""
    vals = df[col].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]
