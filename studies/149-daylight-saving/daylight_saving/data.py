"""Data layer for Study 149 (Daylight-Saving).

Two tapes, one logical shape (a daily return Series with a DST-Monday flag):

- ``synthetic_daily`` — a *deterministic, offline* generator.  A ``dst_effect``
  knob injects an extra drift on post-DST Mondays; set ``dst_effect=0`` for the
  null (pure random walk) so the strategy test scores exactly zero.  This is the
  study's null in a bottle.
- ``fetch_daily`` — loads ^GSPC daily closes from Yahoo Finance (cache-first).
  Default ``fetch=False`` raises if no cache file exists; ``fetch=True`` hits
  Yahoo and saves a parquet under ``_cache/``.

US DST change-date rules (all Sunday → effect tested on the following Monday):
  Pre-2007:  spring = last Sunday of April; fall = last Sunday of October.
  2007+  :   spring = 2nd Sunday of March;  fall = 1st Sunday of November.
The helper :func:`dst_mondays` returns the Monday after each DST change for a
given year range, in two labelled buckets (``'spring'`` and ``'fall'``).

No look-ahead: the DST Monday flag is determined purely by the calendar date of
the observation — there is no forward information in the label.

Reference: Kamstra, Kramer & Levi (2000), "Losing Sleep at the Market: The
Daylight-Saving Anomaly", American Economic Review 90(4), 1005-1011.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(_HERE, "..", "_cache")

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# DST date computation
# ---------------------------------------------------------------------------
def dst_mondays(
    start_year: int = 1980,
    end_year: int = 2026,
) -> pd.DataFrame:
    """Return a DataFrame of post-DST Mondays for every year in ``[start_year, end_year]``.

    Columns: ``date`` (the Monday), ``season`` (``'spring'`` or ``'fall'``),
    ``year``, ``pre_2007`` (bool).

    The US DST schedule:
    - 1987-2006: spring on the 1st Sunday of April; fall on the last Sunday of October.
    - Before 1987: spring on the last Sunday of April; fall on the last Sunday of October.
    - 2007+:       spring on the 2nd Sunday of March;  fall on the 1st Sunday of November.

    We return the *Monday after the Sunday change*, which is the first trading
    day on which a sleep-disrupted investor would trade (per Kamstra et al.).
    """
    rows = []
    for yr in range(start_year, end_year + 1):
        if yr >= 2007:
            # Spring: 2nd Sunday of March
            spring_sun = _nth_weekday_of_month(yr, 3, 6, 2)   # month=3, weekday=6 (Sun), n=2
            # Fall: 1st Sunday of November
            fall_sun   = _nth_weekday_of_month(yr, 11, 6, 1)
        elif yr >= 1987:
            # Spring: 1st Sunday of April
            spring_sun = _nth_weekday_of_month(yr, 4, 6, 1)
            # Fall: last Sunday of October
            fall_sun   = _last_weekday_of_month(yr, 10, 6)
        else:
            # Spring: last Sunday of April
            spring_sun = _last_weekday_of_month(yr, 4, 6)
            # Fall: last Sunday of October
            fall_sun   = _last_weekday_of_month(yr, 10, 6)

        for sun, season in [(spring_sun, "spring"), (fall_sun, "fall")]:
            mon = sun + pd.Timedelta(days=1)
            rows.append({
                "date":     mon,
                "season":   season,
                "year":     yr,
                "pre_2007": yr < 2007,
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """Return the n-th occurrence (1-indexed) of ``weekday`` (0=Mon, 6=Sun) in ``year/month``."""
    first = pd.Timestamp(year, month, 1)
    offset = (weekday - first.dayofweek) % 7
    first_match = first + pd.Timedelta(days=offset)
    return first_match + pd.Timedelta(weeks=n - 1)


def _last_weekday_of_month(year: int, month: int, weekday: int) -> pd.Timestamp:
    """Return the last occurrence of ``weekday`` (0=Mon, 6=Sun) in ``year/month``."""
    # Start from the last day of the month and step back.
    if month == 12:
        last = pd.Timestamp(year + 1, 1, 1) - pd.Timedelta(days=1)
    else:
        last = pd.Timestamp(year, month + 1, 1) - pd.Timedelta(days=1)
    offset = (last.dayofweek - weekday) % 7
    return last - pd.Timedelta(days=offset)


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    start_year: int = 1980,
    end_year: int = 2025,
    dst_effect: float = 0.0,
    base_ret: float = 4e-4,
    daily_vol: float = 0.011,
    seed: int = 149,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily-return series with a tunable post-DST Monday effect.

    Log returns are i.i.d. normal with mean ``base_ret`` and std ``daily_vol``
    on ordinary days; on post-DST Mondays the mean is shifted by ``dst_effect``.
    Setting ``dst_effect=0`` gives a pure random walk — the null — so the
    post-DST vs other-Monday test scores exactly zero in expectation.

    Returns ``(frame, truth)`` where ``frame`` has columns ``ret`` and
    ``is_dst_monday`` (bool), indexed by trading date.
    """
    rng = np.random.default_rng(seed)
    start = f"{start_year}-01-01"
    end   = f"{end_year}-12-31"
    idx   = pd.bdate_range(start=start, end=end, name="date")

    dst_mon = set(
        dst_mondays(start_year, end_year)["date"].dt.normalize().values
    )
    is_dst = pd.array(
        [pd.Timestamp(d).normalize() in dst_mon for d in idx],
        dtype=bool,
    )

    rets = base_ret + daily_vol * rng.standard_normal(len(idx))
    rets[is_dst] += dst_effect

    frame = pd.DataFrame(
        {"ret": rets.astype(float), "is_dst_monday": is_dst},
        index=idx,
    )
    n_dst = int(is_dst.sum())
    truth = {
        "dst_effect": dst_effect,
        "base_ret": base_ret,
        "daily_vol": daily_vol,
        "n_days": len(idx),
        "n_dst": n_dst,
        "n_other_monday": int((idx.dayofweek == 0).sum()) - n_dst,
        "seed": seed,
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — ^GSPC daily closes from Yahoo Finance
# ---------------------------------------------------------------------------
def _gspc_cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "gspc_daily.parquet")


def fetch_daily(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "1980-01-01",
) -> pd.DataFrame:
    """^GSPC daily close-to-close returns with DST-Monday labels; cache-first.

    Returns a DataFrame with columns ``ret`` (daily pct-change), ``is_dst_monday``
    (bool), ``is_monday`` (bool), ``season`` (``'spring'``, ``'fall'``, or ``''``),
    indexed by trading date. Network is touched only with ``fetch=True``.
    """
    cache = _gspc_cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(cache):
            raise FileNotFoundError(
                f"No cached ^GSPC data at {cache}. "
                "Call fetch_daily(fetch=True) once to populate, or run "
                "examples/verify.py --fetch."
            )
        raw = pd.read_parquet(cache)
        ret = raw["ret"]
    else:
        import yfinance as yf

        px = yf.download("^GSPC", start=start, auto_adjust=True, progress=False)
        if px.empty:
            raise RuntimeError("yfinance returned no data for ^GSPC")
        if isinstance(px.columns, pd.MultiIndex):
            px.columns = px.columns.get_level_values(0)
        close_col = next(c for c in px.columns if str(c).lower() in ("close", "adj close"))
        ret = px[close_col].pct_change().dropna()
        ret.name = "ret"
        os.makedirs(cache_dir, exist_ok=True)
        ret.to_frame().to_parquet(cache)

    ret.index = pd.DatetimeIndex(ret.index).tz_localize(None)
    ret = ret[ret.index >= start].dropna()
    ret.index.name = "date"

    # Build DST-Monday flags
    start_yr = ret.index.year.min()
    end_yr   = ret.index.year.max()
    dst_tbl  = dst_mondays(int(start_yr), int(end_yr))
    dst_map  = {
        row["date"].normalize(): row["season"]
        for _, row in dst_tbl.iterrows()
    }

    idx_norm = pd.DatetimeIndex(ret.index).normalize()
    is_dst   = pd.array([d in dst_map for d in idx_norm], dtype=bool)
    season   = [dst_map.get(d, "") for d in idx_norm]

    frame = pd.DataFrame(
        {
            "ret":           ret.values,
            "is_dst_monday": is_dst,
            "is_monday":     ret.index.dayofweek == 0,
            "season":        season,
        },
        index=ret.index,
    )
    return frame


def fingerprint(frame: pd.DataFrame) -> str:
    """A short content fingerprint of the return column, for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(frame["ret"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
