"""Data layer for Study 297 (March-Madness).

The folklore: the NCAA Division I men's basketball tournament ("March Madness")
is a three-week national distraction — billions of dollars in (mostly illegal)
bracket pools, employees streaming day-session games at their desks, productivity
surveys breathlessly quoted every March. The natural Wall-Street-folklore corollary
is that this distraction must show up in the market: lower returns, thinner volume,
maybe higher volatility while half the country watches basketball instead of
trading.

Two tapes, one shape (a daily close-to-close return Series indexed by date):

- ``synthetic_daily`` — a *deterministic, offline* generator. A ``madness_penalty_bp``
  knob adds a downward drift to in-tournament days; ``madness_penalty_bp=0`` is a
  pure random walk with no tournament seasonality. This is the study's null in a
  bottle, so the tests can confirm the machinery is truthful before looking at
  real data.

- ``fetch_daily`` — the real S&P 500 (^GSPC) daily tape, **cache-first by default**.
  It reads the repo-level ``_cache/^GSPC_split_only.parquet`` that ships with the
  repo (split-adjusted OHLC; close-to-close *price* return, no dividends — labelled
  as price-only on the Signal axis). Network is touched only on ``fetch=True`` via a
  lazy ``yfinance`` import, so the reproducible core and the test suite never hit the
  network.

The NCAA tournament windows are HARDCODED here from the well-documented bracket
schedule (NCAA.com / Sports Reference / Wikipedia "NCAA Division I men's basketball
tournament" by year). The modern 64+ team bracket began in 1985; we cover 1985–2025.
Each window runs from the **first-round Thursday** (the famous opening day, the
first full slate of games) through **championship Monday** (the title game), both
inclusive. This is the ~3-week stretch the "distraction" claim is about.

No look-ahead is baked in: each window is assigned to the *actual* calendar days it
spans, and the strategy that trades on the window applies an explicit one-day
execution lag (see ``strategy.py``).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
STUDY_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))

# Repo-level cached ^GSPC daily OHLC (split-adjusted), ships with the repo.
_REPO_GSPC_FILE = "^GSPC_split_only.parquet"
# Study-local cache (written on fetch=True).
_CACHE_FILE = "march_madness_gspc_daily.parquet"


# ---------------------------------------------------------------------------
# Hardcoded NCAA men's tournament windows 1985-2025 (64+ team era)
# Source: NCAA.com bracket archives; Sports-Reference / Wikipedia
#   "<year> NCAA Division I men's basketball tournament".
# Format: (first_round_thursday, championship_monday), both inclusive.
#   - first-round Thursday = the opening full day of games
#   - championship Monday  = the national title game
# The tournament begins Selection Sunday + a "First Four" (since 2011) on the
# Tue/Wed, but the canonical "madness" window is the Thu first round through
# the Monday final, which is what employees stream and what the distraction
# surveys cover.
# ---------------------------------------------------------------------------
TOURNAMENT_WINDOWS: list[tuple[str, str]] = [
    ("1985-03-14", "1985-04-01"),
    ("1986-03-13", "1986-03-31"),
    ("1987-03-12", "1987-03-30"),
    ("1988-03-17", "1988-04-04"),
    ("1989-03-16", "1989-04-03"),
    ("1990-03-15", "1990-04-02"),
    ("1991-03-14", "1991-04-01"),
    ("1992-03-19", "1992-04-06"),
    ("1993-03-18", "1993-04-05"),
    ("1994-03-17", "1994-04-04"),
    ("1995-03-16", "1995-04-03"),
    ("1996-03-14", "1996-04-01"),
    ("1997-03-13", "1997-03-31"),
    ("1998-03-12", "1998-03-30"),
    ("1999-03-11", "1999-03-29"),
    ("2000-03-16", "2000-04-03"),
    ("2001-03-15", "2001-04-02"),
    ("2002-03-14", "2002-04-01"),
    ("2003-03-20", "2003-04-07"),
    ("2004-03-18", "2004-04-05"),
    ("2005-03-17", "2005-04-04"),
    ("2006-03-16", "2006-04-03"),
    ("2007-03-15", "2007-04-02"),
    ("2008-03-20", "2008-04-07"),
    ("2009-03-19", "2009-04-06"),
    ("2010-03-18", "2010-04-05"),
    ("2011-03-17", "2011-04-04"),
    ("2012-03-15", "2012-04-02"),
    ("2013-03-21", "2013-04-08"),
    ("2014-03-20", "2014-04-07"),
    ("2015-03-19", "2015-04-06"),
    ("2016-03-17", "2016-04-04"),
    ("2017-03-16", "2017-04-03"),
    ("2018-03-15", "2018-04-02"),
    ("2019-03-21", "2019-04-08"),
    # 2020: tournament CANCELLED (COVID-19). No window — a documented gap.
    ("2021-03-19", "2021-04-05"),
    ("2022-03-17", "2022-04-04"),
    ("2023-03-16", "2023-04-03"),
    ("2024-03-21", "2024-04-08"),
    ("2025-03-20", "2025-04-07"),
]


def tournament_table() -> pd.DataFrame:
    """Return a clean DataFrame of the hardcoded tournament windows.

    Columns: ``year`` (int), ``start`` (Timestamp, first-round Thursday),
    ``end`` (Timestamp, championship Monday), ``n_calendar_days`` (int).
    The 2020 tournament was cancelled (COVID-19) and is intentionally absent.
    """
    rows = []
    for s, e in TOURNAMENT_WINDOWS:
        ts, te = pd.Timestamp(s), pd.Timestamp(e)
        rows.append(
            {
                "year": ts.year,
                "start": ts,
                "end": te,
                "n_calendar_days": (te - ts).days + 1,
            }
        )
    return pd.DataFrame(rows)


def tournament_dates(
    start: str = "1985-01-01",
    end: str = "2026-12-31",
) -> pd.DatetimeIndex:
    """All calendar days that fall inside any NCAA tournament window.

    Both ends of each window are inclusive. Purely deterministic and offline —
    derived from the hardcoded ``TOURNAMENT_WINDOWS`` table.
    """
    days: list[pd.Timestamp] = []
    window_start = pd.Timestamp(start)
    window_end = pd.Timestamp(end)
    for s, e in TOURNAMENT_WINDOWS:
        ts, te = pd.Timestamp(s), pd.Timestamp(e)
        ts = max(ts, window_start)
        te = min(te, window_end)
        if ts > te:
            continue
        days.extend(pd.date_range(ts, te, freq="D"))
    return pd.DatetimeIndex(sorted(set(days)))


def is_tournament(dates: pd.DatetimeIndex) -> pd.Series:
    """Boolean mask: True if the date falls inside any NCAA tournament window.

    Parameters
    ----------
    dates :
        DatetimeIndex to label (calendar dates).

    Returns a boolean Series aligned to ``dates`` named ``in_tournament``.
    """
    if len(dates) == 0:
        return pd.Series([], index=dates, name="in_tournament", dtype=bool)
    tset = set(
        tournament_dates(
            start=str(dates.min().date()),
            end=str(dates.max().date()),
        ).normalize()
    )
    mask = pd.Series(
        [d.normalize() in tset for d in dates],
        index=dates,
        name="in_tournament",
    )
    return mask


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 41,
    annual_drift_bp: float = 700.0,
    vol_ann: float = 0.17,
    madness_penalty_bp: float = 0.0,
    seed: int = 297,
) -> tuple[pd.Series, pd.Series, dict]:
    """A reproducible daily return series with a planted tournament drag (or none).

    Daily simple returns are i.i.d. normal with an annualised drift of
    ``annual_drift_bp`` basis points and annual vol ``vol_ann``, except days inside
    an NCAA tournament window, which receive an additional ``madness_penalty_bp``
    per-day bias (negative = the "distraction" drag the folklore predicts). The
    only forecastable structure is the tournament-calendar label: when
    ``madness_penalty_bp = 0`` (the null) all days are identical draws.

    Returns ``(returns, tournament_mask, truth)``.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("1985-01-02")
    all_days = pd.bdate_range(start=start, periods=int(n_years * 252))
    n = len(all_days)
    daily_drift = annual_drift_bp * 1e-4 / 252
    daily_vol = vol_ann / np.sqrt(252)
    r = rng.normal(daily_drift, daily_vol, n)

    mask = is_tournament(pd.DatetimeIndex(all_days))
    r[mask.values] += madness_penalty_bp * 1e-4

    ret = pd.Series(r, index=all_days, name="ret")
    truth = {
        "n_years": n_years,
        "annual_drift_bp": annual_drift_bp,
        "vol_ann": vol_ann,
        "madness_penalty_bp": madness_penalty_bp,
        "seed": seed,
        "n_obs": int(n),
        "n_tournament_days": int(mask.sum()),
        "frac_tournament": float(mask.mean()),
    }
    return ret, mask, truth


# ---------------------------------------------------------------------------
# Real tape — ^GSPC daily, cache-first
# ---------------------------------------------------------------------------
def _read_gspc_close(path: str) -> pd.Series:
    """Read a split-only ^GSPC parquet and return the close-to-close return series."""
    df = pd.read_parquet(path)
    if "Close" in df.columns:
        px = df["Close"].copy()
    elif "ret" in df.columns:
        # Already a return series (study-local cache)
        ret = df["ret"].copy()
        ret.index = pd.DatetimeIndex(ret.index)
        if ret.index.tz is not None:
            ret.index = ret.index.tz_localize(None)
        return ret.sort_index()
    else:
        px = df.iloc[:, 0].copy()
    px.index = pd.DatetimeIndex(px.index)
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)
    px = px.sort_index().dropna()
    ret = px.pct_change().dropna()
    ret.name = "ret"
    ret.index.name = "date"
    return ret


def fetch_daily(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "1985-01-01",
    end: str = "2025-12-31",
) -> tuple[pd.Series, pd.Series]:
    """Daily close-to-close *price* return for ^GSPC; cache-first unless ``fetch=True``.

    Returns ``(returns, tournament_mask)``:

    - ``returns``         — a simple price-return Series (split-adjusted, no dividends).
    - ``tournament_mask`` — a boolean Series, True on NCAA-tournament trading days.

    Order of preference:
      1. study-local cache ``_cache/march_madness_gspc_daily.parquet`` (if present),
      2. repo-level cache ``_cache/^GSPC_split_only.parquet`` (ships with the repo),
      3. only if ``fetch=True``: a lazy ``yfinance`` download, then write the cache.

    Network is touched only when ``fetch=True``.
    """
    # 1) study-local cache (a return series written by a prior fetch)
    local = os.path.join(STUDY_CACHE, _CACHE_FILE)
    if os.path.exists(local):
        ret = _read_gspc_close(local)
    else:
        # 2) repo-level split-only OHLC parquet
        repo = os.path.join(cache_dir, _REPO_GSPC_FILE)
        if os.path.exists(repo):
            ret = _read_gspc_close(repo)
        elif not fetch:
            raise FileNotFoundError(
                f"No cached daily tape for ^GSPC at {local} or {repo}. "
                f"Call fetch_daily(fetch=True) once to populate the cache."
            )
        else:
            # 3) network fetch (explicit opt-in)
            import yfinance as yf  # lazy import

            raw = yf.download(
                "^GSPC", start=start, interval="1d",
                auto_adjust=True, progress=False,
            )
            if raw.empty:
                raise RuntimeError("yfinance returned no data for ^GSPC")
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            px = raw["Close"].dropna()
            px.index = pd.DatetimeIndex(px.index)
            if px.index.tz is not None:
                px.index = px.index.tz_localize(None)
            px = px.sort_index()
            ret = px.pct_change().dropna()
            ret.name = "ret"
            ret.index.name = "date"
            os.makedirs(STUDY_CACHE, exist_ok=True)
            ret.to_frame().to_parquet(local)

    # Clip to the study window.
    ret = ret[(ret.index >= pd.Timestamp(start)) & (ret.index <= pd.Timestamp(end))]
    ret = ret.sort_index()
    mask = is_tournament(pd.DatetimeIndex(ret.index))
    return ret, mask


def fingerprint(ret: pd.Series) -> str:
    """A short content fingerprint of the return series, for the as-of stamp."""
    vals = np.ascontiguousarray(pd.Series(ret).dropna().to_numpy(dtype=float))
    h = hashlib.sha1(vals.tobytes())
    return h.hexdigest()[:12]
