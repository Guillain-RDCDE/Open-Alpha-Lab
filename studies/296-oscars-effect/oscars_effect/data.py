"""Data layer for Study 296 (Oscars-Effect).

Two tapes, one shape (a daily close-to-close return Series indexed by date) plus a
hardcoded table of Academy Award ceremonies:

- ``OSCAR_CEREMONIES`` — a hardcoded table of every Academy Award ceremony from
  the 67th (1995, honoring 1994 films) through the 97th (2025, honoring 2024
  films). Each row records the ceremony date, the ceremony number, the year of
  films honored, the Best Picture winner, and a small ``genre`` tag. The dates are
  deterministic, well-documented historical facts (the ceremony is broadcast live;
  its date is not contested data). Sources: the Academy of Motion Picture Arts and
  Sciences (oscars.org), Wikipedia "List of Academy Awards ceremonies." We start at
  1995 because the cached ^GSPC daily tape begins in January 1994, so the first
  ceremony with a clean prior-trading-day return is the March 1995 ceremony.

- ``synthetic_daily`` — a *deterministic, offline* generator. A ``post_oscar_bp``
  knob adds a one-day bump to the trading day *after* each ceremony;
  ``post_oscar_bp=0`` is a pure random walk with no Oscars seasonality. This is the
  study's null in a bottle, used by the positive control.

- ``fetch_daily`` — the real S&P 500 (^GSPC) daily tape back to 1994, via Yahoo
  Finance, **cache-first by default**. The result is stored as parquet under
  ``_cache/`` so the test suite and the reproducible core never touch the network.

No look-ahead: the ceremony happens on a Sunday/Monday evening (US time), so the
first *tradeable* reaction is the next regular session. The event window is built
around that next trading day, never the ceremony day itself.
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

# The study-local cache file (populated by examples/verify.py copying from the
# repo-level ^GSPC daily tape). Cache-first: tests/notebooks never hit the network.
_CACHE_FILE = "oscars_gspc_daily.parquet"
# Repo-level fallback (a ^GSPC daily tape already staged for several studies).
_REPO_GSPC_FILE = "chinese_zodiac_GSPC_daily.parquet"


# ---------------------------------------------------------------------------
# The hardcoded Oscar-ceremony table — the study's deterministic offline core
# ---------------------------------------------------------------------------
# Columns:
#   ceremony   : ordinal ceremony number (67th = 67, ..., 97th = 97)
#   film_year  : year of films honored (the ceremony in year Y honors year Y-1)
#   date       : the calendar date the ceremony was held (ISO string)
#   best_pic   : the Best Picture winner
#   genre      : a coarse genre tag for the winner (drama / biopic / crime /
#                war / fantasy / musical / comedy / thriller / scifi). This is a
#                soft, hand-assigned label used only for a slice-and-dice cell;
#                it is NOT a contested data source and carries no trading meaning.
#
# Sources: oscars.org (Academy of Motion Picture Arts and Sciences);
#          Wikipedia "List of Academy Awards ceremonies."
# As-of: 2026-06-17. Latest entry = 97th ceremony (March 2, 2025), Best Picture
# winner "Anora."

OSCAR_CEREMONIES: list[dict] = [
    {"ceremony": 67, "film_year": 1994, "date": "1995-03-27", "best_pic": "Forrest Gump",                          "genre": "drama"},
    {"ceremony": 68, "film_year": 1995, "date": "1996-03-25", "best_pic": "Braveheart",                            "genre": "war"},
    {"ceremony": 69, "film_year": 1996, "date": "1997-03-24", "best_pic": "The English Patient",                   "genre": "drama"},
    {"ceremony": 70, "film_year": 1997, "date": "1998-03-23", "best_pic": "Titanic",                               "genre": "drama"},
    {"ceremony": 71, "film_year": 1998, "date": "1999-03-21", "best_pic": "Shakespeare in Love",                   "genre": "comedy"},
    {"ceremony": 72, "film_year": 1999, "date": "2000-03-26", "best_pic": "American Beauty",                       "genre": "drama"},
    {"ceremony": 73, "film_year": 2000, "date": "2001-03-25", "best_pic": "Gladiator",                             "genre": "war"},
    {"ceremony": 74, "film_year": 2001, "date": "2002-03-24", "best_pic": "A Beautiful Mind",                      "genre": "biopic"},
    {"ceremony": 75, "film_year": 2002, "date": "2003-03-23", "best_pic": "Chicago",                               "genre": "musical"},
    {"ceremony": 76, "film_year": 2003, "date": "2004-02-29", "best_pic": "The Lord of the Rings: ROTK",           "genre": "fantasy"},
    {"ceremony": 77, "film_year": 2004, "date": "2005-02-27", "best_pic": "Million Dollar Baby",                   "genre": "drama"},
    {"ceremony": 78, "film_year": 2005, "date": "2006-03-05", "best_pic": "Crash",                                 "genre": "drama"},
    {"ceremony": 79, "film_year": 2006, "date": "2007-02-25", "best_pic": "The Departed",                          "genre": "crime"},
    {"ceremony": 80, "film_year": 2007, "date": "2008-02-24", "best_pic": "No Country for Old Men",                "genre": "thriller"},
    {"ceremony": 81, "film_year": 2008, "date": "2009-02-22", "best_pic": "Slumdog Millionaire",                   "genre": "drama"},
    {"ceremony": 82, "film_year": 2009, "date": "2010-03-07", "best_pic": "The Hurt Locker",                       "genre": "war"},
    {"ceremony": 83, "film_year": 2010, "date": "2011-02-27", "best_pic": "The King's Speech",                     "genre": "biopic"},
    {"ceremony": 84, "film_year": 2011, "date": "2012-02-26", "best_pic": "The Artist",                            "genre": "comedy"},
    {"ceremony": 85, "film_year": 2012, "date": "2013-02-24", "best_pic": "Argo",                                  "genre": "thriller"},
    {"ceremony": 86, "film_year": 2013, "date": "2014-03-02", "best_pic": "12 Years a Slave",                      "genre": "drama"},
    {"ceremony": 87, "film_year": 2014, "date": "2015-02-22", "best_pic": "Birdman",                               "genre": "comedy"},
    {"ceremony": 88, "film_year": 2015, "date": "2016-02-28", "best_pic": "Spotlight",                             "genre": "drama"},
    {"ceremony": 89, "film_year": 2016, "date": "2017-02-26", "best_pic": "Moonlight",                             "genre": "drama"},
    {"ceremony": 90, "film_year": 2017, "date": "2018-03-04", "best_pic": "The Shape of Water",                    "genre": "fantasy"},
    {"ceremony": 91, "film_year": 2018, "date": "2019-02-24", "best_pic": "Green Book",                            "genre": "drama"},
    {"ceremony": 92, "film_year": 2019, "date": "2020-02-09", "best_pic": "Parasite",                              "genre": "thriller"},
    {"ceremony": 93, "film_year": 2020, "date": "2021-04-25", "best_pic": "Nomadland",                             "genre": "drama"},
    {"ceremony": 94, "film_year": 2021, "date": "2022-03-27", "best_pic": "CODA",                                  "genre": "drama"},
    {"ceremony": 95, "film_year": 2022, "date": "2023-03-12", "best_pic": "Everything Everywhere All at Once",     "genre": "scifi"},
    {"ceremony": 96, "film_year": 2023, "date": "2024-03-10", "best_pic": "Oppenheimer",                           "genre": "biopic"},
    {"ceremony": 97, "film_year": 2024, "date": "2025-03-02", "best_pic": "Anora",                                 "genre": "comedy"},
]

OSCAR_DF: pd.DataFrame = pd.DataFrame(OSCAR_CEREMONIES)


def oscar_table() -> pd.DataFrame:
    """Return a clean copy of the hardcoded Oscar-ceremony table.

    Columns: ``ceremony`` (int), ``film_year`` (int), ``date`` (datetime64),
    ``best_pic`` (str), ``genre`` (str). The ``date`` is the calendar date of the
    ceremony broadcast — the *event* whose market reaction the next session we test.
    """
    df = OSCAR_DF.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Event-window construction
# ---------------------------------------------------------------------------
def next_trading_day(date: pd.Timestamp, trading_index: pd.DatetimeIndex) -> pd.Timestamp | None:
    """The first trading session strictly after ``date`` (the first tradeable reaction).

    The ceremony airs on a Sunday or Monday evening US time, so the earliest a
    trader can act on the result is the *next* regular session. Returns ``None`` if
    no such session exists in ``trading_index``.
    """
    later = trading_index[trading_index > pd.Timestamp(date).normalize()]
    if len(later) == 0:
        return None
    return later[0]


def event_windows(
    ret: pd.Series,
    ceremonies: pd.DataFrame | None = None,
    pre: int = 1,
    post: int = 3,
) -> pd.DataFrame:
    """Build a tidy event-window panel around each Oscar ceremony.

    For each ceremony we locate the *next* trading session (event day 0, the first
    tradeable reaction) and record returns for relative days ``-pre .. +post`` around
    it. One execution lag is baked in: the ceremony day is never used; event day 0 is
    the first full session after the broadcast.

    Returns a long DataFrame with columns:
      ``ceremony``, ``film_year``, ``best_pic``, ``genre``, ``rel_day`` (int),
      ``date`` (the actual session), ``ret`` (that session's simple return).
    """
    r = pd.Series(ret).astype(float).dropna().sort_index()
    idx = pd.DatetimeIndex(r.index)
    if ceremonies is None:
        ceremonies = oscar_table()

    rows: list[dict] = []
    for _, c in ceremonies.iterrows():
        day0 = next_trading_day(pd.Timestamp(c["date"]), idx)
        if day0 is None:
            continue
        loc = idx.get_loc(day0)
        for rel in range(-pre, post + 1):
            j = loc + rel
            if j < 0 or j >= len(idx):
                continue
            sess = idx[j]
            rows.append({
                "ceremony": int(c["ceremony"]),
                "film_year": int(c["film_year"]),
                "best_pic": c["best_pic"],
                "genre": c["genre"],
                "rel_day": int(rel),
                "date": sess,
                "ret": float(r.iloc[j]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 31,
    annual_drift_bp: float = 700.0,
    vol_ann: float = 0.16,
    post_oscar_bp: float = 0.0,
    seed: int = 296,
) -> tuple[pd.Series, pd.DataFrame, dict]:
    """A reproducible daily return series with a planted post-Oscar bump (or none).

    Returns are i.i.d. normal with an annualised drift of ``annual_drift_bp`` basis
    points and annual vol ``vol_ann``, except the single trading day *after* each
    ceremony, which receives an extra ``post_oscar_bp`` basis points. When
    ``post_oscar_bp = 0`` (the null) every day is identical and the ceremony label
    carries no information. This is the study's null in a bottle.

    Returns ``(returns, ceremonies, truth)`` where ``ceremonies`` is the hardcoded
    table clipped to the synthetic window and ``truth`` records planted parameters.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("1994-01-03")
    all_days = pd.bdate_range(start=start, periods=n_years * 252)
    n = len(all_days)
    daily_drift = annual_drift_bp * 1e-4 / 252
    daily_vol = vol_ann / np.sqrt(252)
    r = rng.normal(daily_drift, daily_vol, n)
    ret = pd.Series(r, index=all_days, name="ret")

    # Plant the bump on the trading day after each in-window ceremony.
    cer = oscar_table()
    cer = cer[
        (cer["date"] >= all_days.min()) & (cer["date"] <= all_days.max())
    ].copy()
    n_planted = 0
    idx = pd.DatetimeIndex(ret.index)
    for _, c in cer.iterrows():
        day0 = next_trading_day(pd.Timestamp(c["date"]), idx)
        if day0 is None:
            continue
        ret.loc[day0] += post_oscar_bp * 1e-4
        n_planted += 1

    truth = {
        "n_years": n_years,
        "annual_drift_bp": annual_drift_bp,
        "vol_ann": vol_ann,
        "post_oscar_bp": post_oscar_bp,
        "seed": seed,
        "n_obs": int(n),
        "n_ceremonies": int(len(cer)),
        "n_planted": int(n_planted),
    }
    return ret, cer.reset_index(drop=True), truth


# ---------------------------------------------------------------------------
# Real tape — ^GSPC daily, cache-first
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str, fname: str = _CACHE_FILE) -> str:
    return os.path.join(cache_dir, fname)


def _load_cached_returns() -> pd.Series | None:
    """Try the study-local cache, then the repo-level ^GSPC tape. Returns None if absent."""
    candidates = [
        _cache_path(STUDY_CACHE, _CACHE_FILE),
        _cache_path(DEFAULT_CACHE, _CACHE_FILE),
        _cache_path(DEFAULT_CACHE, _REPO_GSPC_FILE),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            ret = df["ret"] if "ret" in df.columns else df.iloc[:, 0]
            ret = ret.copy()
            ret.index = pd.DatetimeIndex(ret.index)
            if ret.index.tz is not None:
                ret.index = ret.index.tz_localize(None)
            ret.name = "ret"
            return ret.sort_index()
    return None


def fetch_daily(
    cache_dir: str = STUDY_CACHE,
    fetch: bool = False,
    start: str = "1994-01-01",
) -> tuple[pd.Series, pd.DataFrame]:
    """Daily close-to-close return for ^GSPC since ``start``; cache-first unless ``fetch=True``.

    Returns ``(returns, ceremonies)``:

    - ``returns``    — a simple-return Series aligned to trading days.
    - ``ceremonies`` — the hardcoded Oscar table clipped to the tape's window.

    Network is touched only when ``fetch=True``; the cached parquet persists so every
    downstream consumer — tests, notebooks, verify.py — is offline by default.
    Raises ``FileNotFoundError`` when no cache is present and ``fetch=False``.
    """
    ret = _load_cached_returns()
    if ret is not None:
        cer = oscar_table()
        cer = cer[
            (cer["date"] >= ret.index.min()) & (cer["date"] <= ret.index.max())
        ].reset_index(drop=True)
        return ret, cer

    if not fetch:
        raise FileNotFoundError(
            "No cached daily tape for ^GSPC. "
            "Call fetch_daily(fetch=True) once to populate the cache, "
            "or run examples/verify.py which stages it from the repo-level _cache."
        )

    import yfinance as yf  # lazy: only when we actually go to the network

    raw = yf.download("^GSPC", start=start, interval="1d", auto_adjust=True, progress=False)
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
    os.makedirs(cache_dir, exist_ok=True)
    ret.to_frame().to_parquet(_cache_path(cache_dir))
    cer = oscar_table()
    cer = cer[
        (cer["date"] >= ret.index.min()) & (cer["date"] <= ret.index.max())
    ].reset_index(drop=True)
    return ret, cer


def fingerprint(ret: pd.Series) -> str:
    """A short content fingerprint of the return series, for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(ret.to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]
