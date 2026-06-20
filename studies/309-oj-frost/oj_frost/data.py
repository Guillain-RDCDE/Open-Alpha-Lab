"""Data layer for Study 309 (OJ-Frost).

Two tapes, one shape (a tz-naive daily close-price frame), plus a hardcoded table of
Florida hard-freeze events:

- ``synthetic_oj`` — a *deterministic, offline* generator. A daily random-walk OJ price
  with two knobs that plant exactly the things a "freeze trade" could possibly harvest:
  ``freeze_jump`` plants a positive price spike in the window right after each (synthetic)
  freeze date — the *positive control* — and ``winter_drift`` plants a mild Dec–Feb
  seasonal tilt. With both knobs at zero the tape is a pure random walk — the *null* — so
  a test can assert the event-study fires *only* when a freeze effect is actually planted.
- ``fetch_oj`` — the real Yahoo! daily tape for ``OJ=F`` (frozen-concentrate OJ futures
  continuous front month), cache-only by default so the test-suite and the reproducible
  core never touch the network.

The freeze table (``FREEZE_EVENTS``) is hardcoded from the historical record of severe
Florida citrus-belt freezes (the events that drove the *Trading Places* lore). It is a
*calendar-known* list, so the event study needs **no execution lag** for entries timed
relative to a freeze date that is in the past — the lag discipline lives in
``strategy.py`` for the price-reactive variant.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# The hardcoded freeze table — severe Florida citrus-belt freezes
# ---------------------------------------------------------------------------
# Dates of notable hard freezes that damaged the Florida orange crop. These are the
# events the "freeze trade" folklore is built on. We use the date the freeze *struck*
# (the cold night); a trade can only act on the first *tradable* session at or after it.
# The list is deliberately conservative — only widely-documented severe freezes.
FREEZE_EVENTS: list[str] = [
    "1977-01-19",  # Jan 1977 freeze (snow in Miami)
    "1981-01-13",  # Jan 1981 freeze
    "1982-01-11",  # Jan 1982 freeze
    "1983-12-25",  # "Christmas freeze" of 1983 — severe crop loss
    "1985-01-21",  # Jan 1985 freeze — severe crop loss
    "1989-12-23",  # Dec 1989 freeze
    "1996-02-05",  # Feb 1996 freeze
    "2001-01-04",  # Jan 2001 freeze
    "2003-01-24",  # Jan 2003 freeze
    "2009-01-22",  # Jan 2009 freeze
    "2010-01-11",  # Jan 2010 prolonged cold snap
    "2022-01-30",  # Jan 2022 cold snap
]


def freeze_dates() -> pd.DatetimeIndex:
    """The hardcoded freeze table as a tz-naive DatetimeIndex (sorted)."""
    return pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in FREEZE_EVENTS), name="freeze")


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_oj(
    start: str = "1975-01-02",
    n_days: int = 12_000,
    daily_vol: float = 0.018,
    freeze_jump: float = 0.0,
    winter_drift: float = 0.0,
    n_freezes: int = 12,
    jump_window: int = 5,
    seed: int = 309,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OJ close tape with optionally-planted freeze spikes.

    Log returns are i.i.d. normal of standard deviation ``daily_vol`` (a random walk),
    plus two optional planted structures:

    - ``freeze_jump`` — on the ``jump_window`` sessions *following* each synthetic freeze
      date, add ``freeze_jump / jump_window`` to the daily log-return (a positive spike
      spread over the window). ``freeze_jump = 0`` plants nothing → the null.
    - ``winter_drift`` — add ``winter_drift`` (per day) to the log-return on every Dec,
      Jan and Feb session (a mild winter seasonal tilt). ``winter_drift = 0`` → none.

    The synthetic freeze dates are placed deterministically: ``n_freezes`` of them, one
    per year picked from the winter months, seeded. ``truth`` records the planted
    parameters and the synthetic freeze dates so a test can target the event window.

    Returns ``(frame, truth)`` where ``frame`` has a single ``close`` column.
    """
    rng = np.random.default_rng(seed)
    # Decorative-but-real business-day index; n_days kept well under any ns overflow.
    sessions = pd.bdate_range(start=start, periods=n_days)

    log_ret = rng.normal(0.0, daily_vol, n_days)

    # Winter (DJF) tilt.
    if winter_drift:
        months = sessions.month.to_numpy()
        winter = (months == 12) | (months == 1) | (months == 2)
        log_ret[winter] += winter_drift

    # Place n_freezes synthetic freeze dates, one per distinct winter, deterministically.
    syn_freezes: list[pd.Timestamp] = []
    years = sorted(set(sessions.year))
    winter_years = [y for y in years if (sessions.year == y).sum() > 200]
    chosen_years = winter_years[: n_freezes] if n_freezes <= len(winter_years) else winter_years
    for y in chosen_years:
        # A January session in year y.
        jan = sessions[(sessions.year == y) & (sessions.month == 1)]
        if len(jan) == 0:
            continue
        pick = jan[rng.integers(0, len(jan))]
        syn_freezes.append(pick)

    if freeze_jump and syn_freezes:
        # Front-load the spike: most of the pop lands on the FIRST session after the
        # freeze (markets react fast to the cold-night news), with a geometric decay over
        # the window. The weights sum to 1, so `freeze_jump` is the total planted log-move.
        decay = 0.5
        raw_w = np.array([decay ** k for k in range(jump_window)], dtype=float)
        weights = raw_w / raw_w.sum()
        pos = {ts: i for i, ts in enumerate(sessions)}
        for ts in syn_freezes:
            i0 = pos.get(ts)
            if i0 is None:
                continue
            for k in range(1, jump_window + 1):  # the sessions AFTER the freeze
                j = i0 + k
                if j < n_days:
                    log_ret[j] += freeze_jump * weights[k - 1]

    close = 100.0 * np.exp(np.cumsum(log_ret))
    frame = pd.DataFrame({"close": close},
                         index=pd.DatetimeIndex(sessions, name="date"))
    truth = {
        "daily_vol": daily_vol,
        "freeze_jump": freeze_jump,
        "winter_drift": winter_drift,
        "jump_window": jump_window,
        "n_days": n_days,
        "seed": seed,
        "syn_freezes": pd.DatetimeIndex(syn_freezes, name="freeze"),
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"close_{safe}_daily.parquet")


def fetch_oj(
    ticker: str = "OJ=F",
    start: str = "2000-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily close for ``ticker`` (default OJ=F); cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as a
    parquet under ``_cache/``). ``OJ=F`` on Yahoo is the continuous front-month
    frozen-concentrate OJ future; its history on Yahoo is short (roughly the 2000s on),
    which is itself a load-bearing fact for the study — most of the famous freezes
    predate the tape.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached OJ tape for {ticker} at {path}. "
                f"Call fetch_oj({ticker!r}, fetch=True) once to populate the cache."
            )
        frame = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            ticker, start=start, interval="1d", auto_adjust=True, progress=False
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        frame = raw.rename(columns=str.lower)[["close"]]
        frame.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        frame.to_parquet(path)

    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    return frame


def fingerprint(frame: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(frame["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
