"""Data access for the first-hour study — the **session panel** and where it comes from.

One object runs the whole study: a per-**session** features frame, one row per trading
day, carrying just enough of the morning and the close to test the claim:

    oc_ret    first-hour (09:30->10:30 ET) return         -> the "opening candle"
    rest_ret  rest-of-day (10:30->16:00 ET) return        -> the part still unknown at 10:30
    day_ret   full RTH session (09:30->16:00 ET) return    -> what "closes red" means
    ib_high_rejected  did the first-hour HIGH print before its LOW?  -> the IB-rejection flag

The booleans the edgeful pitch trades on are derived from those:
``oc_red = oc_ret < 0``, ``session_red = day_ret < 0``, ``ib_high_rejected`` as above.
This module produces the panel two ways, the desk's standing split:

    * :func:`synthetic_sessions` — fully **offline**. A toy tape with a *baked-in* amount of
      intraday momentum (``momentum``) and an IB-rejection flag that is **conditionally
      independent of the afternoon** — i.e. it carries, by construction, *zero* predictive
      power beyond the sign of the morning. That ground truth is exactly what the tests assert
      the decomposition recovers, and the thesis the real run then confirms. No network in CI.
    * :func:`fetch_intraday` + :func:`daily_features` — pull real intraday bars from Yahoo!
      Finance, cache each (ticker, interval) to parquet, and reduce them to the session panel.
      **Cache-only** unless ``fetch=True``: a missing cache is skipped, never a silent stall.

The resolution decision, named up front as a house rule (see METHODOLOGY.md). The
**IB-rejection** flag needs *intra-hour* ordering (which of the first hour's high / low printed
first), which only sub-hourly bars can answer — and Yahoo serves sub-hourly history for **at
most 60 days**. So the faithful full-confluence replication runs on **ES=F / NQ=F at 5-minute**
fidelity over ~60 days (~58 sessions, mirroring edgeful's own 6-month / 128-session window and
its tiny conditional sub-samples). The high-statistical-power test of the *opening-candle* leg
— which needs only the 09:30->10:30 sign and the close — runs on **SPY / QQQ at 1-hour**
fidelity, whose bars align exactly to the 09:30 RTH open and reach back ~730 sessions. SPY/QQQ
are the cash proxies for ES/NQ; edgeful's own members built the same dashboard on QQQ/SPY.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(_HERE, "..", "_cache")

NY_TZ = "America/New_York"

# The canonical session-panel columns every consumer (signals, decompose) expects.
FEATURE_COLUMNS = [
    "session_open", "oc_close", "session_close",
    "oc_ret", "rest_ret", "day_ret",
    "oc_red", "session_red", "rest_red",
    "ib_high", "ib_low", "ib_high_rejected",
]


# --------------------------------------------------------------------------- #
# Synthetic tape — offline, with a baked-in momentum and a null IB flag
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GroundTruth:
    """What the synthetic generator baked in, so a test can check the decomposition."""
    momentum: float          # afternoon's loading on the morning return (the real, modest edge)
    baseline_red: float      # unconditional P(session closes red), by construction ~0.5 - drift
    n_sessions: int
    ib_is_null: bool         # True: IB-rejection carries no info beyond the morning's sign


def synthetic_sessions(
    n_sessions: int = 520,
    momentum: float = 0.12,
    first_hour_steps: int = 12,
    drift_step: float = 0.00004,
    sigma_step: float = 0.0016,
    sigma_rest: float = 0.0060,
    drift_rest: float = 0.00020,
    seed: int = 0,
) -> tuple[pd.DataFrame, GroundTruth]:
    """A toy book of trading sessions with a **known** amount of intraday momentum.

    Each session's **first hour** is a short random walk of ``first_hour_steps`` log-return
    increments (drift ``drift_step``, vol ``sigma_step``); its sum is the opening-candle return
    ``oc_ret`` and the *timing* of the walk's running max vs running min gives the IB-rejection
    flag (high-before-low). The **afternoon** is

        rest_ret = momentum * oc_ret + drift_rest + sigma_rest * N(0, 1)

    so it continues the morning with strength ``momentum`` — a genuine but modest edge — and,
    crucially, depends on the morning *only through* ``oc_ret``: it is **independent of the
    IB-rejection flag given the morning's sign**. That is the baked-in null the study exists to
    expose: the "confluence" cannot beat the opening candle alone, because the second signal
    adds no information. ``day_ret = oc_ret + rest_ret``.

    Returns ``(features, truth)`` — ``features`` is the canonical session panel (one row per
    day) on a business-day index; ``truth`` carries the baked-in momentum and baseline.
    Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2024-01-02", periods=n_sessions, name="date")

    # First hour as a short random walk; its sum is the opening-candle (log) move.
    steps = rng.normal(drift_step, sigma_step, size=(n_sessions, first_hour_steps))
    path = np.cumsum(steps, axis=1)                    # running first-hour log-price (from 0)
    oc_log = path[:, -1]
    # Afternoon continues the morning with strength `momentum`, depending on it ONLY through
    # oc_log. day (log) move is the two legs summed.
    rest_log = momentum * oc_log + drift_rest + sigma_rest * rng.standard_normal(n_sessions)
    day_log = oc_log + rest_log

    session_open = np.full(n_sessions, 100.0)
    oc_close = session_open * np.exp(oc_log)
    session_close = session_open * np.exp(day_log)
    # Simple returns, exactly as the real reducer computes them (close/open - 1).
    oc_ret = oc_close / session_open - 1.0
    rest_ret = session_close / oc_close - 1.0
    day_ret = session_close / session_open - 1.0

    # The IB-rejection flag is drawn **independently** of everything else — the baked-in null:
    # it carries no information about the close beyond the opening candle's own sign. (The
    # first-hour high/low *levels* are still the path's, for display realism.)
    ib_high_rejected = rng.random(n_sessions) < 0.5

    feat = pd.DataFrame(
        {
            "session_open": session_open,
            "oc_close": oc_close,
            "session_close": session_close,
            "oc_ret": oc_ret,
            "rest_ret": rest_ret,
            "day_ret": day_ret,
            "oc_red": oc_ret < 0,
            "session_red": day_ret < 0,
            "rest_red": rest_ret < 0,
            "ib_high": session_open * np.exp(path.max(axis=1)),
            "ib_low": session_open * np.exp(path.min(axis=1)),
            "ib_high_rejected": pd.array(ib_high_rejected, dtype="boolean"),
        },
        index=idx,
    )[FEATURE_COLUMNS]

    truth = GroundTruth(
        momentum=momentum,
        baseline_red=float((day_ret < 0).mean()),
        n_sessions=n_sessions,
        ib_is_null=True,
    )
    return feat, truth


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! intraday bars, cached per (ticker, interval), reduced to sessions
# --------------------------------------------------------------------------- #

def _cache_path(ticker: str, interval: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "_").replace("^", "_")
    return os.path.join(cache_dir, f"bars_{safe}_{interval}.parquet")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance's columns to Open/High/Low/Close and put the index in NY time."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    df = df[keep].copy()
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    df.index = idx.tz_convert(NY_TZ)
    df.index.name = "ts"
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_intraday(
    ticker: str,
    interval: str = "1h",
    period: str = "730d",
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
) -> pd.DataFrame:
    """Return (and cache) intraday OHLC bars for ``ticker`` at ``interval``, NY-tz indexed.

    **Cache-only by default**: if a parquet exists it is returned; otherwise an empty frame is
    returned *unless* ``fetch=True``, in which case Yahoo is hit once and the result cached. The
    network import is lazy, so the offline core (and CI) never imports ``yfinance``. Yahoo's
    intraday history is capped by interval (~60 days for sub-hourly, ~730 for ``1h``), so
    ``period`` is a *rolling* window ending ~now — pin the headline with ``quantlab.repro.as_of``.
    """
    path = _cache_path(ticker, interval, cache_dir)
    if os.path.exists(path) and not fetch:
        return pd.read_parquet(path)

    if not fetch:
        return pd.DataFrame()

    import yfinance as yf  # lazy: offline core never imports it

    raw = yf.download(ticker, period=period, interval=interval,
                      auto_adjust=False, prepost=False, progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = _normalize(raw)
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(path)
    return df


def _hhmm(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(idx.strftime("%H:%M"), index=idx)


def daily_features(
    bars: pd.DataFrame,
    first_hour_open: str = "09:30",
    first_hour_end: str = "10:30",
    session_end: str = "16:00",
    has_subhour: bool = True,
) -> pd.DataFrame:
    """Reduce intraday RTH ``bars`` to the canonical one-row-per-session panel.

    Restricts to the regular session (``first_hour_open``..``session_end`` ET), then per date:

    * **opening candle** = the bars in ``[first_hour_open, first_hour_end)``. ``session_open`` is
      that block's first open; ``oc_close`` its last close; ``oc_ret = oc_close/session_open - 1``.
    * **IB-rejection** (only if ``has_subhour``): within the first-hour block, ``ib_high_rejected``
      is True when the block's HIGH prints *before* its LOW (``idxmax(High) < idxmin(Low)``) — the
      edgeful definition (high rejected first, low breaks second). With hourly bars the first hour
      is a single bar, so the flag is ``NA``.
    * **close** = the last RTH bar of the day; ``session_close`` its close;
      ``day_ret = session_close/session_open - 1``; ``rest_ret = session_close/oc_close - 1``.

    A day is kept only if it has a full first-hour block and at least one later bar (so half-days
    with a truncated open are dropped, never silently mislabelled). Returns the ``FEATURE_COLUMNS``
    frame on a ``date`` index.
    """
    if bars.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    df = bars.copy()
    hhmm = _hhmm(df.index)
    in_session = (hhmm >= first_hour_open) & (hhmm <= session_end)
    df = df[in_session.to_numpy()]
    hhmm = hhmm[in_session.to_numpy()]
    in_fh = ((hhmm >= first_hour_open) & (hhmm < first_hour_end)).to_numpy()

    dates = pd.Index(df.index.date, name="date")
    rows = []
    for d, sl in df.groupby(dates):
        fh = sl[in_fh[dates == d]]
        if fh.empty:
            continue
        rest = sl[sl.index > fh.index[-1]]
        if rest.empty:                       # need something after the first hour to grade
            continue
        s_open = float(fh["Open"].iloc[0])
        oc_close = float(fh["Close"].iloc[-1])
        s_close = float(rest["Close"].iloc[-1])
        ib_high = float(fh["High"].max())
        ib_low = float(fh["Low"].min())
        if has_subhour and len(fh) > 1:
            rejected = bool(fh["High"].idxmax() < fh["Low"].idxmin())
        else:
            rejected = pd.NA
        oc_ret = oc_close / s_open - 1.0
        day_ret = s_close / s_open - 1.0
        rest_ret = s_close / oc_close - 1.0
        rows.append((d, s_open, oc_close, s_close, oc_ret, rest_ret, day_ret,
                     oc_ret < 0, day_ret < 0, rest_ret < 0, ib_high, ib_low, rejected))

    out = pd.DataFrame(rows, columns=["date"] + FEATURE_COLUMNS).set_index("date")
    out.index = pd.DatetimeIndex(out.index, name="date")
    out["ib_high_rejected"] = out["ib_high_rejected"].astype("boolean")
    return out
