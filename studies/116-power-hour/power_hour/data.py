"""Data layer for Study 116 (Power-Hour).

Two tapes, one shape — a per-**session** features frame with one row per trading day:

- ``synthetic_sessions`` — a *deterministic, offline* generator. A scalar ``continuation``
  knob controls how strongly the morning/midday return (09:30→15:00 ET) predicts the last-hour
  return (15:00→16:00 ET). ``continuation = 0`` is the null: the last hour is independent of
  everything that happened before it, so a trend-following signal is a fair coin. This is the
  study's null in a bottle and the only forecastable structure in the tape.

- ``fetch_daily_panel`` — pulls **1-hour bars** from Yahoo! Finance for the basket (SPY, QQQ,
  IWM), converts them to the canonical session panel, and caches as parquet. 1-hour bars are
  chosen deliberately: they give ~730 days of history (vs ~60 for 5-minute), which yields
  enough power to call a weak continuation effect absent. Yahoo's 1-hour cap on US tickers is
  approximately 730 calendar days. Cache-only by default (``fetch=False`` raises if no cache);
  pass ``fetch=True`` once to populate.

No look-ahead is baked in here — the morning/midday return is computed from bars *before*
15:00 ET; the last-hour return uses bars at 15:00–16:00 ET only.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

NY_TZ = "America/New_York"

# The canonical session-panel columns every consumer expects.
FEATURE_COLUMNS = [
    "morning_ret",    # 09:30 open → 15:00 close (the signal, known before last hour)
    "last_ret",       # 15:00 open → 16:00 close (the outcome — last hour's net)
    "full_ret",       # 09:30 open → 16:00 close (full-day return)
]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_sessions(
    n_sessions: int = 500,
    continuation: float = 0.0,
    sigma_morning: float = 0.008,
    sigma_last: float = 0.004,
    drift: float = 0.0002,
    seed: int = 116,
) -> tuple[pd.DataFrame, dict]:
    """Reproducible per-session panel with a controllable last-hour continuation effect.

    Each session draws:
    - ``morning_ret``   ~ N(drift, sigma_morning²) — the pre-15:00 move.
    - ``last_ret``      = continuation * morning_ret + eps_last
      where  eps_last  ~ N(0, sigma_last²).

    ``continuation`` is the *only* forecastable structure in the tape:
    - ``continuation = 0``  → last hour is independent of the morning; a signal based on
      the morning direction is a fair coin.
    - ``continuation > 0``  → the day's trend continues into the last hour; a momentum
      (follow-the-morning) signal can harvest it.
    - ``continuation < 0``  → the day reverses in the last hour; a contrarian signal wins.

    Returns ``(panel, truth)`` where ``panel`` is the canonical session DataFrame on a
    business-day index and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2022-01-03", periods=n_sessions, name="date")

    morning_ret = rng.normal(drift, sigma_morning, n_sessions)
    eps_last = rng.normal(0.0, sigma_last, n_sessions)
    last_ret = continuation * morning_ret + eps_last
    full_ret = morning_ret + last_ret + morning_ret * last_ret  # simple-return chain

    panel = pd.DataFrame(
        {
            "morning_ret": morning_ret,
            "last_ret": last_ret,
            "full_ret": full_ret,
        },
        index=idx,
    )[FEATURE_COLUMNS]

    truth = {
        "continuation": continuation,
        "sigma_morning": sigma_morning,
        "sigma_last": sigma_last,
        "drift": drift,
        "n_sessions": n_sessions,
        "seed": seed,
    }
    return panel, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo! 1-hour bars, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"panel_{safe}_1h.parquet")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns, lowercase, tz-convert to NY."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].copy()
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    df.index = idx.tz_convert(NY_TZ)
    df.index.name = "ts"
    return df[~df.index.duplicated(keep="last")].sort_index()


def _bars_to_panel(bars: pd.DataFrame) -> pd.DataFrame:
    """Reduce 1-hour RTH bars to the per-session feature panel.

    Yahoo Finance serves 1-hour US-equity bars stamped at their *open* time, with exactly
    seven bars per regular session: 09:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30.
    The 15:30 bar covers the last 30 minutes of the regular session (15:30→16:00) — this
    is the "power hour" (last-half-hour) bar we treat as the outcome.

    ``morning_ret``:  open of the 09:30 bar → close of the 14:30 bar (the pre-last-bar move).
    ``last_ret``   :  open of the 15:30 bar → close of the 15:30 bar (the power-hour move).
    ``full_ret``   :  open of the 09:30 bar → close of the 15:30 bar (the full-day move).

    A day is kept only if it has both an opening 09:30 bar and a closing 15:30 bar.
    """
    if bars.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    # RTH bars only: 09:30 through 15:55 (the 15:30 bar is the last one)
    rth = bars.between_time("09:30", "15:55")
    if rth.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    dates = pd.Index(rth.index.date, name="date")
    rows = []
    for d, sl in rth.groupby(dates):
        sl = sl.sort_index()
        times = sl.index.strftime("%H:%M")
        # Opening bar must start at 09:30
        open_bars = sl[times == "09:30"]
        # Morning: all bars strictly before the 15:30 power-hour bar
        morning_bars = sl[times < "15:30"]
        # Last bar (power hour): the 15:30 bar
        last_bars = sl[times == "15:30"]
        if open_bars.empty or morning_bars.empty or last_bars.empty:
            continue
        session_open = float(open_bars["open"].iloc[0])
        morning_close = float(morning_bars["close"].iloc[-1])
        last_open = float(last_bars["open"].iloc[0])
        last_close = float(last_bars["close"].iloc[-1])
        morning_ret = morning_close / session_open - 1.0
        last_ret = last_close / last_open - 1.0
        full_ret = last_close / session_open - 1.0
        rows.append((d, morning_ret, last_ret, full_ret))

    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    out = pd.DataFrame(rows, columns=["date"] + FEATURE_COLUMNS).set_index("date")
    out.index = pd.DatetimeIndex(out.index, name="date")
    return out


def fetch_daily_panel(
    ticker: str,
    period: str = "730d",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Per-session (morning_ret, last_ret, full_ret) panel for ``ticker``.

    Hits Yahoo! 1-hour bars (cache-only by default). With 1-hour fidelity Yahoo typically
    serves ~500-730 sessions — substantially more power than the ~60 days that 5-minute bars
    afford. Pass ``fetch=True`` once to populate the cache; subsequent calls read the parquet.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached 1h panel for {ticker} at {path}. "
                f"Call fetch_daily_panel({ticker!r}, fetch=True) once to populate the cache."
            )
        return pd.read_parquet(path)

    import yfinance as yf  # lazy: only when we actually go to the network

    raw = yf.download(
        ticker, period=period, interval="1h", auto_adjust=True, progress=False
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no 1h bars for {ticker}")
    bars = _normalize(raw)
    panel = _bars_to_panel(bars)
    os.makedirs(cache_dir, exist_ok=True)
    panel.to_parquet(path)
    return panel


def fingerprint(panel: pd.DataFrame) -> str:
    """A short content fingerprint of a session panel (last_ret column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(panel["last_ret"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
