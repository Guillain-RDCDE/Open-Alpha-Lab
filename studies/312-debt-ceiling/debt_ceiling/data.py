"""Data layer for Study 312 (Debt-Ceiling) — the *volatility* angle.

Two tapes, one shape per series (a tz-naive daily OHLCV frame). Unlike Study 311
(Government-Shutdown), which works on SPY *total-return* prices and a directional
forward-return test, this study works on the **VIX** (and SPY for realized vol) and asks
a question about *volatility behaviour* around the binding US **debt-ceiling X-dates**:

- ``synthetic_vix`` — a *deterministic, offline* generator of a mean-reverting VIX-like
  series with an optional **planted event hump**: on a list of "X-date" sessions we add a
  positive bump to the level over the days *before* the deadline that decays *after* it
  (the long-vol-into / short-vol-out positive control), or nothing at all (the null).
  This lets a test assert the event-study engine recovers a planted vol hump only when
  one is really there.
- ``load_real`` — the real daily tape, cache-first via the shared ``_cache`` parquet
  panel (``^VIX_raw`` for the VIX, ``SPY_total_return`` for realized vol), with a graceful
  ``quantlab.data`` fallback and, only on an explicit ``fetch=True``, a network pull. The
  test-suite and reproducible core never touch the network.

THE X-DATE TABLE is hardcoded below: US debt-ceiling episodes that went to the wire (a
real "X-date" — the Treasury's projected cash-exhaustion date — was published and markets
priced a default tail), with the trading-relevant DEADLINE date. Routine, quietly-raised
ceilings are excluded and the exclusion is stated, in the open.

No look-ahead is baked in here — the event-window bookkeeping lives in ``strategy.py``.
The X-date is *calendar-known* in advance (Treasury announces a projected date weeks
ahead), so the canonical pre-deadline window needs no extra execution lag; a ``+1``
session entry lag is offered as a robustness check in the engine.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# The hardcoded debt-ceiling X-date table
# ---------------------------------------------------------------------------
# US debt-ceiling episodes since the VIX's 1990 inception that actually went to the
# brink — a Treasury "X-date" (projected cash-exhaustion) was published and markets
# priced a non-trivial technical-default tail. The ``deadline`` is the date the binding
# constraint was relieved (the bill signed / extraordinary-measures cliff), snapped by
# the engine to the first trading session on or after it. ``x_date`` is the contemporary
# Treasury-projected exhaustion date, for context. Routine, quietly-raised ceilings
# (dozens since 1990) are deliberately EXCLUDED — the canonical list keeps the standoffs
# markets actually feared.
#
# Sources: US Treasury "Debt Limit" press releases; Congressional Research Service
# "The Debt Limit Since 2011" (R43389) and "Debt Limit" primers; Bipartisan Policy
# Center debt-limit analyses; contemporaneous Reuters/WSJ wrap-ups.
X_DATES = pd.DataFrame(
    [
        # name,                          deadline,      x_date,        ramp
        ("2011 (Aug, S&P downgrade)",    "2011-08-02", "2011-08-02", True),
        ("2013 (Oct, fiscal cliff redux)", "2013-10-17", "2013-10-17", True),
        ("2015 (Nov, Boehner exit)",     "2015-11-03", "2015-11-03", False),
        ("2017 (Dec, Harvey supplemental)", "2017-12-08", "2017-12-08", False),
        ("2021 (Oct, short-term raise)", "2021-10-18", "2021-10-18", True),
        ("2021 (Dec, $2.5T raise)",      "2021-12-15", "2021-12-15", False),
        ("2023 (Jun, FRA debt deal)",    "2023-06-05", "2023-06-01", True),
    ],
    columns=["name", "deadline", "x_date", "ramp"],
)
X_DATES["deadline"] = pd.to_datetime(X_DATES["deadline"])
X_DATES["x_date"] = pd.to_datetime(X_DATES["x_date"])


def deadlines() -> pd.DatetimeIndex:
    """The binding debt-ceiling deadline dates as a tz-naive DatetimeIndex."""
    return pd.DatetimeIndex(X_DATES["deadline"].to_numpy(), name="deadline")


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_vix(
    n_days: int = 8400,
    event_effect: float = 0.0,
    pre: int = 20,
    post: int = 20,
    n_events: int = 40,
    base: float = 18.0,
    kappa: float = 0.05,
    sigma: float = 0.06,
    start: str = "1993-01-29",
    seed: int = 312,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict]:
    """A reproducible daily VIX-like tape with optional planted event humps.

    The level follows a discrete Ornstein-Uhlenbeck (mean-reverting) process in log
    space around ``base``, which is the right shape for an implied-vol index (sticky,
    spiky, mean-reverting). On ``n_events`` evenly-spaced "X-date" sessions we add a
    **tent-shaped** bump to the log level: it ramps up over the ``pre`` sessions *before*
    the deadline, peaks at the event bar, and decays over the ``post`` sessions *after*
    it — the planted "vol rises into the deadline, collapses on resolution" control.

    - ``event_effect = 0`` → a plain OU walk; X-dates carry no vol information. This is
      the null: the engine must NOT find a significant pre-deadline vol ramp.
    - ``event_effect > 0`` → a real vol hump around each event. This is the positive
      control: the engine must recover it.

    A companion ``close`` (a price level) is synthesised so the same frame can feed a
    realized-vol calculation; OHLC are decorative wicks around it.

    Returns ``(bars, events, truth)`` where ``bars`` carries a ``vix`` column (the level)
    plus OHLCV, ``events`` is the DatetimeIndex of the planted event sessions, and
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days)

    # OU in log-level: x_t = x_{t-1} + kappa*(log base - x_{t-1}) + sigma*eps.
    log_base = np.log(base)
    x = np.empty(n_days)
    x[0] = log_base
    eps = rng.normal(0.0, sigma, n_days)
    for i in range(1, n_days):
        x[i] = x[i - 1] + kappa * (log_base - x[i - 1]) + eps[i]

    # Plant evenly-spaced events, leaving room for the window on both sides.
    lo, hi = pre + 5, n_days - post - 5
    if hi > lo and n_events > 0:
        event_locs = np.unique(np.linspace(lo, hi, n_events).round().astype(int))
        for e in event_locs:
            # Tent: 0 -> peak over [e-pre, e], peak -> 0 over [e, e+post].
            for k in range(-pre, post + 1):
                j = e + k
                if 0 <= j < n_days:
                    w = (1.0 - abs(k) / (pre if k < 0 else post)) if (pre and post) else 0.0
                    x[j] += event_effect * max(w, 0.0)
    else:
        event_locs = np.array([], dtype=int)

    vix = np.exp(x)

    # A companion price level for realized vol: returns scaled by the vol level.
    daily_ret = rng.normal(0.0, 1.0, n_days) * (vix / 100.0) / np.sqrt(TRADING_DAYS_PER_YEAR)
    close = 100.0 * np.exp(np.cumsum(daily_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, 0.003, n_days)) * close
    hi_px = np.maximum(open_, close) + wick
    lo_px = np.minimum(open_, close) - wick
    vol = rng.integers(1_000_000, 50_000_000, n_days).astype(float)

    bars = pd.DataFrame(
        {
            "open": open_,
            "high": hi_px,
            "low": lo_px,
            "close": close,
            "volume": vol,
            "vix": vix,
        },
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    events = (
        pd.DatetimeIndex(bars.index[event_locs], name="event")
        if event_locs.size
        else pd.DatetimeIndex([], name="event")
    )
    truth = {
        "event_effect": event_effect,
        "pre": pre,
        "post": post,
        "n_events": int(events.size),
        "base": base,
        "kappa": kappa,
        "sigma": sigma,
        "n_days": n_days,
        "seed": seed,
    }
    return bars, events, truth


def synthetic_null_events(
    bars: pd.DataFrame,
    n_events: int = 40,
    window: int = 20,
    seed: int = 7,
) -> pd.DatetimeIndex:
    """Draw ``n_events`` random session dates from ``bars`` — the placebo/event-null control.

    These are *random* dates with no special meaning, used to ask: does the vol behaviour
    the engine measures around the real X-dates differ from what you'd measure around any
    random date at all? (The synthetic-event null in the brief.)
    """
    rng = np.random.default_rng(seed)
    lo, hi = window + 5, len(bars) - window - 5
    locs = rng.choice(np.arange(lo, hi), size=min(n_events, hi - lo), replace=False)
    return pd.DatetimeIndex(sorted(bars.index[locs]), name="null_event")


# ---------------------------------------------------------------------------
# Real tape — daily VIX / SPY, cache-first
# ---------------------------------------------------------------------------
def _cache_path(ticker: str) -> str:
    """Path to the shared cache parquet for a ticker.

    The VIX is cached as ``^VIX_raw.parquet`` (a raw index level — no dividends/splits,
    so total-return adjustment is meaningless for it); SPY as ``SPY_total_return.parquet``.
    """
    safe = ticker.replace("/", "")
    if ticker.upper() in ("^VIX", "VIX"):
        return os.path.join(DEFAULT_CACHE, "^VIX_raw.parquet")
    return os.path.join(DEFAULT_CACHE, f"{safe}_total_return.parquet")


def load_real(
    ticker: str = "^VIX",
    start: str = "1990-01-01",
    end: str | None = None,
    use_cache: bool = True,
    fetch: bool = False,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-first, network only on ``fetch``.

    Primary source is the shared cache parquet (``_cache/^VIX_raw.parquet`` for the VIX,
    ``_cache/SPY_total_return.parquet`` for SPY). On a cache miss it falls back to
    :func:`quantlab.data.fetch`; only with ``fetch=True`` does it go to the network (and
    cache the result). Columns are lower-cased ``open/high/low/close/volume``; the index
    is a tz-naive ``date``.

    NB: the VIX is an *index level*, not a tradable total-return series — it is loaded
    raw and labelled as such; you cannot hold the VIX, only VIX-linked products.
    """
    cache_path = _cache_path(ticker)
    bars = None
    if use_cache and os.path.exists(cache_path):
        bars = pd.read_parquet(cache_path)
    elif not fetch:
        try:
            import sys

            sys.path.insert(0, REPO_ROOT)
            from quantlab import data as qdata

            mode = "raw" if ticker.upper() in ("^VIX", "VIX") else "total_return"
            raw = qdata.fetch(ticker, start=start, end=end, mode=mode, use_cache=use_cache)
            bars = raw.rename(columns=str.lower)
        except Exception as exc:  # graceful offline fallback
            raise FileNotFoundError(
                f"No cached tape for {ticker} at {cache_path}, and the quantlab "
                f"fallback failed offline ({exc!r}). Call load_real(fetch=True) once "
                f"to populate the cache."
            ) from exc
    else:
        import yfinance as yf

        auto = ticker.upper() not in ("^VIX", "VIX")
        raw = yf.download(ticker, start=start, interval="1d", auto_adjust=auto,
                          progress=False)
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw
        os.makedirs(DEFAULT_CACHE, exist_ok=True)
        bars.to_parquet(cache_path)

    bars = bars.rename(columns=str.lower)
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in bars.columns]
    bars = bars[keep]
    bars.index = pd.DatetimeIndex(bars.index)
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    if end is not None:
        bars = bars.loc[:end]
    return bars.loc[start:]


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]
