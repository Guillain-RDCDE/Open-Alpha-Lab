"""Data layer for Study 467 (Bump-and-Run Reversal).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  Bulkowski's bump-and-run reversal is: a gentle **lead-in trendline**, then a speculative
  **bump** (price steepens and surges away from the line), then a **break back below the
  trendline** that is supposed to forecast a *downward reversal*. The believers' rule is to
  **short** that break. We plant exactly that: with ``edge > 0`` the path is built so that
  after a genuine bump-and-break the price really does fall (a reversal the short can bank);
  with ``edge = 0`` the log-return series is a pure random walk and the bump-then-break is a
  fair coin. This is the positive control — a harness that cannot bank the planted reversal
  proves nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the lead-in
trendline is fit on a *trailing* window, the bump and break are read on the close of *t*,
and the short is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs bump-and-run proponents draw on: the broad tape, big-cap tech, small
# caps, and a couple of cross-asset charts. Daily, liquid, long history.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    lead_in: int = 60,
    start: str = "2010-01-04",
    seed: int = 467,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of bump-and-run reversal.

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    On top of that we plant a recurring bump-and-run *cycle*: a gentle up-sloping lead-in, a
    steep speculative bump, and then — with ``edge > 0`` — a genuine **downward reversal** once
    the bump rolls over and price breaks back below the lead-in slope. At ``edge = 0`` the bump
    still happens (so the geometry detector still fires), but the post-break drift is zero, so
    a short on the break is a fair coin; at ``edge > 0`` the break is followed by a real fall
    that the short should bank.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    close = np.empty(n_days)
    log_p = np.log(100.0)

    # A repeating bump cycle of fixed period. Within each cycle: a gentle up-drift lead-in, then
    # an accelerating bump that peaks, then a **reversal window** followed by a long flat quiet
    # stretch. The crucial decoupling: the lead-in and bump drifts are *independent of edge* (so
    # the bump always forms and the detector always fires the same way), and the planted edge
    # injects a downward drift ONLY in the reversal window — the ~``reversal`` bars right after
    # the bump peak, exactly where the break entry + its forward horizon land. At edge=0 that
    # window is flat (a fair coin for the short); at edge>0 it falls (a reversal the short banks).
    # The quiet stretch (flat) is long enough that no later cycle's up-drift contaminates a
    # forward return measured just after a break.
    reversal = 70                             # post-peak window carrying the planted reversal
    quiet = 130                               # flat buffer so the next lead-in cannot leak in
    period = lead_in + lead_in + reversal + quiet
    lead_drift = daily_vol * 0.12             # gentle lead-in up-slope (per bar)
    bump_drift = daily_vol * 0.55             # steep speculative bump up-slope (per bar)
    for i in range(n_days):
        phase = i % period
        if phase < lead_in:                   # gentle lead-in
            mu = lead_drift
        elif phase < 2 * lead_in:             # speculative bump (steepening rally)
            mu = bump_drift
        elif phase < 2 * lead_in + reversal:  # reversal window (planted, edge-scaled)
            mu = -edge * bump_drift
        else:                                 # long flat quiet buffer
            mu = 0.0
        eps = rng.normal(0.0, daily_vol)
        log_p += eps + mu
        close[i] = np.exp(log_p)

    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, close.size)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {"edge": edge, "annual_vol": annual_vol, "lead_in": lead_in,
             "n_days": n_days, "seed": seed}
    return bars, truth


# --------------------------------------------------------------------------- #
# Real tape — Yahoo daily, cache-first
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_1d.parquet")


def load_real(
    ticker: str = "SPY",
    start: str = "2005-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    allow_fetch: bool = True,
) -> pd.DataFrame:
    """Real daily OHLC for ``ticker``; **cache-first** (network only on a cache miss).

    Reads a cached parquet if present. Otherwise — and only if ``allow_fetch`` — downloads
    from yfinance (with a couple of retries + back-off on rate limits) and caches the parquet,
    so every subsequent call is fully offline.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path):
        bars = pd.read_parquet(path)
    elif allow_fetch:
        bars = _download(ticker, start, end)
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)
    else:
        raise FileNotFoundError(
            f"No cached daily tape for {ticker} at {path}. "
            f"Call load_real({ticker!r}) once (network) to populate the cache."
        )

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars[["open", "high", "low", "close"]]


def _download(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf  # lazy: only on a real cache miss

    last_err = None
    for attempt in range(3):
        try:
            raw = yf.download(ticker, start=start, end=end, interval="1d",
                              auto_adjust=True, progress=False)
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                bars = raw.rename(columns=str.lower)[["open", "high", "low", "close"]]
                bars.index.name = "date"
                return bars
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"yfinance returned no daily bars for {ticker}: {last_err}")


def have_real(tickers: list[str] | None = None, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every cached parquet for ``tickers`` is present (offline-safe check)."""
    tickers = tickers or DEFAULT_TICKERS
    return all(os.path.exists(_cache_path(t, cache_dir)) for t in tickers)


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
