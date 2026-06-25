"""Data layer for Study 499 (Trendline-Break).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  An uptrend line is least-squares-fit through the recent confirmed swing lows; a break is a
  close below it. Because the break fires at a fresh local low, the genuinely *bankable*
  conditional effect at that point is a **post-break bounce** (mean reversion up). We plant
  exactly that: with ``edge > 0`` the path gets a real upward push over the bars the trade
  spans right after it pierces the fitted rising line, so the break — read as a **fade**
  (long, ``short=False``) — banks a genuine bounce; with ``edge = 0`` the log-return series is
  a pure random walk and the break is a fair coin. This is the positive control — a harness
  that cannot bank the planted bounce proves nothing by finding nothing on the real tape. The
  folklore's *bearish* reading (``short=True``) is the one tested on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the trendline is
fit on swing lows that are *confirmed* (a fractal needs ``k`` bars on each side), a break is
detected on the close of *t*, and the trade is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs trendline-break proponents draw on: the broad tape, big-cap tech, small
# caps, and a couple of cross-asset charts. Daily, liquid, long history.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    pivot_k: int = 10,
    start: str = "2010-01-04",
    seed: int = 499,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known*, bankable trendline-break effect.

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    The planted effect is driven off the **same object the detector uses**: an online,
    lag-confirmed swing-low OLS trendline. When a close first pierces *below* that fitted rising
    line (the mechanical "support broke" event), with ``edge > 0`` we schedule a short bounded
    upward push over the bars the trade actually spans (onset at the entry bar, decaying over
    ``decay`` days) — a **real post-break bounce** (mean reversion off the broken support).

    Why a *bounce*, not a "continues down" drop? The break fires at a fresh local low (price
    dipping beneath rising swing-low support); on this geometry a genuinely *bankable*
    conditional effect at that point is the reversion **up** — which the rule banks if read as a
    **fade** (long the breakdown, ``short=False``). That is the honest positive control: at
    ``edge = 0`` the tape is a pure martingale (a break is a fair coin → t ≈ 0, no false
    positive); at ``edge > 0`` the planted bounce lights up the fade with a high win-rate. The
    folklore's *bearish* reading (``short=True``) is then tested on the real tape, where it must
    beat a drift-matched random baseline to count.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    close = np.empty(n_days)
    log_close = np.empty(n_days)
    log_p = np.log(100.0)
    # A gently *rising* tape so genuine uptrends and rising swing-low support lines form and
    # price normally holds above them.
    base_drift = daily_vol * 0.22
    onset = 1                            # push begins at the entry bar (t+1 of the break)
    decay = 10                           # days the planted post-break bounce persists
    push_mult = 1.5                      # strength of the planted bounce (per unit edge)
    k = pivot_k
    sched = np.zeros(n_days)             # exogenous scheduled push per bar
    # online confirmed swing lows (position, log-price) and a "currently below the line" flag
    low_pos: list[int] = []
    low_val: list[float] = []
    below_prev = False

    def _line_at(t: int) -> float:
        """OLS rising trendline through the 3 latest confirmed lows, evaluated at bar t; or nan."""
        if len(low_pos) < 3:
            return float("nan")
        xs = np.array(low_pos[-3:], dtype=float)
        ys = np.array(low_val[-3:], dtype=float)
        xm = xs.mean()
        dx = xs - xm
        den = float(dx @ dx)
        if den == 0.0:
            return float("nan")
        slope = float(dx @ (ys - ys.mean())) / den
        if slope <= 0:
            return float("nan")
        intercept = float(ys.mean()) - slope * xm
        return slope * t + intercept

    for i in range(n_days):
        # confirm a swing low at bar i-k (needs k strictly-higher bars each side); online & lagged
        c_idx = i - k
        if c_idx >= k:
            seg = log_close[c_idx - k:c_idx + k + 1]
            cv = log_close[c_idx]
            if cv == seg.min() and (seg[:k] > cv).all() and (seg[k + 1:] > cv).all():
                low_pos.append(c_idx)
                low_val.append(float(cv))

        line_t = _line_at(i)
        is_below = np.isfinite(line_t) and (log_p < line_t)
        fresh_break = is_below and not below_prev
        below_prev = is_below

        if edge > 0.0 and fresh_break:
            for j in range(i + onset, min(i + onset + decay, n_days)):
                sched[j] = edge * daily_vol * push_mult   # planted UP bounce off the break

        pull = float(sched[i])
        # base up-drift applies on quiet bars; suspended while the planted bounce is active so
        # the effect is clean. At edge=0 the tape is a pure martingale (fair coin).
        drift = 0.0 if (edge == 0.0 or pull != 0.0) else base_drift
        eps = rng.normal(0.0, daily_vol)
        log_p += eps + drift + pull
        close[i] = np.exp(log_p)
        log_close[i] = log_p

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
    truth = {"edge": edge, "annual_vol": annual_vol, "pivot_k": pivot_k,
             "n_days": n_days, "seed": seed, "planted": "post-break bounce (fade)",
             "bank_with_short": False}
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
