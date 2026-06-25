"""Data layer for Study 485 (STARC Bands).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  A STARC band is an ATR envelope around a short SMA: ``lower = SMA − k·ATR``. The believers'
  claim is that when the close pierces *below the lower band it reverts upward* toward the SMA.
  We plant exactly that: with ``edge > 0`` the path is pulled back toward the rolling SMA
  whenever the close strays below ``SMA − k·ATR`` (and symmetrically faded above the upper
  band), so a lower-band-pierce entry harvests a real bounce; with ``edge = 0`` the log-return
  series is a pure random walk and the pierce is a fair coin. This is the positive control — a
  harness that cannot bank the planted bounce proves nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the SMA and ATR
that define the bands use only data through bar *t*, the pierce is detected on the close of
*t*, and the trade is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs STARC-band proponents draw on: the broad tape, big-cap tech, small caps,
# and a couple of cross-asset charts. Daily, liquid, long history.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    sma_n: int = 6,
    atr_n: int = 15,
    k_atr: float = 2.0,
    start: str = "2010-01-04",
    seed: int = 485,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of STARC-band mean reversion.

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    On top of that we plant a STARC-respecting force: we keep a rolling short SMA and a rolling
    range-based half-width (a proxy ATR). Whenever the close pierces *below* ``SMA − k·ATR`` we
    add a small upward pull back toward the SMA, proportional to ``edge`` (and a symmetric
    downward fade above ``SMA + k·ATR``). At ``edge = 0`` the tape is a pure martingale and a
    lower-band pierce is a fair coin; at ``edge > 0`` a pierce is followed by a real reversion
    that the detector should bank.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    close = np.empty(n_days)
    open_ = np.empty(n_days)
    hi = np.empty(n_days)
    lo = np.empty(n_days)
    log_p = np.log(100.0)
    closes: list[float] = []
    tr_ema = daily_vol * 100.0       # running Wilder-style ATR proxy (price units)
    prev_close = 100.0
    revert_left = 0                  # bars of post-pierce reversion still owed (the planted edge)

    # Symmetric fat-tailed innovations (a Gaussian/jump mixture) so closes frequently overshoot
    # a 2-ATR band — without occasional big moves a constant-vol Gaussian walk almost never
    # pierces SMA-2*ATR. The jump is symmetric and zero-mean, so at edge=0 a lower-band pierce
    # is a genuine fair coin (no asymmetry sneaking in through the back door).
    jump_p, jump_scale = 0.06, daily_vol * 4.0
    # Per-bar Jensen correction so the *arithmetic* expected one-bar return is ~0 (a symmetric
    # log walk otherwise drifts up in price terms, which would bias the edge=0 control upward).
    var_bar = daily_vol ** 2 + jump_p * jump_scale ** 2
    drift_adj = -0.5 * var_bar

    def _innov():
        z = rng.normal(0.0, daily_vol)
        if rng.random() < jump_p:                     # symmetric jump on a fraction of bars
            z += rng.choice([-1.0, 1.0]) * abs(rng.normal(0.0, jump_scale))
        return z + drift_adj

    for i in range(n_days):
        # --- planted force: a persistent pull toward the SMA after a fresh LOWER pierce -----
        pull = 0.0
        sma = float(np.mean(closes[-sma_n:])) if len(closes) >= sma_n else np.exp(log_p)
        if edge > 0.0 and revert_left > 0:
            pull = edge * np.log(sma / np.exp(log_p))   # ease back UP toward the moving average
            revert_left -= 1

        eps = _innov()
        log_p += eps + pull
        p = np.exp(log_p)

        o = prev_close
        wick = abs(rng.normal(0.0, daily_vol * 0.5)) * p
        h = max(o, p) + wick
        lw = min(o, p) - wick

        close[i] = p; open_[i] = o; hi[i] = h; lo[i] = lw

        # --- update the causal ATR proxy and arm the next reversion on a fresh pierce ------
        tr = max(h - lw, abs(h - prev_close), abs(lw - prev_close))
        tr_ema += (tr - tr_ema) / atr_n
        if edge > 0.0 and len(closes) >= sma_n:
            lower = sma - k_atr * tr_ema
            if p < lower and revert_left == 0:        # the exact event the rule trades
                revert_left = 6                       # plant a multi-day reversion to be banked
        prev_close = p
        closes.append(p)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {"edge": edge, "annual_vol": annual_vol, "sma_n": sma_n, "atr_n": atr_n,
             "k_atr": k_atr, "n_days": n_days, "seed": seed}
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
