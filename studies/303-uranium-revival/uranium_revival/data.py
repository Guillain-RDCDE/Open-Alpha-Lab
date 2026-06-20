"""Data layer for Study 303 (Uranium-Revival).

Two tapes, one shape (a tz-naive daily OHLCV frame):

- ``synthetic_daily`` — a *deterministic, offline* regime-switching generator. A single
  ``regime`` knob picks what the tape *is*, so a test can assert the trend rule wins only
  when there is a genuine, persistent trend to ride — and not on a hype rocket or a coin:

    * ``regime="trend"``  — *persistent* multi-month bull and bear runs (a Markov
      regime with serial correlation in direction). This is the structure a
      trend-following rule is *built* to harvest: stay long through the bulls, step
      aside through the bears. The *positive control*: the harness must bank a timing
      edge over buy-and-hold here, or it proves nothing by finding nothing.
    * ``regime="random"`` — a pure random walk, no drift, no direction-persistence. The
      *null*: a fair coin; a trend rule cannot reliably beat buy-and-hold here.
    * ``regime="hype"``   — a boom-bust bubble: a fast parabolic run-up followed by a
      sharp crash that gives most of it back, wrapped in high vol. This is the "uranium
      rocket" caricature — a trend rule can *look* great in-sample if it happens to ride
      the boom, but the regime carries no exploitable persistence once you account for
      the crash. The realistic-pessimist control.

  Uranium-miner ETFs (URA from 2010, URNM from 2019) are *single, highly volatile,
  thematic* baskets — exactly the kind of tape where a trend rule's edge is most easily
  confused with one lucky boom. That is the whole point of the study.

- ``load_real`` / ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``), cache-only
  by default, so the test-suite and the reproducible core never touch the network. A
  cache-first ``load_real`` tries the study's local ``_cache`` parquet first, then the
  shared ``quantlab.data`` cache, and raises cleanly on a miss (offline-safe).

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (one explicit
``shift``: a signal known at the close of *t* earns the return of *t+1*).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Trading days per year approximation.
TRADING_DAYS_PER_YEAR = 252

# The thematic uranium-miner ETFs the claim is about.
URANIUM_TICKERS = ("URA", "URNM")


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 2500,
    regime: str = "trend",
    daily_vol: float = 0.025,
    drift: float = 0.0008,
    start: str = "2011-01-03",
    seed: int = 303,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape whose *character* is set by ``regime``.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters and a
    ``trendable`` flag — the ground truth a test checks the engine against.

    Regimes
    -------
    ``"trend"``
        A two-state Markov chain over direction: long stretches of positive drift (bull)
        and long stretches of negative drift (bear), each persisting ~80 trading days on
        average. A trend rule can ride the bulls and dodge the bears, so timing *beats*
        buy-and-hold. ``trendable=True``.
    ``"random"``
        Pure random walk (drift forced to ~0). A fair coin. ``trendable=False``.
    ``"hype"``
        A boom-bust: the first ~45% of the sample is a parabolic ramp (rising drift),
        then a sharp crash phase gives most of it back, then it drifts sideways. High
        vol throughout. A trend rule may ride the boom but the regime has no durable
        edge once the crash is in-sample. ``trendable=False``.

    The volatility is deliberately high (``daily_vol`` ~2.5%/day) to mimic a single
    thematic miner ETF rather than a broad index.
    """
    if regime not in ("trend", "random", "hype"):
        raise ValueError(f"unknown regime {regime!r}")

    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days)

    eps = rng.normal(0.0, daily_vol, n_days)

    if regime == "trend":
        # Two-state Markov chain over direction with ~100-day mean dwell time, so
        # bull/bear runs persist long enough for a trend rule to harvest them.
        p_switch = 1.0 / 100.0
        amp = 0.0025  # per-day drift magnitude inside a bull/bear run
        state = 1  # +1 bull, -1 bear
        mu = np.empty(n_days)
        for i in range(n_days):
            if rng.random() < p_switch:
                state = -state
            mu[i] = state * amp
        trendable = True
    elif regime == "random":
        mu = np.zeros(n_days)
        trendable = False
    else:  # hype — a boom-bust bubble
        mu = np.zeros(n_days)
        boom_end = int(0.45 * n_days)
        crash_end = int(0.60 * n_days)
        # Boom: drift ramps up to a parabolic peak.
        ramp = np.linspace(0.5, 3.0, boom_end)
        mu[:boom_end] = drift * 4.0 * ramp
        # Crash: a sharp negative drift gives most of the boom back.
        boom_total = float(mu[:boom_end].sum())
        crash_len = crash_end - boom_end
        mu[boom_end:crash_end] = -(boom_total * 0.85) / crash_len
        # After: sideways drift ~0 (already zero).
        trendable = False

    log_ret = mu + eps
    close = 10.0 * np.exp(np.cumsum(log_ret))

    open_ = np.empty_like(close)
    open_[0] = 10.0
    open_[1:] = close[:-1] * np.exp(rng.normal(0.0, daily_vol * 0.3, n_days - 1))
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(1_000_000, 30_000_000, n_days).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {
        "regime": regime,
        "trendable": trendable,
        "daily_vol": daily_vol,
        "drift": drift,
        "n_days": n_days,
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str = DEFAULT_CACHE) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_daily.parquet")


def fetch_daily(
    ticker: str,
    start: str = "2010-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as a
    parquet under ``_cache/``). URA lists 2010; URNM 2019 — both shorter than a broad
    index, which is itself part of the story (a thin sample for a momentum claim).
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call fetch_daily({ticker!r}, fetch=True) once to populate the cache."
            )
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            ticker, start=start, interval="1d", auto_adjust=True, progress=False
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    return bars


def load_real(ticker: str, cache_dir: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Cache-first real loader: study ``_cache`` parquet → shared ``quantlab.data`` → raise.

    Never touches the network. Returns a tz-naive daily OHLCV frame. Raises
    ``FileNotFoundError`` if no cache is available anywhere (so callers can fall back to
    the synthetic tape, offline).
    """
    # 1) study-local cache.
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path):
        bars = pd.read_parquet(path)
        if bars.index.tz is not None:
            bars.index = bars.index.tz_localize(None)
        return bars

    # 2) shared quantlab cache (graceful: only if importable and a cache hit exists).
    try:
        from quantlab import data as qld  # noqa: WPS433

        bars = qld.fetch(ticker, mode="adjusted", use_cache=True)
        bars = bars.rename(columns=str.lower)
        keep = [c for c in ("open", "high", "low", "close", "volume") if c in bars.columns]
        bars = bars[keep]
        if bars.index.tz is not None:
            bars.index = bars.index.tz_localize(None)
        bars.index.name = "date"
        return bars
    except Exception:  # noqa: BLE001 — any failure means "no real tape", fall through
        pass

    raise FileNotFoundError(
        f"No cached real tape for {ticker} (looked in {path} and the shared quantlab "
        "cache). Populate one with fetch_daily(fetch=True) or run on the synthetic tape."
    )


def have_real_cache(tickers=URANIUM_TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every ticker has a study-local cached parquet (offline-detectable)."""
    return all(os.path.exists(_cache_path(t, cache_dir)) for t in tickers)


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
