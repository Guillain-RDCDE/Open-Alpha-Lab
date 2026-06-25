"""Data layer for Study 484 (Vertical-Horizontal-Filter).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**
  built specifically for the VHF gate. The VHF claim is: *momentum works when the market is
  trending (high VHF) and fails when it is ranging (low VHF)*. So we plant exactly that
  regime structure: the tape switches between **trending blocks** (momentum persists — an up
  day tends to be followed by more up days) and **ranging blocks** (the path churns, returns
  mean-revert). With ``edge > 0`` the persistence inside trending blocks is real, and — because
  trending blocks are *also* where the VHF reads high — a VHF-gated momentum entry harvests a
  real edge that the ungated entry dilutes with the ranging blocks. With ``edge = 0`` every day
  is a fair coin regardless of regime, so the gate buys nothing. This is the positive control:
  a gate that cannot bank the planted regime-conditional momentum proves nothing by finding
  nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the VHF and the
momentum signal are both read on the close of *t* (rolling windows that end at *t*), and the
trade is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs a VHF/momentum proponent draws on: the broad tape, big-cap tech, small
# caps, the Dow, and gold. Daily, liquid, long history.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 4000,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    vhf_n: int = 28,
    block: int = 40,
    start: str = "2010-01-04",
    seed: int = 484,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with *known* regime-conditional momentum.

    The tape is built in blocks of ``block`` days that alternate between a **trending** regime
    and a **ranging** regime. Baseline log-returns are i.i.d. Gaussian with daily sigma
    ``annual_vol/sqrt(252)``. On top of that, with ``edge > 0``:

      * a **trending** block carries a persistent **directional drift** (a random sign, fixed for
        the whole block) of magnitude ``edge * daily_vol`` — so the path travels far in one
        direction (VHF reads high) *and* the next few weeks keep going that way (forward momentum
        is real). A momentum entry made here is followed by a genuine continuation.
      * a **ranging** block carries **mean reversion** (yesterday's deviation is pulled back) and
        no net drift — the path churns in place (VHF reads low), and a momentum entry here fades.

    So a *high-VHF gate* on a momentum entry selectively enters during the trending blocks where
    the continuation is real, while the ungated momentum entry is diluted by the (forward-fading)
    ranging blocks. At ``edge = 0`` every block is a pure martingale and the gate buys nothing.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    # one persistent drift sign per trending block (deterministic given the seed)
    n_blocks = n_days // block + 2
    block_sign = rng.choice([-1.0, 1.0], size=n_blocks)

    close = np.empty(n_days)
    log_p = np.log(100.0)
    prev_ret = 0.0
    for i in range(n_days):
        b = i // block
        trending = b % 2 == 0      # even blocks trend, odd blocks range
        eps = rng.normal(0.0, daily_vol)
        if edge > 0.0:
            if trending:
                ret = eps + edge * daily_vol * block_sign[b]    # persistent directional drift
            else:
                ret = eps - edge * prev_ret                     # mean reversion (churn, no drift)
        else:
            ret = eps
        log_p += ret
        close[i] = np.exp(log_p)
        prev_ret = ret

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
    truth = {"edge": edge, "annual_vol": annual_vol, "vhf_n": vhf_n,
             "block": block, "n_days": n_days, "seed": seed}
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
