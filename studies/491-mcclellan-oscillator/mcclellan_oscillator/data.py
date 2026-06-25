"""Data layer for Study 491 (McClellan Oscillator).

Three things live here, all sharing one shape (a tz-naive daily OHLC frame,
calendar-date indexed):

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads
  a cached parquet if present and only touches the network on an explicit cache miss
  (with a short back-off + retry), then caches the parquet so re-runs are offline.

- ``load_breadth_basket`` — the breadth proxy. The true McClellan oscillator is built
  on NYSE *exchange* advance/decline data (thousands of issues). We have no offline
  exchange-breadth feed, so we approximate it with a small basket of liquid US ETFs:
  each day we count how many basket members closed up vs down — the *net advances* —
  exactly the input the oscillator wants, just on a coarse basket instead of the full
  exchange. **This is a proxy and it caps the test**: a 5–10 name basket is a noisy
  estimate of true exchange breadth, so a genuine edge could be blurred. We state that
  loudly in the docs; the Signal test (breadth-cross vs random-day entries on SPY) is
  unaffected by the proxy's coarseness in the sense that *if* the proxy carried a real
  signal it would still have to beat random — coarseness only makes a false positive
  less likely, never more.

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge
  knob** specific to THIS indicator. It emits a market close *and* a synthetic net-
  breadth series; with ``edge > 0`` an up-cross of the McClellan oscillator from
  negative is genuinely followed by a forward market bounce (the positive control);
  with ``edge = 0`` the breadth series is independent noise and the up-cross is a fair
  coin. A detector that cannot bank the planted effect proves nothing by finding none.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the
oscillator is read on the close of *t*, an up-cross from negative is confirmed at *t*,
and the trade is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# The instruments the study trades / fingerprints. Reused from the desk's standing
# offline cache (SPY QQQ IWM DIA GLD) so the whole study runs with no network.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]

# The breadth-proxy basket. Ideally a spread of sector ETFs (XLK XLF XLE XLV XLI XLY
# XLP XLU XLB) plus SPY; whichever of these is cached is used. To stay 100% offline on
# the desk's standing cache we fall back to the 5 default tapes, which still give a
# valid (coarser) net-advances breadth series.
BREADTH_BASKET = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]
BREADTH_FALLBACK = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    start: str = "2010-01-04",
    seed: int = 491,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily tape with a *known* amount of breadth-momentum forecasting.

    Returns a frame with the usual OHLC columns **plus** a ``net_adv`` column (the daily
    net advances that feed the McClellan oscillator). The market close is a random walk
    in log-returns with daily sigma ``annual_vol/sqrt(252)``. We build a latent breadth
    momentum ``b_t`` (a slow AR(1) process) and set ``net_adv`` = noisy readout of it.

    With ``edge > 0`` the *future* market return is pulled up in proportion to ``edge``
    right after the oscillator (the EMA19-EMA39 of ``net_adv``) crosses up through zero
    from negative — so a McClellan up-cross genuinely forecasts a bounce. With
    ``edge = 0`` the breadth series and the market are independent and the up-cross is a
    fair coin. ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    # latent breadth momentum: a mean-reverting AR(1) that swings positive/negative
    b = np.zeros(n_days)
    phi = 0.93
    for i in range(1, n_days):
        b[i] = phi * b[i - 1] + rng.normal(0.0, 1.0)
    # net advances: a noisy linear readout of the latent breadth (range-ish +/- basket)
    basket = 10
    net_adv = np.clip(np.round(b * 2.0 + rng.normal(0.0, 1.5, n_days)), -basket, basket)

    # McClellan oscillator of the *synthetic* breadth, computed causally to plant the edge
    osc = _ema(net_adv, 19) - _ema(net_adv, 39)
    up_cross = np.zeros(n_days, dtype=bool)
    up_cross[1:] = (osc[1:] > 0) & (osc[:-1] <= 0)

    close = np.empty(n_days)
    log_p = np.log(100.0)
    drift = 0.0  # pure martingale baseline (no free beta in the control)
    pend = 0  # number of remaining days to apply the planted post-cross bounce
    for i in range(n_days):
        if edge > 0.0 and up_cross[i]:
            pend = 20  # plant a 20-day forward bounce after each up-cross
        pull = (edge * daily_vol * 1.2) if (edge > 0.0 and pend > 0) else 0.0
        if pend > 0:
            pend -= 1
        eps = rng.normal(0.0, daily_vol)
        log_p += eps + drift + pull
        close[i] = np.exp(log_p)

    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, close.size)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "net_adv": net_adv},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {"edge": edge, "annual_vol": annual_vol, "n_days": n_days, "seed": seed}
    return bars, truth


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """Causal EMA with pandas' span convention (alpha = 2/(span+1)); pure-numpy output."""
    return pd.Series(np.asarray(x, dtype=float)).ewm(span=span, adjust=False).mean().to_numpy()


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

    Reads a cached parquet if present. Otherwise — and only if ``allow_fetch`` —
    downloads from yfinance (with a couple of retries + back-off on rate limits) and
    caches the parquet, so every subsequent call is fully offline.
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


def breadth_members(cache_dir: str = DEFAULT_CACHE, allow_fetch: bool = False) -> list[str]:
    """Which breadth-basket tickers are actually available (cache-first).

    Prefers the wide sector basket; falls back to the 5 default tapes if the sector
    ETFs aren't cached (so the study is fully offline on the desk's standing cache).
    """
    if allow_fetch:
        return BREADTH_BASKET
    have = [t for t in BREADTH_BASKET if os.path.exists(_cache_path(t, cache_dir))]
    if len(have) >= 3:
        return have
    return [t for t in BREADTH_FALLBACK if os.path.exists(_cache_path(t, cache_dir))]


def load_breadth(
    members: list[str] | None = None,
    cache_dir: str = DEFAULT_CACHE,
    allow_fetch: bool = False,
    asof: str | None = None,
) -> pd.Series:
    """Daily **net advances** = (# basket members up) − (# down), the McClellan input.

    Each member's close is loaded cache-first; we align them on the common calendar,
    take the sign of each day's close-to-close change, and sum across members. The
    result is a daily integer series in ``[-N, +N]`` for an N-name basket — a coarse
    proxy for true NYSE exchange breadth.
    """
    members = members or breadth_members(cache_dir, allow_fetch)
    cols = {}
    for t in members:
        b = load_real(t, cache_dir=cache_dir, allow_fetch=allow_fetch)
        cols[t] = b["close"]
    px = pd.DataFrame(cols).dropna(how="all")
    if asof is not None:
        px = px[px.index <= asof]
    chg = px.diff()
    net = np.sign(chg).sum(axis=1)  # +1 per up name, -1 per down, 0 unchanged
    net = net.iloc[1:]  # drop the first NaN-diff row
    net.name = "net_adv"
    return net


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    col = "close" if "close" in bars.columns else bars.columns[0]
    h = hashlib.sha1(np.ascontiguousarray(bars[col].to_numpy()).tobytes())
    return h.hexdigest()[:12]
