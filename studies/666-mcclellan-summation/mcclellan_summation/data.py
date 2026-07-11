"""Data layer for Study 666 (McClellan Summation Index).

Three things live here, all sharing one shape (a tz-naive daily OHLC frame,
calendar-date indexed):

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads
  a cached parquet if present and only touches the network on an explicit cache miss
  (with a short back-off + retry), then caches the parquet so re-runs are offline.

- ``load_breadth`` — the breadth proxy the Summation Index is built on. The true
  McClellan family is built on NYSE *exchange* advance/decline data (thousands of
  issues). We have no offline exchange-breadth feed, so we approximate it with a
  basket of liquid US sector ETFs: each day we count how many basket members closed
  up vs down — the *net advances* — exactly the input the oscillator (and hence the
  Summation Index, its running integral) wants, just on a coarse basket instead of
  the full exchange. **This is a proxy and it caps the test** — stated loudly in the
  docs, exactly as sibling study 491 (the oscillator itself) states it.

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge
  knob** specific to the Summation Index's claim: a regime timer (long while the
  Summation Index sits above zero) should out-earn a coin-flip regime when the knob
  is on, and be a fair coin when it's off. A detector that cannot bank a planted
  regime effect proves nothing by finding none on the real tape.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the
Summation Index is read on the close of *t*, any cross/threshold event is confirmed
at *t*, and any trade/regime switch is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# The traded instrument (the "index" the regime timer is deployed on).
TRADED = "SPY"

# The breadth-proxy basket — the 9 SPDR sector ETFs plus SPY (a coarse stand-in for
# true NYSE exchange breadth; identical basket to sibling 491-mcclellan-oscillator,
# so the two studies' breadth inputs are directly comparable). Fallback to the
# desk's standing 5-ticker cache if the sector ETFs can't be fetched.
BREADTH_BASKET = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]
BREADTH_FALLBACK = ["SPY", "QQQ", "IWM", "DIA", "GLD"]
DEFAULT_TICKERS = BREADTH_BASKET


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    start: str = "2010-01-04",
    seed: int = 666,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily tape with a *known* amount of Summation-regime forecasting.

    Returns a frame with the usual OHLC columns **plus** a ``net_adv`` column (the daily
    net advances the McClellan family is built on). The market close is a random walk in
    log-returns with daily sigma ``annual_vol/sqrt(252)``. A latent breadth momentum
    ``b_t`` (a slow AR(1) process) drives ``net_adv``.

    With ``edge > 0`` the *future* drift is pulled up whenever the causal Summation
    Index (cumsum of EMA19-EMA39 of ``net_adv``) sits **above zero** and pulled down
    when it sits below — i.e. the Summation *regime* genuinely forecasts the market's
    near-term drift. With ``edge = 0`` the breadth series and the market are
    independent and the regime is a fair coin. ``truth`` records the planted params.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    # latent breadth momentum: a mean-reverting AR(1) that swings positive/negative
    b = np.zeros(n_days)
    phi = 0.97  # slower than 491's oscillator control — the Summation Index has memory
    for i in range(1, n_days):
        b[i] = phi * b[i - 1] + rng.normal(0.0, 1.0)
    basket = 10
    net_adv = np.clip(np.round(b * 2.0 + rng.normal(0.0, 1.5, n_days)), -basket, basket)

    # Causal McClellan oscillator + its running Summation Index, to plant the regime
    osc = _ema(net_adv, 19) - _ema(net_adv, 39)
    summ = np.cumsum(osc)
    regime_bull = summ > 0.0

    close = np.empty(n_days)
    log_p = np.log(100.0)
    for i in range(n_days):
        pull = (edge * daily_vol * 0.5) if (edge > 0.0 and regime_bull[i]) else 0.0
        pull -= (edge * daily_vol * 0.5) if (edge > 0.0 and not regime_bull[i]) else 0.0
        eps = rng.normal(0.0, daily_vol)
        log_p += eps + pull
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
    ticker: str = TRADED,
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


def fetch_all(tickers: list[str] | None = None, start: str = "2005-01-01",
             end: str | None = None, cache_dir: str = DEFAULT_CACHE) -> None:
    """Populate the local cache for every basket ticker. Network; run once."""
    for t in (tickers or DEFAULT_TICKERS):
        load_real(t, start=start, end=end, cache_dir=cache_dir, allow_fetch=True)


def breadth_members(cache_dir: str = DEFAULT_CACHE, allow_fetch: bool = False) -> list[str]:
    """Which breadth-basket tickers are actually available (cache-first).

    Prefers the 9-sector-ETF + SPY basket; falls back to the 5 default tapes if the
    sector ETFs aren't cached (so the study can still run fully offline).
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
    """Daily **net advances** = (# basket members up) - (# down), the McClellan input.

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
