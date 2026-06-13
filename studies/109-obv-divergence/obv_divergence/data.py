"""Data layer for Study 109 (OBV-Divergence).

Two tapes, one shape (a daily OHLCV frame):

- ``synthetic_daily`` — a *deterministic, offline* generator. Two knobs govern
  the planted structure:

  * ``price_momentum`` (float) — daily log-return AR(1) coefficient. At 0 the
    price series is a martingale; positive values add persistence (the condition
    under which an OBV trend-cross might find something).
  * ``vol_lead`` (float in [0, 1]) — the fraction of the day's return that is
    *anticipated* by the volume signal: when ``vol_lead > 0``, OBV divergence
    from price (OBV rising while price falls, or vice versa) genuinely precedes
    a reversal. Set ``vol_lead = 0`` for the strict null (volume is contemporaneous
    noise — no forecast content). This is the study's null in a bottle.

- ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``), cache-only by
  default so the test-suite and reproducible core never touch the network.

No look-ahead is baked in here — that discipline lives in ``strategy.py``
(signals are formed on closes up to *t*; positions entered at *t+1*).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 500,
    price_momentum: float = 0.0,
    vol_lead: float = 0.0,
    daily_vol_bps: float = 100.0,
    start: str = "2020-01-02",
    seed: int = 109,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with tunable price momentum and volume lead.

    Price log-returns follow an AR(1):
        ``r_t = price_momentum * r_{t-1} + eps_t``
    with ``eps_t`` i.i.d. Normal(0, daily_vol_bps * 1e-4).

    Volume is generated as:
        ``v_t = base_vol * exp(vol_scale * |eps_t| + vol_lead * future_sign)``
    where ``future_sign`` is the sign of ``r_{t+1}`` (the next day's return).
    When ``vol_lead > 0``, today's volume is higher before a big move tomorrow,
    making OBV a genuine lead indicator. When ``vol_lead = 0``, volume carries
    no predictive information (the strict null).

    ``price_momentum = 0, vol_lead = 0`` is the double null: no trend and no
    predictive volume structure — neither OBV signal type should work here.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)

    eps = rng.normal(0.0, daily_vol_bps * 1e-4, n_days)

    # AR(1) price returns
    log_ret = np.empty(n_days)
    prev = 0.0
    for i, e in enumerate(eps):
        r = price_momentum * prev + e
        log_ret[i] = r
        prev = r

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]

    # Intrabar high/low: small wick around the open-close range
    wick = np.abs(rng.normal(0.0, daily_vol_bps * 0.3e-4, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick

    # Volume: scales with abs(return); vol_lead adds a preview of next day's sign
    future_sign = np.sign(np.append(log_ret[1:], [0.0]))  # 0 padding at the end
    base_vol = rng.integers(500_000, 5_000_000, n_days).astype(float)
    vol_scale = 5.0  # how much abs(return) drives volume
    volume = base_vol * np.exp(vol_scale * np.abs(log_ret) + vol_lead * np.abs(log_ret) * future_sign)
    volume = np.clip(volume, 100_000, 50_000_000).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": volume},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "price_momentum": price_momentum,
        "vol_lead": vol_lead,
        "daily_vol_bps": daily_vol_bps,
        "n_days": n_days,
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_1d.parquet")


def fetch_daily(
    ticker: str,
    start: str = "2010-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is
    cached as a parquet under ``_cache/``). With ``end=None`` the frame runs
    to today. Returns a tz-naive index at date precision.
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
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        if bars.index.tz is not None:
            bars.index = bars.index.tz_localize(None)
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
