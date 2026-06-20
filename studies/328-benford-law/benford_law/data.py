"""Data layer for Study 328 (Benford-Law).

Benford's Law says the leading (most-significant) digit ``d`` of a "natural" set of
numbers appears with probability ``log10(1 + 1/d)`` — ~30.1% ones, ~17.6% twos, …,
~4.6% nines. It holds for quantities that span *several orders of magnitude on a
roughly log-uniform scale*. A stock price that drifts from \\$10 to \\$1000 over decades
is exactly that kind of quantity; a daily *return* (±a few percent, tightly clustered)
is not. The whole study turns on that distinction, so the data layer hands us two
synthetic tapes with a *known* leading-digit law and a cache-first real loader.

Two synthetic generators, one shape (a tz-naive daily price Series):

- ``synthetic_benford`` — a geometric random walk whose drift makes the price wander
  across many orders of magnitude. Over a long enough span its leading digits are
  Benford by construction (the log of a wide-range positive series is ~uniform mod 1).
  This is the **positive control**: a tape we *know* should conform.
- ``synthetic_nonbenford`` — a mean-reverting price pinned to a narrow band (e.g.
  oscillating around \\$50, range ~[40, 60]). It never spans an order of magnitude, so
  its leading digits are dominated by 4s and 5s — emphatically **not** Benford. This is
  the **null/anti-control**: a tape we *know* should fail the test, so a harness that
  flags it proves the test discriminates.

Plus ``load_real`` — the real Yahoo daily tape via the shared ``quantlab.data`` cache
(or this study's local ``_cache/``), cache-only by default so the offline core and the
test-suite never touch the network. Graceful fallback to the synthetic tape is the
caller's job (see the notebooks); here a cache miss raises, loudly.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))
# The desk's shared parquet cache (repo-root/_cache), where quantlab.data writes.
SHARED_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Synthetic tapes — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_benford(
    n_days: int = 6000,
    drift: float = 0.0011,
    daily_vol: float = 0.013,
    start_price: float = 5.0,
    seed: int = 328,
) -> tuple[pd.Series, dict]:
    """A geometric random walk that wanders across orders of magnitude → Benford.

    Log returns are i.i.d. normal with mean ``drift`` and sd ``daily_vol``; the price
    is ``start_price * exp(cumsum(log_ret))``. With a small positive drift over several
    thousand sessions the price climbs from ~\\$10 through \\$100 to ~\\$1000+, so its
    *log* is approximately uniform over an integer number of decades, which is exactly
    the condition for Benford's first-digit law to hold.

    Returns ``(price, truth)`` where ``truth`` records the planted parameters and the
    flag ``benford=True``.
    """
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(drift, daily_vol, n_days)
    price = start_price * np.exp(np.cumsum(log_ret))
    idx = _decorative_index(n_days, start="2000-01-03")
    s = pd.Series(price, index=idx, name="close")
    truth = {
        "benford": True,
        "drift": drift,
        "daily_vol": daily_vol,
        "n_days": n_days,
        "seed": seed,
        "log_range_decades": float(np.log10(price.max() / price.min())),
    }
    return s, truth


def synthetic_nonbenford(
    n_days: int = 6000,
    center: float = 50.0,
    half_band: float = 10.0,
    reversion: float = 0.05,
    daily_vol: float = 0.012,
    seed: int = 328,
) -> tuple[pd.Series, dict]:
    """A range-bound, mean-reverting price → NOT Benford (digits cluster on 4s/5s).

    Log-price follows an Ornstein-Uhlenbeck-style pull toward ``log(center)`` with a
    reflecting feel from the strong reversion, so the price oscillates in roughly
    ``[center - half_band, center + half_band]`` and never spans an order of magnitude.
    A price living in [40, 60] has leading digits that are almost all 4s and 5s — the
    opposite of Benford. This is the anti-control: the test *must* flag it.

    Returns ``(price, truth)`` with ``truth['benford'] = False``.
    """
    rng = np.random.default_rng(seed + 1)
    log_center = np.log(center)
    log_p = np.empty(n_days)
    x = log_center
    for i in range(n_days):
        x = x + reversion * (log_center - x) + rng.normal(0.0, daily_vol)
        log_p[i] = x
    price = np.exp(log_p)
    # Hard-clip into the band so the "narrow range" property is guaranteed offline.
    price = np.clip(price, center - half_band, center + half_band)
    idx = _decorative_index(n_days, start="2000-01-03")
    s = pd.Series(price, index=idx, name="close")
    truth = {
        "benford": False,
        "center": center,
        "half_band": half_band,
        "reversion": reversion,
        "n_days": n_days,
        "seed": seed,
        "log_range_decades": float(np.log10(price.max() / price.min())),
    }
    return s, truth


def _decorative_index(n: int, start: str = "2000-01-03") -> pd.DatetimeIndex:
    """A purely decorative daily index of length ``n``.

    Built from a bounded ``period_range`` of calendar days (NOT
    ``date_range(periods=BIG, freq=...)``) so even a multi-thousand-day synthetic span
    stays far inside the ns-Timestamp range (n days << 290 years). The dates are a
    label only — nothing in the Benford analysis depends on the calendar.
    """
    periods = pd.period_range(start=start, periods=n, freq="D")
    return periods.to_timestamp()


# ---------------------------------------------------------------------------
# Real tape — cache-first, network only on explicit opt-in
# ---------------------------------------------------------------------------
def _local_path(ticker: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(LOCAL_CACHE, f"{safe}_close.parquet")


def _shared_path(ticker: str, mode: str = "split_only") -> str:
    return os.path.join(SHARED_CACHE, f"{ticker}_{mode}.parquet")


def cache_available(ticker: str = "SPY", mode: str = "split_only") -> bool:
    """True if a usable cached tape exists locally or in the shared cache (no network)."""
    return os.path.exists(_local_path(ticker)) or os.path.exists(_shared_path(ticker, mode))


def load_real(
    ticker: str = "SPY",
    mode: str = "split_only",
    fetch: bool = False,
) -> pd.Series:
    """Real daily close for ``ticker`` as a tz-naive Series; cache-first, offline-safe.

    Resolution order:

    1. This study's local ``_cache/<ticker>_close.parquet`` (a one-column close tape).
    2. The desk's shared ``_cache/<ticker>_<mode>.parquet`` (``quantlab.data`` OHLC).
    3. Only if ``fetch=True``: go to the network via ``quantlab.data.fetch`` and cache
       the close locally.

    ``mode`` is ``'split_only'`` by default — Benford cares about the *level*, and a
    split chops the price by an integer, which mechanically rewrites leading digits, so
    we explicitly use the split-adjusted level (the price a holder actually saw between
    splits). A cache miss with ``fetch=False`` raises ``FileNotFoundError``.
    """
    lp = _local_path(ticker)
    if os.path.exists(lp):
        s = pd.read_parquet(lp)["close"]
    else:
        sp = _shared_path(ticker, mode)
        if os.path.exists(sp):
            df = pd.read_parquet(sp)
            col = "Close" if "Close" in df.columns else df.columns[-1]
            s = df[col].rename("close")
        elif fetch:
            from quantlab import data as qdata  # lazy: only on an explicit network hit

            df = qdata.fetch(ticker, mode=mode)
            s = df["Close"].rename("close")
            os.makedirs(LOCAL_CACHE, exist_ok=True)
            s.to_frame().to_parquet(lp)
        else:
            raise FileNotFoundError(
                f"No cached tape for {ticker} (looked in {lp} and {sp}). "
                f"Call load_real({ticker!r}, fetch=True) once to populate the cache."
            )
    s = s.dropna()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s.astype(float)


def fingerprint(s: pd.Series) -> str:
    """A short content fingerprint of a price tape, for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(s.to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]
