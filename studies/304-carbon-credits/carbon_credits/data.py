"""Data layer for Study 304 (Carbon-Credits).

Two tapes, one shape — a tz-naive daily **total-return price** frame (one column per
asset; ``auto_adjust=True`` so distributions are folded in):

- ``synthetic_carbon`` — a *deterministic, offline* generator. Daily log returns follow a
  trend-stationary AR(1): ``r_t = mu/252 + momentum * (r_{t-1} - mu/252) + eps_t``. Two
  knobs plant exactly the two things the study can possibly harvest:

    * ``mu`` — the *drift*. ``mu > 0`` is a genuine rewarded buy-and-hold (the positive
      control for the "green bet pays" claim); ``mu = 0`` is the null — a driftless tape
      where holding earns nothing but vol.
    * ``momentum`` — short-horizon return *persistence*. ``momentum > 0`` makes trends
      autocorrelate, so a trend-following overlay can add value over buy-and-hold;
      ``momentum = 0`` is a martingale where any timing rule is a coin and must *lose* to
      buy-and-hold after costs. This is the timing-overlay null in a bottle.

  A second column ``CASH`` is a near-zero-vol money-market proxy so the engine can race
  **excess-of-cash to excess-of-cash** honestly.

- ``load_real`` — the real daily **total-return** tape for KRBN (KraneShares Global Carbon
  Strategy ETF, the headline carbon-allowance vehicle) plus a cash proxy (BIL, 1-3m
  T-bills). Cache-first: it reads a per-ticker parquet under this study's ``_cache/`` and
  falls back to ``quantlab.data.fetch(mode='total_return')`` on a miss. KRBN listed
  **2020-07-30**, which bounds the joint window — a short tape, named honestly everywhere.

No look-ahead is baked in here — the one-bar execution lag lives in ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

TRADING_DAYS_PER_YEAR = 252

# KRBN (the headline carbon-allowance ETF) listed 2020-07-30 — bounds the real window.
KRBN_INCEPTION = "2020-07-30"


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core (drift + momentum controls)
# ---------------------------------------------------------------------------
def synthetic_carbon(
    n_days: int = 1500,
    mu: float = 0.10,
    momentum: float = 0.0,
    daily_vol: float = 0.018,
    cash_rate: float = 0.02,
    start: str = "2020-07-30",
    seed: int = 304,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily total-return frame with a known drift and persistence.

    Log returns follow a trend-stationary AR(1) around the drift:

        ``r_t = mu/252 + momentum * (r_{t-1} - mu/252) + eps_t``

    with ``eps_t`` i.i.d. normal of standard deviation ``daily_vol``.

    - ``mu > 0``        → a genuinely rewarded hold (the "green bet pays" positive control).
    - ``mu = 0``        → the null: holding earns nothing but volatility.
    - ``momentum > 0``  → return persistence a trend overlay can exploit (timing positive
                          control).
    - ``momentum = 0``  → a martingale; any timing rule is a coin and loses to B&H on cost.

    Returns ``(frame, truth)`` where ``frame`` has columns ``['KRBN', 'CASH']`` (price
    levels starting at 100) and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days)
    momentum = float(np.clip(momentum, -0.99, 0.99))

    drift = mu / TRADING_DAYS_PER_YEAR
    log_ret = np.empty(n_days)
    prev = 0.0  # deviation of the previous return from its mean
    for i in range(n_days):
        eps = rng.normal(0.0, daily_vol)
        dev = momentum * prev + eps
        log_ret[i] = drift + dev
        prev = dev

    krbn = 100.0 * np.exp(np.cumsum(log_ret))
    # A near-deterministic cash leg compounding at cash_rate (tiny noise for realism).
    cash_ret = cash_rate / TRADING_DAYS_PER_YEAR + rng.normal(0.0, 1e-5, n_days)
    cash = 100.0 * np.exp(np.cumsum(cash_ret))

    frame = pd.DataFrame(
        {"KRBN": krbn, "CASH": cash},
        index=pd.DatetimeIndex(sessions, name="Date"),
    )
    truth = {
        "mu": mu,
        "momentum": momentum,
        "daily_vol": daily_vol,
        "cash_rate": cash_rate,
        "n_days": n_days,
        "seed": seed,
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — daily total-return prices, cache-first
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"{safe}_total_return.parquet")


def load_real(
    tickers: tuple[str, ...] = ("KRBN", "BIL"),
    start: str = KRBN_INCEPTION,
    end: str | None = None,
    use_cache: bool = True,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily **total-return** price frame for ``tickers`` (one column each).

    Cache-first: each ticker is read from ``_cache/<ticker>_total_return.parquet`` if it
    exists; on a miss it falls back to :func:`quantlab.data.fetch` with
    ``mode="total_return"`` (``auto_adjust=True`` ⇒ distributions folded in ⇒ the fair
    series for a buy-and-hold comparison). The frame is the inner join across tickers
    (dropna), so it begins at the latest inception — KRBN (2020-07-30) bounds the window.
    Columns are upper-case tickers; index is tz-naive.
    """
    cols: dict[str, pd.Series] = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if use_cache and os.path.exists(path):
            s = pd.read_parquet(path)
            ser = s["Close"] if "Close" in s.columns else s.iloc[:, 0]
            cols[tk] = ser.rename(tk)
        else:
            import sys

            sys.path.insert(0, REPO_ROOT)
            from quantlab import data as qdata

            raw = qdata.fetch(tk, start=start, end=end, mode="total_return",
                              use_cache=use_cache)
            cols[tk] = raw["Close"].rename(tk)

    frame = pd.concat(cols.values(), axis=1, join="inner").dropna()
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    frame = frame.loc[start:]
    if end is not None:
        frame = frame.loc[:end]
    frame.index.name = "Date"
    return frame


def fingerprint(frame: pd.DataFrame) -> str:
    """A short content fingerprint of a price frame, for the as-of stamp.

    Hashes the column names and the raw float bytes of every column, so a different
    window, a different asset, or drifted data all change the fingerprint loudly.
    """
    h = hashlib.sha1()
    for c in frame.columns:
        h.update(str(c).encode())
        h.update(np.ascontiguousarray(frame[c].to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]
