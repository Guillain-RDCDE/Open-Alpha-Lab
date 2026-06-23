"""Data layer for Study 370 — the 0DTE-options boom and the SPX tape.

Two sources, both offline-friendly:

* **Real tape.** Daily SPY OHLC (yfinance, no key), cached under ``_cache/spy_ohlc.csv``,
  plus a one-shot **option-chain snapshot** for the nearest SPY expiry cached under
  ``_cache/spy_option_chain.csv``. From the daily OHLC we build two *proxies* for the
  intraday-tape behaviour the 0DTE story is about:

    - ``oc_ret``  — close-to-close log return (the only return a daily feed sees);
    - ``intraday`` — open→close log return, and ``overnight`` — prevClose→open log return,
      the daily decomposition that lets us watch the *intraday* leg separately;

  and a same-day **realised-range** vol proxy from the high/low. The decisive caveat: a
  daily feed CANNOT see true 0DTE intraday microstructure (no free intraday option tape),
  so every cross-2022 contrast here is a labelled **proxy**, not a measurement of the SPX
  intraday tape itself.

* **Synthetic.** A deterministic, fixed-seed generator that produces an intraday return
  series with a **pinning regime switch** at a chosen date: after the switch, intraday
  returns become more **mean-reverting** (AR(1) coefficient = ``-pin``) — the positive
  control. With ``pin = 0`` no regime change exists and the detector must NOT manufacture a
  significant break; with a large ``pin`` the break must light up.

Pure numpy + pandas + stdlib for the offline path. ``fetch_spy`` / ``fetch_option_chain``
(network) are only used once to build the cache and are never imported by the notebooks'
offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
SPY_CACHE = os.path.join(CACHE_DIR, "spy_ohlc.csv")
OPT_CACHE = os.path.join(CACHE_DIR, "spy_option_chain.csv")

# The structural break the 0DTE story hangs on: same-day SPX/SPY options went from a Friday
# curiosity to a daily product in 2022 (Mon/Wed/Fri rollout through 2022). We split the tape
# at the start of 2022.
BREAK_DATE = pd.Timestamp("2022-01-01")


# --------------------------------------------------------------------------- #
# Real tape (network — used once to build the cache)
# --------------------------------------------------------------------------- #
def fetch_spy(start: str = "2010-01-01", end: str | None = None,
              path: str = SPY_CACHE) -> pd.DataFrame:
    """Download SPY daily OHLC via yfinance and cache it (network-only, used once)."""
    import yfinance as yf

    raw = yf.download("SPY", start=start, end=end, auto_adjust=False, progress=False)
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    raw = raw.dropna(how="all")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def fetch_option_chain(path: str = OPT_CACHE) -> pd.DataFrame:
    """Download the nearest-expiry SPY option chain snapshot (network-only, used once)."""
    import yfinance as yf

    spy = yf.Ticker("SPY")
    near = spy.options[0]
    ch = spy.option_chain(near)
    cols = ["strike", "lastPrice", "bid", "ask", "volume", "openInterest",
            "impliedVolatility"]
    calls = ch.calls[cols].copy(); calls["type"] = "call"
    puts = ch.puts[cols].copy(); puts["type"] = "put"
    both = pd.concat([calls, puts], ignore_index=True)
    both["expiry"] = near
    os.makedirs(os.path.dirname(path), exist_ok=True)
    both.to_csv(path, index=False)
    return both


def have_real(path: str = SPY_CACHE) -> bool:
    return os.path.exists(path)


def have_option_chain(path: str = OPT_CACHE) -> bool:
    return os.path.exists(path)


def load_option_chain(path: str = OPT_CACHE) -> pd.DataFrame:
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Build the proxy frame from daily OHLC
# --------------------------------------------------------------------------- #
def _tape_from_ohlc(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Daily OHLC -> proxy frame with intraday/overnight/close-to-close legs + range vol.

    Columns (all log returns unless noted):
      * ``close``     — raw close (level)
      * ``oc_ret``    — close[t]/close[t-1] - 1's log analogue (close-to-close)
      * ``intraday``  — open→close   log return (the daily 'intraday' leg)
      * ``overnight`` — prevClose→open log return (the gap)
      * ``rng``       — Parkinson-style same-day range proxy log(high/low)
    """
    o = ohlc["Open"].astype(float)
    h = ohlc["High"].astype(float)
    low = ohlc["Low"].astype(float)
    c = ohlc["Close"].astype(float)
    out = pd.DataFrame(index=ohlc.index)
    out["close"] = c
    out["oc_ret"] = np.log(c / c.shift(1))
    out["intraday"] = np.log(c / o)
    out["overnight"] = np.log(o / c.shift(1))
    out["rng"] = np.log(h / low)
    return out.dropna()


def load_prices(path: str = SPY_CACHE) -> pd.DataFrame:
    """Load the cached SPY OHLC frame (index = date)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df


def load_real(path: str = SPY_CACHE) -> pd.DataFrame:
    """Convenience: cached SPY OHLC -> proxy tape frame in one call."""
    return _tape_from_ohlc(load_prices(path))


# --------------------------------------------------------------------------- #
# Synthetic positive control — an intraday-tape pinning regime switch
# --------------------------------------------------------------------------- #
def synthetic_tape(n_days: int = 4_000, pin: float = 0.0, seed: int = 370,
                   sig: float = 0.008, switch_frac: float = 0.5) -> pd.DataFrame:
    """Deterministic intraday return series with a *pinning* (mean-reversion) regime switch.

    Before the switch the daily **intraday** (open→close) leg is white noise. After the
    switch (at ``switch_frac`` of the sample) the leg becomes an AR(1) with negative
    coefficient ``-pin``: a higher ``pin`` means a more *mean-reverting / pinned* tape — the
    exact behaviour the 0DTE story predicts. With ``pin = 0`` there is **no** regime change
    and the break test must NOT manufacture one; with a large ``pin`` the break must light up.

    Returns the same column schema as the real proxy frame, so the same engine runs on both.
    """
    rng = np.random.default_rng(seed)
    sw = int(n_days * switch_frac)
    intraday = np.empty(n_days)
    eps = rng.normal(0.0, sig, size=n_days)
    intraday[0] = eps[0]
    for t in range(1, n_days):
        phi = -pin if t >= sw else 0.0          # mean-reversion only after the switch
        intraday[t] = phi * intraday[t - 1] + eps[t]
    overnight = rng.normal(0.0002, sig * 0.6, size=n_days)   # gap leg, regime-independent
    close = 100.0 * np.exp(np.cumsum(intraday + overnight))
    rng_proxy = np.abs(intraday) + rng.uniform(0.001, 0.004, size=n_days)
    # decorative business-day index; pd.period_range keeps it inside datetime bounds for
    # any plausible n_days (the date is only a label — the regime switch is positional).
    idx = pd.period_range("2008-01-01", periods=n_days, freq="D").to_timestamp()
    return pd.DataFrame(
        {"close": close, "oc_ret": intraday + overnight, "intraday": intraday,
         "overnight": overnight, "rng": rng_proxy},
        index=idx,
    )
