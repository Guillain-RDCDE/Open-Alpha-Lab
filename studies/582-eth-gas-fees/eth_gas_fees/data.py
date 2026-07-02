"""Data layer for Study 582 (ETH-Gas-Fees) — gas-fee spikes as a top signal.

The folk claim (crypto Twitter / on-chain analytics blogs): when Ethereum **gas fees** spike —
the price of doing anything on-chain, denominated in gwei and dollars — the network is congested
because everyone is minting, aping, and leveraging at once. That euphoric congestion is read as a
*top signal*: pay-anything-to-transact behaviour marks the blow-off, so **forward ETH returns
should be LOW after a gas-fee spike** (a contrarian read). The bullish counter-read is that high
gas is *genuine demand* for blockspace, so it should be neutral-to-positive for forward returns.

DATA AVAILABILITY (stated openly on the SIGNAL axis). A clean, daily, survivorship-free tape of
Ethereum gas fees back to genesis requires an archive node or a keyed API (Etherscan / Dune /
Owlracle). A no-key retail stack (yfinance) reaches **ETH-USD daily prices** but *not* a gas-fee
series. So this study is **synthetic-only** on the signal side: we cannot compute a REAL-tape
headline for the gas→return relation, which — per the desk's house rule for lego-returns /
whisky-cask / sneaker-resale — caps the SIGNAL axis at `WEAK`/`NONE` (a REAL stamp needs a robust
*t* ≥ 2 on a REAL tape, which we do not have). What we *can* do honestly is:

- **``synthetic_series``** — a deterministic, offline generator that builds a coupled
  (gas_gwei, eth_price) daily world with a single knob, ``top_signal_beta``, that plants the
  contrarian effect: the more a *gas spike* (deviation of gas from its trailing baseline) exceeds
  normal, the LOWER the next-window ETH return. ``top_signal_beta = 0`` is the null (gas carries
  no forward information). This is the positive control and the null in one bottle; tests never
  touch the network.
- **``fetch_eth_prices``** — the real yfinance ETH-USD daily tape, cache-first into this study's
  own ``_cache/``. It is used ONLY to characterise the real ETH return distribution the synthetic
  generator is calibrated against (vol, drift), and to state the data-availability limitation with
  a real anchor — NOT to compute a gas→return headline (there is no real gas series here).

No look-ahead: the gas *spike* at day *t* is measured from a trailing baseline ending at day *t*
(a backward-looking rolling median), so it is fully known at the close of day *t*; the outcome is
the ETH return earned over the FORWARD window (t, t+H]. ``forward_returns`` shifts the realised
return so the signal at row *t* never sees its own outcome — one documented execution lag.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

TRADING_DAYS = 365  # crypto trades every day
SEED = 582


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic gas/price world."""

    top_signal_beta: float  # forward-return response per unit of gas-spike (negative = top signal)

    @property
    def has_signal(self) -> bool:
        return self.top_signal_beta != 0.0


# ---------------------------------------------------------------------------
# Synthetic coupled gas / price world — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_series(
    n_days: int = 1461,           # ~4 years of daily data
    top_signal_beta: float = -0.020,
    daily_drift: float = 0.0008,  # ETH mean daily log return (calibrated to real tape)
    daily_vol: float = 0.045,     # ETH daily vol (calibrated to real tape)
    gas_baseline: float = 30.0,   # baseline gas price, gwei
    gas_vol: float = 0.35,        # gas log-shock vol (gas is very spiky)
    coupling: float = 0.6,        # how much a price run-up drags gas up (congestion)
    seed: int = SEED,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible coupled (gas, ETH price) daily world with a tunable *top signal*.

    The mechanism, on purpose, matches the folk story: **congestion follows euphoria**. A latent
    "euphoria" state ``u_t`` (a persistent AR(1)) drives BOTH a contemporaneous price run-up AND a
    gas spike (people transact frantically when price is ripping). The planted forward effect is:

        eth_fwd_ret(t) = daily_drift * H + top_signal_beta * gas_spike(t) + idio

    where ``gas_spike(t)`` is the standardised deviation of gas from its trailing baseline. With
    ``top_signal_beta < 0`` a *high* gas spike predicts a *low* forward ETH return (the contrarian
    top signal); ``= 0`` is the null (gas spikes carry no forward information — the run-up and the
    congestion are contemporaneous coincidence, and tomorrow is a coin flip).

    Returns ``(df, truth)`` with a daily DatetimeIndex and columns:
      - ``eth_price`` — synthetic ETH-USD level
      - ``gas_gwei``  — synthetic daily average gas price, gwei
    exactly the two series ``strategy`` consumes. Deterministic and offline.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_days, freq="D")

    # Latent persistent euphoria state (AR(1)): drives the *contemporaneous* price kick and gas
    # together. Crucially, price responds to the euphoria *innovation* (the shock ``xi``), not the
    # persistent level, so that at the null (``top_signal_beta = 0``) a gas spike — which tracks
    # the persistent *level* ``u`` — carries NO information about the FORWARD return. The run-up
    # and the congestion are contemporaneous; tomorrow is a coin flip. This keeps the positive
    # control honest: the engine must read ~0 at the null and only light up when we plant a signal.
    u = np.zeros(n_days)
    xi = np.zeros(n_days)
    phi = 0.94
    for t in range(1, n_days):
        xi[t] = rng.standard_normal() * 0.3
        u[t] = phi * u[t - 1] + xi[t]

    # Daily log returns: base + a contemporaneous euphoria *innovation* kick + idio noise.
    eps = rng.standard_normal(n_days) * daily_vol
    log_ret = daily_drift + 0.030 * xi + eps
    log_price = np.cumsum(log_ret) + np.log(1200.0)  # start near a plausible ETH level
    eth_price = np.exp(log_price)

    # Gas tracks the persistent euphoria LEVEL (congestion builds and lingers) + its own spikes.
    # It deliberately does NOT depend on future returns, so the null has no forward leakage.
    gas_log = (
        np.log(gas_baseline)
        + coupling * u
        + rng.standard_normal(n_days) * gas_vol
    )
    gas_gwei = np.clip(np.exp(gas_log), 3.0, 2000.0)

    df = pd.DataFrame({"eth_price": eth_price, "gas_gwei": gas_gwei}, index=dates)
    df.index.name = "date"

    # Bake the planted forward effect INTO the realised prices so the tradable test is honest:
    # we perturb each day's log return by top_signal_beta * (yesterday's gas spike), i.e. a high
    # gas spike today drags the NEXT day's return. gas_spike uses only trailing data (no look-ahead).
    if top_signal_beta != 0.0:
        spike = _gas_spike_from_series(pd.Series(gas_gwei, index=dates)).to_numpy()
        # shift the effect forward by one day: spike at t moves return at t+1
        eff = np.zeros(n_days)
        eff[1:] = top_signal_beta * np.nan_to_num(spike[:-1])
        log_ret2 = log_ret + eff
        eth_price = np.exp(np.cumsum(log_ret2) + np.log(1200.0))
        df["eth_price"] = eth_price

    return df, WorldTruth(top_signal_beta=top_signal_beta)


def _gas_spike_from_series(gas: pd.Series, baseline_window: int = 30) -> pd.Series:
    """Standardised trailing gas spike: (gas - rolling median) / rolling MAD-scale.

    Backward-looking only (uses data up to and including day *t*), so it is public at the close of
    day *t*. Higher = gas is spiking relative to its recent baseline.
    """
    med = gas.rolling(baseline_window, min_periods=baseline_window // 2).median()
    mad = (gas - med).abs().rolling(baseline_window, min_periods=baseline_window // 2).median()
    scale = (mad * 1.4826).replace(0.0, np.nan)
    return ((gas - med) / scale).rename("gas_spike")


# ---------------------------------------------------------------------------
# Real ETH tape — yfinance daily prices, cache-first (for the caveat anchor only)
# ---------------------------------------------------------------------------
def _eth_cache() -> str:
    return os.path.join(DEFAULT_CACHE, "eth_usd_daily.parquet")


def _retry(fn, tries: int = 3, pause: float = 1.0):
    last = None
    for _ in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(pause)
    raise last  # type: ignore[misc]


def fetch_eth_prices(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "2018-01-01",
    end: str = "2026-06-30",
) -> pd.DataFrame:
    """Daily ETH-USD close, cache-first.

    Cache-only by default (``fetch=False``): returns the cached parquet if present, else an empty
    frame. Network is touched only on an explicit ``fetch=True`` (then cached). Used ONLY to
    characterise the real ETH return distribution behind the synthetic calibration — there is no
    real gas-fee series in a no-key retail stack, so no real gas→return headline is computed.
    """
    path = _eth_cache()
    if os.path.exists(path):
        px = pd.read_parquet(path)
        if px.index.tz is not None:
            px.index = px.index.tz_localize(None)
        return px
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf  # lazy

    raw = _retry(
        lambda: yf.download("ETH-USD", start=start, end=end, auto_adjust=True, progress=False)
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    out = close.to_frame("eth_price")
    out.index = pd.DatetimeIndex(out.index).tz_localize(None)
    out = out.dropna()
    os.makedirs(cache_dir, exist_ok=True)
    out.to_parquet(path)
    return out


def real_return_moments(cache_dir: str = DEFAULT_CACHE) -> dict:
    """Daily log-return mean/vol of the cached real ETH tape (empty-safe).

    Anchors the synthetic calibration to reality; returns NaNs offline with no cache.
    """
    px = fetch_eth_prices(cache_dir=cache_dir, fetch=False)
    if px.empty:
        return {"n": 0, "mean": float("nan"), "vol": float("nan"),
                "start": None, "end": None}
    lr = np.log(px["eth_price"]).diff().dropna()
    return {
        "n": int(len(lr)),
        "mean": float(lr.mean()),
        "vol": float(lr.std(ddof=1)),
        "start": str(px.index[0].date()),
        "end": str(px.index[-1].date()),
    }


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
