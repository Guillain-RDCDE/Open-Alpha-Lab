"""Data layer for Study 584 (Exchange-Netflows) — the on-chain "coins-to-exchanges = bearish" claim.

The folklore, from crypto on-chain analytics desks (CryptoQuant, Glassnode, Nansen): when BTC
**flows onto** exchanges (positive *net-inflow* = deposits minus withdrawals), holders are moving
coins to venues where they can *sell*, so a rising exchange net-inflow is read as a **bearish**
lead for BTC's forward return; a net-*outflow* (coins leaving to cold storage / self-custody) is
read as bullish accumulation. The tradable version: go flat/short when net-inflows spike, long
when net-outflows spike.

### Why this study is SYNTHETIC-ONLY (the data-availability wall)

Per-entity, exchange-tagged on-chain flow — the actual "how many BTC moved to Binance/Coinbase
wallets today" series — is a **proprietary, address-clustering product**. It requires labelling
millions of on-chain addresses to known exchange cold/hot wallets, which is exactly the moat that
Glassnode / CryptoQuant / Nansen sell behind paid API keys (hundreds of USD/month, no free tier
with usable history). A no-key retail stack **cannot reach a real exchange-netflow tape** — the
raw blockchain gives you transfers, not the *exchange labels* that define a "netflow". So there is
no honest real BTC-netflow series to certify a signal on. This study is therefore built on a
**deterministic synthetic world** and is **capped at WEAK/NONE** by house rule: a `REAL` stamp
needs a robust *t* >= 2 on a REAL tape, and no such tape exists here. The data-availability
limitation is stated openly on the SIGNAL axis (like the repo's lego-returns / whisky-cask /
sneaker-resale studies).

### The synthetic world (deterministic, seeded)

``synthetic_series`` builds a daily panel over ``n_days`` with:

- a BTC daily-return series with realistic crypto volatility and fat-ish tails,
- an exchange **net-inflow** series (positive = coins TO exchanges) with its own persistence,
- a single knob, ``bear_beta``: today's net-inflow depresses the *forward* BTC return by
  ``bear_beta`` per unit of standardised inflow. ``bear_beta < 0`` plants the bearish folklore
  (inflows -> lower forward returns); ``bear_beta = 0`` is the null (netflow tells you nothing).

The engine's positive control and null live in this one generator. Tests never touch the network;
there is no network path at all (no real fetch exists to write).

No look-ahead: the signal on day *t* is the netflow **known at the close of day *t*** (a public
on-chain quantity), and it is tested against the return over ``(t, t+lag]`` — a strictly forward
window. The one documented execution lag is ``lag`` days (default 1: you see today's netflow, you
trade tomorrow's return).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Fixed study seed (== study number) so the deterministic core is reproducible.
STUDY_SEED = 584


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic world."""

    bear_beta: float  # forward-return response per unit standardised net-inflow (negative = folklore)

    @property
    def has_signal(self) -> bool:
        return self.bear_beta != 0.0


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    """A zero-mean AR(1) path (persistence ``phi``, innovation sd ``sigma``)."""
    x = np.empty(n)
    x[0] = rng.standard_normal() * sigma / np.sqrt(max(1e-6, 1 - phi * phi))
    for i in range(1, n):
        x[i] = phi * x[i - 1] + sigma * rng.standard_normal()
    return x


def synthetic_series(
    n_days: int = 1500,
    bear_beta: float = -0.0025,
    btc_vol: float = 0.035,
    drift: float = 0.0007,
    flow_phi: float = 0.6,
    seed: int = STUDY_SEED,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible daily BTC + exchange-netflow world with a tunable bearish coupling.

    Construction (all offline, deterministic given ``seed``):

    - ``netflow`` — a persistent AR(1) series (positive = coins moving TO exchanges), then
      standardised to ``z_flow`` (mean 0, sd 1) so ``bear_beta`` is in "per-1-sigma-of-inflow"
      units.
    - ``btc_ret`` — the *realised* daily BTC return. The forward return over the next ``lag`` days
      carries the planted effect, so we build returns as::

          btc_ret[t] = drift + bear_beta * z_flow[t-1] + idio[t]

      i.e. **yesterday's** standardised net-inflow shifts **today's** return by
      ``bear_beta * z_flow`` (a strictly forward, no-look-ahead coupling). With ``bear_beta < 0``,
      a net-inflow spike is followed by *lower* BTC returns — the bearish folklore. With
      ``bear_beta = 0`` the return is pure noise around the drift (the null).

    Returns ``(frame, truth)`` where ``frame`` has a ``DatetimeIndex`` and columns
    ``netflow`` (raw), ``z_flow`` (standardised) and ``btc_ret`` (daily total return).
    """
    rng = np.random.default_rng(seed)

    # Persistent net-inflow series (arbitrary "BTC/day" units; sign is what matters).
    raw_flow = _ar1(rng, n_days, phi=flow_phi, sigma=1.0)
    z_flow = (raw_flow - raw_flow.mean()) / (raw_flow.std(ddof=0) or 1.0)

    idio = btc_vol * rng.standard_normal(n_days)
    btc_ret = np.full(n_days, drift) + idio
    # yesterday's standardised inflow shifts today's return (forward coupling, no look-ahead)
    btc_ret[1:] += bear_beta * z_flow[:-1]

    idx = pd.bdate_range("2019-01-01", periods=n_days, name="date")
    frame = pd.DataFrame(
        {"netflow": raw_flow, "z_flow": z_flow, "btc_ret": btc_ret},
        index=idx,
    )
    return frame, WorldTruth(bear_beta=bear_beta)


def forward_return(frame: pd.DataFrame, lag: int = 1) -> pd.Series:
    """Forward cumulative BTC return over ``(t, t+lag]`` — the target the netflow predicts.

    For ``lag = 1`` this is simply next-day return. Uses only future returns relative to the
    signal at ``t``; the last ``lag`` rows are NaN (no forward window).
    """
    r = frame["btc_ret"]
    fwd = pd.Series(np.nan, index=frame.index, name=f"fwd_ret_{lag}d")
    if lag < 1:
        raise ValueError("lag must be >= 1")
    # cumulative return over the next `lag` days
    log_r = np.log1p(r)
    cum = log_r.rolling(lag).sum().shift(-lag)
    fwd.loc[:] = np.expm1(cum)
    return fwd


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
