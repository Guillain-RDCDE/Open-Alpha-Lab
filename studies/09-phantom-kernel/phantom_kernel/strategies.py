"""The Avellaneda-Stoikov quotes, three rival market-makers, and the tournament.

The two AS equations, verbatim from the 2008 paper:

    reservation price   r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
    optimal half-spread (delta_bid + delta_ask)/2, with
    delta_bid + delta_ask = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)

The whole point of the model is the *skew*: the reservation price slides away from the mid in
proportion to inventory, so quotes lean to flatten the book. The rivals isolate how much of
that is worth anything:

* :class:`ASQuoter`        — the full model (fixed sigma, fixed k).
* :class:`AdaptiveASQuoter`— AS with rolling realised-vol (the production fix the article
  praises): same skew, but sigma tracks the market.
* :class:`SymmetricQuoter` — a constant spread centred on the mid, no inventory skew at all.
* :class:`ClampQuoter`     — a constant spread, but it simply *stops quoting the side that
  grows inventory* past a band. The trivial "don't hold inventory" rule — the beta against
  which the AS skew's "alpha" must be measured.

:func:`run_market` replays one exogenous :class:`~phantom_kernel.sim.Flow` through a quoter and
returns a per-step ledger; :func:`metrics` summarises it. All deterministic given the flow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .sim import Flow


# --------------------------------------------------------------------------- #
# The two AS equations
# --------------------------------------------------------------------------- #
def reservation_price(s: float, q: float, t: float, gamma: float, sigma: float, T: float) -> float:
    """AS reservation (indifference) price: the mid, skewed by inventory risk."""
    return s - q * gamma * sigma * sigma * (T - t)


def optimal_spread(t: float, gamma: float, sigma: float, k: float, T: float) -> float:
    """AS optimal *total* spread = inventory-risk term + order-arrival term."""
    return gamma * sigma * sigma * (T - t) + (2.0 / gamma) * math.log(1.0 + gamma / k)


# --------------------------------------------------------------------------- #
# The quoters
# --------------------------------------------------------------------------- #
class ASQuoter:
    """Full Avellaneda-Stoikov with fixed parameters."""

    def __init__(self, gamma: float, sigma: float, k: float, T: float):
        self.gamma, self.sigma, self.k, self.T = gamma, sigma, k, T

    def observe(self, s: float) -> None:  # no state to update
        pass

    def quote(self, s: float, q: float, t: float) -> tuple[float | None, float | None]:
        r = reservation_price(s, q, t, self.gamma, self.sigma, self.T)
        half = optimal_spread(t, self.gamma, self.sigma, self.k, self.T) / 2.0
        return r - half, r + half


class AdaptiveASQuoter(ASQuoter):
    """AS with rolling realised-volatility — the first production fix the article recommends.

    The skew and spread formulas are untouched; only ``sigma`` is re-estimated on a rolling
    window of recent mid moves, so the quotes breathe with the volatility regime.
    """

    def __init__(self, gamma: float, sigma: float, k: float, T: float, vol_window: int = 200, dt: float = 0.01):
        super().__init__(gamma, sigma, k, T)
        self.vol_window = vol_window
        self.dt = dt
        self._prices: list[float] = []

    def observe(self, s: float) -> None:
        self._prices.append(s)
        if len(self._prices) > self.vol_window + 1:
            self._prices.pop(0)
        if len(self._prices) >= 20:
            d = np.diff(np.asarray(self._prices))
            rv = d.std()
            if rv > 0:
                self.sigma = float(rv / math.sqrt(self.dt))  # per-sqrt-time vol


class SymmetricQuoter:
    """Constant half-spread, centred on the mid. No inventory skew — the naive baseline."""

    def __init__(self, half_spread: float):
        self.half = half_spread

    def observe(self, s: float) -> None:
        pass

    def quote(self, s: float, q: float, t: float) -> tuple[float | None, float | None]:
        return s - self.half, s + self.half


class ClampQuoter:
    """Constant spread, but stop quoting the side that would grow |inventory| past a band.

    This is the cheapest possible inventory control — no reservation price, no maths — and so
    the honest control for "is the AS skew real alpha, or just keeping inventory near zero?".
    """

    def __init__(self, half_spread: float, max_inventory: float):
        self.half = half_spread
        self.q_max = max_inventory

    def observe(self, s: float) -> None:
        pass

    def quote(self, s: float, q: float, t: float) -> tuple[float | None, float | None]:
        bid = s - self.half if q < self.q_max else None     # long enough -> stop buying
        ask = s + self.half if q > -self.q_max else None     # short enough -> stop selling
        return bid, ask


# --------------------------------------------------------------------------- #
# The market loop
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Ledger:
    """Per-step record of one quoter's run through one market."""

    pnl: np.ndarray          # mark-to-market wealth path, shape (n_steps + 1,)
    inventory: np.ndarray    # inventory path, shape (n_steps + 1,)
    n_fills: int
    n_adverse: int           # fills immediately followed by an informed move against us


def run_market(flow: Flow, quoter, T: float | None = None) -> Ledger:
    """Replay ``flow`` through ``quoter``, returning the wealth and inventory paths.

    Each step: the quoter posts (bid, ask) around the *current* mid; an incoming market sell
    with reach >= (mid - bid) lifts our bid (we buy, inventory up); an incoming market buy with
    reach >= (ask - mid) lifts our ask (we sell, inventory down). Wealth is marked to the mid:
    ``pnl = cash + inventory * s``. A fill tagged informed that is then run over by the mid is
    counted as an adverse selection.
    """
    n = flow.n_steps
    dt = flow.dt
    T = T if T is not None else n * dt
    s = flow.s
    cash = 0.0
    q = 0.0
    pnl = np.empty(n + 1)
    inv = np.empty(n + 1)
    pnl[0] = 0.0
    inv[0] = 0.0
    n_fills = 0
    n_adverse = 0

    for t in range(n):
        quoter.observe(s[t])
        bid, ask = quoter.quote(s[t], q, t * dt)
        # Market sell hits our bid (we buy at bid).
        if bid is not None and flow.sell_arr[t] and flow.sell_reach[t] >= (s[t] - bid):
            q += 1.0
            cash -= bid
            n_fills += 1
            if flow.sell_informed[t]:
                n_adverse += 1
        # Market buy lifts our ask (we sell at ask).
        if ask is not None and flow.buy_arr[t] and flow.buy_reach[t] >= (ask - s[t]):
            q -= 1.0
            cash += ask
            n_fills += 1
            if flow.buy_informed[t]:
                n_adverse += 1
        pnl[t + 1] = cash + q * s[t + 1]
        inv[t + 1] = q

    return Ledger(pnl=pnl, inventory=inv, n_fills=n_fills, n_adverse=n_adverse)


def metrics(led: Ledger) -> dict:
    """Summarise a ledger: terminal P&L, per-step Sharpe, inventory risk, fills."""
    dpnl = np.diff(led.pnl)
    sd = dpnl.std(ddof=1)
    sharpe = float(dpnl.mean() / sd) if sd > 0 else float("nan")
    return {
        "terminal_pnl": float(led.pnl[-1]),
        "pnl_per_step": float(dpnl.mean()),
        "sharpe_step": sharpe,                 # per-step Sharpe of the wealth increments
        "inv_std": float(led.inventory.std()),
        "inv_absmax": float(np.abs(led.inventory).max()),
        "n_fills": int(led.n_fills),
        "n_adverse": int(led.n_adverse),
    }


def daily_returns(led: Ledger, block: int = 200) -> pd.Series:
    """Block-summed P&L increments — a coarser return series for bootstrap Sharpe CIs."""
    dpnl = np.diff(led.pnl)
    n_blocks = len(dpnl) // block
    trimmed = dpnl[: n_blocks * block].reshape(n_blocks, block).sum(axis=1)
    return pd.Series(trimmed)
