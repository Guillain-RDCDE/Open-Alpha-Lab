"""The order-flow simulator — a transparent market a market-maker can be tested inside.

The reproducible core of this study is a simulator, not a cached data dump, for one honest
reason: the Avellaneda-Stoikov optimum is a *theorem about a model world*, so the cleanest
way to ask "does the theorem's load-bearing assumption survive reality?" is to build two
worlds — one that obeys the assumptions (where the model *must* work, validating the code)
and one wired with the documented frictions the paper omits — and run the same machinery in
both. Everything is deterministic given a seed.

The one assumption everything hangs on
--------------------------------------
AS assume market orders arrive at a rate that fades exponentially with how far your quote
sits from the mid::

    lambda(delta) = A * exp(-k * delta)

That exponential is not a stylised convenience — it is *what makes the optimal half-spread a
closed form* (the ``(2/gamma) ln(1 + gamma/k)`` term). It has a clean microstructure
reading: a resting quote at distance ``delta`` is filled by an incoming market order iff that
order "reaches" at least ``delta`` into the book, so

    lambda(delta) = Lambda * P(reach >= delta) = Lambda * S_reach(delta),

i.e. *the arrival kernel is the survival function of order reach, scaled by the arrival
rate.* This makes the assumption falsifiable with a single question: **is order reach
exponentially distributed?** If ``reach ~ Exponential(k)`` then ``S(delta) = exp(-k delta)``
and AS holds exactly. If reach is heavy-tailed (the empirically documented case — trade
sizes and the volume to walk the book follow power laws; Gopikrishnan et al. 2000, Gabaix
et al. 2003), the kernel is a power law, ``k`` is not a stable number, and the "optimal"
spread is computed from a phantom parameter.

Two worlds
----------
* :data:`WORLD_A` — textbook AS: exponential reach (so the kernel is genuinely
  ``A e^{-k delta}``), constant volatility, no jumps, all flow uninformed. The model *should*
  estimate cleanly and the skew *should* pay here. If it doesn't, the code is wrong.
* :data:`WORLD_B` — the same market with the frictions the paper sets aside: **heavy-tailed
  (Pareto) reach**, **price jumps**, **stochastic volatility**, and a fraction of **informed**
  orders that move the mid against whoever just filled them (adverse selection).

Used by :mod:`phantom_kernel.estimator` (which only needs the reach/fill side) and
:mod:`phantom_kernel.strategies` (which needs the full mid path + order stream).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# World definition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class World:
    """A market the simulator can realise. All rates are per unit of model time.

    ``reach_kind`` selects the order-reach law that *generates* the arrival kernel:
    ``"exponential"`` (mean ``1/reach_k`` -> the kernel is exactly ``A e^{-reach_k delta}``,
    the AS assumption) or ``"pareto"`` (a heavy tail -> a power-law kernel, the documented
    reality). The mid-price frictions (``jump_rate``, ``vol_of_vol``, ``informed_frac``) are
    zero in the textbook world and switched on in the friction world.
    """

    name: str
    # --- order arrivals (per side) ---
    arrival_rate: float = 40.0      # Lambda: expected market orders per unit time, each side
    reach_kind: str = "exponential"  # "exponential" | "pareto"
    reach_k: float = 0.6            # exponential rate; mean reach = 1/reach_k
    pareto_alpha: float = 1.5       # Pareto/Lomax tail exponent (heavier as alpha -> 1)
    pareto_xm: float = 0.55         # Pareto scale (sets the mean reach alongside alpha)
    # --- mid-price dynamics ---
    sigma: float = 0.4              # base volatility (per unit sqrt-time)
    jump_rate: float = 0.0          # Poisson jumps per unit time (0 = none)
    jump_size: float = 0.0          # jump magnitude (std of the jump)
    vol_of_vol: float = 0.0         # amplitude of the (mean-reverting) vol process (0 = const)
    # --- information / adverse selection ---
    informed_frac: float = 0.0      # share of orders that are informed
    informed_edge: float = 0.0      # mid move an informed order drags behind it
    informed_reach_boost: float = 0.0  # extra reach an informed order carries (it picks off quotes)

    def mean_reach(self) -> float:
        """Expected order reach under this world's law (for matching the two worlds)."""
        if self.reach_kind == "exponential":
            return 1.0 / self.reach_k
        # Lomax (Pareto type II) shifted to start at xm: mean = xm * (1 + 1/(alpha-1)).
        return self.pareto_xm * (1.0 + 1.0 / (self.pareto_alpha - 1.0))


# The textbook world: every AS assumption holds. The kernel is exactly A e^{-k delta}.
WORLD_A = World(
    name="A (textbook)",
    arrival_rate=40.0,
    reach_kind="exponential",
    reach_k=0.6,
    sigma=0.4,
    jump_rate=0.0,
    vol_of_vol=0.0,
    informed_frac=0.0,
)

# The friction world: heavy-tailed reach (power-law kernel), jumps, stochastic vol, and
# informed flow. Mean reach is matched to WORLD_A (1/0.6 ~ 1.67) so the two markets trade at a
# comparable *volume* and only the *shape* of the kernel differs. With alpha = 1.2 the reach
# has infinite variance — the occasional book-sweeping order the exponential law cannot
# produce, and the documented reality of market-order size (Gopikrishnan 2000, Gabaix 2003).
WORLD_B = World(
    name="B (frictions)",
    arrival_rate=40.0,
    reach_kind="pareto",
    pareto_alpha=1.2,
    pareto_xm=0.28,                 # mean reach = 0.28 * (1 + 1/0.2) = 1.68 ~ 1/0.6 (WORLD_A)
    sigma=0.4,
    jump_rate=2.0,
    jump_size=1.2,
    vol_of_vol=0.6,
    informed_frac=0.30,
    informed_edge=0.9,
    informed_reach_boost=2.5,
)


# --------------------------------------------------------------------------- #
# Order reach — the object whose survival function IS the AS kernel
# --------------------------------------------------------------------------- #
def sample_reach(world: World, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``size`` order reaches under ``world``'s law (always non-negative).

    Exponential reach gives ``S(delta) = e^{-k delta}`` — the AS kernel exactly. Pareto
    (Lomax) reach gives a power-law survival — the heavy-tailed reality that breaks it.
    """
    if world.reach_kind == "exponential":
        return rng.exponential(1.0 / world.reach_k, size)
    if world.reach_kind == "pareto":
        # numpy's pareto(a) is Lomax: support [0, inf), survival (1 + x)^(-a). Shift by xm so
        # the smallest reaches are xm and the mean matches the exponential world.
        return world.pareto_xm * (1.0 + rng.pareto(world.pareto_alpha, size))
    raise ValueError(f"unknown reach_kind {world.reach_kind!r}")


def fill_counts(
    world: World,
    deltas: np.ndarray,
    n_orders: int = 200_000,
    seed: int = 0,
) -> np.ndarray:
    """Empirical fill counts at each quote distance in ``deltas``.

    Simulates ``n_orders`` incoming market orders and, for each candidate quote distance
    ``delta``, counts how many would fill a resting quote there (reach >= delta). This is the
    raw material the estimator fits: ``count(delta) ~ n_orders * S_reach(delta)``, i.e. a
    sampled, Poisson-noisy version of the arrival kernel ``lambda(delta)``.
    """
    rng = np.random.default_rng(seed)
    reach = sample_reach(world, n_orders, rng)
    # count(delta_j) = #{i : reach_i >= delta_j}; vectorised via a sorted-search on reach.
    reach_sorted = np.sort(reach)
    # number with reach >= d  =  n - (index of first reach >= d)
    idx = np.searchsorted(reach_sorted, deltas, side="left")
    return (reach.size - idx).astype(float)


# --------------------------------------------------------------------------- #
# The full exogenous stream — mid path + order flow for the MM tournament
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Flow:
    """One realised market: a mid path plus the per-step incoming order stream.

    The stream is *exogenous* — it does not depend on which market-maker is being tested — so
    every quoter in the tournament faces the identical market and the comparison is clean. The
    mid moves on information regardless of whether our MM filled the informed order (the
    informed trader trades anyway); the *adverse selection* bites only when our quote was the
    one that got hit just before the move. ``dt`` is the per-step time increment.
    """

    s: np.ndarray            # mid path, shape (n_steps + 1,)
    dt: float
    buy_arr: np.ndarray      # bool (n_steps,): a market buy (lifts asks) arrived this step
    buy_reach: np.ndarray    # its reach from mid
    buy_informed: np.ndarray
    sell_arr: np.ndarray     # bool: a market sell (hits bids) arrived this step
    sell_reach: np.ndarray
    sell_informed: np.ndarray

    @property
    def n_steps(self) -> int:
        return self.buy_arr.size


def simulate_flow(
    world: World,
    n_steps: int = 20_000,
    dt: float = 0.01,
    s0: float = 100.0,
    seed: int = 0,
) -> Flow:
    """Realise one market under ``world``: the mid path and the two-sided order stream.

    Mid dynamics per step: a diffusion ``sigma_t sqrt(dt) Z`` (with ``sigma_t`` a
    mean-reverting stochastic-vol process when ``vol_of_vol > 0``), optional Poisson jumps,
    and an informed-flow drag — each informed order pulls the mid ``informed_edge`` in its own
    direction *after* it could have filled a quote, which is exactly the mechanism that turns
    a fresh fill into an adverse-selection loss.
    """
    rng = np.random.default_rng(seed)
    sqdt = np.sqrt(dt)

    # Stochastic volatility: log-vol as a discrete Ornstein-Uhlenbeck around 0 (so the median
    # instantaneous vol stays ~ sigma). Flat when vol_of_vol == 0 (the textbook world).
    log_v = np.zeros(n_steps)
    if world.vol_of_vol > 0:
        kappa = 2.0  # mean-reversion speed
        v = 0.0
        for t in range(n_steps):
            v += -kappa * v * dt + world.vol_of_vol * sqdt * rng.standard_normal()
            log_v[t] = v
    sigma_t = world.sigma * np.exp(log_v)

    # Order arrivals (thinned Poisson: per-step Bernoulli at rate arrival_rate * dt).
    p_arr = world.arrival_rate * dt
    buy_arr = rng.random(n_steps) < p_arr
    sell_arr = rng.random(n_steps) < p_arr
    buy_reach = sample_reach(world, n_steps, rng)
    sell_reach = sample_reach(world, n_steps, rng)
    buy_informed = buy_arr & (rng.random(n_steps) < world.informed_frac)
    sell_informed = sell_arr & (rng.random(n_steps) < world.informed_frac)
    # Informed orders are aggressive — they reach further into the book to pick off quotes.
    buy_reach = buy_reach + world.informed_reach_boost * buy_informed
    sell_reach = sell_reach + world.informed_reach_boost * sell_informed

    # Jumps (Poisson in time, Gaussian in size).
    jump = np.zeros(n_steps)
    if world.jump_rate > 0:
        jhit = rng.random(n_steps) < world.jump_rate * dt
        jump = jhit * world.jump_size * rng.standard_normal(n_steps)

    # Build the mid path. Informed flow drags the mid in the order's direction.
    s = np.empty(n_steps + 1)
    s[0] = s0
    z = rng.standard_normal(n_steps)
    informed_move = world.informed_edge * (buy_informed.astype(float) - sell_informed.astype(float))
    for t in range(n_steps):
        s[t + 1] = s[t] + sigma_t[t] * sqdt * z[t] + jump[t] + informed_move[t]

    return Flow(
        s=s, dt=dt,
        buy_arr=buy_arr, buy_reach=buy_reach, buy_informed=buy_informed,
        sell_arr=sell_arr, sell_reach=sell_reach, sell_informed=sell_informed,
    )
