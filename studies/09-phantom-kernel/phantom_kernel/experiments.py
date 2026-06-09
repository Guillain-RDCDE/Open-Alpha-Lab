"""The teardown — the experiments behind the verdict, each returning a tidy table.

Three questions, in the desk's order:

1. **Does the machine work?** On the textbook world the estimator must recover ``k`` and the
   AS skew must pay. :func:`estimator_recovery`, and the WORLD_A row of :func:`tournament`.
2. **Is the kernel real (H1)?** Exponential vs power-law goodness-of-fit on each world's fills
   (:func:`kernel_gof_table`), and the spread error a static ``k`` pays when the true one
   drifts 4x intraday (:func:`k_instability`).
3. **Is the skew alpha or beta (H2)?** The market-making tournament (:func:`tournament`) in
   both worlds: AS vs adaptive-AS vs a skew-free symmetric quoter vs a brainless inventory
   clamp. The gap between AS and the *clamp* is the part of the model that isn't just "don't
   hold inventory".

Everything is deterministic given the seeds. The numbers these functions print are the
numbers in ``docs/results.md`` and the README.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import sim
from .estimator import fit_exponential, goodness_of_fit, static_k_spread_error
from .strategies import (
    AdaptiveASQuoter,
    ASQuoter,
    ClampQuoter,
    JumpRobustASQuoter,
    SymmetricQuoter,
    metrics,
    optimal_spread,
    daily_returns,
    run_market,
)

try:  # the desk's shared bootstrap; keep the study importable even if run standalone
    from quantlab.stats import sharpe_ci_bootstrap
except Exception:  # pragma: no cover
    sharpe_ci_bootstrap = None


# Quote-distance grid the kernel is measured on (price units, same scale as order reach).
# Reaches far into the tail (out to 40) where an exponential predicts ~zero fills but a
# power-law kernel keeps delivering them — the range that exposes the wrong shape.
def default_deltas() -> np.ndarray:
    return np.linspace(0.25, 40.0, 80)


# --------------------------------------------------------------------------- #
# 1. Machinery sanity — recover k where it genuinely exists
# --------------------------------------------------------------------------- #
def estimator_recovery(world: sim.World = sim.WORLD_A, n_orders: int = 400_000, seed: int = 0) -> dict:
    """On an exponential world, the fitted ``k`` must match the planted ``k`` (and R^2 ~ 1)."""
    deltas = default_deltas()
    counts = sim.fill_counts(world, deltas, n_orders=n_orders, seed=seed)
    fit = fit_exponential(deltas, counts)
    return {
        "k_true": world.reach_k,
        "k_recovered": round(fit["k"], 4),
        "rel_error_pct": round(100.0 * (fit["k"] - world.reach_k) / world.reach_k, 2),
        "r2": round(fit["r2"], 5),
    }


# --------------------------------------------------------------------------- #
# 2a. Is the kernel exponential at all? (H1)
# --------------------------------------------------------------------------- #
def kernel_gof_table(n_orders: int = 400_000, seed: int = 0) -> pd.DataFrame:
    """Exponential-vs-power-law fit on each world's fill kernel.

    WORLD_A should crown the exponential (it is one); WORLD_B's heavy-tailed reach should make
    the power law win, i.e. the AS form is the wrong shape — H1 falsified.
    """
    deltas = default_deltas()
    rows = {}
    for w in (sim.WORLD_A, sim.WORLD_B):
        counts = sim.fill_counts(w, deltas, n_orders=n_orders, seed=seed)
        g = goodness_of_fit(deltas, counts)
        rows[w.name] = {
            "r2_exp": round(g["r2_exp"], 4),
            "r2_pow": round(g["r2_pow"], 4),
            "k_exp": round(g["k_exp"], 4),
            "alpha_pow": round(g["alpha_pow"], 4),
            "aic_gap": round(g["aic_gap"], 1),     # >0 => power-law preferred
            "winner": g["winner"],
        }
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# 2b. The phantom parameter — a static k while the true one drifts (H1)
# --------------------------------------------------------------------------- #
def k_instability(seed: int = 0) -> dict:
    """Four intraday regimes with k spanning 4x; the static fit's spread error per regime."""
    deltas = default_deltas()
    k_values = np.array([0.3, 0.6, 0.9, 1.2])   # 4x spread, the article's admitted range
    return static_k_spread_error(k_values, deltas, n_orders=200_000, seed=seed)


# --------------------------------------------------------------------------- #
# 3. The tournament — is the AS skew alpha, or just inventory control? (H2)
# --------------------------------------------------------------------------- #
def _k_for_as(world: sim.World, seed: int) -> float:
    """The k an AS desk would actually plug in: the true rate if exponential, else the
    (misspecified) exponential fit a practitioner would estimate from this world's fills."""
    if world.reach_kind == "exponential":
        return world.reach_k
    deltas = default_deltas()
    counts = sim.fill_counts(world, deltas, n_orders=400_000, seed=seed)
    return fit_exponential(deltas, counts)["k"]


def tournament(
    world: sim.World,
    n_steps: int = 60_000,
    dt: float = 0.01,
    gamma: float = 0.1,
    max_inventory: float = 30.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Run four market-makers through one realisation of ``world`` and rank them.

    The symmetric and clamp baselines quote a constant spread set to AS's mid-session
    half-spread, so the comparison isolates the *skew / clamp logic*, not who quotes tighter.
    Bootstrap Sharpe CIs (on block-summed P&L) come from the desk's shared estimator.
    """
    flow = sim.simulate_flow(world, n_steps=n_steps, dt=dt, seed=seed)
    T = n_steps * dt
    k_as = _k_for_as(world, seed)
    # Representative half-spread (AS at mid-session, zero inventory) for the flat baselines.
    half = optimal_spread(T / 2.0, gamma, world.sigma, k_as, T) / 2.0

    quoters = {
        "AS (fixed)": ASQuoter(gamma, world.sigma, k_as, T),
        "AS (adaptive vol)": AdaptiveASQuoter(gamma, world.sigma, k_as, T, dt=dt),
        "Symmetric (no skew)": SymmetricQuoter(half),
        "Inventory clamp": ClampQuoter(half, max_inventory),
    }

    rows = {}
    for name, q in quoters.items():
        led = run_market(flow, q, T=T)
        m = metrics(led)
        if sharpe_ci_bootstrap is not None:
            ci = sharpe_ci_bootstrap(daily_returns(led), n_boot=2000, periods_per_year=1, seed=seed)
            m["pnl_sharpe"] = round(ci["sharpe"], 3)
            m["pnl_sharpe_ci_lo"] = round(ci["ci_low"], 3)
            m["pnl_sharpe_ci_hi"] = round(ci["ci_high"], 3)
        rows[name] = {
            "terminal_pnl": round(m["terminal_pnl"], 2),
            "pnl_sharpe": m.get("pnl_sharpe", round(m["sharpe_step"], 4)),
            "ci_lo": m.get("pnl_sharpe_ci_lo", float("nan")),
            "ci_hi": m.get("pnl_sharpe_ci_hi", float("nan")),
            "inv_std": round(m["inv_std"], 2),
            "inv_absmax": round(m["inv_absmax"], 0),
            "n_fills": m["n_fills"],
            "n_adverse": m["n_adverse"],
        }
    out = pd.DataFrame(rows).T
    out.attrs["k_as"] = k_as
    out.attrs["half_spread"] = half
    return out


def tournament_both(**kw) -> dict[str, pd.DataFrame]:
    """Run the tournament in both worlds; returns ``{world_name: table}``."""
    return {sim.WORLD_A.name: tournament(sim.WORLD_A, **kw),
            sim.WORLD_B.name: tournament(sim.WORLD_B, **kw)}


def tournament_multiseed(world: sim.World, seeds=(0, 1, 2, 3, 4), **kw) -> pd.DataFrame:
    """P&L Sharpe per quoter across ``seeds`` — proves the ranking isn't a single-draw fluke."""
    rows = [tournament(world, seed=s, **kw)["pnl_sharpe"] for s in seeds]
    df = pd.DataFrame(rows, index=[f"seed{s}" for s in seeds])
    df.loc["mean"] = df.mean()
    return df


# --------------------------------------------------------------------------- #
# Beat-7 worked complement — does a jump-robust vol rescue the "production fix"?
# --------------------------------------------------------------------------- #
def extension_rescue(
    world: sim.World,
    n_steps: int = 60_000,
    dt: float = 0.01,
    gamma: float = 0.1,
    seed: int = 0,
) -> pd.DataFrame:
    """Three AS variants on one market: fixed sigma, naive rolling-vol, jump-robust rolling-vol.

    The base study found the article's recommended rolling realised-vol fix (``AdaptiveASQuoter``)
    collapses in the jumpy World B. This swaps in a **bipower-variation** vol estimate
    (``JumpRobustASQuoter``) and asks whether the fix is salvageable. All three face the identical
    exogenous flow, so differences are pure estimator effects.
    """
    flow = sim.simulate_flow(world, n_steps=n_steps, dt=dt, seed=seed)
    T = n_steps * dt
    k_as = _k_for_as(world, seed)
    quoters = {
        "AS (fixed sigma)": ASQuoter(gamma, world.sigma, k_as, T),
        "AS (naive RV)": AdaptiveASQuoter(gamma, world.sigma, k_as, T, dt=dt),
        "AS (jump-robust BV)": JumpRobustASQuoter(gamma, world.sigma, k_as, T, dt=dt),
    }
    rows = {}
    for name, q in quoters.items():
        led = run_market(flow, q, T=T)
        m = metrics(led)
        sharpe = m["sharpe_step"]
        ci_lo = ci_hi = float("nan")
        if sharpe_ci_bootstrap is not None:
            ci = sharpe_ci_bootstrap(daily_returns(led), n_boot=2000, periods_per_year=1, seed=seed)
            sharpe, ci_lo, ci_hi = ci["sharpe"], ci["ci_low"], ci["ci_high"]
        rows[name] = {
            "terminal_pnl": round(m["terminal_pnl"], 2),
            "pnl_sharpe": round(sharpe, 3),
            "ci_lo": round(ci_lo, 3),
            "ci_hi": round(ci_hi, 3),
            "inv_std": round(m["inv_std"], 2),
            "n_fills": m["n_fills"],
        }
    return pd.DataFrame(rows).T


def extension_rescue_multiseed(world: sim.World, seeds=(0, 1, 2, 3, 4), **kw) -> pd.DataFrame:
    """P&L Sharpe of the three AS variants across ``seeds`` (+ a mean row)."""
    rows = [extension_rescue(world, seed=s, **kw)["pnl_sharpe"] for s in seeds]
    df = pd.DataFrame(rows, index=[f"seed{s}" for s in seeds])
    df.loc["mean"] = df.mean()
    return df
