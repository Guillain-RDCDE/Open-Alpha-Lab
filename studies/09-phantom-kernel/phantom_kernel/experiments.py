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
   hold inventory". The :func:`k_ablation` then swaps only ``k`` between its two calibrations
   to prove where the World-B win lives — the experiment behind the MISATTRIBUTED stamp.

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
def kernel_gof_table(n_orders: int = 400_000, seed: int = 0, worlds=None) -> pd.DataFrame:
    """Exponential-vs-power-law on each world's order reach.

    WORLD_A should crown the exponential (it is one); WORLD_B's heavy-tailed reach should make
    the power law win, i.e. the AS form is the wrong shape — H1 falsified. WORLD_B_STRESS
    (alpha = 1.2, heavier than anything the study measured on real books) rides along as a
    labelled stress row. The verdict columns (``aic_gap``, ``V``, ``winner``) come from the
    per-observation likelihood test (each order counted once); the R^2 columns are the
    descriptive binned fits.
    """
    deltas = default_deltas()
    if worlds is None:
        worlds = (sim.WORLD_A, sim.WORLD_B, sim.WORLD_B_STRESS)
    rows = {}
    for w in worlds:
        # The same draw that generates fill_counts(w, ..., seed): one reach per order.
        reach = sim.sample_reach(w, n_orders, np.random.default_rng(seed))
        counts = sim.fill_counts(w, deltas, n_orders=n_orders, seed=seed)
        g = goodness_of_fit(reach, deltas, counts)
        rows[w.name] = {
            "r2_exp": round(g["r2_exp"], 4),
            "r2_pow": round(g["r2_pow"], 4),
            "k_exp": round(g["k_exp"], 4),
            "alpha_mle": round(g["alpha_mle"], 4),
            "aic_gap": round(g["aic_gap"], 1),     # >0 => power-law preferred (per-order test)
            "ll_per_order": round(g["ll_per_obs"], 4),
            "V": round(g["V"], 1),
            "winner": g["winner"],
        }
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# 2b. The phantom parameter — a static k while the true one drifts (H1)
# --------------------------------------------------------------------------- #
def k_instability(seed: int = 0, n_steps: int = 60_000, dt: float = 0.01) -> dict:
    """Four intraday regimes with k spanning 4x; the static fit's spread error per regime.

    Two parameterisations of the *same* mis-calibration, because the spread error depends on
    how much non-k spread surrounds the k-term:

    * **headline** — the tournament's own session (``T = n_steps * dt``), evaluated
      mid-session, i.e. the configuration every other number in this study trades at;
    * **bound_T1** — ``T = 1``, where the k-term is essentially the whole spread. This is the
      worst case by construction (it maximises the k-error's share) and is quoted only as a
      labelled upper bound.
    """
    deltas = default_deltas()
    k_values = np.array([0.3, 0.6, 0.9, 1.2])   # 4x spread, the article's admitted range
    T = n_steps * dt
    headline = static_k_spread_error(k_values, deltas, n_orders=200_000, seed=seed,
                                     horizon=T, eval_t=T / 2.0)
    bound = static_k_spread_error(k_values, deltas, n_orders=200_000, seed=seed,
                                  horizon=1.0, eval_t=0.0)
    out = dict(headline)            # headline keys at top level (back-compatible shape)
    out["bound_T1"] = {k: bound[k] for k in
                       ("horizon", "eval_t", "spread_pct_error_per_regime",
                        "max_abs_spread_pct_error")}
    return out


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
# 3b. The k-ablation — the experiment the MISATTRIBUTED stamp rests on
# --------------------------------------------------------------------------- #
def k_ablation(
    world: sim.World = sim.WORLD_B,
    k_textbook: float = 0.6,
    seeds=(0, 1, 2, 3, 4),
    n_steps: int = 60_000,
    dt: float = 0.01,
    gamma: float = 0.1,
    base_seed: int = 0,
) -> pd.DataFrame:
    """Same AS machinery, same World-B flow — only ``k`` changes. Identical everything else.

    The two values are the two calibrations the study itself puts on the table: the textbook
    ``k`` (0.6, the rate World A is built on) and the phantom ``k`` a practitioner would fit
    on World B's own fills. If the closed-form spread width were where AS's World-B win lives,
    swapping one for the other (a ~2x move in the arrival parameter, hence a materially
    different quoted width) would move the P&L materially. If the win lives in the k-free
    inventory skew, it barely moves. This is the ablation behind the MISATTRIBUTED stamp.
    """
    k_fit = _k_for_as(world, base_seed)
    labels = {f"k={k_textbook:.3f} (textbook)": k_textbook,
              f"k={k_fit:.3f} (fitted on B)": k_fit}
    T = n_steps * dt
    rows = {}
    for s in seeds:
        flow = sim.simulate_flow(world, n_steps=n_steps, dt=dt, seed=s)
        row = {}
        for label, k in labels.items():
            led = run_market(flow, ASQuoter(gamma, world.sigma, k, T), T=T)
            m = metrics(led)
            sharpe = m["sharpe_step"]
            if sharpe_ci_bootstrap is not None:
                ci = sharpe_ci_bootstrap(daily_returns(led), n_boot=2000, periods_per_year=1, seed=s)
                sharpe = ci["sharpe"]
            row[f"sharpe {label}"] = round(sharpe, 3)
            row[f"pnl {label}"] = round(m["terminal_pnl"], 1)
        rows[f"seed{s}"] = row
    df = pd.DataFrame(rows).T
    df.loc["mean"] = df.mean()
    df.attrs["k_textbook"] = k_textbook
    df.attrs["k_fit"] = k_fit
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
