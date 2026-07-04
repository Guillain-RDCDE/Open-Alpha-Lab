"""The retirement glidepath engine — Study 596 (Bond Tent / Rising Equity).

The claim under test (Kitces & Pfau 2014): a retiree should hold **more bonds
on the day of retirement** and *re-raise* the equity weight through retirement
(a "rising equity glidepath", the back half of the "bond tent"). The pitch:
sequence-of-returns risk is concentrated in the first retirement decade, so
carrying less equity exactly then — and more later — cuts the failure risk of
a 4%-rule retirement without giving up the long-run equity engine.

Strategies (equity weight as a function of years-since-retirement, 30y):

- ``s6040``      — static 60/40, annually rebalanced (the planner benchmark).
- ``s5050``      — static 50/50 (the *matched-average* control: the rising
                   tent below averages exactly 50% equity).
- ``s4060``      — static 40/60 ("just hold more bonds forever").
- ``rising``     — the bond tent's retirement half: 30% -> 70% equity, linear
                   over the 30 years (average 50%).
- ``declining``  — the mirror image, 70% -> 30% (average 50%): identical
                   average allocation, *reversed timing*. Rising vs declining
                   is the pure sequence-TIMING test.

Decomposition identity (the third axis):

    rising - s6040  =  (rising - s5050)   [shape, at matched average equity]
                     + (s5050 - s6040)    [just holding less equity]

Mechanics: 30 annual steps per cohort, vectorised across all monthly-start
cohorts. Withdrawal of ``wr`` (real, of initial wealth = 1) at the START of
each year, then rebalance to the year's target weight, then the year's real
returns. Weights are a deterministic function of the year index, fixed at
retirement — the year-j target is set at the end of year j-1 (one clean lag;
no information beyond the calendar is used). Costs: one-way ``cost_bps`` x
traded value (withdrawal sales + rebalancing turnover).

Inference: cohorts overlap massively (monthly starts, 360-month windows), so
mean differences carry a Newey-West HAC t with the bandwidth forced to the
full 360-month overlap; distribution stats (success rates, 5th percentiles)
carry circular block-bootstrap CIs (120-month blocks on the joint monthly
tape). The synthetic control averages over >= 20 independent seeded worlds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as _data

HORIZON = 30                 # retirement years
COST_BPS = 10.0              # one-way, x traded value
STRATEGIES = ("s6040", "s5050", "s4060", "rising", "declining")


# ---------------------------------------------------------------------------
# Weight schedules
# ---------------------------------------------------------------------------

def weights_for(name: str, horizon: int = HORIZON) -> np.ndarray:
    """Equity-weight path (length ``horizon``) for a named strategy."""
    j = np.arange(horizon) / max(horizon - 1, 1)
    table = {
        "s6040": np.full(horizon, 0.60),
        "s5050": np.full(horizon, 0.50),
        "s4060": np.full(horizon, 0.40),
        "rising": 0.30 + 0.40 * j,               # 30% -> 70%, avg 50%
        "declining": 0.70 - 0.40 * j,            # 70% -> 30%, avg 50%
        # robustness variants
        "rising_3060": 0.30 + 0.30 * j,          # Kitces-Pfau 30 -> 60
        "rising_fast": np.minimum(0.30 + 0.40 * np.arange(horizon) / 14.0, 0.70),
        "rising_2080": 0.20 + 0.60 * j,          # a deeper tent
    }
    if name not in table:
        raise ValueError(f"unknown strategy {name!r}")
    return table[name]


# ---------------------------------------------------------------------------
# Cohort construction — vectorised annual gross returns per cohort-year
# ---------------------------------------------------------------------------

def cohort_year_returns(
    eq: np.ndarray, bd: np.ndarray, horizon: int = HORIZON, step: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(C, horizon) matrices of annual gross real returns from a monthly tape.

    Cohort i starts at month ``i*step``; its year j spans months
    ``[start + 12j, start + 12j + 12)``. Built from cumulative log returns,
    so the whole panel is two vectorised gathers.
    """
    le = np.concatenate([[0.0], np.cumsum(np.log1p(eq))])
    lb = np.concatenate([[0.0], np.cumsum(np.log1p(bd))])
    n = len(eq)
    starts = np.arange(0, n - 12 * horizon + 1, step)
    a = starts[:, None] + 12 * np.arange(horizon)[None, :]
    b = a + 12
    return np.exp(le[b] - le[a]), np.exp(lb[b] - lb[a]), starts


def annual_cohorts(
    eq_annual: np.ndarray, bd_annual: np.ndarray, horizon: int = HORIZON
) -> tuple[np.ndarray, np.ndarray]:
    """(C, horizon) gross-return matrices from an *annual* tape (synthetic)."""
    n = len(eq_annual)
    starts = np.arange(0, n - horizon + 1)
    idx = starts[:, None] + np.arange(horizon)[None, :]
    return 1.0 + eq_annual[idx], 1.0 + bd_annual[idx]


# ---------------------------------------------------------------------------
# The simulator — vectorised across cohorts
# ---------------------------------------------------------------------------

def simulate(
    ge: np.ndarray,
    gb: np.ndarray,
    weights: np.ndarray,
    wr: float = 0.04,
    cost_bps: float = COST_BPS,
) -> dict:
    """Run one weight schedule over all cohorts.

    ``ge``/``gb``: (C, H) annual gross real returns. ``weights``: (H,) equity
    weights. ``wr``: real withdrawal (fraction of initial wealth = 1) taken at
    the START of each year. Returns terminal wealth (real multiple of initial),
    the success flag (never depleted) and the failure year (-1 if none).
    """
    c, h = ge.shape
    cost = cost_bps * 1e-4
    terminal = np.zeros(c)
    fail_year = np.full(c, -1, dtype=int)
    alive = np.ones(c, dtype=bool)

    # Year 0: withdraw from initial wealth 1, deploy at the initial weight.
    w0 = weights[0]
    wealth = np.full(c, 1.0 - wr)
    if wr >= 1.0:
        alive[:] = False
    veq = wealth * w0
    vbd = wealth * (1.0 - w0)
    veq, vbd = veq * ge[:, 0], vbd * gb[:, 0]

    for j in range(1, h):
        wealth = veq + vbd
        # withdrawal (real, fixed) at the start of year j
        newly_dead = alive & (wealth <= wr)
        fail_year[newly_dead] = j
        alive &= ~newly_dead
        wealth = np.where(alive, wealth - wr, 0.0)
        # rebalance to the year-j target (set at the end of year j-1),
        # paying one-way costs on withdrawal sales + rebalancing turnover
        wj = weights[j]
        scale = np.where(veq + vbd > 0, wealth / np.maximum(veq + vbd, 1e-300), 0.0)
        veq_after = veq * scale
        turnover = np.abs(wj * wealth - veq_after)
        wealth = np.maximum(wealth - cost * (wr * alive + turnover), 0.0)
        veq = wj * wealth * ge[:, j]
        vbd = (1.0 - wj) * wealth * gb[:, j]

    terminal = np.where(alive, veq + vbd, 0.0)
    return {"terminal": terminal, "success": alive.copy(), "fail_year": fail_year}


def run_all(
    ge: np.ndarray, gb: np.ndarray, wr: float,
    strategies: tuple[str, ...] = STRATEGIES, cost_bps: float = COST_BPS,
) -> dict[str, dict]:
    return {s: simulate(ge, gb, weights_for(s, ge.shape[1]), wr=wr,
                        cost_bps=cost_bps) for s in strategies}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def hac_tstat(diff: np.ndarray, lags: int) -> float:
    """Newey-West HAC t for mean(diff) with an explicit Bartlett bandwidth.

    Monthly-start 30-year cohorts overlap by up to 359 months, so the serial
    correlation is enormous; we force ``lags`` to the full overlap instead of
    the usual automatic selector (which would be absurdly short here).
    """
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n < 4:
        return float("nan")
    lags = int(min(lags, n - 1))
    mu = diff.mean()
    e = diff - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 1e-300) / n)
    return float(mu / se)


def summarize(res: dict) -> dict:
    """Per-strategy cohort summary."""
    t = res["terminal"]
    return {
        "success_pct": 100.0 * float(res["success"].mean()),
        "n_fail": int((~res["success"]).sum()),
        "mean": float(t.mean()),
        "median": float(np.median(t)),
        "p05": float(np.quantile(t, 0.05)),
        "worst": float(t.min()),
        "n_cohorts": int(len(t)),
    }


def safemax(
    ge: np.ndarray, gb: np.ndarray, name: str,
    lo: float = 0.005, hi: float = 0.12, tol: float = 1e-4,
    cost_bps: float = COST_BPS,
) -> float:
    """Highest real withdrawal rate at which EVERY cohort survives 30 years."""
    w = weights_for(name, ge.shape[1])
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if simulate(ge, gb, w, wr=mid, cost_bps=cost_bps)["success"].all():
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# Circular block bootstrap on the joint monthly tape
# ---------------------------------------------------------------------------

def block_bootstrap(
    eq: np.ndarray,
    bd: np.ndarray,
    pairs: tuple[tuple[str, str], ...] = (
        ("rising", "s6040"), ("rising", "s5050"), ("rising", "declining"),
        ("s5050", "s6040"),
    ),
    wr_terminal: float = 0.04,
    wr_success: float = 0.05,
    n_boot: int = 1000,
    block: int = 120,
    horizon: int = HORIZON,
    cost_bps: float = COST_BPS,
    seed: int = 596,
) -> dict:
    """Bootstrap distribution of strategy differences.

    Circular block bootstrap (joint EQ/BD rows, ``block``-month blocks —
    within-decade dynamics preserved, cross-decade mean reversion destroyed;
    that is stated, not hidden). For each replicate the full cohort panel is
    rebuilt and each pair's differences recorded:

    - ``dmean``    — mean terminal wealth diff at ``wr_terminal``
    - ``dp05``     — 5th-percentile terminal wealth diff at ``wr_terminal``
    - ``dsucc``    — success-rate diff (pct points) at ``wr_success``

    Returns ``{pair: {metric: np.array(n_boot)}}``.
    """
    rng = np.random.default_rng(seed)
    n = len(eq)
    n_blocks = int(np.ceil(n / block))
    names = sorted({s for p in pairs for s in p})
    out = {p: {"dmean": np.empty(n_boot), "dp05": np.empty(n_boot),
               "dsucc": np.empty(n_boot)} for p in pairs}

    for it in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        ge, gb, _ = cohort_year_returns(eq[idx], bd[idx], horizon=horizon)
        res_t = {s: simulate(ge, gb, weights_for(s, horizon), wr=wr_terminal,
                             cost_bps=cost_bps) for s in names}
        res_s = {s: simulate(ge, gb, weights_for(s, horizon), wr=wr_success,
                             cost_bps=cost_bps) for s in names}
        for p in pairs:
            a, b = p
            ta, tb = res_t[a]["terminal"], res_t[b]["terminal"]
            out[p]["dmean"][it] = ta.mean() - tb.mean()
            out[p]["dp05"][it] = np.quantile(ta, 0.05) - np.quantile(tb, 0.05)
            out[p]["dsucc"][it] = 100.0 * (res_s[a]["success"].mean()
                                           - res_s[b]["success"].mean())
    return out


def ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1.0 - level) / 2.0
    return float(np.quantile(x, a)), float(np.quantile(x, 1.0 - a))


# ---------------------------------------------------------------------------
# Synthetic control — >= 20 independent seeded worlds per setting
# ---------------------------------------------------------------------------

def control_shape_effect(
    reversion: float,
    wr: float,
    n_seeds: int = 20,
    n_years: int = 200,
    horizon: int = HORIZON,
    cost_bps: float = COST_BPS,
    base_seed: int = 596_000,
    **world_kwargs,
) -> dict:
    """The pure-timing detector on synthetic worlds: rising vs declining.

    For each of ``n_seeds`` independent worlds, run the rising and declining
    schedules (same 50% average equity) and record the 5th-percentile terminal
    wealth difference and the success-rate difference. Returns the across-seed
    means and one-sample t-stats (n_seeds independent worlds, so a plain t is
    valid). The exact null is ``reversion = 0, wr = 0`` — with no withdrawals
    and i.i.d. returns, a weight path and its mirror give the *same*
    terminal-wealth distribution, so the detector must read ~0 there.
    """
    dp05 = np.empty(n_seeds)
    dsucc = np.empty(n_seeds)
    for k in range(n_seeds):
        world, _ = _data.synthetic_world(
            n_years=n_years, reversion=reversion, seed=base_seed + k,
            **world_kwargs)
        ge, gb = annual_cohorts(world["EQ"].to_numpy(),
                                world["BD"].to_numpy(), horizon=horizon)
        up = simulate(ge, gb, weights_for("rising", horizon), wr=wr,
                      cost_bps=cost_bps)
        dn = simulate(ge, gb, weights_for("declining", horizon), wr=wr,
                      cost_bps=cost_bps)
        dp05[k] = np.quantile(up["terminal"], 0.05) - np.quantile(dn["terminal"], 0.05)
        dsucc[k] = 100.0 * (up["success"].mean() - dn["success"].mean())

    def _t(x):
        s = x.std(ddof=1)
        return float(x.mean() / (s / np.sqrt(len(x)))) if s > 0 else float("nan")

    return {
        "dp05_mean": float(dp05.mean()), "dp05_t": _t(dp05),
        "dsucc_mean": float(dsucc.mean()), "dsucc_t": _t(dsucc),
        "n_seeds": n_seeds, "reversion": reversion, "wr": wr,
    }
