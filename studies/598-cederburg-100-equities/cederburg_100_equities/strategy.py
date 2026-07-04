"""The lifecycle allocation horse race — Study 598 (Cederburg 100% Equities).

The claim under test (Anarkulova, Cederburg & O'Doherty 2023, "Beyond the
Status Quo"): a lifecycle invested **100% in stocks — half domestic, half
international — for one's entire life** beats stock/bond balanced strategies
and target-date glidepaths on BOTH retirement wealth AND ruin risk.

Strategies (weights [w_US, w_INTL] per lifecycle year; bonds get the rest):

- ``alleq``      — 100% equity, 50/50 domestic/international, for life
                   (the paper's headline strategy).
- ``alleq_dom``  — 100% domestic equity for life (the pure-tape variant —
                   no simulated international leg anywhere).
- ``s6040``      — static 60/40 stocks/bonds, annually rebalanced (the
                   balanced benchmark; domestic equity leg).
- ``tdf``        — a target-date glidepath: 90% equity until age 40, sliding
                   to 45% at retirement (65) and 30% by age 80, flat after;
                   the equity leg is split 50/50 dom/intl like ``alleq`` so
                   TDF-vs-alleq is a pure bonds+glidepath contrast.
- ``ageinbonds`` — "your age in bonds" (equity = (100 − age)%, domestic),
                   the folk benchmark (robustness).
- ``tdf_dom``    — the TDF with a domestic-only equity leg (robustness).

Lifecycle mechanics (Cederburg's design, simplified and stated): save for
**40 years** (age 25→65), contributing **1 unit of real consumption per year**
at the start of each year (flat real earnings — a stated simplification of
the paper's earnings process); retire at 65 and withdraw **4% of the wealth
at retirement**, fixed in real terms, for **30 years** (age 65→95, a fixed
horizon instead of the paper's mortality tables — stated). **Ruin** =
depletion before age 95. Weights are a deterministic function of age, fixed
at the start of the lifecycle: the year-*j* target is set at the end of year
*j−1* (the one clean execution lag; the only signal is the calendar). Costs:
``COST_BPS`` one-way × traded value (contribution deployment, rebalancing
turnover, withdrawal sales) plus an ``ER_BPS`` annual expense ratio × NAV —
index-fund reality for all strategies.

Inference: monthly-start 70-year cohorts overlap by up to 839 months, so mean
differences carry a Newey-West HAC t with the bandwidth forced to the full
840-month overlap; distribution stats (ruin rates, percentiles) carry an
outer circular block-bootstrap of the tape. The Cederburg-style bootstrap
(``block_lifetimes``) resamples 120-month blocks into 840-month lifetimes —
the paper's own machinery. The synthetic control averages over >= 20
independent seeded worlds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as _data

SAVE_YEARS = 40
RET_YEARS = 30
HORIZON = SAVE_YEARS + RET_YEARS          # 70 lifecycle years
WR = 0.04                                  # of wealth at retirement, real
COST_BPS = 10.0                            # one-way, x traded value
ER_BPS = 5.0                               # annual expense ratio, x NAV
STRATEGIES = ("alleq", "alleq_dom", "s6040", "tdf")


# ---------------------------------------------------------------------------
# Weight schedules — (HORIZON, 2) arrays of [w_US, w_INTL]
# ---------------------------------------------------------------------------

def weights_for(name: str, horizon: int = HORIZON) -> np.ndarray:
    """Weight path [w_US, w_INTL] per lifecycle year (age 25 + j)."""
    j = np.arange(horizon)
    age = 25 + j

    def _split(eq: np.ndarray, intl_share: float) -> np.ndarray:
        w = np.empty((len(eq), 2))
        w[:, 0] = eq * (1.0 - intl_share)
        w[:, 1] = eq * intl_share
        return w

    def _tdf_eq() -> np.ndarray:
        eq = np.empty(horizon)
        eq[age < 40] = 0.90
        m1 = (age >= 40) & (age <= 65)
        eq[m1] = 0.90 + (0.45 - 0.90) * (age[m1] - 40) / 25.0
        m2 = (age > 65) & (age <= 80)
        eq[m2] = 0.45 + (0.30 - 0.45) * (age[m2] - 65) / 15.0
        eq[age > 80] = 0.30
        return eq

    table = {
        "alleq": lambda: _split(np.ones(horizon), 0.5),
        "alleq_dom": lambda: _split(np.ones(horizon), 0.0),
        "s6040": lambda: _split(np.full(horizon, 0.60), 0.0),
        "tdf": lambda: _split(_tdf_eq(), 0.5),
        "tdf_dom": lambda: _split(_tdf_eq(), 0.0),
        "ageinbonds": lambda: _split(
            np.clip((100.0 - age) / 100.0, 0.0, 1.0), 0.0),
    }
    if name not in table:
        raise ValueError(f"unknown strategy {name!r}")
    return table[name]()


# ---------------------------------------------------------------------------
# Cohort construction
# ---------------------------------------------------------------------------

def cohort_year_returns(
    us: np.ndarray, intl: np.ndarray, bd: np.ndarray,
    horizon: int = HORIZON, step: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(C, horizon) matrices of annual gross real returns from a monthly tape.

    Cohort i starts at month ``i*step``; its lifecycle year j spans months
    ``[start + 12j, start + 12j + 12)``. Built from cumulative log returns.
    """
    n = len(us)
    starts = np.arange(0, n - 12 * horizon + 1, step)
    a = starts[:, None] + 12 * np.arange(horizon)[None, :]
    b = a + 12
    out = []
    for x in (us, intl, bd):
        lx = np.concatenate([[0.0], np.cumsum(np.log1p(x))])
        out.append(np.exp(lx[b] - lx[a]))
    return out[0], out[1], out[2], starts


def annual_cohorts(
    us: np.ndarray, intl: np.ndarray, bd: np.ndarray, horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(C, horizon) gross-return matrices from an *annual* tape (synthetic)."""
    n = len(us)
    starts = np.arange(0, n - horizon + 1)
    idx = starts[:, None] + np.arange(horizon)[None, :]
    return 1.0 + us[idx], 1.0 + intl[idx], 1.0 + bd[idx]


# ---------------------------------------------------------------------------
# The lifecycle simulator — vectorised across cohorts / draws
# ---------------------------------------------------------------------------

def simulate(
    gu: np.ndarray,
    gi: np.ndarray,
    gb: np.ndarray,
    weights: np.ndarray,
    wr: float = WR,
    cost_bps: float = COST_BPS,
    er_bps: float = ER_BPS,
    save_years: int = SAVE_YEARS,
) -> dict:
    """Run one weight schedule over all cohorts, savings + retirement.

    ``gu``/``gi``/``gb``: (C, H) annual gross real returns (US equity, intl
    equity, bonds). ``weights``: (H, 2) [w_US, w_INTL]. Savings phase
    (years 0..save_years-1): contribute 1.0 real at the start of each year.
    Retirement: withdraw ``wr`` × wealth-at-retirement (fixed real) at the
    start of each year. Costs: ``cost_bps`` one-way × traded value plus
    ``er_bps`` annual × NAV. Returns wealth at retirement, terminal wealth
    (units of the annual contribution), ruin flag and failure year.
    """
    c, h = gu.shape
    cost = cost_bps * 1e-4
    er = er_bps * 1e-4

    vu = np.zeros(c)
    vi = np.zeros(c)
    vb = np.zeros(c)
    alive = np.ones(c, dtype=bool)
    fail_year = np.full(c, -1, dtype=int)
    wret = np.zeros(c)
    wd = np.zeros(c)

    for j in range(h):
        wealth = vu + vi + vb
        if j < save_years:
            wealth = wealth + 1.0                       # real contribution
        else:
            if j == save_years:
                wret = wealth.copy()                    # pot at retirement
                wd = wr * wret                          # fixed real withdrawal
            newly_dead = alive & (wealth <= wd)
            fail_year[newly_dead] = j - save_years
            alive &= ~newly_dead
            wealth = np.where(alive, wealth - wd, 0.0)
        # rebalance to the year-j target (set at the end of year j-1)
        tu = weights[j, 0] * wealth
        ti = weights[j, 1] * wealth
        tb = wealth - tu - ti
        traded = np.abs(tu - vu) + np.abs(ti - vi) + np.abs(tb - vb)
        wealth = np.maximum(wealth - cost * traded - er * wealth, 0.0)
        vu = weights[j, 0] * wealth * gu[:, j]
        vi = weights[j, 1] * wealth * gi[:, j]
        vb = (wealth - weights[j, 0] * wealth - weights[j, 1] * wealth) * gb[:, j]

    terminal = np.where(alive, vu + vi + vb, 0.0)
    return {"terminal": terminal, "wret": wret, "success": alive.copy(),
            "fail_year": fail_year}


def run_all(
    gu: np.ndarray, gi: np.ndarray, gb: np.ndarray, wr: float = WR,
    strategies: tuple[str, ...] = STRATEGIES, cost_bps: float = COST_BPS,
    er_bps: float = ER_BPS,
) -> dict[str, dict]:
    return {s: simulate(gu, gi, gb, weights_for(s, gu.shape[1]), wr=wr,
                        cost_bps=cost_bps, er_bps=er_bps) for s in strategies}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def hac_tstat(diff: np.ndarray, lags: int) -> float:
    """Newey-West HAC t for mean(diff) with an explicit Bartlett bandwidth.

    Monthly-start 70-year cohorts overlap by up to 839 months; ``lags`` is
    forced to the full overlap by the caller (the automatic selector would be
    absurdly short here).
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
        "ruin_pct": 100.0 * float((~res["success"]).mean()),
        "n_fail": int((~res["success"]).sum()),
        "mean": float(t.mean()),
        "median": float(np.median(t)),
        "p05": float(np.quantile(t, 0.05)),
        "wret_mean": float(res["wret"].mean()),
        "wret_median": float(np.median(res["wret"])),
        "n_cohorts": int(len(t)),
    }


# ---------------------------------------------------------------------------
# Cederburg-style block bootstrap — resampled 70-year lifetimes
# ---------------------------------------------------------------------------

def block_lifetimes(
    us: np.ndarray, intl: np.ndarray, bd: np.ndarray,
    n_draws: int = 2000, block: int = 120, horizon: int = HORIZON,
    seed: int = 598,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The paper's machinery: bootstrap lifetimes of ``horizon`` years.

    Each draw stitches circular ``block``-month blocks (joint across the three
    assets — cross-correlations preserved) into an 840-month lifetime, then
    aggregates to annual gross returns. Returns three (n_draws, horizon)
    matrices. All strategies are evaluated on the SAME draws (paired).
    """
    rng = np.random.default_rng(seed)
    n = len(us)
    months = 12 * horizon
    n_blocks = int(np.ceil(months / block))
    starts = rng.integers(0, n, size=(n_draws, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(
        n_draws, -1)[:, :months] % n
    out = []
    for x in (us, intl, bd):
        lx = np.log1p(x)
        out.append(np.exp(lx[idx].reshape(n_draws, horizon, 12).sum(axis=2)))
    return out[0], out[1], out[2]


# ---------------------------------------------------------------------------
# Outer block bootstrap — tape-level uncertainty of the cohort metrics
# ---------------------------------------------------------------------------

def outer_bootstrap(
    us: np.ndarray, intl: np.ndarray, bd: np.ndarray,
    pairs: tuple[tuple[str, str], ...] = (("alleq", "s6040"),
                                          ("alleq", "tdf"),
                                          ("alleq_dom", "s6040")),
    wr: float = WR,
    n_boot: int = 600,
    block: int = 120,
    horizon: int = HORIZON,
    seed: int = 598,
) -> dict:
    """Bootstrap distribution of cohort-panel strategy differences.

    Circular block bootstrap of the joint monthly tape (``block``-month
    blocks — within-decade dynamics preserved, cross-decade mean reversion
    destroyed; stated, not hidden). For each replicate the full cohort panel
    is rebuilt and each pair's differences recorded:

    - ``dmean``  — mean terminal-wealth difference
    - ``dmed``   — median terminal-wealth difference
    - ``druin``  — ruin-rate difference (pct points)

    Returns ``{pair: {metric: np.array(n_boot)}}``.
    """
    rng = np.random.default_rng(seed)
    n = len(us)
    n_blocks = int(np.ceil(n / block))
    names = sorted({s for p in pairs for s in p})
    out = {p: {"dmean": np.empty(n_boot), "dmed": np.empty(n_boot),
               "druin": np.empty(n_boot)} for p in pairs}

    for it in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        gu, gi, gb, _ = cohort_year_returns(us[idx], intl[idx], bd[idx],
                                            horizon=horizon)
        res = {s: simulate(gu, gi, gb, weights_for(s, horizon), wr=wr)
               for s in names}
        for p in pairs:
            a, b = p
            ta, tb = res[a]["terminal"], res[b]["terminal"]
            out[p]["dmean"][it] = ta.mean() - tb.mean()
            out[p]["dmed"][it] = np.median(ta) - np.median(tb)
            out[p]["druin"][it] = 100.0 * ((~res[a]["success"]).mean()
                                           - (~res[b]["success"]).mean())
    return out


def ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1.0 - level) / 2.0
    return float(np.quantile(x, a)), float(np.quantile(x, 1.0 - a))


# ---------------------------------------------------------------------------
# Synthetic control — >= 20 independent seeded worlds per setting
# ---------------------------------------------------------------------------

def control_premium_effect(
    premium: float,
    wr: float = WR,
    n_seeds: int = 20,
    n_years: int = 200,
    horizon: int = HORIZON,
    base_seed: int = 598_000,
    **world_kwargs,
) -> dict:
    """The all-equity-vs-60/40 detector on synthetic worlds.

    For each of ``n_seeds`` independent worlds, race ``alleq`` against
    ``s6040`` and record the mean terminal-wealth difference and the
    ruin-rate difference. Returns across-seed means and one-sample t-stats
    (independent worlds, so a plain t is valid). The exact null is
    ``premium = 0``: equal arithmetic means make the expected terminal wealth
    of every weight schedule identical, so ``dmean`` must read ~0 there.
    """
    dmean = np.empty(n_seeds)
    druin = np.empty(n_seeds)
    for k in range(n_seeds):
        world, _ = _data.synthetic_world(n_years=n_years, premium=premium,
                                         seed=base_seed + k, **world_kwargs)
        gu, gi, gb = annual_cohorts(world["US"].to_numpy(),
                                    world["INTL"].to_numpy(),
                                    world["BD"].to_numpy(), horizon=horizon)
        a = simulate(gu, gi, gb, weights_for("alleq", horizon), wr=wr)
        b = simulate(gu, gi, gb, weights_for("s6040", horizon), wr=wr)
        dmean[k] = a["terminal"].mean() - b["terminal"].mean()
        druin[k] = 100.0 * ((~a["success"]).mean() - (~b["success"]).mean())

    def _t(x):
        s = x.std(ddof=1)
        return float(x.mean() / (s / np.sqrt(len(x)))) if s > 0 else float("nan")

    return {
        "dmean_mean": float(dmean.mean()), "dmean_t": _t(dmean),
        "druin_mean": float(druin.mean()), "druin_t": _t(druin),
        "n_seeds": n_seeds, "premium": premium, "wr": wr,
    }
