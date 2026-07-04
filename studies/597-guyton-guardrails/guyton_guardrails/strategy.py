"""The dynamic-withdrawal engine — Study 597 (Guyton-Klinger Guardrails).

The claim under test (Guyton 2004; Guyton & Klinger 2006): a retiree who
adjusts withdrawals with **decision rules** — freeze the inflation raise after
a losing year, cut spending 10% when the current withdrawal rate breaches an
upper guardrail, raise it 10% below a lower guardrail — can safely START at a
5%+ initial withdrawal rate, where the fixed (Bengen) rule at 5% fails a
quarter of historical retirements. Sibling study 173 tested the FIXED 4% rule;
this study tests the DYNAMIC variant.

Rules implemented (Guyton & Klinger 2006, applied at each annual withdrawal):

- **Withdrawal (inflation) rule** — the withdrawal grows with last year's
  realised CPI inflation, capped at +6%; the raise is **skipped** ("frozen")
  when last year's nominal portfolio return was negative AND the current
  withdrawal rate is above the initial rate.
- **Capital-preservation rule** — if the current withdrawal rate exceeds
  1.2 x the initial rate, cut the withdrawal by 10% (applied only while more
  than 15 years remain, per the paper; a cut-always variant is a robustness
  knob).
- **Prosperity rule** — if the current withdrawal rate falls below 0.8 x the
  initial rate, raise the withdrawal by 10%.

The simulation runs in **nominal** space (the rules are defined on nominal
amounts) and every outcome is deflated by realised CPI back to real. With all
rules disabled the engine is exactly the Bengen constant-real rule —
algebraically identical to sibling 596's real-space simulator (a printed
machinery identity, and the fixed-4% success rate reproduces 596's number).

Mechanics: 30 annual steps per cohort, vectorised across all monthly-start
cohorts. Withdrawal at the START of each year, then rebalance to the target
equity weight, then the year's nominal returns. One clean execution lag: every
rule input (last year's inflation, last year's portfolio return, the current
portfolio value) is known at the start-of-year withdrawal date — no
information beyond the completed calendar year is used. Costs: one-way
``cost_bps`` x traded value (withdrawal sales + rebalancing turnover).

Inference: cohorts overlap massively (monthly starts, 360-month windows), so
mean differences carry a Newey-West HAC t with the bandwidth forced to the
full 360-month overlap; distribution stats (success rates, medians) carry
circular block-bootstrap CIs (120-month blocks on the joint monthly tape).
The synthetic control averages over >= 20 independent seeded worlds.
"""

from __future__ import annotations

import numpy as np

from . import data as _data

HORIZON = 30                 # retirement years
COST_BPS = 10.0              # one-way, x traded value
EQ_W = 0.60                  # headline equity weight (65% GK variant = knob)

# Rule presets: (freeze, infl_cap, guardrails, cp_first15_only, lower, upper,
#                cut_mult, raise_mult)
PRESETS: dict[str, dict] = {
    # Bengen constant-real rule: plain CPI adjustment, nothing else
    "fixed": dict(freeze=False, infl_cap=None, guardrails=False,
                  cp_first15_only=True, lower=0.8, upper=1.2,
                  cut=0.90, raise_=1.10),
    # full Guyton-Klinger 2006
    "gk": dict(freeze=True, infl_cap=0.06, guardrails=True,
               cp_first15_only=True, lower=0.8, upper=1.2,
               cut=0.90, raise_=1.10),
    # decomposition / robustness variants
    "gk_cut_always": dict(freeze=True, infl_cap=0.06, guardrails=True,
                          cp_first15_only=False, lower=0.8, upper=1.2,
                          cut=0.90, raise_=1.10),
    "gk_noraise": dict(freeze=True, infl_cap=0.06, guardrails=True,
                       cp_first15_only=True, lower=0.0, upper=1.2,
                       cut=0.90, raise_=1.10),
    "gk_freeze_only": dict(freeze=True, infl_cap=0.06, guardrails=False,
                           cp_first15_only=True, lower=0.8, upper=1.2,
                           cut=0.90, raise_=1.10),
    # degenerate identity: guardrails at (0, inf), no freeze, no cap == fixed
    "gk_wide": dict(freeze=False, infl_cap=None, guardrails=True,
                    cp_first15_only=False, lower=0.0, upper=np.inf,
                    cut=0.90, raise_=1.10),
}


# ---------------------------------------------------------------------------
# Cohort construction — vectorised annual gross returns per cohort-year
# ---------------------------------------------------------------------------

def cohort_year_returns(
    eq: np.ndarray, bd: np.ndarray, infl: np.ndarray,
    horizon: int = HORIZON, step: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(C, horizon) matrices of annual gross nominal returns + gross inflation.

    Cohort i starts at month ``i*step``; its year j spans months
    ``[start + 12j, start + 12j + 12)``. Built from cumulative log returns,
    so the whole panel is three vectorised gathers.
    """
    le = np.concatenate([[0.0], np.cumsum(np.log1p(eq))])
    lb = np.concatenate([[0.0], np.cumsum(np.log1p(bd))])
    li = np.concatenate([[0.0], np.cumsum(np.log1p(infl))])
    n = len(eq)
    starts = np.arange(0, n - 12 * horizon + 1, step)
    a = starts[:, None] + 12 * np.arange(horizon)[None, :]
    b = a + 12
    return (np.exp(le[b] - le[a]), np.exp(lb[b] - lb[a]),
            np.exp(li[b] - li[a]), starts)


def annual_cohorts(
    frame, horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(C, horizon) gross matrices from an *annual* ['EQ','BD','INFL'] frame."""
    eq = frame["EQ"].to_numpy()
    bd = frame["BD"].to_numpy()
    infl = frame["INFL"].to_numpy()
    n = len(eq)
    starts = np.arange(0, n - horizon + 1)
    idx = starts[:, None] + np.arange(horizon)[None, :]
    return 1.0 + eq[idx], 1.0 + bd[idx], 1.0 + infl[idx]


# ---------------------------------------------------------------------------
# The simulator — vectorised across cohorts
# ---------------------------------------------------------------------------

def simulate(
    ge: np.ndarray,
    gb: np.ndarray,
    gi: np.ndarray,
    wr0: float = 0.05,
    preset: str = "gk",
    eq_w: float = EQ_W,
    cost_bps: float = COST_BPS,
) -> dict:
    """Run one withdrawal policy over all cohorts.

    ``ge``/``gb``/``gi``: (C, H) annual gross nominal returns and gross CPI
    inflation. ``wr0``: initial withdrawal rate (fraction of initial wealth
    = 1, withdrawn at the START of each year). Returns real terminal wealth,
    the success flag (never depleted), the failure year, the (C, H) matrix of
    **real** income, and per-cohort rule-event counts.
    """
    cfg = PRESETS[preset]
    c, h = ge.shape
    cost = cost_bps * 1e-4

    W = np.full(c, float(wr0))          # nominal withdrawal amount
    cpi = np.ones(c)                    # CPI(level) / CPI(retirement)
    alive = np.ones(c, dtype=bool)
    income = np.zeros((c, h))           # REAL income per cohort-year
    n_cut = np.zeros(c)
    n_raise = np.zeros(c)
    n_freeze = np.zeros(c)
    fail_year = np.full(c, -1, dtype=int)

    # Year 0: withdraw wr0 from initial wealth 1, deploy at the target weight.
    income[:, 0] = wr0
    wealth = np.full(c, 1.0 - wr0)
    veq = wealth * eq_w * ge[:, 0]
    vbd = wealth * (1.0 - eq_w) * gb[:, 0]
    prev_neg = (eq_w * ge[:, 0] + (1.0 - eq_w) * gb[:, 0]) < 1.0

    for j in range(1, h):
        cpi = cpi * gi[:, j - 1]        # CPI at the start of year j
        P = veq + vbd
        infl = gi[:, j - 1] - 1.0
        cur_wr = np.where(P > 0, W / np.maximum(P, 1e-300), np.inf)

        # ---- withdrawal (inflation) rule -------------------------------
        if cfg["freeze"] or cfg["infl_cap"] is not None:
            adj = 1.0 + (np.minimum(infl, cfg["infl_cap"])
                         if cfg["infl_cap"] is not None else infl)
            frozen = (prev_neg & (cur_wr > wr0)) if cfg["freeze"] else \
                np.zeros(c, dtype=bool)
            W = np.where(frozen, W, W * adj)
            n_freeze += frozen & alive
        else:                            # plain Bengen CPI adjustment
            W = W * (1.0 + infl)

        # ---- guardrails --------------------------------------------------
        if cfg["guardrails"]:
            cur_wr = np.where(P > 0, W / np.maximum(P, 1e-300), np.inf)
            in_cp_window = (h - j) > 15 if cfg["cp_first15_only"] else True
            cut = (cur_wr > cfg["upper"] * wr0) & in_cp_window
            raise_ = (cur_wr < cfg["lower"] * wr0) & ~cut
            W = np.where(cut, cfg["cut"] * W,
                         np.where(raise_, cfg["raise_"] * W, W))
            n_cut += cut & alive
            n_raise += raise_ & alive

        # ---- withdraw ----------------------------------------------------
        newly_dead = alive & (P <= W)
        fail_year[newly_dead] = j
        income[:, j] = np.where(
            alive & ~newly_dead, W, np.where(newly_dead, P, 0.0)) / cpi
        alive = alive & ~newly_dead
        wealth = np.where(alive, P - W, 0.0)

        # ---- rebalance to the target weight, pay one-way costs -----------
        scale = np.where(P > 0, wealth / np.maximum(P, 1e-300), 0.0)
        veq_after = veq * scale
        turnover = np.abs(eq_w * wealth - veq_after)
        wealth = np.maximum(wealth - cost * (W * alive + turnover), 0.0)
        veq = eq_w * wealth * ge[:, j]
        vbd = (1.0 - eq_w) * wealth * gb[:, j]
        prev_neg = (eq_w * ge[:, j] + (1.0 - eq_w) * gb[:, j]) < 1.0

    cpi = cpi * gi[:, h - 1]
    terminal = (veq + vbd) / cpi        # REAL terminal wealth
    return {
        "terminal": terminal, "success": alive.copy(), "fail_year": fail_year,
        "income": income, "lti": income.sum(axis=1),
        "min_inc": income.min(axis=1),
        "n_cut": n_cut, "n_raise": n_raise, "n_freeze": n_freeze,
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def summarize(res: dict) -> dict:
    """Per-strategy cohort summary (all quantities REAL, initial wealth = 1)."""
    t, lti, mi = res["terminal"], res["lti"], res["min_inc"]
    return {
        "success_pct": 100.0 * float(res["success"].mean()),
        "n_fail": int((~res["success"]).sum()),
        "n_cohorts": int(len(t)),
        "term_mean": float(t.mean()), "term_median": float(np.median(t)),
        "term_p05": float(np.quantile(t, 0.05)),
        "lti_mean": float(lti.mean()), "lti_median": float(np.median(lti)),
        "lti_p05": float(np.quantile(lti, 0.05)),
        "lti_worst": float(lti.min()),
        "mininc_median": float(np.median(mi)),
        "mininc_p05": float(np.quantile(mi, 0.05)),
        "mininc_worst": float(mi.min()),
        "cuts_mean": float(res["n_cut"].mean()),
        "raises_mean": float(res["n_raise"].mean()),
        "freezes_mean": float(res["n_freeze"].mean()),
    }


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


def safemax(
    ge: np.ndarray, gb: np.ndarray, gi: np.ndarray, preset: str,
    lo: float = 0.005, hi: float = 0.12, tol: float = 1e-4,
    cost_bps: float = COST_BPS, eq_w: float = EQ_W,
) -> float:
    """Highest initial withdrawal rate at which EVERY cohort survives 30y."""
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        r = simulate(ge, gb, gi, wr0=mid, preset=preset,
                     eq_w=eq_w, cost_bps=cost_bps)
        if r["success"].all():
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
    infl: np.ndarray,
    n_boot: int = 1000,
    block: int = 120,
    horizon: int = HORIZON,
    cost_bps: float = COST_BPS,
    seed: int = 597,
) -> dict:
    """Bootstrap distribution of the study's headline differences.

    Circular block bootstrap (joint EQ/BD/INFL rows, ``block``-month blocks —
    within-decade dynamics preserved, cross-decade mean reversion destroyed;
    that is stated, not hidden). Per replicate the full cohort panel is
    rebuilt and four differences recorded:

    - ``dsucc5``   — success-rate gap GK-5% minus fixed-5% (pct points)
    - ``dterm5``   — mean real terminal-wealth gap GK-5% minus fixed-5%
    - ``dlti54``   — mean lifetime real income gap GK-5% minus fixed-4%
    - ``dlti55``   — mean lifetime real income gap GK-5% minus fixed-5%
    """
    rng = np.random.default_rng(seed)
    n = len(eq)
    n_blocks = int(np.ceil(n / block))
    out = {k: np.empty(n_boot) for k in ("dsucc5", "dterm5",
                                         "dlti54", "dlti55")}
    for it in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        ge, gb, gi, _ = cohort_year_returns(eq[idx], bd[idx], infl[idx],
                                            horizon=horizon)
        f4 = simulate(ge, gb, gi, wr0=0.04, preset="fixed", cost_bps=cost_bps)
        f5 = simulate(ge, gb, gi, wr0=0.05, preset="fixed", cost_bps=cost_bps)
        g5 = simulate(ge, gb, gi, wr0=0.05, preset="gk", cost_bps=cost_bps)
        out["dsucc5"][it] = 100.0 * (g5["success"].mean()
                                     - f5["success"].mean())
        out["dterm5"][it] = g5["terminal"].mean() - f5["terminal"].mean()
        out["dlti54"][it] = g5["lti"].mean() - f4["lti"].mean()
        out["dlti55"][it] = g5["lti"].mean() - f5["lti"].mean()
    return out


def ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1.0 - level) / 2.0
    return float(np.quantile(x, a)), float(np.quantile(x, 1.0 - a))


# ---------------------------------------------------------------------------
# Synthetic control — >= 20 independent seeded worlds per setting
# ---------------------------------------------------------------------------

def control_rescue(
    eq_vol: float,
    wr: float,
    n_seeds: int = 20,
    n_years: int = 200,
    horizon: int = HORIZON,
    cost_bps: float = COST_BPS,
    base_seed: int = 597_000,
    **world_kwargs,
) -> dict:
    """The ruin-rescue detector on synthetic worlds: GK vs fixed, same rate.

    For each of ``n_seeds`` independent worlds, run the fixed and the GK rule
    at the same initial rate ``wr`` and record the success-rate difference
    (pp) and the mean lifetime-real-income difference. Returns across-seed
    means and one-sample t-stats (independent worlds, so a plain t is valid).

    The **null** is a calm world (low ``eq_vol``, adequate premium, moderate
    ``wr``) where the fixed rule already succeeds in every cohort: there is
    no ruin to rescue, so ``dsucc`` must read exactly 0 in every seed. The
    planted effect is ruin risk — crank ``eq_vol`` (or the withdrawal rate)
    and the rescue must light up.
    """
    dsucc = np.empty(n_seeds)
    dlti = np.empty(n_seeds)
    fixed_fail = np.empty(n_seeds)
    for k in range(n_seeds):
        world, _ = _data.synthetic_world(
            n_years=n_years, eq_vol=eq_vol, seed=base_seed + k, **world_kwargs)
        ge, gb, gi = annual_cohorts(world, horizon=horizon)
        f = simulate(ge, gb, gi, wr0=wr, preset="fixed", cost_bps=cost_bps)
        g = simulate(ge, gb, gi, wr0=wr, preset="gk", cost_bps=cost_bps)
        dsucc[k] = 100.0 * (g["success"].mean() - f["success"].mean())
        dlti[k] = g["lti"].mean() - f["lti"].mean()
        fixed_fail[k] = 100.0 * (1.0 - f["success"].mean())

    def _t(x):
        s = x.std(ddof=1)
        return float(x.mean() / (s / np.sqrt(len(x)))) if s > 0 else float("nan")

    return {
        "dsucc_mean": float(dsucc.mean()), "dsucc_t": _t(dsucc),
        "dlti_mean": float(dlti.mean()), "dlti_t": _t(dlti),
        "fixed_fail_mean": float(fixed_fail.mean()),
        "n_seeds": n_seeds, "eq_vol": eq_vol, "wr": wr,
    }
