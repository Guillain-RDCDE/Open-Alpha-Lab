"""The Deflated-Sharpe-Ratio engine and its honest controls — Study 833.

The disease, stated plainly (Bailey & López de Prado, 2014, *The Deflated Sharpe Ratio*):
run ``N`` independent strategies on a tape with **zero** true edge and the *best* sample
Sharpe is not zero — it inflates with ``N``. Under the null (no skill) the expected maximum
Sharpe over ``N`` iid trials is

    E[max SR] ≈ sqrt(V) · [ (1-γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e)) ]

where ``V`` is the cross-trial variance of the (per-period) Sharpe estimates, ``γ`` is the
Euler-Mascheroni constant, and ``Z⁻¹`` is the inverse standard-normal CDF. A backtest that
reports "the best of my 1,000 rules had Sharpe 1.4!" without disclosing the 1,000 has said
nothing: 1.4 is *below* what luck alone delivers.

The cure is the **Deflated Sharpe Ratio**: re-express the winner's Sharpe as the probability
it beats the *expected maximum under the null*, correcting for the trial count ``N``, the
sample length ``T``, and the higher moments (skew ``g3`` / kurtosis ``g4``) of the returns:

    DSR = Φ( (SR − SR0)·sqrt(T − 1) / sqrt(1 − g3·SR + (g4 − 1)/4·SR²) )

with ``SR`` the per-period Sharpe of the selected series and ``SR0`` the expected maximum.
For the winner of ``N`` empty trials ``SR ≈ SR0``, so DSR ≈ 0.5 — a coin flip, "consistent
with luck." For an honest *single* hypothesis (``N = 1`` ⇒ ``SR0 = 0``) the DSR collapses to
the Probabilistic Sharpe Ratio (PSR) vs a zero benchmark, which a genuinely-good strategy
clears comfortably.

Inference rails are shared with the desk's cross-sectional studies: a one-sample *t*, a
Welch *t*, a Newey-West HAC *t*, and a Wilson interval on hit rates.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

TRADING_DAYS = 252
GAMMA = 0.5772156649015329     # Euler-Mascheroni constant


# --------------------------------------------------------------------------- #
# Sharpe helpers (vectorised across a T×N panel)
# --------------------------------------------------------------------------- #
def sharpe_ann(r: np.ndarray, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe of a single return stream (excess-of-zero)."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(periods)) if sd > 0 else float("nan")


def sr_per_period(r: np.ndarray) -> float:
    """Non-annualised (per-period) Sharpe of a single return stream."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else float("nan")


def panel_sr_per_period(panel: np.ndarray) -> np.ndarray:
    """Per-period Sharpe of every column of a ``T×N`` panel (vectorised, no python loop)."""
    P = np.asarray(panel, dtype=float)
    mu = P.mean(axis=0)
    sd = P.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(sd > 0, mu / sd, np.nan)
    return out


# --------------------------------------------------------------------------- #
# The expected maximum Sharpe under the null (Bailey & López de Prado 2014)
# --------------------------------------------------------------------------- #
def expected_max_sharpe(n_trials: int, var_sharpe: float = 1.0) -> float:
    """Expected maximum (per-period) Sharpe under the null, over ``n_trials`` iid trials.

        E[max] ≈ sqrt(var_sharpe) · [ (1-γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e)) ]

    ``var_sharpe`` is the cross-trial variance of the per-period Sharpe *estimates* (under
    the null with ``T`` observations, ≈ ``1/T``). Returns ``0`` for ``n_trials < 2`` (a
    single hypothesis has no selection to inflate it — the honest baseline). This is the bar
    the best in-sample Sharpe must clear to be more than luck.
    """
    if n_trials < 2:
        return 0.0
    z1 = float(stats.norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    return math.sqrt(max(var_sharpe, 0.0)) * ((1.0 - GAMMA) * z1 + GAMMA * z2)


def best_sharpe_experiment(
    panel: np.ndarray, periods: int = TRADING_DAYS, var_sharpe: float | None = None,
) -> dict:
    """Take the best column of a null panel; compare its Sharpe to the formula's prediction.

    Reports, in both per-period and annualised units: the *observed* maximum Sharpe across
    the ``N`` columns, the expected maximum under the null (using the empirical cross-trial
    variance of the per-period SR estimates unless ``var_sharpe`` is supplied), and the mean
    column Sharpe (≈ 0, the true edge). This is the whole inflation story in one call.
    """
    P = np.asarray(panel, dtype=float)
    T, N = P.shape
    srs = panel_sr_per_period(P)
    v = float(np.nanvar(srs, ddof=1)) if var_sharpe is None else float(var_sharpe)
    emax = expected_max_sharpe(N, v)
    j = int(np.nanargmax(srs))
    obs = float(srs[j])
    ann = math.sqrt(periods)
    return {
        "n_trials": int(N), "n_days": int(T),
        "var_sharpe": v, "sr_std_across_trials": math.sqrt(v),
        "best_col": j,
        "obs_max_sr": obs, "obs_max_sharpe_ann": obs * ann,
        "exp_max_sr": emax, "exp_max_sharpe_ann": emax * ann,
        "mean_sr": float(np.nanmean(srs)), "mean_sharpe_ann": float(np.nanmean(srs)) * ann,
    }


def inflation_curve(
    n_grid=(2, 5, 10, 25, 50, 100, 250, 500, 1000),
    n_days: int = 1260,
    ann_vol: float = 0.15,
    n_seeds: int = 40,
    base_seed: int = 833,
    periods: int = TRADING_DAYS,
) -> dict:
    """The headline curve: observed best Sharpe vs the formula's E[max], as ``N`` grows.

    For each ``N`` in ``n_grid`` we draw ``n_seeds`` independent null panels of ``N`` empty
    columns, record the best (per-period) Sharpe of each, and average. The observed mean
    tracks the expected-maximum-Sharpe prediction; both climb monotonically in ``N`` while
    the *true* edge stays pinned at zero. Vectorised: one big panel per (seed, N).

    Returns arrays (annualised) of ``N``, the mean observed best Sharpe, its across-seed sd,
    and the formula prediction (evaluated at the theoretical ``var_sharpe ≈ 1/(T-1)``).
    """
    from . import data as _data
    ann = math.sqrt(periods)
    n_grid = list(n_grid)
    obs_mean, obs_sd, pred = [], [], []
    v_theory = 1.0 / (n_days - 1)     # sampling var of a per-period SR estimate under H0
    for N in n_grid:
        bests = np.empty(n_seeds)
        for s in range(n_seeds):
            panel = _data.null_panel(N, n_days, ann_vol, base_seed + s)
            srs = panel_sr_per_period(panel)
            bests[s] = np.nanmax(srs)
        obs_mean.append(float(bests.mean()) * ann)
        obs_sd.append(float(bests.std(ddof=1)) * ann)
        pred.append(expected_max_sharpe(N, v_theory) * ann)
    return {
        "n_grid": np.asarray(n_grid),
        "obs_best_ann": np.asarray(obs_mean),
        "obs_best_sd_ann": np.asarray(obs_sd),
        "pred_ann": np.asarray(pred),
        "n_days": n_days, "n_seeds": n_seeds,
    }


# --------------------------------------------------------------------------- #
# The Deflated / Probabilistic Sharpe Ratio (Bailey & López de Prado 2014)
# --------------------------------------------------------------------------- #
def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    var_sharpe: float | None = None,
    periods: int = TRADING_DAYS,
) -> dict:
    """The Deflated Sharpe Ratio of a *selected* (best-of-N) strategy's return series.

    Steps:
      1. Per-period Sharpe ``SR`` of the selected series and its return skew ``g3`` /
         (non-excess) kurtosis ``g4``.
      2. Expected maximum Sharpe under the null ``SR0 = expected_max_sharpe(n_trials, V)``.
         ``V`` (cross-trial SR variance) defaults to the theoretical ``1/(T-1)`` when not
         supplied — the sampling variance of a per-period SR estimate under H0.
      3. DSR = Φ( (SR − SR0)·sqrt(T − 1) / sqrt(1 − g3·SR + (g4 − 1)/4·SR²) ).

    ``n_trials = 1`` ⇒ ``SR0 = 0`` and the DSR is the ordinary **Probabilistic Sharpe Ratio**
    (PSR) — the deflation an honest single hypothesis gets. Returns a dict with the annualised
    Sharpe, the per-period ``SR``, ``SR0`` (both per-period and annualised), the deflated
    *excess* Sharpe (``SR − SR0``, annualised), and the DSR probability in ``[0, 1]``. A DSR
    below ~0.95 says "not distinguishable from the luck of the search."
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = r.size
    ann = math.sqrt(periods)
    if T < 4 or r.std(ddof=1) == 0:
        nan = float("nan")
        return {"sharpe_ann": nan, "sr": nan, "sr0": nan, "sr0_ann": nan,
                "deflated_excess_ann": nan, "dsr": nan, "n_trials": int(n_trials), "T": int(T)}
    sr = float(r.mean() / r.std(ddof=1))
    g3 = float(stats.skew(r, bias=False))
    g4 = float(stats.kurtosis(r, fisher=False, bias=False))
    v = (1.0 / (T - 1)) if var_sharpe is None else float(var_sharpe)
    sr0 = expected_max_sharpe(n_trials, v)
    denom = math.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr, 1e-12))
    z = (sr - sr0) * math.sqrt(T - 1) / denom
    return {
        "sharpe_ann": sr * ann, "sr": sr,
        "sr0": sr0, "sr0_ann": sr0 * ann,
        "deflated_excess_ann": (sr - sr0) * ann,
        "dsr": float(stats.norm.cdf(z)),
        "n_trials": int(n_trials), "T": int(T),
    }


def probabilistic_sharpe_ratio(returns: np.ndarray, benchmark_sr: float = 0.0,
                               periods: int = TRADING_DAYS) -> float:
    """Probabilistic Sharpe Ratio: P(true per-period SR > ``benchmark_sr``). DSR at N=1."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = r.size
    if T < 4 or r.std(ddof=1) == 0:
        return float("nan")
    sr = float(r.mean() / r.std(ddof=1))
    g3 = float(stats.skew(r, bias=False))
    g4 = float(stats.kurtosis(r, fisher=False, bias=False))
    denom = math.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr, 1e-12))
    z = (sr - benchmark_sr) * math.sqrt(T - 1) / denom
    return float(stats.norm.cdf(z))


# --------------------------------------------------------------------------- #
# The in-sample champion → out-of-sample collapse (the Mirage picture)
# --------------------------------------------------------------------------- #
def in_sample_champion(panel: np.ndarray, frac: float = 0.5, periods: int = TRADING_DAYS) -> dict:
    """Pick the in-sample Sharpe champion of a null panel; report its IS vs OOS Sharpe.

    Split the ``T`` rows into the first ``frac`` (in-sample) / the rest (out-of-sample),
    choose the column with the best in-sample Sharpe, and read that same column's
    out-of-sample Sharpe. On a null panel the gorgeous IS Sharpe evaporates OOS — the
    selection artefact laid bare. Vectorised column Sharpes.
    """
    P = np.asarray(panel, dtype=float)
    T, N = P.shape
    cut = int(T * frac)
    is_srs = panel_sr_per_period(P[:cut])
    j = int(np.nanargmax(is_srs))
    oos = P[cut:, j]
    ann = math.sqrt(periods)
    return {
        "champion": j,
        "is_sharpe_ann": float(is_srs[j]) * ann,
        "oos_sharpe_ann": sharpe_ann(oos, periods),
        "oos_t_nw": newey_west_t(oos),
        "mean_is_sharpe_ann": float(np.nanmean(is_srs)) * ann,
        "is_n": cut, "oos_n": T - cut, "n_trials": N,
    }


# --------------------------------------------------------------------------- #
# The costed timer — is there anything to harvest? (No; the winner is empty.)
# --------------------------------------------------------------------------- #
def timer_stats(returns: np.ndarray, cost_bps: float = 5.0,
                periods: int = TRADING_DAYS) -> dict:
    """Cost the "winning" strategy as if you traded it out-of-sample.

    Charge a one-way ``cost_bps`` × NAV round-trip per period (a long/flat strategy rotates
    daily in the worst case) on the selected series' out-of-sample leg. On a null winner the
    gross OOS mean is ≈ 0 and any friction pushes the net negative — a Mirage by construction.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 3:
        return {"n": n, "gross_bps": float("nan"), "net_bps": float("nan"),
                "sharpe_net": float("nan"), "t_net": float("nan")}
    round_trip = 2.0 * cost_bps / 1e4
    net = r - round_trip
    sd = net.std(ddof=1)
    return {
        "n": n,
        "gross_bps": float(r.mean() * 1e4),
        "net_bps": float(net.mean() * 1e4),
        "cost_bps_per_day": round_trip * 1e4,
        "sharpe_net": float(net.mean() / sd * math.sqrt(periods)) if sd > 0 else float("nan"),
        "t_net": one_sample_t(net),
    }


# --------------------------------------------------------------------------- #
# Inference primitives (shared house rails)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# The synthetic controls — the machinery proof (fires on planted edge, silent on null)
# --------------------------------------------------------------------------- #
def null_dsr_calibration(
    n_trials: int = 1000,
    n_days: int = 1260,
    ann_vol: float = 0.15,
    n_seeds: int = 40,
    base_seed: int = 833,
) -> dict:
    """Across ``n_seeds`` null pools of ``N`` empty strategies: how often the *naive* screen
    and the *DSR* each declare the winner "real."

    The naive screen (best sample Sharpe > 0 with a one-sample |t| ≥ 2) fires on virtually
    every pool — the winner always looks great. The DSR (≥ 0.95) fires at ~the nominal 5%,
    because on a null ``SR ≈ SR0`` ⇒ DSR ≈ 0.5. The gap is the whole point: the deflation is
    calibrated, the naive number is not. Also returns the mean DSR (≈ 0.5) and the mean
    deflated-excess Sharpe (≈ 0 — the winner shrunk back to nothing).
    """
    from . import data as _data
    naive_fire = dsr_fire = 0
    dsrs, excess, best_sharpes = [], [], []
    for s in range(n_seeds):
        panel = _data.null_panel(n_trials, n_days, ann_vol, base_seed + s)
        srs = panel_sr_per_period(panel)
        j = int(np.nanargmax(srs))
        r = panel[:, j]
        best_sharpes.append(sharpe_ann(r))
        naive_t = one_sample_t(r)
        if srs[j] > 0 and abs(naive_t) >= 2.0:
            naive_fire += 1
        d = deflated_sharpe_ratio(r, n_trials)
        dsrs.append(d["dsr"]); excess.append(d["deflated_excess_ann"])
        if d["dsr"] >= 0.95:
            dsr_fire += 1
    return {
        "n_seeds": n_seeds, "n_trials": n_trials,
        "naive_fire": naive_fire, "naive_fire_rate": naive_fire / n_seeds,
        "dsr_fire": dsr_fire, "dsr_fire_rate": dsr_fire / n_seeds,
        "mean_best_sharpe_ann": float(np.mean(best_sharpes)),
        "mean_dsr": float(np.mean(dsrs)),
        "mean_deflated_excess_ann": float(np.mean(excess)),
    }


def honest_control(
    true_ann_sharpe: float = 1.0,
    n_days: int = 1260,
    ann_vol: float = 0.15,
    n_seeds: int = 40,
    base_seed: int = 833,
) -> dict:
    """The positive control: a single honest strategy (true Sharpe > 0), across ``n_seeds``.

    Each seed draws one honest stream and computes its DSR as a *single* hypothesis
    (``n_trials = 1`` ⇒ PSR vs 0). A faithful deflation must keep this one's DSR high (near 1)
    — the correction spares genuine skill, it only punishes *searching*. Returns the mean
    realised annualised Sharpe, the mean DSR, and the fraction with DSR ≥ 0.95.
    """
    from . import data as _data
    sharpes, dsrs, fire = [], [], 0
    for s in range(n_seeds):
        r = _data.honest_strategy(n_days, true_ann_sharpe, ann_vol, base_seed + s)
        sharpes.append(sharpe_ann(r))
        d = deflated_sharpe_ratio(r, n_trials=1)
        dsrs.append(d["dsr"])
        if d["dsr"] >= 0.95:
            fire += 1
    return {
        "n_seeds": n_seeds, "true_ann_sharpe": true_ann_sharpe,
        "mean_sharpe_ann": float(np.mean(sharpes)),
        "mean_dsr": float(np.mean(dsrs)),
        "dsr_fire": fire, "dsr_fire_rate": fire / n_seeds,
    }
