"""The GA engine and its honest controls — Study 589 (Genetic-Algo-Overfit).

The parable: hand a **genetic algorithm** a menu of technical features and let it *evolve* a
trading rule that maximises the in-sample Sharpe. Because the GA searches a huge space of
feature-weight combinations, it will *always* find a rule that looks brilliant on the training
half — even when the tape is a pure random walk with no edge to find. The tell is the
**out-of-sample collapse**: the crowned rule's Sharpe reverts to zero on data it never saw.

A trading rule here is a linear score ``s_t = w . features_t``; the rule goes **long** when
``s_t > 0`` and **flat** otherwise (a long/flat timing rule). The GA's genome is the weight vector
``w`` (unit-norm, so scale is fixed and only the *direction* — which features, which signs —
matters). Fitness = in-sample Sharpe of the rule's returns.

What this module provides:

1. ``evolve_rule`` — a deterministic genetic algorithm (seeded): tournament selection, uniform
   crossover, Gaussian mutation, elitism. Returns the IS-champion genome and the number of *distinct*
   genomes it evaluated (its effective trial count, which feeds the deflated Sharpe).
2. ``rule_returns`` / ``sharpe`` — the long/flat book and its annualised Sharpe.
3. ``is_oos_split`` — the honest protocol: evolve on IS, judge the frozen champion on OOS.
4. ``deflated_sharpe`` — Bailey & López de Prado's Deflated Sharpe Ratio: haircut the observed
   Sharpe for the *number of trials* the GA effectively ran and the return moments. The probability
   the IS Sharpe beats what pure luck would produce given that many trials.
5. ``placebo_pvalue`` — a label-shuffle null: re-run the *whole* evolve-then-validate protocol on
   shuffled forward returns and see how often the OOS Sharpe matches or beats the real one.
6. ``net_sharpe`` — costs: turnover-based one-way cost on the long/flat switches (the OOS book).
7. ``robustness_sweep`` — the IS→OOS shrinkage as the GA's search *budget* (population x
   generations) grows: the more you evolve, the prettier the IS backtest and the deeper the OOS
   collapse — on the null.
8. ``control_oos_sharpe`` / ``seed_robust_control`` — the synthetic positive control: on a tape with
   a *planted* edge the GA-found rule keeps its Sharpe OOS, averaged over >= 20 seeds so no lucky
   RNG seed manufactures the result.

Execution convention: features at the close of day ``t`` are public at that close; the rule takes a
position at that close and earns the return of day ``t+1`` — the single execution lag, applied once
via the ``fwd_ret`` column the data layer already shifted. No future data enters the score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import data as _data

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# The rule and its book
# ---------------------------------------------------------------------------
def rule_position(features: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Long/flat position from a genome ``w``: 1 when the linear score is positive, else 0."""
    score = features @ w
    return (score > 0.0).astype(float)


def rule_returns(features: np.ndarray, fwd_ret: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Daily book returns of the long/flat *timing* rule (position at close of t earns fwd_ret[t]).

    The forward returns are **demeaned** over the evaluation window before the position is applied,
    so the book measures only the rule's *timing* skill (being long on the good days) and cannot
    bank the tape's average drift just by being long-biased. This isolates the overfitting parable:
    a rule with no genuine timing edge scores a Sharpe of ~0, not the market's beta. (A long/flat
    rule on a drifting tape would otherwise inherit that drift's Sharpe for free — a real effect,
    but not the effect this study is about.)
    """
    pos = rule_position(features, w)
    y = np.asarray(fwd_ret, dtype=float)
    finite = np.isfinite(y)
    mu = y[finite].mean() if finite.any() else 0.0
    return pos * (y - mu)


def sharpe(returns: np.ndarray) -> float:
    """Annualised Sharpe of a daily return stream (0 if degenerate)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 5:
        return 0.0
    sd = r.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS))


# ---------------------------------------------------------------------------
# The genetic algorithm
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GAResult:
    genome: np.ndarray          # the IS-champion weight vector (unit norm)
    is_sharpe: float            # its in-sample Sharpe (the "brilliant backtest")
    n_trials: int               # distinct genomes evaluated (effective trial count)
    fitness_history: list       # best IS Sharpe per generation (the "it looks brilliant" curve)


def _normalise(w: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(w)
    return w / n if n > 1e-12 else w


def evolve_rule(
    features: np.ndarray,
    fwd_ret: np.ndarray,
    pop_size: int = 60,
    generations: int = 40,
    mutation_sigma: float = 0.35,
    mutation_rate: float = 0.30,
    tournament: int = 3,
    elite: int = 2,
    seed: int = _data.STUDY_SEED,
) -> GAResult:
    """A deterministic genetic algorithm that evolves a long/flat rule to maximise IS Sharpe.

    Genome = a unit-norm weight vector over the features. Fitness = in-sample Sharpe. Selection is
    a k-way tournament; offspring come from uniform crossover + Gaussian mutation; the top ``elite``
    genomes carry over untouched. Returns the champion, its IS Sharpe, the number of *distinct*
    genomes evaluated (its effective trial count), and the best-Sharpe-per-generation curve.

    Fully seeded — same inputs, same evolution, every run.
    """
    rng = np.random.default_rng(seed)
    n_feat = features.shape[1]

    def fitness(w):
        return sharpe(rule_returns(features, fwd_ret, w))

    # Initial population: random unit-norm genomes.
    pop = [_normalise(rng.standard_normal(n_feat)) for _ in range(pop_size)]
    seen = set()

    def _key(w):
        return tuple(np.round(w, 4))

    fits = []
    for w in pop:
        seen.add(_key(w))
        fits.append(fitness(w))
    fits = np.array(fits)

    history = []
    for _ in range(generations):
        order = np.argsort(-fits)
        pop = [pop[i] for i in order]
        fits = fits[order]
        history.append(float(fits[0]))

        new_pop = [pop[i].copy() for i in range(elite)]  # elitism
        while len(new_pop) < pop_size:
            # tournament selection of two parents
            def pick():
                idx = rng.integers(0, pop_size, size=tournament)
                best = idx[np.argmax(fits[idx])]
                return pop[best]

            p1, p2 = pick(), pick()
            mask = rng.random(n_feat) < 0.5           # uniform crossover
            child = np.where(mask, p1, p2).astype(float)
            mut = rng.random(n_feat) < mutation_rate  # Gaussian mutation
            child = child + mut * rng.standard_normal(n_feat) * mutation_sigma
            new_pop.append(_normalise(child))

        pop = new_pop
        fits = np.empty(pop_size)
        for i, w in enumerate(pop):
            seen.add(_key(w))
            fits[i] = fitness(w)

    best_i = int(np.argmax(fits))
    return GAResult(
        genome=pop[best_i],
        is_sharpe=float(fits[best_i]),
        n_trials=len(seen),
        fitness_history=history + [float(fits[best_i])],
    )


# ---------------------------------------------------------------------------
# The honest IS / OOS protocol
# ---------------------------------------------------------------------------
def _split_arrays(df: pd.DataFrame, is_frac: float = 0.5):
    feats = df[_data.FEATURE_NAMES].to_numpy(dtype=float)
    fwd = df["fwd_ret"].to_numpy(dtype=float)
    k = int(round(len(df) * is_frac))
    return (feats[:k], fwd[:k]), (feats[k:], fwd[k:]), k


def is_oos_split(
    df: pd.DataFrame,
    is_frac: float = 0.5,
    pop_size: int = 60,
    generations: int = 40,
    seed: int = _data.STUDY_SEED,
) -> dict:
    """Evolve on the IS half, then judge the frozen champion, untouched, on the OOS half.

    Returns the IS champion, its IS Sharpe, its OOS Sharpe, the IS - OOS shrinkage, the OOS
    long/flat exposure, the OOS book returns and the effective trial count.
    """
    (fis, yis), (foos, yoos), _ = _split_arrays(df, is_frac)
    ga = evolve_rule(fis, yis, pop_size=pop_size, generations=generations, seed=seed)
    oos_ret = rule_returns(foos, yoos, ga.genome)
    is_s = ga.is_sharpe
    oos_s = sharpe(oos_ret)
    return {
        "genome": ga.genome,
        "is_sharpe": is_s,
        "oos_sharpe": oos_s,
        "shrinkage": float(is_s - oos_s),
        "oos_exposure": float(rule_position(foos, ga.genome).mean()),
        "oos_ret": oos_ret,
        "n_trials": ga.n_trials,
        "fitness_history": ga.fitness_history,
    }


# ---------------------------------------------------------------------------
# The Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
# ---------------------------------------------------------------------------
def expected_max_sharpe(n_trials: int, sr_var: float = 1.0) -> float:
    """Expected maximum Sharpe from ``n_trials`` independent zero-edge strategies.

    E[max SR] ~= sqrt(sr_var) * ( (1-g) * z(1 - 1/N) + g * z(1 - 1/(N e)) ), where g is the
    Euler-Mascheroni constant and z is the Gaussian quantile. The luck bar that a search of N
    trials clears for free.
    """
    if n_trials < 2:
        return 0.0
    g = 0.5772156649015329
    z1 = sps.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_var) * ((1.0 - g) * z1 + g * z2))


def deflated_sharpe(
    returns: np.ndarray, n_trials: int, benchmark_sr: float | None = None
) -> dict:
    """The Deflated Sharpe Ratio: probability the observed Sharpe beats a luck benchmark.

    Haircuts the observed (in-sample) Sharpe for (a) the number of trials the GA effectively ran —
    via the expected-max-Sharpe benchmark — and (b) the non-normal return moments (Lo's
    skew/kurtosis correction to the Sharpe standard error). DSR near 0 means "this Sharpe is what a
    search of this many trials produces from noise"; near 1 means "genuinely beyond luck".
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 10:
        return {"sr": 0.0, "sr_daily": 0.0, "dsr": float("nan"),
                "benchmark_sr": 0.0, "n_trials": int(n_trials)}
    sr_daily = r.mean() / r.std(ddof=1) if r.std(ddof=1) > 1e-12 else 0.0
    skew = float(sps.skew(r))
    kurt = float(sps.kurtosis(r, fisher=True))  # excess kurtosis
    # Lo (2002) SE of the Sharpe with non-normal moments:
    se = np.sqrt((1.0 - skew * sr_daily + (kurt) / 4.0 * sr_daily**2) / (n - 1))
    if benchmark_sr is None:
        # per-trade SR dispersion assumed ~1 (standardised); express benchmark in daily SR units
        sr_ann_bench = expected_max_sharpe(n_trials, sr_var=1.0)
        benchmark_sr = sr_ann_bench / np.sqrt(TRADING_DAYS)
    if se <= 0:
        return {"sr": float(sr_daily * np.sqrt(TRADING_DAYS)), "sr_daily": float(sr_daily),
                "dsr": float("nan"), "benchmark_sr": float(benchmark_sr * np.sqrt(TRADING_DAYS)),
                "n_trials": int(n_trials)}
    dsr = float(sps.norm.cdf((sr_daily - benchmark_sr) / se))
    return {
        "sr": float(sr_daily * np.sqrt(TRADING_DAYS)),
        "sr_daily": float(sr_daily),
        "dsr": dsr,
        "benchmark_sr": float(benchmark_sr * np.sqrt(TRADING_DAYS)),
        "n_trials": int(n_trials),
    }


# ---------------------------------------------------------------------------
# Placebo — re-run the whole search on shuffled forward returns
# ---------------------------------------------------------------------------
def placebo_pvalue(
    df: pd.DataFrame,
    n_perm: int = 40,
    is_frac: float = 0.5,
    pop_size: int = 40,
    generations: int = 25,
    seed: int = _data.STUDY_SEED,
) -> dict:
    """Label-shuffle placebo on the *OOS* Sharpe.

    Shuffle the forward returns against the features (destroying any real predictability), then run
    the entire evolve-then-validate protocol. The p-value is the fraction of shuffles whose OOS
    Sharpe >= the observed OOS Sharpe. If the real OOS edge is genuine it sits in the tail; if it is
    a search artefact it looks just like the shuffles. (A lighter GA budget keeps this tractable.)
    """
    real = is_oos_split(df, is_frac, pop_size=pop_size, generations=generations, seed=seed)
    obs = real["oos_sharpe"]
    rng = np.random.default_rng(seed + 1)
    feats = df[_data.FEATURE_NAMES].to_numpy(dtype=float)
    fwd = df["fwd_ret"].to_numpy(dtype=float)
    count = 0
    perm_oos = []
    for j in range(n_perm):
        perm = rng.permutation(len(fwd))
        shuf = pd.DataFrame(feats, columns=_data.FEATURE_NAMES)
        shuf["fwd_ret"] = fwd[perm]
        res = is_oos_split(shuf, is_frac, pop_size=pop_size, generations=generations,
                           seed=seed + 100 + j)
        perm_oos.append(res["oos_sharpe"])
        if res["oos_sharpe"] >= obs - 1e-12:
            count += 1
    return {
        "obs_oos_sharpe": float(obs),
        "p": (count + 1) / (n_perm + 1),
        "perm_mean_oos": float(np.mean(perm_oos)),
        "n_perm": int(n_perm),
    }


# ---------------------------------------------------------------------------
# Costs — turnover on the OOS long/flat book
# ---------------------------------------------------------------------------
def net_sharpe(oos_features: np.ndarray, oos_fwd: np.ndarray, genome: np.ndarray,
               cost_bps: float = 2.0) -> dict:
    """Gross and net (turnover-cost) OOS Sharpe of the long/flat rule.

    Turnover is one-way x NAV: every switch between long and flat crosses the spread once. Charge
    ``cost_bps`` on each switch. Long/flat has no borrow (no short leg), so cost is purely the
    timing turnover — deliberately mild, to show the OOS collapse is *not* a cost story.
    """
    pos = rule_position(oos_features, genome)
    gross = rule_returns(oos_features, oos_fwd, genome)  # demeaned timing book (same as fitness)
    switches = np.abs(np.diff(np.concatenate([[0.0], pos])))
    cost = switches * cost_bps * 1e-4
    net = gross - cost
    return {
        "gross_sharpe": sharpe(gross),
        "net_sharpe": sharpe(net),
        "turnover_per_day": float(switches.mean()),
        "cost_bps": float(cost_bps),
    }


# ---------------------------------------------------------------------------
# Robustness — IS/OOS shrinkage vs the GA's search budget
# ---------------------------------------------------------------------------
def robustness_sweep(df: pd.DataFrame, budgets, is_frac: float = 0.5,
                     seed: int = _data.STUDY_SEED) -> pd.DataFrame:
    """IS Sharpe, OOS Sharpe and shrinkage as the search budget (pop x generations) grows.

    ``budgets`` is a list of (pop_size, generations). The more the GA searches, the higher the IS
    Sharpe it can manufacture — while the OOS Sharpe stays pinned to the tape's true edge (zero on
    the null). Returns a tidy frame.
    """
    rows = []
    for (pop, gen) in budgets:
        res = is_oos_split(df, is_frac, pop_size=pop, generations=gen, seed=seed)
        rows.append({
            "trials": res["n_trials"],
            "pop": pop, "gen": gen,
            "is_sharpe": res["is_sharpe"],
            "oos_sharpe": res["oos_sharpe"],
            "shrinkage": res["shrinkage"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Synthetic positive control — the GA banks a REAL planted edge OOS
# ---------------------------------------------------------------------------
def control_oos_sharpe(signal_strength: float, n_days: int = 3000,
                       pop_size: int = 60, generations: int = 40,
                       seed: int = _data.STUDY_SEED) -> dict:
    """Run evolve-then-validate on a tape with a *planted* edge of ``signal_strength``.

    On the null (0) the OOS Sharpe collapses to ~0; with a real planted edge the GA-found rule keeps
    its Sharpe OOS (shrinkage ~0). Returns IS/OOS Sharpe and shrinkage.
    """
    df, _ = _data.synthetic_series(n_days=n_days, signal_strength=signal_strength, seed=seed)
    res = is_oos_split(df, pop_size=pop_size, generations=generations, seed=seed)
    return {
        "signal_strength": signal_strength,
        "is_sharpe": res["is_sharpe"],
        "oos_sharpe": res["oos_sharpe"],
        "shrinkage": res["shrinkage"],
    }


def seed_robust_control(signal_strength: float, n_seeds: int = 20,
                        n_days: int = 2000, pop_size: int = 40, generations: int = 25,
                        base_seed: int = _data.STUDY_SEED) -> dict:
    """Average the OOS Sharpe over ``n_seeds`` synthetic worlds for a planted ``signal_strength``.

    House rule: any synthetic-dependent claim averages the statistic over >= 20 seeds, so no single
    lucky RNG seed manufactures the result. Returns the mean IS and OOS Sharpe (and their spread)
    across seeds — the null must stay flat OOS, the control must stay up OOS.
    """
    is_list, oos_list = [], []
    for s in range(base_seed, base_seed + n_seeds):
        df, _ = _data.synthetic_series(n_days=n_days, signal_strength=signal_strength, seed=s)
        res = is_oos_split(df, pop_size=pop_size, generations=generations, seed=s)
        is_list.append(res["is_sharpe"])
        oos_list.append(res["oos_sharpe"])
    return {
        "signal_strength": signal_strength,
        "n_seeds": int(n_seeds),
        "mean_is_sharpe": float(np.mean(is_list)),
        "mean_oos_sharpe": float(np.mean(oos_list)),
        "mean_shrinkage": float(np.mean(is_list) - np.mean(oos_list)),
        "oos_t": float(np.mean(oos_list) / (np.std(oos_list, ddof=1) / np.sqrt(n_seeds)))
        if np.std(oos_list, ddof=1) > 1e-12 else float("nan"),
    }
