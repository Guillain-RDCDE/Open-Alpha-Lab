"""The overfitting engine and its honest controls — Study 344 (Backtest-Overfitting).

The disease, stated plainly (Bailey, Borwein, López de Prado & Zhu, 2014): if you try
enough strategy configurations on the *same* data, one of them will look spectacular by
chance alone — and the more you try, the more spectacular the luckiest looks, and the
harder it falls out of sample. Two diagnostics put a number on it:

1. **Deflated Sharpe Ratio (DSR).** The maximum Sharpe found over ``N`` trials is biased
   upward; under the null (no skill) the expected max Sharpe grows with ``N``. The DSR
   re-expresses the best Sharpe as the probability it exceeds the *expected maximum under
   the null*, correcting for the number of trials, the sample length, and the higher
   moments (skew/kurtosis) of the returns. A naked Sharpe of 1.5 from a 1000-config sweep
   can deflate to a DSR near zero — i.e. "consistent with luck."
   (Bailey & López de Prado, 2014, *The Deflated Sharpe Ratio*.)

2. **Probability of Backtest Overfitting (PBO).** Via Combinatorially-Symmetric
   Cross-Validation (CSCV): split the (T × N) trial-return matrix into ``S`` time blocks,
   form every balanced in-sample/out-of-sample partition, pick the in-sample champion in
   each, and ask how often that champion lands in the *bottom half* out-of-sample. PBO is
   that frequency — the probability that the strategy you'd have selected is below median
   live. PBO ≈ 0.5 means selection is worthless; PBO > 0.5 means it is *worse* than random.
   (Bailey, Borwein, López de Prado & Zhu, 2017, *The Probability of Backtest Overfitting*.)

The strategy zoo we grid-search is deliberately bland — long/flat moving-average crossovers
with a swept (fast, slow) pair — because on a martingale tape NONE of them has an edge, so
the entire in-sample spread is overfitting we can measure.

Honesty discipline (the audits that keep catching us):
  * One execution lag, applied once: the signal known at the close of *t* takes the return
    of *t+1* (a single ``shift(1)``). Crossover rules are not calendar-known, so they need
    the lag.
  * Costs are one-way turnover × NAV at ``cost_bps`` per side; a flat-to-long flip is one
    one-way trade.
  * Sharpe is excess-of-cash where a real risk-free is supplied, and excess-vs-excess for
    any race; here the flat leg sits in cash earning zero by construction (synthetic), so
    the long/flat Sharpe is already an excess-of-zero number on the synthetic tape.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# The strategy zoo — long/flat moving-average crossovers
# ---------------------------------------------------------------------------
def _crossover_returns(
    prices: pd.Series, fast: int, slow: int, cost_bps: float
) -> pd.Series:
    """Net daily returns of a long/flat fast-vs-slow MA crossover, one execution lag.

    Rule: hold long when fast-MA > slow-MA, else flat (cash, zero return). The position is
    formed from MAs known at the close of *t* and earns the asset return of *t+1* — one
    ``shift(1)``. Turnover at a flip is ``|pos_t - pos_{t-1}|`` (one-way, in {0,1}) charged
    at ``cost_bps``.
    """
    asset_ret = prices.pct_change().fillna(0.0)
    f = prices.rolling(fast, min_periods=fast).mean()
    s = prices.rolling(slow, min_periods=slow).mean()
    raw_pos = (f > s).astype(float)            # signal known at close of t
    pos = raw_pos.shift(1).fillna(0.0)         # the one and only execution lag
    gross = pos * asset_ret
    turnover = pos.diff().abs().fillna(pos.abs())   # one-way turnover, first day = open cost
    cost = turnover * cost_bps * 1e-4
    return (gross - cost).rename(f"f{fast}_s{slow}")


def _trial_grid(n_trials: int) -> list[tuple[int, int]]:
    """A deterministic list of ``(fast, slow)`` crossover pairs, ``fast < slow``.

    Built by taking the Cartesian product of fast and slow windows and keeping the first
    ``n_trials`` valid pairs — a stable, seed-independent enumeration so the same grid
    appears every run (the 'researcher's parameter zoo')."""
    fasts = list(range(2, 60))
    slows = list(range(5, 260, 3))
    pairs = [(f, s) for f, s in itertools.product(fasts, slows) if f < s]
    return pairs[:n_trials]


def sweep(
    prices: pd.Series, n_trials: int = 200, cost_bps: float = 0.0, seed: int = 344
) -> pd.DataFrame:
    """Run ``n_trials`` crossover configurations on one price path → (T × N) return matrix.

    The columns are the trials (the 'configurations the researcher tried'); the rows are
    days. This is the object the Deflated Sharpe Ratio and PBO/CSCV consume. The ``seed``
    is accepted for signature symmetry but the grid is deterministic by construction.
    """
    pairs = _trial_grid(n_trials)
    cols = {f"f{f}_s{s}": _crossover_returns(prices, f, s, cost_bps) for f, s in pairs}
    mat = pd.DataFrame(cols, index=prices.index)
    # Drop the warm-up rows where the longest slow MA is undefined (all-NaN columns aligned).
    return mat.dropna(how="all").fillna(0.0)


def buy_and_hold(prices: pd.Series) -> pd.Series:
    """The honest benchmark: own the asset, no timing. Daily returns."""
    return prices.pct_change().fillna(0.0).rename("buy_and_hold")


# ---------------------------------------------------------------------------
# Honest metrics & inference
# ---------------------------------------------------------------------------
def sharpe(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised Sharpe (excess-of-zero; the synthetic flat leg earns zero by design)."""
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * math.sqrt(periods_per_year))


def hac_tstat(returns: pd.Series) -> dict:
    """Newey-West HAC t-stat that the mean per-period return differs from zero.

    Used to ask whether the *selected champion's* out-of-sample mean is distinguishable
    from zero with autocorrelation-robust standard errors — the bar a 'real' edge must
    clear (|t| >= 2).
    """
    x = returns.dropna().to_numpy(dtype=float)
    n = x.size
    if n < 3:
        return {"mean": float("nan"), "tstat": float("nan"), "n": n}
    mu = x.mean()
    e = x - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        wk = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * wk * float(e[k:] @ e[:-k]) / n
    se = math.sqrt(max(lrv, 0.0) / n)
    return {"mean": float(mu), "tstat": float(mu / se) if se > 0 else float("nan"), "n": n}


def block_bootstrap_ci(
    returns: pd.Series, block: int = 21, n_boot: int = 2000, seed: int = 344, alpha: float = 0.05
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the *mean* per-period return.

    Block resampling preserves the autocorrelation i.i.d. resampling would destroy.
    Returns the (lower, upper) percentile interval on the mean per-period return.
    """
    x = returns.dropna().to_numpy(dtype=float)
    n = x.size
    if n < block + 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[b] = x[idx][:n].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


# ---------------------------------------------------------------------------
# The two overfitting diagnostics (Bailey & López de Prado)
# ---------------------------------------------------------------------------
def expected_max_sharpe(n_trials: int, var_sharpe: float = 1.0) -> float:
    """Expected maximum Sharpe under the null (no skill), over ``n_trials`` independent trials.

    From Bailey & López de Prado (2014): with i.i.d. trial Sharpes of variance
    ``var_sharpe``, the expected maximum is approximately

        E[max] ≈ sqrt(var_sharpe) * [ (1-γ) Z⁻¹(1 - 1/N) + γ Z⁻¹(1 - 1/(N·e)) ]

    where ``γ`` is the Euler-Mascheroni constant and ``Z⁻¹`` the inverse standard normal
    CDF. This is the bar the *best* in-sample Sharpe must clear to be more than luck."""
    if n_trials < 2:
        return 0.0
    gamma = 0.5772156649015329
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(var_sharpe) * ((1.0 - gamma) * z1 + gamma * z2)


def deflated_sharpe_ratio(
    returns: pd.Series, n_trials: int, periods_per_year: int = TRADING_DAYS
) -> dict:
    """The Deflated Sharpe Ratio for a *selected* (best-of-N) strategy's return series.

    Steps (Bailey & López de Prado, 2014):
      1. Estimate the (non-annualised, per-period) Sharpe ``SR`` of the selected series and
         its sampling variance via the higher-moment-aware Lo (2002) / Mertens estimator
         that accounts for skew ``g3`` and kurtosis ``g4``.
      2. Compute the expected maximum Sharpe under the null, ``SR0``, over ``n_trials``,
         using the cross-trial dispersion of per-period Sharpes as ``var_sharpe`` proxy
         (here taken as 1 in non-annualised SR units — a conservative, standard default).
      3. DSR = Φ( (SR - SR0) * sqrt(T-1) / sqrt(1 - g3·SR + (g4-1)/4 · SR²) ).

    Returns a dict with the raw annualised Sharpe, the per-period SR, SR0, and the DSR
    (a probability in [0,1]). DSR < 0.5 (and especially near 0) says "the best of these N
    is consistent with luck."
    """
    r = returns.dropna().to_numpy(dtype=float)
    T = r.size
    if T < 4 or r.std(ddof=1) == 0:
        return {"sharpe_ann": float("nan"), "sr_per_period": float("nan"),
                "sr0": float("nan"), "dsr": float("nan"), "n_trials": n_trials, "T": T}
    sr = r.mean() / r.std(ddof=1)                       # per-period Sharpe
    g3 = float(stats.skew(r, bias=False))
    g4 = float(stats.kurtosis(r, fisher=False, bias=False))  # non-excess kurtosis
    sr0 = expected_max_sharpe(n_trials, var_sharpe=1.0)
    denom = math.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr, 1e-12))
    z = (sr - sr0) * math.sqrt(T - 1) / denom
    dsr = float(stats.norm.cdf(z))
    return {
        "sharpe_ann": float(sr * math.sqrt(periods_per_year)),
        "sr_per_period": float(sr),
        "sr0": float(sr0),
        "dsr": dsr,
        "n_trials": int(n_trials),
        "T": int(T),
    }


def pbo_cscv(returns_matrix: pd.DataFrame, n_splits: int = 14, seed: int = 344) -> dict:
    """Probability of Backtest Overfitting via Combinatorially-Symmetric Cross-Validation.

    Bailey, Borwein, López de Prado & Zhu (2017). Given an (T × N) matrix of trial returns:

      1. Partition the T rows into ``S`` (=``n_splits``, even) contiguous, equal blocks.
      2. For every way of choosing S/2 blocks as in-sample (the rest out-of-sample) — a
         balanced, symmetric split — pick the trial with the highest in-sample Sharpe.
      3. Find that champion's *rank* among all trials out-of-sample; its relative rank
         ``ω = rank/(N+1)``. Define the logit ``λ = ln(ω/(1-ω))``.
      4. PBO = fraction of splits where the in-sample champion lands in the bottom half
         out-of-sample (``λ < 0``).

    PBO near 0 → in-sample selection generalises; PBO near 0.5 → selection is worthless;
    PBO > 0.5 → selection is *anti-predictive* (the in-sample winner tends to be an
    out-of-sample loser — the signature of overfitting).

    Returns ``{"pbo": ..., "n_combos": ..., "lambdas": [...], "median_oos_rank": ...}``.
    """
    M = returns_matrix.dropna(how="any")
    T, N = M.shape
    if N < 2 or T < n_splits:
        return {"pbo": float("nan"), "n_combos": 0, "lambdas": [], "median_oos_rank": float("nan")}
    if n_splits % 2 == 1:
        n_splits += 1

    # Contiguous equal-ish blocks.
    bounds = np.linspace(0, T, n_splits + 1).astype(int)
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_splits)]
    arr = M.to_numpy(dtype=float)

    half = n_splits // 2
    combos = list(itertools.combinations(range(n_splits), half))
    # Symmetric: each (IS, OOS) appears with its complement; enumerate all C(S, S/2) and let
    # the symmetry fall out (the canonical CSCV uses all such partitions).
    lambdas: list[float] = []
    oos_ranks: list[float] = []
    for is_blocks in combos:
        is_set = set(is_blocks)
        is_idx = np.concatenate([blocks[b] for b in range(n_splits) if b in is_set])
        oos_idx = np.concatenate([blocks[b] for b in range(n_splits) if b not in is_set])
        is_r = arr[is_idx]
        oos_r = arr[oos_idx]

        def _sr(x):
            mu = x.mean(axis=0)
            sd = x.std(axis=0, ddof=1)
            sd = np.where(sd > 0, sd, np.inf)
            return mu / sd

        is_sr = _sr(is_r)
        champ = int(np.nanargmax(is_sr))
        oos_sr = _sr(oos_r)
        # Rank of champion OOS (1 = worst, N = best); relative rank ω in (0,1).
        order = np.argsort(np.argsort(oos_sr))  # 0..N-1, higher = better
        rank = order[champ] + 1
        omega = rank / (N + 1.0)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        lambdas.append(float(math.log(omega / (1.0 - omega))))
        oos_ranks.append(float(rank))

    lam = np.array(lambdas)
    pbo = float((lam < 0).mean()) if lam.size else float("nan")
    return {
        "pbo": pbo,
        "n_combos": len(combos),
        "lambdas": lambdas,
        "median_oos_rank": float(np.median(oos_ranks)) if oos_ranks else float("nan"),
        "n_trials": int(N),
    }


# ---------------------------------------------------------------------------
# The headline demonstration — in-sample champion → out-of-sample collapse
# ---------------------------------------------------------------------------
def in_sample_champion(returns_matrix: pd.DataFrame, frac: float = 0.5) -> dict:
    """Split a trial matrix into the first ``frac`` (in-sample) / rest (out-of-sample),
    pick the in-sample Sharpe champion, and report its IS vs OOS Sharpe.

    This is the picture the whole study is about: the gorgeous IS Sharpe of the *selected*
    rule, and what it does live (OOS). Returns the champion's column name and both Sharpes.
    """
    M = returns_matrix.dropna(how="any")
    T = len(M)
    cut = int(T * frac)
    is_part = M.iloc[:cut]
    oos_part = M.iloc[cut:]
    is_sr = is_part.apply(sharpe)
    champ = is_sr.idxmax()
    return {
        "champion": str(champ),
        "is_sharpe": float(is_sr.loc[champ]),
        "oos_sharpe": float(sharpe(oos_part[champ])),
        "is_n": int(cut),
        "oos_n": int(T - cut),
        "oos_returns": oos_part[champ],
        "mean_is_sharpe": float(is_sr.mean()),
    }
