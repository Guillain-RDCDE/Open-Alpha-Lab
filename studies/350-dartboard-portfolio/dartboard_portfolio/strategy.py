"""The dartboard engine and its honest controls — Study 350 (Dartboard-Portfolio).

Burton Malkiel's quip (*A Random Walk Down Wall Street*, 1973): *"A blindfolded monkey
throwing darts at a newspaper's financial pages could select a portfolio that would do
just as well as one carefully selected by experts."* The famous follow-up — the Wall
Street Journal's dartboard contest, and the Research Affiliates monkey study — found the
random portfolios *beating* the experts and the cap-weighted index startlingly often.

We run the experiment as three competitors on the *same* universe and the *same*
rebalance schedule:

- **Dartboard** — pick ``k`` names at random, hold them **equal-weight**, rebalance each
  window with a fresh random draw. We run ``n_darts`` independent monkeys (seeded) to get
  a distribution, not one lucky throw.
- **Cap-weighted index** — hold every name at its market-cap weight. The benchmark the
  monkey is racing.
- **Expert (concentrated)** — hold the ``k`` *largest* names equal-weight, a stand-in for
  the consensus "obvious winners" pick. (A documented, deterministic proxy for the
  experts the dartboard is supposed to embarrass.)

The decisive question is **not** "does the dartboard win?" (it usually does) but **why**:
random equal-weight selection over-weights the many small names the cap index barely
holds, so a positive small-cap premium hands the monkey a *size tilt*, not skill. We
therefore measure the dartboard-minus-index spread, its block-bootstrap CI, its HAC
*t*-stat, and — the honest control — whether the spread *survives equalising the size
exposure* (an equal-weight index, which removes the tilt the dartboard is exploiting).

Costs: one-way turnover × NAV at ``cost_bps`` per rebalance (the dartboard's full random
re-draw is high-turnover — a cost the legend never mentions). One execution lag is not
needed: the index weights are calendar-known and the dart picks are random.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTHS_PER_YEAR = 12


# ---------------------------------------------------------------------------
# Weight builders
# ---------------------------------------------------------------------------
def cap_weights(caps: pd.Series) -> np.ndarray:
    """Market-cap weights — the index. Falls back to equal weight on a degenerate cap."""
    w = caps.to_numpy(dtype=float)
    total = w.sum()
    if total <= 0:
        return np.full(len(w), 1.0 / len(w))
    return w / total


def equal_weights(n: int) -> np.ndarray:
    """The 1/N vector for an n-name basket."""
    return np.full(n, 1.0 / n)


def dart_weights(n_assets: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """One blindfolded throw: pick ``k`` of ``n_assets`` at random, equal-weight them."""
    k = min(k, n_assets)
    pick = rng.choice(n_assets, size=k, replace=False)
    w = np.zeros(n_assets)
    w[pick] = 1.0 / k
    return w


def expert_weights(caps: pd.Series, k: int) -> np.ndarray:
    """The 'expert' pick: the ``k`` largest names, equal-weight (consensus mega-caps)."""
    n = len(caps)
    k = min(k, n)
    order = np.argsort(caps.to_numpy(dtype=float))[::-1]  # largest first
    w = np.zeros(n)
    w[order[:k]] = 1.0 / k
    return w


# ---------------------------------------------------------------------------
# Portfolio return engine (with one-way turnover costs)
# ---------------------------------------------------------------------------
def _book_returns(
    returns: pd.DataFrame,
    weight_fn,
    rebalance_every: int,
    cost_bps: float,
) -> pd.Series:
    """Net portfolio returns for a weight schedule, charging one-way turnover × NAV.

    ``weight_fn(t)`` returns the target weight vector to hold from rebalance period ``t``
    onward. Turnover at a rebalance is ``sum(|w_new - w_drifted|)`` (one-way), charged at
    ``cost_bps``. Between rebalances weights drift with the assets (buy-and-hold inside a
    window), so the cost is realistic, not per-period.
    """
    R = returns.to_numpy(dtype=float)
    T, n = R.shape
    out = np.full(T, np.nan)

    w = None          # current (drifted) weights
    for t in range(T):
        if t % rebalance_every == 0:
            target = weight_fn(t)
            if w is None:
                cost = 0.0  # no cost to establish the very first book
            else:
                cost = float(np.abs(target - w).sum()) * cost_bps * 1e-4
            w = target.copy()
            net_t = float(w @ R[t]) - cost
        else:
            net_t = float(w @ R[t])
        out[t] = net_t
        # Drift the weights by this period's returns.
        grown = w * (1.0 + R[t])
        s = grown.sum()
        w = grown / s if s > 0 else equal_weights(n)
    return pd.Series(out, index=returns.index)


def index_returns(
    returns: pd.DataFrame, caps: pd.Series, rebalance_every: int = 12, cost_bps: float = 0.0
) -> pd.Series:
    """Cap-weighted index return series."""
    base = cap_weights(caps)
    return _book_returns(returns, lambda t: base, rebalance_every, cost_bps)


def equal_index_returns(
    returns: pd.DataFrame, rebalance_every: int = 12, cost_bps: float = 0.0
) -> pd.Series:
    """Equal-weight index (the whole universe at 1/N) — the size-tilt control."""
    n = returns.shape[1]
    base = equal_weights(n)
    return _book_returns(returns, lambda t: base, rebalance_every, cost_bps)


def expert_returns(
    returns: pd.DataFrame, caps: pd.Series, k: int = 10,
    rebalance_every: int = 12, cost_bps: float = 0.0
) -> pd.Series:
    """Concentrated 'expert' (k largest names, equal-weight) return series."""
    base = expert_weights(caps, k)
    return _book_returns(returns, lambda t: base, rebalance_every, cost_bps)


def dartboard_returns(
    returns: pd.DataFrame, k: int = 10, n_darts: int = 200,
    rebalance_every: int = 12, cost_bps: float = 0.0, seed: int = 350
) -> pd.DataFrame:
    """Run ``n_darts`` independent random-selection monkeys; return a (T x n_darts) frame.

    Each monkey re-draws ``k`` random names at every rebalance (a fresh blindfolded
    throw), holds them equal-weight, and drifts between rebalances. The cross-monkey
    distribution is the dartboard's sampling distribution — one throw is luck, the median
    is the strategy.
    """
    n_assets = returns.shape[1]
    cols = {}
    for d in range(n_darts):
        rng = np.random.default_rng(seed + d)
        # A per-rebalance fresh draw: capture rng in a closure keyed on period.
        draws: dict[int, np.ndarray] = {}

        def wfn(t, _rng=rng, _draws=draws):
            if t not in _draws:
                _draws[t] = dart_weights(n_assets, k, _rng)
            return _draws[t]

        cols[f"dart{d:04d}"] = _book_returns(returns, wfn, rebalance_every, cost_bps)
    return pd.DataFrame(cols, index=returns.index)


# ---------------------------------------------------------------------------
# Metrics & inference
# ---------------------------------------------------------------------------
def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def summarize(monthly_ret: pd.Series, periods_per_year: int = MONTHS_PER_YEAR) -> dict:
    """Headline metrics for one monthly return series.

    Returns n (months), CAGR, annualised Sharpe, max drawdown, and the win flag is left
    to the caller. Sharpe is excess-of-zero here; race comparisons use excess-vs-excess
    differences in ``hac_tstat_diff``.
    """
    r = monthly_ret.dropna()
    n = len(r)
    if n < 2:
        return {k: float("nan") for k in ("n", "cagr", "sharpe", "max_dd", "mean_ann")}
    total = float((1.0 + r).prod())
    n_years = n / periods_per_year
    cagr = total ** (1.0 / n_years) - 1.0 if (n_years > 0 and total > 0) else float("nan")
    ann_mean = r.mean() * periods_per_year
    ann_vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = ann_mean / ann_vol if ann_vol > 0 else float("nan")
    equity = (1.0 + r).cumprod().to_numpy()
    return {
        "n": int(n),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_dd": _max_drawdown(equity),
        "mean_ann": float(ann_mean),
    }


def hac_tstat_diff(r1: pd.Series, r2: pd.Series) -> dict:
    """Newey-West HAC t-stat on the per-period difference ``r1 - r2``.

    Tests whether series 1 (e.g. the median dartboard) significantly out- or under-earns
    series 2 (e.g. the cap index). Excess-vs-excess: both legs are fully invested, so the
    raw difference *is* the excess-of-each-other return — no cash leg to net out.
    """
    aligned = pd.concat([r1, r2], axis=1, join="inner").dropna()
    diff = (aligned.iloc[:, 0] - aligned.iloc[:, 1]).to_numpy(dtype=float)
    n = diff.size
    if n < 3:
        return {"mean_diff": float("nan"), "tstat": float("nan"), "n": n}
    mu = diff.mean()
    e = diff - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        wk = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * wk * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return {
        "mean_diff": float(mu),
        "tstat": float(mu / se) if se > 0 else float("nan"),
        "n": n,
        "mean_diff_ann": float(mu * MONTHS_PER_YEAR),
    }


def block_bootstrap_ci(
    diff: pd.Series, block: int = 6, n_boot: int = 2000, seed: int = 350, alpha: float = 0.05
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the *mean* of a difference series.

    Block resampling preserves the autocorrelation that i.i.d. resampling would destroy.
    Returns the (lower, upper) percentile interval on the mean per-period difference.
    """
    x = diff.dropna().to_numpy(dtype=float)
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


def dartboard_win_rate(darts: pd.DataFrame, benchmark: pd.Series) -> float:
    """Fraction of monkeys whose terminal wealth beats the benchmark's terminal wealth."""
    aligned = darts.join(benchmark.rename("bench"), how="inner").dropna()
    if aligned.empty:
        return float("nan")
    term = (1.0 + aligned).prod()
    bench_term = term["bench"]
    dart_terms = term.drop("bench")
    return float((dart_terms > bench_term).mean())


def race(
    returns: pd.DataFrame,
    caps: pd.Series,
    k: int = 10,
    n_darts: int = 200,
    rebalance_every: int = 12,
    cost_bps: float = 0.0,
    seed: int = 350,
) -> dict:
    """Run the full dartboard vs index vs expert race and return a results bundle.

    Keys: ``darts`` (T x n_darts), ``index``, ``equal_index``, ``expert`` (Series),
    ``median_dart`` (Series), ``win_rate_vs_index``, ``win_rate_vs_expert``, and the HAC
    test of the *median* dartboard minus the cap index (and minus the equal index — the
    size-tilt control).
    """
    darts = dartboard_returns(returns, k=k, n_darts=n_darts,
                              rebalance_every=rebalance_every, cost_bps=cost_bps, seed=seed)
    idx = index_returns(returns, caps, rebalance_every=rebalance_every, cost_bps=cost_bps)
    eq = equal_index_returns(returns, rebalance_every=rebalance_every, cost_bps=cost_bps)
    exp = expert_returns(returns, caps, k=k, rebalance_every=rebalance_every, cost_bps=cost_bps)
    median_dart = darts.median(axis=1)

    return {
        "darts": darts,
        "index": idx,
        "equal_index": eq,
        "expert": exp,
        "median_dart": median_dart,
        "win_rate_vs_index": dartboard_win_rate(darts, idx),
        "win_rate_vs_expert": dartboard_win_rate(darts, exp),
        "test_vs_index": hac_tstat_diff(median_dart, idx),
        "test_vs_equal_index": hac_tstat_diff(median_dart, eq),
    }
