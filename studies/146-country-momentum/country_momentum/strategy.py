"""Strategy and honest controls — Study 146 (Country-Momentum).

The claim: rank country ETFs by their 12-1 month total return (skip the most recent month
to avoid reversal-contamination), go **long the top-K** each month, rebalanced monthly.
We benchmark against two controls:

1. **Equal-weight (EW) all-country basket** — the naive benchmark; "momentum add value
   vs just holding all countries equally?"
2. **Random-rotation control** — each month pick K countries at random (seeded) and hold
   them equal-weight; "does the 12-1 signal add anything over an uninformed strategy of the
   same size?"

The honest verdict requires beating the random control on a HAC t-stat on monthly returns.

Implementation choices (no look-ahead):
- The 12-1 score for month ``t`` uses closes up to month ``t-1`` (skip=1) and
  compares to the close 12 months earlier (lookback=12, so the score at month ``t``
  uses returns from ``t-13`` to ``t-1``). The position is entered at the start of
  month ``t+1`` (one period lag after scoring), exits at the end of ``t+1``.
- Costs: a round-trip ``cost_bps`` is charged proportional to turnover each month.
- Equal-weight within the top-K; no leverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_MONTHS_PER_YEAR = 12


# ---------------------------------------------------------------------------
# Signal: 12-1 momentum score
# ---------------------------------------------------------------------------
def momentum_score(
    panel: pd.DataFrame,
    lookback: int = 12,
    skip: int = 1,
) -> pd.DataFrame:
    """Cross-sectional 12-1 momentum score (total return over lookback-skip months).

    At each month ``t``, the score for country ``i`` is::

        score_{i,t} = (price_{i,t-skip} / price_{i,t-skip-lookback}) - 1

    Computed from cumulative returns. Returns a frame of the same shape as ``panel``,
    with NaN where there is insufficient history.
    """
    # cumulative-return index from the monthly return panel
    cum = (1.0 + panel).cumprod()
    # shift(skip) gives price at t-skip; shift(skip+lookback) gives t-skip-lookback
    score = cum.shift(skip) / cum.shift(skip + lookback) - 1.0
    return score


# ---------------------------------------------------------------------------
# Portfolio weights
# ---------------------------------------------------------------------------
def top_k_weights(
    score_row: pd.Series,
    k: int,
    min_count: int | None = None,
) -> pd.Series:
    """Equal-weight top-K countries by score, returning a weight vector.

    ``score_row`` is the cross-section of scores at one month; NaN entries are
    dropped before ranking so a country with no history is never included. If
    fewer than ``min_count`` (default ``k``) valid scores are available,
    returns an all-zero vector.
    """
    valid = score_row.dropna()
    mc = min_count if min_count is not None else k
    if len(valid) < mc:
        return pd.Series(0.0, index=score_row.index)
    ranked = valid.nlargest(k)
    w = pd.Series(0.0, index=score_row.index)
    w[ranked.index] = 1.0 / k
    return w


def ew_weights(score_row: pd.Series) -> pd.Series:
    """Equal-weight all countries with a valid score at this month."""
    valid = score_row.dropna()
    if valid.empty:
        return pd.Series(0.0, index=score_row.index)
    w = pd.Series(0.0, index=score_row.index)
    w[valid.index] = 1.0 / len(valid)
    return w


def random_weights(
    score_row: pd.Series,
    k: int,
    rng: np.random.Generator,
) -> pd.Series:
    """Pick K countries uniformly at random from those with a valid score."""
    valid = score_row.dropna().index.tolist()
    w = pd.Series(0.0, index=score_row.index)
    if len(valid) < k:
        return w
    chosen = rng.choice(valid, size=k, replace=False)
    w[chosen] = 1.0 / k
    return w


# ---------------------------------------------------------------------------
# Monthly return streams
# ---------------------------------------------------------------------------
def _turnover(w_prev: pd.Series, w_cur: pd.Series) -> float:
    """One-way turnover = 0.5 * sum(|w_cur - w_prev|)."""
    return 0.5 * float((w_cur - w_prev).abs().sum())


def run_strategy(
    panel: pd.DataFrame,
    k: int = 4,
    lookback: int = 12,
    skip: int = 1,
    cost_bps: float = 10.0,
    seed: int | None = None,
    mode: str = "momentum",
) -> pd.Series:
    """Backtest monthly country momentum, EW, or random-rotation, net of costs.

    Parameters
    ----------
    panel : pd.DataFrame
        Monthly-return panel (rows = months, columns = country ETF tickers).
    k : int
        Number of countries to hold (top-K for momentum/random; ignored for EW).
    lookback : int
        Look-back horizon for the 12-1 score (months of total return, default 12).
    skip : int
        Skip-month (to avoid 1-month reversal contamination, default 1).
    cost_bps : float
        Round-trip trading cost per rebalance, in basis points of notional traded.
    seed : int, optional
        Seed for the random-rotation control (``mode='random'`` only).
    mode : str
        ``'momentum'`` (top-K), ``'ew'`` (equal-weight all), or ``'random'`` (random-K).

    Returns
    -------
    pd.Series
        Monthly net returns, indexed to the same period index as ``panel``.
    """
    score = momentum_score(panel, lookback=lookback, skip=skip)
    rng = np.random.default_rng(seed if seed is not None else 0)

    w_prev = pd.Series(0.0, index=panel.columns)
    rets_out: list[tuple] = []
    c = cost_bps * 1e-4

    for t in panel.index:
        # weights are formed on the score known BEFORE month t's return is observed
        sc = score.loc[t]
        if mode == "momentum":
            w_cur = top_k_weights(sc, k)
        elif mode == "ew":
            w_cur = ew_weights(sc)
        elif mode == "random":
            w_cur = random_weights(sc, k, rng)
        else:
            raise ValueError(f"mode must be 'momentum', 'ew', or 'random'; got {mode!r}")

        # gross return for this month: enter at start-of-month (previous weight shift)
        r_gross = float((w_prev * panel.loc[t].fillna(0.0)).sum())
        to = _turnover(w_prev, w_cur)
        r_net = r_gross - c * to * 2.0  # round-trip = enter + exit
        rets_out.append((t, r_gross, r_net))
        w_prev = w_cur

    df = pd.DataFrame(rets_out, columns=["date", "r_gross", "r_net"]).set_index("date")
    return df


# ---------------------------------------------------------------------------
# Summary statistics (HAC t-stat on monthly returns)
# ---------------------------------------------------------------------------
def summarize(
    returns: pd.Series,
    periods_per_year: int = TRADING_MONTHS_PER_YEAR,
) -> dict:
    """Headline statistics for a monthly net-return series.

    Returns trade count, annualised mean, Sharpe, max-drawdown, and — most
    importantly — the Newey-West HAC t-stat on the mean (the inference bar).
    """
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: float("nan") for k in
                ("n", "ann_ret", "ann_vol", "sharpe", "max_drawdown", "skew", "tstat")}
    mean_m = float(r.mean())
    std_m = float(r.std(ddof=1))
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()

    # Newey-West long-run variance
    e = r.to_numpy() - mean_m
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        lrv += 2.0 * w * float(e[lag:] @ e[:-lag]) / n
    se = np.sqrt(max(lrv, 0.0) / n)

    sharpe = (mean_m / std_m * np.sqrt(periods_per_year)) if std_m > 0 else float("nan")
    return {
        "n": int(n),
        "ann_ret": float(mean_m * periods_per_year),
        "ann_vol": float(std_m * np.sqrt(periods_per_year)),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd),
        "skew": float(r.skew()),
        "tstat": float(mean_m / se) if se > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Cost sweep helper
# ---------------------------------------------------------------------------
def cost_sweep(
    panel: pd.DataFrame,
    k: int = 4,
    lookback: int = 12,
    skip: int = 1,
    costs_bps: tuple = (0, 5, 10, 20, 40),
) -> pd.DataFrame:
    """Momentum net Sharpe and annual return at each cost level vs the EW benchmark."""
    rows = []
    ew_df = run_strategy(panel, k=k, lookback=lookback, skip=skip, cost_bps=0, mode="ew")
    ew_s = summarize(ew_df["r_net"])
    for c in costs_bps:
        mom_df = run_strategy(panel, k=k, lookback=lookback, skip=skip, cost_bps=c, mode="momentum")
        mom_s = summarize(mom_df["r_net"])
        rows.append({
            "cost_bps": c,
            "mom_sharpe": mom_s["sharpe"],
            "mom_ann_ret": mom_s["ann_ret"],
            "ew_sharpe": ew_s["sharpe"],
            "ew_ann_ret": ew_s["ann_ret"],
            "active_sharpe": mom_s["sharpe"] - ew_s["sharpe"],
        })
    return pd.DataFrame(rows).set_index("cost_bps")


# ---------------------------------------------------------------------------
# Top-K vs random control (edge vs uninformed rotation)
# ---------------------------------------------------------------------------
def momentum_vs_random(
    panel: pd.DataFrame,
    k: int = 4,
    lookback: int = 12,
    skip: int = 1,
    cost_bps: float = 0.0,
    n_seeds: int = 50,
) -> dict:
    """Compare momentum's monthly returns to a distribution of random-rotation controls.

    Returns the momentum mean and HAC t, plus the mean and std of random-rotation means
    over ``n_seeds`` random seeds — the 'can it beat an uninformed picker?' test.
    """
    mom_df = run_strategy(panel, k=k, lookback=lookback, skip=skip, cost_bps=cost_bps,
                          mode="momentum")
    mom_s = summarize(mom_df["r_gross"])

    rand_means = []
    for s in range(n_seeds):
        rd = run_strategy(panel, k=k, lookback=lookback, skip=skip, cost_bps=cost_bps,
                          mode="random", seed=s)
        rand_means.append(float(rd["r_gross"].mean()))

    rand_arr = np.array(rand_means)
    return {
        "mom_mean_bps": mom_s["ann_ret"] / 12 * 1e4,
        "mom_tstat": mom_s["tstat"],
        "mom_sharpe": mom_s["sharpe"],
        "rand_mean_bps": float(rand_arr.mean()) * 1e4 / 12 * 12,  # annualised
        "rand_mean_monthly_bps": float(rand_arr.mean() * 1e4),
        "rand_std_bps": float(rand_arr.std() * 1e4),
        "n_rand_seeds": n_seeds,
        "edge_vs_random_bps": float(mom_df["r_gross"].mean() * 1e4 - rand_arr.mean() * 1e4),
    }
