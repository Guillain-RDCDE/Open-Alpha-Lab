"""The Mag-7 engine and its honest controls — Study 355 (Magnificent-Seven).

The pitch: hold the seven (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA) equal-weight and watch
them crush the S&P 500. They did. The question is **why**, and whether you could have done it
*forward*:

- **Mag 7 (named basket)** — the seven, equal-weight, monthly-rebalanced. Its spread over the
  index is real on the 2015-2026 tape (that part is not in dispute).
- **Cap index / SPY** — the benchmark the basket is raced against.
- **Equal-weight field** — the whole mega-cap universe at 1/N, the *weighting* control: it
  strips out "equal-weight beat cap-weight" so the residual spread is *name selection*.
- **Ex-post placebo ("pick the 7 winners")** — with the full sample in view, select the seven
  *highest-realised-return* names of the field and hold them. This is the Mag-7 *selection rule*
  made explicit. It manufactures a large positive spread **even when no name has any true edge**
  (the synthetic null proves it). This is the look-ahead the legend hides.
- **Random 7-baskets** — many seeded random draws of seven names: the sampling distribution a
  *blindly chosen* seven would have produced. Where the Mag 7 sits in that distribution, vs
  where the ex-post placebo sits, is the whole story.

The decisive comparison is therefore **Mag-7 spread vs the ex-post-placebo spread**: if almost
all of the Mag-7 outperformance is reproduced by "pick the winners in hindsight" on a *no-edge*
tape, the basket is selection, not a factor. We report each leg's CAGR/Sharpe, the HAC t-stat of
the Mag-7-minus-benchmark spread (the Signal-axis test), the spread net of one-way costs on NAV,
and the placebo decomposition.

Costs: one-way turnover × NAV at ``cost_bps`` per rebalance. One execution lag where a basket is
formed from a signal; the named/random baskets need no signal lag (membership is fixed), and the
look-ahead the study measures is the *selection*, not the accounting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


# ---------------------------------------------------------------------------
# Portfolio return engine (with one-way turnover costs)
# ---------------------------------------------------------------------------
def _book_returns(
    returns: pd.DataFrame, weight_fn, rebalance_every: int, cost_bps: float
) -> pd.Series:
    """Net portfolio returns for a weight schedule, charging one-way turnover × NAV.

    ``weight_fn(t)`` returns the target weight vector held from rebalance period ``t`` onward.
    Turnover at a rebalance is ``sum(|w_new - w_drifted|)`` (one-way), charged at ``cost_bps``.
    Between rebalances weights drift with the assets (buy-and-hold inside a window).
    """
    R = returns.to_numpy(dtype=float)
    T, n = R.shape
    out = np.full(T, np.nan)
    w = None
    for t in range(T):
        if t % rebalance_every == 0:
            target = weight_fn(t)
            cost = 0.0 if w is None else float(np.abs(target - w).sum()) * cost_bps * 1e-4
            w = target.copy()
            out[t] = float(w @ R[t]) - cost
        else:
            out[t] = float(w @ R[t])
        grown = w * (1.0 + R[t])
        s = grown.sum()
        w = grown / s if s > 0 else np.full(n, 1.0 / n)
    return pd.Series(out, index=returns.index)


def basket_weights(columns, members) -> np.ndarray:
    """Equal-weight vector over ``columns`` placing 1/k on each name in ``members``."""
    cols = list(columns)
    members = [m for m in members if m in cols]
    w = np.zeros(len(cols))
    if members:
        idx = [cols.index(m) for m in members]
        w[idx] = 1.0 / len(members)
    return w


def basket_returns(
    returns: pd.DataFrame, members, rebalance_every: int = 1, cost_bps: float = 0.0
) -> pd.Series:
    """Equal-weight, monthly-rebalanced return series for a fixed named basket."""
    base = basket_weights(returns.columns, members)
    return _book_returns(returns, lambda t: base, rebalance_every, cost_bps)


def equal_field_returns(
    returns: pd.DataFrame, rebalance_every: int = 1, cost_bps: float = 0.0
) -> pd.Series:
    """Equal-weight 1/N over the WHOLE field — the weighting control."""
    n = returns.shape[1]
    base = np.full(n, 1.0 / n)
    return _book_returns(returns, lambda t: base, rebalance_every, cost_bps)


# ---------------------------------------------------------------------------
# The ex-post placebo — "pick the k winners, in hindsight"
# ---------------------------------------------------------------------------
def expost_winners(
    returns: pd.DataFrame, k: int = 7, allow_lookahead_selection: bool = False
) -> list[str]:
    """The ``k`` names with the highest *total realised return over the whole sample*.

    This is the Mag-7 selection rule made explicit: it uses the ENTIRE tape (look-ahead) to
    name the basket. Requires the opt-in flag, because the resulting "edge" is manufactured by
    selection — exactly the bias the study exposes.
    """
    if not allow_lookahead_selection:
        raise PermissionError(
            "expost_winners() ranks names by their FULL-SAMPLE realised return — pure "
            "look-ahead. Pass allow_lookahead_selection=True to opt in; the spread it earns "
            "is a selection artefact, not a tradable edge."
        )
    total = (1.0 + returns).prod(axis=0)            # terminal wealth per name (look-ahead)
    return list(total.sort_values(ascending=False).index[:k])


def random_baskets(
    returns: pd.DataFrame, k: int = 7, n_draws: int = 2000,
    rebalance_every: int = 1, cost_bps: float = 0.0, seed: int = 355
) -> pd.DataFrame:
    """``n_draws`` seeded random equal-weight k-baskets — the blind-pick sampling distribution.

    Each draw is a *fixed* random set of ``k`` names held the whole sample (a basket you could
    have named at the start by luck). Returns a (T x n_draws) frame of net return series.
    """
    cols = list(returns.columns)
    out = {}
    for d in range(n_draws):
        rng = np.random.default_rng(seed + d)
        members = list(rng.choice(cols, size=min(k, len(cols)), replace=False))
        out[f"rand{d:04d}"] = basket_returns(returns, members, rebalance_every, cost_bps)
    return pd.DataFrame(out, index=returns.index)


# ---------------------------------------------------------------------------
# Metrics & inference
# ---------------------------------------------------------------------------
def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def summarize(monthly_ret: pd.Series, periods_per_year: int = MONTHS_PER_YEAR) -> dict:
    """Headline metrics for one monthly return series (n, CAGR, Sharpe, max-dd, ann mean)."""
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
        "n": int(n), "cagr": float(cagr), "sharpe": float(sharpe),
        "max_dd": _max_drawdown(equity), "mean_ann": float(ann_mean),
    }


def hac_tstat_diff(r1: pd.Series, r2: pd.Series) -> dict:
    """Newey-West HAC t-stat on the per-period difference ``r1 - r2`` (the Signal-axis test).

    Both legs are fully invested, so the raw difference *is* the excess-of-each-other return
    (no cash leg to net out). Returns the mean monthly spread, its annualised value, the HAC
    t-stat, and n.
    """
    aligned = pd.concat([r1, r2], axis=1, join="inner").dropna()
    diff = (aligned.iloc[:, 0] - aligned.iloc[:, 1]).to_numpy(dtype=float)
    n = diff.size
    if n < 3:
        return {"mean_diff": float("nan"), "tstat": float("nan"), "n": n,
                "mean_diff_ann": float("nan")}
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


def percentile_of(value: float, sample: np.ndarray) -> float:
    """Percentile rank (0-100) of ``value`` within ``sample`` — where a basket sits."""
    s = np.asarray(sample, dtype=float)
    s = s[~np.isnan(s)]
    if s.size == 0:
        return float("nan")
    return float((s < value).mean() * 100.0)


def race(
    returns: pd.DataFrame,
    bench: pd.Series,
    members,
    k: int = 7,
    n_draws: int = 2000,
    rebalance_every: int = 1,
    cost_bps: float = 0.0,
    seed: int = 355,
    allow_lookahead_selection: bool = False,
) -> dict:
    """The full Mag-7 teardown: named basket vs index vs equal-field vs ex-post placebo.

    Returns a bundle:
      - ``mag7`` / ``mag7_net`` — the named basket gross / net of ``cost_bps``.
      - ``bench`` — the benchmark (cap index / SPY) series.
      - ``equal_field`` — 1/N over the whole universe (the weighting control).
      - ``expost`` — the look-ahead "pick the k winners" basket (requires the opt-in).
      - ``rand_spread`` — terminal-CAGR spread vs bench for each random blind basket.
      - ``test_vs_bench`` — HAC t-stat of the named basket minus the benchmark (Signal axis).
      - ``test_vs_equal`` — HAC t-stat of the named basket minus the equal-weight field.
      - ``mag7_cagr_spread`` / ``expost_cagr_spread`` / ``selection_share`` — the decomposition:
        how much of the Mag-7 CAGR spread the *ex-post selection* alone reproduces.
      - ``mag7_pctile`` — where the Mag-7 spread sits in the random-basket distribution.
    """
    mag7 = basket_returns(returns, members, rebalance_every, 0.0)
    mag7_net = basket_returns(returns, members, rebalance_every, cost_bps)
    eqf = equal_field_returns(returns, rebalance_every, 0.0)
    rands = random_baskets(returns, k=k, n_draws=n_draws,
                           rebalance_every=rebalance_every, cost_bps=0.0, seed=seed)

    s_bench = summarize(bench)["cagr"]
    s_mag7 = summarize(mag7)["cagr"]
    rand_cagr = np.array([summarize(rands[c])["cagr"] for c in rands.columns])
    rand_spread = rand_cagr - s_bench

    out = {
        "mag7": mag7, "mag7_net": mag7_net, "bench": bench, "equal_field": eqf,
        "rand_spread": rand_spread,
        "test_vs_bench": hac_tstat_diff(mag7, bench),
        "test_vs_equal": hac_tstat_diff(mag7, eqf),
        "mag7_cagr_spread": float(s_mag7 - s_bench),
        "mag7_pctile": percentile_of(s_mag7 - s_bench, rand_spread),
        "mag7_net_cagr": summarize(mag7_net)["cagr"],
    }

    if allow_lookahead_selection:
        winners = expost_winners(returns, k=k, allow_lookahead_selection=True)
        expost = basket_returns(returns, winners, rebalance_every, 0.0)
        s_expost = summarize(expost)["cagr"]
        out["expost"] = expost
        out["expost_members"] = winners
        out["expost_cagr_spread"] = float(s_expost - s_bench)
        # Selection share: how much of the Mag-7 spread the blind ex-post rule reproduces.
        denom = out["mag7_cagr_spread"]
        out["selection_share"] = float(out["expost_cagr_spread"] / denom) if denom else float("nan")
    return out
