"""The AI-Datacenter-Basket engine and its honest controls — Study 393.

The pitch: hold the eight (NVDA, VRT, ETN, CEG, VST, SMCI, ANET, DELL) equal-weight and ride the
AI build-out as the market and tech crawl behind. They did. The question is **why**, and whether
you could have done it *forward*:

- **Basket (named)** — the eight, equal-weight, monthly-rebalanced. Its spread over the index is
  real on the 2019-2026 tape (that part is not in dispute).
- **SPY / QQQ** — the market and the tech tape the basket is raced against (excess-vs-excess: both
  legs fully invested, so the raw difference *is* the excess).
- **Equal-weight field** — the whole candidate universe at 1/N, the *theme-beta* control: it
  strips out "datacenter stuff went up" so the residual spread is *name selection within the theme*.
- **Ex-post placebo ("pick the 8 winners")** — with the full sample in view, select the eight
  *highest-realised-return* names of the field and hold them. This is the basket's *selection rule*
  made explicit. It manufactures a large positive spread **even when no name has any true edge**
  (the synthetic null proves it). This is the look-ahead the legend hides.
- **Random 8-baskets** — many seeded random draws of eight names: the sampling distribution a
  *blindly chosen* eight would have produced. Where the headline basket sits in that distribution,
  vs where the ex-post placebo sits, is the whole story.

The decisive comparison is therefore **basket spread vs the ex-post-placebo spread**: if almost
all of the basket's outperformance is reproduced by "pick the winners in hindsight" on a *no-edge*
tape, the basket is selection, not a theme factor. We report each leg's CAGR/Sharpe/maxDD, the HAC
t-stat of the basket-minus-benchmark spread (the Signal-axis test, vs both SPY and QQQ), the spread
net of one-way costs on NAV, and the placebo decomposition.

Costs: one-way turnover × NAV at ``cost_bps`` per rebalance. The fixed named / random baskets need
no signal lag (membership is constant); the look-ahead the study measures is the *selection*, not
the accounting.
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
    """Equal-weight 1/N over the WHOLE field — the theme-beta control."""
    n = returns.shape[1]
    base = np.full(n, 1.0 / n)
    return _book_returns(returns, lambda t: base, rebalance_every, cost_bps)


# ---------------------------------------------------------------------------
# The ex-post placebo — "pick the k winners, in hindsight"
# ---------------------------------------------------------------------------
def expost_winners(
    returns: pd.DataFrame, k: int = 8, allow_lookahead_selection: bool = False
) -> list[str]:
    """The ``k`` names with the highest *total realised return over the whole sample*.

    This is the basket's selection rule made explicit: it uses the ENTIRE tape (look-ahead) to name
    the basket. Requires the opt-in flag, because the resulting "edge" is manufactured by selection
    — exactly the bias the study exposes.
    """
    if not allow_lookahead_selection:
        raise PermissionError(
            "expost_winners() ranks names by their FULL-SAMPLE realised return — pure look-ahead. "
            "Pass allow_lookahead_selection=True to opt in; the spread it earns is a selection "
            "artefact, not a tradable edge."
        )
    total = (1.0 + returns).prod(axis=0)            # terminal wealth per name (look-ahead)
    return list(total.sort_values(ascending=False).index[:k])


def random_baskets(
    returns: pd.DataFrame, k: int = 8, n_draws: int = 2000,
    rebalance_every: int = 1, cost_bps: float = 0.0, seed: int = 393
) -> pd.DataFrame:
    """``n_draws`` seeded random equal-weight k-baskets — the blind-pick sampling distribution.

    Each draw is a *fixed* random set of ``k`` names held the whole sample (a basket you could have
    named at the start by luck). Returns a (T x n_draws) frame of net return series.
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

    Both legs are fully invested, so the raw difference *is* the excess-of-each-other return (no
    cash leg to net out). Returns the mean monthly spread, its annualised value, the HAC t-stat,
    and n.
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


def single_name_tstats(returns: pd.DataFrame, bench: pd.Series, members) -> dict:
    """HAC t of each named member's monthly excess over ``bench`` — is the basket one name?

    Returns ``{ticker: {"mean_ann": .., "tstat": ..}}`` for each member present in the panel.
    """
    out = {}
    for m in members:
        if m in returns.columns:
            t = hac_tstat_diff(returns[m], bench)
            out[m] = {"mean_ann": t["mean_diff_ann"], "tstat": t["tstat"]}
    return out


def race(
    returns: pd.DataFrame,
    bench: pd.DataFrame,
    members,
    k: int = 8,
    n_draws: int = 2000,
    rebalance_every: int = 1,
    cost_bps: float = 0.0,
    seed: int = 393,
    allow_lookahead_selection: bool = False,
) -> dict:
    """The full teardown: named basket vs market vs tech vs equal-field vs ex-post placebo.

    ``bench`` is a 2-column frame with ``spy`` and ``qqq`` monthly returns. Returns a bundle:
      - ``basket`` / ``basket_net`` — the named basket gross / net of ``cost_bps``.
      - ``spy`` / ``qqq`` — the benchmark series (market / tech tape).
      - ``equal_field`` — 1/N over the whole field (the theme-beta control).
      - ``expost`` — the look-ahead "pick the k winners" basket (requires the opt-in).
      - ``rand_spread`` — terminal-CAGR spread vs SPY for each random blind basket.
      - ``test_vs_spy`` / ``test_vs_qqq`` / ``test_vs_equal`` — HAC t-stats of the basket spread.
      - ``basket_cagr_spread`` / ``expost_cagr_spread`` / ``selection_share`` — the decomposition:
        how much of the basket CAGR spread the *ex-post selection* alone reproduces.
      - ``basket_pctile`` — where the basket spread sits in the random-basket distribution.
      - ``single_names`` — per-member HAC t of excess over SPY (is it one name?).
    """
    spy = bench["spy"]
    qqq = bench["qqq"]
    basket = basket_returns(returns, members, rebalance_every, 0.0)
    basket_net = basket_returns(returns, members, rebalance_every, cost_bps)
    eqf = equal_field_returns(returns, rebalance_every, 0.0)
    rands = random_baskets(returns, k=k, n_draws=n_draws,
                           rebalance_every=rebalance_every, cost_bps=0.0, seed=seed)

    s_spy = summarize(spy)["cagr"]
    s_basket = summarize(basket)["cagr"]
    rand_cagr = np.array([summarize(rands[c])["cagr"] for c in rands.columns])
    rand_spread = rand_cagr - s_spy

    out = {
        "basket": basket, "basket_net": basket_net, "spy": spy, "qqq": qqq,
        "equal_field": eqf, "rand_spread": rand_spread,
        "test_vs_spy": hac_tstat_diff(basket, spy),
        "test_vs_qqq": hac_tstat_diff(basket, qqq),
        "test_vs_equal": hac_tstat_diff(basket, eqf),
        "basket_cagr_spread": float(s_basket - s_spy),
        "basket_pctile": percentile_of(s_basket - s_spy, rand_spread),
        "basket_net_cagr": summarize(basket_net)["cagr"],
        "single_names": single_name_tstats(returns, spy, members),
    }

    if allow_lookahead_selection:
        winners = expost_winners(returns, k=k, allow_lookahead_selection=True)
        expost = basket_returns(returns, winners, rebalance_every, 0.0)
        s_expost = summarize(expost)["cagr"]
        out["expost"] = expost
        out["expost_members"] = winners
        out["expost_cagr_spread"] = float(s_expost - s_spy)
        denom = out["basket_cagr_spread"]
        out["selection_share"] = float(out["expost_cagr_spread"] / denom) if denom else float("nan")
    return out
