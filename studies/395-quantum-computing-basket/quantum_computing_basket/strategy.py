"""The Quantum-Computing-Basket engine and its honest controls — Study 395.

The pitch: hold the four pure-play quantum-computing stocks (IONQ, RGTI, QBTS, QUBT) equal-weight
and ride the "next big thing" while the market crawls behind. On the raw tape the basket's CAGR is
enormous — which is exactly what makes it a hype-cycle case study rather than a strategy. The
question is whether *any* of that survives the obvious risk adjustments:

- **Basket (the four pure-plays)** — equal-weight, monthly-rebalanced. Its *raw* spread over the
  market is huge; its *Sharpe* is the test.
- **SPY / QQQ** — the market and the tech tape the basket is raced against (excess-vs-excess: both
  legs fully invested, so the raw difference *is* the excess).
- **QTUM ETF** — the diversified "play quantum" fund (a ~70-name disruptive-tech index). The
  control for "could you have ridden the theme *without* the lottery?" — the sane-basket benchmark.
- **Per-name decomposition** — is the basket's number a broad theme, or one or two moon-shots?

The decisive comparisons are **(a) the HAC t-stat of the basket-minus-benchmark spread** (does the
huge raw spread clear t = 2, given the enormous volatility?) and **(b) the Sharpe / drawdown race**
(is the risk-adjusted return even competitive with SPY, let alone the diversified ETF?). A big CAGR
with a sub-market Sharpe, an 80%-class drawdown, and a sub-2 t is a hype cycle, not an edge.

Costs: one-way turnover × NAV at ``cost_bps`` per rebalance. The fixed equal-weight basket needs no
signal lag (membership is constant); costs are negligible relative to the volatility, which is
itself a finding (frequency/cost is not what kills this — risk is).
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


# ---------------------------------------------------------------------------
# Metrics & inference
# ---------------------------------------------------------------------------
def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def summarize(monthly_ret: pd.Series, periods_per_year: int = MONTHS_PER_YEAR) -> dict:
    """Headline metrics for one monthly return series (n, CAGR, Sharpe, vol, max-dd, ann mean)."""
    r = monthly_ret.dropna()
    n = len(r)
    if n < 2:
        return {k: float("nan") for k in
                ("n", "cagr", "sharpe", "vol", "max_dd", "mean_ann")}
    total = float((1.0 + r).prod())
    n_years = n / periods_per_year
    cagr = total ** (1.0 / n_years) - 1.0 if (n_years > 0 and total > 0) else float("nan")
    ann_mean = r.mean() * periods_per_year
    ann_vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = ann_mean / ann_vol if ann_vol > 0 else float("nan")
    equity = (1.0 + r).cumprod().to_numpy()
    return {
        "n": int(n), "cagr": float(cagr), "sharpe": float(sharpe),
        "vol": float(ann_vol), "max_dd": _max_drawdown(equity),
        "mean_ann": float(ann_mean),
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


def single_name_tstats(returns: pd.DataFrame, bench: pd.Series, members) -> dict:
    """HAC t of each named member's monthly excess over ``bench`` (+ CAGR / vol / max-dd).

    Is the basket a broad theme, or one or two moon-shots that no single one clears t = 2?
    Returns ``{ticker: {"mean_ann", "tstat", "cagr", "vol", "max_dd"}}`` for each present member.
    """
    out = {}
    for m in members:
        if m in returns.columns:
            t = hac_tstat_diff(returns[m], bench)
            s = summarize(returns[m])
            out[m] = {"mean_ann": t["mean_diff_ann"], "tstat": t["tstat"],
                      "cagr": s["cagr"], "vol": s["vol"], "max_dd": s["max_dd"]}
    return out


def race(
    returns: pd.DataFrame,
    bench: pd.DataFrame,
    members,
    rebalance_every: int = 1,
    cost_bps: float = 0.0,
    etf: str = "QTUM",
    allow_survivorship: bool = False,
) -> dict:
    """The full teardown: the pure-play basket vs market vs tech vs the diversified ETF.

    ``bench`` is a 2-column frame with ``spy`` and ``qqq`` monthly returns; ``returns`` is the
    field panel (pure-plays + the ``etf`` column). Requires ``allow_survivorship=True`` — the
    basket is a surviving-cohort selection, named on the axis. Returns a bundle:

      - ``basket`` / ``basket_net`` — the equal-weight pure-play basket gross / net of ``cost_bps``.
      - ``spy`` / ``qqq`` / ``etf`` — the benchmark series (market / tech / diversified quantum ETF).
      - ``summ`` — ``summarize`` for each leg (CAGR, Sharpe, vol, max-dd).
      - ``test_vs_spy`` / ``test_vs_qqq`` / ``test_vs_etf`` — HAC t-stats of the basket spread.
      - ``single_names`` — per-member HAC t of excess over SPY + risk stats (broad or moon-shot?).
    """
    if not allow_survivorship:
        raise PermissionError(
            "race() uses the surviving-cohort quantum basket (de-SPAC names that made it to a "
            "listing). Pass allow_survivorship=True to opt in; the bias runs bullish and is named "
            "on the Signal axis."
        )
    spy = bench["spy"]
    qqq = bench["qqq"]
    etf_r = returns[etf] if etf in returns.columns else None
    members = [m for m in members if m in returns.columns]
    basket = basket_returns(returns, members, rebalance_every, 0.0)
    basket_net = basket_returns(returns, members, rebalance_every, cost_bps)

    summ = {
        "basket": summarize(basket),
        "basket_net": summarize(basket_net),
        "spy": summarize(spy),
        "qqq": summarize(qqq),
    }
    if etf_r is not None:
        summ["etf"] = summarize(etf_r)

    out = {
        "basket": basket, "basket_net": basket_net, "spy": spy, "qqq": qqq, "etf": etf_r,
        "summ": summ,
        "test_vs_spy": hac_tstat_diff(basket, spy),
        "test_vs_qqq": hac_tstat_diff(basket, qqq),
        "single_names": single_name_tstats(returns, spy, members),
    }
    if etf_r is not None:
        out["test_vs_etf"] = hac_tstat_diff(basket, etf_r)
    return out
