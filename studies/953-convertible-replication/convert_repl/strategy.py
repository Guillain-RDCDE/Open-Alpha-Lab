"""Strategy + inference for Study 953 — Replicating the Convert.

The exercise: take a convertible-bond fund's daily **total return in excess of cash**
and fit it, *in-sample only*, to a **static long-only mix** of three liquid ETFs —
SPY (large-cap equity), QQQ (the growth/tech tilt convertible issuers actually have)
and LQD (investment-grade credit) — with the unspent weight sitting in T-bills. The
fit is a constrained least-squares problem: weights are non-negative and sum to at most
one, so the replica is a portfolio anyone can actually hold (long-only means **no borrow
cost arises**; an unconstrained variant is run as a robustness check and, if it goes
short, is charged borrow).

The weights are then **frozen** and held out-of-sample. Two questions follow:

1. **Residual alpha.** Is the mean out-of-sample difference (fund minus frozen replica)
   distinguishable from zero, with a Newey-West (HAC) *t* and a circular block-bootstrap
   CI, after the replica is charged realistic rebalance costs and its own fee drag? The
   fund's return is already net of its expense ratio, so a positive residual is what the
   wrapper genuinely adds; a negative one is what it costs.
2. **Convexity.** The convertible pitch is not "more return", it is a *shape*: catch the
   rally, cushion the fall. So we sort out-of-sample **months** by the equity factor's
   excess return into terciles and ask whether the fund beats its own replica in *both*
   tails (the smile) — plus a Treynor-Mazuy quadratic on the fund-minus-replica
   difference, with an HAC *t* on the curvature term.

Execution lag — exactly one, and it is documented here: the replication weights are
estimated on data **through** the in-sample end date and applied from the **next**
trading day onward; the monthly rebalance trade implied by a month-end drift is
executed at the **close of the following day**. Nothing else in the pipeline looks
forward.

Returns are **simple** (arithmetic) throughout so the portfolio return is exactly
``w . r`` and the wealth path compounds correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Inference primitives (shared house implementations)
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


def newey_west_t(x: np.ndarray, lags: int | None = None) -> float:
    """HAC (Newey-West, Bartlett kernel) *t* of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(int(lags), 0)
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        var += 2.0 * w * float(u[lag:] @ u[:-lag]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def hac_ols(y: np.ndarray, X: np.ndarray, lags: int | None = None) -> dict:
    """OLS with an intercept and Newey-West standard errors. Returns coefs and *t*s."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[keep], X[keep]
    n, k = X.shape
    Z = np.column_stack([np.ones(n), X])
    XtX_inv = np.linalg.pinv(Z.T @ Z)
    beta = XtX_inv @ (Z.T @ y)
    resid = y - Z @ beta
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    S = (Z * resid[:, None]).T @ (Z * resid[:, None])
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        A = (Z[lag:] * resid[lag:, None]).T @ (Z[:-lag] * resid[:-lag, None])
        S += w * (A + A.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstats = np.where(se > 0, beta / se, np.nan)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return {"beta": beta, "se": se, "t": tstats, "n": int(n), "k": int(k), "r2": r2}


def block_bootstrap_mean_ci(
    x: pd.Series | np.ndarray,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 953,
    alpha: float = 0.05,
    scale: float = TRADING_DAYS,
) -> dict:
    """Circular block-bootstrap CI for the *annualised* mean of a daily series.

    Blocks of ``block`` consecutive days preserve autocorrelation and vol clustering.
    ``scale`` annualises the mean (252 for daily, 12 for monthly).
    """
    r = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"mean_ann": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_positive": float("nan"), "n_obs": int(n)}
    point = float(r.mean() * scale)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[b] = r[idx].mean() * scale
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean_ann": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_positive": float((boots > 0).mean()), "n_obs": int(n),
            "n_boot": int(n_boot), "block": int(block)}


# --------------------------------------------------------------------------- #
# Constrained least squares (the replication fit)
# --------------------------------------------------------------------------- #
def project_simplex(v: np.ndarray, z: float = 1.0) -> np.ndarray:
    """Euclidean projection of ``v`` onto ``{w >= 0, sum(w) = z}``."""
    v = np.asarray(v, dtype=float)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - z
    idx = np.arange(1, v.size + 1)
    cond = u - css / idx > 0
    rho = idx[cond][-1]
    theta = css[rho - 1] / rho
    return np.maximum(v - theta, 0.0)


def project_capped(v: np.ndarray, max_sum: float = 1.0) -> np.ndarray:
    """Euclidean projection onto ``{w >= 0, sum(w) <= max_sum}``.

    Clipping at zero already gives the projection onto the non-negative orthant; if the
    clipped vector satisfies the budget it *is* the projection onto the intersection,
    otherwise the projection lands on the ``sum = max_sum`` face.
    """
    w = np.maximum(np.asarray(v, dtype=float), 0.0)
    if w.sum() <= max_sum:
        return w
    return project_simplex(np.asarray(v, dtype=float), z=max_sum)


def constrained_ls(
    y: np.ndarray,
    X: np.ndarray,
    nonneg: bool = True,
    max_sum: float = 1.0,
    iters: int = 1500,
) -> tuple[float, np.ndarray]:
    """Least squares of ``y`` on ``X`` with an intercept and a budget-constrained slope.

    Minimises ``||y - a - X w||^2`` over a free intercept ``a`` and weights ``w`` in
    ``{w >= 0, sum(w) <= max_sum}`` (when ``nonneg``), by concentrating the intercept
    out (demeaning) and running deterministic accelerated (FISTA) projected gradient
    descent with a Lipschitz step. With ``nonneg=False`` this is plain OLS. Returns
    ``(a, w)``, where ``a`` is the per-period intercept — the *in-sample* alpha of the
    fit. Deterministic: no randomness, fixed iteration count, fixed starting point.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[keep], X[keep]
    ybar, Xbar = y.mean(), X.mean(axis=0)
    yc, Xc = y - ybar, X - Xbar

    if not nonneg:
        w = np.linalg.pinv(Xc.T @ Xc) @ (Xc.T @ yc)
        return float(ybar - Xbar @ w), w

    G = Xc.T @ Xc
    c = Xc.T @ yc
    lip = float(np.linalg.eigvalsh(G).max())
    step = 1.0 / lip if lip > 0 else 1.0
    w = np.full(X.shape[1], max_sum / X.shape[1])
    z = w.copy()
    t_k = 1.0
    for _ in range(iters):
        w_new = project_capped(z - step * (G @ z - c), max_sum=max_sum)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k))
        z = w_new + ((t_k - 1.0) / t_next) * (w_new - w)
        w, t_k = w_new, t_next
    return float(ybar - Xbar @ w), w


# --------------------------------------------------------------------------- #
# The replica portfolio path (monthly rebalance, one-way cost, fee drag)
# --------------------------------------------------------------------------- #
def replica_excess(
    factor_rets: pd.DataFrame,
    cash_ret: pd.Series,
    weights: np.ndarray,
    cost_bps: float = 3.0,
    fee_ann: float = 0.0010,
    borrow_bps_ann: float = 50.0,
) -> pd.Series:
    """Path of a static-weight replica, rebalanced monthly, in **excess-of-cash** terms.

    Weights drift with the market inside each month and are traded back to target on the
    **first trading day of the next month** — i.e. the month-end drift observed at the
    close of month-end ``t`` is carried through the return of ``t+1`` and only corrected
    at the close of ``t+1`` (the single execution lag; a month-end-close rebalance would
    be the zero-lag variant and is deliberately *not* what this does — it moves the
    out-of-sample alpha by 5 bps/yr and the *t* by 0.02, either way).
    Costs are ``cost_bps`` one-way x NAV on the traded turnover; ``fee_ann`` is a
    straight-line management-fee drag on the whole sleeve (the replica ETFs are gross of
    their own fees in the tape); any *negative* weight pays ``borrow_bps_ann`` on its
    absolute size (only reachable in the unconstrained robustness variant).
    """
    idx = factor_rets.index
    R = factor_rets.to_numpy(dtype=float)
    rc = cash_ret.reindex(idx).to_numpy(dtype=float)
    tgt = np.asarray(weights, dtype=float)
    tgt_full = np.append(tgt, 1.0 - tgt.sum())          # last slot = cash
    cost = cost_bps * 1e-4
    fee_d = fee_ann / TRADING_DAYS
    borrow_d = borrow_bps_ann * 1e-4 / TRADING_DAYS

    months = idx.to_period("M").to_numpy()
    cur = tgt_full.copy()
    out = np.empty(len(idx))
    for i in range(len(idx)):
        r_all = np.append(R[i], rc[i])
        port = float(cur @ r_all)
        short_leg = float(np.abs(np.minimum(cur[:-1], 0.0)).sum())
        port -= fee_d + borrow_d * short_leg
        # drift the weights with the realised returns
        grown = cur * (1.0 + r_all)
        denom = grown.sum()
        cur = grown / denom if denom != 0 else tgt_full.copy()
        # The single execution lag: the month-end drift is *not* corrected at the
        # month-end close. Day ``i`` is the first trading day of a new month, so its
        # return above was earned on the drifted weights, and only now — at that day's
        # close — is the portfolio traded back to target.
        if i > 0 and months[i] != months[i - 1]:
            port -= cost * float(np.abs(tgt_full - cur).sum())
            cur = tgt_full.copy()
        out[i] = port - rc[i]
    return pd.Series(out, index=idx, name="replica_excess")


# --------------------------------------------------------------------------- #
# Fit + hold-out
# --------------------------------------------------------------------------- #
def excess_returns(prices: pd.DataFrame, cash_col: str) -> pd.DataFrame:
    """Daily simple returns of every column, minus the cash column (excess-of-cash)."""
    rets = prices.pct_change()
    ex = rets.sub(rets[cash_col], axis=0)
    ex[cash_col] = rets[cash_col]
    return ex.dropna()


def fit_replication(
    prices: pd.DataFrame,
    fund_col: str,
    factor_cols,
    cash_col: str,
    is_end: str,
    nonneg: bool = True,
    max_sum: float = 1.0,
) -> dict:
    """Fit the static replication weights on the IN-SAMPLE window only.

    Returns the fitted weights (one per factor, plus the implied cash weight), the
    in-sample intercept (annualised), the in-sample R^2 and tracking error, and the
    in-sample window's day count. The weights are the only thing carried out-of-sample.
    """
    rets = prices.pct_change().dropna()
    ex = rets.sub(rets[cash_col], axis=0)
    is_mask = ex.index <= pd.Timestamp(is_end)
    y = ex.loc[is_mask, fund_col].to_numpy()
    X = ex.loc[is_mask, list(factor_cols)].to_numpy()
    a, w = constrained_ls(y, X, nonneg=nonneg, max_sum=max_sum)
    resid = y - a - X @ w
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "weights": {c: float(v) for c, v in zip(factor_cols, w)},
        "w_vec": w,
        "w_cash": float(1.0 - w.sum()),
        "alpha_is_ann": float(a * TRADING_DAYS),
        "r2_is": 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan"),
        "te_is_ann": float(resid.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "n_is": int(is_mask.sum()),
        "is_end": str(pd.Timestamp(is_end).date()),
        "nonneg": bool(nonneg),
    }


def holdout(
    prices: pd.DataFrame,
    fund_col: str,
    factor_cols,
    cash_col: str,
    is_end: str,
    cost_bps: float = 3.0,
    fee_ann: float = 0.0010,
    nonneg: bool = True,
    max_sum: float = 1.0,
    borrow_bps_ann: float = 50.0,
) -> dict:
    """Fit in-sample, freeze the weights, and score the OUT-OF-SAMPLE residual alpha.

    The replica is charged ``cost_bps`` one-way on its monthly rebalance turnover and a
    ``fee_ann`` fee drag; the fund side needs no fee adjustment because its total return
    is already net of its expense ratio. Every series is excess-of-cash. The reported
    alpha is ``mean(fund_excess - replica_excess)`` annualised, with a Newey-West *t*.
    """
    fit = fit_replication(prices, fund_col, factor_cols, cash_col, is_end,
                          nonneg=nonneg, max_sum=max_sum)
    rets = prices.pct_change().dropna()
    ex_fund = rets[fund_col] - rets[cash_col]
    repl = replica_excess(rets[list(factor_cols)], rets[cash_col], fit["w_vec"],
                          cost_bps=cost_bps, fee_ann=fee_ann,
                          borrow_bps_ann=borrow_bps_ann)
    diff = (ex_fund - repl).dropna()

    oos_mask = diff.index > pd.Timestamp(is_end)
    out = {"fit": fit, "ex_fund": ex_fund, "replica": repl, "diff": diff,
           "oos_mask": oos_mask}
    for tag, mask in (("is", ~oos_mask), ("oos", oos_mask)):
        d = diff[mask]
        f = ex_fund.reindex(d.index)
        r = repl.reindex(d.index)
        out[tag] = {
            "n_days": int(len(d)),
            "alpha_ann": float(d.mean() * TRADING_DAYS),
            "t_hac": newey_west_t(d.to_numpy()),
            "te_ann": float(d.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            "corr": float(np.corrcoef(f, r)[0, 1]) if len(d) > 2 else float("nan"),
            "sharpe_fund": _sharpe(f),
            "sharpe_replica": _sharpe(r),
            "vol_fund": float(f.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            "vol_replica": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            "dd_fund": max_drawdown(f),
            "dd_replica": max_drawdown(r),
            "start": str(d.index[0].date()) if len(d) else "",
            "end": str(d.index[-1].date()) if len(d) else "",
        }
    out["oos"]["sharpe_gap"] = out["oos"]["sharpe_fund"] - out["oos"]["sharpe_replica"]
    return out


def _sharpe(x: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    r = pd.Series(x).dropna().astype(float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")


def max_drawdown(returns: pd.Series) -> float:
    r = pd.Series(returns).astype(float).dropna()
    if r.empty:
        return float("nan")
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Headline annualised stats for a daily simple-return series (pass an excess series)."""
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    sd = r.std(ddof=1)
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    return {
        "n_days": int(n),
        "sharpe": float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan"),
        "vol_ann": float(sd * np.sqrt(periods_per_year)),
        "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and wealth.iloc[-1] > 0 else float("nan"),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "mean_daily_bps": float(r.mean() * 1e4),
        "tstat": newey_west_t(r.to_numpy()),
    }


# --------------------------------------------------------------------------- #
# The convexity test (monthly, fund vs its own replica)
# --------------------------------------------------------------------------- #
def to_monthly(x: pd.Series) -> pd.Series:
    """Compound a daily simple-return series into complete calendar-month returns."""
    s = pd.Series(x).dropna()
    m = (1.0 + s).groupby(s.index.to_period("M")).prod() - 1.0
    m.index.name = "month"
    return m


def convexity_test(
    ex_fund: pd.Series,
    replica: pd.Series,
    ex_market: pd.Series,
    mask: pd.Series | None = None,
) -> dict:
    """Does the fund beat its replica in the tails? (the convexity claim, monthly)

    Months are sorted by the **equity factor's** excess return into terciles. A convex
    wrapper should beat its linear replica in the *top* tercile (converts ride the rally)
    **and** in the *bottom* one (the bond floor cushions) — a smile. We report the mean
    fund-minus-replica difference in each tercile, the smile (tails minus middle) with a
    Welch *t*, up-/down-capture of the fund against its replica, and a Treynor-Mazuy
    quadratic of the difference on the market with an HAC *t* on the curvature.

    **This is a shape *description*, not a tradable rule, and it is labelled as such:** the
    tercile breakpoints and the up/down split are computed on the *whole* window being
    described, so a month's bucket is known only in arrears. Nothing downstream trades on
    it — it exists to answer "what shape did the payoff have?", never "what should I buy?".
    """
    if mask is not None:
        ex_fund, replica, ex_market = ex_fund[mask], replica[mask], ex_market[mask]
    f = to_monthly(ex_fund)
    r = to_monthly(replica)
    m = to_monthly(ex_market)
    common = f.index.intersection(r.index).intersection(m.index)
    f, r, m = f.loc[common], r.loc[common], m.loc[common]
    d = (f - r).to_numpy()
    mv = m.to_numpy()

    q1, q2 = np.percentile(mv, [100 / 3.0, 200 / 3.0])
    low, mid, high = mv <= q1, (mv > q1) & (mv <= q2), mv > q2
    tails = low | high

    up, dn = mv > 0, mv < 0
    cap_up = float(f[up].sum() / r[up].sum()) if r[up].sum() != 0 else float("nan")
    cap_dn = float(f[dn].sum() / r[dn].sum()) if r[dn].sum() != 0 else float("nan")

    tm = hac_ols(d, np.column_stack([mv, mv ** 2]))
    return {
        "n_months": int(len(common)),
        "mean_low_bps": float(d[low].mean() * 1e4),
        "mean_mid_bps": float(d[mid].mean() * 1e4),
        "mean_high_bps": float(d[high].mean() * 1e4),
        "t_low": one_sample_t(d[low]),
        "t_high": one_sample_t(d[high]),
        "smile_bps": float(d[tails].mean() * 1e4 - d[mid].mean() * 1e4),
        "t_smile": welch_t(d[tails], d[mid]),
        "up_capture": cap_up,
        "down_capture": cap_dn,
        "capture_asymmetry": cap_up - cap_dn,
        "tm_gamma": float(tm["beta"][2]),
        "t_tm_gamma": float(tm["t"][2]),
        "tm_beta": float(tm["beta"][1]),
        "tm_alpha_ann": float(tm["beta"][0] * 12),
        "t_tm_alpha": float(tm["t"][0]),
        "diff_monthly": pd.Series(d, index=common, name="fund_minus_replica"),
        "mkt_monthly": m,
    }


# --------------------------------------------------------------------------- #
# Robustness — era cut, cost sweep, fee sweep
# --------------------------------------------------------------------------- #
def era_cut(res: dict, split: str = "2022-01-01") -> dict:
    """Split the out-of-sample window at ``split`` and re-score the residual alpha."""
    diff = res["diff"][res["oos_mask"]]
    out = {}
    for tag, m in (("early", diff.index < pd.Timestamp(split)),
                   ("late", diff.index >= pd.Timestamp(split))):
        d = diff[m]
        out[tag] = None if len(d) < 60 else {
            "n_days": int(len(d)),
            "alpha_ann": float(d.mean() * TRADING_DAYS),
            "t_hac": newey_west_t(d.to_numpy()),
            "start": str(d.index[0].date()),
            "end": str(d.index[-1].date()),
        }
    return out


def cost_sweep(
    prices: pd.DataFrame,
    fund_col: str,
    factor_cols,
    cash_col: str,
    is_end: str,
    cost_grid=(0.0, 1.0, 3.0, 10.0),
    fee_grid=(0.0, 0.0010, 0.0020),
) -> list[dict]:
    """Out-of-sample residual alpha across one-way rebalance costs and replica fee drags.

    The fee drag is a PROXY (published expense ratios for a SPY/QQQ/LQD sleeve are
    ~3-20 bps/yr) so it is swept rather than asserted; the rebalance cost is charged
    one-way x NAV on the traded turnover only.
    """
    rows = []
    for c in cost_grid:
        for fee in fee_grid:
            res = holdout(prices, fund_col, factor_cols, cash_col, is_end,
                          cost_bps=c, fee_ann=fee)
            rows.append({"cost_bps": c, "fee_ann_bps": fee * 1e4,
                         "alpha_ann": res["oos"]["alpha_ann"],
                         "t_hac": res["oos"]["t_hac"],
                         "te_ann": res["oos"]["te_ann"]})
    return rows


def factor_set_sweep(
    prices: pd.DataFrame,
    fund_col: str,
    factor_sets,
    cash_col: str,
    is_end: str,
    cost_bps: float = 3.0,
    fee_ann: float = 0.0010,
) -> list[dict]:
    """Out-of-sample residual alpha across alternative replication kits.

    **Why this exists.** The three-leg kit (SPY + QQQ + LQD) is a *judgement*, and a
    hindsight-flavoured one: we know, today, that convertible issuers skew growth/tech, and
    a richer kit mechanically shrinks the very residual the study is trying to measure. The
    honest thing is not to hide the choice but to show what it bought. Sweeping the kit all
    the way down to a single equity leg reports how much of the blank is the kit's doing.
    """
    rows = []
    for fs in factor_sets:
        fs = tuple(fs)
        res = holdout(prices, fund_col, fs, cash_col, is_end,
                      cost_bps=cost_bps, fee_ann=fee_ann)
        rows.append({
            "factors": "+".join(fs),
            "n_factors": len(fs),
            "r2_is": res["fit"]["r2_is"],
            "weights": res["fit"]["weights"],
            "w_cash": res["fit"]["w_cash"],
            "alpha_ann": res["oos"]["alpha_ann"],
            "t_hac": res["oos"]["t_hac"],
            "n_oos": res["oos"]["n_days"],
        })
    return rows


def split_sweep(
    prices: pd.DataFrame,
    fund_col: str,
    factor_cols,
    cash_col: str,
    is_ends,
    cost_bps: float = 3.0,
    fee_ann: float = 0.0010,
) -> list[dict]:
    """Out-of-sample residual alpha across alternative in-sample / hold-out boundaries.

    The 2017-12-31 split is likewise a choice (roughly half of CWB's life). Moving it shows
    whether the headline *t* is a property of the fund or of the boundary.
    """
    rows = []
    for cut in is_ends:
        res = holdout(prices, fund_col, tuple(factor_cols), cash_col, cut,
                      cost_bps=cost_bps, fee_ann=fee_ann)
        rows.append({"is_end": str(cut), "n_oos": res["oos"]["n_days"],
                     "alpha_ann": res["oos"]["alpha_ann"], "t_hac": res["oos"]["t_hac"]})
    return rows


def rolling_refit(
    prices: pd.DataFrame,
    fund_col: str,
    factor_cols,
    cash_col: str,
    first_is_end: str,
    step_years: int = 1,
    cost_bps: float = 3.0,
    fee_ann: float = 0.0010,
) -> dict:
    """Expanding-window variant: refit the weights each year, always applied forward.

    Guards against the single fixed in-sample split being lucky or unlucky. The
    concatenated out-of-sample difference is scored with the same HAC *t*.
    """
    rets = prices.pct_change().dropna()
    ex_fund = rets[fund_col] - rets[cash_col]
    start_year = pd.Timestamp(first_is_end).year
    end_year = int(rets.index[-1].year)
    pieces = []
    weights_path = []
    for y in range(start_year, end_year, step_years):
        cut = f"{y}-12-31"
        fit = fit_replication(prices, fund_col, factor_cols, cash_col, cut)
        repl = replica_excess(rets[list(factor_cols)], rets[cash_col], fit["w_vec"],
                              cost_bps=cost_bps, fee_ann=fee_ann)
        seg = (ex_fund - repl)
        seg = seg[(seg.index > pd.Timestamp(cut)) &
                  (seg.index <= pd.Timestamp(f"{y + step_years}-12-31"))]
        if len(seg):
            pieces.append(seg)
            weights_path.append({"through": cut, **fit["weights"],
                                 "cash": fit["w_cash"]})
    if not pieces:
        return {"n_days": 0}
    d = pd.concat(pieces).sort_index()
    return {"n_days": int(len(d)), "alpha_ann": float(d.mean() * TRADING_DAYS),
            "t_hac": newey_west_t(d.to_numpy()),
            "te_ann": float(d.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            "weights_path": weights_path,
            "start": str(d.index[0].date()), "end": str(d.index[-1].date())}


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports a real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, is_end: str | None = None,
                     cost_bps: float = 0.0, fee_ann: float = 0.0) -> dict:
    """Run the whole pipeline on a synthetic panel and report what it recovers.

    On the planted world (``signal_strength=1``) the harness must recover a positive
    out-of-sample alpha and a positive Treynor-Mazuy curvature; on the null
    (``signal_strength=0``) it must find neither. Never evidence about markets.
    """
    if is_end is None:
        is_end = str(prices.index[len(prices) // 2].date())
    res = holdout(prices, "fund", ("eq", "growth", "credit"), "cash", is_end,
                  cost_bps=cost_bps, fee_ann=fee_ann)
    rets = prices.pct_change().dropna()
    cx = convexity_test(res["ex_fund"], res["replica"],
                        rets["eq"] - rets["cash"], mask=res["oos_mask"])
    return {
        "weights": res["fit"]["weights"],
        "w_cash": res["fit"]["w_cash"],
        "r2_is": res["fit"]["r2_is"],
        "alpha_is_ann": res["fit"]["alpha_is_ann"],
        "alpha_oos_ann": res["oos"]["alpha_ann"],
        "t_oos": res["oos"]["t_hac"],
        "te_oos_ann": res["oos"]["te_ann"],
        "tm_gamma": cx["tm_gamma"],
        "t_tm_gamma": cx["t_tm_gamma"],
        "smile_bps": cx["smile_bps"],
        "n_oos": res["oos"]["n_days"],
    }
