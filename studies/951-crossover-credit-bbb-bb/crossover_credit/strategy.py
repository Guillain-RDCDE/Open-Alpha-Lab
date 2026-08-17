"""Strategy + inference for Study 951 — The Crossover Rung.

The claim: the **BBB/BB boundary** is the best-paid rung on the credit ladder. Investment
grade is squeezed by mandated buyers; deep high yield is where defaults live; the crossover
zone — bonds that have just been pushed out of investment grade — is supposedly where the
forced-seller premium sits without the default cost.

The race is between *static holdings*, not a timing rule, so the honest comparison is a
**risk-adjusted** one:

1. Every rung's daily total return is turned **excess-of-cash** (minus BIL's total return).
2. Each excess series is regressed on two excess factors — ``IEF`` (duration) and ``SPY``
   (equity beta) — and the intercept is the annualised **alpha**, with a Newey-West HAC
   *t*. Without that adjustment the ladder race is rigged: high yield simply carries more
   equity beta and less duration than investment grade, so a raw excess-return comparison
   rewards risk, not skill.
3. The headline statistic is the alpha on the **difference** series (crossover minus a
   benchmark rung), which is the fair head-to-head: same factors, one regression, one *t*.

**Execution lag.** The long-only rungs are buy-and-hold funds: there is no signal to lag,
and the comparison is a holdings race. The optional long/short spread *does* drift away
from dollar-neutral, so it is reset monthly: the weights implied by the closes through the
**last trading day of month t** are acted on at **t+1** (the first trading day of the next
month), when the rebalancing cost is charged. That is the study's single execution lag.

**Costs.** ETF total-return closes are already net of the fund expense ratio, so no fee is
charged twice. The long-only expression is a one-off fund swap inside a sleeve you already
own — turnover is effectively nil, but the switch is still charged on *both* sides (sell the
benchmark rung, buy the crossover rung). The long/short expression pays a one-way cost x NAV
on each month's rebalancing turnover and a **borrow rate on the short leg**, swept 0 / 25 /
50 / 100 bps per year because that rate is an assumption, not tape. Because both legs are
*total-return* series, shorting the benchmark rung automatically pays its distributions
across to the lender — that cost is in the tape, not an assumption.

**The factor betas are fitted in sample.** The IEF/SPY loadings are estimated by OLS over the
whole window and the alpha is the *in-sample* intercept. That is the standard attribution
form and it is not traded — the tradable expressions below hold raw funds, not a fitted
hedge — but it means the reported alpha is a description of the sample, not an out-of-sample
return anyone could have banked.

Returns are **simple** (arithmetic) throughout so weighted portfolio returns are exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Returns & excess-of-cash
# --------------------------------------------------------------------------- #
def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily total returns from a frame of total-return close levels."""
    return prices.sort_index().pct_change()


def excess_returns(prices: pd.DataFrame, cash: str = "BIL") -> pd.DataFrame:
    """Excess-of-cash daily simple returns for every column of ``prices``.

    Rows where any column is missing are dropped first, so every leg of the race sees the
    identical calendar — the common window is set by the *latest* fund inception.
    """
    r = daily_returns(prices).dropna()
    return r.sub(r[cash], axis=0)


def default_hac_lags(n: int) -> int:
    """Newey-West bandwidth, the standard ``4 (n/100)^(2/9)`` rule of thumb."""
    return max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def newey_west_t(x: np.ndarray, lags: int | None = None) -> float:
    """HAC (Newey-West, Bartlett kernel) *t* of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = default_hac_lags(n)
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        var += 2.0 * w * float(u[lag:] @ u[:-lag]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def ols_hac(y, X, lags: int | None = None) -> dict:
    """OLS of ``y`` on ``[1, X]`` with Newey-West HAC standard errors.

    ``X`` may be a 2-D array, a DataFrame, or an empty array (intercept only, in which
    case the intercept *t* reduces to the plain HAC mean *t*). Returns the coefficient
    vector, the HAC *t* vector, the R-squared and the bandwidth actually used.
    """
    y = np.asarray(y, dtype=float).ravel()
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa.reshape(-1, 1)
    if Xa.size == 0:
        Xa = np.empty((len(y), 0))
    Z = np.column_stack([np.ones(len(y)), Xa])
    n, k = Z.shape
    if n <= k + 2:
        return {"beta": np.full(k, np.nan), "tstat": np.full(k, np.nan),
                "r2": float("nan"), "n": n, "lags": 0}
    if lags is None:
        lags = default_hac_lags(n)

    beta = np.linalg.lstsq(Z, y, rcond=None)[0]
    resid = y - Z @ beta
    ZtZ_inv = np.linalg.pinv(Z.T @ Z)
    g = Z * resid[:, None]
    S = g.T @ g
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        A = g[lag:].T @ g[:-lag]
        S += w * (A + A.T)
    V = ZtZ_inv @ S @ ZtZ_inv
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = float(1.0 - (resid @ resid) / ss_tot) if ss_tot > 0 else float("nan")
    return {"beta": beta, "tstat": t, "r2": r2, "n": int(n), "lags": int(lags)}


# --------------------------------------------------------------------------- #
# The two-factor alpha (the fair-race statistic)
# --------------------------------------------------------------------------- #
def factor_alpha(
    y_excess: pd.Series,
    factors: pd.DataFrame,
    lags: int | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> dict:
    """Regress an excess-of-cash series on the excess factor tapes; report annualised alpha.

    ``factors`` is a frame of excess-of-cash factor returns (here: IEF for duration, SPY
    for equity beta). The intercept is the daily alpha; ``alpha_ann`` is it x 252 and
    ``t_alpha`` is its Newey-West *t*. |*t*| >= 2 is the desk's bar for a real premium.
    """
    df = pd.concat([y_excess.rename("y"), factors], axis=1).dropna()
    if len(df) < 30:
        return {"alpha_ann": float("nan"), "t_alpha": float("nan"),
                "betas": {}, "r2": float("nan"), "n": len(df), "lags": 0}
    y = df["y"].to_numpy()
    names = list(factors.columns)
    res = ols_hac(y, df[names].to_numpy(), lags=lags)
    return {
        "alpha_ann": float(res["beta"][0] * periods_per_year),
        "t_alpha": float(res["tstat"][0]),
        "betas": {nm: float(b) for nm, b in zip(names, res["beta"][1:])},
        "t_betas": {nm: float(t) for nm, t in zip(names, res["tstat"][1:])},
        "r2": res["r2"], "n": res["n"], "lags": res["lags"],
    }


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Headline annualised stats for a daily simple-return series (pass the *excess* one)."""
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    mu = r.mean()
    std = r.std(ddof=1)
    sharpe = float(mu / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    return {
        "n_days": int(n),
        "ann_return": float(mu * periods_per_year),
        "sharpe": sharpe,
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "tstat": newey_west_t(r.to_numpy()),
    }


def max_drawdown(returns: pd.Series) -> float:
    """Max drawdown of the *absolute* (lived) wealth path of a simple-return series."""
    r = pd.Series(returns).astype(float).dropna()
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


# --------------------------------------------------------------------------- #
# The ladder race
# --------------------------------------------------------------------------- #
def ladder_table(
    prices: pd.DataFrame,
    rungs=("AGG", "LQD", "ANGL", "HYG"),
    factors=("IEF", "SPY"),
    cash: str = "BIL",
    lags: int | None = None,
) -> pd.DataFrame:
    """Excess-of-cash stats + two-factor alpha for every rung, on one common calendar.

    Also carries the *absolute* (not excess) max drawdown, because that is the loss an
    investor actually lives through. Rows are in ladder order, safest first.
    """
    cols = list(dict.fromkeys(list(rungs) + list(factors) + [cash]))
    e = excess_returns(prices[cols], cash=cash)
    raw = daily_returns(prices[cols]).loc[e.index]
    F = e[list(factors)]
    rows = []
    for rung in rungs:
        s = summary(e[rung])
        a = factor_alpha(e[rung], F, lags=lags)
        rows.append({
            "rung": rung,
            "n_days": s["n_days"],
            "excess_sharpe": s["sharpe"],
            "excess_return_ann": s["ann_return"],
            "vol_ann": s["vol_ann"],
            "max_drawdown_abs": max_drawdown(raw[rung]),
            "alpha_ann": a["alpha_ann"],
            "t_alpha": a["t_alpha"],
            "beta_dur": a["betas"].get(factors[0], float("nan")),
            "beta_eq": a["betas"].get(factors[1], float("nan")),
            "r2": a["r2"],
        })
    return pd.DataFrame(rows).set_index("rung")


def pair_alpha(
    prices: pd.DataFrame,
    long: str,
    short: str,
    factors=("IEF", "SPY"),
    cash: str = "BIL",
    lags: int | None = None,
) -> dict:
    """The head-to-head: two-factor alpha of the ``long - short`` excess-return difference.

    Because both legs are excess-of-cash, the cash leg cancels in the difference and the
    intercept is exactly the duration- and equity-adjusted premium of one rung over the
    other. This is the study's headline statistic.
    """
    cols = list(dict.fromkeys([long, short] + list(factors) + [cash]))
    e = excess_returns(prices[cols], cash=cash)
    d = (e[long] - e[short]).rename("diff")
    res = factor_alpha(d, e[list(factors)], lags=lags)
    res["raw_diff_ann"] = float(d.mean() * TRADING_DAYS)
    res["t_raw_diff"] = newey_west_t(d.to_numpy())
    res["start"] = e.index[0]
    res["end"] = e.index[-1]
    res["diff"] = d
    res["factors"] = e[list(factors)]
    return res


# --------------------------------------------------------------------------- #
# Bootstrap CI on the pair alpha
# --------------------------------------------------------------------------- #
def bootstrap_alpha_ci(
    diff: pd.Series,
    factors: pd.DataFrame,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 951,
    alpha_level: float = 0.05,
    periods_per_year: int = TRADING_DAYS,
) -> dict:
    """Circular block bootstrap CI for the annualised two-factor alpha of ``diff``.

    Blocks of ``block`` consecutive days are resampled *jointly* across the dependent
    series and the factor tape, so the factor exposure is re-estimated on each draw and
    the interval reflects estimation error in the betas as well as in the intercept.
    """
    df = pd.concat([diff.rename("y"), factors], axis=1).dropna()
    y = df["y"].to_numpy(dtype=float)
    Z = np.column_stack([np.ones(len(df)), df[factors.columns].to_numpy(dtype=float)])
    n = len(y)
    if n < block + 5:
        return {"alpha_ann": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "frac_negative": float("nan"), "n_obs": n}
    point = float(np.linalg.lstsq(Z, y, rcond=None)[0][0] * periods_per_year)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.full(n_boot, np.nan)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        try:
            boots[b] = np.linalg.lstsq(Z[idx], y[idx], rcond=None)[0][0] * periods_per_year
        except np.linalg.LinAlgError:
            continue
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha_level / 2, 100 * (1 - alpha_level / 2)])
    return {
        "alpha_ann": point, "ci_low": float(lo), "ci_high": float(hi),
        "frac_negative": float((valid < 0).mean()), "n_obs": int(n),
        "n_boot": int(valid.size), "block": int(block),
    }


# --------------------------------------------------------------------------- #
# Robustness — eras, year jackknife, calendar table
# --------------------------------------------------------------------------- #
def era_cut(
    prices: pd.DataFrame,
    long: str,
    short: str,
    split: str = "2019-01-01",
    factors=("IEF", "SPY"),
    cash: str = "BIL",
) -> dict:
    """Re-run the head-to-head alpha on each side of ``split``.

    A real premium should survive in *both* halves; a lucky one lives in only one.
    """
    out = {}
    for tag, sl in [("early", slice(None, split)), ("late", slice(split, None))]:
        sub = prices.loc[sl]
        if len(sub) < 200:
            out[tag] = None
            continue
        res = pair_alpha(sub, long, short, factors=factors, cash=cash)
        out[tag] = {"n": res["n"], "alpha_ann": res["alpha_ann"], "t_alpha": res["t_alpha"],
                    "raw_diff_ann": res["raw_diff_ann"]}
    return out


def year_jackknife(
    prices: pd.DataFrame,
    long: str,
    short: str,
    factors=("IEF", "SPY"),
    cash: str = "BIL",
) -> pd.DataFrame:
    """Drop one calendar year at a time and re-estimate the head-to-head alpha.

    An edge carried by one or two crisis-recovery years collapses here; a broad one
    barely moves. Reported as annualised alpha and its HAC *t* per dropped year.
    """
    cols = list(dict.fromkeys([long, short] + list(factors) + [cash]))
    e = excess_returns(prices[cols], cash=cash)
    d = e[long] - e[short]
    F = e[list(factors)]
    rows = []
    for y in sorted(set(d.index.year)):
        keep = d.index.year != y
        res = factor_alpha(d[keep], F[keep])
        rows.append({"dropped_year": int(y), "n": res["n"],
                     "alpha_ann": res["alpha_ann"], "t_alpha": res["t_alpha"]})
    return pd.DataFrame(rows).set_index("dropped_year")


def drop_years(
    prices: pd.DataFrame,
    long: str,
    short: str,
    years,
    factors=("IEF", "SPY"),
    cash: str = "BIL",
) -> dict:
    """Head-to-head alpha with a *set* of calendar years excluded (e.g. the crisis years)."""
    cols = list(dict.fromkeys([long, short] + list(factors) + [cash]))
    e = excess_returns(prices[cols], cash=cash)
    keep = ~pd.Index(e.index.year).isin(list(years))
    d = (e[long] - e[short])[keep]
    res = factor_alpha(d, e[list(factors)][keep])
    res["dropped"] = list(years)
    return res


def calendar_year_table(
    prices: pd.DataFrame,
    long: str,
    short: str,
    cash: str = "BIL",
) -> pd.DataFrame:
    """Per-calendar-year compounded excess-return difference (%), long rung minus short rung.

    Unadjusted (no factor regression) — the lived year-by-year experience of owning the
    crossover fund instead of the benchmark rung.
    """
    e = excess_returns(prices[[long, short, cash]], cash=cash)
    d = e[long] - e[short]
    rows = []
    for y, g in d.groupby(d.index.year):
        rows.append({"year": int(y), "n_days": len(g),
                     "diff_pct": float(((1.0 + g).prod() - 1.0) * 100.0)})
    return pd.DataFrame(rows).set_index("year")


def lag_sensitivity(
    diff: pd.Series,
    factors: pd.DataFrame,
    lag_grid=(5, 10, 21, 42),
) -> pd.DataFrame:
    """HAC bandwidth sweep on the head-to-head alpha — is the *t* an artefact of the kernel?"""
    rows = []
    for lg in lag_grid:
        res = factor_alpha(diff, factors, lags=lg)
        rows.append({"hac_lags": int(lg), "alpha_ann": res["alpha_ann"],
                     "t_alpha": res["t_alpha"]})
    return pd.DataFrame(rows).set_index("hac_lags")


# --------------------------------------------------------------------------- #
# The tradable expressions
# --------------------------------------------------------------------------- #
def long_only_swap(
    prices: pd.DataFrame,
    long: str,
    short: str,
    cost_bps: float = 5.0,
    cash: str = "BIL",
) -> dict:
    """Expression A — own the crossover fund *instead of* the benchmark rung.

    No shorting, no timing, no borrow: a one-off fund swap inside a sleeve you already
    hold. The switch is still **two** one-way trades — sell the benchmark rung, buy the
    crossover rung — so ``2 x cost_bps`` x NAV is charged once and amortised over the
    sample, entirely against the *long* (crossover) leg, which is the leg you would not
    have held otherwise. Reports both legs' excess-of-cash Sharpe and the absolute
    drawdown. (At 5 bps one-way over 14 years the whole drag is ~0.007%/yr — the point of
    charging both sides is honesty, not magnitude.)
    """
    e = excess_returns(prices[[long, short, cash]], cash=cash)
    raw = daily_returns(prices[[long, short, cash]]).loc[e.index]
    n = len(e)
    amortised = 2.0 * cost_bps * 1e-4 / n if n else 0.0
    net_long = e[long] - amortised
    s_long, s_short = summary(net_long), summary(e[short])
    return {
        "n_days": n,
        "sharpe_long": s_long["sharpe"], "sharpe_short": s_short["sharpe"],
        "sharpe_gain": s_long["sharpe"] - s_short["sharpe"],
        "excess_ann_long": s_long["ann_return"], "excess_ann_short": s_short["ann_return"],
        "vol_long": s_long["vol_ann"], "vol_short": s_short["vol_ann"],
        "dd_long_abs": max_drawdown(raw[long]), "dd_short_abs": max_drawdown(raw[short]),
        "cost_bps": cost_bps,
    }


def long_short_spread(
    prices: pd.DataFrame,
    long: str,
    short: str,
    borrow_bps: float = 50.0,
    cost_bps: float = 5.0,
    factors=("IEF", "SPY"),
    cash: str = "BIL",
) -> dict:
    """Expression B — the pure spread: long the crossover rung, short the benchmark rung.

    Dollar-neutral, reset at each month boundary. Weights implied by the closes through
    the **last trading day of month t** are acted on at **t+1** — the study's single
    execution lag — and that is where the rebalancing turnover cost is charged (one-way
    ``cost_bps`` x NAV on the absolute weight change of both legs). The short leg pays
    ``borrow_bps`` per year, accrued daily; that rate is an **assumption**, so it is swept.
    """
    cols = list(dict.fromkeys([long, short] + list(factors) + [cash]))
    e = excess_returns(prices[cols], cash=cash)
    ra = e[long].to_numpy()
    rb = e[short].to_numpy()
    idx = e.index
    periods = idx.to_period("M")
    # A new month starts at bar i: the month-end signal of bar i-1 is acted on here.
    new_month = np.r_[False, periods[1:] != periods[:-1]]

    n = len(idx)
    w_long = w_short = 1.0
    out = np.zeros(n)
    turnover = 0.0
    daily_borrow = borrow_bps * 1e-4 / TRADING_DAYS
    for i in range(n):
        if new_month[i]:
            trade = abs(w_long - 1.0) + abs(w_short - 1.0)
            turnover += trade
            out[i] -= trade * cost_bps * 1e-4
            w_long = w_short = 1.0
        out[i] += w_long * ra[i] - w_short * rb[i] - w_short * daily_borrow
        w_long *= (1.0 + ra[i])
        w_short *= (1.0 + rb[i])

    net = pd.Series(out, index=idx, name="spread")
    res = factor_alpha(net, e[list(factors)])
    s = summary(net)
    return {
        "borrow_bps": borrow_bps, "cost_bps": cost_bps,
        "alpha_ann": res["alpha_ann"], "t_alpha": res["t_alpha"],
        "mean_ann": s["ann_return"], "sharpe": s["sharpe"], "vol_ann": s["vol_ann"],
        "max_drawdown": max_drawdown(net),
        "turnover_per_year": float(turnover / (n / TRADING_DAYS)) if n else float("nan"),
        "n_days": n, "net": net,
    }


def borrow_sweep(
    prices: pd.DataFrame,
    long: str,
    short: str,
    borrow_grid=(0.0, 25.0, 50.0, 100.0),
    cost_bps: float = 5.0,
    factors=("IEF", "SPY"),
    cash: str = "BIL",
) -> pd.DataFrame:
    """Sweep the assumed short-leg borrow rate — the one number that is not tape."""
    rows = []
    for b in borrow_grid:
        r = long_short_spread(prices, long, short, borrow_bps=b, cost_bps=cost_bps,
                              factors=factors, cash=cash)
        rows.append({"borrow_bps": b, "alpha_ann": r["alpha_ann"], "t_alpha": r["t_alpha"],
                     "mean_ann": r["mean_ann"], "sharpe": r["sharpe"],
                     "turnover_per_year": r["turnover_per_year"]})
    return pd.DataFrame(rows).set_index("borrow_bps")


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports a real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame) -> dict:
    """Run the head-to-head alpha estimator on a synthetic ladder.

    On the planted world the crossover rung must show an alpha close to the planted value
    with |*t*| >= 2; on the null it must show ~0 with no reliable *t*. Proves the estimator
    is unbiased — it never supports a real-tape stamp.
    """
    res = pair_alpha(prices, "crossover", "hy", factors=("dur", "eq"), cash="cash")
    res_ig = pair_alpha(prices, "crossover", "ig", factors=("dur", "eq"), cash="cash")
    return {
        "alpha_ann": res["alpha_ann"], "t_alpha": res["t_alpha"],
        "alpha_vs_ig": res_ig["alpha_ann"], "t_vs_ig": res_ig["t_alpha"],
        "raw_diff_ann": res["raw_diff_ann"], "n": res["n"],
        "betas": res["betas"],
    }
