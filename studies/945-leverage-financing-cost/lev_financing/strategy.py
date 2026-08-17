"""Estimation + inference for Study 945 — The Hidden Financing.

A 2x fund borrows one dollar for every dollar of your NAV; a 3x fund borrows two. The
prospectus never quotes the rate. The accounting, however, is rigid:

    r_fund(t)  =  L * r_index(t)  -  (L - 1) * f / 252  -  ER / 252  +  tracking noise

so a plain OLS of the fund's daily total return on the benchmark's recovers *both*
unknowns: the slope is the realised leverage ``L`` and the intercept is the entire daily
drag. Annualise the intercept, strip the published expense ratio, divide by the ``L - 1``
dollars actually borrowed, and what is left is the **implied financing rate**. Set it
next to ``^IRX`` (the 13-week bill) and you have the spread the wrapper charges you over
T-bills.

**What the residual actually contains (read this before calling it an interest rate).**
The intercept is the fund's *whole* daily shortfall against ``L`` times the benchmark.
Subtracting the published expense ratio leaves financing **plus every other daily friction
the wrapper runs**: swap spreads and counterparty fees, the slippage of the daily reset
itself, and any residual tracking loss. None of that is separable from interest with a
return regression alone, so ``implied_financing_pct`` is an **upper bound on the pure
borrowing rate**, not the rate itself. The study reports it as such; the number that needs
no such caveat is the all-in ``cost_per_borrowed_dollar_pct``, which is what a holder pays
whatever its internal name.

Two accounting details that decide whether the answer is right or nonsense:

1. **Benchmark basis.** The fund tracks the *index*; the tradable benchmark SPY runs
   ``SPY_EXPENSE_RATIO`` below it. Measuring the drag on an index basis therefore adds
   ``beta * SPY_ER`` back. Using the S&P 500 **price** index instead of a total-return
   benchmark mis-states the drag by ``L`` times the dividend yield — a ~3.5 pp/yr error
   that would make the wrappers look extortionate. Both are computed and labelled.
2. **Total return, both sides.** The funds distribute part of their collateral income;
   ``auto_adjust=True`` closes put that back, so the measured drag is the fund's *net*
   cost and not an artefact of what it paid out.

The second half of the module answers the practical question: is the wrapper a cheap
loan? It races the fund against a **self-financed replication** — hold ``L`` times SPY in
a margin account at ``benchmark rate + margin_spread``, reset daily like the fund does —
excess-of-cash on both sides, with the one-way trading cost of that daily reset charged
against NAV, and reports the **break-even margin spread** at which do-it-yourself ties
the wrapper.

**One execution lag, and only one.** The replication's exposure for day ``t+1`` is set
from the close of day ``t`` and earns day ``t+1``'s return; the borrow rate charged on
day ``t+1`` is the rate quoted at the close of day ``t``. The estimation half is a
*measurement*, not a trade: fund and benchmark returns are contemporaneous by
construction, which is the whole point of a same-day regression.

**No short leg.** Nothing here is sold short, so no borrow fee applies; the only
borrowing in the study is the *long* leverage itself, which is the object being priced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
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
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        var += 2.0 * w * float(u[lag:] @ u[:-lag]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def auto_lags(n: int) -> int:
    """Newey-West bandwidth rule of thumb, 4 * (n/100)^(2/9)."""
    return int(np.floor(4.0 * (max(n, 1) / 100.0) ** (2.0 / 9.0)))


def ols_hac(y: np.ndarray, x: np.ndarray, lags: int | None = None) -> dict:
    """OLS of ``y`` on a constant and ``x`` with Newey-West standard errors.

    Returns ``alpha``, ``beta`` and their HAC standard errors and *t* statistics, the
    residual standard deviation and the R-squared. The HAC sandwich uses the Bartlett
    kernel with the ``auto_lags`` bandwidth unless ``lags`` is given.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = len(y)
    if n < 10:
        return {k: float("nan") for k in
                ("alpha", "beta", "alpha_se", "beta_se", "alpha_t", "beta_t",
                 "resid_sd", "r2", "n")}
    X = np.column_stack([np.ones(n), x])
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ (X.T @ y)
    e = y - X @ b
    if lags is None:
        lags = auto_lags(n)
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        A = Xe[lag:].T @ Xe[:-lag]
        S = S + w * (A + A.T)
    V = XtXi @ S @ XtXi
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "alpha": float(b[0]), "beta": float(b[1]),
        "alpha_se": float(se[0]), "beta_se": float(se[1]),
        "alpha_t": float(b[0] / se[0]) if se[0] > 0 else float("nan"),
        "beta_t": float(b[1] / se[1]) if se[1] > 0 else float("nan"),
        "resid_sd": float(e.std(ddof=2)), "n": int(n),
        "r2": float(1.0 - (e @ e) / ss_tot) if ss_tot > 0 else float("nan"),
    }


def block_bootstrap_ci(
    x: pd.Series | np.ndarray,
    stat=np.mean,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 945,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap percentile CI for a statistic of a daily series.

    Blocks of ``block`` consecutive days preserve the serial dependence of tracking
    error and of the rate path. Returns the point estimate, the (1-alpha) interval and
    the share of resamples below zero.
    """
    r = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_below_zero": float("nan"), "n_obs": int(n)}
    point = float(stat(r))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[b] = stat(r[idx])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_below_zero": float((boots < 0).mean()), "n_obs": int(n),
            "n_boot": int(n_boot), "block": int(block)}


# --------------------------------------------------------------------------- #
# The estimator — back the financing rate out of the wrapper
# --------------------------------------------------------------------------- #
def implied_financing(
    fund_ret: pd.Series,
    bench_ret: pd.Series,
    leverage: int,
    expense_ratio: float,
    bench_fee: float = 0.0,
    rate_pct: pd.Series | None = None,
    lags: int | None = None,
) -> dict:
    """Regress a levered fund's daily return on the benchmark's and solve for financing.

    Parameters
    ----------
    fund_ret, bench_ret:
        Aligned daily **simple total returns** of the levered fund and of the benchmark.
    leverage:
        The wrapper's stated daily leverage ``L`` (2 for SSO, 3 for UPRO).
    expense_ratio:
        The fund's published expense ratio in **percent per annum** (a PROXY read off the
        prospectus — swept in :func:`expense_ratio_sweep`).
    bench_fee:
        The benchmark ETF's own expense ratio in percent per annum. The drag is reported
        on an *index* basis, so ``beta * bench_fee`` is added back. Pass 0.0 when the
        benchmark is the index itself.
    rate_pct:
        Optional daily series of the benchmark short rate in percent per annum (``^IRX``).
        When given, the mean over the window is used to report the financing **spread**.

    Returns the fitted regression, the annualised drag on an index basis, the implied
    financing rate, the all-in cost per borrowed dollar (drag divided by ``L - 1``, i.e.
    *including* the expense ratio), and the spread over the benchmark rate. All rates are
    in percent per annum; annualisation is the simple ``252 x`` convention throughout, so
    every quoted rate is directly comparable with a quoted money-market rate.

    **Which t statistic answers which question.** ``alpha_t`` tests only that the drag is
    non-zero — a claim the expense ratio alone guarantees, so it is *not* evidence for a
    mark-up over bills and must never be quoted as such. The study's actual claims are the
    two spreads, and each gets its own HAC t built from the same intercept standard error
    rescaled by the ``L - 1`` borrowed dollars: ``spread_t`` (financing above the benchmark
    rate, which additionally leans on the assumed expense ratio) and ``all_in_spread_t``
    (the all-in charge above the benchmark rate, which does not).
    """
    common = fund_ret.index.intersection(bench_ret.index)
    y = fund_ret.reindex(common).astype(float)
    x = bench_ret.reindex(common).astype(float)
    fit = ols_hac(y.to_numpy(), x.to_numpy(), lags=lags)

    alpha_ann = fit["alpha"] * TRADING_DAYS * 100.0
    alpha_se_ann = fit["alpha_se"] * TRADING_DAYS * 100.0
    drag = -alpha_ann + fit["beta"] * bench_fee          # index basis, % per annum
    drag_se = alpha_se_ann
    f_implied = (drag - expense_ratio) / (leverage - 1)
    cost_per_borrowed = drag / (leverage - 1)

    mean_rate = float("nan")
    if rate_pct is not None:
        mean_rate = float(pd.Series(rate_pct).reindex(common).dropna().mean())

    # The spreads inherit the intercept's sampling error, scaled by the borrowed dollars.
    # The benchmark rate is a realised sample mean over the same window and is held fixed:
    # the uncertainty priced here is the wrapper's, not the Fed's.
    spread_se = drag_se / (leverage - 1)
    spread = f_implied - mean_rate
    all_in_spread = cost_per_borrowed - mean_rate

    return {
        "leverage": int(leverage),
        "beta": fit["beta"], "beta_se": fit["beta_se"],
        "beta_t_vs_L": (fit["beta"] - leverage) / fit["beta_se"] if fit["beta_se"] > 0 else float("nan"),
        "alpha_ann_pct": alpha_ann, "alpha_se_ann_pct": alpha_se_ann,
        "alpha_t": fit["alpha_t"],
        "r2": fit["r2"], "resid_sd_bps": fit["resid_sd"] * 1e4, "n_days": fit["n"],
        "expense_ratio_pct": expense_ratio,
        "drag_ann_pct": drag, "drag_se_ann_pct": drag_se,
        "implied_financing_pct": f_implied,
        "cost_per_borrowed_dollar_pct": cost_per_borrowed,
        "mean_rate_pct": mean_rate,
        "spread_over_rate_pct": spread,
        "all_in_spread_over_rate_pct": all_in_spread,
        "spread_se_ann_pct": spread_se,
        "spread_t": spread / spread_se if spread_se > 0 else float("nan"),
        "all_in_spread_t": all_in_spread / spread_se if spread_se > 0 else float("nan"),
        "start": str(common[0].date()) if len(common) else None,
        "end": str(common[-1].date()) if len(common) else None,
    }


def constrained_drag_series(
    fund_ret: pd.Series,
    bench_ret: pd.Series,
    leverage: int,
    bench_fee: float = 0.0,
) -> pd.Series:
    """Daily drag with the slope pinned at ``L``: ``L * r_bench - r_fund`` (index basis).

    The free-beta regression and this pinned version answer the same question two ways.
    They agree whenever the realised beta is statistically indistinguishable from ``L``
    (it is, on this tape), and the pinned series is what the block bootstrap resamples —
    a mean of it, annualised, *is* the drag.
    """
    common = fund_ret.index.intersection(bench_ret.index)
    d = leverage * bench_ret.reindex(common) - fund_ret.reindex(common)
    d = d + leverage * bench_fee / 100.0 / TRADING_DAYS
    return d.dropna().rename("drag")


def bootstrap_spread(
    fund_ret: pd.Series,
    bench_ret: pd.Series,
    leverage: int,
    expense_ratio: float,
    mean_rate_pct: float,
    bench_fee: float = 0.0,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 945,
) -> dict:
    """Block-bootstrap CI for the implied financing **spread** over the benchmark rate.

    Resamples the pinned daily drag series in 21-day blocks and pushes each resample
    through the same arithmetic (annualise, strip the expense ratio, divide by the
    borrowed dollars, subtract the mean benchmark rate). The benchmark rate is held at
    its sample mean — the uncertainty priced here is the wrapper's, not the Fed's.
    """
    d = constrained_drag_series(fund_ret, bench_ret, leverage, bench_fee=bench_fee)

    def stat(arr: np.ndarray) -> float:
        drag = float(arr.mean()) * TRADING_DAYS * 100.0
        return (drag - expense_ratio) / (leverage - 1) - mean_rate_pct

    return block_bootstrap_ci(d, stat=stat, n_boot=n_boot, block=block, seed=seed)


def rolling_financing(
    fund_ret: pd.Series,
    bench_ret: pd.Series,
    rate_pct: pd.Series,
    leverage: int,
    expense_ratio: float,
    bench_fee: float = 0.0,
    window: int = 252,
) -> pd.DataFrame:
    """Rolling one-year implied financing rate, benchmark rate and spread.

    Each window runs the same free-beta OLS over the trailing ``window`` sessions and is
    stamped on that window's **last** day, so nothing at date ``t`` uses data after ``t``.
    The companion benchmark rate is the trailing mean of ``rate_pct`` over the identical
    window, which keeps the spread an apples-to-apples difference.
    """
    common = fund_ret.index.intersection(bench_ret.index).intersection(rate_pct.index)
    y = fund_ret.reindex(common).to_numpy(dtype=float)
    x = bench_ret.reindex(common).to_numpy(dtype=float)
    n = len(y)
    if n < window + 1:
        return pd.DataFrame(columns=["beta", "implied_financing", "rate", "spread"])
    ones = np.ones(window)
    rows = []
    for i in range(window, n + 1):
        yy, xx = y[i - window:i], x[i - window:i]
        X = np.column_stack([ones, xx])
        b = np.linalg.lstsq(X, yy, rcond=None)[0]
        drag = -b[0] * TRADING_DAYS * 100.0 + b[1] * bench_fee
        rows.append((b[1], (drag - expense_ratio) / (leverage - 1)))
    idx = common[window - 1:]
    out = pd.DataFrame(rows, index=idx, columns=["beta", "implied_financing"])
    out["rate"] = pd.Series(rate_pct).reindex(common).rolling(window).mean().reindex(idx)
    out["spread"] = out["implied_financing"] - out["rate"]
    return out.dropna()


def pass_through_regression(rolling: pd.DataFrame, lags: int | None = None) -> dict:
    """Regress the rolling implied financing rate on the rolling benchmark rate.

    A wrapper that simply passes the policy rate through at a constant mark-up gives a
    slope of 1 and an intercept equal to that mark-up. A slope well below 1 would mean
    the wrapper keeps part of a rate rise; well above 1, that it over-charges as rates
    climb. HAC standard errors, with the honest caveat that overlapping 252-day windows
    make these residuals hugely autocorrelated — the slope is descriptive, and the
    inference bar is carried by the full-sample regression, not by this one.
    """
    if len(rolling) < 50:
        return {k: float("nan") for k in ("slope", "slope_se", "intercept", "corr", "n")}
    fit = ols_hac(rolling["implied_financing"].to_numpy(),
                  rolling["rate"].to_numpy(), lags=lags or 252)
    return {
        "slope": fit["beta"], "slope_se": fit["beta_se"],
        "slope_t_vs_one": (fit["beta"] - 1.0) / fit["beta_se"] if fit["beta_se"] > 0 else float("nan"),
        "intercept": fit["alpha"], "intercept_se": fit["alpha_se"],
        "corr": float(np.corrcoef(rolling["rate"], rolling["implied_financing"])[0, 1]),
        "n": int(len(rolling)),
    }


# --------------------------------------------------------------------------- #
# Robustness — eras, rate regimes, assumption sweeps
# --------------------------------------------------------------------------- #
def era_cut(
    fund_ret: pd.Series,
    bench_ret: pd.Series,
    rate_pct: pd.Series,
    leverage: int,
    expense_ratio: float,
    bench_fee: float = 0.0,
    split: str = "2018-01-01",
) -> dict:
    """Re-estimate the financing spread on each side of a calendar split."""
    out = {}
    for tag, sl in [("early", slice(None, split)), ("late", slice(split, None))]:
        f, b = fund_ret.loc[sl], bench_ret.loc[sl]
        if len(f) < 250:
            out[tag] = None
            continue
        out[tag] = implied_financing(f, b, leverage, expense_ratio,
                                     bench_fee=bench_fee, rate_pct=rate_pct.loc[sl])
    return out


def rate_regime_cut(
    fund_ret: pd.Series,
    bench_ret: pd.Series,
    rate_pct: pd.Series,
    leverage: int,
    expense_ratio: float,
    bench_fee: float = 0.0,
    threshold: float = 1.0,
) -> dict:
    """Split the sample by the *level* of the benchmark rate, not by the calendar.

    A near-zero-rate day and a 5%-rate day are different worlds for a financing spread:
    at the zero bound the spread is most of the cost, at 5% it is a garnish. Days are
    assigned by that day's quoted rate, which is known at the close of that day — no
    look-ahead, and no trading decision hangs on it (this is a measurement split).
    """
    r = pd.Series(rate_pct).reindex(fund_ret.index)
    out = {}
    for tag, mask in [("zirp", r < threshold), ("hiked", r >= threshold)]:
        idx = r.index[mask.fillna(False)]
        if len(idx) < 250:
            out[tag] = None
            continue
        out[tag] = implied_financing(fund_ret.loc[idx], bench_ret.loc[idx], leverage,
                                     expense_ratio, bench_fee=bench_fee,
                                     rate_pct=r.loc[idx])
    return out


def expense_ratio_sweep(
    fund_ret: pd.Series,
    bench_ret: pd.Series,
    rate_pct: pd.Series,
    leverage: int,
    expense_ratios=(0.75, 0.85, 0.89, 0.95, 1.05),
    bench_fee: float = 0.0,
) -> list[dict]:
    """Sweep the assumed expense ratio — the study's one soft, non-tape input.

    The *drag* is measured, not assumed; only its split between "fee" and "financing"
    depends on the prospectus number. A 15 bp error in the assumed fee moves the implied
    financing rate by 15 bp / (L - 1), so the 3x wrapper is the more robust estimate.
    """
    base = implied_financing(fund_ret, bench_ret, leverage, expense_ratios[0],
                             bench_fee=bench_fee, rate_pct=rate_pct)
    rows = []
    for er in expense_ratios:
        f = (base["drag_ann_pct"] - er) / (leverage - 1)
        rows.append({
            "expense_ratio_pct": er,
            "implied_financing_pct": f,
            "spread_over_rate_pct": f - base["mean_rate_pct"],
        })
    return rows


# --------------------------------------------------------------------------- #
# The practical read — wrapper vs a self-financed margin replication
# --------------------------------------------------------------------------- #
def replicate(
    bench_ret: pd.Series,
    rate_pct: pd.Series,
    leverage: int,
    margin_spread: float,
    cost_bps: float = 1.0,
) -> pd.Series:
    """Daily total return of a self-financed, daily-reset ``L``x position in the benchmark.

    The investor holds ``L`` times NAV in SPY and borrows ``L - 1`` times NAV at
    ``rate + margin_spread`` (percent per annum, ACT/252). The exposure is reset to ``L``
    every close — the same mechanic the wrapper runs — which forces a daily trade of
    ``L * (L - 1) * |r|`` of NAV; that turnover is charged at ``cost_bps`` one-way
    **against NAV**.

    **One execution lag:** the position that earns day ``t``'s return was set at the
    close of ``t-1``, and the rate charged over day ``t`` is the one quoted at the close
    of ``t-1``. Both legs are shifted together, so no information from day ``t`` enters
    day ``t``'s own return.

    **ELIGIBILITY, not just price (a labelled limit of this arm).** Reg T caps a US retail
    margin account's *initial* leverage at 2x, so the ``L = 3`` replication is not
    available in an ordinary margin account at all — it needs portfolio margin, futures or
    a prime broker. The 3x race is therefore a *counterfactual price comparison*, not an
    executable alternative for a retail holder, and the sub-1% margin spreads in the grid
    belong to desks that same holder cannot open an account with. Neither is a margin call
    modelled here: the wrapper's loss is capped at NAV and the margin account's is not.
    Both omissions flatter the do-it-yourself arm, so the break-even below is a
    conservative line for the wrapper.
    """
    common = bench_ret.index.intersection(rate_pct.index)
    x = bench_ret.reindex(common).astype(float)
    rate = pd.Series(rate_pct).reindex(common).astype(float).shift(1)
    borrow = (leverage - 1) * (rate + margin_spread) / 100.0 / TRADING_DAYS
    turnover = leverage * (leverage - 1) * x.abs()
    friction = turnover * cost_bps * 1e-4
    return (leverage * x - borrow - friction).dropna().rename(f"rep_{leverage}x")


def replication_race(
    fund_ret: pd.Series,
    bench_ret: pd.Series,
    cash_ret: pd.Series,
    rate_pct: pd.Series,
    leverage: int,
    margin_spread: float,
    cost_bps: float = 1.0,
) -> dict:
    """Race the wrapper against the self-financed replication, excess-of-cash both sides.

    Both arms have the same cash leg (BIL's realised total return) subtracted, so the
    Sharpe comparison is excess-of-cash versus excess-of-cash and the cash leg cancels in
    the difference. Reports both arms' annualised excess Sharpe and CAGR, the annualised
    return difference, and the HAC *t* on the daily difference.

    **Do not read ``t_diff`` as strength of evidence for an edge.** Both arms hold the same
    ``L`` times the same benchmark, so the difference is a near-deterministic cost
    differential plus tracking noise; its t grows mechanically with the assumed margin
    spread and with sample length. A t of 30 here means "the arithmetic is stable", not
    "the finding is thirty-sigma". The economically meaningful outputs are the annualised
    gap and the break-even spread.
    """
    rep = replicate(bench_ret, rate_pct, leverage, margin_spread, cost_bps=cost_bps)
    common = fund_ret.index.intersection(rep.index).intersection(cash_ret.index)
    f = fund_ret.reindex(common)
    p = rep.reindex(common)
    c = cash_ret.reindex(common)
    e_fund, e_rep = (f - c).dropna(), (p - c).dropna()
    diff = (f - p).dropna()
    return {
        "margin_spread_pct": margin_spread, "cost_bps": cost_bps,
        "fund": summary(e_fund), "replication": summary(e_rep),
        "diff_ann_pct": float(diff.mean()) * TRADING_DAYS * 100.0,
        "t_diff": newey_west_t(diff.to_numpy(), lags=auto_lags(len(diff))),
        "sharpe_gap": summary(e_fund)["sharpe"] - summary(e_rep)["sharpe"],
        "n_days": int(len(diff)),
    }


def breakeven_margin_spread(
    fund_ret: pd.Series,
    bench_ret: pd.Series,
    rate_pct: pd.Series,
    leverage: int,
    cost_bps: float = 1.0,
) -> float:
    """The margin spread at which the do-it-yourself replication ties the wrapper.

    Solved in closed form from the mean daily returns: below this spread (and paying no
    more than ``cost_bps`` to run the daily reset) the margin account wins; above it, the
    wrapper does. Expressed in percent per annum over the benchmark rate.
    """
    rep0 = replicate(bench_ret, rate_pct, leverage, 0.0, cost_bps=cost_bps)
    common = fund_ret.index.intersection(rep0.index)
    gap = float((rep0.reindex(common) - fund_ret.reindex(common)).dropna().mean())
    return gap * TRADING_DAYS * 100.0 / (leverage - 1)


# --------------------------------------------------------------------------- #
# Performance summary
# --------------------------------------------------------------------------- #
def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Headline annualised stats for a daily simple-return series (pass excess-of-cash)."""
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n < 3:
        return {k: float("nan") for k in
                ("n_days", "cagr", "sharpe", "vol_ann", "max_drawdown", "mean_ann_pct", "tstat")}
    mu, sd = r.mean(), r.std(ddof=1)
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    return {
        "n_days": int(n),
        "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if wealth.iloc[-1] > 0 else float("nan"),
        "sharpe": float(mu / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan"),
        "vol_ann": float(sd * np.sqrt(periods_per_year)),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "mean_ann_pct": float(mu * periods_per_year * 100.0),
        "tstat": newey_west_t(r.to_numpy(), lags=auto_lags(n)),
    }


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports the stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, truth: dict, leverage: int = 2) -> dict:
    """Run the estimator on a synthetic tape whose financing rate is known exactly.

    On the planted world (``signal_strength = 1``) the recovered spread must land on the
    planted 75 bp; on the null (``signal_strength = 0``) it must land on zero. Proves the
    arithmetic is unbiased — it never supports a real-tape stamp.
    """
    r_idx = prices["index"].pct_change()
    r_fund = prices[f"fund_{leverage}"].pct_change()
    est = implied_financing(
        r_fund.dropna(), r_idx.dropna(), leverage,
        expense_ratio=truth["expense_ratio_pct"], bench_fee=0.0,
        rate_pct=prices["rate"],
    )
    est["planted_spread_pct"] = truth["planted_spread_pct"]
    est["spread_error_pct"] = est["spread_over_rate_pct"] - truth["planted_spread_pct"]
    return est
