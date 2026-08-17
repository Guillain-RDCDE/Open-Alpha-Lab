"""Strategy + inference for Study 917 — Stale NAV.

The claim: a US-listed single-country ETF (EWJ, EWG, FXI, EWA, EWU) tracks a market that
is **shut** while Wall Street trades, so its price is supposed to be *stale* with respect
to the US session that has just finished — and should therefore **catch up** the next day.
If so, today's SPY return predicts tomorrow's country-ETF return with a **positive** slope,
and a rule that goes long the fund after a top-decile SPY day should make money.

What we run:

1. **The predictive regression.** ``r_fund[t+1] = a + b · r_spy[t] + e``, HAC (Newey-West)
   standard errors, per fund and per era. The catch-up story predicts ``b > 0``.
2. **The domestic control.** The identical regression with SPY on itself,
   ``r_spy[t+1] = a + b · r_spy[t]``. The US market has its own short-horizon reversal; a
   country ETF with beta ≈ 1 inherits it mechanically. Any *timezone* effect must be what
   is left after that — so we also run the **relative** regression
   ``(r_fund[t+1] − r_spy[t+1]) = a + b · r_spy[t]``, which nets the shared beta out.
3. **The tradable rule.** Form the signal at the close of day ``t`` (was SPY's day-``t``
   return in the top decile of its own *expanding* history?) and hold the fund for the
   day ``t+1`` return. **Exactly one execution lag.** Costs are charged one-way × NAV on
   every change of position; the mirror (short) version additionally pays a borrow fee.
4. **Robustness.** Era cut at 2010-01-01 (fair-value pricing and 24-hour arbitrage were
   supposed to have killed this by then), a cost sweep, a borrow sweep for the short
   mirror, a decile-threshold sweep, a conservative *extra*-lag variant (in case an
   at-the-close execution on the signal day is too generous), and a circular block
   bootstrap on the headline mean.

Returns are **simple** (arithmetic) so a binary in/out portfolio return is exactly
``position · r_fund + (1 − |position|) · r_cash`` and the wealth path compounds correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def _auto_lags(n: int) -> int:
    return int(np.floor(4.0 * (max(n, 1) / 100.0) ** (2.0 / 9.0)))


def one_sample_t(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a, b) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x, lags: int | None = None) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = _auto_lags(n)
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def ols_hac(y, x, lags: int | None = None) -> dict:
    """Univariate OLS ``y = a + b·x`` with Newey-West (HAC) standard errors.

    Rows where either side is NaN are dropped pairwise. Returns ``alpha``, ``beta``,
    ``t_beta``, ``se_beta``, the sample size and the lag truncation actually used.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    m = ~(np.isnan(y) | np.isnan(x))
    y, x = y[m], x[m]
    n = len(y)
    if n < 30 or x.std(ddof=1) == 0:
        return {"alpha": float("nan"), "beta": float("nan"), "t_beta": float("nan"),
                "se_beta": float("nan"), "n": int(n), "lags": 0}
    if lags is None:
        lags = _auto_lags(n)
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    Z = X * e[:, None]
    S = Z.T @ Z / n
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        A = Z[l:].T @ Z[:-l] / n
        S += w * (A + A.T)
    XtX_inv = np.linalg.inv(X.T @ X / n)
    V = XtX_inv @ S @ XtX_inv / n
    se = np.sqrt(np.diag(V))
    return {
        "alpha": float(beta[0]), "beta": float(beta[1]),
        "se_beta": float(se[1]),
        "t_beta": float(beta[1] / se[1]) if se[1] > 0 else float("nan"),
        "n": int(n), "lags": int(lags),
    }


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# The predictive regressions
# --------------------------------------------------------------------------- #
def lagged_regression(fund_ret: pd.Series, spy_ret: pd.Series, lag: int = 1) -> dict:
    """Regress the fund's day ``t+lag`` return on SPY's day ``t`` return, HAC t.

    The catch-up claim predicts ``beta > 0``: a strong US session today is "owed" to the
    country fund tomorrow. A negative beta means the fund *gives back* part of the US move
    — reversal, not catch-up.
    """
    y = pd.Series(fund_ret).shift(-lag)
    x = pd.Series(spy_ret)
    common = y.index.intersection(x.index)
    return ols_hac(y.reindex(common), x.reindex(common))


def relative_regression(fund_ret: pd.Series, spy_ret: pd.Series, lag: int = 1) -> dict:
    """Same regression on the fund's return **net of SPY's own next-day return**.

    ``(r_fund[t+1] − r_spy[t+1]) = a + b·r_spy[t]``. This is the timezone-specific test:
    it removes whatever the US market itself does the next day (SPY has its own
    short-horizon reversal), which a beta-1 country ETF would inherit for free.

    **ASSUMPTION — unit beta.** Subtracting *one* unit of SPY assumes the fund's
    contemporaneous US beta is exactly 1. It is not: on the real tape it runs 0.75 (EWJ)
    to 1.16 (FXI), so this hedge leaves ``(beta − 1)`` units of SPY in the residual, which
    then re-imports ``(beta − 1) × b_domestic`` of the market's own reversal. The
    advantage is that nothing is fitted — no in-sample parameter enters the headline.
    :func:`beta_hedged_regression` drops the assumption at the price of a fitted beta;
    both are reported, and the study's conclusion must not depend on which you read.
    """
    rel = pd.Series(fund_ret) - pd.Series(spy_ret)
    return lagged_regression(rel, spy_ret, lag=lag)


def beta_hedged_regression(fund_ret: pd.Series, spy_ret: pd.Series, lag: int = 1) -> dict:
    """:func:`relative_regression` with the unit-beta assumption replaced by a fitted beta.

    Estimates the fund's **contemporaneous** US beta by OLS on the same sample
    (``r_fund[t] = a + beta·r_spy[t]``) and regresses ``(r_fund[t+1] − beta·r_spy[t+1])``
    on ``r_spy[t]``. This is the *correctly hedged* timezone-specific slope.

    **CAVEAT — the beta is fitted in-sample and applied in-sample.** It is not a tradable
    rule and is never used to stamp a badge; it exists to show whether the headline's
    unit-beta shortcut is load-bearing. (It is: on the real tape the hedge ratio moves
    EWJ's slope from *t* = −1.51 to *t* = −3.19.) Reported alongside, never instead of,
    the unfitted relative slope. Returns the usual ``ols_hac`` dict plus ``beta_contemp``.
    """
    f = pd.Series(fund_ret).astype(float)
    s = pd.Series(spy_ret).astype(float)
    common = f.index.intersection(s.index)
    f, s = f.reindex(common), s.reindex(common)
    contemp = ols_hac(f, s)
    b_c = contemp["beta"]
    if not np.isfinite(b_c):
        out = ols_hac([], [])
        out["beta_contemp"] = float("nan")
        return out
    out = lagged_regression(f - b_c * s, s, lag=lag)
    out["beta_contemp"] = float(b_c)
    return out


def confound_decomposition(fund_ret: pd.Series, spy_ret: pd.Series, lag: int = 1) -> dict:
    """How much of a fund's raw slope is just SPY's own reversal arriving through beta?

    ``b_raw ≈ beta_contemp · b_domestic + b_hedged``. The first term is the **confound**
    (the US market's short-horizon reversal, inherited mechanically); the second is what
    is left for a timezone story. ``share`` is the confound as a fraction of the raw
    slope — above 1.0 means the confound *over*-explains it (the fund reverses *less*
    than its beta says it should).
    """
    raw = lagged_regression(fund_ret, spy_ret, lag=lag)
    hedged = beta_hedged_regression(fund_ret, spy_ret, lag=lag)
    dom = domestic_control(spy_ret, lag=lag)
    confound = hedged["beta_contemp"] * dom["beta"]
    share = confound / raw["beta"] if raw["beta"] not in (0.0,) else float("nan")
    return {"beta_raw": raw["beta"], "t_raw": raw["t_beta"],
            "beta_contemp": hedged["beta_contemp"], "beta_domestic": dom["beta"],
            "confound": float(confound), "share": float(share),
            "beta_hedged": hedged["beta"], "t_hedged": hedged["t_beta"]}


def per_fund_regressions(returns: pd.DataFrame, funds, benchmark: str = "SPY",
                         lag: int = 1) -> pd.DataFrame:
    """Raw, unit-beta relative, and fitted-beta hedged lagged regressions per fund.

    ``beta_rel`` assumes a unit hedge (nothing fitted); ``beta_hedged`` uses the fund's
    own in-sample contemporaneous beta (fitted — see :func:`beta_hedged_regression`).
    Both are carried so a reader can see whether the conclusion turns on the hedge ratio.
    """
    spy = returns[benchmark]
    rows = []
    for f in funds:
        raw = lagged_regression(returns[f], spy, lag=lag)
        rel = relative_regression(returns[f], spy, lag=lag)
        hed = beta_hedged_regression(returns[f], spy, lag=lag)
        rows.append({"fund": f, "n": raw["n"],
                     "beta_raw": raw["beta"], "t_raw": raw["t_beta"],
                     "beta_rel": rel["beta"], "t_rel": rel["t_beta"],
                     "beta_contemp": hed["beta_contemp"],
                     "beta_hedged": hed["beta"], "t_hedged": hed["t_beta"]})
    return pd.DataFrame(rows).set_index("fund")


def domestic_control(spy_ret: pd.Series, lag: int = 1) -> dict:
    """SPY on itself: ``r_spy[t+1] = a + b·r_spy[t]``. The confound to beat."""
    return lagged_regression(spy_ret, spy_ret, lag=lag)


# --------------------------------------------------------------------------- #
# Signal & backtest
# --------------------------------------------------------------------------- #
def trigger_signal(spy_ret: pd.Series, top_q: float = 0.90, min_obs: int = 250) -> pd.Series:
    """1 on days whose SPY return is in the top ``top_q`` of its own *expanding* history.

    The threshold at day ``t`` uses only returns through ``t`` (an expanding quantile with
    a ``min_obs`` warmup), so the decile cut is never chosen with hindsight. The series
    returned is **unlagged**: it is the state of the world at the close of ``t``. The one
    execution lag is applied in :func:`run_rule`.
    """
    r = pd.Series(spy_ret).astype(float)
    thr = r.expanding(min_obs).quantile(top_q)
    sig = (r >= thr).astype(float)
    sig[thr.isna()] = np.nan
    return sig.rename("signal")


def run_rule(
    fund_ret: pd.Series,
    cash_ret: pd.Series,
    signal: pd.Series,
    direction: int = 1,
    cost_bps: float = 10.0,
    borrow_ann: float = 0.0,
    extra_lag: int = 0,
) -> pd.DataFrame:
    """Hold the fund for the day-``t+1`` return whenever the day-``t`` signal fired.

    Parameters
    ----------
    fund_ret, cash_ret:
        Daily simple total returns of the country ETF and of the cash leg.
    signal:
        The unlagged {0, 1} trigger series. Shifted by ``1 + extra_lag`` days here — that
        single shift IS the study's execution lag (signal formed through the close of
        ``t``, position on for the ``t+1`` return).
    direction:
        ``+1`` = the claim (long the fund after a strong US day); ``-1`` = the mirror
        (short it).
    cost_bps:
        One-way transaction cost in bps of NAV, charged on every change of position (so a
        one-day round trip pays it twice). Country ETFs are wider than SPY: 10 bps one-way
        is our baseline **ASSUMPTION**, swept in :func:`cost_sweep`.
    borrow_ann:
        Annual stock-borrow fee, charged daily on the short leg only (an **ASSUMPTION**,
        swept in :func:`borrow_sweep`). Zero when ``direction = +1``, which has no short.
    extra_lag:
        Extra days of delay, for the conservative variant that refuses to assume an
        at-the-close execution on the signal day itself.

    Returns a frame with ``position``, ``r_fund``, ``r_cash``, ``r_strategy`` (net) and
    ``excess`` (strategy minus cash).
    """
    f = pd.Series(fund_ret).astype(float)
    c = pd.Series(cash_ret).astype(float)
    common = f.index.intersection(c.index).intersection(signal.index)
    f, c = f.reindex(common), c.reindex(common)
    pos = (pd.Series(signal).reindex(common).shift(1 + int(extra_lag))
           .fillna(0.0) * float(direction))

    turnover = pos.diff().abs()
    turnover.iloc[0] = abs(pos.iloc[0]) if len(pos) else 0.0
    cost = turnover * cost_bps * 1e-4
    borrow = (pos < 0).astype(float) * (borrow_ann / TRADING_DAYS)

    # Uninvested NAV earns cash; a short position still earns cash on the full NAV
    # (the short proceeds sit in the box) and pays borrow.
    r_strat = (pos * f + (1.0 - pos.abs()) * c + (pos < 0).astype(float) * c
               - cost - borrow).rename("r_strategy")

    out = pd.concat([pos.rename("position"), f.rename("r_fund"), c.rename("r_cash"),
                     r_strat], axis=1).dropna()
    out["excess"] = out["r_strategy"] - out["r_cash"]
    return out


def conditional_stats(excess: pd.Series, position: pd.Series) -> dict:
    """Mean excess return on trigger days vs the rest, with HAC t on the trigger leg."""
    e = pd.Series(excess).astype(float)
    p = pd.Series(position).astype(float).reindex(e.index)
    on = e[p != 0].dropna()
    off = e[p == 0].dropna()
    return {
        "n_on": int(len(on)), "n_off": int(len(off)),
        "mean_on_bps": float(on.mean() * 1e4) if len(on) else float("nan"),
        "mean_off_bps": float(off.mean() * 1e4) if len(off) else float("nan"),
        "diff_bps": float((on.mean() - off.mean()) * 1e4) if len(on) and len(off) else float("nan"),
        "t_on": newey_west_t(on.to_numpy()),
        "t_diff_welch": welch_t(on.to_numpy(), off.to_numpy()),
    }


# --------------------------------------------------------------------------- #
# Performance summary
# --------------------------------------------------------------------------- #
def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Headline annualised stats for a daily simple-return series.

    Pass an *excess-of-cash* series to get the excess Sharpe (the race this study runs).
    """
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n < 3:
        return {"n_days": int(n), "cagr": float("nan"), "sharpe": float("nan"),
                "vol_ann": float("nan"), "max_drawdown": float("nan"),
                "mean_daily_bps": float("nan"), "tstat": float("nan")}
    mu, std = r.mean(), r.std(ddof=1)
    sharpe = float(mu / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    cagr = (float(wealth.iloc[-1] ** (1.0 / years) - 1.0)
            if years > 0 and wealth.iloc[-1] > 0 else float("nan"))
    return {
        "n_days": int(n), "cagr": cagr, "sharpe": sharpe,
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "mean_daily_bps": float(mu * 1e4),
        "tstat": newey_west_t(r.to_numpy()),
    }


def equal_weight_basket(returns: pd.DataFrame, funds) -> pd.Series:
    """Equal-weight daily return of the country funds **available on each day**.

    FXI only lists in 2004, so the basket is a 4-fund basket until then and a 5-fund
    basket after — documented rather than hidden, and cross-checked per fund.
    """
    return returns[list(funds)].mean(axis=1, skipna=True).rename("basket")


# --------------------------------------------------------------------------- #
# The headline race
# --------------------------------------------------------------------------- #
def compare(
    fund_ret: pd.Series,
    spy_ret: pd.Series,
    cash_ret: pd.Series,
    top_q: float = 0.90,
    cost_bps: float = 10.0,
    borrow_ann: float = 0.0,
    min_obs: int = 250,
) -> dict:
    """Race the long rule, the short mirror and buy-and-hold, all excess-of-cash.

    Returns the two regressions (raw and relative), the conditional day statistics, the
    net excess-Sharpe of each arm, and the underlying series for the bootstrap.
    """
    sig = trigger_signal(spy_ret, top_q=top_q, min_obs=min_obs)
    long_bt = run_rule(fund_ret, cash_ret, sig, direction=+1, cost_bps=cost_bps)
    short_bt = run_rule(fund_ret, cash_ret, sig, direction=-1, cost_bps=cost_bps,
                        borrow_ann=borrow_ann)
    gross_bt = run_rule(fund_ret, cash_ret, sig, direction=+1, cost_bps=0.0)

    idx = long_bt.index
    e_bh = (pd.Series(fund_ret).reindex(idx) - pd.Series(cash_ret).reindex(idx)).dropna()

    return {
        "reg_raw": lagged_regression(fund_ret, spy_ret),
        "reg_rel": relative_regression(fund_ret, spy_ret),
        "cond": conditional_stats(e_bh.reindex(idx), long_bt["position"]),
        "long": summary(long_bt["excess"]),
        "short": summary(short_bt["excess"]),
        "gross_long": summary(gross_bt["excess"]),
        "bh": summary(e_bh),
        "trigger_frac": float((long_bt["position"] != 0).mean()),
        "turnover": float(long_bt["position"].diff().abs().sum()),
        "e_long": long_bt["excess"], "e_short": short_bt["excess"], "e_bh": e_bh,
        "position": long_bt["position"],
    }


# --------------------------------------------------------------------------- #
# Bootstrap (circular block)
# --------------------------------------------------------------------------- #
def bootstrap_mean_ci(
    x: pd.Series,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 917,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the mean of a daily series (reported in bps)."""
    r = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"mean_bps": float("nan"), "ci_low_bps": float("nan"),
                "ci_high_bps": float("nan"), "frac_negative": float("nan"), "n_obs": int(n)}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[b] = r[idx].mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean_bps": float(r.mean() * 1e4), "ci_low_bps": float(lo * 1e4),
            "ci_high_bps": float(hi * 1e4), "frac_negative": float((boots < 0).mean()),
            "n_obs": int(n), "n_boot": int(n_boot), "block": int(block)}


def bootstrap_sharpe_ci(
    excess: pd.Series,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 917,
    periods_per_year: int = TRADING_DAYS,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the annualised Sharpe of an excess-return series."""
    r = np.asarray(pd.Series(excess).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"sharpe": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": int(n)}
    ann = np.sqrt(periods_per_year)
    sd = r.std(ddof=1)
    point = float(r.mean() / sd * ann) if sd > 0 else float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.full(n_boot, np.nan)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        s = r[idx]
        sdb = s.std(ddof=1)
        if sdb > 0:
            boots[b] = s.mean() / sdb * ann
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((valid < 0).mean()), "n_obs": int(n),
            "n_boot": int(valid.size), "block": int(block)}


# --------------------------------------------------------------------------- #
# Robustness: eras, costs, borrow, threshold, extra lag
# --------------------------------------------------------------------------- #
def era_cut(fund_ret: pd.Series, spy_ret: pd.Series, cash_ret: pd.Series,
            split: str = "2010-01-01", **kw) -> dict:
    """Split at ``split`` and re-run the whole race on each half.

    2010 is the natural cut: US fair-value pricing rules and round-the-clock ETF
    arbitrage were both well established by then, so a genuine stale-price effect should
    be visibly weaker (or gone) in the late half.
    """
    out = {}
    for tag, sl in [("early", slice(None, split)), ("late", slice(split, None))]:
        f = pd.Series(fund_ret).loc[sl]
        s = pd.Series(spy_ret).loc[sl]
        c = pd.Series(cash_ret).loc[sl]
        if len(f.dropna()) < 400:
            out[tag] = None
            continue
        cmp = compare(f, s, c, **kw)
        hed = beta_hedged_regression(f, s)     # beta refit *within* the era
        out[tag] = {
            "n": cmp["reg_raw"]["n"],
            "beta_raw": cmp["reg_raw"]["beta"], "t_raw": cmp["reg_raw"]["t_beta"],
            "beta_rel": cmp["reg_rel"]["beta"], "t_rel": cmp["reg_rel"]["t_beta"],
            "beta_hedged": hed["beta"], "t_hedged": hed["t_beta"],
            "beta_contemp": hed["beta_contemp"],
            "mean_on_bps": cmp["cond"]["mean_on_bps"], "t_on": cmp["cond"]["t_on"],
            "n_on": cmp["cond"]["n_on"],
            "sharpe_long": cmp["long"]["sharpe"], "sharpe_short": cmp["short"]["sharpe"],
            "sharpe_bh": cmp["bh"]["sharpe"],
        }
    return out


def cost_sweep(fund_ret: pd.Series, spy_ret: pd.Series, cash_ret: pd.Series,
               grid=(0.0, 5.0, 10.0, 25.0, 50.0), top_q: float = 0.90) -> list:
    """Net excess-Sharpe of both directions across one-way costs (bps of NAV)."""
    sig = trigger_signal(spy_ret, top_q=top_q)
    rows = []
    for c in grid:
        lo = run_rule(fund_ret, cash_ret, sig, direction=+1, cost_bps=c)
        sh = run_rule(fund_ret, cash_ret, sig, direction=-1, cost_bps=c)
        rows.append({"cost_bps": c,
                     "sharpe_long": summary(lo["excess"])["sharpe"],
                     "t_long": newey_west_t(lo["excess"].to_numpy()),
                     "sharpe_short": summary(sh["excess"])["sharpe"],
                     "t_short": newey_west_t(sh["excess"].to_numpy())})
    return rows


def borrow_sweep(fund_ret: pd.Series, spy_ret: pd.Series, cash_ret: pd.Series,
                 grid=(0.0, 0.005, 0.02, 0.05), cost_bps: float = 10.0,
                 top_q: float = 0.90) -> list:
    """Short-mirror net excess-Sharpe across annual borrow fees (the short leg's rent)."""
    sig = trigger_signal(spy_ret, top_q=top_q)
    rows = []
    for b in grid:
        sh = run_rule(fund_ret, cash_ret, sig, direction=-1, cost_bps=cost_bps,
                      borrow_ann=b)
        rows.append({"borrow_ann": b,
                     "sharpe_short": summary(sh["excess"])["sharpe"],
                     "t_short": newey_west_t(sh["excess"].to_numpy())})
    return rows


def threshold_sweep(fund_ret: pd.Series, spy_ret: pd.Series, cash_ret: pd.Series,
                    grid=(0.80, 0.90, 0.95, 0.99), cost_bps: float = 10.0) -> list:
    """Is the decile cut knife-edge? Sweep the quantile that defines a "strong US day"."""
    rows = []
    for q in grid:
        sig = trigger_signal(spy_ret, top_q=q)
        bt = run_rule(fund_ret, cash_ret, sig, direction=+1, cost_bps=cost_bps)
        e_bh = (pd.Series(fund_ret).reindex(bt.index)
                - pd.Series(cash_ret).reindex(bt.index))
        cond = conditional_stats(e_bh, bt["position"])
        rows.append({"top_q": q, "n_on": cond["n_on"], "mean_on_bps": cond["mean_on_bps"],
                     "t_on": cond["t_on"],
                     "sharpe_long": summary(bt["excess"])["sharpe"]})
    return rows


def extra_lag_check(fund_ret: pd.Series, spy_ret: pd.Series, cash_ret: pd.Series,
                    cost_bps: float = 10.0, top_q: float = 0.90) -> dict:
    """The conservative variant: wait one *more* day before taking the position.

    The base rule assumes you can execute at the close of the signal day (a market-on-close
    order placed on information that is only final at that close — optimistic). If any
    effect is real and merely slow, some of it should survive one extra day.
    """
    sig = trigger_signal(spy_ret, top_q=top_q)
    bt = run_rule(fund_ret, cash_ret, sig, direction=+1, cost_bps=cost_bps, extra_lag=1)
    e_bh = pd.Series(fund_ret).reindex(bt.index) - pd.Series(cash_ret).reindex(bt.index)
    cond = conditional_stats(e_bh, bt["position"])
    return {"mean_on_bps": cond["mean_on_bps"], "t_on": cond["t_on"],
            "n_on": cond["n_on"], "sharpe_long": summary(bt["excess"])["sharpe"]}


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports a real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, truth: dict, cost_bps: float = 10.0) -> dict:
    """Run the study's machinery on a synthetic panel with a known planted catch-up.

    On the planted world the lagged regression must recover a **positive** beta close to
    ``truth['stale_beta_effective']`` with a large HAC t, and the long rule must earn a
    positive mean on trigger days. On the null both must be quiet. This proves the
    harness is unbiased; it never supports a real-tape verdict.
    """
    rets = prices.pct_change()
    spy = rets["SPY"]
    cash = rets["cash"]
    funds = truth["funds"]
    betas, ts, means, t_means = [], [], [], []
    for f in funds:
        reg = lagged_regression(rets[f], spy)
        betas.append(reg["beta"]); ts.append(reg["t_beta"])
        sig = trigger_signal(spy)
        bt = run_rule(rets[f], cash, sig, direction=+1, cost_bps=cost_bps)
        e_bh = rets[f].reindex(bt.index) - cash.reindex(bt.index)
        cond = conditional_stats(e_bh, bt["position"])
        means.append(cond["mean_on_bps"]); t_means.append(cond["t_on"])
    return {
        "planted_beta": truth["stale_beta_effective"],
        "beta_mean": float(np.mean(betas)), "t_beta_mean": float(np.mean(ts)),
        "t_beta_max_abs": float(np.max(np.abs(ts))),
        "mean_on_bps": float(np.mean(means)), "t_on_mean": float(np.mean(t_means)),
        "n_funds": len(funds),
    }
