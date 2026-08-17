"""Replication + inference for Study 954 — High Yield in Disguise.

The construction. A high-yield credit fund is supposed to be a *bond* — but its returns
famously behave like a blend of equity risk and interest-rate duration. So build that
blend explicitly and hold it out of sample:

1. On every trading day ``t`` fit, on the trailing ``window`` days of **total-return**
   daily returns, the constrained regression

       r_HY - r_IEF  =  w * (r_SPY - r_IEF)  +  residual

   whose slope ``w`` is exactly the equity share of a fully funded
   ``w * SPY + (1 - w) * IEF`` blend (the ``(1 - w)`` on the duration leg is implied, so
   the two weights always sum to 1 and no cash is created or destroyed).
2. Freeze ``w`` at the **last trading day of each calendar month** and apply it to the
   whole of the *following* month. That month-end freeze is the study's **single
   execution lag**: the weight used on day ``t`` was fitted on returns ending at least
   one trading day before ``t`` and is never re-fitted intra-month. No second lag is
   stacked on top of it.
3. Charge the replication a one-way cost on each rebalance (turnover x NAV, both legs)
   and a borrow charge on any leg the fitted weight pushes short. The high-yield fund
   itself is bought once and held, so it pays no rebalancing friction at all — the
   comparison is deliberately generous to the fund.

Then two questions, in this order:

- **Is HY a costume?** How much of the fund's daily variation does the replication
  actually explain, and what is left over (tracking error, residual mean)? The residual
  mean is tested with a Newey-West *t*.
- **Does the costume pay?** At the *same realised volatility*, does the fund or the
  replication deliver more **excess-of-cash** return? That is precisely the excess-Sharpe
  difference, so we test it as the HAC *t* on the vol-matched daily return difference and
  put a circular block-bootstrap CI around it. Then we look at what each side actually
  cost an owner in 2008, 2020 and 2022.

All returns are **simple** (arithmetic) daily total returns, so a fully funded blend's
return is exactly ``w * r_equity + (1 - w) * r_duration`` and the wealth path compounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# The three crisis windows the practical question asks about. Fixed calendar windows,
# not an event-detection rule, so nothing is picked with hindsight beyond the choice of
# the three episodes everybody already names.
CRISES = {
    "2008 GFC": ("2008-06-01", "2009-06-30"),
    "2020 Covid": ("2020-02-01", "2020-04-30"),
    "2022 rate shock": ("2022-01-01", "2022-12-31"),
}


# --------------------------------------------------------------------------- #
# Inference primitives (mirror of Study 803)
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
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# The fitted blend weight
# --------------------------------------------------------------------------- #
def rolling_beta(
    r_hy: pd.Series, r_eq: pd.Series, r_dur: pd.Series, window: int = 252
) -> pd.Series:
    """Trailing-``window`` OLS slope of ``(r_hy - r_dur)`` on ``(r_eq - r_dur)``.

    That slope *is* the equity share ``w`` of the fully funded blend
    ``w * equity + (1 - w) * duration`` that best fits the fund, because subtracting the
    duration leg from both sides imposes the "weights sum to one" constraint exactly.
    The value at day ``t`` uses only returns through ``t``.
    """
    y = (r_hy - r_dur)
    x = (r_eq - r_dur)
    cov = y.rolling(window, min_periods=window).cov(x)
    var = x.rolling(window, min_periods=window).var()
    return (cov / var).rename("beta")


def held_out_weights(beta: pd.Series) -> pd.Series:
    """Freeze ``beta`` at each month-end and apply it to the *following* month.

    This is the study's single execution lag: the weight in force on day ``t`` was
    estimated on returns ending at the previous calendar month-end, i.e. at least one
    trading day (and at most a month) before it is used. Returns a step series aligned
    to ``beta.index``, NaN until the first complete estimation window has passed.
    """
    key = pd.Series(beta.index.to_period("M"), index=beta.index)
    month_end = beta.groupby(key).last()
    return key.map(month_end.shift(1)).astype(float).rename("w")


# --------------------------------------------------------------------------- #
# The replication backtest
# --------------------------------------------------------------------------- #
def replicate(
    hy: pd.Series,
    equity: pd.Series,
    duration: pd.Series,
    cash: pd.Series,
    window: int = 252,
    cost_bps: float = 2.0,
    borrow_bps_ann: float = 50.0,
) -> pd.DataFrame:
    """Build the held-out ``w * equity + (1 - w) * duration`` replication of ``hy``.

    Parameters
    ----------
    hy, equity, duration, cash:
        Daily total-return **close levels** of the high-yield fund, the equity leg
        (SPY), the duration leg (IEF) and the cash proxy (BIL).
    window:
        Estimation window in trading days for the rolling beta (default 252 = one year).
        A design choice, swept in :func:`window_sweep`.
    cost_bps:
        One-way transaction cost in bps charged on the replication's turnover at each
        monthly rebalance (both legs move, so turnover is ``2 * |dw|``). A PROXY: 2 bps
        is generous for two of the most liquid ETFs alive; swept in :func:`cost_sweep`.
        The high-yield fund pays nothing — it is bought once and held.
    borrow_bps_ann:
        Annualised borrow spread charged on any leg the fitted weight pushes short
        (``w < 0`` shorts equity, ``w > 1`` shorts duration). A PROXY. On the real tape
        the fitted ``w`` never leaves ``[0, 1]``, so this charge is identically zero —
        :func:`compare` reports the realised short notional so the reader can check.

    Returns a frame with ``r_hy``, ``r_repl`` (net), ``r_cash``, ``w``, ``turnover`` and
    ``short_notional``. Simple returns; rows before the first held-out weight are dropped.
    """
    idx = hy.index
    for s in (equity, duration, cash):
        idx = idx.intersection(s.index)
    hy, eq, du, ca = (s.reindex(idx).sort_index() for s in (hy, equity, duration, cash))

    # ``fill_method=None`` is explicit on purpose: pandas 2.x forward-fills gaps by
    # default (turning a missing close into a fabricated 0% day) while pandas 3.x does
    # not. Pinning it keeps the headline identical on both, and a genuine hole in a tape
    # stays a hole — it is dropped, not invented.
    r_hy = hy.pct_change(fill_method=None).rename("r_hy")
    r_eq = eq.pct_change(fill_method=None)
    r_du = du.pct_change(fill_method=None)
    r_cash = ca.pct_change(fill_method=None).rename("r_cash")

    beta = rolling_beta(r_hy, r_eq, r_du, window=window)
    w = held_out_weights(beta)

    turnover = (w.diff().abs().fillna(0.0) * 2.0).rename("turnover")
    short_notional = (
        np.maximum(-w, 0.0) + np.maximum(w - 1.0, 0.0)
    ).rename("short_notional")

    gross = w * r_eq + (1.0 - w) * r_du
    friction = turnover * cost_bps * 1e-4 + short_notional * (borrow_bps_ann * 1e-4) / TRADING_DAYS
    r_repl = (gross - friction).rename("r_repl")

    out = pd.concat([r_hy, r_repl, r_cash, w, turnover, short_notional], axis=1)
    return out.dropna()


# --------------------------------------------------------------------------- #
# Performance summary
# --------------------------------------------------------------------------- #
def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Headline annualised stats for a daily simple-return series.

    Pass an *excess-of-cash* series to get the excess Sharpe — in which case ``cagr`` is
    the compounded **excess-of-cash** growth rate, **not** what the holder's statement
    showed; :func:`compare` reports the lived (absolute) CAGR separately so the two are
    never confused. ``max_drawdown`` is the drawdown of the series as given (pass the raw,
    not excess, series for the lived one).
    """
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    mu, std = r.mean(), r.std(ddof=1)
    sharpe = float(mu / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    cagr = (
        float(wealth.iloc[-1] ** (1.0 / years) - 1.0)
        if years > 0 and wealth.iloc[-1] > 0 else float("nan")
    )
    return {
        "n_days": int(n),
        "cagr": cagr,
        "sharpe": sharpe,
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "mean_daily_bps": float(mu * 1e4),
        "tstat": newey_west_t(r.to_numpy()),
    }


def max_drawdown(returns: pd.Series) -> float:
    r = pd.Series(returns).astype(float).dropna()
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def vol_matched_diff(e_a: pd.Series, e_b: pd.Series) -> pd.Series:
    """Daily difference of two excess-return series each scaled to unit realised vol.

    Scaling both arms to the same realised volatility before differencing answers the
    practical question directly — *for the same risk, which one paid more?* — and its
    mean is (up to the annualisation constant) the excess-Sharpe difference, so its HAC
    *t* is the Jobson-Korkie Sharpe-comparison test in Newey-West form.

    The two scalars are **full-sample realised volatilities**, i.e. computed *ex post*.
    That is deliberate and standard for a Sharpe-comparison statistic, but it means the
    vol-matched series is a **test statistic, not a tradable path**: nobody could have
    levered to those constants in advance. Every euro-denominated claim in this study
    (CAGR, drawdown, crisis table) comes from the unscaled series instead.
    """
    a, b = e_a.align(e_b, join="inner")
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    if not (sa > 0 and sb > 0):
        return pd.Series(dtype=float)
    return (a / sa - b / sb).dropna().rename("vol_matched_diff")


def sharpe_diff_tstat(e_a: pd.Series, e_b: pd.Series) -> float:
    """HAC *t* on the vol-matched difference: does arm A out-Sharpe arm B?"""
    d = vol_matched_diff(e_a, e_b)
    return newey_west_t(d.to_numpy()) if len(d) > 2 else float("nan")


# --------------------------------------------------------------------------- #
# The race (the headline)
# --------------------------------------------------------------------------- #
def compare(
    hy: pd.Series,
    equity: pd.Series,
    duration: pd.Series,
    cash: pd.Series,
    window: int = 252,
    cost_bps: float = 2.0,
    borrow_bps_ann: float = 50.0,
) -> dict:
    """Race the high-yield fund against its held-out equity + duration replication.

    Both arms are reported **excess-of-cash** (minus BIL's total return) so the Sharpe
    race is apples-to-apples; drawdowns are the *absolute* (lived) ones. Returns the two
    summaries, the replication quality (correlation, R-squared, tracking error, residual
    mean and its HAC *t*), the excess-Sharpe gap with its HAC *t*, and the realised
    weight path.
    """
    bt = replicate(hy, equity, duration, cash, window=window,
                   cost_bps=cost_bps, borrow_bps_ann=borrow_bps_ann)
    e_hy = (bt["r_hy"] - bt["r_cash"]).rename("e_hy")
    e_rp = (bt["r_repl"] - bt["r_cash"]).rename("e_repl")
    resid = (bt["r_hy"] - bt["r_repl"]).rename("residual")

    s_hy, s_rp = summary(e_hy), summary(e_rp)
    a_hy, a_rp = summary(bt["r_hy"]), summary(bt["r_repl"])
    corr = float(bt["r_hy"].corr(bt["r_repl"]))

    return {
        "hy": s_hy,
        "repl": s_rp,
        # Lived (absolute, total-return) growth rates — what the statement showed. The
        # ``cagr`` inside ``hy``/``repl`` is the *excess-of-cash* one; keep them apart.
        "cagr_hy_abs": a_hy["cagr"],
        "cagr_repl_abs": a_rp["cagr"],
        "cagr_cash_abs": summary(bt["r_cash"])["cagr"],
        "dd_hy_abs": max_drawdown(bt["r_hy"]),
        "dd_repl_abs": max_drawdown(bt["r_repl"]),
        "excess_sharpe_gap": s_hy["sharpe"] - s_rp["sharpe"],
        "t_gap": sharpe_diff_tstat(e_hy, e_rp),
        "corr": corr,
        "r2": corr ** 2,
        "tracking_error_ann": float(resid.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "residual_ann": float(resid.mean() * TRADING_DAYS),
        "t_residual": newey_west_t(resid.to_numpy()),
        "w_mean": float(bt["w"].mean()),
        "w_min": float(bt["w"].min()),
        "w_max": float(bt["w"].max()),
        "turnover_per_year": float(bt["turnover"].sum() / (len(bt) / TRADING_DAYS)),
        "short_notional_max": float(bt["short_notional"].max()),
        "start": bt.index[0], "end": bt.index[-1],
        "e_hy": e_hy, "e_repl": e_rp, "resid": resid, "bt": bt,
    }


def replication_r2_by_horizon(bt: pd.DataFrame, horizons=("W", "ME", "QE")) -> pd.DataFrame:
    """R-squared and tracking error of the replication at several return horizons.

    A fund whose bonds are marked stale would look badly replicated daily and well
    replicated monthly. If R-squared barely moves with the horizon, the unexplained part
    is genuine credit risk, not a marking artefact.
    """
    per_year = {"W": 52, "ME": 12, "QE": 4}
    rows = [{
        "horizon": "D", "n": int(len(bt)),
        "r2": float(bt["r_hy"].corr(bt["r_repl"]) ** 2),
        "te_ann": float((bt["r_hy"] - bt["r_repl"]).std(ddof=1) * np.sqrt(TRADING_DAYS)),
    }]
    for h in horizons:
        a = (1.0 + bt["r_hy"]).resample(h).prod() - 1.0
        b = (1.0 + bt["r_repl"]).resample(h).prod() - 1.0
        rows.append({
            "horizon": h, "n": int(len(a)),
            "r2": float(a.corr(b) ** 2),
            "te_ann": float((a - b).std(ddof=1) * np.sqrt(per_year[h])),
        })
    return pd.DataFrame(rows).set_index("horizon")


# --------------------------------------------------------------------------- #
# Bootstrap CI (circular block bootstrap)
# --------------------------------------------------------------------------- #
def bootstrap_sharpe_ci(
    excess: pd.Series,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 954,
    periods_per_year: int = TRADING_DAYS,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the annualised Sharpe of a daily series.

    Blocks of ``block`` consecutive days preserve volatility clustering. Works equally on
    an excess-return series (giving the excess Sharpe) and on the vol-matched difference
    (giving a CI on the Sharpe *gap*).
    """
    r = np.asarray(pd.Series(excess).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"sharpe": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": int(n), "n_boot": 0, "block": block}
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
        ssd = s.std(ddof=1)
        if ssd > 0:
            boots[b] = s.mean() / ssd * ann
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((valid < 0).mean()), "n_obs": int(n),
            "n_boot": int(valid.size), "block": block}


# --------------------------------------------------------------------------- #
# Robustness — eras, crises, costs, estimation window
# --------------------------------------------------------------------------- #
def era_cut(cmp: dict, split: str = "2017-01-01") -> dict:
    """Split the realised daily series at ``split`` and re-run the vol-matched race.

    Re-uses the *already held-out* series rather than re-fitting, so both halves inherit
    exactly the same out-of-sample weights the full run used.
    """
    out = {}
    for tag, mask in [
        ("early", cmp["e_hy"].index < pd.Timestamp(split)),
        ("late", cmp["e_hy"].index >= pd.Timestamp(split)),
    ]:
        a, b, res = cmp["e_hy"][mask], cmp["e_repl"][mask], cmp["resid"][mask]
        if len(a) < 60:
            out[tag] = None
            continue
        sa, sb = summary(a), summary(b)
        out[tag] = {
            "n_days": sa["n_days"],
            "sharpe_hy": sa["sharpe"], "sharpe_repl": sb["sharpe"],
            "excess_sharpe_gap": sa["sharpe"] - sb["sharpe"],
            "t_gap": sharpe_diff_tstat(a, b),
            "residual_ann": float(res.mean() * TRADING_DAYS),
            "t_residual": newey_west_t(res.to_numpy()),
        }
    return out


def crisis_table(cmp: dict, crises: dict = CRISES) -> pd.DataFrame:
    """Peak-to-trough drawdown and total return of each arm inside each crisis window.

    Absolute (lived) numbers, not excess-of-cash — this is what the holder actually saw
    on the statement.
    """
    bt = cmp["bt"]
    rows = []
    for tag, (a, b) in crises.items():
        m = (bt.index >= pd.Timestamp(a)) & (bt.index <= pd.Timestamp(b))
        if m.sum() < 20:
            continue
        row = {"crisis": tag, "n_days": int(m.sum())}
        for name, col in [("hy", "r_hy"), ("repl", "r_repl")]:
            w = (1.0 + bt[col][m]).cumprod()
            row[f"dd_{name}"] = float((w / w.cummax() - 1.0).min())
            row[f"ret_{name}"] = float(w.iloc[-1] - 1.0)
        rows.append(row)
    return pd.DataFrame(rows).set_index("crisis")


def cost_sweep(
    hy: pd.Series, equity: pd.Series, duration: pd.Series, cash: pd.Series,
    window: int = 252, grid=(0.0, 1.0, 2.0, 5.0, 10.0, 25.0),
    borrow_bps_ann: float = 50.0,
) -> pd.DataFrame:
    """Sweep the one-way cost PROXY charged to the replication's monthly rebalance."""
    rows = []
    for c in grid:
        cmp = compare(hy, equity, duration, cash, window=window,
                      cost_bps=c, borrow_bps_ann=borrow_bps_ann)
        rows.append({
            "cost_bps": c,
            "sharpe_repl": cmp["repl"]["sharpe"],
            "excess_sharpe_gap": cmp["excess_sharpe_gap"],
            "t_gap": cmp["t_gap"],
            "residual_ann": cmp["residual_ann"],
            "turnover_per_year": cmp["turnover_per_year"],
        })
    return pd.DataFrame(rows).set_index("cost_bps")


def leg_sweep(
    hy: pd.Series, equity: pd.Series, duration_legs: dict, cash: pd.Series,
    window: int = 252, cost_bps: float = 2.0,
) -> pd.DataFrame:
    """Re-run the whole race with a different **Treasury maturity** as the duration leg.

    Picking IEF (7-10y) as *the* duration leg is a design choice, not a fact of the tape:
    a high-yield fund's own duration is nearer 3-4 years, so SHY (1-3y) or IEI (3-7y) are
    at least as defensible, and TLT (20y+) is the aggressive end. The replication weight
    ``w`` re-fits itself to whatever leg it is given, so the *fit* survives — the question
    this sweep answers is whether the **conclusion** does. Anything that only holds at one
    maturity was a maturity bet, not a statement about credit.

    Pass Treasury legs only: a leg that itself contains corporate credit (AGG, BND) would
    smuggle the thing under test into the benchmark.
    """
    rows = []
    for name, dur in duration_legs.items():
        cmp = compare(hy, equity, dur, cash, window=window, cost_bps=cost_bps)
        rows.append({
            "leg": name,
            "start": cmp["start"].date().isoformat(),
            "n_days": cmp["hy"]["n_days"],
            "w_mean": cmp["w_mean"],
            "r2": cmp["r2"],
            "residual_ann": cmp["residual_ann"],
            "sharpe_hy": cmp["hy"]["sharpe"],
            "sharpe_repl": cmp["repl"]["sharpe"],
            "excess_sharpe_gap": cmp["excess_sharpe_gap"],
            "t_gap": cmp["t_gap"],
            "dd_repl_abs": cmp["dd_repl_abs"],
        })
    return pd.DataFrame(rows).set_index("leg")


def window_sweep(
    hy: pd.Series, equity: pd.Series, duration: pd.Series, cash: pd.Series,
    grid=(126, 252, 504, 756), cost_bps: float = 2.0,
) -> pd.DataFrame:
    """Sweep the beta estimation window — the one design choice with real bite.

    A longer window is a steadier weight but starts the out-of-sample record later (and
    so drops more of the 2008 crash), which is exactly the trade-off a reader should see.
    """
    rows = []
    for w in grid:
        cmp = compare(hy, equity, duration, cash, window=w, cost_bps=cost_bps)
        rows.append({
            "window": w,
            "start": cmp["start"].date().isoformat(),
            "n_days": cmp["hy"]["n_days"],
            "w_mean": cmp["w_mean"],
            "sharpe_hy": cmp["hy"]["sharpe"],
            "sharpe_repl": cmp["repl"]["sharpe"],
            "excess_sharpe_gap": cmp["excess_sharpe_gap"],
            "t_gap": cmp["t_gap"],
            "r2": cmp["r2"],
        })
    return pd.DataFrame(rows).set_index("window")


# --------------------------------------------------------------------------- #
# Synthetic control (machinery proof — never supports a real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, window: int = 252, cost_bps: float = 0.0) -> dict:
    """Run the whole race on a synthetic (fund, equity, duration, cash) tape.

    On the planted world (``signal_strength = 1``) the harness must recover ``w_true``,
    find a negative residual and hand the replication the higher Sharpe. On the null
    (``signal_strength = 0``) it must recover ``w_true`` just as well and report a gap
    indistinguishable from zero.
    """
    cmp = compare(prices["fund"], prices["equity"], prices["duration"], prices["cash"],
                  window=window, cost_bps=cost_bps)
    return {
        "w_mean": cmp["w_mean"],
        "r2": cmp["r2"],
        "residual_ann": cmp["residual_ann"],
        "t_residual": cmp["t_residual"],
        "excess_sharpe_gap": cmp["excess_sharpe_gap"],
        "t_gap": cmp["t_gap"],
        "sharpe_hy": cmp["hy"]["sharpe"],
        "sharpe_repl": cmp["repl"]["sharpe"],
        "n_days": cmp["hy"]["n_days"],
    }
