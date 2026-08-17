"""Estimator + inference for Study 916 — Withholding Drag.

The measurement
---------------
For any fund with a total-return close ``TR`` and a price-only close ``PX`` (both
split-adjusted, only the first dividend-adjusted), the daily difference

    d_t = TR_t / TR_{t-1} - PX_t / PX_{t-1}

is zero on ordinary days and equals ``distribution_t / PX_{t-1}`` on ex-dates. Summed
over a calendar year it is the fund's **realised distribution yield**: the income that
actually reached a US holder, already net of foreign withholding *and* of the fund's
own expenses. That number is on the tape, exactly.

What is *not* on the tape is the **gross** dividend the underlying companies declared,
so the withholding rate itself is never observed. Everything this module reports about
"drag" therefore comes in two clearly separated flavours:

1. **Measured** — realised net income yields, the gap between a broad developed-ex-US
   fund and a single-country blend of the same market, HAC *t*, block-bootstrap CI,
   era cut, and the total-return race between the two wrappers.
2. **Inferred** — the implied gross yield and withholding drag under an *assumed*
   effective withholding rate. Swept across a grid; never asserted at a point.

All three non-tape inputs get their own grid: the withholding rate
(:func:`implied_withholding`), the blend country weights (:func:`blend_weight_sweep`)
and the expense ratios (:func:`fee_gap_sweep` — the fees are *today's*, applied to a
19-year window, and that anachronism runs in the hypothesis's favour).

The ruler is calibrated on the tape before it is used: EFA and IEFA track nearly the
same market from the same issuer, so their realised income-yield gap should reproduce
their **known 26 bp fee difference**. If the estimator recovers that, it can be trusted
on the harder comparison.

Conventions
-----------
* **One execution lag.** The only traded object here is the single-country blend: its
  target weights are known at the close of month-end ``t`` and the rebalance is
  executed at ``t+1``. Nothing in the study forecasts anything, so there is no second
  lag to hide.
* **Costs** are one-way × NAV, charged on the rebalance turnover only.
* **No short leg** anywhere, so no borrow cost applies (stated, not assumed away).
* Sharpe races are **excess-of-cash vs excess-of-cash** (minus BIL's total return),
  reported gross and net.
* Everything is **total return** unless a series is explicitly named price-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
BP = 1e4


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
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def auto_lags(n: int) -> int:
    """Newey-West lag rule of thumb, floor(4 (n/100)^(2/9))."""
    return max(1, int(np.floor(4.0 * (max(n, 1) / 100.0) ** (2.0 / 9.0))))


def hac_t(x) -> float:
    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    return newey_west_t(arr, lags=auto_lags(len(arr)))


def block_bootstrap_mean_ci(
    x,
    n_boot: int = 2000,
    block: int = 63,
    seed: int = 916,
    alpha: float = 0.05,
    scale: float = TRADING_DAYS * BP,
) -> dict:
    """Circular block-bootstrap CI for the mean of a daily series, in bp/yr.

    A 63-day (one quarter) block spans a full ex-dividend cycle, so each resampled
    block carries whole payment events rather than slicing one in half. ``scale``
    converts a daily mean into annualised basis points.
    """
    r = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": int(n), "block": block}
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
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((boots < 0).mean()), "n_obs": int(n),
            "block": int(block), "n_boot": int(n_boot)}


def bootstrap_sharpe_ci(excess, n_boot: int = 2000, block: int = 21,
                        seed: int = 916, alpha: float = 0.05) -> dict:
    """Circular block-bootstrap CI for the annualised Sharpe of an excess series."""
    r = np.asarray(pd.Series(excess).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"sharpe": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": int(n)}
    ann = np.sqrt(TRADING_DAYS)
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
            "block": int(block)}


def summary(returns) -> dict:
    """Annualised stats for a daily simple-return series (pass excess for excess Sharpe)."""
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    sd = r.std(ddof=1)
    wealth = (1.0 + r).cumprod()
    years = n / TRADING_DAYS
    return {
        "n_days": int(n),
        "sharpe": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan"),
        "vol_ann": float(sd * np.sqrt(TRADING_DAYS)),
        "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and wealth.iloc[-1] > 0 else float("nan"),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "mean_daily_bps": float(r.mean() * BP),
        "tstat": hac_t(r),
    }


# --------------------------------------------------------------------------- #
# The measurement — realised (net) distribution yield
# --------------------------------------------------------------------------- #
def income_yield_daily(tr: pd.Series, px: pd.Series) -> pd.Series:
    """Daily realised distribution accrual: ``r_total_return - r_price_only``.

    Zero on ordinary sessions, ``distribution / previous close`` on ex-dates. Both
    inputs must be split-adjusted; only ``tr`` is dividend-adjusted. Negative dust of
    order 1e-6 is Yahoo's rounding in the adjustment factor and is left in — it is
    part of the honest measurement error, not something to clip away.
    """
    common = tr.index.intersection(px.index)
    a = tr.reindex(common).sort_index().pct_change()
    b = px.reindex(common).sort_index().pct_change()
    return (a - b).dropna().rename("income")


def realised_yield_bp(tr: pd.Series, px: pd.Series) -> float:
    """Annualised realised distribution yield, bp/yr (mean daily accrual x 252)."""
    d = income_yield_daily(tr, px)
    return float(d.mean() * TRADING_DAYS * BP)


def annual_income_yield(tr: pd.Series, px: pd.Series,
                        min_days: int = 240) -> pd.DataFrame:
    """Per-calendar-year realised distribution yield (%), complete years only.

    Years with fewer than ``min_days`` sessions (the first and the as-of year) are
    dropped, so no partial year is compared with a full one.
    """
    d = income_yield_daily(tr, px)
    g = d.groupby(d.index.year)
    rows = []
    for y, s in g:
        if len(s) < min_days:
            continue
        rows.append({"year": int(y), "yield_pct": float(s.sum() * 100), "n_days": int(len(s))})
    return pd.DataFrame(rows).set_index("year") if rows else pd.DataFrame(
        columns=["yield_pct", "n_days"]
    )


# --------------------------------------------------------------------------- #
# The single-country blend (the benchmark wrapper)
# --------------------------------------------------------------------------- #
def blend_returns(returns: pd.DataFrame, weights: dict,
                  cost_bps: float = 0.0) -> pd.Series:
    """Monthly-rebalanced weighted portfolio return from a frame of daily returns.

    Weights drift with the constituents inside a month; at the first session of each
    new month the book is reset to the targets. The targets are known at the close of
    the previous month (they are static), and the trade is executed on the following
    session — the study's single execution lag. Turnover is charged at ``cost_bps``
    one-way × NAV.
    """
    cols = [c for c in weights if c in returns.columns]
    r = returns[cols].dropna()
    if r.empty:
        return pd.Series(dtype=float)
    w_target = np.array([weights[c] for c in cols], dtype=float)
    w_target = w_target / w_target.sum()

    months = r.index.to_period("M")
    new_month = np.r_[True, months[1:] != months[:-1]]

    w = w_target.copy()
    out = np.empty(len(r))
    vals = r.to_numpy()
    for i in range(len(r)):
        if new_month[i] and i > 0:
            turnover = np.abs(w_target - w).sum()
            cost = turnover * cost_bps * 1e-4
            w = w_target.copy()
        else:
            cost = 0.0
        port = float(w @ vals[i])
        out[i] = port - cost
        grown = w * (1.0 + vals[i])
        tot = grown.sum()
        w = grown / tot if tot > 0 else w_target.copy()
    return pd.Series(out, index=r.index, name="blend")


def blend_index(tr: pd.DataFrame, px: pd.DataFrame, weights: dict,
                cost_bps: float = 0.0) -> tuple[pd.Series, pd.Series]:
    """Build blended total-return and price-only *index levels* from constituents.

    Both legs use the same monthly-rebalanced weight path, so the difference between
    them is the blend's realised distribution yield — exactly the quantity the single
    funds' own legs measure.
    """
    common = tr.index.intersection(px.index)
    r_tr = tr.reindex(common).pct_change().dropna()
    r_px = px.reindex(common).pct_change().dropna()
    idx = r_tr.index.intersection(r_px.index)
    b_tr = blend_returns(r_tr.loc[idx], weights, cost_bps=cost_bps)
    b_px = blend_returns(r_px.loc[idx], weights, cost_bps=cost_bps)
    lvl_tr = 100.0 * (1.0 + b_tr).cumprod()
    lvl_px = 100.0 * (1.0 + b_px).cumprod()
    return lvl_tr.rename("blend_tr"), lvl_px.rename("blend_px")


# --------------------------------------------------------------------------- #
# The headline gap
# --------------------------------------------------------------------------- #
def yield_gap(
    fund_tr: pd.Series, fund_px: pd.Series,
    bench_tr: pd.Series, bench_px: pd.Series,
    fund_expense: float = 0.0, bench_expense: float = 0.0,
    seed: int = 916, n_boot: int = 2000, block: int = 63,
) -> dict:
    """Realised income-yield gap (benchmark − fund) in bp/yr, with HAC *t* and CI.

    ``*_expense`` are the funds' prospectus expense ratios (an ASSUMPTION, not tape).
    Distributions are paid net of expenses, so a fee difference shows up inside the
    raw gap; ``gap_fee_adj_bp`` adds the fee difference back, isolating the part of
    the gap that fees cannot explain. Both numbers are reported.
    """
    d_fund = income_yield_daily(fund_tr, fund_px)
    d_bench = income_yield_daily(bench_tr, bench_px)
    idx = d_fund.index.intersection(d_bench.index)
    diff = (d_bench.loc[idx] - d_fund.loc[idx]).rename("gap")

    raw_bp = float(diff.mean() * TRADING_DAYS * BP)
    fee_gap_bp = (bench_expense - fund_expense) * BP
    boot = block_bootstrap_mean_ci(diff, n_boot=n_boot, block=block, seed=seed)
    # The fee adjustment is a constant, so it shifts the mean but not the variance:
    # re-centre the daily difference and re-run the same HAC t on the residual gap.
    adj = diff + fee_gap_bp / (TRADING_DAYS * BP)
    boot_adj = block_bootstrap_mean_ci(adj, n_boot=n_boot, block=block, seed=seed)

    return {
        "start": str(idx[0].date()) if len(idx) else None,
        "end": str(idx[-1].date()) if len(idx) else None,
        "n_days": int(len(idx)),
        "fund_yield_bp": float(d_fund.loc[idx].mean() * TRADING_DAYS * BP),
        "bench_yield_bp": float(d_bench.loc[idx].mean() * TRADING_DAYS * BP),
        "gap_bp": raw_bp,
        "fee_gap_bp": fee_gap_bp,
        "gap_fee_adj_bp": raw_bp + fee_gap_bp,
        "t_hac": hac_t(diff),
        "ci_low_bp": boot["ci_low"], "ci_high_bp": boot["ci_high"],
        "t_hac_fee_adj": hac_t(adj),
        "ci_low_fee_adj_bp": boot_adj["ci_low"], "ci_high_fee_adj_bp": boot_adj["ci_high"],
        "frac_negative": boot["frac_negative"],
        "diff": diff,
    }


def distribution_cross_check(tr: pd.Series, px: pd.Series,
                             dividend: pd.Series) -> dict:
    """Validate the ruler against the fund's *actual* per-share cash distributions.

    ``dividend`` is the declared cash amount from the price-only tape. Its yield,
    ``dividend_t / PX_{t-1}``, is an independent measurement of the same quantity the
    total-return-minus-price-only difference recovers. If the two agree to a basis
    point the estimator has no systematic bias and the residual is Yahoo's rounding.
    """
    d_meas = income_yield_daily(tr, px)
    idx = d_meas.index
    d_cash = (dividend.reindex(px.index) / px.shift(1)).reindex(idx).fillna(0.0)
    a = float(d_meas.mean() * TRADING_DAYS * BP)
    b = float(d_cash.mean() * TRADING_DAYS * BP)
    return {"differenced_bp": a, "cash_dividend_bp": b, "residual_bp": a - b,
            "n_days": int(len(idx)), "n_ex_dates": int((d_cash > 0).sum())}


def implied_withholding(net_yield_bp: float, w_grid=(0.05, 0.10, 0.12, 0.15, 0.20, 0.25)) -> list:
    """INFERENCE: implied gross yield and withholding drag for a grid of rates.

    Given a *measured* net distribution yield and an *assumed* effective withholding
    rate ``w``, the gross yield the underlying companies declared would have been
    ``net / (1 - w)`` and the drag ``net * w / (1 - w)`` bp/yr. The tape cannot
    identify ``w``; this is arithmetic conditional on an assumption, and is reported
    as a sweep so the reader sees how much the headline depends on it.
    """
    out = []
    for w in w_grid:
        gross = net_yield_bp / (1.0 - w)
        out.append({"w": float(w), "gross_bp": float(gross),
                    "drag_bp": float(gross - net_yield_bp)})
    return out


def fee_calibration(tr: pd.DataFrame, px: pd.DataFrame, a: str, b: str,
                    expense: dict, seed: int = 916) -> dict:
    """Calibrate the ruler on a pair whose *true* yield difference is (mostly) known.

    ``a`` and ``b`` should track near-identical markets from the same issuer, so the
    only large, documented difference between their realised distribution yields is
    the fee gap. Recovering that known gap validates the estimator on the real tape
    before it is pointed at the harder question.
    """
    common = tr.index.intersection(px.index)
    sub_tr, sub_px = tr.loc[common], px.loc[common]
    g = yield_gap(sub_tr[a], sub_px[a], sub_tr[b], sub_px[b],
                  fund_expense=expense.get(a, 0.0), bench_expense=expense.get(b, 0.0),
                  seed=seed)
    g["expected_gap_bp"] = (expense.get(a, 0.0) - expense.get(b, 0.0)) * BP
    g["residual_bp"] = g["gap_bp"] - g["expected_gap_bp"]
    return g


# --------------------------------------------------------------------------- #
# The total-return race — is the gap bankable?
# --------------------------------------------------------------------------- #
def total_return_race(fund_tr: pd.Series, bench_tr: pd.Series, cash_tr: pd.Series,
                      cost_bps: float = 0.0, seed: int = 916) -> dict:
    """Excess-of-cash race: the broad fund vs the single-country blend.

    Both legs are measured excess of BIL's own total return. ``cost_bps`` is charged
    on the blend's monthly rebalance turnover only (the broad fund is buy-and-hold, so
    it pays nothing after the entry trade). No short leg → no borrow. Returns per-leg
    summaries, the Sharpe advantage of the blend over the fund, and the HAC *t* on the
    daily return difference.
    """
    idx = fund_tr.index.intersection(bench_tr.index).intersection(cash_tr.index)
    r_f = fund_tr.reindex(idx).pct_change()
    r_b = bench_tr.reindex(idx).pct_change()
    r_c = cash_tr.reindex(idx).pct_change()
    # bench_tr arrives as a level already built at the requested cost; charge nothing twice.
    e_f = (r_f - r_c).dropna().rename("e_fund")
    e_b = (r_b - r_c).dropna().rename("e_bench")
    j = e_f.index.intersection(e_b.index)
    e_f, e_b = e_f.loc[j], e_b.loc[j]
    diff = (e_b - e_f).rename("diff")
    s_f, s_b = summary(e_f), summary(e_b)
    return {
        "fund": s_f, "bench": s_b,
        "sharpe_adv": s_b["sharpe"] - s_f["sharpe"],
        "ret_adv_bp": float(diff.mean() * TRADING_DAYS * BP),
        "t_diff": hac_t(diff),
        "cost_bps": cost_bps,
        "e_fund": e_f, "e_bench": e_b, "diff": diff,
        "start": str(j[0].date()) if len(j) else None,
        "end": str(j[-1].date()) if len(j) else None,
    }


# --------------------------------------------------------------------------- #
# Robustness — eras, cost sweep, weight sweep
# --------------------------------------------------------------------------- #
def era_cut(fund_tr, fund_px, bench_tr, bench_px, split: str = "2017-01-01",
            fund_expense: float = 0.0, bench_expense: float = 0.0) -> dict:
    """Split the sample and re-measure the income-yield gap on each half."""
    out = {}
    for tag, sl in [("early", slice(None, split)), ("late", slice(split, None))]:
        f_tr, f_px = fund_tr.loc[sl], fund_px.loc[sl]
        b_tr, b_px = bench_tr.loc[sl], bench_px.loc[sl]
        if min(len(f_tr), len(b_tr)) < 300:
            out[tag] = None
            continue
        g = yield_gap(f_tr, f_px, b_tr, b_px, fund_expense=fund_expense,
                      bench_expense=bench_expense, n_boot=800)
        out[tag] = {k: g[k] for k in
                    ("n_days", "fund_yield_bp", "bench_yield_bp", "gap_bp",
                     "gap_fee_adj_bp", "t_hac", "t_hac_fee_adj",
                     "ci_low_bp", "ci_high_bp")}
    return out


def cost_sweep(fund_tr: pd.Series, tr: pd.DataFrame, px: pd.DataFrame,
               cash_tr: pd.Series, weights: dict,
               cost_grid=(0.0, 2.0, 5.0, 10.0, 25.0)) -> list:
    """Re-run the total-return race with the blend rebuilt at each one-way cost."""
    rows = []
    for c in cost_grid:
        b_tr, _ = blend_index(tr, px, weights, cost_bps=c)
        race = total_return_race(fund_tr, b_tr, cash_tr, cost_bps=c)
        rows.append({"cost_bps": c, "sharpe_fund": race["fund"]["sharpe"],
                     "sharpe_bench": race["bench"]["sharpe"],
                     "sharpe_adv": race["sharpe_adv"],
                     "ret_adv_bp": race["ret_adv_bp"], "t_diff": race["t_diff"]})
    return rows


def fee_gap_sweep(gap_result: dict,
                  fee_gap_grid_bp=(35.0, 40.0, 44.0, 47.0, 51.0, 55.0),
                  seed: int = 916, n_boot: int = 2000) -> list:
    """Sensitivity of the fee-adjusted gap to the expense-ratio ASSUMPTION.

    ``EXPENSE_RATIO`` holds *today's* (2026) fact-sheet fees, but they are applied to a
    19-year window and ETF fees have fallen over it — VEA's expense ratio was well above
    10 bp at its 2007 launch against 3 bp today, while the iShares single-country funds
    have barely moved. The time-averaged fee gap is therefore *smaller* than the 47 bp
    the headline adds back, so the headline fee adjustment is an **upper bound** and the
    fee-adjusted gap it produces is an upper bound too (i.e. it flatters the withholding
    hypothesis, which needs a positive gap).

    This re-centres the same daily difference on a grid of assumed fee gaps — the shift
    is a constant, so only the mean moves — and re-reports the HAC *t* and bootstrap CI
    at each point. Takes the dict returned by :func:`yield_gap` (it needs its ``diff``).
    """
    diff = gap_result["diff"]
    rows = []
    for f in fee_gap_grid_bp:
        adj = diff + float(f) / (TRADING_DAYS * BP)
        boot = block_bootstrap_mean_ci(adj, n_boot=n_boot, seed=seed)
        rows.append({
            "fee_gap_bp": float(f),
            "gap_fee_adj_bp": float(gap_result["gap_bp"] + float(f)),
            "t_hac": hac_t(adj),
            "ci_low_bp": boot["ci_low"], "ci_high_bp": boot["ci_high"],
        })
    return rows


def blend_weight_sweep(fund_tr: pd.Series, fund_px: pd.Series,
                       tr: pd.DataFrame, px: pd.DataFrame,
                       weight_sets: dict, fund_expense: float = 0.0,
                       bench_expense: float = 0.0) -> list:
    """How much does the headline gap depend on the assumed blend weights?"""
    rows = []
    for name, w in weight_sets.items():
        b_tr, b_px = blend_index(tr, px, w, cost_bps=0.0)
        g = yield_gap(fund_tr, fund_px, b_tr, b_px, fund_expense=fund_expense,
                      bench_expense=bench_expense, n_boot=500)
        rows.append({"weights": name, "gap_bp": g["gap_bp"],
                     "gap_fee_adj_bp": g["gap_fee_adj_bp"], "t_hac": g["t_hac"],
                     "bench_yield_bp": g["bench_yield_bp"]})
    return rows


# --------------------------------------------------------------------------- #
# Synthetic control (machinery proof — never supports a real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(tr: pd.DataFrame, px: pd.DataFrame, truth: dict,
                     n_boot: int = 600) -> dict:
    """Run the income-yield gap estimator on a synthetic panel with a known answer.

    On the planted world the estimator must recover ``truth['true_gap_bp']``; on the
    null (``signal_strength = 0``) it must report ~0 bp/yr with a CI covering zero.
    """
    bench_cols = truth["bench_cols"]
    w = {c: 1.0 / len(bench_cols) for c in bench_cols}
    b_tr, b_px = blend_index(tr[bench_cols], px[bench_cols], w, cost_bps=0.0)
    g = yield_gap(tr["broad"], px["broad"], b_tr, b_px, n_boot=n_boot)
    return {
        "true_gap_bp": truth["true_gap_bp"],
        "measured_gap_bp": g["gap_bp"],
        "t_hac": g["t_hac"],
        "ci_low_bp": g["ci_low_bp"], "ci_high_bp": g["ci_high_bp"],
        "fund_yield_bp": g["fund_yield_bp"],
        "bench_yield_bp": g["bench_yield_bp"],
        "n_days": g["n_days"],
    }
