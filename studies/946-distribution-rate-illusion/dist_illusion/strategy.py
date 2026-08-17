"""Strategy + inference for Study 946 — Distribution is not Return.

The sort: at the close of month ``t`` rank every fund with a full twelve months of history
by its **trailing-12-month distribution rate** (reconstructed from the total-return /
price-only gap; a PROXY, see ``data``). Buy an equal-weight top tercile, sell an
equal-weight bottom tercile, and hold both through month ``t+1``. **Exactly one execution
lag**: the ranking variable uses data through the close of ``t``, the return earned is
month ``t+1``'s — nothing else is lagged anywhere in this module.

The same sort is scored against three different left-hand sides, which is the entire point
of the study:

1. ``total`` — the fund's **total** return. This is the claim: does a fat payout predict a
   fat return? If distributions were free money it would.
2. ``price`` — the fund's **price-only** return. This is the mechanism: the payout leaves
   the NAV, so a fat payout should predict a *falling quoted price*.
3. ``dist``  — next month's payout itself. This is the sanity leg: if the ranking variable
   cannot even forecast the next distribution, nothing else here means anything.

Everything is measured on **simple monthly returns**. Long-only legs are raced
**excess-of-cash** (minus BIL) against SPY excess-of-cash; the high-minus-low leg is
self-financing, so it carries no cash term but *does* pay a borrow fee on its short side
(swept, since the fee is an ASSUMPTION and not tape). Costs are one-way bps × NAV traded,
charged on the realised name turnover of each leg.

Inference: Newey-West HAC *t* on every mean, a moving-block bootstrap CI, a CAPM control
(``r = α + β·r_SPY`` in excess-of-cash space) because the high-payout cohort is structurally
lower-beta, an era cut, and a cost / borrow sweep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def newey_west_t(x, lags: int = 6) -> float:
    """HAC (Newey-West, Bartlett kernel) *t* of mean(x) vs 0. 6 lags = half a year."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * (float(u[l:] @ u[:-l]) / n)
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def one_sample_t(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(x.size)
    return float(x.mean() / se) if se > 0 else float("nan")


def block_bootstrap_ci(x, n_boot: int = 2000, block: int = 6, seed: int = 946,
                       alpha: float = 0.05) -> dict:
    """Moving-block bootstrap CI for the mean of a monthly series, reported in bps.

    Six-month blocks preserve the autocorrelation a rolling-12-month ranking variable
    inevitably induces. Returns the point mean, the percentile interval and the share of
    resamples above zero.
    """
    r = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"mean_bps": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "frac_positive": float("nan"), "n_obs": n}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    starts = rng.integers(0, n, (n_boot, n_blocks))
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    means = r[idx.reshape(n_boot, -1)[:, :n]].mean(axis=1) * 1e4
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean_bps": float(r.mean() * 1e4), "ci_low": float(lo), "ci_high": float(hi),
            "frac_positive": float((means > 0).mean()), "n_obs": int(n),
            "n_boot": int(n_boot), "block": int(block)}


def capm(y, mkt_excess, lags: int = 6) -> dict:
    """OLS ``y = α + β·mkt_excess`` with a HAC *t* on α (both sides excess-of-cash).

    The high-payout cohort is structurally lower-beta than the dividend-equity cohort, so a
    raw high-minus-low return in a bull market is a beta bet unless β is taken out. This is
    the control that decides whether anything survives.
    """
    y = np.asarray(pd.Series(y).astype(float))
    m = np.asarray(pd.Series(mkt_excess).astype(float))
    ok = np.isfinite(y) & np.isfinite(m)
    y, m = y[ok], m[ok]
    if y.size < 5:
        return {"alpha_bps": float("nan"), "beta": float("nan"),
                "t_alpha": float("nan"), "r2": float("nan"), "n": int(y.size)}
    X = np.column_stack([np.ones_like(m), m])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ b
    r2 = 1.0 - resid.var() / y.var() if y.var() > 0 else float("nan")
    return {"alpha_bps": float(b[0] * 1e4), "beta": float(b[1]),
            "t_alpha": newey_west_t(resid + b[0], lags=lags), "r2": float(r2),
            "n": int(y.size)}


def summary(returns, periods_per_year: int = MONTHS) -> dict:
    """Headline stats for a monthly simple-return series (pass an *excess* series for
    an excess Sharpe): mean in bps, annualised Sharpe / vol / CAGR, max drawdown, HAC *t*."""
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n < 3:
        return {"n_months": n, "mean_bps": float("nan"), "sharpe": float("nan"),
                "vol_ann": float("nan"), "cagr": float("nan"),
                "max_drawdown": float("nan"), "tstat": float("nan")}
    mu, sd = r.mean(), r.std(ddof=1)
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    return {
        "n_months": int(n),
        "mean_bps": float(mu * 1e4),
        "sharpe": float(mu / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan"),
        "vol_ann": float(sd * np.sqrt(periods_per_year)),
        "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if wealth.iloc[-1] > 0 else float("nan"),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "tstat": newey_west_t(r.to_numpy()),
    }


# --------------------------------------------------------------------------- #
# Fama-MacBeth — the cross-sectional slope, month by month
# --------------------------------------------------------------------------- #
def fama_macbeth(panel: dict, target: str = "total", min_funds: int = 6,
                 lags: int = 6) -> dict:
    """Month-by-month cross-sectional slope of ``target`` on the trailing payout rank.

    Each month ``t`` the ranking variable ``dist_rate`` is z-scored **within the
    cross-section** (so the slope reads as "bps of next-month return per 1 sd of trailing
    distribution rate") and regressed on month ``t+1``'s ``target``. The Fama-MacBeth
    statistic is the time-series mean of those slopes with a HAC *t*.

    The one execution lag lives here and nowhere else: ``dist_rate`` at ``t``, ``target``
    at ``t+1``.
    """
    D = panel["dist_rate"]
    Y = panel[target]
    idx = list(D.index)
    dates, slopes = [], []
    for j in range(len(idx) - 1):
        t, nxt = idx[j], idx[j + 1]
        x = D.loc[t].to_numpy(dtype=float)
        y = Y.loc[nxt].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < min_funds:
            continue
        xs, ys = x[ok], y[ok]
        sd = xs.std(ddof=1)
        if sd <= 0:
            continue
        z = (xs - xs.mean()) / sd
        slopes.append(float(np.dot(z - z.mean(), ys) / np.dot(z - z.mean(), z)))
        dates.append(nxt)
    s = pd.Series(slopes, index=pd.DatetimeIndex(dates, name="date"), name=f"slope_{target}")
    return {
        "slopes": s,
        "n_months": int(len(s)),
        "mean_bps": float(s.mean() * 1e4) if len(s) else float("nan"),
        "tstat": newey_west_t(s.to_numpy(), lags=lags) if len(s) else float("nan"),
        "target": target,
    }


# --------------------------------------------------------------------------- #
# The tercile sort — high payout minus low payout
# --------------------------------------------------------------------------- #
def sorted_legs(panel: dict, frac: float = 1.0 / 3.0, min_funds: int = 6) -> pd.DataFrame:
    """Equal-weight high-payout and low-payout legs, held for the month after the ranking.

    Returns one row per held month with, for each leg, the realised **total** return
    (``hi``/``lo``), the **price-only** return (``hi_p``/``lo_p``), the realised
    **distribution** (``hi_d``/``lo_d``), the leg's average trailing payout rate at
    formation (``dhi``/``dlo``), the cross-section width (``n``, ``k``) and the leg's
    one-way name turnover (``to_hi``/``to_lo``). The ``hml*`` columns are high minus low.
    """
    D, R, P, DI = panel["dist_rate"], panel["total"], panel["price"], panel["dist"]
    names = np.asarray(D.columns)
    idx = list(D.index)
    rows = []
    prev_hi = prev_lo = None
    for j in range(len(idx) - 1):
        t, nxt = idx[j], idx[j + 1]
        x = D.loc[t].to_numpy(dtype=float)
        y = R.loc[nxt].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < min_funds:
            continue
        pos = np.flatnonzero(ok)
        order = pos[np.argsort(x[pos], kind="stable")]
        k = max(2, int(round(pos.size * frac)))
        lo_i, hi_i = order[:k], order[-k:]
        p = P.loc[nxt].to_numpy(dtype=float)
        di = DI.loc[nxt].to_numpy(dtype=float)
        hi_set, lo_set = frozenset(names[hi_i]), frozenset(names[lo_i])
        rows.append({
            "date": nxt,
            "hi": float(np.nanmean(y[hi_i])), "lo": float(np.nanmean(y[lo_i])),
            "hi_p": float(np.nanmean(p[hi_i])), "lo_p": float(np.nanmean(p[lo_i])),
            "hi_d": float(np.nanmean(di[hi_i])), "lo_d": float(np.nanmean(di[lo_i])),
            "dhi": float(np.nanmean(x[hi_i])), "dlo": float(np.nanmean(x[lo_i])),
            "n": int(pos.size), "k": int(k),
            "to_hi": 0.0 if prev_hi is None else len(hi_set - prev_hi) / k,
            "to_lo": 0.0 if prev_lo is None else len(lo_set - prev_lo) / k,
        })
        prev_hi, prev_lo = hi_set, lo_set
    cols = ["date", "hi", "lo", "hi_p", "lo_p", "hi_d", "lo_d", "dhi", "dlo",
            "n", "k", "to_hi", "to_lo"]
    legs = pd.DataFrame(rows, columns=cols).set_index("date")
    legs["hml"] = legs["hi"] - legs["lo"]
    legs["hml_p"] = legs["hi_p"] - legs["lo_p"]
    legs["hml_d"] = legs["hi_d"] - legs["lo_d"]
    return legs


def giveback_ratio(legs: pd.DataFrame) -> float:
    """How much of the extra payout comes straight back as extra NAV erosion.

    ``−mean(hml_p) / mean(hml_d)``. A value of 1.0 is the pure return-of-capital world:
    every extra basis point distributed is an extra basis point off the quoted price, and
    total return is untouched. Above 1.0 the high-payout leg erodes *more* than it pays.
    """
    denom = float(legs["hml_d"].mean())
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return float(-legs["hml_p"].mean() / denom)


# --------------------------------------------------------------------------- #
# The race and the costed net
# --------------------------------------------------------------------------- #
def race(panel: dict, legs: pd.DataFrame) -> dict:
    """Excess-of-cash summaries for the high leg, the low leg and the benchmark.

    Both long legs and the benchmark are measured minus the cash leg (BIL total return),
    so the lower-beta high-payout basket cannot look good just by carrying less risk — or
    bad just by carrying less market. Also returns the CAPM read of each leg and of the
    self-financing high-minus-low spread.
    """
    cash = panel["cash"].reindex(legs.index)
    bench_ex = panel["bench"].reindex(legs.index) - cash
    out = {
        "hi": summary(legs["hi"] - cash),
        "lo": summary(legs["lo"] - cash),
        "bench": summary(bench_ex),
        "hi_abs": summary(legs["hi"]),
        "lo_abs": summary(legs["lo"]),
        "bench_abs": summary(panel["bench"].reindex(legs.index)),
        "hml": summary(legs["hml"]),
        "hml_price": summary(legs["hml_p"]),
        "capm_hi": capm(legs["hi"] - cash, bench_ex),
        "capm_lo": capm(legs["lo"] - cash, bench_ex),
        "capm_hml": capm(legs["hml"], bench_ex),
    }
    out["sharpe_hi_minus_bench"] = out["hi"]["sharpe"] - out["bench"]["sharpe"]
    out["sharpe_hi_minus_lo"] = out["hi"]["sharpe"] - out["lo"]["sharpe"]
    return out


def net_hml(legs: pd.DataFrame, cost_bps: float = 5.0,
            borrow_bps_annual: float = 0.0) -> pd.Series:
    """The high-minus-low spread net of trading cost and short-leg borrow.

    Cost is ``cost_bps`` one-way × the NAV actually traded, charged on both legs' realised
    name turnover. Borrow is an **ASSUMPTION** (not tape): ``borrow_bps_annual / 12``
    charged every month on the full short notional. Both are swept in the results.
    """
    trade = (legs["to_hi"] + legs["to_lo"]) * (cost_bps * 1e-4)
    borrow = borrow_bps_annual * 1e-4 / MONTHS
    return (legs["hml"] - trade - borrow).rename("hml_net")


def cost_borrow_sweep(legs: pd.DataFrame,
                      cost_grid=(0.0, 5.0, 10.0, 25.0),
                      borrow_grid=(0.0, 50.0, 100.0, 200.0)) -> pd.DataFrame:
    """Grid of the net high-minus-low mean and HAC *t* over cost × borrow assumptions."""
    rows = []
    for c in cost_grid:
        for b in borrow_grid:
            net = net_hml(legs, cost_bps=c, borrow_bps_annual=b)
            rows.append({"cost_bps": c, "borrow_bps": b,
                         "mean_bps": float(net.mean() * 1e4),
                         "tstat": newey_west_t(net.to_numpy())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Era cut
# --------------------------------------------------------------------------- #
def era_cut(panel: dict, legs: pd.DataFrame, split: str = "2020-06-30") -> dict:
    """Split the held months at ``split`` and re-read the three slopes in each half.

    A mechanical identity (payout out, price down) should hold in *both* halves. An edge
    that lives in one half only is a regime artefact.
    """
    out = {}
    for tag, m in (("early", legs.index <= pd.Timestamp(split)),
                   ("late", legs.index > pd.Timestamp(split))):
        s = legs[m]
        if len(s) < 12:
            out[tag] = None
            continue
        cash = panel["cash"].reindex(s.index)
        bench_ex = panel["bench"].reindex(s.index) - cash
        out[tag] = {
            "n_months": int(len(s)),
            "start": str(s.index[0].date()), "end": str(s.index[-1].date()),
            "hml_bps": float(s["hml"].mean() * 1e4), "t_hml": newey_west_t(s["hml"].to_numpy()),
            "hml_p_bps": float(s["hml_p"].mean() * 1e4), "t_hml_p": newey_west_t(s["hml_p"].to_numpy()),
            "hml_d_bps": float(s["hml_d"].mean() * 1e4),
            "giveback": giveback_ratio(s),
            "alpha_bps": capm(s["hml"], bench_ex)["alpha_bps"],
            "t_alpha": capm(s["hml"], bench_ex)["t_alpha"],
        }
    return out


# --------------------------------------------------------------------------- #
# Synthetic control (machinery proof — never supports a real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: dict, frac: float = 1.0 / 3.0, min_funds: int = 4) -> dict:
    """Run the whole pipeline on a synthetic panel and report the three slopes.

    On the null (``signal_strength=0``) the total-return slope must be flat while the price
    slope is a big negative and the payout slope a big positive, with a give-back ratio near
    1. On the planted world the total-return slope must fire. This proves the estimator is
    unbiased — it never supports a real-tape verdict.
    """
    legs = sorted_legs(panel, frac=frac, min_funds=min_funds)
    cash = panel["cash"].reindex(legs.index)
    bench_ex = panel["bench"].reindex(legs.index) - cash
    fm_total = fama_macbeth(panel, "total", min_funds=min_funds)
    fm_price = fama_macbeth(panel, "price", min_funds=min_funds)
    fm_dist = fama_macbeth(panel, "dist", min_funds=min_funds)
    return {
        "n_months": int(len(legs)),
        "fm_total_bps": fm_total["mean_bps"], "t_total": fm_total["tstat"],
        "fm_price_bps": fm_price["mean_bps"], "t_price": fm_price["tstat"],
        "fm_dist_bps": fm_dist["mean_bps"], "t_dist": fm_dist["tstat"],
        "hml_bps": float(legs["hml"].mean() * 1e4), "t_hml": newey_west_t(legs["hml"].to_numpy()),
        "hml_p_bps": float(legs["hml_p"].mean() * 1e4), "t_hml_p": newey_west_t(legs["hml_p"].to_numpy()),
        "hml_d_bps": float(legs["hml_d"].mean() * 1e4),
        "giveback": giveback_ratio(legs),
        "capm_hml": capm(legs["hml"], bench_ex),
    }
