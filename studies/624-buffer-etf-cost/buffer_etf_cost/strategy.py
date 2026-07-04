"""Strategy + inference for Study 624 — Buffer ETFs: the Cost of Comfort.

The claim under test: defined-outcome ("buffer") ETFs deliver exactly the buffer they
promise — and charge you the upside cap plus a fat fee for insurance you could build
cheaper with a plain SPY/BIL mix. Three measurements:

* **Mechanical delivery check.** For each completed 12-month outcome period of each
  single-series fund, pair the fund's total return with SPY's PRICE return (the stated
  reference) and check the promised payoff: floor ≈ -fee inside the buffer, ref + buffer
  beyond it, capped upside above it. This is the "comfort delivered" leg.

* **The cost of comfort (Signal).** Race each fund against the *dumb mix it replaces*: a
  beta-matched w·SPY + (1-w)·BIL portfolio, rebalanced monthly with explicit one-way costs
  on the rebalancing turnover. The headline statistic is a Newey-West (HAC) t on the
  monthly mix-minus-fund return difference — serial correlation robust, excess-vs-excess
  by construction (both legs are total returns of fully-funded portfolios over the same
  months).

* **Third axis.** Did ANY buffer cohort beat its own dumb mix? Count winners across
  BUFR + the four Power Buffer vintages.

Execution convention (the one documented lag): the mix is formed at the first month-end
close after the fund's listing and holds fixed weights, rebalanced at each month-end
close to weights known in advance — no information is used before it exists (the beta is
a full-sample descriptive estimate; the w-grid robustness shows the verdict does not
depend on it). Costs: one-way bps × traded NAV fraction on every rebalance trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_MONTHS = 12.0


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
def nw_tstat(x: np.ndarray, lags: int = 6) -> float:
    """Newey-West (HAC) t of mean(x) vs 0 with Bartlett weights."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 8:
        return float("nan")
    mu = x.mean()
    e = x - mu
    s = float(e @ e) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        s += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(s / n)
    return float(mu / se) if se > 0 else float("nan")


def beta_vs(fund_m: pd.Series, spy_m: pd.Series, bil_m: pd.Series) -> float:
    """OLS beta of the fund's monthly excess return on SPY's monthly excess return."""
    df = pd.concat([fund_m, spy_m, bil_m], axis=1, keys=["f", "s", "b"]).dropna()
    fe = df["f"] - df["b"]
    se = df["s"] - df["b"]
    return float(np.cov(fe, se, ddof=1)[0, 1] / np.var(se, ddof=1))


# --------------------------------------------------------------------------- #
# The dumb mix
# --------------------------------------------------------------------------- #
def mix_returns(spy_m: pd.Series, bil_m: pd.Series, w: float,
                cost_bps: float = 2.0) -> pd.Series:
    """Monthly returns of a w·SPY + (1-w)·BIL mix, rebalanced monthly, net of costs.

    Each month the portfolio earns w·SPY + (1-w)·BIL; weights then drift with the two
    returns and are traded back to (w, 1-w) at the month-end close. The rebalance trades
    a NAV fraction |drift| out of one leg and into the other; each side pays ``cost_bps``
    one-way, so the month's cost is 2 · |drift| · cost_bps. (No shorts, no leverage.)
    """
    df = pd.concat([spy_m, bil_m], axis=1, keys=["s", "b"]).dropna()
    c = cost_bps / 1e4
    out = np.empty(len(df))
    for i, (rs, rb) in enumerate(zip(df["s"].values, df["b"].values)):
        gross = w * rs + (1.0 - w) * rb
        w_drift = w * (1.0 + rs) / (1.0 + gross)
        turn = abs(w_drift - w)
        out[i] = gross - 2.0 * turn * c
    return pd.Series(out, index=df.index)


def gap_stats(fund_m: pd.Series, mix_m: pd.Series, lags: int = 6) -> dict:
    """The cost of comfort: mix-minus-fund monthly gap, annualised, with a HAC t."""
    df = pd.concat([fund_m, mix_m], axis=1, keys=["f", "m"]).dropna()
    d = (df["m"] - df["f"]).values
    up = df["m"].values >= 0  # regime split on the mix's own sign
    return {
        "n_months": int(len(d)),
        "gap_bps_mo": float(d.mean() * 1e4),
        "gap_pp_yr": float(d.mean() * TRADING_MONTHS * 100.0),
        "hac_t": nw_tstat(d, lags=lags),
        "corr": float(np.corrcoef(df["f"], df["m"])[0, 1]),
        "gap_up_bps": float(d[up].mean() * 1e4) if up.any() else float("nan"),
        "gap_dn_bps": float(d[~up].mean() * 1e4) if (~up).any() else float("nan"),
        "start": str(df.index[0].date()), "end": str(df.index[-1].date()),
    }


def perf_stats(monthly: pd.Series, bil_m: pd.Series | None = None) -> dict:
    """CAGR / vol / max drawdown / Sharpe (excess vs BIL) from monthly total returns."""
    r = monthly.dropna()
    n = len(r)
    wealth = (1.0 + r).cumprod()
    cagr = float(wealth.iloc[-1] ** (TRADING_MONTHS / n) - 1.0)
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_MONTHS))
    dd = float((wealth / wealth.cummax() - 1.0).min())
    sharpe = float("nan")
    if bil_m is not None:
        ex = (r - bil_m.reindex(r.index)).dropna()
        sd = ex.std(ddof=1)
        if sd > 0:
            sharpe = float(ex.mean() / sd * np.sqrt(TRADING_MONTHS))
    return {"cagr_pct": cagr * 100, "vol_pct": vol * 100, "maxdd_pct": dd * 100,
            "sharpe": sharpe, "n_months": n}


# --------------------------------------------------------------------------- #
# Mechanical delivery check (outcome periods)
# --------------------------------------------------------------------------- #
def outcome_periods(fund_adj: pd.Series, spy_raw: pd.Series,
                    reset_month: int) -> pd.DataFrame:
    """Completed 12-month outcome periods: fund TOTAL return vs SPY PRICE return.

    The outcome period of a month-``m`` series runs from the last close before the first
    trading day of month ``m`` to the same close one year later (issuers reset on the
    first trading day). We take month-end closes of month ``m-1`` → month ``m-1`` + 12.
    Only periods fully inside both series are kept.
    """
    f_me = fund_adj.dropna().resample("ME").last()
    s_me = spy_raw.dropna().resample("ME").last()
    rows = []
    anchor_month = 12 if reset_month == 1 else reset_month - 1
    for y in range(f_me.index[0].year - 1, f_me.index[-1].year + 1):
        start = pd.Timestamp(y, anchor_month, 1) + pd.offsets.MonthEnd(0)
        end = start + pd.offsets.MonthEnd(12)
        if start < f_me.index[0] - pd.offsets.MonthEnd(1) or start not in f_me.index:
            continue
        if end not in f_me.index or start not in s_me.index or end not in s_me.index:
            continue
        fr = float(f_me[end] / f_me[start] - 1.0)
        sr = float(s_me[end] / s_me[start] - 1.0)
        rows.append({"period": f"{start.date()} -> {end.date()}",
                     "spy_price_ret_pct": sr * 100, "fund_ret_pct": fr * 100})
    return pd.DataFrame(rows)


def delivery_check(periods: pd.DataFrame, buffer_pct: float, er_pct: float,
                   floor_slack_pct: float = 2.0) -> dict:
    """Was the promised payoff delivered, period by period?

    * inside the buffer (ref in [-buffer, 0)): fund should lose no more than roughly the
      fee — floor = -(er + slack) where slack absorbs option pricing/NAV noise;
    * beyond the buffer (ref < -buffer): fund ≥ ref + buffer - er - slack;
    * up periods (ref ≥ 0): fund participates but is capped — the *realized cap* proxy is
      the fund's return in periods where it lagged the reference by ≥ 2 pp (cap binding).
    """
    ref = periods["spy_price_ret_pct"].values
    fnd = periods["fund_ret_pct"].values
    buf, er = buffer_pct, er_pct
    inside = (ref < 0) & (ref >= -buf)
    beyond = ref < -buf
    up = ref >= 0
    floor = -(er + floor_slack_pct)
    ok_inside = fnd[inside] >= floor
    ok_beyond = fnd[beyond] >= (ref[beyond] + buf - er - floor_slack_pct)
    capped = up & ((ref - fnd) >= 2.0)
    return {
        "n_periods": int(len(ref)),
        "n_up": int(up.sum()), "n_inside": int(inside.sum()), "n_beyond": int(beyond.sum()),
        "inside_honored": int(ok_inside.sum()), "beyond_honored": int(ok_beyond.sum()),
        "worst_inside_pct": float(fnd[inside].min()) if inside.any() else float("nan"),
        "n_capped": int(capped.sum()),
        "mean_capped_ret_pct": float(fnd[capped].mean()) if capped.any() else float("nan"),
        "mean_up_giveup_pp": float((ref[up] - fnd[up]).mean()) if up.any() else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Cost decomposition (descriptive)
# --------------------------------------------------------------------------- #
def cost_decomposition(gap_pp_yr: float, er_pct: float, beta: float,
                       spy_div_yield_pct: float) -> dict:
    """Split the annualised mix-minus-fund gap into named parts.

    fee (stated ER) + dividends forgone (the fund's options are on SPY's PRICE index, so
    it never earns the dividend the mix's SPY leg collects: ≈ beta × SPY dividend yield)
    + a residual (option structuring, cap asymmetry realisation, trading inside the
    wrapper). Descriptive arithmetic on the measured gap — no free parameters.
    """
    div_forgone = beta * spy_div_yield_pct
    return {"gap_pp_yr": gap_pp_yr, "fee_pp": er_pct, "div_forgone_pp": div_forgone,
            "residual_pp": gap_pp_yr - er_pct - div_forgone}
