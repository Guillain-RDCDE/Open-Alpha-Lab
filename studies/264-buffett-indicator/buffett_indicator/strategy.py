"""Strategy and honest controls -- Study 264 (Buffett-Indicator).

The Buffett claim, made precise: the level of market-cap-to-GDP at year-end
predicts the *following* year's equity return with a NEGATIVE slope -- high
valuation means low (or negative) forward returns ("leave the party"), low
valuation means high forward returns.

We test it four ways, each with its honest control:

1. **Predictive regression.**  OLS of next-year return on the indicator level,
   with a Newey-West (HAC) t-stat.  This is the spine: a real signal needs
   |t| >= 2 on the REAL tape.

2. **High vs low terciles.**  Split years into expensive / mid / cheap by the
   indicator and compare mean forward returns.  The Buffett claim is
   cheap >> expensive.

3. **A real-time timing strategy.**  Be long the S&P only when the indicator is
   below its *expanding* (point-in-time) median, else sit in cash; compare to
   buy-and-hold.  The expanding median is the crucial honesty fix: the famous
   "200% of GDP is expensive" thresholds are calibrated *with hindsight* on the
   whole sample, which is look-ahead.  In real time you never knew 70% would one
   day look cheap.

4. **In-sample vs out-of-sample slope.**  The full-sample regression slope is
   an upper bound; a walk-forward (expanding-window) version is the honest one.

**Price-only**: forward returns are the S&P 500 *price* index (no dividends);
the cash leg of the timing strategy earns zero (a conservative, not a generous,
assumption -- real T-bills would have helped the timing strategy in the high-rate
1970s-80s).  Gross of costs; turnover is low (one trade per regime flip) so costs
are a footnote, reported explicitly.

**No look-ahead**: the indicator at year-end Y forecasts the return in Y+1.  The
timing strategy at the start of Y+1 uses only the expanding history through Y.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# HAC summary statistics for a regression slope
# ---------------------------------------------------------------------------
def _hac_se(x: np.ndarray, resid: np.ndarray, lags: int | None = None) -> float:
    """Newey-West HAC standard error of an OLS slope (single regressor + const)."""
    n = len(x)
    xc = x - x.mean()
    sxx = float(xc @ xc)
    if sxx <= 0:
        return float("nan")
    if lags is None:
        lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    # Score for slope: u_t = xc_t * resid_t
    u = xc * resid
    s = float(u @ u)
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        s += 2.0 * w * float(u[k:] @ u[:-k])
    var_beta = s / (sxx ** 2)
    return float(np.sqrt(max(var_beta, 0.0)))


def predictive_regression(
    df: pd.DataFrame,
    x_col: str = "ratio",
    y_col: str = "fwd_ret",
) -> dict:
    """OLS of next-year return on the valuation level, with a HAC t-stat.

    Returns a dict with ``beta`` (slope, per 1 percentage-point of GDP),
    ``beta_per_100`` (the more readable slope per 100pp of GDP),
    ``alpha`` (intercept), ``r2``, ``hac_se``, ``tstat`` (HAC), ``n``,
    and ``corr`` (Pearson correlation of x and y).
    """
    d = df[[x_col, y_col]].dropna()
    x = d[x_col].to_numpy(dtype=float)
    y = d[y_col].to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return {k: float("nan") for k in
                ("beta", "beta_per_100", "alpha", "r2", "hac_se", "tstat", "corr")} | {"n": n}

    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    resid = y - X @ coef
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    hac_se = _hac_se(x, resid)
    tstat = beta / hac_se if hac_se and hac_se > 0 else float("nan")
    corr = float(np.corrcoef(x, y)[0, 1]) if n >= 2 else float("nan")

    return {
        "beta": beta,
        "beta_per_100": beta * 100.0,
        "alpha": alpha,
        "r2": float(r2),
        "hac_se": float(hac_se),
        "tstat": float(tstat),
        "corr": corr,
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Tercile comparison
# ---------------------------------------------------------------------------
def tercile_returns(
    df: pd.DataFrame,
    x_col: str = "ratio",
    y_col: str = "fwd_ret",
) -> pd.DataFrame:
    """Mean forward return in cheap / mid / expensive valuation terciles.

    Splits years by the *full-sample* tercile breakpoints of the indicator
    (this is in-sample, so a generous upper bound -- the honest version is the
    expanding-quantile timing strategy below).  Returns a DataFrame indexed by
    bucket ('cheap', 'mid', 'expensive') with columns ``n``, ``mean_fwd``,
    ``vol_fwd``, ``up_rate``.
    """
    d = df[[x_col, y_col]].dropna().copy()
    q1, q2 = d[x_col].quantile([1 / 3, 2 / 3])
    bucket = np.where(d[x_col] <= q1, "cheap",
                      np.where(d[x_col] <= q2, "mid", "expensive"))
    d["bucket"] = bucket
    rows = []
    for b in ("cheap", "mid", "expensive"):
        sub = d.loc[d["bucket"] == b, y_col]
        rows.append({
            "bucket": b,
            "n": int(sub.size),
            "mean_fwd": float(sub.mean()) if sub.size else float("nan"),
            "vol_fwd": float(sub.std(ddof=1)) if sub.size > 1 else float("nan"),
            "up_rate": float((sub > 0).mean()) if sub.size else float("nan"),
        })
    return pd.DataFrame(rows).set_index("bucket")


# ---------------------------------------------------------------------------
# Real-time timing strategy (expanding-median rule, no look-ahead)
# ---------------------------------------------------------------------------
def timing_strategy(
    df: pd.DataFrame,
    x_col: str = "ratio",
    y_col: str = "fwd_ret",
    min_history: int = 10,
    cash_ret: float = 0.0,
) -> pd.DataFrame:
    """Long-the-S&P-when-cheap timing rule using an EXPANDING (point-in-time) median.

    At the start of each year Y+1 we only know the indicator history through
    year-end Y.  We go long the S&P (earn ``fwd_ret``) if the indicator at
    year-end Y is at or below the median of all *prior* indicator readings
    (expanding window of at least ``min_history`` years); otherwise we sit in
    cash (earn ``cash_ret``, default 0 -- price-only, conservative).

    This expanding rule is the honest version of "the market is expensive": it
    never uses a threshold calibrated on the future.

    Returns a DataFrame indexed by ``year`` (the forecast year, = Y+1) with
    columns: ``ratio`` (the indicator that drove the decision, observed at Y),
    ``in_market`` (bool), ``strat_ret`` (the realised strategy return),
    ``bah_ret`` (buy-and-hold = fwd_ret).
    """
    d = df[["year", x_col, y_col]].dropna().sort_values("year").reset_index(drop=True)
    rows = []
    for i in range(len(d)):
        ratio_i = float(d.loc[i, x_col])
        fwd = float(d.loc[i, y_col])
        if i < min_history:
            # Burn-in: not enough history to form a point-in-time threshold ->
            # default to invested (the null prior is "be in the market").
            in_mkt = True
        else:
            hist = d.loc[: i - 1, x_col].to_numpy(dtype=float)  # strictly prior
            thresh = float(np.median(hist))
            in_mkt = ratio_i <= thresh
        rows.append({
            "year": int(d.loc[i, "year"]) + 1,
            "ratio": ratio_i,
            "in_market": bool(in_mkt),
            "strat_ret": fwd if in_mkt else cash_ret,
            "bah_ret": fwd,
        })
    out = pd.DataFrame(rows).set_index("year")
    return out


def strategy_stats(timing: pd.DataFrame, one_way_bps: float = 10.0) -> dict:
    """Headline stats for the timing strategy vs buy-and-hold (gross and net of costs).

    Turnover = a regime flip (in<->out) which costs ``one_way_bps`` once per flip.
    Net subtracts that cost in flip years.  Returns annualised geometric returns,
    a HAC t-stat on the (strat - bah) excess return, fraction of years invested,
    and the number of regime flips.
    """
    s = timing["strat_ret"].to_numpy(dtype=float)
    b = timing["bah_ret"].to_numpy(dtype=float)
    in_mkt = timing["in_market"].to_numpy(dtype=bool)
    n = len(s)

    flips = int((in_mkt[1:] != in_mkt[:-1]).sum()) + (1 if in_mkt[0] else 0)
    cost = np.zeros(n)
    prev = False
    for i in range(n):
        if in_mkt[i] != prev:
            cost[i] = one_way_bps * 1e-4
        prev = in_mkt[i]
    s_net = s - cost

    def _cagr(r):
        r = np.asarray(r, dtype=float)
        if r.size == 0:
            return float("nan")
        return float(np.prod(1.0 + r) ** (1.0 / r.size) - 1.0)

    excess = s - b
    mu = float(excess.mean())
    if n >= 3 and excess.std(ddof=1) > 0:
        e = excess - mu
        lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
        lrv = float(e @ e) / n
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        excess_t = float(mu / se) if se > 0 else float("nan")
    else:
        excess_t = float("nan")

    return {
        "n": int(n),
        "frac_invested": float(in_mkt.mean()),
        "n_flips": flips,
        "strat_cagr": _cagr(s),
        "strat_cagr_net": _cagr(s_net),
        "bah_cagr": _cagr(b),
        "excess_mean": mu,
        "excess_t": excess_t,
    }


# ---------------------------------------------------------------------------
# One-stop summary dict -- mirrors the R dict and docs/results.md
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame, one_way_bps: float = 10.0) -> dict:
    """Flat summary dict for the real merged DataFrame (mirrors results.md / R)."""
    reg = predictive_regression(df)
    terc = tercile_returns(df)
    tim = timing_strategy(df)
    ss = strategy_stats(tim, one_way_bps=one_way_bps)

    return {
        "n": reg["n"],
        "year_start": int(df["year"].min()),
        "year_end": int(df["year"].max()),
        # Predictive regression
        "beta_per_100": round(reg["beta_per_100"] * 100, 2),  # in pp of return per 100pp GDP
        "reg_r2": round(reg["r2"], 3),
        "reg_t": round(reg["tstat"], 3),
        "corr": round(reg["corr"], 3),
        # Terciles (mean forward return, in pct)
        "cheap_mean": round(float(terc.loc["cheap", "mean_fwd"]) * 100, 1),
        "mid_mean": round(float(terc.loc["mid", "mean_fwd"]) * 100, 1),
        "exp_mean": round(float(terc.loc["expensive", "mean_fwd"]) * 100, 1),
        "cheap_up": round(float(terc.loc["cheap", "up_rate"]) * 100, 1),
        "exp_up": round(float(terc.loc["expensive", "up_rate"]) * 100, 1),
        # Timing strategy
        "strat_cagr": round(ss["strat_cagr"] * 100, 2),
        "strat_cagr_net": round(ss["strat_cagr_net"] * 100, 2),
        "bah_cagr": round(ss["bah_cagr"] * 100, 2),
        "frac_invested": round(ss["frac_invested"] * 100, 1),
        "n_flips": ss["n_flips"],
        "excess_t": round(ss["excess_t"], 3),
    }
