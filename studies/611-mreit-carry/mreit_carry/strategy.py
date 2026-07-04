"""Strategy + inference for Study 611 — mREIT Carry.

The claim under test: *"Mortgage REITs pay 10-14% — leveraged MBS carry that is real income
(and, fine, real drawdowns). Harvest the dividend stream and the carry compounds."* We take
it apart in four moves, all on monthly total-return vs price-only tapes:

    * **Carry decomposition.** Monthly dividend return = total-return minus price-only
      return, per name. Its mean (with a Newey-West HAC t — dividend streams are serially
      correlated) is the carry the pitch sells. This is where "the income is real" is
      settled.

    * **NAV erosion.** Price-only CAGR per name. If the price leg is deeply negative, the
      dividend is partly a return OF capital dressed as a return ON capital.

    * **Carry premium vs a duration-matched benchmark.** Regress the mREIT's monthly
      *excess* return (over BIL) on IEF and SPY *excess* returns with HAC (Newey-West)
      standard errors. The intercept is the carry premium over a passive levered
      Treasuries-plus-equity-beta mix an investor could hold instead; its HAC t decides
      whether the packaged carry adds anything a DIY levered IEF doesn't.

    * **Crisis autopsies.** Peak-to-trough total-return drawdowns inside four named windows
      (GFC 2007-09, taper tantrum 2013, COVID 2020, rate shock 2022), vs SPY and IEF.

Inference: HAC/Newey-West t everywhere a mean of a serially-correlated monthly series is
tested (lags = 6). Sharpe races are excess-vs-excess (over BIL). No random baseline is used
anywhere. Execution: the study is a passive buy-and-hold decomposition — entry is the first
month-end close of the common window, one documented (and here trivial) lag; there is no
timing signal to lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12.0
HAC_LAGS = 6

# Crisis windows (fixed, documented calendar dates — not fitted):
#   GFC: REM inception (2007-06) through the March-2009 equity trough.
#   Taper tantrum: Bernanke's 2013-05-22 testimony through year-end (the mREIT washout).
#   COVID: the 2020 dash-for-cash (mREIT repo margin spiral).
#   2022: the Fed hiking cycle / MBS spread blowout through the October trough.
CRISES = [
    ("GFC 2007-09", "2007-06-01", "2009-03-31"),
    ("Taper tantrum 2013", "2013-05-01", "2013-12-31"),
    ("COVID 2020", "2020-02-01", "2020-04-30"),
    ("Rate shock 2022", "2022-01-01", "2022-10-31"),
]


# --------------------------------------------------------------------------- #
# Panel plumbing
# --------------------------------------------------------------------------- #
def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Wide monthly simple returns (month-end to month-end), partial last month dropped."""
    m = prices.resample("ME").last()
    ret = m.pct_change()
    last_px = prices.index.max()
    last_bucket = ret.index.max()
    if last_px < (last_bucket - pd.offsets.MonthEnd(0)) or last_px.day < last_bucket.day:
        ret = ret.iloc[:-1]
    return ret


def aligned_monthly(tr: pd.DataFrame, px: pd.DataFrame,
                    tickers: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly TR and PX return frames on the common complete-data window."""
    mtr = monthly_returns(tr)
    mpx = monthly_returns(px)
    if tickers is None:
        tickers = [c for c in mtr.columns if c in mpx.columns]
    mtr, mpx = mtr[tickers], mpx[tickers]
    ok = mtr.dropna().index.intersection(mpx.dropna().index)
    return mtr.loc[ok], mpx.loc[ok]


def dividend_component(mtr: pd.DataFrame, mpx: pd.DataFrame) -> pd.DataFrame:
    """Monthly dividend return = total-return minus price-only return, per name."""
    return mtr - mpx


# --------------------------------------------------------------------------- #
# HAC (Newey-West) inference
# --------------------------------------------------------------------------- #
def nw_tstat(x: np.ndarray | pd.Series, lags: int = HAC_LAGS) -> tuple[float, float]:
    """Newey-West HAC t of the mean of ``x`` against zero. Returns (mean, t)."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    if n < 8:
        return float("nan"), float("nan")
    mu = x.mean()
    e = x - mu
    s = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        s += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(s / n)
    return float(mu), float(mu / se) if se > 0 else float("nan")


def nw_regression(y: pd.Series, X: pd.DataFrame, lags: int = HAC_LAGS) -> dict:
    """OLS of ``y`` on ``X`` (+ intercept) with Newey-West HAC standard errors.

    Returns alpha (intercept), betas, their HAC t's and the R². ``y`` and ``X`` should be
    *excess* returns so the intercept is a genuine risk-adjusted spread.
    """
    df = pd.concat([y.rename("_y"), X], axis=1).dropna()
    yv = df["_y"].to_numpy(dtype=float)
    Xm = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(dtype=float) for c in X.columns])
    n, k = Xm.shape
    XtX_inv = np.linalg.inv(Xm.T @ Xm)
    coef = XtX_inv @ Xm.T @ yv
    resid = yv - Xm @ coef
    # Newey-West covariance
    S = np.zeros((k, k))
    for i in range(n):
        xi = Xm[i][:, None]
        S += resid[i] ** 2 * (xi @ xi.T)
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        for i in range(lag, n):
            xi = Xm[i][:, None]
            xj = Xm[i - lag][:, None]
            G = resid[i] * resid[i - lag] * (xi @ xj.T)
            S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    tss = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / tss if tss > 0 else float("nan")
    names = ["alpha"] + list(X.columns)
    return {
        "n": n, "r2": r2,
        "coef": dict(zip(names, coef.tolist())),
        "t": dict(zip(names, (coef / se).tolist())),
        "se": dict(zip(names, se.tolist())),
        "resid": resid,
    }


# --------------------------------------------------------------------------- #
# Performance stats
# --------------------------------------------------------------------------- #
def cagr(monthly: pd.Series) -> float:
    m = monthly.dropna()
    yrs = len(m) / MONTHS
    growth = float(np.exp(np.log1p(m).sum()))
    return growth ** (1.0 / yrs) - 1.0 if yrs > 0 else float("nan")


def ann_vol(monthly: pd.Series) -> float:
    return float(monthly.dropna().std(ddof=1) * np.sqrt(MONTHS))


def sharpe_excess(monthly: pd.Series, cash: pd.Series) -> float:
    """Annualised Sharpe of the EXCESS return over the cash proxy (excess-vs-excess)."""
    ex = (monthly - cash).dropna()
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(MONTHS)) if sd > 0 else float("nan")


def max_drawdown(prices: pd.Series) -> float:
    p = prices.dropna()
    dd = p / p.cummax() - 1.0
    return float(dd.min())


def perf_table(mtr: pd.DataFrame, cash_col: str = "BIL") -> pd.DataFrame:
    """CAGR / vol / excess Sharpe / max-DD per name, on the aligned monthly TR frame."""
    cash = mtr[cash_col]
    rows = {}
    for c in mtr.columns:
        idxp = (1.0 + mtr[c]).cumprod()
        rows[c] = {
            "cagr_pct": cagr(mtr[c]) * 100,
            "vol_pct": ann_vol(mtr[c]) * 100,
            "sharpe_excess": sharpe_excess(mtr[c], cash),
            "max_dd_pct": max_drawdown(idxp) * 100,
        }
    return pd.DataFrame(rows).T


def decompose(mtr: pd.DataFrame, mpx: pd.DataFrame, name: str) -> dict:
    """Total-return vs price-only vs dividend-component summary for one name."""
    div = dividend_component(mtr, mpx)[name]
    mu, t = nw_tstat(div)
    return {
        "tr_cagr_pct": cagr(mtr[name]) * 100,
        "px_cagr_pct": cagr(mpx[name]) * 100,
        "div_bps_mo": mu * 1e4,
        "div_ann_pct": float(np.expm1(np.log1p(mu) * MONTHS)) * 100,
        "div_hac_t": t,
        "n_months": int(mtr[name].dropna().shape[0]),
    }


# --------------------------------------------------------------------------- #
# Carry premium vs the duration-matched benchmark
# --------------------------------------------------------------------------- #
def carry_premium(mtr: pd.DataFrame, name: str, cash_col: str = "BIL",
                  factors: tuple[str, ...] = ("IEF", "SPY"), lags: int = HAC_LAGS) -> dict:
    """HAC alpha of the mREIT's excess return on IEF + SPY excess returns.

    The benchmark an investor could hold instead of the packaged carry: a passive mix of
    levered 7-10y Treasuries (the duration) and equity beta, financed at bills. The
    intercept IS the total-return spread vs that duration-matched levered-IEF benchmark;
    its Newey-West t is the study's decisive tradability statistic.
    """
    cash = mtr[cash_col]
    y = mtr[name] - cash
    X = pd.DataFrame({f: mtr[f] - cash for f in factors})
    reg = nw_regression(y, X, lags=lags)
    out = {
        "alpha_bps_mo": reg["coef"]["alpha"] * 1e4,
        "alpha_ann_pct": float(np.expm1(np.log1p(reg["coef"]["alpha"]) * MONTHS)) * 100,
        "t_alpha": reg["t"]["alpha"], "r2": reg["r2"], "n": reg["n"],
    }
    for f in factors:
        out[f"beta_{f}"] = reg["coef"][f]
        out[f"t_{f}"] = reg["t"][f]
    return out


def benchmark_race(mtr: pd.DataFrame, name: str, cash_col: str = "BIL",
                   factors: tuple[str, ...] = ("IEF", "SPY")) -> dict:
    """Race the mREIT against its own beta-matched benchmark (cash + betas x factors).

    The benchmark's betas come from the full-sample regression — a risk-decomposition
    benchmarking choice (stated openly), not a tradable timing signal; nothing here is a
    trading rule with a look-ahead.
    """
    cp = carry_premium(mtr, name, cash_col=cash_col, factors=factors)
    cash = mtr[cash_col]
    bench = cash.copy()
    for f in factors:
        bench = bench + cp[f"beta_{f}"] * (mtr[f] - cash)
    spread = (mtr[name] - bench).dropna()
    mu, t = nw_tstat(spread)
    idx_name = (1.0 + mtr[name]).cumprod()
    idx_bench = (1.0 + bench).cumprod()
    return {
        **cp,
        "spread_bps_mo": mu * 1e4, "t_spread": t,
        "name_cagr_pct": cagr(mtr[name]) * 100,
        "bench_cagr_pct": cagr(bench) * 100,
        "name_dd_pct": max_drawdown(idx_name) * 100,
        "bench_dd_pct": max_drawdown(idx_bench) * 100,
        "bench_series": bench,
    }


# --------------------------------------------------------------------------- #
# Crisis autopsies
# --------------------------------------------------------------------------- #
def crisis_table(tr_daily: pd.DataFrame, tickers: list[str],
                 crises: list[tuple[str, str, str]] = CRISES) -> pd.DataFrame:
    """Peak-to-trough total-return drawdown per name inside each named window (daily tape)."""
    rows = []
    for label, a, b in crises:
        w = tr_daily.loc[a:b]
        row = {"crisis": label}
        for tk in tickers:
            s = w[tk].dropna()
            row[tk] = max_drawdown(s) * 100 if len(s) > 10 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("crisis")
