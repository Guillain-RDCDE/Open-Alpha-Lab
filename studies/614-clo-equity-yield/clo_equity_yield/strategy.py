"""Strategy + inference for Study 614 — CLO Equity Yield ("the 15% machine").

The claim under test: *"ECC and OXLC pay ~15%+ distributions from CLO equity — the levered
first-loss tranche of leveraged-loan securitizations throws off enormous cash flows, and the
funds pass them to you monthly. The income is real."* We take it apart in five moves, all on
monthly total-return vs price-only tapes:

    * **Distribution decomposition.** Monthly distribution return = total-return minus
      price-only return, per fund. Its mean (with a Newey-West HAC t — payout streams are
      serially correlated) is the cash stream the pitch sells. This settles "the 15% is
      really paid".

    * **NAV/price erosion.** Price-only CAGR per fund. If the price leg is deeply negative,
      part of the distribution is a return OF capital dressed as a return ON capital.

    * **Total-return spread vs HYG (the headline test).** The fund's monthly total return
      minus HYG's, mean tested with a HAC t. HYG is the plain, unlevered, one-click credit
      alternative the same income investor could buy instead.

    * **Alpha vs a credit/equity-matched benchmark.** Regress the fund's monthly *excess*
      return (over BIL) on HYG and SPY *excess* returns with HAC (Newey-West) standard
      errors. The intercept is the premium over a passive levered-credit-plus-equity mix;
      its HAC t decides whether the packaged CLO-equity carry adds anything DIY leverage
      doesn't.

    * **Crisis autopsies.** Peak-to-trough total-return drawdowns inside four named windows
      (2015-16 credit crunch, Q4-2018, COVID 2020, rate shock 2022), vs HYG and SPY.

Inference: HAC/Newey-West t everywhere a mean of a serially-correlated monthly series is
tested (lags = 6; 3/12 as robustness). Sharpe races are excess-vs-excess (over BIL). No
random baseline is used anywhere. Execution: the study is a passive buy-and-hold
decomposition — entry is the first month-end close of each fund's window, one documented
(and here trivial) lag; there is no timing signal to lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12.0
HAC_LAGS = 6

# Crisis windows (fixed, documented calendar dates — not fitted):
#   2015-16: the energy/HY credit crunch (loan and CLO tranche spreads blew out).
#   Q4 2018: the leveraged-loan outflow quarter (loan mutual-fund run, Dec-24 trough).
#   COVID 2020: the dash-for-cash; CLO equity NAVs marked toward zero cash-flow scenarios.
#   2022: the Fed hiking cycle / risk-asset repricing through the October trough.
CRISES = [
    ("Credit crunch 2015-16", "2015-06-01", "2016-02-29"),
    ("Q4 2018", "2018-09-01", "2018-12-31"),
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


def distribution_component(mtr: pd.DataFrame, mpx: pd.DataFrame) -> pd.DataFrame:
    """Monthly distribution return = total-return minus price-only return, per name."""
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
    """Total-return vs price-only vs distribution-component summary for one fund."""
    dist = distribution_component(mtr, mpx)[name]
    mu, t = nw_tstat(dist)
    return {
        "tr_cagr_pct": cagr(mtr[name]) * 100,
        "px_cagr_pct": cagr(mpx[name]) * 100,
        "dist_bps_mo": mu * 1e4,
        "dist_ann_pct": float(np.expm1(np.log1p(mu) * MONTHS)) * 100,
        "dist_hac_t": t,
        "n_months": int(mtr[name].dropna().shape[0]),
    }


# --------------------------------------------------------------------------- #
# Headline test — total-return spread vs HYG
# --------------------------------------------------------------------------- #
def tr_spread_vs(mtr: pd.DataFrame, name: str, bench: str = "HYG",
                 lags: int = HAC_LAGS) -> dict:
    """HAC t of the fund's monthly TOTAL-return spread over the plain credit ETF.

    The one-click alternative test: same investor, same account, HYG instead of the CLO
    fund. Both legs are total return; the spread's Newey-West t is the study's headline
    Signal statistic.
    """
    spread = (mtr[name] - mtr[bench]).dropna()
    mu, t = nw_tstat(spread, lags=lags)
    return {
        "spread_bps_mo": mu * 1e4,
        "spread_ann_pct": float(np.expm1(np.log1p(mu) * MONTHS)) * 100,
        "spread_t": t,
        "n": int(len(spread)),
        "name_cagr_pct": cagr(mtr[name]) * 100,
        "bench_cagr_pct": cagr(mtr[bench]) * 100,
    }


# --------------------------------------------------------------------------- #
# Alpha vs the credit/equity-matched benchmark
# --------------------------------------------------------------------------- #
def carry_premium(mtr: pd.DataFrame, name: str, cash_col: str = "BIL",
                  factors: tuple[str, ...] = ("HYG", "SPY"), lags: int = HAC_LAGS) -> dict:
    """HAC alpha of the fund's excess return on HYG + SPY excess returns.

    The benchmark an investor could hold instead of the packaged CLO-equity carry: a
    passive mix of levered high-yield credit and equity beta, financed at bills. The
    intercept IS the risk-adjusted spread vs that mix; its Newey-West t is the study's
    decisive tradability statistic.
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
                   factors: tuple[str, ...] = ("HYG", "SPY")) -> dict:
    """Race the fund against its own beta-matched benchmark (cash + betas x factors).

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
# Third axis — return OF capital vs return ON capital (returns arithmetic)
# --------------------------------------------------------------------------- #
def capital_return_split(mtr: pd.DataFrame, mpx: pd.DataFrame, name: str) -> dict:
    """How much of the distribution stream was financed by the price leg?

    Pure returns arithmetic on log-CAGRs: the distribution stream (TR minus price-only)
    is the cash paid; the negative price-only CAGR is the capital consumed while paying
    it. ``financed_by_price_pct`` = the share of the annualised distribution stream offset
    by price erosion (0% = fully earned on top of a flat price; 100% = the entire payout
    was matched one-for-one by capital shrinkage). Clipped to [0, 100+] only by the tape.
    """
    d = decompose(mtr, mpx, name)
    dist_ann = d["dist_ann_pct"]
    px_ann = d["px_cagr_pct"]
    financed = (-px_ann / dist_ann * 100.0) if dist_ann > 0 else float("nan")
    return {
        **d,
        "financed_by_price_pct": financed,
        "kept_ann_pct": d["tr_cagr_pct"],
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
