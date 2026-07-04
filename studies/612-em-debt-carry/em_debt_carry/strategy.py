"""Strategy + inference for Study 612 — EM Debt Carry.

The claim under test: *"Emerging-market sovereign bonds pay a fat spread over Treasuries —
carry you actually collect."* We take it apart in five moves, all on monthly total-return
vs price-only tapes:

    * **Total-return spread.** Monthly EMB total return minus IEF total return — the
      carry you *actually collect* after every default, restructuring and spread blowout
      on the tape. Its mean (with a Newey-West HAC t — bond spreads are serially
      correlated) is the decisive Signal statistic.

    * **Yield pickup.** Monthly coupon component = total-return minus price-only return,
      per fund; EMB's coupon minus IEF's coupon is the *promised* pickup. A rolling-12m
      version shows how the promise moved. The gap between the promised pickup and the
      collected spread is what defaults and risk-off ate.

    * **Local-currency contrast (EMLC, 2010-08+).** Same borrowers, but you also own
      their currencies. If the carry survives in USD but dies in local terms, the "fat
      spread" is a hard-currency phenomenon and the FX leg is where it leaks
      (see [364-fx-carry-trade] for the currency side of that story).

    * **Risk-off coincidence (the carry-crash profile).** Spread conditional on equity
      tape: mean EMB-minus-IEF in SPY's worst-decile months vs all other months (Welch t
      of the gap), plus the spread's correlation with SPY and its skew. Carry that pays
      steadily and then hands it back exactly when everything else is falling is
      insurance-selling, not free money.

    * **Crisis ledger (third axis).** Five fixed, documented risk-off windows (GFC 2008,
      taper tantrum 2013, EM stress 2018, COVID 2020, rate/Russia shock 2022): cumulative
      spread surrendered inside them vs collected outside them — does the spread survive
      the crises it insures?

Inference: HAC/Newey-West t everywhere a mean of a serially-correlated monthly series is
tested (lags = 6); Welch t for the risk-off group split. Sharpe races are
excess-vs-excess (over BIL). No random baseline is used anywhere. Execution: the study is
a passive buy-and-hold decomposition — entry is the first month-end close of the common
window, one documented (and here trivial) lag; there is no timing signal to lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12.0
HAC_LAGS = 6

# Risk-off windows (fixed, documented calendar dates — not fitted):
#   GFC: the Lehman shock through the March-2009 trough (EMBI spreads > 800 bp).
#   Taper tantrum: Bernanke's 2013-05-22 testimony through the September FOMC.
#   EM stress 2018: the Argentina/Turkey currency crises and the Fed-hike EM washout.
#   COVID: the 2020 dash-for-cash (EM debt funds gapped with equities).
#   2022: the Fed hiking cycle + Russia's default/ejection through the October trough.
CRISES = [
    ("GFC 2008",            "2008-09-01", "2009-03-31"),
    ("Taper tantrum 2013",  "2013-05-01", "2013-09-30"),
    ("EM stress 2018",      "2018-04-01", "2018-11-30"),
    ("COVID 2020",          "2020-02-01", "2020-04-30"),
    ("Rate/Russia 2022",    "2022-01-01", "2022-10-31"),
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
    """Monthly coupon/distribution return = total-return minus price-only return."""
    return mtr - mpx


# --------------------------------------------------------------------------- #
# HAC (Newey-West) inference + Welch
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


def welch_t(a: np.ndarray | pd.Series, b: np.ndarray | pd.Series) -> float:
    """Welch t of mean(a) - mean(b) (unequal variance). NaN if either < 2 obs."""
    a = np.asarray(pd.Series(a).dropna(), dtype=float)
    b = np.asarray(pd.Series(b).dropna(), dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def nw_regression(y: pd.Series, X: pd.DataFrame, lags: int = HAC_LAGS) -> dict:
    """OLS of ``y`` on ``X`` (+ intercept) with Newey-West HAC standard errors.

    ``y`` and ``X`` should be *excess* returns so the intercept is a genuine
    risk-adjusted spread.
    """
    df = pd.concat([y.rename("_y"), X], axis=1).dropna()
    yv = df["_y"].to_numpy(dtype=float)
    Xm = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(dtype=float) for c in X.columns])
    n, k = Xm.shape
    XtX_inv = np.linalg.inv(Xm.T @ Xm)
    coef = XtX_inv @ Xm.T @ yv
    resid = yv - Xm @ coef
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


# --------------------------------------------------------------------------- #
# The carry spread (the decisive Signal statistic)
# --------------------------------------------------------------------------- #
def spread_stats(mtr: pd.DataFrame, name: str = "EMB", bench: str = "IEF",
                 lags: int = HAC_LAGS) -> dict:
    """Total-return spread name-minus-bench: mean, HAC t, growth, skew.

    This is the carry you actually collect: both legs are total return, both in USD,
    both duration ~7y, so the spread is the EM credit/carry premium net of everything
    the tape threw at it.
    """
    sp = (mtr[name] - mtr[bench]).dropna()
    mu, t = nw_tstat(sp, lags=lags)
    w_name = float((1.0 + mtr[name].loc[sp.index]).prod()) * 100
    w_bench = float((1.0 + mtr[bench].loc[sp.index]).prod()) * 100
    return {
        "spread_bps_mo": mu * 1e4,
        "spread_ann_pct": float(np.expm1(np.log1p(mu) * MONTHS)) * 100,
        "hac_t": t, "n_months": int(len(sp)),
        "skew": float(pd.Series(sp).skew()),
        "wealth_name": w_name, "wealth_bench": w_bench,
        "start": sp.index.min(), "end": sp.index.max(),
        "series": sp,
    }


def yield_pickup(mtr: pd.DataFrame, mpx: pd.DataFrame,
                 name: str = "EMB", bench: str = "IEF") -> dict:
    """The PROMISED carry: coupon component of name minus coupon component of bench.

    Coupon component = TR minus price-only return per fund. The pickup's mean (HAC t) is
    the yield spread the pitch quotes; ``rolling`` is its trailing-12m annualised path.
    """
    div = dividend_component(mtr, mpx)
    pick = (div[name] - div[bench]).dropna()
    mu, t = nw_tstat(pick)
    mu_n, t_n = nw_tstat(div[name].loc[pick.index])
    mu_b, _ = nw_tstat(div[bench].loc[pick.index])
    rolling = (div[name].rolling(12).sum() - div[bench].rolling(12).sum()).dropna() * 100
    return {
        "pickup_bps_mo": mu * 1e4,
        "pickup_ann_pct": float(np.expm1(np.log1p(mu) * MONTHS)) * 100,
        "hac_t": t,
        "coupon_name_ann_pct": float(np.expm1(np.log1p(mu_n) * MONTHS)) * 100,
        "coupon_name_t": t_n,
        "coupon_bench_ann_pct": float(np.expm1(np.log1p(mu_b) * MONTHS)) * 100,
        "rolling": rolling, "n_months": int(len(pick)),
    }


# --------------------------------------------------------------------------- #
# Risk-off coincidence — the carry-crash profile
# --------------------------------------------------------------------------- #
def riskoff_profile(mtr: pd.DataFrame, name: str = "EMB", bench: str = "IEF",
                    spy: str = "SPY", q: float = 0.10) -> dict:
    """Spread conditional on the equity tape: worst-decile SPY months vs the rest.

    If the carry is insurance-selling, the spread should be strongly negative exactly in
    the months equities crash (Welch t of the risk-off-vs-rest gap), positively
    correlated with SPY, and left-skewed.
    """
    df = pd.concat([(mtr[name] - mtr[bench]).rename("sp"), mtr[spy].rename("spy")],
                   axis=1).dropna()
    cut = df["spy"].quantile(q)
    off = df[df["spy"] <= cut]["sp"]
    rest = df[df["spy"] > cut]["sp"]
    return {
        "q": q, "cut_pct": float(cut) * 100,
        "n_off": int(len(off)), "n_rest": int(len(rest)),
        "off_bps_mo": float(off.mean()) * 1e4,
        "rest_bps_mo": float(rest.mean()) * 1e4,
        "gap_bps_mo": float(off.mean() - rest.mean()) * 1e4,
        "welch_t": welch_t(off, rest),
        "corr_spy": float(df["sp"].corr(df["spy"])),
    }


def carry_premium(mtr: pd.DataFrame, name: str = "EMB", cash_col: str = "BIL",
                  factors: tuple[str, ...] = ("IEF", "SPY"), lags: int = HAC_LAGS) -> dict:
    """HAC alpha of the fund's excess return on IEF + SPY excess returns.

    Strips duration AND the equity beta: is anything left of the carry once you account
    for the fact that part of the 'spread' is just hidden equity risk?
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


# --------------------------------------------------------------------------- #
# Crisis autopsies + the third-axis ledger
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


def crisis_mask(index: pd.DatetimeIndex,
                crises: list[tuple[str, str, str]] = CRISES) -> pd.Series:
    """Boolean mask over month-end stamps: True inside any named crisis window."""
    m = pd.Series(False, index=index)
    for _label, a, b in crises:
        m |= (index >= pd.Timestamp(a)) & (index <= pd.Timestamp(b) + pd.offsets.MonthEnd(0))
    return m


def crisis_ledger(mtr: pd.DataFrame, name: str = "EMB", bench: str = "IEF",
                  crises: list[tuple[str, str, str]] = CRISES) -> dict:
    """Third axis: does the spread survive the crises it insures?

    Splits the monthly TR spread into crisis months (inside the five fixed windows) and
    calm months. Reports cumulative spread surrendered in-crisis, collected out-of-crisis,
    the whole-sample total, per-window cumulative spreads, and a Welch t of the
    crisis-vs-calm monthly gap.
    """
    sp = (mtr[name] - mtr[bench]).dropna()
    mask = crisis_mask(sp.index, crises)
    in_c, out_c = sp[mask], sp[~mask]
    per_window = []
    for label, a, b in crises:
        w = sp[(sp.index >= pd.Timestamp(a)) &
               (sp.index <= pd.Timestamp(b) + pd.offsets.MonthEnd(0))]
        if len(w) == 0:
            continue
        per_window.append((label, float(w.sum()) * 100, int(len(w))))
    return {
        "n_crisis": int(len(in_c)), "n_calm": int(len(out_c)),
        "crisis_cum_pct": float(in_c.sum()) * 100,
        "calm_cum_pct": float(out_c.sum()) * 100,
        "total_cum_pct": float(sp.sum()) * 100,
        "crisis_bps_mo": float(in_c.mean()) * 1e4,
        "calm_bps_mo": float(out_c.mean()) * 1e4,
        "welch_t": welch_t(in_c, out_c),
        "calm_hac_t": nw_tstat(out_c)[1],
        "per_window": per_window,
    }
