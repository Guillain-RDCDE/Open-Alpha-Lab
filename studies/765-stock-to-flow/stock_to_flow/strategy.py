"""Strategy + inference for Study 765 — Stock-to-Flow.

The claim (PlanB, 2019): Bitcoin's price is governed by its scarcity, measured as the
stock-to-flow ratio SF = stock / annual-flow, via a power law

    ln(price) = a + b * ln(SF)     (reported in-sample R^2 ~ 0.95).

This module runs the honest teardown of that claim in four movements:

* **The fit, and why it flatters.** OLS of ln(price) on ln(SF), full sample and — the honest
  version — fit only through the model's **publication date** and projected forward
  (genuine out-of-sample). The catch, made explicit: ln(SF) is ~96% correlated with calendar
  time (it is a near-deterministic staircase of the halving schedule), so regressing a trending
  log-price on it is a textbook **spurious regression** of two non-stationary series — almost any
  smoothly rising asset would "fit". We show it by racing the SF fit against a plain
  ln(price) ~ time trend.

* **Out-of-sample divergence.** With coefficients frozen at publication, how far does the model's
  predicted price drift from the realised tape — the six-figure predictions vs the 2022 crash and
  the years after.

* **Is the valuation residual tradable?** Treat "price below the model line" as an
  undervaluation buy signal. Predictive regression of forward BTC returns on the residual
  (Newey-West / HAC t), a residual-timer backtest vs buy-and-hold net of costs (gross AND net),
  and a matched-exposure random placebo.

* **Synthetic positive control.** A deterministic world where a mean-reversion-to-model effect is
  *planted*, to prove the predictive machinery recovers a real valuation signal when one exists
  and reads ~zero when it doesn't.

The decisive numbers: the out-of-sample R^2 and predicted/actual divergence (the "busted" axis),
the HAC t on the residual->return regression (the Signal axis), and the net timer-vs-HODL race
(the Tradability axis).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365   # BTC trades every calendar day


# --------------------------------------------------------------------------- #
# The S2F power-law fit
# --------------------------------------------------------------------------- #
def fit_s2f(df: pd.DataFrame, train_end: str | None = None,
            use_marketcap: bool = False) -> dict:
    """OLS of ln(value) on ln(SF) over the rows up to ``train_end`` (all rows if None).

    ``value`` is price (default) or market cap (price * supply, PlanB's own dependent variable).
    Returns the slope/intercept, in-sample R^2 and the residual standard deviation (log units).
    No look-ahead: when ``train_end`` is a pre-publication date the coefficients only see data
    that existed then.
    """
    d = df if train_end is None else df[df.index <= pd.Timestamp(train_end)]
    x = np.log(d["sf"].to_numpy())
    val = (d["price"] * d["supply"]) if use_marketcap else d["price"]
    y = np.log(val.to_numpy())
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    resid_sd = float(np.std(y - yhat, ddof=2))
    return {"a": float(a), "b": float(b), "r2": r2, "resid_sd": resid_sd,
            "n": int(len(d)), "use_marketcap": use_marketcap}


def model_price(df: pd.DataFrame, a: float, b: float) -> pd.Series:
    """The model's predicted BTC price path, exp(a + b ln SF)."""
    return pd.Series(np.exp(a + b * np.log(df["sf"].to_numpy())), index=df.index, name="model")


def valuation_residual(df: pd.DataFrame, a: float, b: float) -> pd.Series:
    """ln(price) - (a + b ln SF): >0 means price ABOVE the model (rich), <0 BELOW (cheap)."""
    r = np.log(df["price"].to_numpy()) - (a + b * np.log(df["sf"].to_numpy()))
    return pd.Series(r, index=df.index, name="resid")


def oos_fit_stats(df: pd.DataFrame, train_end: str, use_marketcap: bool = False) -> dict:
    """In-sample vs out-of-sample fit with coefficients FROZEN at ``train_end``.

    The honest test of a predictive model: fit only on data that existed at publication, then
    score the untouched future. Reports in-sample R^2, out-of-sample R^2 (frozen coefficients,
    so it can go negative), and the log RMSE on each side.
    """
    fit = fit_s2f(df, train_end=train_end, use_marketcap=use_marketcap)
    a, b = fit["a"], fit["b"]
    tr = df[df.index <= pd.Timestamp(train_end)]
    oos = df[df.index > pd.Timestamp(train_end)]

    def _score(d):
        x = np.log(d["sf"].to_numpy())
        val = (d["price"] * d["supply"]) if use_marketcap else d["price"]
        y = np.log(val.to_numpy())
        yhat = a + b * x
        ss_res = float(((y - yhat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return 1.0 - ss_res / ss_tot, float(np.sqrt(((y - yhat) ** 2).mean()))

    r2_in, rmse_in = _score(tr)
    r2_oos, rmse_oos = _score(oos)
    return {"a": a, "b": b, "train_end": train_end, "n_train": len(tr), "n_oos": len(oos),
            "r2_in": r2_in, "r2_oos": r2_oos, "rmse_in": rmse_in, "rmse_oos": rmse_oos}


def spurious_trend_race(df: pd.DataFrame) -> dict:
    """Race the S2F fit against a plain ln(price) ~ calendar-time trend.

    If S2F only "works" because ln(SF) is a proxy for time, a trend regression should fit almost
    as well. Returns both R^2's and the correlation of ln(SF) with time — the crux of the
    spurious-regression critique.
    """
    x_sf = np.log(df["sf"].to_numpy())
    y = np.log(df["price"].to_numpy())
    t = np.arange(len(df), dtype=float)

    def _r2(x):
        b, a = np.polyfit(x, y, 1)
        yhat = a + b * x
        return 1.0 - float(((y - yhat) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())

    return {"r2_sf": _r2(x_sf), "r2_time": _r2(t),
            "corr_sf_time": float(np.corrcoef(x_sf, t)[0, 1])}


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def _forward_returns(price: pd.Series, horizon: int) -> pd.Series:
    """Simple forward return over ``horizon`` days at each date (NaN where it runs off the end)."""
    logp = np.log(price.to_numpy())
    n = len(logp)
    out = np.full(n, np.nan)
    out[: n - horizon] = np.exp(logp[horizon:] - logp[: n - horizon]) - 1.0
    return pd.Series(out, index=price.index, name=f"fwd_{horizon}")


def newey_west_t(x: np.ndarray, y: np.ndarray, lag: int) -> dict:
    """OLS slope of y on x with a Newey-West (HAC) t-stat at the given lag.

    Robust to the heavy serial correlation induced by overlapping forward-return windows — the
    right standard error for a residual->forward-return regression.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = len(x)
    if n < lag + 5:
        return {"slope": float("nan"), "t": float("nan"), "n": n}
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for L in range(1, lag + 1):
        w = 1.0 - L / (lag + 1.0)
        G = (X[L:] * resid[L:, None]).T @ (X[:-L] * resid[:-L, None])
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se_slope = float(np.sqrt(cov[1, 1]))
    return {"slope": float(beta[1]), "t": float(beta[1] / se_slope) if se_slope > 0 else float("nan"),
            "n": n}


def predictive_regression(df: pd.DataFrame, resid: pd.Series,
                          horizons: tuple[int, ...] = (30, 90, 180, 365),
                          window_start: str | None = None) -> pd.DataFrame:
    """Does the S2F valuation residual predict forward BTC returns?

    For each horizon, regress the (overlapping) forward return on the *lagged* residual (the
    residual is known at day t; the return runs t -> t+h), Newey-West t at lag = horizon. A
    genuine "buy when cheap vs model" signal needs a **negative** slope (a low/negative residual
    -> high forward return) clearing |t| >= 2. ``window_start`` restricts to the out-of-sample
    era.
    """
    d = df if window_start is None else df[df.index >= pd.Timestamp(window_start)]
    r = resid.reindex(d.index)
    rows = []
    for h in horizons:
        fwd = _forward_returns(d["price"], h)
        nw = newey_west_t(r.to_numpy(), fwd.to_numpy(), lag=int(round(1.5 * h)))
        rows.append({"horizon": h, "slope": nw["slope"], "hac_t": nw["t"], "n": nw["n"]})
    return pd.DataFrame(rows).set_index("horizon")


# --------------------------------------------------------------------------- #
# Residual timer vs buy-and-hold
# --------------------------------------------------------------------------- #
def timer_backtest(df: pd.DataFrame, resid: pd.Series, threshold: float = 0.0,
                   cost_bps: float = 10.0, window_start: str | None = None) -> dict:
    """Long BTC when the residual is below ``threshold`` (price cheap vs the model), else cash.

    One-day execution lag (residual known at t's close -> position held from t+1). Costs charged
    one-way x NAV on every switch. Reports GROSS and NET total return, CAGR, Sharpe and exposure,
    against continuous buy-and-hold over the *same* window (the only window the rule could act in).
    ``window_start`` pins the out-of-sample / post-2021 era.
    """
    d = df if window_start is None else df[df.index >= pd.Timestamp(window_start)]
    ret = d["price"].pct_change().dropna()
    r = resid.reindex(ret.index)
    raw_pos = (r < threshold).astype(float)          # 1 = long (cheap), 0 = cash
    pos = raw_pos.shift(1).fillna(0.0)               # one-day execution lag
    switches = pos.diff().abs().fillna(pos.abs())
    cost = cost_bps / 1e4

    gross = pos * ret
    net = gross - switches * cost
    n_years = (d.index[-1] - d.index[0]).days / 365.25

    def _tot(x):
        return float((1.0 + x).prod() - 1.0)

    def _cagr(x):
        w = float((1.0 + x).prod())
        return (w ** (1.0 / n_years) - 1.0) * 100 if n_years > 0 else float("nan")

    def _sharpe(x):
        return float(x.mean() / x.std() * np.sqrt(DAYS_PER_YEAR)) if x.std() > 0 else float("nan")

    return {
        "window_start": str(d.index[0].date()), "window_end": str(d.index[-1].date()),
        "years": n_years, "n_switches": int(switches.sum()),
        "exposure_pct": float(pos.mean()) * 100,
        "gross_total_pct": _tot(gross) * 100, "net_total_pct": _tot(net) * 100,
        "gross_cagr_pct": _cagr(gross), "net_cagr_pct": _cagr(net),
        "gross_sharpe": _sharpe(gross), "net_sharpe": _sharpe(net),
        "bh_total_pct": _tot(ret) * 100, "bh_cagr_pct": _cagr(ret), "bh_sharpe": _sharpe(ret),
    }


def random_placebo(df: pd.DataFrame, exposure: float, cost_bps: float = 10.0,
                   window_start: str | None = None, n_draws: int = 2000,
                   seed: int = 765) -> dict:
    """Matched-exposure random-timing null for the residual timer.

    Draw random long/cash paths at the timer's realised exposure many times; report the share of
    random books whose NET total return matches or beats the real timer. Answers: is the timer's
    result distinguishable from spending the same fraction of days long at random?
    """
    d = df if window_start is None else df[df.index >= pd.Timestamp(window_start)]
    ret = d["price"].pct_change().dropna().to_numpy()
    n = len(ret)
    k = int(round(exposure / 100.0 * n))
    cost = cost_bps / 1e4
    rng = np.random.default_rng(seed)
    totals = np.empty(n_draws)
    for i in range(n_draws):
        pos = np.zeros(n)
        pos[rng.choice(n, size=k, replace=False)] = 1.0
        switches = np.abs(np.diff(np.r_[0.0, pos]))
        net = pos * ret - switches * cost
        totals[i] = float(np.prod(1.0 + net) - 1.0)
    return {"mean_total_pct": float(totals.mean()) * 100,
            "p95_total_pct": float(np.percentile(totals, 95)) * 100,
            "n_draws": n_draws, "exposure_pct": exposure}


# --------------------------------------------------------------------------- #
# Synthetic positive control (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(df: pd.DataFrame, horizon: int = 30) -> float:
    """HAC t of the residual->forward-return regression on a synthetic world.

    On a planted world the residual is the negative of ``true_resid`` up to scale; a genuine
    mean-reversion effect should give a significant NEGATIVE slope (cheap -> higher forward
    return). Uses the same ``newey_west_t`` machinery as the real tape.
    """
    resid = pd.Series(df["true_resid"].to_numpy(), index=df.index)
    fwd = _forward_returns(df["price"], horizon)
    nw = newey_west_t(resid.to_numpy(), fwd.to_numpy(), lag=int(round(1.5 * horizon)))
    return nw["t"]
