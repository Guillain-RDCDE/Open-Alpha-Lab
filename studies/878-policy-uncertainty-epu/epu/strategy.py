"""Strategy + inference for Study 878 — Economic Policy Uncertainty (EPU).

The claim, operationalised on a monthly frame of ``unc`` (the uncertainty signal — real
Baker-Bloom-Davis EPU, or the labelled VIX proxy), ``spy`` (month-end close), and ``rv``
(the month's annualised realized vol):

    High uncertainty should PREDICT the future — (a) higher forward equity **volatility**,
    and (b), per the risk-premium story, higher forward **returns** as compensation.

We test both legs with **predictive regressions** and honest inference:

  * **Forward realized vol on uncertainty.** Regress the H-month-ahead realized vol on the
    uncertainty level (and on its month-over-month change), Newey-West HAC *t*, R^2. A
    positive, significant slope is the vol leg. (Caveat: vol clusters, so a vol->vol
    prediction is close to mechanical; we say so.)
  * **Forward SPY return on uncertainty.** Regress the H-month-ahead SPY return on the
    uncertainty level (and change), Newey-West HAC *t*, R^2. A positive, significant slope
    is the risk-premium leg — the desk's real question.
  * **A permutation placebo** breaks the uncertainty->outcome link (block-shuffle) to check
    the slope isn't a lucky alignment given the heavy autocorrelation in both series.
  * **A two-era cut** asks whether any slope is one regime's artefact.
  * **A costed timer** (the Tradability axis): a long/flat SPY rule that leans INTO high
    uncertainty (the risk-premium bet), net of a one-way cost, raced against buy-and-hold.

The decisive confound: an uncertainty index is a *contemporaneous* stress gauge — it spikes
WITH drawdowns and vol. A forward-vol slope is mostly vol-clustering; a forward-return slope
is the genuinely surprising claim, and it is where the honesty rails bite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 12  # months per year


# --------------------------------------------------------------------------- #
# The monthly frame (shared by the real and synthetic paths)
# --------------------------------------------------------------------------- #
def monthly_frame(spy_daily: pd.Series, unc_monthly: pd.Series) -> pd.DataFrame:
    """Aligned monthly ``[unc, spy, rv]`` frame from a daily SPY price and a monthly
    uncertainty series. ``spy`` = month-end close; ``rv`` = the month's annualised realized
    vol (std of daily log returns * sqrt(252)); ``unc`` = the uncertainty level at month end.
    """
    spy_daily = spy_daily.sort_index()
    spy = spy_daily.resample("ME").last()
    lr = np.log(spy_daily / spy_daily.shift(1)).dropna()
    rv = lr.groupby(lr.index.to_period("M")).std(ddof=0) * np.sqrt(252)
    rv.index = rv.index.to_timestamp(how="end").normalize() + pd.offsets.MonthEnd(0)
    unc = unc_monthly.copy()
    unc.index = unc.index + pd.offsets.MonthEnd(0)
    frame = pd.DataFrame({"unc": unc}).join(
        pd.DataFrame({"spy": spy.rename("spy")}), how="inner").join(
        pd.DataFrame({"rv": rv.rename("rv")}), how="inner")
    return frame.dropna()


# --------------------------------------------------------------------------- #
# Forward outcomes
# --------------------------------------------------------------------------- #
def forward_return(frame: pd.DataFrame, horizon: int) -> pd.Series:
    """``horizon``-month forward simple SPY return, stamped at the *formation* month end.

    Value on row ``t`` is ``spy[t+h]/spy[t]-1`` — realised strictly *after* ``t`` (the
    uncertainty at ``t`` is known at ``t``, so there is a built-in one-month execution lag)."""
    spy = frame["spy"]
    return spy.shift(-horizon) / spy - 1.0


def forward_rv(frame: pd.DataFrame, horizon: int) -> pd.Series:
    """Mean annualised realized vol over the next ``horizon`` months (``t+1 .. t+h``)."""
    rv = frame["rv"]
    fwd = rv.shift(-1).rolling(horizon).mean().shift(-(horizon - 1))
    fwd.name = "fwd_rv"
    return fwd


def unc_change(frame: pd.DataFrame) -> pd.Series:
    """Month-over-month change in the uncertainty level (a stationary transform)."""
    return frame["unc"].diff()


# --------------------------------------------------------------------------- #
# Inference primitives (canonical desk set)
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


def newey_west_t(x: np.ndarray, lags: int = 6) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    gamma0 = float(u @ u) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        cov = float(u[l:] @ u[:-l]) / n
        var += 2.0 * w * cov
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# The predictive regression with Newey-West HAC slope t
# --------------------------------------------------------------------------- #
def predictive_reg(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray,
                   lags: int = 6) -> dict:
    """OLS ``y = a + b*x + u`` with a **Newey-West HAC** *t* on the slope ``b``.

    Overlapping forward outcomes and a persistent regressor make plain OLS *t* far too
    optimistic; the Bartlett-kernel HAC sandwich corrects the slope's standard error.
    Returns ``slope``, HAC ``t``, ``r2``, ``n``, and the standardised slope ``beta_sd`` (the
    slope in units of forward-outcome-sd per 1-sd move in ``x``)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 5:
        return {"slope": float("nan"), "t": float("nan"), "r2": float("nan"),
                "n": n, "beta_sd": float("nan")}
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    # HAC "meat": weighted autocovariances of the score g_t = X_t * u_t
    g = X * resid[:, None]
    S = g.T @ g
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        G = g[l:].T @ g[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se_slope = float(np.sqrt(V[1, 1])) if V[1, 1] > 0 else float("nan")
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    return {
        "slope": float(beta[1]),
        "t": float(beta[1] / se_slope) if se_slope and se_slope > 0 else float("nan"),
        "r2": float(r2),
        "n": n,
        "beta_sd": float(beta[1] * sx / sy) if sy > 0 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# The two headline regressions across horizons
# --------------------------------------------------------------------------- #
def regress_forward(frame: pd.DataFrame, outcome: str, horizons=(1, 3, 6, 12),
                    on: str = "level", lags: int = 6) -> pd.DataFrame:
    """Predictive regressions of a forward outcome on the uncertainty ``on`` in {"level",
    "change"} across ``horizons``. ``outcome`` in {"ret", "rv"}. One row per horizon with
    slope, HAC *t*, R^2, standardised slope, n."""
    x = frame["unc"] if on == "level" else unc_change(frame)
    rows = []
    for h in horizons:
        y = forward_return(frame, h) if outcome == "ret" else forward_rv(frame, h)
        r = predictive_reg(x.reindex(frame.index), y.reindex(frame.index), lags=lags)
        rows.append({"horizon": h, **r})
    return pd.DataFrame(rows).set_index("horizon")


# --------------------------------------------------------------------------- #
# Placebo — is the slope a lucky alignment given the autocorrelation?
# --------------------------------------------------------------------------- #
def placebo_pvalue(frame: pd.DataFrame, outcome: str, horizon: int, on: str = "level",
                   block: int = 12, n_draws: int = 2000, seed: int = 878) -> dict:
    """Block-shuffle the regressor (preserving its short-run autocorrelation via ``block``-
    month blocks), re-fit the slope many times, and ask how often ``|placebo slope| >=
    |observed slope|``. A two-sided p on whether the real slope beats a broken-link null."""
    x = (frame["unc"] if on == "level" else unc_change(frame)).to_numpy(dtype=float)
    y = (forward_return(frame, horizon) if outcome == "ret"
         else forward_rv(frame, horizon)).to_numpy(dtype=float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    xo, yo = x[ok], y[ok]
    n = len(xo)
    if n < 20:
        return {"obs_slope": float("nan"), "p_value": float("nan"), "n_draws": 0}
    obs = predictive_reg(xo, yo, lags=6)["slope"]
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    hits = 0
    slopes = np.empty(n_draws)
    for i in range(n_draws):
        starts = rng.integers(0, max(1, n - block), size=nb)
        perm = np.concatenate([np.arange(s, s + block) for s in starts])[:n] % n
        xp = xo[perm]
        b = predictive_reg(xp, yo, lags=6)["slope"]
        slopes[i] = b
        if abs(b) >= abs(obs):
            hits += 1
    return {"obs_slope": float(obs), "placebo_mean": float(np.nanmean(slopes)),
            "placebo_sd": float(np.nanstd(slopes, ddof=1)),
            "p_value": float(hits / n_draws), "n_draws": n_draws}


# --------------------------------------------------------------------------- #
# The costed timer (the Tradability axis)
# --------------------------------------------------------------------------- #
def monthly_returns(frame: pd.DataFrame) -> pd.Series:
    return (frame["spy"] / frame["spy"].shift(1) - 1.0).dropna()


def timer_stats(frame: pd.DataFrame, thr_q: float = 0.66, lag: int = 1,
                cost_bps: float = 10.0, lean_in: bool = True) -> dict:
    """Long/flat SPY rule keyed off the uncertainty level, net of costs.

    ``lean_in=True`` (the risk-premium bet): hold SPY when uncertainty known ``lag`` months
    earlier is in its top ``1-thr_q`` (above the trailing-expanding ``thr_q`` quantile),
    else flat. ``lean_in=False`` de-risks in high uncertainty. A one-way ``cost_bps`` is
    charged on each change of position. Raced against buy-and-hold on the same tape."""
    ret = monthly_returns(frame)
    u = frame["unc"]
    thr = u.expanding(min_periods=24).quantile(thr_q)
    hot = (u > thr)
    want = hot if lean_in else ~hot
    pos_raw = want.shift(lag).astype(float)
    pos = pos_raw.reindex(ret.index).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs())
    c = cost_bps / 1e4
    gross = pos * ret
    net = gross - turn * c

    def _stats(r: pd.Series) -> dict:
        mu, sd = r.mean() * ANN, r.std(ddof=1) * np.sqrt(ANN)
        return {"ann_ret": float(mu), "ann_vol": float(sd),
                "sharpe": float(mu / sd) if sd > 0 else float("nan")}

    return {
        "n_months": int(len(ret)), "n_turns": float(turn.sum()),
        "exposure": float((pos != 0).mean()),
        "gross": _stats(gross), "net": _stats(net), "buy_hold": _stats(ret),
        "cost_bps": cost_bps, "lean_in": lean_in,
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(spy_daily: pd.Series, unc_monthly: pd.Series,
                     horizon: int = 3) -> dict:
    """Run the two headline slopes on a synthetic (spy, unc) world at one horizon."""
    frame = monthly_frame(spy_daily, unc_monthly)
    ret = predictive_reg(frame["unc"].reindex(frame.index),
                         forward_return(frame, horizon).reindex(frame.index))
    rv = predictive_reg(frame["unc"].reindex(frame.index),
                        forward_rv(frame, horizon).reindex(frame.index))
    return {"ret_t": ret["t"], "ret_slope": ret["slope"],
            "rv_t": rv["t"], "rv_slope": rv["slope"], "n": int(len(frame))}
