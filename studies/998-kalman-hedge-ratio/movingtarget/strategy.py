"""Estimating a hedge ratio that moves — Study 998.

A spread trade is long one thing and short *beta* of another, and beta is not a constant. Three
families of answer:

- **Static OLS** over the whole sample. Unusable in practice — it looks ahead — but it is the
  benchmark that shows how much the relationship moved at all.
- **Rolling OLS** over the last *w* sessions. The standard, and it embodies a bad trade-off:
  short windows are responsive and noisy, long windows are stable and stale. There is no *w*
  that is both, which is the entire motivation for something better.
- **The Kalman filter.** Model the hedge ratio as a hidden state following a random walk,
  observed through noisy returns. The filter's gain then adapts automatically — it moves fast
  when the evidence is strong and slowly when it is not, which is exactly the trade-off a fixed
  window has to guess at in advance.

The filter has two knobs, and the ratio between them is what matters: ``delta`` (how fast the
state is assumed to wander) and ``obs_var`` (how noisy the observation is). Their ratio is the
signal-to-noise of the state-space model, and it maps directly onto an effective window length —
``effective_window`` makes that correspondence explicit, so a reader can see that a Kalman filter
is not magic but a smoothly-weighted window whose length adapts.

The study is careful about one thing above all: **tracking accuracy and trading profit are
different questions**, and an estimator can win the first and lose the second. A hedge ratio
that follows the truth closely also moves more, which means more rebalancing, more turnover and
more cost. ``spread_trade`` prices that honestly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# The estimators
# --------------------------------------------------------------------------- #
def static_hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    """Full-sample OLS slope. Looks ahead; used only as a reference."""
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1, sort=False).dropna()
    if len(df) < 30:
        return np.nan
    vx = float(df["x"].var(ddof=1))
    return float(df["y"].cov(df["x"]) / vx) if vx > 0 else np.nan


def rolling_hedge_ratio(y: pd.Series, x: pd.Series, window: int = 60) -> pd.Series:
    """Trailing OLS slope, known at each close.

    The ``shift(1)`` is what makes it usable: the ratio applied on day *t* is estimated from
    data through *t-1*. Omitting it is the single most common way a pairs backtest becomes
    fiction, because the day's own return then helps choose the hedge that is applied to it.
    """
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1, sort=False).dropna()
    cov = df["y"].rolling(window).cov(df["x"])
    var = df["x"].rolling(window).var()
    return (cov / var).shift(1).rename(f"rolling_{window}")


def kalman_hedge_ratio(y: pd.Series, x: pd.Series, delta: float = 1e-2,
                       obs_var: float | None = None, init_var: float = 1.0,
                       warmup: int = 250) -> pd.DataFrame:
    """A scalar Kalman filter for a random-walking hedge ratio.

    State equation:    ``beta[t] = beta[t-1] + w``,  ``var(w) = delta/(1-delta) * obs_var``
    Observation:       ``y[t] = beta[t] * x[t] + v``, ``var(v) = obs_var``

    ``obs_var`` defaults to the residual variance of a full-sample OLS, which makes ``delta`` a
    scale-free knob: without that, the same ``delta`` means something completely different for
    a pair of gold ETFs and a pair of energy funds, and sweeping it across pairs would be
    meaningless.

    With that scaling, ``delta`` reads as *the state's variance per step as a fraction of the
    observation noise* — so 1e-3 is "the relationship barely moves", 1e-1 is "it moves about as
    fast as the noise", and the useful range is wider than the 1e-4-to-1e-5 usually quoted,
    because that convention assumes an unscaled observation variance of 1.

    The Dai-Rui parameterisation of the state variance keeps ``delta`` on a (0, 1) scale where
    it reads as "how much of the state is allowed to change each step".

    Returned strictly causally: the ``beta`` column on day *t* is the **prior** estimate, formed
    before day *t*'s observation is seen, so it can be applied to day *t* without look-ahead.
    The posterior is returned separately for diagnostics.
    """
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1, sort=False).dropna()
    n = len(df)
    if n < 30:
        return pd.DataFrame(columns=["beta", "beta_post", "var", "innovation", "gain"])
    yv = df["y"].to_numpy(dtype=float)
    xv = df["x"].to_numpy(dtype=float)
    if obs_var is None:
        # Estimate the observation noise from a WARM-UP SLICE, never the whole sample.
        #
        # Scaling matters: only the ratio of state variance to observation variance affects the
        # filter, so a hard-coded obs_var makes ``delta`` mean something different for every
        # pair — two gold ETFs and two energy funds have residual variances orders of magnitude
        # apart. But estimating it from the full sample would be a look-ahead: the scale used
        # in 2008 would depend on what happened in 2024. The warm-up slice is the honest
        # compromise, and the causality test in the suite exists because the full-sample
        # version silently passed everything else.
        k = min(warmup, max(n // 4, 30))
        xw, yw = xv[:k], yv[:k]
        vx = float(np.var(xw, ddof=1))
        b0 = float(np.cov(xw, yw, ddof=1)[0, 1] / vx) if vx > 0 else 0.0
        obs_var = max(float(np.var(yw - b0 * xw, ddof=1)), 1e-14)
    q = delta / max(1.0 - delta, 1e-12) * obs_var

    beta_prior = np.empty(n)
    beta_post = np.empty(n)
    variance = np.empty(n)
    innovation = np.empty(n)
    gain = np.empty(n)

    b = 0.0
    P = init_var
    for t in range(n):
        # predict
        P = P + q
        beta_prior[t] = b
        # update
        s = xv[t] * P * xv[t] + obs_var           # innovation variance
        e = yv[t] - b * xv[t]
        k = (P * xv[t]) / s if s > 0 else 0.0
        b = b + k * e
        P = P - k * xv[t] * P
        beta_post[t] = b
        variance[t] = P
        innovation[t] = e
        gain[t] = k
    return pd.DataFrame({"beta": beta_prior, "beta_post": beta_post, "var": variance,
                         "innovation": innovation, "gain": gain}, index=df.index)


def effective_window(delta: float, obs_var: float, x_var: float = 1.0) -> float:
    """The rolling window a Kalman filter is roughly equivalent to.

    A Kalman filter with a random-walk state is an exponentially-weighted estimator in steady
    state, and its steady-state gain implies a half-life. Converting that to an equivalent
    equal-weighted window makes the comparison with rolling OLS concrete: a reader can see that
    a filter is not doing something categorically different, it is choosing the weighting
    scheme that the window length was guessing at.
    """
    q = delta / max(1.0 - delta, 1e-12) * obs_var
    # steady-state P solves P = P + q - (P x)^2 / (x^2 P + r); with x_var normalised:
    disc = q * q + 4 * q * obs_var / max(x_var, 1e-12)
    p_ss = 0.5 * (q + np.sqrt(disc))
    k_ss = p_ss / (p_ss + obs_var / max(x_var, 1e-12))
    if k_ss <= 0 or k_ss >= 1:
        return np.nan
    return float(2.0 / k_ss - 1.0)      # the EWMA-to-SMA equivalence


# --------------------------------------------------------------------------- #
# Grading them
# --------------------------------------------------------------------------- #
def tracking_error(estimate: pd.Series, truth: pd.Series) -> dict:
    """How closely an estimator follows a known hedge ratio."""
    df = pd.concat([estimate.rename("e"), truth.rename("t")], axis=1,
                   sort=False).dropna()
    if len(df) < 30:
        return {"n": int(len(df))}
    err = df["e"] - df["t"]
    return {"n": int(len(df)), "rmse": float(np.sqrt((err ** 2).mean())),
            "mae": float(err.abs().mean()), "bias": float(err.mean()),
            "correlation": float(df["e"].corr(df["t"])),
            "estimate_vol": float(df["e"].diff().std(ddof=1)),
            "truth_vol": float(df["t"].diff().std(ddof=1)),
            "excess_movement": float(df["e"].diff().std(ddof=1)
                                     / max(df["t"].diff().std(ddof=1), 1e-12))}


def spread_series(y: pd.Series, x: pd.Series, beta: pd.Series) -> pd.Series:
    """The hedged spread ``y - beta * x``, using a strictly prior beta."""
    df = pd.concat([y.rename("y"), x.rename("x"), beta.rename("b")], axis=1,
                   sort=False).dropna()
    return (df["y"] - df["b"] * df["x"]).rename("spread")


def spread_quality(spread: pd.Series) -> dict:
    """Is the hedged spread actually more stationary than the raw series?

    The measurement a pairs trader cares about, expressed three ways: the variance of the
    spread (lower is a better hedge), its autocorrelation half-life (shorter means it reverts
    faster), and a variance-ratio statistic that is below 1 for a mean-reverting series and
    at 1 for a random walk.
    """
    s = spread.dropna()
    n = len(s)
    if n < 200:
        return {"n": int(n)}
    lag1 = float(s.autocorr(1))
    hl = float(-np.log(2) / np.log(abs(lag1))) if 0 < abs(lag1) < 1 else np.nan
    out = {"n": int(n), "std": float(s.std(ddof=1)), "autocorr_1": lag1,
           "halflife": hl, "mean": float(s.mean())}
    for q in (5, 21):
        if n > q * 4:
            vr = float(s.diff(q).var(ddof=1) / (q * s.diff().var(ddof=1)))
            out[f"variance_ratio_{q}"] = vr
    return out


def spread_trade(y_ret: pd.Series, x_ret: pd.Series, beta: pd.Series,
                 entry_z: float = 2.0, exit_z: float = 0.5, lookback: int = 60,
                 cost_bps: float = 5.0) -> dict:
    """Trade the hedged spread on a *z*-score band, charging the rebalancing the hedge implies.

    The cost model is the point of this function. A hedge ratio that moves has to be *traded* to
    be maintained: every change in beta is a trade in the second leg, whether or not the spread
    position itself changed. An adaptive estimator therefore pays for its adaptiveness, and a
    comparison that charges only entry and exit costs will always flatter it.
    """
    df = pd.concat([y_ret.rename("y"), x_ret.rename("x"), beta.rename("b")], axis=1,
                   sort=False).dropna()
    if len(df) < lookback * 3:
        return {"n": int(len(df))}
    sp = df["y"] - df["b"] * df["x"]
    cum = sp.cumsum()
    mu = cum.rolling(lookback).mean()
    sd = cum.rolling(lookback).std(ddof=1)
    z = ((cum - mu) / sd).shift(1)

    pos = np.zeros(len(df))
    state = 0.0
    for i, zi in enumerate(z.to_numpy()):
        if not np.isfinite(zi):
            state = 0.0
        elif state == 0.0:
            if zi > entry_z:
                state = -1.0
            elif zi < -entry_z:
                state = 1.0
        elif abs(zi) < exit_z:
            state = 0.0
        pos[i] = state
    position = pd.Series(pos, index=df.index)

    gross = position * sp
    # Turnover: the spread leg trades when the position flips, and the hedge leg trades
    # whenever beta moves while a position is open.
    pos_turn = position.diff().abs().fillna(0.0) * (1 + df["b"].abs())
    hedge_turn = (position.abs() * df["b"].diff().abs()).fillna(0.0)
    cost = (pos_turn + hedge_turn) * cost_bps / 1e4
    net = (gross - cost).dropna()
    years = len(net) / TRADING_DAYS
    sd_n = float(net.std(ddof=1))
    return {"n": int(len(net)), "years": float(years),
            "gross_ann": float(gross.mean() * TRADING_DAYS),
            "net_ann": float(net.mean() * TRADING_DAYS),
            "cost_ann": float(cost.mean() * TRADING_DAYS),
            "vol": sd_n * np.sqrt(TRADING_DAYS),
            "sharpe": float(net.mean() / sd_n * np.sqrt(TRADING_DAYS))
            if sd_n > 0 else np.nan,
            "time_in_market": float((position != 0).mean()),
            "n_trades": int(position.diff().abs().sum() / 2),
            "hedge_turnover_ann": float(hedge_turn.mean() * TRADING_DAYS),
            "returns": net}


def compare_estimators(y: pd.Series, x: pd.Series, y_ret: pd.Series, x_ret: pd.Series,
                       windows=(20, 60, 120, 250), deltas=(1e-3, 1e-2, 1e-1, 3e-1),
                       cost_bps: float = 5.0) -> pd.DataFrame:
    """Every estimator, graded on both spread quality and traded performance."""
    rows = []
    static = static_hedge_ratio(y_ret, x_ret)
    if np.isfinite(static):
        b = pd.Series(static, index=y_ret.index)
        q = spread_quality(spread_series(y_ret, x_ret, b).cumsum())
        t = spread_trade(y_ret, x_ret, b, cost_bps=cost_bps)
        rows.append({"estimator": "static OLS (look-ahead)", "beta_mean": static,
                     "beta_vol": 0.0, **_pick(q, t)})
    for w in windows:
        b = rolling_hedge_ratio(y_ret, x_ret, w)
        if b.dropna().empty:
            continue
        q = spread_quality(spread_series(y_ret, x_ret, b).cumsum())
        t = spread_trade(y_ret, x_ret, b, cost_bps=cost_bps)
        rows.append({"estimator": f"rolling {w}d", "beta_mean": float(b.mean()),
                     "beta_vol": float(b.diff().std(ddof=1)), **_pick(q, t)})
    for d in deltas:
        kf = kalman_hedge_ratio(y_ret, x_ret, delta=d)
        if kf.empty:
            continue
        b = kf["beta"]
        q = spread_quality(spread_series(y_ret, x_ret, b).cumsum())
        t = spread_trade(y_ret, x_ret, b, cost_bps=cost_bps)
        rows.append({"estimator": f"Kalman delta={d:g}", "beta_mean": float(b.mean()),
                     "beta_vol": float(b.diff().std(ddof=1)), **_pick(q, t)})
    return pd.DataFrame(rows).set_index("estimator")


def _pick(q: dict, t: dict) -> dict:
    return {"spread_std": q.get("std", np.nan), "halflife": q.get("halflife", np.nan),
            "variance_ratio_21": q.get("variance_ratio_21", np.nan),
            "gross_ann": t.get("gross_ann", np.nan), "net_ann": t.get("net_ann", np.nan),
            "cost_ann": t.get("cost_ann", np.nan), "sharpe": t.get("sharpe", np.nan),
            "n_trades": t.get("n_trades", 0),
            "hedge_turnover": t.get("hedge_turnover_ann", np.nan)}


def synthetic_pair(n: int = 4000, beta_start: float = 1.0, beta_vol: float = 0.002,
                   noise: float = 0.006, x_vol: float = 0.012,
                   seed: int = 998) -> dict:
    """Two return series whose true hedge ratio random-walks with a known variance.

    ``beta_vol = 0`` gives a constant relationship, where a static estimator is optimal and any
    adaptive one is fitting noise. Turning it up makes adaptation genuinely valuable. Because
    the truth is returned, estimators can be graded against it rather than against each other.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2006-01-02", periods=n)
    beta = beta_start + np.cumsum(rng.normal(0, beta_vol, n))
    x = rng.normal(0, x_vol, n)
    y = beta * x + rng.normal(0, noise, n)
    return {"y": pd.Series(y, index=idx, name="y"),
            "x": pd.Series(x, index=idx, name="x"),
            "beta": pd.Series(beta, index=idx, name="beta")}


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Confirmed** if the Kalman filter tracks a *known* moving hedge ratio with
      lower RMSE than the best rolling window **and** produces a tighter spread on the real
      pairs; **Partial** if it wins one; **Busted** if a rolling window is at least as good on
      both.
    - **Tradability**: **Useful** if the best Kalman configuration beats the best rolling one on
      net Sharpe after the hedge-rebalancing cost is charged; **Partial** if it wins gross and
      loses net; **Mirage** if it loses both.
    """
    tracks = h["kalman_rmse"] < h["best_rolling_rmse"]
    tighter = h["kalman_wins_spread"] > 0.5
    signal = ("Confirmed" if (tracks and tighter)
              else ("Partial" if (tracks or tighter) else "Busted"))
    if h["kalman_net_sharpe"] > h["rolling_net_sharpe"]:
        trad = "Useful"
    elif h["kalman_gross_sharpe"] > h["rolling_gross_sharpe"]:
        trad = "Partial"
    else:
        trad = "Mirage"
    return {
        "signal": signal,
        "signal_why": (
            f"Graded against a hedge ratio that is **known** because it was planted, the Kalman "
            f"filter tracked it with an RMSE of **{h['kalman_rmse']:.4f}** against the best "
            f"rolling window's {h['best_rolling_rmse']:.4f} (a {h['best_rolling_window']}-day "
            f"window; shorter ones were too noisy and longer ones too slow). The reason is "
            f"visible in one diagnostic: the filter's estimate moved "
            f"**{h['kalman_excess_movement']:.2f}×** as much as the truth did, against "
            f"{h['rolling_excess_movement']:.2f}× for the rolling window — adaptiveness without "
            f"the thrashing. On the {h['n_pairs']} real pairs the filter produced the tighter "
            f"spread in **{h['kalman_wins_spread']:.0%}** of them. The control matters as much "
            f"as the result: on the pairs that should barely move at all "
            f"({h['static_pair']}), the filter's advantage {h['static_verdict']}, which is what "
            f"distinguishes an estimator that adapts from one that merely wobbles."),
        "trad_why": (
            f"Tracking is not trading, and the gap between them is the hedge-rebalancing cost "
            f"that most comparisons omit. A hedge ratio that follows the truth has to be "
            f"*traded* to be maintained: every move in beta is a trade in the second leg, "
            f"whether or not the spread position changed. Charged properly, the filter's annual "
            f"hedge turnover was **{h['kalman_turnover']:.2f}** against the rolling window's "
            f"{h['rolling_turnover']:.2f}, costing {h['kalman_cost']:.2%} a year versus "
            f"{h['rolling_cost']:.2%}. Gross, the filter earned a Sharpe of "
            f"{h['kalman_gross_sharpe']:.2f} against {h['rolling_gross_sharpe']:.2f}; **net, "
            f"{h['kalman_net_sharpe']:.2f} against {h['rolling_net_sharpe']:.2f}**. And a "
            f"caveat that outranks all of it: the best pairs-trade Sharpe here is "
            f"{h['best_sharpe']:.2f} across {h['years']:.0f} years, which is not a business."),
        "trad": trad,
        "one_sentence": (
            f"A Kalman filter tracks a moving hedge ratio {h['best_rolling_rmse'] / max(h['kalman_rmse'], 1e-9):.1f}× "
            f"more accurately than the best rolling window — and once you charge for the "
            f"rebalancing that accuracy requires, the trading advantage is "
            f"{h['kalman_net_sharpe'] - h['rolling_net_sharpe']:+.2f} of Sharpe."),
    }
