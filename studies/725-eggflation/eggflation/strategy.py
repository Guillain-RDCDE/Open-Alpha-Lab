"""Strategy + inference for Study 725 — "Eggflation" (trade the egg spike via CALM).

The claim: egg prices explode on every avian-flu cull, so you can *front-run* the spike
in the pure-play egg stock, Cal-Maine (CALM). We test the strongest tradable version:

1. **Contemporaneous vs predictive regression.** Regress CALM's monthly return on the
   egg-price change — *same month* (the look-ahead illusion) and *lagged* (the only
   thing a trader could actually act on, since the retail egg print is public only after
   the month closes). A Newey-West (HAC) *t* on the lagged slope is the Signal statistic;
   ``REAL`` needs |t| ≥ 2 on the *predictive* slope.
2. **A circular-shift placebo.** Re-run the predictive regression on hundreds of
   circularly-shifted egg series to get an empirical null for the *t*-stat — a slope that
   doesn't beat its own placebo is noise, whatever its nominal *t*.
3. **The reverse-lead test.** Regress *next* month's egg-price change on *this* month's
   CALM return — does the **stock lead the government print**? If it does, the published
   egg number is stale news by the time you could trade it.
4. **The egg-momentum timer.** Long CALM when trailing egg momentum is positive (acted on
   with a one-month execution lag), cash otherwise; gross and net of costs, vs
   buy-and-hold CALM and vs SPY, on excess-of-cash Sharpe.

One execution lag, documented: a signal built from data through month *t* trades month
*t+1* — one ``shift``, applied once. Pure numpy/pandas; scipy only for an optional p-value.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MONTHS = 12.0


# --------------------------------------------------------------------------- #
# HAC (Newey-West) OLS
# --------------------------------------------------------------------------- #
def hac_ols(y: np.ndarray, xs: list[np.ndarray], lags: int = 6) -> dict:
    """OLS of ``y`` on a constant + the regressors in ``xs`` with Newey-West (HAC) SEs.

    Returns ``coef`` (incl. intercept at index 0), ``t`` (HAC *t*-stats), ``se`` and
    ``n``. Bartlett kernel, ``lags`` truncation — the honest SE when residuals cluster
    (egg/flu shocks persist for months). The slope of interest is ``t[1]``.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    X = np.column_stack([np.ones(n)] + [np.asarray(x, dtype=float) for x in xs])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    Xe = X * resid[:, None]
    S = Xe.T @ Xe
    for L in range(1, lags + 1):
        if L >= n:
            break
        w = 1.0 - L / (lags + 1.0)
        G = Xe[L:].T @ Xe[:-L]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    return {"coef": beta, "t": t, "se": se, "n": int(n)}


def _p_value(t: float, df: int) -> float:
    """Two-sided p-value for a t-stat (scipy if available, else normal approx)."""
    if not np.isfinite(t):
        return float("nan")
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t), df))
    except Exception:  # pragma: no cover
        return float(math.erfc(abs(t) / math.sqrt(2)))


# --------------------------------------------------------------------------- #
# The regressions — contemporaneous, predictive, reverse-lead
# --------------------------------------------------------------------------- #
def contemporaneous(panel: pd.DataFrame, lags: int = 6) -> dict:
    """Same-month regression ``calm_ret[t] = a + b·egg_chg[t] + e`` (the look-ahead cell).

    This is what people *feel* ("eggs spiked and CALM ripped") — but it is untradable: you
    cannot know the month's egg-price change before the month is over. Reported to expose
    the illusion, not to support a stamp.
    """
    d = panel.dropna(subset=["calm_ret", "egg_chg"])
    r = hac_ols(d["calm_ret"].to_numpy(), [d["egg_chg"].to_numpy()], lags)
    return {"slope": float(r["coef"][1]), "t": float(r["t"][1]),
            "p": _p_value(r["t"][1], r["n"] - 2), "n": r["n"],
            "corr_ret": float(np.corrcoef(d["calm_ret"], d["egg_chg"])[0, 1])}


def predictive(panel: pd.DataFrame, lag: int = 1, control_spy: bool = False,
               lags: int = 6) -> dict:
    """Predictive regression ``calm_ret[t+lag] = a + b·egg_chg[t] (+ c·spy_ret[t+lag]) + e``.

    The tradable test: does the egg-price change *known now* forecast CALM's return ``lag``
    months later? ``control_spy=True`` adds the contemporaneous market to strip common
    beta. One execution lag is baked in by the ``shift``. ``REAL`` needs HAC |t| ≥ 2.
    """
    p = panel.copy()
    p["y"] = p["calm_ret"].shift(-lag)
    # the contemporaneous-to-the-return market leg (same month as y = t+lag)
    p["spy_fwd"] = p["spy_ret"].shift(-lag)
    cols = ["y", "egg_chg"] + (["spy_fwd"] if control_spy else [])
    d = p.dropna(subset=cols)
    xs = [d["egg_chg"].to_numpy()]
    if control_spy:
        xs.append(d["spy_fwd"].to_numpy())
    r = hac_ols(d["y"].to_numpy(), xs, lags)
    return {"slope": float(r["coef"][1]), "t": float(r["t"][1]),
            "p": _p_value(r["t"][1], r["n"] - len(xs) - 1), "n": r["n"], "lag": lag,
            "control_spy": control_spy}


def reverse_lead(panel: pd.DataFrame, lag: int = 1, lags: int = 6) -> dict:
    """Does the **stock lead the print**? ``egg_chg[t+lag] = a + b·calm_ret[t] + e``.

    A significant positive ``b`` means CALM's move *foreshadows* the next retail egg-price
    change — i.e. the government egg number is lagging, stale-by-the-time-you-read-it news.
    That is the mechanism that kills the trade.
    """
    p = panel.copy()
    p["y"] = p["egg_chg"].shift(-lag)
    d = p.dropna(subset=["y", "calm_ret"])
    r = hac_ols(d["y"].to_numpy(), [d["calm_ret"].to_numpy()], lags)
    return {"slope": float(r["coef"][1]), "t": float(r["t"][1]),
            "p": _p_value(r["t"][1], r["n"] - 2), "n": r["n"], "lag": lag}


def circular_shift_placebo(panel: pd.DataFrame, lag: int = 1, n_boot: int = 5000,
                           seed: int = 725, lags: int = 6) -> dict:
    """Empirical null for the predictive *t*: circularly shift the egg series, re-fit.

    Preserves each series' own autocorrelation while destroying any real egg→CALM
    alignment. Returns the observed *t*, the two-sided placebo *p* (share of |null t| ≥
    |obs t|), and the null's 2.5/97.5 percentiles. A signal must beat its own placebo.
    """
    p = panel.copy()
    p["y"] = p["calm_ret"].shift(-lag)
    d = p.dropna(subset=["y", "egg_chg"])
    y = d["y"].to_numpy()
    x = d["egg_chg"].to_numpy()
    n = len(x)
    obs = hac_ols(y, [x], lags)["t"][1]
    rng = np.random.default_rng(seed)
    null = np.empty(n_boot)
    for i in range(n_boot):
        k = int(rng.integers(1, n))
        null[i] = hac_ols(y, [np.roll(x, k)], lags)["t"][1]
    return {"obs_t": float(obs), "p": float((np.abs(null) >= abs(obs)).mean()),
            "lo": float(np.quantile(null, 0.025)), "hi": float(np.quantile(null, 0.975)),
            "n_boot": int(n_boot)}


# --------------------------------------------------------------------------- #
# The egg-momentum timer
# --------------------------------------------------------------------------- #
def egg_momentum_signal(panel: pd.DataFrame, window: int = 3) -> pd.Series:
    """1.0 when trailing ``window``-month egg momentum is positive, else 0.0 — lagged 1 mo.

    The signal is built from egg prices through month *t* and traded in *t+1* (one
    ``shift`` = the single documented execution lag). Returns a 0/1 position aligned to
    the panel.
    """
    mom = panel["egg"].pct_change(window)
    return (mom > 0).astype(float).shift(1).rename("signal")


def egg_momentum_timer(panel: pd.DataFrame, window: int = 3,
                       rf_annual: float = 0.02) -> pd.DataFrame:
    """Long CALM when egg momentum is positive (1-mo lag), cash otherwise.

    Returns a frame with ``signal``, ``timer`` (gross monthly return), ``calm`` (buy-hold)
    and ``spy`` returns aligned. Cash earns a flat ``rf_annual`` monthly. Costs are applied
    separately by :func:`apply_costs`.
    """
    sig = egg_momentum_signal(panel, window)
    rf_m = rf_annual / MONTHS
    out = pd.DataFrame(index=panel.index)
    out["signal"] = sig
    out["timer"] = sig * panel["calm_ret"] + (1.0 - sig) * rf_m
    out["calm"] = panel["calm_ret"]
    out["spy"] = panel["spy_ret"]
    return out.dropna(subset=["signal"])


def apply_costs(timer: pd.DataFrame, cost_bps_one_way: float = 10.0) -> pd.Series:
    """Charge ``cost_bps_one_way`` × NAV on every position change (one-way, per switch).

    A flip in/out of CALM is a one-way trade each side; we debit the cost on the month the
    ``signal`` changes. Returns the net monthly return series.
    """
    switch = timer["signal"].diff().abs().fillna(0.0)
    return (timer["timer"] - switch * cost_bps_one_way / 1e4).rename("net")


# --------------------------------------------------------------------------- #
# Summary stats
# --------------------------------------------------------------------------- #
def summarize(returns: pd.Series, rf_annual: float = 0.02,
              periods_per_year: float = MONTHS) -> dict:
    """Annualised excess-of-cash Sharpe, CAGR, vol and max-drawdown of monthly returns.

    Sharpe is **excess of a flat ``rf_annual``** so a timer that parks in cash is compared
    like-for-like against buy-and-hold. CAGR/vol/MDD describe the raw series.
    """
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: float("nan") for k in ("sharpe", "cagr", "vol", "mdd", "n")}
    rf_m = rf_annual / periods_per_year
    ex = r - rf_m
    sd = ex.std(ddof=1)
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    yrs = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / yrs) - 1.0 if eq.iloc[-1] > 0 else float("nan")
    return {
        "sharpe": float(ex.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else float("nan"),
        "cagr": float(cagr),
        "vol": float(r.std(ddof=1) * math.sqrt(periods_per_year)),
        "mdd": float(dd),
        "n": int(len(r)),
    }


def capm_alpha(calm_ret: pd.Series, spy_ret: pd.Series, lags: int = 6) -> dict:
    """CALM's monthly CAPM alpha/beta vs SPY (HAC) — is the 'egg stock' even market-beta?"""
    j = pd.concat([calm_ret, spy_ret], axis=1, keys=["y", "x"]).dropna()
    r = hac_ols(j["y"].to_numpy(), [j["x"].to_numpy()], lags)
    return {"alpha_m": float(r["coef"][0]), "t_alpha": float(r["t"][0]),
            "beta": float(r["coef"][1]), "n": r["n"]}


def control_recovers(panel: pd.DataFrame, lag: int = 1) -> dict:
    """Positive control: on a planted-lead world the predictive slope must clear |t| ≥ 2."""
    pr = predictive(panel, lag=lag)
    return {"slope": pr["slope"], "t": pr["t"], "recovered": int(abs(pr["t"]) >= 2.0)}
