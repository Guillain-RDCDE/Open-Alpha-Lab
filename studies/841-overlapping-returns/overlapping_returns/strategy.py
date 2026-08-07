"""Strategy + inference for Study 841 — Overlapping-Returns Inflation.

The claim (Hansen & Hodrick 1980; Hodrick 1992): a predictive regression that forecasts a
**long-horizon** return — the cumulative return over the next ``h`` months — from a predictor known
today, sampled **monthly**, uses returns that *overlap*. Consecutive left-hand-side observations
``y_t = r_{t+1}+...+r_{t+h}`` and ``y_{t+1} = r_{t+2}+...+r_{t+h+1}`` share ``h-1`` months, so the
regression residual is a moving average of order ``h-1``. Ordinary-least-squares standard errors
assume the residuals are serially uncorrelated; they are not. The result is a t-statistic and an R²
that are **grossly inflated** — long-horizon "predictability" that is, under the null, a pure artefact
of the overlap. The size distortion grows with ``h``.

The corrections this study puts against the naive OLS t:

* **Newey-West / HAC** (Newey & West 1987) — a heteroskedasticity- and autocorrelation-consistent
  sandwich estimator with a Bartlett kernel; with lags ≈ ``h-1`` it absorbs the induced MA(h-1)
  structure. It helps a lot, but is known to remain somewhat over-sized for very persistent
  regressors and small samples.
* **Hodrick (1992) "1B" standard errors** — an exact reformulation that pushes the summation onto the
  *regressor* instead of the dependent variable, so the moments are built from **non-overlapping**
  one-period returns. Under the null it is the best-sized of the three (Ang & Bekaert 2007;
  Wei & Wright 2013).

This is distinct from:

* [838-hac-necessity](../../838-hac-necessity/) — HAC on a **strategy's own daily P&L** (the
  autocorrelation of a trading rule's returns), not the long-horizon *predictive-regression* overlap
  trap studied here;
* [835-spurious-regression](../../835-spurious-regression/) — the Granger-Newbold spurious regression
  between two independent **unit-root / trending** series, a different mechanism (non-stationarity),
  not the overlap of a stationary predictor with cumulative returns;
* [346-multiple-testing](../../346-multiple-testing/) — inflated significance from **many
  hypotheses**, corrected by a trial-count haircut, not by an autocorrelation-consistent covariance.

Method:

* **Overlapping returns.** ``overlapping_returns(r, h)`` builds the monthly-sampled cumulative
  ``h``-month forward return, vectorised.
* **The regression.** ``predictive_regression`` runs OLS of ``y_t`` on ``x_t`` and reports the slope,
  the naive (homoskedastic) t, the Newey-West t, the Hodrick 1B t, and the naive R².
* **The Monte Carlo.** ``size_experiment`` repeats the regression over many synthetic worlds and
  reports the **rejection rate** of each t at the 5% level — the size (under the null) or the power
  (under a planted ``beta``). ``horizon_sweep`` traces this as ``h`` grows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# Overlapping long-horizon returns
# --------------------------------------------------------------------------- #
def overlapping_returns(r: np.ndarray | pd.Series, h: int) -> np.ndarray:
    """Monthly-sampled cumulative ``h``-month **forward** return.

    ``y[t] = r[t+1] + r[t+2] + ... + r[t+h]`` (simple sum of log-like monthly returns), for every
    month ``t`` for which the full window exists — so consecutive values overlap by ``h-1`` months.
    Returns an array of length ``len(r) - h`` aligned to ``t = 0 .. len(r)-h-1``. Vectorised via a
    cumulative-sum difference (no per-row loop).
    """
    r = np.asarray(r, dtype=float)
    n = r.size
    if h < 1 or n <= h:
        return np.empty(0)
    cs = np.concatenate([[0.0], np.cumsum(r)])  # cs[k] = sum(r[:k])
    # forward window (t+1 .. t+h] = cs[t+1+h] - cs[t+1]
    t = np.arange(0, n - h)
    return cs[t + 1 + h] - cs[t + 1]


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    """OLS of y on [1, x], closed form. Returns (intercept, slope, residuals)."""
    xbar = x.mean()
    ybar = y.mean()
    xd = x - xbar
    sxx = float(xd @ xd)
    b = float(xd @ (y - ybar) / sxx) if sxx > 0 else 0.0
    a = float(ybar - b * xbar)
    resid = y - (a + b * x)
    return a, b, resid


def ols_slope_t(x: np.ndarray, y: np.ndarray) -> dict:
    """Slope, **naive** homoskedastic-OLS t of the slope, and the naive R².

    This is the number the trap inflates: the classical OLS standard error assumes iid residuals,
    which the overlap violates.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 3:
        return {"slope": float("nan"), "t_naive": float("nan"), "r2": float("nan"), "n": n}
    a, b, u = _ols(x, y)
    sse = float(u @ u)
    xd = x - x.mean()
    sxx = float(xd @ xd)
    if sxx <= 0 or n <= 2:
        return {"slope": b, "t_naive": float("nan"), "r2": float("nan"), "n": n}
    sigma2 = sse / (n - 2)
    se_b = np.sqrt(sigma2 / sxx)
    yd = y - y.mean()
    tss = float(yd @ yd)
    r2 = 1.0 - sse / tss if tss > 0 else float("nan")
    return {"slope": b, "t_naive": (b / se_b if se_b > 0 else float("nan")), "r2": r2, "n": n}


def newey_west_slope_t(x: np.ndarray, y: np.ndarray, lags: int) -> float:
    """Newey-West (HAC, Bartlett kernel) t of the OLS slope of ``y`` on ``[1, x]``.

    A textbook heteroskedasticity- and autocorrelation-consistent sandwich: the "meat" sums the
    autocovariances of the score ``g_t = X_t * u_t`` out to ``lags`` with Bartlett weights. Choosing
    ``lags ≈ h-1`` targets the MA(h-1) structure the overlap induces.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 3:
        return float("nan")
    a, b, u = _ols(x, y)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.array([a, b])
    g = X * u[:, None]                       # score contributions, shape (n, 2)
    S = (g.T @ g) / n                        # lag 0
    L = max(0, int(lags))
    for l in range(1, min(L, n - 1) + 1):
        w = 1.0 - l / (L + 1.0)
        cov = (g[l:].T @ g[:-l]) / n
        S += w * (cov + cov.T)
    XtX_inv = np.linalg.inv(X.T @ X / n)
    V = XtX_inv @ S @ XtX_inv / n            # Cov(beta_hat)
    se_b = np.sqrt(V[1, 1]) if V[1, 1] > 0 else float("nan")
    return float(beta[1] / se_b) if se_b and se_b > 0 else float("nan")


def hodrick_1b_slope_t(x_full: np.ndarray, r_full: np.ndarray, h: int) -> float:
    """Hodrick (1992) "1B" t of the long-horizon slope, built from **non-overlapping** one-period
    returns.

    The long-horizon slope has the algebraic identity (demeaning ``x``)

        b_hat = [ sum_t x~_t * y_t ] / [ sum_t x~_t^2 ]
              = [ sum_s r_s * XS_s ] / [ sum_t x~_t^2 ],

    where ``XS_s = sum_{i=1}^{h} x~_{s-i}`` is the sum of the ``h`` demeaned predictors that
    *precede* month ``s`` (Hodrick's trick: the summation moves from the overlapping dependent
    variable onto the regressor). Because the one-period returns ``r_s`` are serially uncorrelated
    under the null, the numerator is a sum of (near-)uncorrelated terms and its variance needs **no**
    autocovariance corrections — only a heteroskedasticity-robust core:

        Var(b_hat) = [ sum_s XS_s^2 * u_s^2 ] / [ sum_t x~_t^2 ]^2 ,   u_s = r_s - mean(r).

    ``x_full`` and ``r_full`` are the full monthly series (``x[t]`` predicts ``r[t+1]``). Returns the
    Hodrick t of the slope against 0.
    """
    x = np.asarray(x_full, dtype=float)
    r = np.asarray(r_full, dtype=float)
    n = r.size
    if h < 1 or n <= h + 1:
        return float("nan")

    # Regression sample: t = 0 .. N-1 (N = n - h), regressor x[t], y_t = sum r[t+1..t+h].
    N = n - h
    xreg = x[:N]
    xbar = xreg.mean()
    xd = xreg - xbar
    y = overlapping_returns(r, h)            # length N
    sxx = float(xd @ xd)
    if sxx <= 0:
        return float("nan")
    b_hat = float((xd @ (y - y.mean())) / sxx)

    # Summed regressor XS_s = sum of the h demeaned predictors before month s (indices in [0, N-1]).
    # Build via a sliding cumulative sum of the (zero-padded) demeaned regressor.
    xd_pad = np.zeros(n)
    xd_pad[:N] = xd                          # demeaned predictors live on indices 0..N-1
    cs = np.concatenate([[0.0], np.cumsum(xd_pad)])   # cs[k] = sum(xd_pad[:k])
    s = np.arange(n)
    lo = np.clip(s - h, 0, n)                # window start (inclusive) = max(0, s-h)
    hi = np.clip(s, 0, n)                    # window end   (exclusive) = min(n, s)  -> indices < s
    XS = cs[hi] - cs[lo]                     # sum of xd_pad over [s-h, s-1] ∩ [0, N-1]

    u = r - r.mean()
    var_num = float(np.sum((XS ** 2) * (u ** 2)))
    var_b = var_num / (sxx ** 2)
    se_b = np.sqrt(var_b) if var_b > 0 else float("nan")
    return float(b_hat / se_b) if se_b and se_b > 0 else float("nan")


# --------------------------------------------------------------------------- #
# One regression, three t-stats
# --------------------------------------------------------------------------- #
def predictive_regression(x_full: np.ndarray, r_full: np.ndarray, h: int,
                          nw_lags: int | None = None) -> dict:
    """Run the overlapping long-horizon predictive regression at horizon ``h`` and return every
    diagnostic: the slope, the naive R², and the naive / Newey-West / Hodrick t of the slope.

    ``x[t]`` predicts ``y_t = r[t+1]+...+r[t+h]`` (one-period execution lag). ``nw_lags`` defaults to
    ``h-1`` — the exact order of the MA structure the overlap induces.
    """
    x = np.asarray(x_full, dtype=float)
    r = np.asarray(r_full, dtype=float)
    N = r.size - h
    if N < 3:
        return {"h": h, "slope": float("nan"), "r2": float("nan"),
                "t_naive": float("nan"), "t_nw": float("nan"), "t_hodrick": float("nan"), "n": N}
    xreg = x[:N]
    y = overlapping_returns(r, h)
    base = ols_slope_t(xreg, y)
    lags = (h - 1) if nw_lags is None else int(nw_lags)
    t_nw = newey_west_slope_t(xreg, y, lags)
    t_hod = hodrick_1b_slope_t(x, r, h)
    return {
        "h": h,
        "slope": base["slope"],
        "r2": base["r2"],
        "t_naive": base["t_naive"],
        "t_nw": t_nw,
        "t_hodrick": t_hod,
        "n": base["n"],
    }


# --------------------------------------------------------------------------- #
# The Monte Carlo — rejection rates (size under the null, power under an edge)
# --------------------------------------------------------------------------- #
def size_experiment(data_mod, h: int, beta: float = 0.0, rho: float = 0.95,
                    n_months: int = 600, n_sims: int = 2000, base_seed: int = 841,
                    nw_lags: int | None = None, crit: float = 1.96) -> dict:
    """Repeat the horizon-``h`` regression over ``n_sims`` synthetic worlds and report, for each t,
    the **rejection rate** at the two-sided 5% level (``|t| > crit``).

    Under ``beta = 0`` the rejection rate is the test's **size** — it should sit near 0.05; a naive
    OLS rate far above 0.05 is the overlap trap. Under ``beta > 0`` it is the **power**. Also returns
    the mean ``|t|`` per method and the mean naive R² (which inflates with ``h`` even under the null).
    """
    naive_t, nw_t, hod_t, r2s = [], [], [], []
    rej_naive = rej_nw = rej_hod = 0
    valid = 0
    for s in range(n_sims):
        df, _ = data_mod.simulate_world(n_months=n_months, beta=beta, rho=rho, seed=base_seed + s)
        out = predictive_regression(df["x"].to_numpy(), df["r"].to_numpy(), h, nw_lags=nw_lags)
        tn, tw, th, r2 = out["t_naive"], out["t_nw"], out["t_hodrick"], out["r2"]
        if not (np.isfinite(tn) and np.isfinite(tw) and np.isfinite(th)):
            continue
        valid += 1
        naive_t.append(abs(tn)); nw_t.append(abs(tw)); hod_t.append(abs(th)); r2s.append(r2)
        rej_naive += abs(tn) > crit
        rej_nw += abs(tw) > crit
        rej_hod += abs(th) > crit
    v = max(1, valid)
    return {
        "h": h, "beta": beta, "rho": rho, "n_sims": valid, "n_months": n_months,
        "reject_naive": rej_naive / v,
        "reject_nw": rej_nw / v,
        "reject_hodrick": rej_hod / v,
        "mean_abs_t_naive": float(np.mean(naive_t)) if naive_t else float("nan"),
        "mean_abs_t_nw": float(np.mean(nw_t)) if nw_t else float("nan"),
        "mean_abs_t_hodrick": float(np.mean(hod_t)) if hod_t else float("nan"),
        "mean_r2": float(np.mean(r2s)) if r2s else float("nan"),
    }


def horizon_sweep(data_mod, horizons=(1, 3, 6, 12, 24), beta: float = 0.0, rho: float = 0.95,
                  n_months: int = 600, n_sims: int = 2000, base_seed: int = 841,
                  nw_lags: int | None = None) -> pd.DataFrame:
    """Trace the rejection rates and mean naive R² as the horizon ``h`` grows — the signature plot of
    the trap. ``nw_lags=None`` uses ``h-1`` at each horizon (the MA order)."""
    rows = [size_experiment(data_mod, h=h, beta=beta, rho=rho, n_months=n_months,
                            n_sims=n_sims, base_seed=base_seed, nw_lags=nw_lags)
            for h in horizons]
    return pd.DataFrame(rows).set_index("h")
