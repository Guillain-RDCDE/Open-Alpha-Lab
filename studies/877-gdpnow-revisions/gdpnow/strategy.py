"""Strategy + inference for Study 877 — GDPNow Revisions.

The claim, operationalised on a frame of ``rev`` (the day-over-day change of the
current-quarter GDPNow nowcast — a real-time growth surprise) and ``fwd1`` / ``fwd5`` (the
1- and 5-trading-day forward SPY returns anchored to that forecast date):

    An **upward** revision (data came in above the model's running estimate) should predict a
    **higher** forward SPY return; a large **downward** revision should precede **weakness**.

We test it four ways:

  * **Predictive regression.** OLS of the forward return on the revision with a Newey-West
    (HAC) *t* on the slope and the regression *R²* — the direct form of the claim.
  * **Decile conditional test.** Mean forward return after the top-decile (biggest **up**)
    revisions vs the bottom-decile (biggest **down**) revisions, each with a HAC *t*, and a
    Welch *t* on up-minus-down — does the tail of the revision distribution move stocks?
  * **Era cut.** The same regression on two sub-eras (split 2019-01-01) — a real edge should
    not flip sign across halves.
  * **A costed timer.** A long/flat SPY rule that holds the market for one day after an
    up-revision, charged a one-way cost per turn, raced against buy-and-hold.

The honest stamp comes from the real-tape Newey-West *t*: a robust ``|t| >= 2`` that also
holds across eras is "Real"; a significant *wrong-sign* estimate, or one that fails the era
cut / evaporates under the execution lag, is "None".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Inference primitives (shared house kit — cf. study 803)
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


def newey_west_t(x: np.ndarray, lags: int = 5) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# The predictive regression — forward return on the revision, HAC t
# --------------------------------------------------------------------------- #
def ols_nw(x: np.ndarray, y: np.ndarray, lags: int = 5) -> dict:
    """OLS ``y = a + b*x`` with a Newey-West (HAC) covariance on the slope.

    Returns ``beta`` (slope, per 1pp of revision), ``t`` (HAC *t* of the slope), ``r2``,
    ``alpha`` and ``n``. Vectorised; the HAC sandwich uses the Bartlett kernel so serial
    correlation in overlapping forward returns does not overstate significance.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    n = len(x)
    if n < 3:
        return {"n": n, "beta": float("nan"), "t": float("nan"),
                "r2": float("nan"), "alpha": float("nan")}
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    Xe = X * resid[:, None]
    S = Xe.T @ Xe / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        G = Xe[l:].T @ Xe[:-l] / n
        S += w * (G + G.T)
    XtX_inv = np.linalg.inv(X.T @ X / n)
    cov = XtX_inv @ S @ XtX_inv / n
    se = np.sqrt(np.diag(cov))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return {"n": int(n), "beta": float(beta[1]),
            "t": float(beta[1] / se[1]) if se[1] > 0 else float("nan"),
            "r2": float(r2), "alpha": float(beta[0])}


def predict_stats(frame: pd.DataFrame, ycol: str = "fwd1", xcol: str = "rev",
                  lags: int = 5) -> dict:
    """Predictive regression of ``ycol`` on ``xcol`` on the whole frame."""
    r = ols_nw(frame[xcol].to_numpy(), frame[ycol].to_numpy(), lags=lags)
    r["ycol"], r["xcol"] = ycol, xcol
    return r


# --------------------------------------------------------------------------- #
# Decile conditional test — do the biggest up / down revisions move stocks?
# --------------------------------------------------------------------------- #
def decile_conditional(frame: pd.DataFrame, ycol: str = "fwd1", xcol: str = "rev",
                       q: float = 0.10, nw_lags: int = 5) -> dict:
    """Forward returns after the top-decile (biggest up) vs bottom-decile (biggest down)
    revisions, each with a HAC *t*, plus a Welch *t* on up-minus-down and the base rate."""
    sub = frame[[xcol, ycol]].dropna()
    x = sub[xcol].to_numpy(dtype=float)
    y = sub[ycol].to_numpy(dtype=float)
    if len(x) < 20:
        return {"n": len(x)}
    q_lo = float(np.quantile(x, q))
    q_hi = float(np.quantile(x, 1 - q))
    up = y[x >= q_hi]
    down = y[x <= q_lo]
    return {
        "n": int(len(x)),
        "base_bps": float(y.mean() * 1e4),
        "up_thr": q_hi, "down_thr": q_lo,
        "n_up": int(len(up)), "n_down": int(len(down)),
        "up_bps": float(up.mean() * 1e4), "up_t": newey_west_t(up, nw_lags),
        "down_bps": float(down.mean() * 1e4), "down_t": newey_west_t(down, nw_lags),
        "up_minus_down_welch_t": welch_t(up, down),
    }


# --------------------------------------------------------------------------- #
# Era cut
# --------------------------------------------------------------------------- #
def era_stats(frame: pd.DataFrame, split: str = "2019-01-01", ycol: str = "fwd1",
              xcol: str = "rev", lags: int = 5) -> dict:
    """The predictive regression on two eras split at ``split`` (needs a DatetimeIndex)."""
    idx = frame.index
    early = frame[idx < pd.Timestamp(split)]
    late = frame[idx >= pd.Timestamp(split)]
    return {
        "split": split,
        "early": ols_nw(early[xcol].to_numpy(), early[ycol].to_numpy(), lags),
        "late": ols_nw(late[xcol].to_numpy(), late[ycol].to_numpy(), lags),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the regression slope a lucky alignment?
# --------------------------------------------------------------------------- #
def placebo_pvalue(frame: pd.DataFrame, ycol: str = "fwd1", xcol: str = "rev",
                   n_draws: int = 5000, seed: int = 877) -> dict:
    """Shuffle the forward returns against the revisions (breaking any link) and ask how
    often the shuffled |slope| matches or beats the observed |slope| — a two-sided null."""
    sub = frame[[xcol, ycol]].dropna()
    x = sub[xcol].to_numpy(dtype=float)
    y = sub[ycol].to_numpy(dtype=float)
    n = len(x)
    if n < 20:
        return {"obs_beta": float("nan"), "p_value": float("nan"), "n_draws": 0}
    xc = x - x.mean()
    denom = float(xc @ xc)
    obs = float(xc @ (y - y.mean())) / denom
    rng = np.random.default_rng(seed)
    betas = np.empty(n_draws)
    for i in range(n_draws):
        yp = y[rng.permutation(n)]
        betas[i] = float(xc @ (yp - yp.mean())) / denom
    p = float((np.abs(betas) >= abs(obs)).mean())
    return {"obs_beta_bps": obs * 1e4, "placebo_sd_bps": float(betas.std(ddof=1) * 1e4),
            "p_value": p, "n_draws": int(n_draws)}


# --------------------------------------------------------------------------- #
# The costed timer (the Tradability axis)
# --------------------------------------------------------------------------- #
def timer_stats(frame: pd.DataFrame, ycol: str = "fwd1", xcol: str = "rev",
                cost_bps: float = 1.0) -> dict:
    """Long/flat SPY-timing rule: hold the market for the forward window after an
    *up*-revision (``rev > 0``), flat otherwise. A one-way ``cost_bps`` is charged on each
    change of position. Compared to buy-and-hold on the same set of forecast dates."""
    sub = frame[[xcol, ycol]].dropna()
    x = sub[xcol].to_numpy(dtype=float)
    r = sub[ycol].to_numpy(dtype=float)
    n = len(r)
    pos = (x > 0).astype(float)
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
    c = cost_bps / 1e4
    net = pos * r - turn * c

    def _stats(v: np.ndarray) -> dict:
        mu = float(v.mean())
        sd = float(v.std(ddof=1)) if n > 1 else float("nan")
        return {"mean_bps": mu * 1e4, "ann_pct": mu * TRADING_DAYS * 100,
                "sharpe": mu / sd * np.sqrt(TRADING_DAYS) if sd and sd > 0 else float("nan")}

    return {
        "n": int(n), "exposure": float(pos.mean()), "cost_bps": cost_bps,
        "gross": _stats(pos * r), "net": _stats(net), "buy_hold": _stats(r),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(frame: pd.DataFrame, ycol: str = "fwd1", xcol: str = "rev",
                     lags: int = 5) -> dict:
    """Run the headline predictive regression on a synthetic frame."""
    r = predict_stats(frame, ycol=ycol, xcol=xcol, lags=lags)
    return {"n": r["n"], "beta_bps": r["beta"] * 1e4, "t": r["t"], "r2": r["r2"]}
