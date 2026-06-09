"""The teardown engine — turn the regime panel into the numbers that earn the verdict.

Two questions, in the order an honest investigation asks them:

1. **Is the raw claim even there?** :func:`regime_gap` measures the plain effect the pitch sells:
   are negative-gamma ("amplifier") days more volatile / more trending than positive-gamma
   ("absorber") days? It returns the two group means, their gap, and a HAC-robust t-stat on the
   gap (an OLS of the outcome on the regime dummy, Newey-West errors). On the synthetic this is
   positive by construction — VIX alone guarantees it.

2. **Does the GEX sign add anything over the VIX level — or is it the vol regime relabeled?**
   :func:`partial_over_vix` is the load-bearing move. It nests two regressions, ``y ~ vix`` and
   ``y ~ vix + neg_gamma``, and reports the negative-gamma coefficient *after* VIX is partialled
   out, its HAC t-stat, the incremental R-squared, and the **share of the raw gap that survives**.
   If the surviving coefficient collapses toward zero, the GEX "regime" was the volatility regime
   wearing a trenchcoat — the synthetic's ``beta = 0`` case, and the question the real run answers.

All inference is autocorrelation-robust (Newey-West / Bartlett); the panel is daily and the
regimes are persistent, so the naive OLS t-stat would overstate significance. NumPy only;
deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# HAC (Newey-West) OLS — the one workhorse
# --------------------------------------------------------------------------- #

def _nw_lags(n: int) -> int:
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def hac_ols(y: np.ndarray, X: np.ndarray, lags: int | None = None) -> dict:
    """OLS of ``y`` on ``X`` (include your own intercept column) with Newey-West HAC errors.

    Daily outcomes conditioned on a persistent regime are autocorrelated and heteroskedastic, so
    the long-run variance uses the Bartlett kernel ``w_l = 1 - l/(L+1)``. Returns the coefficient
    vector, HAC standard errors, t-stats, R-squared and ``n``. ``lags=None`` uses the standard
    rule of thumb ``floor(4*(n/100)^(2/9))``.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    if lags is None:
        lags = _nw_lags(n)
    # Meat matrix S = sum x_t x_t' e_t^2 + Bartlett-weighted lead/lag cross terms.
    xe = X * resid[:, None]
    S = xe.T @ xe
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = xe[l:].T @ xe[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    return {"beta": beta, "se": se, "t": t, "r2": r2, "n": n, "lags": lags}


def _xy(panel: pd.DataFrame, y_col: str, with_vix: bool) -> tuple[np.ndarray, np.ndarray]:
    """Build (y, X) dropping NA rows. X = [const, (vix), neg_gamma] — neg_gamma is the last column."""
    cols = ["vix", y_col] if with_vix else [y_col]
    sub = panel[["neg_gamma"] + cols].copy()
    sub["neg_gamma"] = sub["neg_gamma"].astype("boolean")
    sub = sub.dropna()
    y = sub[y_col].to_numpy(float)
    neg = sub["neg_gamma"].to_numpy(dtype=bool).astype(float)
    parts = [np.ones(len(sub))]
    if with_vix:
        parts.append(sub["vix"].to_numpy(float))
    parts.append(neg)
    return y, np.column_stack(parts)


# --------------------------------------------------------------------------- #
# 1. The raw regime gap
# --------------------------------------------------------------------------- #

def regime_gap(panel: pd.DataFrame, y_col: str, lags: int | None = None) -> dict:
    """The plain claim: mean outcome on negative- vs positive-gamma days, with a HAC t on the gap.

    Regresses ``y_col`` on ``[const, neg_gamma]``; the slope is the gap ``mean|neg - mean|pos`` and
    its Newey-West t-stat is the honest significance. Also returns each group mean and ``n``.
    """
    y, X = _xy(panel, y_col, with_vix=False)
    fit = hac_ols(y, X, lags=lags)
    neg = X[:, -1].astype(bool)
    return {
        "mean_pos": float(y[~neg].mean()) if (~neg).any() else float("nan"),
        "mean_neg": float(y[neg].mean()) if neg.any() else float("nan"),
        "gap": float(fit["beta"][-1]),
        "t": float(fit["t"][-1]),
        "n": fit["n"], "n_neg": int(neg.sum()), "lags": fit["lags"],
    }


# --------------------------------------------------------------------------- #
# 2. The load-bearing move — does GEX sign survive the VIX control?
# --------------------------------------------------------------------------- #

def partial_over_vix(panel: pd.DataFrame, y_col: str, lags: int | None = None) -> dict:
    """Nest ``y ~ vix`` inside ``y ~ vix + neg_gamma``: what does the GEX sign add over VIX?

    Returns the raw gap (no control), the **surviving** negative-gamma coefficient after VIX is
    partialled out and its HAC t-stat, the two R-squareds and their increment, and the
    ``survival_share`` = surviving / raw. A survival_share near 0 (and an insignificant t) means
    the regime was the volatility level relabeled — the trenchcoat verdict.
    """
    raw = regime_gap(panel, y_col, lags=lags)

    y0, X0 = _xy(panel, y_col, with_vix=False)              # const + neg_gamma  (already in raw)
    yv, Xv = _xy(panel, y_col, with_vix=True)               # const + vix + neg_gamma
    base = hac_ols(yv, Xv[:, :-1], lags=lags)               # const + vix only
    full = hac_ols(yv, Xv, lags=lags)

    surviving = float(full["beta"][-1])
    return {
        "raw_gap": raw["gap"], "raw_t": raw["t"],
        "surviving_coef": surviving, "surviving_t": float(full["t"][-1]),
        "vix_only_r2": base["r2"], "full_r2": full["r2"],
        "delta_r2": full["r2"] - base["r2"],
        "survival_share": (surviving / raw["gap"]) if raw["gap"] not in (0.0, float("nan")) else float("nan"),
        "n": full["n"], "lags": full["lags"],
    }


def summary(panel: pd.DataFrame, lags: int | None = None) -> dict:
    """Both outcomes at once: the raw gap and the VIX-controlled survival for ``rv`` and ``de``.

    The one call the demo and verify scripts use. ``rv`` (range vol) and ``de`` (directional
    efficiency / trend-vs-chop) each get a :func:`regime_gap` and a :func:`partial_over_vix`,
    so the verdict can read both "is the raw effect there?" and "does it survive VIX?" off one dict.
    """
    out = {"n": int(len(panel.dropna(subset=["neg_gamma", "rv", "de", "vix"])))}
    for y_col in ("rv", "de"):
        out[y_col] = {"raw": regime_gap(panel, y_col, lags=lags),
                      "partial": partial_over_vix(panel, y_col, lags=lags)}
    return out
