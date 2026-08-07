"""Strategy + inference for Study 835 — Spurious Regression (Granger & Newbold 1974).

The claim, at full strength: **regress one independent random walk on another** and OLS
will hand you a large *t*-statistic and a high R² — a "significant relation" that is
pure fiction, manufactured by the nonstationarity of the two series. The demonstration:

* **The pitfall.** Simulate many pairs of *independent* random walks; regress ``y`` on
  ``x`` in **levels**; record the slope *t*-stat and R². The share of pairs with
  ``|t| > 1.96`` — which a correctly sized 5% test would produce ~5% of the time — is
  grossly inflated (and *grows with the sample size*, because the spurious *t* scales
  with √T). The R² is high far more often than any real-null should allow.
* **The fix — first-difference.** Regress ``Δy`` on ``Δx``. The differences of two
  independent random walks are two independent **white-noise** streams, so the
  differenced regression is correctly sized: ``|t| > 1.96`` in ~5% of pairs, R² ~ 0.
* **The other fix — test for cointegration.** Before trusting a levels regression, test
  whether ``y - β x`` is stationary (Engle-Granger). On independent walks the test
  *fails to reject* a unit root in the residual (correctly: no cointegration); on a
  genuinely cointegrated pair it *rejects* (a real long-run relation exists).
* **Specificity.** The same level OLS on two *stationary* series is correctly sized —
  proving the inflation is a property of the **unit root**, not of OLS.

All estimators are closed-form and **vectorised over pairs** (no per-pair Python loop
for the OLS headline); the cointegration leg uses ``statsmodels`` on a modest sub-sample.
The inference primitives (one-sample / Welch / Newey-West HAC / Wilson) mirror the desk's
canonical template (study 803) so the machinery reads the same across the lab.
"""

from __future__ import annotations

import numpy as np

TRADING_DAYS = 252
CRIT_5PCT = 1.959963984540054  # two-sided 5% normal critical value


# --------------------------------------------------------------------------- #
# Vectorised OLS: regress each row of Y on the matching row of X (with intercept)
# --------------------------------------------------------------------------- #
def ols_batch(X: np.ndarray, Y: np.ndarray) -> dict:
    """Row-wise simple OLS ``y = a + b·x + e`` for every pair, fully vectorised.

    ``X`` and ``Y`` are ``(n_pairs, n_obs)``. Returns a dict of ``(n_pairs,)`` arrays:
    ``beta`` (slope), ``t`` (classical OLS *t* on the slope, ``b / se(b)``), ``r2``
    (regression R²), and ``se`` (the slope standard error). Uses the closed-form
    ``se(b) = sqrt( SSE/(n-2) / Sxx )`` — the same textbook OLS *t* Granger & Newbold's
    critique is about (it is precisely this *t* that over-rejects on unit-root data).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = X.shape[1]
    xbar = X.mean(axis=1, keepdims=True)
    ybar = Y.mean(axis=1, keepdims=True)
    xc = X - xbar
    yc = Y - ybar
    Sxx = np.einsum("ij,ij->i", xc, xc)
    Sxy = np.einsum("ij,ij->i", xc, yc)
    Syy = np.einsum("ij,ij->i", yc, yc)
    beta = np.divide(Sxy, Sxx, out=np.zeros_like(Sxy), where=Sxx > 0)
    # SSE = Syy - beta * Sxy  (equivalently Syy - Sxy^2/Sxx)
    sse = Syy - beta * Sxy
    sse = np.clip(sse, 0.0, None)
    dof = max(n - 2, 1)
    sigma2 = sse / dof
    var_b = np.divide(sigma2, Sxx, out=np.full_like(sigma2, np.inf), where=Sxx > 0)
    se = np.sqrt(var_b)
    t = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    r2 = np.divide(Syy - sse, Syy, out=np.zeros_like(Syy), where=Syy > 0)
    return {"beta": beta, "t": t, "r2": r2, "se": se}


def _first_diff(A: np.ndarray) -> np.ndarray:
    return np.diff(np.asarray(A, dtype=float), axis=1)


# --------------------------------------------------------------------------- #
# The headline experiment — level vs differenced regression on a batch of pairs
# --------------------------------------------------------------------------- #
def regression_experiment(X: np.ndarray, Y: np.ndarray, crit: float = CRIT_5PCT) -> dict:
    """Run **level** and **first-differenced** OLS on the same batch of pairs.

    Returns, for each specification, the rejection rate (share with ``|t| > crit``, which
    a valid 5% test caps near 0.05), the mean/median ``|t|``, the mean R², and the share
    of pairs with R² > 0.25. The gap between ``level`` and ``diff`` *is* the pitfall.
    """
    lvl = ols_batch(X, Y)
    dif = ols_batch(_first_diff(X), _first_diff(Y))

    def _summ(res):
        at = np.abs(res["t"])
        return {
            "n_pairs": int(len(at)),
            "reject_rate": float(np.mean(at > crit)),
            "mean_abs_t": float(np.mean(at)),
            "median_abs_t": float(np.median(at)),
            "mean_r2": float(np.mean(res["r2"])),
            "share_r2_gt_25": float(np.mean(res["r2"] > 0.25)),
        }

    return {"level": _summ(lvl), "diff": _summ(dif),
            "level_t": lvl["t"], "diff_t": dif["t"],
            "level_r2": lvl["r2"], "diff_r2": dif["r2"]}


def sample_size_sweep(
    data_mod, n_obs_grid=(50, 125, 250, 500, 1000),
    n_pairs: int = 4000, seed_base: int = 835, crit: float = CRIT_5PCT,
) -> list[dict]:
    """The killer table: level OLS over-rejects *more* as the sample grows; diff stays ~5%.

    For each ``n_obs`` build a fresh batch of independent random walks (seed offset by the
    grid position so each panel is deterministic and disjoint) and report the level and
    differenced rejection rates. The spurious *t* scales with √T, so more data makes the
    level test *worse* — the opposite of the usual "more data → better inference".
    """
    out = []
    for k, n in enumerate(n_obs_grid):
        X, Y = data_mod.independent_walks(n_pairs, n_obs=n, seed=seed_base + 1000 + k)
        ex = regression_experiment(X, Y, crit=crit)
        out.append({
            "n_obs": int(n),
            "level_reject": ex["level"]["reject_rate"],
            "level_mean_abs_t": ex["level"]["mean_abs_t"],
            "level_mean_r2": ex["level"]["mean_r2"],
            "diff_reject": ex["diff"]["reject_rate"],
        })
    return out


# --------------------------------------------------------------------------- #
# The cointegration leg — tell a spurious relation from a genuine one
# --------------------------------------------------------------------------- #
def cointegration_reject_rate(X: np.ndarray, Y: np.ndarray, alpha: float = 0.05) -> dict:
    """Share of pairs where Engle-Granger rejects *no*-cointegration at level ``alpha``.

    Uses ``statsmodels.tsa.stattools.coint`` (Engle-Granger two-step). On independent
    random walks the residual ``y - β x`` has a unit root, so the test should reject only
    ~``alpha`` of the time (correctly: no genuine relation). On a truly cointegrated pair
    the residual is stationary, so the test rejects far more often. Loops in Python (the
    test itself is not vectorisable) — keep ``n_pairs`` modest.
    """
    from statsmodels.tsa.stattools import coint

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    pvals = np.empty(X.shape[0])
    for i in range(X.shape[0]):
        try:
            _, p, _ = coint(Y[i], X[i], trend="c", maxlag=0, autolag=None)
        except Exception:
            p = 1.0
        pvals[i] = p
    return {
        "n_pairs": int(X.shape[0]),
        "reject_rate": float(np.mean(pvals < alpha)),
        "median_pvalue": float(np.median(pvals)),
        "alpha": alpha,
    }


# --------------------------------------------------------------------------- #
# Tradability — can you trade the spurious "relationship"?  (a pairs timer)
# --------------------------------------------------------------------------- #
def pairs_timer(
    X: np.ndarray, Y: np.ndarray, window: int = 60, entry_z: float = 1.0,
    cost_bps: float = 5.0, borrow_bps_yr: float = 50.0, sigma_hint: float = 1.0,
) -> dict:
    """A naive mean-reversion pairs trade on the spurious spread, costed, NO look-ahead.

    A quant who trusts the level regression bets the spread ``s = y - β x`` mean-reverts.
    The honest version re-estimates the hedge ratio ``β`` on a **trailing** ``window``
    (known at close ``t-1``) — *not* the full-sample OLS, which would itself be a
    look-ahead that manufactures reversion (a second spurious result). Each day the
    contrarian position is ``-z_{t-1}`` (capped at ±3σ) where ``z`` is the trailing
    z-score of the spread; the position is held day ``t`` and earns ``pos · Δs``, with
    ``Δs = Δy - β Δx`` a *future* white-noise increment independent of the position. So
    the gross edge is ≈ 0; a one-way ``cost_bps`` on ``|Δpos|`` traded plus short borrow
    on ``|pos|`` push it negative. PnL is scaled to the shock size (``sigma_hint``) so bps
    are cross-pair comparable. Vectorised over pairs; loops only over the ``n_obs`` steps.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    P, n = X.shape
    scale = 1.0 / max(sigma_hint, 1e-12)
    round_trip = cost_bps / 1e4
    borrow_daily = (borrow_bps_yr / 1e4) / 365.0

    daily_gross, daily_net = [], []
    prev_pos = np.zeros(P)
    # beta & z-normalisation come from an estimation window ending at t-2 (rows known at
    # t-1); the SIGNAL point (spread at t-1) is strictly OUT of that window — otherwise the
    # last in-sample OLS residual carries a boundary artefact that fakes reversion.
    for t in range(window + 1, n - 1):
        xw = X[:, t - 1 - window:t - 1]         # rows t-1-window .. t-2, known at close t-1
        yw = Y[:, t - 1 - window:t - 1]
        xbar = xw.mean(axis=1); ybar = yw.mean(axis=1)
        xc = xw - xbar[:, None]; yc = yw - ybar[:, None]
        Sxx = np.einsum("ij,ij->i", xc, xc)
        Sxy = np.einsum("ij,ij->i", xc, yc)
        beta = np.divide(Sxy, Sxx, out=np.zeros(P), where=Sxx > 0)   # trailing hedge ratio
        intercept = ybar - beta * xbar
        s_win = yw - intercept[:, None] - beta[:, None] * xw
        mu = s_win.mean(axis=1)
        sd = s_win.std(axis=1, ddof=1)
        s_sig = Y[:, t - 1] - intercept - beta * X[:, t - 1]         # spread at t-1 (OOS)
        z = np.divide(s_sig - mu, sd, out=np.zeros(P), where=sd > 0)
        pos = -np.clip(z, -3.0, 3.0)
        pos = np.where(np.abs(z) >= entry_z, pos, 0.0)               # held over day t
        # spread change realised t-1 -> t at the FROZEN trailing beta (future increment)
        ds = (Y[:, t] - beta * X[:, t]) - (Y[:, t - 1] - beta * X[:, t - 1])
        gross = pos * ds * scale
        traded = np.abs(pos - prev_pos)
        cost = traded * scale * round_trip + np.abs(pos) * scale * borrow_daily
        daily_gross.append(float(np.mean(gross)))
        daily_net.append(float(np.mean(gross - cost)))
        prev_pos = pos

    g = np.asarray(daily_gross)
    ntt = np.asarray(daily_net)
    nm = float(np.mean(ntt)) if len(ntt) else float("nan")
    gm = float(np.mean(g)) if len(g) else float("nan")
    sd = float(np.std(ntt, ddof=1)) if len(ntt) > 1 else float("nan")
    sharpe = nm / sd * np.sqrt(TRADING_DAYS) if sd and sd > 0 else float("nan")
    return {
        "n_pairs": int(P),
        "gross_bps": gm * 1e4,
        "net_bps": nm * 1e4,
        "cost_bps_per_day": (gm - nm) * 1e4,
        "t_net": one_sample_t(ntt),
        "sharpe_net": sharpe,
        "ann_net_pct": nm * TRADING_DAYS * 100,
    }


# --------------------------------------------------------------------------- #
# Inference primitives (mirrors study 803's template)
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


def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
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
    """Score interval for a binomial share — used on the rejection *rate* (a proportion)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Synthetic control summary — the machinery is unbiased / correctly sized
# --------------------------------------------------------------------------- #
def size_control(data_mod, n_pairs: int = 4000, n_obs: int = 250, phi: float = 0.0,
                 seed: int = 835, crit: float = CRIT_5PCT) -> dict:
    """Level OLS on two INDEPENDENT STATIONARY series — must be correctly sized (~5%).

    This is the specificity proof: the same OLS *t* that over-rejects on random walks
    behaves itself on stationary data. If this fires far above 0.05 the machinery is
    broken; it should land near the nominal size.
    """
    X, Y = data_mod.stationary_pairs(n_pairs, n_obs=n_obs, phi=phi, seed=seed)
    ex = regression_experiment(X, Y, crit=crit)
    return {"reject_rate": ex["level"]["reject_rate"],
            "mean_abs_t": ex["level"]["mean_abs_t"],
            "mean_r2": ex["level"]["mean_r2"], "n_pairs": n_pairs}


__all__ = [
    "TRADING_DAYS", "CRIT_5PCT",
    "ols_batch", "regression_experiment", "sample_size_sweep",
    "cointegration_reject_rate", "pairs_timer", "size_control",
    "one_sample_t", "welch_t", "newey_west_t", "wilson_interval",
]
