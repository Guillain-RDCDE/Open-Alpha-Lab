"""Strategy + inference for Study 828 — FX Dollar Factor (DOL) + dollar-timing.

The construction, on the monthly USD-per-foreign panel:

* **DOL — the dollar factor.** Each month, compute the equal-weight average of the
  foreign-currency spot returns (a USD-funded long basket; a *positive* DOL means the
  dollar depreciated against the basket). Optionally attach a fixed carry proxy to make
  it an *excess* return (spot + average forward discount / 12), the LRV DOL. The
  **premium** test is the Newey-West (HAC) *t* of the monthly DOL mean vs zero.

* **Dollar-timing.** LRV argue DOL is only priced *conditionally*: the average forward
  discount times it. True forward discounts need rate data absent from yfinance, so the
  time-varying conditioning variable is a **spot-only proxy** — the trailing 12-month
  DOL (dollar trend). We run the predictive regression ``DOL_{t+1} = a + b·signal_t + e``
  with HAC (Newey-West) standard errors on the slope ``b``, plus a block-shuffle placebo
  that breaks the signal→outcome link, a two-era robustness cut, a costed
  conditional-DOL overlay, and a seeded synthetic positive control.

Execution: one documented lag — the signal is known at the close of month ``t`` and the
position is held over month ``t+1``. No future data enters the signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# DOL factor construction
# --------------------------------------------------------------------------- #
def monthly_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-currency monthly simple spot returns from the month-end USD-per-foreign panel."""
    return panel.sort_index().pct_change().dropna(how="all")


def dol_series(panel: pd.DataFrame, carry: dict[str, float] | None = None) -> pd.DataFrame:
    """Equal-weight dollar factor DOL (index = month-end).

    Columns:
      * ``spot``   — equal-weight mean of the foreign-currency spot returns (USD-funded
                     long basket; positive => dollar depreciated).
      * ``excess`` — ``spot`` + the average forward-discount carry / 12 (LRV DOL as an
                     excess return), when a ``carry`` map is supplied; else NaN.
    """
    ret = monthly_returns(panel)
    spot = ret.mean(axis=1, skipna=True)
    out = pd.DataFrame({"spot": spot})
    if carry is not None:
        afd_ann = np.mean([carry.get(c, 0.0) for c in panel.columns]) / 100.0
        out["excess"] = spot + afd_ann / MONTHS_PER_YEAR
    else:
        out["excess"] = np.nan
    return out.dropna(subset=["spot"])


# --------------------------------------------------------------------------- #
# Inference primitives (shared house set — cf. study 803)
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


def _hac_slope_t(x: np.ndarray, y: np.ndarray, lags: int = 6) -> dict:
    """OLS slope of y on x with Newey-West (HAC) standard error; pure numpy.

    Returns ``{alpha, beta, t_beta, r2, n}``. The HAC covariance uses a Bartlett kernel
    on the score ``z_t = x_t * u_t`` (u = residual), matching statsmodels' ``cov_hac``.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    n = len(x)
    if n < 5:
        return {"alpha": float("nan"), "beta": float("nan"), "t_beta": float("nan"),
                "r2": float("nan"), "n": n}
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    u = y - X @ b
    # HAC meat: sum of score autocovariances, Bartlett-weighted
    Z = X * u[:, None]                      # n x 2 scores
    S = (Z.T @ Z)
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        G = Z[l:].T @ Z[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se_beta = np.sqrt(max(cov[1, 1], 0.0))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((u ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return {"alpha": float(b[0]), "beta": float(b[1]),
            "t_beta": float(b[1] / se_beta) if se_beta > 0 else float("nan"),
            "r2": r2, "n": n}


# --------------------------------------------------------------------------- #
# The DOL premium
# --------------------------------------------------------------------------- #
def dol_stats(dol: pd.DataFrame, leg: str = "spot", nw_lags: int = 6) -> dict:
    """Headline premium stats for the DOL series (default the spot leg)."""
    x = dol[leg].to_numpy(dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    mu = float(x.mean()) if n else float("nan")
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    return {
        "n_months": n,
        "mean_bps": mu * 1e4,
        "ann_pct": mu * MONTHS_PER_YEAR * 100,
        "t_nw": newey_west_t(x, nw_lags),
        "t_1s": one_sample_t(x),
        "sharpe": (mu / sd * np.sqrt(MONTHS_PER_YEAR)) if sd and sd > 0 else float("nan"),
        "vol_ann_pct": sd * np.sqrt(MONTHS_PER_YEAR) * 100 if sd == sd else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Dollar-timing — trailing dollar trend predicts next-month DOL
# --------------------------------------------------------------------------- #
def timing_signal(dol: pd.DataFrame, leg: str = "spot", window: int = 12) -> pd.Series:
    """Spot-only proxy for the LRV average-forward-discount conditioning variable: the
    trailing ``window``-month cumulative DOL (the dollar trend), known at month ``t``.

    A high (positive) value = the dollar has been weak (basket appreciating) over the
    past year. The predictive test asks whether that persists into month ``t+1``.
    """
    r = dol[leg]
    return (1.0 + r).rolling(window, min_periods=window).apply(np.prod, raw=True).sub(1.0).rename("signal")


def timing_regression(dol: pd.DataFrame, leg: str = "spot", window: int = 12,
                      nw_lags: int = 6) -> dict:
    """Predictive regression ``DOL_{t+1} = a + b·signal_t + e`` with HAC slope *t*.

    ``signal_t`` is the trailing-dollar-trend proxy known at the close of month ``t``
    (one lag; no look-ahead). Returns the slope, its Newey-West *t*, R², and n.
    """
    sig = timing_signal(dol, leg, window)
    y = dol[leg].shift(-1)                       # next-month DOL
    d = pd.DataFrame({"x": sig, "y": y}).dropna()
    res = _hac_slope_t(d["x"].to_numpy(), d["y"].to_numpy(), nw_lags)
    res["window"] = window
    return res


def block_shuffle_placebo(dol: pd.DataFrame, leg: str = "spot", window: int = 12,
                          n_perm: int = 1000, block: int = 6, seed: int = 828) -> dict:
    """Block-shuffle placebo for the timing slope.

    Circularly rotate the forward-DOL series in blocks against the signal ``n_perm``
    times; p = share of shuffles whose |slope| >= observed. Block rotation preserves the
    serial structure so the null is honest.
    """
    sig = timing_signal(dol, leg, window)
    y = dol[leg].shift(-1)
    d = pd.DataFrame({"x": sig, "y": y}).dropna().reset_index(drop=True)
    n = len(d)
    if n < 30:
        return {"obs_beta": float("nan"), "p_value": float("nan"), "n_perm": 0}
    xv = d["x"].to_numpy(); yv = d["y"].to_numpy()
    obs = abs(_hac_slope_t(xv, yv)["beta"])
    rng = np.random.default_rng(seed)
    count = 0
    betas = np.empty(n_perm)
    for i in range(n_perm):
        shift = int(rng.integers(block, n - block))
        ys = np.roll(yv, shift)
        b = _hac_slope_t(xv, ys)["beta"]
        betas[i] = b
        if abs(b) >= obs - 1e-15:
            count += 1
    return {"obs_beta": obs, "placebo_beta_sd": float(np.nanstd(betas, ddof=1)),
            "p_value": (count + 1) / (n_perm + 1), "n_perm": n_perm}


# --------------------------------------------------------------------------- #
# Era split
# --------------------------------------------------------------------------- #
def era_stats(dol: pd.DataFrame, split: str = "2015-01-01", leg: str = "spot") -> dict:
    """DOL premium in two eras (early < split <= late)."""
    early = dol[dol.index < split]
    late = dol[dol.index >= split]
    return {"early": dol_stats(early, leg), "late": dol_stats(late, leg), "split": split}


# --------------------------------------------------------------------------- #
# The costed timer
# --------------------------------------------------------------------------- #
def timer_stats(dol: pd.DataFrame, leg: str = "spot", window: int = 12,
                cost_bps: float = 5.0, turnover: float = 0.20) -> dict:
    """Cost two books and compare Sharpe.

    * **Static DOL** — always long the equal-weight foreign basket funded in USD. Monthly
      rebalancing turnover ``turnover`` of the 1.0-NAV book pays one-way ``cost_bps``.
    * **Timed DOL** — long DOL only when the trailing-trend signal (lagged) is positive,
      else flat; a switch trades the full NAV at one-way ``cost_bps``.

    Returns gross/net annualised mean and net Sharpe for both, plus switch count.
    """
    r = dol[leg]
    sig = timing_signal(dol, leg, window).shift(1)     # known at t-1, applied over t
    d = pd.DataFrame({"r": r, "sig": sig}).dropna()
    one_way = cost_bps / 1e4

    # static long basket: monthly rebalance drag = turnover * one_way
    static_gross = d["r"]
    static_net = d["r"] - turnover * one_way

    # timed overlay
    pos = (d["sig"] > 0).astype(float)
    switches = pos.diff().abs().fillna(pos.iloc[0])
    timed_gross = pos * d["r"]
    timed_net = timed_gross - switches * one_way

    def _sh(x):
        x = np.asarray(x, dtype=float)
        s = x.std(ddof=1)
        return float(x.mean() / s * np.sqrt(MONTHS_PER_YEAR)) if s > 0 else float("nan")

    return {
        "n_months": int(len(d)),
        "static_gross_ann_pct": float(static_gross.mean() * MONTHS_PER_YEAR * 100),
        "static_net_ann_pct": float(static_net.mean() * MONTHS_PER_YEAR * 100),
        "static_sharpe_net": _sh(static_net.to_numpy()),
        "static_t_net": one_sample_t(static_net.to_numpy()),
        "timed_gross_ann_pct": float(timed_gross.mean() * MONTHS_PER_YEAR * 100),
        "timed_net_ann_pct": float(timed_net.mean() * MONTHS_PER_YEAR * 100),
        "timed_sharpe_net": _sh(timed_net.to_numpy()),
        "timed_t_net": one_sample_t(timed_net.to_numpy()),
        "switches_per_yr": float(switches.sum() / len(d) * MONTHS_PER_YEAR),
        "frac_invested": float(pos.mean()),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: pd.DataFrame, window: int = 12) -> dict:
    """Run the headline DOL premium + timing regression on a synthetic panel."""
    dol = dol_series(panel, carry=None)
    prem = dol_stats(dol, "spot")
    tim = timing_regression(dol, "spot", window)
    return {"prem_mean_bps": prem["mean_bps"], "prem_t_nw": prem["t_nw"],
            "timing_beta": tim["beta"], "timing_t": tim["t_beta"],
            "n_months": prem["n_months"]}
