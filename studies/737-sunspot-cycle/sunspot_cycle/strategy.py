"""The event-study / regime engine and its honest controls — Study 737 (Sunspot-Cycle).

The claim under test, steelmanned: **the ~11-year solar/sunspot cycle drives stock
returns.** In its Jevons (1875) form the sunspot cycle drove harvests → trade cycles →
markets; in its modern folklore form the equity market simply "runs on an 11-year solar
clock" — high solar activity coincides with (or leads) good returns, low activity with
bad. If true, that is a *free, century-long, publicly-broadcast* calendar for timing
equities.

The machinery (one execution convention documented throughout):

* ``monthly_returns`` — simple month-over-month price-only returns of ^GSPC.
* ``abnormal_returns`` — constant-mean model (Brown & Warner 1985): return minus its own
  full-sample mean, so a "solar" effect is measured against the market's ordinary drift,
  not on top of it.
* ``forward_return`` / ``turning_point_stats`` — the PRIMARY, independent-events test.
  For each solar minimum and each solar maximum, the cumulative forward price-only
  return over the next ``horizon`` months. The unit of analysis is one number per
  turning point (turning points are ≈ 5–6 years apart → independent, non-overlapping),
  so a **one-sample t across events** is the right test — not a HAC regression on an
  autocorrelated monthly panel. Maxima vs minima are contrasted with a Welch t.
  **Generosity, stated loudly:** this uses the *true* turning-point dates, i.e. perfect
  hindsight knowledge of the cycle you would never possess live.
* ``regime_split`` — the classic "high-activity months vs low-activity months" cut on
  the labelled proxy, with a **circular block bootstrap** of the difference (months are
  autocorrelated; the resampling unit must respect that, unlike the independent-event
  test above).
* ``phase_regression`` — regress monthly abnormal returns on ``cos φ`` and ``sin φ`` of
  the solar phase: the direct "is there an 11-year sinusoid in returns?" test, with a
  Newey-West (HAC) covariance and an R² that says how much variance the cycle explains.
* ``placebo_regime_spread`` — the falsification control: recompute the high−low regime
  spread on **random** cycle calendars (phases shifted by a random offset), thousands of
  times; a real effect must sit in the tail.
* ``solar_timer`` — the tradable overlay. Long the index in the rising (min→max) half of
  the cycle, in cash in the falling half — acting on the proxy phase **lagged by
  ``smooth_lag`` months** (the SILSO smoothing lag: you only *know* the current phase
  once the smoothed sunspot number is published, ≈ 6 months late). One documented lag,
  costs one-way × NAV per switch, long-or-cash (no shorting, no borrow), vs buy-and-hold.

Costs are one-way × NAV per rebalance; the overlay is long-or-flat (never short).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #
def monthly_returns(monthly_close: pd.Series) -> pd.Series:
    """Simple month-over-month price-only returns."""
    return monthly_close.pct_change()


def abnormal_returns(ret: pd.Series) -> pd.Series:
    """Abnormal return = monthly return minus its own full-sample mean (constant mean).

    Demeaning removes equities' ordinary up-drift so a "solar" effect is not just the
    market rising over the decades a particular phase happened to cover.
    """
    return ret - ret.mean(skipna=True)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> tuple[float, float]:
    """Mean and one-sample t-stat of ``x`` (events treated as independent)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2 or x.std(ddof=1) == 0:
        return float(np.nan if n == 0 else x.mean()), float("nan")
    se = x.std(ddof=1) / np.sqrt(n)
    return float(x.mean()), float(x.mean() / se)


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson (1927) score interval for a binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def newey_west_t(y: np.ndarray, X: np.ndarray, lags: int = 12) -> np.ndarray:
    """OLS of y on X (design already includes an intercept column); HAC (Newey-West) t's.

    Returns the vector of t-statistics, one per column of X. Used for the phase
    regression, where the residuals of a monthly series are autocorrelated and a plain
    OLS t would overstate significance.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    # Newey-West meat matrix
    S = (resid[:, None] * X).T @ (resid[:, None] * X)
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        u = resid[:, None] * X
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        return beta / se


# --------------------------------------------------------------------------- #
# PRIMARY — forward return around independent turning points
# --------------------------------------------------------------------------- #
def forward_return(monthly_close: pd.Series, event_date: pd.Timestamp,
                   horizon: int = 12) -> float | None:
    """Cumulative price-only return from the month-end on/after ``event_date`` to
    ``horizon`` months later. ``None`` if the window runs off the tape.

    ``searchsorted`` snaps the turning point (a month label) to the first month-end
    on/after it — the study's single documented alignment convention.
    """
    idx = monthly_close.index
    ev = pd.Timestamp(event_date)
    if ev < idx[0]:                       # event predates the tape — never snap forward to the start
        return None
    pos = idx.searchsorted(ev)
    if pos >= len(idx) or pos + horizon >= len(idx):
        return None
    return float(monthly_close.iat[pos + horizon] / monthly_close.iat[pos] - 1.0)


def turning_point_stats(monthly_close: pd.Series, tps: pd.DataFrame,
                        horizon: int = 12) -> dict:
    """Forward ``horizon``-month returns after solar maxima vs minima.

    Each turning point contributes one forward return (independent events). Reports the
    mean + one-sample t for maxima, for minima, and the Welch t of the (max − min)
    contrast — the direct Jevons prediction being max-forward > min-forward.
    """
    fwd = {"max": [], "min": []}
    for _, row in tps.iterrows():
        r = forward_return(monthly_close, row["date"], horizon)
        if r is not None:
            fwd[row["kind"]].append(r)
    mx = np.array(fwd["max"], dtype=float)
    mn = np.array(fwd["min"], dtype=float)
    mx_mean, mx_t = one_sample_t(mx)
    mn_mean, mn_t = one_sample_t(mn)
    return {
        "n_max": mx.size, "max_mean": mx_mean, "max_t": mx_t,
        "n_min": mn.size, "min_mean": mn_mean, "min_t": mn_t,
        "diff_mean": (mx_mean - mn_mean), "welch_t": welch_t(mx, mn),
        "max_fwd": mx, "min_fwd": mn,
    }


def forward_placebo(monthly_close: pd.Series, n_events: int, horizon: int = 12,
                    n_draws: int = 5000, seed: int = 737) -> np.ndarray:
    """Mean forward ``horizon``-month return of ``n_events`` RANDOM month-ends, ``n_draws``
    times. Places a turning-point group's mean forward return in context: the market
    drifts up, so *any* set of dates earns a positive forward return — the question is
    whether the solar dates beat a random calendar of the same size.
    """
    idx = monthly_close.index
    n = len(idx)
    hi = n - horizon - 1
    if hi <= 0 or n_events <= 0:
        return np.array([])
    px = monthly_close.to_numpy()
    rng = np.random.default_rng(seed)
    out = np.empty(n_draws)
    for d in range(n_draws):
        locs = rng.integers(0, hi, size=n_events)
        out[d] = float(np.mean(px[locs + horizon] / px[locs] - 1.0))
    return out


# --------------------------------------------------------------------------- #
# Regime split — high-activity vs low-activity months (the classic cut)
# --------------------------------------------------------------------------- #
def regime_split(ret: pd.Series, proxy: pd.DataFrame, q: float = 1 / 3,
                 block: int = 12, n_boot: int = 5000, seed: int = 737) -> dict:
    """Mean monthly return in high-activity vs low-activity months + block-bootstrap CI.

    "High" = proxy activity in the top ``q`` quantile, "low" = bottom ``q``. Because
    consecutive monthly returns are autocorrelated and the regime label is itself highly
    persistent, the difference is given a **circular block bootstrap** CI (block length
    ``block`` months) on the paired (return, is_high, is_low) series — the honest unit,
    not an i.i.d. monthly t. Returns means, the high−low spread (annualised bps too), and
    the bootstrap 95% CI + two-sided p of the spread.
    """
    df = pd.DataFrame({"ret": ret}).join(proxy[["activity"]]).dropna()
    a = df["activity"].to_numpy()
    r = df["ret"].to_numpy()
    hi_thr = np.quantile(a, 1 - q)
    lo_thr = np.quantile(a, q)
    is_hi = a >= hi_thr
    is_lo = a <= lo_thr
    hi_mean = float(r[is_hi].mean())
    lo_mean = float(r[is_lo].mean())
    spread = hi_mean - lo_mean

    n = len(r)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        rr, hh, ll = r[idx], is_hi[idx], is_lo[idx]
        if hh.sum() == 0 or ll.sum() == 0:
            boot[b] = np.nan
            continue
        boot[b] = rr[hh].mean() - rr[ll].mean()
    boot = boot[np.isfinite(boot)]
    lo_ci, hi_ci = np.quantile(boot, [0.025, 0.975])
    p_two = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {
        "n": n, "hi_mean": hi_mean, "lo_mean": lo_mean,
        "spread": spread, "spread_ann_bps": spread * 12 * 1e4,
        "ci": (float(lo_ci), float(hi_ci)), "p_boot": float(min(p_two, 1.0)),
        "n_hi": int(is_hi.sum()), "n_lo": int(is_lo.sum()),
    }


# --------------------------------------------------------------------------- #
# Phase regression — the direct 11-year sinusoid test
# --------------------------------------------------------------------------- #
def phase_regression(ar: pd.Series, proxy: pd.DataFrame, lags: int = 12) -> dict:
    """Regress monthly abnormal returns on (1, cos φ, sin φ); HAC t's + R².

    A genuine 11-year cycle in returns shows up as a jointly-significant (cos, sin) pair
    and a non-trivial R². The amplitude of the fitted sinusoid (in annualised bps) is
    reported so a reader can see how big any "effect" would be even if it were real.
    """
    df = pd.DataFrame({"ar": ar}).join(proxy[["phase"]]).dropna()
    y = df["ar"].to_numpy()
    ph = df["phase"].to_numpy()
    X = np.column_stack([np.ones_like(ph), np.cos(ph), np.sin(ph)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    t = newey_west_t(y, X, lags=lags)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    amp = float(np.hypot(beta[1], beta[2]))          # sinusoid amplitude, monthly
    return {
        "n": len(y), "beta_cos": float(beta[1]), "beta_sin": float(beta[2]),
        "t_cos": float(t[1]), "t_sin": float(t[2]), "r2": r2,
        "amp_month": amp, "amp_ann_bps": amp * 12 * 1e4,
    }


# --------------------------------------------------------------------------- #
# Falsification — random-calendar placebo on the regime spread
# --------------------------------------------------------------------------- #
def placebo_regime_spread(ret: pd.Series, proxy: pd.DataFrame, q: float = 1 / 3,
                          n_draws: int = 2000, seed: int = 737) -> np.ndarray:
    """High−low regime spread under RANDOM cycle calendars (phase circularly shifted).

    The activity series is rolled by a random offset thousands of times, breaking its
    real alignment to returns while preserving its exact 11-year shape and persistence.
    The observed spread must sit in the tail of this null; sitting in the bulk means a
    random 11-year clock reproduces the "effect" just as well.
    """
    df = pd.DataFrame({"ret": ret}).join(proxy[["activity"]]).dropna()
    r = df["ret"].to_numpy()
    a = df["activity"].to_numpy()
    n = len(r)
    rng = np.random.default_rng(seed)
    out = np.empty(n_draws)
    for d in range(n_draws):
        shift = rng.integers(1, n)
        a_s = np.roll(a, shift)
        hi_thr = np.quantile(a_s, 1 - q)
        lo_thr = np.quantile(a_s, q)
        is_hi = a_s >= hi_thr
        is_lo = a_s <= lo_thr
        out[d] = r[is_hi].mean() - r[is_lo].mean()
    return out


def placebo_pvalue(observed: float, placebo: np.ndarray, tail: str = "two") -> float:
    """Empirical p-value of ``observed`` within the placebo draws."""
    if placebo.size == 0 or not np.isfinite(observed):
        return float("nan")
    if tail == "right":
        return float((placebo >= observed).mean())
    if tail == "left":
        return float((placebo <= observed).mean())
    med = np.median(placebo)
    return float((np.abs(placebo - med) >= abs(observed - med)).mean())


# --------------------------------------------------------------------------- #
# The tradable overlay — "solar cycle timing", paying the smoothing lag
# --------------------------------------------------------------------------- #
def solar_timer(monthly_close: pd.Series, proxy: pd.DataFrame,
                smooth_lag: int = 6, cost_bps: float = 0.0) -> dict:
    """Long-or-cash overlay: hold the index in the rising (min→max) half, else cash.

    The phase is only *known* once the smoothed sunspot number is published, ≈ 6 months
    after the fact — so the signal for month ``t`` is the ``rising`` flag as of month
    ``t − smooth_lag`` (one documented lag, applied once; zero look-ahead). Each time the
    position flips, one-way ``cost_bps`` × NAV is charged. Cash earns nothing (a
    conservative, single, explicit assumption — no risk-free-rate double count). Reported
    against buy-and-hold over the identical window; monthly returns, so an annualised
    figure and a monthly-return t (Newey-West) are given.
    """
    ret = monthly_returns(monthly_close)
    sig = proxy["rising"].astype(float).shift(smooth_lag)      # the lag, applied once
    df = pd.DataFrame({"ret": ret, "sig": sig}).dropna()
    pos = df["sig"].to_numpy()                                 # 1 = long, 0 = cash
    r = df["ret"].to_numpy()
    switches = np.abs(np.diff(np.concatenate([[0.0], pos])))
    cost = switches * cost_bps * 1e-4
    timer_ret = pos * r - cost
    bh_ret = r
    n = len(r)
    months_per_year = 12.0
    yrs = n / months_per_year

    def cagr(x):
        g = float(np.prod(1.0 + x))
        return g ** (1.0 / yrs) - 1.0 if g > 0 else float("nan")

    def sharpe(x):
        sd = x.std(ddof=1)
        return float(x.mean() / sd * np.sqrt(months_per_year)) if sd > 0 else float("nan")

    diff = timer_ret - bh_ret
    _, tdiff = one_sample_t(diff)
    return {
        "n": n, "years": yrs,
        "timer_cagr": cagr(timer_ret), "bh_cagr": cagr(bh_ret),
        "timer_sharpe": sharpe(timer_ret), "bh_sharpe": sharpe(bh_ret),
        "exposure": float(pos.mean()), "n_switches": int(switches.sum()),
        "excess_cagr": cagr(timer_ret) - cagr(bh_ret), "t_diff": tdiff,
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(monthly_close: pd.Series, proxy: pd.DataFrame) -> dict:
    """Run the headline regime split + phase regression on a synthetic world."""
    ret = monthly_returns(monthly_close)
    ar = abnormal_returns(ret)
    rs = regime_split(ret, proxy, n_boot=1000)
    pr = phase_regression(ar, proxy)
    return {"regime_spread_ann_bps": rs["spread_ann_bps"], "regime_p": rs["p_boot"],
            "phase_r2": pr["r2"], "t_cos": pr["t_cos"], "t_sin": pr["t_sin"]}
