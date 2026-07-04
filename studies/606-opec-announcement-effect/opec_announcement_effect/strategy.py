"""Statistics for Study 606 — OPEC Announcement Effect.

Everything is numpy + pandas only, deterministic given a seed.

* ``map_event_days``    — meeting date -> first tradable session at-or-after it, per
                          asset (weekend/holiday meetings roll forward; deduped).
* ``vol_stats``         — decision-day |return| and intraday range vs an event-free
                          baseline: Welch t on |r|, Brown-Forsythe spread test,
                          variance ratio, vol multiple + bootstrap CI, and a
                          random-calendar placebo (2,000 seeded draws).
* ``drift_stats``       — signed cumulative drift close(-1) -> close(+k), k = 0..5:
                          per-event one-sample t AND a Newey-West dummy regression on
                          the full daily tape (the HAC t quoted on the Signal axis).
* ``continuation_stats``— the "trade the announcement" rule: sign of the day-0 move,
                          held to close(+k); gross/net of futures costs, plus the
                          one-day-lagged variant (enter close(+1)).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HALO = 5  # sessions on each side of an event day excluded from the baseline


# --------------------------------------------------------------------------- #
# Small inference helpers (no scipy)
# --------------------------------------------------------------------------- #
def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch's t for a mean difference between two independent samples."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return float((a.mean() - b.mean()) / np.sqrt(va + vb))


def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def nw_dummy_t(r: np.ndarray, dummy: np.ndarray, lags: int) -> tuple[float, float]:
    """OLS of r on [1, dummy] with Newey-West (HAC) errors; returns (beta, t_beta)."""
    r = np.asarray(r, float)
    d = np.asarray(dummy, float)
    X = np.column_stack([np.ones_like(d), d])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ r)
    u = r - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    n = len(r)
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = Xu[L:].T @ Xu[:-L]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    return float(beta[1]), float(beta[1] / np.sqrt(cov[1, 1]))


# --------------------------------------------------------------------------- #
# Event mapping
# --------------------------------------------------------------------------- #
def map_event_days(px: pd.DataFrame, dates: pd.DatetimeIndex,
                   max_roll: int = 7) -> np.ndarray:
    """Positions (iloc) of each meeting's day 0 on this asset's tape.

    Day 0 = the first session at-or-after the decision date (weekend meetings roll to
    Monday; the 2014 Thanksgiving meeting rolls to the next CL session). Meetings
    before the tape starts, after it ends, or more than ``max_roll`` calendar days from
    the next session (a data gap, not a roll) are dropped; duplicates are deduped.
    """
    idx = px.index
    pos = []
    for d in dates:
        i = int(idx.searchsorted(d))
        if i >= len(idx):
            continue
        if (idx[i] - d).days > max_roll:
            continue
        pos.append(i)
    return np.unique(np.asarray(pos, dtype=int))


def _returns(px: pd.DataFrame) -> np.ndarray:
    return px["Close"].pct_change().to_numpy()


def _ranges(px: pd.DataFrame) -> np.ndarray:
    """Intraday range (High-Low) / previous Close — a spread measure per session."""
    prev = px["Close"].shift(1)
    return ((px["High"] - px["Low"]) / prev).to_numpy()


def _baseline_mask(n: int, ev: np.ndarray, halo: int = HALO) -> np.ndarray:
    """True on sessions further than ``halo`` sessions from every event day 0."""
    mask = np.ones(n, dtype=bool)
    for e in ev:
        mask[max(0, e - halo):min(n, e + halo + 1)] = False
    mask[0] = False  # first return/range is NaN
    return mask


# --------------------------------------------------------------------------- #
# 1) Volatility on decision days
# --------------------------------------------------------------------------- #
def vol_stats(px: pd.DataFrame, dates: pd.DatetimeIndex, halo: int = HALO,
              n_boot: int = 5000, n_placebo: int = 2000, seed: int = 606) -> dict:
    """Decision-day volatility vs the event-free baseline.

    * ``vol_multiple``  — mean |r| on day 0 / mean |r| on baseline days, with a
      bootstrap 95% CI (events resampled i.i.d. — they are months apart; the baseline
      resampled in circular blocks of 10 to respect vol clustering).
    * ``welch_t_abs``   — Welch t of |r| event vs baseline.
    * ``bf_t``          — Brown-Forsythe-style spread test: Welch t on absolute
      deviations from each group's median return.
    * ``var_ratio``     — variance ratio event/baseline.
    * ``p_placebo``     — share of ``n_placebo`` random same-size day sets (drawn from
      the baseline) whose mean |r| beats the event days'.
    * the same level/ratio for the intraday **range**.
    """
    rng = np.random.default_rng(seed)
    r = _returns(px)
    g = _ranges(px)
    ev = map_event_days(px, dates)
    base = _baseline_mask(len(r), ev, halo)
    ok = ~np.isnan(r)
    ev = ev[~np.isnan(r[ev])]
    r_ev, r_b = r[ev], r[base & ok]
    g_ev, g_b = g[ev], g[base & ok & ~np.isnan(g)]
    g_ev = g_ev[~np.isnan(g_ev)]

    abs_ev, abs_b = np.abs(r_ev), np.abs(r_b)
    ratio = abs_ev.mean() / abs_b.mean()

    # bootstrap CI on the vol multiple
    boots = np.empty(n_boot)
    nb, block = len(abs_b), 10
    for i in range(n_boot):
        e = abs_ev[rng.integers(0, len(abs_ev), len(abs_ev))]
        starts = rng.integers(0, nb, nb // block + 1)
        picks = (starts[:, None] + np.arange(block)[None, :]).ravel() % nb
        b = abs_b[picks[:nb]]
        boots[i] = e.mean() / b.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])

    # random-calendar placebo: same number of days drawn from the baseline pool
    base_idx = np.flatnonzero(base & ok)
    hits = 0
    for _ in range(n_placebo):
        fake = np.abs(r[rng.choice(base_idx, size=len(ev), replace=False)])
        if fake.mean() >= abs_ev.mean():
            hits += 1
    p_placebo = hits / n_placebo

    bf = welch_t(np.abs(r_ev - np.median(r_ev)), np.abs(r_b - np.median(r_b)))
    return dict(
        n_events=len(ev), n_base=len(r_b),
        abs_ev_bps=abs_ev.mean() * 1e4, abs_base_bps=abs_b.mean() * 1e4,
        vol_multiple=ratio, ci_lo=lo, ci_hi=hi,
        welch_t_abs=welch_t(abs_ev, abs_b), bf_t=bf,
        var_ratio=r_ev.var(ddof=1) / r_b.var(ddof=1),
        p_placebo=p_placebo,
        rng_ev_bps=g_ev.mean() * 1e4, rng_base_bps=g_b.mean() * 1e4,
        rng_multiple=g_ev.mean() / g_b.mean(),
        welch_t_rng=welch_t(g_ev, g_b),
    )


# --------------------------------------------------------------------------- #
# 2) Signed post-decision drift
# --------------------------------------------------------------------------- #
def drift_stats(px: pd.DataFrame, dates: pd.DatetimeIndex,
                horizons: tuple[int, ...] = (0, 1, 2, 3, 5)) -> list[dict]:
    """Signed cumulative drift close(-1) -> close(+k) after each decision.

    Per horizon: the per-event mean cumulative return + one-sample t (events are
    months apart, so per-event observations are treated as independent — the two
    overlapping windows of April 2020 are the documented exception), AND a Newey-West
    dummy regression of the daily return on "session inside the day-0..+k window"
    (lags = k+5), whose HAC t is the Signal-axis statistic.
    """
    r = _returns(px)
    ev = map_event_days(px, dates)
    n = len(r)
    out = []
    for k in horizons:
        cums = []
        for e in ev:
            if e + k >= n or np.isnan(r[e:e + k + 1]).any():
                continue
            cums.append(np.prod(1.0 + r[e:e + k + 1]) - 1.0)
        cums = np.asarray(cums)
        dummy = np.zeros(n)
        for e in ev:
            dummy[e:min(n, e + k + 1)] = 1.0
        ok = ~np.isnan(r)
        beta, hac_t = nw_dummy_t(r[ok], dummy[ok], lags=k + 5)
        out.append(dict(k=k, n=len(cums), mean_bps=cums.mean() * 1e4,
                        t_event=one_sample_t(cums),
                        beta_day_bps=beta * 1e4, hac_t=hac_t))
    return out


# --------------------------------------------------------------------------- #
# 3) Surprise-continuation ("trade the announcement")
# --------------------------------------------------------------------------- #
def continuation_stats(px: pd.DataFrame, dates: pd.DatetimeIndex, k: int = 5,
                       cost_bps_side: float = 2.0) -> dict:
    """Go with the day-0 sign, hold to close(+k).

    Primary: signal = sign of the day-0 close-to-close return (the decision lands
    intraday, hours before the settle); enter AT the day-0 close, exit at close(+k)
    — the single documented execution convention. Robustness: enter one full session
    later, at close(+1), exit close(+k+1). Costs: ``cost_bps_side`` one-way x 2 legs.
    """
    r = _returns(px)
    ev = map_event_days(px, dates)
    n = len(r)
    day0, fwd, fwd_lag = [], [], []
    for e in ev:
        if e + k + 1 >= n or np.isnan(r[e]):
            continue
        seg = r[e + 1:e + k + 1]
        seg_lag = r[e + 2:e + k + 2]
        if np.isnan(seg).any() or np.isnan(seg_lag).any():
            continue
        day0.append(r[e])
        fwd.append(np.prod(1.0 + seg) - 1.0)
        fwd_lag.append(np.prod(1.0 + seg_lag) - 1.0)
    day0, fwd, fwd_lag = map(np.asarray, (day0, fwd, fwd_lag))
    sgn = np.sign(day0)
    strat = sgn * fwd
    strat_lag = sgn * fwd_lag
    rt_cost = 2.0 * cost_bps_side
    return dict(
        n=len(strat), gross_bps=strat.mean() * 1e4, t=one_sample_t(strat),
        hit=float((strat > 0).mean()),
        net_bps=strat.mean() * 1e4 - rt_cost,
        gross_lag_bps=strat_lag.mean() * 1e4, t_lag=one_sample_t(strat_lag),
        rt_cost_bps=rt_cost,
        corr_day0_fwd=float(np.corrcoef(day0, fwd)[0, 1]),
    )
