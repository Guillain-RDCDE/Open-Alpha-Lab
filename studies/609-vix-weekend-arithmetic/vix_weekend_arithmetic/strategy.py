"""Tests + inference for Study 609 — VIX Weekend Arithmetic.

The claim: the VIX carries a *guaranteed* day-of-week seesaw by pure calendar-variance
arithmetic — 30 **calendar** days in the window, variance accruing mostly on **trading**
days, so the quoted index is mechanically marked down as the weekend enters the window
(Thursday/Friday) and pops back once it has passed (Monday). We measure it four ways:

    * **Day-of-week table.** Mean close-to-close Δlog VIX (in %) by the weekday the close
      lands on, with per-weekday one-sample t's.

    * **The headline contrast.** Monday vs Friday: a **Welch t** on the two groups *and* a
      **Newey-West (HAC)** t on the Monday-minus-Friday contrast from a dummy regression on
      the full daily series (Δlog VIX is serially correlated; lags = 10).

    * **The arithmetic race.** Fit the single free parameter of the variance-day-count model
      — the weekend-day variance fraction ``f`` — to the five observed weekday means by
      least squares. ``f = 1`` would mean no arithmetic; ``f = 0`` the full trading-time
      bound (Monday +4.8 %). The fitted ``f`` is the market's implied weekend discount.

    * **The leak test (third axis).** VIXY (tradable VIX-futures ETP): does the Monday pop
      survive in the vehicle? Mean Monday return vs other days (Welch t) and a literal
      "hold over the weekend only" trade — buy Friday close, sell Monday close — net of
      one-way costs. The calendar is known in advance, so entering at Friday's close *is*
      the one documented execution lag (no look-ahead by construction).

Placebo: weekday labels shuffled 20,000 times (seeded), p = P[shuffled Mon−Fri spread ≥
observed]. All randomness is a fixed-seed placebo; every real-tape statistic is otherwise a
deterministic function of the cached series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as _data


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def dlog_pct(close: pd.Series) -> pd.Series:
    """Close-to-close log change in percent, indexed by the day the close lands on."""
    return (np.log(close) - np.log(close.shift(1))).dropna() * 100.0


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) − mean(b) (unequal variance)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def ttest_vs_zero(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def _ols_nw(y: np.ndarray, X: np.ndarray, lags: int = 10):
    """OLS with Newey-West (HAC, Bartlett kernel) covariance. Returns (beta, V)."""
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = Xu[lag:].T @ Xu[:-lag]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return beta, V


# --------------------------------------------------------------------------- #
# Day-of-week table + headline contrast
# --------------------------------------------------------------------------- #
def weekday_table(d: pd.Series) -> list[dict]:
    """Per-weekday mean Δlog VIX (%) with n and a one-sample t (obs within a weekday are
    ~1 week apart, so the plain t is the natural within-group statistic)."""
    out = []
    wd = d.index.weekday
    for k in range(5):
        g = d.values[wd == k]
        out.append({"weekday": _data.WEEKDAYS[k], "n": int(len(g)),
                    "mean_pct": float(g.mean()), "t": ttest_vs_zero(g)})
    return out


def mon_fri_contrast(d: pd.Series, lags: int = 10) -> dict:
    """The headline: Monday vs Friday Δlog VIX.

    * ``welch_t`` — Welch t of the Monday group vs the Friday group.
    * ``hac_t``   — Newey-West t of the Monday-minus-Friday contrast from the dummy
      regression  d = a + b·1{Mon} + c·1{Fri} + e  on the full daily series (base =
      Tue/Wed/Thu), Bartlett lags = ``lags``. The autocorrelation-robust statistic.
    """
    wd = d.index.weekday
    mon = d.values[wd == 0]
    fri = d.values[wd == 4]
    y = d.values.astype(float)
    X = np.column_stack([np.ones(len(y)), (wd == 0).astype(float), (wd == 4).astype(float)])
    beta, V = _ols_nw(y, X, lags=lags)
    c = np.array([0.0, 1.0, -1.0])
    spread_hat = float(c @ beta)
    hac_t = spread_hat / float(np.sqrt(c @ V @ c))
    return {
        "mon_mean": float(mon.mean()), "fri_mean": float(fri.mean()),
        "spread": float(mon.mean() - fri.mean()),
        "n_mon": int(len(mon)), "n_fri": int(len(fri)),
        "welch_t": welch_t(mon, fri),
        "hac_t": float(hac_t),
        "hac_mon_t": float(beta[1] / np.sqrt(V[1, 1])),
        "hac_fri_t": float(beta[2] / np.sqrt(V[2, 2])),
    }


def placebo_spread(d: pd.Series, n_draws: int = 20_000, seed: int = 609) -> dict:
    """Label-shuffle placebo for the Monday-minus-Friday spread.

    Shuffle which days carry the 'Monday' and 'Friday' tags (keeping the group sizes) and
    ask how often a random tagging produces a spread at least as large as the observed one.
    """
    rng = np.random.default_rng(seed)
    wd = d.index.weekday
    y = d.values.astype(float)
    n_mon = int((wd == 0).sum())
    n_fri = int((wd == 4).sum())
    obs = float(y[wd == 0].mean() - y[wd == 4].mean())
    n = len(y)
    draws = np.empty(n_draws)
    for i in range(n_draws):
        sel = rng.choice(n, size=n_mon + n_fri, replace=False)
        draws[i] = y[sel[:n_mon]].mean() - y[sel[n_mon:]].mean()
    return {"obs": obs, "p_value": float((draws >= obs).mean()), "draws": draws}


# --------------------------------------------------------------------------- #
# The arithmetic race — implied weekend fraction
# --------------------------------------------------------------------------- #
def implied_weekend_fraction(d: pd.Series) -> dict:
    """Fit the day-count model's single parameter f to the five observed weekday means.

    Least squares over the (demeaned) five weekday means vs the model's predicted
    day-of-week changes at each f on a 0…1 grid (step 0.0005). Deterministic. Returns the
    fitted f, the RMSE at the fit, and the observed/model tables (in %).
    """
    obs = np.array([r["mean_pct"] for r in weekday_table(d)])
    obs_c = obs - obs.mean()
    grid = np.arange(0.0, 1.0001, 0.0005)
    best_f, best_sse = float("nan"), np.inf
    for f in grid:
        m = np.array([_data.model_weekday_changes(float(f))[k] for k in range(5)])
        sse = float(((obs_c - (m - m.mean())) ** 2).sum())
        if sse < best_sse:
            best_sse, best_f = sse, float(f)
    model_fit = {k: v for k, v in _data.model_weekday_changes(best_f).items()}
    model_full = {k: v for k, v in _data.model_weekday_changes(0.0).items()}
    return {"f": best_f, "rmse_pct": float(np.sqrt(best_sse / 5.0)),
            "obs": {k: float(obs[k]) for k in range(5)},
            "model_at_fit": model_fit, "model_full_arithmetic": model_full}


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #
def by_decade(d: pd.Series) -> list[dict]:
    """Monday/Friday means + Welch t inside each decade bucket."""
    out = []
    for lo, hi, lbl in [(1990, 1999, "1990-1999"), (2000, 2009, "2000-2009"),
                        (2010, 2019, "2010-2019"), (2020, 2026, "2020-2026")]:
        g = d[(d.index.year >= lo) & (d.index.year <= hi)]
        wd = g.index.weekday
        mon, fri = g.values[wd == 0], g.values[wd == 4]
        out.append({"decade": lbl, "n": int(len(g)),
                    "mon_mean": float(mon.mean()), "fri_mean": float(fri.mean()),
                    "spread": float(mon.mean() - fri.mean()),
                    "welch_t": welch_t(mon, fri)})
    return out


def gap_table(d: pd.Series) -> dict:
    """Holiday-robust cut: post-gap vs pre-gap days by *calendar gap*, not weekday label.

    * post-gap — the previous trading close is ≥ 3 calendar days back (Mondays, plus e.g.
      Tuesdays after a Monday holiday): the market-closed span has just LEFT the realized
      past and the day-count window has re-filled with trading days → arithmetic says UP.
    * pre-gap — the NEXT trading day is ≥ 3 calendar days ahead (Fridays, pre-holiday
      days): the closed span is about to dominate the front of the window → arithmetic says
      the mark-down has just happened (these closes carry the DOWN legs).

    The exchange calendar is known in advance, so the pre-gap flag is not look-ahead — it is
    the same public calendar the arithmetic runs on.
    """
    idx = d.index
    prev_gap = np.r_[np.nan, (idx[1:] - idx[:-1]).days]
    next_gap = np.r_[(idx[1:] - idx[:-1]).days, np.nan]
    post = d.values[prev_gap >= 3]
    pre = d.values[next_gap >= 3]
    mid = d.values[(prev_gap < 3) & (next_gap < 3)]
    return {"post_mean": float(post.mean()), "n_post": int(len(post)),
            "pre_mean": float(pre.mean()), "n_pre": int(len(pre)),
            "mid_mean": float(mid.mean()), "n_mid": int(len(mid)),
            "welch_t_post_vs_pre": welch_t(post, pre),
            "welch_t_post_vs_mid": welch_t(post, mid)}


# --------------------------------------------------------------------------- #
# Third axis — does the arithmetic leak into a tradable vehicle?
# --------------------------------------------------------------------------- #
def vixy_weekend(vixy: pd.Series, vix: pd.Series, cost_bps: float = 5.0) -> dict:
    """Can a VIX-futures ETP harvest the index's Monday pop?

    The trade: buy VIXY at Friday's close, sell at Monday's close — the calendar signal is
    known in advance, so the Friday-close entry is the single documented execution lag.
    One round trip per weekend = 2 × ``cost_bps`` one-way. We compare VIXY's mean Monday
    (close-over-weekend) simple return vs its mean return on all other days (Welch t), show
    the ^VIX index's Monday mean on the *matched* sample for contrast, and annualize the
    literal weekend-hold P&L gross and net.
    """
    r = vixy.pct_change().dropna()
    wd = r.index.weekday
    mon = r.values[wd == 0]
    rest = r.values[wd != 0]
    # matched ^VIX Monday mean over the VIXY sample window
    vix_d = dlog_pct(vix.loc[vixy.index.min():vixy.index.max()])
    vix_mon = vix_d.values[vix_d.index.weekday == 0]
    years = (r.index.max() - r.index.min()).days / 365.25
    trades_per_year = len(mon) / years
    c = 2.0 * cost_bps / 1e4                       # full round trip per weekend
    gross_ann = float(mon.mean()) * trades_per_year * 100
    net_ann = (float(mon.mean()) - c) * trades_per_year * 100
    return {
        "n_mon": int(len(mon)), "n_rest": int(len(rest)), "years": float(years),
        "mon_mean_pct": float(mon.mean() * 100), "rest_mean_pct": float(rest.mean() * 100),
        "welch_t": welch_t(mon, rest), "mon_t": ttest_vs_zero(mon),
        "vix_mon_mean_pct": float(vix_mon.mean()),
        "trades_per_year": float(trades_per_year), "cost_bps": float(cost_bps),
        "gross_ann_pct": gross_ann, "net_ann_pct": net_ann,
    }
