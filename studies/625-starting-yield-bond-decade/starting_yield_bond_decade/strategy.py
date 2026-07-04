"""Stats layer for Study 625 (Starting-Yield-Bond-Decade).

The tested rule is a *forecast-accuracy* claim, not an alpha rule: the
month-*t* average 10-year yield predicts the annualised nominal total return
of a constant-maturity 10y Treasury roll over the following 120 months.

Inference, per the desk's bar:

- **Primary** — non-overlapping decades (unit = decade, like study 120):
  plain OLS on ~15 independent points; R², t on the slope, and a t-test of
  slope = 1 / intercept = 0 (the "duration arithmetic" identity).
- **Secondary** — all overlapping monthly windows with a Newey-West (HAC)
  t using 120 lags (the windows share up to 119 months).
- **Phase sweep** — the non-overlapping anchor is arbitrary, so every one of
  the 120 possible monthly phases is reported (min / median / max R²,
  min slope-t) instead of a cherry-picked anchor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# OLS + HAC machinery (numpy only)
# ---------------------------------------------------------------------------
def ols_stats(x: np.ndarray, y: np.ndarray) -> dict:
    """Univariate OLS y = a + b·x: slope/intercept, R², classic t-stats.

    Also reports ``t_slope_vs1`` (H0: slope = 1) and ``t_intercept`` (H0: a = 0)
    — the duration-arithmetic identity says (b, a) ≈ (1, 0).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    b = ((x - xm) * (y - ym)).sum() / sxx
    a = ym - b * xm
    resid = y - a - b * x
    dof = n - 2
    s2 = (resid ** 2).sum() / dof
    se_b = np.sqrt(s2 / sxx)
    se_a = np.sqrt(s2 * (1.0 / n + xm ** 2 / sxx))
    ss_tot = ((y - ym) ** 2).sum()
    r2 = 1.0 - (resid ** 2).sum() / ss_tot
    return {
        "n": n, "slope": b, "intercept": a, "r2": r2,
        "t_slope": b / se_b, "se_slope": se_b,
        "t_slope_vs1": (b - 1.0) / se_b,
        "t_intercept": a / se_a, "se_intercept": se_a,
        "resid_std": float(np.sqrt(s2)),
    }


def newey_west_ols(x: np.ndarray, y: np.ndarray, lags: int) -> dict:
    """OLS with Newey-West (Bartlett kernel) HAC standard error on the slope.

    Used for the overlapping monthly regression, where 10-year windows share
    up to 119 months — plain OLS t's there are fiction; the HAC t is the
    honest secondary statistic.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    # HAC meat: S = sum_l w_l * (Gamma_l + Gamma_l')
    u = X * resid[:, None]
    S = u.T @ u
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        gamma = u[lag:].T @ u[:-lag]
        S += w * (gamma + gamma.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    ym = y.mean()
    r2 = 1.0 - (resid ** 2).sum() / ((y - ym) ** 2).sum()
    return {
        "n": n, "slope": beta[1], "intercept": beta[0], "r2": r2,
        "hac_t_slope": beta[1] / se[1], "hac_se_slope": se[1],
        "hac_t_intercept": beta[0] / se[0], "lags": lags,
    }


# ---------------------------------------------------------------------------
# Decade construction
# ---------------------------------------------------------------------------
def decade_table(panel: pd.DataFrame, xcol: str = "gs10", ycol: str = "fwd10",
                 phase: int = 0, step: int = 120) -> pd.DataFrame:
    """Non-overlapping decade observations: every ``step``-th month from ``phase``.

    Keeps only rows where both the starting yield and the full forward decade
    exist. ``phase = 0`` anchors on the first usable month (1871-01 for the
    real tape).
    """
    df = panel.dropna(subset=[xcol, ycol])
    rows = df.iloc[phase::step]
    return rows[[xcol, ycol]].copy()


def decade_ols(panel: pd.DataFrame, xcol: str = "gs10", ycol: str = "fwd10",
               phase: int = 0) -> dict:
    """Primary stat: OLS over the non-overlapping decade table at ``phase``."""
    tab = decade_table(panel, xcol=xcol, ycol=ycol, phase=phase)
    out = ols_stats(tab[xcol].to_numpy(), tab[ycol].to_numpy())
    out["mae_identity"] = float((tab[ycol] - tab[xcol]).abs().mean())
    out["starts"] = [d.strftime("%Y-%m") for d in tab.index]
    return out


def phase_sweep(panel: pd.DataFrame, xcol: str = "gs10", ycol: str = "fwd10",
                step: int = 120) -> dict:
    """R² and slope-t across ALL monthly phases of the non-overlapping grid.

    The anchor month is arbitrary; the claim must not depend on it. Returns
    min / median / max R², the minimum slope-t, and the n range.
    """
    r2s, ts, ns = [], [], []
    for ph in range(step):
        tab = decade_table(panel, xcol=xcol, ycol=ycol, phase=ph, step=step)
        if len(tab) < 8:
            continue
        s = ols_stats(tab[xcol].to_numpy(), tab[ycol].to_numpy())
        r2s.append(s["r2"])
        ts.append(s["t_slope"])
        ns.append(s["n"])
    r2s, ts = np.array(r2s), np.array(ts)
    return {
        "n_phases": len(r2s),
        "r2_min": float(r2s.min()), "r2_med": float(np.median(r2s)),
        "r2_max": float(r2s.max()),
        "t_min": float(ts.min()), "t_med": float(np.median(ts)),
        "n_min": int(min(ns)), "n_max": int(max(ns)),
    }


def overlapping_hac(panel: pd.DataFrame, xcol: str = "gs10", ycol: str = "fwd10",
                    lags: int = 120) -> dict:
    """Secondary stat: all overlapping monthly windows, NW(120) t on the slope."""
    df = panel.dropna(subset=[xcol, ycol])
    return newey_west_ols(df[xcol].to_numpy(), df[ycol].to_numpy(), lags=lags)


# ---------------------------------------------------------------------------
# Cohort autopsy — the 2020 buyer
# ---------------------------------------------------------------------------
def cohort_autopsy(panel: pd.DataFrame, start: str = "2020-12-01") -> dict:
    """The decade-so-far of a cohort that bought at ``start`` (partial decade).

    Predicted decade = the starting yield. Realised-so-far = annualised total
    return of the constant-maturity roll from end of ``start`` month to the
    last complete month, plus the worst drawdown of the TR index en route.
    Explicitly PARTIAL — never pooled with the completed decades.
    """
    ts = pd.Timestamp(start)
    tr = panel["tr"].dropna()
    y0 = float(panel.loc[ts, "gs10"])
    seg = tr.loc[ts:]
    months = len(seg) - 1
    years = months / 12.0
    realized = (seg.iloc[-1] / seg.iloc[0]) ** (1.0 / years) - 1.0
    dd = float((seg / seg.cummax() - 1.0).min())
    # yield needed over the remaining years for the decade to land on target
    total_target = (1.0 + y0) ** 10.0
    got = seg.iloc[-1] / seg.iloc[0]
    rem_years = 10.0 - years
    required = (total_target / got) ** (1.0 / rem_years) - 1.0 if rem_years > 0 else np.nan
    return {
        "start": ts.strftime("%Y-%m"), "start_yield": y0,
        "months_elapsed": months, "years_elapsed": years,
        "realized_ann": float(realized), "max_drawdown": dd,
        "end": seg.index[-1].strftime("%Y-%m"),
        "required_rest_ann": float(required),
    }
