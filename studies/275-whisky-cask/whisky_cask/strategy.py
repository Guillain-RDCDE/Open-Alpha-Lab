"""Strategy and analysis layer for Study 275 (Whisky-Cask).

The pitch: "rare whisky casks are an uncorrelated alternative asset that beat the
stock market with lower volatility." We take that claim apart on three axes:

  1. **Raw performance.** Annualised return and naive Sharpe of the appraisal-based
     whisky index vs the S&P 500, plus the correlation that the pitch sells as
     'diversification'.

  2. **The appraisal-smoothing trap (the killer).** Annual appraisal indices are
     serially autocorrelated by construction — last year's mark anchors this year's.
     That *understates* true volatility and inflates the measured Sharpe. We apply
     the Geltner (1991) first-order unsmoothing to recover an estimate of the
     *true* return series and its honest Sharpe. A high raw Sharpe that collapses
     after unsmoothing is the signature of a mirage.

  3. **The cost wedge (the second killer).** The index level is what a *seller of
     the rarest bottles at auction* realises. A retail cask buyer pays a stack of
     frictions the index never sees: a dealer markup on entry (often 30-50%),
     annual warehouse storage + insurance, the Angels' Share (evaporation), and a
     selling commission on exit. We net these out and show what is left.

Inference. With ~16 annual observations the honest test of "whisky beats equities"
is a HAC (Newey-West) t-test on the annual *excess* return (whisky minus S&P). We
report it with the small-sample caveat front and centre: n=16 cannot certify a
REAL signal even if the point estimate is large.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 275


# ---------------------------------------------------------------------------
# Basic performance stats
# ---------------------------------------------------------------------------
def _ann_return(returns: np.ndarray) -> float:
    """Geometric annualised return from a vector of annual simple returns."""
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return float("nan")
    return float(np.prod(1.0 + returns) ** (1.0 / len(returns)) - 1.0)


def _sharpe(returns: np.ndarray, rf: float = 0.0) -> float:
    """Naive (annual-frequency) Sharpe ratio of an annual return vector."""
    returns = np.asarray(returns, dtype=float)
    ex = returns - rf
    sd = ex.std(ddof=1)
    if sd == 0 or len(ex) < 2:
        return float("nan")
    return float(ex.mean() / sd)


def performance_stats(df: pd.DataFrame, rf: float = 0.02) -> dict:
    """Raw whisky-vs-S&P performance: ann return, vol, Sharpe, correlation.

    ``df`` must carry ``whisky_return`` and ``sp500_return`` columns. ``rf`` is the
    annual risk-free rate used for the Sharpe (default 2%).
    """
    w = df["whisky_return"].to_numpy(dtype=float)
    s = df["sp500_return"].to_numpy(dtype=float)

    return {
        "n": int(len(df)),
        "whisky_ann": round(_ann_return(w) * 100, 2),
        "sp_ann": round(_ann_return(s) * 100, 2),
        "whisky_vol": round(float(np.std(w, ddof=1)) * 100, 2),
        "sp_vol": round(float(np.std(s, ddof=1)) * 100, 2),
        "whisky_sharpe": round(_sharpe(w, rf), 3),
        "sp_sharpe": round(_sharpe(s, rf), 3),
        "corr": round(float(np.corrcoef(w, s)[0, 1]), 3),
        "whisky_beats_frac": round(float((w > s).mean()), 3),
    }


# ---------------------------------------------------------------------------
# The appraisal-smoothing trap — Geltner unsmoothing
# ---------------------------------------------------------------------------
def autocorr1(returns: np.ndarray) -> float:
    """Lag-1 autocorrelation of an annual return series."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 3:
        return float("nan")
    r0 = r[:-1] - r[:-1].mean()
    r1 = r[1:] - r[1:].mean()
    denom = np.sqrt((r0 ** 2).sum() * (r1 ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((r0 * r1).sum() / denom)


def geltner_unsmooth(reported: np.ndarray, rho: float | None = None) -> tuple[np.ndarray, float]:
    """Geltner (1991) first-order unsmoothing of an appraisal-based return series.

    Given reported (smoothed) returns ``r_t`` with lag-1 autocorrelation ``rho``,
    the implied *true* return is

        r_true_t = (r_t - rho * r_{t-1}) / (1 - rho)

    which inflates the variance back toward its un-anchored level. If ``rho`` is
    not supplied it is estimated from the data. Returns ``(true_returns, rho)``;
    ``true_returns`` has the same length (the first element is set equal to the
    reported first element, which has no predecessor).
    """
    r = np.asarray(reported, dtype=float)
    if rho is None:
        rho = autocorr1(r)
    if not np.isfinite(rho) or rho >= 0.999:
        rho = 0.0
    out = np.empty_like(r)
    out[0] = r[0]
    if abs(1.0 - rho) < 1e-9:
        return r.copy(), float(rho)
    out[1:] = (r[1:] - rho * r[:-1]) / (1.0 - rho)
    return out, float(rho)


def smoothing_stats(df: pd.DataFrame, rf: float = 0.02) -> dict:
    """Compare the *measured* whisky Sharpe to the *unsmoothed* (honest) Sharpe.

    The appraisal index understates volatility via serial correlation. We estimate
    the lag-1 autocorrelation, unsmooth the series (Geltner 1991), and report how
    much of the headline Sharpe survives once the manufactured smoothness is removed.
    """
    w = df["whisky_return"].to_numpy(dtype=float)
    rho = autocorr1(w)
    w_true, rho_used = geltner_unsmooth(w, rho)

    raw_sharpe = _sharpe(w, rf)
    true_sharpe = _sharpe(w_true, rf)
    raw_vol = float(np.std(w, ddof=1))
    true_vol = float(np.std(w_true, ddof=1))

    return {
        "rho_lag1": round(float(rho), 3),
        "raw_vol_pct": round(raw_vol * 100, 2),
        "unsmoothed_vol_pct": round(true_vol * 100, 2),
        "vol_inflation_x": round(true_vol / raw_vol, 2) if raw_vol > 0 else float("nan"),
        "raw_sharpe": round(raw_sharpe, 3),
        "unsmoothed_sharpe": round(true_sharpe, 3),
        "sharpe_retained_pct": round(true_sharpe / raw_sharpe * 100, 1)
        if raw_sharpe not in (0, float("nan")) and np.isfinite(raw_sharpe) and raw_sharpe != 0
        else float("nan"),
    }


# ---------------------------------------------------------------------------
# The cost wedge — what a retail cask buyer actually keeps
# ---------------------------------------------------------------------------
def net_of_costs(
    df: pd.DataFrame,
    entry_markup: float = 0.40,
    annual_storage_insurance: float = 0.015,
    annual_angels_share: float = 0.020,
    exit_commission: float = 0.10,
) -> dict:
    """Net the appraisal-index gross return down to what a retail cask owner keeps.

    Frictions an index never charges but a real cask owner pays:
      - ``entry_markup``: dealer markup over fair value when you buy (one-off, amortised).
      - ``annual_storage_insurance``: bonded-warehouse storage + insurance, per year.
      - ``annual_angels_share``: evaporative loss of alcohol, per year (the cask
        literally shrinks). Only the value-bearing share is affected, modelled as a
        drag on the asset return.
      - ``exit_commission``: broker/auction commission when you finally sell (one-off).

    We amortise the one-off entry markup and exit commission over the holding period
    (the full sample length) and subtract the recurring drags each year. Returns the
    gross vs net annualised return and the implied annual drag.
    """
    w = df["whisky_return"].to_numpy(dtype=float)
    n = len(w)
    gross_ann = _ann_return(w)

    # Recurring annual drag.
    annual_drag = annual_storage_insurance + annual_angels_share
    net_year = w - annual_drag

    # One-off entry/exit, amortised geometrically over the holding period.
    one_off = (1.0 - exit_commission) / (1.0 + entry_markup)
    one_off_ann = one_off ** (1.0 / n) - 1.0

    net_ann = _ann_return(net_year) + one_off_ann

    return {
        "gross_ann_pct": round(gross_ann * 100, 2),
        "net_ann_pct": round(net_ann * 100, 2),
        "annual_recurring_drag_pct": round(annual_drag * 100, 2),
        "entry_markup_pct": round(entry_markup * 100, 1),
        "exit_commission_pct": round(exit_commission * 100, 1),
        "one_off_drag_ann_pct": round(one_off_ann * 100, 2),
        "total_cost_wedge_pct": round((gross_ann - net_ann) * 100, 2),
    }


# ---------------------------------------------------------------------------
# Inference — HAC (Newey-West) t-test on the annual excess return
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, n_lags: int = 1) -> tuple[float, float]:
    """Newey-West HAC t-stat for H0: mean(x) = 0, with ``n_lags`` lags.

    Returns ``(mean, t_stat)``. Conservative for small samples — the whole point
    is that n=16 cannot deliver a |t| >= 2 even when the point estimate is large.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return float(np.mean(x)) if n else float("nan"), float("nan")
    mu = float(x.mean())
    e = x - mu
    gamma0 = float((e @ e) / n)
    var = gamma0
    for k in range(1, min(n_lags, n - 1) + 1):
        w = 1.0 - k / (n_lags + 1.0)
        gk = float((e[k:] @ e[:-k]) / n)
        var += 2.0 * w * gk
    if var <= 0:
        return mu, float("nan")
    se = np.sqrt(var / n)
    return mu, float(mu / se) if se > 0 else float("nan")


def excess_return_test(df: pd.DataFrame, n_lags: int = 1) -> dict:
    """HAC t-test on the annual whisky-minus-S&P excess return.

    H0: whisky does not beat the S&P on a per-year basis. We report the mean annual
    excess, its HAC t-stat, and (because survivorship/smoothing both bias the index
    upward) flag that even a positive point estimate fails the REAL bar at n=16.
    """
    w = df["whisky_return"].to_numpy(dtype=float)
    s = df["sp500_return"].to_numpy(dtype=float)
    excess = w - s
    mean_excess, t = hac_tstat(excess, n_lags=n_lags)
    return {
        "n": int(len(df)),
        "mean_excess_pct": round(float(mean_excess) * 100, 2),
        "hac_t": round(float(t), 3),
        "hac_lags": int(n_lags),
        "win_frac": round(float((excess > 0).mean()), 3),
    }


# ---------------------------------------------------------------------------
# Summarize — compact R-dict for notebooks / results.md
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame, rf: float = 0.02, n_lags: int = 1) -> dict:
    """One-stop summary dict for the real merged DataFrame.

    Returns a flat dict mirroring the R = dict(...) in build_notebooks.py and
    docs/results.md.
    """
    perf = performance_stats(df, rf=rf)
    sm = smoothing_stats(df, rf=rf)
    cost = net_of_costs(df)
    test = excess_return_test(df, n_lags=n_lags)

    return {
        "n": perf["n"],
        "whisky_ann": perf["whisky_ann"],
        "sp_ann": perf["sp_ann"],
        "whisky_vol": perf["whisky_vol"],
        "sp_vol": perf["sp_vol"],
        "whisky_sharpe": perf["whisky_sharpe"],
        "sp_sharpe": perf["sp_sharpe"],
        "corr": perf["corr"],
        "whisky_beats_frac": perf["whisky_beats_frac"],
        "rho_lag1": sm["rho_lag1"],
        "raw_vol_pct": sm["raw_vol_pct"],
        "unsmoothed_vol_pct": sm["unsmoothed_vol_pct"],
        "vol_inflation_x": sm["vol_inflation_x"],
        "raw_sharpe": sm["raw_sharpe"],
        "unsmoothed_sharpe": sm["unsmoothed_sharpe"],
        "gross_ann_pct": cost["gross_ann_pct"],
        "net_ann_pct": cost["net_ann_pct"],
        "total_cost_wedge_pct": cost["total_cost_wedge_pct"],
        "mean_excess_pct": test["mean_excess_pct"],
        "hac_t": test["hac_t"],
        "win_frac": test["win_frac"],
    }
