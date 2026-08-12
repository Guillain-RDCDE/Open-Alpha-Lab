"""Strategy + inference for Study 901 — Profitable Small-Caps.

The tradable claim (Asness-Frazzini-Israel-Moskowitz-Pedersen 2018): the "cleaned" size
effect — small caps once you hold quality fixed — is a real risk-adjusted advantage. So a
**profitable small-cap** ETF (CALF, XSHQ) should beat **plain** small caps (IWM, IJR) and
hold up against **large** caps (SPY) on an **excess-of-cash Sharpe** basis, net of costs,
and the advantage should not be a mere size or value beta.

Everything is measured **excess-of-cash** (fund return minus BIL, the T-bill leg), so a
Sharpe race is a race of risk-adjusted *premia*, not of cash-rate luck. Method:

* **Common-window Sharpe race.** Slice every leg to the dates all contestants have data
  (CALF/XSHQ are young), annualise each excess-Sharpe, and bootstrap a CI on the winner's
  Sharpe and on the **Sharpe difference** (paired block bootstrap).
* **HAC t on the daily return difference.** Newey-West (Bartlett) t on
  ``x_quality - x_plain`` — is the quality leg's daily excess return higher, robust to the
  clustering that a Sharpe number hides?
* **Size / value beta decomposition.** Regress the quality leg's excess return on the plain
  small-cap excess return (the size/SMB-like factor) and on SPY excess (the market): the
  residual **alpha** is the part of the edge that is *not* just small-cap or market beta.
* **Drawdown, calendar-year table, era cut** (pre-/post-2021) for era-robustness.
* **Costed net version.** Charge the ER gap vs the cheapest plain baseline plus a one-way
  spread on a documented annual rebalance; a long-quality / short-plain isolation trade also
  pays borrow on the short leg.

Reuses ``quantlab.stats`` / ``quantlab.analytics`` where they fit (annualized_sharpe,
sharpe_ci_bootstrap, sharpe_with_se, mean_tstat_hac); the paired Sharpe-difference bootstrap
and the size/value decomposition are local.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
NW_LAGS = 10  # Newey-West lag window for daily series


# --------------------------------------------------------------------------- #
# Inference primitives (self-contained; quantlab mirrors these where it fits)
# --------------------------------------------------------------------------- #
def annualized_sharpe(x: np.ndarray, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe of an already-excess return series."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(periods)) if sd > 0 else float("nan")


def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = NW_LAGS) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
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


def sharpe_ci_bootstrap(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                        seed: int = 901, periods: int = TRADING_DAYS) -> dict:
    """Circular-block-bootstrap CI for the annualised excess-Sharpe of ``x``."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    point = annualized_sharpe(x, periods)
    if n < 8:
        return {"sharpe": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n": n}
    blk = max(1, round(n ** (1.0 / 3.0)))
    n_blocks = int(np.ceil(n / blk))
    offsets = np.arange(blk)
    ann = np.sqrt(periods)
    rng = np.random.default_rng(seed)
    boots = np.full(n_boot, np.nan)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        s = x[idx]
        sd = s.std(ddof=1)
        if sd > 0:
            boots[b] = s.mean() / sd * ann
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((valid < 0).mean()), "n": n}


def sharpe_diff_ci(xa: np.ndarray, xb: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                   seed: int = 901, periods: int = TRADING_DAYS) -> dict:
    """Paired circular-block bootstrap of the Sharpe **difference** ``SR(xa) - SR(xb)``.

    ``xa`` and ``xb`` are aligned excess-return series (same dates). The block resample uses
    the SAME index draws for both legs so the paired dependence is preserved — the honest way
    to ask "is A's Sharpe higher than B's?" on overlapping tape.
    """
    xa = np.asarray(xa, dtype=float)
    xb = np.asarray(xb, dtype=float)
    ok = np.isfinite(xa) & np.isfinite(xb)
    xa, xb = xa[ok], xb[ok]
    n = len(xa)
    diff = annualized_sharpe(xa, periods) - annualized_sharpe(xb, periods)
    if n < 8:
        return {"diff": diff, "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_le0": float("nan"), "n": n}
    blk = max(1, round(n ** (1.0 / 3.0)))
    n_blocks = int(np.ceil(n / blk))
    offsets = np.arange(blk)
    ann = np.sqrt(periods)
    rng = np.random.default_rng(seed)
    boots = np.full(n_boot, np.nan)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        sa, sb = xa[idx], xb[idx]
        da, db = sa.std(ddof=1), sb.std(ddof=1)
        if da > 0 and db > 0:
            boots[b] = sa.mean() / da * ann - sb.mean() / db * ann
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": diff, "ci_low": float(lo), "ci_high": float(hi),
            "frac_le0": float((valid <= 0).mean()), "n": n}


# --------------------------------------------------------------------------- #
# Common-window helpers
# --------------------------------------------------------------------------- #
def common_window(frame: pd.DataFrame, cols: list[str],
                  start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Rows where every column in ``cols`` is present (the funds' overlapping window)."""
    df = frame[cols].dropna()
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def max_drawdown(x: np.ndarray) -> float:
    """Max drawdown (fraction, negative) of a return series compounded from 1."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    curve = np.cumprod(1.0 + x)
    peak = np.maximum.accumulate(curve)
    return float((curve / peak - 1.0).min())


# --------------------------------------------------------------------------- #
# The Sharpe race
# --------------------------------------------------------------------------- #
def leg_stats(x: np.ndarray, periods: int = TRADING_DAYS) -> dict:
    """Headline stats for one excess-of-cash return leg."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    sr = annualized_sharpe(x, periods)
    return {
        "n": n,
        "ann_excess_pct": float(np.nanmean(x)) * periods * 100,
        "ann_vol_pct": float(np.nanstd(x, ddof=1)) * np.sqrt(periods) * 100,
        "sharpe": sr,
        "maxdd_pct": max_drawdown(x) * 100,
    }


def race(frame: pd.DataFrame, legs: dict[str, str], start: str | None = None,
         end: str | None = None, periods: int = TRADING_DAYS) -> dict:
    """Excess-of-cash Sharpe race over the COMMON window of all ``legs``.

    ``legs`` maps a display name -> excess-return column (``x_<ticker>``). Returns per-leg
    ``leg_stats`` plus the window (start/end/n), computed on the dates every leg is present.
    """
    cols = list(legs.values())
    win = common_window(frame, cols, start, end)
    out = {"start": str(win.index.min().date()) if len(win) else None,
           "end": str(win.index.max().date()) if len(win) else None,
           "n": len(win), "legs": {}}
    for name, col in legs.items():
        out["legs"][name] = leg_stats(win[col].to_numpy(), periods)
    return out


def pair_test(frame: pd.DataFrame, a_col: str, b_col: str, start: str | None = None,
              end: str | None = None, lags: int = NW_LAGS, seed: int = 901) -> dict:
    """Head-to-head A vs B on the common window: excess-Sharpes, HAC t on the daily return
    difference, and a paired Sharpe-difference bootstrap CI. ``a_col``/``b_col`` are
    ``x_<ticker>`` excess-return columns; positive ⇒ A (the quality leg) wins."""
    win = common_window(frame, [a_col, b_col], start, end)
    a = win[a_col].to_numpy(dtype=float)
    b = win[b_col].to_numpy(dtype=float)
    d = a - b
    sd = sharpe_diff_ci(a, b, seed=seed)
    return {
        "n": len(win),
        "start": str(win.index.min().date()) if len(win) else None,
        "end": str(win.index.max().date()) if len(win) else None,
        "sharpe_a": annualized_sharpe(a), "sharpe_b": annualized_sharpe(b),
        "sharpe_diff": sd["diff"], "diff_ci_low": sd["ci_low"], "diff_ci_high": sd["ci_high"],
        "diff_frac_le0": sd["frac_le0"],
        "mean_diff_bps": float(np.nanmean(d)) * 1e4,
        "t_nw_diff": newey_west_t(d, lags),
        "t_1s_diff": one_sample_t(d),
    }


# --------------------------------------------------------------------------- #
# Size / value beta decomposition — is the edge just a small-cap/market tilt?
# --------------------------------------------------------------------------- #
def beta_decomp(frame: pd.DataFrame, y_col: str, factor_cols: list[str],
                start: str | None = None, end: str | None = None,
                lags: int = NW_LAGS) -> dict:
    """OLS of the quality leg's excess return on factor excess returns, HAC t's.

    ``y = alpha + sum_k beta_k * factor_k + eps``. With ``factor_cols = [x_IWM, x_SPY]`` the
    residual **alpha** is the daily excess return the quality leg earns beyond its small-cap
    (IWM) and market (SPY) betas — the part that is NOT explained by loading up on size or
    the market. Annualised alpha and its HAC t are the headline.
    """
    cols = [y_col] + factor_cols
    win = common_window(frame, cols, start, end)
    y = win[y_col].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(win))] + [win[c].to_numpy(dtype=float)
                                               for c in factor_cols])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    n, k = X.shape
    # Newey-West HAC covariance of the OLS coefficients.
    Z = X * e[:, None]
    S = Z.T @ Z
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        G = Z[l:].T @ Z[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    r2 = 1.0 - float(e @ e) / float(((y - y.mean()) ** 2).sum())
    betas = {factor_cols[i]: float(beta[i + 1]) for i in range(len(factor_cols))}
    t_betas = {factor_cols[i]: float(beta[i + 1] / se[i + 1]) for i in range(len(factor_cols))}
    return {
        "n": n,
        "alpha_bps": float(beta[0]) * 1e4,
        "alpha_ann_pct": float(beta[0]) * TRADING_DAYS * 100,
        "t_alpha": float(beta[0] / se[0]),
        "betas": betas, "t_betas": t_betas, "r2": r2,
    }


# --------------------------------------------------------------------------- #
# Calendar-year table + era cut
# --------------------------------------------------------------------------- #
def calendar_years(frame: pd.DataFrame, legs: dict[str, str],
                   start: str | None = None) -> pd.DataFrame:
    """Per-calendar-year total return (%) for each leg over its own common window.

    Uses the RAW fund return columns (``r_<ticker>``), compounded within each year — a
    reader-facing table, not an excess-Sharpe. ``legs`` maps name -> ``r_<ticker>``.
    """
    cols = list(legs.values())
    win = common_window(frame, cols, start)
    yearly = {}
    for name, col in legs.items():
        g = (1.0 + win[col]).groupby(win.index.year).prod() - 1.0
        yearly[name] = g * 100
    return pd.DataFrame(yearly)


def era_races(frame: pd.DataFrame, legs: dict[str, str], split: str,
              start: str | None = None) -> dict:
    """Sharpe race on the pre-``split`` and post-``split`` sub-eras (era-robustness)."""
    return {
        "pre": race(frame, legs, start=start, end=split),
        "post": race(frame, legs, start=split),
    }


# --------------------------------------------------------------------------- #
# Costed / tradable version
# --------------------------------------------------------------------------- #
def costed_race(frame: pd.DataFrame, quality_col: str, plain_col: str,
                er_quality: float, er_plain: float, cost_bps_oneway: float = 5.0,
                rebalances_per_year: float = 1.0, start: str | None = None,
                end: str | None = None, periods: int = TRADING_DAYS) -> dict:
    """Charge the quality leg its realistic frictions and re-race the excess-Sharpe.

    Two charges, both as a daily haircut on the quality leg's excess return:

    * the **ER gap** vs the cheaper plain baseline: ``(er_quality - er_plain)%/yr`` (you
      already implicitly pay ER inside a total-return NAV, but ETFs report NAV net of ER, so
      to keep the leg comparable to a same-ER baseline we add back only the *difference*;
      this is conservative — it charges the whole gap to quality);
    * a **one-way spread** on the fund's own rebalance: the ETF turns its book over ~annually,
      costing ``cost_bps_oneway`` × ``rebalances_per_year`` of NAV / yr (small caps trade
      wider, so 5 bps one-way is not generous).

    Reports the gross vs net quality Sharpe and the net quality-minus-plain Sharpe gap.
    """
    win = common_window(frame, [quality_col, plain_col], start, end)
    q = win[quality_col].to_numpy(dtype=float)
    p = win[plain_col].to_numpy(dtype=float)
    er_gap_daily = (er_quality - er_plain) / 100.0 / periods
    spread_daily = cost_bps_oneway / 1e4 * rebalances_per_year / periods
    charge = er_gap_daily + spread_daily
    q_net = q - charge
    return {
        "n": len(win),
        "charge_ann_pct": charge * periods * 100,
        "sharpe_q_gross": annualized_sharpe(q, periods),
        "sharpe_q_net": annualized_sharpe(q_net, periods),
        "sharpe_plain": annualized_sharpe(p, periods),
        "net_gap": annualized_sharpe(q_net, periods) - annualized_sharpe(p, periods),
        "net_excess_ann_pct": float(np.nanmean(q_net)) * periods * 100,
    }


def isolation_trade(frame: pd.DataFrame, quality_col: str, plain_col: str,
                    borrow_annual_bps: float = 50.0, cost_bps_oneway: float = 5.0,
                    rebalances_per_year: float = 2.0, start: str | None = None,
                    end: str | None = None, lags: int = NW_LAGS,
                    periods: int = TRADING_DAYS) -> dict:
    """Long quality / short plain, dollar-neutral: isolate the "cleaned size" leg.

    Gross daily P&L = ``x_quality - x_plain`` (the cash legs cancel). Charges: borrow on the
    short plain leg (``borrow_annual_bps``/yr) + one-way spread on both legs at
    ``rebalances_per_year`` rebalances/yr. Reports gross/net annualised return and HAC t.
    """
    win = common_window(frame, [quality_col, plain_col], start, end)
    d = (win[quality_col] - win[plain_col]).to_numpy(dtype=float)
    borrow_daily = borrow_annual_bps / 1e4 / periods
    spread_daily = 2.0 * cost_bps_oneway / 1e4 * rebalances_per_year / periods
    charge = borrow_daily + spread_daily
    net = d - charge
    return {
        "n": len(win),
        "gross_ann_pct": float(np.nanmean(d)) * periods * 100,
        "net_ann_pct": float(np.nanmean(net)) * periods * 100,
        "charge_ann_pct": charge * periods * 100,
        "t_nw_gross": newey_west_t(d, lags),
        "t_nw_net": newey_west_t(net, lags),
    }
