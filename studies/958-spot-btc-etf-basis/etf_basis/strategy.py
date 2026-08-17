"""Estimators + inference for Study 958 — Spot ETF Basis.

The object of study is a **tracking difference**: the cumulative log gap between a
bitcoin wrapper and a bitcoin reference,

    c_t = log(W_t / W_0) - log(S_t / S_0),

sampled on the wrapper's trading days. If the wrapper simply bleeds a constant annual
rate (a fee, a roll cost, or both), ``c_t`` is a straight line with a negative slope,
and that slope *is* the annualised drag.

Two estimators of that slope, and the whole methodological point of this study:

- :func:`naive_drag` — ``252 x mean(daily log difference)``. Because the daily
  differences telescope, this is *exactly* the endpoint estimator
  ``(c_last - c_first) / years``: it throws away every observation in between. On a
  reference quoted at a different hour from the wrapper (a 24/7 coin at 00:00 UTC
  against ETFs marked at 16:00 New York) the two endpoints each carry a fat intraday
  offset, so the estimate inherits several percent per year of pure timestamp noise.
- :func:`trend_drag` — the HAC-robust OLS slope of ``c_t`` on time in years. Same
  target, but it uses all *n* observations, so the offset noise averages down instead
  of sitting undiluted in two endpoints. On the real tape this is the difference
  between an unusable ruler and one that reads a 25 bp fee to the basis point.

Everything else hangs off those two: :func:`piecewise_drag` puts a break at the spot-ETF
launch and reports the change in slope with a HAC *t*; :func:`placebo_split_sweep` asks
whether that break is at all special by re-running it at dozens of arbitrary dates;
:func:`implied_basis` turns a drag into an annualised futures basis given the fee and
collateral-yield ASSUMPTIONS; and :func:`pair_trade` asks whether the residual is worth
shorting, charging borrow on the short leg and one-way costs on turnover.

Conventions
-----------
* **One execution lag, once:** the only forecast-free choice in the study is the pair's
  rebalance. Weights are reset to equal notional using the **close of day t** and are
  live from **day t+1**; nothing else in the study conditions on anything.
* **Total return throughout** (``auto_adjust=True``): BITO's distributions are
  reinvested. Price-only would manufacture most of the "drag".
* **Log returns for drags, simple returns for the harvest.** A drag is a compounding
  rate, so it belongs in logs; a portfolio return is arithmetic, so the pair is built
  from simple returns and its annualised spread is not identical to the log slope
  (the gap is the usual variance term).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# HAC machinery
# --------------------------------------------------------------------------- #
def hac_ols(X: np.ndarray, y: np.ndarray, lags: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """OLS with Newey-West (Bartlett) standard errors. Returns ``(beta, se)``."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    S = np.zeros((k, k))
    lags = int(max(0, min(lags, n - 2)))
    for lag in range(lags + 1):
        w = 1.0 if lag == 0 else 1.0 - lag / (lags + 1.0)
        A = X[lag:] * resid[lag:, None]
        B = X[: n - lag] * resid[: n - lag, None]
        M = A.T @ B
        S += w * (M + M.T) if lag > 0 else M
    V = XtX_inv @ S @ XtX_inv
    return beta, np.sqrt(np.abs(np.diag(V)))


def newey_west_t(x, lags: int = 20) -> float:
    """HAC (Newey-West, Bartlett kernel) *t* of mean(x) against zero."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    u = x - x.mean()
    var = float(u @ u) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        var += 2.0 * w * float(u[lag:] @ u[:-lag]) / n
    if var <= 0:
        return float("nan")
    return float(x.mean() / np.sqrt(var / n))


def auto_lags(n: int) -> int:
    """Newey-West rule-of-thumb bandwidth ``floor(4 (n/100)^(2/9))``."""
    return int(np.floor(4.0 * (max(n, 1) / 100.0) ** (2.0 / 9.0)))


# --------------------------------------------------------------------------- #
# Tracking difference
# --------------------------------------------------------------------------- #
def align(wrapper: pd.Series, reference: pd.Series,
          lo=None, hi=None) -> tuple[pd.Series, pd.Series]:
    """Intersect two close series on common dates, optionally windowed to [lo, hi]."""
    w = pd.Series(wrapper).dropna()
    r = pd.Series(reference).dropna()
    idx = w.index.intersection(r.index)
    if lo is not None:
        idx = idx[idx >= pd.Timestamp(lo)]
    if hi is not None:
        idx = idx[idx <= pd.Timestamp(hi)]
    idx = idx.sort_values()
    return w.reindex(idx), r.reindex(idx)


def cumulative_diff(wrapper: pd.Series, reference: pd.Series,
                    lo=None, hi=None) -> pd.Series:
    """Cumulative log tracking difference ``log(W/W0) - log(S/S0)``, zeroed at the start."""
    w, r = align(wrapper, reference, lo, hi)
    c = np.log(w) - np.log(r)
    if len(c) == 0:
        return c.rename("cum_diff")
    return (c - c.iloc[0]).rename("cum_diff")


def daily_diff(wrapper: pd.Series, reference: pd.Series,
               lo=None, hi=None) -> pd.Series:
    """Daily log return difference (the first difference of :func:`cumulative_diff`)."""
    return cumulative_diff(wrapper, reference, lo, hi).diff().dropna().rename("d_diff")


def naive_drag(wrapper: pd.Series, reference: pd.Series,
               lo=None, hi=None, lags: int | None = None) -> dict:
    """Annualised drag as ``252 x mean(daily log diff)``, with a HAC *t*.

    Exactly the endpoint estimator (the daily differences telescope), reported so the
    reader can see for themselves how much precision the trend slope buys back.
    """
    d = daily_diff(wrapper, reference, lo, hi)
    n = len(d)
    if n < 30:
        return {"drag_pct": float("nan"), "t": float("nan"), "n": n}
    lg = auto_lags(n) if lags is None else lags
    return {
        "drag_pct": float(d.mean() * TRADING_DAYS * 100.0),
        "t": newey_west_t(d.to_numpy(), lags=lg),
        "n": int(n),
        "sd_daily_pct": float(d.std(ddof=1) * 100.0),
        "ac1": float(d.autocorr(1)) if n > 3 else float("nan"),
    }


def trend_drag(wrapper: pd.Series, reference: pd.Series,
               lo=None, hi=None, lags: int = 20) -> dict:
    """Annualised drag as the HAC-robust OLS slope of the cumulative diff on time.

    Returns the slope in %/yr, its Newey-West standard error and *t*, the window and
    the observation count. A negative slope means the wrapper bleeds against the
    reference.
    """
    c = cumulative_diff(wrapper, reference, lo, hi)
    n = len(c)
    if n < 30:
        return {"drag_pct": float("nan"), "se_pct": float("nan"), "t": float("nan"), "n": n}
    t_years = np.asarray((c.index - c.index[0]).days / 365.25, dtype=float)
    X = np.column_stack([np.ones_like(t_years), t_years])
    beta, se = hac_ols(X, c.to_numpy(), lags=lags)
    return {
        "drag_pct": float(beta[1] * 100.0),
        "se_pct": float(se[1] * 100.0),
        "t": float(beta[1] / se[1]) if se[1] > 0 else float("nan"),
        "n": int(n),
        "start": str(c.index[0].date()),
        "end": str(c.index[-1].date()),
        "years": float(t_years[-1]),
    }


# --------------------------------------------------------------------------- #
# The era test — a break in the drag at the spot-ETF launch
# --------------------------------------------------------------------------- #
def piecewise_drag(wrapper: pd.Series, reference: pd.Series, split: str,
                   lo=None, hi=None, lags: int = 20) -> dict:
    """Fit a broken-trend model to the cumulative diff and test the change in slope.

    ``c_t = a + b t + g 1{t>t0} + h (t - t0) 1{t>t0}``. ``b`` is the pre-event drag,
    ``b + h`` the post-event drag, and ``h`` — the coefficient the HAC *t* is reported
    for — is the **compression**: positive ``h`` means the drag shrank (the carry
    compressed), negative ``h`` means it widened. The level dummy ``g`` absorbs any
    one-off jump on the event day so it cannot leak into the slope.
    """
    c = cumulative_diff(wrapper, reference, lo, hi)
    n = len(c)
    t_years = np.asarray((c.index - c.index[0]).days / 365.25, dtype=float)
    t0 = (pd.Timestamp(split) - c.index[0]).days / 365.25 if n else float("nan")
    if n < 60 or not (t_years[0] < t0 < t_years[-1]):
        return {"pre_pct": float("nan"), "post_pct": float("nan"),
                "change_pct": float("nan"), "se_pct": float("nan"),
                "t": float("nan"), "n": n}
    d = (t_years > t0).astype(float)
    X = np.column_stack([np.ones_like(t_years), t_years, d, (t_years - t0) * d])
    beta, se = hac_ols(X, c.to_numpy(), lags=lags)
    return {
        "pre_pct": float(beta[1] * 100.0),
        "post_pct": float((beta[1] + beta[3]) * 100.0),
        "change_pct": float(beta[3] * 100.0),
        "se_pct": float(se[3] * 100.0),
        "t": float(beta[3] / se[3]) if se[3] > 0 else float("nan"),
        "n": int(n),
        "n_pre": int((~d.astype(bool)).sum()),
        "n_post": int(d.sum()),
        "split": str(pd.Timestamp(split).date()),
    }


def trend_residual_diagnostics(wrapper: pd.Series, reference: pd.Series,
                               lo=None, hi=None, lags: int = 20) -> dict:
    """How trustworthy is :func:`trend_drag`'s *t* on this particular pair?

    The trend slope is only as good as the assumption that the residual around it is
    *stationary*. Reports the residual's AR(1) and a Dickey-Fuller *t* (the HAC *t* on
    ``rho - 1`` in ``d resid ~ a + (rho-1) resid_{t-1}``; roughly, below −3.4 rejects a
    unit root at 5%). A residual pinned near one means the HAC *t* on the slope is
    flattering the evidence and the honest read is :func:`monthly_drag` instead.
    """
    c = cumulative_diff(wrapper, reference, lo, hi)
    n = len(c)
    if n < 60:
        return {"ar1": float("nan"), "df_t": float("nan"), "n": n}
    t_years = np.asarray((c.index - c.index[0]).days / 365.25, dtype=float)
    X = np.column_stack([np.ones_like(t_years), t_years])
    beta, _ = hac_ols(X, c.to_numpy(), lags=lags)
    resid = c.to_numpy() - X @ beta
    d = np.diff(resid)
    Z = np.column_stack([np.ones_like(d), resid[:-1]])
    b, se = hac_ols(Z, d, lags=lags)
    return {
        "ar1": float(np.corrcoef(resid[1:], resid[:-1])[0, 1]),
        "df_t": float(b[1] / se[1]) if se[1] > 0 else float("nan"),
        "n": int(n),
    }


def monthly_drag(wrapper: pd.Series, reference: pd.Series,
                 lo=None, hi=None) -> dict:
    """Annualised drag from **non-overlapping monthly** gaps, with a plain *t*.

    The inference-clean cross-check on :func:`trend_drag`. The trend slope regresses a
    *serially dependent* level on time; even with a HAC covariance its *t* can flatter
    the evidence when the residual basis is persistent (on the real tape the residual of
    the IBIT-referenced fit has an AR(1) near 0.98). This estimator instead takes the
    month-end values of the cumulative diff, differences them once — 29 to 56
    **non-overlapping** observations, so no HAC is needed at all — and reports the mean
    with an ordinary *t*. It is far less efficient than the trend slope, which is the
    point: whatever survives it is not an artefact of the inference.
    """
    c = cumulative_diff(wrapper, reference, lo, hi)
    if len(c) < 90:
        return {"drag_pct": float("nan"), "t": float("nan"), "n_months": 0}
    m = c.groupby([c.index.year, c.index.month]).last()
    d = m.diff().dropna()
    if len(d) < 6:
        return {"drag_pct": float("nan"), "t": float("nan"), "n_months": int(len(d))}
    years = (c.index[-1] - c.index[0]).days / 365.25
    sd = float(d.std(ddof=1))
    return {
        "drag_pct": float(d.mean() * len(d) / years * 100.0) if years > 0 else float("nan"),
        "t": float(d.mean() / sd * np.sqrt(len(d))) if sd > 0 else float("nan"),
        "n_months": int(len(d)),
    }


def cycle_regression(wrapper: pd.Series, reference: pd.Series, window: int = 126,
                     lags: int = 250, lo=None, hi=None) -> dict:
    """Is the drag a function of where bitcoin is in its cycle?

    Regresses the **rolling ``window``-session trend drag** (annualised, %/yr) of the
    wrapper on the **trailing ``window``-session log return of the reference**. Both
    series are built from overlapping windows, so the residuals are autocorrelated by
    construction out to ``window`` lags: the reported *t* uses a Newey-West bandwidth of
    ``lags`` (250 by default, a full trading year — deliberately wider than the overlap)
    and is the only *t* worth reading here. The naive OLS *t* is reported alongside so
    the size of the overlap inflation is visible.

    This is a **descriptive, non-certified** regression: with roughly ``n/window``
    independent windows the effective sample is under ten, so it can suggest that the
    carry follows the price cycle but can never establish it.
    """
    c = cumulative_diff(wrapper, reference, lo, hi)
    n = len(c)
    if n < 3 * window:
        return {"corr": float("nan"), "slope": float("nan"), "t": float("nan"),
                "t_ols": float("nan"), "n": n, "n_eff": 0.0, "window": int(window)}
    t_years = np.asarray((c.index - c.index[0]).days / 365.25, dtype=float)
    y = c.to_numpy()
    slopes = np.full(n, np.nan)
    for i in range(window, n):
        xs = t_years[i - window:i + 1]
        ys = y[i - window:i + 1]
        xs = xs - xs.mean()
        denom = float(xs @ xs)
        if denom > 0:
            slopes[i] = float(xs @ (ys - ys.mean()) / denom) * 100.0
    ref = np.log(pd.Series(reference).dropna().reindex(c.index).to_numpy())
    trail = np.full(n, np.nan)
    trail[window:] = ref[window:] - ref[:-window]
    ok = np.isfinite(slopes) & np.isfinite(trail)
    if ok.sum() < 60:
        return {"corr": float("nan"), "slope": float("nan"), "t": float("nan"),
                "t_ols": float("nan"), "n": int(ok.sum()), "n_eff": 0.0,
                "window": int(window)}
    X = np.column_stack([np.ones(int(ok.sum())), trail[ok]])
    beta, se = hac_ols(X, slopes[ok], lags=lags)
    _, se0 = hac_ols(X, slopes[ok], lags=0)
    return {
        "corr": float(np.corrcoef(slopes[ok], trail[ok])[0, 1]),
        "slope": float(beta[1]),
        "t": float(beta[1] / se[1]) if se[1] > 0 else float("nan"),
        "t_ols": float(beta[1] / se0[1]) if se0[1] > 0 else float("nan"),
        "n": int(ok.sum()),
        "n_eff": float(ok.sum() / window),
        "window": int(window),
        "lags": int(lags),
    }


def matched_window_sweep(wrapper: pd.Series, reference: pd.Series, split: str,
                         months=(6, 9, 12, 18, 24), lags: int = 20) -> pd.DataFrame:
    """Re-run the era test on **symmetric windows of several widths** around ``split``.

    A single "matched twelve months either side" is a defensible choice, but it is *a*
    choice, and one window is exactly the kind of number that gets picked because it
    reads well. This sweep publishes the whole family: one row per half-width, with the
    pre and post drag, the change in slope and its HAC *t*. Read the *sign column*, not
    the significance — if the change flips sign as the width moves, no window is
    evidence of anything; if every width agrees on the sign, the choice of window did
    not manufacture the answer.
    """
    rows = []
    for m in months:
        lo = pd.Timestamp(split) - pd.DateOffset(months=int(m))
        hi = pd.Timestamp(split) + pd.DateOffset(months=int(m)) - pd.Timedelta(days=1)
        res = piecewise_drag(wrapper, reference, split, lo=str(lo.date()),
                             hi=str(hi.date()), lags=lags)
        rows.append({"half_width_months": int(m), "pre_pct": res["pre_pct"],
                     "post_pct": res["post_pct"], "change_pct": res["change_pct"],
                     "t": res["t"], "n": res["n"]})
    return pd.DataFrame(rows).set_index("half_width_months")


def placebo_split_sweep(wrapper: pd.Series, reference: pd.Series,
                        freq: str = "MS", min_years: float = 0.5,
                        lo=None, hi=None, lags: int = 20) -> pd.DataFrame:
    """Re-run :func:`piecewise_drag` at every calendar date on a ``freq`` grid.

    A slowly-varying drag produces a "significant" broken trend at almost *any* split
    date. This sweep says how unusual the real event date's *t* actually is: if the
    launch ranks mid-pack among arbitrary dates, the break is a property of the curve,
    not of the event. Returns one row per split with the change and its HAC *t*.
    """
    c = cumulative_diff(wrapper, reference, lo, hi)
    if len(c) < 120:
        return pd.DataFrame(columns=["split", "change_pct", "t"])
    grid = pd.date_range(c.index[0], c.index[-1], freq=freq)
    rows = []
    for sp in grid:
        years_before = (sp - c.index[0]).days / 365.25
        years_after = (c.index[-1] - sp).days / 365.25
        if years_before < min_years or years_after < min_years:
            continue
        res = piecewise_drag(wrapper, reference, str(sp.date()), lo=lo, hi=hi, lags=lags)
        if np.isfinite(res["t"]):
            rows.append({"split": sp, "change_pct": res["change_pct"], "t": res["t"]})
    return pd.DataFrame(rows)


def placebo_rank(sweep: pd.DataFrame, t_real: float) -> dict:
    """Where a real |*t*| sits in the placebo distribution (rank 1 = most extreme)."""
    if len(sweep) == 0 or not np.isfinite(t_real):
        return {"rank": float("nan"), "n": 0, "median_abs_t": float("nan"),
                "max_abs_t": float("nan"), "frac_more_extreme": float("nan")}
    a = np.abs(sweep["t"].to_numpy())
    more = int((a >= abs(t_real)).sum())
    return {
        "rank": int(max(more, 1)),
        "n": int(len(a)),
        "median_abs_t": float(np.median(a)),
        "max_abs_t": float(a.max()),
        "frac_more_extreme": float(more / len(a)),
    }


# --------------------------------------------------------------------------- #
# Drag -> implied basis (uses the fee / collateral ASSUMPTIONS)
# --------------------------------------------------------------------------- #
def implied_basis(drag_pct: float, fee: float, cash_rate: float) -> dict:
    """Back out the annualised futures basis implied by a measured wrapper drag.

    A fully collateralised long futures wrapper earns, per year,
    ``spot return - basis + collateral yield - fee``. So its drag against spot is
    ``-basis + cash - fee``, and inverting,

        basis = cash - fee - drag,           excess_basis = basis - cash = -fee - drag.

    ``excess_basis`` — the carry *above* the risk-free rate — is the economically
    interesting quantity and, usefully, does not depend on the cash assumption at all.
    Both ``fee`` and ``cash_rate`` are ASSUMPTIONS (decimal, annual); sweep them with
    :func:`fee_sweep` before quoting a level.
    """
    drag = drag_pct / 100.0
    basis = cash_rate - fee - drag
    return {
        "basis_pct": float(basis * 100.0),
        "excess_basis_pct": float((-fee - drag) * 100.0),
        "fee_pct": float(fee * 100.0),
        "cash_pct": float(cash_rate * 100.0),
        "drag_pct": float(drag_pct),
    }


def fee_sweep(drag_pct: float, cash_rate: float,
              fee_grid=(0.0075, 0.0095, 0.0115)) -> list[dict]:
    """Implied basis across a grid of assumed expense ratios (the fee is a PROXY)."""
    return [implied_basis(drag_pct, f, cash_rate) for f in fee_grid]


# --------------------------------------------------------------------------- #
# Calendar-year table
# --------------------------------------------------------------------------- #
def annual_drag_table(wrapper: pd.Series, reference: pd.Series,
                      lags: int = 20, min_days: int = 60) -> pd.DataFrame:
    """Per-calendar-year trend drag (%/yr) of ``wrapper`` against ``reference``.

    Years with fewer than ``min_days`` common observations are dropped, so a partial
    first or last year never masquerades as a full one. The as-of slice already removes
    the incomplete current month.
    """
    w, r = align(wrapper, reference)
    rows = []
    for y in sorted(set(w.index.year)):
        lo = pd.Timestamp(f"{y}-01-01")
        hi = pd.Timestamp(f"{y}-12-31")
        sub = w.index[(w.index >= lo) & (w.index <= hi)]
        if len(sub) < min_days:
            continue
        res = trend_drag(w, r, lo=lo, hi=hi, lags=lags)
        rows.append({"year": int(y), "drag_pct": res["drag_pct"],
                     "se_pct": res["se_pct"], "t": res["t"], "n": res["n"]})
    return pd.DataFrame(rows).set_index("year") if rows else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Block bootstrap CI for a drag
# --------------------------------------------------------------------------- #
def bootstrap_drag_ci(wrapper: pd.Series, reference: pd.Series,
                      lo=None, hi=None, n_boot: int = 2000, block: int = 21,
                      seed: int = 958, alpha: float = 0.05) -> dict:
    """Circular block-bootstrap CI for the annualised drag (mean-based estimator).

    Blocks of ``block`` consecutive days preserve the strong negative autocorrelation
    the timestamp offset induces in the daily differences. Reported alongside the HAC
    interval as an independent read on the same uncertainty.
    """
    d = daily_diff(wrapper, reference, lo, hi).to_numpy()
    n = d.size
    if n < block + 2:
        return {"drag_pct": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_obs": n}
    point = float(d.mean() * TRADING_DAYS * 100.0)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[b] = d[idx].mean() * TRADING_DAYS * 100.0
    lo_q, hi_q = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"drag_pct": point, "ci_low": float(lo_q), "ci_high": float(hi_q),
            "frac_positive": float((boots > 0).mean()), "n_obs": int(n),
            "block": int(block), "n_boot": int(n_boot)}


def bootstrap_sharpe_ci(returns: pd.Series, n_boot: int = 2000, block: int = 21,
                        seed: int = 958, alpha: float = 0.05) -> dict:
    """Circular block-bootstrap CI for the annualised Sharpe of a daily return series."""
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"sharpe": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    ann = np.sqrt(TRADING_DAYS)
    sd = r.std(ddof=1)
    point = float(r.mean() / sd * ann) if sd > 0 else float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.full(n_boot, np.nan)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        s = r[idx]
        sdb = s.std(ddof=1)
        if sdb > 0:
            boots[b] = s.mean() / sdb * ann
    valid = boots[np.isfinite(boots)]
    lo_q, hi_q = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe": point, "ci_low": float(lo_q), "ci_high": float(hi_q),
            "frac_negative": float((valid < 0).mean()), "n_obs": int(n)}


# --------------------------------------------------------------------------- #
# Tradability — long the spot wrapper, short the futures wrapper
# --------------------------------------------------------------------------- #
def pair_trade(spot_etf: pd.Series, futures_etf: pd.Series,
               borrow_ann: float = 0.02, cost_bps: float = 5.0,
               rebalance_days: int = 21) -> pd.DataFrame:
    """Long 1 unit of the spot wrapper, short 1 unit of the futures wrapper.

    The book is reset to equal notional every ``rebalance_days`` sessions. **The one
    execution lag:** the reset uses the close of day ``t`` and the new weights are live
    from day ``t+1``; between resets the weights drift with the legs, exactly as a real
    book does. The short leg pays ``borrow_ann`` on its notional every day it is on
    (an ASSUMPTION — sweep it with :func:`borrow_sweep`), and both legs pay ``cost_bps``
    one-way on the notional traded at each reset.

    **Financing convention (an ASSUMPTION, stated so the Sharpe is readable).** The book
    is dollar-neutral and self-financing: the long leg is paid for with the short leg's
    proceeds, and the short rebate earned on those proceeds is assumed to offset exactly
    the cash the long leg gives up. The reported return is therefore already a spread
    **over cash on both legs** — there is no risk-free rate left to subtract, and none is
    subtracted. ``borrow_ann`` is the *incremental* stock-loan fee charged on top of that
    wash (the hard-to-borrow spread), which is why it is swept from 0 to 5%: a 5% borrow
    is the honest stand-in for the case where the rebate collapses. Both legs are total
    return, so the short pays away every BITO distribution.

    Returns a daily frame with the two leg returns, the weights actually in force, the
    gross spread and the net return.
    """
    a, b = align(spot_etf, futures_etf)
    r_long = a.pct_change().rename("r_long")
    r_short = b.pct_change().rename("r_short")
    df = pd.concat([r_long, r_short], axis=1).dropna()
    n = len(df)
    cost = cost_bps * 1e-4

    w_l = np.empty(n)
    w_s = np.empty(n)
    turnover = np.zeros(n)
    cur_l, cur_s = 1.0, -1.0
    for i in range(n):
        # Weights in force on day i were set at the close of day i-1 (the single lag).
        w_l[i], w_s[i] = cur_l, cur_s
        cur_l *= (1.0 + df["r_long"].iat[i])
        cur_s *= (1.0 + df["r_short"].iat[i])
        if (i + 1) % rebalance_days == 0:
            turnover[i] = abs(cur_l - 1.0) + abs(cur_s + 1.0)
            cur_l, cur_s = 1.0, -1.0

    gross = w_l * df["r_long"].to_numpy() + w_s * df["r_short"].to_numpy()
    borrow_cost = np.abs(w_s) * borrow_ann / TRADING_DAYS
    net = gross - borrow_cost - turnover * cost
    out = df.copy()
    out["w_long"] = w_l
    out["w_short"] = w_s
    out["gross"] = gross
    out["borrow"] = borrow_cost
    out["cost"] = turnover * cost
    out["net"] = net
    return out


def pair_summary(pair: pd.DataFrame, col: str = "net") -> dict:
    """Annualised return, vol, Sharpe, HAC *t* and worst drawdown of a pair leg."""
    r = pair[col].dropna()
    n = len(r)
    if n < 10:
        return {"ann_pct": float("nan"), "sharpe": float("nan"), "t": float("nan"), "n": n}
    sd = r.std(ddof=1)
    wealth = (1.0 + r).cumprod()
    return {
        "ann_pct": float(r.mean() * TRADING_DAYS * 100.0),
        "vol_pct": float(sd * np.sqrt(TRADING_DAYS) * 100.0),
        "sharpe": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan"),
        "t": newey_west_t(r.to_numpy(), lags=auto_lags(n)),
        "max_dd_pct": float((wealth / wealth.cummax() - 1.0).min() * 100.0),
        "n": int(n),
    }


def borrow_sweep(spot_etf: pd.Series, futures_etf: pd.Series,
                 borrow_grid=(0.0, 0.01, 0.02, 0.05),
                 cost_grid=(0.0, 5.0)) -> list[dict]:
    """The harvest across assumed borrow rates and one-way costs (both ASSUMPTIONS)."""
    rows = []
    for borrow in borrow_grid:
        for cost in cost_grid:
            pr = pair_trade(spot_etf, futures_etf, borrow_ann=borrow, cost_bps=cost)
            s = pair_summary(pr, "net")
            s.update({"borrow_pct": borrow * 100.0, "cost_bps": cost})
            rows.append(s)
    return rows


def pair_by_year(spot_etf: pd.Series, futures_etf: pd.Series,
                 borrow_ann: float = 0.02, cost_bps: float = 5.0,
                 min_days: int = 60) -> pd.DataFrame:
    """Per-calendar-year net harvest — does the spread decay year over year?"""
    pr = pair_trade(spot_etf, futures_etf, borrow_ann=borrow_ann, cost_bps=cost_bps)
    rows = []
    for y in sorted(set(pr.index.year)):
        sub = pr[pr.index.year == y]
        if len(sub) < min_days:
            continue
        s = pair_summary(sub, "net")
        g = pair_summary(sub, "gross")
        rows.append({"year": int(y), "gross_pct": g["ann_pct"], "net_pct": s["ann_pct"],
                     "sharpe": s["sharpe"], "t": s["t"], "n": s["n"]})
    return pd.DataFrame(rows).set_index("year") if rows else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Synthetic control — the machinery proof (never supports a real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, truth: dict, lags: int = 20) -> dict:
    """Run the study's estimators on a synthetic panel and compare with the truth.

    Checks three things at once: the spot-wrapper drag recovers the planted fee (the
    ruler calibration), the futures-wrapper drag recovers the planted carry, and the
    era test recovers the planted compression — while the naive endpoint estimator is
    reported alongside so its extra noise is visible.
    """
    split = truth["event_date"]
    fut_trend = trend_drag(prices["futures_etf"], prices["spot"], lags=lags)
    fut_naive = naive_drag(prices["futures_etf"], prices["spot"])
    spot_trend = trend_drag(prices["spot_etf"], prices["spot"], lags=lags)
    era = piecewise_drag(prices["futures_etf"], prices["spot"], split=split, lags=lags)
    return {
        "planted_fee_pct": truth["drag_spot_etf_pct"],
        "spot_etf_drag_pct": spot_trend["drag_pct"],
        "planted_drag_pre_pct": truth["drag_fut_pre_pct"],
        "planted_drag_post_pct": truth["drag_fut_post_pct"],
        "planted_change_pct": truth["drag_change_pct"],
        "fut_drag_trend_pct": fut_trend["drag_pct"],
        "fut_drag_trend_t": fut_trend["t"],
        "fut_drag_naive_pct": fut_naive["drag_pct"],
        "fut_drag_naive_t": fut_naive["t"],
        "era_pre_pct": era["pre_pct"],
        "era_post_pct": era["post_pct"],
        "era_change_pct": era["change_pct"],
        "era_t": era["t"],
        "n": fut_trend["n"],
    }
