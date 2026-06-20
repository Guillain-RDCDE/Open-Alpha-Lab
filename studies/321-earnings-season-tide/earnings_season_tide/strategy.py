"""The engine and its honest controls — Study 321 (Earnings-Season-Tide).

The folk claim, steelmanned: *the whole market drifts differently during the four peak
earnings weeks each year.* When the big-cap index constituents are all reporting at once —
mid-late January, April, July and October — the aggregate index is supposedly pulled by a
distinct seasonal "tide", up or down, that you could lean on by being long (or flat) only
inside those windows.

We test it as a **calendar contrast**, the cleanest possible form:

1. Tag every trading day as in-window (inside one of the four peak earnings windows) or
   out-of-window, by date alone — see ``data.in_earnings_window``. Because the rule is
   *calendar-known*, there is **no execution lag**: you know on 1 January exactly which
   days will be in-window all year, so no ``shift`` is applied (a calendar rule earns the
   same-day return it conditions on).
2. Compare the mean daily log return in-window vs out-of-window, and put a robust,
   autocorrelation-aware standard error on the *difference* (Newey-West / HAC *t*) plus a
   circular-block-bootstrap CI on the in-window mean. A raw split-of-means is not enough;
   the difference has to clear the inference bar.
3. Race a "long only in the earnings windows, in cash otherwise" overlay against plain
   buy-and-hold — **excess-of-cash vs excess-of-cash**, because the overlay sits in cash
   most of the year, so a raw-vs-raw Sharpe race would be rigged. Costs are charged
   one-way × NAV on each entry/exit of a window.

The synthetic positive control plants a known in-window tide so we can prove the harness
recovers it; on the ``tide_bps = 0`` tape the same machinery must read insignificant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TRADING_DAYS_PER_YEAR, in_earnings_window

# A nominal cash (risk-free) drift used only for the excess-vs-excess race. Per trading
# day; 0 by default so the offline core is self-contained. The real run passes the
# realised T-bill day-rate if available.
DEFAULT_RF_DAILY = 0.0


# ---------------------------------------------------------------------------
# Returns & the calendar contrast
# ---------------------------------------------------------------------------
def log_returns(bars: pd.DataFrame) -> pd.Series:
    """Daily log returns of the close, first bar dropped."""
    return np.log(bars["close"]).diff().dropna()


def calendar_contrast(bars: pd.DataFrame, windows=None) -> dict:
    """Mean in-window vs out-of-window daily return, with a HAC *t* on the difference.

    Returns a dict of the two group means (bps/day), counts, the in-window minus
    out-of-window difference (bps/day), and the Newey-West *t* on that difference. The *t*
    is computed by regressing daily returns on the in-window dummy with a HAC covariance,
    which is exactly a robust two-sample mean test (the dummy's slope = the difference).
    """
    r = log_returns(bars)
    mask = (in_earnings_window(r.index) if windows is None
            else in_earnings_window(r.index, windows)).reindex(r.index).to_numpy()
    r_in = r.to_numpy()[mask]
    r_out = r.to_numpy()[~mask]

    diff = float(r_in.mean() - r_out.mean()) if r_in.size and r_out.size else float("nan")
    t = _hac_diff_tstat(r.to_numpy(), mask)
    return {
        "n_in": int(mask.sum()),
        "n_out": int((~mask).sum()),
        "mean_in_bps": float(r_in.mean() * 1e4) if r_in.size else float("nan"),
        "mean_out_bps": float(r_out.mean() * 1e4) if r_out.size else float("nan"),
        "diff_bps": diff * 1e4,
        "tstat_diff": t,
        "ann_in_pct": float(r_in.mean() * TRADING_DAYS_PER_YEAR * 100.0) if r_in.size else float("nan"),
    }


def _hac_diff_tstat(r: np.ndarray, mask: np.ndarray) -> float:
    """HAC (Newey-West) t-stat on the in-window − out-of-window mean difference.

    Implemented as a HAC-robust OLS of ``r`` on ``[1, dummy]``; the dummy slope is the mean
    difference and its HAC standard error gives the t. No hard dependency on quantlab.
    """
    r = np.asarray(r, dtype=float)
    n = r.size
    if n < 10 or mask.sum() == 0 or (~mask).sum() == 0:
        return float("nan")
    X = np.column_stack([np.ones(n), mask.astype(float)])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ r)
    resid = r - X @ beta
    u = X * resid[:, None]               # score contributions

    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    S = u.T @ u
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        G = u[k:].T @ u[:-k]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se_slope = np.sqrt(max(cov[1, 1], 0.0))
    return float(beta[1] / se_slope) if se_slope > 0 else float("nan")


def block_bootstrap_ci(
    bars: pd.DataFrame,
    windows=None,
    block: int = 21,
    n_boot: int = 2000,
    seed: int = 321,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the in-window − out-of-window mean diff (bps/day).

    Resamples whole blocks of the daily-return series (preserving volatility clustering and
    the calendar alignment by resampling the *paired* (return, in-window) stream) and
    recomputes the difference each draw. Returns ``(lo, hi)`` in bps/day at level
    ``1 - alpha``.
    """
    r = log_returns(bars)
    mask = (in_earnings_window(r.index) if windows is None
            else in_earnings_window(r.index, windows)).reindex(r.index).to_numpy()
    rv = r.to_numpy()
    n = rv.size
    if n < block * 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        idx = idx[:n]
        rr = rv[idx]
        mm = mask[idx]
        if mm.any() and (~mm).any():
            diffs[b] = rr[mm].mean() - rr[~mm].mean()
        else:
            diffs[b] = np.nan
    diffs = diffs[np.isfinite(diffs)]
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return (float(lo * 1e4), float(hi * 1e4))


# ---------------------------------------------------------------------------
# The tradable overlay: long only inside the windows, cash otherwise
# ---------------------------------------------------------------------------
def overlay_returns(
    bars: pd.DataFrame,
    windows=None,
    cost_bps: float = 0.0,
    rf_daily: float = DEFAULT_RF_DAILY,
) -> pd.DataFrame:
    """Daily returns of a "long index in-window, cash out-of-window" overlay.

    The position is calendar-known, so the day-*t* return is earned on day *t* with **no
    lag** when in-window. Out-of-window days earn the cash rate ``rf_daily``. ``cost_bps``
    is a one-way cost charged on NAV each time the position *changes* (window entry and
    exit), i.e. the spread paid twice per window.

    Columns: ``ret_gross, ret_net, in_window, traded`` indexed by date.
    """
    r = log_returns(bars)
    mask = (in_earnings_window(r.index) if windows is None
            else in_earnings_window(r.index, windows)).reindex(r.index).astype(bool)
    m = mask.to_numpy()
    rv = r.to_numpy()

    ret_gross = np.where(m, rv, rf_daily)
    # Trade when the in-window state flips (counting the first day as a flip from flat).
    prev = np.concatenate([[False], m[:-1]])
    traded = m != prev
    cost = traded.astype(float) * cost_bps * 1e-4
    ret_net = ret_gross - cost

    return pd.DataFrame(
        {"ret_gross": ret_gross, "ret_net": ret_net,
         "in_window": m, "traded": traded},
        index=r.index,
    )


def sharpe(returns: np.ndarray, rf_daily: float = 0.0) -> float:
    """Annualised Sharpe of an *excess* (over-cash) daily return stream."""
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)] - rf_daily
    if x.size < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def race_vs_buyhold(
    bars: pd.DataFrame,
    windows=None,
    cost_bps: float = 1.0,
    rf_daily: float = DEFAULT_RF_DAILY,
) -> dict:
    """Excess-vs-excess Sharpe race: in-window overlay vs plain buy-and-hold.

    Both legs are reduced to *excess of cash* before the Sharpe is taken, because the
    overlay sits in cash most of the year and a raw-Sharpe comparison would flatter it.
    Returns Sharpes (net for the overlay, gross for buy-and-hold), the overlay's annual
    turnover (one-way × NAV) and time-in-market.
    """
    ov = overlay_returns(bars, windows, cost_bps=cost_bps, rf_daily=rf_daily)
    r = log_returns(bars)
    n_years = max(len(r) / TRADING_DAYS_PER_YEAR, 1e-9)

    return {
        "overlay_sharpe_net": sharpe(ov["ret_net"].to_numpy(), rf_daily),
        "buyhold_sharpe": sharpe(r.to_numpy(), rf_daily),
        "overlay_ann_pct": float(ov["ret_net"].mean() * TRADING_DAYS_PER_YEAR * 100.0),
        "buyhold_ann_pct": float(r.mean() * TRADING_DAYS_PER_YEAR * 100.0),
        "time_in_market": float(ov["in_window"].mean()),
        "turnover_per_yr": float(ov["traded"].sum() / n_years),  # one-way × NAV
    }


# ---------------------------------------------------------------------------
# Per-window breakdown (which quarter, if any, carries the tide)
# ---------------------------------------------------------------------------
def per_window_means(bars: pd.DataFrame, windows=None) -> pd.DataFrame:
    """Mean daily return (bps/day) and HAC t for each of the four windows separately.

    Lets us see whether one quarter (e.g. October) drives any apparent tide, or whether it
    is spread across all four — a guard against a one-season fluke masquerading as a
    year-round seasonal.
    """
    from .data import PEAK_WINDOWS

    wins = windows if windows is not None else PEAK_WINDOWS
    r = log_returns(bars)
    rv = r.to_numpy()
    labels = {1: "Jan (Q4)", 4: "Apr (Q1)", 7: "Jul (Q2)", 10: "Oct (Q3)"}
    rows = []
    for w in wins:
        m = in_earnings_window(r.index, (w,)).reindex(r.index).to_numpy()
        n_in = int(m.sum())
        rows.append({
            "window": labels.get(w[0], f"month {w[0]}"),
            "n_days": n_in,
            "mean_bps": float(rv[m].mean() * 1e4) if n_in else float("nan"),
            "tstat": _hac_diff_tstat(rv, m),
        })
    return pd.DataFrame(rows)
