"""Strategy + inference for Study 944 — How Much Leverage.

The instrument: a **constant-leverage, daily-reset** position in SPY. Each day the
portfolio is rebalanced back to a fixed multiple ``L`` of net asset value, the borrowed
sleeve ``(L - 1)`` is financed at the bill rate plus a spread, and the notional traded by
the reset pays a one-way cost::

    r_L(t) = L * r_spy(t) - (L - 1) * (r_cash(t) + spread) - cost * turnover(t)

This is exactly the mechanic of a levered ETF or a futures overlay, and it is *not* a
timing rule: at every ``L`` the position is always fully invested. Consequently

* the **excess-of-cash** return is ``L * (r_spy - r_cash) - (L-1) * spread - cost``, so
  gross of the spread the Sharpe ratio is **invariant** in ``L`` by construction — the
  Sharpe axis of this study is degenerate and only moves through the financing spread
  and the reset cost, which is stated plainly rather than dressed up as a result;
* the **geometric growth rate** is *not* invariant. It is concave in ``L``:
  ``g(L) ~ L*mu - L^2*sigma^2/2 - (L-1)*spread``, peaking at the Kelly / Merton multiple
  ``L* = (mu - spread) / sigma^2``. Terminal wealth is therefore the object of interest.

The study maps the realised ``g(L)`` on the financed, costed tape, locates the realised
optimum, and then asks the only question a leverage user cares about: **is that optimum
stable enough to have been known in advance?** Three ways of asking it:

1. a circular block bootstrap of the full-sample optimum (how wide is the peak's CI);
2. a rolling five-year ex-post optimum (how far does it wander through history);
3. a *tradable* ex-ante rule — size at the trailing-window Kelly estimate formed through
   day ``t``, act at ``t+1`` — raced against plain unlevered buy-and-hold, with an HAC
   *t* on the daily **log**-return difference (growth is additive in logs, so that is the
   correct statistic for a terminal-wealth claim) and a block-bootstrap CI.

**One execution lag, once:** the only lagged quantity in the study is the ex-ante Kelly
estimate (``ex_ante_kelly``), which uses returns through day ``t`` and is applied to day
``t+1``. Fixed-``L`` sweeps carry no lag because they carry no forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# The leverage grid the study sweeps on the real tape.
DEFAULT_GRID = np.round(np.arange(1.0, 3.0001, 0.05), 4)


# --------------------------------------------------------------------------- #
# Leg construction — one shape for both tapes
# --------------------------------------------------------------------------- #
def prepare_real(px: pd.DataFrame, asset: str = "SPY", rate: str = "IRX") -> pd.DataFrame:
    """Build the ``(r_asset, r_cash, acc)`` leg frame from the real cached tape.

    ``r_asset`` is SPY's daily **total-return** change (``auto_adjust=True`` closes);
    ``r_cash`` is the ^IRX-implied bill accrual over the calendar days spanned by the bar
    (act/360, previous close's rate — see ``data.cash_rate_daily``); ``acc`` is that same
    year-fraction, used to accrue the financing spread on the identical day-count.
    """
    from . import data as _data

    a = pd.Series(px[asset]).dropna().astype(float)
    r_cash = _data.cash_rate_daily(pd.Series(px[rate]).dropna().astype(float))
    common = a.index.intersection(r_cash.dropna().index)
    a = a.loc[common].sort_index()
    r_cash = r_cash.loc[common].sort_index()
    acc = pd.Series(common, index=common).diff().dt.days.astype(float) / 360.0
    out = pd.DataFrame({
        "r_asset": a.pct_change(),
        "r_cash": r_cash,
        "acc": acc,
    }).dropna()
    return out


def prepare_synth(prices: pd.DataFrame) -> pd.DataFrame:
    """Build the same leg frame from a synthetic ``(asset, cash)`` tape (act/252)."""
    r_a = prices["asset"].pct_change()
    r_c = prices["cash"].pct_change()
    out = pd.DataFrame({
        "r_asset": r_a,
        "r_cash": r_c,
        "acc": 1.0 / TRADING_DAYS,
    }).dropna()
    return out


# --------------------------------------------------------------------------- #
# The daily-reset constant-leverage engine
# --------------------------------------------------------------------------- #
def levered_returns(
    legs: pd.DataFrame,
    lev: float,
    spread_bps: float = 50.0,
    cost_bps: float = 1.0,
) -> pd.DataFrame:
    """Daily returns of a constant-``lev``, daily-reset position with financing and cost.

    Parameters
    ----------
    legs:
        Frame with ``r_asset`` (daily total return of the risky asset), ``r_cash`` (the
        bill accrual over the same bar) and ``acc`` (that bar's year-fraction).
    lev:
        The constant leverage multiple. ``lev = 1`` is unlevered buy-and-hold (no
        financing, no reset trade, zero cost — the engine reproduces that exactly).
    spread_bps:
        **PROXY.** Annualised financing spread over the bill rate paid on the borrowed
        sleeve ``(lev - 1)``, in bps. Not a measured quantity; swept in ``spread_sweep``.
    cost_bps:
        One-way cost in bps of the notional the daily reset trades. Turnover is measured
        on the pre-cost path (the feedback of cost into turnover is O(1e-8)/day).

    Returns a frame with ``r_lev`` (net), ``e_lev`` (excess of cash), ``turnover``.
    Returns are floored at −100%: a levered vehicle that loses its whole NAV is wiped
    out, it does not go negative. On this sample the floor never binds (the worst day at
    ``lev = 3`` is about −33%), and ``wipeout_days`` in :func:`sweep` reports it.
    """
    r_a = legs["r_asset"].to_numpy(dtype=float)
    r_c = legs["r_cash"].to_numpy(dtype=float)
    acc = np.broadcast_to(np.asarray(legs["acc"], dtype=float), r_a.shape)

    borrow = (lev - 1.0) * (r_c + spread_bps * 1e-4 * acc)
    gross = lev * r_a - borrow

    # Daily reset: exposure drifts to lev*(1+r_a) per unit of equity (1+gross); the reset
    # trades the gap back to lev. Exactly zero when lev == 1.
    eq = 1.0 + gross
    eq = np.where(eq <= 1e-8, 1e-8, eq)
    turnover = np.abs(lev * (1.0 + r_a) / eq - lev)
    net = gross - cost_bps * 1e-4 * turnover
    net = np.maximum(net, -1.0)

    return pd.DataFrame(
        {"r_lev": net, "e_lev": net - r_c, "turnover": turnover},
        index=legs.index,
    )


def _ann_stats(r_lev: pd.Series, e_lev: pd.Series, periods: int = TRADING_DAYS) -> dict:
    r = pd.Series(r_lev).astype(float).dropna()
    e = pd.Series(e_lev).astype(float).dropna()
    n = len(r)
    years = n / periods
    wealth = (1.0 + r).cumprod()
    terminal = float(wealth.iloc[-1]) if n else float("nan")
    cagr = float(terminal ** (1.0 / years) - 1.0) if years > 0 and terminal > 0 else float("nan")
    log_growth = float(np.log1p(r.clip(lower=-0.999999)).mean() * periods)
    sd_e = e.std(ddof=1)
    sharpe = float(e.mean() / sd_e * np.sqrt(periods)) if sd_e > 0 else float("nan")
    dd = float((wealth / wealth.cummax() - 1.0).min())
    return {
        "terminal_wealth": terminal,
        "cagr": cagr,
        "log_growth_ann": log_growth,
        "excess_sharpe": sharpe,
        "vol_ann": float(r.std(ddof=1) * np.sqrt(periods)),
        "max_drawdown": dd,
        "n_days": int(n),
    }


#: Public alias — annualised stats for one arm's ``(r_lev, e_lev)`` pair.
annual_stats = _ann_stats


def sweep(
    legs: pd.DataFrame,
    grid=DEFAULT_GRID,
    spread_bps: float = 50.0,
    cost_bps: float = 1.0,
) -> pd.DataFrame:
    """Map the leverage grid: terminal wealth, CAGR, log growth, excess Sharpe, max DD.

    One row per multiple. ``excess_sharpe`` is excess-of-cash on both sides of any
    comparison by construction (the cash leg is subtracted inside the engine); it is
    near-flat in ``L`` gross of the spread, which is the point.
    """
    rows = []
    for lev in np.asarray(grid, dtype=float):
        bt = levered_returns(legs, lev, spread_bps=spread_bps, cost_bps=cost_bps)
        s = _ann_stats(bt["r_lev"], bt["e_lev"])
        s["lev"] = float(lev)
        s["turnover_ann"] = float(bt["turnover"].mean() * TRADING_DAYS)
        s["wipeout_days"] = int((bt["r_lev"] <= -0.999999).sum())
        rows.append(s)
    out = pd.DataFrame(rows).set_index("lev")
    return out


def realised_optimum(
    legs: pd.DataFrame,
    grid=DEFAULT_GRID,
    spread_bps: float = 50.0,
    cost_bps: float = 1.0,
) -> float:
    """The multiple on ``grid`` that maximises realised log growth (= terminal wealth)."""
    tab = sweep(legs, grid=grid, spread_bps=spread_bps, cost_bps=cost_bps)
    return float(tab["log_growth_ann"].idxmax())


def _fast_log_growth(r_a: np.ndarray, r_c: np.ndarray, acc: np.ndarray,
                     grid: np.ndarray, spread_bps: float, cost_bps: float) -> np.ndarray:
    """Mean log return for every multiple on ``grid`` at once (the bootstrap inner loop).

    Broadcast over a ``(grid, days)`` block — identical arithmetic to
    :func:`levered_returns`, just vectorised so a 1,000-draw bootstrap of the whole grid
    stays in the seconds range.
    """
    lev = np.asarray(grid, dtype=float)[:, None]
    fin = (r_c + spread_bps * 1e-4 * acc)[None, :]
    gross = lev * r_a[None, :] - (lev - 1.0) * fin
    eq = np.where(1.0 + gross <= 1e-8, 1e-8, 1.0 + gross)
    turnover = np.abs(lev * (1.0 + r_a)[None, :] / eq - lev)
    net = np.maximum(gross - cost_bps * 1e-4 * turnover, -0.999999)
    return np.log1p(net).mean(axis=1)


# --------------------------------------------------------------------------- #
# Kelly / Merton theory
# --------------------------------------------------------------------------- #
def kelly_leverage(excess: pd.Series | np.ndarray) -> float:
    """The theoretical growth-optimal multiple ``mu / sigma^2`` from an excess series.

    Computed on daily excess returns; the time units cancel, so the result is a pure
    leverage number. This is the continuous-time (Merton / Kelly) optimum for a
    log-utility investor facing a Gaussian tape with no financing spread. Fat tails and
    a positive spread both push the *realised* optimum below it — never above.
    """
    x = np.asarray(pd.Series(excess).dropna(), dtype=float)
    if x.size < 3:
        return float("nan")
    v = x.var(ddof=1)
    return float(x.mean() / v) if v > 0 else float("nan")


def kelly_from_legs(legs: pd.DataFrame) -> float:
    """Kelly multiple implied by the raw (unlevered, unfinanced) excess tape."""
    return kelly_leverage(legs["r_asset"] - legs["r_cash"])


# --------------------------------------------------------------------------- #
# Inference primitives (mirror of Study 912)
# --------------------------------------------------------------------------- #
def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        var += 2.0 * w * float(u[lag:] @ u[:-lag]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def auto_nw_t(x) -> float:
    """Newey-West t with the usual ``4*(n/100)^(2/9)`` bandwidth."""
    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = arr.size
    if n < 3:
        return float("nan")
    return newey_west_t(arr, lags=int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def max_drawdown(returns) -> float:
    r = pd.Series(returns).astype(float).dropna()
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def block_bootstrap_ci(
    x,
    stat=np.mean,
    n_boot: int = 2000,
    block: int = 63,
    seed: int = 944,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap percentile CI of ``stat`` on a 1-D series."""
    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = arr.size
    if n < block + 2:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_obs": n}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[b] = stat(arr[idx])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(stat(arr)), "ci_low": float(lo), "ci_high": float(hi),
            "n_obs": int(n), "block": int(block), "n_boot": int(n_boot)}


# --------------------------------------------------------------------------- #
# How wide is the peak? — block bootstrap of the realised optimum
# --------------------------------------------------------------------------- #
def bootstrap_optimum(
    legs: pd.DataFrame,
    grid=DEFAULT_GRID,
    n_boot: int = 1000,
    block: int = 63,
    spread_bps: float = 50.0,
    cost_bps: float = 1.0,
    seed: int = 944,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap distribution of the realised growth-optimal multiple.

    Resamples 63-day (quarterly) blocks of the joint ``(r_asset, r_cash, acc)`` rows so
    vol clustering survives, re-runs the whole grid on each resample and records the
    argmax. A *knowable* optimum has a tight CI; an unknowable one spans the grid.
    """
    r_a = legs["r_asset"].to_numpy(dtype=float)
    r_c = legs["r_cash"].to_numpy(dtype=float)
    acc = np.broadcast_to(np.asarray(legs["acc"], dtype=float), r_a.shape).copy()
    g = np.asarray(grid, dtype=float)
    n = r_a.size
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)

    point = float(g[np.argmax(_fast_log_growth(r_a, r_c, acc, g, spread_bps, cost_bps))])
    opts = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        opts[b] = g[np.argmax(_fast_log_growth(r_a[idx], r_c[idx], acc[idx], g,
                                               spread_bps, cost_bps))]
    lo, hi = np.percentile(opts, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "opt": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "sd": float(opts.std(ddof=1)),
        "frac_at_floor": float((opts <= g[0] + 1e-9).mean()),
        "frac_at_cap": float((opts >= g[-1] - 1e-9).mean()),
        "n_boot": int(n_boot),
        "block": int(block),
        "grid_lo": float(g[0]),
        "grid_hi": float(g[-1]),
    }


# --------------------------------------------------------------------------- #
# Does the optimum sit still? — rolling ex-post optimum
# --------------------------------------------------------------------------- #
def rolling_optimum(
    legs: pd.DataFrame,
    grid=DEFAULT_GRID,
    window: int = 1260,          # five years
    step: int = 21,              # monthly stride (cheap, and the series is smooth)
    spread_bps: float = 50.0,
    cost_bps: float = 1.0,
) -> pd.DataFrame:
    """Ex-post growth-optimal multiple and Kelly estimate over rolling windows.

    Each row is a *backward-looking* window ending on that date — this is deliberately
    the easiest possible problem (perfect hindsight within the window). If the optimum
    still wanders across the whole grid under hindsight, no forecast can pin it down.
    """
    r_a = legs["r_asset"].to_numpy(dtype=float)
    r_c = legs["r_cash"].to_numpy(dtype=float)
    acc = np.broadcast_to(np.asarray(legs["acc"], dtype=float), r_a.shape).copy()
    g = np.asarray(grid, dtype=float)
    idx = legs.index
    rows = []
    for end in range(window, len(idx) + 1, step):
        sl = slice(end - window, end)
        lg = _fast_log_growth(r_a[sl], r_c[sl], acc[sl], g, spread_bps, cost_bps)
        e = r_a[sl] - r_c[sl]
        v = e.var(ddof=1)
        rows.append({
            "date": idx[end - 1],
            "opt_lev": float(g[int(np.argmax(lg))]),
            "kelly": float(e.mean() / v) if v > 0 else np.nan,
            "vol_ann": float(r_a[sl].std(ddof=1) * np.sqrt(TRADING_DAYS)),
            "exret_ann": float(e.mean() * TRADING_DAYS),
        })
    return pd.DataFrame(rows).set_index("date")


def instability(roll: pd.DataFrame, grid=DEFAULT_GRID) -> dict:
    """Summarise how far the rolling ex-post optimum wanders."""
    g = np.asarray(grid, dtype=float)
    o = roll["opt_lev"].dropna()
    return {
        "n_windows": int(len(o)),
        "mean": float(o.mean()),
        "sd": float(o.std(ddof=1)),
        "min": float(o.min()),
        "max": float(o.max()),
        "range": float(o.max() - o.min()),
        "frac_at_floor": float((o <= g[0] + 1e-9).mean()),
        "frac_at_cap": float((o >= g[-1] - 1e-9).mean()),
        "kelly_min": float(roll["kelly"].min()),
        "kelly_max": float(roll["kelly"].max()),
    }


# --------------------------------------------------------------------------- #
# Does the answer survive the sample you happen to hold? — start-date sensitivity
# --------------------------------------------------------------------------- #
def start_sensitivity(
    legs: pd.DataFrame,
    starts=("2003-06-04", "2004-01-06", "2005-01-03", "2007-01-03", "2010-01-04"),
    grid=DEFAULT_GRID,
    split: str = "2015-01-01",
    spread_bps: float = 50.0,
    cost_bps: float = 1.0,
) -> pd.DataFrame:
    """Re-run the headline for a set of sample **start dates** and watch it move.

    This test exists because it caught a real drift: the study was first written on a
    cache whose ^IRX history began 2004-01-06, and a later refresh pushed the start back
    to 2003-06-04. Seven extra months — an eighth of a percent of the sample's calendar
    span at the front — moved the realised optimum and *reversed the sign* of the era
    hand-off conclusion. Nothing about the world changed; only the arbitrary left edge of
    the window did.

    For each start we report the realised optimum, the unlevered and optimum CAGR, and
    the **hand-off**: the late era's optimum applied in the early era, against simply not
    levering there. ``handoff_edge`` is the difference — positive means "one decade's
    answer still beat 1x in the other decade", negative means it lost to doing nothing.
    The split date is held fixed so the eras are comparable across rows.
    """
    rows = []
    for s in starts:
        sub = legs.loc[pd.Timestamp(s):]
        early, late = sub.loc[:split], sub.loc[split:]
        if len(early) < 260 or len(late) < 260:
            continue
        tab = sweep(sub, grid=grid, spread_bps=spread_bps, cost_bps=cost_bps)
        t_e = sweep(early, grid=grid, spread_bps=spread_bps, cost_bps=cost_bps)
        t_l = sweep(late, grid=grid, spread_bps=spread_bps, cost_bps=cost_bps)
        opt = float(tab["log_growth_ann"].idxmax())
        late_opt = float(t_l["log_growth_ann"].idxmax())
        hand = float(t_e.loc[late_opt, "cagr"])
        unlev = float(t_e.loc[t_e.index[0], "cagr"])
        rows.append({
            "start": str(sub.index[0].date()),
            "n_days": int(len(sub)),
            "opt_lev": opt,
            "kelly": kelly_from_legs(sub),
            "cagr_opt": float(tab.loc[opt, "cagr"]),
            "cagr_l1": float(tab.loc[tab.index[0], "cagr"]),
            "early_opt": float(t_e["log_growth_ann"].idxmax()),
            "late_opt": late_opt,
            "handoff_cagr": hand,
            "handoff_unlev": unlev,
            "handoff_edge": hand - unlev,
        })
    cols = ["start", "n_days", "opt_lev", "kelly", "cagr_opt", "cagr_l1",
            "early_opt", "late_opt", "handoff_cagr", "handoff_unlev", "handoff_edge"]
    return pd.DataFrame(rows, columns=cols).set_index("start")


# --------------------------------------------------------------------------- #
# The tradable version — ex-ante Kelly sizing (the one execution lag)
# --------------------------------------------------------------------------- #
def ex_ante_kelly(
    legs: pd.DataFrame,
    window: int = 756,           # three years of trailing daily data
    lo: float = 1.0,
    hi: float = 3.0,
    spread_bps: float = 50.0,
    cost_bps: float = 1.0,
) -> pd.DataFrame:
    """Size at the trailing-window Kelly estimate, formed through ``t``, applied at ``t+1``.

    The estimate ``mu/sigma^2`` is computed on the trailing ``window`` of daily excess
    returns ending at day ``t``, clipped to ``[lo, hi]``, then **shifted one day** — the
    study's single execution lag. Trading the leverage change costs the same one-way
    ``cost_bps`` as the daily reset, charged on the notional the change itself moves.

    Returns a frame with ``lev`` (the applied multiple), ``r_lev``, ``e_lev``.
    """
    e = (legs["r_asset"] - legs["r_cash"])
    mu = e.rolling(window, min_periods=window).mean()
    var = e.rolling(window, min_periods=window).var(ddof=1)
    raw = (mu / var).clip(lower=lo, upper=hi)
    lev = raw.shift(1).dropna()

    sub = legs.loc[lev.index]
    r_a = sub["r_asset"].to_numpy(dtype=float)
    r_c = sub["r_cash"].to_numpy(dtype=float)
    acc = np.broadcast_to(np.asarray(sub["acc"], dtype=float), r_a.shape)
    lv = lev.to_numpy(dtype=float)

    borrow = (lv - 1.0) * (r_c + spread_bps * 1e-4 * acc)
    gross = lv * r_a - borrow
    eq = np.where(1.0 + gross <= 1e-8, 1e-8, 1.0 + gross)
    # Reset back to *today's* target, plus the notional moved by the target's own change.
    drift = np.abs(lv * (1.0 + r_a) / eq - lv)
    retarget = np.abs(np.diff(lv, prepend=lv[0]))
    net = np.maximum(gross - cost_bps * 1e-4 * (drift + retarget), -1.0)

    return pd.DataFrame({"lev": lv, "r_lev": net, "e_lev": net - r_c}, index=lev.index)


def race_vs_fixed(
    legs: pd.DataFrame,
    arms: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Annualised stats for a set of arms already aligned on a common index."""
    rows = []
    for name, bt in arms.items():
        s = _ann_stats(bt["r_lev"], bt["e_lev"])
        s["arm"] = name
        rows.append(s)
    return pd.DataFrame(rows).set_index("arm")


def growth_diff_test(a: pd.DataFrame, b: pd.DataFrame, seed: int = 944) -> dict:
    """HAC *t* and block-bootstrap CI on the daily **log**-return difference ``a − b``.

    Growth (and hence terminal wealth) is additive in log returns, so the mean log-return
    difference is the statistic that a "this leverage compounds faster" claim must clear.
    We report the arithmetic-excess HAC *t* alongside, because the two can disagree in
    sign when the arms differ in volatility — that disagreement is itself informative.
    """
    idx = a.index.intersection(b.index)
    la = np.log1p(a.loc[idx, "r_lev"].clip(lower=-0.999999))
    lb = np.log1p(b.loc[idx, "r_lev"].clip(lower=-0.999999))
    d_log = (la - lb).dropna()
    d_ex = (a.loc[idx, "e_lev"] - b.loc[idx, "e_lev"]).dropna()
    ci = block_bootstrap_ci(d_log * TRADING_DAYS, stat=np.mean, seed=seed)
    return {
        "n_days": int(len(d_log)),
        "log_growth_diff_ann": float(d_log.mean() * TRADING_DAYS),
        "t_log_diff": auto_nw_t(d_log),
        "ci_low_ann": ci["ci_low"],
        "ci_high_ann": ci["ci_high"],
        "excess_diff_ann": float(d_ex.mean() * TRADING_DAYS),
        "t_excess_diff": auto_nw_t(d_ex),
    }


def sharpe_diff_test(
    a: pd.DataFrame,
    b: pd.DataFrame,
    n_boot: int = 1000,
    block: int = 63,
    seed: int = 944,
    alpha: float = 0.05,
) -> dict:
    """Paired circular block-bootstrap CI on the **excess-Sharpe difference** ``a − b``.

    Constant leverage cannot move Sharpe (it is a pure scaling of the excess stream), so
    the sweep's Sharpe column is flat by construction. A *time-varying* multiple can move
    it, and the ex-ante Kelly arm does move it a little — which is exactly the number a
    reader would seize on. This test asks whether that gap is distinguishable from noise.
    Blocks are drawn once and applied to **both** arms, so the pairing (and the common
    market factor) survives the resample.
    """
    idx = a.index.intersection(b.index)
    ea = a.loc[idx, "e_lev"].to_numpy(dtype=float)
    eb = b.loc[idx, "e_lev"].to_numpy(dtype=float)
    n = ea.size

    def _sh(x):
        sd = x.std(ddof=1)
        return float(x.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")

    point = _sh(ea) - _sh(eb)
    if n < block + 2:
        return {"sharpe_a": _sh(ea), "sharpe_b": _sh(eb), "diff": point,
                "ci_low": float("nan"), "ci_high": float("nan"), "n_obs": int(n)}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        j = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[i] = _sh(ea[j]) - _sh(eb[j])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "sharpe_a": _sh(ea), "sharpe_b": _sh(eb), "diff": point,
        "ci_low": float(lo), "ci_high": float(hi),
        "frac_positive": float((boots > 0).mean()),
        "n_obs": int(n), "block": int(block), "n_boot": int(n_boot),
    }


# --------------------------------------------------------------------------- #
# Sweeps — the PROXY assumptions, tested rather than trusted
# --------------------------------------------------------------------------- #
def spread_sweep(
    legs: pd.DataFrame,
    grid=DEFAULT_GRID,
    spreads=(0.0, 25.0, 50.0, 100.0, 200.0),
    cost_bps: float = 1.0,
) -> pd.DataFrame:
    """Realised optimum and its growth edge over ``L=1`` at several financing spreads."""
    rows = []
    for s in spreads:
        tab = sweep(legs, grid=grid, spread_bps=s, cost_bps=cost_bps)
        opt = float(tab["log_growth_ann"].idxmax())
        rows.append({
            "spread_bps": float(s),
            "opt_lev": opt,
            "cagr_opt": float(tab.loc[opt, "cagr"]),
            "cagr_l1": float(tab.loc[tab.index[0], "cagr"]),
            "dd_opt": float(tab.loc[opt, "max_drawdown"]),
            "sharpe_opt": float(tab.loc[opt, "excess_sharpe"]),
            "sharpe_l1": float(tab.loc[tab.index[0], "excess_sharpe"]),
        })
    return pd.DataFrame(rows).set_index("spread_bps")


def cost_sweep(
    legs: pd.DataFrame,
    grid=DEFAULT_GRID,
    costs=(0.0, 0.5, 1.0, 2.0, 5.0),
    spread_bps: float = 50.0,
) -> pd.DataFrame:
    """Realised optimum and its growth at several one-way reset costs."""
    rows = []
    for c in costs:
        tab = sweep(legs, grid=grid, spread_bps=spread_bps, cost_bps=c)
        opt = float(tab["log_growth_ann"].idxmax())
        rows.append({
            "cost_bps": float(c),
            "opt_lev": opt,
            "cagr_opt": float(tab.loc[opt, "cagr"]),
            "cagr_l1": float(tab.loc[tab.index[0], "cagr"]),
            "turnover_ann_opt": float(tab.loc[opt, "turnover_ann"]),
        })
    return pd.DataFrame(rows).set_index("cost_bps")


def era_cut(
    legs: pd.DataFrame,
    split: str = "2015-01-01",
    grid=DEFAULT_GRID,
    spread_bps: float = 50.0,
    cost_bps: float = 1.0,
) -> dict:
    """Split the sample and re-locate the realised optimum in each half."""
    out = {}
    for tag, sl in [("early", slice(None, split)), ("late", slice(split, None))]:
        sub = legs.loc[sl]
        if len(sub) < 260:
            out[tag] = None
            continue
        tab = sweep(sub, grid=grid, spread_bps=spread_bps, cost_bps=cost_bps)
        opt = float(tab["log_growth_ann"].idxmax())
        out[tag] = {
            "n_days": int(len(sub)),
            "start": str(sub.index[0].date()),
            "end": str(sub.index[-1].date()),
            "opt_lev": opt,
            "kelly": kelly_from_legs(sub),
            "cagr_opt": float(tab.loc[opt, "cagr"]),
            "cagr_l1": float(tab.loc[tab.index[0], "cagr"]),
            "cagr_l3": float(tab.loc[tab.index[-1], "cagr"]),
            "dd_l1": float(tab.loc[tab.index[0], "max_drawdown"]),
            "dd_l3": float(tab.loc[tab.index[-1], "max_drawdown"]),
            "sharpe_l1": float(tab.loc[tab.index[0], "excess_sharpe"]),
            "sharpe_l3": float(tab.loc[tab.index[-1], "excess_sharpe"]),
        }
    return out


# --------------------------------------------------------------------------- #
# Financing cross-check — is the ^IRX construction fair to BIL?
# --------------------------------------------------------------------------- #
def financing_crosscheck(px: pd.DataFrame) -> dict:
    """Compare the ^IRX-implied cash accrual against BIL's realised total return.

    BIL is the *tradable* cash leg but only starts in 2007; ^IRX covers the whole window.
    If the two agree over the overlap, the ^IRX construction is a fair stand-in for the
    earlier years.
    """
    from . import data as _data

    r_irx = _data.cash_rate_daily(px["IRX"].dropna())
    r_bil = px["BIL"].dropna().pct_change()
    idx = r_irx.dropna().index.intersection(r_bil.dropna().index)
    a, b = r_irx.loc[idx], r_bil.loc[idx]
    return {
        "n_days": int(len(idx)),
        "start": str(idx[0].date()),
        "end": str(idx[-1].date()),
        "irx_ann_pct": float(a.mean() * TRADING_DAYS * 100),
        "bil_ann_pct": float(b.mean() * TRADING_DAYS * 100),
        "gap_bps": float((a.mean() - b.mean()) * TRADING_DAYS * 1e4),
    }


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports the real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(
    prices: pd.DataFrame,
    grid=np.round(np.arange(0.0, 3.0001, 0.25), 4),
    spread_bps: float = 0.0,
    cost_bps: float = 0.0,
) -> dict:
    """Recover the planted growth-optimal multiple from a synthetic tape.

    With a planted Kelly of 2.0 the sweep must peak near 2.0; on the null (zero excess
    drift) every unit of leverage is pure variance drag, so the peak must sit on the
    grid floor. The grid starts at 0 here precisely so the null has somewhere to go —
    the real-tape sweep starts at 1.0 because the study's question is *how much*
    leverage, not whether to hold equities at all.
    """
    legs = prepare_synth(prices)
    tab = sweep(legs, grid=grid, spread_bps=spread_bps, cost_bps=cost_bps)
    opt = float(tab["log_growth_ann"].idxmax())
    return {
        "opt_lev": opt,
        "kelly": kelly_from_legs(legs),
        "cagr_opt": float(tab.loc[opt, "cagr"]),
        "n_days": int(len(legs)),
        "table": tab,
    }
