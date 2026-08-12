"""The overlay + inference for Study 893 — Vol-Target 60/40.

The static book is the fixed **60% SPY / 40% IEF** blend, rebalanced annually (the object
[Study 97](../../97-balancing-act/) grades a Real/Investable default). The thermostat sits on top:

    w_t = sigma_target / sigma_hat_{t-1},   clipped to [0, max_leverage]

``sigma_hat_{t-1}`` is the trailing realized **portfolio** volatility of the static blend, known
only through yesterday's close (one explicit ``shift`` — zero look-ahead). The vol-targeted book
holds ``w_t`` of the blend and parks ``(1 − w_t)`` in cash (earning the bill rate when de-risked,
paying it to finance leverage when w>1). A clean identity falls out for the fair race:

    excess_of_cash(vol_target)_t = w_t * excess_of_cash(blend)_t

so the *excess-of-cash* Sharpe comparison automatically finances leverage at cash — and the
**costed** variant then adds a realistic borrow spread on the levered fraction plus a per-unit
turnover charge on the overlay's daily ``|Δw|``.

We race excess-of-cash vs excess-of-cash (both minus BIL), put the daily excess-return *difference*
through a Newey–West (HAC) *t*, interval the Sharpe difference with a circular block bootstrap
(resampled jointly so the vol clustering survives), and cut the sample into eras. The synthetic
control only proves the machinery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Return helpers
# --------------------------------------------------------------------------- #
def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily total returns from a price frame (first row dropped)."""
    return prices.pct_change().dropna()


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def _ann_sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


# --------------------------------------------------------------------------- #
# The static 60/40 blend — fixed weights, annual rebalance (the baseline)
# --------------------------------------------------------------------------- #
def static_blend(
    returns: pd.DataFrame,
    weights: dict[str, float] | None = None,
    rebalance: str = "annual",
    cost_bps: float = 0.0,
) -> pd.Series:
    """Daily net return of the fixed-weight blend, reset on a calendar schedule.

    Weights drift with the assets between rebalances; on each rebalance date the book is reset to
    ``weights`` and the one-way turnover (sum of absolute weight changes) is charged at
    ``cost_bps``. ``rebalance='annual'`` resets on the first trading day of each calendar year.
    Default weights are 60% SPY / 40% IEF. Vectorised drift; a single pass over rebalance marks.
    """
    if weights is None:
        weights = {"SPY": 0.6, "IEF": 0.4}
    cols = list(weights.keys())
    R = returns[cols].to_numpy(dtype=float)
    w_target = np.array([weights[c] for c in cols], dtype=float)
    n = R.shape[0]
    idx = returns.index

    if rebalance == "annual":
        marks = idx.to_series().groupby(idx.year).head(1).index
    elif rebalance == "monthly":
        marks = idx.to_series().groupby([idx.year, idx.month]).head(1).index
    elif rebalance == "none":
        marks = idx[:1]
    else:
        raise ValueError(f"unknown rebalance schedule: {rebalance!r}")
    rebal = set(pd.Index(marks))

    w = w_target.copy()
    out = np.empty(n)
    cost = cost_bps * 1e-4
    for t in range(n):
        if idx[t] in rebal:
            turn = np.abs(w_target - w).sum()
            w = w_target.copy()
        else:
            turn = 0.0
        out[t] = float(w @ R[t]) - turn * cost
        w = w * (1.0 + R[t])
        s = w.sum()
        if s != 0:
            w = w / s
    return pd.Series(out, index=idx, name="static_6040")


# --------------------------------------------------------------------------- #
# The thermostat — past-only inverse-vol exposure on the blend
# --------------------------------------------------------------------------- #
def realized_vol(blend_ret: pd.Series, window: int = 21) -> pd.Series:
    """Trailing annualised realized **portfolio** volatility of the blend return."""
    return (blend_ret.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)).rename("rv")


def target_weight(
    blend_ret: pd.Series,
    target_vol: float = 0.10,
    window: int = 21,
    max_leverage: float = 2.0,
) -> pd.Series:
    """Past-only exposure ``w_t = target_vol / sigma_hat_{t-1}``, clipped to ``[0, max_leverage]``.

    The vol forecast is the trailing realized portfolio vol **lagged one bar**, so the weight
    applied to day ``t`` uses only returns through ``t−1``. The cap is the realistic leverage
    ceiling a balanced book faces; without it the rule demands unbounded size in dead-calm tape.
    """
    fc = realized_vol(blend_ret, window).shift(1)
    return (target_vol / fc).clip(lower=0.0, upper=max_leverage).rename("weight")


def vol_targeted(
    blend_ret: pd.Series,
    cash_ret: pd.Series,
    target_vol: float = 0.10,
    window: int = 21,
    max_leverage: float = 2.0,
    cost_bps: float = 0.0,
    borrow_bps_yr: float = 0.0,
) -> pd.Series:
    """Daily net return of the vol-targeted 60/40 book.

    Holds ``w_t`` of the static blend and ``(1 − w_t)`` in cash::

        gross_t = w_t * blend_t + (1 − w_t) * cash_t

    The **costed** version subtracts (a) a per-unit turnover charge ``cost_bps * |Δw_t|`` on the
    overlay's daily exposure change and (b) a borrow spread ``borrow_bps_yr`` (annual, over cash)
    on the levered fraction ``max(w_t − 1, 0)``. Returns a daily net-return Series aligned to the
    overlap of ``blend_ret`` and ``cash_ret`` (the vol warm-up rows drop out).
    """
    w = target_weight(blend_ret, target_vol, window, max_leverage)
    idx = w.dropna().index.intersection(blend_ret.index).intersection(cash_ret.index)
    w = w.reindex(idx)
    b = blend_ret.reindex(idx)
    c = cash_ret.reindex(idx).fillna(0.0)

    gross = w * b + (1.0 - w) * c
    turn = w.diff().abs().fillna(0.0)
    tc = (cost_bps * 1e-4) * turn
    borrow = (borrow_bps_yr * 1e-4 / TRADING_DAYS) * (w - 1.0).clip(lower=0.0)
    return (gross - tc - borrow).rename("vol_target_6040")


# --------------------------------------------------------------------------- #
# Headline stats — excess-of-cash Sharpe, CAGR, vol, drawdown
# --------------------------------------------------------------------------- #
def stats(net: pd.Series, cash: pd.Series | None = None) -> dict:
    """Headline stats. Sharpe is **excess-of-cash** when ``cash`` is given; CAGR/vol/DD raw."""
    net = net.astype(float).dropna()
    equity = (1.0 + net).cumprod()
    years = len(net) / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else float("nan")
    vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    ex = (net - cash.reindex(net.index).fillna(0.0)) if cash is not None else net
    return {
        "sharpe": _ann_sharpe(ex.to_numpy()),
        "cagr": cagr,
        "vol": vol,
        "max_dd": _max_drawdown(equity.to_numpy()),
        "final": float(equity.iloc[-1]),
        "n_days": int(len(net)),
    }


def calendar_year_returns(net: pd.Series) -> pd.Series:
    """Compounded calendar-year total return of a daily return series."""
    g = (1.0 + net.astype(float)).groupby(net.index.year).prod() - 1.0
    g.index.name = "year"
    return g.rename("year_return")


# --------------------------------------------------------------------------- #
# Inference primitives (self-contained; mirror the desk's shared helpers)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int | None = None) -> float:
    """HAC (Newey–West, Bartlett kernel) *t* of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def spanning_alpha(
    managed_excess: np.ndarray,
    static_excess: np.ndarray,
    lags: int | None = None,
) -> dict:
    """Moreira–Muir (2017) spanning test: regress the managed book's excess return on the static
    book's excess return; the intercept ``alpha`` is the risk-adjusted pickup that *survives*
    matching the static's beta — the **leverage-invariant** significance read.

    A raw mean-return difference is contaminated: because ``E[1/sigma_hat] > 1/E[sigma_hat]``
    (Jensen), the thermostat runs average exposure slightly above 1, so a plain *t* on the return
    difference picks up that level tilt, not the re-timing skill. The spanning alpha nets out the
    static beta and asks the honest question — does re-timing risk earn more than loading the same
    risk statically? HAC (Newey–West) *t* on the intercept.
    """
    y = np.asarray(managed_excess, dtype=float)
    x = np.asarray(static_excess, dtype=float)
    m = np.isfinite(y) & np.isfinite(x)
    y, x = y[m], x[m]
    n = y.size
    if n < 20:
        return {"alpha_daily": float("nan"), "alpha_ann": float("nan"),
                "beta": float("nan"), "t_alpha": float("nan"), "n": n}
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ (X.T @ y)
    resid = y - X @ coef
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    # Newey–West HAC covariance of the OLS coefficients.
    S = (X * resid[:, None]).T @ (X * resid[:, None]) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        Xu = X * resid[:, None]
        G = Xu[k:].T @ Xu[:-k] / n
        S += w * (G + G.T)
    cov = XtX_inv @ (n * S) @ XtX_inv
    se_alpha = float(np.sqrt(max(cov[0, 0], 0.0)))
    alpha = float(coef[0])
    return {
        "alpha_daily": alpha,
        "alpha_ann": alpha * TRADING_DAYS,
        "beta": float(coef[1]),
        "t_alpha": float(alpha / se_alpha) if se_alpha > 0 else float("nan"),
        "n": n,
    }


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def bootstrap_sharpe_diff(
    a: pd.Series,
    b: pd.Series,
    cash: pd.Series | None = None,
    block: int = 21,
    n_boot: int = 2000,
    seed: int = 893,
) -> dict:
    """Circular block bootstrap CI for the excess-of-cash Sharpe **difference** (``a`` − ``b``).

    Resamples the two aligned excess-return series **jointly** in circular blocks (so the
    cross-correlation and the volatility clustering both survive), recomputes each arm's annualised
    Sharpe per resample, and returns the point difference, a 95% percentile CI, and the bootstrap
    fraction of resamples in which ``a`` beats ``b``.
    """
    idx = a.dropna().index.intersection(b.dropna().index)
    ra = a.reindex(idx).to_numpy(dtype=float)
    rb = b.reindex(idx).to_numpy(dtype=float)
    if cash is not None:
        f = cash.reindex(idx).fillna(0.0).to_numpy(dtype=float)
        ra, rb = ra - f, rb - f
    n = len(idx)
    point = _ann_sharpe(ra) - _ann_sharpe(rb)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    a_wins = 0
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        offsets = (starts[:, None] + np.arange(block)[None, :]) % n
        sel = offsets.reshape(-1)[:n]
        sa, sb = _ann_sharpe(ra[sel]), _ann_sharpe(rb[sel])
        diffs[i] = sa - sb
        if sa > sb:
            a_wins += 1
    lo, hi = np.percentile(diffs[np.isfinite(diffs)], [2.5, 97.5])
    return {
        "point": float(point),
        "ci95": (float(lo), float(hi)),
        "frac_a_wins": a_wins / n_boot,
        "n": n, "block": block, "n_boot": n_boot,
    }


# --------------------------------------------------------------------------- #
# The full race — static 60/40 vs vol-targeted 60/40, gross and costed
# --------------------------------------------------------------------------- #
def race(
    returns: pd.DataFrame,
    cash_col: str = "BIL",
    weights: dict[str, float] | None = None,
    target_vol: float | None = None,
    window: int = 21,
    max_leverage: float = 2.0,
    cost_bps: float = 0.0,
    borrow_bps_yr: float = 0.0,
    rebalance_cost_bps: float = 0.0,
) -> dict:
    """Static 60/40 vs vol-targeted 60/40 on a common sample, both excess-of-cash.

    ``target_vol=None`` sets the thermostat's target to the static blend's own realized vol, so the
    vol-targeted book runs the **same average risk** and only *re-times* it — making the drawdown
    comparison fair (the Sharpe difference is already leverage-invariant). Returns each arm's stats,
    the excess-Sharpe advantage, the HAC *t* on the daily excess-return difference, realized
    leverage stats, and the overlay's annualised turnover.
    """
    if weights is None:
        weights = {"SPY": 0.6, "IEF": 0.4}
    cash = returns[cash_col]

    blend_gross = static_blend(returns, weights, "annual", cost_bps=rebalance_cost_bps)
    if target_vol is None:
        target_vol = float(blend_gross.std(ddof=1) * np.sqrt(TRADING_DAYS))

    vt = vol_targeted(blend_gross, cash, target_vol, window, max_leverage,
                      cost_bps=cost_bps, borrow_bps_yr=borrow_bps_yr)

    idx = vt.dropna().index
    static = blend_gross.reindex(idx)
    cash_al = cash.reindex(idx)
    w = target_weight(blend_gross, target_vol, window, max_leverage).reindex(idx)

    s_static = stats(static, cash_al)
    s_vt = stats(vt, cash_al)

    ex_static = (static - cash_al).to_numpy(dtype=float)
    ex_vt = (vt - cash_al).to_numpy(dtype=float)
    diff = ex_vt - ex_static
    span = spanning_alpha(ex_vt, ex_static)

    return {
        "target_vol": float(target_vol),
        "static": s_static,
        "vol_target": s_vt,
        "sharpe_gain": float(s_vt["sharpe"] - s_static["sharpe"]),
        "dd_static": s_static["max_dd"],
        "dd_vt": s_vt["max_dd"],
        "alpha_ann": span["alpha_ann"],
        "alpha_beta": span["beta"],
        "t_alpha": span["t_alpha"],
        "diff_t_nw": newey_west_t(diff),
        "diff_t_1s": one_sample_t(diff),
        "avg_leverage": float(w.mean()),
        "max_leverage_used": float(w.max()),
        "frac_levered": float((w > 1.0).mean()),
        "frac_capped": float((w >= max_leverage - 1e-9).mean()),
        "turnover_ann": float(w.diff().abs().mean() * TRADING_DAYS),
        "n_days": int(len(idx)),
        "static_net": static, "vt_net": vt, "cash_net": cash_al, "weight": w,
    }


def cost_sweep(
    returns: pd.DataFrame,
    one_way_bps=(0.0, 1.0, 2.0, 5.0, 10.0),
    borrow_bps_yr: float = 50.0,
    cash_col: str = "BIL",
    weights: dict[str, float] | None = None,
    window: int = 21,
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    """Excess-Sharpe advantage of the thermostat vs cost per unit of exposure traded (bps).

    Each row charges ``one_way`` bps on the overlay turnover **and** ``borrow_bps_yr`` on the
    levered fraction (the honest tradability test). The static baseline pays the same one-way cost
    on its annual rebalance, so the race stays apples-to-apples.
    """
    rows = {}
    for c in one_way_bps:
        r = race(returns, cash_col=cash_col, weights=weights, window=window,
                 max_leverage=max_leverage, cost_bps=c, borrow_bps_yr=borrow_bps_yr,
                 rebalance_cost_bps=c)
        rows[c] = {
            "static_sharpe": r["static"]["sharpe"],
            "vt_sharpe": r["vol_target"]["sharpe"],
            "sharpe_gain": r["sharpe_gain"],
            "dd_static": r["dd_static"],
            "dd_vt": r["dd_vt"],
        }
    out = pd.DataFrame(rows).T
    out.index.name = "one_way_bps"
    return out


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof — never cited for the stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(
    frame: pd.DataFrame,
    target_vol: float | None = None,
    window: int = 21,
    max_leverage: float = 2.0,
) -> dict:
    """Run the headline race on a synthetic price frame (offline)."""
    ret = to_returns(frame)
    r = race(ret, cash_col="BIL", target_vol=target_vol, window=window,
             max_leverage=max_leverage)
    return {
        "sharpe_static": r["static"]["sharpe"],
        "sharpe_vt": r["vol_target"]["sharpe"],
        "sharpe_gain": r["sharpe_gain"],
        "diff_t_nw": r["diff_t_nw"],
        "t_alpha": r["t_alpha"],
        "alpha_ann": r["alpha_ann"],
        "dd_static": r["dd_static"],
        "dd_vt": r["dd_vt"],
        "avg_leverage": r["avg_leverage"],
        "n_days": r["n_days"],
    }
