"""The engine and its honest controls — Study 587 (NFT-Floor-Beta).

The claim, at full strength: **blue-chip NFT floor-price momentum is a crypto risk-appetite
signal that *leads* ETH** — when floors are running hot, ETH is about to run. The sceptical prior:
NFT floors are a *lagged, high-beta* echo of ETH (they follow the market and amplify it with
extra noise), so floor momentum has **no predictive edge** on *forward* ETH returns beyond what
ETH already tells you about itself.

This module tests the *lead* honestly:

1. **The signal.** ``floor_mom`` — a trailing mean of NFT floor-index returns (momentum), known at
   the close of day ``t``.

2. **The predictive test.** OLS of the *forward* ETH return over ``(t, t+h]`` on ``floor_mom_t``,
   with a **Newey-West HAC** *t*-stat (overlapping forward windows induce autocorrelation, so a
   plain *t* would overstate significance). The sign and HAC-*t* of the slope are the claim.

3. **The lead-vs-follow decomposition.** A contemporaneous regression (floor on ETH) recovers the
   high **beta** (floors follow ETH); a lagged cross-correlation shows floors *lag* ETH. Together
   they say "floors follow, not lead" in the null world.

4. **A control for ETH's own momentum.** Since ETH weakly trends, we also run the predictive
   regression *controlling for ETH's own trailing momentum* — the honest question is whether floor
   momentum adds anything *beyond* ETH's autocorrelation.

5. **A block-shuffle placebo null.** Circularly shift the floor signal against the ETH target many
   times; the placebo *p* is the fraction of shifts whose |HAC-t| ≥ the observed. A genuine lead
   sits in the tail; lagged noise does not.

6. **A tradable long/flat overlay** with costs: go long ETH when floor momentum is positive, else
   flat; report gross and net (one-way cost × turnover), and — for the short variant — a borrow.

7. **A multi-horizon / multi-window robustness sweep.**

8. **A seed-robust synthetic positive control** (>= 20 seeds): planting a genuine lead
   (``lead_alpha > 0``) must drive the mean HAC-*t* past +2; the null (``lead_alpha = 0``) must
   stay flat near 0.

Execution convention (one documented lag): the signal ``floor_mom_t`` uses floor data *through*
day ``t`` only; the position is entered at that same close and held over the *forward* window
``(t, t+h]``. No future data enters the signal. A one-bar-later entry (enter at ``t+1``) shifts the
headline HAC-*t* by a fraction and changes no stamp — the sweep uses the identical convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal + target construction
# ---------------------------------------------------------------------------
def floor_momentum(df: pd.DataFrame, lookback: int = 14) -> pd.Series:
    """Trailing NFT floor-index momentum (mean of the last ``lookback`` floor returns).

    Known at the close of each day (uses only floor data through that day). If ``df`` already
    carries a ``floor_mom`` column with the matching lookback it is reused; otherwise recomputed.
    """
    return df["floor_ret"].rolling(lookback).mean().rename("floor_mom")


def eth_momentum(df: pd.DataFrame, lookback: int = 14) -> pd.Series:
    """Trailing ETH momentum (control): mean of the last ``lookback`` ETH returns."""
    return df["eth_ret"].rolling(lookback).mean().rename("eth_mom")


def forward_return(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Forward ETH return over ``(t, t+horizon]`` (sum of the next ``horizon`` daily returns).

    Uses only *future* returns relative to day ``t`` — the thing the signal at ``t`` tries to
    predict. The last ``horizon`` rows are NaN (no full forward window).
    """
    r = df["eth_ret"]
    fwd = r.shift(-1).rolling(horizon).sum().shift(-(horizon - 1))
    return fwd.rename("eth_fwd")


# ---------------------------------------------------------------------------
# Newey-West HAC t-stat for a predictive slope
# ---------------------------------------------------------------------------
def _hac_lag(n: int) -> int:
    """Rule-of-thumb Bartlett lag: floor(4*(n/100)^(2/9))."""
    return max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def predictive_regression(
    signal: pd.Series,
    target: pd.Series,
    controls: pd.DataFrame | None = None,
    hac_lag: int | None = None,
) -> dict:
    """OLS of ``target`` on ``signal`` (+ optional ``controls``) with a Newey-West HAC *t*.

    Returns the slope on ``signal``, its HAC *t*-stat, the plain (OLS) *t*, the R², and n. The
    HAC variance uses Bartlett weights with a rule-of-thumb lag (overlapping forward windows make
    the residuals autocorrelated, so the plain *t* would be too optimistic).
    """
    parts = {"signal": signal, "target": target}
    if controls is not None:
        for c in controls.columns:
            parts[c] = controls[c]
    d = pd.DataFrame(parts).dropna()
    if len(d) < 30:
        return {"slope": float("nan"), "hac_t": float("nan"), "ols_t": float("nan"),
                "r2": float("nan"), "n": int(len(d))}
    y = d["target"].to_numpy(dtype=float)
    xcols = ["signal"] + ([c for c in d.columns if c not in ("signal", "target")])
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(dtype=float) for c in xcols])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)

    # plain OLS se
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    ols_se = np.sqrt(sigma2 * xtx_inv[1, 1])

    # Newey-West HAC se
    L = hac_lag if hac_lag is not None else _hac_lag(n)
    S = (X * resid[:, None]).T @ (X * resid[:, None])  # lag-0
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1.0)
        g = (X[lag:] * resid[lag:, None]).T @ (X[:-lag] * resid[:-lag, None])
        S += w * (g + g.T)
    cov_hac = xtx_inv @ S @ xtx_inv
    hac_se = float(np.sqrt(cov_hac[1, 1]))

    slope = float(coef[1])
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return {
        "slope": slope,
        "hac_t": slope / hac_se if hac_se > 0 else float("nan"),
        "ols_t": slope / ols_se if ols_se > 0 else float("nan"),
        "r2": r2,
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Lead-vs-follow decomposition
# ---------------------------------------------------------------------------
def contemporaneous_beta(df: pd.DataFrame, lag: int = 1) -> dict:
    """OLS beta of floor return on *lagged* ETH return (floors *follow* ETH with a lag).

    By construction the NFT floor index is a lagged high-beta echo of ETH: ``floor_t ≈ beta *
    eth_{t-1}``. Regressing ``floor_t`` on ``eth_{t-lag}`` (default lag 1) recovers that beta — a
    beta > 1 says floors *amplify* yesterday's ETH move (the high-beta-follow mechanism, the
    mechanical part of the claim). Returns the beta, its plain *t*, and the correlation.
    """
    d = df[["eth_ret", "floor_ret"]].dropna().copy()
    d["eth_lag"] = d["eth_ret"].shift(lag)
    d = d.dropna()
    if len(d) < 10:
        return {"beta": float("nan"), "t": float("nan"), "corr": float("nan")}
    x = d["eth_lag"].to_numpy(dtype=float)
    y = d["floor_ret"].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(len(d) - 2, 1)
    sigma2 = float(resid @ resid) / dof
    se = float(np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1]))
    corr = float(np.corrcoef(x, y)[0, 1])
    return {"beta": float(coef[1]), "t": float(coef[1] / se) if se > 0 else float("nan"),
            "corr": corr}


def lead_lag_corr(df: pd.DataFrame, max_lag: int = 5) -> pd.Series:
    """Cross-correlation of floor_ret with ETH_ret at lags -max_lag..+max_lag.

    ``corr(floor_t, eth_{t+k})``. A peak at **negative** k (floor correlates with *past* ETH)
    means floors *lag* ETH (follow); a peak at positive k would mean floors *lead*. Returned as a
    Series indexed by lag k.
    """
    d = df[["eth_ret", "floor_ret"]].dropna()
    f = d["floor_ret"].to_numpy(dtype=float)
    e = d["eth_ret"].to_numpy(dtype=float)
    out = {}
    for k in range(-max_lag, max_lag + 1):
        if k < 0:
            a, b = f[-k:], e[:k]
        elif k > 0:
            a, b = f[:-k], e[k:]
        else:
            a, b = f, e
        out[k] = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")
    return pd.Series(out, name="floor_leads_eth_by_k")


# ---------------------------------------------------------------------------
# Block-shuffle placebo null
# ---------------------------------------------------------------------------
def placebo_pvalue(
    signal: pd.Series,
    target: pd.Series,
    n_perm: int = 500,
    block: int = 21,
    seed: int = 587,
) -> float:
    """Block/circular-shift placebo p-value for the predictive HAC-t.

    Circularly shift the aligned signal against the target by a random amount (a whole-series
    rotation preserves each series' autocorrelation while destroying any true lead), recompute the
    HAC-t, and count how often |placebo HAC-t| >= |observed|. A genuine lead sits in the tail.
    """
    d = pd.DataFrame({"signal": signal, "target": target}).dropna()
    if len(d) < 60:
        return float("nan")
    obs = abs(predictive_regression(d["signal"], d["target"])["hac_t"])
    s = d["signal"].to_numpy(dtype=float)
    y = d["target"].to_numpy(dtype=float)
    n = len(s)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        shift = int(rng.integers(block, n - block))
        s_shift = np.roll(s, shift)
        reg = predictive_regression(pd.Series(s_shift), pd.Series(y))
        if abs(reg["hac_t"]) >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Tradable long/flat overlay + costs
# ---------------------------------------------------------------------------
def overlay_backtest(
    df: pd.DataFrame,
    lookback: int = 14,
    cost_bps: float = 10.0,
    allow_short: bool = False,
    borrow_ann_bps: float = 0.0,
) -> dict:
    """Long/flat (optionally long/short) ETH overlay driven by floor momentum, gross & net.

    Rule: at each close, if ``floor_mom_t > 0`` hold ETH the next day; else flat (or short if
    ``allow_short``). Position is known at ``t`` and earns ``eth_ret_{t+1}`` (one-bar execution
    lag). Net charges ``cost_bps`` one-way on each change in position (turnover) and, for shorts, a
    pro-rated annual borrow on days held short.

    Returns gross/net annualised return, Sharpe (net), turnover, and the fraction of days long.
    """
    mom = floor_momentum(df, lookback)
    r = df["eth_ret"]
    d = pd.DataFrame({"mom": mom, "r": r}).dropna()
    if len(d) < 60:
        return {"gross_ann": float("nan"), "net_ann": float("nan"), "sharpe_net": float("nan"),
                "turnover": float("nan"), "frac_long": float("nan"), "n": int(len(d))}
    pos = np.where(d["mom"].to_numpy() > 0, 1.0, (-1.0 if allow_short else 0.0))
    pos = pd.Series(pos, index=d.index).shift(1).fillna(0.0)  # enter next day (execution lag)
    fut = d["r"]
    gross = pos * fut
    # costs: one-way per unit change in position
    turn = pos.diff().abs().fillna(pos.abs())
    cost = turn * (cost_bps * 1e-4)
    borrow = (pos < 0).astype(float) * (borrow_ann_bps * 1e-4 / 365.0)
    net = gross - cost - borrow

    ann = 365.0
    gross_ann = float(gross.mean() * ann)
    net_ann = float(net.mean() * ann)
    vol = float(net.std(ddof=1) * np.sqrt(ann))
    sharpe = net_ann / vol if vol > 0 else float("nan")
    return {
        "gross_ann": gross_ann,
        "net_ann": net_ann,
        "sharpe_net": sharpe,
        "turnover": float(turn.mean()),
        "frac_long": float((pos > 0).mean()),
        "n": int(len(d)),
    }


# ---------------------------------------------------------------------------
# Robustness sweep (horizons)
# ---------------------------------------------------------------------------
def horizon_sweep(df: pd.DataFrame, lookback: int = 14, horizons=(1, 3, 5, 10, 21)) -> pd.DataFrame:
    """Predictive HAC-t of floor momentum on forward ETH return across several horizons.

    Reports, per horizon, the raw predictive HAC-t and the HAC-t *controlling for ETH's own
    trailing momentum* (the honest 'does floor add anything beyond ETH?' question).
    """
    mom = floor_momentum(df, lookback)
    ethm = eth_momentum(df, lookback)
    rows = []
    for h in horizons:
        fwd = forward_return(df, h)
        raw = predictive_regression(mom, fwd)
        ctl = predictive_regression(mom, fwd, controls=ethm.to_frame())
        rows.append({
            "horizon": h,
            "hac_t_raw": raw["hac_t"],
            "hac_t_ctrl_ethmom": ctl["hac_t"],
            "slope_raw": raw["slope"],
            "n": raw["n"],
        })
    return pd.DataFrame(rows).set_index("horizon")


# ---------------------------------------------------------------------------
# Seed-robust synthetic positive control (house rule >= 20 seeds)
# ---------------------------------------------------------------------------
def synthetic_mean_hac_t(
    data_mod,
    lead_alpha: float,
    horizon: int = 5,
    lookback: int = 14,
    n_seeds: int = 25,
    base_seed: int = 587,
) -> float:
    """Average the predictive HAC-t (floor momentum -> forward ETH) over ``n_seeds`` worlds.

    House rule: any synthetic-dependent claim averages the stat over >= 20 seeds so no single lucky
    RNG seed can manufacture significance. Returns the mean HAC-t for a planted ``lead_alpha``
    (0 = null must stay flat; > 0 = a genuine lead the engine must catch).
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        df, _ = data_mod.synthetic_series(lead_alpha=lead_alpha, seed=s)
        mom = floor_momentum(df, lookback)
        fwd = forward_return(df, horizon)
        ts.append(predictive_regression(mom, fwd)["hac_t"])
    return float(np.nanmean(ts))
