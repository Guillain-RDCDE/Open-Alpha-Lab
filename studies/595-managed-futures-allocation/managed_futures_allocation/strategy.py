"""Strategy + inference for Study 595 — Managed-Futures Sleeve.

The claim (the allocator's pitch, not the trend-follower's): carving **10-20%** of a 60/40
portfolio into a managed-futures (trend-following) sleeve improves risk-adjusted returns —
near-zero correlation to stocks/bonds plus crisis alpha of the 2022 kind. Siblings
31-trade-winds and 518-time-series-momentum tested the STANDALONE premium (both Weak); here
the unit under test is the **portfolio improvement**, so the decisive statistics are:

  * the **Newey-West t of the MF book's alpha versus the 60/40** (monthly excess-on-excess
    regression) — the textbook mean-variance criterion: a positive alpha at t >= 2 means the
    sleeve raises the achievable Sharpe, whatever its standalone Sharpe;
  * a **paired moving-block bootstrap of ΔSharpe** (60/40+15% sleeve minus plain 60/40),
    resampling the JOINT monthly rows so cross-correlation survives — the direct CI on the
    claimed improvement;
  * the **correlation** of the sleeve to the 60/40 (with a Fisher-z CI) — the "near-zero
    correlation" half of the pitch;
  * the calendar-**2022 attribution** — the crisis-alpha half.

Book construction (replication sleeve): simple 12-month time-series momentum on the shared
18-contract futures panel — sign of the trailing 12-month return, inverse-vol sizing
(40%/sigma per contract / N_active, the Moskowitz-Ooi-Pedersen scheme), monthly rebalance.
Exactly ONE execution lag: the signal and vol are measured through month-end t-1 and the
position is held for month t. Costs are one-way bps x traded notional (|Δw|); futures shorts
post margin and pay NO stock-borrow fee (they are unfunded margin instruments — documented
here, not silently assumed). The sleeve is CASH-COLLATERALISED when blended: funded sleeve
return = rf + book excess return - ETF fee. Sharpe races are excess-vs-excess throughout.

Any random baseline (the random-sign placebo book) is averaged over >= 20 seeds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12.0


# --------------------------------------------------------------------------- #
# The TSMOM replication book (monthly, one-month lag)
# --------------------------------------------------------------------------- #
def tsmom_book(fut_monthly: pd.DataFrame, lookback: int = 12, vol_window: int = 12,
               per_asset_target: float = 0.40, cost_bps: float = 5.0) -> dict:
    """Simple 12-month TSMOM book on a wide monthly futures-return frame.

    Signal at month-end t-1: sign of the trailing ``lookback``-month compounded return
    (months t-lookback .. t-1). Size: ``per_asset_target``/sigma_i / N_active, sigma_i =
    trailing ``vol_window``-month annualised vol (floored at 2%/yr against degenerate sizing).
    Position held for month t — exactly one month of execution lag, no look-ahead.

    Returns dict with ``gross``/``net`` monthly excess-return Series, ``turnover`` (sum |Δw|
    per month), ``weights`` (the held weight frame) and ``leverage`` (mean sum |w|).
    Futures returns are excess by construction (unfunded margin instruments; shorts pay no
    borrow). Costs: ``cost_bps`` one-way x traded notional |Δw|.
    """
    m = fut_monthly.copy()
    trail = (1.0 + m).rolling(lookback, min_periods=lookback).apply(np.prod, raw=True) - 1.0
    sig = np.sign(trail)
    vol = m.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(MONTHS)
    vol = vol.clip(lower=0.02)
    raw_w = sig * (per_asset_target / vol)
    n_active = raw_w.notna().sum(axis=1).replace(0, np.nan)
    w_signal = raw_w.div(n_active, axis=0)          # weight decided at month-end t
    w_held = w_signal.shift(1)                      # ... and HELD for month t (the one lag)

    contrib = (w_held * m)
    gross = contrib.sum(axis=1, min_count=1).dropna()
    w_filled = w_held.fillna(0.0)
    turnover = (w_filled - w_filled.shift(1)).abs().sum(axis=1).reindex(gross.index).fillna(0.0)
    net = gross - turnover * (cost_bps / 1e4)
    lev = w_filled.abs().sum(axis=1).reindex(gross.index)
    return {"gross": gross, "net": net, "turnover": turnover,
            "weights": w_held.reindex(gross.index), "leverage": float(lev.mean()),
            "contrib": contrib.reindex(gross.index)}


def random_sign_book(fut_monthly: pd.DataFrame, seeds: range,
                     vol_window: int = 12, per_asset_target: float = 0.40,
                     cost_bps: float = 5.0) -> pd.DataFrame:
    """Placebo: same sizing/lag/costs but iid RANDOM signs instead of the trend sign.

    Averaged over the given seeds (>= 20 by house rule). Returns one row per seed with the
    standalone net Sharpe of the placebo book (the caller blends and diffs as needed).
    """
    m = fut_monthly.copy()
    vol = m.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(MONTHS)
    vol = vol.clip(lower=0.02)
    size = per_asset_target / vol
    rows = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        signs = pd.DataFrame(rng.choice([-1.0, 1.0], size=m.shape),
                             index=m.index, columns=m.columns).where(size.notna())
        raw_w = signs * size
        n_active = raw_w.notna().sum(axis=1).replace(0, np.nan)
        w_held = raw_w.div(n_active, axis=0).shift(1)
        gross = (w_held * m).sum(axis=1, min_count=1).dropna()
        w_filled = w_held.fillna(0.0)
        to = (w_filled - w_filled.shift(1)).abs().sum(axis=1).reindex(gross.index).fillna(0.0)
        net = gross - to * (cost_bps / 1e4)
        rows.append({"seed": sd, "net": net})
    return rows


# --------------------------------------------------------------------------- #
# Portfolio arithmetic (excess-vs-excess, funded blend)
# --------------------------------------------------------------------------- #
def portfolio_6040(stock_tr: pd.Series, bond_tr: pd.Series) -> pd.Series:
    """Monthly-rebalanced 60% stock / 40% bond total-return series."""
    df = pd.concat([stock_tr, bond_tr], axis=1).dropna()
    return 0.6 * df.iloc[:, 0] + 0.4 * df.iloc[:, 1]


def blend_returns(p6040_tr: pd.Series, mf_excess: pd.Series, rf: pd.Series,
                  sleeve: float = 0.15, fee_annual: float = 0.0085) -> pd.Series:
    """Total return of (1-sleeve) x 60/40 + sleeve x funded MF, monthly rebalanced.

    The MF sleeve is cash-collateralised: funded sleeve return = rf + excess - fee/12
    (fee = the live ETF expense ratio charged against the replication, headline 0.85%/yr).
    """
    idx = p6040_tr.index.intersection(mf_excess.index).intersection(rf.dropna().index)
    funded = rf.loc[idx] + mf_excess.loc[idx] - fee_annual / MONTHS
    return (1.0 - sleeve) * p6040_tr.loc[idx] + sleeve * funded


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def ann_stats(total_ret: pd.Series, rf: pd.Series) -> dict:
    """CAGR, ann vol, EXCESS Sharpe, max drawdown, MAR — on monthly total returns."""
    r = total_ret.dropna()
    rf_a = rf.reindex(r.index)
    ex = (r - rf_a).dropna()
    wealth = (1.0 + r).cumprod()
    yrs = len(r) / MONTHS
    cagr = float(wealth.iloc[-1] ** (1.0 / yrs) - 1.0)
    vol = float(r.std() * np.sqrt(MONTHS))
    sharpe = float(ex.mean() / ex.std() * np.sqrt(MONTHS)) if ex.std() > 0 else np.nan
    dd = float((wealth / wealth.cummax() - 1.0).min())
    mar = cagr / abs(dd) if dd != 0 else np.nan
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "maxdd": dd, "mar": mar,
            "n_months": len(r)}


def sharpe_excess(excess: pd.Series) -> float:
    e = excess.dropna()
    return float(e.mean() / e.std() * np.sqrt(MONTHS)) if e.std() > 0 else np.nan


def nw_tstat(x: pd.Series, lags: int = 6) -> float:
    """Newey-West (HAC) t of the mean of a monthly series."""
    v = np.asarray(x.dropna(), dtype=float)
    n = len(v)
    if n < 10:
        return np.nan
    e = v - v.mean()
    s0 = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        s0 += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(s0 / n)
    return float(v.mean() / se) if se > 0 else np.nan


def nw_alpha(y: pd.Series, x: pd.Series, lags: int = 6) -> dict:
    """OLS y = a + b x on monthly EXCESS returns, Newey-West t of the alpha.

    Positive alpha at t >= 2 is the mean-variance criterion for the sleeve improving the
    core portfolio's attainable Sharpe.
    """
    df = pd.concat([y, x], axis=1).dropna()
    yy = df.iloc[:, 0].to_numpy(dtype=float)
    xx = df.iloc[:, 1].to_numpy(dtype=float)
    n = len(yy)
    X = np.column_stack([np.ones(n), xx])
    beta = np.linalg.lstsq(X, yy, rcond=None)[0]
    resid = yy - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    # HAC covariance of the moment conditions
    G = X * resid[:, None]
    S = (G.T @ G) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        gk = (G[k:].T @ G[:-k]) / n
        S += w * (gk + gk.T)
    cov = XtX_inv @ (n * S) @ XtX_inv
    se_a = float(np.sqrt(cov[0, 0]))
    r2 = 1.0 - float(resid @ resid) / float(((yy - yy.mean()) ** 2).sum())
    return {"alpha_m": float(beta[0]), "alpha_ann": float(beta[0] * MONTHS),
            "beta": float(beta[1]), "t_alpha": float(beta[0] / se_a) if se_a > 0 else np.nan,
            "r2": r2, "n": n}


def corr_ci(x: pd.Series, y: pd.Series) -> dict:
    """Pearson correlation with a 95% Fisher-z CI."""
    df = pd.concat([x, y], axis=1).dropna()
    n = len(df)
    r = float(df.iloc[:, 0].corr(df.iloc[:, 1]))
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    return {"r": r, "lo": float(lo), "hi": float(hi), "n": n}


def bootstrap_dsharpe(e6040: pd.Series, mf_excess: pd.Series, sleeve: float = 0.15,
                      fee_annual: float = 0.0085, n_draws: int = 5000, block: int = 6,
                      seed: int = 595) -> dict:
    """Paired moving-block bootstrap of ΔSharpe = Sharpe(blend) - Sharpe(60/40).

    Resamples JOINT monthly rows (blocks of ``block`` months) of the (60/40 excess, MF excess)
    pair, so the cross-correlation structure survives; recomputes both Sharpes per draw.
    The blend excess is (1-sleeve) x e6040 + sleeve x (mf_excess - fee/12) — the funded-sleeve
    identity (the rf legs cancel in excess space except the fee). Returns observed ΔSharpe,
    the 95% percentile CI and the one-sided p (share of draws with ΔSharpe <= 0).
    """
    df = pd.concat([e6040, mf_excess], axis=1).dropna()
    a = df.iloc[:, 0].to_numpy(dtype=float)
    b = df.iloc[:, 1].to_numpy(dtype=float) - fee_annual / MONTHS
    n = len(a)

    def dsharpe(av, bv):
        bl = (1.0 - sleeve) * av + sleeve * bv
        sa = av.mean() / av.std() if av.std() > 0 else np.nan
        sb = bl.mean() / bl.std() if bl.std() > 0 else np.nan
        return (sb - sa) * np.sqrt(MONTHS)

    obs = dsharpe(a, b)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    draws = np.empty(n_draws)
    for i in range(n_draws):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        draws[i] = dsharpe(a[idx], b[idx])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = float((draws <= 0.0).mean())
    return {"obs": float(obs), "lo": float(lo), "hi": float(hi), "p_onesided": p,
            "n_draws": n_draws, "block": block, "n_months": n}


def calendar_year_return(total_ret: pd.Series, year: int) -> float:
    r = total_ret.dropna()
    r = r[r.index.year == year]
    return float((1.0 + r).prod() - 1.0) if len(r) else np.nan
