"""Strategy + inference for Study 830 — BAB Across Asset Classes.

The claim (Frazzini & Pedersen 2014, at the multi-asset level): rank a basket of
**asset classes** by their beta to a common equal-weight multi-asset market
portfolio; the low-beta assets are over-compensated per unit of risk and the
high-beta under-compensated (a *flat* cross-asset security-market line). Build the
**BAB factor**: long the low-beta assets **levered to unit beta**, short the high-beta
assets **de-levered to unit beta**, so the book is ex-ante beta-neutral and its mean
return is (up to the leverage) an alpha.

This is distinct from:

* [238-betting-against-beta](../../238-betting-against-beta/) — BAB **in the US stock
  cross-section** (thousands of single names), the original Frazzini-Pedersen test.
  Here the "assets" are nine *asset classes*, not stocks.
* [660-carry-everywhere](../../660-carry-everywhere/) — carry (yield/roll) sorted
  **everywhere**; the sort variable is *carry*, not *beta to a common market*.
* [68-all-weather-risk-parity](../../68-all-weather-risk-parity/) — inverse-vol /
  risk-parity weighting of asset classes (a long-only allocation); BAB is a
  **long-short, beta-neutral** factor, not an allocation.

Method:

* **Total-return closes -> daily simple returns**, one column per asset class.
* **Multi-asset market** = the equal-weight average of the asset returns each day.
* **Rolling beta** of each asset to the market (Frazzini-Pedersen construction:
  correlation over a long window x vol ratio over a short window, shrunk toward 1),
  vectorised via pandas ``rolling``.
* **Point-in-time BAB.** On each day ``t`` the betas known at the close of ``t-1``
  (one ``shift``) rank the assets; rank weights load long on the low-beta side and
  short on the high-beta side; each leg is scaled to unit beta and the two legs
  netted::

        r_BAB,t = (1/beta_L) * (w_L . r_t) - (1/beta_H) * (w_H . r_t)

  (risk-free approximated at 0 — a self-financing, beta-neutral daily book).
* **Inference.** Newey-West (HAC) *t* on the daily BAB return; a one-sample *t* and a
  CAPM alpha (HAC) cross-check that the mean is not just market exposure; a
  permutation placebo breaks the beta->return link; a costed timer charges realized
  turnover on the levered book plus borrow on the short leg.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Return frame + rolling betas
# --------------------------------------------------------------------------- #
def close_returns(panel_or_frame) -> pd.DataFrame:
    """Daily simple returns (index=date, columns=asset).

    Accepts either a ``{ticker: DataFrame[Close]}`` panel or a wide close frame.
    """
    if isinstance(panel_or_frame, dict):
        closes = pd.DataFrame(
            {s: panel_or_frame[s]["Close"] for s in panel_or_frame}
        ).sort_index()
    else:
        closes = panel_or_frame.sort_index()
    return closes.astype(float).pct_change()


def market_return(ret: pd.DataFrame) -> pd.Series:
    """Equal-weight multi-asset market return (mean across asset columns)."""
    return ret.mean(axis=1).rename("MKT")


def rolling_betas(
    ret: pd.DataFrame,
    corr_window: int = 252,
    vol_window: int = 63,
    shrink: float = 0.6,
) -> pd.DataFrame:
    """Frazzini-Pedersen rolling betas of each asset to the equal-weight market.

    ``beta_i = shrink * (rho_i * sigma_i / sigma_m) + (1 - shrink) * 1``, where the
    correlation ``rho_i`` uses the long ``corr_window`` and the volatilities the short
    ``vol_window`` (FP's asymmetric estimator), shrunk toward the cross-sectional prior
    of 1. Row ``t`` uses returns through ``t`` (inclusive); the BAB build shifts by one
    day so a day-``t`` position is formed on information known at ``t-1``.
    """
    m = market_return(ret)
    sig_m = m.rolling(vol_window, min_periods=vol_window).std()
    sig_i = ret.rolling(vol_window, min_periods=vol_window).std()
    rho = ret.rolling(corr_window, min_periods=corr_window).corr(m)
    beta_ts = rho.mul(sig_i).div(sig_m, axis=0)
    return shrink * beta_ts + (1.0 - shrink)


# --------------------------------------------------------------------------- #
# The BAB construction — long low-beta (levered), short high-beta (de-levered)
# --------------------------------------------------------------------------- #
def bab_series(
    ret: pd.DataFrame,
    corr_window: int = 252,
    vol_window: int = 63,
    shrink: float = 0.6,
    min_names: int = 6,
) -> pd.DataFrame:
    """Daily beta-neutral BAB factor return + book diagnostics.

    On each day ``t`` the assets are ranked by the beta known at the close of ``t-1``
    (one ``shift``). Frazzini-Pedersen rank weights: with ranks ``z`` (1..n) and mean
    rank ``zbar``, ``w_L ∝ max(zbar - z, 0)`` (low beta, the long) and
    ``w_H ∝ max(z - zbar, 0)`` (high beta, the short), each normalised to sum to 1.
    Long leg is levered by ``1/beta_L`` and short leg by ``1/beta_H`` so both carry
    unit beta::

        r_BAB = (1/beta_L)(w_L . r) - (1/beta_H)(w_H . r)

    Returns a frame indexed by date with columns:
      * ``bab``   — the beta-neutral factor return,
      * ``lo``/``hi`` — the (unlevered) long/short leg returns,
      * ``beta_L``/``beta_H`` — the leg betas,
      * ``turnover`` — sum |Δ position| on the levered net book (for the timer),
      * ``short_notional`` — gross short exposure (for borrow).
    Days with fewer than ``min_names`` ranked assets are dropped.
    """
    B = rolling_betas(ret, corr_window, vol_window, shrink).shift(1).to_numpy(dtype=float)
    R = ret.to_numpy(dtype=float)
    idx = ret.index
    n_assets = R.shape[1]

    dates, out = [], []
    prev_pos = np.zeros(n_assets)
    for i in range(len(idx)):
        b = B[i]
        valid = np.where(np.isfinite(b))[0]
        if len(valid) < min_names:
            prev_pos = np.zeros(n_assets)
            continue
        bv = b[valid]
        # ranks 1..n (ties -> average rank)
        z = _avg_rank(bv)
        zbar = z.mean()
        wL = np.maximum(zbar - z, 0.0)
        wH = np.maximum(z - zbar, 0.0)
        sL, sH = wL.sum(), wH.sum()
        if sL <= 0 or sH <= 0:
            prev_pos = np.zeros(n_assets)
            continue
        wL /= sL
        wH /= sH
        beta_L = float(wL @ bv)
        beta_H = float(wH @ bv)
        if not (beta_L > 0 and beta_H > 0):
            prev_pos = np.zeros(n_assets)
            continue
        rr = R[i, valid]
        lo = float(wL @ rr)
        hi = float(wH @ rr)
        bab = lo / beta_L - hi / beta_H
        # net levered position vector (for turnover / borrow)
        pos = np.zeros(n_assets)
        pos[valid] = wL / beta_L - wH / beta_H
        turnover = float(np.abs(pos - prev_pos).sum())
        short_notional = float(-pos[pos < 0].sum())
        prev_pos = pos
        dates.append(idx[i])
        out.append((bab, lo, hi, beta_L, beta_H, turnover, short_notional))

    return pd.DataFrame(
        out,
        index=pd.Index(dates, name=idx.name),
        columns=["bab", "lo", "hi", "beta_L", "beta_H", "turnover", "short_notional"],
    )


def _avg_rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (1..n), ties averaged — a scipy-free rankdata."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # average ties
    uniq, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


# --------------------------------------------------------------------------- #
# Inference primitives (shared with the sibling studies)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
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


def capm_alpha(bab: np.ndarray, mkt: np.ndarray, lags: int = 10) -> dict:
    """CAPM regression of the BAB return on the market: alpha, realized beta, HAC t.

    Confirms the mean BAB return is an *alpha* (a beta-neutral book), not disguised
    market exposure. HAC (Newey-West) standard errors via statsmodels.
    """
    import statsmodels.api as sm

    y = np.asarray(bab, dtype=float)
    x = np.asarray(mkt, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    if len(y) < 30:
        return {"alpha_bps": float("nan"), "alpha_t": float("nan"),
                "beta": float("nan"), "beta_t": float("nan"), "n": int(len(y))}
    X = sm.add_constant(x)
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {
        "alpha_bps": float(res.params[0] * 1e4),
        "alpha_t": float(res.tvalues[0]),
        "beta": float(res.params[1]),
        "beta_t": float(res.tvalues[1]),
        "n": int(len(y)),
    }


# --------------------------------------------------------------------------- #
# Headline stats
# --------------------------------------------------------------------------- #
def bab_stats(book: pd.DataFrame, mkt: pd.Series | None = None, nw_lags: int = 10) -> dict:
    b = book["bab"].to_numpy(dtype=float)
    b = b[~np.isnan(b)]
    n = len(b)
    mean = float(np.mean(b)) if n else float("nan")
    sd = float(np.std(b, ddof=1)) if n > 1 else float("nan")
    sharpe = mean / sd * np.sqrt(TRADING_DAYS) if sd and sd > 0 else float("nan")
    out = {
        "n_days": n,
        "bab_bps": mean * 1e4,
        "t_nw": newey_west_t(b, nw_lags),
        "t_1s": one_sample_t(b),
        "sharpe": sharpe,
        "lo_bps": float(np.nanmean(book["lo"].to_numpy()) * 1e4),
        "hi_bps": float(np.nanmean(book["hi"].to_numpy()) * 1e4),
        "beta_L": float(np.nanmean(book["beta_L"].to_numpy())),
        "beta_H": float(np.nanmean(book["beta_H"].to_numpy())),
    }
    if mkt is not None:
        m = mkt.reindex(book.index).to_numpy(dtype=float)
        ca = capm_alpha(book["bab"].to_numpy(), m, nw_lags)
        out.update(alpha_bps=ca["alpha_bps"], alpha_t=ca["alpha_t"],
                   realized_beta=ca["beta"])
    return out


# --------------------------------------------------------------------------- #
# Placebo — is the alpha real, or a lucky alignment of the beta sort?
# --------------------------------------------------------------------------- #
def placebo_pvalue(
    ret: pd.DataFrame,
    corr_window: int = 252,
    vol_window: int = 63,
    shrink: float = 0.6,
    min_names: int = 6,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 830,
) -> dict:
    """Keep the beta sort but read each day's asset returns from a **column-permuted**
    panel (beta->return link broken, each day's cross-asset return distribution
    preserved). Two-sided: p = share of permuted BAB means at least as extreme in
    absolute value as observed.
    """
    obs = float(bab_series(ret, corr_window, vol_window, shrink, min_names)["bab"].mean())
    B = rolling_betas(ret, corr_window, vol_window, shrink).shift(1).to_numpy(dtype=float)
    R = ret.to_numpy(dtype=float)
    n_assets = R.shape[1]

    rows, WL, WH, betaL, betaH = [], [], [], [], []
    for i in range(len(ret.index)):
        b = B[i]
        valid = np.where(np.isfinite(b))[0]
        if len(valid) < min_names:
            continue
        bv = b[valid]
        z = _avg_rank(bv)
        zbar = z.mean()
        wL = np.maximum(zbar - z, 0.0)
        wH = np.maximum(z - zbar, 0.0)
        if wL.sum() <= 0 or wH.sum() <= 0:
            continue
        wL /= wL.sum(); wH /= wH.sum()
        bL = float(wL @ bv); bH = float(wH @ bv)
        if not (bL > 0 and bH > 0):
            continue
        # embed leg weights back into full-width vectors (aligned to columns)
        fL = np.zeros(n_assets); fL[valid] = wL
        fH = np.zeros(n_assets); fH[valid] = wH
        rows.append(i); WL.append(fL); WH.append(fH); betaL.append(bL); betaH.append(bH)

    rows = np.asarray(rows)
    means = []
    if len(rows):
        M = R[rows]                       # (T, N) forward returns on active days
        WL = np.asarray(WL); WH = np.asarray(WH)
        betaL = np.asarray(betaL); betaH = np.asarray(betaH)
        for seed in range(n_seeds):
            rng = np.random.default_rng(base_seed + seed)
            for _ in range(n_draws_per_seed):
                perm = rng.permutation(n_assets)
                Mp = M[:, perm]
                lo = (WL * Mp).sum(axis=1)
                hi = (WH * Mp).sum(axis=1)
                bab = lo / betaL - hi / betaH
                means.append(float(np.mean(bab)))
    means = np.asarray(means)
    if len(means):
        sd = means.std(ddof=1) if len(means) > 1 else float("nan")
        p = float((np.abs(means) >= abs(obs)).mean())
        sigma = abs(obs - means.mean()) / sd if sd and sd > 0 else float("nan")
    else:
        sd, p, sigma = float("nan"), float("nan"), float("nan")
    return {
        "obs_bps": obs * 1e4,
        "placebo_mean_bps": float(means.mean() * 1e4) if len(means) else float("nan"),
        "placebo_sd_bps": float(sd * 1e4) if len(means) else float("nan"),
        "p_value": p,
        "sigma": sigma,
        "n_draws": int(len(means)),
        "draws_bps": means * 1e4,
    }


# --------------------------------------------------------------------------- #
# The costed timer
# --------------------------------------------------------------------------- #
def timer_stats(
    book: pd.DataFrame,
    cost_bps: float = 5.0,
    borrow_bps_yr: float = 50.0,
) -> dict:
    """Cost the levered BAB book.

    The BAB book is levered (``1/beta_L`` on the long low-beta leg, ``1/beta_H`` on the
    short high-beta leg), so gross notional exceeds 1. We charge the realized daily
    **turnover** of the net levered position (``sum|Δ weight|``) at ``cost_bps`` one-way,
    plus borrow on the short notional. Net Sharpe / *t* tell you whether a small
    beta-neutral alpha survives the friction of running a levered multi-asset book.
    """
    b = book["bab"].to_numpy(dtype=float)
    turn = book["turnover"].to_numpy(dtype=float)
    shortn = book["short_notional"].to_numpy(dtype=float)
    ok = np.isfinite(b)
    b, turn, shortn = b[ok], turn[ok], shortn[ok]
    n = len(b)
    cost = turn * (cost_bps / 1e4)
    borrow = shortn * ((borrow_bps_yr / 1e4) / 365.0)
    net = b - cost - borrow
    gross_mean = float(np.mean(b)) if n else float("nan")
    net_mean = float(np.mean(net)) if n else float("nan")
    sd = float(np.std(net, ddof=1)) if n > 1 else float("nan")
    sharpe = net_mean / sd * np.sqrt(TRADING_DAYS) if sd and sd > 0 else float("nan")
    return {
        "n_days": n,
        "gross_bps": gross_mean * 1e4,
        "net_bps": net_mean * 1e4,
        "cost_bps_per_day": float(np.mean(cost + borrow) * 1e4) if n else float("nan"),
        "avg_turnover": float(np.mean(turn)) if n else float("nan"),
        "avg_gross": float(np.mean(1.0 / book["beta_L"].to_numpy()[ok]
                                   + 1.0 / book["beta_H"].to_numpy()[ok])) if n else float("nan"),
        "ann_net_pct": net_mean * TRADING_DAYS * 100,
        "sharpe_net": sharpe,
        "t_net": one_sample_t(net),
        "t_net_nw": newey_west_t(net, 10),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(
    panel_or_frame,
    corr_window: int = 252,
    vol_window: int = 63,
    shrink: float = 0.6,
    min_names: int = 6,
) -> dict:
    """Run the headline BAB stats on a synthetic panel/frame."""
    ret = close_returns(panel_or_frame)
    book = bab_series(ret, corr_window, vol_window, shrink, min_names)
    ts = bab_stats(book, market_return(ret))
    return {"bab_bps": ts["bab_bps"], "t_nw": ts["t_nw"],
            "sharpe": ts["sharpe"], "alpha_t": ts.get("alpha_t", float("nan")),
            "n_days": ts["n_days"]}
