"""Strategy + inference for Study 826 — Treasury Duration BAB.

Frazzini & Pedersen (2014) *betting-against-beta*, run inside the Treasury curve.

Method:

* **Total-return panel.** Build a per-ETF daily simple-return panel from adjusted Close
  (SHY, IEI, IEF, TLH, TLT).
* **Duration factor.** The equal-weight mean of the five daily returns — a single
  systematic "duration" factor that all five ETFs load on, monotone in maturity.
* **Rolling betas.** For each ETF, the trailing ``window``-day beta to the duration
  factor, ``beta_i = Cov(r_i, f) / Var(f)`` (a vectorised rolling regression). The value
  on row ``t`` uses data through ``t``; the book shifts it one day so a day-``t`` position
  is formed on the beta known at the close of ``t-1``.
* **The BAB book (Frazzini-Pedersen rank weights).** Each day, rank the five ETFs by
  their (lagged) beta. Long weights are proportional to ``(z_bar - z_i)`` for the
  below-median (low-beta) legs, short weights to ``(z_i - z_bar)`` for the above-median
  (high-beta) legs, each side normalised to sum to one. The long leg is then levered to
  **unit factor beta** (``1/beta_L``) and the short leg to **unit factor beta**
  (``1/beta_H``); the BAB return is ``(r_L - rf)/beta_L - (r_H - rf)/beta_H`` — beta-neutral
  by construction.
* **Inference.** Newey-West (HAC) *t* on the daily BAB return; a one-sample *t* and a
  pooled Welch *t* (levered-low leg vs levered-high leg) cross-check; a factor regression
  recovers the alpha and confirms the residual factor beta is ~0; a permutation placebo
  breaks the beta->return link; a costed timer charges the leveraged round-trip friction
  plus short borrow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
RF_DAILY = 0.0   # daily risk-free proxy; ~0 at the daily horizon (documented simplification)


# --------------------------------------------------------------------------- #
# Return panel + duration factor + betas
# --------------------------------------------------------------------------- #
def close_returns(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily simple total-return returns (index=date, columns=ticker)."""
    closes = pd.DataFrame({s: panel[s]["Close"] for s in panel}).sort_index()
    return closes.pct_change()


def duration_factor(ret: pd.DataFrame) -> pd.Series:
    """Equal-weight mean of the five ETF daily returns — the duration factor."""
    return ret.mean(axis=1).rename("factor")


def rolling_betas(ret: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Trailing ``window``-day beta of each ETF to the duration factor.

    ``beta_i = Cov(r_i, f) / Var(f)`` over the trailing window (value on row ``t`` uses
    returns through ``t`` inclusive). Vectorised across dates; the loop is over the five
    columns only (not over dates)."""
    f = duration_factor(ret)
    var_f = f.rolling(window, min_periods=window).var()
    out = {}
    for c in ret.columns:
        cov = ret[c].rolling(window, min_periods=window).cov(f)
        out[c] = cov / var_f
    betas = pd.DataFrame(out, index=ret.index)
    bad = np.broadcast_to((var_f <= 0).to_numpy()[:, None], betas.shape)
    return betas.mask(pd.DataFrame(bad, index=betas.index, columns=betas.columns))


# --------------------------------------------------------------------------- #
# The BAB book — rank-weighted, levered to unit beta, beta-neutral
# --------------------------------------------------------------------------- #
def _rank_weights(betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frazzini-Pedersen rank weights for one cross-section of betas.

    Returns ``(w_long, w_short)`` each summing to 1 (long on low-beta, short on
    high-beta). Median asset gets zero weight."""
    n = len(betas)
    z = betas.argsort().argsort().astype(float) + 1.0     # ranks 1..n
    zbar = z.mean()
    dev = z - zbar
    denom = np.abs(dev).sum()
    if denom <= 0:
        return np.zeros(n), np.zeros(n)
    k = 2.0 / denom
    w_long = k * np.maximum(-dev, 0.0)                    # low beta -> long
    w_short = k * np.maximum(dev, 0.0)                    # high beta -> short
    return w_long, w_short


def bab_book(ret: pd.DataFrame, window: int = 252, rf: float = RF_DAILY) -> pd.DataFrame:
    """Daily Frazzini-Pedersen BAB return inside the Treasury curve.

    On each day ``t`` the five ETFs are ranked by the beta known at the close of ``t-1``
    (one ``shift``). The low-beta legs form the long book (levered to unit factor beta
    ``1/beta_L``), the high-beta legs the short book (levered to unit factor beta
    ``1/beta_H``); ``bab = (r_L - rf)/beta_L - (r_H - rf)/beta_H``. Also returns the
    per-leg levered returns, the leg betas, and the gross leverage (long+short notional)
    for the timer. Fully vectorised across dates.
    """
    B = rolling_betas(ret, window).shift(1).to_numpy(dtype=float)   # known at t-1
    R = ret.to_numpy(dtype=float)
    idx = ret.index
    T = len(idx)

    valid = np.isfinite(B).all(axis=1) & np.isfinite(R).all(axis=1)
    rows = np.where(valid)[0]
    if len(rows) == 0:
        return pd.DataFrame(
            columns=["bab", "lev_lo", "lev_hi", "beta_lo", "beta_hi",
                     "gross_lev", "w"]
        )

    Bv, Rv = B[rows], R[rows]
    n = Bv.shape[1]

    # ranks per row (1..n), vectorised
    order = np.argsort(Bv, axis=1, kind="stable")
    z = np.empty_like(Bv)
    cols = np.arange(n)[None, :]
    np.put_along_axis(z, order, np.broadcast_to(cols + 1.0, Bv.shape), axis=1)
    zbar = z.mean(axis=1, keepdims=True)
    dev = z - zbar
    denom = np.abs(dev).sum(axis=1, keepdims=True)
    kk = np.where(denom > 0, 2.0 / denom, 0.0)
    w_long = kk * np.maximum(-dev, 0.0)     # (m, n) low beta -> long
    w_short = kk * np.maximum(dev, 0.0)      # (m, n) high beta -> short

    r_L = (w_long * Rv).sum(axis=1)
    r_H = (w_short * Rv).sum(axis=1)
    beta_L = (w_long * Bv).sum(axis=1)
    beta_H = (w_short * Bv).sum(axis=1)

    good = (beta_L > 1e-6) & (beta_H > 1e-6)
    lev_lo = np.where(good, (r_L - rf) / beta_L, np.nan)
    lev_hi = np.where(good, (r_H - rf) / beta_H, np.nan)
    bab = lev_lo - lev_hi

    # signed levered weight vector per asset (for turnover): long +w/beta_L, short -w/beta_H
    inv_bL = np.where(good, 1.0 / beta_L, 0.0)[:, None]
    inv_bH = np.where(good, 1.0 / beta_H, 0.0)[:, None]
    W = w_long * inv_bL - w_short * inv_bH   # (m, n)
    gross_lev = np.where(good, 1.0 / beta_L + 1.0 / beta_H, np.nan)

    out = pd.DataFrame(
        {
            "bab": bab, "lev_lo": lev_lo, "lev_hi": lev_hi,
            "beta_lo": np.where(good, beta_L, np.nan),
            "beta_hi": np.where(good, beta_H, np.nan),
            "gross_lev": gross_lev,
        },
        index=idx[rows],
    )
    out = out.dropna(subset=["bab"])
    # keep the levered weight matrix aligned for the timer
    out.attrs["W"] = pd.DataFrame(W, index=idx[rows], columns=ret.columns).loc[out.index]
    return out


# --------------------------------------------------------------------------- #
# Inference primitives (shared house kit — mirror of study 803)
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


def nw_alpha(bab: np.ndarray, factor: np.ndarray, lags: int = 10) -> dict:
    """Regress the BAB return on the duration factor; return alpha (bps/day), its NW t,
    and the residual factor beta (should be ~0 for a beta-neutral book)."""
    y = np.asarray(bab, dtype=float)
    x = np.asarray(factor, dtype=float)
    m = np.isfinite(y) & np.isfinite(x)
    y, x = y[m], x[m]
    n = len(y)
    if n < 5:
        return {"alpha_bps": float("nan"), "t_alpha": float("nan"), "beta_resid": float("nan")}
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    # NW t on the intercept via HAC on the residual + design (simple: HAC t of alpha
    # using the fact that alpha is mean of (y - beta*x)); use one-sample-style NW on the
    # partialled series y - beta*x.
    partialled = y - coef[1] * x
    t_alpha = newey_west_t(partialled, lags)
    return {
        "alpha_bps": float(coef[0] * 1e4),
        "t_alpha": float(t_alpha),
        "beta_resid": float(coef[1]),
    }


# --------------------------------------------------------------------------- #
# Headline stats
# --------------------------------------------------------------------------- #
def bab_stats(book: pd.DataFrame, ret: pd.DataFrame | None = None,
              nw_lags: int = 10) -> dict:
    b = book["bab"].to_numpy(dtype=float)
    b = b[np.isfinite(b)]
    n = len(b)
    sd = b.std(ddof=1) if n > 1 else float("nan")
    sharpe = b.mean() / sd * np.sqrt(TRADING_DAYS) if sd and sd > 0 else float("nan")
    out = {
        "n_days": int(len(book)),
        "bab_bps": float(np.nanmean(book["bab"].to_numpy()) * 1e4),
        "t_nw": newey_west_t(book["bab"].to_numpy(), nw_lags),
        "t_1s": one_sample_t(book["bab"].to_numpy()),
        "sharpe": float(sharpe),
        "lev_lo_bps": float(np.nanmean(book["lev_lo"].to_numpy()) * 1e4),
        "lev_hi_bps": float(np.nanmean(book["lev_hi"].to_numpy()) * 1e4),
        "welch_t": welch_t(book["lev_lo"].to_numpy(), book["lev_hi"].to_numpy()),
        "beta_lo": float(np.nanmean(book["beta_lo"].to_numpy())),
        "beta_hi": float(np.nanmean(book["beta_hi"].to_numpy())),
        "gross_lev": float(np.nanmean(book["gross_lev"].to_numpy())),
    }
    if ret is not None:
        f = duration_factor(ret).reindex(book.index).to_numpy()
        out.update(nw_alpha(book["bab"].to_numpy(), f, nw_lags))
    return out


# --------------------------------------------------------------------------- #
# Placebo — is the BAB alpha real, or a lucky alignment of the beta sort?
# --------------------------------------------------------------------------- #
def placebo_pvalue(
    ret: pd.DataFrame,
    window: int = 252,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 826,
) -> dict:
    """Keep the beta-rank book construction but read each day's forward returns from a
    **column-permuted** panel (beta->return link broken, each day's cross-sectional
    return distribution preserved). p = share of permuted worlds whose BAB mean is >=
    observed (right-tail test, the claim predicts a positive BAB)."""
    obs = float(bab_book(ret, window)["bab"].mean())
    B = rolling_betas(ret, window).shift(1).to_numpy(dtype=float)
    R = ret.to_numpy(dtype=float)
    ncol = R.shape[1]

    valid = np.isfinite(B).all(axis=1) & np.isfinite(R).all(axis=1)
    rows = np.where(valid)[0]
    Bv = B[rows]
    n = Bv.shape[1]

    order = np.argsort(Bv, axis=1, kind="stable")
    z = np.empty_like(Bv)
    cols = np.arange(n)[None, :]
    np.put_along_axis(z, order, np.broadcast_to(cols + 1.0, Bv.shape), axis=1)
    zbar = z.mean(axis=1, keepdims=True)
    dev = z - zbar
    denom = np.abs(dev).sum(axis=1, keepdims=True)
    kk = np.where(denom > 0, 2.0 / denom, 0.0)
    w_long = kk * np.maximum(-dev, 0.0)
    w_short = kk * np.maximum(dev, 0.0)
    beta_L = (w_long * Bv).sum(axis=1)
    beta_H = (w_short * Bv).sum(axis=1)
    good = (beta_L > 1e-6) & (beta_H > 1e-6)

    means = []
    Rrows = R[rows]
    m = len(rows)
    row_ar = np.arange(m)[:, None]
    for seed in range(n_seeds):
        rng = np.random.default_rng(base_seed + seed)
        for _ in range(n_draws_per_seed):
            perm = rng.permutation(ncol)
            Rp = Rrows[row_ar, perm[None, :]]         # permute the return columns
            r_L = (w_long * Rp).sum(axis=1)
            r_H = (w_short * Rp).sum(axis=1)
            lev_lo = np.where(good, (r_L - RF_DAILY) / beta_L, np.nan)
            lev_hi = np.where(good, (r_H - RF_DAILY) / beta_H, np.nan)
            means.append(np.nanmean(lev_lo - lev_hi))
    means = np.asarray(means)
    return {
        "obs_bps": obs * 1e4,
        "placebo_mean_bps": float(means.mean() * 1e4) if len(means) else float("nan"),
        "placebo_sd_bps": float(means.std(ddof=1) * 1e4) if len(means) > 1 else float("nan"),
        "p_value": float((means >= obs).mean()) if len(means) else float("nan"),
        "n_draws": len(means),
        "draws_bps": means * 1e4,
    }


# --------------------------------------------------------------------------- #
# The costed timer
# --------------------------------------------------------------------------- #
def timer_stats(
    book: pd.DataFrame,
    cost_bps: float = 2.0,
    borrow_bps_yr: float = 50.0,
) -> dict:
    """Cost the levered BAB book.

    The BAB book carries a **large gross notional** — the low-beta long leg is levered by
    ``1/beta_L`` (SHY's beta to the curve is small, so this leverage is big). We charge a
    one-way cost on the daily **turnover** of the signed levered weight vector, plus a
    borrow charge on the short notional (``1/beta_H``). Turnover is the sum of absolute
    day-over-day changes in the per-asset levered weights."""
    W = book.attrs.get("W")
    if W is None or W.empty:
        return {"n_days": 0, "gross_bps": float("nan"), "net_bps": float("nan")}
    b = book["bab"].to_numpy(dtype=float)
    turnover = W.diff().abs().sum(axis=1).to_numpy().copy()
    turnover[0] = np.nansum(np.abs(W.iloc[0].to_numpy()))   # day-1 = build the book
    short_notional = (1.0 / book["beta_hi"]).to_numpy()      # 1/beta_H = short leverage
    cost = (cost_bps / 1e4) * turnover
    borrow = (borrow_bps_yr / 1e4 / TRADING_DAYS) * short_notional
    net = b - cost - borrow
    n = len(net)
    sd = np.nanstd(net, ddof=1) if n > 1 else float("nan")
    sharpe = np.nanmean(net) / sd * np.sqrt(TRADING_DAYS) if sd and sd > 0 else float("nan")
    return {
        "n_days": n,
        "gross_bps": float(np.nanmean(b) * 1e4),
        "net_bps": float(np.nanmean(net) * 1e4),
        "cost_bps_per_day": float(np.nanmean(cost) * 1e4),
        "borrow_bps_per_day": float(np.nanmean(borrow) * 1e4),
        "avg_turnover": float(np.nanmean(turnover)),
        "ann_net_pct": float(np.nanmean(net) * TRADING_DAYS * 100),
        "sharpe_net": float(sharpe),
        "t_net": one_sample_t(net),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: dict[str, pd.DataFrame], window: int = 252) -> dict:
    """Run the headline BAB stats on a synthetic panel."""
    ret = close_returns(panel)
    book = bab_book(ret, window)
    ts = bab_stats(book, ret)
    return {"bab_bps": ts["bab_bps"], "t_nw": ts["t_nw"],
            "welch_t": ts["welch_t"], "sharpe": ts["sharpe"],
            "beta_resid": ts.get("beta_resid", float("nan")),
            "n_days": ts["n_days"]}
