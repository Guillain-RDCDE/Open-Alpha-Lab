"""Strategy + inference for Study 875 — Idiosyncratic-Vol Change.

The claim under test: sort a cross-section not on the **level** of idiosyncratic
volatility (the Ang-Hodrick-Xing-Zhang puzzle, study 501) but on its **change** — is a
name's residual (market-model) vol **rising** or **falling**? The story: a rising
idio-vol signals a deteriorating information environment / rising disagreement, so those
names go on to earn **less**, while names whose idio-vol is **falling** re-rate. So a
long **falling-idio-vol** / short **rising-idio-vol** book should earn a positive spread.
The honest question is whether this delta-IVOL is anything beyond the idio-vol *level*
effect (501) or the *total*-vol trend (817).

This is distinct from:

* [501-idiosyncratic-volatility](../../501-idiosyncratic-volatility/) — the **level** of
  residual (market-model) vol (low-idio-vol names out-earn high-idio-vol names). This
  study sorts on the **change** in that residual vol, not its level; we regress the
  delta *out of* the level below (the additivity test).
* [817-realized-volatility-trend](../../817-realized-volatility-trend/) — the trend in
  **total** realized vol (`vol21/vol63 - 1`). This study uses the **residual**
  (idiosyncratic, market-model) vol and a recent-vs-prior **change**, stripping out the
  common market-vol move that a total-vol measure still carries.
* [330-low-volatility](../../330-low-volatility/) — the low-**total**-vol *level*
  anomaly, again a level and a total (not residual) vol.

Method:

* **Close-to-close returns.** Build a per-name daily simple-return panel from adjusted
  Close; the **market** is the equal-weight cross-sectional mean return each day.
* **Idiosyncratic vol.** On each name, the rolling ``window``-day market-model residual
  vol, computed vectorised via the identity
  ``resid_var = var(r) - cov(r, mkt)**2 / var(mkt)`` (the variance of the CAPM residual
  ``r - a - b*mkt``), value on row ``t`` uses returns through ``t``.
* **Delta-IVOL.** The **change** in idio-vol: the recent ``window``-day residual vol
  minus the prior (non-overlapping) ``window``-day residual vol (``ivol - ivol.shift(window)``).
  Positive = idio-vol **rising**, negative = **falling**.
* **Point-in-time sort.** On each day ``t`` rank the cross-section by the delta-IVOL
  known at the close of ``t-1`` (one ``shift``) and hold day ``t``. Long the bottom
  ``frac`` (most falling idio-vol), short the top ``frac`` (most rising); equal weight.
* **Inference.** Newey-West (HAC) *t* on the daily long-short spread; a one-sample *t*
  and a pooled Welch *t* (falling book vs rising book) cross-check; a permutation placebo
  breaks the signal->outcome link; an additivity regression against the idio-vol *level*
  sort (501); a costed timer charges the round-trip friction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Return panel + market + signal
# --------------------------------------------------------------------------- #
def close_returns(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily simple close-to-close returns (index=date, columns=ticker)."""
    closes = pd.DataFrame({s: panel[s]["Close"] for s in panel}).sort_index()
    return closes.pct_change()


def market_return(ret: pd.DataFrame) -> pd.Series:
    """Equal-weight cross-sectional mean return each day — the market-model factor."""
    return ret.mean(axis=1, skipna=True)


def idio_vol(ret: pd.DataFrame, mkt: pd.Series, window: int = 21) -> pd.DataFrame:
    """Rolling ``window``-day market-model **residual** (idiosyncratic) vol, per name.

    For each name the CAPM residual is ``r - a - b*mkt`` with ``b = cov(r,mkt)/var(mkt)``;
    its variance is the identity ``var(r) - cov(r,mkt)**2 / var(mkt)``. We compute the
    rolling second moments vectorised (no per-date regression) so the whole panel's
    residual vol falls out at once. Value on row ``t`` uses returns through ``t``
    (inclusive); the sort shifts by one day so a day-``t`` position is formed on
    information known at ``t-1``.
    """
    r = ret
    rm = r.mul(mkt, axis=0)
    er = r.rolling(window, min_periods=window).mean()
    er2 = (r ** 2).rolling(window, min_periods=window).mean()
    em = mkt.rolling(window, min_periods=window).mean()
    em2 = (mkt ** 2).rolling(window, min_periods=window).mean()
    erm = rm.rolling(window, min_periods=window).mean()

    var_r = er2 - er ** 2
    var_m = (em2 - em ** 2)
    cov_rm = erm.sub(er.mul(em, axis=0))
    # residual variance = var(r) - cov^2/var(m); broadcast var_m (a Series) over columns
    resid_var = var_r.sub(cov_rm.pow(2).div(var_m, axis=0))
    resid_var = resid_var.clip(lower=0.0)
    out = np.sqrt(resid_var)
    return out.where(var_m.gt(0), axis=0)


def delta_ivol(ret: pd.DataFrame, mkt: pd.Series, window: int = 21) -> pd.DataFrame:
    """Delta-IVOL: recent ``window``-day residual vol minus the prior ``window``-day one.

    ``ivol - ivol.shift(window)`` — the change between two **non-overlapping** windows
    (recent ``[t-window, t]`` vs prior ``[t-2*window, t-window]``). Positive = idio-vol
    **rising**, negative = idio-vol **falling**. Value on row ``t`` uses returns through
    ``t``; :func:`delta_spreads` shifts by one day for a point-in-time position.
    """
    iv = idio_vol(ret, mkt, window)
    return iv - iv.shift(window)


# --------------------------------------------------------------------------- #
# The cross-sectional sort -> long-falling-ivol / short-rising-ivol spread
# --------------------------------------------------------------------------- #
def _fractile_spreads(sig: pd.DataFrame, ret: pd.DataFrame, frac: float,
                      min_names: int) -> pd.DataFrame:
    """Daily equal-weight bottom-minus-top fractile spread on a (shifted) signal.

    ``lo`` = mean forward day-``t`` return of the bottom ``frac`` (the long); ``hi`` =
    mean of the top ``frac`` (the short); ``spread = lo - hi``. Days with fewer than
    ``min_names`` ranked names are dropped. Fully vectorised over names per day.
    """
    S = sig.to_numpy(dtype=float)
    R = ret.to_numpy(dtype=float)
    idx = ret.index
    out_spread, out_lo, out_hi, out_n, out_t = [], [], [], [], []
    for i in range(len(idx)):
        row = S[i]
        valid = np.where(~np.isnan(row))[0]
        n = len(valid)
        if n < min_names:
            continue
        k = max(1, int(np.floor(n * frac)))
        order = valid[np.argsort(row[valid], kind="stable")]
        low = order[:k]        # bottom of signal  -> long
        high = order[-k:]      # top of signal     -> short
        rr = R[i]
        lo = float(np.nanmean(rr[low]))
        hi = float(np.nanmean(rr[high]))
        out_spread.append(lo - hi); out_lo.append(lo); out_hi.append(hi)
        out_n.append(n); out_t.append(idx[i])
    return pd.DataFrame(
        {"spread": out_spread, "lo": out_lo, "hi": out_hi, "n": out_n}, index=out_t
    ).sort_index()


def delta_spreads(
    ret: pd.DataFrame,
    mkt: pd.Series | None = None,
    window: int = 21,
    frac: float = 0.3,
    min_names: int = 10,
) -> pd.DataFrame:
    """Daily equal-weight bottom-minus-top delta-IVOL fractile spread.

    On each day ``t`` names are ranked by the delta-IVOL known at the close of ``t-1``
    (one ``shift``). ``lo`` = mean forward day-``t`` return of the bottom ``frac`` (most
    falling idio-vol, the long); ``hi`` = mean of the top ``frac`` (most rising idio-vol,
    the short). ``spread = lo - hi`` (long falling-idio-vol, short rising-idio-vol).
    """
    if mkt is None:
        mkt = market_return(ret)
    sig = delta_ivol(ret, mkt, window).shift(1)  # known at close t-1
    return _fractile_spreads(sig, ret, frac, min_names)


# --------------------------------------------------------------------------- #
# Inference primitives
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


# --------------------------------------------------------------------------- #
# Headline stats
# --------------------------------------------------------------------------- #
def delta_stats(spreads: pd.DataFrame, nw_lags: int = 10) -> dict:
    sp = spreads["spread"].to_numpy(dtype=float)
    return {
        "n_days": int(len(spreads)),
        "spread_bps": float(np.nanmean(sp) * 1e4),
        "t_nw": newey_west_t(sp, nw_lags),
        "t_1s": one_sample_t(sp),
        "lo_bps": float(np.nanmean(spreads["lo"].to_numpy()) * 1e4),
        "hi_bps": float(np.nanmean(spreads["hi"].to_numpy()) * 1e4),
        "welch_t": welch_t(spreads["lo"].to_numpy(), spreads["hi"].to_numpy()),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the spread real, or a lucky alignment of the sort?
# --------------------------------------------------------------------------- #
def placebo_pvalue(
    ret: pd.DataFrame,
    window: int = 21,
    frac: float = 0.3,
    min_names: int = 10,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 875,
) -> dict:
    """Keep the delta-IVOL sort but read each day's forward return from a
    **column-permuted** panel (signal->outcome link broken, each day's cross-sectional
    distribution preserved). p = share of permuted worlds whose spread mean is >=
    observed (right-tail test on the long-falling / short-rising spread)."""
    cols = list(ret.columns)
    ncol = len(cols)
    mkt = market_return(ret)
    sig = delta_ivol(ret, mkt, window).shift(1)
    obs = float(delta_spreads(ret, mkt, window, frac, min_names)["spread"].mean())

    ret_mat = ret.to_numpy(dtype=float)
    pos_of = {c: i for i, c in enumerate(cols)}
    rows_idx, lows, highs = [], [], []
    row_lookup = {t: r for r, t in enumerate(ret.index)}
    for t in ret.index:
        s = sig.loc[t].dropna()
        if len(s) < min_names:
            continue
        k = max(1, int(np.floor(len(s) * frac)))
        order = s.sort_values()
        rows_idx.append(row_lookup[t])
        lows.append(np.array([pos_of[c] for c in order.index[:k]]))
        highs.append(np.array([pos_of[c] for c in order.index[-k:]]))
    rows_idx = np.asarray(rows_idx)

    means = []
    if len(rows_idx):
        M = ret_mat[rows_idx]
        kl = max(len(a) for a in lows)
        kh = max(len(a) for a in highs)

        def _pad(books, kmax):
            P = np.zeros((len(books), kmax), dtype=int)
            V = np.zeros((len(books), kmax), dtype=bool)
            for j, a in enumerate(books):
                P[j, :len(a)] = a
                V[j, :len(a)] = True
            return P, V

        LOW, LOWv = _pad(lows, kl)
        HIGH, HIGHv = _pad(highs, kh)
        rows_ar = np.arange(len(rows_idx))[:, None]

        def _masked_mean(pos, valid, perm):
            vals = M[rows_ar, perm[pos]]
            vals = np.where(valid, vals, np.nan)
            return np.nanmean(vals, axis=1)

        for seed in range(n_seeds):
            rng = np.random.default_rng(base_seed + seed)
            for _ in range(n_draws_per_seed):
                perm = rng.permutation(ncol)
                lo_v = _masked_mean(LOW, LOWv, perm)
                hi_v = _masked_mean(HIGH, HIGHv, perm)
                means.append(np.nanmean(lo_v - hi_v))
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
# Additivity vs the idio-vol LEVEL effect (study 501)
# --------------------------------------------------------------------------- #
def level_spreads(
    ret: pd.DataFrame,
    mkt: pd.Series | None = None,
    window: int = 21,
    frac: float = 0.3,
    min_names: int = 10,
) -> pd.DataFrame:
    """The idio-vol LEVEL sort (501-style) — long low idio-vol, short high idio-vol.

    Same machinery as :func:`delta_spreads`, but the ranking signal is the *level* of
    trailing ``window``-day residual vol (not its change). Used to ask whether the
    idio-vol *change* is additive to the classic idio-vol level puzzle (study 501).
    """
    if mkt is None:
        mkt = market_return(ret)
    sig = idio_vol(ret, mkt, window).shift(1)
    return _fractile_spreads(sig, ret, frac, min_names)


def additivity(ret: pd.DataFrame, window: int = 21, frac: float = 0.3,
               nw_lags: int = 10) -> dict:
    """Regress the delta-IVOL spread on the idio-vol LEVEL spread; is the delta additive?

    Fits ``delta_spread_t = a + b * level_spread_t + e_t`` on the overlapping days.
    ``a`` (intercept) is the part of the delta spread orthogonal to the level puzzle — a
    Newey-West *t* on the residual ``delta - b*level`` says whether the change carries
    anything the level does not. Also reports the correlation of the two spreads.
    """
    mkt = market_return(ret)
    dl = delta_spreads(ret, mkt, window, frac)
    lv = level_spreads(ret, mkt, window, frac)
    joined = dl[["spread"]].join(lv[["spread"]], how="inner", lsuffix="_dl", rsuffix="_lv")
    y = joined["spread_dl"].to_numpy(dtype=float)
    x = joined["spread_lv"].to_numpy(dtype=float)
    n = len(y)
    if n < 3:
        return {"n_days": n, "corr": float("nan"), "beta": float("nan"),
                "alpha_bps": float("nan"), "alpha_t_nw": float("nan"),
                "level_bps": float("nan"), "level_t_nw": float("nan")}
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    resid = y - b * x            # delta spread net of its level-explained part
    return {
        "n_days": n,
        "corr": float(np.corrcoef(x, y)[0, 1]),
        "beta": b,
        "alpha_bps": a * 1e4,
        "alpha_t_nw": newey_west_t(resid, nw_lags),
        "level_bps": float(np.nanmean(x) * 1e4),
        "level_t_nw": newey_west_t(x, nw_lags),
    }


# --------------------------------------------------------------------------- #
# The costed timer
# --------------------------------------------------------------------------- #
def timer_stats(
    spreads: pd.DataFrame,
    cost_bps: float = 5.0,
    borrow_bps_yr: float = 50.0,
) -> dict:
    """Cost the long-falling-idio-vol / short-rising-idio-vol book.

    The signal is a trailing residual-vol change that turns over roughly monthly, but
    names drift across the fractile boundary daily; we charge a conservative daily
    round-trip on the fraction of the 2x-NAV book that rotates. To stay comparable to
    the desk's other cross-sectional timers we charge 2 sides x one-way cost x NAV per
    day on the long-short book, plus borrow on the short leg.
    """
    sp = spreads["spread"].to_numpy(dtype=float)
    sp = sp[~np.isnan(sp)]
    n = len(sp)
    round_trip_cost = 2.0 * cost_bps / 1e4
    borrow_daily = (borrow_bps_yr / 1e4) / 365.0
    net = sp - round_trip_cost - borrow_daily
    gross_mean = float(sp.mean())
    net_mean = float(net.mean())
    sd = float(net.std(ddof=1)) if n > 1 else float("nan")
    sharpe = net_mean / sd * np.sqrt(TRADING_DAYS) if sd and sd > 0 else float("nan")
    return {
        "n_days": n,
        "gross_bps": gross_mean * 1e4,
        "net_bps": net_mean * 1e4,
        "cost_bps_per_day": (round_trip_cost + borrow_daily) * 1e4,
        "ann_net_pct": net_mean * TRADING_DAYS * 100,
        "sharpe_net": sharpe,
        "t_net": one_sample_t(net),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: dict[str, pd.DataFrame], window: int = 21,
                     frac: float = 0.3) -> dict:
    """Run the headline delta-IVOL stats on a synthetic panel."""
    ret = close_returns(panel)
    sp = delta_spreads(ret, None, window, frac)
    ts = delta_stats(sp)
    return {"spread_bps": ts["spread_bps"], "t_nw": ts["t_nw"],
            "welch_t": ts["welch_t"], "n_days": ts["n_days"]}
