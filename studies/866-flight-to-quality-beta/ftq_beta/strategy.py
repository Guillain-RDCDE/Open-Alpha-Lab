"""Strategy + inference for Study 866 — Flight-to-Quality Beta.

The claim: some stocks are *true defensives* — they rally with long Treasuries on
**risk-off** days. For each name we estimate a **flight-to-quality beta** (``beta_ftq``)
— its beta to the **TLT** daily return, computed **only on down-SPY days**::

    beta_ftq = cov(r_i, r_TLT | r_SPY < 0) / var(r_TLT | r_SPY < 0)

A high ``beta_ftq`` name reliably co-moves with the safe-haven bid when the market
falls (a good crash hedge). The CAPM-of-insurance prediction is two-sided:

1. **Return penalty.** Investors overpay for the hedge, so high-FTQ-beta names earn a
   **lower** average return. We test the equal-weight **long low-FTQ / short high-FTQ**
   monthly spread (betting *against* the insurance) — the claim predicts it is
   **positive**.
2. **Crash protection.** On the worst market days the high-FTQ book should lose **less**
   than the low-FTQ book. We compare the two books' mean return on the risk-off tail.

This is distinct from:

* [332-downside-beta](../../332-downside-beta/) — beta to the **equity market** on down
  days (Ang-Chen-Xing β⁻). This study's conditioner is the **Treasury (TLT)** return —
  co-movement with the *safe haven*, not with the falling market itself.
* [238-betting-against-beta](../../238-betting-against-beta/) — the Frazzini-Pedersen
  BAB tilt on **market** beta. FTQ beta is a *cross-asset* (equity↔bond) loading
  measured only in sell-offs.
* [246-defensive-sectors](../../246-defensive-sectors/) — a **sector** label (staples /
  utilities / health-care) as the defensive proxy. Here "defensive" is *revealed* from
  the tape by each name's own bond co-movement in sell-offs, not assigned by GICS.
* [69-safe-haven](../../69-safe-haven/) — whether a whole **asset class** (gold, bonds)
  hedges equities. This is a *within-equity cross-section* sort on how each stock loads
  on that safe-haven bid.

Method:

* **Return panels.** Daily simple close-to-close returns for the cross-section; daily
  simple returns for TLT and SPY.
* **Conditional FTQ beta, vectorised across names.** On the trailing ``lookback_days``
  window ending at each month-end, restrict to days where ``r_SPY < threshold``
  (default 0 = a down market day) and regress every name on ``r_TLT`` at once (one
  matrix contraction), yielding the cross-section of FTQ betas known at that month-end.
* **Point-in-time sort.** The FTQ beta for month ``t`` is formed on the window ending
  at the last day of month ``t-1``; the book holds month ``t``'s realised returns. One
  documented execution lag, applied once.
* **Inference.** Newey-West (HAC) *t* on the monthly long-short spread; one-sample *t*;
  pooled Welch *t* (low-FTQ vs high-FTQ book); a permutation placebo that breaks the
  signal→outcome link; a costed timer; a crash-day drawdown comparison; a seeded
  synthetic positive control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTHS = 12


# --------------------------------------------------------------------------- #
# Return panels
# --------------------------------------------------------------------------- #
def close_returns(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily simple close-to-close returns (index=date, columns=ticker)."""
    closes = pd.DataFrame({s: panel[s]["Close"] for s in panel}).sort_index()
    return closes.pct_change()


def market_returns(market_close: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Daily simple returns for (TLT, SPY) from the cached close DataFrame."""
    r = market_close.sort_index().pct_change()
    return r["TLT"], r["SPY"]


# --------------------------------------------------------------------------- #
# Conditional flight-to-quality beta (vectorised across names)
# --------------------------------------------------------------------------- #
def ftq_beta_panel(
    returns_df: pd.DataFrame,
    tlt: pd.Series,
    spy: pd.Series,
    window_end: pd.Timestamp,
    lookback_days: int = 252,
    threshold: float = 0.0,
    min_down: int = 40,
) -> pd.Series:
    """Cross-section of flight-to-quality betas on the window *ending at* ``window_end``.

    Restrict the trailing ``lookback_days`` window to days where ``r_SPY < threshold``
    (a risk-off day), then for every name compute the OLS slope of its return on the
    **TLT** return over just those days::

        beta_ftq_i = cov(r_i, r_TLT | down) / var(r_TLT | down)

    Vectorised: with the down-day TLT vector centred as ``bc``, the whole cross-section
    is ``(R_dn_centred.T @ bc) / (bc @ bc)`` — one matrix contraction, no per-name loop.
    Returns a Series indexed by ticker (NaN where the down-day TLT variance is zero or
    there are fewer than ``min_down`` down days).
    """
    win = returns_df.loc[:window_end].tail(lookback_days)
    b = tlt.reindex(win.index).to_numpy(dtype=float)
    s = spy.reindex(win.index).to_numpy(dtype=float)
    dn = np.isfinite(s) & (s < threshold) & np.isfinite(b)
    if int(dn.sum()) < min_down:
        return pd.Series(np.nan, index=win.columns, dtype=float)

    R = win.to_numpy(dtype=float)[dn]          # (Td, N)
    bd = b[dn]                                  # (Td,)
    bc = bd - bd.mean()
    denom = float(bc @ bc)
    if denom <= 0:
        return pd.Series(np.nan, index=win.columns, dtype=float)

    Rc = R - np.nanmean(R, axis=0, keepdims=True)
    # a name with any NaN in the down window would poison the contraction; guard it
    good = np.isfinite(R).all(axis=0)
    betas = np.full(R.shape[1], np.nan, dtype=float)
    if good.any():
        betas[good] = (Rc[:, good].T @ bc) / denom
    return pd.Series(betas, index=win.columns, dtype=float)


def month_ends(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Last available trading day of each month in ``index``."""
    s = pd.Series(index, index=index)
    return list(s.groupby([index.year, index.month]).last())


# --------------------------------------------------------------------------- #
# The cross-sectional sort -> long low-FTQ / short high-FTQ spread
# --------------------------------------------------------------------------- #
def ftq_spreads(
    returns_df: pd.DataFrame,
    tlt: pd.Series,
    spy: pd.Series,
    lookback_days: int = 252,
    threshold: float = 0.0,
    q: float = 0.20,
    min_stocks: int = 20,
    min_down: int = 40,
) -> pd.DataFrame:
    """Monthly long-low-FTQ / short-high-FTQ quintile spread.

    Each month-end ``t-1`` estimate the FTQ-beta cross-section on the trailing window,
    sort, form an equal-weight **low** quintile (bottom ``q``, the long — cheap risky
    names) and **high** quintile (top ``q``, the short — the expensive hedges), and earn
    each name's realised simple return over the *next* calendar month ``t``. Signal at
    end of ``t-1`` → return of ``t``: one execution lag.

    ``spread = lo - hi`` (long low-FTQ, short high-FTQ) — the claim predicts a positive
    "pay-for-the-hedge" spread. Returns a monthly DataFrame with columns
    ``spread, lo, hi, n, n_lo, n_hi``.
    """
    me = month_ends(returns_df.index)
    rows: list[dict] = []
    for k in range(len(me) - 1):
        form_end, hold_start, hold_end = me[k], me[k], me[k + 1]
        sig = ftq_beta_panel(
            returns_df, tlt, spy, window_end=form_end,
            lookback_days=lookback_days, threshold=threshold, min_down=min_down,
        ).dropna()
        if len(sig) < min_stocks:
            continue
        nxt = returns_df.loc[(returns_df.index > hold_start) & (returns_df.index <= hold_end)]
        if nxt.empty:
            continue
        fwd = (1.0 + nxt).prod() - 1.0
        common = sig.index.intersection(fwd.dropna().index)
        if len(common) < min_stocks:
            continue
        sig = sig.loc[common]
        fwd = fwd.loc[common].to_numpy(dtype=float)
        sv = sig.to_numpy(dtype=float)
        lo_thr = np.quantile(sv, q)
        hi_thr = np.quantile(sv, 1.0 - q)
        low_mask = sv <= lo_thr        # low FTQ  -> long
        high_mask = sv >= hi_thr       # high FTQ -> short
        lo = float(fwd[low_mask].mean())
        hi = float(fwd[high_mask].mean())
        rows.append({
            "date": hold_end, "spread": lo - hi, "lo": lo, "hi": hi,
            "n": int(len(common)), "n_lo": int(low_mask.sum()),
            "n_hi": int(high_mask.sum()),
        })
    if not rows:
        return pd.DataFrame(columns=["spread", "lo", "hi", "n", "n_lo", "n_hi"])
    res = pd.DataFrame(rows).set_index("date")
    res.index = pd.DatetimeIndex(res.index)
    return res


# --------------------------------------------------------------------------- #
# Inference primitives  (copied from the desk canon — study 803)
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


def newey_west_t(x: np.ndarray, lags: int = 6) -> float:
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
def ftq_stats(spreads: pd.DataFrame, nw_lags: int = 6) -> dict:
    sp = spreads["spread"].to_numpy(dtype=float)
    return {
        "n_months": int(len(spreads)),
        "spread_pct": float(np.nanmean(sp) * 100),
        "t_nw": newey_west_t(sp, nw_lags),
        "t_1s": one_sample_t(sp),
        "lo_pct": float(np.nanmean(spreads["lo"].to_numpy()) * 100),
        "hi_pct": float(np.nanmean(spreads["hi"].to_numpy()) * 100),
        "welch_t": welch_t(spreads["lo"].to_numpy(), spreads["hi"].to_numpy()),
        "spread_ann_pct": float(np.nanmean(sp) * MONTHS * 100),
    }


# --------------------------------------------------------------------------- #
# Crash protection — do high-FTQ names really lose less on the worst days?
# --------------------------------------------------------------------------- #
def crash_protection(
    returns_df: pd.DataFrame,
    tlt: pd.Series,
    spy: pd.Series,
    lookback_days: int = 252,
    threshold: float = 0.0,
    q: float = 0.20,
    crash_pct: float = 0.05,
    min_stocks: int = 20,
    min_down: int = 40,
) -> dict:
    """Compare the low-FTQ and high-FTQ books' mean return on the worst SPY days.

    Point-in-time: on each day ``t`` (after a full first year) rank the cross-section by
    the FTQ beta known at the previous month-end, form the low/high books, and record
    each book's day-``t`` return. Then restrict to the worst ``crash_pct`` of SPY days
    (the risk-off tail) and compare the two books' mean returns — a positive
    ``hi_minus_lo`` means the high-FTQ (hedge) book *cushioned* the crash.
    """
    me = month_ends(returns_df.index)
    if len(me) < 13:
        return {}
    idx = returns_df.index
    cols = list(returns_df.columns)
    R = returns_df.to_numpy(dtype=float)
    pos = {c: i for i, c in enumerate(cols)}

    # precompute the FTQ book membership formed at each month-end, mapped to columns
    book_lo: dict[pd.Timestamp, np.ndarray] = {}
    book_hi: dict[pd.Timestamp, np.ndarray] = {}
    for k in range(len(me)):
        sig = ftq_beta_panel(returns_df, tlt, spy, window_end=me[k],
                             lookback_days=lookback_days, threshold=threshold,
                             min_down=min_down).dropna()
        if len(sig) < min_stocks:
            continue
        sv = sig.to_numpy(dtype=float)
        lo_thr = np.quantile(sv, q); hi_thr = np.quantile(sv, 1.0 - q)
        book_lo[me[k]] = np.array([pos[c] for c in sig.index[sv <= lo_thr]])
        book_hi[me[k]] = np.array([pos[c] for c in sig.index[sv >= hi_thr]])

    me_sorted = sorted(book_lo)
    if not me_sorted:
        return {}
    me_arr = np.array([t.value for t in me_sorted])

    lo_ret, hi_ret = np.full(len(idx), np.nan), np.full(len(idx), np.nan)
    for i in range(len(idx)):
        # most recent month-end strictly before day i
        j = np.searchsorted(me_arr, idx[i].value, side="left") - 1
        if j < 0:
            continue
        t0 = me_sorted[j]
        lo_ret[i] = np.nanmean(R[i, book_lo[t0]])
        hi_ret[i] = np.nanmean(R[i, book_hi[t0]])

    spy_r = spy.reindex(idx).to_numpy(dtype=float)
    ok = np.isfinite(spy_r) & np.isfinite(lo_ret) & np.isfinite(hi_ret)
    spy_r, lo_ret, hi_ret = spy_r[ok], lo_ret[ok], hi_ret[ok]
    if len(spy_r) < 50:
        return {}
    thr = np.quantile(spy_r, crash_pct)
    crash = spy_r <= thr
    return {
        "n_crash_days": int(crash.sum()),
        "spy_crash_pct": float(spy_r[crash].mean() * 100),
        "lo_book_crash_pct": float(lo_ret[crash].mean() * 100),
        "hi_book_crash_pct": float(hi_ret[crash].mean() * 100),
        "hi_minus_lo_crash_pct": float((hi_ret[crash] - lo_ret[crash]).mean() * 100),
        "crash_welch_t": welch_t(hi_ret[crash], lo_ret[crash]),
        "all_days_hi_minus_lo_pct": float((hi_ret - lo_ret).mean() * 100),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the spread real, or a lucky alignment of the sort?
# --------------------------------------------------------------------------- #
def placebo_pvalue(
    returns_df: pd.DataFrame,
    tlt: pd.Series,
    spy: pd.Series,
    lookback_days: int = 252,
    threshold: float = 0.0,
    q: float = 0.20,
    min_stocks: int = 20,
    min_down: int = 40,
    n_draws: int = 1000,
    base_seed: int = 866,
) -> dict:
    """Keep the FTQ sort but read each month's forward returns from a **name-permuted**
    cross-section (signal→outcome link broken, each month's return distribution
    preserved). p = share of permuted worlds whose spread mean is >= observed
    (right-tail test on the long-low/short-high spread)."""
    obs = float(ftq_spreads(returns_df, tlt, spy, lookback_days, threshold, q,
                            min_stocks, min_down)["spread"].mean())

    me = month_ends(returns_df.index)
    forwards: list[np.ndarray] = []
    lo_idx: list[np.ndarray] = []
    hi_idx: list[np.ndarray] = []
    for k in range(len(me) - 1):
        form_end, hold_start, hold_end = me[k], me[k], me[k + 1]
        sig = ftq_beta_panel(returns_df, tlt, spy, window_end=form_end,
                             lookback_days=lookback_days, threshold=threshold,
                             min_down=min_down).dropna()
        if len(sig) < min_stocks:
            continue
        nxt = returns_df.loc[(returns_df.index > hold_start) & (returns_df.index <= hold_end)]
        if nxt.empty:
            continue
        fwd = (1.0 + nxt).prod() - 1.0
        common = sig.index.intersection(fwd.dropna().index)
        if len(common) < min_stocks:
            continue
        sig = sig.loc[common]
        f = fwd.loc[common].to_numpy(dtype=float)
        sv = sig.to_numpy(dtype=float)
        lo_thr = np.quantile(sv, q); hi_thr = np.quantile(sv, 1.0 - q)
        forwards.append(f)
        lo_idx.append(np.where(sv <= lo_thr)[0])
        hi_idx.append(np.where(sv >= hi_thr)[0])

    rng = np.random.default_rng(base_seed)
    means = np.empty(n_draws)
    for d in range(n_draws):
        tot = 0.0
        for f, li, hi in zip(forwards, lo_idx, hi_idx):
            perm = rng.permutation(len(f))
            fp = f[perm]
            tot += fp[li].mean() - fp[hi].mean()
        means[d] = tot / len(forwards) if forwards else np.nan

    return {
        "obs_pct": obs * 100,
        "placebo_mean_pct": float(means.mean() * 100),
        "placebo_sd_pct": float(means.std(ddof=1) * 100) if n_draws > 1 else float("nan"),
        "p_value": float((means >= obs).mean()),
        "n_draws": int(n_draws),
        "draws_pct": means * 100,
    }


# --------------------------------------------------------------------------- #
# The costed timer
# --------------------------------------------------------------------------- #
def timer_stats(
    spreads: pd.DataFrame,
    one_way_bps: float = 10.0,
    monthly_turnover: float = 1.0,
    borrow_bps_yr: float = 50.0,
) -> dict:
    """Cost the long-low-FTQ / short-high-FTQ book.

    The long-short rebalances both legs monthly. A round-trip on each leg at
    ``one_way_bps`` one-way, scaled by ``monthly_turnover``, costs ``2 legs × 2
    (round-trip) × one_way_bps`` per month; the short leg pays ``borrow_bps_yr`` borrow
    (monthly slice). Net = gross spread − trading drag − borrow.
    """
    sp = spreads["spread"].to_numpy(dtype=float)
    sp = sp[~np.isnan(sp)]
    n = len(sp)
    trade_drag = monthly_turnover * 2.0 * 2.0 * one_way_bps * 1e-4
    borrow_monthly = (borrow_bps_yr / 1e4) / MONTHS
    net = sp - trade_drag - borrow_monthly
    gross_mean = float(sp.mean()) if n else float("nan")
    net_mean = float(net.mean()) if n else float("nan")
    sd = float(net.std(ddof=1)) if n > 1 else float("nan")
    sharpe = net_mean / sd * np.sqrt(MONTHS) if sd and sd > 0 else float("nan")
    return {
        "n_months": n,
        "gross_pct": gross_mean * 100,
        "net_pct": net_mean * 100,
        "cost_pct_per_month": (trade_drag + borrow_monthly) * 100,
        "ann_net_pct": net_mean * MONTHS * 100,
        "sharpe_net": sharpe,
        "t_net": one_sample_t(net),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(world: dict, lookback_days: int = 252, q: float = 0.20,
                     min_stocks: int = 10, min_down: int = 40) -> dict:
    """Run the headline FTQ stats on a synthetic world dict from
    ``data.synthetic_panel`` (keys ``panel``, ``market``, ``tlt``)."""
    ret = close_returns(world["panel"])
    sp = ftq_spreads(ret, world["tlt"], world["market"], lookback_days=lookback_days,
                     q=q, min_stocks=min_stocks, min_down=min_down)
    ts = ftq_stats(sp)
    return {"spread_pct": ts["spread_pct"], "t_nw": ts["t_nw"],
            "welch_t": ts["welch_t"], "n_months": ts["n_months"]}
