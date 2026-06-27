"""The engine and its honest controls -- Study 512 (High-Volume-Return-Premium).

The claim (Gervais, Kaniel & Mingelgrin 2001, *Journal of Finance*): a stock that trades on
unusually **high** volume over a day or week tends to *appreciate* over the following weeks;
one that trades on unusually **low** volume tends to *depreciate*. GKM call it the
"high-volume return premium" and attribute it to a visibility / attention shock that shifts
the stock's investor base.

This module builds the trade honestly:

1. **The signal.** Each Friday, form every name's *abnormal volume* = this week's average daily
   volume divided by its own trailing 8-week average, minus 1. Rank the cross-section; the top
   quintile is "high volume", the bottom quintile "low volume".

2. **The long-short book.** Long the high-volume quintile, short the low-volume quintile,
   equal-weight, dollar-neutral. Held for ``horizon`` weeks (GKM's short-horizon drift). The
   single execution lag: the signal is known at Friday's close; we enter at the **next**
   week's open-to-close return (no same-bar fill, no look-ahead). Returns one weekly series.

3. **Inference.** One-sample t and a Newey-West (HAC) t on the weekly long-short mean -- the
   inference-bar number -- plus a **label-shuffle placebo**: permute the volume labels across
   names within each week and re-run, building a null distribution for the mean.

4. **Costs.** One-way ``cost_bps`` x NAV x turnover, charged on both legs, plus an annual
   ``borrow_ann_bps`` on the short (low-volume) leg. Gross vs net are both reported.

5. **Synthetic positive control.** ``sweep_synthetic`` plants a known premium and verifies the
   engine recovers a monotone, significant long-short -- a faithful-engine check only, never
   cited for the real-tape stamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEKS_PER_YEAR = 52
TRAIL_WEEKS = 8          # trailing window for the "normal" volume baseline
MIN_NAMES = 10           # need a sane cross-section to sort
QUINTILE = 0.20          # top/bottom 20% by abnormal volume


# ---------------------------------------------------------------------------
# Weekly aggregation + the abnormal-volume signal
# ---------------------------------------------------------------------------
def weekly_frames(
    returns: pd.DataFrame,
    volume: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse daily (returns, volume) into weekly (compounded return, mean daily volume).

    Returns ``(weekly_ret, weekly_vol)`` indexed by week-ending Friday (``W-FRI``):
    ``weekly_ret`` is the compounded simple return over the week, ``weekly_vol`` is the
    average daily share volume that week.
    """
    common = returns.index.intersection(volume.index)
    r = returns.reindex(common)
    v = volume.reindex(common)
    wk_ret = (1.0 + r).resample("W-FRI").prod() - 1.0
    wk_vol = v.resample("W-FRI").mean()
    # A fully-empty resampled week (holidays) -> drop.
    wk_ret = wk_ret.dropna(how="all")
    wk_vol = wk_vol.reindex(wk_ret.index)
    return wk_ret, wk_vol


def abnormal_volume(weekly_vol: pd.DataFrame, trail: int = TRAIL_WEEKS) -> pd.DataFrame:
    """Abnormal volume = (this week's mean daily volume) / (trailing ``trail``-week mean) - 1.

    The trailing mean is **strictly past** (``.shift(1)``) so no look-ahead leaks in. Rows
    before the trailing window is full are NaN.
    """
    trail_mean = weekly_vol.rolling(trail, min_periods=max(3, trail // 2)).mean().shift(1)
    return weekly_vol / trail_mean - 1.0


# ---------------------------------------------------------------------------
# The weekly long-short book
# ---------------------------------------------------------------------------
def long_short(
    returns: pd.DataFrame,
    volume: pd.DataFrame,
    horizon: int = 1,
    trail: int = TRAIL_WEEKS,
    quintile: float = QUINTILE,
    cost_bps: float = 0.0,
    borrow_ann_bps: float = 0.0,
    min_names: int = MIN_NAMES,
    shuffle_seed: int | None = None,
) -> pd.DataFrame:
    """Weekly long-high-volume / short-low-volume book, GKM (2001) style.

    For each week *t* with a full trailing baseline:
      - rank names by ``abnormal_volume``; long the top ``quintile``, short the bottom
        ``quintile`` (equal weight, dollar-neutral);
      - **one execution lag**: hold the forward ``horizon``-week compounded return starting at
        week *t+1* (the signal is public at week *t*'s Friday close; we trade the next week --
        no same-bar fill);
      - net return = long-leg mean - short-leg mean, then charge costs: both legs trade at
        formation so a one-way ``cost_bps`` x NAV is applied to each side (``2*cost_bps``), and
        an annual ``borrow_ann_bps`` on the short leg pro-rated over the holding horizon.

    ``shuffle_seed`` (not None) permutes the abnormal-volume labels *across names within each
    week* -- the label-shuffle placebo null. Returns a DataFrame indexed by formation-week with
    columns ``long_ret, short_ret, ls_gross, ls_net, n_long, n_short``.
    """
    wk_ret, wk_vol = weekly_frames(returns, volume)
    abn = abnormal_volume(wk_vol, trail=trail)
    weeks = abn.index
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None

    rows: list[dict] = []
    for i, wk in enumerate(weeks):
        sig = abn.loc[wk].dropna()
        if len(sig) < min_names:
            continue
        # forward horizon must exist
        fwd_idx = weeks[i + 1 : i + 1 + horizon]
        if len(fwd_idx) < horizon:
            continue

        if rng is not None:
            # Shuffle labels across names: break the (volume -> future return) link.
            vals = sig.to_numpy().copy()
            rng.shuffle(vals)
            sig = pd.Series(vals, index=sig.index)

        n = len(sig)
        k = max(1, int(round(n * quintile)))
        order = sig.sort_values()
        low_names = order.index[:k]            # bottom = low volume -> short
        high_names = order.index[-k:]          # top = high volume   -> long

        # Forward compounded return over the holding horizon, entered next week.
        fwd = (1.0 + wk_ret.loc[fwd_idx]).prod() - 1.0   # per-name compounded
        long_ret = float(fwd.reindex(high_names).dropna().mean())
        short_ret = float(fwd.reindex(low_names).dropna().mean())
        if np.isnan(long_ret) or np.isnan(short_ret):
            continue

        ls_gross = long_ret - short_ret
        # Costs: both legs trade at formation; borrow on the short leg over the horizon.
        cost = 2.0 * cost_bps * 1e-4
        borrow = borrow_ann_bps * 1e-4 * (horizon / WEEKS_PER_YEAR)
        ls_net = ls_gross - cost - borrow

        rows.append(
            {
                "date": wk,
                "long_ret": long_ret,
                "short_ret": short_ret,
                "ls_gross": ls_gross,
                "ls_net": ls_net,
                "n_long": int(len(high_names)),
                "n_short": int(len(low_names)),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _hac_t(x: np.ndarray) -> float:
    """Newey-West (HAC) t-stat on the mean of a return series (desk lag rule)."""
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def summary(weekly: pd.Series, periods_per_year: int = WEEKS_PER_YEAR) -> dict:
    """Annualised stats for a weekly long-short series.

    Returns annualised mean, vol, Sharpe, a **plain one-sample t** (the inference-bar number),
    a Newey-West HAC t, hit-rate, max-drawdown and *n* weeks.
    """
    r = pd.Series(weekly).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: float("nan") for k in
                ("mean", "vol", "sharpe", "t_os", "t_hac", "hit_rate", "max_dd", "n")}
    arr = r.to_numpy()
    mean_p = float(arr.mean())
    vol_p = float(arr.std(ddof=1))
    mean_ann = mean_p * periods_per_year
    vol_ann = vol_p * np.sqrt(periods_per_year)
    sr = mean_ann / vol_ann if vol_ann > 0 else float("nan")
    t_os = float(mean_p / (vol_p / np.sqrt(n))) if vol_p > 0 else float("nan")
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    return {
        "mean": mean_ann,
        "vol": vol_ann,
        "sharpe": sr,
        "t_os": t_os,
        "t_hac": _hac_t(arr),
        "hit_rate": float((r > 0).mean()),
        "max_dd": dd,
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Label-shuffle placebo null
# ---------------------------------------------------------------------------
def _signal_and_forward(
    returns: pd.DataFrame,
    volume: pd.DataFrame,
    horizon: int,
    trail: int,
    min_names: int,
):
    """Precompute, once, the per-week (abnormal-volume row, forward-return row) numpy matrices.

    Returns ``(abn_mat, fwd_mat)`` aligned (week x name) numpy arrays for every formation week
    that has both a full trailing baseline and a complete forward horizon. NaNs mark missing
    names. This is the shared kernel the fast placebo resamples without rebuilding pandas.
    """
    wk_ret, wk_vol = weekly_frames(returns, volume)
    abn = abnormal_volume(wk_vol, trail=trail)
    weeks = abn.index
    growth = (1.0 + wk_ret)
    abn_rows, fwd_rows = [], []
    for i, wk in enumerate(weeks):
        row = abn.loc[wk]
        if row.notna().sum() < min_names:
            continue
        fwd_idx = weeks[i + 1 : i + 1 + horizon]
        if len(fwd_idx) < horizon:
            continue
        fwd = growth.loc[fwd_idx].prod() - 1.0
        abn_rows.append(row.to_numpy(dtype=float))
        fwd_rows.append(fwd.to_numpy(dtype=float))
    if not abn_rows:
        return np.empty((0, 0)), np.empty((0, 0))
    return np.vstack(abn_rows), np.vstack(fwd_rows)


def _long_short_means(abn_mat: np.ndarray, fwd_mat: np.ndarray, quintile: float) -> np.ndarray:
    """Per-week long-short returns from precomputed (abn, fwd) matrices (numpy-only, fast)."""
    out = np.full(abn_mat.shape[0], np.nan)
    for w in range(abn_mat.shape[0]):
        a = abn_mat[w]
        f = fwd_mat[w]
        valid = ~np.isnan(a)
        idx = np.where(valid)[0]
        if idx.size < MIN_NAMES:
            continue
        order = idx[np.argsort(a[idx])]
        k = max(1, int(round(idx.size * quintile)))
        low = order[:k]
        high = order[-k:]
        lr = np.nanmean(f[high])
        sr = np.nanmean(f[low])
        if np.isnan(lr) or np.isnan(sr):
            continue
        out[w] = lr - sr
    return out


def placebo_pvalue(
    returns: pd.DataFrame,
    volume: pd.DataFrame,
    real_mean: float,
    n_shuffles: int = 200,
    horizon: int = 1,
    trail: int = TRAIL_WEEKS,
    quintile: float = QUINTILE,
    min_names: int = MIN_NAMES,
    base_seed: int = 512,
) -> dict:
    """Two-sided placebo p-value: how often a *label-shuffled* book beats the real mean?

    For each of ``n_shuffles`` draws, permute the volume labels across names within each week
    and recompute the mean weekly long-short. The p-value is the share of shuffles whose
    |mean| >= |real mean| -- a faithful null for "did volume actually carry the sort, or could
    random labels have done as well?". The (abnormal-volume, forward-return) matrices are
    precomputed once and only the per-week label order is permuted, so this is fast and exact.
    Returns the p-value, the null mean and null std (both weekly).
    """
    abn_mat, fwd_mat = _signal_and_forward(returns, volume, horizon, trail, min_names)
    if abn_mat.size == 0:
        return {"p": float("nan"), "null_mean": float("nan"), "null_std": float("nan"),
                "n_shuffles": 0}
    real_abs = abs(real_mean)
    rng = np.random.default_rng(base_seed)
    null_means = np.empty(n_shuffles)
    nrow = abn_mat.shape[0]
    for j in range(n_shuffles):
        shuffled = abn_mat.copy()
        for w in range(nrow):
            valid = ~np.isnan(shuffled[w])
            vals = shuffled[w, valid]
            rng.shuffle(vals)
            shuffled[w, valid] = vals
        ls = _long_short_means(shuffled, fwd_mat, quintile)
        null_means[j] = np.nanmean(ls)
    p = float((np.abs(null_means) >= real_abs).mean())
    return {
        "p": p,
        "null_mean": float(null_means.mean()),
        "null_std": float(null_means.std(ddof=1)) if null_means.size > 1 else float("nan"),
        "n_shuffles": int(n_shuffles),
    }


# ---------------------------------------------------------------------------
# Turnover (for the cost charge sanity-check)
# ---------------------------------------------------------------------------
def avg_turnover(
    returns: pd.DataFrame,
    volume: pd.DataFrame,
    horizon: int = 1,
    trail: int = TRAIL_WEEKS,
    quintile: float = QUINTILE,
    min_names: int = MIN_NAMES,
) -> float:
    """Average one-sided name turnover of the long leg between consecutive formation weeks.

    A weekly-rebalanced quintile book turns over a large fraction of its names; this reports the
    mean Jaccard-complement of the long basket week-over-week, for the cost narrative.
    """
    wk_ret, wk_vol = weekly_frames(returns, volume)
    abn = abnormal_volume(wk_vol, trail=trail)
    weeks = abn.index
    prev_long: set | None = None
    turns: list[float] = []
    for wk in weeks:
        sig = abn.loc[wk].dropna()
        if len(sig) < min_names:
            prev_long = None
            continue
        k = max(1, int(round(len(sig) * quintile)))
        high = set(sig.sort_values().index[-k:])
        if prev_long is not None and prev_long:
            changed = len(high - prev_long) / max(1, len(high))
            turns.append(changed)
        prev_long = high
    return float(np.mean(turns)) if turns else float("nan")


# ---------------------------------------------------------------------------
# Synthetic positive control (faithful-engine check ONLY)
# ---------------------------------------------------------------------------
def sweep_synthetic(
    premiums=(-0.06, -0.03, 0.0, 0.03, 0.06, 0.10),
    n_stocks: int = 40,
    n_days: int = 2600,
    horizon: int = 1,
    seed: int = 512,
) -> pd.DataFrame:
    """Plant each premium in a synthetic panel and report the recovered long-short mean & t.

    Imported lazily to avoid a circular import. This is a power/faithfulness check only --
    never cited for the real-tape verdict.
    """
    from . import data as _data

    rows = []
    for prem in premiums:
        ret, vol, _ = _data.synthetic_panel(
            n_stocks=n_stocks, n_days=n_days, hv_premium=prem, seed=seed
        )
        df = long_short(ret, vol, horizon=horizon)
        if df.empty:
            rows.append({"premium": prem, "ls_mean_ann_%": float("nan"), "t_os": float("nan")})
            continue
        s = summary(df["ls_gross"])
        rows.append({"premium": prem, "ls_mean_ann_%": s["mean"] * 100, "t_os": s["t_os"]})
    return pd.DataFrame(rows)
