"""The Industry-Momentum engine and its honest controls -- Study 506.

Moskowitz & Grinblatt (1999): cross-sectional momentum is largely an *industry* effect.
This module builds the sector-ETF embodiment and the single-name comparison, honestly:

1. **The 12-1 signal.** Each month *t*, rank every asset by its trailing 12-1 month return
   -- the cumulative return over months t-12..t-1, **skipping the most recent month** (the
   standard Jegadeesh-Titman / Moskowitz-Grinblatt convention that dodges short-term reversal).

2. **The long-short book.** Long the top ``top_k`` winners, short the bottom ``top_k`` losers,
   equal-weighted, dollar-neutral. Held over month t+1 -- ONE documented execution lag: the
   signal is public at the close of t, we hold the *realised* next-month (t+1) return. No
   look-ahead, no same-bar fill.

3. **Costs.** One-way ``cost_bps`` x NAV x turnover each rebalance (both legs trade), plus a
   ``borrow_ann_bps`` annual short-borrow charge on the short leg (pro-rated monthly). Gross
   and net are both reported.

4. **Inference.** A one-sample Newey-West (HAC) *t* on the monthly long-short series is the
   bar number. A label-shuffle **placebo** null (permute the asset labels before ranking,
   destroying any real cross-sectional signal while preserving the marginal distributions)
   gives an empirical p-value: how often does pure noise beat the real book's mean?

5. **The race.** The same machine runs on the 11 SPDR sectors (industry momentum) and on a
   large-cap survivor basket (single-name momentum). Moskowitz-Grinblatt predict the industry
   expression carries the punch.

The synthetic positive control (``data.synthetic_world``) plants a persistent industry
component and proves the engine recovers a positive long-short when -- and only when -- one is
planted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12

# 12-1 momentum convention: look back 12 months, skip the most recent 1.
LOOKBACK = 12
SKIP = 1


# ---------------------------------------------------------------------------
# The 12-1 momentum signal + long-short book
# ---------------------------------------------------------------------------
def momentum_signal(monthly_ret: pd.DataFrame, lookback: int = LOOKBACK, skip: int = SKIP) -> pd.DataFrame:
    """Trailing 12-1 cumulative return for each asset, as of each month.

    The value at row *t* is the cumulative simple return over months [t-lookback, t-skip]
    (inclusive of t-lookback, exclusive of the last ``skip`` months). It is therefore known at
    the close of month *t* and is used to rank for the *t+1* holding period.

    Returns a (date x asset) DataFrame of momentum scores; rows before enough history are NaN.
    """
    logret = np.log1p(monthly_ret)
    # Cumulative log return over the trailing window ending `skip` months ago.
    # rolling sum of the last `lookback` months, then subtract the last `skip` months.
    full = logret.rolling(lookback, min_periods=lookback).sum()
    recent = logret.rolling(skip, min_periods=skip).sum() if skip > 0 else 0.0
    score = full - recent
    return np.expm1(score)


def long_short(
    monthly_ret: pd.DataFrame,
    top_k: int = 3,
    lookback: int = LOOKBACK,
    skip: int = SKIP,
    cost_bps: float = 0.0,
    borrow_ann_bps: float = 0.0,
    shuffle_seed: int | None = None,
) -> pd.DataFrame:
    """Monthly long-(top_k winners) / short-(bottom_k losers) 12-1 momentum book.

    For each formation month *t* with a complete momentum score, rank assets, go long the
    ``top_k`` highest and short the ``top_k`` lowest (equal-weight, dollar-neutral), and earn
    the *realised* return of month *t+1* (one execution lag). Costs: one-way ``cost_bps`` per
    leg per side that trades, charged on turnover vs the previous month's holdings; short borrow
    is ``borrow_ann_bps``/12 on the short leg each month it is held.

    ``shuffle_seed`` (the PLACEBO): if set, the asset labels on the score row are permuted with
    that RNG before ranking -- the ranks become meaningless, so any positive mean is luck. Used
    to build the label-shuffle null.

    Returns a DataFrame indexed by the *holding* month with columns:
        ``long_ret``  -- equal-weight realised return of the winner leg
        ``short_ret`` -- equal-weight realised return of the loser leg
        ``ls_gross``  -- long_ret - short_ret (gross, dollar-neutral)
        ``ls_net``    -- ls_gross minus costs and borrow
        ``turnover``  -- one-sided name turnover at this rebalance (0..1 per leg)
    """
    score = momentum_signal(monthly_ret, lookback=lookback, skip=skip)
    dates = monthly_ret.index
    rows: list[dict] = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None

    for i in range(len(dates) - 1):
        t = dates[i]
        t1 = dates[i + 1]
        s = score.loc[t].dropna()
        if len(s) < 2 * top_k:
            prev_long, prev_short = set(), set()
            continue

        if rng is not None:
            # Label shuffle: permute which asset each score belongs to.
            perm = rng.permutation(s.index.to_numpy())
            s = pd.Series(s.to_numpy(), index=perm)

        ranked = s.sort_values(ascending=False)
        winners = list(ranked.index[:top_k])
        losers = list(ranked.index[-top_k:])

        fwd = monthly_ret.loc[t1]
        long_ret = float(fwd.reindex(winners).dropna().mean())
        short_ret = float(fwd.reindex(losers).dropna().mean())
        if np.isnan(long_ret) or np.isnan(short_ret):
            continue
        ls_gross = long_ret - short_ret

        # Turnover: fraction of names that changed on each leg (one-sided).
        new_long, new_short = set(winners), set(losers)
        if prev_long or prev_short:
            to_long = len(new_long - prev_long) / max(len(new_long), 1)
            to_short = len(new_short - prev_short) / max(len(new_short), 1)
            turnover = 0.5 * (to_long + to_short)
        else:
            turnover = 1.0
        prev_long, prev_short = new_long, new_short

        # Costs: both legs trade `turnover` of their names; one-way cost per name traded.
        # Charge cost on both legs (long+short) -> 2 * turnover * cost_bps.
        cost = 2.0 * turnover * cost_bps * 1e-4
        borrow = borrow_ann_bps * 1e-4 / MONTHS_PER_YEAR
        ls_net = ls_gross - cost - borrow

        rows.append({
            "date": t1,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "ls_gross": ls_gross,
            "ls_net": ls_net,
            "turnover": turnover,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def hac_tstat(r: pd.Series) -> float:
    """Newey-West (HAC) t-stat on the mean of a monthly return series.

    Lag rule floor(4*(n/100)^(2/9)) -- the desk's standard.
    """
    x = pd.Series(r).dropna().to_numpy(dtype=float)
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


def summary(r: pd.Series, periods_per_year: int = MONTHS_PER_YEAR) -> dict:
    """Annualised stats for a monthly return series: mean, vol, Sharpe, HAC t, hit, maxDD, n."""
    r = pd.Series(r).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: float("nan") for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n")}
    mean_ann = float(r.mean()) * periods_per_year
    vol_ann = float(r.std(ddof=1)) * np.sqrt(periods_per_year)
    sr = mean_ann / vol_ann if vol_ann > 0 else float("nan")
    nav = (1.0 + r).cumprod()
    dd = float((nav / nav.cummax() - 1.0).min())
    return {
        "mean": mean_ann,
        "vol": vol_ann,
        "sharpe": sr,
        "tstat": hac_tstat(r),
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": dd,
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(
    monthly_ret: pd.DataFrame,
    real_mean: float,
    top_k: int = 3,
    n_shuffles: int = 500,
    base_seed: int = 506,
    **ls_kwargs,
) -> dict:
    """Empirical p-value from a label-shuffle null.

    Re-runs the long-short ``n_shuffles`` times with the asset labels permuted before each
    ranking (``shuffle_seed`` varied). Each shuffle destroys the real cross-sectional signal
    while preserving the marginal return distributions. The p-value is the fraction of shuffles
    whose mean long-short return >= the real book's mean.

    Returns ``{p_value, null_mean, null_std, n_shuffles}``.

    Fast path: the 12-1 score matrix is rolling-expensive, so we compute it ONCE and, for each
    shuffle, only permute the (score, next-month-return) pairing per formation month and take the
    top_k/bottom_k means. This is the same gross long-short statistic the slow loop produces (the
    label permutation breaks any real cross-sectional alignment) but ~100x cheaper -- so the
    placebo runs in seconds, not minutes. ``ls_kwargs`` (costs/borrow) do not affect the GROSS
    placebo statistic and are ignored here by design.
    """
    lookback = ls_kwargs.get("lookback", LOOKBACK)
    skip = ls_kwargs.get("skip", SKIP)
    score = momentum_signal(monthly_ret, lookback=lookback, skip=skip)
    dates = monthly_ret.index

    # Pre-extract, per formation month with a complete score, the score vector and the realised
    # t+1 return vector over the SAME assets present in both. The placebo permutes which score
    # is glued to which forward return, then ranks.
    months: list[tuple[np.ndarray, np.ndarray, int]] = []
    for i in range(len(dates) - 1):
        s = score.loc[dates[i]].dropna()
        if len(s) < 2 * top_k:
            continue
        fwd = monthly_ret.loc[dates[i + 1]].reindex(s.index)
        ok = fwd.notna().to_numpy()
        sv = s.to_numpy()[ok]
        rv = fwd.to_numpy()[ok]
        if len(sv) < 2 * top_k:
            continue
        months.append((sv, rv, top_k))

    if not months:
        return {"p_value": float("nan"), "null_mean": float("nan"),
                "null_std": float("nan"), "n_shuffles": 0}

    rng = np.random.default_rng(base_seed)
    null_means = []
    for _ in range(n_shuffles):
        ls_sum = 0.0
        for sv, rv, k in months:
            # Permute the label->score map: glue each forward return to a random score.
            perm = rng.permutation(len(sv))
            order = np.argsort(sv[perm])  # ascending by shuffled score
            ls_sum += rv[order[-k:]].mean() - rv[order[:k]].mean()
        null_means.append(ls_sum / len(months))

    if not null_means:
        return {"p_value": float("nan"), "null_mean": float("nan"),
                "null_std": float("nan"), "n_shuffles": 0}
    arr = np.array(null_means)
    p = float((arr >= real_mean).mean())
    return {
        "p_value": p,
        "null_mean": float(arr.mean() * MONTHS_PER_YEAR),
        "null_std": float(arr.std(ddof=1) * MONTHS_PER_YEAR),
        "n_shuffles": int(len(arr)),
    }


# ---------------------------------------------------------------------------
# Synthetic positive control (seed-robust)
# ---------------------------------------------------------------------------
def control_sweep(data_module, moms, n_years: int = 18, top_k: int = 3, n_seeds: int = 20) -> pd.DataFrame:
    """Faithful-engine power check: plant a range of industry-momentum strengths and recover them.

    For each planted ``industry_mom`` we average the long-short HAC t and annualised mean over
    ``n_seeds`` RNG seeds (a Welch-style multi-seed average -- never a single lucky seed). This
    is ONLY a power/faithfulness check; it never supports a real-tape stamp.

    Returns a DataFrame: industry_mom, mean_ann, tstat (both seed-averaged).
    """
    rows = []
    for mom in moms:
        means, ts = [], []
        for s in range(n_seeds):
            sec, _mkt, _truth = data_module.synthetic_world(
                n_years=n_years, industry_mom=mom, seed=506 + s,
            )
            res = long_short(sec, top_k=top_k)
            if res.empty:
                continue
            stt = summary(res["ls_gross"])
            means.append(stt["mean"])
            ts.append(stt["tstat"])
        rows.append({
            "industry_mom": mom,
            "mean_ann": float(np.nanmean(means)) if means else float("nan"),
            "tstat": float(np.nanmean(ts)) if ts else float("nan"),
        })
    return pd.DataFrame(rows)
