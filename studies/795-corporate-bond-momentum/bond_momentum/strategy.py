"""The cross-sectional bond-momentum engine and its honest controls — Study 795.

Jostova, Nikolova, Philipov & Stahel (2013): past 6-month total-return winners among
corporate bonds keep out-performing past losers over the next months, an effect they find
concentrated in *high-yield* names. We test the ETF-panel version: each month-end, rank a
credit + Treasury bond-ETF basket on its trailing K-month total return, go long the top
group and short the bottom group (equal-weight, dollar-neutral), hold one month, rebalance.

This is CROSS-SECTIONAL momentum (rank assets AGAINST EACH OTHER), distinct from:
  * 518-time-series-momentum — each asset traded on the sign of ITS OWN trailing return;
  * 247-bond-seasonality — a calendar (month-of-year) claim on bonds, not a rank sort;
  * 611-mREIT-carry / 612-EM-debt-carry — packaged *carry* (yield you collect), not a
    momentum rank.

This module measures, honestly:

1. **The signal.** The monthly winners-minus-losers portfolio return, GROSS, with a robust
   one-sample *t* (Newey-West HAC), Sharpe, hit-rate + Wilson interval, and max-drawdown.
2. **The null.** A rank-shuffle placebo: randomly permute which asset gets which momentum
   rank each month (destroying the genuine past-return -> future-return link while keeping
   the realised return cross-section intact), recompute the WML mean many times, and report
   the share of placebos that beat the real mean (a permutation p-value).
3. **Costs.** One-way bps x NAV x turnover charged at each monthly rebalance, plus an
   annual borrow on the short leg pro-rated monthly. Reported as gross vs net.
4. **The positive control.** A deterministic synthetic panel with planted cross-sectional
   momentum the engine must recover (and a no-momentum null it must score at zero) — a
   faithful-engine / power check ONLY, never cited for the real-tape stamp.

Execution lag (documented exactly, ONE shift): the rank is computed from prices up to and
including month-end *m*; we do NOT trade on that close. We enter at the close one trading
day later (first session of month *m+1*) and earn the realised return of month *m+1*. A
single forward-only lag — no same-bar fill, no look-ahead.

Survivorship: the basket is broad ETFs still trading in 2026, chosen ex post — named on the
SIGNAL axis (see ``data.py``). A falling ETF is shorted, never deleted, so this is milder
than a single-name credit sort, but any premium here is a mild upper bound.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
TRADING_DAYS = 252

# Formation window: Jostova et al. use a 6-month formation for bond momentum. We default to
# 6 months with no skip (bond momentum, unlike equity momentum, is not dominated by the
# short-term-reversal month, so the standard 12-1 skip is unnecessary; we test skip as a
# robustness cut). Hold one month, rebalance monthly.
LOOKBACK_M = 6
SKIP_M = 0
# Group size for the long / short legs. With ~11 names, the top/bottom third is ~3-4 names
# a side — a real spread with enough diversification not to be one ETF's story.
GROUP_FRAC = 1.0 / 3.0


# ---------------------------------------------------------------------------
# Monthly resampling and the cross-sectional momentum signal
# ---------------------------------------------------------------------------
def to_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end total-return prices (last observation in each calendar month)."""
    return prices.resample("ME").last().dropna(how="all")


def momentum_signal(
    monthly_prices: pd.DataFrame, lookback: int = LOOKBACK_M, skip: int = SKIP_M
) -> pd.DataFrame:
    """Each asset's trailing ``lookback``-month total return, optionally skipping the most
    recent ``skip`` months, at each month-end.

    For month-end *t* and asset *i* the signal is ``P[i,t-skip] / P[i,t-skip-lookback] - 1``
    — built purely from past prices (no look-ahead). Returns a (month x ticker) DataFrame,
    NaN where insufficient history.
    """
    p = monthly_prices
    num = p.shift(skip)
    den = p.shift(skip + lookback)
    return num / den - 1.0


def n_group(n_valid: int, group_frac: float = GROUP_FRAC) -> int:
    """How many names on each leg: ``round(group_frac * n_valid)``, at least 1, and never
    more than half the cross-section (so the long and short legs never overlap)."""
    g = int(round(group_frac * n_valid))
    return max(1, min(g, n_valid // 2))


# ---------------------------------------------------------------------------
# The winners-minus-losers book
# ---------------------------------------------------------------------------
def wml_book(
    monthly_prices: pd.DataFrame,
    lookback: int = LOOKBACK_M,
    skip: int = SKIP_M,
    group_frac: float = GROUP_FRAC,
    cost_bps: float = 0.0,
    borrow_ann_bps: float = 0.0,
    min_names: int = 6,
) -> pd.DataFrame:
    """Monthly cross-sectional winners-minus-losers bond-momentum portfolio, gross and net.

    Construction at each month-end *t*:
      1. Rank valid assets by their trailing ``lookback``-month return (skipping ``skip``).
      2. Long the top ``n_group`` names (equal-weight, total long notional = 1), short the
         bottom ``n_group`` (equal-weight, total short notional = 1) — a dollar-neutral
         winners-minus-losers spread.
      3. The WML return for month *t+1* is (long-leg mean fwd return) - (short-leg mean fwd
         return); single forward execution lag (form on the *t* close, hold month *t+1*).
      4. Costs: one-way turnover x cost_bps x NAV at the rebalance (turnover measured on the
         2-unit gross book), plus an annual borrow on the 1-unit short leg pro-rated monthly.
         Net = gross - costs - borrow.

    Returns a DataFrame indexed by the HOLDING month (*t+1*) with columns:
        ``wml_gross`` long-minus-short forward return
        ``long_ret`` / ``short_ret`` the two legs' forward returns
        ``turnover`` one-way fraction of the 2-unit gross book turned over vs the prior month
        ``wml_net`` wml_gross net of costs and borrow
        ``n_names`` valid names ranked this month · ``n_leg`` names per leg
    """
    fwd = monthly_prices.pct_change().shift(-1)          # month t -> realised return of t+1
    sig = momentum_signal(monthly_prices, lookback, skip)

    rows: list[dict] = []
    prev_w: pd.Series | None = None

    for t in sig.index:
        s = sig.loc[t]
        r = fwd.loc[t] if t in fwd.index else pd.Series(dtype=float)
        valid = s.notna() & r.notna()
        names = list(s.index[valid])
        if len(names) < min_names:
            continue

        ss = s.loc[names].sort_values()
        g = n_group(len(names), group_frac)
        losers = list(ss.index[:g])
        winners = list(ss.index[-g:])

        w = pd.Series(0.0, index=names)
        w.loc[winners] = 1.0 / g
        w.loc[losers] = -1.0 / g

        r_named = r.loc[names].astype(float)
        long_ret = float(r_named.loc[winners].mean())
        short_ret = float(r_named.loc[losers].mean())
        wml_gross = long_ret - short_ret

        if prev_w is not None:
            allnames = w.index.union(prev_w.index)
            wa = w.reindex(allnames).fillna(0.0)
            pa = prev_w.reindex(allnames).fillna(0.0)
            turnover = float((wa - pa).abs().sum() / 2.0)   # one-way, on the 2-unit book
        else:
            turnover = 1.0
        prev_w = w

        cost = turnover * cost_bps * 1e-4
        borrow = 1.0 * borrow_ann_bps * 1e-4 / MONTHS_PER_YEAR   # 1-unit short leg
        wml_net = wml_gross - cost - borrow

        # The holding month is the NEXT month-end after t (label the realised-return period).
        hold = fwd.index[fwd.index.get_loc(t)] if t in fwd.index else t
        rows.append({
            "date": hold, "wml_gross": wml_gross, "long_ret": long_ret,
            "short_ret": short_ret, "turnover": turnover, "wml_net": wml_net,
            "n_names": len(names), "n_leg": g,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date").dropna(subset=["wml_gross"])


# ---------------------------------------------------------------------------
# Inference primitives
# ---------------------------------------------------------------------------
def hac_tstat(r: pd.Series) -> float:
    """Newey-West (HAC, Bartlett kernel, auto lag) one-sample t on a monthly-mean series."""
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


def one_sample_t(r: pd.Series) -> float:
    """Plain (i.i.d.) one-sample t on the mean of a return series."""
    x = pd.Series(r).dropna().to_numpy(dtype=float)
    if x.size < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(x.size)
    return float(x.mean() / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def summary(r: pd.Series, periods_per_year: int = MONTHS_PER_YEAR) -> dict:
    """Annualised statistics for a monthly return series.

    Returns mean(ann), vol(ann), Sharpe, HAC t, plain t, hit-rate + Wilson CI, max-drawdown,
    worst month, n. The HAC *t* is the inference-bar number.
    """
    s = pd.Series(r).astype(float).dropna()
    n = len(s)
    if n < 2:
        return {k: float("nan") for k in
                ("mean", "vol", "sharpe", "tstat", "t_iid", "hit_rate",
                 "hit_lo", "hit_hi", "max_dd", "worst")} | {"n": int(n)}
    mean_ann = float(s.mean() * periods_per_year)
    vol_ann = float(s.std(ddof=1) * np.sqrt(periods_per_year))
    sr = mean_ann / vol_ann if vol_ann > 0 else float("nan")
    eq = (1.0 + s).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    k_hit = int((s > 0).sum())
    lo, hi = wilson_interval(k_hit, n)
    return {
        "mean": mean_ann, "vol": vol_ann, "sharpe": sr,
        "tstat": hac_tstat(s), "t_iid": one_sample_t(s),
        "hit_rate": k_hit / n, "hit_lo": lo, "hit_hi": hi,
        "max_dd": dd, "worst": float(s.min()), "n": int(n),
    }


# ---------------------------------------------------------------------------
# The rank-shuffle placebo / null
# ---------------------------------------------------------------------------
def placebo_pvalue(
    monthly_prices: pd.DataFrame,
    lookback: int = LOOKBACK_M,
    skip: int = SKIP_M,
    group_frac: float = GROUP_FRAC,
    n_perm: int = 2000,
    seed: int = 795,
) -> dict:
    """Permutation p-value: does the real WML mean beat a rank-shuffled null?

    Each month we keep the realised forward-return cross-section exactly, but randomly
    PERMUTE which asset gets which momentum rank — destroying the genuine past->future link
    while preserving the return distribution and the leg sizes. We recompute the WML mean
    ``n_perm`` times and report the one-sided share of placebos whose mean >= the real mean.
    A real edge gives a small p; a data-mined one gives p ~ 0.5.

    Returns ``{"real_mean_ann", "p_value", "placebo_mean_ann", "placebo_sd_ann", "n_perm"}``.
    """
    real = wml_book(monthly_prices, lookback=lookback, skip=skip, group_frac=group_frac)
    if real.empty:
        return {"real_mean_ann": float("nan"), "p_value": float("nan"),
                "placebo_mean_ann": float("nan"), "placebo_sd_ann": float("nan"), "n_perm": 0}
    real_mean = float(real["wml_gross"].mean())

    fwd = monthly_prices.pct_change().shift(-1)
    sig = momentum_signal(monthly_prices, lookback, skip)

    months: list[np.ndarray] = []
    for t in sig.index:
        s = sig.loc[t]
        r = fwd.loc[t] if t in fwd.index else pd.Series(dtype=float)
        valid = s.notna() & r.notna()
        names = list(s.index[valid])
        if len(names) < 6:
            continue
        months.append(r.loc[names].to_numpy(dtype=float))

    if not months:
        return {"real_mean_ann": real_mean * MONTHS_PER_YEAR, "p_value": float("nan"),
                "placebo_mean_ann": float("nan"), "placebo_sd_ann": float("nan"), "n_perm": 0}

    rng = np.random.default_rng(seed)
    placebo = np.empty(n_perm)
    for b in range(n_perm):
        msum = 0.0
        for rv in months:
            k = n_group(rv.size, group_frac)
            perm = rng.permutation(rv.size)   # random rank assignment
            winners = perm[-k:]
            losers = perm[:k]
            msum += float(rv[winners].mean() - rv[losers].mean())
        placebo[b] = msum / len(months)

    p = float((placebo >= real_mean).mean())
    return {
        "real_mean_ann": real_mean * MONTHS_PER_YEAR,
        "p_value": p,
        "placebo_mean_ann": float(placebo.mean() * MONTHS_PER_YEAR),
        "placebo_sd_ann": float(placebo.std(ddof=1) * MONTHS_PER_YEAR),
        "n_perm": int(n_perm),
    }


# ---------------------------------------------------------------------------
# Deterministic synthetic positive control
# ---------------------------------------------------------------------------
def synthetic_control(
    strengths: tuple[float, ...] = (0.0, 0.06, 0.12, 0.20),
    n_assets: int = 11,
    n_days: int = 3000,
    seed: int = 314,
) -> pd.DataFrame:
    """Plant a known cross-sectional momentum and verify the WML engine recovers it.

    Sweeps ``mom_strength`` on the deterministic synthetic panel and reports the portfolio
    mean and HAC *t*. The engine should score ~0 at strength 0 (null) and rise monotonically
    as the planted rank persistence grows. Faithful-engine / power check ONLY — never cited
    to support a real-tape stamp.
    """
    from . import data as _data

    rows: list[dict] = []
    for strength in strengths:
        prices, _ = _data.synthetic_panel(
            n_assets=n_assets, n_days=n_days, mom_strength=strength, seed=seed
        )
        bk = wml_book(to_monthly(prices))
        if bk.empty:
            rows.append({"mom_strength": strength, "mean_ann": float("nan"),
                         "tstat": float("nan"), "n": 0})
            continue
        srow = summary(bk["wml_gross"])
        rows.append({"mom_strength": strength, "mean_ann": srow["mean"],
                     "tstat": srow["tstat"], "n": srow["n"]})
    return pd.DataFrame(rows)


def synthetic_null_robustness(
    n_seeds: int = 20, n_assets: int = 11, n_days: int = 3000, seed0: int = 2000
) -> dict:
    """Average the null (mom_strength=0) HAC *t* across many seeds — seed-robustness guard.

    A single lucky seed can hand a synthetic null a |t|>2 by chance. We sweep ``n_seeds``
    independent panels at the null and report the mean null *t* and the share with |t|>2.
    """
    from . import data as _data

    ts: list[float] = []
    for i in range(n_seeds):
        prices, _ = _data.synthetic_panel(
            n_assets=n_assets, n_days=n_days, mom_strength=0.0, seed=seed0 + i
        )
        bk = wml_book(to_monthly(prices))
        if not bk.empty:
            ts.append(summary(bk["wml_gross"])["tstat"])
    arr = np.array([t for t in ts if np.isfinite(t)])
    if arr.size == 0:
        return {"mean_null_t": float("nan"), "frac_abs_t_gt2": float("nan"), "n_seeds": 0}
    return {"mean_null_t": float(arr.mean()),
            "frac_abs_t_gt2": float((np.abs(arr) > 2).mean()), "n_seeds": int(arr.size)}
