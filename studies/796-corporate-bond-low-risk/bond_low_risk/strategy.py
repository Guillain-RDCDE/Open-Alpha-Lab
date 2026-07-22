"""The bond low-risk (Betting-Against-Beta) engine and its honest controls — Study 796.

Frazzini & Pedersen (2014), *Betting Against Beta*: because leverage-constrained investors
bid up risky assets, the lowest-risk assets earn the highest *risk-adjusted* returns. A
"BAB" portfolio that **levers up the safe leg** and **de-levers the risky leg** to equal
ex-ante risk earns a positive premium. We test the bond version on a credit + Treasury
bond-ETF basket: each month-end, rank the universe on trailing *volatility*, then build a
vol-scaled **low-minus-high** spread — long the low-vol names (levered up to a common risk
target) and short the high-vol names (levered down) — and ask whether the safe leg really
does deliver more return per unit of risk.

This is the LOW-RISK / volatility-anomaly leg, distinct from:
  * 238-betting-against-beta — BAB in **equities** (single stocks, market-beta sort);
  * 330-low-volatility-anomaly — the low-vol anomaly in **equities**;
  * 795-corporate-bond-momentum — a trailing-**return rank** (momentum) on the SAME bond
    universe, an orthogonal signal (change, not risk-level).

This module measures, honestly:

1. **The signal.** The monthly vol-scaled low-minus-high (BAB) spread return, GROSS, with a
   robust one-sample *t* (Newey-West HAC), Sharpe, hit-rate + Wilson interval, max-drawdown;
   plus the descriptive Sharpe of the low-vol vs high-vol basket the claim is really about.
2. **The null.** A vol-rank-shuffle placebo: randomly permute which asset carries which
   trailing volatility each month (destroying the genuine low-vol -> high-risk-adjusted-
   return link while keeping the realised return cross-section and the vol distribution
   intact), recompute the BAB mean many times, and report the share of placebos that beat
   the real mean (a permutation p-value).
3. **Costs.** One-way bps x NAV x turnover at each monthly rebalance, plus a single annual
   financing/borrow rate charged once on the borrowed notional (the levered long portion
   above 1x plus the short leg). Reported as gross vs net.
4. **The positive control.** A deterministic synthetic panel with a planted low-risk tilt
   the engine must recover (and a no-anomaly null it must score at zero) — a faithful-engine
   / power check ONLY, never cited for the real-tape stamp.

Execution lag (documented exactly, ONE shift): the trailing volatility and the leg scaling
are computed from daily prices up to and including month-end *t*; we do NOT trade on that
close. We enter at the close one trading day later (first session of month *m+1*) and earn
the realised return of month *m+1*. A single forward-only lag — no same-bar fill, no
look-ahead. The vol used to *scale* each leg is the ex-ante (trailing) vol, known at *t*.

Survivorship: the basket is broad ETFs still trading in 2026, chosen ex post — named on the
SIGNAL axis (see ``data.py``). A survivor set that dropped blown-up high-vol vehicles would
*understate* the risky leg's tail and thus *understate* the low-risk premium, so a null here
is not manufactured by the bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
TRADING_DAYS = 252

# Risk proxy: trailing ~1-year (252 trading-day) daily-return volatility, annualised, at
# each month-end (Frazzini-Pedersen estimate risk over ~1y of daily data). Robustness cuts
# use 63d (~3m) and 126d (~6m).
VOL_WINDOW = 252
MIN_WINDOW = 120          # need at least this many daily obs to estimate a vol
# Group size for the low / high legs (top/bottom third of ~11 names ~ 3-4 a side).
GROUP_FRAC = 1.0 / 3.0
# Common ex-ante risk target both legs are scaled to (annualised). The scaling is what makes
# this a *risk-adjusted* (BAB) comparison rather than a raw return spread.
TARGET_VOL = 0.06
MAX_LEVERAGE = 8.0        # cap on either leg's scale factor (short-duration vol is tiny)


# ---------------------------------------------------------------------------
# Monthly resampling and the trailing-volatility signal
# ---------------------------------------------------------------------------
def to_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end total-return prices (last observation in each calendar month)."""
    return prices.resample("ME").last().dropna(how="all")


def trailing_vol_at(
    daily_ret: pd.DataFrame, t: pd.Timestamp, window: int = VOL_WINDOW,
    min_window: int | None = None,
) -> pd.Series:
    """Annualised trailing daily-return volatility per asset as of month-end ``t``.

    Uses only returns up to and including ``t`` (no look-ahead). Names with fewer than
    ``min_window`` valid observations in the window get NaN. When ``min_window`` is None it
    scales with the window (``min(MIN_WINDOW, 0.75*window)``) so short windows still qualify.
    """
    if min_window is None:
        min_window = min(MIN_WINDOW, max(20, int(0.75 * window)))
    win = daily_ret.loc[:t].tail(window)
    n_obs = win.notna().sum()
    vol = win.std(ddof=1) * np.sqrt(TRADING_DAYS)
    vol[n_obs < min_window] = np.nan
    return vol


def n_group(n_valid: int, group_frac: float = GROUP_FRAC) -> int:
    """How many names on each leg: ``round(group_frac * n_valid)``, at least 1, and never
    more than half the cross-section (so the low and high legs never overlap)."""
    g = int(round(group_frac * n_valid))
    return max(1, min(g, n_valid // 2))


# ---------------------------------------------------------------------------
# The vol-scaled low-minus-high (BAB) book
# ---------------------------------------------------------------------------
def bab_book(
    prices: pd.DataFrame,
    vol_window: int = VOL_WINDOW,
    group_frac: float = GROUP_FRAC,
    target_vol: float = TARGET_VOL,
    max_leverage: float = MAX_LEVERAGE,
    cost_bps: float = 0.0,
    fin_ann_bps: float = 0.0,
    min_names: int = 6,
) -> pd.DataFrame:
    """Monthly vol-scaled low-minus-high bond low-risk (BAB) portfolio, gross and net.

    Construction at each month-end *t* (from DAILY total-return prices):
      1. Estimate each asset's trailing ``vol_window``-day annualised volatility.
      2. Rank valid assets by that volatility. Low leg = the ``n_group`` lowest-vol names,
         high leg = the ``n_group`` highest-vol names (equal-weight within each leg).
      3. Scale each leg to a common ex-ante ``target_vol``: the leg's own trailing basket
         volatility gives a scale ``k = target_vol / leg_vol`` (capped at ``max_leverage``).
         The low-vol leg is levered UP (k>1), the high-vol leg levered DOWN (k<1) — the BAB
         signature. The spread return for month *t+1* is
         ``k_low * low_leg_fwd_ret - k_high * high_leg_fwd_ret`` (single forward lag).
      4. Costs: one-way turnover x cost_bps x NAV at the rebalance (turnover on the signed
         per-name weight book), plus a single annual financing/borrow on the borrowed
         notional — the levered-long portion above 1x plus the short leg — pro-rated monthly.
         Net = gross - costs - carry.

    Returns a DataFrame indexed by the HOLDING month (*t+1*) with columns:
        ``bab_gross`` vol-scaled low-minus-high forward return
        ``low_ret`` / ``high_ret`` the two legs' *unlevered* equal-weight forward returns
        ``k_low`` / ``k_high`` the leg scale factors (ex-ante risk parity)
        ``turnover`` one-way fraction of the gross book turned over vs the prior month
        ``bab_net`` bab_gross net of costs and carry
        ``n_names`` valid names ranked this month · ``n_leg`` names per leg
    """
    daily_ret = prices.pct_change()
    monthly_prices = to_monthly(prices)
    fwd = monthly_prices.pct_change().shift(-1)          # month t -> realised return of t+1

    rows: list[dict] = []
    prev_w: pd.Series | None = None

    for t in monthly_prices.index:
        vol = trailing_vol_at(daily_ret, t, window=vol_window)
        r = fwd.loc[t] if t in fwd.index else pd.Series(dtype=float)
        valid = vol.notna() & r.notna() & (vol > 0)
        names = list(vol.index[valid])
        if len(names) < min_names:
            continue

        vv = vol.loc[names].sort_values()
        g = n_group(len(names), group_frac)
        low = list(vv.index[:g])           # lowest vol
        high = list(vv.index[-g:])         # highest vol

        # Ex-ante leg vols from the trailing daily returns of an equal-weight leg basket.
        win = daily_ret.loc[:t].tail(vol_window)
        low_vol = float(win[low].mean(axis=1).std(ddof=1) * np.sqrt(TRADING_DAYS))
        high_vol = float(win[high].mean(axis=1).std(ddof=1) * np.sqrt(TRADING_DAYS))
        if not (low_vol > 0 and high_vol > 0):
            continue
        k_low = min(target_vol / low_vol, max_leverage)
        k_high = min(target_vol / high_vol, max_leverage)

        r_named = r.loc[names].astype(float)
        low_ret = float(r_named.loc[low].mean())
        high_ret = float(r_named.loc[high].mean())
        bab_gross = k_low * low_ret - k_high * high_ret

        # Signed per-name weights (for turnover): long low leg, short high leg.
        w = pd.Series(0.0, index=names)
        w.loc[low] = k_low / g
        w.loc[high] = -k_high / g
        if prev_w is not None:
            allnames = w.index.union(prev_w.index)
            wa = w.reindex(allnames).fillna(0.0)
            pa = prev_w.reindex(allnames).fillna(0.0)
            turnover = float((wa - pa).abs().sum() / 2.0)   # one-way, on the gross book
        else:
            turnover = float((k_low + k_high) / 2.0)
        prev_w = w

        cost = turnover * cost_bps * 1e-4
        borrowed = max(k_low - 1.0, 0.0) + k_high          # levered long + short notional
        carry = borrowed * fin_ann_bps * 1e-4 / MONTHS_PER_YEAR
        bab_net = bab_gross - cost - carry

        hold = fwd.index[fwd.index.get_loc(t)] if t in fwd.index else t
        rows.append({
            "date": hold, "bab_gross": bab_gross, "low_ret": low_ret, "high_ret": high_ret,
            "k_low": k_low, "k_high": k_high, "turnover": turnover, "bab_net": bab_net,
            "n_names": len(names), "n_leg": g,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date").dropna(subset=["bab_gross"])


def leg_returns(
    prices: pd.DataFrame, vol_window: int = VOL_WINDOW, group_frac: float = GROUP_FRAC,
    min_names: int = 6,
) -> pd.DataFrame:
    """The *unlevered* equal-weight low-vol and high-vol basket monthly returns.

    This is the raw material for the claim's plain-language form: "does the low-vol basket
    earn a higher **Sharpe** than the high-vol basket?" (no leverage, no spread). Columns
    ``low`` / ``high``, indexed by holding month.
    """
    bk = bab_book(prices, vol_window=vol_window, group_frac=group_frac, min_names=min_names)
    if bk.empty:
        return pd.DataFrame()
    return bk[["low_ret", "high_ret"]].rename(columns={"low_ret": "low", "high_ret": "high"})


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


def break_even_bps(bk: pd.DataFrame) -> float:
    """One-way cost (bps) that zeroes the gross BAB mean given the realised turnover.

    ``mean(bab_gross) = mean(turnover) * be * 1e-4``  ->  be = gross_mean / turnover / 1e-4.
    Financing is not included here (it is charged separately in the net timer); this is the
    turnover break-even the notebooks quote alongside the cost sweep.
    """
    if bk.empty:
        return float("nan")
    g = float(bk["bab_gross"].mean())
    tv = float(bk["turnover"].mean())
    if tv <= 0:
        return float("nan")
    return g / tv / 1e-4


# ---------------------------------------------------------------------------
# The vol-rank-shuffle placebo / null
# ---------------------------------------------------------------------------
def _spread_from(fwd: np.ndarray, vol: np.ndarray, group_frac: float,
                 target_vol: float, max_leverage: float) -> float:
    """One month's vol-scaled low-minus-high spread from realised returns ``fwd`` and
    per-asset trailing vols ``vol`` (both aligned, finite). Legs are scaled by the mean
    per-asset vol of the leg (a fast proxy for the leg-basket vol used in ``bab_book``)."""
    order = np.argsort(vol)                     # ascending vol
    g = n_group(vol.size, group_frac)
    low = order[:g]
    high = order[-g:]
    low_vol = float(vol[low].mean())
    high_vol = float(vol[high].mean())
    if not (low_vol > 0 and high_vol > 0):
        return np.nan
    k_low = min(target_vol / low_vol, max_leverage)
    k_high = min(target_vol / high_vol, max_leverage)
    return k_low * float(fwd[low].mean()) - k_high * float(fwd[high].mean())


def placebo_pvalue(
    prices: pd.DataFrame,
    vol_window: int = VOL_WINDOW,
    group_frac: float = GROUP_FRAC,
    target_vol: float = TARGET_VOL,
    max_leverage: float = MAX_LEVERAGE,
    n_perm: int = 2000,
    seed: int = 796,
) -> dict:
    """Permutation p-value: does the real BAB mean beat a vol-rank-shuffled null?

    Each month we keep the realised forward-return cross-section and the vector of trailing
    volatilities exactly, but randomly PERMUTE which asset carries which volatility —
    destroying the genuine low-vol -> high-risk-adjusted-return link while preserving both
    marginal distributions and the leg sizes. We recompute the vol-scaled spread mean
    ``n_perm`` times and report the one-sided share of placebos whose mean >= the real mean.
    A real low-risk edge gives a small p; a data-mined one gives p ~ 0.5.

    Returns ``{"real_mean_ann", "p_value", "placebo_mean_ann", "placebo_sd_ann", "n_perm"}``.
    """
    daily_ret = prices.pct_change()
    monthly_prices = to_monthly(prices)
    fwd_all = monthly_prices.pct_change().shift(-1)

    months: list[tuple[np.ndarray, np.ndarray]] = []
    for t in monthly_prices.index:
        vol = trailing_vol_at(daily_ret, t, window=vol_window)
        r = fwd_all.loc[t] if t in fwd_all.index else pd.Series(dtype=float)
        valid = vol.notna() & r.notna() & (vol > 0)
        names = list(vol.index[valid])
        if len(names) < 6:
            continue
        months.append((r.loc[names].to_numpy(dtype=float),
                       vol.loc[names].to_numpy(dtype=float)))

    if not months:
        return {"real_mean_ann": float("nan"), "p_value": float("nan"),
                "placebo_mean_ann": float("nan"), "placebo_sd_ann": float("nan"), "n_perm": 0}

    real_vals = [_spread_from(f, v, group_frac, target_vol, max_leverage) for f, v in months]
    real_vals = [x for x in real_vals if np.isfinite(x)]
    real_mean = float(np.mean(real_vals))

    rng = np.random.default_rng(seed)
    placebo = np.empty(n_perm)
    for b in range(n_perm):
        msum = 0.0
        cnt = 0
        for f, v in months:
            vs = v[rng.permutation(v.size)]       # shuffle vols across names
            x = _spread_from(f, vs, group_frac, target_vol, max_leverage)
            if np.isfinite(x):
                msum += x
                cnt += 1
        placebo[b] = msum / cnt if cnt else np.nan

    p = float((placebo >= real_mean).mean())
    return {
        "real_mean_ann": real_mean * MONTHS_PER_YEAR,
        "p_value": p,
        "placebo_mean_ann": float(np.nanmean(placebo) * MONTHS_PER_YEAR),
        "placebo_sd_ann": float(np.nanstd(placebo, ddof=1) * MONTHS_PER_YEAR),
        "n_perm": int(n_perm),
    }


# ---------------------------------------------------------------------------
# Deterministic synthetic positive control
# ---------------------------------------------------------------------------
def synthetic_control(
    strengths: tuple[float, ...] = (0.0, 0.30, 0.60, 1.00),
    n_assets: int = 11,
    n_days: int = 3000,
    seed: int = 314,
) -> pd.DataFrame:
    """Plant a known low-risk (BAB) tilt and verify the engine recovers it.

    Sweeps ``lowrisk_strength`` on the deterministic synthetic panel and reports the BAB
    mean and HAC *t*. The engine should score ~0 at strength 0 (all assets on one Sharpe
    line) and rise monotonically as the low-vol names are given the higher Sharpe.
    Faithful-engine / power check ONLY — never cited to support a real-tape stamp.
    """
    from . import data as _data

    rows: list[dict] = []
    for strength in strengths:
        prices, _ = _data.synthetic_panel(
            n_assets=n_assets, n_days=n_days, lowrisk_strength=strength, seed=seed
        )
        bk = bab_book(prices)
        if bk.empty:
            rows.append({"lowrisk_strength": strength, "mean_ann": float("nan"),
                         "tstat": float("nan"), "n": 0})
            continue
        srow = summary(bk["bab_gross"])
        rows.append({"lowrisk_strength": strength, "mean_ann": srow["mean"],
                     "tstat": srow["tstat"], "n": srow["n"]})
    return pd.DataFrame(rows)


def synthetic_null_robustness(
    n_seeds: int = 20, n_assets: int = 11, n_days: int = 3000, seed0: int = 2000
) -> dict:
    """Average the null (lowrisk_strength=0) HAC *t* across many seeds — seed-robustness guard.

    A single lucky seed can hand a synthetic null a |t|>2 by chance. We sweep ``n_seeds``
    independent panels at the null and report the mean null *t* and the share with |t|>2.
    """
    from . import data as _data

    ts: list[float] = []
    for i in range(n_seeds):
        prices, _ = _data.synthetic_panel(
            n_assets=n_assets, n_days=n_days, lowrisk_strength=0.0, seed=seed0 + i
        )
        bk = bab_book(prices)
        if not bk.empty:
            ts.append(summary(bk["bab_gross"])["tstat"])
    arr = np.array([t for t in ts if np.isfinite(t)])
    if arr.size == 0:
        return {"mean_null_t": float("nan"), "frac_abs_t_gt2": float("nan"), "n_seeds": 0}
    return {"mean_null_t": float(arr.mean()),
            "frac_abs_t_gt2": float((np.abs(arr) > 2).mean()), "n_seeds": int(arr.size)}
