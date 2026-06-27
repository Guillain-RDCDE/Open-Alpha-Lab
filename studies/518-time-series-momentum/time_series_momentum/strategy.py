"""The Time-Series-Momentum engine and its honest controls -- Study 518.

Moskowitz, Ooi & Pedersen (2012): an asset's OWN trailing 12-month return sign predicts its
next-month return, in every asset class. Trade each instrument long if its own past 12-month
return is positive, short if negative; scale each position by the inverse of its recent
realised volatility (a common risk target), then average across a diversified cross-asset
basket. This is the canonical "trend everywhere" factor.

This is distinct from cross-sectional momentum (Study 507: rank assets AGAINST EACH OTHER) and
from simple SMA trend timing (Study 110 Faber: price-vs-200d-average, long-flat single asset).
Here the signal is the *sign of the asset's own trailing return*, the book is *long AND short*,
and positions are *vol-scaled and diversified across asset classes*.

This module measures, honestly:

1. **The signal.** The monthly vol-scaled, diversified TSMOM portfolio return, GROSS, with a
   robust one-sample *t* (Newey-West HAC), Sharpe, hit-rate and max-drawdown.
2. **The null.** A sign-shuffle placebo: randomly flip each asset's position sign each month
   (destroying the genuine own-trend link while preserving the per-asset return magnitudes and
   the vol-scaling), recompute the portfolio mean many times, and report the share of placebos
   that beat the real mean (a permutation p-value). A real edge survives; a data-mined one does
   not.
3. **Costs.** One-way bps x NAV x turnover charged at each monthly rebalance, plus an annual
   borrow on the (time-varying) net short exposure pro-rated monthly. Reported as gross vs net.
4. **The positive control.** A deterministic synthetic panel with a planted own-asset trend the
   engine must recover (and a no-trend null it must score at zero) -- a faithful-engine / power
   check ONLY, never cited for the real-tape stamp.

Execution lag (documented exactly, ONE shift): the sign signal is computed from prices up to
and including month-end *m*; we do NOT trade on that close. We enter at the close one trading
day later (the first session of month *m+1*) and earn the realised return of month *m+1*. This
is a single, forward-only lag -- no same-bar fill, no look-ahead. The vol estimate used to size
month *m+1* uses only data through month-end *m*.

Survivorship: the basket is broad ETFs still trading in 2026, chosen ex post. TSMOM's own-sign
construction makes this far milder than a cross-sectional equity sort (a falling ETF is shorted,
never deleted), but a vehicle that survived to 2026 had no terminal blow-up, so any TSMOM
premium here is a mild upper bound. Named on the SIGNAL axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
TRADING_DAYS = 252

# 12-month own-return lookback (Moskowitz-Ooi-Pedersen use trailing 12 months, no skip).
LOOKBACK_M = 12
# Vol-scaling: target each position to a common annualised risk and cap leverage so a low-vol
# bond/FX leg cannot blow the book to 40x notional. MOP target ~40% per-position vol then scale
# the WHOLE portfolio to ~12%; equivalently here we target a modest per-asset vol and cap the
# per-position notional, keeping the diversified book near 1x gross-per-asset on average.
VOL_TARGET = 0.15        # annualised per-asset vol target (~median asset vol -> ~1x avg notional)
VOL_CAP = 3.0            # max absolute notional per asset (leverage cap on the inverse-vol size)
VOL_LOOKBACK_M = 12      # months of monthly returns used to estimate each asset's vol


# ---------------------------------------------------------------------------
# Monthly resampling and the own-sign signal
# ---------------------------------------------------------------------------
def to_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end adjusted-close prices (last observation in each calendar month)."""
    m = prices.resample("ME").last()
    return m.dropna(how="all")


def own_sign_signal(monthly_prices: pd.DataFrame, lookback: int = LOOKBACK_M) -> pd.DataFrame:
    """The sign of each asset's OWN trailing ``lookback``-month return at each month-end.

    For month-end *t* and asset *i*, the signal is ``sign(P[i,t] / P[i,t-lookback] - 1)`` --
    +1 if the asset's own trailing 12-month return is positive, -1 if negative, built purely
    from past prices (no look-ahead). Returns a (month x ticker) DataFrame of +/-1, NaN where
    insufficient history.
    """
    p = monthly_prices
    trail = p / p.shift(lookback) - 1.0
    return np.sign(trail)


def realised_vol(
    monthly_prices: pd.DataFrame, lookback: int = VOL_LOOKBACK_M
) -> pd.DataFrame:
    """Annualised trailing realised vol per asset, from monthly returns (no look-ahead).

    At month-end *t* the vol uses the trailing ``lookback`` monthly returns through *t* only.
    Used to size the *next* month's position -- a single forward step, no peeking.
    """
    mret = monthly_prices.pct_change()
    vol_m = mret.rolling(lookback, min_periods=max(3, lookback // 2)).std()
    return vol_m * np.sqrt(MONTHS_PER_YEAR)


# ---------------------------------------------------------------------------
# The vol-scaled, diversified TSMOM book
# ---------------------------------------------------------------------------
def tsmom_book(
    monthly_prices: pd.DataFrame,
    lookback: int = LOOKBACK_M,
    vol_lookback: int = VOL_LOOKBACK_M,
    vol_target: float = VOL_TARGET,
    vol_cap: float = VOL_CAP,
    cost_bps: float = 0.0,
    borrow_ann_bps: float = 0.0,
    vol_scale: bool = True,
    min_names: int = 4,
) -> pd.DataFrame:
    """Monthly diversified time-series-momentum portfolio, gross and net.

    Construction at each month-end *t*:
      1. For each asset, take the SIGN of its own trailing ``lookback``-month return.
      2. Size the position as ``sign * vol_target / realised_vol`` (inverse-vol scaling to a
         common risk target) if ``vol_scale`` else just the ``sign`` (equal-notional).
      3. Average the per-asset positions x next-month asset returns across all assets with a
         valid signal -> the diversified TSMOM portfolio return for month *t+1* (single forward
         execution lag -- form on the *t* close, hold the *t+1* month; never a same-bar fill).
      4. Costs: one-way turnover x cost_bps x NAV at the rebalance, plus an annual borrow on the
         average net SHORT notional pro-rated monthly. Net = gross - costs - borrow.

    Returns a DataFrame indexed by the HOLDING month with columns:
        ``port_gross`` -- diversified portfolio return (mean of position x asset return)
        ``turnover``   -- one-way fraction of book notional turned over vs the prior month
        ``short_frac`` -- average absolute short notional this month (for the borrow charge)
        ``port_net``   -- port_gross net of costs and borrow
        ``n_assets``   -- assets carrying a valid signal
    """
    fwd = monthly_prices.pct_change().shift(-1)   # month t -> realised return of month t+1
    sign = own_sign_signal(monthly_prices, lookback)
    vol = realised_vol(monthly_prices, vol_lookback)

    rows: list[dict] = []
    prev_w: pd.Series | None = None

    for t in sign.index:
        s = sign.loc[t]
        v = vol.loc[t] if t in vol.index else pd.Series(dtype=float)
        r = fwd.loc[t] if t in fwd.index else pd.Series(dtype=float)
        # Valid where we have a sign, a positive vol estimate and a forward return.
        valid = s.notna() & v.notna() & (v > 0) & r.notna()
        names = list(s.index[valid])
        if len(names) < min_names:
            continue

        sgn = s.loc[names].astype(float)
        if vol_scale:
            size = (vol_target / v.loc[names]).clip(upper=vol_cap)  # cap per-position leverage
            w = sgn * size
        else:
            w = sgn.copy()

        ret = r.loc[names].astype(float)
        # Diversified portfolio: average of position-weighted asset returns (1/N over assets).
        port_gross = float((w * ret).mean())

        # Turnover: average absolute change in per-asset weight vs prior month (one-way).
        if prev_w is not None:
            allnames = w.index.union(prev_w.index)
            wa = w.reindex(allnames).fillna(0.0)
            pa = prev_w.reindex(allnames).fillna(0.0)
            # one-way notional change, normalised by current gross notional
            gross_notional = wa.abs().sum()
            turnover = float((wa - pa).abs().sum() / (2.0 * gross_notional)) if gross_notional > 0 else 1.0
        else:
            turnover = 1.0
        prev_w = w

        # Borrow on the average short notional (vol-scaled) this month.
        short_frac = float((-w.clip(upper=0.0)).mean())
        cost = turnover * cost_bps * 1e-4
        borrow = short_frac * borrow_ann_bps * 1e-4 / MONTHS_PER_YEAR
        port_net = port_gross - cost - borrow

        rows.append(
            {
                "date": t,
                "port_gross": port_gross,
                "turnover": turnover,
                "short_frac": short_frac,
                "port_net": port_net,
                "n_assets": len(names),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date").dropna(subset=["port_gross"])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def hac_tstat(r: pd.Series) -> float:
    """Newey-West (HAC) t-stat on the mean of a monthly return series."""
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
    """Annualised statistics for a monthly return series.

    Returns mean (ann), vol (ann), Sharpe, HAC t-stat, hit-rate, max-drawdown, worst month, n.
    The HAC *t* is the inference-bar number.
    """
    s = pd.Series(r).astype(float).dropna()
    n = len(s)
    if n < 2:
        return {k: float("nan") for k in
                ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_dd", "worst", "n")}
    mean_ann = float(s.mean() * periods_per_year)
    vol_ann = float(s.std(ddof=1) * np.sqrt(periods_per_year))
    sr = mean_ann / vol_ann if vol_ann > 0 else float("nan")
    eq = (1.0 + s).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    return {
        "mean": mean_ann,
        "vol": vol_ann,
        "sharpe": sr,
        "tstat": hac_tstat(s),
        "hit_rate": float((s > 0).mean()),
        "max_dd": dd,
        "worst": float(s.min()),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# The placebo / sign-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(
    monthly_prices: pd.DataFrame,
    lookback: int = LOOKBACK_M,
    vol_lookback: int = VOL_LOOKBACK_M,
    vol_target: float = VOL_TARGET,
    vol_cap: float = VOL_CAP,
    n_perm: int = 1000,
    seed: int = 518,
) -> dict:
    """Permutation p-value: does the real TSMOM mean beat a sign-shuffled null?

    At each month we keep each asset's vol-scaled *magnitude* and its realised next-month
    return, but randomly FLIP each position's sign -- destroying the genuine own-trend link
    while preserving the position sizes and return distribution. We recompute the portfolio
    mean ``n_perm`` times and report the one-sided share of placebos whose mean >= the real
    mean. A real edge gives a small p; a data-mined one gives p ~ 0.5.

    Returns ``{"real_mean_ann", "p_value", "placebo_mean_ann", "n_perm"}``.
    """
    real = tsmom_book(monthly_prices, lookback=lookback, vol_lookback=vol_lookback,
                      vol_target=vol_target)
    if real.empty:
        return {"real_mean_ann": float("nan"), "p_value": float("nan"),
                "placebo_mean_ann": float("nan"), "n_perm": 0}
    real_mean = float(real["port_gross"].mean())

    fwd = monthly_prices.pct_change().shift(-1)
    sign = own_sign_signal(monthly_prices, lookback)
    vol = realised_vol(monthly_prices, vol_lookback)
    rng = np.random.default_rng(seed)

    # Pre-extract per-month aligned (|weight|, forward-return) arrays once.
    months: list[tuple[np.ndarray, np.ndarray]] = []
    for t in sign.index:
        s = sign.loc[t]
        v = vol.loc[t] if t in vol.index else pd.Series(dtype=float)
        r = fwd.loc[t] if t in fwd.index else pd.Series(dtype=float)
        valid = s.notna() & v.notna() & (v > 0) & r.notna()
        names = list(s.index[valid])
        if len(names) < 4:
            continue
        mag = (vol_target / v.loc[names]).clip(upper=vol_cap).to_numpy()  # capped magnitude
        rv = r.loc[names].to_numpy()
        months.append((mag, rv))

    if not months:
        return {"real_mean_ann": real_mean * MONTHS_PER_YEAR, "p_value": float("nan"),
                "placebo_mean_ann": float("nan"), "n_perm": 0}

    placebo_means = np.empty(n_perm)
    for b in range(n_perm):
        msum = 0.0
        for mag, rv in months:
            flips = rng.choice((-1.0, 1.0), size=mag.shape[0])  # random sign per asset
            msum += float((flips * mag * rv).mean())
        placebo_means[b] = msum / len(months)

    p = float((placebo_means >= real_mean).mean())
    return {
        "real_mean_ann": real_mean * MONTHS_PER_YEAR,
        "p_value": p,
        "placebo_mean_ann": float(placebo_means.mean() * MONTHS_PER_YEAR),
        "n_perm": int(n_perm),
    }


# ---------------------------------------------------------------------------
# Deterministic synthetic positive control
# ---------------------------------------------------------------------------
def synthetic_control(
    strengths: tuple[float, ...] = (0.0, 0.15, 0.30, 0.50),
    n_assets: int = 12,
    n_days: int = 2600,
    seed: int = 518,
) -> pd.DataFrame:
    """Plant a known own-asset trend and verify the TSMOM engine recovers it.

    Sweeps ``trend_strength`` on the deterministic synthetic panel and reports the portfolio
    mean and HAC *t* at each. The engine should score ~0 at strength 0 (null) and rise
    monotonically as the planted own-trend grows. Faithful-engine / power check ONLY -- never
    cited to support a real-tape stamp.
    """
    from . import data as _data

    rows: list[dict] = []
    for strength in strengths:
        prices, _truth = _data.synthetic_panel(
            n_assets=n_assets, n_days=n_days, trend_strength=strength, seed=seed
        )
        mp = to_monthly(prices)
        bk = tsmom_book(mp)
        if bk.empty:
            rows.append({"trend_strength": strength, "port_mean_ann": float("nan"),
                         "tstat": float("nan"), "n": 0})
            continue
        s = summary(bk["port_gross"])
        rows.append({"trend_strength": strength, "port_mean_ann": s["mean"],
                     "tstat": s["tstat"], "n": s["n"]})
    return pd.DataFrame(rows)


def synthetic_null_robustness(
    n_seeds: int = 20, n_assets: int = 12, n_days: int = 2600, seed0: int = 1000
) -> dict:
    """Average the null (trend_strength=0) HAC *t* across many seeds -- seed-robustness guard.

    A single lucky seed can hand a synthetic null a |t|>2 by chance. We sweep ``n_seeds``
    independent panels at the null and report the mean null *t* and the share with |t|>2.
    """
    ts: list[float] = []
    for i in range(n_seeds):
        from . import data as _data
        prices, _ = _data.synthetic_panel(
            n_assets=n_assets, n_days=n_days, trend_strength=0.0, seed=seed0 + i
        )
        mp = to_monthly(prices)
        bk = tsmom_book(mp)
        if bk.empty:
            continue
        ts.append(summary(bk["port_gross"])["tstat"])
    arr = np.array([t for t in ts if np.isfinite(t)])
    if arr.size == 0:
        return {"mean_null_t": float("nan"), "frac_abs_t_gt2": float("nan"), "n_seeds": 0}
    return {
        "mean_null_t": float(arr.mean()),
        "frac_abs_t_gt2": float((np.abs(arr) > 2).mean()),
        "n_seeds": int(arr.size),
    }
