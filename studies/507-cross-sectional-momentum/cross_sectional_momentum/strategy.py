"""The Cross-Sectional-Momentum engine and its honest controls -- Study 507.

Jegadeesh & Titman (1993): rank the cross-section by trailing 12-month return *skipping the
most recent month* (the classic "12-1" to dodge short-term reversal), go long the top fraction
(decile or quintile), short the bottom, rebalance monthly, hold one month. The canonical equity
momentum factor.

This module measures, honestly:

1. **The signal.** The monthly winners-minus-losers (WML) long-short spread, GROSS, with a
   robust one-sample *t* (Newey-West HAC), Sharpe, hit-rate and max-drawdown.
2. **The null.** A label-shuffle / sign-flip placebo: shuffle which stock gets which
   forward return, recompute the WML mean many times, and report the share of placebos that
   beat the real mean (a permutation p-value). A real edge survives; a data-mined one does not.
3. **Costs.** One-way bps x NAV x turnover charged at each monthly rebalance, plus an annual
   borrow on the short (loser) leg pro-rated monthly. Reported as gross vs net.
4. **The positive control.** A deterministic synthetic panel with a planted relative-strength
   drift the engine must recover (and a no-momentum null it must score at zero) -- a
   faithful-engine / power check ONLY, never cited for the real-tape stamp.

Execution lag (documented exactly, ONE shift): the signal is computed from prices up to and
including month-end *m*; we do NOT trade on that close. We enter at the close one trading day
later (the first session of month *m+1*) and earn the realised return of month *m+1*. This is a
single, forward-only lag -- no same-bar fill, no look-ahead.

Survivorship: the basket is names still trading in 2026. The loser leg's natural short
candidates -- firms that trended into delisting -- are absent, so any WML premium here is an
upper bound. Named on the SIGNAL axis. Opt-in guard: pass a delisting-complete panel to
``long_short`` to lift the bias; we cannot from yfinance, so we flag it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
TRADING_DAYS = 252

# 12-1 lookback: 12 months trailing, skip the most recent 1 month (reversal guard).
LOOKBACK_M = 12
SKIP_M = 1


# ---------------------------------------------------------------------------
# Monthly resampling and the 12-1 signal
# ---------------------------------------------------------------------------
def to_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end adjusted-close prices (last observation in each calendar month)."""
    m = prices.resample("ME").last()
    return m.dropna(how="all")


def momentum_signal(
    monthly_prices: pd.DataFrame, lookback: int = LOOKBACK_M, skip: int = SKIP_M
) -> pd.DataFrame:
    """The 12-1 trailing return signal at each month-end.

    For month-end *t*, the signal is the cumulative return from *t - lookback* to *t - skip*
    (i.e. trailing 12 months, dropping the most recent ``skip`` months). Built purely from
    past prices -- no look-ahead. Returns a (month x ticker) DataFrame of signal values,
    NaN where insufficient history.
    """
    p = monthly_prices
    # price at t-skip divided by price at t-lookback, minus 1
    sig = p.shift(skip) / p.shift(lookback) - 1.0
    return sig


# ---------------------------------------------------------------------------
# The winners-minus-losers (WML) long-short book
# ---------------------------------------------------------------------------
def long_short(
    monthly_prices: pd.DataFrame,
    frac: float = 0.2,
    lookback: int = LOOKBACK_M,
    skip: int = SKIP_M,
    cost_bps: float = 0.0,
    borrow_ann_bps: float = 0.0,
    min_names: int = 10,
) -> pd.DataFrame:
    """Monthly winners-minus-losers spread on a fraction sort, gross and net.

    Construction at each month-end *t*:
      1. Rank names by the 12-1 ``momentum_signal``.
      2. Long the top ``frac`` (winners), short the bottom ``frac`` (losers), equal-weight
         within each leg, dollar-neutral ($1 long / $1 short).
      3. Earn the realised return of month *t+1* (the single forward execution lag -- we form
         on the *t* close and hold the *t+1* month; never a same-bar fill).
      4. Costs: turnover x cost_bps x NAV at the rebalance, plus an annual borrow on the short
         leg pro-rated monthly. Net = gross - costs - borrow.

    ``frac=0.2`` is the quintile sort; ``frac=0.1`` the decile. On a ~40-name basket the
    decile leg holds only ~4 names -- the thin-leg fragility this study's third axis probes.

    Returns a DataFrame indexed by the HOLDING month with columns:
        ``win`` / ``los`` -- equal-weight realised return of each leg
        ``wml_gross``     -- win - los (dollar-neutral, gross)
        ``turnover``      -- one-way fraction of book turned over vs the prior month
        ``wml_net``       -- wml_gross net of costs and borrow
        ``n_leg``         -- names per leg
    """
    fwd = monthly_prices.pct_change().shift(-1)  # month t -> realised return of month t+1
    sig = momentum_signal(monthly_prices, lookback, skip)

    rows: list[dict] = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()

    for t in sig.index:
        s = sig.loc[t].dropna()
        r = fwd.loc[t].dropna() if t in fwd.index else pd.Series(dtype=float)
        common = s.index.intersection(r.index)
        s = s.loc[common]
        r = r.loc[common]
        n = len(s)
        if n < min_names:
            continue
        k = max(1, int(round(frac * n)))
        ranked = s.sort_values()
        los_names = list(ranked.index[:k])
        win_names = list(ranked.index[-k:])
        if set(los_names) & set(win_names):
            continue  # too few names to separate the legs

        win_ret = float(r[win_names].mean())
        los_ret = float(r[los_names].mean())
        wml_gross = win_ret - los_ret

        # Turnover: fraction of each leg replaced vs the prior rebalance (one-way, both legs).
        long_set, short_set = set(win_names), set(los_names)
        if prev_long or prev_short:
            long_to = len(long_set ^ prev_long) / (2 * k)
            short_to = len(short_set ^ prev_short) / (2 * k)
            turnover = 0.5 * (long_to + short_to)  # one-way per leg, averaged
        else:
            turnover = 1.0  # first month: full establishment
        prev_long, prev_short = long_set, short_set

        # Costs: both legs trade -> 2 x turnover x cost_bps; borrow on the short leg.
        cost = 2.0 * turnover * cost_bps * 1e-4
        borrow = borrow_ann_bps * 1e-4 / MONTHS_PER_YEAR
        wml_net = wml_gross - cost - borrow

        rows.append(
            {
                "date": t,
                "win": win_ret,
                "los": los_ret,
                "wml_gross": wml_gross,
                "turnover": turnover,
                "wml_net": wml_net,
                "n_leg": k,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date").dropna(subset=["wml_gross"])


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
# The placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(
    monthly_prices: pd.DataFrame,
    frac: float = 0.2,
    lookback: int = LOOKBACK_M,
    skip: int = SKIP_M,
    n_perm: int = 1000,
    seed: int = 507,
) -> dict:
    """Permutation p-value: does the real WML mean beat a shuffled-label null?

    At each month we keep the same forward returns but SHUFFLE which stock the momentum signal
    points at -- destroying any genuine winner/loser persistence while preserving the
    cross-sectional return distribution and the leg sizes. We recompute the WML mean
    ``n_perm`` times and report the one-sided share of placebos whose mean >= the real mean.
    A real edge gives a small p; a data-mined one gives p ~ 0.5.

    Returns ``{"real_mean_ann", "p_value", "placebo_mean_ann", "n_perm"}``.
    """
    real = long_short(monthly_prices, frac=frac, lookback=lookback, skip=skip)
    if real.empty:
        return {"real_mean_ann": float("nan"), "p_value": float("nan"),
                "placebo_mean_ann": float("nan"), "n_perm": 0}
    real_mean = float(real["wml_gross"].mean())

    fwd = monthly_prices.pct_change().shift(-1)
    sig = momentum_signal(monthly_prices, lookback, skip)
    rng = np.random.default_rng(seed)

    # Pre-extract per-month aligned (signal, forward-return) arrays once.
    months: list[tuple[np.ndarray, np.ndarray, int]] = []
    for t in sig.index:
        s = sig.loc[t].dropna()
        r = fwd.loc[t].dropna() if t in fwd.index else pd.Series(dtype=float)
        common = s.index.intersection(r.index)
        if len(common) < 10:
            continue
        sv = s.loc[common].to_numpy()
        rv = r.loc[common].to_numpy()
        k = max(1, int(round(frac * len(common))))
        if 2 * k > len(common):
            continue
        months.append((sv, rv, k))

    if not months:
        return {"real_mean_ann": real_mean * MONTHS_PER_YEAR, "p_value": float("nan"),
                "placebo_mean_ann": float("nan"), "n_perm": 0}

    placebo_means = np.empty(n_perm)
    for b in range(n_perm):
        spread_sum = 0.0
        for sv, rv, k in months:
            perm = rng.permutation(len(sv))  # shuffle the signal-> stock mapping
            order = np.argsort(sv[perm])     # rank by the shuffled signal
            los = rv[order[:k]].mean()
            win = rv[order[-k:]].mean()
            spread_sum += win - los
        placebo_means[b] = spread_sum / len(months)

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
    n_stocks: int = 40,
    n_days: int = 2600,
    frac: float = 0.2,
    seed: int = 507,
) -> pd.DataFrame:
    """Plant a known relative-strength drift and verify the WML engine recovers it.

    Sweeps ``mom_strength`` on the deterministic synthetic panel and reports the WML mean and
    HAC *t* at each. The engine should score ~0 at strength 0 (null) and rise monotonically as
    the planted drift grows. This is a faithful-engine / power check ONLY -- never cited to
    support a real-tape stamp.
    """
    from . import data as _data

    rows: list[dict] = []
    for strength in strengths:
        prices, truth = _data.synthetic_panel(
            n_stocks=n_stocks, n_days=n_days, mom_strength=strength, seed=seed
        )
        mp = to_monthly(prices)
        ls = long_short(mp, frac=frac)
        if ls.empty:
            rows.append({"mom_strength": strength, "wml_mean_ann": float("nan"),
                         "tstat": float("nan"), "n": 0})
            continue
        s = summary(ls["wml_gross"])
        rows.append({"mom_strength": strength, "wml_mean_ann": s["mean"],
                     "tstat": s["tstat"], "n": s["n"]})
    return pd.DataFrame(rows)
