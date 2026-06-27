"""The signal and its honest controls -- Study 531 (Enterprise-Multiple).

The claim (Loughran & Wellman 2011): sort stocks by the *enterprise multiple* EV/EBITDA
and go long the cheap (low-multiple) names, short the expensive (high-multiple) names.
EV/EBITDA normalises for capital structure (it adds net debt to the price) and uses an
operating-earnings denominator, so it is distinct from book-to-market (Study 530) or
cash-flow yield (Study 124).

We run the cross-section *monthly*: each month rank the available names by their most
recent already-public EV/EBITDA, long the cheap tercile, short the expensive tercile,
equal-weight, hold one month. The signal updates only once a year (fundamentals are
annual) but trades monthly, so turnover -- and therefore cost -- is low.

Decided by:
1. The dollar-neutral long-short hedge return, one-sample HAC t-stat vs zero.
2. A label-shuffle **placebo**: permute the cross-sectional signal within each month and
   re-run; the real t must beat this null.
3. **Costs**: one-way bps x turnover + borrow on the short leg -> gross vs net.
4. A deterministic **synthetic positive control**: the same engine recovers a planted
   premium (faithful-engine check only -- never cited for the real-tape stamp).

No look-ahead, one execution lag: the data layer makes the signal public only after a
reporting lag, and the monthly return is the *next* month's, so the position is entered
after the signal is known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core monthly cross-sectional sort
# ---------------------------------------------------------------------------
def long_short_hedge(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    frac: float = 1.0 / 3.0,
    min_firms: int = 12,
) -> pd.DataFrame:
    """Monthly long-cheap / short-expensive hedge on the enterprise multiple.

    For each month, rank names by ``signal`` (EV/EBITDA), long the cheapest ``frac``,
    short the most expensive ``frac``, equal-weight each leg. **Low multiple = cheap =
    long** (the value direction).

    Returns a month-indexed DataFrame with columns ``long`` (cheap-leg return),
    ``short`` (expensive-leg return), ``hedge`` (= long - short, dollar-neutral),
    ``market`` (equal-weight all names), ``turnover`` (fraction of the book that
    changed names vs last month, 0..2), and ``n`` (names used).
    """
    rows = {}
    prev_long: set = set()
    prev_short: set = set()
    for m in signal.index:
        if m not in fwd_ret.index:
            continue
        s = signal.loc[m].dropna()
        r = fwd_ret.loc[m].dropna()
        common = s.index.intersection(r.index)
        if len(common) < min_firms:
            continue
        s_, r_ = s.loc[common], r.loc[common]
        k = max(1, int(np.floor(len(common) * frac)))
        order = s_.sort_values()  # ascending multiple: cheapest first
        long_names = list(order.index[:k])          # low multiple -> long
        short_names = list(order.index[-k:])        # high multiple -> short
        long_ret = float(r_.loc[long_names].mean())
        short_ret = float(r_.loc[short_names].mean())

        # Turnover: names rotated out of each leg / leg size (each leg counted).
        if prev_long or prev_short:
            t_long = len(set(long_names) - prev_long) / max(1, len(long_names))
            t_short = len(set(short_names) - prev_short) / max(1, len(short_names))
            turnover = t_long + t_short
        else:
            turnover = 2.0  # initial build of both legs
        prev_long, prev_short = set(long_names), set(short_names)

        rows[m] = {
            "long": long_ret,
            "short": short_ret,
            "hedge": long_ret - short_ret,
            "market": float(r_.mean()),
            "turnover": turnover,
            "n": len(common),
        }
    out = pd.DataFrame(rows).T.sort_index()
    out.index.name = "month"
    return out.astype(float)


# ---------------------------------------------------------------------------
# Summary statistics (one-sample mean t + Newey-West HAC t)
# ---------------------------------------------------------------------------
def summary(monthly_series: pd.Series, ann_factor: int = 12) -> dict:
    """Stats for a monthly return series.

    ``mean``/``vol`` are annualised; ``sharpe`` annualised. ``t_simple`` is the plain
    one-sample t (mean / (sd/sqrt(n))) on the monthly series; ``t_hac`` is the
    Newey-West HAC t (lag = floor(4*(n/100)^(2/9))). ``hit_rate`` is fraction > 0,
    ``max_drawdown`` from the compounded equity curve.
    """
    r = pd.Series(monthly_series).astype(float).dropna()
    n = len(r)
    keys = ("mean", "vol", "sharpe", "t_simple", "t_hac", "hit_rate", "max_drawdown", "n")
    if n < 3:
        return {k: (np.nan if k != "n" else n) for k in keys}
    mu_m = float(r.mean())
    sd_m = float(r.std(ddof=1))

    t_simple = mu_m / (sd_m / np.sqrt(n)) if sd_m > 0 else np.nan

    e = r.to_numpy(dtype=float) - mu_m
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, min(lags + 1, n)):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    t_hac = mu_m / se if se > 0 else np.nan

    eq = (1.0 + r).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())

    return {
        "mean": mu_m * ann_factor,
        "vol": sd_m * np.sqrt(ann_factor),
        "sharpe": (mu_m / sd_m * np.sqrt(ann_factor)) if sd_m > 0 else np.nan,
        "t_simple": t_simple,
        "t_hac": t_hac,
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": max_dd,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Placebo: label-shuffle null
# ---------------------------------------------------------------------------
def placebo_null(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    n_shuffles: int = 500,
    seed: int = 531,
    frac: float = 1.0 / 3.0,
    min_firms: int = 12,
) -> dict:
    """Permute the cross-sectional signal *within each month* and re-run the hedge.

    Destroys any genuine signal-return alignment while preserving the marginal
    distributions. Returns ``{mean_t, p_value, n_shuffles}`` where ``p_value`` is the
    fraction of shuffled |t_hac| >= the real |t_hac| (two-sided). A real signal should
    have a small p-value.
    """
    real = summary(long_short_hedge(signal, fwd_ret, frac=frac, min_firms=min_firms)["hedge"])
    real_t = abs(real["t_hac"])
    if not np.isfinite(real_t):
        return {"real_t": np.nan, "p_value": np.nan, "n_shuffles": 0}

    rng = np.random.default_rng(seed)
    cols = list(signal.columns)
    count = 0
    ok = 0
    for _ in range(n_shuffles):
        shuf = signal.copy()
        vals = shuf.to_numpy(copy=True)
        for i in range(vals.shape[0]):
            row = vals[i]
            mask = ~np.isnan(row)
            idx = np.where(mask)[0]
            if idx.size > 1:
                perm = rng.permutation(idx)
                row[idx] = row[perm]
            vals[i] = row
        shuf = pd.DataFrame(vals, index=signal.index, columns=cols)
        s = summary(long_short_hedge(shuf, fwd_ret, frac=frac, min_firms=min_firms)["hedge"])
        if np.isfinite(s["t_hac"]):
            count += 1
            if abs(s["t_hac"]) >= real_t:
                ok += 1
    p = ok / count if count else np.nan
    return {"real_t": real["t_hac"], "p_value": p, "n_shuffles": count}


# ---------------------------------------------------------------------------
# Costs: one-way bps x turnover + borrow on the short leg
# ---------------------------------------------------------------------------
def apply_costs(
    hedge_df: pd.DataFrame,
    one_way_bps: float = 5.0,
    borrow_bps_annual: float = 50.0,
) -> pd.DataFrame:
    """Net the monthly hedge for trading + borrow cost.

    ``one_way_bps`` is charged on the dollar of book traded each month (turnover x NAV,
    both legs already counted in ``turnover``). ``borrow_bps_annual`` is the annual stock-
    loan fee on the short leg (notional ~1x NAV), charged monthly. Returns a copy with a
    ``net`` column added (= hedge - trade cost - borrow cost).
    """
    df = hedge_df.copy()
    trade_cost = (one_way_bps / 1e4) * df["turnover"]
    borrow_cost = (borrow_bps_annual / 1e4) / 12.0
    df["cost"] = trade_cost + borrow_cost
    df["net"] = df["hedge"] - df["cost"]
    return df


# ---------------------------------------------------------------------------
# Information coefficient (rank correlation of -EM with next-month return)
# ---------------------------------------------------------------------------
def monthly_ic(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    min_firms: int = 12,
) -> pd.Series:
    """Monthly Spearman IC between *cheapness* (-EV/EBITDA) and next-month return.

    Positive IC means cheap names (low multiple) tend to win -- the value direction.
    """
    from scipy.stats import spearmanr

    out = {}
    for m in signal.index:
        if m not in fwd_ret.index:
            continue
        s = signal.loc[m].dropna()
        r = fwd_ret.loc[m].dropna()
        common = s.index.intersection(r.index)
        if len(common) < min_firms:
            continue
        ic, _ = spearmanr(-s.loc[common], r.loc[common])  # negate: cheap = high
        if np.isfinite(ic):
            out[m] = ic
    return pd.Series(out).sort_index()


# ---------------------------------------------------------------------------
# Deterministic synthetic positive control (faithful-engine / power check)
# ---------------------------------------------------------------------------
def synthetic_control(
    premiums=(0.0, 0.002, 0.004, 0.006, 0.008),
    n_firms: int = 40,
    n_months: int = 180,
    n_seeds: int = 20,
    base_seed: int = 100,
) -> pd.DataFrame:
    """Average the hedge HAC t over ``n_seeds`` for each planted premium.

    Returns a DataFrame indexed by ``em_premium`` with the mean hedge return (%/yr) and
    the **seed-averaged** HAC t (so a "Real" synthetic claim never rests on one lucky
    seed). At premium 0 the engine should produce ~0 mean and |t|<2; as the premium rises
    the recovered mean and t should climb -- proof the engine is faithful. This is a
    power/faithfulness check ONLY; it never supports the real-tape stamp.
    """
    from . import data as _data

    rows = []
    for p in premiums:
        means, ts = [], []
        for s in range(n_seeds):
            em, fwd, _ = _data.synthetic_panel(
                n_firms=n_firms, n_months=n_months, em_premium=p, seed=base_seed + s
            )
            h = long_short_hedge(em, fwd)
            stt = summary(h["hedge"])
            means.append(stt["mean"])
            ts.append(stt["t_hac"])
        rows.append(
            {
                "em_premium": p,
                "mean_pct_yr": float(np.nanmean(means)) * 100.0,
                "t_hac_mean": float(np.nanmean(ts)),
                "n_seeds": n_seeds,
            }
        )
    return pd.DataFrame(rows).set_index("em_premium")
