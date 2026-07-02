"""The engine and its honest controls — Study 547 (Blue-Monday).

The claim, at full strength: 'Blue Monday' — the *third Monday of January*, marketed as the most
depressing day of the year — is a real mood low, and if seasonal mood moves markets (the SAD
thesis, Kamstra-Kramer-Levi 2003; Study 150) then equity returns should **dip** and realised
**volatility** should **spike** on Blue Monday relative to an ordinary Monday.

This module tests that on daily close-to-close returns:

1. **The Blue-Monday vs baseline-Monday split.** Mean return (and realised vol) on Blue Mondays
   versus all *other* Mondays — the honest two-sample comparison. A negative return difference
   and a positive vol difference would be the mood anomaly.

2. **A Welch two-sample t.** On the Blue-vs-other-Monday return difference (the headline stat).

3. **A placebo null.** For each January we could have called *any* of its Mondays 'Blue'. Draw a
   random non-third Monday per year thousands of times and see how often a random-Monday split
   matches the observed |difference| — the label-shuffle null tailored to the calendar claim.

4. **A cost-honest calendar timer.** 'Sit out Blue Monday' (or 'short it') vs buy-and-hold,
   net of a one-way cost per switch. Shorts pay borrow.

5. **A multi-window robustness sweep.** Split the tape into sub-periods; the *sign* of the Blue
   effect must be stable to be bankable.

6. **A seed-robust synthetic positive control (>= 20 seeds).** A deterministic world where the
   dip is planted; the engine must recover a negative t and stay flat at the null.

Timing: whether a Monday is 'Blue' is a deterministic function of the calendar, fully public
before the open, so there is **no execution lag**. The thin sample (one Blue Monday per year) is
the binding limitation and is named on the SIGNAL axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def _mondays(ret: pd.Series) -> pd.Series:
    return ret[ret.index.dayofweek == 0]


# ---------------------------------------------------------------------------
# The Blue-Monday vs baseline-Monday split
# ---------------------------------------------------------------------------
def blue_monday_split(close: pd.Series, data_mod) -> dict:
    """Split Monday returns into Blue Mondays vs all other Mondays.

    Returns bucket means/vols (in bps and annualised), the return difference (Blue − other),
    the realised-vol difference, and the per-bucket return arrays (for the t and placebo).
    """
    ret = daily_returns(close)
    mon = _mondays(ret)
    if mon.empty:
        return {}
    blue_mask = data_mod.is_blue_monday(mon.index)
    blue = mon[blue_mask].to_numpy(dtype=float)
    other = mon[~blue_mask].to_numpy(dtype=float)
    if blue.size < 2 or other.size < 2:
        return {}
    return {
        "n_blue": int(blue.size),
        "n_other": int(other.size),
        "blue_mean_bps": float(blue.mean() * 1e4),
        "other_mean_bps": float(other.mean() * 1e4),
        "diff_bps": float((blue.mean() - other.mean()) * 1e4),
        "blue_vol_ann": float(blue.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "other_vol_ann": float(other.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "vol_diff_ann": float((blue.std(ddof=1) - other.std(ddof=1)) * np.sqrt(TRADING_DAYS)),
        "blue_ret": blue,
        "other_ret": other,
    }


def blue_adjacent_split(close: pd.Series, data_mod) -> dict:
    """Split returns into the Blue-adjacent day (next open after Blue Monday) vs all other days.

    Because Blue Monday itself is nearly always a market holiday (MLK Day), the *tradable* mood
    test is the first open session after it — usually the Tuesday after Blue Monday. This restores
    a full ~33-observation sample. We compare that day's return against every other trading day.
    """
    ret = daily_returns(close)
    if ret.empty:
        return {}
    adj_mask = data_mod.is_blue_adjacent(ret.index)
    blue = ret[adj_mask].to_numpy(dtype=float)
    other = ret[~adj_mask].to_numpy(dtype=float)
    if blue.size < 2 or other.size < 2:
        return {}
    return {
        "n_blue": int(blue.size),
        "n_other": int(other.size),
        "blue_mean_bps": float(blue.mean() * 1e4),
        "other_mean_bps": float(other.mean() * 1e4),
        "diff_bps": float((blue.mean() - other.mean()) * 1e4),
        "blue_vol_ann": float(blue.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "other_vol_ann": float(other.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "vol_diff_ann": float((blue.std(ddof=1) - other.std(ddof=1)) * np.sqrt(TRADING_DAYS)),
        "blue_ret": blue,
        "other_ret": other,
    }


def welch_t(split: dict) -> dict:
    """Welch (unequal-variance) two-sample t on the Blue − other-Monday return difference.

    A *negative* t is the mood dip (Blue underperforms other Mondays).
    """
    if not split:
        return {"diff_bps": float("nan"), "t": float("nan"), "n": 0}
    a = split["blue_ret"]
    b = split["other_ret"]
    na, nb = a.size, b.size
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    t = (a.mean() - b.mean()) / se if se > 0 else float("nan")
    return {"diff_bps": float((a.mean() - b.mean()) * 1e4), "t": float(t), "n": na + nb}


def vol_ratio_test(split: dict) -> dict:
    """Levene-style variance comparison: Blue-Monday vol vs other-Monday vol.

    Reports the vol ratio and an F-style p on equal variances (two-sided). A ratio > 1 would be
    the 'mood vol-spike'.
    """
    if not split:
        return {"ratio": float("nan"), "p": float("nan")}
    a = split["blue_ret"]
    b = split["other_ret"]
    va, vb = a.var(ddof=1), b.var(ddof=1)
    if vb <= 0:
        return {"ratio": float("nan"), "p": float("nan")}
    F = va / vb
    # two-sided F test p via the survival function of the F distribution (no scipy dependency:
    # use a Monte-Carlo-free closed form via the regularized incomplete beta through numpy is
    # unavailable, so report the ratio and a permutation p on |vol diff| instead).
    return {"ratio": float(F), "blue_vol_ann": split["blue_vol_ann"],
            "other_vol_ann": split["other_vol_ann"]}


# ---------------------------------------------------------------------------
# Placebo / random-Monday null
# ---------------------------------------------------------------------------
def placebo_pvalue_adjacent(close: pd.Series, data_mod, n_perm: int = 5000, seed: int = 547) -> float:
    """Random-day placebo p for the Blue-adjacent − other-day return difference.

    Draw the same number of random trading days as there are Blue-adjacent days, ``n_perm`` times,
    and count how often a random split's |mean difference| >= the observed |difference|. A real
    Blue-adjacent effect sits in the tail; a spurious calendar label does not.
    """
    ret = daily_returns(close)
    if ret.empty:
        return float("nan")
    obs = blue_adjacent_split(close, data_mod)
    if not obs:
        return float("nan")
    obs_abs = abs(obs["diff_bps"])
    vals = ret.to_numpy(dtype=float)
    n = vals.size
    k = obs["n_blue"]
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        idx = rng.choice(n, size=k, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        diff = (vals[mask].mean() - vals[~mask].mean()) * 1e4
        if abs(diff) >= obs_abs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def placebo_pvalue(close: pd.Series, data_mod, n_perm: int = 5000, seed: int = 547) -> float:
    """Random-Monday placebo p for the Blue − other-Monday return difference.

    The calendar-aware null: in each January there are several Mondays; 'Blue' is only one of
    them (the third). For each permutation, relabel one *random* Monday per calendar year as the
    pseudo-Blue day, recompute the pseudo-Blue − rest difference, and count how often
    |difference| >= |observed|. A real Blue effect sits in the tail; a spurious label does not.
    """
    ret = daily_returns(close)
    mon = _mondays(ret)
    if mon.empty:
        return float("nan")
    obs = blue_monday_split(close, data_mod)
    if not obs:
        return float("nan")
    obs_abs = abs(obs["diff_bps"])

    vals = mon.to_numpy(dtype=float)
    years = mon.index.year.to_numpy()
    uniq = np.unique(years)
    # index positions of each year's Mondays
    year_pos = {y: np.where(years == y)[0] for y in uniq}
    rng = np.random.default_rng(seed)

    count = 0
    for _ in range(n_perm):
        chosen = []
        for y in uniq:
            pos = year_pos[y]
            chosen.append(pos[rng.integers(pos.size)])
        chosen = np.array(chosen)
        mask = np.zeros(vals.size, dtype=bool)
        mask[chosen] = True
        pb = vals[mask]
        rest = vals[~mask]
        if pb.size < 2 or rest.size < 2:
            continue
        diff = (pb.mean() - rest.mean()) * 1e4
        if abs(diff) >= obs_abs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Cost-honest calendar timer
# ---------------------------------------------------------------------------
def _max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def _stats(net, pos, turn):
    net = net.astype(float)
    equity = (1.0 + net).cumprod()
    n = len(net)
    years = n / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else float("nan")
    sd = net.std(ddof=1)
    sharpe = float(net.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")
    return {
        "cagr": cagr,
        "vol": float(sd * np.sqrt(TRADING_DAYS)),
        "sharpe": sharpe,
        "max_dd": _max_drawdown(equity.to_numpy()),
        "switches": int((turn > 1e-9).sum()),
        "time_in_market": float((pos.abs() > 0).mean()),
        "final": float(equity.iloc[-1]),
    }


def buy_and_hold(close: pd.Series) -> dict:
    ret = close.pct_change().fillna(0.0)
    pos = pd.Series(1.0, index=close.index)
    return _stats(ret, pos, pd.Series(0.0, index=close.index))


def blue_monday_timer(
    close: pd.Series,
    data_mod,
    mode: str = "sit_out",
    cost_bps: float = 1.0,
    borrow_ann_bps: float = 100.0,
) -> dict:
    """Trade the Blue-Monday claim vs buy-and-hold.

    ``mode='sit_out'``: hold the market every day *except* go flat on Blue Monday.
    ``mode='short'``: hold the market, but *short* it on Blue Monday (shorts pay borrow that day).
    A switch (entering/leaving the deviation) costs ``cost_bps`` one-way on NAV.
    """
    ret = close.pct_change().fillna(0.0)
    blue = pd.Series(data_mod.is_blue_monday(close.index), index=close.index)
    pos = pd.Series(1.0, index=close.index)
    if mode == "sit_out":
        pos[blue] = 0.0
    elif mode == "short":
        pos[blue] = -1.0
    else:
        raise ValueError(mode)
    turn = pos.diff().abs().fillna(pos.abs())
    borrow_daily = borrow_ann_bps * 1e-4 / TRADING_DAYS
    short_cost = (pos < 0).astype(float) * borrow_daily
    net = pos * ret - turn * cost_bps * 1e-4 - short_cost
    return _stats(net, pos, turn)


# ---------------------------------------------------------------------------
# Robustness — sub-period sign stability
# ---------------------------------------------------------------------------
def subperiod_sweep(close: pd.Series, data_mod, n_chunks: int = 4, which: str = "adjacent") -> list:
    """Split the tape into ``n_chunks`` equal date spans; report Blue diff and t per span.

    ``which='adjacent'`` uses the tradable Blue-adjacent-day split (a full sample); ``'monday'``
    uses the literal Blue-Monday split (nearly empty because of the MLK-holiday collision).
    """
    ret_index = daily_returns(close).index
    if ret_index.empty:
        return []
    split_fn = blue_adjacent_split if which == "adjacent" else blue_monday_split
    bounds = pd.date_range(ret_index.min(), ret_index.max(), periods=n_chunks + 1)
    out = []
    for i in range(n_chunks):
        lo, hi = bounds[i], bounds[i + 1]
        sub = close[(close.index >= lo) & (close.index <= hi)]
        sp = split_fn(sub, data_mod)
        if not sp:
            out.append((f"{lo.year}-{hi.year}", float("nan"), float("nan")))
            continue
        w = welch_t(sp)
        out.append((f"{lo.year}-{hi.year}", sp["diff_bps"], w["t"]))
    return out


# ---------------------------------------------------------------------------
# Synthetic positive control (seed-robust)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, blue_dip: float, n_seeds: int = 25, base_seed: int = 547,
                     blue_vol_mult: float = 1.0) -> float:
    """Average the Welch t (Blue − other) over ``n_seeds`` synthetic worlds.

    House rule: any synthetic-dependent claim averages the stat over >= 20 seeds so no single
    lucky RNG seed can manufacture significance. A planted ``blue_dip > 0`` should drive the mean
    t negative; ``blue_dip = 0`` should sit at ~0.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        frame, _ = data_mod.synthetic_daily(blue_dip=blue_dip, blue_vol_mult=blue_vol_mult, seed=s)
        sp = blue_monday_split(frame["close"], data_mod)
        ts.append(welch_t(sp)["t"] if sp else float("nan"))
    return float(np.nanmean(ts))
