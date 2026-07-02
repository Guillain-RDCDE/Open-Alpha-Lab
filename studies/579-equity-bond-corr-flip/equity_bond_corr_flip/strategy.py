"""The engine and its honest controls — Study 579 (Equity-Bond-Corr-Flip).

The claim, at full strength: the *sign* of the trailing stock-bond return correlation is a regime
timing signal for the 60/40 (equity/bond) portfolio. In the negative-correlation regime bonds
hedge equity drawdowns and 60/40 diversification works; when the correlation *flips positive* (as
in 2022) the hedge fails, so a timer should cut its 60/40 (or bond-hedge) exposure. The testable
version: **conditional on a positive trailing-correlation sign, the forward 60/40 (or equity)
return is lower than after a negative-correlation sign** — enough to time on.

This module measures that, honestly:

1. **The regime signal.** Each month, the sign of the trailing 6-month SPY/TLT return correlation
   (+1 positive-corr regime, -1 negative). Fully public at the month's close (no look-ahead).

2. **The regime split.** Group the *forward* one-month 60/40 (and equity) returns by the current
   correlation sign and compare their means — a two-sample (Welch) *t* on
   ``fwd | corr<0`` minus ``fwd | corr>0``. The claim predicts a *positive* difference (60/40
   does better after negative-corr months).

3. **The timing overlay.** A rules timer that holds the 60/40 when corr < 0 and de-risks (to a
   bond-only or cash sleeve) when corr > 0, benchmarked against static 60/40 — gross and net of a
   one-way switching cost, with the de-risked sleeve paying nothing extra (long-only, no borrow;
   if a short-bond variant is used, it pays borrow).

4. **A label-shuffle placebo** null on the regime split, and a **multi-window robustness** sweep.

5. **A synthetic positive control.** A deterministic world where the effect is planted
   (``regime_edge > 0``); the engine must recover a positive negative-minus-positive spread /
   timer edge and stay flat at the null — averaged over >= 20 seeds (house rule).

Execution convention: the correlation sign at month *t* is computed from the trailing window ending
at *t*'s close, so it is public at that close; the timer's position for month *t → t+1* is entered
at *t*'s close and held one month. One documented execution lag; a one-month-later entry is tested
in the robustness sweep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# The regime split — the sign of the correlation vs forward returns
# ---------------------------------------------------------------------------
def regime_split(monthly: pd.DataFrame, col: str = "fwd_6040") -> dict:
    """Split forward returns by the trailing-correlation sign and compare means.

    Groups ``col`` (forward 60/40 or equity return) into the negative-corr regime (corr_sign < 0)
    and positive-corr regime (corr_sign > 0), and returns the two group means, the spread
    (neg - pos, the claim predicts > 0), the group sizes, and the raw group arrays for the t-test.
    Months with no signal (corr_sign == 0, the warm-up window) are dropped.
    """
    if monthly.empty:
        return {}
    df = monthly.dropna(subset=[col])
    neg = df.loc[df["corr_sign"] < 0, col].to_numpy(dtype=float)
    pos = df.loc[df["corr_sign"] > 0, col].to_numpy(dtype=float)
    if len(neg) == 0 or len(pos) == 0:
        return {}
    return {
        "neg_mean": float(neg.mean()),
        "pos_mean": float(pos.mean()),
        "spread": float(neg.mean() - pos.mean()),  # claim: > 0
        "n_neg": int(len(neg)),
        "n_pos": int(len(pos)),
        "neg_ret": neg,
        "pos_ret": pos,
    }


def split_tstat(split: dict) -> dict:
    """Welch two-sample t on (forward return | corr<0) minus (forward return | corr>0).

    The claim's directional prediction is a *positive* t (the 60/40 does better after
    negative-correlation months). Returns the spread, the t-stat and total n.
    """
    if not split:
        return {"spread": float("nan"), "t": float("nan"), "n": 0}
    a = split["neg_ret"]
    b = split["pos_ret"]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"spread": float("nan"), "t": float("nan"), "n": na + nb}
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    t = (a.mean() - b.mean()) / se if se > 0 else float("nan")
    return {"spread": float(a.mean() - b.mean()), "t": float(t), "n": na + nb}


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(monthly: pd.DataFrame, col: str = "fwd_6040",
                   n_perm: int = 2000, seed: int = 579) -> float:
    """Label-shuffle placebo p-value for the regime spread.

    Shuffle the correlation-sign labels against the forward returns ``n_perm`` times; the p-value
    is the fraction of shuffles whose |spread| >= the observed |spread|. A real regime signal sits
    in the tail; noise does not.
    """
    if monthly.empty:
        return float("nan")
    base = regime_split(monthly, col)
    if not base:
        return float("nan")
    obs = abs(base["spread"])
    df = monthly.dropna(subset=[col])
    signs = df["corr_sign"].to_numpy(dtype=float)
    signs = signs[signs != 0]
    y = df.loc[df["corr_sign"] != 0, col].to_numpy(dtype=float)
    n = len(y)
    if n < 6:
        return float("nan")
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        s = signs[perm]
        neg = y[s < 0]
        pos = y[s > 0]
        if len(neg) == 0 or len(pos) == 0:
            continue
        spread = neg.mean() - pos.mean()
        if abs(spread) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# The timing overlay — hold 60/40 when corr<0, de-risk when corr>0
# ---------------------------------------------------------------------------
def timer_overlay(monthly: pd.DataFrame, cost_bps: float = 5.0,
                  derisk_to: str = "bond") -> dict:
    """A rules correlation-regime timer vs static 60/40, gross and net of switching costs.

    Each month: if the trailing-corr sign is negative (hedge working), hold the static 60/40 at
    full weight; if positive (hedge broken, the claim says), de-risk to a reduced stance.
    ``derisk_to='bond'`` scales the book down to 0.4x (a defensive, long-only tilt — no borrow);
    ``derisk_to='cash'`` moves to a flat 0% (cash) sleeve. Both stances apply their weight to the
    realised forward one-month 60/40 return (``fwd_6040``); nothing is shorted, so no borrow is
    charged. A one-way ``cost_bps`` is charged each time the stance changes (entering the new
    stance) — the single friction of a monthly timer.

    Returns dict with the timer's and static's cumulative growth, annualised means, the switch
    count, and gross/net timer-minus-static spread (annualised).
    """
    if monthly.empty:
        return {}
    df = monthly.dropna(subset=["fwd_6040"]).copy()
    static_fwd = df["fwd_6040"].to_numpy(dtype=float)
    sign = df["corr_sign"].to_numpy(dtype=float)

    # Stance: 1.0 = full 60/40 (neg-corr or warm-up), else de-risked weight.
    derisk_w = 0.4 if derisk_to == "bond" else 0.0
    stance = np.where(sign > 0, derisk_w, 1.0)
    timer_fwd_gross = stance * static_fwd

    # Switching cost: charge cost_bps one-way each time stance changes (entering the new stance).
    switches = np.zeros(len(stance))
    switches[1:] = (stance[1:] != stance[:-1]).astype(float)
    switches[0] = 0.0
    n_switch = int(switches.sum())
    cost = switches * (cost_bps * 1e-4)
    timer_fwd_net = timer_fwd_gross - cost

    months = len(df)
    ann = 12.0
    static_ann = float(np.mean(static_fwd) * ann)
    timer_gross_ann = float(np.mean(timer_fwd_gross) * ann)
    timer_net_ann = float(np.mean(timer_fwd_net) * ann)

    static_cum = float(np.prod(1.0 + static_fwd) - 1.0)
    timer_gross_cum = float(np.prod(1.0 + timer_fwd_gross) - 1.0)
    timer_net_cum = float(np.prod(1.0 + timer_fwd_net) - 1.0)

    static_vol = float(np.std(static_fwd, ddof=1) * np.sqrt(ann))
    timer_vol = float(np.std(timer_fwd_net, ddof=1) * np.sqrt(ann))
    static_sharpe = static_ann / static_vol if static_vol > 0 else float("nan")
    timer_sharpe = timer_net_ann / timer_vol if timer_vol > 0 else float("nan")

    return {
        "months": months,
        "n_switch": n_switch,
        "static_ann": static_ann,
        "timer_gross_ann": timer_gross_ann,
        "timer_net_ann": timer_net_ann,
        "static_cum": static_cum,
        "timer_gross_cum": timer_gross_cum,
        "timer_net_cum": timer_net_cum,
        "static_vol": static_vol,
        "timer_vol": timer_vol,
        "static_sharpe": static_sharpe,
        "timer_sharpe": timer_sharpe,
        "edge_gross_ann": timer_gross_ann - static_ann,
        "edge_net_ann": timer_net_ann - static_ann,
    }


# ---------------------------------------------------------------------------
# Robustness — multi-window regime split
# ---------------------------------------------------------------------------
def window_sweep(monthly: pd.DataFrame, windows, col: str = "fwd_6040") -> list:
    """Regime spread + t over several date windows.

    ``windows`` is a list of ``(label, start, end)`` string triples. Returns a list of
    ``(label, spread, t, n_neg, n_pos)`` — the sign and strength of the regime split per window.
    """
    out = []
    for lab, start, end in windows:
        sub = monthly.loc[start:end]
        sp = regime_split(sub, col)
        ts = split_tstat(sp)
        out.append((lab, ts["spread"], ts["t"], sp.get("n_neg", 0), sp.get("n_pos", 0)))
    return out


# ---------------------------------------------------------------------------
# Seed-robust synthetic control (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, regime_edge: float, n_seeds: int = 25,
                     base_seed: int = 579, col: str = "fwd_6040") -> float:
    """Average the regime-split t over ``n_seeds`` synthetic worlds.

    The house rule: any synthetic-dependent claim averages the stat over >= 20 seeds so no single
    lucky RNG seed can manufacture significance. Returns the mean split-t across seeds for a
    planted ``regime_edge``.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        monthly, _ = data_mod.synthetic_series(regime_edge=regime_edge, seed=s)
        ts.append(split_tstat(regime_split(monthly, col))["t"])
    return float(np.nanmean(ts))
