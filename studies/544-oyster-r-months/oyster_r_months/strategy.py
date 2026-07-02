"""The engine and its honest controls — Study 544 (Oyster-R-Months).

The claim, dressed up: eat oysters only in months with the letter **R** (Sep–Apr); skip the
R-less months (May–Aug). Transposed to markets: do R-months out-earn R-less months? Because the
R/non-R split is (Sep–Apr) vs (May–Aug), this is calendrically almost identical to the
**sell-in-May / Halloween** seasonal (Nov–Apr vs May–Oct), just one month wider — so the honest
comparison is *both* the R-month split and its sell-in-May cousin, side by side.

This module measures that, honestly:

1. **The split statistic.** Two-sample (Welch) *t* on the difference in mean monthly return
   between R-months and R-less months. The claim predicts R > non-R (a positive gap).

2. **The sell-in-May cousin.** The same statistic on the Halloween split (Nov–Apr vs May–Oct),
   so the reader can see the R-month "signal" is the sell-in-May signal in a costume.

3. **A placebo / label-shuffle null.** Shuffle the R/non-R month labels against the returns many
   times; the p-value is the fraction of shuffles whose |gap| ≥ the observed. A real calendar
   effect sits in the tail; noise does not.

4. **The tradable rule + costs.** "Hold the market in R-months, sit in cash in R-less months."
   Compare its compound growth and Sharpe to buy-and-hold, and charge a one-way cost on every
   switch (into cash each May, back in each September). There is no short leg (you sit in cash),
   so no borrow — but if you *shorted* the R-less months you would pay borrow (reported as a
   sensitivity). Gross and net are both labelled.

5. **A synthetic positive control.** A deterministic world where an R-month premium is planted
   (``r_premium_bp > 0``); the engine must recover a positive, significant gap and stay flat at
   the null — averaged over ≥ 20 seeds (house rule) so no single lucky seed can fake it.

Execution convention: the R/non-R label is a calendar fact known before each month begins, so
there is no estimation and no look-ahead. The tradable rule switches at month boundaries (in on
the first day of an R-month, out on the first day of an R-less month); a one-day-later fill is
immaterial on a monthly bar (the single documented convention).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
R_MONTHS = {1, 2, 3, 4, 9, 10, 11, 12}       # months with an 'R' (Sep–Apr)
HALLOWEEN_WINTER = {11, 12, 1, 2, 3, 4}      # sell-in-May "winter" (Nov–Apr)


# ---------------------------------------------------------------------------
# The split statistic (generic two-sample Welch on a month mask)
# ---------------------------------------------------------------------------
def _welch(a: np.ndarray, b: np.ndarray) -> dict:
    """Welch two-sample t on mean(a) - mean(b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"gap": float("nan"), "t": float("nan"), "na": na, "nb": nb}
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    gap = a.mean() - b.mean()
    t = gap / se if se > 0 else float("nan")
    return {"gap": float(gap), "t": float(t), "na": na, "nb": nb,
            "mean_a": float(a.mean()), "mean_b": float(b.mean())}


def split_stat(ret: pd.Series, in_months: set[int] = R_MONTHS) -> dict:
    """Two-sample Welch t on (in-months) minus (out-months) mean monthly return.

    ``in_months`` defaults to the eight R-months (Sep–Apr). Returns the monthly gap, its Welch t,
    both group means (monthly and annualised), and the group sizes. The claim predicts a positive
    gap (in-months earn more).
    """
    if ret.empty:
        return {"gap": float("nan"), "t": float("nan"), "n": 0}
    m = pd.DatetimeIndex(ret.index).month
    is_in = np.isin(m, list(in_months))
    a = ret.to_numpy(dtype=float)[is_in]      # in-months
    b = ret.to_numpy(dtype=float)[~is_in]     # out-months
    res = _welch(a, b)
    res["n"] = int(len(ret))
    res["ann_in"] = float((1 + a.mean()) ** MONTHS_PER_YEAR - 1) if len(a) else float("nan")
    res["ann_out"] = float((1 + b.mean()) ** MONTHS_PER_YEAR - 1) if len(b) else float("nan")
    return res


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(ret: pd.Series, in_months: set[int] = R_MONTHS,
                   n_perm: int = 5000, seed: int = 544) -> float:
    """Label-shuffle placebo p-value for the R-month gap.

    Shuffle the R/non-R month labels against the returns ``n_perm`` times, preserving the count of
    "in" months; the p-value is the fraction of shuffles whose |gap| ≥ the observed |gap|.
    """
    if ret.empty or len(ret) < 12:
        return float("nan")
    m = pd.DatetimeIndex(ret.index).month
    is_in = np.isin(m, list(in_months))
    y = ret.to_numpy(dtype=float)
    n_in = int(is_in.sum())
    obs = abs(y[is_in].mean() - y[~is_in].mean())
    rng = np.random.default_rng(seed)
    n = len(y)
    count = 0
    for _ in range(n_perm):
        idx = rng.permutation(n)
        sel = idx[:n_in]
        rest = idx[n_in:]
        gap = y[sel].mean() - y[rest].mean()
        if abs(gap) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# The tradable rule + costs
# ---------------------------------------------------------------------------
def rule_vs_hold(ret: pd.Series, in_months: set[int] = R_MONTHS,
                 cost_bps: float = 5.0) -> dict:
    """'In during R-months, cash during R-less months' vs buy-and-hold.

    A monthly rule: hold the instrument (full return) in R-months, sit in cash (0%) in R-less
    months. Charge a one-way ``cost_bps`` on each *switch* (any month whose in/out state differs
    from the previous month). Returns compound annual growth (CAGR), annualised Sharpe (excess
    over 0), and total wealth multiple for both the rule (gross & net) and buy-and-hold.
    """
    if ret.empty:
        return {}
    m = pd.DatetimeIndex(ret.index).month
    is_in = np.isin(m, list(in_months))
    y = ret.to_numpy(dtype=float)

    rule_gross = np.where(is_in, y, 0.0)
    # A switch happens whenever the state changes vs the previous month (and at the very start
    # if we begin "in"). Each switch costs one-way cost_bps on the traded notional.
    state = is_in.astype(int)
    switches = np.abs(np.diff(np.concatenate([[0], state])))  # 1 on each entry/exit
    cost = switches * (cost_bps * 1e-4)
    rule_net = rule_gross - cost

    n_years = len(y) / MONTHS_PER_YEAR

    def _cagr(r):
        w = float(np.prod(1 + r))
        return w ** (1 / n_years) - 1, w

    def _sharpe(r):
        r = np.asarray(r, dtype=float)
        sd = r.std(ddof=1)
        return float(r.mean() / sd * np.sqrt(MONTHS_PER_YEAR)) if sd > 0 else float("nan")

    cg_hold, w_hold = _cagr(y)
    cg_g, w_g = _cagr(rule_gross)
    cg_n, w_n = _cagr(rule_net)
    return {
        "n_years": float(n_years),
        "hold_cagr": float(cg_hold), "hold_wealth": float(w_hold), "hold_sharpe": _sharpe(y),
        "rule_gross_cagr": float(cg_g), "rule_gross_wealth": float(w_g),
        "rule_gross_sharpe": _sharpe(rule_gross),
        "rule_net_cagr": float(cg_n), "rule_net_wealth": float(w_n),
        "rule_net_sharpe": _sharpe(rule_net),
        "n_switches": int(switches.sum()),
    }


def short_rless_net(ret: pd.Series, in_months: set[int] = R_MONTHS,
                    cost_bps: float = 5.0, borrow_ann_bps: float = 50.0) -> dict:
    """Sensitivity: a long-R / short-R-less book (the 'trade the gap' version), net of borrow.

    Long the instrument in R-months, *short* it in R-less months. The short leg pays an annual
    borrow, pro-rated over the ~4 R-less months a year. Returns gross and net CAGR. (This is the
    literal long-short expression of the seasonal; the main rule sits in cash instead.)
    """
    if ret.empty:
        return {}
    m = pd.DatetimeIndex(ret.index).month
    is_in = np.isin(m, list(in_months))
    y = ret.to_numpy(dtype=float)
    pos = np.where(is_in, 1.0, -1.0)             # +1 in R-months, -1 in R-less
    gross = pos * y
    # one-way cost on every flip; borrow charged only in short (R-less) months
    switches = np.abs(np.diff(np.concatenate([[0.0], pos]))) / 2.0
    cost = switches * (cost_bps * 1e-4)
    borrow = np.where(~is_in, borrow_ann_bps * 1e-4 / MONTHS_PER_YEAR, 0.0)
    net = gross - cost - borrow
    n_years = len(y) / MONTHS_PER_YEAR
    return {
        "gross_cagr": float(np.prod(1 + gross) ** (1 / n_years) - 1),
        "net_cagr": float(np.prod(1 + net) ** (1 / n_years) - 1),
    }


# ---------------------------------------------------------------------------
# Sub-period stability (decay)
# ---------------------------------------------------------------------------
def subperiod_stats(ret: pd.Series, cut: str = "2000-01-01",
                    in_months: set[int] = R_MONTHS) -> dict:
    """Split the tape at ``cut`` and report the R-month gap and t on each half."""
    if ret.empty:
        return {}
    early = ret[ret.index < pd.Timestamp(cut)]
    late = ret[ret.index >= pd.Timestamp(cut)]
    return {"early": split_stat(early, in_months), "late": split_stat(late, in_months)}


# ---------------------------------------------------------------------------
# Synthetic positive control — seed-robust (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, r_premium_bp: float, n_seeds: int = 25,
                     base_seed: int = 544, n_years: int = 80) -> float:
    """Average the R-month Welch t over ``n_seeds`` synthetic worlds for a planted premium.

    The house rule: any synthetic-dependent claim averages the stat over ≥ 20 seeds so no single
    lucky RNG seed can manufacture significance. Returns the mean gap-t across seeds.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        ser, _ = data_mod.synthetic_monthly(n_years=n_years, r_premium_bp=r_premium_bp, seed=s)
        ts.append(split_stat(ser, R_MONTHS)["t"])
    return float(np.nanmean(ts))
