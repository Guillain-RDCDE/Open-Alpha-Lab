"""Strategy + inference for Study 526 — Intangible-Value (Lev-Srivastava adjusted book-to-market).

The rule under test (the believers' / academic version):

    Each month, rank the field by **book-to-market** (B/M). Hold the **top tertile equal-weight
    (long the high-B/M = cheap value names)** and **short the bottom tertile (low-B/M = expensive
    growth names)** for the next month; rebalance monthly. The long-cheap / short-expensive spread
    is the classic *value premium*.

The Lev-Srivastava decisive claim is that the value premium has decayed because GAAP expenses
intangible building (R&D, brand/SG&A), so reported book understates the true capital of
intangible-heavy firms and pollutes the B/M sort. Capitalising those outlays into an
**intangible-adjusted book** is supposed to *sharpen* the value sort. So the study's third axis is
the **adjustment contrast**: does the intangible-adjusted-B/M long-short beat the *plain*-B/M
long-short on the same basket, as Lev-Srivastava predict?

We measure it honestly:

  * **One execution lag, exact.** The signal at month-end ``t`` (book/intangibles from the most
    recent reported fiscal year, price as of ``t``) selects the book entered at the **next** month's
    open and held through month ``t+1``. No same-bar fill, no look-ahead in the accounting (1-year
    reporting lag).
  * **HAC inference.** Newey-West t-stat of the monthly long-short (and long-minus-SPY) spread —
    the Signal-axis test. ``REAL`` needs ``t >= 2`` on the real tape **and** survival of a
    label-shuffle placebo null.
  * **Placebo / label-shuffle null.** Permute the cross-sectional signal labels across names many
    times and rebuild the long-short; the real long-short must sit in the tail of that null.
  * **Costs + borrow.** One-way turnover × NAV at ``cost_bps`` per rebalance, plus an annual
    **borrow** fee on the short leg (you pay to be short the expensive growth names).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# Monthly tertile membership from a cross-sectional signal panel
# --------------------------------------------------------------------------- #
def tertile_members(signal_row: pd.Series, frac: float = 1 / 3) -> tuple[list[str], list[str]]:
    """``(long, short)`` names from one month's signal row: top-``frac`` (high B/M = cheap, long)
    and bottom-``frac`` (low B/M = expensive, short)."""
    x = signal_row.dropna()
    if len(x) < 3:
        return [], []
    k = max(1, int(round(len(x) * frac)))
    ranked = x.sort_values(ascending=False)
    return list(ranked.index[:k]), list(ranked.index[-k:])


def _leg_returns(signal: pd.DataFrame, returns: pd.DataFrame, frac: float,
                 which: str) -> tuple[pd.Series, pd.Series]:
    """Monthly equal-weight returns of the long (or short) leg, with a ONE-MONTH execution lag and
    one-way turnover (fraction of names that changed) per month.

    The signal as of month ``t`` (its row in ``signal``) selects the book that earns month ``t+1``'s
    return. Returns ``(leg_ret, turnover)`` aligned to the *holding* months (t+1 onward)."""
    sig_months = signal.index
    out_ret, out_turn, out_idx = [], [], []
    prev_set: set[str] = set()
    for i in range(len(sig_months) - 1):
        t = sig_months[i]
        hold = sig_months[i + 1]
        if hold not in returns.index:
            continue
        lo, sh = tertile_members(signal.loc[t], frac=frac)
        members = lo if which == "long" else sh
        members = [m for m in members if m in returns.columns]
        if not members:
            prev_set = set()
            continue
        r = returns.loc[hold, members].astype(float)
        leg = float(np.nanmean(r.to_numpy()))
        cur = set(members)
        # one-way turnover = fraction of weight traded = |new ^ old| / (2*size), equal-weight
        if prev_set:
            changed = len(cur ^ prev_set) / (2.0 * max(len(cur), 1))
        else:
            changed = 1.0
        out_ret.append(leg)
        out_turn.append(changed)
        out_idx.append(hold)
        prev_set = cur
    idx = pd.DatetimeIndex(out_idx, name="date")
    return (pd.Series(out_ret, index=idx, name=f"{which}_ret"),
            pd.Series(out_turn, index=idx, name=f"{which}_turn"))


def signal_books(signal: pd.DataFrame, returns: pd.DataFrame, frac: float = 1 / 3,
                 cost_bps: float = 0.0, borrow_bps: float = 0.0) -> dict:
    """Build long, short, long-short books from a monthly cross-sectional signal panel.

    Returns a dict of monthly return series (gross + net of ``cost_bps`` turnover; ``ls_net`` also
    nets an annual ``borrow_bps`` on the short leg) plus the realised average turnover.
    """
    lr, lt = _leg_returns(signal, returns, frac, "long")
    sr, st_ = _leg_returns(signal, returns, frac, "short")
    df = pd.concat([lr, sr, lt, st_], axis=1, join="inner").dropna()
    long_g = df["long_ret"]
    short_g = df["short_ret"]
    ls_g = long_g - short_g
    c = cost_bps * 1e-4
    long_c = long_g - df["long_turn"] * c
    short_c = short_g - df["short_turn"] * c
    borrow_monthly = borrow_bps * 1e-4 / MONTHS_PER_YEAR
    ls_n = long_c - short_c - borrow_monthly
    return {
        "long": long_g, "short": short_g, "long_short": ls_g,
        "long_net": long_c, "ls_net": ls_n,
        "avg_turnover": float((df["long_turn"].mean() + df["short_turn"].mean()) / 2.0),
    }


# --------------------------------------------------------------------------- #
# Placebo / label-shuffle null
# --------------------------------------------------------------------------- #
def placebo_null(signal: pd.DataFrame, returns: pd.DataFrame, frac: float = 1 / 3,
                 n_shuffles: int = 400, seed: int = 526) -> np.ndarray:
    """Sampling distribution of the long-short under **shuffled signal labels**.

    Each draw permutes the column labels of the signal panel (same per-name signal *values*, wrong
    names), rebuilds the long-short, and records its annualised mean. The real long-short must sit
    in the tail of this null for the signal — not the field's heterogeneity — to be doing the work.
    """
    cols = list(signal.columns)
    out = np.empty(n_shuffles)
    for d in range(n_shuffles):
        rng = np.random.default_rng(seed + d)
        perm = rng.permutation(len(cols))
        shuffled = signal.copy()
        shuffled.columns = [cols[p] for p in perm]
        shuffled = shuffled[cols]                       # realign to canonical order
        b = signal_books(shuffled, returns, frac=frac)
        out[d] = b["long_short"].mean() * MONTHS_PER_YEAR
    return out


def percentile_of(value: float, sample: np.ndarray) -> float:
    s = np.asarray(sample, dtype=float)
    s = s[~np.isnan(s)]
    if s.size == 0:
        return float("nan")
    return float((s < value).mean() * 100.0)


def placebo_pvalue(value: float, sample: np.ndarray) -> float:
    """Two-sided empirical p of |value| against the shuffled null (fraction at least as extreme)."""
    s = np.asarray(sample, dtype=float)
    s = s[~np.isnan(s)]
    if s.size == 0:
        return float("nan")
    return float((np.abs(s) >= abs(value)).mean())


# --------------------------------------------------------------------------- #
# Metrics & inference
# --------------------------------------------------------------------------- #
def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def summarize(monthly_ret: pd.Series, periods_per_year: int = MONTHS_PER_YEAR) -> dict:
    """Headline metrics for one monthly series (n, CAGR, Sharpe, max-dd, ann mean)."""
    r = monthly_ret.dropna()
    n = len(r)
    if n < 2:
        return {k: float("nan") for k in ("n", "cagr", "sharpe", "max_dd", "mean_ann")}
    total = float((1.0 + r).prod())
    n_years = n / periods_per_year
    cagr = total ** (1.0 / n_years) - 1.0 if (n_years > 0 and total > 0) else float("nan")
    ann_mean = r.mean() * periods_per_year
    ann_vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = ann_mean / ann_vol if ann_vol > 0 else float("nan")
    equity = (1.0 + r).cumprod().to_numpy()
    return {"n": int(n), "cagr": float(cagr), "sharpe": float(sharpe),
            "max_dd": _max_drawdown(equity), "mean_ann": float(ann_mean)}


def hac_tstat(series: pd.Series) -> dict:
    """Newey-West HAC t-stat of the mean of a monthly return series (H0: mean = 0)."""
    x = series.dropna().to_numpy(dtype=float)
    n = x.size
    if n < 3:
        return {"mean": float("nan"), "mean_ann": float("nan"), "tstat": float("nan"), "n": n}
    mu = x.mean()
    e = x - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        wk = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * wk * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return {"mean": float(mu), "mean_ann": float(mu * MONTHS_PER_YEAR),
            "tstat": float(mu / se) if se > 0 else float("nan"), "n": n}


def hac_tstat_diff(r1: pd.Series, r2: pd.Series) -> dict:
    """HAC t of the per-period spread ``r1 - r2`` (both fully invested ⇒ raw diff is the excess)."""
    a = pd.concat([r1, r2], axis=1, join="inner").dropna()
    return hac_tstat(a.iloc[:, 0] - a.iloc[:, 1])


def race(real: dict, signals: dict, frac: float = 1 / 3, cost_bps: float = 10.0,
         borrow_bps: float = 100.0, n_shuffles: int = 400, seed: int = 526) -> dict:
    """The full teardown: intangible-adjusted-B/M long-short vs SPY vs a label-shuffle placebo, plus
    the plain-B/M contrast (the Lev-Srivastava adjustment question).

    ``signals`` is the dict from :func:`data.build_signals` (keys ``bm_intan`` and ``bm_plain``).
    """
    rets, spy = real["returns"], real["spy"]
    intan_books = signal_books(signals["bm_intan"], rets, frac=frac, cost_bps=cost_bps,
                               borrow_bps=borrow_bps)
    plain_books = signal_books(signals["bm_plain"], rets, frac=frac, cost_bps=cost_bps,
                               borrow_bps=borrow_bps)
    null = placebo_null(signals["bm_intan"], rets, frac=frac, n_shuffles=n_shuffles, seed=seed)
    ls_mean_ann = intan_books["long_short"].mean() * MONTHS_PER_YEAR
    return {
        **{k: intan_books[k] for k in ("long", "short", "long_short", "long_net", "ls_net")},
        "avg_turnover": intan_books["avg_turnover"],
        "spy": spy,
        "test_ls": hac_tstat(intan_books["long_short"]),
        "test_long_vs_spy": hac_tstat_diff(intan_books["long"], spy),
        "test_short_vs_spy": hac_tstat_diff(intan_books["short"], spy),
        "test_ls_net": hac_tstat(intan_books["ls_net"]),
        # the Lev-Srivastava contrast: plain B/M
        "plain_long_short": plain_books["long_short"],
        "test_plain_ls": hac_tstat(plain_books["long_short"]),
        # head-to-head: does the adjustment ADD anything over plain B/M? (per-period spread)
        "test_intan_minus_plain": hac_tstat_diff(intan_books["long_short"],
                                                 plain_books["long_short"]),
        # placebo
        "placebo_null": null,
        "placebo_pctile": percentile_of(ls_mean_ann, null),
        "placebo_p": placebo_pvalue(ls_mean_ann, null),
    }
