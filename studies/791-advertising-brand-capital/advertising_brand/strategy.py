"""Strategy + inference for Study 791 — Advertising-Brand-Capital.

The rule under test (the believers' / academic version — Belo-Lin-Vitorino 2014;
Chan-Lakonishok-Sougiannis 2001):

    Each month, rank the field by **advertising intensity** (AdvertisingExpense / Sales).
    Hold the **top tertile equal-weight (long the heavy advertisers)** and **short the bottom
    tertile (the light advertisers)** for the next month; rebalance monthly. The long-heavy /
    short-light spread is the claimed *brand-capital premium* — compensation for holding the
    off-balance-sheet intangible the market under-prices.

We measure it honestly:

  * **One execution lag, exact.** The signal at month-end ``t`` (advertising / sales from the
    most-recent reported fiscal year) selects the book entered next month and held through
    month ``t+1``. No same-bar fill; a one-year reporting lag on the accounting on top of the
    one-month execution lag.
  * **HAC inference.** Newey-West t-stat of the monthly long-short (and long-minus-SPY, and
    heavy-minus-light Welch) spread — the Signal-axis test. ``REAL`` needs ``t >= 2`` on the
    real tape **and** survival of a label-shuffle placebo null.
  * **Placebo / label-shuffle null.** Permute the cross-sectional signal labels across names
    many times and rebuild the long-short; the real long-short must sit in the tail.
  * **Costs + borrow.** One-way turnover x NAV at ``cost_bps`` per rebalance, plus an annual
    **borrow** fee on the short leg (you pay to be short the light advertisers).

Advertising intensity is a *slow* characteristic (it moves once a year, with the reporting
lag), so monthly turnover is low — the honest cost story here is short-borrow, not churn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# Inference primitives (Welch / one-sample / Wilson) — used where relevant
# --------------------------------------------------------------------------- #
def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances). NaN if either < 2 obs."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def one_sample_t(x: np.ndarray) -> float:
    """One-sample t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
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


# --------------------------------------------------------------------------- #
# Monthly tertile membership from a cross-sectional signal panel
# --------------------------------------------------------------------------- #
def tertile_members(signal_row: pd.Series, frac: float = 1 / 3) -> tuple[list[str], list[str]]:
    """``(long, short)`` names from one month's signal row: top-``frac`` / bottom-``frac``."""
    x = signal_row.dropna()
    if len(x) < 3:
        return [], []
    k = max(1, int(round(len(x) * frac)))
    ranked = x.sort_values(ascending=False)
    return list(ranked.index[:k]), list(ranked.index[-k:])


def _tertile_idx(row: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Integer positions of the top-``frac`` / bottom-``frac`` names in one signal row.

    Mirrors :func:`tertile_members` exactly (rank the non-NaN values descending, take the
    first ``k`` and last ``k``, ``k = max(1, round(n_valid * frac))``) but on a numpy row —
    the vectorised inner loop of the sorts, so a 400-shuffle placebo runs in seconds."""
    valid = np.where(~np.isnan(row))[0]
    if valid.size < 3:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    order = valid[np.argsort(-row[valid], kind="stable")]   # descending, stable (ties = input order)
    k = max(1, int(round(valid.size * frac)))
    return order[:k], order[-k:]


def _leg_returns(signal: pd.DataFrame, returns: pd.DataFrame, frac: float,
                 which: str) -> tuple[pd.Series, pd.Series]:
    """Monthly equal-weight returns of the long (or short) leg, with a ONE-MONTH execution lag
    and one-way turnover (fraction of weight traded) per month.

    The signal as of month ``t`` selects the book that earns month ``t+1``'s return. Vectorised
    over names with numpy (identical membership/turnover to a per-name pandas loop, just fast)."""
    cols = list(signal.columns)
    ret_al = returns.reindex(columns=cols)                  # align returns to the signal's names
    S = signal.to_numpy(dtype=float)
    Rm = ret_al.to_numpy(dtype=float)
    ret_pos = {ts: i for i, ts in enumerate(returns.index)}
    sig_months = signal.index
    is_long = which == "long"
    out_ret, out_turn, out_idx = [], [], []
    prev: np.ndarray | None = None
    for i in range(len(sig_months) - 1):
        hold = sig_months[i + 1]
        hpos = ret_pos.get(hold)
        if hpos is None:
            continue
        lo_i, sh_i = _tertile_idx(S[i], frac)
        members = lo_i if is_long else sh_i
        if members.size == 0:
            prev = None
            continue
        leg = float(np.nanmean(Rm[hpos, members]))
        cur = set(members.tolist())
        if prev is not None:
            changed = len(cur ^ prev) / (2.0 * max(len(cur), 1))
        else:
            changed = 1.0
        out_ret.append(leg)
        out_turn.append(changed)
        out_idx.append(hold)
        prev = cur
    idx = pd.DatetimeIndex(out_idx, name="date")
    return (pd.Series(out_ret, index=idx, name=f"{which}_ret"),
            pd.Series(out_turn, index=idx, name=f"{which}_turn"))


def signal_books(signal: pd.DataFrame, returns: pd.DataFrame, frac: float = 1 / 3,
                 cost_bps: float = 0.0, borrow_bps: float = 0.0) -> dict:
    """Build long, short, long-short books from a monthly cross-sectional signal panel.

    Returns a dict of monthly return series (gross + net of ``cost_bps`` one-way turnover;
    ``ls_net`` also nets an annual ``borrow_bps`` on the short leg) plus the average turnover.
    """
    lr, lt = _leg_returns(signal, returns, frac, "long")
    sr, st_ = _leg_returns(signal, returns, frac, "short")
    df = pd.concat([lr, sr, lt, st_], axis=1, join="inner").dropna()
    long_g = df["long_ret"]; short_g = df["short_ret"]
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
                 n_shuffles: int = 400, seed: int = 791) -> np.ndarray:
    """Sampling distribution of the long-short under **shuffled signal labels**.

    Each draw permutes the column labels of the signal panel (same per-name signal *values*,
    wrong names), rebuilds the long-short, and records its annualised mean. The real long-short
    must sit in the tail for the signal — not the field's heterogeneity — to be doing the work.
    """
    cols = list(signal.columns)
    out = np.empty(n_shuffles)
    for d in range(n_shuffles):
        rng = np.random.default_rng(seed + d)
        perm = rng.permutation(len(cols))
        shuffled = signal.copy()
        shuffled.columns = [cols[p] for p in perm]
        shuffled = shuffled[cols]
        b = signal_books(shuffled, returns, frac=frac)
        out[d] = b["long_short"].mean() * MONTHS_PER_YEAR
    return out


def percentile_of(value: float, sample: np.ndarray) -> float:
    s = np.asarray(sample, dtype=float); s = s[~np.isnan(s)]
    if s.size == 0:
        return float("nan")
    return float((s < value).mean() * 100.0)


def placebo_pvalue(value: float, sample: np.ndarray) -> float:
    """Two-sided empirical p of |value| against the shuffled null."""
    s = np.asarray(sample, dtype=float); s = s[~np.isnan(s)]
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
    """HAC t of the per-period spread ``r1 - r2`` (both fully invested => raw diff is excess)."""
    a = pd.concat([r1, r2], axis=1, join="inner").dropna()
    return hac_tstat(a.iloc[:, 0] - a.iloc[:, 1])


def race(real: dict, signal: pd.DataFrame, frac: float = 1 / 3, cost_bps: float = 10.0,
         borrow_bps: float = 100.0, n_shuffles: int = 400, seed: int = 791) -> dict:
    """The full teardown: advertising-intensity long-short vs SPY vs a label-shuffle placebo,
    plus a hit rate (Wilson) on months the spread is positive.

    ``signal`` is the monthly advertising-intensity panel from :func:`data.build_signal`.
    """
    rets, spy = real["returns"], real["spy"]
    books = signal_books(signal, rets, frac=frac, cost_bps=cost_bps, borrow_bps=borrow_bps)
    null = placebo_null(signal, rets, frac=frac, n_shuffles=n_shuffles, seed=seed)
    ls = books["long_short"]
    ls_mean_ann = ls.mean() * MONTHS_PER_YEAR
    k_pos = int((ls > 0).sum()); n_ls = int(ls.notna().sum())
    hit_lo, hit_hi = wilson_interval(k_pos, n_ls)
    return {
        **{k: books[k] for k in ("long", "short", "long_short", "long_net", "ls_net")},
        "avg_turnover": books["avg_turnover"],
        "spy": spy,
        "test_ls": hac_tstat(books["long_short"]),
        "test_long_vs_spy": hac_tstat_diff(books["long"], spy),
        "test_short_vs_spy": hac_tstat_diff(books["short"], spy),
        "test_ls_net": hac_tstat(books["ls_net"]),
        # Welch on the pooled name-month returns of the two legs (cross-section, not time series)
        "welch_legs": welch_t(books["long"].to_numpy(), books["short"].to_numpy()),
        # hit rate of positive long-short months
        "hit": k_pos, "n_ls": n_ls, "hit_rate": (k_pos / n_ls) if n_ls else float("nan"),
        "hit_lo": hit_lo, "hit_hi": hit_hi,
        # placebo
        "placebo_null": null,
        "placebo_pctile": percentile_of(ls_mean_ann, null),
        "placebo_p": placebo_pvalue(ls_mean_ann, null),
    }
