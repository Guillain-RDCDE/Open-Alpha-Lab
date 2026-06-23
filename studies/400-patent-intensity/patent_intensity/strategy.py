"""Strategy + inference for Study 400 — Patent-Intensity (the "innovation premium").

The rule under test (the believers' version):

    Each year, rank the field by **R&D intensity** (R&D expense / revenue, the proxy for
    patent-and-invention intensity). Hold the **top tertile equal-weight (long the innovators)**
    and **short the bottom tertile (the R&D-light dinosaurs)** for the next 12 months; rebalance
    annually. The long-high / short-low spread is the claimed *innovation premium*.

We measure it honestly:

  * **One reporting lag, exact.** A fiscal-year intensity is only *known* after the 10-K is filed.
    We form the year-Y portfolio from intensity reported for fiscal year **Y-1** (a name's most
    recent fully-reported intensity), and the book earns year-Y returns. No look-ahead.
  * **HAC inference.** Newey-West t-stat of the monthly long-short (and long-minus-SPY) spread —
    the Signal-axis test. ``REAL`` needs ``t >= 2`` on the real tape.
  * **Sector/size confound control.** The decisive critique: R&D-intensive ≈ tech/pharma/semis,
    R&D-light ≈ banks/staples/energy. We race the intensity long-short against a *random*
    long-short of the same field (same leg sizes, names drawn blindly) to see how much of the
    spread is "this particular split" vs any split of an already-heterogeneous field, and we
    report the long leg's spread over the *equal-weight field* (strip "the basket went up").
  * **Costs + borrow.** One-way turnover × NAV at ``cost_bps`` per rebalance, plus an annual
    **borrow** fee on the short leg (you pay to be short the R&D-light names).

The decisive question is not whether the long leg made money (in a bull tape it did) but whether
the long-MINUS-short *innovation spread* is statistically real and survives being a sector tilt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# Annual intensity ranking -> yearly long / short membership (with reporting lag)
# --------------------------------------------------------------------------- #
def known_intensity(intensity: pd.DataFrame, year: int, report_lag: int = 1) -> pd.Series:
    """The R&D intensity *known* at the start of ``year`` — the most recent fiscal year on or
    before ``year - report_lag`` for which each name has data (forward-filled within the lag).

    ``report_lag = 1`` ⇒ the year-Y book uses fiscal-year-(Y-1) intensity, the conservative
    convention (a 10-K for FY Y-1 is filed early in year Y; using FY Y-1 has no look-ahead)."""
    avail = intensity[intensity.index <= year - report_lag]
    if avail.empty:
        return pd.Series(dtype=float)
    return avail.ffill().iloc[-1].dropna()


def tertile_members(intensity: pd.DataFrame, year: int, frac: float = 1 / 3,
                    report_lag: int = 1) -> tuple[list[str], list[str]]:
    """``(long, short)`` names for ``year``: top-``frac`` and bottom-``frac`` by known intensity."""
    x = known_intensity(intensity, year, report_lag=report_lag)
    if len(x) < 3:
        return [], []
    k = max(1, int(round(len(x) * frac)))
    ranked = x.sort_values(ascending=False)
    return list(ranked.index[:k]), list(ranked.index[-k:])


# --------------------------------------------------------------------------- #
# Book construction — yearly membership, monthly compounding, costs + borrow
# --------------------------------------------------------------------------- #
def _equal_weight_book(returns: pd.DataFrame, yearly_members: dict[int, list[str]],
                       cost_bps: float = 0.0) -> pd.Series:
    """Monthly returns of an equal-weight book whose membership is set once per calendar year.

    ``yearly_members[y]`` is the list held through calendar year ``y``. Turnover at each January
    rebalance is ``sum |w_new - w_drifted|`` (one-way), charged at ``cost_bps``. Empty membership
    ⇒ flat (0) that year.
    """
    cols = list(returns.columns)
    pos = {c: i for i, c in enumerate(cols)}
    R = returns.to_numpy(dtype=float)
    out = np.zeros(len(returns))
    w = np.zeros(len(cols))
    prev_year = None
    for t, ts in enumerate(returns.index):
        y = ts.year
        if y != prev_year:                              # annual rebalance (January)
            members = [m for m in yearly_members.get(y, []) if m in pos]
            target = np.zeros(len(cols))
            if members:
                for m in members:
                    target[pos[m]] = 1.0 / len(members)
            cost = float(np.abs(target - w).sum()) * cost_bps * 1e-4
            w = target
            out[t] = float(np.nansum(w * np.nan_to_num(R[t]))) - cost
            prev_year = y
        else:
            out[t] = float(np.nansum(w * np.nan_to_num(R[t])))
        grown = w * (1.0 + np.nan_to_num(R[t]))
        s = grown.sum()
        if s > 0:
            w = grown / s
    return pd.Series(out, index=returns.index)


def intensity_books(intensity: pd.DataFrame, returns: pd.DataFrame, frac: float = 1 / 3,
                    report_lag: int = 1, cost_bps: float = 0.0,
                    borrow_bps: float = 0.0) -> dict:
    """Build the long, short and long-short books from the intensity ranking.

    Returns a dict of monthly return series:
      * ``long``        — top-tertile R&D-intensity, equal-weight, annually rebalanced (gross).
      * ``short``       — bottom-tertile (the leg you *short*), as a long series (gross).
      * ``long_short``  — ``long - short`` (the innovation spread), gross.
      * ``long_net`` / ``ls_net`` — net of ``cost_bps`` turnover; ``ls_net`` also nets an annual
        ``borrow_bps`` on the short leg (≈ ``borrow_bps/1e4`` per year spread over 12 months).
      * ``members``     — ``{year: (long, short)}`` actually held.
    """
    years = sorted({ts.year for ts in returns.index})
    long_m, short_m, members = {}, {}, {}
    for y in years:
        lo, sh = tertile_members(intensity, y, frac=frac, report_lag=report_lag)
        long_m[y], short_m[y], members[y] = lo, sh, (lo, sh)

    long_g = _equal_weight_book(returns, long_m, cost_bps=0.0)
    short_g = _equal_weight_book(returns, short_m, cost_bps=0.0)
    long_c = _equal_weight_book(returns, long_m, cost_bps=cost_bps)
    short_c = _equal_weight_book(returns, short_m, cost_bps=cost_bps)

    borrow_monthly = borrow_bps * 1e-4 / MONTHS_PER_YEAR
    ls_g = long_g - short_g
    ls_n = long_c - short_c - borrow_monthly
    return {
        "long": long_g, "short": short_g, "long_short": ls_g,
        "long_net": long_c, "ls_net": ls_n, "members": members,
    }


def random_long_short(returns: pd.DataFrame, k_long: int, k_short: int, n_draws: int = 2000,
                      seed: int = 400) -> np.ndarray:
    """Sampling distribution of a *blind* long/short spread: per draw, pick ``k_long`` names long
    and ``k_short`` other names short (held the whole sample), return each draw's annualised mean
    spread. This is the sector/heterogeneity control — how big a spread does *any* split of this
    already-mixed field produce, with no intensity information used?
    """
    cols = list(returns.columns)
    R = returns.to_numpy(dtype=float)
    out = np.empty(n_draws)
    for d in range(n_draws):
        rng = np.random.default_rng(seed + d)
        pick = rng.permutation(len(cols))
        li, si = pick[:k_long], pick[k_long:k_long + k_short]
        lr = np.nanmean(R[:, li], axis=1)
        sr = np.nanmean(R[:, si], axis=1)
        out[d] = np.nanmean(lr - sr) * MONTHS_PER_YEAR
    return out


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
    """Newey-West HAC t-stat of the mean of a monthly return series (H0: mean = 0).

    Used on the long-short spread and on long-minus-SPY (the Signal-axis tests). Returns the mean
    monthly value, its annualised value, the HAC t-stat, and n."""
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


def percentile_of(value: float, sample: np.ndarray) -> float:
    s = np.asarray(sample, dtype=float)
    s = s[~np.isnan(s)]
    if s.size == 0:
        return float("nan")
    return float((s < value).mean() * 100.0)


def race(intensity: pd.DataFrame, returns: pd.DataFrame, spy: pd.Series, frac: float = 1 / 3,
         report_lag: int = 1, cost_bps: float = 10.0, borrow_bps: float = 100.0,
         n_draws: int = 2000, seed: int = 400) -> dict:
    """The full teardown: intensity long/short vs SPY vs a blind random long/short.

    Returns a bundle with the books, the SPY series, the HAC tests of the innovation spread and of
    long-minus-SPY, the net (cost + borrow) numbers, and where the intensity spread sits in the
    distribution of random blind long/shorts of the same field (the sector/heterogeneity control).
    """
    books = intensity_books(intensity, returns, frac=frac, report_lag=report_lag,
                            cost_bps=cost_bps, borrow_bps=borrow_bps)
    # leg sizes (use the last year's membership counts; constant across years here)
    last = max(books["members"])
    k_long = max(1, len(books["members"][last][0]))
    k_short = max(1, len(books["members"][last][1]))
    rand = random_long_short(returns, k_long, k_short, n_draws=n_draws, seed=seed)

    ls_mean_ann = hac_tstat(books["long_short"])["mean_ann"]
    out = {
        **{k: books[k] for k in ("long", "short", "long_short", "long_net", "ls_net")},
        "spy": spy,
        "test_ls": hac_tstat(books["long_short"]),
        "test_long_vs_spy": hac_tstat_diff(books["long"], spy),
        "test_short_vs_spy": hac_tstat_diff(books["short"], spy),
        "rand_ls_spread": rand,
        "ls_pctile": percentile_of(ls_mean_ann, rand),
        "members": books["members"],
    }
    return out
