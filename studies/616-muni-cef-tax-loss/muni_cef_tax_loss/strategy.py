"""Strategy + inference for Study 616 — Muni-CEF-Tax-Loss ("Muni CEF Tax-Loss Season").

The claim (Starks, Yong & Zheng 2006, JF): **municipal-bond closed-end funds get dumped for
tax losses in December — their discounts widen — and snap back in January**, because their
holders are almost exclusively taxable retail investors. The seasonal cousin of the CEF
discount *level* edge (study 367).

Measurement. Everything runs on the fund-minus-benchmark monthly **excess total return**
(the discount-motion proxy from ``data.monthly_excess``):

  * **Per-winter basket snap-back (primary).** For each winter ``y``, average the December
    excess across available funds (one basket December number) and the following January's
    excess (one basket January number); the snap-back is January minus December. One
    observation per winter kills the cross-fund pseudo-replication (all muni CEFs move
    together in a given January), then a one-sample *t* across winters is the primary test.
  * **Pooled fund-month Welch t (secondary).** December stock-months vs all-other months and
    January vs all-other months, quoted with the pseudo-replication caveat.
  * **Exact seasonal placebo.** The observed statistic is mean(January) - mean(December) of
    the basket excess. Enumerate ALL 132 ordered pairs of distinct calendar months (a, b) and
    compute mean(b) - mean(a); the placebo p is the share of pairs at least as large as the
    Dec→Jan pair (minimum attainable 1/132 ≈ 0.0076). Deterministic — no RNG at all.
  * **Third axis (the actionable version).** Buy the basket on the first close on/after
    **Dec-15** vs on/after **Jan-15**, both exiting at the last February close: the paired
    per-winter difference is exactly the excess earned between Dec-15 and Jan-15 — does the
    early-January snap-back outweigh the residual late-December dump?
  * **Costs.** The tradable version swaps MUB → CEF basket at the December close and back at
    the January close: 4 one-way legs per winter, charged against the January excess.

Execution: the signal is a **fixed calendar rule** (known arbitrarily in advance); every entry
and exit fills at the first close **after** the decision point (the month-boundary or the
mid-month date) — the study's single execution lag. No return earned before an entry is
counted. Winter observations are non-overlapping annual data (lag-1 autocorrelation reported),
so the plain one-sample *t* is the appropriate robust statistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Shared inference helpers
# --------------------------------------------------------------------------- #
def ttest_vs_zero(sample) -> float:
    """One-sample t of ``sample`` against 0."""
    sample = np.asarray(sample, dtype=float)
    sample = sample[~np.isnan(sample)]
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    if se == 0:
        return float("nan")
    return float(sample.mean() / se)


def welch_t(sample, base) -> float:
    """Welch t of mean(sample) - mean(base) (unequal variance)."""
    sample = np.asarray(sample, dtype=float)
    base = np.asarray(base, dtype=float)
    sample = sample[~np.isnan(sample)]
    base = base[~np.isnan(base)]
    if len(sample) < 2 or len(base) < 2:
        return float("nan")
    se = np.sqrt(sample.var(ddof=1) / len(sample) + base.var(ddof=1) / len(base))
    if se == 0:
        return float("nan")
    return float((sample.mean() - base.mean()) / se)


# --------------------------------------------------------------------------- #
# Pooled fund-month view (secondary — carries cross-fund pseudo-replication)
# --------------------------------------------------------------------------- #
def pooled_month_stats(excess: pd.DataFrame) -> dict:
    """Pool all fund-months; December vs rest, January vs rest, January vs December (Welch).

    Secondary evidence only: within a calendar month all muni CEFs move together, so the
    pooled n overstates the effective sample. The primary test is the per-winter basket.
    """
    stacked = excess.stack()
    months = stacked.index.get_level_values(0).month
    vals = stacked.values
    dec = vals[months == 12]
    jan = vals[months == 1]
    rest = vals[(months != 12) & (months != 1)]
    return {
        "dec_mean_bps": float(np.nanmean(dec) * 1e4), "n_dec": int(len(dec)),
        "jan_mean_bps": float(np.nanmean(jan) * 1e4), "n_jan": int(len(jan)),
        "rest_mean_bps": float(np.nanmean(rest) * 1e4), "n_rest": int(len(rest)),
        "welch_dec_vs_rest": welch_t(dec, rest),
        "welch_jan_vs_rest": welch_t(jan, rest),
        "welch_jan_vs_dec": welch_t(jan, dec),
    }


# --------------------------------------------------------------------------- #
# Per-winter basket (primary)
# --------------------------------------------------------------------------- #
def basket_series(excess: pd.DataFrame) -> pd.Series:
    """Equal-weight basket monthly excess (mean across available funds each month)."""
    return excess.mean(axis=1)


def winter_table(excess: pd.DataFrame, min_funds: int = 3) -> pd.DataFrame:
    """One row per winter: basket December excess, following-January excess, snap-back.

    Winter ``y`` pairs December of year ``y`` with January of year ``y+1``. Requires at least
    ``min_funds`` funds present in both months. Columns in simple returns (not bps).
    """
    b = basket_series(excess)
    n_funds = excess.notna().sum(axis=1)
    rows = []
    for ts, val in b.items():
        if ts.month != 12:
            continue
        jan_ts = ts + pd.offsets.MonthEnd(1)
        if jan_ts not in b.index:
            continue
        if n_funds.loc[ts] < min_funds or n_funds.loc[jan_ts] < min_funds:
            continue
        rows.append({"winter": ts.year, "dec": float(val), "jan": float(b.loc[jan_ts]),
                     "n_funds_dec": int(n_funds.loc[ts]), "n_funds_jan": int(n_funds.loc[jan_ts])})
    t = pd.DataFrame(rows).set_index("winter")
    t["snap"] = t["jan"] - t["dec"]
    return t


def winter_stats(wt: pd.DataFrame) -> dict:
    """Headline per-winter stats: means (bps), one-sample t's, hit rates, lag-1 autocorr."""
    snap = wt["snap"].values
    ac1 = float("nan")
    if len(snap) > 3:
        ac1 = float(np.corrcoef(snap[:-1], snap[1:])[0, 1])
    return {
        "n_winters": int(len(wt)),
        "dec_mean_bps": float(wt["dec"].mean() * 1e4), "t_dec": ttest_vs_zero(wt["dec"]),
        "jan_mean_bps": float(wt["jan"].mean() * 1e4), "t_jan": ttest_vs_zero(wt["jan"]),
        "snap_mean_bps": float(wt["snap"].mean() * 1e4), "t_snap": ttest_vs_zero(wt["snap"]),
        "hit_jan": float((wt["jan"] > 0).mean()),
        "hit_snap": float((wt["snap"] > 0).mean()),
        "snap_ac1": ac1,
    }


# --------------------------------------------------------------------------- #
# Exact seasonal placebo — all 132 ordered month pairs (deterministic, no RNG)
# --------------------------------------------------------------------------- #
def seasonal_means(excess: pd.DataFrame) -> pd.Series:
    """Basket mean excess by calendar month (1..12), in simple returns."""
    b = basket_series(excess)
    return b.groupby(b.index.month).mean()


def exact_pair_placebo(excess: pd.DataFrame) -> dict:
    """Rank the Dec→Jan contrast among ALL 132 ordered pairs of distinct calendar months.

    Statistic for pair (a, b): seasonal mean(month b) - seasonal mean(month a) of the basket
    excess. Observed = pair (12, 1). p = share of the 132 pairs with a statistic >= observed
    (the pair itself included, so the minimum attainable p is 1/132 ≈ 0.0076). Deterministic.
    """
    mu = seasonal_means(excess)
    stats = {}
    for a in range(1, 13):
        for b in range(1, 13):
            if a == b:
                continue
            stats[(a, b)] = float(mu.get(b, np.nan) - mu.get(a, np.nan))
    obs = stats[(12, 1)]
    vals = np.array(list(stats.values()))
    p = float((vals >= obs).mean())
    rank = int((vals >= obs).sum())
    return {"obs": obs, "p_value": p, "rank": rank, "n_pairs": len(vals), "pairs": stats}


def adjacent_pair_grid(excess: pd.DataFrame) -> pd.Series:
    """The 12 adjacent month-pair contrasts (Jan→Feb, ..., Dec→Jan) for the placebo chart."""
    mu = seasonal_means(excess)
    out = {}
    for a in range(1, 13):
        b = a % 12 + 1
        out[f"{a:02d}->{b:02d}"] = float(mu.get(b, np.nan) - mu.get(a, np.nan))
    return pd.Series(out)


# --------------------------------------------------------------------------- #
# Tradable January swap — MUB -> CEF basket -> MUB, with costs
# --------------------------------------------------------------------------- #
def january_swap(wt: pd.DataFrame, cost_bps: float = 10.0, n_legs: int = 4) -> dict:
    """Net January excess of the swap trade: hold MUB, swap into the basket for January.

    Enter at the last December close, exit at the last January close (fills at the close
    after each calendar decision point — the single execution lag). Each winter costs
    ``n_legs`` one-way trades (sell MUB, buy basket, sell basket, buy MUB) at ``cost_bps``
    one-way × NAV. The trade's gross P&L over just-holding-MUB is exactly the basket January
    excess, so net = January excess - n_legs × cost.
    """
    c = n_legs * cost_bps / 1e4
    net = wt["jan"] - c
    return {
        "cost_bps": cost_bps, "n_legs": n_legs, "drag_bps": float(c * 1e4),
        "gross_mean_bps": float(wt["jan"].mean() * 1e4),
        "net_mean_bps": float(net.mean() * 1e4),
        "t_net": ttest_vs_zero(net),
        "hit_net": float((net > 0).mean()),
        "n_winters": int(len(wt)),
    }


# --------------------------------------------------------------------------- #
# Third axis — Dec-15 entry vs Jan-15 entry (the actionable version)
# --------------------------------------------------------------------------- #
def _first_close_on_or_after(idx: pd.DatetimeIndex, year: int, month: int, day: int):
    """First trading day on/after (year, month, day); None if beyond the tape."""
    target = pd.Timestamp(year=year, month=month, day=day)
    pos = idx.searchsorted(target)
    if pos >= len(idx):
        return None
    ts = idx[pos]
    # guard: the fill must stay in the intended month-window (never drift a month late)
    if (ts - target).days > 10:
        return None
    return ts


def dec15_vs_jan15(prices: pd.DataFrame, bench: str, funds: list[str],
                   min_funds: int = 3) -> pd.DataFrame:
    """Per-winter basket excess of a Dec-15 entry vs a Jan-15 entry, common exit = last Feb close.

    For each winter: entry A = first close on/after Dec-15, entry B = first close on/after
    Jan-15, exit = last February close. For each fund present at all three dates, the window
    excess is (fund TR over the window) minus (benchmark TR over the same window); basket =
    equal-weight mean. ``diff = winA - winB`` is the paired, per-winter answer to "did buying
    a month earlier pay?" — economically the excess earned between Dec-15 and Jan-15.
    """
    idx = prices.index
    years = sorted({ts.year for ts in idx})
    rows = []
    for y in years:
        e0 = _first_close_on_or_after(idx, y, 12, 15)
        e1 = _first_close_on_or_after(idx, y + 1, 1, 15)
        try:
            feb = prices.loc[str(y + 1) + "-02"]
        except KeyError:
            continue
        if e0 is None or e1 is None or len(feb) == 0:
            continue
        ex = feb.index[-1]
        # exit must be a genuine end-of-February close (tape must reach Feb 24+)
        if ex.day < 24:
            continue
        wa, wb = [], []
        b0, b1, bx = prices.at[e0, bench], prices.at[e1, bench], prices.at[ex, bench]
        if np.isnan(b0) or np.isnan(b1) or np.isnan(bx):
            continue
        for f in funds:
            if f not in prices.columns:
                continue
            p0, p1, px = prices.at[e0, f], prices.at[e1, f], prices.at[ex, f]
            if np.isnan(p0) or np.isnan(p1) or np.isnan(px):
                continue
            wa.append((px / p0 - 1.0) - (bx / b0 - 1.0))
            wb.append((px / p1 - 1.0) - (bx / b1 - 1.0))
        if len(wa) < min_funds:
            continue
        rows.append({"winter": y, "winA_dec15": float(np.mean(wa)),
                     "winB_jan15": float(np.mean(wb)),
                     "diff": float(np.mean(wa) - np.mean(wb)), "n_funds": len(wa)})
    return pd.DataFrame(rows).set_index("winter")


def dec15_stats(dj: pd.DataFrame) -> dict:
    return {
        "n_winters": int(len(dj)),
        "dec15_mean_bps": float(dj["winA_dec15"].mean() * 1e4),
        "jan15_mean_bps": float(dj["winB_jan15"].mean() * 1e4),
        "diff_mean_bps": float(dj["diff"].mean() * 1e4),
        "t_diff": ttest_vs_zero(dj["diff"]),
        "hit_diff": float((dj["diff"] > 0).mean()),
    }
