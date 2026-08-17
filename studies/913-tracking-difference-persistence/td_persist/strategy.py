"""Strategy + inference for Study 913 — Tracking-Difference Persistence.

The claim: index funds tracking the *same* index differ in **tracking difference** (TD),
the annual gap between the fund's total return and the index's, and that gap is
*persistent* — so last calendar year's best tracker is a good bet for next year, and a
fund-picker can harvest a few basis points by rotating into it.

How we test it:

1. **The measurement.** For each family (S&P 500 trackers; the Nasdaq-100 pair) we compute
   each fund's **complete-calendar-year total return** from adjusted closes, then take the
   TD as the fund's return minus a common index proxy. The default proxy is the
   *cross-sectional mean of the family* (equivalently: the family leader, or any other
   common reference — the **ranking is invariant to the choice**, since a common constant
   shifts every fund equally). ^GSPC is available as an alternative proxy but is
   **price-only**, so it identifies the dividend yield, not the TD level: see
   ``data.price_only_note``.

2. **The persistence test.** Pooled year-over-year Spearman rank correlation of TD across
   consecutive calendar years, with a *t* on the mean of the per-pair rank correlations and
   a **year-label permutation** null. Repeated on fund-demeaned ("residual") TD, which
   strips out the constant fee ladder and asks whether anything *beyond fees* persists.

3. **The money question.** Four annually-rebalanced rules on the same family, each holding
   one fund (or an equal split of the tied cheapest):
   ``winner`` (last year's top TD), ``loser`` (last year's bottom), ``cheapest`` (the lowest
   published expense ratio, which never *switches* fund), ``leader`` (the family's flagship,
   likewise), plus ``eqw``. Exactly **one execution lag**: the ranking uses returns through
   the last close of year ``y`` and the new weights are in force from the *second* trading
   day of year ``y+1`` (a one-trading-day shift of the daily weight vector). Costs are
   one-way x NAV on realised turnover; there is no short leg, so no borrow. A
   **capital-gains tax break-even** is computed separately as a labelled ASSUMPTION, because
   switching funds in a taxable account realises the embedded gain.

   Two disclosures on those rules, both audited and both small but real:

   - **The `cheapest` rule reads TODAY's fee sheet.** ``data.EXPENSE_RATIO_BPS`` holds the
     ratios published at build time and applies them to the whole sample, so the fund the
     rule holds is chosen with information the early years did not have (FXAIX's 1.5 bp
     dates from 2019, SWPPX's 2 bp from 2017). That is **hindsight in the selection**, and
     it is priced rather than argued away: ``per_fund_gap_vs_leader`` re-runs the same
     question with no fee sheet at all.
   - **A tied `cheapest` (or `eqw`) is held at target weight**, i.e. implicitly rebalanced
     daily at zero cost, because the weight vector is a constant rather than a drifting
     book. Measured on the real trio this diversification freebie is **+0.06 bp/yr** —
     immaterial at this resolution, but it is a freebie, and only the single-fund arms are
     literally buy-and-hold.

Two units of measurement are reported for every gap, and they disagree in an instructive
way. The **daily** HAC *t* is the conservative one but is swamped by same-day
cross-fund print noise; the **annual** paired *t* (n = one observation per complete
calendar year) is the natural unit for a quantity that is defined annually. Both are shown.

Returns are **simple** (arithmetic) so a one-fund portfolio return is exactly the fund's
return net of any switch cost, and wealth compounds correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MIN_DAYS_FOR_COMPLETE_YEAR = 200   # drops partial first/last calendar years


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def newey_west_t(x, lags: int | None = None) -> float:
    """HAC (Newey-West, Bartlett kernel) *t* of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    u = x - x.mean()
    var = float(u @ u) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        var += 2.0 * w * float(u[lag:] @ u[:-lag]) / n
    if var <= 0:
        return float("nan")
    return float(x.mean() / np.sqrt(var / n))


def spearman(a, b) -> float:
    """Spearman rank correlation (Pearson on ranks; average ranks for ties). No scipy."""
    sa = pd.Series(np.asarray(a, dtype=float)).rank()
    sb = pd.Series(np.asarray(b, dtype=float)).rank()
    if sa.std(ddof=0) == 0 or sb.std(ddof=0) == 0:
        return float("nan")
    return float(np.corrcoef(sa.to_numpy(), sb.to_numpy())[0, 1])


def block_bootstrap_ci(x, n_boot: int = 5000, block: int = 2, seed: int = 913,
                       alpha: float = 0.05) -> dict:
    """Circular block-bootstrap CI for the mean of an annual series (blocks of years).

    A block of 2 consecutive years is the mildest defence against the fact that a fund's
    fee advantage is not an independent draw each year — the whole point of the study.
    """
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = x.size
    if n < 3:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": int(n)}
    block = max(1, min(int(block), n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    draws = x[idx.reshape(n_boot, -1)[:, :n]]
    means = draws.mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(x.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((means < 0).mean()), "n_obs": int(n),
            "n_boot": int(n_boot), "block": int(block)}


# --------------------------------------------------------------------------- #
# The measurement: complete-calendar-year total returns and tracking differences
# --------------------------------------------------------------------------- #
def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Aligned daily simple total returns for a panel of funds (inner-joined dates)."""
    px = prices.dropna(how="any").sort_index()
    return px.pct_change().dropna(how="any")


def complete_years(returns: pd.DataFrame, min_days: int = MIN_DAYS_FOR_COMPLETE_YEAR) -> list:
    """Calendar years with at least ``min_days`` observations — partial years are dropped.

    This is the as-of discipline at the calendar-year level: a fund listed in October, or a
    sample that ends on 30 June, must not contribute a stub "year" to a persistence test.
    """
    counts = pd.Series(1, index=returns.index).groupby(returns.index.year).sum()
    return [int(y) for y in counts.index if counts.loc[y] >= min_days]


def annual_returns(prices: pd.DataFrame, min_days: int = MIN_DAYS_FOR_COMPLETE_YEAR) -> pd.DataFrame:
    """Complete-calendar-year compounded total returns, one row per year, one column per fund."""
    r = daily_returns(prices)
    yrs = complete_years(r, min_days=min_days)
    r = r[np.isin(r.index.year, yrs)]
    ann = (1.0 + r).groupby(r.index.year).prod() - 1.0
    ann.index.name = "year"
    return ann


def tracking_difference(annual: pd.DataFrame, proxy: str = "mean") -> pd.DataFrame:
    """Annual tracking difference in **basis points**, one row per year, one per fund.

    ``proxy`` selects the common index reference subtracted from every fund in a year:

    - ``"mean"`` (default) — the cross-sectional mean of the family, i.e. a *relative* TD
      that sums to zero each year. Symmetric and estimator-free.
    - ``"leader"`` — the family's flagship (the first column), which then carries TD == 0.
    - any column name — that fund as the reference.

    Ranks are **identical** under every choice, because a common per-year constant shifts
    all funds equally; only the reported *level* moves. The level against a real, dividend-
    paying index is not identified from a public price tape (see ``data.price_only_note``).
    """
    if proxy == "mean":
        ref = annual.mean(axis=1)
    elif proxy == "leader":
        ref = annual.iloc[:, 0]
    else:
        ref = annual[proxy]
    return annual.sub(ref, axis=0) * 1e4


# --------------------------------------------------------------------------- #
# The persistence test
# --------------------------------------------------------------------------- #
def _pairwise_rank_corr_matrix(td: pd.DataFrame) -> np.ndarray:
    """All-pairs Spearman between the calendar years of a TD frame (years x years).

    Precomputed once so the permutation null costs a table lookup per draw rather than a
    fresh rank correlation. For a two-fund family the entry is the sign-agreement
    indicator (+1 / -1), which is what Spearman degenerates to there.
    """
    if td.shape[1] == 2:
        s = np.sign(td.iloc[:, 0].to_numpy())
        return np.outer(s, s)
    ranks = td.rank(axis=1).to_numpy(dtype=float)
    ranks = ranks - ranks.mean(axis=1, keepdims=True)
    norm = np.sqrt((ranks ** 2).sum(axis=1, keepdims=True))
    norm[norm == 0] = np.nan
    z = ranks / norm
    return z @ z.T


def yoy_rank_correlations(td: pd.DataFrame) -> pd.Series:
    """Spearman rank correlation of TD in year *y* against TD in year *y+1*.

    With a two-fund family the Spearman is degenerate (+1 or -1) and the series is simply
    a sign-agreement indicator — reported honestly as such rather than dressed up.
    """
    rows = {}
    years = list(td.index)
    for i, y in enumerate(years[:-1]):
        nxt = years[i + 1]
        if nxt != y + 1:
            continue          # never chain across a dropped (partial) year
        a, b = td.loc[y], td.loc[nxt]
        if td.shape[1] == 2:
            rows[nxt] = 1.0 if np.sign(a.iloc[0]) == np.sign(b.iloc[0]) else -1.0
        else:
            rows[nxt] = spearman(a.to_numpy(), b.to_numpy())
    return pd.Series(rows, name="yoy_spearman").astype(float)


def persistence(td: pd.DataFrame, n_perm: int = 5000, seed: int = 913) -> dict:
    """Pooled year-over-year rank persistence of tracking difference.

    Returns the mean year-over-year Spearman, its *t* against zero, a **year-label
    permutation** p-value (shuffle the order of the years and recompute — this preserves
    each year's cross-sectional shape and destroys only the time linkage), the number of
    pairs, and the same statistics for **residual** TD (each fund demeaned over the sample,
    which removes the constant fee ladder). If residual persistence disappears while raw
    persistence survives, the "persistence" is nothing but the published fee ordering.
    """
    raw = yoy_rank_correlations(td)
    res_td = td.sub(td.mean(axis=0), axis=1)
    res = yoy_rank_correlations(res_td)

    # Permutation null: shuffle the ORDER of the calendar years (each year keeps its own
    # cross-sectional shape; only the time linkage is destroyed) and recompute the mean
    # consecutive-pair rank correlation.
    rng = np.random.default_rng(seed)
    corr = _pairwise_rank_corr_matrix(td)
    n_years = corr.shape[0]
    perm_means = np.full(n_perm, np.nan)
    if n_years >= 3:
        for b in range(n_perm):
            order = rng.permutation(n_years)
            perm_means[b] = np.nanmean(corr[order[:-1], order[1:]])
    obs = float(raw.mean())
    valid = perm_means[np.isfinite(perm_means)]
    p_perm = float((np.abs(valid) >= abs(obs)).mean()) if valid.size else float("nan")

    return {
        "mean_spearman": obs,
        "t_spearman": one_sample_t(raw.to_numpy()),
        "n_pairs": int(raw.size),
        "p_permutation": p_perm,
        "mean_spearman_residual": float(res.mean()),
        "t_spearman_residual": one_sample_t(res.to_numpy()),
        "degenerate_two_fund": bool(td.shape[1] == 2),
        "yoy": raw,
        "yoy_residual": res,
    }


# --------------------------------------------------------------------------- #
# The rules
# --------------------------------------------------------------------------- #
RULES = ("winner", "loser", "cheapest", "leader", "eqw")


def _cheapest_weights(funds, expense_ratio_bps: dict) -> pd.Series:
    """Equal weight across the funds tied at the minimum published expense ratio."""
    er = {f: float(expense_ratio_bps[f]) for f in funds}
    lo = min(er.values())
    tied = [f for f in funds if er[f] == lo]
    return pd.Series([1.0 / len(tied) if f in tied else 0.0 for f in funds], index=list(funds))


def rule_weights(td: pd.DataFrame, expense_ratio_bps: dict, rule: str) -> pd.DataFrame:
    """Target weights per *year* for one rule (rows = years the rule is live, cols = funds).

    ``winner``/``loser`` use year ``y-1``'s TD to set year ``y``'s holding, so the first
    complete year of the sample is consumed by the ranking and is not live. The static
    rules (``cheapest``, ``leader``, ``eqw``) are given the same live years so every arm is
    raced over an identical window.
    """
    funds = list(td.columns)
    years = list(td.index)
    live = [y for i, y in enumerate(years) if i > 0 and years[i - 1] == y - 1]
    rows = {}
    for y in live:
        prev = td.loc[y - 1]
        if rule == "winner":
            w = pd.Series(0.0, index=funds); w[prev.idxmax()] = 1.0
        elif rule == "loser":
            w = pd.Series(0.0, index=funds); w[prev.idxmin()] = 1.0
        elif rule == "cheapest":
            w = _cheapest_weights(funds, expense_ratio_bps)
        elif rule == "leader":
            w = pd.Series(0.0, index=funds); w[funds[0]] = 1.0
        elif rule == "eqw":
            w = pd.Series(1.0 / len(funds), index=funds)
        else:
            raise ValueError(f"unknown rule {rule!r}")
        rows[y] = w
    return pd.DataFrame(rows).T.reindex(columns=funds)


def run_rules(prices: pd.DataFrame, expense_ratio_bps: dict, cost_bps: float = 1.0,
              min_days: int = MIN_DAYS_FOR_COMPLETE_YEAR) -> pd.DataFrame:
    """Daily net simple returns of every rule, on a common live window.

    Exactly **one execution lag**: the annual weight vector is broadcast to the days of its
    calendar year and then shifted forward one trading day, so the ranking formed at the
    last close of year ``y`` first earns money on the *second* trading day of year ``y+1``.
    Turnover is charged at ``cost_bps`` one-way x NAV on each leg of a switch (a full
    rotation therefore pays twice), on the day the weights actually change. No short leg,
    so no borrow. Mutual-fund legs trade at NAV; the cost sweep covers the case where a
    switch costs more than a penny ETF spread.

    Weights are **targets**, so a multi-fund arm (a tied ``cheapest``, or ``eqw``) is held at
    constant weight rather than allowed to drift — an implicit daily rebalance whose cost is
    not charged. Between funds tracking the same index that freebie measures +0.06 bp/yr on
    the real trio; the single-fund arms are unaffected.
    """
    r = daily_returns(prices)
    ann = annual_returns(prices, min_days=min_days)
    td = tracking_difference(ann, proxy="mean")

    out = {}
    live_index = None
    for rule in RULES:
        wy = rule_weights(td, expense_ratio_bps, rule)
        live_years = list(wy.index)
        mask = np.isin(r.index.year, live_years)
        rr = r[mask]
        w = pd.DataFrame(
            wy.reindex(rr.index.year).to_numpy(), index=rr.index, columns=rr.columns
        )
        w_held = w.shift(1).dropna(how="any")             # the single execution lag
        gross = (w_held * rr.reindex(w_held.index)).sum(axis=1)
        turnover = w_held.diff().abs().sum(axis=1).fillna(0.0)
        out[rule] = gross - turnover * cost_bps * 1e-4
        live_index = w_held.index if live_index is None else live_index
    frame = pd.DataFrame(out).dropna(how="any")
    frame.index.name = "date"
    return frame


def switch_count(td: pd.DataFrame, expense_ratio_bps: dict, rule: str = "winner") -> int:
    """How many times the rule changes fund over the live years (turnover in trades)."""
    wy = rule_weights(td, expense_ratio_bps, rule)
    if len(wy) < 2:
        return 0
    changed = wy.diff().abs().sum(axis=1).iloc[1:]
    return int((changed > 1e-12).sum())


# --------------------------------------------------------------------------- #
# The gaps, in both units
# --------------------------------------------------------------------------- #
def annual_rule_returns(prices: pd.DataFrame, expense_ratio_bps: dict,
                        cost_bps: float = 1.0) -> pd.DataFrame:
    """Per-calendar-year compounded return of each rule, in basis points."""
    daily = run_rules(prices, expense_ratio_bps, cost_bps=cost_bps)
    ann = (1.0 + daily).groupby(daily.index.year).prod() - 1.0
    ann.index.name = "year"
    return ann * 1e4


def gap_stats(prices: pd.DataFrame, expense_ratio_bps: dict, arm_a: str, arm_b: str,
              cost_bps: float = 1.0) -> dict:
    """The ``arm_a - arm_b`` gap in bp/yr, tested in the daily *and* the annual unit.

    - **daily**: HAC (Newey-West) *t* on the daily return difference, annualised mean.
      Conservative, and deliberately so — but the daily cross-fund difference is dominated
      by same-day print and dividend-timing noise that reverses within days, so this test
      has very little power against a quantity defined once a year.
    - **annual**: paired *t* on the per-calendar-year gap (n = live years), plus a HAC
      (Newey-West, one annual lag) *t* on the same series, a block bootstrap CI and the
      count of positive years. This is the natural unit. Calendar-year returns do **not**
      overlap, so the i.i.d. *t* is not mechanically inflated the way an overlapping-window
      *t* would be; the HAC version is reported anyway because a fee gap is a persistent
      quantity and the reader should not have to take independence on trust.
    """
    daily = run_rules(prices, expense_ratio_bps, cost_bps=cost_bps)
    d = (daily[arm_a] - daily[arm_b]).dropna()
    ann = annual_rule_returns(prices, expense_ratio_bps, cost_bps=cost_bps)
    a = (ann[arm_a] - ann[arm_b]).dropna()
    boot = block_bootstrap_ci(a.to_numpy())
    return {
        "arm_a": arm_a, "arm_b": arm_b, "cost_bps": float(cost_bps),
        "daily_gap_bp_yr": float(d.mean() * TRADING_DAYS * 1e4),
        "t_daily_hac": newey_west_t(d.to_numpy()),
        "n_days": int(d.size),
        "annual_gap_bp_yr": float(a.mean()),
        "t_annual": one_sample_t(a.to_numpy()),
        "t_annual_hac": newey_west_t(a.to_numpy(), lags=1),
        "n_years": int(a.size),
        "years_positive": int((a > 0).sum()),
        "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
        "annual_series": a,
    }


def per_fund_gap_vs_leader(prices: pd.DataFrame,
                           min_days: int = MIN_DAYS_FOR_COMPLETE_YEAR) -> pd.DataFrame:
    """Every non-flagship fund's annual gap vs the leader — the **hindsight-free** check.

    The ``cheapest`` arm picks its fund from the fee sheet published *today* and applies it
    to the whole sample (see the ASSUMPTION block in ``data.EXPENSE_RATIO_BPS``): FXAIX has
    charged 1.5 bp only since 2019, SWPPX 2 bp only since 2017. An investor standing in 2012
    could not have known which rung of the low-fee ladder would end up lowest, so the
    headline ``cheapest - leader`` number is contaminated by selection hindsight even though
    nothing in the *return* series looks ahead.

    This function removes the selection: it reports the gap of **each** non-leader fund
    against the flagship, one row each, plus ``EQW_non_flagship`` — the equal-weight blend of
    all of them, a rule that requires no fee sheet, no ranking and no forecast. If the gap
    survives there, the effect is not an artefact of picking the ex-post cheapest fund; the
    spread between the rows shows how much of the headline *is* the pick.

    No costs are charged here: every arm is a buy-and-hold of a fund that already exists at
    the start of the window, so realised turnover is zero by construction (the EQW row is
    rebalanced to target, which on near-identical funds is worth well under 0.1 bp/yr).
    """
    ann = annual_returns(prices, min_days=min_days) * 1e4
    leader = ann.columns[0]
    gaps = ann.drop(columns=[leader]).sub(ann[leader], axis=0)
    rows = []
    series = [(f, gaps[f].dropna()) for f in gaps.columns]
    series.append(("EQW_non_flagship", gaps.mean(axis=1).dropna()))
    for name, g in series:
        boot = block_bootstrap_ci(g.to_numpy())
        rows.append({
            "vs_leader": f"{name}-{leader}",
            "annual_gap_bp_yr": float(g.mean()),
            "t_annual": one_sample_t(g.to_numpy()),
            "t_annual_hac": newey_west_t(g.to_numpy(), lags=1),
            "years_positive": int((g > 0).sum()),
            "n_years": int(g.size),
            "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
        })
    return pd.DataFrame(rows).set_index("vs_leader")


def excess_of_cash_race(prices: pd.DataFrame, cash: pd.Series, expense_ratio_bps: dict,
                        cost_bps: float = 1.0) -> pd.DataFrame:
    """Excess-of-cash Sharpe of every rule, both arms measured excess-of-cash.

    Every fund here tracks the same index, so the arms share ~99.99% of their variance and
    the Sharpe column is essentially the index's Sharpe repeated — which is exactly the
    point worth showing: a tracking-difference edge is invisible in a Sharpe race and can
    only be read as a return *gap*.
    """
    daily = run_rules(prices, expense_ratio_bps, cost_bps=cost_bps)
    rc = cash.reindex(daily.index).pct_change()
    rows = {}
    for rule in daily.columns:
        e = (daily[rule] - rc).dropna()
        sd = e.std(ddof=1)
        rows[rule] = {
            "excess_sharpe": float(e.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
            "cagr": float((1.0 + daily[rule]).prod() ** (TRADING_DAYS / len(daily)) - 1.0),
            "vol_ann": float(daily[rule].std(ddof=1) * np.sqrt(TRADING_DAYS)),
            "n_days": int(len(daily)),
        }
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Robustness: eras, costs, and the tax break-even (an ASSUMPTION, swept)
# --------------------------------------------------------------------------- #
def era_cut(prices: pd.DataFrame, expense_ratio_bps: dict, arm_a: str, arm_b: str,
            cost_bps: float = 1.0, split_year: int | None = None) -> dict:
    """Split the live years in two halves and re-run the annual gap test on each."""
    ann = annual_rule_returns(prices, expense_ratio_bps, cost_bps=cost_bps)
    a = (ann[arm_a] - ann[arm_b]).dropna()
    years = list(a.index)
    if not years:
        return {}
    if split_year is None:
        split_year = int(years[len(years) // 2])
    out = {}
    for tag, sel in [("early", a.index < split_year), ("late", a.index >= split_year)]:
        x = a[sel]
        out[tag] = {
            "years": f"{int(x.index.min())}-{int(x.index.max())}" if len(x) else "-",
            "n_years": int(len(x)),
            "annual_gap_bp_yr": float(x.mean()) if len(x) else float("nan"),
            "t_annual": one_sample_t(x.to_numpy()) if len(x) > 1 else float("nan"),
            "years_positive": int((x > 0).sum()),
        }
    out["split_year"] = int(split_year)
    return out


def cost_sweep(prices: pd.DataFrame, expense_ratio_bps: dict, arm_a: str, arm_b: str,
               grid=(0.0, 1.0, 5.0, 10.0, 25.0)) -> pd.DataFrame:
    """The gap at several one-way switching costs (bp x NAV charged on realised turnover)."""
    rows = []
    for c in grid:
        g = gap_stats(prices, expense_ratio_bps, arm_a, arm_b, cost_bps=c)
        rows.append({"cost_bps": c, "annual_gap_bp_yr": g["annual_gap_bp_yr"],
                     "t_annual": g["t_annual"], "t_daily_hac": g["t_daily_hac"],
                     "years_positive": g["years_positive"], "n_years": g["n_years"]})
    return pd.DataFrame(rows).set_index("cost_bps")


def tax_breakeven_years(gap_bp_yr: float, embedded_gain_pct=(10.0, 25.0, 50.0, 100.0),
                        tax_rate=(0.0, 0.15, 0.20, 0.238)) -> pd.DataFrame:
    """Years of gap needed to repay the tax on a switch — an ASSUMPTION sweep, not a measurement.

    Neither the embedded gain nor the investor's marginal rate is on the tape. This grid
    makes the arithmetic explicit: switching an appreciated holding realises
    ``gain x rate`` immediately, and the fee gap repays it at ``gap_bp_yr`` a year. The
    0% row is the tax-sheltered account, where the switch is genuinely free.
    """
    rows = []
    for g in embedded_gain_pct:
        row = {"embedded_gain_pct": g}
        for t in tax_rate:
            cost_bp = g * 1e2 * t          # % gain -> bp, times the rate
            row[f"tax_{int(round(t * 1000)):03d}"] = (
                float("inf") if gap_bp_yr <= 0 else cost_bp / gap_bp_yr
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("embedded_gain_pct")


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports a real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, truth: dict, cost_bps: float = 1.0) -> dict:
    """Run the whole pipeline on a synthetic panel with a known fee ladder.

    On the planted world (``signal_strength = 1``) the TD rank must persist and the
    ``cheapest`` rule must beat the ``leader`` (the most expensive fund is generated last,
    so ``leader`` = ``fund_0`` would be wrong — the synthetic leader is taken as the *most
    expensive* fund to mirror SPY/QQQ being the priciest flagships). On the null
    (``signal_strength = 0``) neither must happen.
    """
    funds = [c for c in prices.columns if c != "cash"]
    er = {f: truth["fees_bps"][f] for f in funds}
    # Mirror the real families: the "leader" is the expensive flagship.
    ordered = sorted(funds, key=lambda f: -er[f])
    panel = prices[ordered]
    ann = annual_returns(panel)
    td = tracking_difference(ann, proxy="mean")
    pers = persistence(td, n_perm=500, seed=truth["seed"])
    g_cheap = gap_stats(panel, er, "cheapest", "leader", cost_bps=cost_bps)
    g_win = gap_stats(panel, er, "winner", "cheapest", cost_bps=cost_bps)
    return {
        "mean_spearman": pers["mean_spearman"],
        "t_spearman": pers["t_spearman"],
        "p_permutation": pers["p_permutation"],
        "mean_spearman_residual": pers["mean_spearman_residual"],
        "cheapest_minus_leader_bp": g_cheap["annual_gap_bp_yr"],
        "t_cheapest_minus_leader": g_cheap["t_annual"],
        "winner_minus_cheapest_bp": g_win["annual_gap_bp_yr"],
        "t_winner_minus_cheapest": g_win["t_annual"],
        "n_years": g_cheap["n_years"],
        "fee_spread_bps_eff": truth["fee_spread_bps_eff"],
    }
