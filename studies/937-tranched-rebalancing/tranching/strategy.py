"""Strategy + inference for Study 937 — Tranches.

The sleeve under test is deliberately ordinary: a **monthly** two-asset trend rule.
On its rebalance day the sleeve looks at the risky leg (SPY) against its 200-day simple
moving average using closes **through that day**, and holds SPY for the next month if it
is above, IEF if it is below. That state is the position that earns the return of ``t+1``
(one execution lag, house form — see the conventions below) and is then left alone until
the next rebalance day.

"Monthly" is not one schedule but ``period = 21`` of them: a book rebalanced on offset 0
of the 21-day cycle is a different book from the one rebalanced on offset 7, even though
the *rule* is identical. That arbitrary choice is **rebalance timing luck** (Hoffstein et
al.); Study 836 measured the phantom dispersion it creates. This study tests the proposed
fix — **tranching**, a.k.a. overlapping portfolios: split the capital into ``N`` sleeves
rebalanced on ``N`` offsets spaced evenly round the cycle, so no single date decides the
book.

We measure, on the real tape:

1. the **cone** — the dispersion of outcomes across the 21 possible placements of an
   N-tranche book, for ``N = 1, 4, 12, 21``;
2. what tranching **costs**: annual traded notional, and the net-of-cost Sharpe;
3. whether the ex-post lucky offset is **forecastable** (it must not be, or the cone is
   an edge rather than an artefact);
4. an era cut, a cost sweep, and a cross-check on a 12-1 momentum sleeve.

Accounting conventions, stated once:

- **One** execution lag, in the house form: the state read from closes through the
  rebalance day ``t`` is the position that earns the return of ``t+1`` — one ``shift``,
  applied once, in ``offset_positions`` and nowhere else. (Execution is therefore at the
  ``t`` close, not the ``t+1`` close; ``verify.py`` also reports the whole cone with a
  second day of delay, which does not move the answer.)
- **Traded notional** on a switch is ``2 x |delta position|`` (sell one leg, buy the
  other), charged at ``cost_bps`` **one-way** on NAV. No shorting anywhere, so there is
  no borrow leg.
- Each tranche is a self-financing sleeve that compounds on its own; the book return is
  the **NAV-share weighted** average of the tranche returns (the standard overlapping-
  portfolio implementation), so the tranches are never silently rebalanced against each
  other.
- Sharpes are **excess-of-cash** on both sides of every race.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
PERIOD = 21  # trading days in a "month" — the number of distinct rebalance offsets


# --------------------------------------------------------------------------- #
# Cash legs
# --------------------------------------------------------------------------- #
def cash_returns_from_yield(irx: pd.Series) -> pd.Series:
    """Daily cash return from the ``^IRX`` 13-week bill **discount yield** (a PROXY).

    The quoted figure is an annualised percentage; we accrue it as ``y/100/252`` and lag
    it one day so the rate known at ``t-1`` is what is earned over ``t``. This is a proxy
    for a tradable bill ladder, not a fund total return — BIL is carried as the tradable
    cross-check on its own (shorter) window.
    """
    y = pd.Series(irx).astype(float).ffill()
    return (y / 100.0 / TRADING_DAYS).shift(1).rename("r_cash")


def cash_returns_from_index(level: pd.Series) -> pd.Series:
    """Daily cash return from an accumulation index (BIL's total-return close, or the
    synthetic cash leg)."""
    return pd.Series(level).astype(float).pct_change().rename("r_cash")


# --------------------------------------------------------------------------- #
# Signals (daily state; the rebalance schedule samples them)
# --------------------------------------------------------------------------- #
def sma(close: pd.Series, n: int) -> pd.Series:
    """Simple moving average requiring a full ``n``-observation window (no partials)."""
    return close.rolling(n, min_periods=n).mean()


def trend_state(risky: pd.Series, sma_n: int = 200) -> pd.Series:
    """Daily 1/0 state: 1 = risky leg above its ``sma_n``-day average, 0 = below.

    Formed from closes through day ``t`` only. The schedule samples it on rebalance days
    and ``offset_positions`` applies the one-day execution lag.
    """
    return (risky > sma(risky, sma_n)).astype(float).where(sma(risky, sma_n).notna())


def momentum_state(risky: pd.Series, safe: pd.Series,
                   lookback: int = 252, gap: int = 21) -> pd.Series:
    """Daily 1/0 state for a 12-1 relative-momentum sleeve (the cross-check rule).

    Compares the risky and defensive legs' total return from ``t-lookback`` to ``t-gap``
    (the classic 12-month-minus-most-recent-month window, which skips the short-term
    reversal month). 1 = risky leg ahead.
    """
    r_mom = risky.shift(gap) / risky.shift(lookback) - 1.0
    s_mom = safe.shift(gap) / safe.shift(lookback) - 1.0
    state = (r_mom > s_mom).astype(float)
    return state.where(r_mom.notna() & s_mom.notna())


def random_state(state: pd.Series, seed: int = 0) -> pd.Series:
    """Exposure-matched null control: the same risky-leg *frequency* on random days.

    Used only by the synthetic control. If the sleeve's advantage came from merely being
    in the defensive leg part of the time, this control would show it too; a sleeve that
    beats it is doing genuine timing. Matching exposure sidesteps any Sharpe mismatch
    between the two legs, so the null world's expected advantage is zero by construction.
    """
    in_frac = float(state.dropna().mean())
    rng = np.random.default_rng(seed)
    rand = pd.Series((rng.random(len(state)) < in_frac).astype(float),
                     index=state.index, name="random_state")
    return rand.where(state.notna())


# --------------------------------------------------------------------------- #
# One offset = one book
# --------------------------------------------------------------------------- #
def offset_positions(state: pd.Series, offset: int, period: int = PERIOD) -> pd.Series:
    """Daily risky-leg weight for the book rebalanced on ``offset`` of the cycle.

    The state is sampled on days whose position in the cycle equals ``offset``, held flat
    in between, then shifted one day: the state known at the close of the rebalance day
    ``t`` is the position that earns the return of ``t+1``. That is the house convention —
    one ``shift``, applied here and nowhere else — and it means execution happens at the
    ``t`` close (the closing print that formed the signal), not a day later. Delaying a
    further day is a one-line change (pass ``state.shift(1)``) and ``verify.py`` reports
    that variant.
    """
    n = len(state)
    is_rebal = (np.arange(n) % period) == (offset % period)
    sampled = state.where(pd.Series(is_rebal, index=state.index))
    return sampled.ffill().shift(1).rename(f"pos_{offset}")


def offset_return_matrix(
    risky: pd.Series,
    safe: pd.Series,
    state: pd.Series,
    period: int = PERIOD,
    cost_bps: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every one-tranche book, one column per rebalance offset.

    Returns ``(R, TV)``: net daily simple returns and daily **traded notional** (in units
    of the sleeve's NAV, both legs counted), each ``T x period``, restricted to the dates
    on which *all* offsets are live so the 21 books are strictly comparable.
    """
    r_risky = risky.pct_change()
    r_safe = safe.pct_change()
    cost = cost_bps * 1e-4
    rets, turns = {}, {}
    for o in range(period):
        pos = offset_positions(state, o, period=period)
        traded = 2.0 * pos.diff().abs()          # sell one leg, buy the other
        traded.iloc[0] = 0.0
        gross = pos * r_risky + (1.0 - pos) * r_safe
        rets[o] = gross - traded * cost
        turns[o] = traded
    R = pd.DataFrame(rets)
    TV = pd.DataFrame(turns)
    keep = R.notna().all(axis=1) & TV.notna().all(axis=1)
    return R[keep], TV[keep]


def tranche_offsets(n_tranches: int, base: int = 0, period: int = PERIOD) -> list[int]:
    """The ``n_tranches`` offsets of a book anchored at ``base``, spread round the cycle.

    ``period`` (21) is not divisible by 4 or 12, so the spacing is the evenly-floored
    ``floor(k * period / n)`` — as close to equal as an integer calendar allows.
    """
    return sorted({(base + int(k * period / n_tranches)) % period
                   for k in range(n_tranches)})


def combine(R: pd.DataFrame, TV: pd.DataFrame, offsets) -> tuple[pd.Series, pd.Series]:
    """Aggregate the chosen tranches into one book (NAV-share weighted, self-financing).

    Each tranche starts with equal capital and compounds on its own; the book's daily
    return is the NAV-weighted average of the tranche returns. Same for traded notional,
    so a book of N tranches trades at most 1/N of NAV per tranche rebalance.
    """
    cols = list(offsets)
    r = R[cols]
    nav = (1.0 + r).cumprod()
    w = nav.shift(1)
    w.iloc[0] = 1.0
    w = w.div(w.sum(axis=1), axis=0)
    return (w * r).sum(axis=1).rename("r_book"), (w * TV[cols]).sum(axis=1).rename("turnover")


def rotation_books(R: pd.DataFrame, TV: pd.DataFrame, n_tranches: int,
                   period: int = PERIOD) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All ``period`` placements of an N-tranche book (rotate the anchor round the cycle).

    For ``n_tranches = period`` every rotation is the same set, so the returned frame has
    identical columns — the cone has collapsed to a point by construction.
    """
    rets, turns = {}, {}
    memo: dict[tuple, tuple] = {}
    for base in range(period):
        offs = tuple(tranche_offsets(n_tranches, base, period))
        # Different anchors can select the same set of offsets — at N = period every
        # anchor does. Reuse the computed book rather than rebuilding an identical one.
        if offs not in memo:
            memo[offs] = combine(R, TV, list(offs))
        rets[base], turns[base] = memo[offs]
    return pd.DataFrame(rets), pd.DataFrame(turns)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int | None = None) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        var += 2.0 * w * float(u[lag:] @ u[:-lag]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def sharpe_diff_tstat(r1: pd.Series, r2: pd.Series) -> float:
    """HAC *t* on the daily return **difference** r1 - r2 (Jobson-Korkie, NW form).

    Both arms are excess-of-cash, so the cash leg cancels in the difference.
    """
    diff = (pd.Series(r1) - pd.Series(r2)).dropna()
    if len(diff) < 3:
        return float("nan")
    return newey_west_t(diff.to_numpy())


def spearman(a, b) -> float:
    """Rank correlation without a SciPy dependency."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def sharpe(excess: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    r = pd.Series(excess).astype(float).dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")


def max_drawdown(returns: pd.Series) -> float:
    r = pd.Series(returns).astype(float).dropna()
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def summary(returns: pd.Series, r_cash: pd.Series,
            periods_per_year: int = TRADING_DAYS) -> dict:
    """Headline stats for one book: excess-of-cash Sharpe plus lived CAGR / vol / DD."""
    r = pd.Series(returns).astype(float).dropna()
    e = (r - pd.Series(r_cash).reindex(r.index)).dropna()
    n = len(r)
    years = n / periods_per_year
    wealth = float((1.0 + r).prod())
    return {
        "n_days": int(n),
        "sharpe_excess": sharpe(e, periods_per_year),
        "cagr": float(wealth ** (1.0 / years) - 1.0) if years > 0 and wealth > 0 else float("nan"),
        "vol_ann": float(r.std(ddof=1) * np.sqrt(periods_per_year)),
        "max_drawdown": max_drawdown(r),
        "terminal_wealth": wealth,
        "tstat_excess": newey_west_t(e.to_numpy()),
    }


# --------------------------------------------------------------------------- #
# The cone
# --------------------------------------------------------------------------- #
def cone(
    R: pd.DataFrame,
    TV: pd.DataFrame,
    r_cash: pd.Series,
    n_list=(1, 4, 12, 21),
    period: int = PERIOD,
) -> dict:
    """Dispersion of outcomes across the 21 placements of an N-tranche book.

    For each ``N`` returns the mean / sd / min / max / range of the excess-of-cash Sharpe
    and of the CAGR across rotations, plus mean annual traded notional and mean drawdown.

    Read the two ends of the table together: ``sharpe_mean`` at ``N = 1`` is what an
    investor gets *in expectation* from picking one rebalance date blind, and ``sharpe_sd``
    is the lottery they are entering; at ``N = period`` every rotation is the same book, so
    ``sharpe_mean`` is that book's Sharpe and ``sharpe_sd`` is exactly zero.
    """
    out = {}
    for n in n_list:
        BR, BT = rotation_books(R, TV, n, period=period)
        rows = [summary(BR[c], r_cash) for c in BR.columns]
        sh = np.array([r["sharpe_excess"] for r in rows], dtype=float)
        cg = np.array([r["cagr"] for r in rows], dtype=float)
        tw = np.array([r["terminal_wealth"] for r in rows], dtype=float)
        dd = np.array([r["max_drawdown"] for r in rows], dtype=float)
        vol = np.array([r["vol_ann"] for r in rows], dtype=float)
        turn = np.array([BT[c].mean() * TRADING_DAYS for c in BT.columns], dtype=float)
        out[n] = {
            "n_tranches": n,
            "sharpe_mean": float(sh.mean()),
            "sharpe_sd": float(sh.std(ddof=1)) if len(sh) > 1 else 0.0,
            "sharpe_min": float(sh.min()), "sharpe_max": float(sh.max()),
            "sharpe_range": float(sh.max() - sh.min()),
            "cagr_mean": float(cg.mean()),
            "cagr_sd": float(cg.std(ddof=1)) if len(cg) > 1 else 0.0,
            "cagr_range": float(cg.max() - cg.min()),
            "tw_min": float(tw.min()), "tw_max": float(tw.max()),
            "tw_spread_pct": float((tw.max() / tw.min() - 1.0) * 100.0),
            "dd_mean": float(dd.mean()), "dd_min": float(dd.min()), "dd_max": float(dd.max()),
            "vol_mean": float(vol.mean()),
            "turnover_mean": float(turn.mean()),
            "sharpes": sh, "cagrs": cg,
        }
    return out


def ticket_stats(TV: pd.DataFrame, n_tranches: int, base: int = 0,
                 period: int = PERIOD, n_days: int | None = None) -> dict:
    """Trade *tickets* per year for an N-tranche book (a fixed-cost, not a spread, count).

    Proportional cost is charged on notional and barely moves with N (each tranche trades
    1/N of NAV). Broker tickets do not work that way: a switch in each of N tranches is N
    separate pairs of orders. This counts them so a fixed per-ticket fee can be priced —
    the number is an input to an **ASSUMPTION** sweep, not a tape measurement.
    """
    n = n_days if n_days is not None else len(TV)
    offs = tranche_offsets(n_tranches, base, period)
    switches = sum(float((TV[o] > 0).sum()) for o in offs)
    return {
        "n_tranches": n_tranches,
        "switch_days_per_year": switches / (n / TRADING_DAYS),
        "tickets_per_year": 2.0 * switches / (n / TRADING_DAYS),  # sell + buy per switch
    }


def dispersion_bootstrap(
    R: pd.DataFrame,
    TV: pd.DataFrame,
    r_cash: pd.Series,
    n_boot: int = 1000,
    block: int = 21,
    seed: int = 937,
    period: int = PERIOD,
    alpha: float = 0.05,
) -> dict:
    """Block-bootstrap CI for (a) the cone width at N=1 and (b) the tranching Sharpe gain.

    Resamples *rows* (blocks of ``block`` consecutive days) jointly across the 21
    single-date books and the fully-tranched book, so the cross-sectional dependence is
    preserved. Reports the sd of the 21 single-date Sharpes and the gain
    ``Sharpe(tranched) - mean(Sharpe(single-date))``.
    """
    singles, _ = rotation_books(R, TV, 1, period=period)
    full, _ = rotation_books(R, TV, period, period=period)
    idx = singles.index
    cash = pd.Series(r_cash).reindex(idx)
    E = singles.sub(cash, axis=0).to_numpy(dtype=float)
    ef = (full[0] - cash).to_numpy(dtype=float)
    n = len(idx)
    ann = np.sqrt(TRADING_DAYS)

    def _stats(rows):
        e = E[rows]
        sd = e.std(axis=0, ddof=1)
        sh = np.where(sd > 0, e.mean(axis=0) / np.where(sd > 0, sd, 1.0) * ann, np.nan)
        f = ef[rows]
        shf = f.mean() / f.std(ddof=1) * ann if f.std(ddof=1) > 0 else np.nan
        return float(np.nanstd(sh, ddof=1)), float(shf - np.nanmean(sh))

    point_sd, point_gain = _stats(np.arange(n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    sds = np.full(n_boot, np.nan)
    gains = np.full(n_boot, np.nan)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        rows = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        sds[b], gains[b] = _stats(rows)
    lo_p, hi_p = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "sd_point": point_sd,
        "sd_ci": (float(np.nanpercentile(sds, lo_p)), float(np.nanpercentile(sds, hi_p))),
        "gain_point": point_gain,
        "gain_ci": (float(np.nanpercentile(gains, lo_p)), float(np.nanpercentile(gains, hi_p))),
        "gain_frac_negative": float(np.nanmean(gains < 0)),
        "n_boot": int(n_boot), "block": int(block), "n_obs": int(n),
    }


def best_worst_test(R: pd.DataFrame, r_cash: pd.Series) -> dict:
    """The ex-post best minus ex-post worst single-date book (an in-sample upper bound).

    This *t* is selection-biased by construction — the two arms are the argmax and argmin
    of 21 correlated books — and is reported only to show how large the cone looks when
    you let hindsight pick the ends of it.
    """
    cash = pd.Series(r_cash).reindex(R.index)
    E = R.sub(cash, axis=0)
    sh = E.apply(sharpe)
    best, worst = int(sh.idxmax()), int(sh.idxmin())
    return {
        "best_offset": best, "worst_offset": worst,
        "sharpe_best": float(sh[best]), "sharpe_worst": float(sh[worst]),
        "sharpe_gap": float(sh[best] - sh[worst]),
        "t_diff": sharpe_diff_tstat(E[best], E[worst]),
    }


def persistence(R: pd.DataFrame, r_cash: pd.Series, split: str | None = None) -> dict:
    """Is the lucky offset forecastable? Rank the 21 books in each half and compare.

    If the winning rebalance date were skill, its first-half ranking would carry into the
    second half. Reports the Spearman rank correlation of the two halves' Sharpes, the
    second-half rank (1 = best of 21) of the first-half winner, and what picking the
    first-half winner earned in the second half.

    The comparison that matters is **same-window**: the second-half edge of the chased
    date against the tranching gain *measured on that same second half* (``gain_late``),
    never against the full-sample gain. Both are reported here so nobody has to line up
    two different windows by hand.
    """
    cash = pd.Series(r_cash).reindex(R.index)
    E = R.sub(cash, axis=0)
    if split is None:
        cut = E.index[len(E) // 2]
    else:
        cut = pd.Timestamp(split)
    e1, e2 = E[E.index < cut], E[E.index >= cut]
    s1, s2 = e1.apply(sharpe), e2.apply(sharpe)
    win1 = int(s1.idxmax())
    rank2 = int(s2.rank(ascending=False)[win1])
    # The fully-tranched book on the same rows (TV is only needed for turnover here).
    full, _ = rotation_books(R, R * 0.0, R.shape[1])
    ef = (full[0] - cash)
    return {
        "split": str(pd.Timestamp(cut).date()),
        "n_early": int(len(e1)), "n_late": int(len(e2)),
        "rho": spearman(s1.to_numpy(), s2.to_numpy()),
        "early_winner": win1,
        "late_rank_of_early_winner": rank2,
        "late_sharpe_of_early_winner": float(s2[win1]),
        "late_sharpe_mean": float(s2.mean()),
        "late_sharpe_best": float(s2.max()),
        "late_winner": int(s2.idxmax()),
        "chase_edge_late": float(s2[win1] - s2.mean()),
        "gain_early": float(sharpe(ef[ef.index < cut]) - s1.mean()),
        "gain_late": float(sharpe(ef[ef.index >= cut]) - s2.mean()),
    }


# --------------------------------------------------------------------------- #
# Sweeps
# --------------------------------------------------------------------------- #
def cost_sweep(
    risky: pd.Series,
    safe: pd.Series,
    state: pd.Series,
    r_cash: pd.Series,
    cost_grid=(0.0, 1.0, 5.0, 10.0, 25.0),
    n_list=(1, 21),
    period: int = PERIOD,
) -> list[dict]:
    """Cone statistics at several one-way costs (no shorting → no borrow leg)."""
    rows = []
    for c in cost_grid:
        R, TV = offset_return_matrix(risky, safe, state, period=period, cost_bps=c)
        cn = cone(R, TV, r_cash, n_list=n_list, period=period)
        row = {"cost_bps": c}
        for n in n_list:
            row[f"sharpe_mean_n{n}"] = cn[n]["sharpe_mean"]
            row[f"sharpe_sd_n{n}"] = cn[n]["sharpe_sd"]
            row[f"turnover_n{n}"] = cn[n]["turnover_mean"]
        row["gain"] = cn[max(n_list)]["sharpe_mean"] - cn[min(n_list)]["sharpe_mean"]
        rows.append(row)
    return rows


def random_cone_sd_sweep(
    risky: pd.Series,
    safe: pd.Series,
    state: pd.Series,
    r_cash: pd.Series,
    seeds=range(20),
    cost_bps: float = 5.0,
    period: int = PERIOD,
) -> np.ndarray:
    """Cone sd of the exposure-matched **random-timing** rule, once per seed.

    The "an edge-free rule has a cone too (and a wider one)" claim must not rest on one
    arbitrary draw: a single random schedule is itself a lottery ticket. This runs the
    whole cone measurement across ``seeds`` and returns the array of N=1 Sharpe sds, so
    the claim can be stated as a distribution instead of a lucky number.
    """
    out = []
    for s in seeds:
        rnd = random_state(state, seed=int(s))
        Rr, TVr = offset_return_matrix(risky, safe, rnd, period=period, cost_bps=cost_bps)
        out.append(cone(Rr, TVr, r_cash, n_list=(1,), period=period)[1]["sharpe_sd"])
    return np.array(out, dtype=float)


def era_cut(
    risky: pd.Series,
    safe: pd.Series,
    state: pd.Series,
    r_cash: pd.Series,
    split: str = "2014-01-01",
    cost_bps: float = 5.0,
    n_list=(1, 21),
    period: int = PERIOD,
) -> dict:
    """Re-measure the cone on each half of the sample."""
    out = {}
    for tag, sl in [("early", slice(None, split)), ("late", slice(split, None))]:
        r, s, st_ = risky.loc[sl], safe.loc[sl], state.loc[sl]
        if len(r) < 3 * period + 60:
            out[tag] = None
            continue
        R, TV = offset_return_matrix(r, s, st_, period=period, cost_bps=cost_bps)
        cn = cone(R, TV, r_cash, n_list=n_list, period=period)
        out[tag] = {
            "n_days": int(len(R)),
            "sharpe_mean_n1": cn[min(n_list)]["sharpe_mean"],
            "sharpe_sd_n1": cn[min(n_list)]["sharpe_sd"],
            "sharpe_range_n1": cn[min(n_list)]["sharpe_range"],
            "sharpe_full": cn[max(n_list)]["sharpe_mean"],
            "gain": cn[max(n_list)]["sharpe_mean"] - cn[min(n_list)]["sharpe_mean"],
            "turnover_n1": cn[min(n_list)]["turnover_mean"],
            "turnover_full": cn[max(n_list)]["turnover_mean"],
        }
    return out


# --------------------------------------------------------------------------- #
# Synthetic control (machinery proof — never supports a real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, sma_n: int = 200, cost_bps: float = 5.0,
                     period: int = PERIOD) -> dict:
    """Run the whole cone measurement on a synthetic (risky, safe, cash) tape.

    Two things must hold for the harness to be trusted:

    - the **sleeve** must earn something when a regime is planted
      (``signal_strength=1``) and nothing on the null (``signal_strength=0``);
    - the **cone** must be present in *both* worlds and must collapse with N in both —
      timing luck is an artefact of the arbitrary date, not a symptom of edge.
    """
    state = trend_state(prices["risky"], sma_n=sma_n)
    r_cash = cash_returns_from_index(prices["cash"])
    R, TV = offset_return_matrix(prices["risky"], prices["safe"], state,
                                 period=period, cost_bps=cost_bps)
    cn = cone(R, TV, r_cash, n_list=(1, period), period=period)
    bh = summary(prices["risky"].pct_change(), r_cash)
    rnd = random_state(state, seed=int(prices["risky"].iloc[0] * 1e6) % 10_000)
    Rr, TVr = offset_return_matrix(prices["risky"], prices["safe"], rnd,
                                   period=period, cost_bps=cost_bps)
    cnr = cone(Rr, TVr, r_cash, n_list=(1, period), period=period)
    return {
        "sharpe_mean_n1": cn[1]["sharpe_mean"],
        "sharpe_sd_n1": cn[1]["sharpe_sd"],
        "sharpe_range_n1": cn[1]["sharpe_range"],
        "sharpe_full": cn[period]["sharpe_mean"],
        "sharpe_sd_full": cn[period]["sharpe_sd"],
        "gain": cn[period]["sharpe_mean"] - cn[1]["sharpe_mean"],
        "shrink_ratio": (cn[period]["sharpe_sd"] / cn[1]["sharpe_sd"]
                         if cn[1]["sharpe_sd"] > 0 else float("nan")),
        "turnover_n1": cn[1]["turnover_mean"],
        "turnover_full": cn[period]["turnover_mean"],
        "sleeve_minus_buyhold": cn[period]["sharpe_mean"] - bh["sharpe_excess"],
        "sleeve_minus_random": cn[period]["sharpe_mean"] - cnr[period]["sharpe_mean"],
        "sharpe_random": cnr[period]["sharpe_mean"],
        "sharpe_buyhold": bh["sharpe_excess"],
        "cone_sd_random": cnr[1]["sharpe_sd"],
        "n_days": int(len(R)),
    }
