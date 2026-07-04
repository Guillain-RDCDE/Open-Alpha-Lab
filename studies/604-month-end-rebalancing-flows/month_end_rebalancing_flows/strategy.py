"""Strategy + inference for Study 604 — Month-End Rebalancing Flows.

The claim (institutional folklore, formalised by Etula-Rinne-Suominen-Vaittinen's
*Dash for Cash* and the target-date/pension rebalancing literature): after a month in
which **stocks trounce bonds**, calendar rebalancers must SELL equity / BUY bonds to
restore their policy mix — depressing the equity-minus-bond spread over the **last
few trading days** of the month — and the pressure **reverses over the first days**
of the new month.

The measurement:

  * **Month table.** For every calendar month: the month-to-date equity-minus-bond
    cumulative-return gap measured through the close ``WINDOW`` (=3) trading days
    before month-end (``gap_pre`` — strictly prior information for the last-3-day
    window), the last-3-day eq/bd/spread returns, the full-month gap (``gap_full`` —
    known at the month-end close, strictly prior for the first-3-day window), and the
    NEXT month's first-3-day eq/bd/spread returns.

  * **Conditional split (the claim's own test).** Top vs bottom quintile of the
    conditioning gap: the flow story predicts the last-3-day spread is LOWER after
    equity-rich months (rebalancers sell the winner) and the first-3-day spread is
    HIGHER (the pressure reverses). Welch t on top-vs-bottom; one-sample t per
    quintile. Monthly windows are non-overlapping, so plain/Welch t's are the right
    tool; the tradability daily stream gets a Newey-West (HAC) t.

  * **Placebo.** Randomly permute the conditioning gap across months (killing the
    gap→window link while preserving both marginals), recompute the top-minus-bottom
    spread; the p-value is averaged over ≥ 20 seeds (single-seed baselines are
    banned on this desk).

  * **Tradability.** An expanding-quintile rule (no look-ahead thresholds): when the
    month's ``gap_pre`` exceeds the 80th percentile of all PRIOR months, short
    SPY / long AGG over the last 3 days, then flip long SPY / short AGG over the
    first 3 days of the new month. Signals use only closes already printed — entry at
    the signal close, first return earned the next day (exactly one execution lag).
    Costs are one-way × NAV per leg; the short leg pays borrow.

Third axis: is the desk's unconditional turn-of-the-month drift (study 89) just this
flow story in disguise? If TOM were rebalancing flows, the first-days equity pop
should CONCENTRATE in months with a big prior-month gap and vanish in the middle
quintiles. We test exactly that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 3           # trading days in the last-/first-of-month windows
TOP_Q = 0.8          # top-quintile threshold
BOT_Q = 0.2
NW_LAGS = 5          # HAC lags for the daily tradability stream
BORROW_ANN = 0.0030  # 30 bps/yr stock-borrow on the short leg
TDAYS = 252.0


# --------------------------------------------------------------------------- #
# The month table
# --------------------------------------------------------------------------- #
def month_table(rets: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """One row per month m: conditioning gaps + last-window and next-first-window returns.

    ``rets`` has daily simple-return columns ``eq`` and ``bd``. For month m:

      - ``gap_pre``   : cum eq − cum bd from m's first day through the close
                        ``window`` days before month-end (info available BEFORE the
                        last-window trade).
      - ``last_eq/last_bd/last_spread`` : cum returns over m's final ``window`` days.
      - ``gap_full``  : the full-month cum gap (known at m's last close).
      - ``first_eq/first_bd/first_spread`` : cum returns over the FIRST ``window``
                        days of month m+1 (the reversal window conditioned on m).

    Months shorter than ``window + 5`` days are dropped; the last month needs a
    complete following first-window to fill the ``first_*`` columns (else NaN).
    """
    months = rets.index.to_period("M")
    uniq = list(months.unique())
    rows = []
    for i, m in enumerate(uniq):
        r = rets[months == m]
        if len(r) < window + 5:
            continue
        pre = r.iloc[:-window]
        tail = r.iloc[-window:]
        cum = lambda s: float(np.expm1(np.log1p(s).sum()))
        row = {
            "month": m,
            "n_days": len(r),
            "gap_pre": cum(pre["eq"]) - cum(pre["bd"]),
            "last_eq": cum(tail["eq"]),
            "last_bd": cum(tail["bd"]),
            "gap_full": cum(r["eq"]) - cum(r["bd"]),
        }
        row["last_spread"] = row["last_eq"] - row["last_bd"]
        if i + 1 < len(uniq):
            nxt = rets[months == uniq[i + 1]]
            if len(nxt) >= window:
                head = nxt.iloc[:window]
                row["first_eq"] = cum(head["eq"])
                row["first_bd"] = cum(head["bd"])
                row["first_spread"] = row["first_eq"] - row["first_bd"]
        rows.append(row)
    tab = pd.DataFrame(rows).set_index("month")
    return tab


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
def ttest_vs_zero(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) − mean(b), unequal variances."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def hac_t(x: np.ndarray, lags: int = NW_LAGS) -> float:
    """Newey-West (HAC) t of mean(x) vs 0 with Bartlett weights."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 10:
        return float("nan")
    e = x - x.mean()
    s = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        s += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(s / n)
    return float(x.mean() / se) if se > 0 else float("nan")


# --------------------------------------------------------------------------- #
# The conditional split — the claim's own test
# --------------------------------------------------------------------------- #
def quintile_split(tab: pd.DataFrame, cond: str, out: str,
                   top_q: float = TOP_Q, bot_q: float = BOT_Q) -> dict:
    """Top-vs-bottom-quintile means of ``out`` conditioned on ``cond`` (full-sample
    quintiles — a descriptive event-study split; the tradability rule re-does the
    thresholds expanding, without look-ahead)."""
    t = tab.dropna(subset=[cond, out])
    hi_thr = t[cond].quantile(top_q)
    lo_thr = t[cond].quantile(bot_q)
    top = t.loc[t[cond] >= hi_thr, out].values
    bot = t.loc[t[cond] <= lo_thr, out].values
    return {
        "n": len(t), "n_top": len(top), "n_bot": len(bot),
        "top_mean_bps": float(np.mean(top)) * 1e4,
        "bot_mean_bps": float(np.mean(bot)) * 1e4,
        "uncond_mean_bps": float(t[out].mean()) * 1e4,
        "diff_bps": float(np.mean(top) - np.mean(bot)) * 1e4,
        "welch_t": welch_t(top, bot),
        "t_top": ttest_vs_zero(top),
        "t_bot": ttest_vs_zero(bot),
        "t_uncond": ttest_vs_zero(t[out].values),
        "top_vs_uncond_t": welch_t(top, t[out].values),
    }


def quintile_profile(tab: pd.DataFrame, cond: str, out: str,
                     n_q: int = 5) -> pd.DataFrame:
    """Mean of ``out`` (bps) + one-sample t per quintile of ``cond`` — the full
    dose-response profile behind the top/bottom headline."""
    t = tab.dropna(subset=[cond, out]).copy()
    t["q"] = pd.qcut(t[cond], n_q, labels=False) + 1
    rows = []
    for q, g in t.groupby("q"):
        rows.append({"quintile": int(q), "n": len(g),
                     "mean_bps": float(g[out].mean()) * 1e4,
                     "t": ttest_vs_zero(g[out].values),
                     "cond_mean_pct": float(g[cond].mean()) * 100})
    return pd.DataFrame(rows).set_index("quintile")


def placebo_pvalue(tab: pd.DataFrame, cond: str, out: str, side: str,
                   n_draws: int = 2_000, n_seeds: int = 20,
                   base_seed: int = 604) -> dict:
    """Permutation placebo: shuffle ``cond`` across months, recompute the
    top-minus-bottom-quintile difference in ``out``. ``side='le'`` tests how often a
    random tagging is as NEGATIVE as observed (the sell-the-winner leg);
    ``side='ge'`` as positive (the reversal leg). The p-value is the MEAN over
    ``n_seeds`` independent seeds (single-seed baselines are banned); the seed range
    is reported.
    """
    t = tab.dropna(subset=[cond, out]).reset_index(drop=True)
    obs = quintile_split(tab, cond, out)["diff_bps"]
    outv = t[out].values
    condv = t[cond].values
    n = len(t)
    k_top = int(np.ceil(n * (1 - TOP_Q)))
    k_bot = int(np.ceil(n * BOT_Q))
    ps = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        hits = 0
        for _ in range(n_draws):
            perm = rng.permutation(n)
            c = condv[perm]
            order = np.argsort(c)
            d = (outv[order[-k_top:]].mean() - outv[order[:k_bot]].mean()) * 1e4
            if side == "le":
                hits += (d <= obs)
            else:
                hits += (d >= obs)
        ps.append(hits / n_draws)
    ps = np.asarray(ps)
    return {"obs_diff_bps": obs, "p_mean": float(ps.mean()),
            "p_min": float(ps.min()), "p_max": float(ps.max()),
            "n_seeds": n_seeds, "n_draws": n_draws}


# --------------------------------------------------------------------------- #
# Tradability — expanding-quintile month-end reversal trade
# --------------------------------------------------------------------------- #
def flow_trade(rets: pd.DataFrame, window: int = WINDOW, burn_in: int = 60,
               cost_bps: float = 2.0, top_q: float = TOP_Q,
               borrow_ann: float = BORROW_ANN) -> dict:
    """The tradable version of the claim, with no look-ahead anywhere.

    For each month m (after ``burn_in`` months of history): compute ``gap_pre`` at
    the close ``window`` days before month-end. If it exceeds the ``top_q`` quantile
    of all PRIOR months' gaps (expanding threshold — no full-sample snooping), then

      * SELL leg: short eq / long bd from that close through the month-end close
        (``window`` daily returns, the first earned the day after the signal close —
        exactly one execution lag), then
      * REVERSAL leg: flip to long eq / short bd from the month-end close through
        the close of day ``window`` of the new month.

    Costs: each leg of each instrument pays ``cost_bps`` one-way × NAV; the flip at
    month-end trades 2× notional in both instruments (close + reopen), so a full
    event costs 8 one-way tickets (2 in + 4 flip + 2 out). The short side pays
    ``borrow_ann`` pro-rata for the days it is on. Returns the daily strategy net
    return stream (flat days = 0) plus summary stats with a HAC t on event days.
    """
    months = rets.index.to_period("M")
    uniq = list(months.unique())
    pos_by_month = {m: np.where(months == m)[0] for m in uniq}
    eqv = rets["eq"].values
    bdv = rets["bd"].values
    n = len(rets)
    strat = np.zeros(n)
    active = np.zeros(n, dtype=bool)
    gaps_hist = []
    n_events = 0
    daily_borrow = borrow_ann / TDAYS

    for i, m in enumerate(uniq):
        loc = pos_by_month[m]
        if len(loc) < window + 5:
            continue
        pre = loc[:-window]
        gap = float(np.expm1(np.log1p(eqv[pre]).sum())
                    - np.expm1(np.log1p(bdv[pre]).sum()))
        fire = False
        if len(gaps_hist) >= burn_in:
            thr = float(np.quantile(gaps_hist, top_q))
            fire = gap >= thr
        gaps_hist.append(gap)
        if not fire:
            continue
        n_events += 1
        tail = loc[-window:]                     # sell-the-winner leg
        for j in tail:
            strat[j] += (bdv[j] - eqv[j]) - daily_borrow
            active[j] = True
        if i + 1 < len(uniq):
            head = pos_by_month[uniq[i + 1]][:window]   # reversal leg
            for j in head:
                strat[j] += (eqv[j] - bdv[j]) - daily_borrow
                active[j] = True
        # 8 one-way tickets per event, spread over the event's days
        ev_days = list(tail) + ([] if i + 1 >= len(uniq)
                                else list(pos_by_month[uniq[i + 1]][:window]))
        cost_total = 8.0 * cost_bps / 1e4
        for j in ev_days:
            strat[j] -= cost_total / len(ev_days)

    net = pd.Series(strat, index=rets.index)
    gross_ann = float(net[active].sum() + n_events * 8.0 * cost_bps / 1e4) \
        / (n / TDAYS)
    net_ann = float(net.sum()) / (n / TDAYS)
    return {
        "daily": net, "active": pd.Series(active, index=rets.index),
        "n_events": n_events, "n_active_days": int(active.sum()),
        "share_active": float(active.mean()),
        "gross_ann_pct": gross_ann * 100, "net_ann_pct": net_ann * 100,
        "per_event_gross_bps": (float(net[active].sum()) + n_events * 8.0
                                * cost_bps / 1e4) / max(n_events, 1) * 1e4,
        "per_event_cost_bps": 8.0 * cost_bps,
        "t_net_active": hac_t(net[active].values),
        "cost_bps": cost_bps,
    }


# --------------------------------------------------------------------------- #
# Third axis — is turn-of-the-month just rebalancing flows in disguise?
# --------------------------------------------------------------------------- #
def tom_vs_flows(tab: pd.DataFrame) -> dict:
    """Does the unconditional first-days equity pop (study 89's TOM) live only in
    big-gap months?

    If TOM *were* the flow reversal, the first-3-day EQUITY return should be big
    after top-quintile gap months and gone in the middle. We report: the
    unconditional first-3 equity mean (the TOM drift on this tape), the same mean in
    the middle three quintiles of ``gap_full`` (where rebalancing pressure is
    smallest), the top and bottom quintile means, and the Welch t of extremes-vs-
    middle. TOM surviving in the middle quintiles = the flow story does NOT explain
    it.
    """
    t = tab.dropna(subset=["gap_full", "first_eq"]).copy()
    t["q"] = pd.qcut(t["gap_full"], 5, labels=False) + 1
    mid = t.loc[t["q"].isin([2, 3, 4]), "first_eq"].values
    ext = t.loc[t["q"].isin([1, 5]), "first_eq"].values
    top = t.loc[t["q"] == 5, "first_eq"].values
    bot = t.loc[t["q"] == 1, "first_eq"].values
    return {
        "uncond_bps": float(t["first_eq"].mean()) * 1e4,
        "t_uncond": ttest_vs_zero(t["first_eq"].values),
        "mid_bps": float(mid.mean()) * 1e4, "t_mid": ttest_vs_zero(mid),
        "ext_bps": float(ext.mean()) * 1e4,
        "top_bps": float(top.mean()) * 1e4, "t_top": ttest_vs_zero(top),
        "bot_bps": float(bot.mean()) * 1e4, "t_bot": ttest_vs_zero(bot),
        "welch_ext_vs_mid": welch_t(ext, mid),
        "n_mid": len(mid), "n_ext": len(ext),
    }


def era_split(tab: pd.DataFrame, cond: str, out: str,
              breaks: tuple = ("2003-09-30", "2015-12-31")) -> list[dict]:
    """The conditional split per era (splice era / mid / recent)."""
    eras = []
    b0, b1 = pd.Period(breaks[0], "M"), pd.Period(breaks[1], "M")
    idx = tab.index
    masks = [("pre-AGG (VBMFX era)", idx <= b0),
             (f"{b0.year+1}-{b1.year}", (idx > b0) & (idx <= b1)),
             (f"{b1.year+1}+", idx > b1)]
    for name, mk in masks:
        sub = tab[mk]
        if len(sub.dropna(subset=[cond, out])) < 40:
            continue
        s = quintile_split(sub, cond, out)
        s["era"] = name
        eras.append(s)
    return eras
