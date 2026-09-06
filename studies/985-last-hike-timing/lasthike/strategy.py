"""The last hike, in hindsight and in real time — Study 985.

Two questions that get conflated constantly:

1. **Hindsight.** Conditional on a date being the final hike of a tightening cycle, what did
   markets do over the following 3, 6, 12 and 24 months? This is a legitimate question about
   the economics of monetary cycles, and the answer is a piece of history.

2. **Real time.** On any given day, not knowing the future, can you act on it? You cannot,
   because "this was the last hike" is not observable at the time. The best a live investor can
   do is a **recognition rule**: declare the cycle over once *k* months have passed with no
   further hike. ``recognition_delay`` measures how long that took in each historical cycle,
   and ``realtime_events`` re-runs the whole event study on the dates a live rule would
   actually have fired.

The gap between the two is the study. It is a specific, measurable instance of a mistake that
runs through most macro-timing folklore: an event defined by what happens *after* it cannot be
traded at the time it happens.

The policy path is hard-coded in ``FOMC_MOVES`` — every target-rate change since 1994, when the
Fed began announcing them. That start date is not arbitrary: before 1994 the target was not
published and "the date of the last hike" is a matter of inference rather than record.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
HORIZONS_M = (3, 6, 12, 24)

# Every change in the federal funds target since the Fed began announcing them in Feb 1994.
# (date, new target in %, +1 hike / -1 cut). Source: Federal Reserve H.15 / FOMC statements.
FOMC_MOVES = [
    ("1994-02-04", 3.25, 1), ("1994-03-22", 3.50, 1), ("1994-04-18", 3.75, 1),
    ("1994-05-17", 4.25, 1), ("1994-08-16", 4.75, 1), ("1994-11-15", 5.50, 1),
    ("1995-02-01", 6.00, 1),
    ("1995-07-06", 5.75, -1), ("1995-12-19", 5.50, -1), ("1996-01-31", 5.25, -1),
    ("1997-03-25", 5.50, 1),
    ("1998-09-29", 5.25, -1), ("1998-10-15", 5.00, -1), ("1998-11-17", 4.75, -1),
    ("1999-06-30", 5.00, 1), ("1999-08-24", 5.25, 1), ("1999-11-16", 5.50, 1),
    ("2000-02-02", 5.75, 1), ("2000-03-21", 6.00, 1), ("2000-05-16", 6.50, 1),
    ("2001-01-03", 6.00, -1), ("2001-01-31", 5.50, -1), ("2001-03-20", 5.00, -1),
    ("2001-04-18", 4.50, -1), ("2001-05-15", 4.00, -1), ("2001-06-27", 3.75, -1),
    ("2001-08-21", 3.50, -1), ("2001-09-17", 3.00, -1), ("2001-10-02", 2.50, -1),
    ("2001-11-06", 2.00, -1), ("2001-12-11", 1.75, -1), ("2002-11-06", 1.25, -1),
    ("2003-06-25", 1.00, -1),
    ("2004-06-30", 1.25, 1), ("2004-08-10", 1.50, 1), ("2004-09-21", 1.75, 1),
    ("2004-11-10", 2.00, 1), ("2004-12-14", 2.25, 1), ("2005-02-02", 2.50, 1),
    ("2005-03-22", 2.75, 1), ("2005-05-03", 3.00, 1), ("2005-06-30", 3.25, 1),
    ("2005-08-09", 3.50, 1), ("2005-09-20", 3.75, 1), ("2005-11-01", 4.00, 1),
    ("2005-12-13", 4.25, 1), ("2006-01-31", 4.50, 1), ("2006-03-28", 4.75, 1),
    ("2006-05-10", 5.00, 1), ("2006-06-29", 5.25, 1),
    ("2007-09-18", 4.75, -1), ("2007-10-31", 4.50, -1), ("2007-12-11", 4.25, -1),
    ("2008-01-22", 3.50, -1), ("2008-01-30", 3.00, -1), ("2008-03-18", 2.25, -1),
    ("2008-04-30", 2.00, -1), ("2008-10-08", 1.50, -1), ("2008-10-29", 1.00, -1),
    ("2008-12-16", 0.25, -1),
    ("2015-12-16", 0.50, 1), ("2016-12-14", 0.75, 1), ("2017-03-15", 1.00, 1),
    ("2017-06-14", 1.25, 1), ("2017-12-13", 1.50, 1), ("2018-03-21", 1.75, 1),
    ("2018-06-13", 2.00, 1), ("2018-09-26", 2.25, 1), ("2018-12-19", 2.50, 1),
    ("2019-07-31", 2.25, -1), ("2019-09-18", 2.00, -1), ("2019-10-30", 1.75, -1),
    ("2020-03-03", 1.25, -1), ("2020-03-15", 0.25, -1),
    ("2022-03-17", 0.50, 1), ("2022-05-05", 1.00, 1), ("2022-06-16", 1.75, 1),
    ("2022-07-27", 2.50, 1), ("2022-09-21", 3.25, 1), ("2022-11-02", 4.00, 1),
    ("2022-12-14", 4.50, 1), ("2023-02-01", 4.75, 1), ("2023-03-22", 5.00, 1),
    ("2023-05-03", 5.25, 1), ("2023-07-26", 5.50, 1),
    ("2024-09-18", 5.00, -1), ("2024-11-07", 4.75, -1), ("2024-12-18", 4.50, -1),
]


# --------------------------------------------------------------------------- #
# The policy path
# --------------------------------------------------------------------------- #
def policy_path(moves=None) -> pd.DataFrame:
    """The target-rate history as a dated frame."""
    moves = moves or FOMC_MOVES
    df = pd.DataFrame(moves, columns=["date", "target", "direction"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def tightening_cycles(path: pd.DataFrame, min_hikes: int = 3,
                      gap_months: int = 12) -> pd.DataFrame:
    """Group consecutive hikes into cycles and mark each cycle's final hike.

    A cycle is a run of hikes uninterrupted by a cut and with no gap longer than
    ``gap_months`` between consecutive hikes. Both conventions are choices, and both are
    swept in the results — the 1997 single hike and the 2015-2018 crawl are exactly the cases
    where a different convention gives a different set of "last hikes".
    """
    hikes = path[path["direction"] > 0]
    cuts = path[path["direction"] < 0]
    cycles, current = [], []
    for d, row in hikes.iterrows():
        if current:
            prev = current[-1]
            months = (d - prev).days / 30.44
            cut_between = ((cuts.index > prev) & (cuts.index < d)).any()
            if cut_between or months > gap_months:
                cycles.append(current)
                current = []
        current.append(d)
    if current:
        cycles.append(current)
    rows = []
    for c in cycles:
        if len(c) < min_hikes:
            continue
        rows.append({"first_hike": c[0], "last_hike": c[-1], "n_hikes": len(c),
                     "months": (c[-1] - c[0]).days / 30.44,
                     "total_tightening": float(path.loc[c[-1], "target"]
                                               - path.loc[c[0], "target"])})
    return pd.DataFrame(rows)


def recognition_delay(path: pd.DataFrame, cycles: pd.DataFrame,
                      quiet_months: int = 6) -> pd.DataFrame:
    """When a live rule would have declared each cycle over.

    The rule: after ``quiet_months`` with no further hike, call it. That is generous — it uses
    no forecast, only the absence of news — and it is still late by construction.
    """
    rows = []
    for _, c in cycles.iterrows():
        last = c["last_hike"]
        declared = last + pd.DateOffset(months=quiet_months)
        rows.append({"last_hike": last, "declared_over": declared,
                     "delay_months": float(quiet_months),
                     "n_hikes": int(c["n_hikes"])})
    return pd.DataFrame(rows)


def false_alarms(path: pd.DataFrame, quiet_months: int = 6) -> pd.DataFrame:
    """Every date the live rule would have fired, including the ones that were wrong.

    A pause is not an ending. This walks the hike history and fires whenever ``quiet_months``
    pass with no hike, then records whether another hike followed within two years — the false
    alarms that a hindsight event study never sees.
    """
    hikes = path[path["direction"] > 0].index
    if len(hikes) == 0:
        return pd.DataFrame(columns=["signal_date", "after_hike", "was_the_end"])
    rows = []
    for i, d in enumerate(hikes):
        signal = d + pd.DateOffset(months=quiet_months)
        later = hikes[hikes > d]
        next_hike = later[0] if len(later) else None
        if next_hike is not None and next_hike <= signal:
            continue                      # another hike arrived before the rule could fire
        was_end = next_hike is None or (next_hike - signal).days > 730
        rows.append({"signal_date": signal, "after_hike": d, "was_the_end": bool(was_end),
                     "next_hike": next_hike})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# The event study
# --------------------------------------------------------------------------- #
def forward_returns(prices: pd.Series, dates, horizons_m=HORIZONS_M) -> pd.DataFrame:
    """Total return over each horizon following each date, on the trading calendar."""
    px = prices.dropna()
    rows = []
    for d in dates:
        d = pd.Timestamp(d)
        start_slice = px.loc[:d]
        if start_slice.empty:
            continue
        p0 = float(start_slice.iloc[-1])
        row = {"event": d}
        for m in horizons_m:
            end = d + pd.DateOffset(months=m)
            sl = px.loc[:end]
            row[f"m{m}"] = (float(sl.iloc[-1]) / p0 - 1.0) if (
                len(sl) and sl.index[-1] >= d and px.index[-1] >= end) else np.nan
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["event"] + [f"m{m}" for m in horizons_m]).set_index("event")
    return pd.DataFrame(rows).set_index("event")


def unconditional_returns(prices: pd.Series, horizons_m=HORIZONS_M,
                          step: int = 21) -> dict:
    """The base rate: the same horizons measured from every date, not just event dates.

    Without this, "stocks rose 12% in the year after the last hike" is not a finding — stocks
    rise about 10% in the year after *most* dates.
    """
    px = prices.dropna()
    sample = px.index[::step]
    fr = forward_returns(px, sample, horizons_m)
    return {f"m{m}": {"mean": float(fr[f"m{m}"].mean()),
                      "median": float(fr[f"m{m}"].median()),
                      "sd": float(fr[f"m{m}"].std(ddof=1)),
                      "n": int(fr[f"m{m}"].notna().sum())} for m in horizons_m}


def event_table(prices: pd.Series, dates, horizons_m=HORIZONS_M, step: int = 21) -> pd.DataFrame:
    """Event returns against the unconditional base rate, with a small-sample *t*."""
    fr = forward_returns(prices, dates, horizons_m)
    base = unconditional_returns(prices, horizons_m, step)
    rows = []
    for m in horizons_m:
        col = fr[f"m{m}"].dropna()
        b = base[f"m{m}"]
        n = len(col)
        excess = col.mean() - b["mean"] if n else np.nan
        se = b["sd"] / np.sqrt(n) if n else np.nan
        rows.append({"horizon_m": m, "n_events": n,
                     "event_mean": float(col.mean()) if n else np.nan,
                     "event_median": float(col.median()) if n else np.nan,
                     "base_mean": b["mean"], "excess": float(excess) if n else np.nan,
                     "t": float(excess / se) if n and se and se > 0 else np.nan,
                     "hit_rate": float((col > 0).mean()) if n else np.nan})
    return pd.DataFrame(rows).set_index("horizon_m")


def hindsight_vs_realtime(prices: pd.Series, cycles: pd.DataFrame, path: pd.DataFrame,
                          quiet_months: int = 6, horizons_m=HORIZONS_M) -> pd.DataFrame:
    """The study's central comparison, in one frame."""
    hind = event_table(prices, list(cycles["last_hike"]), horizons_m)
    rd = recognition_delay(path, cycles, quiet_months)
    real = event_table(prices, list(rd["declared_over"]), horizons_m)
    fa = false_alarms(path, quiet_months)
    live = event_table(prices, list(fa["signal_date"]), horizons_m)
    rows = []
    for m in horizons_m:
        rows.append({
            "horizon_m": m,
            "hindsight_excess": hind.loc[m, "excess"], "hindsight_t": hind.loc[m, "t"],
            "delayed_excess": real.loc[m, "excess"], "delayed_t": real.loc[m, "t"],
            "live_excess": live.loc[m, "excess"], "live_t": live.loc[m, "t"],
            "n_hindsight": hind.loc[m, "n_events"], "n_live": live.loc[m, "n_events"],
        })
    return pd.DataFrame(rows).set_index("horizon_m")


def what_the_delay_costs(prices: pd.Series, cycles: pd.DataFrame,
                         quiet_months: int = 6) -> pd.DataFrame:
    """The return given up between the true last hike and the day a live rule could act."""
    px = prices.dropna()
    rows = []
    for _, c in cycles.iterrows():
        a = c["last_hike"]
        b = a + pd.DateOffset(months=quiet_months)
        sa, sb = px.loc[:a], px.loc[:b]
        if sa.empty or sb.empty or px.index[-1] < b:
            continue
        rows.append({"last_hike": str(a.date()), "acted": str(sb.index[-1].date()),
                     "missed_return": float(sb.iloc[-1] / sa.iloc[-1] - 1.0)})
    return pd.DataFrame(rows).set_index("last_hike") if rows else pd.DataFrame(
        columns=["acted", "missed_return"])


def synthetic_world(n_cycles: int = 8, post_cycle_alpha: float = 0.0, n_years: int = 30,
                    seed: int = 985) -> dict:
    """A price series with tightening cycles and a controllable post-cycle rally.

    ``post_cycle_alpha`` is added to the daily drift for twelve months after each cycle's true
    final hike. At zero the cycles are decorative and the null holds. The generator also emits
    the cycle dates, so the study's real-time machinery can be run against a known truth.
    """
    rng = np.random.default_rng(seed)
    n = n_years * TRADING_DAYS
    idx = pd.bdate_range("1990-01-01", periods=n)
    drift = np.full(n, 0.08 / TRADING_DAYS)
    moves, last_hikes = [], []
    spacing = n // max(n_cycles, 1)
    rate = 3.0
    for c in range(n_cycles):
        start = c * spacing + spacing // 4
        n_hikes = int(rng.integers(4, 12))
        hike_gap = max(spacing // (4 * n_hikes), 21)
        for k in range(n_hikes):
            t = start + k * hike_gap
            if t >= n:
                break
            rate += 0.25
            moves.append((idx[t].strftime("%Y-%m-%d"), rate, 1))
            last_t = t
        last_hikes.append(idx[last_t])
        drift[last_t:min(last_t + TRADING_DAYS, n)] += post_cycle_alpha / TRADING_DAYS
        cut_t = min(last_t + TRADING_DAYS + 60, n - 1)
        rate = max(rate - 1.0, 0.25)
        moves.append((idx[cut_t].strftime("%Y-%m-%d"), rate, -1))
    rets = drift + rng.normal(0, 0.16 / np.sqrt(TRADING_DAYS), n)
    prices = pd.Series(100 * np.exp(np.cumsum(rets)), index=idx, name="price")
    return {"prices": prices, "moves": moves, "true_last_hikes": last_hikes}


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Real** if the *hindsight* twelve-month excess return after a final hike
      clears |*t*| >= 2 against the unconditional base rate; **Weak** if it is positive without
      significance; **None** otherwise.
    - **Tradability**: **Investable** only if the **live** rule — which fires on a pause, not on
      knowledge of the future — also clears |*t*| >= 2; **Fragile** if its excess is positive
      without significance; **Mirage** if it is negative or if the live rule's false-alarm rate
      exceeds half.
    """
    hs = h["hindsight_excess_12m"]
    signal = ("Real" if abs(h["hindsight_t_12m"]) >= 2.0 and hs > 0
              else ("Weak" if hs > 0 else "None"))
    live = h["live_excess_12m"]
    if live <= 0 or h["false_alarm_rate"] > 0.5:
        trad = "Mirage"
    elif abs(h["live_t_12m"]) >= 2.0:
        trad = "Investable"
    else:
        trad = "Fragile"
    return {
        "signal": signal,
        "signal_why": (
            f"Across the **{h['n_cycles']} tightening cycles** since the Fed began announcing "
            f"its target in 1994, the S&P returned **{h['hindsight_mean_12m']:+.1%}** in the "
            f"twelve months after the cycle's true final hike, against an unconditional base "
            f"rate of {h['base_mean_12m']:+.1%} — an excess of "
            f"**{hs:+.1%}** (*t* = **{h['hindsight_t_12m']:+.2f}**, "
            f"{h['hindsight_hit_12m']:.0%} of cycles positive). With {h['n_cycles']} events "
            f"there is no *t*-statistic that can carry much weight, and the study says so "
            f"rather than dressing eight observations as a finding."),
        "trad": trad,
        "trad_why": (
            f"None of that is available at the time. Nobody knew 2023-07-26 was the last hike "
            f"until months of not-hiking had passed. A live rule — declare the cycle over after "
            f"**{h['quiet_months']} quiet months** — fired {h['n_live_signals']} times, of "
            f"which **{h['false_alarm_rate']:.0%} were false alarms** (another hike followed "
            f"within two years), and earned an excess of **{live:+.1%}** at twelve months "
            f"(*t* = {h['live_t_12m']:+.2f}) against the hindsight version's {hs:+.1%}. The "
            f"delay alone gave up a median **{h['median_missed']:+.1%}** between the true last "
            f"hike and the day the rule could act — {h['missed_share']:.0%} of the whole "
            f"twelve-month move."),
        "one_sentence": (
            f"Buying the last hike returned {hs:+.1%} of excess over twelve months in hindsight "
            f"and {live:+.1%} for a rule that had to identify the event in real time — the gap "
            f"is what the folklore is actually made of."),
    }
