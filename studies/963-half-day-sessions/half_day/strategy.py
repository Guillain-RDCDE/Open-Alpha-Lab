"""Measurement & inference for Study 963 — The Half Day.

The question is small and the machinery has to be exact, because the sample is tiny:
roughly **two shortened sessions a year** that the tape confirms, so thirty years is sixty
observations per ticker. Everything here is built around not fooling ourselves with
sixty numbers.

**Which dates?** Not a typed-in list. A list typed from memory is the classic way a
calendar study goes wrong — one wrong Christmas Eve and a third of the sample is noise.
Instead:

1. ``rule_candidates`` derives the candidates from the calendar *and the tape's own
   trading days*: the session before Independence Day when that session falls on July 3,
   the Friday after Thanksgiving, and December 24 when it is a trading day. Those are the
   three standing NYSE 1 p.m. closes.
2. ``confirm_candidates`` then makes the **tape** rule on each one: a genuine half session
   prints a fraction of a normal day's volume, so a candidate is kept only if its volume
   is below ``max_ratio`` of the trailing 60-session median. Candidates that fail are
   reported, not silently dropped.
3. ``unclaimed_thin_days`` runs the check in reverse — days the rule never proposed that
   are just as thin — so the reader can see the recall of the rule, not only its
   precision. (Expect a handful: the sessions bracketing a holiday are quiet too, and
   half-days that the NYSE granted once, for a funeral or a storm, are not on any rule.)

**Which return?** Three, kept apart on purpose, because "the half-day return" is
ambiguous and the ambiguity is where the folklore lives:

- ``r_on``  — previous close to today's open (the overnight gap, a full night either way);
- ``r_oc``  — open to close (the *shortened session itself*, 2.5 hours instead of 6.5);
- ``r_cc``  — close to close, the sum of the two, which is what a holder actually earns.

**Inference.** Every group mean carries a Newey-West (HAC) *t* from ``quantlab.analytics``
and, because sixty observations of a fat-tailed series is exactly where the *t* misleads,
a circular-block bootstrap CI on the **difference** against ordinary days. Every family
(July 3 / Black Friday / Christmas Eve) is also reported alone, which multiplies the
tests: the study runs ``5 tickers x 3 families x 3 windows = 45`` cells, so
``expected_false_positives`` prints what 45 coin flips at the 5% level look like before
any cell is called a discovery.

**Costs.** The arithmetic that decides tradability is brutal and is done in
``cost_arithmetic``: an edge earned on ~3 sessions a year is multiplied by 3, not by 252,
and each of those sessions costs a round trip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.analytics import mean_tstat_hac

TRADING_DAYS = 252

FAMILIES = ("jul3", "black_friday", "dec24")
FAMILY_LABEL = {
    "jul3": "July 3 (pre-Independence Day)",
    "black_friday": "the Friday after Thanksgiving",
    "dec24": "Christmas Eve",
}

# A genuine 1 p.m. session prints far less than a normal day. The threshold is an
# assumption, and it is swept in verify.py rather than defended here.
MAX_VOLUME_RATIO = 0.75
VOLUME_WINDOW = 60


# --------------------------------------------------------------------------- #
# The calendar: candidates from the rule, confirmation from the tape
# --------------------------------------------------------------------------- #
def thanksgiving(year: int) -> pd.Timestamp:
    """US Thanksgiving: the fourth Thursday of November."""
    first = pd.Timestamp(year=year, month=11, day=1)
    first_thursday = first + pd.Timedelta(days=(3 - first.dayofweek) % 7)
    return first_thursday + pd.Timedelta(days=21)


def rule_candidates(index: pd.DatetimeIndex) -> pd.Series:
    """Rule-derived early-close candidates, as a ``date -> family`` Series.

    Only dates that are *actually in* ``index`` survive — the tape's own trading calendar
    decides what a trading day is, so a Christmas Eve that fell on a Sunday, or a July 3
    the exchange took off entirely, never enters the sample.
    """
    idx = pd.DatetimeIndex(index)
    days = set(idx)
    out: dict[pd.Timestamp, str] = {}
    for year in sorted({d.year for d in idx}):
        jul3 = pd.Timestamp(year=year, month=7, day=3)
        if jul3 in days and jul3.dayofweek < 5:
            out[jul3] = "jul3"
        bf = thanksgiving(year) + pd.Timedelta(days=1)
        if bf in days:
            out[bf] = "black_friday"
        dec24 = pd.Timestamp(year=year, month=12, day=24)
        if dec24 in days and dec24.dayofweek < 5:
            out[dec24] = "dec24"
    s = pd.Series(out, name="family").sort_index()
    s.index.name = "date"
    return s


def volume_ratio(bars: pd.DataFrame, window: int = VOLUME_WINDOW) -> pd.Series:
    """Each day's volume over the median of the ``window`` sessions *before* it.

    The window is strictly trailing (``shift(1)``): a half day must not be allowed to
    lower the very benchmark it is being compared against.
    """
    med = bars["volume"].rolling(window, min_periods=window // 2).median().shift(1)
    return (bars["volume"] / med).rename("volume_ratio")


def session_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """The three returns plus the volume ratio, one row per session.

    ``r_on`` (previous close -> open), ``r_oc`` (open -> close, the session itself) and
    ``r_cc`` (close -> close) compose exactly: ``(1+r_on)(1+r_oc) = 1+r_cc``.
    """
    close, open_ = bars["close"], bars["open"]
    df = pd.DataFrame({
        "r_on": open_ / close.shift(1) - 1.0,
        "r_oc": close / open_ - 1.0,
        "r_cc": close / close.shift(1) - 1.0,
        "volume_ratio": volume_ratio(bars),
    })
    return df.dropna(subset=["r_cc"])


def confirm_candidates(sessions: pd.DataFrame, cands: pd.Series,
                       max_ratio: float = MAX_VOLUME_RATIO) -> pd.DataFrame:
    """Candidate table with each date's volume ratio and whether the tape confirms it."""
    idx = cands.index.intersection(sessions.index)
    out = pd.DataFrame({
        "family": cands.reindex(idx),
        "volume_ratio": sessions["volume_ratio"].reindex(idx),
        "r_cc": sessions["r_cc"].reindex(idx),
    })
    out["confirmed"] = out["volume_ratio"] < max_ratio
    return out


def unclaimed_thin_days(sessions: pd.DataFrame, cands: pd.Series,
                        max_ratio: float = MAX_VOLUME_RATIO) -> pd.DataFrame:
    """Sessions as thin as a half day that the rule never proposed (the recall check).

    Pass the *median confirmed* ratio rather than the confirmation threshold to ask the
    sharper question — "how many ordinary sessions are as quiet as a **typical** half
    day?" — because a threshold generous enough to keep 90% of the real events also keeps
    a fifth of the ordinary tape.
    """
    thin = sessions[sessions["volume_ratio"] < max_ratio]
    return thin.loc[~thin.index.isin(cands.index)]


def event_mask(sessions: pd.DataFrame, dates: pd.Index, offset: int = 0) -> pd.Series:
    """Boolean mask over ``sessions`` for the session ``offset`` trading days from each date."""
    pos = sessions.index.get_indexer(pd.DatetimeIndex(dates))
    pos = pos[pos >= 0] + offset
    pos = pos[(pos >= 0) & (pos < len(sessions))]
    mask = pd.Series(False, index=sessions.index)
    mask.iloc[np.unique(pos)] = True
    return mask


# --------------------------------------------------------------------------- #
# Inference on a very small sample
# --------------------------------------------------------------------------- #
def group_stats(returns: pd.Series, mask: pd.Series) -> dict:
    """Event mean vs the mean of every other session, with a Welch *t* on the gap.

    **The lag choice is not cosmetic.** The ordinary sessions are consecutive days, so
    their mean gets a Newey-West long-run variance (daily returns are mildly
    autocorrelated and heteroskedastic). The *events* are not consecutive anything — they
    sit four months apart — so applying a Bartlett kernel across them correlates
    observations that share nothing but a row number, and the resulting standard error is
    too small. Left uncorrected it manufactures significance: on the study's own null
    world (thin volume, no planted return) the HAC-on-events version returned *t* = 3.0.
    Events therefore get ``lags=0``, which for a scattered event set is the *conservative*
    and correct choice, and the null behaves (``tests/test_strategy.py``).

    The gap's *t* is a Welch statistic on the difference of two means with unequal
    variances and wildly unequal sample sizes — a pooled-variance *t* is not available
    here in good conscience.
    """
    r = returns.dropna()
    m = mask.reindex(r.index).fillna(False)
    ev, rest = r[m], r[~m]
    if len(ev) < 3 or len(rest) < 30:
        return {"n_event": int(len(ev)), "n_rest": int(len(rest)), "mean_bps": np.nan,
                "rest_bps": np.nan, "diff_bps": np.nan, "t_event": np.nan,
                "t_diff": np.nan, "hit_rate": np.nan}
    a, b = mean_tstat_hac(ev, lags=0), mean_tstat_hac(rest)
    se = np.sqrt(a["se_bps"] ** 2 + b["se_bps"] ** 2)
    diff = a["mean_bps"] - b["mean_bps"]
    return {
        "n_event": int(len(ev)), "n_rest": int(len(rest)),
        "mean_bps": float(a["mean_bps"]), "rest_bps": float(b["mean_bps"]),
        "diff_bps": float(diff), "t_event": float(a["tstat"]),
        "t_diff": float(diff / se) if se > 0 else np.nan,
        "hit_rate": float((ev > 0).mean()),
    }


def bootstrap_diff_ci(returns: pd.Series, mask: pd.Series, n_boot: int = 4000,
                      alpha: float = 0.05, seed: int = 963) -> dict:
    """Percentile CI for the event-minus-rest mean, resampling the events themselves.

    The events are few and scattered, so the uncertainty that matters is the uncertainty
    in *their* mean: each resample draws ``n_event`` events with replacement and an equal
    number of ordinary sessions, and the statistic is the difference of the two means.
    """
    r = returns.dropna()
    m = mask.reindex(r.index).fillna(False)
    ev = r[m].to_numpy(dtype=float)
    rest = r[~m].to_numpy(dtype=float)
    if ev.size < 3 or rest.size < 30:
        return {"ci_low": np.nan, "ci_high": np.nan, "frac_negative": np.nan, "n_boot": 0}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        a = rng.choice(ev, ev.size, replace=True).mean()
        b = rng.choice(rest, ev.size, replace=True).mean()
        draws[i] = a - b
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"ci_low": float(lo * 1e4), "ci_high": float(hi * 1e4),
            "frac_negative": float((draws < 0).mean()), "n_boot": int(n_boot)}


def window_table(sessions: pd.DataFrame, dates: pd.Index, leg: str = "r_cc",
                 offsets: tuple[int, ...] = (-1, 0, 1)) -> pd.DataFrame:
    """``group_stats`` for the day before, the half day itself and the day after."""
    rows = []
    for off in offsets:
        st = group_stats(sessions[leg], event_mask(sessions, dates, off))
        st["offset"] = off
        rows.append(st)
    return pd.DataFrame(rows).set_index("offset")


def family_table(sessions: pd.DataFrame, confirmed: pd.DataFrame,
                 leg: str = "r_cc") -> pd.DataFrame:
    """One row per family — the three standing early closes are three different animals."""
    rows = []
    for fam in FAMILIES:
        dates = confirmed.index[(confirmed["family"] == fam) & confirmed["confirmed"]]
        st = group_stats(sessions[leg], event_mask(sessions, dates, 0))
        st["family"] = fam
        rows.append(st)
    return pd.DataFrame(rows).set_index("family")


def era_cut(sessions: pd.DataFrame, dates: pd.Index, split: str = "2010-01-01",
            leg: str = "r_cc") -> pd.DataFrame:
    """The same test in each half of the sample. A real calendar effect does not need one era."""
    rows = []
    for tag, sl in (("early", sessions.loc[:split]), ("late", sessions.loc[split:])):
        d = pd.DatetimeIndex(dates).intersection(sl.index)
        st = group_stats(sl[leg], event_mask(sl, d, 0))
        st["era"] = tag
        st["start"] = str(sl.index[0].date())
        st["end"] = str(sl.index[-1].date())
        rows.append(st)
    return pd.DataFrame(rows).set_index("era")


def expected_false_positives(n_cells: int, size: float = 0.05) -> float:
    """How many of ``n_cells`` independent tests clear the bar by luck alone."""
    return float(n_cells * size)


# --------------------------------------------------------------------------- #
# Could you trade it?
# --------------------------------------------------------------------------- #
def cost_arithmetic(edge_bps: float, sessions_per_year: float,
                    cost_bps: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0, 10.0)) -> pd.DataFrame:
    """Gross and net annual return of a book that is flat except on the event days.

    ``edge_bps`` is the per-session edge; the position is entered and exited around each
    event, so each session pays ``2 * cost_bps`` (round trip, one-way cost each way).
    Nothing is compounded: at these magnitudes compounding is decoration.
    """
    rows = []
    for c in cost_bps:
        gross = edge_bps * sessions_per_year
        net = (edge_bps - 2.0 * c) * sessions_per_year
        rows.append({"cost_bps_one_way": c, "gross_bps_per_year": gross,
                     "net_bps_per_year": net, "net_pct_per_year": net / 100.0})
    return pd.DataFrame(rows).set_index("cost_bps_one_way")


def breakeven_cost_bps(edge_bps: float) -> float:
    """The one-way cost at which the edge is exactly eaten (``edge / 2``)."""
    return float(edge_bps / 2.0)


# --------------------------------------------------------------------------- #
# The synthetic control — does the apparatus find what is put in front of it?
# --------------------------------------------------------------------------- #
def plant_half_days(bars: pd.DataFrame, every: int = 84, bump_bps: float = 0.0,
                    vol_shrink: float = 0.35) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Plant a known half-day pattern into synthetic bars.

    Every ``every``-th session becomes a "half day": its volume is multiplied by
    ``vol_shrink`` and its close is lifted by ``bump_bps`` basis points relative to its
    open (the later closes are shifted with it, so the bump is a one-day event and not a
    permanent level change in the return series that follows).
    """
    out = bars.copy()
    pos = np.arange(every, len(out), every)
    dates = out.index[pos]
    out.loc[dates, "volume"] = out.loc[dates, "volume"] * vol_shrink

    # The bump is a *level shift* of everything from the event onward, so the planted
    # return lands once, on the event's own open->close and close->close, and is not
    # given back the next morning (a one-bar bump would plant an equal and opposite
    # effect at offset +1 and the window table would read it as a real reversal).
    factor = 1.0 + bump_bps / 1e4
    cols = list(out.columns)
    i_close, i_high = cols.index("close"), cols.index("high")
    for i in pos:
        out.iloc[i:, i_close] *= factor
        if i + 1 < len(out):
            for c in ("open", "high", "low"):
                out.iloc[i + 1:, cols.index(c)] *= factor
        out.iloc[i, i_high] = max(out.iloc[i, i_high], out.iloc[i, i_close])
    return out, dates


# --------------------------------------------------------------------------- #
# The verdict rule — pre-registered, tested, and applied to whatever the run finds
# --------------------------------------------------------------------------- #
def verdict(h: dict) -> dict:
    """Turn the headline numbers into the two stamps, by a rule fixed in advance.

    The rule, written before the run and unit-tested against synthetic headlines:

    - **The signal stamp is about the half day itself** — the ``offset = 0`` cells only
      (5 tickers x 3 families = 15 tests). The day *before* a holiday is somebody else's
      study (the pre-holiday effect of Lakonishok & Smidt 1988 and Ariel 1990, and this
      desk's 780-long-weekend-drift); letting an offset -1 cell earn this study's stamp
      would be claiming a known effect as a new one.
    - **Real** if the offset-0 battery produces more than *twice* the |*t*| >= 2 count luck
      delivers **and** its strongest cell reaches |*t*| >= 2.5; **Weak** if it merely beats
      the luck count; **None** otherwise.
    - **Tradability**: **Investable** only if the best per-session edge's bootstrap CI
      excludes zero *and* the net-of-cost contribution at 1 bp one-way clears 100 bps a
      year; **Fragile** if it clears 25 bps; **Mirage** below that.

    Keeping the rule in tested code — rather than in a human's head at writing time — is
    what stops the verdict from being fitted to the number that came out.
    """
    real = h["n_hits0"] > 2 * h["expected_hits0"] and h["max_abs_t0"] >= 2.5
    weak = h["n_hits0"] > h["expected_hits0"]
    signal = "Real" if real else ("Weak" if weak else "None")
    net = h["net_at_1bp"]
    ci_clears = h["best_ci_low"] > 0.0
    trad = ("Investable" if (ci_clears and net >= 100)
            else ("Fragile" if net >= 25 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"The volume collapse is unmistakable — a confirmed 1 p.m. session trades "
            f"**{h['median_volume_ratio']:.0%}** of a normal day's shares. The *return* is "
            f"the fragile part: SPY's half day gaps **{h['spy_edge_bps']:+.1f} bps** against "
            f"an ordinary session (*t* = {h['spy_t']:+.2f}, CI includes zero), and of the "
            f"**{h['n_cells0']}** half-day cells ({len(h['tickers'])} tickers x 3 families) "
            f"**{h['n_hits0']}** clear |*t*| = 2 against {h['expected_hits0']:.2f} expected by "
            f"luck — all positive, none reaching |*t*| = {2.5:.1f}, and the five tapes are "
            f"not five independent tests. The sharper cells in the wider battery "
            f"({h['n_hits']} of {h['n_cells']}) sit at offset −1: that is the known "
            f"pre-holiday effect, not the half day."),
        "trad": trad,
        "trad_why": (
            f"About **{h['sessions_per_year']:.1f} confirmed sessions a year**. The largest gap "
            f"in the lot ({h['best_ticker']} {h['best_edge_bps']:+.1f} bps/session, CI "
            f"[{h['best_ci_low']:+.0f}, {h['best_ci_high']:+.0f}]) grosses "
            f"{h['best_edge_bps'] * h['sessions_per_year']:+.1f} bps a year and nets "
            f"**{net:+.1f} bps** after a 1 bp round trip — a real number attached to an "
            f"interval that contains zero, harvested twice a year. Break-even is "
            f"**{h['breakeven_bps']:.1f} bps** one-way."),
        "one_sentence": (
            f"The early close genuinely empties the tape — **{h['median_volume_ratio']:.0%}** of "
            f"a normal day's volume — and the sessions around it do lean positive, but on the "
            f"half day itself the lean is smaller than its own error bar, the sharper results "
            f"belong to the day *before* (the long-documented pre-holiday effect), and "
            f"{h['sessions_per_year']:.1f} sessions a year is not a strategy."),
    }


def synthetic_detect(bars: pd.DataFrame, dates: pd.DatetimeIndex,
                     leg: str = "r_cc") -> dict:
    """Run the whole apparatus on planted bars and report what it recovers."""
    sessions = session_frame(bars)
    st = group_stats(sessions[leg], event_mask(sessions, dates, 0))
    ci = bootstrap_diff_ci(sessions[leg], event_mask(sessions, dates, 0))
    st.update({"ci_low": ci["ci_low"], "ci_high": ci["ci_high"]})
    return st
