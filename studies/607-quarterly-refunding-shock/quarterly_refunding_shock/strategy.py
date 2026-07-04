"""Event-study machinery for Study 607 — Quarterly Refunding Shock.

The claim splits into three testable pieces:

* **Volatility (the "macro event" claim).** If the QRA moves the long end, the 10Y
  yield's absolute day-0 change on QRA days must exceed ordinary days. Primary test:
  Welch *t* of |Δy| on **FOMC-clean** QRA days vs an FOMC-clean, event-free baseline —
  crucial because ~¼ of QRA Wednesdays are ALSO FOMC statement days (2023-11-01
  included), and daily bars cannot split an 08:30 statement from a 14:00 one. Events
  sit ~63 trading days apart, so event-day observations are serially uncorrelated and
  the plain Welch *t* is the appropriate robust statistic (no overlapping windows —
  nothing for HAC to fix). A random-calendar placebo (2,000 same-size draws from the
  baseline days) re-asks the question without distributional assumptions.

* **Direction.** Supply surprises cut both ways, so the signed day-0 mean should be ~0;
  a one-sample *t* on signed Δy checks there is no exploitable drift *on* the day.

* **Tradability.** The announcement lands at 08:30 ET; the first honest fill is the
  day-0 close (the ONE execution lag, documented everywhere). We test an unconditional
  3-day TLT hold (close day-0 → close day+3) and a conditional continuation version
  (ride the sign of the day-0 TLT move), net of one-way costs × 2 legs per event.

Era analysis (the third axis): vol-normalise each day's |Δy| by the trailing 60-day
mean |Δy| (ending the day before), so a 2023 QRA day is judged against 2023's own noise
level, not 2004's — then Welch the QRA-day normalised moves across eras (2000-2022 vs
2023+), with the honest small-n caveat and the drop-the-two-loud-days rerun.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_OFFSETS = (-1, 0, 1, 2, 3)
ERA_SPLIT = "2023-01-01"           # the "QRA became a macro event" era
NORM_LOOKBACK = 60                 # trailing days for the vol normalisation


# --------------------------------------------------------------------------- #
# Basic helpers
# --------------------------------------------------------------------------- #
def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b), unequal variances. NaN if either side < 2."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def ttest_vs_zero(x: np.ndarray) -> float:
    """One-sample t of x against 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def dy_bps(level: pd.Series) -> pd.Series:
    """Daily 10Y yield change in basis points from a yield-in-percent level series."""
    return level.diff().mul(100.0).dropna()


def event_positions(index: pd.DatetimeIndex, dates: list[str],
                    max_slip: int = 3) -> list[int]:
    """Map ISO event dates to integer positions on the tape (first session >= the date).

    QRA statements land at 08:30 ET on trading Wednesdays, so day-0 is normally an exact
    hit; the >= mapping only guards against a missing bar. Events more than ``max_slip``
    sessions away from a bar (i.e. before the tape starts / after it ends) are dropped.
    """
    out = []
    for d in dates:
        ts = pd.Timestamp(d)
        i = int(index.searchsorted(ts))
        if i >= len(index):
            continue
        if (index[i] - ts).days > max_slip:
            continue
        out.append(i)
    return sorted(set(out))


def clean_split(index: pd.DatetimeIndex, qra_dates: list[str],
                fomc_dates: list[str]) -> dict:
    """Split QRA day-0 positions into FOMC-clean vs FOMC-overlap; build the baseline mask.

    Baseline days exclude: every QRA window day (offsets -1..+3), every FOMC statement
    day, and the first NORM_LOOKBACK sessions (so the vol-normalised series exists
    everywhere it is used).
    """
    qra_pos = event_positions(index, qra_dates)
    fomc_pos = set(event_positions(index, fomc_dates))
    clean = [p for p in qra_pos if p not in fomc_pos]
    overlap = [p for p in qra_pos if p in fomc_pos]

    base = np.ones(len(index), dtype=bool)
    base[:NORM_LOOKBACK] = False
    for p in qra_pos:
        lo, hi = max(0, p - 1), min(len(index), p + 4)
        base[lo:hi] = False
    for p in fomc_pos:
        base[p] = False
    return {"qra_pos": qra_pos, "clean_pos": clean, "overlap_pos": overlap,
            "fomc_pos": sorted(fomc_pos), "base_mask": base}


# --------------------------------------------------------------------------- #
# Day-0 volatility + direction (the Signal axis)
# --------------------------------------------------------------------------- #
def day0_stats(dy: pd.Series, event_pos: list[int], base_mask: np.ndarray) -> dict:
    """|move| and signed-move day-0 statistics for a set of event positions.

    ``dy`` must be aligned to the same index the positions were computed on (pass the
    level's index positions through :func:`align_positions` first — see verify.py).
    """
    v = dy.values
    ev = np.asarray(event_pos, dtype=int)
    ev = ev[(ev >= 0) & (ev < len(v))]
    e = v[ev]
    b = v[base_mask[:len(v)]]
    e, b = e[~np.isnan(e)], b[~np.isnan(b)]
    return {
        "n_event": int(len(e)), "n_base": int(len(b)),
        "abs_event": float(np.abs(e).mean()), "abs_base": float(np.abs(b).mean()),
        "abs_ratio": float(np.abs(e).mean() / np.abs(b).mean()),
        "welch_abs": welch_t(np.abs(e), np.abs(b)),
        "signed_mean": float(e.mean()), "t_signed": ttest_vs_zero(e),
    }


def align_positions(level_index: pd.DatetimeIndex, dy_index: pd.DatetimeIndex,
                    pos: list[int]) -> list[int]:
    """Translate positions on the level index into positions on the (diff'd) dy index."""
    dates = level_index[pos]
    keep = dy_index.get_indexer(dates)
    return [int(i) for i in keep if i >= 0]


def window_profile(dy: pd.Series, event_pos: list[int], base_mask: np.ndarray,
                   offsets: tuple = WINDOW_OFFSETS) -> list[dict]:
    """Mean |Δy| (and Welch t vs baseline) at each event offset in [-1..+3]."""
    v = dy.values
    b = v[base_mask[:len(v)]]
    b = b[~np.isnan(b)]
    out = []
    for k in offsets:
        idx = [p + k for p in event_pos if 0 <= p + k < len(v)]
        e = v[np.asarray(idx, dtype=int)]
        e = e[~np.isnan(e)]
        out.append({"offset": int(k), "n": int(len(e)),
                    "abs_mean": float(np.abs(e).mean()),
                    "abs_base": float(np.abs(b).mean()),
                    "welch_abs": welch_t(np.abs(e), np.abs(b)),
                    "signed_mean": float(e.mean()),
                    "t_signed": ttest_vs_zero(e)})
    return out


def day2_jobs_diagnostic(dy: pd.Series, event_pos: list[int],
                         base_mask: np.ndarray) -> dict:
    """Is the day+2 'aftershock' just the jobs report? (the calendar collision check)

    QRA lands on an early-Feb/May/Aug/Nov Wednesday; the BLS Employment Situation lands
    on the FIRST FRIDAY of the month at 08:30 ET — which is day+2 of most QRA windows.
    We flag day+2 sessions that are first-Fridays (weekday Friday, day-of-month <= 7,
    the standard Employment-Situation slot) and re-test day+2 |Δy| without them.
    """
    v = dy.values
    dates = dy.index
    b = v[base_mask[:len(v)]]
    b = b[~np.isnan(b)]
    d2 = [p + 2 for p in event_pos if 0 <= p + 2 < len(v)]
    is_ff = np.array([dates[p].weekday() == 4 and dates[p].day <= 7 for p in d2])
    all_abs = np.abs(v[np.asarray(d2, dtype=int)])
    keep = np.asarray(d2, dtype=int)[~is_ff]
    kept_abs = np.abs(v[keep])
    ff_abs = np.abs(v[np.asarray(d2, dtype=int)[is_ff]]) if is_ff.any() else np.array([])
    return {
        "n_day2": len(d2), "n_first_friday": int(is_ff.sum()),
        "share_first_friday": float(is_ff.mean()),
        "abs_all": float(all_abs.mean()), "welch_all": welch_t(all_abs, np.abs(b)),
        "abs_ff": float(ff_abs.mean()) if len(ff_abs) else float("nan"),
        "abs_ex_ff": float(kept_abs.mean()) if len(kept_abs) else float("nan"),
        "welch_ex_ff": welch_t(kept_abs, np.abs(b)),
        "n_ex_ff": int(len(kept_abs)),
    }


def placebo_pvalue(dy: pd.Series, event_pos: list[int], base_mask: np.ndarray,
                   n_draws: int = 2000, seed: int = 607) -> dict:
    """Random-calendar placebo for the day-0 |move|.

    Draw ``len(event_pos)`` pseudo-event days from the baseline days, ``n_draws`` times
    (2,000 draws ≫ the ≥20-seed floor for any random baseline), and ask how often a
    random calendar's mean |Δy| beats the observed QRA-day mean |Δy|.
    """
    rng = np.random.default_rng(seed)
    v = dy.values
    ev = np.asarray(event_pos, dtype=int)
    obs = float(np.nanmean(np.abs(v[ev])))
    cand = np.flatnonzero(base_mask[:len(v)])
    cand = cand[~np.isnan(v[cand])]
    k = len(ev)
    draws = np.empty(n_draws)
    for d in range(n_draws):
        sel = rng.choice(cand, size=k, replace=False)
        draws[d] = np.abs(v[sel]).mean()
    return {"obs": obs, "p_value": float((draws >= obs).mean()),
            "draw_mean": float(draws.mean()), "draws": draws}


# --------------------------------------------------------------------------- #
# Vol-normalised era comparison (the third axis)
# --------------------------------------------------------------------------- #
def normalized_abs(dy: pd.Series, lookback: int = NORM_LOOKBACK) -> pd.Series:
    """|Δy| divided by the trailing ``lookback``-day mean |Δy| ending the DAY BEFORE.

    Judges each day against its own era's noise level: 1.0 = an ordinary day for the
    period, 3.0 = three times the local noise. The shift(1) keeps day-0 out of its own
    denominator (no contamination)."""
    a = dy.abs()
    base = a.rolling(lookback).mean().shift(1)
    return (a / base).dropna()


def era_compare(dy: pd.Series, event_pos: list[int], base_mask: np.ndarray,
                split: str = ERA_SPLIT, drop_dates: list[str] | None = None) -> dict:
    """QRA-day vol-normalised |move| in the pre-2023 vs 2023+ eras, Welch'd.

    ``drop_dates`` re-runs the late era without specific loud days (the "two loud data
    points" check). Also reports each era's raw excess ratio vs its own-era baseline.
    """
    norm = normalized_abs(dy)
    v = dy.values
    dates = dy.index
    split_ts = pd.Timestamp(split)
    drop = {pd.Timestamp(d) for d in (drop_dates or [])}

    def _norm_at(pos_list):
        vals, kept = [], []
        for p in pos_list:
            d = dates[p]
            if d in norm.index:
                vals.append(float(norm.loc[d]))
                kept.append(d)
        return np.asarray(vals), kept

    early = [p for p in event_pos if dates[p] < split_ts]
    late = [p for p in event_pos if dates[p] >= split_ts]
    late_trim = [p for p in late if dates[p] not in drop]

    ne, _ = _norm_at(early)
    nl, late_dates = _norm_at(late)
    nt, _ = _norm_at(late_trim)

    base_pos = np.flatnonzero(base_mask[:len(v)])
    base_dates = dates[base_pos]
    nb = norm.reindex(base_dates).dropna()
    nb_early = nb[nb.index < split_ts].values
    nb_late = nb[nb.index >= split_ts].values

    def _ratio(ev_abs_pos, b_vals):
        e = np.abs(v[np.asarray(ev_abs_pos, dtype=int)]) if len(ev_abs_pos) else np.array([])
        return float(e.mean() / b_vals.mean()) if len(e) and len(b_vals) else float("nan")

    b_early_abs = np.abs(v[base_pos[base_dates < split_ts]])
    b_late_abs = np.abs(v[base_pos[base_dates >= split_ts]])
    return {
        "n_early": len(ne), "n_late": len(nl), "n_late_trim": len(nt),
        "norm_early": float(ne.mean()), "norm_late": float(nl.mean()),
        "norm_late_trim": float(nt.mean()) if len(nt) else float("nan"),
        "norm_base_early": float(nb_early.mean()), "norm_base_late": float(nb_late.mean()),
        "welch_late_vs_early": welch_t(nl, ne),
        "welch_trim_vs_early": welch_t(nt, ne),
        "welch_late_vs_own_base": welch_t(nl, nb_late),
        "welch_early_vs_own_base": welch_t(ne, nb_early),
        "ratio_early": _ratio(early, b_early_abs), "ratio_late": _ratio(late, b_late_abs),
        "late_dates": [d.strftime("%Y-%m-%d") for d in late_dates],
        "late_norm_values": [float(x) for x in nl],
    }


# --------------------------------------------------------------------------- #
# Tradability — TLT after the announcement (ONE lag: entry at the day-0 close)
# --------------------------------------------------------------------------- #
def tlt_event_trades(tlt: pd.Series, qra_dates: list[str], hold: int = 3) -> dict:
    """Post-QRA TLT holds, unconditional and sign-conditional.

    Entry at the **day-0 close** (the announcement is out at 08:30, so the close is the
    first honest fill — the single documented execution lag), exit at the close ``hold``
    sessions later. Event windows are non-overlapping (~63 sessions apart vs hold=3), so
    plain one-sample *t*s are valid. The conditional leg rides the SIGN of the day-0 TLT
    move (continuation): position = sign(r0), same window.

    The baseline for the unconditional Welch is every non-overlapping ``hold``-day
    return outside the event windows (stride = hold, so no overlap inflation).
    """
    lr = np.log(tlt).diff().dropna()
    idx = lr.index
    pos = event_positions(pd.DatetimeIndex(tlt.index), qra_dates)
    # translate to lr positions (lr is tlt.index minus the first bar)
    pos = align_positions(pd.DatetimeIndex(tlt.index), idx, pos)

    ev_ret, cond_ret, used = [], [], []
    for p in pos:
        if p + hold >= len(lr):
            continue
        w = lr.values[p + 1: p + 1 + hold]          # entry day-0 close -> exit day+hold close
        r = float(w.sum())
        r0 = float(lr.values[p])                    # the day-0 move (known at the fill)
        ev_ret.append(r)
        cond_ret.append(np.sign(r0) * r if r0 != 0 else 0.0)
        used.append(p)

    # non-overlapping baseline windows outside events
    excl = np.zeros(len(lr), dtype=bool)
    for p in used:
        excl[max(0, p - 1): p + 1 + hold] = True
    base = []
    i = 0
    while i + hold < len(lr):
        if not excl[i: i + 1 + hold].any():
            base.append(float(lr.values[i + 1: i + 1 + hold].sum()))
            i += hold
        else:
            i += 1
    ev, cd, ba = np.asarray(ev_ret), np.asarray(cond_ret), np.asarray(base)
    years = (idx[-1] - idx[0]).days / 365.25
    per_year = len(ev) / years
    return {
        "n_events": int(len(ev)), "hold": hold, "per_year": float(per_year),
        "uncond_mean_bps": float(ev.mean() * 1e4), "uncond_t": ttest_vs_zero(ev),
        "base_mean_bps": float(ba.mean() * 1e4), "n_base": int(len(ba)),
        "welch_vs_base": welch_t(ev, ba),
        "cond_mean_bps": float(cd.mean() * 1e4), "cond_t": ttest_vs_zero(cd),
    }


def tlt_net(stats: dict, cost_bps_oneway: float) -> dict:
    """Net the per-event mean of a TLT leg: 2 legs (in + out) x one-way cost per event."""
    cost = 2.0 * cost_bps_oneway
    net_u = stats["uncond_mean_bps"] - cost
    net_c = stats["cond_mean_bps"] - cost
    return {
        "cost_bps_oneway": cost_bps_oneway,
        "uncond_net_bps": net_u, "uncond_net_ann_bps": net_u * stats["per_year"],
        "cond_net_bps": net_c, "cond_net_ann_bps": net_c * stats["per_year"],
    }
