"""Tested rules + statistics for Study 605 — VIX Settlement Day.

The event unit is a DAY, and the groups are (monthly VIX final-settlement days) vs (all other
Wednesdays). Days are non-overlapping single-day observations pooled across 22.5 years, so the
group splits use **Welch t** and the regressions use **White (HC1) heteroskedasticity-robust
t** — the appropriate robust statistics for independent, fat-tailed daily events (no
overlapping-window serial correlation to Newey-West away).

Metrics per day (from ^VIX OHLC, all in log units of the index level):

* ``gap``   = ln(Open_t / Close_{t-1})   — the overnight jump into the settlement morning
* ``intr``  = ln(Close_t / Open_t)       — how the session digests the open
* ``cc``    = ln(Close_t / Close_{t-1})  — the whole day
* ``rng``   = (High - Low) / Close_{t-1} — intraday whipsaw
* ``spx``   = ln(SPX Close_t / Close_{t-1})

The distinctive Griffin-Shams signature is STRUCTURAL, not a level shift: on normal days VIX
opening gaps *mean-revert* intraday; if the settlement auction pushes the morning print through
the actual option book, the distorted open should *carry into the day* — a positive
settlement x gap **interaction** in ``intr = a + b*gap + c*sett + d*(gap*sett)``.

The placebo re-draws one fake "settlement" Wednesday per month at random and re-runs the same
interaction regression; it is averaged over >= 20 seeds (25 seeds x 80 draws = 2,000 draws).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as _data


# --------------------------------------------------------------------------- #
# Day frame
# --------------------------------------------------------------------------- #
def day_frame(vix: pd.DataFrame, spx: pd.DataFrame | None = None,
              settlements: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Per-day metrics + settlement / Wednesday / FOMC flags."""
    if settlements is None:
        settlements = _data.settlement_calendar(str(vix.index.min().date()),
                                                str(vix.index.max().date()))
    out = pd.DataFrame({
        "gap": np.log(vix["Open"] / vix["Close"].shift()),
        "intr": np.log(vix["Close"] / vix["Open"]),
        "cc": np.log(vix["Close"] / vix["Close"].shift()),
        "rng": (vix["High"] - vix["Low"]) / vix["Close"].shift(),
    })
    if spx is not None:
        out["spx"] = np.log(spx["Close"] / spx["Close"].shift()).reindex(out.index)
    out = out.dropna(subset=["gap", "intr", "cc", "rng"])
    out["sett"] = out.index.isin(settlements)
    out["wed"] = out.index.weekday == 2
    out["fomc"] = out.index.isin(_data.FOMC_DATES)
    return out


def event_sample(df: pd.DataFrame, ex_fomc: bool = False) -> pd.DataFrame:
    """Settlement days (incl. the holiday-shifted Tuesdays) + all other Wednesdays."""
    d = df[~df["fomc"]] if ex_fomc else df
    return d[d["sett"] | (d["wed"] & ~d["sett"])]


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def welch_t(a, b) -> float:
    """Welch two-sample t (unequal variances)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return float((a.mean() - b.mean()) / np.sqrt(va + vb))


def ols_white(cols: list[np.ndarray], y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """OLS with White (HC1) robust t-stats; returns (betas, t-stats, n). Intercept first."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in cols])
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    n, k = X.shape
    meat = (X * (e ** 2)[:, None]).T @ X
    V = XtXi @ meat @ XtXi * n / (n - k)
    return b, b / np.sqrt(np.diag(V)), n


# --------------------------------------------------------------------------- #
# Level tests — the planned Welch splits
# --------------------------------------------------------------------------- #
def level_tests(df: pd.DataFrame, ex_fomc: bool = False) -> dict:
    """Settlement days vs other Wednesdays: mean and |.| of each metric, Welch t."""
    d = event_sample(df, ex_fomc=ex_fomc)
    s, o = d[d["sett"]], d[~d["sett"]]
    out = {"n_sett": len(s), "n_other": len(o)}
    for col in ("gap", "intr", "cc", "rng", "spx"):
        if col not in d.columns:
            continue
        sc, oc = s[col].dropna(), o[col].dropna()
        out[col] = dict(sett=float(sc.mean()), other=float(oc.mean()),
                        t=welch_t(sc, oc))
        if col != "rng":
            out["abs_" + col] = dict(sett=float(sc.abs().mean()),
                                     other=float(oc.abs().mean()),
                                     t=welch_t(sc.abs(), oc.abs()))
    return out


# --------------------------------------------------------------------------- #
# The signature — settlement x gap continuation interaction
# --------------------------------------------------------------------------- #
def interaction(df: pd.DataFrame, ex_fomc: bool = False,
                drop_zero_gaps: bool = False, winsor: float | None = None) -> dict:
    """intr = a + b*gap + c*sett + d*(gap x sett) on settlement days + other Wednesdays.

    ``d`` > 0 means settlement-day gaps CONTINUE into the close relative to the normal-
    Wednesday baseline ``b`` (which is negative: gaps normally mean-revert). White (HC1) t.
    ``drop_zero_gaps`` removes days whose Yahoo open exactly equals the prior close (a known
    stale-open artefact, ~4% of days, ~30% in 2008-09). ``winsor`` clips gap and intr at the
    given two-sided quantile (e.g. 0.01) to show the tails aren't driving the result.
    """
    d = event_sample(df, ex_fomc=ex_fomc)
    if drop_zero_gaps:
        d = d[d["gap"] != 0.0]
    g, i = d["gap"].to_numpy(), d["intr"].to_numpy()
    if winsor:
        g = np.clip(g, np.quantile(g, winsor), np.quantile(g, 1 - winsor))
        i = np.clip(i, np.quantile(i, winsor), np.quantile(i, 1 - winsor))
    s = d["sett"].astype(float).to_numpy()
    b, t, n = ols_white([g, s, g * s], i)
    return dict(n=n, n_sett=int(s.sum()),
                base_slope=float(b[1]), base_t=float(t[1]),
                sett_dummy=float(b[2]), sett_dummy_t=float(t[2]),
                inter=float(b[3]), inter_t=float(t[3]),
                sett_slope=float(b[1] + b[3]))


def placebo_interaction(df: pd.DataFrame, ex_fomc: bool = True,
                        n_seeds: int = 25, n_per: int = 80,
                        base_seed: int = 605) -> dict:
    """Random-calendar placebo for the interaction, averaged over >= 20 seeds.

    Restricted to NON-settlement Wednesdays, one fake settlement drawn per calendar month;
    the same interaction regression is re-run per draw. Returns the placebo distribution
    stats and the two-sided p of the observed interaction.
    """
    obs = interaction(df, ex_fomc=ex_fomc)["inter"]
    d = event_sample(df, ex_fomc=ex_fomc)
    wed = d[~d["sett"]]
    codes = pd.PeriodIndex(wed.index, freq="M").astype(str)
    groups = [np.where(codes == k)[0] for k in pd.unique(codes)]
    g, i = wed["gap"].to_numpy(), wed["intr"].to_numpy()
    draws = np.empty(n_seeds * n_per)
    q = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(base_seed + seed)
        for _ in range(n_per):
            f = np.zeros(len(wed))
            for idx in groups:
                f[idx[rng.integers(len(idx))]] = 1.0
            b, _, _ = ols_white([g, f, g * f], i)
            draws[q] = b[3]
            q += 1
    return dict(obs=float(obs), mean=float(draws.mean()), sd=float(draws.std()),
                p_two_sided=float((np.abs(draws) >= abs(obs)).mean()),
                n_draws=len(draws), n_seeds=n_seeds)


def subperiod_interactions(df: pd.DataFrame, split: str = _data.PAPER_SPLIT,
                           ex_fomc: bool = True) -> dict:
    """The signature before vs after the Griffin-Shams publication/lawsuit year."""
    pre = interaction(df[df.index < split], ex_fomc=ex_fomc)
    post = interaction(df[df.index >= split], ex_fomc=ex_fomc)
    return dict(pre=pre, post=post)


def subperiod_levels(df: pd.DataFrame, split: str = _data.PAPER_SPLIT) -> dict:
    """|cc| level effect (with and without FOMC) before vs after the split — the fade check."""
    out = {}
    for tag, ex in (("all", False), ("ex_fomc", True)):
        d_pre = event_sample(df[df.index < split], ex_fomc=ex)
        d_post = event_sample(df[df.index >= split], ex_fomc=ex)
        out[tag] = {}
        for lbl, d in (("pre", d_pre), ("post", d_post)):
            s, o = d[d["sett"]], d[~d["sett"]]
            out[tag][lbl] = dict(sett=float(s["cc"].abs().mean()),
                                 other=float(o["cc"].abs().mean()),
                                 t=welch_t(s["cc"].abs(), o["cc"].abs()),
                                 n_sett=len(s))
    return out


# --------------------------------------------------------------------------- #
# Tradability arithmetic — the honest cash-out attempt
# --------------------------------------------------------------------------- #
def continuation_pnl(df: pd.DataFrame, ex_fomc: bool = False,
                     cost_bps_oneway: float = 25.0) -> dict:
    """Ride the settlement-morning gap open->close, in INDEX log units.

    Rule: at the settlement-day open (one clean lag: the calendar is known years ahead and the
    signal is the just-printed open vs yesterday's close), take the gap's direction; exit at
    the close. Position = sign(gap): P&L_t = sign(gap_t) * intr_t. The same rule on other
    Wednesdays is the baseline. THE INDEX IS NOT TRADABLE — this is the paper value of the
    signature; costs are charged per round trip (2 x one-way, in % of notional) at VIX-futures
    retail levels to show what survives even before the futures-vs-spot basis wedge.
    """
    d = event_sample(df, ex_fomc=ex_fomc)
    d = d[d["gap"] != 0.0]
    pnl = np.sign(d["gap"]) * d["intr"]
    s, o = pnl[d["sett"]], pnl[~d["sett"]]
    gross = float(s.mean())
    net = gross - 2.0 * cost_bps_oneway / 1e4
    return dict(n=len(s), gross_per_event=gross, net_per_event=net,
                other_wed=float(o.mean()), t_vs_zero=float(s.mean() / s.sem()),
                t_vs_otherwed=welch_t(s, o),
                events_per_year=12.0, net_per_year=net * 12.0)
