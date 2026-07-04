"""Statistics for Study 608 — Friday News Dump.

Everything is numpy + pandas only, deterministic given a seed.

* ``assign_day0``   — acceptance timestamp -> the first trading session whose CLOSE
                      reflects the news (accepted before 16:00 ET on a trading day =
                      that day; otherwise the next session). Friday-after-close
                      filings therefore land on Monday.
* ``event_ars``     — per-event daily abnormal returns (stock minus SPY) for
                      sessions day 0..+10, as an (n x 11) matrix.
* ``group_stats``   — the DellaVigna-Pollet test: day-0 reaction and post-event
                      drift CAR[+1..+10], Friday vs Mon-Thu filers — Welch t on the
                      gap (the Signal-axis statistic), one-sample t per group,
                      1%-winsorized robustness.
* ``placebo_gap``   — label-permutation placebo on the drift gap (seeded).
* ``share_stats``   — the third axis: Friday / after-close / Friday-PM filing
                      shares, bad news vs the earnings control, two-proportion z.
* ``trade_stats``   — short-the-dump overlay: enter at the day-0 close (the single
                      documented execution lag — the news is public before that
                      close by construction), cover at close(+10), SPY-hedged;
                      costs one-way x 2 legs + borrow on the short.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 10  # drift horizon in sessions after day 0


# --------------------------------------------------------------------------- #
# Inference helpers (no scipy)
# --------------------------------------------------------------------------- #
def welch_t(a, b) -> float:
    """Welch's t for a mean difference between two independent samples."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return float((a.mean() - b.mean()) / np.sqrt(va + vb))


def one_sample_t(x) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def winsorize(x, p: float = 0.01) -> np.ndarray:
    x = np.asarray(x, float)
    lo, hi = np.quantile(x, [p, 1 - p])
    return np.clip(x, lo, hi)


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """Unpooled two-proportion z for share1 - share2; returns (p1, p2, z)."""
    p1, p2 = k1 / n1, k2 / n2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return float(p1), float(p2), float((p1 - p2) / se)


# --------------------------------------------------------------------------- #
# Event-time mapping and abnormal returns
# --------------------------------------------------------------------------- #
def assign_day0(panel: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.Series:
    """First session whose close reflects the filing (see module docstring)."""
    out = []
    for ts in panel["accept_ts"]:
        d = ts.normalize()
        i = int(calendar.searchsorted(d))
        if i < len(calendar) and calendar[i] == d and ts.hour < 16:
            out.append(calendar[i])
        else:
            j = int(calendar.searchsorted(d, side="right"))
            out.append(calendar[j] if j < len(calendar) else pd.NaT)
    return pd.Series(out, index=panel.index, name="day0")


def event_ars(panel: pd.DataFrame, prices: pd.DataFrame, mkt: str = "SPY",
              window: int = WINDOW, min_px: float = 1.0
              ) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-event abnormal returns day 0..+window (stock minus market, daily).

    Keeps only events whose full window (and the day-0 prior close) exists on the
    tape with no missing prices, and whose day(-1) close is >= ``min_px`` (the
    standard penny-stock screen: sub-$1 adjusted quotes carry data artifacts and
    bid-ask bounce measured in hundreds of percent — one sub-penny name in this
    panel "drifts" +32,000% in ten sessions). Returns the filtered panel + the
    (n x 11) matrix.
    """
    cal = prices.index[prices[mkt].notna()]
    px = prices.loc[cal]
    ret = px.pct_change()
    rmkt = ret[mkt].to_numpy()
    day0 = assign_day0(panel, cal)
    pos = {d: i for i, d in enumerate(cal)}
    keep_rows, ars = [], []
    for row, d0 in zip(panel.itertuples(), day0):
        if pd.isna(d0) or row.ticker not in ret.columns:
            continue
        i = pos[d0]
        if i - 1 < 0 or i + window >= len(cal):
            continue
        seg = ret[row.ticker].to_numpy()[i:i + window + 1] - rmkt[i:i + window + 1]
        p_1 = px[row.ticker].iloc[i - 1]
        if np.isnan(seg).any() or not np.isfinite(p_1) or p_1 < min_px:
            continue
        keep_rows.append(row.Index)
        ars.append(seg)
    kept = panel.loc[keep_rows].copy()
    kept["day0"] = day0.loc[keep_rows]
    return kept.reset_index(drop=True), np.asarray(ars)


# --------------------------------------------------------------------------- #
# The DellaVigna-Pollet drift test
# --------------------------------------------------------------------------- #
def group_stats(friday: np.ndarray, armat: np.ndarray, wins: float = 0.01) -> dict:
    """Day-0 reaction and drift CAR[+1..+10], Friday vs Mon-Thu (Welch t on gap)."""
    friday = np.asarray(friday, bool)
    ar0 = armat[:, 0]
    drift = armat[:, 1:].sum(axis=1)
    f, o = friday, ~friday
    dw = winsorize(drift, wins)
    return dict(
        n_fri=int(f.sum()), n_oth=int(o.sum()),
        ar0_fri_bps=ar0[f].mean() * 1e4, ar0_oth_bps=ar0[o].mean() * 1e4,
        ar0_t_fri=one_sample_t(ar0[f]), ar0_t_oth=one_sample_t(ar0[o]),
        ar0_gap_t=welch_t(ar0[f], ar0[o]),
        drift_fri_bps=drift[f].mean() * 1e4, drift_oth_bps=drift[o].mean() * 1e4,
        drift_t_fri=one_sample_t(drift[f]), drift_t_oth=one_sample_t(drift[o]),
        gap_bps=(drift[f].mean() - drift[o].mean()) * 1e4,
        gap_t=welch_t(drift[f], drift[o]),
        gap_t_wins=welch_t(dw[f], dw[o]),
        pooled_drift_bps=drift.mean() * 1e4, pooled_drift_t=one_sample_t(drift),
        pooled_drift_t_wins=one_sample_t(winsorize(drift, wins)),
        car010_fri_bps=armat[f].sum(axis=1).mean() * 1e4,
        car010_oth_bps=armat[o].sum(axis=1).mean() * 1e4,
        car010_gap_t=welch_t(armat[f].sum(axis=1), armat[o].sum(axis=1)),
    )


def placebo_gap(friday: np.ndarray, armat: np.ndarray, n_perm: int = 2000,
                seed: int = 608) -> float:
    """Label-permutation placebo: one-sided p that a random same-size 'Friday' set
    shows a drift gap at least as NEGATIVE as the observed one."""
    rng = np.random.default_rng(seed)
    friday = np.asarray(friday, bool)
    drift = armat[:, 1:].sum(axis=1)
    obs = drift[friday].mean() - drift[~friday].mean()
    k, hits = int(friday.sum()), 0
    for _ in range(n_perm):
        idx = rng.choice(len(drift), size=k, replace=False)
        m = np.zeros(len(drift), bool)
        m[idx] = True
        if drift[m].mean() - drift[~m].mean() <= obs:
            hits += 1
    return hits / n_perm


# --------------------------------------------------------------------------- #
# Third axis — do firms actually dump bad news on Friday (evening)?
# --------------------------------------------------------------------------- #
def friday_null_share(start: str, end: str) -> float:
    """Share of Fridays among EDGAR-open days (weekdays net of US federal
    holidays) over [start, end] — the uniform-timing null for the Friday share."""
    from pandas.tseries.holiday import USFederalHolidayCalendar

    days = pd.bdate_range(start, end)
    hol = USFederalHolidayCalendar().holidays(days[0], days[-1])
    days = days[~days.isin(hol)]
    return float((days.dayofweek == 4).mean())


def share_stats(bad: pd.DataFrame, ctrl: pd.DataFrame) -> dict:
    """Friday / after-close / Friday-PM shares, bad vs control, two-prop z —
    plus a binomial z of the bad-news Friday share against the uniform-calendar
    null (the share of Fridays among EDGAR-open days)."""
    out = dict(n_bad=len(bad), n_ctrl=len(ctrl))
    for flag in ("friday", "after_close", "friday_pm"):
        p1, p2, z = two_prop_z(int(bad[flag].sum()), len(bad),
                               int(ctrl[flag].sum()), len(ctrl))
        out[flag] = dict(bad=p1, ctrl=p2, z=z)
    p0 = friday_null_share(str(bad["accept_ts"].min().date()),
                           str(bad["accept_ts"].max().date()))
    p1 = bad["friday"].mean()
    out["friday_null"] = float(p0)
    out["friday_vs_null_z"] = float(
        (p1 - p0) / np.sqrt(p0 * (1 - p0) / len(bad)))
    wd = bad["weekday"].value_counts(normalize=True).reindex(range(5)).fillna(0)
    out["weekday_shares_bad"] = [float(wd[i]) for i in range(5)]
    wdc = ctrl["weekday"].value_counts(normalize=True).reindex(range(5)).fillna(0)
    out["weekday_shares_ctrl"] = [float(wdc[i]) for i in range(5)]
    return out


# --------------------------------------------------------------------------- #
# Tradability — short the Friday dump
# --------------------------------------------------------------------------- #
def trade_stats(panel: pd.DataFrame, armat: np.ndarray, flag: str = "friday",
                cost_bps_side: float = 10.0, borrow_apr: float = 0.05,
                mkt_cost_bps_side: float = 1.0) -> dict:
    """Short each flagged name at the day-0 close, cover at close(+10), SPY-hedged.

    Execution lag (the ONE documented lag): the filing is public before the day-0
    close by construction of ``assign_day0``; entry is AT that close, so the day-0
    reaction is never captured — only the drift window [+1..+10]. Gross per event =
    -CAR[+1..+10]. Costs: ``cost_bps_side`` one-way x 2 stock legs +
    ``mkt_cost_bps_side`` x 2 hedge legs + borrow at ``borrow_apr`` for 10 sessions
    (shorts pay borrow; many of these names are small-cap hard-to-borrow — 5%/yr is
    the honest generic HTB level, shown at 2%/yr as robustness by the caller).
    """
    m = panel[flag].to_numpy(bool)
    gross = -armat[m, 1:].sum(axis=1)
    borrow = borrow_apr * WINDOW / 252.0
    cost = (2 * cost_bps_side + 2 * mkt_cost_bps_side) / 1e4 + borrow
    net = gross - cost
    yrs = (panel["accept_ts"].max() - panel["accept_ts"].min()).days / 365.25
    return dict(
        n=int(m.sum()), per_year=m.sum() / yrs,
        gross_bps=gross.mean() * 1e4, t_gross=one_sample_t(gross),
        hit=float((gross > 0).mean()),
        cost_bps=cost * 1e4, net_bps=net.mean() * 1e4, t_net=one_sample_t(net),
    )
