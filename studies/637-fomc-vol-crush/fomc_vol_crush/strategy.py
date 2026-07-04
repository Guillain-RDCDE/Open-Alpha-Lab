"""Strategy + inference for Study 637 — FOMC Vol Crush.

The claim: **the VIX collapses the afternoon of a scheduled Fed decision** — the statement
lands mid-afternoon ET (2:15 pm, 2:00 pm since 2013), the ^VIX close prints at 4:15 pm, so the
decision-day close-to-close change should show a sharp, systematic drop: scheduled uncertainty
resolving on a calendar you can set your watch to.

Measurements:

* **Decision-day change vs all other days** — ΔVIX in points (close − prev close) and in log
  terms, Welch *t* for the group split (single-day, non-overlapping events; Welch is the
  planned primary). A Newey-West *t* on the decision-day dummy regression is reported as the
  serial-correlation-robust cross-check, plus a hit rate with a Wilson interval and a
  20,000-draw random-calendar placebo (20 seeds x 1,000 draws).
* **Event window [-5..+3]** — the pre-meeting run-up (does the VIX *build* into the decision?)
  and the post-day persistence (does the crush stick or bounce back?). Per-offset Welch *t*
  vs far-from-meeting days.
* **Realized SPY high-low range** on the same days — is the decision day actually a *calm* day
  (which would make the VIX drop trivial) or a loud day whose *implied* vol still collapses?
* **Third axis (retail capture)** — SVXY held for the decision day only (enter the prior
  close: the calendar is public years ahead, so this is a zero-look-ahead scheduled entry;
  exit the decision-day close). Gross/net at one-way costs x 2 per event, Welch *t* vs all
  other SVXY days.

The decisive number is the decision-day Welch *t* on the REAL ^VIX tape; the honest question
is whether any tradable instrument actually pays you for it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Day frame
# --------------------------------------------------------------------------- #
def day_frame(vix: pd.DataFrame, spy: pd.DataFrame | None,
              fomc: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per ^VIX trading day: ΔVIX (points + log), SPY range, and the FOMC flag."""
    df = pd.DataFrame(index=vix.index)
    df["vix_close"] = vix["Close"]
    df["dvix_pts"] = vix["Close"].diff()
    df["dvix_log"] = np.log(vix["Close"]).diff()
    if spy is not None:
        spy = spy.reindex(vix.index)
        df["spy_range"] = (spy["High"] - spy["Low"]) / spy["Close"].shift(1)
        df["spy_ret"] = np.log(spy["AdjClose"]).diff()
    df["fomc"] = df.index.isin(fomc)
    return df.dropna(subset=["dvix_pts"])


def event_offsets(df: pd.DataFrame, fomc: pd.DatetimeIndex,
                  lo: int = -5, hi: int = 3) -> pd.Series:
    """Business-day offset of each tape day relative to the nearest FOMC decision day.

    Offset 0 = the decision day itself, -1 = the session before, +1 = the session after, etc.
    Days farther than [lo, hi] from every meeting get NaN (the "far" control group). Meetings
    are >= 5 weeks apart, so windows of +-5 sessions never collide.
    """
    idx = df.index
    off = pd.Series(np.nan, index=idx)
    pos_of = {d: i for i, d in enumerate(idx)}
    for d in fomc:
        if d not in pos_of:
            continue
        p = pos_of[d]
        for k in range(lo, hi + 1):
            q = p + k
            if 0 <= q < len(idx):
                off.iloc[q] = k
    return off


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances). NaN if either < 2 obs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(y: np.ndarray, d: np.ndarray, lags: int = 5) -> float:
    """HAC (Newey-West, Bartlett kernel) t of the slope in y = a + b*d.

    b is exactly the treated-minus-rest mean difference; the NW t is the
    serial-correlation-robust cross-check for the daily ΔVIX series.
    """
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=float)
    keep = ~np.isnan(y)
    y, d = y[keep], d[keep]
    n = len(y)
    X = np.column_stack([np.ones(n), d])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    s = X * u[:, None]                       # score contributions (n x 2)
    S = s.T @ s
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = s[l:].T @ s[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(V[1, 1])
    return float(beta[1] / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# The headline split
# --------------------------------------------------------------------------- #
def decision_day_stats(df: pd.DataFrame, nw_lags: int = 5) -> dict:
    """Decision-day ΔVIX vs all-other-days: means, Welch t, NW t, hit rate + Wilson CI."""
    f = df["fomc"].values
    pts = df["dvix_pts"].values
    lg = df["dvix_log"].values
    a_pts, b_pts = pts[f], pts[~f]
    a_lg, b_lg = lg[f], lg[~f]
    k_down = int((a_pts < 0).sum())
    lo, hi = wilson_interval(k_down, len(a_pts))
    return {
        "n_fomc": int(f.sum()), "n_rest": int((~f).sum()),
        "fomc_pts": float(np.nanmean(a_pts)), "rest_pts": float(np.nanmean(b_pts)),
        "gap_pts": float(np.nanmean(a_pts) - np.nanmean(b_pts)),
        "welch_t_pts": welch_t(a_pts, b_pts),
        "fomc_logpct": float(np.nanmean(a_lg) * 100),
        "rest_logpct": float(np.nanmean(b_lg) * 100),
        "welch_t_log": welch_t(a_lg, b_lg),
        "nw_t_pts": newey_west_t(pts, f.astype(float), lags=nw_lags),
        "hit_down": k_down, "hit_rate": k_down / len(a_pts),
        "hit_lo": lo, "hit_hi": hi,
    }


def placebo_pvalue(df: pd.DataFrame, n_draws_per_seed: int = 1_000,
                   n_seeds: int = 20, base_seed: int = 637) -> dict:
    """Random-calendar placebo: draw |FOMC| random non-FOMC days, mean ΔVIX (points).

    p = share of draws whose mean is <= the observed FOMC-day mean (a LEFT-tail test — the
    claim is a drop). Averaged over ``n_seeds`` independent seeds x ``n_draws_per_seed`` draws
    so no single lucky stream decides it.
    """
    f = df["fomc"].values
    pts = df["dvix_pts"].values
    obs = float(np.nanmean(pts[f]))
    pool = pts[~f]
    pool = pool[~np.isnan(pool)]
    k = int(f.sum())
    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            means.append(pool[rng.choice(len(pool), size=k, replace=False)].mean())
    means = np.asarray(means)
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)),
            "p_value": float((means <= obs).mean()),
            "n_draws": len(means), "draws": means}


# --------------------------------------------------------------------------- #
# Event window: run-up and persistence
# --------------------------------------------------------------------------- #
def event_study(df: pd.DataFrame, fomc: pd.DatetimeIndex,
                lo: int = -5, hi: int = 3) -> pd.DataFrame:
    """Mean ΔVIX (points) by event offset, Welch t vs far-from-meeting days."""
    off = event_offsets(df, fomc, lo, hi)
    far = df.loc[off.isna(), "dvix_pts"].values
    rows = []
    for k in range(lo, hi + 1):
        x = df.loc[off == k, "dvix_pts"].values
        rows.append({"offset": k, "n": len(x), "mean_pts": float(np.nanmean(x)),
                     "welch_t": welch_t(x, far)})
    return pd.DataFrame(rows).set_index("offset")


def runup_stats(df: pd.DataFrame, fomc: pd.DatetimeIndex) -> dict:
    """Cumulative ΔVIX over the pre-meeting window [-5..-1] per meeting, one-sample t."""
    off = event_offsets(df, fomc, -5, 3)
    pre = df.loc[off.between(-5, -1), ["dvix_pts"]].copy()
    pre["off"] = off[off.between(-5, -1)]
    # group the 5 pre-days by their meeting: consecutive runs — use the cumsum trick on offset
    # resets; simpler: assign each pre-day to the next FOMC date
    idx = df.index
    nxt = pd.Series(pd.NaT, index=pre.index)
    fpos = pd.DatetimeIndex(sorted(d for d in fomc if d in set(idx)))
    j = np.searchsorted(fpos.values, pre.index.values, side="left")
    nxt[:] = fpos.values[np.minimum(j, len(fpos) - 1)]
    per_meeting = pre.groupby(nxt.values)["dvix_pts"].sum()
    x = per_meeting.values
    se = x.std(ddof=1) / np.sqrt(len(x))
    return {"n_meetings": len(x), "mean_runup_pts": float(x.mean()),
            "t": float(x.mean() / se) if se > 0 else float("nan")}


# --------------------------------------------------------------------------- #
# Realized SPY range on the same days
# --------------------------------------------------------------------------- #
def spy_range_stats(df: pd.DataFrame) -> dict:
    """SPY (H-L)/prev-close on decision days vs all other days, Welch t."""
    f = df["fomc"].values
    rng_ = df["spy_range"].values
    a, b = rng_[f], rng_[~f]
    return {"fomc_range_pct": float(np.nanmean(a) * 100),
            "rest_range_pct": float(np.nanmean(b) * 100),
            "welch_t": welch_t(a, b)}


# --------------------------------------------------------------------------- #
# Sub-period contrast (justified split: the press-conference era)
# --------------------------------------------------------------------------- #
def era_contrast(df: pd.DataFrame, split: str) -> dict:
    """FOMC-day ΔVIX before vs since ``split``: within-era Welch t's + Welch t OF THE
    DIFFERENCE between the two eras' FOMC-day changes."""
    f = df["fomc"]
    early = df.loc[f & (df.index < split), "dvix_pts"].values
    late = df.loc[f & (df.index >= split), "dvix_pts"].values
    rest_early = df.loc[~f & (df.index < split), "dvix_pts"].values
    rest_late = df.loc[~f & (df.index >= split), "dvix_pts"].values
    return {"n_early": len(early), "n_late": len(late),
            "early_pts": float(np.nanmean(early)), "late_pts": float(np.nanmean(late)),
            "welch_t_early": welch_t(early, rest_early),
            "welch_t_late": welch_t(late, rest_late),
            "welch_t_diff": welch_t(late, early)}


# --------------------------------------------------------------------------- #
# Third axis — the retail short-vol expression (SVXY, one day)
# --------------------------------------------------------------------------- #
def svxy_capture(svxy: pd.DataFrame, fomc: pd.DatetimeIndex,
                 cost_bps: float = 5.0, start: str | None = None,
                 end: str | None = None) -> dict:
    """SVXY held for the decision day only: enter prior close, exit decision-day close.

    The FOMC calendar is public years in advance, so entering at the prior session's close is
    a zero-look-ahead scheduled entry (the study's single documented execution convention).
    Each event costs one round trip = 2 x ``cost_bps`` one-way x NAV. Long-only (no borrow).
    Gross/net per event, Welch t vs all other SVXY days, annual contribution at 8 events/yr.
    """
    px = svxy["Close"]
    if start:
        px = px[px.index >= start]
    if end:
        px = px[px.index <= end]
    ret = px.pct_change().dropna()
    f = ret.index.isin(fomc)
    a, b = ret.values[f], ret.values[~f]
    gross = float(a.mean())
    net = gross - 2.0 * cost_bps / 1e4
    return {"n_fomc": int(f.sum()), "n_rest": int((~f).sum()),
            "gross_bps": gross * 1e4, "net_bps": net * 1e4,
            "rest_bps": float(b.mean()) * 1e4,
            "welch_t": welch_t(a, b),
            "ann_net_pct": net * 8 * 100,       # 8 scheduled decisions per year
            "worst_day_pct": float(a.min()) * 100,
            "hit_rate": float((a > 0).mean())}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(close: pd.DataFrame, decisions: pd.DatetimeIndex) -> dict:
    """Run the headline Welch split on a synthetic world."""
    df = pd.DataFrame(index=close.index)
    df["dvix_pts"] = close["Close"].diff()
    df["dvix_log"] = np.log(close["Close"]).diff()
    df["fomc"] = df.index.isin(decisions)
    df = df.dropna(subset=["dvix_pts"])
    return decision_day_stats(df)
