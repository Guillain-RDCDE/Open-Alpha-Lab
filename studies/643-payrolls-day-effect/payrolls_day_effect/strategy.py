"""Strategy + inference for Study 643 — Payrolls-Day-Effect.

The claim: **US equities behave systematically on Nonfarm-Payrolls release
mornings** (usually the first Friday of the month, 8:30 am ET) — the single most
watched print on the macro calendar should move the tape more than an average day,
and folklore insists it moves it in a *particular direction* (a "relief rally" once
the uncertainty clears, or steady pre-release drift as positioning builds).

Measurements:

* **Release-day return vs all other days** — SPY close-to-close log return (the
  release lands at 8:30 am ET, before the 9:30 am open, so the release-day bar
  fully contains the reaction). Welch *t* for the group split (single-day,
  non-overlapping events), a Newey-West dummy-regression *t* as the
  serial-correlation-robust cross-check, a hit-rate with a Wilson interval and a
  random-calendar placebo.
* **Event window [-3..+3]** — does a pre-release drift show up, and does whatever
  happens on the day persist or mean-revert?
* **Realized SPY high-low range** on release days vs other days — is the release
  morning *louder* (the resolution-cross-check that made the FOMC vol-crush study,
  637, real-mechanical)?
* **Third axis (timer capture)** — a naive "own SPY only on NFP day" timer, entered
  at the prior close (the calendar is public months ahead — zero look-ahead) and
  exited at the release-day close, net of one-way costs x 2 per event.

The decisive number is the release-day Welch *t* on the REAL SPY tape; the honest
question is whether the "loud morning" is also a *directional, bankable* one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Day frame
# --------------------------------------------------------------------------- #
def day_frame(spy: pd.DataFrame, nfp_sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per SPY trading day: log return, high-low range, and the NFP flag."""
    df = pd.DataFrame(index=spy.index)
    df["close"] = spy["AdjClose"]
    df["ret"] = np.log(spy["AdjClose"]).diff()
    df["range_pct"] = (spy["High"] - spy["Low"]) / spy["Close"].shift(1)
    df["nfp"] = df.index.isin(nfp_sessions)
    return df.dropna(subset=["ret"])


def event_offsets(df: pd.DataFrame, nfp_sessions: pd.DatetimeIndex,
                  lo: int = -3, hi: int = 3) -> pd.Series:
    """Business-day offset of each tape day relative to the nearest NFP session.

    Offset 0 = the release day itself, -1 = the session before, +1 = the session
    after, etc. Days farther than [lo, hi] from every release get NaN (the "far"
    control group). Releases are >= 3 weeks apart, so windows of +-3 sessions never
    collide.
    """
    idx = df.index
    off = pd.Series(np.nan, index=idx)
    pos_of = {d: i for i, d in enumerate(idx)}
    for d in nfp_sessions:
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
    serial-correlation-robust cross-check for the daily return series.
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
def nfp_day_stats(df: pd.DataFrame, nw_lags: int = 5) -> dict:
    """NFP-day return vs all-other-days: means (bps), Welch t, NW t, hit rate + Wilson CI."""
    f = df["nfp"].values
    r = df["ret"].values
    a, b = r[f], r[~f]
    k_up = int((a > 0).sum())
    lo, hi = wilson_interval(k_up, len(a))
    return {
        "n_nfp": int(f.sum()), "n_rest": int((~f).sum()),
        "nfp_bps": float(np.nanmean(a) * 1e4), "rest_bps": float(np.nanmean(b) * 1e4),
        "gap_bps": float((np.nanmean(a) - np.nanmean(b)) * 1e4),
        "welch_t": welch_t(a, b),
        "nw_t": newey_west_t(r, f.astype(float), lags=nw_lags),
        "hit_up": k_up, "hit_rate": k_up / len(a),
        "hit_lo": lo, "hit_hi": hi,
    }


def placebo_pvalue(df: pd.DataFrame, n_draws_per_seed: int = 1_000,
                   n_seeds: int = 20, base_seed: int = 643) -> dict:
    """Random-calendar placebo: draw |NFP| random non-NFP days, mean return (bps).

    Two-sided: p = share of |draw means| >= |observed mean| (the claim has no
    a-priori direction — "behaves systematically" could be up or down). Averaged
    over ``n_seeds`` independent seeds x ``n_draws_per_seed`` draws so no single
    lucky stream decides it.
    """
    f = df["nfp"].values
    r = df["ret"].values
    obs = float(np.nanmean(r[f]))
    pool = r[~f]
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
            "p_value": float((np.abs(means) >= abs(obs)).mean()),
            "n_draws": len(means), "draws": means}


# --------------------------------------------------------------------------- #
# Event window: pre-release drift and persistence
# --------------------------------------------------------------------------- #
def event_study(df: pd.DataFrame, nfp_sessions: pd.DatetimeIndex,
                lo: int = -3, hi: int = 3) -> pd.DataFrame:
    """Mean return (bps) by event offset, Welch t vs far-from-release days."""
    off = event_offsets(df, nfp_sessions, lo, hi)
    far = df.loc[off.isna(), "ret"].values
    rows = []
    for k in range(lo, hi + 1):
        x = df.loc[off == k, "ret"].values
        rows.append({"offset": k, "n": len(x), "mean_bps": float(np.nanmean(x) * 1e4),
                     "welch_t": welch_t(x, far)})
    return pd.DataFrame(rows).set_index("offset")


def runup_stats(df: pd.DataFrame, nfp_sessions: pd.DatetimeIndex,
                lo: int = -3, hi_excl: int = 0) -> dict:
    """Cumulative return over the pre-release window [lo..-1] per event, one-sample t."""
    off = event_offsets(df, nfp_sessions, lo, 3)
    pre = df.loc[off.between(lo, -1), ["ret"]].copy()
    pre["off"] = off[off.between(lo, -1)]
    idx = df.index
    fpos = pd.DatetimeIndex(sorted(d for d in nfp_sessions if d in set(idx)))
    j = np.searchsorted(fpos.values, pre.index.values, side="left")
    nxt = fpos.values[np.minimum(j, len(fpos) - 1)]
    per_event = pre.groupby(nxt)["ret"].sum()
    x = per_event.values
    se = x.std(ddof=1) / np.sqrt(len(x))
    return {"n_events": len(x), "mean_runup_bps": float(x.mean() * 1e4),
            "t": float(x.mean() / se) if se > 0 else float("nan")}


# --------------------------------------------------------------------------- #
# Realized range on the same days
# --------------------------------------------------------------------------- #
def spy_range_stats(df: pd.DataFrame) -> dict:
    """SPY (H-L)/prev-close on NFP days vs all other days, Welch t."""
    f = df["nfp"].values
    rng_ = df["range_pct"].values
    a, b = rng_[f], rng_[~f]
    return {"nfp_range_pct": float(np.nanmean(a) * 100),
            "rest_range_pct": float(np.nanmean(b) * 100),
            "welch_t": welch_t(a, b)}


# --------------------------------------------------------------------------- #
# Sub-period contrast (justified split: pre vs post the ZIRP/QE-and-Twitter era)
# --------------------------------------------------------------------------- #
def era_contrast(df: pd.DataFrame, split: str) -> dict:
    """NFP-day return before vs since ``split``: within-era Welch t's + Welch t of
    the DIFFERENCE between the two eras' NFP-day returns."""
    f = df["nfp"]
    early = df.loc[f & (df.index < split), "ret"].values
    late = df.loc[f & (df.index >= split), "ret"].values
    rest_early = df.loc[~f & (df.index < split), "ret"].values
    rest_late = df.loc[~f & (df.index >= split), "ret"].values
    return {"n_early": len(early), "n_late": len(late),
            "early_bps": float(np.nanmean(early) * 1e4),
            "late_bps": float(np.nanmean(late) * 1e4),
            "welch_t_early": welch_t(early, rest_early),
            "welch_t_late": welch_t(late, rest_late),
            "welch_t_diff": welch_t(late, early)}


# --------------------------------------------------------------------------- #
# Third axis — the naive timer (own SPY only on NFP day)
# --------------------------------------------------------------------------- #
def timer_capture(df: pd.DataFrame, cost_bps: float = 5.0,
                  start: str | None = None, end: str | None = None) -> dict:
    """SPY held for the NFP release day only: enter prior close, exit release close.

    The NFP calendar is public months in advance, so entering at the prior
    session's close is a zero-look-ahead scheduled entry (the study's single
    documented execution convention). Each event costs one round trip = 2 x
    ``cost_bps`` one-way x NAV. Long-only (no borrow). Gross/net per event, Welch t
    vs all other days, annual contribution at 12 events/yr.
    """
    d = df
    if start:
        d = d[d.index >= start]
    if end:
        d = d[d.index <= end]
    f = d["nfp"].values
    r = d["ret"].values
    a, b = r[f], r[~f]
    gross = float(np.nanmean(a))
    net = gross - 2.0 * cost_bps / 1e4
    return {"n_nfp": int(f.sum()), "n_rest": int((~f).sum()),
            "gross_bps": gross * 1e4, "net_bps": net * 1e4,
            "rest_bps": float(np.nanmean(b)) * 1e4,
            "welch_t": welch_t(a, b),
            "ann_net_pct": net * 12 * 100,       # 12 scheduled releases per year
            "worst_day_pct": float(np.nanmin(a)) * 100,
            "hit_rate": float((a > 0).mean())}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(close: pd.DataFrame, releases: pd.DatetimeIndex) -> dict:
    """Run the headline Welch split (return) on a synthetic world."""
    df = pd.DataFrame(index=close.index)
    df["ret"] = np.log(close["Close"]).diff()
    df["nfp"] = df.index.isin(releases)
    df = df.dropna(subset=["ret"])
    return nfp_day_stats(df)
