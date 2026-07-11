"""Strategy + inference for Study 644 — CPI-Day-Drift.

The claim: **CPI-print mornings (8:30 am ET, before the open) are a scheduled macro
event that moves stocks and bonds systematically** — inflation is the single input the
Fed's post-2021 reaction function watches most closely, so the print should move SPY and
TLT more than an average day, and folklore holds it moves them in a *particular
direction* (a relief rally on a cool print, a selloff on a hot one — direction varies by
print, but the claim is that *something* systematic happens).

Measurements:

* **CPI-day return vs all other days** (SPY and TLT) — close-to-close log return (the
  release lands at 8:30 am ET, before the 9:30 am open, so the release-day bar fully
  contains the reaction). Welch *t* for the group split (single-day, non-overlapping
  events), a Newey-West dummy-regression *t* as the serial-correlation-robust
  cross-check, a hit rate with a Wilson interval and a random-calendar placebo.
* **Event window [-3..+3]** — pre-release drift and post-release persistence.
* **Realized SPY/TLT high-low range** on CPI days vs other days — is the release
  morning *louder* (the resolution cross-check that made the FOMC vol-crush study, 637,
  real-mechanical), independent of which way it moves?
* **Regime split (2022-01-01, the Fed's hawkish pivot)** — is CPI day the loudest,
  most-watched day of the month **specifically in the post-2021 hiking-cycle era**?
  Tested as a Welch *t* of the difference between eras, not eyeballed off a chart.
* **Third axis — "biggest day of the month"?** For every calendar month with a mapped
  CPI session, is that session's |return| the single largest of the month? Hit rate +
  Wilson interval, pre- vs post-regime-split, tested as a difference.
* **Timer capture (tradability)** — a naive "own SPY only on CPI day" timer, entered at
  the prior close (the calendar is public months ahead — zero look-ahead) and exited at
  the release-day close, net of one-way costs x 2 per event.

The decisive numbers are the CPI-day Welch *t*'s on the REAL SPY/TLT tape; the honest
question is whether a "loud morning" is also a *directional, bankable* one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Day frame — generic: works for SPY or TLT
# --------------------------------------------------------------------------- #
def day_frame(px: pd.DataFrame, cpi_sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per trading day: log return, high-low range, and the CPI flag."""
    df = pd.DataFrame(index=px.index)
    df["close"] = px["AdjClose"]
    df["ret"] = np.log(px["AdjClose"]).diff()
    df["range_pct"] = (px["High"] - px["Low"]) / px["Close"].shift(1)
    df["cpi"] = df.index.isin(cpi_sessions)
    return df.dropna(subset=["ret"])


def event_offsets(df: pd.DataFrame, cpi_sessions: pd.DatetimeIndex,
                  lo: int = -3, hi: int = 3) -> pd.Series:
    """Business-day offset of each tape day relative to the nearest CPI session.

    Offset 0 = the release day itself, -1 = the session before, +1 = the session after,
    etc. Days farther than [lo, hi] from every release get NaN (the "far" control
    group). Releases are >= 3 weeks apart, so windows of +-3 sessions never collide.
    """
    idx = df.index
    off = pd.Series(np.nan, index=idx)
    pos_of = {d: i for i, d in enumerate(idx)}
    for d in cpi_sessions:
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
def cpi_day_stats(df: pd.DataFrame, nw_lags: int = 5) -> dict:
    """CPI-day return vs all-other-days: means (bps), Welch t, NW t, hit rate + Wilson CI."""
    f = df["cpi"].values
    r = df["ret"].values
    a, b = r[f], r[~f]
    k_up = int((a > 0).sum())
    lo, hi = wilson_interval(k_up, len(a))
    return {
        "n_cpi": int(f.sum()), "n_rest": int((~f).sum()),
        "cpi_bps": float(np.nanmean(a) * 1e4), "rest_bps": float(np.nanmean(b) * 1e4),
        "gap_bps": float((np.nanmean(a) - np.nanmean(b)) * 1e4),
        "welch_t": welch_t(a, b),
        "nw_t": newey_west_t(r, f.astype(float), lags=nw_lags),
        "hit_up": k_up, "hit_rate": k_up / len(a),
        "hit_lo": lo, "hit_hi": hi,
    }


def placebo_pvalue(df: pd.DataFrame, col: str = "ret", two_sided: bool = True,
                   n_draws_per_seed: int = 1_000, n_seeds: int = 20,
                   base_seed: int = 644) -> dict:
    """Random-calendar placebo: draw |CPI| random non-CPI days, mean of ``col``.

    Two-sided (default, used for return): p = share of |draw means| >= |observed mean|
    (the direction claim has no a-priori sign). One-sided (used for range, an
    inherently non-negative "louder" claim): p = share of draw means >= observed mean.
    Averaged over ``n_seeds`` independent seeds x ``n_draws_per_seed`` draws so no
    single lucky stream decides it.
    """
    f = df["cpi"].values
    r = df[col].values
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
    p = float((np.abs(means) >= abs(obs)).mean()) if two_sided else \
        float((means >= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)),
            "p_value": p, "n_draws": len(means), "draws": means}


# --------------------------------------------------------------------------- #
# Event window: pre-release drift and persistence
# --------------------------------------------------------------------------- #
def event_study(df: pd.DataFrame, cpi_sessions: pd.DatetimeIndex,
                lo: int = -3, hi: int = 3) -> pd.DataFrame:
    """Mean return (bps) by event offset, Welch t vs far-from-release days."""
    off = event_offsets(df, cpi_sessions, lo, hi)
    far = df.loc[off.isna(), "ret"].values
    rows = []
    for k in range(lo, hi + 1):
        x = df.loc[off == k, "ret"].values
        rows.append({"offset": k, "n": len(x), "mean_bps": float(np.nanmean(x) * 1e4),
                     "welch_t": welch_t(x, far)})
    return pd.DataFrame(rows).set_index("offset")


def runup_stats(df: pd.DataFrame, cpi_sessions: pd.DatetimeIndex,
                lo: int = -3, hi_excl: int = 0) -> dict:
    """Cumulative return over the pre-release window [lo..-1] per event, one-sample t."""
    off = event_offsets(df, cpi_sessions, lo, 3)
    pre = df.loc[off.between(lo, -1), ["ret"]].copy()
    pre["off"] = off[off.between(lo, -1)]
    idx = df.index
    fpos = pd.DatetimeIndex(sorted(d for d in cpi_sessions if d in set(idx)))
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
def range_stats(df: pd.DataFrame) -> dict:
    """(H-L)/prev-close on CPI days vs all other days, Welch t."""
    f = df["cpi"].values
    rng_ = df["range_pct"].values
    a, b = rng_[f], rng_[~f]
    return {"cpi_range_pct": float(np.nanmean(a) * 100),
            "rest_range_pct": float(np.nanmean(b) * 100),
            "welch_t": welch_t(a, b)}


# --------------------------------------------------------------------------- #
# Sub-period contrast (justified split: the post-2021 hawkish-pivot / hiking-cycle era)
# --------------------------------------------------------------------------- #
def era_contrast(df: pd.DataFrame, split: str, col: str = "ret") -> dict:
    """CPI-day ``col`` before vs since ``split``: within-era Welch t's + Welch t of the
    DIFFERENCE between the two eras' CPI-day values."""
    f = df["cpi"]
    early = df.loc[f & (df.index < split), col].values
    late = df.loc[f & (df.index >= split), col].values
    rest_early = df.loc[~f & (df.index < split), col].values
    rest_late = df.loc[~f & (df.index >= split), col].values
    scale = 1e4 if col == "ret" else 100.0
    return {"n_early": len(early), "n_late": len(late),
            "early": float(np.nanmean(early) * scale),
            "late": float(np.nanmean(late) * scale),
            "welch_t_early": welch_t(early, rest_early),
            "welch_t_late": welch_t(late, rest_late),
            "welch_t_diff": welch_t(late, early)}


# --------------------------------------------------------------------------- #
# Third axis — the myth-check: is CPI day the SINGLE BIGGEST trading day of its month?
# --------------------------------------------------------------------------- #
def biggest_day_of_month(df: pd.DataFrame, cpi_sessions: pd.DatetimeIndex,
                         split: str, metric: str = "ret") -> dict:
    """For every calendar month with a mapped CPI session, is that session's the single
    largest-magnitude day of the month on ``metric`` (``"ret"`` -> |return|, or
    ``"range_pct"`` -> realized range directly, already non-negative)? Hit rate + Wilson
    interval overall and pre/post ``split``, plus a Welch t of the DIFFERENCE
    (post - pre) so the "CPI day became the biggest day of the month" folklore is
    tested, not eyeballed off a chart. A null day drawn at random from an ``n``-session
    month clears this bar with probability ``1/n`` (~4-5% for a typical ~21-session
    month) — that is the honest baseline, not zero.
    """
    d = df.copy()
    d["month"] = d.index.to_period("M")
    d["_m"] = d[metric].abs() if metric == "ret" else d[metric]
    rows = []
    for month, g in d.groupby("month"):
        cpi_days = g.index[g.index.isin(cpi_sessions)]
        if len(cpi_days) == 0:
            continue
        day = cpi_days[0]                      # exactly one CPI day per month by design
        is_biggest = bool((g["_m"] >= g.loc[day, "_m"]).sum() == 1)
        rows.append({"month": month, "date": day, "n_sessions": len(g),
                     "is_biggest": is_biggest})
    res = pd.DataFrame(rows)
    pre = res.loc[res["date"] < split, "is_biggest"].values.astype(float)
    post = res.loc[res["date"] >= split, "is_biggest"].values.astype(float)
    lo_pre, hi_pre = wilson_interval(int(pre.sum()), len(pre))
    lo_post, hi_post = wilson_interval(int(post.sum()), len(post))
    avg_n = float(res["n_sessions"].mean())
    return {
        "n_months": len(res), "n_pre": len(pre), "n_post": len(post),
        "pre_rate": float(pre.mean()), "pre_lo": lo_pre, "pre_hi": hi_pre,
        "post_rate": float(post.mean()), "post_lo": lo_post, "post_hi": hi_post,
        "welch_t_diff": welch_t(post, pre),
        "null_rate": 1.0 / avg_n,
    }


# --------------------------------------------------------------------------- #
# Tradability — the naive timer (own the asset only on CPI day)
# --------------------------------------------------------------------------- #
def timer_capture(df: pd.DataFrame, cost_bps: float = 5.0,
                  start: str | None = None, end: str | None = None) -> dict:
    """Hold the asset for the CPI release day only: enter prior close, exit release
    close. The CPI calendar is public months in advance, so entering at the prior
    session's close is a zero-look-ahead scheduled entry (the study's single documented
    execution convention). Each event costs one round trip = 2 x ``cost_bps`` one-way x
    NAV. Long-only (no borrow). Gross/net per event, Welch t vs all other days, annual
    contribution at 12 events/yr.
    """
    d = df
    if start:
        d = d[d.index >= start]
    if end:
        d = d[d.index <= end]
    f = d["cpi"].values
    r = d["ret"].values
    a, b = r[f], r[~f]
    gross = float(np.nanmean(a))
    net = gross - 2.0 * cost_bps / 1e4
    return {"n_cpi": int(f.sum()), "n_rest": int((~f).sum()),
            "gross_bps": gross * 1e4, "net_bps": net * 1e4,
            "rest_bps": float(np.nanmean(b)) * 1e4,
            "welch_t": welch_t(a, b),
            "ann_net_pct": net * 12 * 100,       # 12 scheduled releases per year
            "worst_day_pct": float(np.nanmin(a)) * 100,
            "hit_rate": float((a > 0).mean())}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof — return AND range, independently)
# --------------------------------------------------------------------------- #
def synthetic_detect(close: pd.DataFrame, cpi_days: pd.DatetimeIndex) -> dict:
    """Run the headline Welch splits (return + range) on a synthetic world."""
    df = pd.DataFrame(index=close.index)
    df["ret"] = np.log(close["Close"]).diff()
    df["range_pct"] = (close["High"] - close["Low"]) / close["Close"].shift(1)
    df["cpi"] = df.index.isin(cpi_days)
    df = df.dropna(subset=["ret"])
    out = cpi_day_stats(df)
    out["range_welch_t"] = range_stats(df)["welch_t"]
    return out
