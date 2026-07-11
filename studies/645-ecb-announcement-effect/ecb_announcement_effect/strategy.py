"""Strategy + inference for Study 645 — ECB Announcement Effect.

The claim: **euro-area equities drift or react systematically around ECB Governing Council
monetary-policy decision days.** The Council announces its rate decision at 13:45 CET, with a
press conference at 14:30 CET (14:15 since 2012) — both *before* FEZ's ~16:00 ET (22:00 CET)
close and well before EURUSD's 24h-market "close" print — so the decision-day bar should
contain the market's reaction, the way the FOMC vol-crush and pre-FOMC-drift studies test the
Fed's own version of this question.

Measurements:

* **Decision-day FEZ return vs all other days** — close-to-close log return, Welch *t* (the
  planned primary, single non-overlapping events) plus a Newey-West dummy-regression *t* as the
  serial-correlation-robust cross-check, and a hit-rate with a Wilson interval.
* **Realized FEZ range on the same days** — (H-L)/prev-close, decision days vs other days: is
  the announcement actually a *loud* day (mechanical, priced-in vol) or does price move without
  extra intraday churn?
* **EURUSD reaction** — |return| on decision days vs other days: does the currency leg react
  more than equities do?
* **Event window [-5..+3]** — a Lucca-Moench-style pre-meeting drift analog (does FEZ ramp up
  into the decision the way SPY ramps into the Fed?) and the post-day persistence.
* **Era contrast** — the monthly era (2005 -> end-2014) vs the 6-week era (2015 ->), split at
  the Governing Council's own structural announcement (2015-01-01) — a justified, not snooped,
  split.
* **Third axis (tradability, "costs on a timer")** — hold FEZ for the decision day only (enter
  the prior close — the calendar is public months ahead, zero look-ahead — exit the
  decision-day close), swept across a realistic one-way-cost ladder.
* **Random-calendar placebo** and a **20-seed synthetic control** (faithful-engine / power
  check only, never cited in support of a real-tape stamp).

The decisive number is the decision-day Welch *t* on the REAL FEZ tape; everything else is
context for how (not whether) to believe it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Day frame
# --------------------------------------------------------------------------- #
def day_frame(fez: pd.DataFrame, eurusd: pd.DataFrame | None,
              ecb: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per FEZ trading day: log return, realized range, EURUSD |return|, ECB flag."""
    df = pd.DataFrame(index=fez.index)
    df["fez_close"] = fez["Close"]
    df["fez_ret"] = np.log(fez["Close"]).diff()
    df["fez_range"] = (fez["High"] - fez["Low"]) / fez["Close"].shift(1)
    if eurusd is not None:
        fx = eurusd.reindex(fez.index)
        df["eurusd_ret"] = np.log(fx["Close"]).diff()
        df["eurusd_absret"] = df["eurusd_ret"].abs()
    df["ecb"] = df.index.isin(ecb)
    return df.dropna(subset=["fez_ret"])


def event_offsets(df: pd.DataFrame, ecb: pd.DatetimeIndex,
                  lo: int = -5, hi: int = 3) -> pd.Series:
    """Business-day offset of each tape day relative to the nearest ECB decision day.

    Offset 0 = the decision day itself, -1 = the session before, +1 = the session after, etc.
    Days farther than [lo, hi] from every meeting get NaN (the "far" control group). Meetings
    are >= 4 weeks apart, so windows of +-5 sessions never collide.
    """
    idx = df.index
    off = pd.Series(np.nan, index=idx)
    pos_of = {d: i for i, d in enumerate(idx)}
    for d in ecb:
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
    serial-correlation-robust cross-check for the daily FEZ-return series.
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
# The headline split — FEZ return
# --------------------------------------------------------------------------- #
def decision_day_stats(df: pd.DataFrame, nw_lags: int = 5) -> dict:
    """Decision-day FEZ log return vs all-other-days: means, Welch t, NW t, hit rate + Wilson CI."""
    f = df["ecb"].values
    r = df["fez_ret"].values
    a, b = r[f], r[~f]
    k_up = int((a > 0).sum())
    lo, hi = wilson_interval(k_up, len(a))
    return {
        "n_ecb": int(f.sum()), "n_rest": int((~f).sum()),
        "ecb_pct": float(np.nanmean(a) * 100), "rest_pct": float(np.nanmean(b) * 100),
        "gap_pct": float((np.nanmean(a) - np.nanmean(b)) * 100),
        "welch_t": welch_t(a, b),
        "nw_t": newey_west_t(r, f.astype(float), lags=nw_lags),
        "hit_up": k_up, "hit_rate": k_up / len(a), "hit_lo": lo, "hit_hi": hi,
    }


def placebo_pvalue(df: pd.DataFrame, column: str = "fez_ret", n_draws_per_seed: int = 1_000,
                   n_seeds: int = 20, base_seed: int = 645) -> dict:
    """Random-calendar placebo: draw |ECB| random non-ECB days, mean of ``column``.

    p = share of draws whose |mean| is >= the observed |ECB-day mean| (a two-sided test — the
    claim, stated broadly, is "systematic reaction", not a signed direction a priori).
    Averaged over ``n_seeds`` independent seeds x ``n_draws_per_seed`` draws so no single lucky
    stream decides it. Reused for both the return headline and the realized-range cross-check.
    """
    f = df["ecb"].values
    r = df[column].values
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
# Realized range and EURUSD reaction on the same days
# --------------------------------------------------------------------------- #
def range_stats(df: pd.DataFrame, nw_lags: int = 5) -> dict:
    """FEZ (H-L)/prev-close on decision days vs all other days: Welch t + NW t.

    Realized range is strongly autocorrelated (vol clustering), so the Newey-West dummy-
    regression t is the load-bearing robustness check here, not a formality.
    """
    f = df["ecb"].values
    rng_ = df["fez_range"].values
    a, b = rng_[f], rng_[~f]
    return {"ecb_range_pct": float(np.nanmean(a) * 100),
            "rest_range_pct": float(np.nanmean(b) * 100),
            "ratio": float(np.nanmean(a) / np.nanmean(b)),
            "welch_t": welch_t(a, b),
            "nw_t": newey_west_t(rng_, f.astype(float), lags=nw_lags)}


def eurusd_stats(df: pd.DataFrame) -> dict:
    """EURUSD |log return| on decision days vs all other days, Welch t."""
    f = df["ecb"].values
    a = df.loc[f, "eurusd_absret"].values
    b = df.loc[~f, "eurusd_absret"].values
    return {"ecb_abs_pct": float(np.nanmean(a) * 100),
            "rest_abs_pct": float(np.nanmean(b) * 100),
            "welch_t": welch_t(a, b)}


# --------------------------------------------------------------------------- #
# Event window: Lucca-Moench-style pre-meeting drift and persistence
# --------------------------------------------------------------------------- #
def event_study(df: pd.DataFrame, ecb: pd.DatetimeIndex,
                lo: int = -5, hi: int = 3) -> pd.DataFrame:
    """Mean FEZ log return by event offset, Welch t vs far-from-meeting days."""
    off = event_offsets(df, ecb, lo, hi)
    far = df.loc[off.isna(), "fez_ret"].values
    rows = []
    for k in range(lo, hi + 1):
        x = df.loc[off == k, "fez_ret"].values
        rows.append({"offset": k, "n": len(x), "mean_pct": float(np.nanmean(x)) * 100,
                     "welch_t": welch_t(x, far)})
    return pd.DataFrame(rows).set_index("offset")


def runup_stats(df: pd.DataFrame, ecb: pd.DatetimeIndex) -> dict:
    """Cumulative FEZ log return over the pre-meeting window [-5..-1] per meeting, one-sample t."""
    off = event_offsets(df, ecb, -5, 3)
    pre = df.loc[off.between(-5, -1), ["fez_ret"]].copy()
    pre["off"] = off[off.between(-5, -1)]
    idx = df.index
    ecb_pos = pd.DatetimeIndex(sorted(d for d in ecb if d in set(idx)))
    j = np.searchsorted(ecb_pos.values, pre.index.values, side="left")
    nxt = ecb_pos.values[np.minimum(j, len(ecb_pos) - 1)]
    per_meeting = pre.groupby(nxt)["fez_ret"].sum()
    x = per_meeting.values
    se = x.std(ddof=1) / np.sqrt(len(x))
    return {"n_meetings": len(x), "mean_runup_pct": float(x.mean()) * 100,
            "t": float(x.mean() / se) if se > 0 else float("nan")}


# --------------------------------------------------------------------------- #
# Sub-period contrast (justified split: the 6-week-cycle era)
# --------------------------------------------------------------------------- #
def era_contrast(df: pd.DataFrame, split: str, column: str = "fez_ret") -> dict:
    """ECB-day ``column`` before vs since ``split``: within-era Welch t's + Welch t OF THE
    DIFFERENCE between the two eras' decision-day values. Used for both the return headline
    (monthly vs 6-week cadence) and the realized-range cross-check (is the vol bump stable?)."""
    f = df["ecb"]
    early = df.loc[f & (df.index < split), column].values
    late = df.loc[f & (df.index >= split), column].values
    rest_early = df.loc[~f & (df.index < split), column].values
    rest_late = df.loc[~f & (df.index >= split), column].values
    scale = 100.0
    return {"n_early": len(early), "n_late": len(late),
            "early_pct": float(np.nanmean(early)) * scale, "late_pct": float(np.nanmean(late)) * scale,
            "welch_t_early": welch_t(early, rest_early),
            "welch_t_late": welch_t(late, rest_late),
            "welch_t_diff": welch_t(late, early)}


# --------------------------------------------------------------------------- #
# Third axis — "costs on a timer": hold FEZ for the decision day only
# --------------------------------------------------------------------------- #
def timer_capture(df: pd.DataFrame, cost_bps: float = 5.0) -> dict:
    """FEZ held for the decision day only: enter prior close, exit decision-day close.

    The ECB calendar is public months in advance, so entering at the prior session's close is
    a zero-look-ahead scheduled entry (the study's single documented execution convention).
    Each event costs one round trip = 2 x ``cost_bps`` one-way x NAV. Long-only (no borrow).
    Gross/net per event, Welch t vs all other FEZ days, annual contribution at 8 events/yr
    (the modern 6-week cadence).
    """
    f = df["ecb"].values
    r = df["fez_ret"].values           # log return ~= simple return at these magnitudes
    a, b = r[f], r[~f]
    gross = float(a.mean())
    net = gross - 2.0 * cost_bps / 1e4
    return {"n_ecb": int(f.sum()), "n_rest": int((~f).sum()),
            "gross_bps": gross * 1e4, "net_bps": net * 1e4,
            "rest_bps": float(b.mean()) * 1e4,
            "welch_t": welch_t(a, b),
            "ann_net_pct": net * 8 * 100,
            "worst_day_pct": float(a.min()) * 100,
            "hit_rate": float((a > 0).mean())}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(close: pd.DataFrame, decisions: pd.DatetimeIndex) -> dict:
    """Run the headline Welch split on a synthetic world."""
    df = pd.DataFrame(index=close.index)
    df["fez_ret"] = np.log(close["Close"]).diff()
    df["ecb"] = df.index.isin(decisions)
    df = df.dropna(subset=["fez_ret"])
    return decision_day_stats(df)
