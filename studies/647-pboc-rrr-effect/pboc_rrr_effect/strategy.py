"""Strategy + inference for Study 647 — PBoC RRR Effect.

The claim: **Chinese equities pop when the PBoC cuts the Reserve Requirement Ratio (RRR)** —
the "stimulus rally" folklore. A cut frees up bank lending capacity system-wide, so the story
goes, and the market should cheer. The mirror claim (rarely stated but implied by symmetry) is
that a hike should do the opposite.

Measurements:

* **Cut-day / hike-day return vs all other days** — FXI (the primary claim vehicle) log
  return on RRR-cut days and on RRR-hike days, each vs every other trading day. Welch *t*
  (single-day, non-overlapping-in-theory events) is the planned primary; a Newey-West *t* on
  the event-day dummy regression is the serial-correlation-robust cross-check, plus a hit rate
  with a Wilson interval and a **one-sided** random-calendar placebo (the claim has a
  pre-committed sign: cuts should push returns UP, hikes DOWN).
* **Cuts vs hikes, directly** — a Welch *t* of the difference between cut-day and hike-day
  returns (the cleanest test of "does direction matter at all").
* **Event window [-5..+10]** — does the market run up into the cut (buying the rumor: the
  PBoC rarely surprises, RRR moves are usually flagged by State Council meetings days ahead)
  and fade afterward (selling the news)? Per-offset Welch *t* vs far-from-event days.
* **MCHI cross-check** — the same headline split replicated on MCHI (a different index
  provider/construction, inception 2011-03-29) restricted to the modern-era events, so the
  FXI result isn't an artefact of FXI's own swap-heavy construction.
* **Era contrast (cut days only)** — the 2008-2012 panic-easing cuts (GFC, Euro crisis; often
  a genuine surprise-adjacent policy pivot) vs the 2015-2025 secular-grind cuts (well
  telegraphed, State-Council-flagged days ahead), split *ex ante* on the regime change itself.
* **Third axis — "buy the rumor, sell the news"?** The pre-event run-up (rumor) vs the
  post-event decay (news) inside the same [-5..+10] window, plus a "buy the cut" timer:
  FXI entered at the prior close, held 1/3/5/10 trading days, net of one-way costs x 2,
  vs a matched-horizon random-window null drawn from the rest of the tape.

The decisive number is the cut-day Welch *t* on the REAL FXI tape; the honest question is
whether the folklore's directional story survives at all, and whether any timing of it pays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Matching RRR announcement dates to trading days (zero-look-ahead convention)
# --------------------------------------------------------------------------- #
def match_trading_days(index: pd.DatetimeIndex, rrr: pd.DataFrame) -> pd.DataFrame:
    """Map each RRR announcement (Beijing calendar date) to the first FXI/MCHI trading
    session on or after it.

    Execution lag (documented, single convention): the PBoC releases RRR announcements during
    Beijing business hours or over a weekend — Beijing is 12-13 hours ahead of New York, so by
    the time FXI/MCHI's NYSE session opens on the first trading day on/after the announcement's
    Beijing calendar date, the news has already been public for hours (often a full weekend).
    Using the close-to-close change of THAT session as "day 0" is therefore zero-look-ahead by
    construction — if anything conservative (the market often has the news even before that
    session opens). Two announcements landing on the same next-trading-day (never happens in
    this table — RRR events are >= 1 week apart) would collapse to the later one.
    """
    idx_vals = index.values
    rows = []
    for _, r in rrr.iterrows():
        p = int(np.searchsorted(idx_vals, np.datetime64(r["date"]), side="left"))
        if p < len(idx_vals):
            rows.append({"trading_date": index[p], "direction": r["direction"],
                        "bps": int(r["bps"]), "announce_date": r["date"]})
    out = pd.DataFrame(rows)
    return out.drop_duplicates(subset="trading_date", keep="last").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Day frame
# --------------------------------------------------------------------------- #
def day_frame(px: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    """One row per trading day: log return, realized range, and the RRR event flag/direction.

    ``px`` is a raw-OHLC + AdjClose frame (FXI or MCHI); ``matched`` is the output of
    ``match_trading_days`` for that same index.
    """
    df = pd.DataFrame(index=px.index)
    df["ret"] = np.log(px["AdjClose"]).diff()
    df["range"] = (px["High"] - px["Low"]) / px["Close"].shift(1)
    dmap = matched.set_index("trading_date")["direction"]
    df["direction"] = df.index.map(dmap)
    df["event"] = df["direction"].notna()
    return df.dropna(subset=["ret"])


def event_offsets(df: pd.DataFrame, event_days: pd.DatetimeIndex,
                  lo: int = -5, hi: int = 10) -> pd.Series:
    """Business-day offset of each tape day relative to the nearest event day.

    Offset 0 = the event day itself. Overlaps (two events closer than the window) resolve to
    the chronologically LATER event (named, minor caveat — same convention as sibling study
    646-boj-announcement-effect); RRR events are typically months apart so this is rare.
    """
    idx = df.index
    off = pd.Series(np.nan, index=idx)
    pos_of = {d: i for i, d in enumerate(idx)}
    for d in event_days:
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
    """HAC (Newey-West, Bartlett kernel) t of the slope in y = a + b*d."""
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=float)
    keep = ~np.isnan(y)
    y, d = y[keep], d[keep]
    n = len(y)
    X = np.column_stack([np.ones(n), d])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    s = X * u[:, None]
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
def decision_day_stats(df: pd.DataFrame, col: str, direction: str, nw_lags: int = 5) -> dict:
    """Event-day return (for the given direction) vs all-other-days."""
    f = (df["direction"] == direction).values
    r = df[col].values
    a, b = r[f], r[~f]
    k_up = int((a > 0).sum())
    lo, hi = wilson_interval(k_up, len(a))
    return {
        "n_event": int(f.sum()), "n_rest": int((~f).sum()),
        "event_pct": float(np.nanmean(a) * 100), "rest_pct": float(np.nanmean(b) * 100),
        "gap_pct": float((np.nanmean(a) - np.nanmean(b)) * 100),
        "welch_t": welch_t(a, b),
        "nw_t": newey_west_t(r, f.astype(float), lags=nw_lags),
        "hit_up": k_up, "hit_rate": k_up / len(a), "hit_lo": lo, "hit_hi": hi,
    }


def cuts_vs_hikes(df: pd.DataFrame, col: str) -> dict:
    """Direct Welch t of cut-day return minus hike-day return."""
    a = df.loc[df["direction"] == "cut", col].values
    b = df.loc[df["direction"] == "hike", col].values
    return {"n_cut": len(a), "n_hike": len(b),
            "cut_pct": float(np.nanmean(a) * 100), "hike_pct": float(np.nanmean(b) * 100),
            "welch_t": welch_t(a, b)}


def placebo_pvalue(df: pd.DataFrame, col: str, direction: str, tail: str = "right",
                   n_draws_per_seed: int = 1_000, n_seeds: int = 20,
                   base_seed: int = 647) -> dict:
    """Random-calendar placebo: draw |direction events| random non-event days, mean return.

    One-sided (``tail``): the claim has a pre-committed sign — cuts should push returns UP
    (right tail), hikes DOWN (left tail). Averaged over ``n_seeds`` seeds x
    ``n_draws_per_seed`` draws. The pool excludes ALL RRR event days (cut and hike alike), so
    neither direction's null pool is contaminated by the other's plausible effect.
    """
    f = (df["direction"] == direction).values
    r = df[col].values
    obs = float(np.nanmean(r[f]))
    pool = r[~df["event"].values]
    pool = pool[~np.isnan(pool)]
    k = int(f.sum())
    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            means.append(pool[rng.choice(len(pool), size=k, replace=False)].mean())
    means = np.asarray(means)
    p = float((means >= obs).mean()) if tail == "right" else float((means <= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)), "p_value": p,
            "n_draws": len(means), "tail": tail}


# --------------------------------------------------------------------------- #
# Event window: run-up ("rumor") and persistence ("news")
# --------------------------------------------------------------------------- #
def event_study(df: pd.DataFrame, col: str, event_days: pd.DatetimeIndex,
                lo: int = -5, hi: int = 10) -> pd.DataFrame:
    """Mean return (%) by event offset, Welch t vs far-from-event days."""
    off = event_offsets(df, event_days, lo, hi)
    far = df.loc[off.isna(), col].values
    rows = []
    for k in range(lo, hi + 1):
        x = df.loc[off == k, col].values
        rows.append({"offset": k, "n": len(x), "mean_pct": float(np.nanmean(x)) * 100,
                     "welch_t": welch_t(x, far)})
    return pd.DataFrame(rows).set_index("offset")


def runup_stats(df: pd.DataFrame, col: str, event_days: pd.DatetimeIndex,
                lo: int = -5, hi: int = 10) -> dict:
    """Cumulative return over the pre-event window [-5..-1] per event, one-sample t."""
    off = event_offsets(df, event_days, lo, hi)
    pre = df.loc[off.between(-5, -1), [col]].copy()
    pre["off"] = off[off.between(-5, -1)]
    idx = df.index
    fpos = pd.DatetimeIndex(sorted(d for d in event_days if d in set(idx)))
    j = np.searchsorted(fpos.values, pre.index.values, side="left")
    nxt = fpos.values[np.minimum(j, len(fpos) - 1)]
    per_event = pre.groupby(nxt)[col].sum()
    x = per_event.values
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return {"n_events": len(x), "mean_runup_pct": float(x.mean()) * 100,
            "t": float(x.mean() / se) if se and se > 0 else float("nan")}


def postevent_stats(df: pd.DataFrame, col: str, event_days: pd.DatetimeIndex,
                    lo: int = -5, hi: int = 10) -> dict:
    """Cumulative return over the post-event window [+1..+10] per event, one-sample t
    ("sell the news": does the pop, if any, give back?)."""
    off = event_offsets(df, event_days, lo, hi)
    post = df.loc[off.between(1, 10), [col]].copy()
    idx = df.index
    fpos = pd.DatetimeIndex(sorted(d for d in event_days if d in set(idx)))
    j = np.searchsorted(fpos.values, post.index.values, side="left") - 1
    j = np.clip(j, 0, len(fpos) - 1)
    nxt = fpos.values[j]
    per_event = post.groupby(nxt)[col].sum()
    x = per_event.values
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return {"n_events": len(x), "mean_postrun_pct": float(x.mean()) * 100,
            "t": float(x.mean() / se) if se and se > 0 else float("nan")}


# --------------------------------------------------------------------------- #
# Realized range — is an RRR day louder than average, regardless of direction?
# --------------------------------------------------------------------------- #
def range_stats(df: pd.DataFrame, range_col: str) -> dict:
    """(H-L)/prev-close on ANY RRR event day vs all other days, Welch t."""
    f = df["event"].values
    rng_ = df[range_col].values
    a, b = rng_[f], rng_[~f]
    return {"event_pct": float(np.nanmean(a) * 100), "rest_pct": float(np.nanmean(b) * 100),
            "welch_t": welch_t(a, b)}


# --------------------------------------------------------------------------- #
# Era contrast (cut days only): panic-easing (2008-2012) vs secular-grind (2015-2025)
# --------------------------------------------------------------------------- #
def era_contrast(df: pd.DataFrame, col: str, split: str) -> dict:
    """Cut-day return before vs since ``split``: within-era Welch t's + Welch t OF THE
    DIFFERENCE between the two eras' cut-day returns."""
    f = df["direction"] == "cut"
    early = df.loc[f & (df.index < split), col].values
    late = df.loc[f & (df.index >= split), col].values
    rest_early = df.loc[~df["event"] & (df.index < split), col].values
    rest_late = df.loc[~df["event"] & (df.index >= split), col].values
    return {"n_early": len(early), "n_late": len(late),
            "early_pct": float(np.nanmean(early)) * 100, "late_pct": float(np.nanmean(late)) * 100,
            "welch_t_early": welch_t(early, rest_early),
            "welch_t_late": welch_t(late, rest_late),
            "welch_t_diff": welch_t(late, early)}


# --------------------------------------------------------------------------- #
# Third axis — the "buy the cut" timer (tradability)
# --------------------------------------------------------------------------- #
def capture_horizon(close: pd.Series, matched: pd.DataFrame, direction: str, horizon: int,
                    cost_bps: float = 5.0) -> dict:
    """Enter at the prior close, hold ``horizon`` trading days (through the close of session
    offset ``horizon - 1``), exit. One round trip = 2 x ``cost_bps`` one-way x NAV. Long-only.

    The rest-of-tape null is every possible ``horizon``-day rolling window on the SAME
    instrument whose [entry, exit] span doesn't touch any RRR event's [-5..+10] window
    (i.e. genuinely "ordinary" days, not just non-event days) — a stricter, cleaner null than
    the plain day-level placebo used for the headline split.
    """
    idx = close.index
    pos_of = {d: i for i, d in enumerate(idx)}
    logc = np.log(close.values)
    n = len(idx)

    touched = set()
    for d in matched["trading_date"]:
        p = pos_of.get(d)
        if p is not None:
            for k in range(-5, 11):
                if 0 <= p + k < n:
                    touched.add(p + k)

    events = matched.loc[matched["direction"] == direction, "trading_date"]
    ev_rets = []
    for d in events:
        p = pos_of.get(d)
        if p is None:
            continue
        entry, exit_ = p - 1, p - 1 + horizon
        if entry < 0 or exit_ >= n:
            continue
        ev_rets.append(logc[exit_] - logc[entry])
    ev_rets = np.asarray(ev_rets)

    rest = []
    for entry in range(0, n - horizon):
        exit_ = entry + horizon
        if entry in touched or exit_ in touched:
            continue
        rest.append(logc[exit_] - logc[entry])
    rest = np.asarray(rest)

    gross = float(ev_rets.mean()) if len(ev_rets) else float("nan")
    net = gross - 2.0 * cost_bps / 1e4
    return {"n_events": len(ev_rets), "horizon": horizon,
            "gross_bps": gross * 1e4, "net_bps": net * 1e4,
            "rest_mean_bps": float(rest.mean()) * 1e4,
            "welch_t": welch_t(ev_rets, rest),
            "hit_rate": float((ev_rets > 0).mean()) if len(ev_rets) else float("nan"),
            "worst_pct": float(ev_rets.min()) * 100 if len(ev_rets) else float("nan"),
            "best_pct": float(ev_rets.max()) * 100 if len(ev_rets) else float("nan")}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(close: pd.DataFrame, decisions: pd.DatetimeIndex) -> dict:
    """Run the headline Welch split on a synthetic world."""
    df = pd.DataFrame(index=close.index)
    df["ret"] = np.log(close["Close"]).diff()
    df["direction"] = pd.Series(
        np.where(df.index.isin(decisions), "cut", None), index=df.index, dtype=object)
    df["event"] = df["direction"].notna()
    df = df.dropna(subset=["ret"])
    return decision_day_stats(df, "ret", "cut")
