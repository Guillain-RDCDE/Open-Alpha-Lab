"""Strategy + inference for Study 646 — BoJ Announcement Effect.

The claim: **Japanese equities (and the yen) react systematically to Bank of Japan policy
decision days** — especially in the "surprise" NIRP (2016) / YCC-tweak (2016, 2022-24) eras.
Unlike the FOMC, most BoJ meetings under YCC were genuinely low-information (rate on hold,
band unchanged) punctuated by a handful of true surprises — so the claim under test is
explicitly a *tail/surprise* story, not a "set your watch to it" scheduled-directional one.

Measurements:

* **Decision-day return vs all other days** — EWJ (unhedged Japan equities, USD) and the yen
  (defined as **minus** the USDJPY / ``JPY=X`` log change, so positive = yen strengthens — the
  same sign convention as sibling study 615-yen-safe-haven) on BoJ decision days vs every other
  trading day. Welch *t* (single-day, non-overlapping-in-theory events; see the event-window
  caveat below) is the planned primary; a Newey-West *t* on the decision-day dummy regression
  is the serial-correlation-robust cross-check, plus a hit rate with a Wilson interval and a
  random-calendar placebo.
* **Realized range** — same-day (High-Low)/prev-close on EWJ and USDJPY: are decision days
  *louder* than average, even when the average directional move is a wash? This is the desk's
  grey third-axis myth-check on this study.
* **Event window [-5..+3]** — build-up and persistence around the decision. BoJ meetings were
  near-monthly through 2015 (occasionally < 5 trading days apart during the 2008 and 2010-11
  crisis clusters), so overlapping windows are possible; the offset assignment below resolves
  overlaps to the chronologically later meeting (documented, not hidden).
* **Era contrast** — pre-NIRP (2005 -> 2016-01-28, rates near-zero but no NIRP/YCC) vs the
  NIRP/YCC "surprise regime" (2016-01-29 onward), split *ex ante* on the NIRP announcement
  itself, not snooped.
* **Named surprise tail events** — the hand-picked table in ``data.SURPRISE_EVENTS``: what did
  EWJ/JPY actually do on the specific days the folklore is about, vs the average-day
  distribution (a z-score)? Illustrative only, never used to certify the average-day stamp.
* **Tradability** — hold EWJ (or the yen) for the decision day only, entering at the prior
  close (the BoJ's annual meeting schedule is published in advance, so this is a
  zero-look-ahead scheduled entry) and exiting at the decision-day close, net of one-way costs
  x 2 per event.

The decisive number is the decision-day Welch *t* on the REAL EWJ/JPY tape; the honest
question is whether the story is a scheduled, tradable edge or a tail-event artifact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Day frame
# --------------------------------------------------------------------------- #
def day_frame(ewj: pd.DataFrame, jpy: pd.DataFrame, boj: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per EWJ trading day: EWJ / yen log returns + realized ranges + the BoJ flag.

    Execution lag (documented, single convention): the BoJ statement hits the tape during the
    Tokyo morning session on decision day D. EWJ's US-session close on calendar date D sits
    AFTER Tokyo's full reaction to the announcement, so the close-to-close change dated D
    contains the resolution with zero look-ahead (same convention as sibling study
    637-fomc-vol-crush: the statement predates the close it is measured against).

    **Named data quirk — ``JPY=X``'s daily ``Close`` field.** From 2022 onward, yfinance's
    ``JPY=X`` daily bars have their ``Close`` silently duplicate the SAME row's ``Open`` on the
    large majority of days (verified: >95% of 2023-2025 rows, vs <5% pre-2022) — a well-known
    Yahoo FX-data limitation, not a study artifact. Using close-to-close would therefore mute or
    mistime exactly the tail events this study cares about (the Dec-2022 YCC shock's ~4%
    intraday move reads as a ~flat day under the broken field). The fix, applied uniformly
    across the WHOLE 2005-2026 sample (not just the broken years, so no era-dependent
    convention change sneaks in): the yen return dated D is **Open[D+1] / Open[D] - 1**, i.e.
    the 24h window from the start of trading day D (~7am JST, pre-announcement) to the start of
    day D+1 (~7am JST next day) — a window that still starts before the ~11:30am JST
    announcement (zero look-ahead) and, if anything, is slightly WIDER than a pure close-to-
    close window, not narrower. ``High``/``Low`` remain reliable across the whole sample
    (checked), so the realized range uses ``(High-Low)/Open`` rather than the broken
    previous-close.
    """
    idx = ewj.index
    df = pd.DataFrame(index=idx)
    df["ewj_close"] = ewj["Close"]
    df["ewj_ret"] = np.log(ewj["Close"]).diff()
    df["ewj_range"] = (ewj["High"] - ewj["Low"]) / ewj["Close"].shift(1)

    jpy = jpy.reindex(idx).ffill()
    df["usdjpy_open"] = jpy["Open"]
    # yen return: MINUS the USDJPY open[D+1]/open[D] change, so positive = yen strengthens
    # (615-yen-safe-haven's sign convention). Dated D via shift(-1) on the forward leg.
    df["jpy_ret"] = -(np.log(jpy["Open"]).shift(-1) - np.log(jpy["Open"]))
    df["jpy_range"] = (jpy["High"] - jpy["Low"]) / jpy["Open"]

    df["boj"] = df.index.isin(boj)
    return df.dropna(subset=["ewj_ret", "jpy_ret"])


def event_offsets(df: pd.DataFrame, boj: pd.DatetimeIndex,
                  lo: int = -5, hi: int = 3) -> pd.Series:
    """Business-day offset of each tape day relative to the nearest BoJ decision day.

    Offset 0 = the decision day itself, -1 = the session before, +1 = the session after, etc.
    Days farther than [lo, hi] from every meeting get NaN (the "far" control group). BoJ
    meetings can sit closer than the window during the 2008/2010-11 crisis clusters; overlaps
    resolve to the chronologically LATER meeting (the loop below iterates meetings in ascending
    order and the later one overwrites) — a named, minor caveat, not hidden.
    """
    idx = df.index
    off = pd.Series(np.nan, index=idx)
    pos_of = {d: i for i, d in enumerate(idx)}
    for d in boj:
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
# The headline split (generic over a return column, so it runs on EWJ and yen alike)
# --------------------------------------------------------------------------- #
def decision_day_stats(df: pd.DataFrame, col: str, nw_lags: int = 5) -> dict:
    """Decision-day return vs all-other-days: means, Welch t, NW t, hit rate + Wilson CI."""
    f = df["boj"].values
    r = df[col].values
    a, b = r[f], r[~f]
    k_up = int((a > 0).sum())
    lo, hi = wilson_interval(k_up, len(a))
    return {
        "n_boj": int(f.sum()), "n_rest": int((~f).sum()),
        "boj_pct": float(np.nanmean(a) * 100), "rest_pct": float(np.nanmean(b) * 100),
        "gap_pct": float((np.nanmean(a) - np.nanmean(b)) * 100),
        "welch_t": welch_t(a, b),
        "nw_t": newey_west_t(r, f.astype(float), lags=nw_lags),
        "hit_up": k_up, "hit_rate": k_up / len(a), "hit_lo": lo, "hit_hi": hi,
    }


def placebo_pvalue(df: pd.DataFrame, col: str, n_draws_per_seed: int = 1_000,
                   n_seeds: int = 20, base_seed: int = 646) -> dict:
    """Random-calendar placebo: draw |BoJ| random non-BoJ days, mean return.

    p = share of |draws| with |mean| >= |observed| (a two-sided test — the claim under test
    has no pre-committed sign; BoJ surprises go both ways). Averaged over ``n_seeds``
    independent seeds x ``n_draws_per_seed`` draws so no single lucky stream decides it.
    """
    f = df["boj"].values
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
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)),
            "p_value": float((np.abs(means) >= abs(obs)).mean()),
            "n_draws": len(means), "draws": means}


# --------------------------------------------------------------------------- #
# Event window: build-up and persistence
# --------------------------------------------------------------------------- #
def event_study(df: pd.DataFrame, col: str, boj: pd.DatetimeIndex,
                lo: int = -5, hi: int = 3) -> pd.DataFrame:
    """Mean return by event offset, Welch t vs far-from-meeting days."""
    off = event_offsets(df, boj, lo, hi)
    far = df.loc[off.isna(), col].values
    rows = []
    for k in range(lo, hi + 1):
        x = df.loc[off == k, col].values
        rows.append({"offset": k, "n": len(x), "mean_pct": float(np.nanmean(x)) * 100,
                     "welch_t": welch_t(x, far)})
    return pd.DataFrame(rows).set_index("offset")


# --------------------------------------------------------------------------- #
# Realized range on the same days — the grey third-axis myth-check
# --------------------------------------------------------------------------- #
def range_stats(df: pd.DataFrame, col: str) -> dict:
    """(H-L)/prev-close on decision days vs all other days, Welch t."""
    f = df["boj"].values
    rng_ = df[col].values
    a, b = rng_[f], rng_[~f]
    return {"boj_pct": float(np.nanmean(a) * 100), "rest_pct": float(np.nanmean(b) * 100),
            "welch_t": welch_t(a, b)}


# --------------------------------------------------------------------------- #
# Sub-period contrast (justified split: the NIRP/YCC "surprise regime" era)
# --------------------------------------------------------------------------- #
def era_contrast(df: pd.DataFrame, col: str, split: str) -> dict:
    """BoJ-day return before vs since ``split``: within-era Welch t's + Welch t OF THE
    DIFFERENCE between the two eras' BoJ-day returns."""
    f = df["boj"]
    early = df.loc[f & (df.index < split), col].values
    late = df.loc[f & (df.index >= split), col].values
    rest_early = df.loc[~f & (df.index < split), col].values
    rest_late = df.loc[~f & (df.index >= split), col].values
    return {"n_early": len(early), "n_late": len(late),
            "early_pct": float(np.nanmean(early)) * 100, "late_pct": float(np.nanmean(late)) * 100,
            "welch_t_early": welch_t(early, rest_early),
            "welch_t_late": welch_t(late, rest_late),
            "welch_t_diff": welch_t(late, early)}


# --------------------------------------------------------------------------- #
# Named surprise tail events — illustrative, never used for a certification
# --------------------------------------------------------------------------- #
def tail_event_table(df: pd.DataFrame, surprises: pd.DataFrame) -> pd.DataFrame:
    """EWJ / yen return + range on each named surprise day, with a z-score vs all BoJ days."""
    f = df["boj"].values
    ewj_mu, ewj_sd = df.loc[f, "ewj_ret"].mean(), df.loc[f, "ewj_ret"].std(ddof=1)
    jpy_mu, jpy_sd = df.loc[f, "jpy_ret"].mean(), df.loc[f, "jpy_ret"].std(ddof=1)
    rows = []
    for d, label in zip(surprises.index, surprises["label"]):
        if d not in df.index:
            continue
        r = df.loc[d]
        rows.append({
            "date": d.date(), "label": label,
            "ewj_pct": r["ewj_ret"] * 100, "ewj_z": (r["ewj_ret"] - ewj_mu) / ewj_sd,
            "jpy_pct": r["jpy_ret"] * 100, "jpy_z": (r["jpy_ret"] - jpy_mu) / jpy_sd,
            "ewj_range_pct": r["ewj_range"] * 100, "jpy_range_pct": r["jpy_range"] * 100,
        })
    return pd.DataFrame(rows).set_index("date")


# --------------------------------------------------------------------------- #
# Third axis — tradability: hold the instrument for the decision day only
# --------------------------------------------------------------------------- #
def decision_day_capture(df: pd.DataFrame, col: str, cost_bps: float = 5.0) -> dict:
    """Hold ``col``'s instrument for the decision day only: enter prior close, exit the
    decision-day close (same convention as day_frame's log-return column).

    The BoJ's meeting schedule for the year ahead is published in advance, so entering at the
    prior close is a zero-look-ahead scheduled entry (this study's single documented execution
    convention). Each event costs one round trip = 2 x ``cost_bps`` one-way x NAV. Long-only
    (no borrow modeled). Gross/net per event, Welch t vs all other days.
    """
    f = df["boj"].values
    r = df[col].values
    a, b = r[f], r[~f]
    gross = float(np.nanmean(a))
    net = gross - 2.0 * cost_bps / 1e4
    n_events = int(f.sum())
    return {"n_events": n_events,
            "gross_bps": gross * 1e4, "net_bps": net * 1e4,
            "rest_bps": float(np.nanmean(b)) * 1e4,
            "welch_t": welch_t(a, b),
            "hit_rate": float((a > 0).mean()),
            "worst_day_pct": float(np.nanmin(a)) * 100,
            "best_day_pct": float(np.nanmax(a)) * 100}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(close: pd.DataFrame, decisions: pd.DatetimeIndex) -> dict:
    """Run the headline Welch split on a synthetic world."""
    df = pd.DataFrame(index=close.index)
    df["ret"] = np.log(close["Close"]).diff()
    df["boj"] = df.index.isin(decisions)
    df = df.dropna(subset=["ret"])
    return decision_day_stats(df, "ret")
