"""Strategy + inference for Study 642 — Turnaround Tuesday.

The claim: **after a down Monday, the market bounces back on Tuesday** — "turnaround
Tuesday", the mean-reversion cousin of the (mostly dead) Monday Effect. This is a
**conditional** claim — E[Tuesday | Monday < 0] vs the *unconditional* Tuesday mean and
vs all days — not a day-of-week *level* claim (that is studies 224/90) and not the
generic "last hour follows/reverses the morning" intraday claim (study 116).

Measurements:

* **The headline split** — mean Tuesday close-to-close return conditional on the prior
  session (Monday) closing down, vs (a) the unconditional Tuesday mean and (b) all
  trading days. Welch *t* for both contrasts (single, weekly, non-overlapping events —
  Welch is the planned primary) plus a Newey-West dummy-regression *t* as the
  serial-correlation-robust cross-check.
* **A random-pair placebo** — reshuffle which Tuesdays get the "down-Monday" label
  (holding the count fixed) and ask how often a random draw of that many Tuesdays
  produces a mean this high.
* **The Monday-specificity check** — is this really about *Monday*, or just the
  generic "reversal after any down day" effect wearing a costume? We run the same
  down-day -> next-day split for the other four weekday pairs (Tue->Wed, Wed->Thu,
  Thu->Fri, Fri->Mon) and Welch-t the pooled result.
* **Era contrast** — pre- vs post-2000 (the brief's named split), within-era Welch *t*
  plus a Welch *t* of the difference.
* **The timer** — enter SPY at the Monday close *after observing that Monday's own
  close was negative* (zero look-ahead: the flag is knowable at the instant of the
  Monday close, since that IS the close being flagged), exit at the Tuesday close.
  One round trip = 2 x one-way cost x NAV. Gross/net Sharpe, hit rate with a Wilson
  interval, one-sample *t* on the net per-event return.

The decisive number is the down-Monday-Tuesday Welch/NW *t* on the REAL SPY tape; the
honest question is whether the edge survives realistic costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


# --------------------------------------------------------------------------- #
# Day frame + Monday/Tuesday pairing
# --------------------------------------------------------------------------- #
def day_frame(close: pd.Series) -> pd.DataFrame:
    """One row per trading day: close-to-close return, weekday, and the previous
    session's weekday/return (for pairing consecutive sessions)."""
    ret = close.pct_change().dropna()
    df = pd.DataFrame(index=ret.index)
    df["ret"] = ret
    df["weekday"] = df.index.dayofweek
    df["prev_weekday"] = df["weekday"].shift(1)
    df["prev_ret"] = df["ret"].shift(1)
    return df


def monday_tuesday_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Tuesday sessions immediately preceded (previous trading session) by a Monday
    session — i.e. no market holiday breaks the adjacency. If the actual calendar
    Monday was a holiday, the previous session is a Friday, and that Tuesday is
    correctly excluded (there is no "Monday" to condition on).

    Columns: ``tuesday_ret`` (the outcome), ``monday_ret`` (the prior session's
    return, known at the instant it closes), ``down_monday`` (the flag).
    """
    mask = (df["weekday"] == 1) & (df["prev_weekday"] == 0)
    pairs = df.loc[mask, ["ret"]].rename(columns={"ret": "tuesday_ret"})
    pairs["monday_ret"] = df.loc[mask, "prev_ret"]
    pairs["down_monday"] = pairs["monday_ret"] < 0
    return pairs


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


def one_sample_t(x: np.ndarray) -> float:
    """One-sample t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def newey_west_t(y: np.ndarray, d: np.ndarray, lags: int = 5) -> tuple[float, float]:
    """HAC (Newey-West, Bartlett kernel) t and coefficient of the slope in y = a + b*d.

    b is exactly the treated-minus-rest mean difference; the NW t is the
    serial-correlation-robust cross-check for the daily-return series.
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
    s = X * u[:, None]
    S = s.T @ s
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = s[l:].T @ s[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(V[1, 1])
    t = float(beta[1] / se) if se > 0 else float("nan")
    return t, float(beta[1])


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
def headline_stats(df: pd.DataFrame, pairs: pd.DataFrame, nw_lags: int = 5) -> dict:
    """Down-Monday-Tuesday vs (a) unconditional Tuesday, (b) up-Monday Tuesday,
    (c) all trading days. Welch t's, Newey-West t's, hit rate + Wilson CI."""
    cond = pairs.loc[pairs["down_monday"], "tuesday_ret"].to_numpy()
    up = pairs.loc[~pairs["down_monday"], "tuesday_ret"].to_numpy()
    uncond_tue = df.loc[df["weekday"] == 1, "ret"].to_numpy()
    allday = df["ret"].to_numpy()

    k_hit = int((cond > 0).sum())
    hit_lo, hit_hi = wilson_interval(k_hit, len(cond))

    is_dm_tue_all = df.index.isin(pairs.index[pairs["down_monday"]])
    nw_all_t, nw_all_b = newey_west_t(allday, is_dm_tue_all.astype(float), lags=nw_lags)

    tue_df = df.loc[df["weekday"] == 1]
    is_dm_tue_sub = tue_df.index.isin(pairs.index[pairs["down_monday"]])
    nw_tue_t, nw_tue_b = newey_west_t(tue_df["ret"].to_numpy(),
                                       is_dm_tue_sub.astype(float), lags=nw_lags)

    return {
        "n_pairs": len(pairs), "n_down": len(cond), "n_up": len(up),
        "n_uncond_tue": len(uncond_tue), "n_allday": len(allday),
        "cond_bps": float(cond.mean() * 1e4), "up_bps": float(up.mean() * 1e4),
        "uncond_tue_bps": float(uncond_tue.mean() * 1e4),
        "allday_bps": float(allday.mean() * 1e4),
        "welch_t_vs_uncond": welch_t(cond, uncond_tue),
        "welch_t_vs_up": welch_t(cond, up),
        "welch_t_vs_allday": welch_t(cond, allday),
        "nw_t_vs_allday": nw_all_t, "nw_coef_vs_allday_bps": nw_all_b * 1e4,
        "nw_t_vs_uncond": nw_tue_t, "nw_coef_vs_uncond_bps": nw_tue_b * 1e4,
        "hit": k_hit, "hit_rate": k_hit / len(cond),
        "hit_lo": hit_lo, "hit_hi": hit_hi,
    }


def placebo_pvalue(pairs: pd.DataFrame, n_draws_per_seed: int = 1_000,
                   n_seeds: int = 20, base_seed: int = 642) -> dict:
    """Random-pair placebo: reshuffle which Tuesdays carry the down-Monday label
    (count held fixed), mean tuesday_ret of the random draw.

    p = share of draws whose mean is >= the observed down-Monday mean (a RIGHT-tail
    test — the claim is a bounce). Averaged over ``n_seeds`` independent seeds x
    ``n_draws_per_seed`` draws so no single lucky stream decides it.
    """
    tue = pairs["tuesday_ret"].to_numpy()
    k = int(pairs["down_monday"].sum())
    obs = float(tue[pairs["down_monday"].to_numpy()].mean())
    n = len(tue)
    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            idx = rng.choice(n, size=k, replace=False)
            means.append(tue[idx].mean())
    means = np.asarray(means)
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)),
            "p_value": float((means >= obs).mean()),
            "n_draws": len(means), "draws": means}


# --------------------------------------------------------------------------- #
# Monday-specificity check — is this Monday, or generic reversal-after-down-day?
# --------------------------------------------------------------------------- #
def weekday_pair_stats(df: pd.DataFrame) -> pd.DataFrame:
    """For every weekday pair (Mon->Tue, Tue->Wed, ..., Fri->Mon): the down-day ->
    next-day mean, the up-day -> next-day mean, and the Welch t between them."""
    rows = []
    for d in range(5):
        nd = (d + 1) % 5
        mask = (df["weekday"] == nd) & (df["prev_weekday"] == d)
        sub = df.loc[mask]
        down = sub["prev_ret"] < 0
        cond = sub.loc[down, "ret"].to_numpy()
        up = sub.loc[~down, "ret"].to_numpy()
        rows.append({
            "pair": f"{WEEKDAYS[d]}->{WEEKDAYS[nd]}", "n": len(sub),
            "n_down": len(cond), "down_bps": float(cond.mean() * 1e4) if len(cond) else float("nan"),
            "up_bps": float(up.mean() * 1e4) if len(up) else float("nan"),
            "welch_t": welch_t(cond, up),
        })
    return pd.DataFrame(rows).set_index("pair")


def other_weekday_check(df: pd.DataFrame) -> dict:
    """Pool the four NON-Monday->Tuesday down-day -> next-day reversals and Welch-t
    the pooled result — the honest test of whether "turnaround Tuesday" is really
    Monday-specific or just generic short-horizon reversal wearing a costume."""
    cond_all, up_all = [], []
    for d in range(1, 5):
        nd = (d + 1) % 5
        mask = (df["weekday"] == nd) & (df["prev_weekday"] == d)
        sub = df.loc[mask]
        down = sub["prev_ret"] < 0
        cond_all.append(sub.loc[down, "ret"].to_numpy())
        up_all.append(sub.loc[~down, "ret"].to_numpy())
    cond_all = np.concatenate(cond_all)
    up_all = np.concatenate(up_all)
    return {"n_down": len(cond_all), "down_bps": float(cond_all.mean() * 1e4),
            "up_bps": float(up_all.mean() * 1e4), "welch_t": welch_t(cond_all, up_all)}


# --------------------------------------------------------------------------- #
# Era contrast (justified split: pre/post-2000, named in the brief)
# --------------------------------------------------------------------------- #
def era_contrast(df: pd.DataFrame, pairs: pd.DataFrame, split: str) -> dict:
    """Down-Monday-Tuesday mean before vs since ``split``: within-era Welch t's
    (vs the era's own unconditional Tuesday) + Welch t OF THE DIFFERENCE between
    the two eras' down-Monday-Tuesday means."""
    early_pairs = pairs[pairs.index < split]
    late_pairs = pairs[pairs.index >= split]
    early_cond = early_pairs.loc[early_pairs["down_monday"], "tuesday_ret"].to_numpy()
    late_cond = late_pairs.loc[late_pairs["down_monday"], "tuesday_ret"].to_numpy()
    early_uncond = df.loc[(df["weekday"] == 1) & (df.index < split), "ret"].to_numpy()
    late_uncond = df.loc[(df["weekday"] == 1) & (df.index >= split), "ret"].to_numpy()
    return {"n_early": len(early_cond), "n_late": len(late_cond),
            "early_bps": float(early_cond.mean() * 1e4), "late_bps": float(late_cond.mean() * 1e4),
            "welch_t_early": welch_t(early_cond, early_uncond),
            "welch_t_late": welch_t(late_cond, late_uncond),
            "welch_t_diff": welch_t(late_cond, early_cond)}


# --------------------------------------------------------------------------- #
# Robustness — drop the two big crisis windows
# --------------------------------------------------------------------------- #
CRISIS_WINDOWS = [("2008-09-01", "2009-06-30"), ("2020-02-15", "2020-04-30")]


def excl_crisis_check(pairs: pd.DataFrame) -> dict:
    """Down-Monday-Tuesday mean with the 2008-09 GFC and 2020-03 COVID crash windows
    dropped, vs up-Monday Tuesdays (all) — is the effect just a few huge crisis days?"""
    cond = pairs.loc[pairs["down_monday"]].copy()
    keep = pd.Series(True, index=cond.index)
    for lo, hi in CRISIS_WINDOWS:
        keep &= ~((cond.index >= lo) & (cond.index <= hi))
    clean = cond.loc[keep, "tuesday_ret"].to_numpy()
    up_all = pairs.loc[~pairs["down_monday"], "tuesday_ret"].to_numpy()
    return {"n": len(clean), "mean_bps": float(clean.mean() * 1e4),
            "welch_t": welch_t(clean, up_all)}


# --------------------------------------------------------------------------- #
# The timer — buy Monday close after a down Monday, exit Tuesday close
# --------------------------------------------------------------------------- #
def timer_stats(pairs: pd.DataFrame, cost_bps: float = 5.0, years: float = 33.4) -> dict:
    """Enter at the Monday close (the down-Monday flag is knowable at that instant —
    it IS that close), exit at the Tuesday close. One round trip = 2 x one-way cost
    x NAV. Long-only, no borrow."""
    cond = pairs.loc[pairs["down_monday"], "tuesday_ret"].to_numpy()
    gross = float(cond.mean())
    net_arr = cond - 2.0 * cost_bps / 1e4
    net = float(net_arr.mean())
    events_per_year = len(cond) / years
    sd = float(net_arr.std(ddof=1))
    sharpe = net / sd * np.sqrt(events_per_year) if sd > 0 else float("nan")
    return {"n": len(cond), "events_per_year": events_per_year,
            "gross_bps": gross * 1e4, "net_bps": net * 1e4,
            "ann_net_pct": net * events_per_year * 100,
            "sharpe_net": sharpe, "t_net": one_sample_t(net_arr),
            "worst_pct": float(net_arr.min() * 100)}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(close: pd.DataFrame) -> dict:
    """Run the headline down-vs-up Welch split on a synthetic world."""
    df = day_frame(close["Close"])
    pairs = monday_tuesday_pairs(df)
    cond = pairs.loc[pairs["down_monday"], "tuesday_ret"].to_numpy()
    up = pairs.loc[~pairs["down_monday"], "tuesday_ret"].to_numpy()
    return {"n_down": len(cond), "cond_bps": float(cond.mean() * 1e4),
            "up_bps": float(up.mean() * 1e4), "welch_t": welch_t(cond, up)}
