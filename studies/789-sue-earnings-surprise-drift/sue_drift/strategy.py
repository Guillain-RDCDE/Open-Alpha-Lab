"""Strategy + inference for Study 789 — SUE Earnings-Surprise Drift.

The claim (Bernard & Thomas 1989): after a quarterly print, prices keep drifting in the
direction of the **standardized unexpected earnings** (SUE) for weeks — top-SUE names keep
rising, bottom-SUE names keep falling. We test it as a clean event study on a fixed large-cap
basket:

    * **Surprise** = **SUE**: the seasonal-random-walk EPS surprise (``EPS_q − EPS_{q-4}``)
      scaled by the rolling volatility of the last ~8 such surprises (see ``data.compute_sue``).
    * **Drift** = the cumulative return *after* the EPS figure is public. We anchor at the first
      session on/after the 10-Q/10-K filing date, then enter **one day later** (the single
      documented execution lag — signal known at the filing, return earned from t+1) and hold
      H ∈ {21, 42, 63} trading days (~1 / 2 / 3 months).
    * **The signal** = a long-short: long the top SUE tercile, short the bottom, and ask whether
      the *drift* spread is positive and significant.

Inference, the desk's shared spirit — and the honest issue with an event study is that filings
**cluster in earnings seasons**, so pooled per-event returns are neither independent nor
serially uncorrelated. We therefore report, in order of authority:

  * a **calendar-time long-short** aggregated to earnings-season buckets, with a **Newey-West
    (HAC) t** — the autocorrelation-robust headline (the desk's t≥2 bar is judged here);
  * a **within-quarter block placebo** — shuffle SUE labels *inside each calendar quarter*,
    respecting the seasonal clustering;
  * a naive **one-sample t** of the pooled long-short sample (reported, but known to overstate);
  * a **Welch t** of top vs bottom bucket, a **win-rate with a Wilson interval**, a global
    label-shuffle placebo;
  * **one-day execution lag** and **one-way costs × turnover** (+ borrow on the short leg).

The synthetic control (``data.synthetic_sue``) plants a known post-event drift; with the edge
set to zero the inference must NOT manufacture significance (a faithful-engine / power check
only — never cited in support of the real-tape stamp).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (21, 42, 63)            # trading-day drift horizons ~ 1 / 2 / 3 months


# --------------------------------------------------------------------------- #
# Drift measurement
# --------------------------------------------------------------------------- #
def event_drift_frame(prices: pd.DataFrame, events: pd.DataFrame,
                      horizon: int, lag: int = 1) -> pd.DataFrame:
    """Events with a ``drift`` column attached (rows whose window overruns are dropped).

    ``events`` carries ``ticker`` and ``t1_idx`` (integer position of the first session after
    the filing in that ticker's own price series). We enter ``lag`` days later (``t1 + lag``
    close) and exit ``horizon`` days after entry — strictly after the EPS figure is public.
    """
    drifts, keep = [], []
    for j, (_, r) in enumerate(events.iterrows()):
        tk = r["ticker"]
        px = prices[tk].dropna().values
        i = int(r["t1_idx"])
        entry = i + lag
        exit_ = entry + horizon
        if entry < 0 or exit_ >= len(px):
            continue
        drifts.append(px[exit_] / px[entry] - 1.0)
        keep.append(j)
    out = events.iloc[keep].copy()
    out["drift"] = drifts
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Long-short by SUE bucket
# --------------------------------------------------------------------------- #
def _bucketize(surprise: np.ndarray, n_buckets: int = 3) -> np.ndarray:
    """Rank ``surprise`` into ``n_buckets`` equal-frequency bins (0 = lowest)."""
    order = np.argsort(np.argsort(surprise))      # dense ranks 0..n-1
    return np.minimum((order * n_buckets // len(surprise)), n_buckets - 1)


def long_short_drift(frame: pd.DataFrame, surprise_col: str = "sue",
                     n_buckets: int = 3) -> dict:
    """Top-minus-bottom SUE-bucket drift on a frame carrying ``drift``.

    Sort events by the surprise into ``n_buckets`` buckets (terciles by default); the long-short
    is the mean drift of the top bucket minus the mean drift of the bottom bucket. Returns the
    bucket means, the long-short spread, the pooled per-event long-short sample (top drifts and
    minus bottom drifts) for the one-sample t, and the long-short win-rate.
    """
    s = frame[surprise_col].values
    d = frame["drift"].values
    b = _bucketize(s, n_buckets)
    top = d[b == n_buckets - 1]
    bot = d[b == 0]
    ls_mean = float(top.mean() - bot.mean())
    ls_sample = np.concatenate([top, -bot])
    win = float((ls_sample > 0).mean())
    return {
        "n_top": int(len(top)), "n_bot": int(len(bot)),
        "top_mean": float(top.mean()), "bot_mean": float(bot.mean()),
        "ls_mean": ls_mean, "ls_sample": ls_sample, "ls_win": win,
        "top": top, "bot": bot,
    }


def bucket_means(frame: pd.DataFrame, surprise_col: str = "sue",
                 n_buckets: int = 5) -> np.ndarray:
    """Mean drift in each surprise bucket (low -> high) — the monotonicity picture."""
    s = frame[surprise_col].values
    d = frame["drift"].values
    b = _bucketize(s, n_buckets)
    return np.array([d[b == k].mean() if (b == k).any() else np.nan
                     for k in range(n_buckets)])


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def welch_t(sample: np.ndarray, base: np.ndarray) -> float:
    """Welch t of mean(sample) - mean(base) (unequal variance). NaN if either < 2."""
    sample = np.asarray(sample, float)
    base = np.asarray(base, float)
    if len(sample) < 2 or len(base) < 2:
        return float("nan")
    se = np.sqrt(sample.var(ddof=1) / len(sample) + base.var(ddof=1) / len(base))
    if se == 0:
        return float("nan")
    return float((sample.mean() - base.mean()) / se)


def ttest_vs_zero(sample: np.ndarray) -> float:
    """One-sample t of ``sample`` against 0 (the long-short edge test)."""
    sample = np.asarray(sample, float)
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    if se == 0:
        return float("nan")
    return float(sample.mean() / se)


def newey_west_t(y: np.ndarray, lags: int = 2) -> tuple[float, float]:
    """HAC (Newey-West, Bartlett kernel) t and mean of a scalar time series ``y`` vs 0.

    Regress ``y`` on a constant; the coefficient is ``mean(y)`` and the NW standard error is
    robust to the serial correlation induced by overlapping drift windows. Returns (t, mean).
    """
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    n = len(y)
    if n < 3:
        return float("nan"), (float(y.mean()) if n else float("nan"))
    u = y - y.mean()
    S = float(u @ u)
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        S += 2.0 * w * float(u[l:] @ u[:-l])
    var = S / (n * n)                    # variance of the mean, HAC
    se = np.sqrt(var) if var > 0 else float("nan")
    return (float(y.mean() / se) if se and se > 0 else float("nan"), float(y.mean()))


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
# Calendar-time long-short (the autocorrelation-robust headline)
# --------------------------------------------------------------------------- #
def calendar_time_ls(frame: pd.DataFrame, surprise_col: str = "sue",
                     n_buckets: int = 3, freq: str = "Q", nw_lags: int = 2) -> dict:
    """Aggregate the top-minus-bottom drift into calendar (earnings-season) buckets, then HAC.

    The events are bucketed **once** on the full-sample SUE distribution (equal-frequency
    terciles). Each event is then assigned to the calendar period (``freq``) of its entry; within
    each period the long-short is (mean top drift − mean bottom drift) using only that period's
    events. This yields a calendar-time long-short series whose autocorrelation (overlapping
    H-day windows across adjacent seasons) is absorbed by a **Newey-West** t. Periods that lack
    both a top and a bottom event are dropped.

    Returns the per-period series, its NW t and mean, and the count of periods used.
    """
    s = frame[surprise_col].values
    d = frame["drift"].values
    b = _bucketize(s, n_buckets)
    fr = frame.copy()
    fr["_top"] = (b == n_buckets - 1)
    fr["_bot"] = (b == 0)
    fr["_drift"] = d
    per = pd.PeriodIndex(pd.to_datetime(fr["filed"]), freq=freq)
    fr["_per"] = per.astype(str)
    series = []
    for p, g in fr.groupby("_per"):
        t = g.loc[g["_top"], "_drift"]
        bo = g.loc[g["_bot"], "_drift"]
        if len(t) >= 1 and len(bo) >= 1:
            series.append((p, float(t.mean() - bo.mean())))
    series = sorted(series)
    vals = np.array([v for _, v in series], float)
    nw_t, mean = newey_west_t(vals, lags=nw_lags)
    return {"n_periods": len(vals), "series": vals, "labels": [p for p, _ in series],
            "nw_t": nw_t, "mean": mean}


# --------------------------------------------------------------------------- #
# Placebos
# --------------------------------------------------------------------------- #
def placebo_pvalue(frame: pd.DataFrame, surprise_col: str = "sue",
                   n_draws: int = 20_000, n_buckets: int = 3, seed: int = 789) -> dict:
    """Global label-shuffle placebo null for the long-short drift.

    Permute the SUE labels across events many times; each shuffle re-forms random buckets and
    recomputes the top-minus-bottom drift. ``p`` = P[shuffled long-short >= observed].
    """
    s = frame[surprise_col].values
    d = frame["drift"].values
    obs = float(long_short_drift(frame, surprise_col, n_buckets)["ls_mean"])
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    cut_hi = n_buckets - 1
    for i in range(n_draws):
        b = _bucketize(rng.permutation(s), n_buckets)
        means[i] = d[b == cut_hi].mean() - d[b == 0].mean()
    p = float((means >= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)), "p_value": p, "draws": means}


def block_placebo_pvalue(frame: pd.DataFrame, surprise_col: str = "sue",
                         n_draws: int = 5_000, n_buckets: int = 3, seed: int = 789) -> float:
    """Within-quarter block placebo: shuffle SUE labels *inside each calendar quarter*.

    Earnings/filings cluster in seasons, so events are not independent. Shuffling labels only
    within calendar quarter respects that clustering — a stricter null than the global shuffle.
    Returns P[block-shuffled long-short >= observed].
    """
    fr = frame.copy()
    fr["_q"] = pd.PeriodIndex(pd.to_datetime(fr["filed"]), freq="Q").astype(str)
    s = fr[surprise_col].values.copy()
    d = fr["drift"].values
    groups = {q: np.where(fr["_q"].values == q)[0] for q in fr["_q"].unique()}
    obs = float(long_short_drift(fr, surprise_col, n_buckets)["ls_mean"])
    rng = np.random.default_rng(seed)
    cut_hi = n_buckets - 1
    hits = 0
    for _ in range(n_draws):
        sh = s.copy()
        for idxs in groups.values():
            if len(idxs) > 1:
                sh[idxs] = rng.permutation(s[idxs])
        b = _bucketize(sh, n_buckets)
        ls = d[b == cut_hi].mean() - d[b == 0].mean()
        if ls >= obs:
            hits += 1
    return float(hits / n_draws)


# --------------------------------------------------------------------------- #
# Headline summary + costs
# --------------------------------------------------------------------------- #
def summarize(prices: pd.DataFrame, events: pd.DataFrame, horizon: int,
              surprise_col: str = "sue", lag: int = 1, n_buckets: int = 3,
              placebo: bool = True, n_draws: int = 20_000, nw_freq: str = "Q") -> dict:
    """Headline stats for one drift horizon: bucket means, long-short mean, one-sample t,
    calendar-time NW t, win rate + Wilson, and (optionally) the global placebo p."""
    fr = event_drift_frame(prices, events, horizon, lag=lag)
    ls = long_short_drift(fr, surprise_col, n_buckets)
    t = ttest_vs_zero(ls["ls_sample"])
    ct = calendar_time_ls(fr, surprise_col, n_buckets=n_buckets, freq=nw_freq)
    k = int((ls["ls_sample"] > 0).sum())
    n_ls = len(ls["ls_sample"])
    lo, hi = wilson_interval(k, n_ls)
    p = placebo_pvalue(fr, surprise_col, n_draws=n_draws,
                       n_buckets=n_buckets)["p_value"] if placebo else float("nan")
    return {
        "horizon": horizon, "n_events": int(len(fr)),
        "n_top": ls["n_top"], "n_bot": ls["n_bot"],
        "top_mean": ls["top_mean"], "bot_mean": ls["bot_mean"],
        "ls_mean": ls["ls_mean"], "ls_win": ls["ls_win"],
        "win_lo": lo, "win_hi": hi,
        "t": t, "nw_t": ct["nw_t"], "nw_periods": ct["n_periods"], "p_placebo": p,
    }


def net_of_costs(prices: pd.DataFrame, events: pd.DataFrame, horizon: int,
                 cost_bps: float = 10.0, borrow_bps_ann: float = 50.0,
                 surprise_col: str = "sue", lag: int = 1, n_buckets: int = 3) -> dict:
    """Long-short drift net of one-way costs × turnover (+ borrow on the short leg).

    Every event is a fresh round trip: ``cost_bps`` one-way per leg per side (2 legs long + 2
    legs short = 4 × one-way on the long-short), and the short leg pays ``borrow_bps_ann``
    annualised over the holding period. Returns gross/net long-short means.
    """
    fr = event_drift_frame(prices, events, horizon, lag=lag)
    ls = long_short_drift(fr, surprise_col, n_buckets)
    c = cost_bps / 1e4
    round_trip = 4.0 * c
    borrow = (borrow_bps_ann / 1e4) * (horizon / 252.0)
    gross = ls["ls_mean"]
    net = gross - round_trip - borrow
    return {"horizon": horizon, "n_top": ls["n_top"], "n_bot": ls["n_bot"],
            "gross": float(gross), "net": float(net),
            "cost_bps": cost_bps, "borrow_bps_ann": borrow_bps_ann}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, events: pd.DataFrame, horizon: int = 63,
                     n_buckets: int = 3) -> dict:
    """Run the headline long-short + one-sample t on a synthetic world."""
    fr = event_drift_frame(prices, events, horizon, lag=1)
    ls = long_short_drift(fr, "sue", n_buckets)
    return {"n_events": int(len(fr)), "ls_mean": ls["ls_mean"],
            "t": ttest_vs_zero(ls["ls_sample"])}
