"""Strategy + inference for Study 363 — Post-Earnings-Announcement Drift (PEAD).

The claim: after an earnings surprise, a stock keeps **drifting the same way** for weeks —
positive surprises keep rising, negative surprises keep falling. We test it with a clean
event study on a fixed large-cap basket:

    * **Surprise proxy** = the post-announcement **gap** (the one-day reaction the session
      after the print). Big positive gap = good surprise; big negative = bad.
    * **Drift** = the cumulative return *after* the reaction is already known. We observe the
      gap at the reaction-session close, then enter **one day later** (no look-ahead) and hold
      H ∈ {1, 5, 20, 60} trading days.
    * **The PEAD signal** = a long-short: go long the top surprise quintile, short the bottom,
      and ask whether the *drift* spread is positive and significant.

Inference, the desk's shared spirit:

  * a **Welch t** of the long-short drift mean against 0 (and of top vs bottom buckets);
  * a **placebo / label-shuffle** null — permute the surprise labels across events many times
    and ask how often a random quintile split yields as large a long-short drift (the honest
    test for a noisy cross-sectional effect);
  * a **win-rate** of the long-short trade vs the 50% coin-flip base rate;
  * **one-day execution lag** and **one-way costs × turnover** (every event is a fresh trade).

The decisive number is the long-short drift net of costs: PEAD is a real, documented effect
gross, but on a tradable large-cap basket the drift is small, concentrated in the first days,
and one-way costs on per-event turnover bite hard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 5, 20, 60)            # trading-day drift horizons


# --------------------------------------------------------------------------- #
# Drift measurement
# --------------------------------------------------------------------------- #
def drift_returns(prices: pd.DataFrame, events: pd.DataFrame, horizon: int,
                  lag: int = 1) -> np.ndarray:
    """Post-event drift return over ``horizon`` trading days, per event, with a 1-day lag.

    ``events`` carries ``ticker`` and ``t1_idx`` (integer position of the reaction session in
    that ticker's own price series). We observe the gap at ``t1`` close, enter ``lag`` days
    later (``t1 + lag`` close), and exit ``horizon`` days after entry. Events whose window
    overruns the tape are dropped. Returns drift aligned to ``events`` row order (NaN where
    dropped is removed — use :func:`event_drift_frame` to keep alignment).
    """
    out = []
    for _, r in events.iterrows():
        tk = r["ticker"]
        px = prices[tk].dropna().values
        i = int(r["t1_idx"])
        entry = i + lag
        exit_ = entry + horizon
        if entry < 0 or exit_ >= len(px):
            continue
        out.append(px[exit_] / px[entry] - 1.0)
    return np.asarray(out, dtype=float)


def event_drift_frame(prices: pd.DataFrame, events: pd.DataFrame,
                      horizon: int, lag: int = 1) -> pd.DataFrame:
    """Events with a ``drift`` column attached (rows whose window overruns are dropped)."""
    drifts = []
    keep = []
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
# Long-short by surprise quintile
# --------------------------------------------------------------------------- #
def _bucketize(surprise: np.ndarray, n_buckets: int = 5) -> np.ndarray:
    """Rank ``surprise`` into ``n_buckets`` equal-frequency bins (0 = lowest)."""
    order = np.argsort(np.argsort(surprise))      # dense ranks 0..n-1
    return np.minimum((order * n_buckets // len(surprise)), n_buckets - 1)


def long_short_drift(frame: pd.DataFrame, surprise_col: str = "gap",
                     n_buckets: int = 5) -> dict:
    """Top-minus-bottom surprise-quintile drift on a frame carrying ``drift``.

    Sort events by the surprise proxy into ``n_buckets`` quintiles; the PEAD long-short is the
    mean drift of the top bucket minus the mean drift of the bottom bucket. Returns the bucket
    means, the long-short spread, its per-event sample (top drifts minus the bottom mean, and
    vice versa, pooled) for the Welch t, and the long-short win-rate.
    """
    s = frame[surprise_col].values
    d = frame["drift"].values
    b = _bucketize(s, n_buckets)
    top = d[b == n_buckets - 1]
    bot = d[b == 0]
    ls_mean = float(top.mean() - bot.mean())
    # pooled per-event long-short sample: longs contribute +drift, shorts contribute -drift
    ls_sample = np.concatenate([top, -bot])
    win = float((ls_sample > 0).mean())
    return {
        "n_top": int(len(top)), "n_bot": int(len(bot)),
        "top_mean": float(top.mean()), "bot_mean": float(bot.mean()),
        "ls_mean": ls_mean, "ls_sample": ls_sample, "ls_win": win,
        "top": top, "bot": bot,
    }


def bucket_means(frame: pd.DataFrame, surprise_col: str = "gap",
                 n_buckets: int = 5) -> np.ndarray:
    """Mean drift in each surprise bucket (low -> high) — the PEAD monotonicity picture."""
    s = frame[surprise_col].values
    d = frame["drift"].values
    b = _bucketize(s, n_buckets)
    return np.array([d[b == k].mean() if (b == k).any() else np.nan
                     for k in range(n_buckets)])


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def welch_t(sample: np.ndarray, base: np.ndarray) -> float:
    """Welch t of mean(sample) - mean(base) (unequal variance). NaN if either < 2."""
    if len(sample) < 2 or len(base) < 2:
        return float("nan")
    m1, m0 = sample.mean(), base.mean()
    v1 = sample.var(ddof=1) / len(sample)
    v0 = base.var(ddof=1) / len(base)
    se = np.sqrt(v1 + v0)
    if se == 0:
        return float("nan")
    return float((m1 - m0) / se)


def ttest_vs_zero(sample: np.ndarray) -> float:
    """One-sample t of ``sample`` against 0 (the long-short edge test)."""
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    if se == 0:
        return float("nan")
    return float(sample.mean() / se)


def placebo_pvalue(frame: pd.DataFrame, horizon_label: str = "gap",
                   n_draws: int = 20_000, n_buckets: int = 5, seed: int = 363) -> dict:
    """Label-shuffle placebo null for the long-short drift.

    Permute the surprise labels across events many times; each shuffle re-forms random
    quintiles and recomputes the top-minus-bottom drift. ``p`` = P[shuffled long-short >=
    observed] — the honest answer to "could a random sort of the same drifts have looked this
    good?".
    """
    s = frame[horizon_label].values
    d = frame["drift"].values
    obs = float(long_short_drift(frame, horizon_label, n_buckets)["ls_mean"])
    n = len(d)
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    cut_hi = n_buckets - 1
    for i in range(n_draws):
        b = _bucketize(rng.permutation(s), n_buckets)
        means[i] = d[b == cut_hi].mean() - d[b == 0].mean()
    p = float((means >= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()), "p_value": p,
            "draws": means}


def summarize(prices: pd.DataFrame, events: pd.DataFrame, horizon: int,
              surprise_col: str = "gap", lag: int = 1, n_buckets: int = 5,
              placebo: bool = True, n_draws: int = 20_000) -> dict:
    """Headline stats for one drift horizon: bucket means, long-short mean/t/win/placebo."""
    fr = event_drift_frame(prices, events, horizon, lag=lag)
    ls = long_short_drift(fr, surprise_col, n_buckets)
    t = ttest_vs_zero(ls["ls_sample"])
    p = placebo_pvalue(fr, surprise_col, n_draws=n_draws,
                       n_buckets=n_buckets)["p_value"] if placebo else float("nan")
    return {
        "horizon": horizon, "n_events": int(len(fr)),
        "n_top": ls["n_top"], "n_bot": ls["n_bot"],
        "top_mean": ls["top_mean"], "bot_mean": ls["bot_mean"],
        "ls_mean": ls["ls_mean"], "ls_win": ls["ls_win"],
        "t": t, "p_placebo": p,
    }


def net_of_costs(prices: pd.DataFrame, events: pd.DataFrame, horizon: int,
                 cost_bps: float = 10.0, borrow_bps_ann: float = 50.0,
                 surprise_col: str = "gap", lag: int = 1, n_buckets: int = 5) -> dict:
    """Long-short drift net of one-way costs × turnover (+ borrow on the short leg).

    Every event is a fresh round trip: ``cost_bps`` one-way per leg per side (2 legs long +
    2 legs short = 4 × one-way ≈ ``4 * cost_bps`` on the long-short), and the short leg pays
    ``borrow_bps_ann`` annualised over the holding period. Returns gross/net long-short means.
    """
    fr = event_drift_frame(prices, events, horizon, lag=lag)
    ls = long_short_drift(fr, surprise_col, n_buckets)
    c = cost_bps / 1e4
    round_trip = 4.0 * c                              # long in/out + short in/out
    borrow = (borrow_bps_ann / 1e4) * (horizon / 252.0)
    gross = ls["ls_mean"]
    net = gross - round_trip - borrow
    return {"horizon": horizon, "n_top": ls["n_top"], "n_bot": ls["n_bot"],
            "gross": float(gross), "net": float(net),
            "cost_bps": cost_bps, "borrow_bps_ann": borrow_bps_ann}
