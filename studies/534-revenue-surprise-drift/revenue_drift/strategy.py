"""Strategy + inference for Study 534 — Revenue-Surprise-Drift (Jegadeesh & Livnat 2006).

The claim: post-earnings drift follows the **revenue (sales) surprise**, not just the EPS
surprise — top-sales-surprise names keep drifting up for weeks, and (Jegadeesh-Livnat's central
point) this revenue signal adds information *beyond* the EPS surprise. We test it with a clean
event study on a fixed large-cap basket:

    * **Surprise** = the **standardized unexpected revenue (SUR)**: the seasonal random-walk
      revenue surprise (revenue minus the same quarter a year ago) scaled by the trailing
      volatility of those seasonal differences (see ``data.compute_sur``). The revenue analogue
      of academic SUE.
    * **Drift** = the cumulative return *after* the revenue figure is public. We anchor at the
      first session on/after the 10-Q/10-K filing date, then enter **one day later** (no
      look-ahead) and hold H ∈ {1, 5, 20, 60} trading days.
    * **The signal** = a long-short: long the top SUR quintile, short the bottom, and ask whether
      the *drift* spread is positive and significant.

Inference, the desk's shared spirit:

  * a **one-sample t** of the pooled long-short drift sample against 0;
  * a **placebo / label-shuffle** null — permute SUR labels across events many times and ask how
    often a random quintile split yields as large a long-short drift;
  * a **win-rate** vs the 50% coin-flip base rate;
  * **one-day execution lag** and **one-way costs × turnover** (+ borrow on the short leg).

Third axis (Jegadeesh-Livnat's contribution): does the revenue drift add information *beyond*
the EPS surprise? ``incremental_to_eps`` re-runs the SUR long-short within EPS-sign strata.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 5, 20, 60)            # trading-day drift horizons


# --------------------------------------------------------------------------- #
# Drift measurement
# --------------------------------------------------------------------------- #
def event_drift_frame(prices: pd.DataFrame, events: pd.DataFrame,
                      horizon: int, lag: int = 1) -> pd.DataFrame:
    """Events with a ``drift`` column attached (rows whose window overruns are dropped).

    ``events`` carries ``ticker`` and ``t1_idx`` (integer position of the first session after the
    filing in that ticker's own price series). We enter ``lag`` days later (``t1 + lag`` close)
    and exit ``horizon`` days after entry — strictly after the revenue figure is public.
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
# Long-short by SUR quintile
# --------------------------------------------------------------------------- #
def _bucketize(surprise: np.ndarray, n_buckets: int = 5) -> np.ndarray:
    """Rank ``surprise`` into ``n_buckets`` equal-frequency bins (0 = lowest)."""
    order = np.argsort(np.argsort(surprise))      # dense ranks 0..n-1
    return np.minimum((order * n_buckets // len(surprise)), n_buckets - 1)


def long_short_drift(frame: pd.DataFrame, surprise_col: str = "sur",
                     n_buckets: int = 5) -> dict:
    """Top-minus-bottom SUR-quintile drift on a frame carrying ``drift``.

    Sort events by the surprise into ``n_buckets`` quintiles; the long-short is the mean drift of
    the top bucket minus the mean drift of the bottom bucket. Returns the bucket means, the
    long-short spread, the pooled per-event long-short sample (top drifts and minus bottom
    drifts) for the one-sample t, and the long-short win-rate.
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


def bucket_means(frame: pd.DataFrame, surprise_col: str = "sur",
                 n_buckets: int = 5) -> np.ndarray:
    """Mean drift in each surprise bucket (low -> high) — the monotonicity picture."""
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
    se = np.sqrt(sample.var(ddof=1) / len(sample) + base.var(ddof=1) / len(base))
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


def placebo_pvalue(frame: pd.DataFrame, surprise_col: str = "sur",
                   n_draws: int = 20_000, n_buckets: int = 5, seed: int = 534) -> dict:
    """Label-shuffle placebo null for the long-short drift.

    Permute the SUR labels across events many times; each shuffle re-forms random quintiles and
    recomputes the top-minus-bottom drift. ``p`` = P[shuffled long-short >= observed].
    """
    s = frame[surprise_col].values
    d = frame["drift"].values
    obs = float(long_short_drift(frame, surprise_col, n_buckets)["ls_mean"])
    n = len(d)
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    cut_hi = n_buckets - 1
    for i in range(n_draws):
        b = _bucketize(rng.permutation(s), n_buckets)
        means[i] = d[b == cut_hi].mean() - d[b == 0].mean()
    p = float((means >= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()), "p_value": p, "draws": means}


def block_placebo_pvalue(frame: pd.DataFrame, surprise_col: str = "sur",
                         n_draws: int = 5_000, n_buckets: int = 5, seed: int = 534) -> float:
    """Within-quarter block placebo: shuffle SUR labels *inside each calendar quarter*.

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


def summarize(prices: pd.DataFrame, events: pd.DataFrame, horizon: int,
              surprise_col: str = "sur", lag: int = 1, n_buckets: int = 5,
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
                 surprise_col: str = "sur", lag: int = 1, n_buckets: int = 5) -> dict:
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
# Third axis — does the revenue drift add information beyond the EPS surprise?
# --------------------------------------------------------------------------- #
def attach_eps(frame: pd.DataFrame, eps_events: pd.DataFrame,
               window_days: int = 25) -> pd.DataFrame:
    """Attach the nearest reported EPS surprise to each revenue event (same ticker, ±window).

    ``eps_events`` is the EPS event table from study 363's cache (columns ticker, date,
    surprise_pct). A revenue filing and its earnings print are days apart, so we match each
    revenue event to the closest EPS event of the same ticker within ``window_days``. Rows
    without a match get NaN ``eps_pct`` and are excluded from the incremental test.
    """
    out = frame.copy()
    eps_pct = np.full(len(out), np.nan)
    ev_by_tk = {tk: g.sort_values("date") for tk, g in eps_events.groupby("ticker")}
    for j, (_, r) in enumerate(out.iterrows()):
        g = ev_by_tk.get(r["ticker"])
        if g is None:
            continue
        f = pd.Timestamp(r["filed"])
        dt = (g["date"] - f).abs()
        k = dt.values.argmin()
        if dt.iloc[k].days <= window_days:
            eps_pct[j] = float(g["surprise_pct"].values[k])
    out["eps_pct"] = eps_pct
    return out


def incremental_to_eps(frame: pd.DataFrame, surprise_col: str = "sur",
                       n_buckets: int = 5) -> dict:
    """SUR long-short drift *within EPS-sign strata* — Jegadeesh-Livnat's incremental claim.

    If the revenue drift were merely the EPS drift in disguise, then once we hold the **sign of
    the EPS surprise** fixed (good-EPS vs bad-EPS events), the SUR long-short should vanish. We
    run the top-minus-bottom SUR sort separately inside the positive-EPS and negative-EPS
    subsets and pool the per-event long-short samples; a surviving t means revenue surprise
    carries information *beyond* EPS.
    """
    fr = frame.dropna(subset=["eps_pct"]).copy()
    pooled = []
    strata = {}
    for label, mask in (("eps+", fr["eps_pct"] > 0), ("eps-", fr["eps_pct"] <= 0)):
        sub = fr[mask]
        if len(sub) < 4 * n_buckets:
            strata[label] = {"n": int(len(sub)), "ls_mean": float("nan"), "t": float("nan")}
            continue
        ls = long_short_drift(sub, surprise_col, n_buckets)
        pooled.append(ls["ls_sample"])
        strata[label] = {"n": int(len(sub)), "ls_mean": ls["ls_mean"],
                         "t": ttest_vs_zero(ls["ls_sample"])}
    pooled_sample = np.concatenate(pooled) if pooled else np.array([])
    return {"n_matched": int(len(fr)), "strata": strata,
            "pooled_ls_mean": float(pooled_sample.mean()) if len(pooled_sample) else float("nan"),
            "pooled_t": ttest_vs_zero(pooled_sample) if len(pooled_sample) else float("nan")}
