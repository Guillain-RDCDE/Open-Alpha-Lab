"""Strategy + inference for Study 873 — Sentiment Beta.

The claim (Baker & Wurgler 2006, 2007): the stocks whose returns **co-move most with
market sentiment** — high *sentiment beta* — are the speculative, hard-to-value names
that get over-priced when sentiment is high and **under-perform afterwards**. Sort a
cross-section on each name's beta to a sentiment gauge; the theory says a long
**low-sentiment-beta** / short **high-sentiment-beta** book earns a *positive* spread,
and the effect is strongest **after sentiment has peaked**.

This is distinct from:

* [258-baker-wurgler](../../258-baker-wurgler/) — tests the **time-series / aggregate**
  Baker-Wurgler contrarian claim (high sentiment level → low *market* return next
  month), sorting nothing in the cross-section. This study tests the **cross-sectional
  sentiment-beta** leg: which *names* under-earn, ranked by their co-movement with the
  gauge.
* [255-fear-greed-index](../../255-fear-greed-index/) — a market-timing signal off a
  composite fear/greed gauge, again a time-series call, not a cross-sectional beta sort.
* [501-idiosyncratic-volatility](../../501-idiosyncratic-volatility/) and
  [330-low-volatility-anomaly](../../330-low-volatility-anomaly/) — sort on a name's
  **own volatility level**. Sentiment beta is the **co-movement** of a name with a
  market-wide sentiment *time series*; a high-vol name whose variance is idiosyncratic
  (not synced to the speculative leg) has a *low* sentiment beta, so the axes differ.

Method:

* **Sentiment gauge (tradable).** Proxy market sentiment with a daily high-minus-low
  realized-volatility spread built from the panel itself: the equal-weight return of
  the most-volatile (speculative) tercile minus the least-volatile (safe) tercile,
  ranked point-in-time on trailing volatility. It rises in risk-on euphoria.
* **Sentiment beta.** For each name, a rolling ``beta_window``-day OLS slope of its
  daily return on the gauge (vectorised via rolling covariance / variance). Value on
  row ``t`` uses data through ``t``.
* **Point-in-time sort.** On each day ``t`` rank the cross-section by the sentiment
  beta known at the close of ``t-1`` (one ``shift``) and hold day ``t``. Long the
  bottom ``frac`` (low beta), short the top ``frac`` (high beta); equal weight. A
  positive spread means the high-beta names under-earn — the claim's direction.
* **Conditional on sentiment level.** The claim says the under-performance is strongest
  *after sentiment peaks*; we split the spread by the trailing level of the gauge
  (a high-sentiment regime vs the rest).
* **Inference.** Newey-West (HAC) *t* on the daily long-short spread; a one-sample *t*
  and a pooled Welch *t* (low-beta book vs high-beta book) cross-check; a permutation
  placebo breaks the signal->outcome link; a costed timer charges the round-trip
  friction and short-leg borrow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Return panel
# --------------------------------------------------------------------------- #
def close_returns(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily simple close-to-close returns (index=date, columns=ticker)."""
    closes = pd.DataFrame({s: panel[s]["Close"] for s in panel}).sort_index()
    return closes.pct_change()


# --------------------------------------------------------------------------- #
# The sentiment gauge — a tradable high-minus-low-volatility spread
# --------------------------------------------------------------------------- #
def sentiment_gauge(
    ret: pd.DataFrame,
    vol_window: int = 63,
    frac: float = 0.3,
    min_names: int = 10,
) -> pd.Series:
    """Daily sentiment gauge: high-vol tercile return minus low-vol tercile return.

    On each day ``t`` names are ranked by their trailing ``vol_window``-day return
    volatility **known at the close of ``t-1``** (one ``shift``, no look-ahead). The
    gauge is the equal-weight day-``t`` return of the top ``frac`` (speculative, high
    vol) minus the bottom ``frac`` (safe, low vol) — it rises in risk-on euphoria when
    the lottery names are bid up. Returned as a date-indexed Series aligned to ``ret``.
    """
    vol = ret.rolling(vol_window, min_periods=vol_window).std().shift(1)
    V = vol.to_numpy(dtype=float)
    R = ret.to_numpy(dtype=float)
    idx = ret.index
    out = np.full(len(idx), np.nan)
    for i in range(len(idx)):
        row = V[i]
        valid = np.where(~np.isnan(row))[0]
        n = len(valid)
        if n < min_names:
            continue
        k = max(1, int(np.floor(n * frac)))
        order = valid[np.argsort(row[valid], kind="stable")]
        low = order[:k]        # low vol  -> safe leg
        high = order[-k:]      # high vol -> speculative leg
        rr = R[i]
        out[i] = float(np.nanmean(rr[high]) - np.nanmean(rr[low]))
    return pd.Series(out, index=idx, name="sentiment")


# --------------------------------------------------------------------------- #
# Sentiment beta — rolling slope of each name's return on the gauge
# --------------------------------------------------------------------------- #
def sentiment_beta(
    ret: pd.DataFrame,
    gauge: pd.Series,
    beta_window: int = 252,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Rolling ``beta_window``-day OLS slope of each name's return on the gauge.

    Vectorised via the covariance identity ``beta = cov(r_i, g) / var(g)`` with all
    moments computed as rolling means over ``beta_window`` days::

        beta_i[t] = (E[r_i g] - E[r_i] E[g]) / (E[g^2] - E[g]^2)

    Value on row ``t`` uses data through ``t`` (inclusive); the sort in
    :func:`beta_spreads` shifts by one day so a day-``t`` position is formed on the
    beta known at ``t-1``.
    """
    if min_periods is None:
        min_periods = beta_window
    g = gauge.reindex(ret.index)
    rg = ret.mul(g, axis=0)
    mean_r = ret.rolling(beta_window, min_periods=min_periods).mean()
    mean_rg = rg.rolling(beta_window, min_periods=min_periods).mean()
    mean_g = g.rolling(beta_window, min_periods=min_periods).mean()
    mean_g2 = (g ** 2).rolling(beta_window, min_periods=min_periods).mean()
    var_g = mean_g2 - mean_g ** 2
    cov = mean_rg.sub(mean_r.mul(mean_g, axis=0))
    beta = cov.div(var_g, axis=0)
    return beta.where(var_g > 0)


# --------------------------------------------------------------------------- #
# The cross-sectional sort -> long-low-beta / short-high-beta spread
# --------------------------------------------------------------------------- #
def beta_spreads(
    ret: pd.DataFrame,
    vol_window: int = 63,
    beta_window: int = 252,
    frac: float = 0.3,
    min_names: int = 10,
    gauge: pd.Series | None = None,
) -> pd.DataFrame:
    """Daily equal-weight bottom-minus-top sentiment-beta fractile spread.

    On each day ``t`` names are ranked by the sentiment beta known at the close of
    ``t-1`` (one ``shift``). ``lo`` = mean forward day-``t`` return of the bottom
    ``frac`` (low beta, the long); ``hi`` = mean of the top ``frac`` (high beta, the
    short). ``spread = lo - hi`` (long low-beta, short high-beta) — positive when the
    high-sentiment-beta names under-earn, the claim's direction. Days with fewer than
    ``min_names`` ranked names are dropped. A ``sent_level`` column carries the trailing
    level of the gauge for the post-peak conditional analysis.
    """
    if gauge is None:
        gauge = sentiment_gauge(ret, vol_window, frac, min_names)
    beta = sentiment_beta(ret, gauge, beta_window)
    sig = beta.shift(1)  # known at close t-1
    # Trailing sentiment level (momentum of the speculative leg), known at t-1.
    sent_level = gauge.rolling(beta_window, min_periods=beta_window // 2).mean().shift(1)
    S = sig.to_numpy(dtype=float)
    R = ret.to_numpy(dtype=float)
    L = sent_level.reindex(ret.index).to_numpy(dtype=float)
    idx = ret.index
    out_spread, out_lo, out_hi, out_n, out_lvl, out_t = [], [], [], [], [], []
    for i in range(len(idx)):
        row = S[i]
        valid = np.where(~np.isnan(row))[0]
        n = len(valid)
        if n < min_names:
            continue
        k = max(1, int(np.floor(n * frac)))
        order = valid[np.argsort(row[valid], kind="stable")]
        low = order[:k]        # low beta  -> long
        high = order[-k:]      # high beta -> short
        rr = R[i]
        lo = float(np.nanmean(rr[low]))
        hi = float(np.nanmean(rr[high]))
        out_spread.append(lo - hi); out_lo.append(lo); out_hi.append(hi)
        out_n.append(n); out_lvl.append(L[i]); out_t.append(idx[i])
    return pd.DataFrame(
        {"spread": out_spread, "lo": out_lo, "hi": out_hi, "n": out_n,
         "sent_level": out_lvl},
        index=out_t,
    ).sort_index()


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    gamma0 = float(u @ u) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        cov = float(u[l:] @ u[:-l]) / n
        var += 2.0 * w * cov
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Headline stats
# --------------------------------------------------------------------------- #
def beta_stats(spreads: pd.DataFrame, nw_lags: int = 10) -> dict:
    sp = spreads["spread"].to_numpy(dtype=float)
    return {
        "n_days": int(len(spreads)),
        "spread_bps": float(np.nanmean(sp) * 1e4),
        "t_nw": newey_west_t(sp, nw_lags),
        "t_1s": one_sample_t(sp),
        "lo_bps": float(np.nanmean(spreads["lo"].to_numpy()) * 1e4),
        "hi_bps": float(np.nanmean(spreads["hi"].to_numpy()) * 1e4),
        "welch_t": welch_t(spreads["lo"].to_numpy(), spreads["hi"].to_numpy()),
    }


def conditional_on_sentiment(spreads: pd.DataFrame, top_q: float = 0.7) -> dict:
    """Split the spread by the trailing sentiment level — the "after sentiment peaks"
    conditional. ``high`` = days whose ``sent_level`` sits in the top ``1-top_q``
    quantile (sentiment has been high); ``rest`` = the others. The claim predicts the
    long-low/short-high spread is *larger* (more positive) in the high-sentiment regime.
    """
    df = spreads.dropna(subset=["sent_level"])
    if df.empty:
        return {"cut": float("nan"), "high_bps": float("nan"),
                "high_t": float("nan"), "rest_bps": float("nan"),
                "rest_t": float("nan"), "n_high": 0, "n_rest": 0}
    cut = float(df["sent_level"].quantile(top_q))
    hi = df[df["sent_level"] >= cut]["spread"].to_numpy(dtype=float)
    lo = df[df["sent_level"] < cut]["spread"].to_numpy(dtype=float)
    return {
        "cut": cut,
        "high_bps": float(np.nanmean(hi) * 1e4), "high_t": newey_west_t(hi),
        "rest_bps": float(np.nanmean(lo) * 1e4), "rest_t": newey_west_t(lo),
        "n_high": int(len(hi)), "n_rest": int(len(lo)),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the spread real, or a lucky alignment of the sort?
# --------------------------------------------------------------------------- #
def placebo_pvalue(
    ret: pd.DataFrame,
    vol_window: int = 63,
    beta_window: int = 252,
    frac: float = 0.3,
    min_names: int = 10,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 873,
    gauge: pd.Series | None = None,
) -> dict:
    """Keep the sentiment-beta sort but read each day's forward return from a
    **column-permuted** panel (signal->outcome link broken, each day's cross-sectional
    distribution preserved). p = share of permuted worlds whose spread mean is >=
    observed (right-tail test on the long-low/short-high spread)."""
    if gauge is None:
        gauge = sentiment_gauge(ret, vol_window, frac, min_names)
    cols = list(ret.columns)
    ncol = len(cols)
    beta = sentiment_beta(ret, gauge, beta_window)
    sig = beta.shift(1)
    obs = float(beta_spreads(ret, vol_window, beta_window, frac, min_names,
                             gauge=gauge)["spread"].mean())

    ret_mat = ret.to_numpy(dtype=float)
    pos_of = {c: i for i, c in enumerate(cols)}
    rows_idx, lows, highs = [], [], []
    row_lookup = {t: r for r, t in enumerate(ret.index)}
    for t in ret.index:
        s = sig.loc[t].dropna()
        if len(s) < min_names:
            continue
        k = max(1, int(np.floor(len(s) * frac)))
        order = s.sort_values()
        rows_idx.append(row_lookup[t])
        lows.append(np.array([pos_of[c] for c in order.index[:k]]))
        highs.append(np.array([pos_of[c] for c in order.index[-k:]]))
    rows_idx = np.asarray(rows_idx)

    means = []
    if len(rows_idx):
        M = ret_mat[rows_idx]
        kl = max(len(a) for a in lows)
        kh = max(len(a) for a in highs)

        def _pad(books, kmax):
            P = np.zeros((len(books), kmax), dtype=int)
            V = np.zeros((len(books), kmax), dtype=bool)
            for j, a in enumerate(books):
                P[j, :len(a)] = a
                V[j, :len(a)] = True
            return P, V

        LOW, LOWv = _pad(lows, kl)
        HIGH, HIGHv = _pad(highs, kh)
        rows_ar = np.arange(len(rows_idx))[:, None]

        def _masked_mean(pos, valid, perm):
            vals = M[rows_ar, perm[pos]]
            vals = np.where(valid, vals, np.nan)
            return np.nanmean(vals, axis=1)

        for seed in range(n_seeds):
            rng = np.random.default_rng(base_seed + seed)
            for _ in range(n_draws_per_seed):
                perm = rng.permutation(ncol)
                lo_v = _masked_mean(LOW, LOWv, perm)
                hi_v = _masked_mean(HIGH, HIGHv, perm)
                means.append(np.nanmean(lo_v - hi_v))
    means = np.asarray(means)
    return {
        "obs_bps": obs * 1e4,
        "placebo_mean_bps": float(means.mean() * 1e4) if len(means) else float("nan"),
        "placebo_sd_bps": float(means.std(ddof=1) * 1e4) if len(means) > 1 else float("nan"),
        "p_value": float((means >= obs).mean()) if len(means) else float("nan"),
        "n_draws": len(means),
        "draws_bps": means * 1e4,
    }


# --------------------------------------------------------------------------- #
# The costed timer
# --------------------------------------------------------------------------- #
def timer_stats(
    spreads: pd.DataFrame,
    cost_bps: float = 5.0,
    borrow_bps_yr: float = 50.0,
) -> dict:
    """Cost the long-low-beta / short-high-beta book.

    The signal is a slow rolling beta that turns over gradually, but names drift across
    the fractile boundary daily; we charge a conservative daily round-trip on the
    2x-NAV long-short book (2 sides x one-way cost x NAV per day), plus borrow on the
    short leg — the same convention as the desk's other cross-sectional timers.
    """
    sp = spreads["spread"].to_numpy(dtype=float)
    sp = sp[~np.isnan(sp)]
    n = len(sp)
    round_trip_cost = 2.0 * cost_bps / 1e4
    borrow_daily = (borrow_bps_yr / 1e4) / 365.0
    net = sp - round_trip_cost - borrow_daily
    gross_mean = float(sp.mean())
    net_mean = float(net.mean())
    sd = float(net.std(ddof=1)) if n > 1 else float("nan")
    sharpe = net_mean / sd * np.sqrt(TRADING_DAYS) if sd and sd > 0 else float("nan")
    return {
        "n_days": n,
        "gross_bps": gross_mean * 1e4,
        "net_bps": net_mean * 1e4,
        "cost_bps_per_day": (round_trip_cost + borrow_daily) * 1e4,
        "ann_net_pct": net_mean * TRADING_DAYS * 100,
        "sharpe_net": sharpe,
        "t_net": one_sample_t(net),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: dict[str, pd.DataFrame], vol_window: int = 63,
                     beta_window: int = 252, frac: float = 0.3) -> dict:
    """Run the headline sentiment-beta stats on a synthetic panel."""
    ret = close_returns(panel)
    sp = beta_spreads(ret, vol_window, beta_window, frac)
    ts = beta_stats(sp)
    return {"spread_bps": ts["spread_bps"], "t_nw": ts["t_nw"],
            "welch_t": ts["welch_t"], "n_days": ts["n_days"]}
