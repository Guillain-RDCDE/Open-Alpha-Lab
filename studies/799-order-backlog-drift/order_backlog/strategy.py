"""Strategy + inference for Study 799 — Order-Backlog Drift (RPO).

The claim: **growth in a firm's order backlog (Remaining Performance Obligations, RPO)
leads its future sales and forward returns, and the market underreacts to it.** RPO is the
ASC-606 dollar value of contracted-but-unrecognised revenue; a swelling backlog is
supposed to front-run both the income statement and the stock.

We test three things, each on the point-in-time signal (known at the 10-Q/10-K filing
date, so zero look-ahead):

1. **Forward returns — the calendar-time long-short (PRIMARY).** Each month-end, rank the
   names carrying a fresh RPO-YoY-growth signal into terciles; go long the top, short the
   bottom, equal-weight; earn the NEXT month's return (one execution lag). The decisive
   number is the **Newey-West (HAC) t** of that monthly long-short return series — the
   autocorrelation-robust statistic the desk's ``REAL`` bar is written against.

2. **Forward returns — the pooled event drift (CROSS-CHECK).** The 534-style pooled sort:
   bucket all (ticker, filing) events by the signal, measure the top-minus-bottom forward
   drift over a horizon, one-sample t + a label-shuffle placebo.

3. **Leads future sales? (3rd axis).** Does this quarter's RPO growth predict NEXT
   quarter's YoY revenue growth? A pooled OLS slope with a t-stat, plus the tercile spread
   in future sales growth. This is the mechanism check: even if the stock doesn't move,
   does the accounting lead actually exist?

Honesty rails: one documented execution lag (signal at month/filing t, return of t+1);
costs = one-way × NAV × turnover, shorts pay borrow; survivorship named on the Signal axis
(current-survivors basket). Given the thin, one-regime (post-2018) panel, expect the
honest answer to be modest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (21, 63, 126)             # trading-day forward-return horizons (≈1m, 1q, 2q)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    """One-sample t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variance). NaN if either < 2 obs."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 6) -> float:
    """HAC (Newey-West, Bartlett kernel) t of the mean of a return series vs 0.

    Regressing ``x`` on a constant, the HAC standard error of the mean uses the Bartlett-
    weighted autocovariances out to ``lags`` — the serial-correlation-robust statistic the
    desk's REAL bar is measured against (overlapping/persistent portfolio returns violate iid).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    u = x - x.mean()
    gamma0 = float(u @ u) / n
    s = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        cov = float(u[l:] @ u[:-l]) / n
        s += 2.0 * w * cov
    if s <= 0:
        return float("nan")
    se = np.sqrt(s / n)
    return float(x.mean() / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def _bucketize(x: np.ndarray, n_buckets: int) -> np.ndarray:
    """Rank ``x`` into ``n_buckets`` equal-frequency bins (0 = lowest)."""
    order = np.argsort(np.argsort(x))
    return np.minimum((order * n_buckets // len(x)), n_buckets - 1)


# --------------------------------------------------------------------------- #
# Calendar-time long-short (PRIMARY)
# --------------------------------------------------------------------------- #
def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end adjusted-close → simple monthly returns (index = month-end)."""
    me = prices.resample("ME").last()
    return me.pct_change()


def _signal_asof(events: pd.DataFrame, tickers, month_end: pd.Timestamp,
                 signal_col: str, staleness_days: int) -> dict:
    """The freshest signal per ticker known at ``month_end`` (filed ≤ month_end, not stale)."""
    out = {}
    cut = month_end
    for tk in tickers:
        g = events[(events["ticker"] == tk) & (events["filed"] <= cut)]
        if g.empty:
            continue
        r = g.iloc[-1]
        if (cut - r["filed"]).days > staleness_days:
            continue
        v = r[signal_col]
        if pd.notna(v):
            out[tk] = float(v)
    return out


def calendar_ls(prices: pd.DataFrame, events: pd.DataFrame, signal_col: str = "rpo_yoy",
                n_buckets: int = 3, min_names: int = 6, staleness_days: int = 200
                ) -> pd.DataFrame:
    """Monthly rebalanced long-short return series (one execution lag).

    At each month-end M: rank names carrying a fresh signal into ``n_buckets`` terciles; long
    the top, short the bottom (equal-weight). The position is held over month M+1 (return
    realised then) — one documented execution lag, signal known at M's close. Requires ≥
    ``min_names`` in the cross-section. Returns a frame indexed by the RETURN month with
    columns: ls (long-short), long, short, n (cross-section size), turnover (one-way, fraction
    of book replaced).
    """
    mret = monthly_returns(prices)
    months = mret.index
    prev_long: set = set()
    prev_short: set = set()
    recs = []
    for i in range(len(months) - 1):
        m, m1 = months[i], months[i + 1]
        sig = _signal_asof(events, prices.columns, m, signal_col, staleness_days)
        sig = {tk: v for tk, v in sig.items() if pd.notna(mret.loc[m1, tk])}
        if len(sig) < min_names:
            prev_long, prev_short = set(), set()
            continue
        tks = np.array(list(sig.keys()))
        vals = np.array([sig[t] for t in tks])
        b = _bucketize(vals, n_buckets)
        long_tk = set(tks[b == n_buckets - 1])
        short_tk = set(tks[b == 0])
        r_long = float(np.mean([mret.loc[m1, t] for t in long_tk]))
        r_short = float(np.mean([mret.loc[m1, t] for t in short_tk]))

        def _turn(new, old):
            if not old:
                return 1.0
            return len(new - old) / max(len(new), 1)
        turn = 0.5 * (_turn(long_tk, prev_long) + _turn(short_tk, prev_short))
        recs.append({"month": m1, "ls": r_long - r_short, "long": r_long,
                     "short": r_short, "n": len(sig), "turnover": turn})
        prev_long, prev_short = long_tk, short_tk
    return pd.DataFrame(recs).set_index("month")


def calendar_ls_stats(ls: pd.DataFrame, nw_lags: int = 6) -> dict:
    """Headline stats of a calendar long-short series: mean, one-sample t, NW t, ann, Sharpe."""
    r = ls["ls"].to_numpy()
    n = len(r)
    if n == 0:
        return {"n_months": 0, "mean_bps": float("nan"), "t_iid": float("nan"),
                "t_nw": float("nan"), "ann_pct": float("nan"), "sharpe": float("nan"),
                "hit": float("nan"), "avg_n": float("nan"), "avg_turnover": float("nan"),
                "long_bps": float("nan"), "short_bps": float("nan")}
    mean = float(np.mean(r))
    sd = float(np.std(r, ddof=1)) if n > 1 else float("nan")
    sharpe = mean / sd * np.sqrt(12) if sd and sd > 0 else float("nan")
    return {
        "n_months": n, "mean_bps": mean * 1e4, "t_iid": one_sample_t(r),
        "t_nw": newey_west_t(r, lags=nw_lags), "ann_pct": mean * 12 * 100,
        "sharpe": sharpe, "hit": float((r > 0).mean()),
        "avg_n": float(ls["n"].mean()), "avg_turnover": float(ls["turnover"].mean()),
        "long_bps": float(ls["long"].mean() * 1e4), "short_bps": float(ls["short"].mean() * 1e4),
    }


def calendar_ls_net(ls: pd.DataFrame, cost_bps: float = 20.0, borrow_bps_ann: float = 100.0,
                    nw_lags: int = 6) -> dict:
    """Long-short net of one-way costs × turnover (both legs) + borrow on the short leg.

    Each month the book turns over ``turnover`` (one-way, fraction replaced); paying
    ``cost_bps`` one-way on both the long and the short leg costs ``2 × cost_bps × turnover``
    per month, and the short leg pays ``borrow_bps_ann`` annualised (÷12 monthly). Net series
    → NW t / Sharpe.
    """
    r = ls["ls"].to_numpy()
    if len(r) == 0:
        return {"cost_bps": cost_bps, "borrow_bps_ann": borrow_bps_ann,
                "net_mean_bps": float("nan"), "net_ann_pct": float("nan"),
                "net_t_nw": float("nan"), "net_sharpe": float("nan")}
    c = cost_bps / 1e4
    borrow_m = (borrow_bps_ann / 1e4) / 12.0
    net = r - 2.0 * c * ls["turnover"].to_numpy() - borrow_m
    mean = float(np.mean(net)); sd = float(np.std(net, ddof=1)) if len(net) > 1 else float("nan")
    return {"cost_bps": cost_bps, "borrow_bps_ann": borrow_bps_ann,
            "net_mean_bps": mean * 1e4, "net_ann_pct": mean * 12 * 100,
            "net_t_nw": newey_west_t(net, lags=nw_lags),
            "net_sharpe": mean / sd * np.sqrt(12) if sd and sd > 0 else float("nan")}


# --------------------------------------------------------------------------- #
# Pooled event-drift sort (CROSS-CHECK, 534-style)
# --------------------------------------------------------------------------- #
def event_drift_frame(prices: pd.DataFrame, events: pd.DataFrame,
                      horizon: int, lag: int = 1, signal_col: str = "rpo_yoy") -> pd.DataFrame:
    """Attach a forward ``drift`` (return over ``horizon`` days, entered ``lag`` days after the
    first session on/after the filing) to each event. Rows whose window overruns are dropped."""
    out = []
    for _, r in events.iterrows():
        tk = r["ticker"]
        if tk not in prices.columns:
            continue
        px = prices[tk].dropna()
        idx = px.index
        f = pd.Timestamp(r["filed"]).normalize()
        pos = idx.searchsorted(f, side="left")
        entry = pos + lag
        exit_ = entry + horizon
        if pos <= 0 or exit_ >= len(idx):
            continue
        sig = r[signal_col]
        if pd.isna(sig):
            continue
        out.append({"ticker": tk, "filed": idx[pos], "signal": float(sig),
                    "drift": float(px.iloc[exit_] / px.iloc[entry] - 1.0)})
    return pd.DataFrame(out).reset_index(drop=True)


def long_short_drift(frame: pd.DataFrame, n_buckets: int = 3) -> dict:
    """Top-minus-bottom signal-bucket drift on a frame carrying ``signal`` and ``drift``."""
    s = frame["signal"].to_numpy(); d = frame["drift"].to_numpy()
    b = _bucketize(s, n_buckets)
    top, bot = d[b == n_buckets - 1], d[b == 0]
    ls_sample = np.concatenate([top, -bot])
    return {"n_top": int(len(top)), "n_bot": int(len(bot)),
            "top_mean": float(top.mean()), "bot_mean": float(bot.mean()),
            "ls_mean": float(top.mean() - bot.mean()), "ls_sample": ls_sample,
            "ls_win": float((ls_sample > 0).mean())}


def bucket_means(frame: pd.DataFrame, n_buckets: int = 3) -> np.ndarray:
    """Mean forward drift in each signal bucket (low → high) — the monotonicity picture."""
    s = frame["signal"].to_numpy(); d = frame["drift"].to_numpy()
    b = _bucketize(s, n_buckets)
    return np.array([d[b == k].mean() if (b == k).any() else np.nan for k in range(n_buckets)])


def placebo_pvalue(frame: pd.DataFrame, n_draws: int = 10_000, n_buckets: int = 3,
                   seed: int = 799) -> dict:
    """Label-shuffle placebo: permute signals, re-form random terciles, recompute the LS drift.
    ``p`` = P[shuffled long-short ≥ observed]."""
    s = frame["signal"].to_numpy(); d = frame["drift"].to_numpy()
    obs = float(long_short_drift(frame, n_buckets)["ls_mean"])
    rng = np.random.default_rng(seed)
    cut = n_buckets - 1
    means = np.empty(n_draws)
    for i in range(n_draws):
        b = _bucketize(rng.permutation(s), n_buckets)
        means[i] = d[b == cut].mean() - d[b == 0].mean()
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "p_value": float((means >= obs).mean()), "draws": means}


def event_summary(prices: pd.DataFrame, events: pd.DataFrame, horizon: int,
                  n_buckets: int = 3, lag: int = 1, placebo: bool = True,
                  n_draws: int = 10_000, signal_col: str = "rpo_yoy") -> dict:
    """Pooled event-drift headline for one horizon: LS mean/t/win + placebo p."""
    fr = event_drift_frame(prices, events, horizon, lag=lag, signal_col=signal_col)
    ls = long_short_drift(fr, n_buckets)
    t = one_sample_t(ls["ls_sample"])
    p = placebo_pvalue(fr, n_draws=n_draws, n_buckets=n_buckets)["p_value"] if placebo else float("nan")
    return {"horizon": horizon, "n_events": int(len(fr)), "n_top": ls["n_top"],
            "n_bot": ls["n_bot"], "top_mean": ls["top_mean"], "bot_mean": ls["bot_mean"],
            "ls_mean": ls["ls_mean"], "ls_win": ls["ls_win"], "t": t, "p_placebo": p}


# --------------------------------------------------------------------------- #
# 3rd axis — does RPO growth lead future sales?
# --------------------------------------------------------------------------- #
def leads_sales(events: pd.DataFrame) -> dict:
    """Does this quarter's RPO YoY growth predict NEXT quarter's revenue YoY growth?

    Pooled OLS: next_rev_yoy = a + b · rpo_yoy. Reports the slope b, its (iid) t, the R², the
    Pearson correlation, and the future-sales-growth spread between the top and bottom
    RPO-growth terciles. A positive, significant b is the accounting lead the whole thesis
    rests on. (t is iid-pooled — events cluster by quarter, so read it as suggestive, not a
    calendar-robust HAC statistic.)
    """
    fr = events.dropna(subset=["rpo_yoy", "next_rev_yoy"]).copy()
    x = fr["rpo_yoy"].to_numpy(); y = fr["next_rev_yoy"].to_numpy()
    n = len(x)
    if n < 10:
        return {"n": n, "slope": float("nan"), "t": float("nan"), "r2": float("nan"),
                "corr": float("nan"), "top_sales": float("nan"), "bot_sales": float("nan"),
                "spread": float("nan")}
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = float(resid @ resid) / (n - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    t = float(beta[1] / np.sqrt(cov[1, 1])) if cov[1, 1] > 0 else float("nan")
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    corr = float(np.corrcoef(x, y)[0, 1])
    b = _bucketize(x, 3)
    top_sales = float(y[b == 2].mean()); bot_sales = float(y[b == 0].mean())
    return {"n": n, "slope": float(beta[1]), "t": t, "r2": r2, "corr": corr,
            "top_sales": top_sales, "bot_sales": bot_sales, "spread": top_sales - bot_sales}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, events: pd.DataFrame) -> dict:
    """Run the primary calendar long-short + its NW t on a synthetic world."""
    ls = calendar_ls(prices, events, signal_col="rpo_yoy", n_buckets=3, min_names=6,
                     staleness_days=120)
    stt = calendar_ls_stats(ls)
    return {"n_months": stt["n_months"], "mean_bps": stt["mean_bps"], "t_nw": stt["t_nw"]}
