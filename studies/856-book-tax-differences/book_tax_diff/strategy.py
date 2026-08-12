"""Strategy + inference for Study 856 — Book-Tax Differences.

The claim (Hanlon 2005): a **large positive book-tax difference** (book pretax income far above
the taxable income implied by the tax expense, grossed up through the statutory rate) marks
**less persistent earnings** and — if the market is slow to price it — **lower future returns**.
The tradeable read-through is a cross-sectional long-short: **long the low-BTD ("clean") names,
short the high-BTD ("aggressive") names**.

To reuse a single "long the top tercile of the signal, short the bottom" engine, the ranking
signal is ``btd_neg = − BTD/Assets``: the *top* tercile of ``btd_neg`` is the *lowest* book-tax
gap (the names Hanlon says you should own), the *bottom* tercile is the *highest* gap (the ones
to short). So a **positive** long-short return means low-BTD beats high-BTD — the Hanlon
direction. A significant **negative** long-short would be a *wrong-sign* result versus the claim.

We test three things, each on the point-in-time signal (known at the 10-K filing date, so zero
look-ahead):

1. **Forward returns — the calendar-time long-short (PRIMARY).** Each month-end, rank the names
   carrying a fresh (≤ ~14-month-old) BTD signal into terciles; long the top (low BTD), short the
   bottom (high BTD), equal-weight; earn the NEXT month's return (one execution lag). The decisive
   number is the **Newey-West (HAC) t** of that monthly long-short return series.

2. **Forward returns — the pooled event drift (CROSS-CHECK).** Bucket all (ticker, filing) events
   by ``btd_neg``, measure top-minus-bottom forward drift over a horizon, one-sample t + a
   label-shuffle placebo, plus the tercile monotonicity picture.

3. **Earnings persistence (the MECHANISM / 3rd axis).** Hanlon's actual result: high-BTD firms
   have *less persistent* earnings. We regress next-year ROA on this-year ROA and let the slope
   (the persistence coefficient) differ between the high-BTD and low-BTD terciles via an
   interaction. Hanlon predicts the high-BTD persistence slope is **lower**.

Honesty rails: one documented execution lag (signal at month/filing t, return of t+1); costs =
one-way × NAV × turnover, shorts pay borrow; survivorship named on the Signal axis (current-
survivors basket). Given a large-cap survivor panel, expect the honest return answer to be modest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (63, 126, 252)            # trading-day forward-return horizons (≈1q, 2q, 1y)


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

    Regressing ``x`` on a constant, the HAC standard error of the mean uses the Bartlett-weighted
    autocovariances out to ``lags`` — the serial-correlation-robust statistic the desk's REAL bar
    is measured against (overlapping/persistent portfolio returns violate iid).
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


def calendar_ls(prices: pd.DataFrame, events: pd.DataFrame, signal_col: str = "btd_neg",
                n_buckets: int = 3, min_names: int = 6, staleness_days: int = 430
                ) -> pd.DataFrame:
    """Monthly rebalanced long-short return series (one execution lag).

    At each month-end M: rank names carrying a fresh signal into ``n_buckets`` terciles; long the
    top, short the bottom (equal-weight). With the default ``signal_col='btd_neg'`` the top tercile
    is the LOWEST book-tax gap and the bottom is the HIGHEST — long clean, short aggressive (the
    Hanlon portfolio). The position is held over month M+1 (return realised then) — one documented
    execution lag, signal known at M's close. The BTD signal is annual, so ``staleness_days`` is
    ~430 (held until the next 10-K). Requires ≥ ``min_names`` in the cross-section. Returns a frame
    indexed by the RETURN month: ls (long-short), long, short, n, turnover (one-way).
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
    mean = float(np.mean(r)) if n else float("nan")
    sd = float(np.std(r, ddof=1)) if n > 1 else float("nan")
    sharpe = mean / sd * np.sqrt(12) if sd and sd > 0 else float("nan")
    return {
        "n_months": n, "mean_bps": mean * 1e4, "t_iid": one_sample_t(r),
        "t_nw": newey_west_t(r, lags=nw_lags), "ann_pct": mean * 12 * 100,
        "sharpe": sharpe, "hit": float((r > 0).mean()) if n else float("nan"),
        "avg_n": float(ls["n"].mean()) if n else float("nan"),
        "avg_turnover": float(ls["turnover"].mean()) if n else float("nan"),
        "long_bps": float(ls["long"].mean() * 1e4) if n else float("nan"),
        "short_bps": float(ls["short"].mean() * 1e4) if n else float("nan"),
    }


def calendar_ls_net(ls: pd.DataFrame, cost_bps: float = 20.0, borrow_bps_ann: float = 100.0,
                    nw_lags: int = 6) -> dict:
    """Long-short net of one-way costs × turnover (both legs) + borrow on the short leg.

    Each month the book turns over ``turnover`` (one-way, fraction replaced); paying ``cost_bps``
    one-way on both the long and the short leg costs ``2 × cost_bps × turnover`` per month, and the
    short leg pays ``borrow_bps_ann`` annualised (÷12 monthly). Net series → NW t / Sharpe.
    """
    c = cost_bps / 1e4
    borrow_m = (borrow_bps_ann / 1e4) / 12.0
    net = ls["ls"].to_numpy() - 2.0 * c * ls["turnover"].to_numpy() - borrow_m
    mean = float(np.mean(net)); sd = float(np.std(net, ddof=1))
    return {"cost_bps": cost_bps, "borrow_bps_ann": borrow_bps_ann,
            "net_mean_bps": mean * 1e4, "net_ann_pct": mean * 12 * 100,
            "net_t_nw": newey_west_t(net, lags=nw_lags),
            "net_sharpe": mean / sd * np.sqrt(12) if sd > 0 else float("nan")}


# --------------------------------------------------------------------------- #
# Pooled event-drift sort (CROSS-CHECK)
# --------------------------------------------------------------------------- #
def event_drift_frame(prices: pd.DataFrame, events: pd.DataFrame,
                      horizon: int, lag: int = 1, signal_col: str = "btd_neg") -> pd.DataFrame:
    """Attach a forward ``drift`` (return over ``horizon`` days, entered ``lag`` days after the
    first session on/after the filing) to each event. Rows whose window overruns are dropped."""
    out = []
    for _, r in events.iterrows():
        tk = r["ticker"]
        if tk not in prices.columns or pd.isna(r[signal_col]):
            continue
        px = prices[tk].dropna()
        idx = px.index
        f = pd.Timestamp(r["filed"]).normalize()
        pos = idx.searchsorted(f, side="left")
        entry = pos + lag
        exit_ = entry + horizon
        if pos <= 0 or exit_ >= len(idx):
            continue
        out.append({"ticker": tk, "filed": idx[pos], "signal": float(r[signal_col]),
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
                   seed: int = 856) -> dict:
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
                  n_draws: int = 10_000, signal_col: str = "btd_neg") -> dict:
    """Pooled event-drift headline for one horizon: LS mean/t/win + placebo p."""
    fr = event_drift_frame(prices, events, horizon, lag=lag, signal_col=signal_col)
    ls = long_short_drift(fr, n_buckets)
    t = one_sample_t(ls["ls_sample"])
    p = placebo_pvalue(fr, n_draws=n_draws, n_buckets=n_buckets)["p_value"] if placebo else float("nan")
    return {"horizon": horizon, "n_events": int(len(fr)), "n_top": ls["n_top"],
            "n_bot": ls["n_bot"], "top_mean": ls["top_mean"], "bot_mean": ls["bot_mean"],
            "ls_mean": ls["ls_mean"], "ls_win": ls["ls_win"], "t": t, "p_placebo": p}


# --------------------------------------------------------------------------- #
# 3rd axis — do large book-tax differences mark LESS PERSISTENT earnings?
# --------------------------------------------------------------------------- #
def earnings_persistence(events: pd.DataFrame) -> dict:
    """Hanlon's mechanism: high-BTD firms have *less persistent* earnings.

    Pool matched (roa, roa_next) pairs and fit an interaction OLS::

        roa_next = a + b·roa + c·HI + d·(roa × HI)

    where HI = 1 for the top (highest positive) book-tax-difference tercile, 0 for the bottom.
    ``b`` is the persistence slope of the LOW-BTD tercile; ``b + d`` the HIGH-BTD tercile. Hanlon
    predicts ``d < 0`` — earnings persist *less* when the book-tax gap is large. Reports the two
    slopes, their difference ``d`` with an (iid-pooled) t, and R². (The pooled t ignores firm/year
    clustering, so read it as suggestive, not a calendar-robust HAC statistic.)
    """
    fr = events.dropna(subset=["roa", "roa_next", "btd_assets"]).copy()
    # keep economically sane firm-years (drop extreme leverage/ROA outliers)
    fr = fr[(fr["roa"].abs() < 1.0) & (fr["roa_next"].abs() < 1.0)]
    n = len(fr)
    if n < 20:
        return {"n": n, "b_low": float("nan"), "b_high": float("nan"), "diff": float("nan"),
                "t_diff": float("nan"), "r2": float("nan"), "b_all": float("nan")}
    btd = fr["btd_assets"].to_numpy()
    roa = fr["roa"].to_numpy()
    roa_n = fr["roa_next"].to_numpy()
    b3 = _bucketize(btd, 3)
    keep = (b3 == 0) | (b3 == 2)                     # bottom (low BTD) vs top (high BTD) terciles
    hi = (b3[keep] == 2).astype(float)
    x = roa[keep]; y = roa_n[keep]
    m = len(x)
    X = np.column_stack([np.ones(m), x, hi, x * hi])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = m - X.shape[1]
    s2 = float(resid @ resid) / dof if dof > 0 else float("nan")
    cov = s2 * np.linalg.inv(X.T @ X)
    t_diff = float(beta[3] / np.sqrt(cov[3, 3])) if cov[3, 3] > 0 else float("nan")
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    # pooled persistence across everything (for reference)
    Xa = np.column_stack([np.ones(n), roa])
    ba, *_ = np.linalg.lstsq(Xa, roa_n, rcond=None)
    return {"n": n, "n_pair": m, "b_low": float(beta[1]), "b_high": float(beta[1] + beta[3]),
            "diff": float(beta[3]), "t_diff": t_diff, "r2": r2, "b_all": float(ba[1])}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, events: pd.DataFrame) -> dict:
    """Run the primary calendar long-short + its NW t on a synthetic world."""
    ls = calendar_ls(prices, events, signal_col="btd_neg", n_buckets=3, min_names=6,
                     staleness_days=430)
    st = calendar_ls_stats(ls)
    return {"n_months": st["n_months"], "mean_bps": st["mean_bps"], "t_nw": st["t_nw"]}
