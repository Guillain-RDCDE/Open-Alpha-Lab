"""Strategy + inference for Study 859 — Return-on-Invested-Capital Premium.

The claim: **ROIC (NOPAT / invested capital) is a cleaner, unlevered quality gauge than ROE, and
sorting on it earns a forward long-short spread** — high-ROIC firms out-return low-ROIC firms.

We test, all on the point-in-time signal (known at the 10-Q/10-K filing date, so zero
look-ahead):

1. **Forward returns — the calendar-time long-short (PRIMARY).** Each month-end, rank the names
   that carry a fresh ROIC signal into terciles; go long the top, short the bottom, equal-weight;
   earn the NEXT month's return (one execution lag). The decisive number is the **Newey-West
   (HAC) t** of that monthly long-short return series — the autocorrelation-robust statistic the
   desk's `REAL` bar is written against. A ROIC-**change** variant (`roic_chg`) is carried too.

2. **Forward returns — the pooled event drift (CROSS-CHECK).** Bucket all (ticker, filing) events
   by the signal, measure top-minus-bottom forward drift over a horizon, one-sample t + a
   label-shuffle placebo + the tercile monotonicity picture.

3. **Does ROIC add anything over ROE / gross profitability? (3rd axis).** Run the *same* calendar
   long-short on `roic`, on plain `roe` (Study 200) and on gross profitability `gp` (Study 122)
   over the identical panel, and report the cross-sectional rank correlation between ROIC and ROE.
   If ROIC's spread is no better than ROE's — or the two are ~collinear — ROIC "adds nothing".

Honesty rails: one documented execution lag (signal at month/filing t, return of t+1);
costs = one-way × NAV × turnover (both legs) + borrow on the short; survivorship named on the
Signal axis (current-survivors mega-cap basket). Given the thin, uneven panel, expect a modest
honest answer.
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


def calendar_ls(prices: pd.DataFrame, events: pd.DataFrame, signal_col: str = "roic",
                n_buckets: int = 3, min_names: int = 6, staleness_days: int = 200
                ) -> pd.DataFrame:
    """Monthly rebalanced long-short return series (one execution lag).

    At each month-end M: rank names carrying a fresh signal into ``n_buckets`` terciles; long the
    top, short the bottom (equal-weight). The position is held over month M+1 (return realised
    then) — one documented execution lag, signal known at M's close. Requires ≥ ``min_names`` in
    the cross-section. Returns a frame indexed by the RETURN month with columns: ls (long-short),
    long, short, n (cross-section size), turnover (one-way, fraction of book replaced).

    Vectorised: monthly returns and the per-ticker filing history are pre-materialised into numpy
    arrays; each month's freshest-non-stale signal is found with a ``searchsorted`` — no
    per-date DataFrame filtering.
    """
    mret = monthly_returns(prices)
    months = mret.index
    tickers = list(mret.columns)
    col = {t: i for i, t in enumerate(tickers)}
    R = mret.to_numpy(dtype=float)                       # (n_months, n_tickers)
    m_i8 = months.values.astype("datetime64[ns]").astype("int64")
    stale_ns = np.int64(staleness_days) * np.int64(86_400_000_000_000)

    # per-ticker sorted (filed, signal) history, restricted to names in the price panel
    hist: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    ev2 = events.dropna(subset=[signal_col])
    for tk, g in ev2.groupby("ticker"):
        ci = col.get(tk)
        if ci is None:
            continue
        g = g.sort_values("filed")
        fd = g["filed"].values.astype("datetime64[ns]").astype("int64")
        sv = g[signal_col].to_numpy(dtype=float)
        hist[ci] = (fd, sv)

    prev_long: set = set()
    prev_short: set = set()
    recs = []
    for i in range(len(months) - 1):
        m_ns = m_i8[i]
        cols_i, vals_i, ret_i = [], [], []
        for ci, (fd, sv) in hist.items():
            j = int(np.searchsorted(fd, m_ns, side="right")) - 1
            if j < 0 or (m_ns - fd[j]) > stale_ns:
                continue
            r1 = R[i + 1, ci]
            if np.isnan(r1):
                continue
            cols_i.append(ci); vals_i.append(sv[j]); ret_i.append(r1)
        if len(cols_i) < min_names:
            prev_long, prev_short = set(), set()
            continue
        cols_a = np.asarray(cols_i)
        vals_a = np.asarray(vals_i, dtype=float)
        ret_a = np.asarray(ret_i, dtype=float)
        b = _bucketize(vals_a, n_buckets)
        long_mask = b == n_buckets - 1
        short_mask = b == 0
        long_tk = set(cols_a[long_mask].tolist())
        short_tk = set(cols_a[short_mask].tolist())
        r_long = float(ret_a[long_mask].mean())
        r_short = float(ret_a[short_mask].mean())

        def _turn(new, old):
            if not old:
                return 1.0
            return len(new - old) / max(len(new), 1)

        turn = 0.5 * (_turn(long_tk, prev_long) + _turn(short_tk, prev_short))
        recs.append({"month": months[i + 1], "ls": r_long - r_short, "long": r_long,
                     "short": r_short, "n": len(cols_i), "turnover": turn})
        prev_long, prev_short = long_tk, short_tk
    return pd.DataFrame(recs, columns=["month", "ls", "long", "short", "n",
                                       "turnover"]).set_index("month")


def calendar_ls_stats(ls: pd.DataFrame, nw_lags: int = 6) -> dict:
    """Headline stats of a calendar long-short series: mean, one-sample t, NW t, ann, Sharpe."""
    if ls.empty:
        return {"n_months": 0, "mean_bps": float("nan"), "t_iid": float("nan"),
                "t_nw": float("nan"), "ann_pct": float("nan"), "sharpe": float("nan"),
                "hit": float("nan"), "avg_n": float("nan"), "avg_turnover": float("nan"),
                "long_bps": float("nan"), "short_bps": float("nan")}
    r = ls["ls"].to_numpy()
    n = len(r)
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

    Each month the book turns over ``turnover`` (one-way, fraction replaced); paying ``cost_bps``
    one-way on both the long and the short leg costs ``2 × cost_bps × turnover`` per month, and
    the short leg pays ``borrow_bps_ann`` annualised (÷12 monthly). Net series → NW t / Sharpe.
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
def event_drift_frame(prices: pd.DataFrame, events: pd.DataFrame, horizon: int,
                      lag: int = 1, signal_col: str = "roic") -> pd.DataFrame:
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
                   seed: int = 859) -> dict:
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
                  n_draws: int = 10_000, signal_col: str = "roic") -> dict:
    """Pooled event-drift headline for one horizon: LS mean/t/win + placebo p."""
    fr = event_drift_frame(prices, events, horizon, lag=lag, signal_col=signal_col)
    ls = long_short_drift(fr, n_buckets)
    t = one_sample_t(ls["ls_sample"])
    p = placebo_pvalue(fr, n_draws=n_draws, n_buckets=n_buckets)["p_value"] if placebo else float("nan")
    return {"horizon": horizon, "n_events": int(len(fr)), "n_top": ls["n_top"],
            "n_bot": ls["n_bot"], "top_mean": ls["top_mean"], "bot_mean": ls["bot_mean"],
            "ls_mean": ls["ls_mean"], "ls_win": ls["ls_win"], "t": t, "p_placebo": p}


# --------------------------------------------------------------------------- #
# 3rd axis — does ROIC add anything over ROE / gross profitability?
# --------------------------------------------------------------------------- #
def contrast(prices: pd.DataFrame, events: pd.DataFrame,
             staleness_days: int = 200) -> dict:
    """Head-to-head calendar long-short of ROIC vs plain ROE (200) vs gross profitability (122)
    on the identical panel, plus the pooled ROIC↔ROE rank correlation.

    If ROIC's Newey-West t is no better than ROE's and the two signals are ~collinear
    (high rank correlation), ROIC "adds nothing" over the cheaper, older signals.
    """
    out = {}
    for col in ("roic", "roic_chg", "roe", "gp"):
        ls = calendar_ls(prices, events, signal_col=col, n_buckets=3, min_names=6,
                         staleness_days=staleness_days)
        s = calendar_ls_stats(ls)
        out[col] = {"mean_bps": s["mean_bps"], "t_nw": s["t_nw"],
                    "sharpe": s["sharpe"], "n_months": s["n_months"]}
    # pooled ROIC vs ROE cross-sectional rank correlation (Spearman via rank Pearson)
    fr = events.dropna(subset=["roic", "roe"])
    if len(fr) >= 10:
        ra = pd.Series(fr["roic"].to_numpy()).rank().to_numpy()
        rb = pd.Series(fr["roe"].to_numpy()).rank().to_numpy()
        rho = float(np.corrcoef(ra, rb)[0, 1])
    else:
        rho = float("nan")
    out["roic_roe_rank_corr"] = rho
    return out


# --------------------------------------------------------------------------- #
# Synthetic-control detector (machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, events: pd.DataFrame,
                     signal_col: str = "roic") -> dict:
    """Run the primary calendar long-short + its NW t on a synthetic world."""
    ls = calendar_ls(prices, events, signal_col=signal_col, n_buckets=3, min_names=6,
                     staleness_days=120)
    st = calendar_ls_stats(ls)
    return {"n_months": st["n_months"], "mean_bps": st["mean_bps"], "t_nw": st["t_nw"]}
