"""Strategy + inference for Study 678 — Random-Walk-Index (Poulos RWI).

The claim (Michael Poulos, *Technical Analysis of Stocks & Commodities*, 1990): a market
that has moved farther, over the last *n* sessions, than a pure random walk of the same
day-to-day noise (its Average True Range) would be *expected* to move is not wandering —
it is trending, and "RWI > 1" is the mechanical trigger for "the trend is statistically
real, ride it."

    RWI_high(n, t) = (High_t - Low_{t-n}) / (ATR_n(t) * sqrt(n))
    RWI_low(n, t)  = (High_{t-n} - Low_t) / (ATR_n(t) * sqrt(n))

``ATR_n(t)`` is the *simple* n-bar average of the True Range (Poulos' original — no
Wilder smoothing). Poulos scans several short lookbacks (we use n = 2..6, the range in
the original write-up) and reports the single highest RWI-high reading across them as
"the" indicator; ``RWI_high > 1`` is the textbook "real uptrend" trigger.

Measurements:

* **Trend-day vs no-trend-day next-session return** — the honest test of "statistically
  non-random predicts returns": does the day *after* an RWI-high>1 flag actually pay
  more, on average, than the day after a non-flagged session? Welch *t* (single-day,
  effectively non-overlapping observations at daily granularity), a Newey-West *t* on
  the flag-dummy regression, a hit rate with a Wilson interval, and a matched-count
  random-day placebo (the honest question: would picking *any* random day-count-matched
  subset of sessions produce as extreme a mean by chance?).
* **The long timer, as an actual book** — position sized 0/1 off the lagged flag, costs
  charged one-way x NAV on every position change, benchmarked against buy & hold *and*
  against a **block-shuffled random-entry control**: the same flag series, chopped into
  contiguous blocks and randomly re-ordered across the calendar, which preserves the
  timer's total days-invested count *and* its turnover/run-length profile while
  destroying any real correlation between "flag fired" and what happens next. That is
  the fair, exposure- and cost-matched benchmark the claim has to beat.
* **Cross-instrument pooling** — the same split repeated on SPY, QQQ, IWM, DIA and GLD,
  pooled and per-ticker, so one lucky tape can't carry the verdict.
* **Synthetic positive control** — a two-regime (trend/chop) Markov world with a
  TUNABLE planted trend-persistence edge proves the machinery: it must not fire on the
  null (edge = 0, both regimes share the same expected drift) and must recover a
  planted edge when one exists.

Execution convention (one documented lag): the flag is computed from data through the
close of day *t* (it needs that day's own High/Low); the position is entered *at that
same close* (a closing-auction fill) and earns the close(t) -> close(t+1) return — a
single shift, applied once, exactly the METHODOLOGY convention "signal known at the
close of *t* earns the return of *t+1*."
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DEFAULT_PERIODS = (2, 3, 4, 5, 6)   # Poulos' own short-lookback scan


# --------------------------------------------------------------------------- #
# The indicator itself
# --------------------------------------------------------------------------- #
def true_range(df: pd.DataFrame) -> pd.Series:
    """Classic True Range: max(H-L, |H-Cprev|, |L-Cprev|)."""
    prev_close = df["Close"].shift(1)
    a = df["High"] - df["Low"]
    b = (df["High"] - prev_close).abs()
    c = (df["Low"] - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def rwi_high(df: pd.DataFrame, tr: pd.Series, n: int) -> pd.Series:
    """RWI-high(n): today's High vs the Low n sessions ago, scaled by ATR_n * sqrt(n)."""
    atr = tr.rolling(n).mean()
    return (df["High"] - df["Low"].shift(n)) / (atr * np.sqrt(n))


def rwi_low(df: pd.DataFrame, tr: pd.Series, n: int) -> pd.Series:
    """RWI-low(n): the High n sessions ago vs today's Low, scaled by ATR_n * sqrt(n)."""
    atr = tr.rolling(n).mean()
    return (df["High"].shift(n) - df["Low"]) / (atr * np.sqrt(n))


def day_frame(df: pd.DataFrame, periods: tuple[int, ...] = DEFAULT_PERIODS) -> pd.DataFrame:
    """One row per session: close, daily return, RWI-high/low (max over ``periods``),
    the raw flag (RWI-high > 1), the lagged position, and the one-day-forward return
    the flag is claimed to predict."""
    tr = true_range(df)
    rh = pd.concat([rwi_high(df, tr, n) for n in periods], axis=1).max(axis=1)
    rl = pd.concat([rwi_low(df, tr, n) for n in periods], axis=1).max(axis=1)
    ret = df["Close"].pct_change()
    out = pd.DataFrame({
        "close": df["Close"], "ret": ret, "rwi_high": rh, "rwi_low": rl,
    })
    out["flag"] = out["rwi_high"] > 1.0
    out["pos"] = out["flag"].shift(1).fillna(False).astype(bool)
    out["fwd_ret"] = out["ret"].shift(-1)
    return out.dropna(subset=["ret", "rwi_high"])


# --------------------------------------------------------------------------- #
# Inference primitives (Welch / Newey-West / Wilson — generic)
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
    keep = ~np.isnan(y) & ~np.isnan(d)
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
# The headline split — flag day vs no-flag day, next-session return
# --------------------------------------------------------------------------- #
def trend_day_stats(df: pd.DataFrame, nw_lags: int = 5) -> dict:
    """Flag day vs no-flag day next-session return: means, Welch t, NW t, hit rate."""
    d = df.dropna(subset=["fwd_ret"])
    f = d["flag"].values
    fwd = d["fwd_ret"].values
    a, b = fwd[f], fwd[~f]
    k_up = int((a > 0).sum())
    lo, hi = wilson_interval(k_up, len(a))
    return {
        "n_flag": int(f.sum()), "n_rest": int((~f).sum()),
        "flag_bps": float(np.nanmean(a) * 1e4), "rest_bps": float(np.nanmean(b) * 1e4),
        "gap_bps": float((np.nanmean(a) - np.nanmean(b)) * 1e4),
        "welch_t": welch_t(a, b),
        "nw_t": newey_west_t(fwd, f.astype(float), lags=nw_lags),
        "hit_up": k_up, "hit_rate": k_up / len(a) if len(a) else float("nan"),
        "hit_lo": lo, "hit_hi": hi,
    }


def placebo_pvalue(df: pd.DataFrame, n_draws_per_seed: int = 1_000,
                    n_seeds: int = 20, base_seed: int = 678) -> dict:
    """Matched-count random-day placebo: draw |flag| random sessions, mean fwd_ret.

    p = share of draws whose mean is >= the observed flag-day mean (a RIGHT-tail test —
    the claim predicts a *better* forward return). Averaged over ``n_seeds`` independent
    seeds x ``n_draws_per_seed`` draws so no single lucky stream decides it.
    """
    d = df.dropna(subset=["fwd_ret"])
    f = d["flag"].values
    fwd = d["fwd_ret"].values
    obs = float(np.nanmean(fwd[f]))
    pool = fwd[~np.isnan(fwd)]
    k = int(f.sum())
    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            means.append(pool[rng.choice(len(pool), size=k, replace=False)].mean())
    means = np.asarray(means)
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)),
            "p_value": float((means >= obs).mean()),
            "n_draws": len(means)}


# --------------------------------------------------------------------------- #
# The long timer as a book — costs, buy & hold, block-shuffled random control
# --------------------------------------------------------------------------- #
def _sharpe(ret: np.ndarray) -> float:
    ret = np.asarray(ret, dtype=float)
    sd = ret.std(ddof=1)
    return float(ret.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def backtest(df: pd.DataFrame, cost_bps: float = 5.0) -> dict:
    """The RWI-high long timer: pos 0/1 lagged flag, cost x turnover, vs buy & hold."""
    pos = df["pos"].astype(float).values
    ret = df["ret"].values
    turnover = np.abs(np.diff(pos, prepend=0.0))
    cost = turnover * (cost_bps / 1e4)
    net = pos * ret - cost
    bh = ret
    n_trades = int((turnover > 0).sum())
    return {
        "n_days": len(df), "n_invested": int(pos.sum()), "n_trades": n_trades,
        "exposure": float(pos.mean()),
        "total_return_pct": float((np.prod(1.0 + net) - 1.0) * 100),
        "bh_total_return_pct": float((np.prod(1.0 + bh) - 1.0) * 100),
        "sharpe": _sharpe(net), "bh_sharpe": _sharpe(bh),
        "ann_cost_pct": float(cost.sum() / (len(df) / TRADING_DAYS) * 100),
    }


def block_shuffle_positions(flag: np.ndarray, block_size: int, seed: int) -> np.ndarray:
    """Chop ``flag`` into contiguous blocks and randomly permute block order.

    Preserves the total True-count (exposure) and the run-length/turnover texture
    within each block, while destroying any real correlation between "flag fired
    here" and what actually happened next at this calendar date.
    """
    n = len(flag)
    n_blocks = int(np.ceil(n / block_size))
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_blocks)
    out = np.empty(n, dtype=flag.dtype)
    pos = 0
    for b in order:
        lo, hi = b * block_size, min((b + 1) * block_size, n)
        seg = flag[lo:hi]
        out[pos:pos + len(seg)] = seg
        pos += len(seg)
    return out


def random_control_backtest(df: pd.DataFrame, cost_bps: float = 5.0, block_size: int = 21,
                             n_seeds: int = 20, base_seed: int = 678) -> dict:
    """Block-shuffled flag applied to the same tape: the exposure/turnover-matched null."""
    flag = df["flag"].astype(float).values
    ret = df["ret"].values
    tot_rets, sharpes = [], []
    for s in range(n_seeds):
        shuf = block_shuffle_positions(flag, block_size, base_seed + s)
        pos = np.roll(shuf, 1)
        pos[0] = 0.0
        turnover = np.abs(np.diff(pos, prepend=0.0))
        net = pos * ret - turnover * (cost_bps / 1e4)
        tot_rets.append(float(np.prod(1.0 + net) - 1.0) * 100)
        sharpes.append(_sharpe(net))
    tot_rets = np.asarray(tot_rets)
    sharpes = np.asarray(sharpes)
    return {"n_seeds": n_seeds, "mean_total_return_pct": float(tot_rets.mean()),
            "sd_total_return_pct": float(tot_rets.std(ddof=1)),
            "mean_sharpe": float(np.nanmean(sharpes)), "sd_sharpe": float(np.nanstd(sharpes, ddof=1)),
            "draws_total_return_pct": tot_rets, "draws_sharpe": sharpes}


# --------------------------------------------------------------------------- #
# Cross-instrument pooling
# --------------------------------------------------------------------------- #
def cross_instrument_stats(basket: dict[str, pd.DataFrame],
                            periods: tuple[int, ...] = DEFAULT_PERIODS) -> dict:
    """Per-ticker trend_day_stats plus a pooled Welch t across all flagged/unflagged days."""
    per_ticker = {}
    flag_pool, rest_pool = [], []
    for t, raw in basket.items():
        d = day_frame(raw, periods).dropna(subset=["fwd_ret"])
        per_ticker[t] = trend_day_stats(d)
        f = d["flag"].values
        fwd = d["fwd_ret"].values
        flag_pool.append(fwd[f])
        rest_pool.append(fwd[~f])
    flag_pool = np.concatenate(flag_pool)
    rest_pool = np.concatenate(rest_pool)
    pooled = {
        "n_flag": len(flag_pool), "n_rest": len(rest_pool),
        "flag_bps": float(flag_pool.mean() * 1e4), "rest_bps": float(rest_pool.mean() * 1e4),
        "welch_t": welch_t(flag_pool, rest_pool),
    }
    return {"per_ticker": per_ticker, "pooled": pooled}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(ohlc: pd.DataFrame, periods: tuple[int, ...] = DEFAULT_PERIODS) -> dict:
    """Run the headline flag-day-vs-rest split on a synthetic OHLC world."""
    df = day_frame(ohlc, periods)
    return trend_day_stats(df)
