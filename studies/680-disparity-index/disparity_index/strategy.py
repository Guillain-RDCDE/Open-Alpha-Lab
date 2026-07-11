"""Strategy + inference for Study 680 — Disparity Index.

The claim: **DI(N) = 100 x Close / SMA(N)** — the percent deviation of the close from its
own trailing N-day moving average, popular in Korean and Japanese technical analysis — is
a mean-reversion gauge. Above 105 (price 5%+ above its N-day average), "the rubber band is
stretched" -> sell/short. Below 95 (price 5%+ below), "it's stretched the other way" ->
buy. N = 10 and the 95/105 thresholds are the textbook short-horizon defaults, tested
exactly as stated.

One documented execution lag throughout: DI is known from closes up to and including bar
*t*; every trade (or forward-return window) enters at bar *t+1*'s **open** and, where a
holding period applies, exits at the close of bar *t + h*. No look-ahead.

Three complementary measurements share the same DI indicator:

* **Conditional-forward-return split** — every day is bucketed by its DI reading
  (oversold / neutral / overbought); the forward *h*-day return of each bucket is compared
  to the neutral bucket with Welch *t* (i.i.d. across non-overlapping events isn't true
  here — the *h*-day windows overlap — so a Newey-West (HAC, *h* lags) cross-check is
  reported alongside). This is the direct test of "forward returns conditioned on DI
  extremes vs unconditional."
* **Zone-trigger trade ledger** — a signal fires only the day DI first *enters* a zone
  (the trigger, not every day inside it, so trades don't pile up on top of each other);
  entered at the next open, held a fixed window, one-way costs x 2 per round trip. Pinned
  against a **random-direction control** on the identical entry bars (the "beats a coin?"
  myth-check).
* **The "just short-term reversal" diagnostic** — DI(N) is, by construction, a smoothed,
  monotonic function of the trailing N-day return (a rubber-band distance from a moving
  average necessarily *is* an accumulated-return measure). We report the pooled Pearson
  correlation between DI and the trailing N-day return directly: if it is very high, the
  "Disparity Index" folklore is not a distinct discovery, it is
  [329-one-month-reversal](../../329-one-month-reversal/) wearing a different name — and
  that sibling already found the effect is largely bid-ask-bounce microstructure, dead
  since 2002.

Both the conditional split and the trade ledger run pooled across the SPY + basket
universe and are swept over a window x threshold grid for robustness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# The indicator
# --------------------------------------------------------------------------- #
def disparity_index(close: pd.Series, window: int = 10) -> pd.Series:
    """DI(window): 100 x close / trailing-``window`` simple moving average.

    100 = sitting exactly on its own average; >100 = above it (rubber band stretched up);
    <100 = below it. Known at the close of bar *t* (the SMA includes bar *t*'s own close)
    — available to trade at bar *t+1*'s open.
    """
    sma = close.rolling(window, min_periods=window).mean()
    di = 100.0 * close / sma
    return di.rename("di")


def trailing_return(close: pd.Series, window: int = 10) -> pd.Series:
    """Trailing ``window``-day simple return as of bar *t* — the "just reversal?" control."""
    return (close / close.shift(window) - 1.0).rename("ret_trailing")


# --------------------------------------------------------------------------- #
# Inference primitives (shared shape with the desk's other technical studies)
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

    b is exactly the treated-minus-rest mean difference. Overlapping *h*-day forward
    windows induce up-to-(h-1)-lag autocorrelation, so ``lags`` should be >= h-1 there.
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
# Conditional forward returns — every day bucketed by its DI reading
# --------------------------------------------------------------------------- #
def forward_returns(bars: pd.DataFrame, h: int = 5) -> pd.Series:
    """Forward return: enter next bar's open, exit the close of bar t+h (or tape end)."""
    open_ = bars["Open"].to_numpy(dtype=float)
    close_ = bars["Close"].to_numpy(dtype=float)
    n = len(bars)
    out = np.full(n, np.nan)
    for t in range(n - 1):
        e = t + 1
        exit_idx = min(e + h - 1, n - 1)
        out[t] = (close_[exit_idx] - open_[e]) / open_[e]
    return pd.Series(out, index=bars.index, name="fwd_ret")


def pooled_conditional(bars_map: dict[str, pd.DataFrame], window: int = 10,
                       oversold: float = 95.0, overbought: float = 105.0,
                       h: int = 5) -> dict:
    """Pool forward returns across the whole universe, bucketed by the DI reading.

    ``low``/``high`` are **trigger days only** — the day DI first *enters* the oversold
    (< ``oversold``) / overbought (> ``overbought``) zone, not every day inside it. DI is a
    trailing rolling window, so consecutive in-zone days share nearly the same underlying
    ``window`` closes and are barely independent draws; scoring only the entry day keeps
    the events close to non-overlapping, which is what a Welch *t* assumes. ``mid`` = every
    other (non-trigger) day, the unconditional control. Welch *t* (planned primary) plus a
    Newey-West *t* with ``h`` lags (the residual overlap-robust cross-check, since the
    *h*-day forward windows themselves still overlap).
    """
    lows, highs, mids = [], [], []
    for _, bars in bars_map.items():
        di = disparity_index(bars["Close"], window)
        fwd = forward_returns(bars, h)
        trig = zone_signals(di, oversold, overbought)
        df = pd.concat([di, fwd], axis=1)
        low_mask = df.index.isin(trig.index[trig["dir"] == 1])
        high_mask = df.index.isin(trig.index[trig["dir"] == -1])
        mid_mask = ~low_mask & ~high_mask & df["fwd_ret"].notna()
        lows.append(df.loc[low_mask, "fwd_ret"].dropna().to_numpy())
        highs.append(df.loc[high_mask, "fwd_ret"].dropna().to_numpy())
        mids.append(df.loc[mid_mask, "fwd_ret"].to_numpy())
    low = np.concatenate(lows) if lows else np.array([])
    high = np.concatenate(highs) if highs else np.array([])
    mid = np.concatenate(mids) if mids else np.array([])
    nw_low = newey_west_t(np.concatenate([low, mid]),
                          np.concatenate([np.ones(len(low)), np.zeros(len(mid))]),
                          lags=h) if len(low) and len(mid) else float("nan")
    nw_high = newey_west_t(np.concatenate([high, mid]),
                           np.concatenate([np.ones(len(high)), np.zeros(len(mid))]),
                           lags=h) if len(high) and len(mid) else float("nan")
    return {
        "n_low": len(low), "n_high": len(high), "n_mid": len(mid),
        "mean_low_bps": float(np.nanmean(low)) * 1e4 if len(low) else float("nan"),
        "mean_high_bps": float(np.nanmean(high)) * 1e4 if len(high) else float("nan"),
        "mean_mid_bps": float(np.nanmean(mid)) * 1e4 if len(mid) else float("nan"),
        "welch_t_low": welch_t(low, mid),
        "welch_t_high": welch_t(high, mid),
        "nw_t_low": nw_low,
        "nw_t_high": nw_high,
    }


# --------------------------------------------------------------------------- #
# Zone-trigger trade ledger — enter only the day DI first crosses into a zone
# --------------------------------------------------------------------------- #
def zone_signals(di: pd.Series, oversold: float = 95.0, overbought: float = 105.0
                 ) -> pd.DataFrame:
    """Trigger days: the first day DI crosses *into* a zone (not every day inside it).

    Contrarian direction: entering oversold (DI < oversold) -> buy (``dir`` = +1);
    entering overbought (DI > overbought) -> sell/short (``dir`` = -1).
    """
    in_low = di < oversold
    in_high = di > overbought
    trig_low = in_low & ~in_low.shift(1).fillna(False)
    trig_high = in_high & ~in_high.shift(1).fillna(False)
    signal = pd.Series(0, index=di.index, dtype=int)
    signal[trig_low.fillna(False)] = 1
    signal[trig_high.fillna(False)] = -1
    entries = signal[signal != 0]
    return pd.DataFrame({"dir": entries.astype(int)})


def random_directions(n: int, seed: int = 0) -> np.ndarray:
    """A reproducible vector of +-1 — the control arm's coin."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=n)


def run_trades(bars: pd.DataFrame, entries: pd.DataFrame, hold_days: int = 5,
               cost_bps: float = 5.0, directions: np.ndarray | None = None) -> pd.DataFrame:
    """Fixed-horizon forward-return trades: enter next open, exit close of t+hold_days-1.

    ``cost_bps`` is one-way x NAV; a round trip charges it twice (entry + exit).
    ``directions`` overrides the entry sign vector (the random-control arm passes it).
    """
    open_ = bars["Open"]
    close_ = bars["Close"]
    idx = bars.index
    n_bars = len(bars)
    pos = {ts: i for i, ts in enumerate(idx)}
    dirs = (np.asarray(directions, dtype=int) if directions is not None
            else entries["dir"].to_numpy(dtype=int))

    rows = []
    for sig_ts, d in zip(entries.index, dirs):
        i = pos.get(sig_ts)
        if i is None or i + 1 >= n_bars:
            continue
        e = i + 1
        entry_px = open_.iat[e]
        exit_idx = min(e + hold_days - 1, n_bars - 1)
        exit_px = close_.iat[exit_idx]
        ret_gross = d * (exit_px - entry_px) / entry_px
        rows.append({
            "entry_ts": idx[e], "dir": int(d), "ret_gross": ret_gross,
            "ret_net": ret_gross - 2.0 * cost_bps / 1e4,
        })
    return pd.DataFrame(rows)


def summarize(ledger: pd.DataFrame, col: str = "ret_net", lags: int = 5) -> dict:
    """Headline per-trade statistics: n, win-rate, mean (bps), HAC t on the mean."""
    r = ledger[col].to_numpy(dtype=float) if len(ledger) > 0 else np.array([])
    r = r[np.isfinite(r)]
    n = r.size
    out = {
        "n_trades": int(n),
        "win_rate": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "tstat": float("nan"),
    }
    if n > 5:
        out["tstat"] = _hac_mean_t(r, lags)
    return out


def _hac_mean_t(r: np.ndarray, lags: int) -> float:
    n = r.size
    mu = r.mean()
    e = r - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def pooled_trade_ledger(bars_map: dict[str, pd.DataFrame], window: int = 10,
                        oversold: float = 95.0, overbought: float = 105.0,
                        hold_days: int = 5, cost_bps: float = 5.0,
                        seed: int = 680) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pooled (strategy ledger, random-control ledger) across the whole universe."""
    strat_parts, rand_parts = [], []
    for _, bars in bars_map.items():
        di = disparity_index(bars["Close"], window)
        entries = zone_signals(di, oversold, overbought)
        if entries.empty:
            continue
        strat = run_trades(bars, entries, hold_days, cost_bps)
        rdirs = random_directions(len(entries), seed=seed)
        rand = run_trades(bars, entries, hold_days, cost_bps, directions=rdirs)
        strat_parts.append(strat)
        rand_parts.append(rand)
    strat_ledger = pd.concat(strat_parts, ignore_index=True) if strat_parts else pd.DataFrame()
    rand_ledger = pd.concat(rand_parts, ignore_index=True) if rand_parts else pd.DataFrame()
    return strat_ledger, rand_ledger


# --------------------------------------------------------------------------- #
# Drift baseline — does the surviving leg beat plain buy-any-random-day?
# --------------------------------------------------------------------------- #
def random_day_baseline(bars_map: dict[str, pd.DataFrame], window: int = 10,
                        oversold: float = 95.0, overbought: float = 105.0,
                        hold_days: int = 5, cost_bps: float = 5.0,
                        seed: int = 680) -> dict:
    """Each single-sided leg (buy-oversold, short-overbought) vs a same-size,
    same-direction, same-ticker RANDOM-DAY entry control — not the random-*direction*
    coin used above, but a random *day*.

    The universe pools two-decade-plus bull-drift names (NVDA, TSLA…), so the honest
    question is not "does DI beat a coin flip" but "does buying oversold beat simply
    buying on an arbitrary day of the same stock" — i.e. is the edge conditional
    information, or just captured drift. For each ticker we draw ``len(entries)`` random
    entry days (no replacement) matched in direction and pool exactly like the strategy
    ledger; HAC(``hold_days``) t on the pooled net returns and Welch *t* of the difference.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for leg, d in (("low", 1), ("high", -1)):
        strat_parts, rand_parts = [], []
        for _, bars in bars_map.items():
            di = disparity_index(bars["Close"], window)
            z = zone_signals(di, oversold, overbought)
            leg_entries = z[z["dir"] == d]
            if leg_entries.empty:
                continue
            strat_parts.append(run_trades(bars, leg_entries, hold_days, cost_bps))
            n_bars = len(bars)
            n_pick = min(len(leg_entries), max(n_bars - hold_days - 1, 0))
            if n_pick <= 0:
                continue
            idxpos = rng.choice(np.arange(n_bars - hold_days - 1), size=n_pick, replace=False)
            rand_entries = pd.DataFrame({"dir": d}, index=bars.index[idxpos])
            rand_parts.append(run_trades(bars, rand_entries, hold_days, cost_bps))
        strat = pd.concat(strat_parts, ignore_index=True) if strat_parts else pd.DataFrame()
        rand = pd.concat(rand_parts, ignore_index=True) if rand_parts else pd.DataFrame()
        ss = summarize(strat, "ret_net", lags=hold_days)
        rs = summarize(rand, "ret_net", lags=hold_days)
        wt = (welch_t(strat["ret_net"].to_numpy(), rand["ret_net"].to_numpy())
              if len(strat) and len(rand) else float("nan"))
        out[leg] = {
            "n": ss["n_trades"], "mean_bps": ss["mean_bps"], "tstat": ss["tstat"],
            "rand_mean_bps": rs["mean_bps"], "rand_tstat": rs["tstat"],
            "delta_bps": (ss["mean_bps"] - rs["mean_bps"]
                         if np.isfinite(ss["mean_bps"]) and np.isfinite(rs["mean_bps"])
                         else float("nan")),
            "welch_t_vs_random_day": wt,
        }
    return out


# --------------------------------------------------------------------------- #
# Parameter robustness grid (window x thresholds)
# --------------------------------------------------------------------------- #
def param_grid(bars_map: dict[str, pd.DataFrame], windows=(5, 10, 20, 25),
              threshold_pairs=((97.0, 103.0), (95.0, 105.0), (93.0, 107.0)),
              h: int = 5) -> pd.DataFrame:
    """Sweep window x (oversold, overbought) and report the conditional-split t-stats."""
    rows = []
    for w in windows:
        for lo, hi in threshold_pairs:
            r = pooled_conditional(bars_map, window=w, oversold=lo, overbought=hi, h=h)
            rows.append({
                "window": w, "oversold": lo, "overbought": hi,
                "n_low": r["n_low"], "n_high": r["n_high"],
                "welch_t_low": r["welch_t_low"], "welch_t_high": r["welch_t_high"],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# "Just short-term reversal?" diagnostic — is DI distinct from trailing return?
# --------------------------------------------------------------------------- #
def di_return_correlation(bars_map: dict[str, pd.DataFrame], window: int = 10) -> dict:
    """Pooled Pearson correlation between DI(window) and the trailing window-day return.

    DI is a moving-average-relative distance; a trailing return is a start/end-point
    distance. Both are monotonic summaries of "how far has price moved lately" — if they
    are almost perfectly correlated, conditioning on DI extremes is mechanically the same
    exercise as conditioning on trailing-return extremes (i.e. short-term/one-month
    reversal), just relabeled.
    """
    di_all, ret_all = [], []
    for _, bars in bars_map.items():
        di = disparity_index(bars["Close"], window)
        ret = trailing_return(bars["Close"], window) * 100.0  # same units as DI-100
        both = pd.concat([di, ret], axis=1).dropna()
        di_all.append(both["di"].to_numpy())
        ret_all.append(both["ret_trailing"].to_numpy())
    di_all = np.concatenate(di_all)
    ret_all = np.concatenate(ret_all)
    corr = float(np.corrcoef(di_all - 100.0, ret_all)[0, 1])
    return {"n": len(di_all), "corr": corr}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(close_bars: pd.DataFrame, window: int = 10,
                     oversold: float = 95.0, overbought: float = 105.0,
                     h: int = 5) -> dict:
    """Run the headline conditional split on one synthetic tape."""
    return pooled_conditional({"SYN": close_bars}, window=window,
                              oversold=oversold, overbought=overbought, h=h)
