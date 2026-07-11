"""Strategy + inference for Study 679 — Psychological Line.

The claim: **PSY(N) = 100 x (up-closes in the last N days) / N** is a crowd-emotion
thermometer. Above 75%, "everyone who wants to buy already has" -> sell. Below 25%,
"everyone who wants to sell already has" -> buy. N = 12 and the 75/25 thresholds are the
textbook (Nison-era Japanese technical analysis) defaults, tested exactly as stated.

One documented execution lag throughout: PSY is known from closes up to and including bar
*t*; every trade (or forward-return window) enters at bar *t+1*'s **open** and, where a
holding period applies, exits at the close of bar *t + h*. No look-ahead.

Two complementary measurements share the same PSY indicator:

* **Conditional-forward-return split** — every day is bucketed by its PSY reading
  (oversold / neutral / overbought); the forward *h*-day return of each bucket is compared
  to the neutral bucket with Welch *t* (i.i.d. across non-overlapping events isn't true
  here — the *h*-day windows overlap — so a Newey-West (HAC, *h* lags) cross-check is
  reported alongside). This is the direct test of "forward returns conditioned on PSY
  extremes vs unconditional."
* **Zone-trigger trade ledger** — a signal fires only the day PSY first *enters* a zone
  (the trigger, not every day inside it, so trades don't pile up on top of each other);
  entered at the next open, held a fixed window, one-way costs x 2 per round trip. Pinned
  against a **random-direction control** on the identical entry bars (the "beats a coin?"
  myth-check).

Both run pooled across the SPY + basket universe and are swept over a window x threshold
grid for robustness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# The indicator
# --------------------------------------------------------------------------- #
def psy_line(close: pd.Series, window: int = 12) -> pd.Series:
    """PSY(window): 100 x (share of up-closes in the trailing ``window`` days).

    Known at the close of bar *t* (uses closes up to and including *t*) — available to
    trade at bar *t+1*'s open.
    """
    up = (close.diff() > 0).astype(float)
    psy = up.rolling(window, min_periods=window).sum() / window * 100.0
    return psy.rename("psy")


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
# Conditional forward returns — every day bucketed by its PSY reading
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


def pooled_conditional(bars_map: dict[str, pd.DataFrame], window: int = 12,
                       oversold: float = 25.0, overbought: float = 75.0,
                       h: int = 5) -> dict:
    """Pool forward returns across the whole universe, bucketed by the PSY reading.

    ``low``/``high`` are **trigger days only** — the day PSY first *enters* the oversold
    (< ``oversold``) / overbought (> ``overbought``) zone, not every day inside it. PSY is a
    trailing rolling window, so consecutive in-zone days share nearly the same underlying
    12 (or however many) closes and are barely independent draws; scoring only the entry day
    keeps the events close to non-overlapping, which is what a Welch *t* assumes. ``mid`` =
    every other (non-trigger) day, the unconditional control. Welch *t* (planned primary)
    plus a Newey-West *t* with ``h`` lags (the residual overlap-robust cross-check, since the
    *h*-day forward windows themselves still overlap).
    """
    lows, highs, mids = [], [], []
    for _, bars in bars_map.items():
        psy = psy_line(bars["Close"], window)
        fwd = forward_returns(bars, h)
        trig = zone_signals(psy, oversold, overbought, min_gap=h)
        df = pd.concat([psy, fwd], axis=1)
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
# Zone-trigger trade ledger — enter only the day PSY first crosses into a zone
# --------------------------------------------------------------------------- #
def zone_signals(psy: pd.Series, oversold: float = 25.0, overbought: float = 75.0,
                 min_gap: int = 0) -> pd.DataFrame:
    """Trigger days: the first day PSY crosses *into* a zone (not every day inside it).

    Contrarian direction: entering oversold (PSY < oversold) -> buy (``dir`` = +1);
    entering overbought (PSY > overbought) -> sell/short (``dir`` = -1).

    ``min_gap`` (in trading days) enforces a cooldown after each accepted trigger before
    the next one counts. PSY is a rolling window, so a reading that "flaps" back and forth
    across a threshold produces several re-entry triggers whose forward-return windows
    overlap in time — pseudo-replication that inflates a Welch *t*'s significance. A
    cooldown of the same length as the forward-return horizon keeps events close to
    non-overlapping, the assumption the Welch split relies on.
    """
    in_low = psy < oversold
    in_high = psy > overbought
    trig_low = in_low & ~in_low.shift(1).fillna(False)
    trig_high = in_high & ~in_high.shift(1).fillna(False)
    signal = pd.Series(0, index=psy.index, dtype=int)
    signal[trig_low.fillna(False)] = 1
    signal[trig_high.fillna(False)] = -1
    entries = signal[signal != 0]
    if min_gap > 0 and len(entries) > 1:
        pos = {ts: i for i, ts in enumerate(psy.index)}
        kept, last_pos = [], -10**9
        for ts, d in entries.items():
            i = pos[ts]
            if i - last_pos >= min_gap:
                kept.append((ts, d))
                last_pos = i
        entries = pd.Series({ts: d for ts, d in kept})
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


def pooled_trade_ledger(bars_map: dict[str, pd.DataFrame], window: int = 12,
                        oversold: float = 25.0, overbought: float = 75.0,
                        hold_days: int = 5, cost_bps: float = 5.0,
                        seed: int = 679) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pooled (strategy ledger, random-control ledger) across the whole universe."""
    strat_parts, rand_parts = [], []
    for _, bars in bars_map.items():
        psy = psy_line(bars["Close"], window)
        entries = zone_signals(psy, oversold, overbought, min_gap=hold_days)
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
# Parameter robustness grid (window x thresholds)
# --------------------------------------------------------------------------- #
def param_grid(bars_map: dict[str, pd.DataFrame], windows=(10, 12, 14, 20),
              threshold_pairs=((20.0, 80.0), (25.0, 75.0), (30.0, 70.0)),
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
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(close_bars: pd.DataFrame, window: int = 12,
                     oversold: float = 25.0, overbought: float = 75.0,
                     h: int = 5) -> dict:
    """Run the headline conditional split on one synthetic tape."""
    return pooled_conditional({"SYN": close_bars}, window=window,
                              oversold=oversold, overbought=overbought, h=h)
