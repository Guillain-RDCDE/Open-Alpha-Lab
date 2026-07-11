"""Strategy + inference for Study 670 — Bollinger-Squeeze.

The claim (the "TTM Squeeze", John Carter / *Mastering the Trade*): when the
**Bollinger Bands contract to sit INSIDE the Keltner Channels**, realized volatility
has compressed to an unusual low and a **big directional move is coming** — trade the
breakout in the direction the price commits to when the bands re-expand.

Bands, both causal (trailing data only, no look-ahead):

* **Bollinger Bands** — SMA(20) of close +- ``bb_std`` * rolling std(20) of close
  (default ``bb_std = 2.0``).
* **Keltner Channel** — EMA(20) of close +- ``kc_mult`` * ATR(20) (default
  ``kc_mult = 1.5``, the canonical TTM parameterisation; ATR is Wilder-style, a
  20-bar rolling mean of True Range).
* **Squeeze ON**: the Bollinger Band sits entirely inside the Keltner Channel
  (``bb_upper < kc_upper`` and ``bb_lower > kc_lower``) — volatility is unusually
  low relative to its own trailing average.
* **Squeeze OFF / release**: the first bar the squeeze condition fails after a run
  of >= ``min_run`` squeeze bars — the "fire" the claim says to trade.
* **Direction**: the sign of a causal OLS slope of close over the trailing
  ``slope_window`` bars, evaluated ON the release bar (no look-ahead) — a simple,
  auditable proxy for the TTM momentum histogram.

Two DISTINCT questions, deliberately kept separate (the point of this study):

1. **Does volatility expand after the release?** (mechanical, expected true —
   the squeeze is *defined* as compressed vol, so some reversion to the mean is
   close to definitional; the honest test is whether the release predicts MORE
   expansion than a random day, not merely that vol is no longer at its low.)
2. **Does the signed breakout call the direction of the next move — profitably,
   net of costs?** (the tradable claim, expected weak-to-none).

Both are pinned against a **matched random-entry control** (same ticker, same event
count, same direction mix for the directional test) so any bull-market drift or
generic vol-clustering the tape carries anyway is subtracted out, not mistaken for
squeeze-specific information.

Execution: signal on the release bar's close, position entered at the **next bar's
open** — one documented lag, no look-ahead. Costs are one-way x NAV per leg (2 legs
per round trip); short legs additionally pay a borrow accrual.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Bands
# --------------------------------------------------------------------------- #
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def atr(bars: pd.DataFrame, n: int = 20) -> pd.Series:
    """Average True Range (simple rolling mean of TR, Wilder-style bar range)."""
    prev_c = bars["close"].shift(1)
    tr = pd.concat(
        [bars["high"] - bars["low"], (bars["high"] - prev_c).abs(), (bars["low"] - prev_c).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(close, window)
    sd = close.rolling(window, min_periods=window).std(ddof=0)
    return pd.DataFrame({"mid": mid, "upper": mid + num_std * sd, "lower": mid - num_std * sd})


def keltner_channel(bars: pd.DataFrame, ema_span: int = 20, atr_n: int = 20,
                     mult: float = 1.5) -> pd.DataFrame:
    mid = ema(bars["close"], span=ema_span)
    a = atr(bars, n=atr_n)
    return pd.DataFrame({"mid": mid, "upper": mid + mult * a, "lower": mid - mult * a})


def rolling_slope(y: pd.Series, window: int) -> pd.Series:
    """Causal OLS slope of ``y`` on {0..window-1} over each trailing window (raw units/bar)."""
    x = np.arange(window, dtype=float)
    x_c = x - x.mean()
    denom = float((x_c ** 2).sum())

    def _slope(arr: np.ndarray) -> float:
        return float(np.dot(arr - arr.mean(), x_c) / denom)

    return y.rolling(window, min_periods=window).apply(_slope, raw=True)


# --------------------------------------------------------------------------- #
# Squeeze detection
# --------------------------------------------------------------------------- #
def squeeze_frame(bars: pd.DataFrame, bb_window: int = 20, bb_std: float = 2.0,
                   kc_span: int = 20, kc_atr_n: int = 20, kc_mult: float = 1.5,
                   slope_window: int = 20) -> pd.DataFrame:
    """One row per bar: bands, the squeeze flag and the causal breakout slope."""
    close = bars["close"]
    bb = bollinger_bands(close, window=bb_window, num_std=bb_std)
    kc = keltner_channel(bars, ema_span=kc_span, atr_n=kc_atr_n, mult=kc_mult)
    df = pd.DataFrame(index=bars.index)
    df["close"] = close
    df["bb_up"], df["bb_lo"] = bb["upper"], bb["lower"]
    df["kc_up"], df["kc_lo"] = kc["upper"], kc["lower"]
    df["squeeze"] = (bb["upper"] < kc["upper"]) & (bb["lower"] > kc["lower"])
    df["slope"] = rolling_slope(close, slope_window)
    ret = np.log(close).diff()
    df["log_ret"] = ret
    return df.dropna(subset=["bb_up", "kc_up", "slope"])


def squeeze_release_events(df: pd.DataFrame, min_run: int = 5) -> pd.DataFrame:
    """Release-bar dates: the first non-squeeze bar after a run of >= ``min_run`` squeeze bars.

    Returns a DataFrame indexed by release date with a ``dir`` column (+-1, the sign
    of the causal slope on the release bar; ties broken long).
    """
    sq = df["squeeze"].to_numpy()
    n = len(sq)
    run = 0
    releases = []
    for i in range(n):
        if sq[i]:
            run += 1
        else:
            if run >= min_run:
                releases.append(i)
            run = 0
    idx = df.index[releases]
    dirs = np.sign(df["slope"].to_numpy()[releases])
    dirs[dirs == 0] = 1.0
    return pd.DataFrame({"dir": dirs.astype(int)}, index=idx)


def random_dates(df: pd.DataFrame, n: int, seed: int, warmup: int = 30) -> pd.DatetimeIndex:
    rng = np.random.default_rng(seed)
    valid = df.index[warmup:]
    n = min(n, len(valid))
    chosen = rng.choice(valid, size=n, replace=False)
    return pd.DatetimeIndex(sorted(chosen))


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def one_sample_t(x: np.ndarray, mu0: float = 0.0) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float((x.mean() - mu0) / se) if se > 0 else float("nan")


def hac_t(x: np.ndarray) -> float:
    """Newey-West HAC t of the sample mean (Bartlett kernel, automatic lag)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Test 1 — forward volatility expansion (the mechanical claim)
# --------------------------------------------------------------------------- #
def _fwd_vol(log_ret: pd.Series, dates: pd.DatetimeIndex, k: int) -> np.ndarray:
    idx = log_ret.index
    pos = {d: i for i, d in enumerate(idx)}
    out = []
    for d in dates:
        i = pos.get(d)
        if i is None or i + k >= len(idx):
            out.append(np.nan)
            continue
        window = log_ret.iloc[i + 1: i + 1 + k].to_numpy()
        out.append(float(np.std(window, ddof=1)) if len(window) >= 2 else np.nan)
    return np.asarray(out)


def _trail_vol(log_ret: pd.Series, dates: pd.DatetimeIndex, k: int) -> np.ndarray:
    idx = log_ret.index
    pos = {d: i for i, d in enumerate(idx)}
    out = []
    for d in dates:
        i = pos.get(d)
        if i is None or i - k + 1 < 0:
            out.append(np.nan)
            continue
        window = log_ret.iloc[i - k + 1: i + 1].to_numpy()
        out.append(float(np.std(window, ddof=1)) if len(window) >= 2 else np.nan)
    return np.asarray(out)


def vol_expansion_stats(df: pd.DataFrame, events: pd.DataFrame, k: int = 10,
                         n_random: int = 2000, seed: int = 670) -> dict:
    """Forward-K-day realized vol after release vs (a) the squeeze itself, (b) random days.

    (a) is close to tautological (a squeeze IS low trailing vol by construction) and
    is reported only as context. (b) is the honest test: does the release TIME the
    expansion better than picking a random day on the same tape?
    """
    dates = events.index
    fwd = _fwd_vol(df["log_ret"], dates, k)
    trail = _trail_vol(df["log_ret"], dates, k)
    ratio = fwd / trail

    rng = np.random.default_rng(seed)
    valid = df.index[30:-k] if k < len(df.index) - 30 else df.index[30:]
    n = min(n_random, max(len(valid) - 1, 1))
    rand_dates = pd.DatetimeIndex(sorted(rng.choice(valid, size=n, replace=False)))
    fwd_rand = _fwd_vol(df["log_ret"], rand_dates, k)

    return {
        "n_events": len(dates),
        "fwd_vol_mean": float(np.nanmean(fwd)), "trail_vol_mean": float(np.nanmean(trail)),
        "ratio_mean": float(np.nanmean(ratio)), "ratio_t_vs_1": one_sample_t(ratio, mu0=1.0),
        "fwd_rand_mean": float(np.nanmean(fwd_rand)), "n_random": len(rand_dates),
        "welch_t_vs_random": welch_t(fwd, fwd_rand),
        "fwd": fwd, "fwd_rand": fwd_rand,
    }


# --------------------------------------------------------------------------- #
# Test 2 — the directional breakout trade (the tradable claim)
# --------------------------------------------------------------------------- #
def breakout_ledger(bars: pd.DataFrame, events: pd.DataFrame, hold_days: int = 10,
                     cost_bps: float = 5.0, borrow_bps_annual: float = 20.0) -> pd.DataFrame:
    """Signed forward trades: enter next bar's open in ``dir``, exit after ``hold_days``.

    Net return = dir * (exit/entry - 1) - round-trip cost - (borrow accrual on shorts).
    """
    close = bars["close"]
    open_ = bars["open"]
    dates = bars.index.tolist()
    pos = {d: i for i, d in enumerate(dates)}
    n_bars = len(bars)
    rows = []
    for sig_date, d in zip(events.index, events["dir"]):
        i = pos.get(sig_date)
        if i is None or i + 1 >= n_bars:
            continue
        e = i + 1
        last = min(e + hold_days - 1, n_bars - 1)
        entry_px, exit_px = open_.iat[e], close.iat[last]
        held = last - e + 1
        ret_gross = d * (exit_px / entry_px - 1.0)
        cost = 2.0 * cost_bps * 1e-4
        borrow = (borrow_bps_annual * 1e-4) * (held / TRADING_DAYS) if d < 0 else 0.0
        rows.append({"entry_date": bars.index[e], "exit_date": bars.index[last],
                     "dir": int(d), "bars_held": held,
                     "ret_gross": ret_gross, "ret_net": ret_gross - cost - borrow})
    return pd.DataFrame(rows)


def random_control_ledger(bars: pd.DataFrame, df: pd.DataFrame, events: pd.DataFrame,
                           hold_days: int, cost_bps: float, borrow_bps_annual: float,
                           seed: int) -> pd.DataFrame:
    """Random-date, direction-mix-matched control: shuffle the SAME dir multiset onto
    ``len(events)`` random dates. Isolates squeeze *timing* from the long/short mix
    (and therefore from generic drift).

    ``bars`` is the raw OHLC frame (execution prices); ``df`` is the squeeze frame
    (for the valid-date universe, post warm-up).
    """
    rd = random_dates(df, len(events), seed=seed, warmup=30)
    rng = np.random.default_rng(seed + 1)
    dirs = events["dir"].to_numpy().copy()
    rng.shuffle(dirs)
    n = min(len(rd), len(dirs))
    ctrl = pd.DataFrame({"dir": dirs[:n]}, index=rd[:n])
    return breakout_ledger(bars, ctrl, hold_days=hold_days, cost_bps=cost_bps,
                            borrow_bps_annual=borrow_bps_annual)


def summarize(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    r = ledger[col].to_numpy(dtype=float) if len(ledger) else np.array([])
    n = r.size
    return {
        "n_trades": int(n),
        "win_rate": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "hac_t": hac_t(r) if n > 5 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Param robustness sweep
# --------------------------------------------------------------------------- #
def param_sweep(bars_by_ticker: dict[str, pd.DataFrame], bb_stds=(1.5, 2.0, 2.5),
                 kc_mults=(1.0, 1.5, 2.0), hold_days_list=(5, 10, 20),
                 cost_bps: float = 5.0, min_run: int = 5, seed: int = 670) -> pd.DataFrame:
    """Pooled (all tickers) directional Welch t vs matched random control, per param combo."""
    rows = []
    for bb_std in bb_stds:
        for kc_mult in kc_mults:
            pooled_events = {}
            frames = {}
            for tkr, bars in bars_by_ticker.items():
                df = squeeze_frame(bars, bb_std=bb_std, kc_mult=kc_mult)
                ev = squeeze_release_events(df, min_run=min_run)
                pooled_events[tkr] = ev
                frames[tkr] = df
            for hold in hold_days_list:
                sig_all, ctrl_all, n_ev = [], [], 0
                for tkr, bars in bars_by_ticker.items():
                    ev = pooled_events[tkr]
                    n_ev += len(ev)
                    if len(ev) == 0:
                        continue
                    led = breakout_ledger(bars, ev, hold_days=hold, cost_bps=cost_bps)
                    ctrl = random_control_ledger(bars, frames[tkr], ev, hold_days=hold,
                                                  cost_bps=cost_bps, borrow_bps_annual=20.0,
                                                  seed=seed)
                    sig_all.append(led["ret_net"].to_numpy())
                    ctrl_all.append(ctrl["ret_net"].to_numpy())
                sig = np.concatenate(sig_all) if sig_all else np.array([])
                ctrl = np.concatenate(ctrl_all) if ctrl_all else np.array([])
                rows.append({
                    "bb_std": bb_std, "kc_mult": kc_mult, "hold_days": hold,
                    "n_events": n_ev,
                    "mean_bps": float(sig.mean() * 1e4) if sig.size else float("nan"),
                    "ctrl_mean_bps": float(ctrl.mean() * 1e4) if ctrl.size else float("nan"),
                    "welch_t": welch_t(sig, ctrl),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(bars: pd.DataFrame, hold_days: int = 10, cost_bps: float = 5.0,
                      min_run: int = 5, seed: int = 670) -> dict:
    """Run the same squeeze-detection + directional test on a synthetic tape."""
    df = squeeze_frame(bars)
    ev = squeeze_release_events(df, min_run=min_run)
    if len(ev) < 5:
        return {"n_events": len(ev), "welch_t": float("nan"), "mean_bps": float("nan")}
    led = breakout_ledger(bars, ev, hold_days=hold_days, cost_bps=cost_bps)
    ctrl = random_control_ledger(bars, df, ev, hold_days=hold_days, cost_bps=cost_bps,
                                  borrow_bps_annual=20.0, seed=seed)
    sig, ctl = led["ret_net"].to_numpy(), ctrl["ret_net"].to_numpy()
    return {
        "n_events": len(ev),
        "mean_bps": float(sig.mean() * 1e4) if sig.size else float("nan"),
        "ctrl_mean_bps": float(ctl.mean() * 1e4) if ctl.size else float("nan"),
        "welch_t": welch_t(sig, ctl),
    }
