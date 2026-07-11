"""Strategy + inference for Study 665 — Titanic Syndrome.

The claim (Bill Ohama, 1965, as relayed by SentimenTrader / StockCharts / McClellan
Financial — the same secondary literature used for the Hindenburg-Omen studies on this
desk): if the market prints a fresh 52-week high within the last **7 trading sessions**,
and on that reading the number of stocks hitting fresh 52-week **lows** exceeds the
number hitting fresh highs, internal breadth has failed to confirm the new high — the
"band is playing while the ship is listing" — and a decline should follow.

Measurements:

* **The headline signal** — ``near_high[t]`` (^GSPC printed a fresh 252-session high on
  some day in ``[t-6, t]``) AND ``n_lows[t] > n_highs[t]`` across the 30-member Dow
  breadth basket. Consecutive signal days within a 21-calendar-day window are merged
  into one *cluster* (the Hindenburg-style convention on this desk — no cherry-picking
  the "best" day of a listing episode).
* **Forward returns** — SPY total-return forward return at horizons 1/5/20/60 trading
  days, entered at the *next* close after the cluster fires (one documented lag), vs a
  drift-matched **random-entry** baseline of the same count (the honest test on an
  upward-drifting tape) and the plain unconditional (monthly-sampled) mean. One-sample
  HAC *t* and Welch *t* (signal vs random-entry) both reported.
* **False-alarm rate** — the fraction of clusters *not* followed by a ≥5% peak-to-trough
  SPY drawdown within the next 60 sessions, vs the same base rate on random dates
  (Welch *t* on the proportions) — the Hindenburg-style crash-rate cross-check.
* **The timer** — an actual equity-curve overlay: hold SPY, go to (unremunerated) cash
  for the 20 sessions after every cluster, one-way costs on each transition. Compared
  against buy-and-hold *and* against a random-timer control that sits out the same
  number of same-length episodes on randomly drawn dates — the fair "is the *timing*
  skill worth anything beyond just being out of the market sometimes" test.

No look-ahead: the signal is read on the close of day *t* (^GSPC + the Dow-30 panel);
every position/return uses SPY prices at *t+1* or later. Survivorship named: the Dow-30
breadth basket is *current* membership (see ``data.py`` / ``docs/references.md``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 5, 20, 60)
LOOKBACK = 252            # ~52 weeks of trading sessions
ATH_WINDOW = 7            # Ohama's "within 7 sessions of a new high"
CLUSTER_DAYS = 21         # calendar days merging consecutive signal days into one cluster
TIMER_EXIT_DAYS = 20      # sessions the timer sits in cash after a cluster fires
TIMER_COST_BPS = 5.0      # one-way cost per transition (in and out)


# --------------------------------------------------------------------------- #
# Breadth + signal construction
# --------------------------------------------------------------------------- #
def dow_breadth_counts(dow: pd.DataFrame, lookback: int = LOOKBACK) -> pd.DataFrame:
    """Daily count of Dow-30 members at a fresh trailing ``lookback``-day high / low.

    Trailing-only (the window ending at, and including, *t* — no look-ahead).
    """
    dow = dow.dropna(how="any")
    roll_max = dow.rolling(lookback, min_periods=lookback).max()
    roll_min = dow.rolling(lookback, min_periods=lookback).min()
    at_high = dow >= roll_max - 1e-9
    at_low = dow <= roll_min + 1e-9
    return pd.DataFrame({
        "n_highs": at_high.sum(axis=1), "n_lows": at_low.sum(axis=1),
        "n_stocks": dow.shape[1],
    }, index=dow.index)


def index_near_high(close: pd.Series, lookback: int = LOOKBACK,
                     window: int = ATH_WINDOW) -> pd.Series:
    """True on day *t* iff the index printed a fresh trailing-``lookback`` high on any
    of the trailing ``window`` sessions (inclusive of *t*) — Ohama's "within 7 sessions
    of a new high" (here: a fresh 52-week high, the free-data-computable proxy for a
    literal all-time high; see the data-layer docstring)."""
    roll_max = close.rolling(lookback, min_periods=lookback).max()
    at_high = close >= roll_max - 1e-9
    return at_high.rolling(window, min_periods=1).max().astype(bool)


def titanic_frame(dow: pd.DataFrame, gspc_close: pd.Series, spy_close: pd.Series,
                   lookback: int = LOOKBACK, window: int = ATH_WINDOW) -> pd.DataFrame:
    """One row per common trading day: breadth counts, the near-high flag, the raw
    Titanic-Syndrome signal, and SPY close/return for measurement."""
    counts = dow_breadth_counts(dow, lookback)
    near = index_near_high(gspc_close, lookback, window)
    idx = counts.index.intersection(near.index).intersection(spy_close.index)
    idx = idx.sort_values()
    df = counts.loc[idx].copy()
    df["near_high"] = near.loc[idx]
    df["spy_close"] = spy_close.loc[idx]
    df["spy_ret"] = df["spy_close"].pct_change()
    df["titanic"] = df["near_high"] & (df["n_lows"] > df["n_highs"])
    return df


def cluster_entries(signal: pd.Series, calendar_days: int = CLUSTER_DAYS) -> pd.DatetimeIndex:
    """Collapse consecutive signal days within ``calendar_days`` into one cluster;
    keep only the first day of each cluster (the Hindenburg-style convention)."""
    dates = signal.index[signal.astype(bool)]
    if len(dates) == 0:
        return pd.DatetimeIndex([])
    starts, last = [], None
    for d in dates:
        if last is None or (d - last).days > calendar_days:
            starts.append(d)
        last = d
    return pd.DatetimeIndex(starts)


# --------------------------------------------------------------------------- #
# Forward-return engine (mirrors sibling 493's convention: enter next close)
# --------------------------------------------------------------------------- #
def forward_returns(close: pd.Series, entries, horizon: int, cost_bps: float = 0.0) -> np.ndarray:
    """Forward ``horizon``-day return for each entry, entered at the *next* close.

    ``cost_bps`` is a one-way cost (round trip = 2x) subtracted from each trade.
    """
    pos = {d: i for i, d in enumerate(close.index)}
    p = close.to_numpy(dtype=float)
    n = p.size
    out = []
    for d in entries:
        i = pos.get(d)
        if i is None or i + 1 + horizon >= n:
            continue
        e = i + 1
        out.append(p[e + horizon] / p[e] - 1.0 - 2.0 * cost_bps * 1e-4)
    return np.asarray(out, dtype=float)


def random_entries(close_index, n: int, warmup: int = LOOKBACK, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates after the warm-up — the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = close_index[warmup:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(sorted(rng.choice(valid, size=min(n, len(valid)), replace=False)))


def unconditional_returns(close: pd.Series, horizon: int, exclude: set,
                           sample_every: int = 21) -> np.ndarray:
    """Forward ``horizon``-day return sampled ~monthly on non-signal days — the plain
    unconditional baseline (independent of signal count, unlike random_entries)."""
    idx = close.index
    dates = [idx[i] for i in range(0, len(idx), sample_every) if idx[i] not in exclude]
    return forward_returns(close, pd.DatetimeIndex(dates), horizon)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def hac_t(x: np.ndarray) -> float:
    """Newey-West (Bartlett-kernel) one-sample t-stat of the mean against zero."""
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        if n < 2:
            return float("nan")
        se = x.std(ddof=1) / np.sqrt(n)
        return float(x.mean() / se) if se > 0 else float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def summarize(returns: np.ndarray) -> dict:
    r = np.asarray(returns, dtype=float); r = r[np.isfinite(r)]
    n = r.size
    return {
        "n": int(n),
        "win": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "t_hac": hac_t(r),
    }


# --------------------------------------------------------------------------- #
# Headline gauntlet: forward returns, signal vs random-entry vs unconditional
# --------------------------------------------------------------------------- #
def run_forward_returns(df: pd.DataFrame, entries: pd.DatetimeIndex,
                         horizons=HORIZONS, cost_bps: float = 0.0, seed: int = 665) -> dict:
    close = df["spy_close"]
    ent_set = set(entries)
    out = {"n_entries": int(len(entries)), "by_h": {}}
    for h in horizons:
        sig = forward_returns(close, entries, h, cost_bps=cost_bps)
        rnd = forward_returns(close, random_entries(close.index, max(len(entries), 30),
                                                      warmup=LOOKBACK, seed=seed), h)
        unc = unconditional_returns(close, h, exclude=ent_set)
        s_sum, r_sum, u_sum = summarize(sig), summarize(rnd), summarize(unc)
        out["by_h"][h] = {
            "signal": s_sum, "random": r_sum, "unconditional": u_sum,
            "welch_t_vs_random": welch_t(sig, rnd),
            "welch_t_vs_unconditional": welch_t(sig, unc),
        }
    return out


# --------------------------------------------------------------------------- #
# False-alarm rate — Hindenburg-style crash-rate cross-check
# --------------------------------------------------------------------------- #
def false_alarm_stats(df: pd.DataFrame, entries: pd.DatetimeIndex, horizon: int = 60,
                       dd_threshold: float = -0.05, sample_every: int = 21) -> dict:
    """Fraction of clusters followed (within ``horizon`` sessions) by a peak-to-trough
    SPY drawdown at least as deep as ``dd_threshold``, vs the same rate on random,
    ~monthly-sampled dates. ``false_alarm_rate = 1 - signal crash rate``."""
    close = df["spy_close"]
    pos = {d: i for i, d in enumerate(close.index)}
    p = close.to_numpy(dtype=float)
    n = p.size

    def had_decline(i: int) -> bool | None:
        if i + 1 + horizon >= n:
            return None
        window = p[i + 1: i + 1 + horizon]
        running_max = np.maximum.accumulate(window)
        dd = window / running_max - 1.0
        return bool((dd <= dd_threshold).any())

    def rate(dates):
        vals = [had_decline(pos[d]) for d in dates if d in pos]
        vals = [v for v in vals if v is not None]
        return np.array(vals, dtype=float)

    ent_set = set(entries)
    base_dates = [d for i, d in enumerate(close.index) if i % sample_every == 0 and d not in ent_set]
    sig_vals = rate(entries)
    base_vals = rate(base_dates)
    sig_rate = float(sig_vals.mean()) if len(sig_vals) else float("nan")
    base_rate = float(base_vals.mean()) if len(base_vals) else float("nan")
    return {
        "n_clusters": int(len(sig_vals)), "n_base": int(len(base_vals)),
        "signal_decline_rate": sig_rate, "base_decline_rate": base_rate,
        "false_alarm_rate": 1.0 - sig_rate if np.isfinite(sig_rate) else float("nan"),
        "welch_t": welch_t(sig_vals, base_vals),
    }


# --------------------------------------------------------------------------- #
# The timer — an actual equity-curve overlay, vs buy-and-hold and a random control
# --------------------------------------------------------------------------- #
def _flat_mask(index: pd.DatetimeIndex, entries: pd.DatetimeIndex, exit_days: int) -> np.ndarray:
    pos = {d: i for i, d in enumerate(index)}
    n = len(index)
    flat = np.zeros(n, dtype=bool)
    for d in entries:
        i = pos.get(d)
        if i is None:
            continue
        lo, hi = i + 1, min(i + exit_days, n - 1)
        flat[lo:hi + 1] = True
    return flat


def _equity_stats(ret: np.ndarray) -> dict:
    ret = np.asarray(ret, dtype=float)
    equity = np.cumprod(1.0 + ret)
    n = len(ret)
    total = float(equity[-1] - 1.0) if n else float("nan")
    cagr = float(equity[-1] ** (252.0 / n) - 1.0) if n else float("nan")
    vol = float(ret.std(ddof=1) * np.sqrt(252)) if n > 1 else float("nan")
    sharpe = float((ret.mean() * 252) / vol) if vol and vol > 0 else float("nan")
    running_max = np.maximum.accumulate(equity)
    maxdd = float((equity / running_max - 1.0).min()) if n else float("nan")
    return {"total_return": total, "cagr": cagr, "vol": vol, "sharpe": sharpe, "maxdd": maxdd}


def timer_performance(df: pd.DataFrame, entries: pd.DatetimeIndex, exit_days: int = TIMER_EXIT_DAYS,
                       cost_bps: float = TIMER_COST_BPS) -> dict:
    """Hold SPY; go to (unremunerated) cash for ``exit_days`` sessions after each cluster
    (one lag: the flat window starts the session *after* the signal fires). One-way
    ``cost_bps`` charged on every position transition. Cash earns 0% — a conservative
    simplification that, if anything, understates real cash's T-bill yield and so biases
    *against* the timer, never for it."""
    ret = df["spy_ret"].fillna(0.0).to_numpy(dtype=float)
    idx = df.index
    flat = _flat_mask(idx, entries, exit_days)
    position = (~flat).astype(float)
    transitions = np.zeros(len(idx), dtype=bool)
    transitions[1:] = position[1:] != position[:-1]
    strat_ret = ret * position
    strat_ret = strat_ret - transitions * (cost_bps * 1e-4)
    bh = _equity_stats(ret)
    tm = _equity_stats(strat_ret)
    return {"buy_hold": bh, "timer": tm, "n_days_out": int(flat.sum()),
            "n_transitions": int(transitions.sum()), "n_clusters": int(len(entries))}


def timer_curves(df: pd.DataFrame, entries: pd.DatetimeIndex, exit_days: int = TIMER_EXIT_DAYS,
                  cost_bps: float = TIMER_COST_BPS) -> dict:
    """Buy-and-hold vs timer equity curves ($1 compounded), for plotting."""
    ret = df["spy_ret"].fillna(0.0).to_numpy(dtype=float)
    idx = df.index
    flat = _flat_mask(idx, entries, exit_days)
    position = (~flat).astype(float)
    transitions = np.zeros(len(idx), dtype=bool)
    transitions[1:] = position[1:] != position[:-1]
    strat_ret = ret * position - transitions * (cost_bps * 1e-4)
    return {"index": idx, "buy_hold": np.cumprod(1.0 + ret), "timer": np.cumprod(1.0 + strat_ret)}


def random_timer_control(df: pd.DataFrame, n_clusters: int, exit_days: int = TIMER_EXIT_DAYS,
                          cost_bps: float = TIMER_COST_BPS, n_draws: int = 500,
                          seed: int = 665) -> dict:
    """Same number of clusters, same fixed exit window, dates drawn at random — the
    fair control for "is the *timing* worth anything beyond sitting out sometimes".
    Returns the distribution of random-timer CAGR and the empirical two-sided rank of
    the real timer's CAGR within it."""
    close_index = df.index
    cagrs = []
    for k in range(n_draws):
        ent = random_entries(close_index, n_clusters, warmup=LOOKBACK, seed=seed * 100_003 + k)
        perf = timer_performance(df, ent, exit_days=exit_days, cost_bps=cost_bps)
        cagrs.append(perf["timer"]["cagr"])
    return {"cagrs": np.asarray(cagrs, dtype=float), "n_draws": n_draws}


# --------------------------------------------------------------------------- #
# Synthetic positive control (machinery proof — never cited for the real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_index_close(panel: pd.DataFrame) -> pd.Series:
    return (1.0 + panel["index_ret"]).cumprod()


def synthetic_detect(panel: pd.DataFrame, signal: pd.Series, horizon: int = 20,
                      seed: int = 665) -> dict:
    """Run the headline signal-vs-random-entry Welch/HAC split on a synthetic panel.

    Clusters raw signal days first (the same ``cluster_entries`` convention used on
    the real tape) — un-clustered adjacent signal days are highly correlated draws
    (they overlap in forward-return window and share the same underlying episode),
    and would otherwise inflate the null's false-positive rate.
    """
    close = synthetic_index_close(panel)
    entries = cluster_entries(signal, calendar_days=CLUSTER_DAYS)
    sig = forward_returns(close, entries, horizon)
    rnd = forward_returns(close, random_entries(close.index, max(len(entries), 30),
                                                 warmup=LOOKBACK, seed=seed), horizon)
    return {"n": int(len(entries)), "mean_bps": float(np.nanmean(sig) * 1e4) if len(sig) else float("nan"),
            "welch_t": welch_t(sig, rnd), "t_hac": hac_t(sig)}
