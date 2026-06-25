"""Chandelier Exit as a falsifiable mechanical rule — Study 479.

Chuck LeBeau's **chandelier exit** is a volatility-scaled trailing stop for a long-flat trend
system. With the canonical parameters ``n = 22``, ``m = 3``:

* compute **ATR(n)** (Wilder's Average True Range over ``n`` bars);
* track the **highest high since the position opened**, ``HH``;
* hang the stop ``m * ATR(n)`` below it: ``stop_t = HH_t - m * ATR_t``.

The position is **long** while the close holds above the trailing stop and goes **flat** the
first time the close pierces below it; a fresh **n-day breakout high** re-arms a long. The
folklore (LeBeau & Lucas, *Technical Traders Guide to Computer Analysis*, 1992): the ATR trail
*lets winners run and cuts losers*, so the chandelier-managed long **beats buy-and-hold** on a
risk-adjusted basis.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **No look-ahead.** ATR and the running high use only bars up to and including *t*; the
   stop/breakout is read on the close of *t*; any position change is executed at the **next**
   close (one documented lag).
2. **Entry events.** A long *entry* is the bar the position flips flat -> long. We measure the
   forward H-day return from the next close, exactly as for the other entry studies.
3. **Controls.**
   - **Buy-and-hold** — the passive long the chandelier claims to beat (the thesis baseline).
   - **Random-stop baseline** — a long-flat schedule with the *same time-in-market* but exits
     placed on random days (the drift-matched "is the ATR trail better than a coin-flip stop?"
     null).
   - **Scrambled-ATR placebo** — rebuild the trail from a permutation of the ATR series,
     destroying the volatility geometry while keeping its marginal: the honest "is the ATR
     trail itself load-bearing?" null.

HORIZONS = (5, 10, 20, 60). All entries are read on close of *t*, executed at close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)
ATR_N = 22
ATR_MULT = 3.0


# --------------------------------------------------------------------------- #
# Average True Range (Wilder)
# --------------------------------------------------------------------------- #
def atr(bars: pd.DataFrame, n: int = ATR_N) -> pd.Series:
    """Wilder's Average True Range over ``n`` bars (uses only past+current bars)."""
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    # Wilder smoothing (RMA): a = 1/n EWMA, seeded by the simple mean of the first n TRs.
    out = np.full(tr.size, np.nan)
    if tr.size >= n:
        out[n - 1] = tr[:n].mean()
        for i in range(n, tr.size):
            out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return pd.Series(out, index=bars.index)


# --------------------------------------------------------------------------- #
# Chandelier position series (long-flat)
# --------------------------------------------------------------------------- #
def chandelier_position(bars: pd.DataFrame, n: int = ATR_N, m: float = ATR_MULT) -> pd.DataFrame:
    """Long-flat chandelier state per bar, with no look-ahead.

    Returns a DataFrame over ``bars.index`` with columns:
      * ``atr``  — Wilder ATR(n)
      * ``hh``   — running highest-high since the current long opened (NaN while flat)
      * ``stop`` — chandelier trailing stop ``hh - m*atr`` (NaN while flat)
      * ``pos``  — 1 while long, 0 while flat (the state *as of the close of t*)

    Rule: re-arm a **long** when the close sets a fresh ``n``-day high (a breakout) while flat;
    while long, trail ``stop = max-high-since-entry - m*ATR``; exit to **flat** the first close
    below the stop. State is computed on close *t*; trading on it is lagged in
    :func:`chandelier_entries` / :func:`forward_returns`.
    """
    a = atr(bars, n=n).to_numpy()
    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    nbar = close.size

    # n-day breakout high (prior n bars, excluding current): close > rolling max of last n highs
    roll_hh = pd.Series(high, index=bars.index).rolling(n).max().shift(1).to_numpy()

    pos = np.zeros(nbar, dtype=int)
    hh = np.full(nbar, np.nan)
    stop = np.full(nbar, np.nan)
    in_long = False
    peak = np.nan
    for i in range(nbar):
        if np.isnan(a[i]):
            continue
        if not in_long:
            # re-arm long on a fresh n-day breakout high
            if not np.isnan(roll_hh[i]) and close[i] > roll_hh[i]:
                in_long = True
                peak = high[i]
        else:
            peak = max(peak, high[i])
            s = peak - m * a[i]
            if close[i] < s:
                in_long = False
                peak = np.nan
        if in_long:
            pos[i] = 1
            hh[i] = peak
            stop[i] = peak - m * a[i]
    return pd.DataFrame(
        {"atr": a, "hh": hh, "stop": stop, "pos": pos}, index=bars.index
    )


def chandelier_entries(bars: pd.DataFrame, n: int = ATR_N, m: float = ATR_MULT) -> pd.DatetimeIndex:
    """Bars where the chandelier flips **flat -> long** (the entry events).

    Entry is executed at the next close by :func:`forward_returns` (one documented lag).
    """
    pos = chandelier_position(bars, n=n, m=m)["pos"]
    flips = (pos == 1) & (pos.shift(1, fill_value=0) == 0)
    return bars.index[flips.to_numpy()]


def random_entries(close: pd.Series, n_ent: int, n: int = ATR_N, seed: int = 0) -> pd.DatetimeIndex:
    """``n_ent`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = close.index[2 * n:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n_ent, len(valid)), replace=False)
    return pd.DatetimeIndex(sorted(chosen))


# --------------------------------------------------------------------------- #
# Forward-return engine
# --------------------------------------------------------------------------- #
def forward_returns(close: pd.Series, entries, horizon: int, cost_bps: float = 0.0) -> np.ndarray:
    """Forward ``horizon``-day return for each entry, entered at the *next* close (one lag).

    ``cost_bps`` is a one-way cost charged on both legs (in + out), subtracted from each trade's
    return. Trades whose window overruns the tape are dropped.
    """
    pos = {d: i for i, d in enumerate(close.index)}
    p = close.to_numpy(dtype=float)
    nbar = p.size
    out = []
    for d in entries:
        i = pos.get(d)
        if i is None or i + 1 + horizon >= nbar:
            continue
        e = i + 1                      # enter at next close
        r = p[e + horizon] / p[e] - 1.0
        out.append(r - 2.0 * cost_bps * 1e-4)
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# Equity curves (the thesis axis: does the ATR trail beat holding?)
# --------------------------------------------------------------------------- #
def strategy_equity(bars: pd.DataFrame, n: int = ATR_N, m: float = ATR_MULT,
                    cost_bps: float = 1.0) -> dict:
    """Compounded long-flat chandelier equity vs buy-and-hold, with per-switch costs.

    The chandelier position (as of close *t*) earns the next day's close-to-close return when
    long, lagged one bar to avoid look-ahead. Every position change pays ``cost_bps`` one-way.
    Returns CAGR, annualised Sharpe, max-drawdown and time-in-market for both strategy and B&H,
    plus the time-in-market the chandelier is long (used to drift-match the random-stop null).
    """
    close = bars["close"].to_numpy(dtype=float)
    pos = chandelier_position(bars, n=n, m=m)["pos"].to_numpy()
    ret = np.concatenate([[0.0], close[1:] / close[:-1] - 1.0])
    # position decided at close t earns return of t->t+1 ; align: pos lagged by 1
    held = np.concatenate([[0], pos[:-1]])
    switch = np.abs(np.diff(np.concatenate([[0], held]))) > 0
    strat_ret = held * ret - switch * (cost_bps * 1e-4)
    bh_ret = ret.copy()
    return {
        "strat": _curve_stats(strat_ret),
        "bh": _curve_stats(bh_ret),
        "time_in_market": float(held.mean()),
        "n_switches": int(switch.sum()),
    }


def _curve_stats(daily_ret: np.ndarray) -> dict:
    r = np.asarray(daily_ret, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return {"cagr": float("nan"), "sharpe": float("nan"), "maxdd": float("nan")}
    eq = np.cumprod(1.0 + r)
    yrs = r.size / 252.0
    cagr = eq[-1] ** (1.0 / yrs) - 1.0 if eq[-1] > 0 else float("nan")
    sharpe = (r.mean() / r.std(ddof=1)) * np.sqrt(252) if r.std() > 0 else float("nan")
    peak = np.maximum.accumulate(eq)
    maxdd = float((eq / peak - 1.0).min())
    return {"cagr": float(cagr), "sharpe": float(sharpe), "maxdd": maxdd}


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def hac_t(x: np.ndarray) -> float:
    """Newey-West (HAC) one-sample t-stat of the mean against zero."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for kk in range(1, lags + 1):
        w = 1.0 - kk / (lags + 1.0)
        lrv += 2.0 * w * float(e[kk:] @ e[:-kk]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def summarize(returns: np.ndarray) -> dict:
    """Headline per-trade stats: count, win-rate, mean (bps), per-trade Sharpe, HAC t."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    return {
        "n": int(n),
        "win": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "sharpe": float(r.mean() / r.std(ddof=1)) if n > 1 and r.std() > 0 else float("nan"),
        "t": hac_t(r),
    }


def scrambled_atr_placebo(bars: pd.DataFrame, horizon: int, n: int = ATR_N, m: float = ATR_MULT,
                          n_draws: int = 1000, seed: int = 479) -> dict:
    """Placebo: rebuild the chandelier trail from a permuted ATR series, killing the geometry.

    Keeps the *marginal* distribution of ATR (same set of widths) but shuffles which width sits
    on which bar, so the trailing stop becomes volatility-blind nonsense. Returns the share of
    placebo runs whose mean chandelier-entry forward return **beats** the real one — the honest
    "is the ATR trail's structure adding anything?" p-value, plus the observed mean.
    """
    obs = float(np.mean(forward_returns(bars["close"], chandelier_entries(bars, n=n, m=m), horizon)))
    a_real = atr(bars, n=n).to_numpy()
    finite = np.isfinite(a_real)
    if finite.sum() < n + 5:
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    roll_hh = pd.Series(high, index=bars.index).rolling(n).max().shift(1).to_numpy()
    idx = bars.index
    nbar = close.size
    beats = 0
    valid = 0
    pool = a_real[finite].copy()
    for _ in range(n_draws):
        a = a_real.copy()
        a[finite] = rng.permutation(pool)
        pos = np.zeros(nbar, dtype=int)
        in_long = False
        peak = np.nan
        for i in range(nbar):
            if np.isnan(a[i]):
                continue
            if not in_long:
                if not np.isnan(roll_hh[i]) and close[i] > roll_hh[i]:
                    in_long = True
                    peak = high[i]
            else:
                peak = max(peak, high[i])
                if close[i] < peak - m * a[i]:
                    in_long = False
                    peak = np.nan
            pos[i] = 1 if in_long else 0
        flips = (pos == 1) & (np.concatenate([[0], pos[:-1]]) == 0)
        ent = idx[flips]
        rr = forward_returns(bars["close"], ent, horizon)
        if rr.size == 0:
            continue
        valid += 1
        if rr.mean() >= obs:
            beats += 1
    p = (beats + 1) / (valid + 1) if valid else float("nan")
    return {"obs": obs, "p_value": float(p), "n_draws": valid}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(bars: pd.DataFrame, n: int = ATR_N, m: float = ATR_MULT,
                   cost_bps: float = 1.0, random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: chandelier entry vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the chandelier-entry summary (gross + net), the
    drift-matched random-entry baseline, and the entry-minus-random delta; plus the equity-curve
    comparison (chandelier-managed long vs buy-and-hold) for the thesis axis.
    """
    close = bars["close"]
    ent = chandelier_entries(bars, n=n, m=m)
    res = {"n_entries": int(len(ent)), "by_h": {}, "equity": strategy_equity(bars, n=n, m=m, cost_bps=cost_bps)}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(close, max(len(ent), 50), n=n, seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
