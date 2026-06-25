"""TD Sequential (DeMark 9-13) as a falsifiable mechanical rule — Study 476.

Tom DeMark's **TD Sequential** is a two-stage exhaustion counter:

* **TD Buy Setup.** A run of **nine consecutive closes**, each *strictly below* the close
  **four bars earlier**. The ninth bar completes the setup ("9") — the first exhaustion signal.
* **TD Buy Countdown.** Once a buy setup completes, a countdown begins: each bar whose close is
  *at or below the low two bars earlier* increments the count; the **thirteenth** such bar
  completes the countdown ("13") — the deep exhaustion signal.

The folklore (DeMark's own teaching, echoed across technical-analysis sites and trading desks):
*a completed buy setup/countdown marks the exhaustion of the down-move* — sellers are spent, a
high-probability reversal **up** is at hand, so you go long.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Setup-9** — long entry fires on the bar that completes a 9-bar buy setup (close[t] <
   close[t-4] for nine consecutive bars). Read on the close of *t*; entered at the **next**
   close (one documented lag).
2. **Countdown-13** — long entry fires on the bar that completes a 13-count buy countdown
   (DeMark's deeper exhaustion). Same one-bar entry lag.
3. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold) that captures
   the tape's drift, and (b) a **scrambled-lookback placebo** that rebuilds the setup using a
   permuted set of comparison offsets (not the canonical 4-bar / 2-bar geometry), destroying
   DeMark's specific count structure while keeping the same number of comparisons and the price
   marginal — the honest "is the 4-bar/2-bar count doing anything?" null.

No look-ahead: every rung of the setup/countdown uses only closes/lows at or before *t*; the
signal is read on the close of *t* and the position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)

SETUP_LEN = 9       # TD Buy Setup: 9 consecutive lower closes vs close-4
SETUP_LOOKBACK = 4  # "close vs close four bars earlier"
COUNTDOWN_LEN = 13  # TD Buy Countdown completes at 13
COUNTDOWN_LB = 2    # countdown rung: close <= low two bars earlier


# --------------------------------------------------------------------------- #
# TD Setup / Countdown detection
# --------------------------------------------------------------------------- #
def buy_setup_count(close: pd.Series, lookback: int = SETUP_LOOKBACK) -> np.ndarray:
    """Running TD Buy-Setup count: consecutive bars whose close < close ``lookback`` bars ago.

    Returns an int array aligned to ``close``; the value at bar ``t`` is the length of the
    current down-streak ending at ``t`` (reset to 0 whenever close[t] >= close[t-lookback]).
    A completed setup is a bar where this count reaches :data:`SETUP_LEN`.
    """
    p = close.to_numpy(dtype=float)
    n = p.size
    cnt = np.zeros(n, dtype=int)
    for t in range(lookback, n):
        if p[t] < p[t - lookback]:
            cnt[t] = cnt[t - 1] + 1
        else:
            cnt[t] = 0
    return cnt


def buy_setup_entries(close: pd.Series, lookback: int = SETUP_LOOKBACK,
                      setup_len: int = SETUP_LEN) -> pd.DatetimeIndex:
    """Bars that complete a TD Buy Setup (count reaches ``setup_len``) — the '9' signal.

    Entry is executed at the next close by :func:`forward_returns`. Only the bar that *hits*
    the setup length is kept (the completion, not every further lower close).
    """
    cnt = buy_setup_count(close, lookback=lookback)
    hit = cnt == setup_len
    return close.index[hit]


def buy_countdown_entries(close: pd.Series, low: pd.Series,
                          lookback: int = SETUP_LOOKBACK, setup_len: int = SETUP_LEN,
                          cd_lb: int = COUNTDOWN_LB,
                          cd_len: int = COUNTDOWN_LEN) -> pd.DatetimeIndex:
    """Bars that complete a TD Buy Countdown ('13') after a completed buy setup.

    After each setup-9 completion, a countdown runs: every bar whose close is at/below the low
    ``cd_lb`` bars earlier increments the count; the ``cd_len``-th such bar completes the
    countdown. A fresh setup completion restarts the countdown (standard DeMark recycling).
    All comparisons use only data at or before the bar — no look-ahead.
    """
    c = close.to_numpy(dtype=float)
    lo = low.to_numpy(dtype=float)
    n = c.size
    setup_cnt = buy_setup_count(close, lookback=lookback)
    out_pos = []
    counting = False
    count = 0
    for t in range(n):
        if setup_cnt[t] == setup_len:      # a fresh buy setup completes -> (re)start countdown
            counting = True
            count = 0
        if counting and t >= cd_lb:
            if c[t] <= lo[t - cd_lb]:
                count += 1
                if count == cd_len:
                    out_pos.append(t)
                    counting = False       # countdown complete; wait for next setup
    return close.index[out_pos]


def random_entries(close: pd.Series, n: int, warmup: int = 20, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = close.index[warmup:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    return pd.DatetimeIndex(sorted(chosen))


# --------------------------------------------------------------------------- #
# Forward-return engine
# --------------------------------------------------------------------------- #
def forward_returns(close: pd.Series, entries, horizon: int, cost_bps: float = 0.0) -> np.ndarray:
    """Forward ``horizon``-day return for each entry, entered at the *next* close (one lag).

    ``cost_bps`` is a one-way cost (charged twice: in + out) subtracted from each trade's
    return. Trades whose window overruns the tape are dropped.
    """
    pos = {d: i for i, d in enumerate(close.index)}
    p = close.to_numpy(dtype=float)
    n = p.size
    out = []
    for d in entries:
        i = pos.get(d)
        if i is None or i + 1 + horizon >= n:
            continue
        e = i + 1                      # enter at next close
        r = p[e + horizon] / p[e] - 1.0
        out.append(r - 2.0 * cost_bps * 1e-4)
    return np.asarray(out, dtype=float)


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


def scrambled_lookback_placebo(close: pd.Series, horizon: int,
                               n_draws: int = 1000, seed: int = 476,
                               setup_len: int = SETUP_LEN) -> dict:
    """Placebo: rebuild the buy setup with a *permuted* comparison-offset, not the 4-bar rule.

    DeMark's setup compares close[t] to close[t-4] specifically; the count's whole identity is
    that 4-bar geometry. We replace the fixed 4-bar lookback with a random offset drawn from a
    plausible range (2..8) per draw, so the comparison still fires on roughly the same kind of
    down-streaks (same price marginal, same setup length) but with a *different* lookback
    geometry. Returns the share of placebo runs whose mean setup-9 forward return **beats** the
    real one — the honest "is the specific 4-bar count load-bearing?" p-value, plus the observed
    mean.
    """
    obs = float(np.mean(forward_returns(close, buy_setup_entries(close, setup_len=setup_len),
                                        horizon)))
    if not np.isfinite(obs):
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    offsets = [2, 3, 5, 6, 7, 8]  # all plausible offsets EXCEPT the canonical 4
    beats = 0
    valid = 0
    for _ in range(n_draws):
        lb = int(rng.choice(offsets))
        ent = buy_setup_entries(close, lookback=lb, setup_len=setup_len)
        rr = forward_returns(close, ent, horizon)
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
def run_experiment(bars: pd.DataFrame, cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: setup-9 vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the setup-9 summary (gross + net), the drift-matched
    random-entry baseline, the touch-minus-random delta, and the countdown-13 summary.
    """
    close = bars["close"]
    low = bars["low"] if "low" in bars else close
    ent = buy_setup_entries(close)
    cd = buy_countdown_entries(close, low)
    res = {"n_entries": int(len(ent)), "n_countdown": int(len(cd)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(close, max(len(ent), 50), seed=random_seed), h))
        cds = summarize(forward_returns(close, cd, h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd, "countdown": cds,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
