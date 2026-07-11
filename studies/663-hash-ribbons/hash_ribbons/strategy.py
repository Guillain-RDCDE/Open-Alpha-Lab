"""Strategy + inference for Study 663 — Hash-Ribbons.

The claim (Capriole Investments / Charles Edwards, 2019): watch the 30-day SMA of Bitcoin's
network hashrate against its 60-day SMA. A "capitulation" is a stretch where the 30-day SMA
sinks below the 60-day (miners going offline, unprofitable rigs shutting down); the "buy
signal" fires the day the 30-day SMA claws back above the 60-day — capitulation ending, the
weakest/most-levered miners already forced to sell their BTC and switch off, right as the
network's economics stabilize. Historically only a **handful** of these signals have ever
fired — this study measures exactly how many, and what happened next.

Measurements:

* **Signal detection** — SMA(30)/SMA(60) crossovers on the (interpolated) daily hashrate,
  filtered to genuine capitulations: the 30-day SMA must sit below the 60-day for at least
  three weeks *and* the hashrate itself must have fallen at least 8% peak-to-trough during
  that stretch (a magnitude filter that separates real miner capitulation from single-digit
  noise blips — consistent with Capriole's own "ribbons visibly compressed" visual criterion).
* **Forward returns after each signal** vs the unconditional forward-return distribution over
  the whole BTC tape, at 30/90/180/365-day horizons: Welch *t* (honestly noted as barely
  interpretable at n≈4) plus a random-date placebo (draw the same number of random dates,
  thousands of times, see how often a random calendar does as well).
* **Timer vs buy-and-hold** — go long BTC for a fixed 180-day window after each signal
  (Hash-Ribbons has no sell rule in the folklore; a fixed hold is the one honest way to build
  a comparable exposure backtest), cash the rest of the time, net of one-way costs × 2 per
  episode; compared against continuous buy-and-hold over the same window.
* **Synthetic positive control** — a deterministic seeded BTC-like return path with the same
  tiny number of signal windows and a TUNABLE planted post-signal drift, to prove the
  detector is unbiased (it must not fire on a null world, and must recover a planted effect).

The decisive number is the real-tape placebo p-value; the honest headline is the sample size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365   # BTC trades every calendar day


# --------------------------------------------------------------------------- #
# Ribbon + signal detection
# --------------------------------------------------------------------------- #
def ribbon(hr_daily: pd.Series, short: int = 30, long: int = 60) -> pd.DataFrame:
    """SMA(short)/SMA(long) of the daily hashrate path, plus the below/above state."""
    df = pd.DataFrame(index=hr_daily.index)
    df["hashrate"] = hr_daily
    df["sma_short"] = hr_daily.rolling(short).mean()
    df["sma_long"] = hr_daily.rolling(long).mean()
    df["below"] = (df["sma_short"] < df["sma_long"]).fillna(False)
    return df


def capitulation_signals(hr_daily: pd.Series, short: int = 30, long: int = 60,
                         min_capitulation_days: int = 21,
                         min_decline_pct: float = 8.0) -> pd.DataFrame:
    """Genuine Hash-Ribbons buy signals: SMA(short) crossing back above SMA(long).

    A raw crossover is kept only if, immediately before it, SMA(short) sat below SMA(long)
    for at least ``min_capitulation_days`` AND the raw hashrate itself fell at least
    ``min_decline_pct`` percent peak-to-trough over that stretch — the magnitude filter that
    turns "any old wobble" into "actual miner capitulation". Returns one row per surviving
    signal: the crossover date, the capitulation window and its peak-to-trough decline.
    """
    df = ribbon(hr_daily, short, long)
    below = df["below"].values
    sma_s, sma_l = df["sma_short"].values, df["sma_long"].values
    idx = df.index
    cross = (sma_s > sma_l) & (np.r_[False, sma_s[:-1] <= sma_l[:-1]])
    cross = np.where(pd.isna(sma_s) | pd.isna(sma_l), False, cross)

    rows = []
    for pos in np.flatnonzero(cross):
        i = pos - 1
        n = 0
        while i >= 0 and below[i]:
            n += 1
            i -= 1
        window = hr_daily.iloc[i + 1: pos]
        if len(window) == 0 or n < min_capitulation_days:
            continue
        peak, trough = float(window.max()), float(window.min())
        decline_pct = (trough / peak - 1.0) * 100.0
        if abs(decline_pct) < min_decline_pct:
            continue
        rows.append({
            "signal_date": idx[pos], "capitulation_days": n,
            "window_start": idx[i + 1], "window_end": idx[pos - 1],
            "decline_pct": decline_pct,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference primitives
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


def _forward_returns_all(logp: np.ndarray, horizon: int) -> np.ndarray:
    """Forward log-to-simple return at every valid start position, for a given horizon."""
    n = len(logp)
    pos = np.arange(0, n - horizon)
    return np.exp(logp[pos + horizon] - logp[pos]) - 1.0


def _entry_positions(index: pd.DatetimeIndex, dates: pd.DatetimeIndex) -> np.ndarray:
    """Map each signal date to its position on ``index`` (searchsorted; next trading day)."""
    return np.asarray([int(index.searchsorted(d)) for d in dates])


# --------------------------------------------------------------------------- #
# Forward-return event study — the headline
# --------------------------------------------------------------------------- #
def forward_return_stats(btc: pd.Series, signal_dates: pd.DatetimeIndex,
                         horizons: tuple[int, ...] = (30, 90, 180, 365)) -> pd.DataFrame:
    """Mean forward BTC return after each signal vs the unconditional distribution.

    Welch t (n≈4 — a formal statistic, but barely interpretable at this sample size, said
    loudly on the front card) and a 20-seed × 1,000-draw random-date placebo p-value
    (right-tailed: share of random-date draws whose mean forward return is >= observed).
    """
    logp = np.log(btc.values)
    pos = _entry_positions(btc.index, signal_dates)
    rows = []
    for h in horizons:
        valid = pos[pos + h < len(logp)]
        obs = np.exp(logp[valid + h] - logp[valid]) - 1.0
        all_fwd = _forward_returns_all(logp, h)
        t = welch_t(obs, all_fwd)

        rng_means = []
        for s in range(20):
            rng = np.random.default_rng(663 + s)
            all_pos = np.arange(0, len(logp) - h)
            for _ in range(1_000):
                pick = rng.choice(all_pos, size=len(valid), replace=False)
                rng_means.append(all_fwd[pick].mean())
        rng_means = np.asarray(rng_means)
        p = float((rng_means >= obs.mean()).mean())

        rows.append({
            "horizon": h, "n": len(valid),
            "signal_mean_pct": float(obs.mean()) * 100,
            "signal_median_pct": float(np.median(obs)) * 100,
            "uncond_mean_pct": float(all_fwd.mean()) * 100,
            "uncond_sd_pct": float(all_fwd.std(ddof=1)) * 100,
            "welch_t": t,
            "placebo_mean_pct": float(rng_means.mean()) * 100,
            "placebo_p": p,
            "n_draws": len(rng_means),
        })
    return pd.DataFrame(rows).set_index("horizon")


# --------------------------------------------------------------------------- #
# Timer vs buy-and-hold
# --------------------------------------------------------------------------- #
def timer_backtest(btc: pd.Series, signal_dates: pd.DatetimeIndex,
                   hold_days: int = 180, cost_bps: float = 10.0) -> dict:
    """Long BTC for ``hold_days`` trading days after each signal (one-day entry lag), else
    cash at 0%; costs charged one-way × NAV on every entry and every exit.

    Overlapping hold windows (two signals close together) merge into one continuous exposure
    episode — no double-counting. Compared against continuous buy-and-hold over the *same*
    [first signal date, tape end] window (the only window the strategy could have acted in).
    """
    ret = btc.pct_change().dropna()
    exposed = pd.Series(False, index=ret.index)
    for d in signal_dates:
        entry_pos = int(ret.index.searchsorted(d)) + 1     # one-day execution lag
        if entry_pos >= len(ret):
            continue
        exit_pos = min(entry_pos + hold_days, len(ret) - 1)
        exposed.iloc[entry_pos: exit_pos + 1] = True

    trans = exposed.astype(int).diff().fillna(0)
    n_entries, n_exits = int((trans == 1).sum()), int((trans == -1).sum())

    strat_ret = ret.where(exposed, 0.0).copy()
    cost = cost_bps / 1e4
    strat_ret[trans == 1] -= cost
    strat_ret[trans == -1] -= cost

    start = signal_dates.min()
    s_idx = int(strat_ret.index.searchsorted(start))
    strat_w = (1.0 + strat_ret.iloc[s_idx:]).cumprod()
    bh_w = (1.0 + ret.iloc[s_idx:]).cumprod()
    strat_w, bh_w = strat_w / strat_w.iloc[0], bh_w / bh_w.iloc[0]
    n_years = (strat_w.index[-1] - strat_w.index[0]).days / 365.25

    strat_ret_w = strat_ret.loc[strat_w.index]
    bh_ret_w = ret.loc[bh_w.index]
    return {
        "n_signals": len(signal_dates), "n_entries": n_entries, "n_exits": n_exits,
        "years": n_years,
        "strat_total_pct": float(strat_w.iloc[-1] - 1) * 100,
        "strat_cagr_pct": float(strat_w.iloc[-1] ** (1 / n_years) - 1) * 100,
        "strat_sharpe": float(strat_ret_w.mean() / strat_ret_w.std() * np.sqrt(DAYS_PER_YEAR)),
        "exposure_pct": float(exposed.loc[strat_w.index].mean()) * 100,
        "bh_total_pct": float(bh_w.iloc[-1] - 1) * 100,
        "bh_cagr_pct": float(bh_w.iloc[-1] ** (1 / n_years) - 1) * 100,
        "bh_sharpe": float(bh_ret_w.mean() / bh_ret_w.std() * np.sqrt(DAYS_PER_YEAR)),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(r: np.ndarray, signal_idx: np.ndarray, horizon: int = 180) -> float:
    """Welch t of mean forward return after the planted signal windows vs unconditional."""
    logp = np.cumsum(np.log1p(r))
    n = len(r)
    valid = signal_idx[signal_idx + horizon < n]
    obs = np.exp(logp[valid + horizon] - logp[valid]) - 1.0
    all_fwd = _forward_returns_all(logp, horizon)
    return welch_t(obs, all_fwd)
