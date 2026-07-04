"""Strategy + inference for Study 594 — Leverage Rotation (TQQQ + 200SMA).

THE RULE (as retail runs it): each day, if **QQQ's close is above its own 200-day SMA**,
hold **TQQQ** (3x daily-reset Nasdaq-100); otherwise hold **T-bills**. Signal on today's
close, position effective on the NEXT trading day's return — exactly ONE execution lag,
documented here and nowhere re-lagged.

Inference, the desk's shared spirit:

* **HAC (Newey-West) t** on the daily return DIFFERENCE (rotation minus benchmark) —
  daily strategy returns are autocorrelated and heteroskedastic, the naive t flatters.
* **Exposure-matched random-timer baseline** — the rotation spends only ~70% of days in
  the market; any un-levered comparison must separate *timing skill* from *exposure*.
  We shuffle the rotation's own in/out RUN LENGTHS (preserving both the exposure
  fraction and the switch count exactly), rebuild the strategy per seed, and average
  the Welch t of (real rotation vs random timer daily returns) over >= 20 seeds.
  Single-seed baselines are banned desk-wide.
* **Regime conditioning** — Welch t of next-day QQQ return (mean channel) and of
  next-day squared return (vol channel) above vs below the SMA: WHY the rule works.
* **Costs** one-way x NAV traded on every switch (a switch trades the full NAV once).
* **Sharpe races are excess-vs-excess** (both legs minus the same T-bill accrual).

All functions are pure numpy/pandas, deterministic given the seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Signal + backtest
# --------------------------------------------------------------------------- #
def sma_signal(close: pd.Series, window: int = 200) -> pd.Series:
    """True where close > its ``window``-day SMA. NaN until the SMA exists."""
    sma = close.rolling(window).mean()
    sig = (close > sma).astype(float)
    return sig.where(sma.notna())


def rotation_returns(asset_ret: pd.Series, cash_ret: pd.Series, signal: pd.Series,
                     cost_bps: float = 0.0) -> tuple[pd.Series, int]:
    """Daily net returns of the rotation + the switch count.

    ``signal`` is evaluated on day t's close; the position holds day t+1's return
    (the single execution lag). Each switch trades the full NAV once and pays
    ``cost_bps`` one-way x NAV.
    """
    pos = signal.shift(1)                      # THE one execution lag
    pos = pos.reindex(asset_ret.index).dropna().astype(bool)
    idx = pos.index
    a = asset_ret.reindex(idx)
    c = cash_ret.reindex(idx).ffill().fillna(0.0)
    ret = np.where(pos.values, a.values, c.values)
    switch = pos.values[1:] != pos.values[:-1]
    n_switch = int(switch.sum())
    cost = np.zeros(len(idx))
    cost[1:][switch] = cost_bps * 1e-4         # one-way x full NAV per switch
    return pd.Series(ret - cost, index=idx, name="rotation"), n_switch


def positions(signal: pd.Series, index: pd.Index) -> pd.Series:
    """The lagged boolean position aligned to a return index (for reuse/shuffles)."""
    pos = signal.shift(1).reindex(index).dropna().astype(bool)
    return pos


# --------------------------------------------------------------------------- #
# Performance stats
# --------------------------------------------------------------------------- #
def perf_stats(ret: pd.Series, rf: pd.Series) -> dict:
    """CAGR, annualised vol, EXCESS Sharpe (vs the same T-bill leg), max drawdown."""
    r = ret.dropna()
    rf_al = rf.reindex(r.index).ffill().fillna(0.0)
    n = len(r)
    yrs = n / TRADING_DAYS
    nav = np.exp(np.log1p(r).cumsum())
    cagr = float(nav.iloc[-1] ** (1.0 / yrs) - 1.0)
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    ex = r - rf_al
    sharpe = float(ex.mean() / ex.std() * np.sqrt(TRADING_DAYS)) if ex.std() > 0 else np.nan
    dd = float((nav / nav.cummax() - 1.0).min())
    return {"cagr_pct": cagr * 100.0, "vol_pct": vol * 100.0,
            "sharpe": sharpe, "maxdd_pct": dd * 100.0, "years": yrs, "n": n,
            "final_mult": float(nav.iloc[-1])}


def hac_t(x: pd.Series | np.ndarray, lags: int | None = None) -> dict:
    """Newey-West (Bartlett) t of the mean of ``x`` — the desk's robust daily t."""
    r = np.asarray(x, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    mu = r.mean()
    e = r - mu
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * (float(e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(lrv, 1e-300) / n)
    return {"mean_bps": mu * 1e4, "t": mu / se, "lags": lags, "n": n}


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch two-sample t of mean(a) - mean(b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return float((a.mean() - b.mean()) / np.sqrt(va + vb))


# --------------------------------------------------------------------------- #
# Exposure-matched random-timer baseline (>= 20 seeds, averaged Welch t)
# --------------------------------------------------------------------------- #
def _runs(pos: np.ndarray) -> tuple[list[int], list[int], bool]:
    """Decompose a boolean position path into (in-run lengths, out-run lengths, starts_in)."""
    change = np.flatnonzero(pos[1:] != pos[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(pos)]))
    lengths = np.diff(bounds)
    starts_in = bool(pos[0])
    ins, outs = [], []
    state = starts_in
    for ln in lengths:
        (ins if state else outs).append(int(ln))
        state = not state
    return ins, outs, starts_in


def shuffled_position(pos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random timer matched on exposure AND switch count: permute the real position's
    own in-run and out-run lengths (separately) and re-interleave them."""
    ins, outs, starts_in = _runs(pos)
    ins = list(rng.permutation(ins))
    outs = list(rng.permutation(outs))
    out = np.empty(len(pos), dtype=bool)
    i = 0
    state = starts_in
    while i < len(pos):
        seq = ins if state else outs
        ln = seq.pop() if seq else len(pos) - i
        out[i:i + ln] = state
        i += ln
        state = not state
    return out


def random_timer_test(asset_ret: pd.Series, cash_ret: pd.Series, pos: pd.Series,
                      cost_bps: float = 0.0, n_seeds: int = 40,
                      seed0: int = 594) -> dict:
    """The real rotation vs >= 20 exposure-and-switch-matched random timers.

    Per seed: rebuild the strategy on a shuffled position path (same exposure, same
    switch count, same costs), then Welch t of (real daily returns vs random daily
    returns). Reports the seed-AVERAGED Welch t (the desk's guard against a lucky
    single seed), the mean random CAGR/Sharpe, and the fraction of seeds the real
    rotation beats on Sharpe.
    """
    idx = pos.index
    a = asset_ret.reindex(idx).values
    c = cash_ret.reindex(idx).ffill().fillna(0.0).values
    p = pos.values.astype(bool)

    def strat(p_arr: np.ndarray) -> np.ndarray:
        ret = np.where(p_arr, a, c)
        sw = np.zeros(len(p_arr))
        sw[1:][p_arr[1:] != p_arr[:-1]] = cost_bps * 1e-4
        return ret - sw

    real = strat(p)
    real_sharpe = float((real - c).mean() / (real - c).std() * np.sqrt(TRADING_DAYS))
    ts, cagrs, sharpes = [], [], []
    yrs = len(idx) / TRADING_DAYS
    for s in range(n_seeds):
        rng = np.random.default_rng(seed0 + s)
        rp = shuffled_position(p, rng)
        rr = strat(rp)
        ts.append(welch_t(real, rr))
        cagrs.append(float(np.exp(np.log1p(rr).sum() / yrs) - 1.0) * 100.0)
        ex = rr - c
        sharpes.append(float(ex.mean() / ex.std() * np.sqrt(TRADING_DAYS)))
    ts = np.asarray(ts)
    sharpes = np.asarray(sharpes)
    return {"avg_welch_t": float(ts.mean()), "std_welch_t": float(ts.std()),
            "n_seeds": n_seeds, "real_sharpe": real_sharpe,
            "rand_sharpe_mean": float(sharpes.mean()),
            "rand_cagr_mean_pct": float(np.mean(cagrs)),
            "beat_frac": float((real_sharpe > sharpes).mean())}


# --------------------------------------------------------------------------- #
# Regime conditioning — WHY the SMA works (mean channel vs vol channel)
# --------------------------------------------------------------------------- #
def sma_conditioning(qqq_ret: pd.Series, signal: pd.Series) -> dict:
    """Next-day QQQ return above vs below the 200SMA (one lag, same as the trade).

    Welch t on the mean (the return channel) and on the squared return (the volatility
    channel). The vol channel is the well-documented one; the mean channel is what a
    CAGR edge would need.
    """
    pos = signal.shift(1).reindex(qqq_ret.index).dropna().astype(bool)
    r = qqq_ret.reindex(pos.index).values
    above, below = r[pos.values], r[~pos.values]
    return {
        "n_above": int(len(above)), "n_below": int(len(below)),
        "mean_above_bps": float(above.mean() * 1e4),
        "mean_below_bps": float(below.mean() * 1e4),
        "welch_t_mean": welch_t(above, below),
        "vol_above_pct": float(above.std() * np.sqrt(TRADING_DAYS) * 100.0),
        "vol_below_pct": float(below.std() * np.sqrt(TRADING_DAYS) * 100.0),
        "welch_t_var": welch_t(below ** 2, above ** 2),
    }


# --------------------------------------------------------------------------- #
# Sub-periods + the 2000 cohort
# --------------------------------------------------------------------------- #
SUBPERIODS = [
    ("dot-com crash 2000-02", "2000-03-24", "2002-10-09"),
    ("GFC 2007-09", "2007-10-31", "2009-03-09"),
    ("COVID crash 2020", "2020-02-19", "2020-04-30"),
    ("2022 bear", "2022-01-01", "2022-12-31"),
]


def window_stats(ret: pd.Series, start: str, end: str) -> dict:
    """Total return + max drawdown of a return series inside [start, end]."""
    r = ret.loc[start:end].dropna()
    if len(r) == 0:
        return {"total_pct": np.nan, "maxdd_pct": np.nan, "n": 0}
    nav = np.exp(np.log1p(r).cumsum())
    return {"total_pct": float((nav.iloc[-1] - 1.0) * 100.0),
            "maxdd_pct": float((nav / nav.cummax() - 1.0).min() * 100.0),
            "n": len(r)}


def cohort_2000(rot: pd.Series, qqq: pd.Series, sim3x: pd.Series,
                start: str = "2000-03-24", stake: float = 10_000.0) -> dict:
    """The third-axis question: $10k at the 2000-03-24 QQQ peak — who survives?

    For each leg (rotation, QQQ B&H, 3x B&H): trough value, max drawdown, the year the
    stake is first made whole again (if ever), and the final value at the as-of.
    """
    out = {}
    for name, r in [("rotation", rot), ("qqq_bh", qqq), ("tqqq_bh", sim3x)]:
        rr = r.loc[start:].dropna()
        nav = stake * np.exp(np.log1p(rr).cumsum())
        trough = float(nav.min())
        dd = float((nav / nav.cummax() - 1.0).min() * 100.0)
        whole = nav[nav >= stake]
        # first recovery AFTER the initial drawdown began
        rec = None
        under = nav < stake
        if under.any():
            first_under = under.idxmax()
            after = nav.loc[first_under:]
            back = after[after >= stake]
            rec = str(back.index[0].date()) if len(back) else None
        out[name] = {"trough_value": trough, "maxdd_pct": dd,
                     "recovered_on": rec, "final_value": float(nav.iloc[-1]),
                     "end": str(nav.index[-1].date())}
    return out
