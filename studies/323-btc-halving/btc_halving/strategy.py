"""Strategy and honest controls — Study 323 (BTC-Halving).

The folk thesis under test, formalised three ways:

1. **Cycle alignment (does the halving really print the top/bottom?).**
   ``cycle_extrema`` finds, for each completed cycle, how many days *after* the
   halving the cycle high and low actually landed. The lore says: bottom a bit
   *before* the halving, top ~12-18 months *after*. We measure the realised
   offsets and ask whether they cluster — or whether each cycle prints its
   extrema at a different phase (which would mean the "cycle" is hindsight).

2. **Phase-bucket forward returns.** ``phase_bucket_returns`` slices the cycle
   into quarters by halving phase and reports the mean daily return earned in
   each. The lore predicts the strongest returns in the post-halving "up" half
   and the worst in the pre-halving "down" half.

3. **A halving-timing rule vs buy-and-hold.** ``timing_rule`` goes long when the
   calendar says we are in the post-halving "bull" phase (phase in
   ``[lo, hi)``), else holds cash, and races the equity net-of-costs against
   **buy-and-hold BTC** — the only honest benchmark for a single trending asset.

**No look-ahead:** the cycle *phase* on day *t* is a calendar fact knowable years
ahead (the halving schedule is deterministic), so the *signal* needs no lag. We
still apply **one** documented execution-lag: a position decided from the close
of *t* earns the return of *t+1* (one ``shift``), and turnover is charged
**one-way × NAV** on every change in position. Long-only, so no borrow.

**Single-asset / tiny-n caveat (named on the Signal axis):** BTC is one
surviving asset that rose ~1000x, and Yahoo covers only the **2020 and 2024**
halvings cleanly — two cycles. Any "cycle" fit to two realisations is
catastrophically over-determined; the synthetic control proves only that the
*engine* can detect a planted cycle, never that the market has one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import HALVINGS, CYCLE_DAYS, cycle_phase, days_since_last_halving


# ---------------------------------------------------------------------------
# 1) Where do the cycle extrema actually fall relative to the halving?
# ---------------------------------------------------------------------------
def cycle_extrema(bars: pd.DataFrame, halvings=HALVINGS,
                  cycle_days: int = CYCLE_DAYS) -> pd.DataFrame:
    """For each halving, the day-offset of the post-halving high and low.

    A "cycle window" runs from a halving to ``cycle_days`` later (or to the end
    of the tape). Within that window we locate the price high and low and report
    how many days *after* the halving each occurred. The lore predicts the high
    clusters around +365..+550 days and the low near the cycle edges.

    Columns: ``halving, n_days_in_window, high_offset_d, low_offset_d,
    high_ret_x, low_ret_x`` (returns are multiples of the price *at* the
    halving). Halvings with no in-window data are dropped.
    """
    close = bars["close"]
    rows = []
    for h in sorted(halvings):
        start = h
        stop = h + pd.Timedelta(days=cycle_days)
        win = close[(close.index >= start) & (close.index < stop)]
        if win.empty:
            continue
        # price at (or just after) the halving — first available bar in window.
        base = float(win.iloc[0])
        hi_date = win.idxmax()
        lo_date = win.idxmin()
        rows.append({
            "halving": h,
            "n_days_in_window": int((win.index[-1] - h).days),
            "high_offset_d": int((hi_date - h).days),
            "low_offset_d": int((lo_date - h).days),
            "high_ret_x": float(win.max() / base),
            "low_ret_x": float(win.min() / base),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2) Forward returns by cycle phase bucket
# ---------------------------------------------------------------------------
def phase_bucket_returns(bars: pd.DataFrame, n_buckets: int = 4,
                         halvings=HALVINGS,
                         cycle_days: int = CYCLE_DAYS) -> pd.DataFrame:
    """Mean daily log-return within each phase bucket of the halving cycle.

    Bucket 0 is the quarter just after a halving, bucket ``n_buckets-1`` the
    quarter just before the next. Reports n, mean daily log-return (bps), and a
    Newey-West HAC *t* per bucket. Days before the first halving (NaN phase) are
    excluded.
    """
    close = bars["close"]
    logret = np.log(close).diff()
    phase = cycle_phase(close.index, halvings, cycle_days)
    bucket = np.floor(phase * n_buckets)
    df = pd.DataFrame({"ret": logret.to_numpy(), "bucket": bucket},
                      index=close.index).dropna()
    rows = []
    for b in range(n_buckets):
        r = df.loc[df["bucket"] == b, "ret"].to_numpy()
        r = r[np.isfinite(r)]
        rows.append({
            "bucket": b,
            "phase_lo": b / n_buckets,
            "phase_hi": (b + 1) / n_buckets,
            "n": int(r.size),
            "mean_bps": float(r.mean() * 1e4) if r.size else float("nan"),
            "tstat": hac_tstat(r),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3) Halving-timing rule vs buy-and-hold
# ---------------------------------------------------------------------------
def timing_rule(bars: pd.DataFrame, phase_lo: float = 0.0, phase_hi: float = 0.5,
                cost_bps: float = 10.0, halvings=HALVINGS,
                cycle_days: int = CYCLE_DAYS) -> pd.DataFrame:
    """Long-when-in-post-halving-window, else flat; one-bar fill lag, costs.

    The position for day *t* is decided by the *calendar* phase at *t-1*'s close
    (no look-ahead — the phase is known years ahead anyway) and earns day *t*'s
    return. Turnover is one-way × NAV: each flip flat<->long is charged
    ``cost_bps`` once. Returns a frame with strategy and buy-and-hold daily net
    log-returns and their cumulative equity.

    Default window ``[0.0, 0.5)`` = "the first half of each cycle after a halving"
    (the lore's bull phase).
    """
    close = bars["close"]
    logret = np.log(close).diff().fillna(0.0)
    phase = cycle_phase(close.index, halvings, cycle_days)
    in_window = (phase >= phase_lo) & (phase < phase_hi)
    raw_pos = pd.Series(np.where(np.isnan(phase), 0.0,
                                 in_window.astype(float)), index=close.index)
    pos = raw_pos.shift(1).fillna(0.0)  # one-bar execution lag (decide t-1, earn t)
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * (cost_bps * 1e-4)
    strat = pos * logret - cost
    out = pd.DataFrame({
        "pos": pos,
        "ret_bh": logret,
        "ret_strat": strat,
        "eq_bh": np.exp(logret.cumsum()),
        "eq_strat": np.exp(strat.cumsum()),
    }, index=close.index)
    return out


def summarize_timing(book: pd.DataFrame) -> dict:
    """Headline stats for a timing book vs buy-and-hold (daily log-returns)."""
    s = book["ret_strat"].to_numpy()
    b = book["ret_bh"].to_numpy()
    n = s.size
    ann = 365.0
    def cagr(r):
        return float(np.exp(r.sum() * ann / max(n, 1)) - 1.0)
    def sharpe(r):
        sd = r.std(ddof=1)
        return float(r.mean() / sd * np.sqrt(ann)) if sd > 0 else float("nan")
    excess = s - b  # strategy minus buy-and-hold, excess-vs-excess race
    return {
        "n_days": int(n),
        "cagr_strat": cagr(s),
        "cagr_bh": cagr(b),
        "sharpe_strat": sharpe(s),
        "sharpe_bh": sharpe(b),
        "excess_mean_bps": float(excess.mean() * 1e4),
        "excess_t": hac_tstat(excess),
        "time_in_market": float((book["pos"] > 0).mean()),
        "n_flips": int(book["pos"].diff().abs().fillna(0).gt(0).sum()),
    }


# ---------------------------------------------------------------------------
# Inference helpers — HAC t and circular block bootstrap CI
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray) -> float:
    """Newey-West HAC t-stat of the mean of ``x`` (no quantlab hard dependency)."""
    r = np.asarray(x, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n <= 5:
        return float("nan")
    mu = r.mean()
    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def block_bootstrap_ci(x: np.ndarray, block: int = 21, n_boot: int = 2000,
                       alpha: float = 0.05, seed: int = 323) -> tuple[float, float]:
    """Circular block-bootstrap CI for the mean of ``x`` (preserves clustering)."""
    r = np.asarray(x, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < block + 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[i] = r[idx][:n].mean()
    lo = float(np.quantile(means, alpha / 2) * 1e4)
    hi = float(np.quantile(means, 1 - alpha / 2) * 1e4)
    return (lo, hi)
