"""Strategy and analysis layer for Study 294 (Coinbase-Rank).

The claim under test: when the Coinbase app spikes toward #1 on the App Store, it
marks a retail-euphoria *top* -- crypto is supposed to weaken afterwards. We test
this two ways, both honestly:

  1. **Forward event study.** Estimate a market model BTC_t = a + b * ETH_t on the
     non-event days (so ETH absorbs the broad crypto move), then measure the
     *abnormal* forward return (residual) of BTC over a [+1, +post] window after
     each rank-spike day. The folk top claim predicts a NEGATIVE cumulative
     abnormal return (CAR) after the spike. Significance is via a permutation test
     (shuffle event dates across the calendar) and a t-stat on the per-event
     forward abnormal returns -- the relevant |t| >= 2 hurdle.

  2. **A tradable strategy.** SHORT BTC at the *close of the spike day* (a one-day
     execution lag -- you cannot trade the chart-position instant at daily
     resolution) and hold for ``hold_days``. Apply one-way transaction costs on
     entry and exit AND a borrow fee against NAV (shorts pay borrow). Compare gross
     and net to BTC buy-and-hold over the same windows. Crypto is price-only =
     total-return (no dividend).

The honesty spine:
- A curated FAMOUS-spike list is a hindsight-selected sample that biases a
  contrarian top study UPWARD (we remember the spikes that preceded a top); named
  on the Signal axis.
- ETH is the market proxy so we never credit a BTC move that was just the whole
  crypto complex moving.
- The one-day lag is the difference between an academic event study (which can use
  the contemporaneous day) and a real trade.
- Shorting pays borrow; the net backtest charges it explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats  # noqa: F401  (kept for parity / extensions)


# ---------------------------------------------------------------------------
# Market model: abnormal BTC returns net of ETH
# ---------------------------------------------------------------------------
def market_model_residuals(
    rets: pd.DataFrame,
    event_mask: pd.Series,
) -> tuple[pd.Series, float, float]:
    """Fit BTC_t = a + b * ETH_t on NON-event days; return abnormal returns for all days.

    Using only non-event days to estimate (a, b) avoids letting the events
    contaminate the benchmark. Returns ``(abnormal, alpha, beta)`` where
    ``abnormal`` is a Series aligned to ``rets`` of BTC minus its market-model fit.
    """
    btc = rets["btc_ret"].to_numpy(dtype=float)
    eth = rets["eth_ret"].to_numpy(dtype=float)
    em = event_mask.to_numpy(dtype=bool)

    est = ~em
    if est.sum() < 3:
        est = np.ones_like(em, dtype=bool)
    X = np.column_stack([np.ones(est.sum()), eth[est]])
    coef, *_ = np.linalg.lstsq(X, btc[est], rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    fitted = alpha + beta * eth
    abnormal = btc - fitted
    return pd.Series(abnormal, index=rets.index, name="abnormal"), alpha, beta


# ---------------------------------------------------------------------------
# Forward event study: abnormal returns in the window after each spike
# ---------------------------------------------------------------------------
def event_window_returns(
    abnormal: pd.Series,
    event_dates: pd.Series | pd.DatetimeIndex,
    pre: int = 1,
    post: int = 10,
) -> pd.DataFrame:
    """Stack abnormal returns in a [-pre, +post] window around each event.

    Events are aligned to the abnormal-return index by position: each event date
    is mapped to the first available index position on or after the date (the
    one-day-lag-safe alignment). Returns a DataFrame, rows = events, columns =
    relative offsets (e.g. -1, 0, 1, ..., post). Events too close to the edges are
    dropped.
    """
    idx = abnormal.index
    offsets = list(range(-pre, post + 1))
    rows = []
    kept = []
    for d in pd.to_datetime(pd.Index(event_dates)):
        pos = int(np.searchsorted(idx.values, np.datetime64(d), side="left"))
        if pos - pre < 0 or pos + post >= len(idx):
            continue
        row = [float(abnormal.iloc[pos + o]) for o in offsets]
        rows.append(row)
        kept.append(d)
    out = pd.DataFrame(rows, columns=offsets, index=pd.to_datetime(kept))
    return out


def event_study_stats(
    rets: pd.DataFrame,
    event_dates: pd.Series | pd.DatetimeIndex,
    pre: int = 1,
    post: int = 10,
    fwd: int = 5,
    n_permutations: int = 10_000,
    seed: int = 294,
) -> dict:
    """Forward event-study summary: forward CAR over [+1, +fwd], t-stat, perm p.

    The folk claim is a contrarian TOP: crypto should fall AFTER the spike. So the
    headline statistic is the cumulative abnormal return over the ``fwd`` trading
    days following the event (offsets +1..+fwd); a NEGATIVE value supports the
    claim.

    The permutation test reassigns the same NUMBER of event days to random
    positions in the sample (keeping the event-day count fixed) and recomputes the
    mean forward CAR, giving a null distribution that controls for the fat-tailed
    BTC return distribution far better than a normal-theory t-test.

    Returns a flat dict with the forward CAR, its t-stat, permutation p-value, and
    the per-offset mean abnormal returns.
    """
    abnormal, alpha, beta = market_model_residuals(
        rets, _event_mask_from_dates(rets.index, event_dates)
    )
    win = event_window_returns(abnormal, event_dates, pre=pre, post=post)
    n_ev = len(win)

    fwd_cols = [o for o in win.columns if 1 <= o <= fwd]
    fwd_car_per_event = win[fwd_cols].sum(axis=1).to_numpy(dtype=float) if n_ev else np.array([])
    car_mean = float(fwd_car_per_event.mean()) if n_ev > 0 else float("nan")
    car_sd = float(fwd_car_per_event.std(ddof=1)) if n_ev > 1 else float("nan")
    car_t = float(car_mean / (car_sd / np.sqrt(n_ev))) if (n_ev > 1 and car_sd > 0) else float("nan")

    # Day-0 abnormal return (the same-day reaction, for context).
    day0 = win[0].to_numpy(dtype=float) if 0 in win.columns else np.array([])
    aar0 = float(day0.mean()) if n_ev > 0 else float("nan")

    mean_by_offset = win.mean(axis=0)
    n_neg = int((fwd_car_per_event < 0).sum()) if n_ev else 0

    # Permutation test on the forward CAR.
    abn = abnormal.to_numpy(dtype=float)
    n = len(abn)
    valid_lo, valid_hi = pre, n - post
    # Precompute the cumulative forward window sum at each candidate position.
    rng = np.random.default_rng(seed)
    perm_car = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        pos = rng.integers(valid_lo, valid_hi, size=n_ev)
        vals = np.array([abn[p + 1:p + fwd + 1].sum() for p in pos])
        perm_car[i] = vals.mean()
    perm_p = float((np.abs(perm_car) >= abs(car_mean)).mean()) if not np.isnan(car_mean) else float("nan")

    return {
        "n_events": int(n_ev),
        "alpha": round(alpha, 6),
        "beta": round(beta, 4),
        "aar_day0": round(aar0, 6),
        "aar_day0_pct": round(aar0 * 100, 3),
        "fwd": fwd,
        "fwd_car": round(car_mean, 6),
        "fwd_car_pct": round(car_mean * 100, 3),
        "fwd_car_t": round(car_t, 3),
        "n_neg": n_neg,
        "perm_p": round(perm_p, 4),
        "mean_by_offset": mean_by_offset,
        "perm_car": perm_car,
        "fwd_car_per_event": fwd_car_per_event,
        "pre": pre,
        "post": post,
    }


def _event_mask_from_dates(idx: pd.DatetimeIndex, event_dates) -> pd.Series:
    """Boolean Series over ``idx``, True at the position on/after each event date."""
    mask = np.zeros(len(idx), dtype=bool)
    vals = idx.values
    for d in pd.to_datetime(pd.Index(event_dates)):
        pos = int(np.searchsorted(vals, np.datetime64(d), side="left"))
        if 0 <= pos < len(idx):
            mask[pos] = True
    return pd.Series(mask, index=idx, name="is_event")


# ---------------------------------------------------------------------------
# Tradable backtest: SHORT BTC at spike-day close, hold N days, with costs+borrow
# ---------------------------------------------------------------------------
def rank_short_backtest(
    rets: pd.DataFrame,
    event_dates: pd.Series | pd.DatetimeIndex,
    hold_days: int = 5,
    cost_bps_oneway: float = 20.0,
    borrow_bps_per_day: float = 5.0,
    seed: int = 294,
) -> dict:
    """SHORT BTC at the close of the spike day (one-day lag), hold ``hold_days``.

    The folk top claim => short crypto after a Coinbase rank spike. Execution lag:
    a spike observed on day t is only actionable at the day-t *close*, so the short
    earns the NEGATIVE of BTC's return from day t+1 through day t+hold_days. One-way
    transaction costs (``cost_bps_oneway``) are charged on entry and exit, and a
    per-day borrow fee (``borrow_bps_per_day``) is charged against NAV for the
    holding period -- shorts pay borrow. Crypto BTC is price-only = total-return.

    A positive net per-trade return means the short MADE money (BTC fell). Returns
    gross/net per-trade stats plus the BTC long buy-and-hold benchmark over the
    same windows (so the reader sees what a buy-and-holder earned instead).
    """
    idx = rets.index
    btc = rets["btc_ret"].to_numpy(dtype=float)
    vals = idx.values

    short_pnl = []
    long_btc = []
    for d in pd.to_datetime(pd.Index(event_dates)):
        pos = int(np.searchsorted(vals, np.datetime64(d), side="left"))
        lo, hi = pos + 1, pos + hold_days
        if lo < 0 or hi >= len(idx):
            continue
        b_ret = float(np.prod(1.0 + btc[lo:hi + 1]) - 1.0)
        short_pnl.append(-b_ret)   # short profits when BTC falls
        long_btc.append(b_ret)

    short_pnl = np.array(short_pnl)
    long_btc = np.array(long_btc)
    n_tr = len(short_pnl)

    trade_cost = 2.0 * cost_bps_oneway * 1e-4          # round-trip on NAV
    borrow_cost = borrow_bps_per_day * 1e-4 * hold_days  # borrow over the hold
    net_short = short_pnl - trade_cost - borrow_cost

    gross_mean = float(short_pnl.mean()) if n_tr else float("nan")
    net_mean = float(net_short.mean()) if n_tr else float("nan")
    btc_mean = float(long_btc.mean()) if n_tr else float("nan")
    gross_sd = float(short_pnl.std(ddof=1)) if n_tr > 1 else float("nan")
    net_t = float(net_mean / (gross_sd / np.sqrt(n_tr))) if (n_tr > 1 and gross_sd > 0) else float("nan")
    hit = float((net_short > 0).mean()) if n_tr else float("nan")

    return {
        "n_trades": int(n_tr),
        "hold_days": hold_days,
        "cost_bps_oneway": cost_bps_oneway,
        "borrow_bps_per_day": borrow_bps_per_day,
        "gross_mean_pct": round(gross_mean * 100, 3),
        "net_mean_pct": round(net_mean * 100, 3),
        "btc_bench_pct": round(btc_mean * 100, 3),
        "net_t": round(net_t, 3),
        "hit_rate": round(hit, 4),
        "excess_vs_btc_pct": round((net_mean - btc_mean) * 100, 3),
        "short_pnl": short_pnl,
        "long_btc": long_btc,
    }


# ---------------------------------------------------------------------------
# Summarize -- compact dict for notebooks / results.md
# ---------------------------------------------------------------------------
def summarize(
    rets: pd.DataFrame,
    spike_df: pd.DataFrame,
    pre: int = 1,
    post: int = 10,
    fwd: int = 5,
    hold_days: int = 5,
    n_permutations: int = 10_000,
    seed: int = 294,
) -> dict:
    """One-stop summary dict mirroring docs/results.md and the R-dict in notebooks.

    Runs the forward event study and the lagged short backtest, returning a flat
    dict of rounded headline numbers.
    """
    dates = spike_df["date"]
    es = event_study_stats(rets, dates, pre=pre, post=post, fwd=fwd,
                           n_permutations=n_permutations, seed=seed)
    bt = rank_short_backtest(rets, dates, hold_days=hold_days, seed=seed)

    return {
        "n_events": es["n_events"],
        "beta_eth": es["beta"],
        "aar_day0_pct": es["aar_day0_pct"],
        "fwd": es["fwd"],
        "fwd_car_pct": es["fwd_car_pct"],
        "fwd_car_t": es["fwd_car_t"],
        "n_neg": es["n_neg"],
        "perm_p": es["perm_p"],
        "bt_n_trades": bt["n_trades"],
        "bt_gross_pct": bt["gross_mean_pct"],
        "bt_net_pct": bt["net_mean_pct"],
        "bt_btc_pct": bt["btc_bench_pct"],
        "bt_net_t": bt["net_t"],
        "bt_hit": bt["hit_rate"],
        "bt_excess_pct": bt["excess_vs_btc_pct"],
    }
