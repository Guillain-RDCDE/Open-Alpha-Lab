"""Strategy and analysis layer for Study 291 (Doge-Tweets).

The claim under test: Elon Musk's Dogecoin tweets move DOGE-USD. We test this two
ways, both honestly:

  1. **Event study.** Estimate a market model DOGE_t = a + b * BTC_t + e_t on the
     non-event days (so BTC absorbs the broad crypto move), then measure the
     *abnormal* return (residual) of DOGE on and around each tweet day. We report
     the average abnormal return (AAR) on the event day and the cumulative
     abnormal return (CAR) over a [-1, +k] window. Significance is via a
     permutation test (shuffle event dates across the calendar) and a t-stat on
     the per-event abnormal returns -- which is the relevant >=2 hurdle.

  2. **A tradable strategy.** Buy DOGE at the *close of the tweet day* (a one-day
     execution lag -- you cannot trade the tweet instant at daily resolution) and
     hold for ``hold_days``. Apply one-way transaction costs on each entry and
     exit, charged against NAV. Compare gross and net to a BTC buy-and-hold over
     the same windows. Crypto is price-only = total-return (no dividend).

The honesty spine:
- A curated FAMOUS-tweet list is a hindsight-selected sample that biases the
  event study UPWARD; named on the Signal axis.
- BTC is the market proxy so we never credit a DOGE move that was just the whole
  crypto complex moving.
- The one-day lag is the difference between an academic event study (which often
  uses the contemporaneous day, unachievable in practice) and a real trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Market model: abnormal DOGE returns net of BTC
# ---------------------------------------------------------------------------
def market_model_residuals(
    rets: pd.DataFrame,
    event_mask: pd.Series,
) -> tuple[pd.Series, float, float]:
    """Fit DOGE_t = a + b * BTC_t on NON-event days; return abnormal returns for all days.

    Using only non-event days to estimate (a, b) avoids letting the events
    contaminate the benchmark. Returns ``(abnormal, alpha, beta)`` where
    ``abnormal`` is a Series aligned to ``rets`` of DOGE minus its market-model
    fit.
    """
    doge = rets["doge_ret"].to_numpy(dtype=float)
    btc = rets["btc_ret"].to_numpy(dtype=float)
    em = event_mask.to_numpy(dtype=bool)

    est = ~em
    if est.sum() < 3:
        est = np.ones_like(em, dtype=bool)
    X = np.column_stack([np.ones(est.sum()), btc[est]])
    coef, *_ = np.linalg.lstsq(X, doge[est], rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    fitted = alpha + beta * btc
    abnormal = doge - fitted
    return pd.Series(abnormal, index=rets.index, name="abnormal"), alpha, beta


# ---------------------------------------------------------------------------
# Event study: AAR / CAR around tweet days
# ---------------------------------------------------------------------------
def event_window_returns(
    abnormal: pd.Series,
    event_dates: pd.Series | pd.DatetimeIndex,
    pre: int = 1,
    post: int = 3,
) -> pd.DataFrame:
    """Stack the abnormal returns in a [-pre, +post] window around each event.

    Events are aligned to the abnormal-return index by position: each event date
    is mapped to the first available index position on or after the date (the
    one-day-lag-safe alignment). Returns a DataFrame, rows = events, columns =
    relative offsets (e.g. -1, 0, 1, 2, 3). Events too close to the edges are
    dropped.
    """
    idx = abnormal.index
    offsets = list(range(-pre, post + 1))
    rows = []
    kept = []
    for d in pd.to_datetime(pd.Index(event_dates)):
        pos_arr = np.searchsorted(idx.values, np.datetime64(d), side="left")
        pos = int(pos_arr)
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
    post: int = 3,
    n_permutations: int = 10_000,
    seed: int = 291,
) -> dict:
    """Full event-study summary: AAR on day 0, CAR over [-pre, +post], t-stat, perm p.

    The permutation test reassigns the same NUMBER of event days to random
    positions in the sample (keeping the event-day count fixed) and recomputes the
    mean day-0 abnormal return, giving a null distribution that controls for the
    fat-tailed DOGE return distribution far better than a normal-theory t-test.

    Returns a flat dict with day-0 AAR, its t-stat, CAR, permutation p-value, and
    the per-offset mean abnormal returns.
    """
    abnormal, alpha, beta = market_model_residuals(
        rets, _event_mask_from_dates(rets.index, event_dates)
    )
    win = event_window_returns(abnormal, event_dates, pre=pre, post=post)
    n_ev = len(win)

    day0 = win[0].to_numpy(dtype=float) if 0 in win.columns else np.array([])
    aar0 = float(day0.mean()) if n_ev > 0 else float("nan")
    sd0 = float(day0.std(ddof=1)) if n_ev > 1 else float("nan")
    t0 = float(aar0 / (sd0 / np.sqrt(n_ev))) if (n_ev > 1 and sd0 > 0) else float("nan")

    car_per_event = win.sum(axis=1).to_numpy(dtype=float)
    car_mean = float(car_per_event.mean()) if n_ev > 0 else float("nan")
    car_sd = float(car_per_event.std(ddof=1)) if n_ev > 1 else float("nan")
    car_t = float(car_mean / (car_sd / np.sqrt(n_ev))) if (n_ev > 1 and car_sd > 0) else float("nan")

    mean_by_offset = win.mean(axis=0)

    # Permutation test on day-0 AAR.
    abn = abnormal.to_numpy(dtype=float)
    n = len(abn)
    valid_lo, valid_hi = pre, n - post
    rng = np.random.default_rng(seed)
    perm_aar = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        pos = rng.integers(valid_lo, valid_hi, size=n_ev)
        perm_aar[i] = abn[pos].mean()
    # two-sided
    perm_p = float((np.abs(perm_aar) >= abs(aar0)).mean()) if not np.isnan(aar0) else float("nan")

    return {
        "n_events": int(n_ev),
        "alpha": round(alpha, 6),
        "beta": round(beta, 4),
        "aar_day0": round(aar0, 6),
        "aar_day0_pct": round(aar0 * 100, 3),
        "t_day0": round(t0, 3),
        "car": round(car_mean, 6),
        "car_pct": round(car_mean * 100, 3),
        "car_t": round(car_t, 3),
        "perm_p": round(perm_p, 4),
        "mean_by_offset": mean_by_offset,
        "perm_aar": perm_aar,
        "car_per_event": car_per_event,
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
# Tradable backtest: buy DOGE at tweet-day close, hold N days, with costs
# ---------------------------------------------------------------------------
def tweet_strategy_backtest(
    rets: pd.DataFrame,
    event_dates: pd.Series | pd.DatetimeIndex,
    hold_days: int = 3,
    cost_bps_oneway: float = 30.0,
    seed: int = 291,
) -> dict:
    """Buy DOGE at the close of the tweet day (one-day lag), hold ``hold_days``.

    Execution lag: a tweet on day t is only actionable at the day-t *close*, so
    the position earns returns from day t+1 through day t+hold_days. One-way
    transaction costs (``cost_bps_oneway``) are charged on entry and exit against
    NAV. Crypto DOGE is price-only = total-return.

    Returns gross and net annualized-equivalent per-trade stats plus the BTC
    buy-and-hold benchmark over the same windows.
    """
    idx = rets.index
    doge = rets["doge_ret"].to_numpy(dtype=float)
    btc = rets["btc_ret"].to_numpy(dtype=float)
    vals = idx.values

    trade_doge = []
    trade_btc = []
    for d in pd.to_datetime(pd.Index(event_dates)):
        pos = int(np.searchsorted(vals, np.datetime64(d), side="left"))
        # earn from pos+1 .. pos+hold_days (one-day lag: enter at close of pos)
        lo, hi = pos + 1, pos + hold_days
        if lo < 0 or hi >= len(idx):
            continue
        d_ret = float(np.prod(1.0 + doge[lo:hi + 1]) - 1.0)
        b_ret = float(np.prod(1.0 + btc[lo:hi + 1]) - 1.0)
        trade_doge.append(d_ret)
        trade_btc.append(b_ret)

    trade_doge = np.array(trade_doge)
    trade_btc = np.array(trade_btc)
    n_tr = len(trade_doge)

    cost = 2.0 * cost_bps_oneway * 1e-4  # round-trip on NAV
    net_doge = trade_doge - cost

    gross_mean = float(trade_doge.mean()) if n_tr else float("nan")
    net_mean = float(net_doge.mean()) if n_tr else float("nan")
    btc_mean = float(trade_btc.mean()) if n_tr else float("nan")
    gross_sd = float(trade_doge.std(ddof=1)) if n_tr > 1 else float("nan")
    net_t = float(net_mean / (gross_sd / np.sqrt(n_tr))) if (n_tr > 1 and gross_sd > 0) else float("nan")
    hit = float((net_doge > 0).mean()) if n_tr else float("nan")

    return {
        "n_trades": int(n_tr),
        "hold_days": hold_days,
        "cost_bps_oneway": cost_bps_oneway,
        "gross_mean_pct": round(gross_mean * 100, 3),
        "net_mean_pct": round(net_mean * 100, 3),
        "btc_bench_pct": round(btc_mean * 100, 3),
        "net_t": round(net_t, 3),
        "hit_rate": round(hit, 4),
        "excess_vs_btc_pct": round((net_mean - btc_mean) * 100, 3),
        "trade_doge": trade_doge,
        "trade_btc": trade_btc,
    }


# ---------------------------------------------------------------------------
# Directional split + summarize
# ---------------------------------------------------------------------------
def summarize(
    rets: pd.DataFrame,
    tweet_df: pd.DataFrame,
    pre: int = 1,
    post: int = 3,
    hold_days: int = 3,
    n_permutations: int = 10_000,
    seed: int = 291,
) -> dict:
    """One-stop summary dict mirroring docs/results.md and the R-dict in notebooks.

    Splits the event study by bullish (direction>0) vs all events, runs the
    tradable backtest, and returns a flat dict of rounded headline numbers.
    """
    all_dates = tweet_df["date"]
    bull_dates = tweet_df.loc[tweet_df["direction"] > 0, "date"]

    es_all = event_study_stats(rets, all_dates, pre=pre, post=post,
                               n_permutations=n_permutations, seed=seed)
    es_bull = event_study_stats(rets, bull_dates, pre=pre, post=post,
                                n_permutations=n_permutations, seed=seed)
    bt = tweet_strategy_backtest(rets, all_dates, hold_days=hold_days, seed=seed)

    return {
        "n_events": es_all["n_events"],
        "n_bull": es_bull["n_events"],
        "beta_btc": es_all["beta"],
        "aar_day0_pct": es_all["aar_day0_pct"],
        "t_day0": es_all["t_day0"],
        "car_pct": es_all["car_pct"],
        "car_t": es_all["car_t"],
        "perm_p": es_all["perm_p"],
        "bull_aar_day0_pct": es_bull["aar_day0_pct"],
        "bull_t_day0": es_bull["t_day0"],
        "bull_perm_p": es_bull["perm_p"],
        "bt_n_trades": bt["n_trades"],
        "bt_gross_pct": bt["gross_mean_pct"],
        "bt_net_pct": bt["net_mean_pct"],
        "bt_btc_pct": bt["btc_bench_pct"],
        "bt_net_t": bt["net_t"],
        "bt_hit": bt["hit_rate"],
        "bt_excess_pct": bt["excess_vs_btc_pct"],
    }
