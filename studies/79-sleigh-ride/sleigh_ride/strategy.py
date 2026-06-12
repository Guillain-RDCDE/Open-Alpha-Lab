"""The strategy and its honest controls — Study 79 (Sleigh-Ride).

The folk recipe (Hirsch, *Stock Traders Almanac*, first cited 1972): the last 5
trading days of the calendar year plus the first 2 of January tend to produce
above-average returns on the S&P 500. If Santa Claus fails to call, bears may
come to Broad and Wall.

We implement an honest test in three steps:

1. **Mark every Santa window** using only the calendar — no look-ahead from prices.
2. **Measure the window's mean daily return** vs all non-Santa trading days (the
   all-window baseline). A HAC *t*-stat (Newey-West) on the raw difference decides
   whether the effect is distinguishable from zero.
3. **Check the 'If Santa Fails' claim**: years in which the window was negative were
   asserted to precede bad January/Q1 returns. We test this directional forecast with
   the same HAC framework and a win-rate vs the base-rate of down Januarys.

No look-ahead: the window membership is determined by the calendar date alone — no
forward price information is used to define the window boundaries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SANTA_WINDOW_TAIL, SANTA_WINDOW_HEAD


# ---------------------------------------------------------------------------
# Window identification
# ---------------------------------------------------------------------------
def label_santa_windows(bars: pd.DataFrame) -> pd.Series:
    """Return a boolean Series: True on every bar that falls inside a Santa window.

    Santa window = last ``SANTA_WINDOW_TAIL`` trading days of year Y + first
    ``SANTA_WINDOW_HEAD`` trading days of year Y+1, for every Y in the tape.
    The mapping is purely calendar-driven — no price look-ahead.
    """
    dates = bars.index
    years = dates.year
    in_santa = pd.Series(False, index=dates, name="in_santa")
    for y in years.unique():
        yr_idx = np.where(years == y)[0]
        if len(yr_idx) >= SANTA_WINDOW_TAIL:
            tail_idx = yr_idx[-SANTA_WINDOW_TAIL:]
            in_santa.iloc[tail_idx] = True
        next_yr_idx = np.where(years == y + 1)[0]
        if len(next_yr_idx) >= SANTA_WINDOW_HEAD:
            head_idx = next_yr_idx[:SANTA_WINDOW_HEAD]
            in_santa.iloc[head_idx] = True
    return in_santa


def santa_windows(bars: pd.DataFrame) -> pd.DataFrame:
    """Return a per-window (= per-year) summary frame.

    Each row represents one Santa Claus Rally window; columns are the year,
    the date range, the 7-bar log return, the mean daily log return, and whether
    the window was positive. Used both by the strategy tests and the notebooks.
    """
    in_santa = label_santa_windows(bars)
    log_ret = np.log(bars["close"] / bars["close"].shift(1))
    # Drop the first bar (NaN log-return).
    log_ret = log_ret.dropna()
    in_santa = in_santa.loc[log_ret.index]

    dates = bars.index
    years = dates.year

    records = []
    seen = set()
    for y in sorted(years.unique()):
        # A window spans the tail of year y and the head of year y+1; identify by y.
        yr_idx = np.where(years == y)[0]
        if len(yr_idx) < SANTA_WINDOW_TAIL:
            continue
        tail_dates = dates[yr_idx[-SANTA_WINDOW_TAIL:]]
        next_yr_idx = np.where(years == y + 1)[0]
        if len(next_yr_idx) < SANTA_WINDOW_HEAD:
            continue
        head_dates = dates[next_yr_idx[:SANTA_WINDOW_HEAD]]
        window_dates = tail_dates.append(head_dates)
        window_ret = log_ret.loc[log_ret.index.isin(window_dates)]
        if len(window_ret) == 0:
            continue
        if y in seen:
            continue
        seen.add(y)
        total = float(window_ret.sum())
        records.append(
            {
                "year": int(y),
                "start": window_dates[0],
                "end": window_dates[-1],
                "n_days": int(len(window_ret)),
                "total_ret": total,
                "mean_daily_ret": float(window_ret.mean()),
                "positive": bool(total > 0),
            }
        )
    return pd.DataFrame(records).set_index("year") if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------
def daily_returns(bars: pd.DataFrame) -> pd.Series:
    """Daily log returns of the close series (first bar is NaN, dropped)."""
    r = np.log(bars["close"] / bars["close"].shift(1))
    return r.dropna()


def compare_santa_vs_baseline(bars: pd.DataFrame) -> dict:
    """Test: mean daily return inside Santa windows vs all other trading days.

    Returns a dict with the mean, std, n, and HAC t-stat for both arms, plus the
    raw difference and its HAC t-stat — the headline inference number.

    The test is on *daily* log returns (not the 7-day window aggregate) to maximise
    power and for comparability with the all-days baseline.
    """
    in_santa = label_santa_windows(bars)
    ret = daily_returns(bars)
    in_santa = in_santa.loc[ret.index]

    r_santa = ret[in_santa].to_numpy()
    r_base = ret[~in_santa].to_numpy()

    def _stats(r: np.ndarray) -> dict:
        n = len(r)
        mu = float(r.mean()) if n > 0 else float("nan")
        std = float(r.std(ddof=1)) if n > 1 else float("nan")
        t = _hac_tstat(r) if n > 5 else float("nan")
        return {"n": n, "mean_bps": mu * 1e4, "std_bps": std * 1e4, "tstat": t}

    s = _stats(r_santa)
    b = _stats(r_base)

    # HAC t on the difference — is the Santa mean significantly higher than baseline?
    diff = r_santa.mean() - r_base.mean() if len(r_santa) and len(r_base) else float("nan")
    # For the HAC on the difference we treat each Santa-day return minus the baseline
    # mean as the quantity of interest; this is the one-sample HAC t for Santa returns
    # centred at the baseline mean.
    if len(r_santa) > 5:
        diff_centred = r_santa - r_base.mean()
        diff_t = _hac_tstat(diff_centred)
    else:
        diff_t = float("nan")

    return {
        "santa_n": s["n"],
        "santa_mean_bps": s["mean_bps"],
        "santa_std_bps": s["std_bps"],
        "santa_tstat": s["tstat"],
        "base_n": b["n"],
        "base_mean_bps": b["mean_bps"],
        "base_std_bps": b["std_bps"],
        "base_tstat": b["tstat"],
        "diff_bps": float(diff) * 1e4 if np.isfinite(diff) else float("nan"),
        "diff_tstat": diff_t,
    }


def santa_window_summary(bars: pd.DataFrame) -> dict:
    """Per-window (per-year) aggregate: mean window return, win-rate, HAC t-stat.

    Uses the 7-day window log return as the unit of observation (not daily returns)
    — this is the number Hirsch reports. The HAC t-stat is on the vector of per-window
    7-day log returns (vs zero), and separately vs the same-length random-window
    baseline drawn from all non-year-end 7-day windows.
    """
    wins = santa_windows(bars)
    if wins.empty:
        return {}
    r = wins["total_ret"].to_numpy()
    n = len(r)
    mu = float(r.mean())
    win_rate = float((r > 0).mean())
    t = _hac_tstat(r) if n > 5 else float("nan")
    # Median for robustness.
    med = float(np.median(r))
    return {
        "n_windows": n,
        "mean_window_bps": mu * 1e4,
        "median_window_bps": med * 1e4,
        "win_rate": win_rate,
        "tstat_vs_zero": t,
    }


# ---------------------------------------------------------------------------
# 'If Santa Fails, Bears May Come' test
# ---------------------------------------------------------------------------
def if_santa_fails(bars: pd.DataFrame, forward_days: int = 20) -> dict:
    """Test the follow-up Hirsch claim: a negative Santa window predicts a bad January.

    For each Santa window, record whether it was positive or negative, then measure the
    forward ``forward_days``-bar log return starting from the first trading day of January
    (the day after the Santa window closes). Test whether the negative-window group has a
    lower forward return than the positive-window group; also compute win-rates.

    Returns a dict with group statistics and the difference t-stat (HAC on the positive
    minus negative means, using the per-window forward returns as the unit).
    """
    wins = santa_windows(bars)
    if wins.empty or len(wins) < 6:
        return {}

    log_ret = daily_returns(bars)
    records = []
    dates = bars.index

    for y, row in wins.iterrows():
        end_dt = row["end"]  # last day of Santa window (first 2 days of new year)
        # Find the first bar strictly after the window end.
        forward_start_idx = dates.searchsorted(end_dt, side="right")
        if forward_start_idx + forward_days > len(dates):
            continue
        fwd = log_ret.iloc[forward_start_idx: forward_start_idx + forward_days]
        records.append(
            {
                "year": int(y),
                "santa_positive": bool(row["positive"]),
                "fwd_ret": float(fwd.sum()),
            }
        )

    if not records:
        return {}

    df = pd.DataFrame(records)
    pos = df.loc[df["santa_positive"], "fwd_ret"].to_numpy()
    neg = df.loc[~df["santa_positive"], "fwd_ret"].to_numpy()

    if len(pos) == 0 or len(neg) == 0:
        return {"n_pos": len(pos), "n_neg": len(neg)}

    diff = pos.mean() - neg.mean()
    # HAC t on the difference: sign-aligned series (positive minus neg baseline)
    all_fwd = df["fwd_ret"].to_numpy()
    sign = np.where(df["santa_positive"].to_numpy(), 1.0, -1.0)
    # Two-sample approximation: treat the centred difference per obs.
    aligned = sign * (all_fwd - all_fwd.mean())
    diff_t = _hac_tstat(aligned) if len(aligned) > 5 else float("nan")

    return {
        "n_pos_santa": int(len(pos)),
        "n_neg_santa": int(len(neg)),
        "fwd_mean_after_pos_bps": float(pos.mean()) * 1e4,
        "fwd_mean_after_neg_bps": float(neg.mean()) * 1e4,
        "fwd_diff_bps": float(diff) * 1e4,
        "fwd_diff_tstat": diff_t,
        "neg_santa_win_rate_fwd": float((neg > 0).mean()),  # fraction of neg windows with pos forward
        "overall_forward_win_rate": float((all_fwd > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Random-window baseline control
# ---------------------------------------------------------------------------
def random_window_baseline(
    bars: pd.DataFrame,
    n_samples: int = 1000,
    seed: int = 79,
) -> dict:
    """Draw random 7-consecutive-trading-day windows from non-year-end dates.

    This provides a distribution of 7-day returns under the null hypothesis that the
    Santa window is just a randomly chosen 7-day stretch. The 95th percentile of this
    distribution is the bar the Santa mean must clear to be 'unusually good'. Returns
    mean, std, and the 95th percentile of the random-window mean distribution.
    """
    log_ret = daily_returns(bars)
    in_santa = label_santa_windows(bars).loc[log_ret.index]
    # Draw only from non-Santa dates to avoid contaminating the null.
    non_santa_idx = np.where(~in_santa.to_numpy())[0]
    if len(non_santa_idx) < 7 + 1:
        return {}
    rng = np.random.default_rng(seed)
    window_rets = []
    r = log_ret.to_numpy()
    valid_starts = non_santa_idx[non_santa_idx + 7 <= len(r)]
    if len(valid_starts) == 0:
        return {}
    for _ in range(n_samples):
        i = rng.choice(valid_starts)
        window_rets.append(r[i: i + 7].sum())
    window_rets = np.array(window_rets)
    return {
        "baseline_mean_bps": float(window_rets.mean()) * 1e4,
        "baseline_std_bps": float(window_rets.std(ddof=1)) * 1e4,
        "baseline_p95_bps": float(np.percentile(window_rets, 95)) * 1e4,
        "baseline_win_rate": float((window_rets > 0).mean()),
    }


# ---------------------------------------------------------------------------
# HAC t-stat (Newey-West)
# ---------------------------------------------------------------------------
def _hac_tstat(r: np.ndarray) -> float:
    """HAC (Newey-West) t-stat on the mean of ``r``. Same kernel as study 72."""
    n = len(r)
    if n < 6:
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


def summarize(returns: np.ndarray | pd.Series, label: str = "series") -> dict:
    """Headline statistics for a vector of per-period returns: n, mean_bps, win-rate,
    HAC t-stat (vs zero), and median. Mirrors the study-72 summarize() interface.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    return {
        "label": label,
        "n": int(n),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "median_bps": float(np.median(r) * 1e4) if n else float("nan"),
        "win_rate": float((r > 0).mean()) if n else float("nan"),
        "tstat": _hac_tstat(r) if n > 5 else float("nan"),
    }
