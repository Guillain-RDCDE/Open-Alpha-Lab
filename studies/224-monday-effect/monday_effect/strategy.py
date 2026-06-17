"""Analysis engine for Study 224 (Monday Effect / full day-of-week close-to-close).

The folk claim: Monday is the worst day of the week -- the Monday Effect -- and Friday or
Tuesday rebound. Avoid Monday / buy the best day and you beat buy-and-hold.

French (1980, JFE) documented significantly negative Monday close-to-close returns on
1953--1977 S&P data. This desk tests the same claim on the full SPY era (1993-2026) using:

- Per-weekday means with HAC (Newey-West) t-stats.
- Monday-vs-rest contrast -- the headline claim as a difference of means.
- Pre-2000 vs post-2000 split WITH a test of the change.
- Literal day-of-week timers vs buy-and-hold net of 1 bp/switch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def hac_tstat(x, lags=None):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 5:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def _hac_se(x, lags=None):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 5:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    e = x - x.mean()
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    return float(np.sqrt(max(lrv, 0.0) / n))


def _hac_diff_tstat(a, b):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float)
    b = b[np.isfinite(b)]
    if a.size <= 5 or b.size <= 5:
        return float("nan")
    se = np.sqrt(_hac_se(a) ** 2 + _hac_se(b) ** 2)
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def weekday_means(close: pd.Series) -> pd.DataFrame:
    ret = daily_returns(close)
    dow = ret.index.dayofweek
    rows = {}
    for d in range(5):
        x = ret[dow == d].to_numpy()
        rows[WEEKDAYS[d]] = {
            "mean_bps": float(x.mean() * 1e4),
            "n": int(x.size),
            "hac_t": hac_tstat(x),
        }
    return pd.DataFrame(rows).T[["mean_bps", "n", "hac_t"]]


def contrast(close: pd.Series, weekday: int) -> dict:
    ret = daily_returns(close)
    dow = ret.index.dayofweek
    inn = ret[dow == weekday].to_numpy()
    rest = ret[dow != weekday].to_numpy()
    return {
        "weekday": WEEKDAYS[weekday],
        "in_bps": float(inn.mean() * 1e4),
        "rest_bps": float(rest.mean() * 1e4),
        "diff_bps": float((inn.mean() - rest.mean()) * 1e4),
        "hac_t": _hac_diff_tstat(inn, rest),
        "n_in": int(inn.size),
    }


def subperiod_effect(close: pd.Series, weekday: int, cut: str = "2000-01-01") -> dict:
    ret = daily_returns(close)
    cut_ts = pd.Timestamp(cut)
    pre = ret[ret.index < cut_ts]
    post = ret[ret.index >= cut_ts]

    def _contrast_arrays(r):
        dow = r.index.dayofweek
        return r[dow == weekday].to_numpy(), r[dow != weekday].to_numpy()

    pre_in, pre_rest = _contrast_arrays(pre)
    post_in, post_rest = _contrast_arrays(post)
    pre_diff = (pre_in.mean() - pre_rest.mean()) * 1e4
    post_diff = (post_in.mean() - post_rest.mean()) * 1e4

    pre_c = pre_in - pre_rest.mean()
    post_c = post_in - post_rest.mean()
    t_change = _hac_diff_tstat(post_c, pre_c)

    return {
        "weekday": WEEKDAYS[weekday],
        "cut": cut,
        "pre_diff_bps": float(pre_diff),
        "post_diff_bps": float(post_diff),
        "change_bps": float(post_diff - pre_diff),
        "hac_t_change": float(t_change),
        "n_pre": int(pre_in.size),
        "n_post": int(post_in.size),
    }


def weekday_position(close: pd.Series, long_days: set) -> pd.Series:
    dow = pd.Series(close.index.dayofweek, index=close.index)
    return dow.isin(long_days).astype(float)


def skip_monday_position(close: pd.Series) -> pd.Series:
    return weekday_position(close, {1, 2, 3, 4})


def monday_only_position(close: pd.Series) -> pd.Series:
    return weekday_position(close, {0})


def best_day_position(close: pd.Series, weekday: int) -> pd.Series:
    return weekday_position(close, {weekday})


def _max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def backtest(close: pd.Series, position: pd.Series, cost_bps: float = 1.0) -> dict:
    ret = close.pct_change().fillna(0.0)
    pos = position.reindex(close.index).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs())
    net = pos * ret - turn * cost_bps * 1e-4
    return _stats(net, pos, turn)


def buy_and_hold(close: pd.Series) -> dict:
    ret = close.pct_change().fillna(0.0)
    pos = pd.Series(1.0, index=close.index)
    return _stats(ret, pos, pd.Series(0.0, index=close.index))


def _stats(net, pos, turn):
    net = net.astype(float)
    equity = (1.0 + net).cumprod()
    n = len(net)
    years = n / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = (
        float(net.mean() / net.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if net.std() > 0
        else float("nan")
    )
    return {
        "net": net,
        "equity": equity,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": _max_drawdown(equity.to_numpy()),
        "switches": int((turn > 1e-9).sum()),
        "time_in_market": float((pos > 0).mean()),
        "final": float(equity.iloc[-1]),
    }
