"""The analysis and its honest controls — Study 90 (Weekend / day-of-week effect).

The folk claim: *"Mondays are negative — the **weekend effect** — and Tuesday rebounds
('**turnaround Tuesday**'). Avoid Monday / buy Tuesday and you beat buy-and-hold."* We
measure the per-weekday mean daily return, test the two headline contrasts with
autocorrelation-robust inference, and pin a literal day-of-week timer against the thing it
claims to beat:

- **Per-weekday means** — the raw effect, one number per weekday, each with a HAC *t*-stat
  (the daily series is mildly autocorrelated, so a plain *t* overstates significance).
- **Monday-vs-rest and Tuesday-vs-rest contrasts** — the two claims stated as a difference
  of means, each with its own HAC *t*.
- **A pre-2000 vs post-2000 split, WITH a test of the *difference*** — the effect is famously
  documented in the pre-2000 record and is reported to have decayed or reversed since, so we
  test the *change* in the Monday (and Tuesday) effect across the split, not just each half.
- **A literal timer vs buy-and-hold** — "long only on Tuesdays" and "skip Mondays", net of
  costs. Day-of-week is **calendar-known**, so there is *no execution lag*: the rule
  "be long on Tuesday" is decidable the night before. Costs are charged one-way on every
  change in position; cash earns **0%** (a stated, conservative choice).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


# ---------------------------------------------------------------------------
# Returns helper
# ---------------------------------------------------------------------------
def daily_returns(close: pd.Series) -> pd.Series:
    """Simple daily returns (close-to-close), first bar dropped (no return defined)."""
    return close.pct_change().dropna()


# ---------------------------------------------------------------------------
# Inference: HAC t-stat on a mean (local, no quantlab dep)
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) t-stat for the mean of ``x`` (local, no quantlab dep)."""
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


def _hac_diff_tstat(a: np.ndarray, b: np.ndarray) -> float:
    """HAC t-stat for the difference in means of two *independent* groups.

    Both groups are autocorrelated within themselves; they are disjoint subsets of the
    daily tape (e.g. Mondays vs all other days, or pre- vs post-2000), so we combine their
    HAC standard errors in quadrature. Used for the weekday contrasts and the sub-period
    difference-of-effects test.
    """
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float)
    b = b[np.isfinite(b)]
    if a.size <= 5 or b.size <= 5:
        return float("nan")
    se_a = _hac_se(a)
    se_b = _hac_se(b)
    se = np.sqrt(se_a**2 + se_b**2)
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def _hac_se(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) standard error of the mean of ``x``."""
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


# ---------------------------------------------------------------------------
# Per-weekday means + contrasts
# ---------------------------------------------------------------------------
def weekday_means(close: pd.Series) -> pd.DataFrame:
    """Per-weekday mean daily return (in bps), count, and HAC *t*-stat of the mean.

    Returns a frame indexed by weekday label (Mon..Fri) with columns ``mean_bps``,
    ``n``, ``hac_t``. The mean is the average close-to-close return on bars *falling on*
    that weekday.
    """
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
    """The mean-difference contrast of one ``weekday`` vs every other weekday.

    ``weekday`` is Monday=0 .. Friday=4. Returns the in-group mean (bps), the rest mean
    (bps), their difference (bps), and a HAC *t*-stat of the difference. This is the claim
    ("Mondays are negative", "Tuesday rebounds") stated as a testable difference of means.
    """
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


# ---------------------------------------------------------------------------
# Sub-period split — pre vs post a cut date, WITH a test of the difference
# ---------------------------------------------------------------------------
def subperiod_effect(close: pd.Series, weekday: int, cut: str = "2000-01-01") -> dict:
    """The ``weekday``-vs-rest effect computed *separately* before and after ``cut``,
    plus a HAC test of whether the effect *changed* across the split.

    The famous story is that the weekday effects were strong pre-2000 and decayed/reversed
    afterward. Reporting two sub-period means is not enough — we test the *difference of the
    differences* (the post-cut contrast minus the pre-cut contrast) so a claim of decay
    carries its own uncertainty. The split date is the conventional, pre-registered one
    (the turn of the millennium), not snooped.

    Returns the pre/post contrast (bps), their change, and a HAC *t* on the change.
    """
    ret = daily_returns(close)
    cut_ts = pd.Timestamp(cut)
    pre = ret[ret.index < cut_ts]
    post = ret[ret.index >= cut_ts]

    def _contrast_series(r: pd.Series) -> np.ndarray:
        """The per-bar contrast series: in-group bars minus the rest-group mean.

        Centring the rest group lets us treat the contrast as the mean of a single series
        on the in-group bars, so its HAC SE is well defined. (Equivalent point estimate to
        the two-group difference of means.)"""
        dow = r.index.dayofweek
        return r[dow == weekday].to_numpy(), r[dow != weekday].to_numpy()

    pre_in, pre_rest = _contrast_series(pre)
    post_in, post_rest = _contrast_series(post)
    pre_diff = (pre_in.mean() - pre_rest.mean()) * 1e4
    post_diff = (post_in.mean() - post_rest.mean()) * 1e4

    # Test of the *change* in the effect: build the two contrast distributions (in-group
    # demeaned by the rest-group mean, per period) and HAC-difference them.
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


# ---------------------------------------------------------------------------
# Literal day-of-week timers
# ---------------------------------------------------------------------------
def weekday_position(close: pd.Series, long_days: set[int]) -> pd.Series:
    """Long-or-flat position that is **1** on bars whose weekday is in ``long_days``, else 0.

    Day-of-week is calendar-known, so there is **no execution lag**: the position for a bar
    is decided by that bar's own weekday (knowable the night before). Indexed like ``close``.
    """
    dow = pd.Series(close.index.dayofweek, index=close.index)
    return dow.isin(long_days).astype(float)


def tuesday_only_position(close: pd.Series) -> pd.Series:
    """'Buy Tuesday': long only on Tuesdays (weekday 1), flat otherwise."""
    return weekday_position(close, {1})


def skip_monday_position(close: pd.Series) -> pd.Series:
    """'Avoid Monday': long every weekday except Monday (weekday 0)."""
    return weekday_position(close, {1, 2, 3, 4})


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def backtest(close: pd.Series, position: pd.Series, cost_bps: float = 1.0) -> dict:
    """Run a long-or-flat position over ``close`` and return net stats + the daily series.

    ``cost_bps`` is charged (one-way) on every change in position. Cash earns 0%. Returns a
    dict with the net daily return series and headline stats (CAGR, vol, Sharpe, max
    drawdown, switches, time-in-market). No execution lag: day-of-week is calendar-known.
    """
    ret = close.pct_change().fillna(0.0)
    pos = position.reindex(close.index).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs())
    net = pos * ret - turn * cost_bps * 1e-4
    return _stats(net, pos, turn)


def buy_and_hold(close: pd.Series) -> dict:
    """Buy-and-hold the tape: position 1 every day, no costs after entry."""
    ret = close.pct_change().fillna(0.0)
    pos = pd.Series(1.0, index=close.index)
    return _stats(ret, pos, pd.Series(0.0, index=close.index))


def _stats(net: pd.Series, pos: pd.Series, turn: pd.Series) -> dict:
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
