"""The strategy and its honest controls — Study 89 (Turn-of-the-Month).

The folk claim: *"almost all of the market's gains happen in a narrow window around the
turn of the month — the last trading day of the old month plus the first three of the
new one. Be invested only in that window and you capture the market with a fraction of
the risk."* (Ariel 1987; Lakonishok & Smidt 1988; McConnell & Xu 2008.)

We implement:

- **TOM day-labelling** by trading-day-of-month position (``tom_mask``). The window is
  *calendar-known* — knowable in advance — so there is **no execution lag**: a calendar
  rule is not a forecast and needs no ``shift`` (per house rules).
- **TOM vs rest engine** (``window_contrast``): mean daily return inside vs outside the
  window, the share of the total cumulative log-return earned inside the window, and a
  HAC *t*-stat on the *difference* of daily means.
- **TOM-only timer** (``backtest`` with the TOM position): long while in the window, in
  cash (earning **0%**, a stated conservative choice) otherwise, charged a switch cost
  on every entry/exit. Pinned against buy-and-hold both on raw terms and — because the
  timer is in cash ~80% of the time — on a **per-day-invested** / Sharpe basis (normalise
  before you marvel).
- **Sub-period contrast** (``subperiod_contrast``): the TOM-minus-rest daily premium in
  an early vs late half, with a HAC *t* on each half and a block-bootstrap test of the
  *difference* between halves (decay test on a justified split).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# TOM day-labelling (calendar-known — no execution lag)
# ---------------------------------------------------------------------------
def tom_mask(index: pd.DatetimeIndex, last: int = 1, first: int = 3) -> pd.Series:
    """Boolean Series: True on the turn-of-month window.

    The window is the **last ``last`` trading day(s) of month t-1** through **trading day
    +``first`` of month t** — i.e. the canonical Lakonishok-Smidt ``[-1, +3]`` window when
    ``last=1, first=3``. It is built purely from each session's position within its
    calendar month, so it is knowable in advance: a calendar rule, not a forecast, hence
    no execution lag.
    """
    idx = pd.DatetimeIndex(index)
    s = pd.Series(np.arange(len(idx)), index=idx)
    ym = idx.to_period("M")
    tdom = s.groupby(ym).cumcount() + 1          # 1-based trading-day-of-month
    n_in_month = s.groupby(ym).transform("size")
    tdfe = n_in_month - tdom + 1                 # 1 == last trading day of the month
    mask = (tdom.to_numpy() <= first) | (tdfe.to_numpy() <= last)
    return pd.Series(mask, index=idx)


def trading_day_of_month(index: pd.DatetimeIndex) -> pd.Series:
    """1-based trading-day-of-month for each session (for the per-day bar chart)."""
    idx = pd.DatetimeIndex(index)
    s = pd.Series(np.arange(len(idx)), index=idx)
    return (s.groupby(idx.to_period("M")).cumcount() + 1).astype(int)


def signed_trading_day(index: pd.DatetimeIndex, around: int = 6) -> pd.Series:
    """A signed day index centred on the month boundary: ... -2, -1 (last days of t-1),
    +1, +2, +3 ... (first days of t). Days deeper than ``around`` inside the month map to
    NaN. Used only for the mean-return-by-day bar chart around the turn."""
    idx = pd.DatetimeIndex(index)
    s = pd.Series(np.arange(len(idx)), index=idx)
    ym = idx.to_period("M")
    tdom = (s.groupby(ym).cumcount() + 1).to_numpy()
    n_in_month = s.groupby(ym).transform("size").to_numpy()
    tdfe = n_in_month - tdom + 1
    out = np.full(len(idx), np.nan)
    # First `around` trading days -> +1..+around
    first_sel = tdom <= around
    out[first_sel] = tdom[first_sel]
    # Last `around` trading days -> -around..-1  (last day == -1)
    last_sel = tdfe <= around
    out[last_sel] = -tdfe[last_sel]
    return pd.Series(out, index=idx)


# ---------------------------------------------------------------------------
# Window contrast: TOM vs rest-of-month
# ---------------------------------------------------------------------------
def window_contrast(close: pd.Series, last: int = 1, first: int = 3) -> dict:
    """Compare TOM-window daily returns against the rest of the month.

    Returns a dict with the two daily means (bps), the HAC *t*-stat on the daily-return
    *difference* (TOM minus rest, via a dummy-regression residual series), the number of
    days in each bucket, the fraction of calendar (trading) days that are TOM, and the
    **share of the total cumulative log-return** earned inside the window.
    """
    ret = close.pct_change().dropna()
    mask = tom_mask(ret.index, last=last, first=first).reindex(ret.index).fillna(False)
    tom_ret = ret[mask.to_numpy()]
    rest_ret = ret[~mask.to_numpy()]

    logret = np.log1p(ret)
    tot_log = float(logret.sum())
    tom_log = float(logret[mask.to_numpy()].sum())

    # HAC t on the difference of means: regress demeaned-by-group? Simplest exact route is
    # a per-day series equal to (ret - mean_rest) on TOM days and 0 elsewhere is NOT the
    # difference of means. Instead build the contrast series used by a two-sample mean
    # difference under HAC: x_t = d_t*(r_t) with the standard dummy-coefficient SE.
    d = mask.to_numpy().astype(float)
    r = ret.to_numpy()
    # OLS slope of r on dummy d (intercept = rest mean, slope = TOM-minus-rest difference);
    # the slope's HAC SE gives the t-stat. We compute it via the residual influence series.
    dbar = d.mean()
    dc = d - dbar
    slope = float((dc @ (r - r.mean())) / (dc @ dc))
    intercept = float(r.mean() - slope * dbar)
    resid = r - (intercept + slope * d)
    # HAC SE of the slope: (sum dc^2)^-1 * S * (sum dc^2)^-1, S = HAC of (dc*resid).
    g = dc * resid
    S = _hac_lrv(g)
    sxx = float(dc @ dc)
    se_slope = np.sqrt(S * len(g) / (sxx ** 2)) if sxx > 0 else float("nan")
    t_diff = slope / se_slope if se_slope and se_slope > 0 else float("nan")

    return {
        "tom_mean_bps": float(tom_ret.mean() * 1e4),
        "rest_mean_bps": float(rest_ret.mean() * 1e4),
        "diff_bps": float((tom_ret.mean() - rest_ret.mean()) * 1e4),
        "t_diff": float(t_diff),
        "n_tom": int(len(tom_ret)),
        "n_rest": int(len(rest_ret)),
        "frac_tom_days": float(mask.mean()),
        "share_of_total_logret": float(tom_log / tot_log) if tot_log != 0 else float("nan"),
        "tom_logret": tom_log,
        "total_logret": tot_log,
    }


# ---------------------------------------------------------------------------
# TOM-only timer vs buy-and-hold
# ---------------------------------------------------------------------------
def tom_position(index: pd.DatetimeIndex, last: int = 1, first: int = 3) -> pd.Series:
    """Long-or-flat position: 1 inside the TOM window, 0 (cash) otherwise.

    No ``shift`` — the window is calendar-known, so the position for day *t* is set by
    *t*'s own calendar slot, decided in advance. (Per the house rule on calendar-known
    rules needing no execution lag.)
    """
    return tom_mask(index, last=last, first=first).astype(float)


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def backtest(close: pd.Series, position: pd.Series, cost_bps: float = 5.0) -> dict:
    """Run a long-or-flat position over ``close`` and return net stats + the daily series.

    ``cost_bps`` is charged one-way on every change in position (each entry and each exit).
    Cash earns 0%. Returns net daily return series and headline stats, including a
    ``sharpe_invested`` (Sharpe computed only over days the book is actually long — the
    fair "per-day-invested" comparison, since the timer sits in cash most of the time).
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
    sd = net.std(ddof=1)
    sharpe = float(net.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")
    # Per-day-invested Sharpe: only over the days the book is long (cash days excluded).
    inv = net[pos.to_numpy() > 0]
    sd_inv = inv.std(ddof=1) if len(inv) > 1 else 0.0
    sharpe_inv = (
        float(inv.mean() / sd_inv * np.sqrt(TRADING_DAYS)) if sd_inv > 0 else float("nan")
    )
    return {
        "net": net,
        "equity": equity,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sharpe_invested": sharpe_inv,
        "max_dd": _max_drawdown(equity.to_numpy()),
        "switches": int((turn > 1e-9).sum()),
        "time_in_market": float((pos > 0).mean()),
        "final": float(equity.iloc[-1]),
    }


# ---------------------------------------------------------------------------
# Inference: HAC long-run variance / t-stat
# ---------------------------------------------------------------------------
def _hac_lrv(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West long-run variance estimate for the mean of ``x`` (per-obs scale)."""
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
    return max(lrv, 0.0)


def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) t-stat for the mean of ``x`` (local, no quantlab dep)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 5:
        return float("nan")
    lrv = _hac_lrv(x, lags=lags)
    se = np.sqrt(lrv / n)
    return float(x.mean() / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# Sub-period decay contrast (with a block-bootstrap test of the difference)
# ---------------------------------------------------------------------------
def _tom_minus_rest_daily(close: pd.Series, last: int, first: int) -> pd.Series:
    """A daily series whose *mean* is (TOM mean - rest mean): r_t demeaned by the OTHER
    group, scaled so its sample mean equals the difference of group means. We use the
    centred-dummy contrast r_t*(d_t-dbar)/Var(d) which has mean exactly the OLS slope =
    difference of means; its block-bootstrapped mean gives a test of the difference."""
    ret = close.pct_change().dropna()
    mask = tom_mask(ret.index, last=last, first=first).reindex(ret.index).fillna(False)
    d = mask.to_numpy().astype(float)
    dbar = d.mean()
    var_d = ((d - dbar) ** 2).mean()
    contrast = ret.to_numpy() * (d - dbar) / var_d
    return pd.Series(contrast, index=ret.index)


def subperiod_contrast(
    close: pd.Series,
    split: str,
    last: int = 1,
    first: int = 3,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 89,
) -> dict:
    """Early-vs-late decay test of the TOM-minus-rest daily premium.

    ``split`` is a justified date that cuts the sample in two (e.g. the midpoint of the
    sample, or a publication-aware date). For each half we report the difference of daily
    means (bps) and its HAC *t*; for the change between halves we report a circular
    block-bootstrap two-sided p-value on (late_diff - early_diff), resampling each half in
    blocks of ``block`` days to respect autocorrelation.
    """
    contrast = _tom_minus_rest_daily(close, last, first)
    early = contrast.loc[:split]
    late = contrast.loc[split:]
    # Avoid double-counting the split day if it lands in both slices.
    if len(early) and len(late) and early.index[-1] == late.index[0]:
        late = late.iloc[1:]

    early_diff = float(early.mean() * 1e4)
    late_diff = float(late.mean() * 1e4)
    early_t = hac_tstat(early.to_numpy())
    late_t = hac_tstat(late.to_numpy())

    rng = np.random.default_rng(seed)
    # The observed change, in the SAME (decimal) units as the bootstrap below.
    obs_delta = float(late.mean() - early.mean())

    def _boot_mean(arr: np.ndarray) -> float:
        n = len(arr)
        if n == 0:
            return float("nan")
        n_blocks = int(np.ceil(n / block))
        starts = rng.integers(0, n, size=n_blocks)
        idxs = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        return float(arr[idxs[:n]].mean())

    ea = early.to_numpy()
    la = late.to_numpy()
    deltas = np.empty(n_boot)
    # Bootstrap under the null that both halves share the pooled mean: centre each half,
    # resample, and compare the bootstrap delta distribution against obs_delta.
    ea_c = ea - ea.mean()
    la_c = la - la.mean()
    for b in range(n_boot):
        deltas[b] = (_boot_mean(la_c) - _boot_mean(ea_c))
    p = float((np.abs(deltas) >= abs(obs_delta)).mean())

    return {
        "split": split,
        "early_diff_bps": early_diff,
        "late_diff_bps": late_diff,
        "early_t": float(early_t),
        "late_t": float(late_t),
        "delta_bps": float(late_diff - early_diff),
        "p_delta": p,
        "n_early": int(len(early)),
        "n_late": int(len(late)),
    }
