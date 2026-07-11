"""The strategy and its honest controls — Study 674 (VIDYA).

Tushar Chande's Variable Index Dynamic Average (*"Adapting Moving Averages to Market
Volatility"*, Technical Analysis of Stocks & Commodities, March 1992) replaces the fixed
smoothing constant of an EMA with one driven by his own Chande Momentum Oscillator
(CMO, Study 185)::

    CMO(t, m)  = 100 * (sum_up(m) - sum_down(m)) / (sum_up(m) + sum_down(m))
    VI(t)      = |CMO(t, m)| / 100                          (volatility index, in [0,1])
    alpha      = 2 / (period + 1)                            (the nominal EMA speed)
    VIDYA(t)   = alpha * VI(t) * P(t) + (1 - alpha * VI(t)) * VIDYA(t-1)

CMO is a pure *momentum* (trend-strength) oscillator, not a volatility measure: it
reads near +-100 when price moves persistently in one direction over the last ``m``
bars (however big or small the moves are) and near 0 when up-moves and down-moves
net out (chop), *regardless of how big the day-to-day swings actually are*. Chande's
own pitch conflates "volatile" with "trending" — the mechanism section below tests
both readings honestly: does VI (the speed knob) actually track realized *volatility*,
and does it track *trend strength*, on the real tape?

We turn that into the same honest race the desk runs for every "smarter MA" claim
(Studies 91, 432, 433, 434, 672, 673):

- **VIDYA price-cross long/flat (or long/short).** Long when close > VIDYA(period,
  cmo_period), flat (or short) otherwise.
- **SMA(N) crossover** and **EMA(N) crossover** — the two "dumb" benchmarks it claims
  to beat, same nominal N.
- **NET Sharpe race, excess-vs-excess** against buy-and-hold; costs one-way x NAV per
  leg on every position change (turnover), shorts pay borrow.
- **HAC (Newey-West) one-sample t** on the daily *active spread* (strategy - buy&hold)
  — the inference-bar number that decides the Signal stamp — plus the same t on the
  VIDYA-vs-SMA and VIDYA-vs-EMA *head-to-head* spreads (is VIDYA better than the
  "dumb" MAs it claims to beat, not just better than a coin).
- **A position-shuffle permutation placebo** that destroys VIDYA's timing while
  keeping its turnover, to ask "is the timing real, or just exposure?"
- **A whipsaw count** (position switches/yr) and a **CMO-period robustness sweep**
  (the CMO lookback ``cmo_period`` is a free knob distinct from the base ``period``;
  the literal claim must survive varying it).

One documented execution lag throughout: the position formed on the close of *t* earns
the return of *t+1* (one ``shift``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def cmo(price: pd.Series, period: int = 9) -> pd.Series:
    """Chande Momentum Oscillator, in [-100, +100] (Study 185's indicator).

    100 * (sum of up-moves - sum of down-moves) / (sum of up-moves + sum of down-moves)
    over a trailing ``period``-bar window of first differences. +100 = every move in the
    window was up, -100 = every move was down, 0 = up-moves and down-moves cancelled
    out (chop, not necessarily *small* moves).
    """
    diff = price.diff()
    up = diff.clip(lower=0.0)
    down = (-diff).clip(lower=0.0)
    sum_up = up.rolling(period).sum()
    sum_down = down.rolling(period).sum()
    denom = sum_up + sum_down
    out = 100.0 * (sum_up - sum_down) / denom.replace(0.0, np.nan)
    return out.fillna(0.0).rename("cmo").where(denom.notna())


def vidya(series: pd.Series, period: int = 14, cmo_period: int = 9) -> pd.Series:
    """Chande's VIDYA: an EMA whose speed is scaled by ``|CMO(cmo_period)| / 100``.

    Recursive by construction, so it is computed in a plain Python loop over numpy
    arrays (at most tens of thousands of bars -> milliseconds). Seeded with the first
    price for which a CMO reading exists; VIDYA is flat (NaN) before that (matches the
    NaN warm-up every other MA in this study serves). When ``VI(t) = 0`` (pure chop
    inside the CMO window) VIDYA does not move at all that bar — the literal "freezes
    in quiet markets" mechanism.
    """
    p = series.to_numpy(dtype=float)
    vi = (cmo(series, cmo_period).abs() / 100.0).to_numpy(dtype=float)
    n = len(p)
    out = np.full(n, np.nan)
    alpha = 2.0 / (period + 1.0)

    start = None
    for i in range(n):
        if np.isfinite(vi[i]):
            start = i
            break
    if start is None:
        return pd.Series(out, index=series.index, name="vidya")

    out[start] = p[start]
    for t in range(start + 1, n):
        k = alpha * (vi[t] if np.isfinite(vi[t]) else 0.0)
        out[t] = k * p[t] + (1.0 - k) * out[t - 1]
    return pd.Series(out, index=series.index, name="vidya")


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (span convention)."""
    return series.ewm(span=period, adjust=False).mean()


# --------------------------------------------------------------------------- #
# Timing rules — each returns a desired position series in {-1,0,+1} (pre-lag)
# --------------------------------------------------------------------------- #
def _cross_position(close: pd.Series, line: pd.Series, long_short: bool = False) -> pd.Series:
    above = close > line
    pos = pd.Series(0.0, index=close.index)
    pos[above] = 1.0
    if long_short:
        pos[~above] = -1.0
    pos[line.isna()] = 0.0
    return pos


def vidya_position(close: pd.Series, period: int = 14, cmo_period: int = 9,
                   long_short: bool = False) -> pd.Series:
    """VIDYA price-cross rule: long when close > VIDYA(period, cmo_period), flat/short otherwise."""
    return _cross_position(close, vidya(close, period, cmo_period), long_short=long_short)


def sma_position(close: pd.Series, period: int = 14, long_short: bool = False) -> pd.Series:
    """SMA price-cross rule: long when close > SMA(period), flat/short otherwise."""
    return _cross_position(close, sma(close, period), long_short=long_short)


def ema_position(close: pd.Series, period: int = 14, long_short: bool = False) -> pd.Series:
    """EMA price-cross rule: long when close > EMA(period), flat/short otherwise."""
    return _cross_position(close, ema(close, period), long_short=long_short)


# --------------------------------------------------------------------------- #
# Diagnostics — does VIDYA actually speed up in trending/volatile regimes?
# --------------------------------------------------------------------------- #
def step_response(period: int = 14, cmo_period: int = 9, pre: int = 30, post: int = 30,
                  base: float = 100.0, jump_pct: float = 0.20) -> pd.DataFrame:
    """A textbook step (flat -> +jump_pct, held) — how fast does each line catch up?

    Deterministic, no randomness: isolates the *mechanism* from any noise. The flat
    pre-jump segment has CMO ~ 0 (no net direction) so VIDYA should barely move — the
    "freezes in quiet markets" half of the claim. The single-direction jump then drives
    CMO toward +-100 inside the window, so VIDYA should speed up and catch price the
    "accelerates in a trend" half. Returns a frame indexed 0..pre+post-1 with columns
    price/VIDYA/SMA/EMA; the jump lands at index ``pre``.
    """
    price = np.concatenate([np.full(pre, base), np.full(post, base * (1.0 + jump_pct))])
    s = pd.Series(price)
    return pd.DataFrame({
        "price": s.to_numpy(),
        "VIDYA": vidya(s, period, cmo_period).to_numpy(),
        "SMA": sma(s, period).to_numpy(),
        "EMA": ema(s, period).to_numpy(),
    })


def tracking_distance(close: pd.Series, period: int = 14, cmo_period: int = 9) -> dict:
    """Mean |close - line| / close (%) for VIDYA vs SMA vs EMA — the literal "hugs price" claim."""
    out = {}
    for name, line in (("VIDYA", vidya(close, period, cmo_period)),
                       ("SMA", sma(close, period)), ("EMA", ema(close, period))):
        dist = ((close - line).abs() / close).dropna()
        out[name] = float(dist.mean() * 100.0)
    return out


def regime_correlations(close: pd.Series, cmo_period: int = 9, vol_window: int = 20,
                        trend_window: int = 20) -> dict:
    """Does VI actually track realized volatility, or trend strength, or neither?

    ``VI(t) = |CMO(t, cmo_period)| / 100`` is VIDYA's speed knob. We correlate it
    (Pearson) against two independent regime proxies computed on the same tape: a
    trailing realized-volatility measure (rolling std of returns, ``vol_window``) and a
    trailing trend-strength measure (a *longer*-window CMO, ``trend_window``, so it is
    not mechanically the same quantity as VI itself). Returns both correlations plus a
    high/low tercile split of VI's own mean level in high- vs low-volatility and
    high- vs low-trend days — the direct read of "speeds up when volatile/trending".
    """
    vi = (cmo(close, cmo_period).abs() / 100.0).rename("vi")
    ret = close.pct_change()
    realized_vol = ret.rolling(vol_window).std().rename("vol")
    trend_strength = (cmo(close, trend_window).abs() / 100.0).rename("trend")

    df = pd.concat([vi, realized_vol, trend_strength], axis=1).dropna()
    corr_vol = float(df["vi"].corr(df["vol"]))
    corr_trend = float(df["vi"].corr(df["trend"]))

    vol_q = df["vol"].quantile([1 / 3, 2 / 3])
    lo_vol = df.loc[df["vol"] <= vol_q.iloc[0], "vi"].mean()
    hi_vol = df.loc[df["vol"] >= vol_q.iloc[1], "vi"].mean()

    trend_q = df["trend"].quantile([1 / 3, 2 / 3])
    lo_trend = df.loc[df["trend"] <= trend_q.iloc[0], "vi"].mean()
    hi_trend = df.loc[df["trend"] >= trend_q.iloc[1], "vi"].mean()

    return {
        "n": int(len(df)),
        "corr_vi_vol": corr_vol, "corr_vi_trend": corr_trend,
        "vi_low_vol_tercile": float(lo_vol), "vi_high_vol_tercile": float(hi_vol),
        "vi_low_trend_tercile": float(lo_trend), "vi_high_trend_tercile": float(hi_trend),
    }


# --------------------------------------------------------------------------- #
# Backtest engine — daily returns, one execution lag, costs one-way x NAV
# --------------------------------------------------------------------------- #
def backtest(
    bars: pd.DataFrame,
    position: pd.Series,
    cost_bps: float = 5.0,
    borrow_bps_yr: float = 50.0,
    rf_annual: float = 0.0,
) -> pd.DataFrame:
    """Run a daily-returns backtest of ``position`` on ``bars`` and return a frame.

    One documented execution lag: the desired position formed on the close of *t* is
    held for the return of *t+1* (a single ``shift(1)``). Daily asset returns are
    close-to-close. Costs are charged one-way x NAV on the change in position
    (``|pos_t - pos_{t-1}|``). A short position pays a daily borrow fee
    (``borrow_bps_yr/252``) on the short notional.
    """
    close = bars["close"]
    asset_ret = close.pct_change().fillna(0.0)
    held = position.shift(1).fillna(0.0)              # the ONE execution lag
    turnover = held.diff().abs().fillna(held.abs())    # one-way change in exposure

    cost = turnover * (cost_bps * 1e-4)
    borrow = np.maximum(-held, 0.0) * (borrow_bps_yr * 1e-4 / TRADING_DAYS)

    strat_gross = held * asset_ret
    strat_net = strat_gross - cost - borrow
    bh = asset_ret

    rf_daily = rf_annual / TRADING_DAYS
    return pd.DataFrame(
        {
            "asset_ret": asset_ret,
            "held": held,
            "turnover": turnover,
            "strat_gross": strat_gross,
            "strat_net": strat_net,
            "bh": bh,
            "strat_net_excess": strat_net - rf_daily * held.abs().clip(upper=1.0),
            "bh_excess": bh - rf_daily,
        },
        index=bars.index,
    )


# --------------------------------------------------------------------------- #
# Statistics — Sharpe, drawdown, HAC t, permutation placebo
# --------------------------------------------------------------------------- #
def annual_sharpe(daily: pd.Series) -> float:
    d = pd.Series(daily).dropna().to_numpy(dtype=float)
    if d.size < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / d.std(ddof=1) * np.sqrt(TRADING_DAYS))


def cagr(daily: pd.Series) -> float:
    d = pd.Series(daily).dropna().to_numpy(dtype=float)
    if d.size == 0:
        return float("nan")
    total = float(np.prod(1.0 + d))
    yrs = d.size / TRADING_DAYS
    return total ** (1.0 / yrs) - 1.0 if yrs > 0 and total > 0 else float("nan")


def max_drawdown(daily: pd.Series) -> float:
    d = pd.Series(daily).dropna().to_numpy(dtype=float)
    if d.size == 0:
        return float("nan")
    eq = np.cumprod(1.0 + d)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def hac_tstat(daily: pd.Series) -> float:
    """Newey-West HAC one-sample t-stat for the mean of a daily series being != 0."""
    r = pd.Series(daily).dropna().to_numpy(dtype=float)
    n = r.size
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


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances). NaN if either < 2 obs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def permutation_pvalue(
    asset_ret: pd.Series,
    position: pd.Series,
    cost_bps: float = 0.0,
    n_perm: int = 2000,
    seed: int = 674,
) -> dict:
    """Block-permutation placebo: is the VIDYA *timing* real, or just exposure?

    The realised position path's block structure is held fixed but circularly shifted
    against returns, destroying the alignment between the rule's calls and the actual
    moves while preserving its turnover and net-long bias. Test statistic: mean daily
    active spread (strategy - buy&hold). Returns the observed statistic, the placebo
    mean, and a one-sided p-value (share of placebo spreads >= observed).
    """
    rng = np.random.default_rng(seed)
    a = asset_ret.fillna(0.0).to_numpy(dtype=float)
    held = position.shift(1).fillna(0.0).to_numpy(dtype=float)
    n = a.size

    def _spread(h):
        turn = np.abs(np.diff(h, prepend=0.0))
        strat = h * a - turn * (cost_bps * 1e-4)
        return float((strat - a).mean())

    obs = _spread(held)
    placebo = np.empty(n_perm)
    for i in range(n_perm):
        shift = int(rng.integers(1, n))
        placebo[i] = _spread(np.roll(held, shift))
    pval = float((placebo >= obs).mean())
    return {"observed_spread_bps": obs * 1e4, "placebo_mean_bps": float(placebo.mean()) * 1e4,
            "p_value": pval, "n_perm": n_perm}


# --------------------------------------------------------------------------- #
# Summary of a backtest frame
# --------------------------------------------------------------------------- #
def summarize(bt: pd.DataFrame, label: str = "") -> dict:
    net = bt["strat_net"]
    spread = bt["strat_net"] - bt["bh"]
    pos_changes = int((bt["held"].diff().abs() > 1e-9).sum())
    yrs = len(bt) / TRADING_DAYS
    return {
        "label": label,
        "n_days": int(len(bt)),
        "sharpe_net": annual_sharpe(net),
        "sharpe_excess": annual_sharpe(bt["strat_net_excess"]),
        "bh_sharpe": annual_sharpe(bt["bh"]),
        "bh_sharpe_excess": annual_sharpe(bt["bh_excess"]),
        "cagr_net": cagr(net),
        "bh_cagr": cagr(bt["bh"]),
        "maxdd_net": max_drawdown(net),
        "bh_maxdd": max_drawdown(bt["bh"]),
        "mean_spread_bps": float(spread.mean() * 1e4),
        "spread_t": hac_tstat(spread),
        "n_switches": pos_changes,
        "switches_per_yr": pos_changes / yrs if yrs > 0 else float("nan"),
        "time_in_market": float((bt["held"].abs() > 1e-9).mean()),
        "ann_turnover": float(bt["turnover"].sum() / yrs) if yrs > 0 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(
    bars: pd.DataFrame,
    period: int = 14,
    cmo_period: int = 9,
    sma_period: int = 14,
    ema_period: int = 14,
    cost_bps: float = 5.0,
    borrow_bps_yr: float = 50.0,
    rf_annual: float = 0.0,
    long_short: bool = False,
    n_perm: int = 2000,
    perm_seed: int = 674,
) -> dict:
    """Run the full VIDYA-vs-SMA-vs-EMA-vs-buy&hold race on one tape.

    Builds three timing rules at the *same nominal N* (VIDYA, SMA, EMA), backtests each
    net of costs with one execution lag, computes buy&hold, runs the permutation placebo
    on the VIDYA rule, and reports the VIDYA-vs-SMA / VIDYA-vs-EMA head-to-head HAC t's
    (the literal "beats a plain MA" claim).
    """
    close = bars["close"]
    rules = {
        "VIDYA": vidya_position(close, period, cmo_period, long_short=long_short),
        "SMA": sma_position(close, sma_period, long_short=long_short),
        "EMA": ema_position(close, ema_period, long_short=long_short),
    }
    results = {}
    bts = {}
    for name, pos in rules.items():
        bt = backtest(bars, pos, cost_bps=cost_bps, borrow_bps_yr=borrow_bps_yr,
                      rf_annual=rf_annual)
        bts[name] = bt
        results[name] = summarize(bt, name)

    # Head-to-head: is VIDYA better than the "dumb" MAs it claims to beat?
    diff_v_sma = bts["VIDYA"]["strat_net"] - bts["SMA"]["strat_net"]
    diff_v_ema = bts["VIDYA"]["strat_net"] - bts["EMA"]["strat_net"]
    results["diff_vidya_sma_bps"] = float(diff_v_sma.mean() * 1e4)
    results["diff_vidya_sma_t"] = hac_tstat(diff_v_sma)
    results["diff_vidya_ema_bps"] = float(diff_v_ema.mean() * 1e4)
    results["diff_vidya_ema_t"] = hac_tstat(diff_v_ema)

    # Permutation placebo on the VIDYA rule (gross spread, so timing not cost is tested).
    perm = permutation_pvalue(close.pct_change(), rules["VIDYA"], cost_bps=0.0,
                              n_perm=n_perm, seed=perm_seed)
    results["VIDYA_permutation"] = perm
    return results
