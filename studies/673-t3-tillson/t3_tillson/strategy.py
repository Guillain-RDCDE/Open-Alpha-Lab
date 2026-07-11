"""The strategy and its honest controls — Study 673 (T3, Tillson).

Tim Tillson's "T3" (*Technical Analysis of Stocks & Commodities*, Jan 1998, "Better Moving
Averages") is built from a "generalized DEMA" (GD) applied three times, i.e. six nested
EMAs of the same nominal length N, recombined with a "volume factor" v::

    GD(x, v) = (1+v)*EMA(x) - v*EMA(EMA(x))
    T3(x, N, v) = GD(GD(GD(x, v), v), v)

Expanding the nesting, T3 is a fixed linear combination of the 3rd through 6th EMA of the
price (``e3..e6``, each ``EMA(., N)`` applied that many times in a row)::

    c1 = -v^3
    c2 = 3v^2 + 3v^3
    c3 = -6v^2 - 3v - 3v^3
    c4 = 1 + 3v + v^3 + 3v^2
    T3 = c1*e6 + c2*e5 + c3*e4 + c4*e3

At v=0, T3 collapses to a plain EMA (well, its 3rd application); Tillson's own pitch is
that a v around 0.7-0.8 "virtually eliminates lag while smoothing the data" more than a
DEMA or TEMA of the same length — the claim under test is that a T3 price-cross rule turns
earlier AND gives cleaner (fewer whipsaw) crossovers than a plain SMA/EMA of the same
nominal N. Unlike McGinley's Dynamic (Study 672), T3 is NOT recursive in price — it is six
ordinary EMAs stacked and linearly recombined — so it is fully vectorized here (no Python
loop needed).

We turn that into the same honest race the desk runs for every "smarter MA" claim
(Studies 91, 432, 433, 481, 672):

- **T3 price-cross long/flat (or long/short).** Long when close > T3(N, v), flat (or
  short) otherwise.
- **T3-slope timer.** Long when T3(N, v) is rising session-over-session, flat/short
  otherwise — the other half of the brief ("T3 crossover / T3-slope timer").
- **SMA(N) crossover** and **EMA(N) crossover** — the two "dumb" benchmarks it claims to
  beat, same nominal N.
- **NET Sharpe race, excess-vs-excess** against buy-and-hold; costs one-way x NAV per
  leg on every position change (turnover), shorts pay borrow.
- **HAC (Newey-West) one-sample t** on the daily *active spread* (strategy - buy&hold) —
  the inference-bar number that decides the Signal stamp — plus the same t on the
  T3-vs-SMA and T3-vs-EMA *head-to-head* spreads (is T3 better than the "dumb" MAs it
  claims to beat, not just better than a coin).
- **A position-shuffle permutation placebo** that destroys the T3 rule's timing while
  keeping its turnover, to ask "is the timing real, or just exposure?"
- **A whipsaw count** (position switches/yr) — the literal "fewer false signals" claim.
- **A volume-factor (v) robustness sweep** — the parameter Tillson himself says is the
  smoothness/lag knob; if the edge (or lack of one) only shows up at one hand-picked v,
  it is not a robust result.

One documented execution lag throughout: the position formed on the close of *t* earns
the return of *t+1* (one ``shift``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Moving-average primitives
# --------------------------------------------------------------------------- #
def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (span convention)."""
    return series.ewm(span=period, adjust=False).mean()


def t3(series: pd.Series, n: int = 14, v: float = 0.7) -> pd.Series:
    """Tillson's T3: six nested EMA(n) stacked, recombined by the volume factor v.

    ``GD(x) = (1+v)*EMA(x) - v*EMA(EMA(x))`` applied three times ("generalized DEMA",
    nested); expanding the nesting collapses to a fixed linear combination of the 3rd
    through 6th EMA of the input. Fully vectorized (no Python loop): six calls to
    ``.ewm().mean()`` plus a linear combination.
    """
    e1 = ema(series, n)
    e2 = ema(e1, n)
    e3 = ema(e2, n)
    e4 = ema(e3, n)
    e5 = ema(e4, n)
    e6 = ema(e5, n)
    c1 = -(v ** 3)
    c2 = 3.0 * v ** 2 + 3.0 * v ** 3
    c3 = -6.0 * v ** 2 - 3.0 * v - 3.0 * v ** 3
    c4 = 1.0 + 3.0 * v + v ** 3 + 3.0 * v ** 2
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


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


def _slope_position(line: pd.Series, long_short: bool = False) -> pd.Series:
    rising = line.diff() > 0
    pos = pd.Series(0.0, index=line.index)
    pos[rising] = 1.0
    if long_short:
        pos[~rising] = -1.0
    pos[line.isna() | line.diff().isna()] = 0.0
    return pos


def t3_position(close: pd.Series, n: int = 14, v: float = 0.7,
                long_short: bool = False) -> pd.Series:
    """T3 price-cross rule: long when close > T3(n, v), flat/short otherwise."""
    return _cross_position(close, t3(close, n, v), long_short=long_short)


def t3_slope_position(close: pd.Series, n: int = 14, v: float = 0.7,
                      long_short: bool = False) -> pd.Series:
    """T3-slope timer: long when T3(n, v) is rising session-over-session, flat/short else."""
    return _slope_position(t3(close, n, v), long_short=long_short)


def sma_position(close: pd.Series, period: int = 14, long_short: bool = False) -> pd.Series:
    """SMA price-cross rule: long when close > SMA(period), flat/short otherwise."""
    return _cross_position(close, sma(close, period), long_short=long_short)


def ema_position(close: pd.Series, period: int = 14, long_short: bool = False) -> pd.Series:
    """EMA price-cross rule: long when close > EMA(period), flat/short otherwise."""
    return _cross_position(close, ema(close, period), long_short=long_short)


# --------------------------------------------------------------------------- #
# Diagnostics — does T3 actually "hug price" / react faster than SMA/EMA?
# --------------------------------------------------------------------------- #
def step_response(n: int = 14, v: float = 0.7, pre: int = 30, post: int = 30,
                  base: float = 100.0, jump_pct: float = 0.20) -> pd.DataFrame:
    """A textbook step (flat -> +jump_pct, held) — how fast does each line catch up?

    Deterministic, no randomness: isolates the *mechanism* (does stacking six EMAs with
    the v recombination make T3 catch up to a step faster than a plain EMA/SMA of the
    same N?) from any noise. Returns a frame indexed 0..pre+post-1 with columns
    price/T3/SMA/EMA; the jump lands at index ``pre``.
    """
    price = np.concatenate([np.full(pre, base), np.full(post, base * (1.0 + jump_pct))])
    s = pd.Series(price)
    return pd.DataFrame({
        "price": s.to_numpy(),
        "T3": t3(s, n, v).to_numpy(),
        "SMA": sma(s, n).to_numpy(),
        "EMA": ema(s, n).to_numpy(),
    })


def tracking_distance(close: pd.Series, n: int = 14, v: float = 0.7) -> dict:
    """Mean |close - line| / close (%) for T3 vs SMA vs EMA — the literal "hugs price" claim."""
    out = {}
    for name, line in (("T3", t3(close, n, v)),
                       ("SMA", sma(close, n)), ("EMA", ema(close, n))):
        dist = ((close - line).abs() / close).dropna()
        out[name] = float(dist.mean() * 100.0)
    return out


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
    seed: int = 673,
) -> dict:
    """Block-permutation placebo: is the T3 *timing* real, or just exposure?

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
    t3_n: int = 14,
    t3_v: float = 0.7,
    sma_period: int = 14,
    ema_period: int = 14,
    cost_bps: float = 5.0,
    borrow_bps_yr: float = 50.0,
    rf_annual: float = 0.0,
    long_short: bool = False,
    rule: str = "cross",
    n_perm: int = 2000,
    perm_seed: int = 673,
) -> dict:
    """Run the full T3-vs-SMA-vs-EMA-vs-buy&hold race on one tape.

    ``rule`` selects the T3 timer: ``"cross"`` (price vs T3, the primary rule, raced
    against price-cross SMA/EMA) or ``"slope"`` (T3 rising/falling — the slope-timer half
    of the brief; still raced against the same price-cross SMA/EMA benchmarks, since that
    is the comparison the folklore actually makes). Builds the three rules at the *same
    nominal N*, backtests each net of costs with one execution lag, computes buy&hold,
    runs the permutation placebo on the T3 rule, and reports the T3-vs-SMA / T3-vs-EMA
    head-to-head HAC t's (the literal "cleaner crossovers than a plain MA" claim).
    """
    close = bars["close"]
    t3_pos = (t3_position(close, t3_n, t3_v, long_short=long_short) if rule == "cross"
             else t3_slope_position(close, t3_n, t3_v, long_short=long_short))
    rules = {
        "T3": t3_pos,
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

    # Head-to-head: is T3 better than the "dumb" MAs it claims to beat?
    diff_t3_sma = bts["T3"]["strat_net"] - bts["SMA"]["strat_net"]
    diff_t3_ema = bts["T3"]["strat_net"] - bts["EMA"]["strat_net"]
    results["diff_t3_sma_bps"] = float(diff_t3_sma.mean() * 1e4)
    results["diff_t3_sma_t"] = hac_tstat(diff_t3_sma)
    results["diff_t3_ema_bps"] = float(diff_t3_ema.mean() * 1e4)
    results["diff_t3_ema_t"] = hac_tstat(diff_t3_ema)

    # Permutation placebo on the T3 rule (gross spread, so timing not cost is tested).
    perm = permutation_pvalue(close.pct_change(), rules["T3"], cost_bps=0.0,
                              n_perm=n_perm, seed=perm_seed)
    results["T3_permutation"] = perm
    return results


def v_sweep(
    bars: pd.DataFrame,
    vs: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
    t3_n: int = 14,
    cost_bps: float = 5.0,
    rule: str = "cross",
) -> pd.DataFrame:
    """Sweep Tillson's own "volume factor" v — the literal robustness the brief asks for.

    v=0 collapses T3 toward a plain (thrice-applied) EMA; v near 1 pushes it toward the
    most aggressive DEMA/TEMA-like overshoot. Returns one row per v with the T3 rule's
    net Sharpe, active-spread HAC t and whipsaw rate, so "does the edge survive across the
    v the indicator is tuned by" is answered directly instead of on one cherry-picked v.
    """
    rows = []
    for v in vs:
        r = run_experiment(bars, t3_n=t3_n, t3_v=v, cost_bps=cost_bps, rule=rule)["T3"]
        rows.append({"v": v, "sharpe_net": r["sharpe_net"], "spread_bps": r["mean_spread_bps"],
                     "spread_t": r["spread_t"], "switches_per_yr": r["switches_per_yr"]})
    return pd.DataFrame(rows).set_index("v")
