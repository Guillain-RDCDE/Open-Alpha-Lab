"""The strategy and its honest controls — Study 672 (McGinley Dynamic).

John McGinley's "Dynamic" (Technically Speaking, 1997) is a recursive moving average
whose own update speed is driven by how far price has run from the line::

    MD(t) = MD(t-1) + ( P(t) - MD(t-1) ) / ( N * (P(t) / MD(t-1))**4 )

When price is running away from the line, ``P/MD`` moves off 1, the quartic term grows,
the denominator grows, and the *increment shrinks* — i.e. the line resists being dragged
too far, too fast. When price sits near the line the ratio is ~1 and the increment behaves
like a fast EMA. McGinley's own pitch (echoed on every retail charting site since): the
Dynamic "auto-adjusts" and "hugs price" more faithfully than a fixed SMA/EMA of the same
nominal length, so its crossovers are cleaner ("no more whipsaws"). A step-response check
(``notebooks``/``examples/verify.py``) shows the mechanism actually cuts the *other* way on
a sudden jump: the quartic brake makes MD react *more slowly* than an EMA or even an SMA
of the same N right after a shock — the "adaptive" part is a brake, not an accelerator.

We turn that into the same honest race the desk runs for every "smarter MA" claim
(Studies 91, 432, 433, 434):

- **McGinley Dynamic price-cross long/flat (or long/short).** Long when close > MD(N),
  flat (or short) otherwise.
- **SMA(N) crossover** and **EMA(N) crossover** — the two "dumb" benchmarks it claims to
  beat, same nominal N.
- **NET Sharpe race, excess-vs-excess** against buy-and-hold; costs one-way x NAV per
  leg on every position change (turnover), shorts pay borrow.
- **HAC (Newey-West) one-sample t** on the daily *active spread* (strategy - buy&hold) —
  the inference-bar number that decides the Signal stamp — plus the same t on the
  MD-vs-SMA and MD-vs-EMA *head-to-head* spreads (is MD better than the "dumb" MAs it
  claims to beat, not just better than a coin).
- **A position-shuffle permutation placebo** that destroys the MD rule's timing while
  keeping its turnover, to ask "is the timing real, or just exposure?"
- **A whipsaw count** (position switches/yr) — the literal "fewer false signals" claim.

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
def mcginley_dynamic(series: pd.Series, n: int = 14) -> pd.Series:
    """John McGinley's "Dynamic": MD(t) = MD(t-1) + (P(t)-MD(t-1)) / (N * (P(t)/MD(t-1))**4).

    Recursive by construction (each value depends on the previous), so it is computed in
    a plain Python loop over numpy arrays — the series is at most tens of thousands of
    bars, so this costs milliseconds. Seeded with the first valid price (the standard
    convention: MD(0) = P(0)); the first bar carries no signal (matches the NaN warm-up
    every other MA in this study serves).
    """
    p = series.to_numpy(dtype=float)
    out = np.empty_like(p)
    out[0] = p[0]
    for t in range(1, len(p)):
        prev = out[t - 1]
        if prev <= 0 or not np.isfinite(prev):
            out[t] = p[t]
            continue
        ratio = p[t] / prev
        denom = n * (ratio ** 4)
        if not np.isfinite(denom) or denom == 0:
            out[t] = prev
        else:
            out[t] = prev + (p[t] - prev) / denom
    md = pd.Series(out, index=series.index)
    md.iloc[0] = np.nan  # no signal on the seed bar itself
    return md


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


def md_position(close: pd.Series, n: int = 14, long_short: bool = False) -> pd.Series:
    """McGinley Dynamic price-cross rule: long when close > MD(n), flat/short otherwise."""
    return _cross_position(close, mcginley_dynamic(close, n), long_short=long_short)


def sma_position(close: pd.Series, period: int = 14, long_short: bool = False) -> pd.Series:
    """SMA price-cross rule: long when close > SMA(period), flat/short otherwise."""
    return _cross_position(close, sma(close, period), long_short=long_short)


def ema_position(close: pd.Series, period: int = 14, long_short: bool = False) -> pd.Series:
    """EMA price-cross rule: long when close > EMA(period), flat/short otherwise."""
    return _cross_position(close, ema(close, period), long_short=long_short)


# --------------------------------------------------------------------------- #
# Diagnostics — does MD actually "hug price" / react faster than SMA/EMA?
# --------------------------------------------------------------------------- #
def step_response(n: int = 14, pre: int = 30, post: int = 30,
                  base: float = 100.0, jump_pct: float = 0.20) -> pd.DataFrame:
    """A textbook step (flat -> +jump_pct, held) — how fast does each line catch up?

    Deterministic, no randomness: isolates the *mechanism* (does the quartic brake in
    the McGinley formula make it faster or slower than an EMA/SMA of the same N right
    after a shock?) from any noise. Returns a frame indexed 0..pre+post-1 with columns
    price/MD/SMA/EMA; the jump lands at index ``pre``.
    """
    price = np.concatenate([np.full(pre, base), np.full(post, base * (1.0 + jump_pct))])
    s = pd.Series(price)
    return pd.DataFrame({
        "price": s.to_numpy(),
        "MD": mcginley_dynamic(s, n).to_numpy(),
        "SMA": sma(s, n).to_numpy(),
        "EMA": ema(s, n).to_numpy(),
    })


def tracking_distance(close: pd.Series, n: int = 14) -> dict:
    """Mean |close - line| / close (%) for MD vs SMA vs EMA — the literal "hugs price" claim."""
    out = {}
    for name, line in (("MD", mcginley_dynamic(close, n)),
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
    seed: int = 672,
) -> dict:
    """Block-permutation placebo: is the MD *timing* real, or just exposure?

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
    md_n: int = 14,
    sma_period: int = 14,
    ema_period: int = 14,
    cost_bps: float = 5.0,
    borrow_bps_yr: float = 50.0,
    rf_annual: float = 0.0,
    long_short: bool = False,
    n_perm: int = 2000,
    perm_seed: int = 672,
) -> dict:
    """Run the full MD-vs-SMA-vs-EMA-vs-buy&hold race on one tape.

    Builds three timing rules at the *same nominal N* (McGinley Dynamic, SMA, EMA),
    backtests each net of costs with one execution lag, computes buy&hold, runs the
    permutation placebo on the MD rule, and reports the MD-vs-SMA / MD-vs-EMA
    head-to-head HAC t's (the literal "cleaner crossovers than a plain MA" claim).
    """
    close = bars["close"]
    rules = {
        "MD": md_position(close, md_n, long_short=long_short),
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

    # Head-to-head: is MD better than the "dumb" MAs it claims to beat?
    md_spread = bts["MD"]["strat_net"] - bts["MD"]["bh"]
    sma_spread = bts["SMA"]["strat_net"] - bts["SMA"]["bh"]
    ema_spread = bts["EMA"]["strat_net"] - bts["EMA"]["bh"]
    diff_md_sma = bts["MD"]["strat_net"] - bts["SMA"]["strat_net"]
    diff_md_ema = bts["MD"]["strat_net"] - bts["EMA"]["strat_net"]
    results["diff_md_sma_bps"] = float(diff_md_sma.mean() * 1e4)
    results["diff_md_sma_t"] = hac_tstat(diff_md_sma)
    results["diff_md_ema_bps"] = float(diff_md_ema.mean() * 1e4)
    results["diff_md_ema_t"] = hac_tstat(diff_md_ema)

    # Permutation placebo on the MD rule (gross spread, so timing not cost is tested).
    perm = permutation_pvalue(close.pct_change(), rules["MD"], cost_bps=0.0,
                              n_perm=n_perm, seed=perm_seed)
    results["MD_permutation"] = perm
    return results
