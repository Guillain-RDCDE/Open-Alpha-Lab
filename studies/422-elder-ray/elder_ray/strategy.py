"""The strategy and its honest controls — Study 422 (Elder Ray).

Dr Alexander Elder's *Elder Ray* (1989, *Trading for a Living*) reads price as a tug of
war around a 13-period EMA "consensus of value":

    EMA13(t)      = 13-period exponential moving average of Close
    Bull Power(t) = High(t) - EMA13(t)     (how far bulls push price above value)
    Bear Power(t) = Low(t)  - EMA13(t)     (how far bears push price below value)

Elder's own trading recipe uses Elder Ray as the *second screen* of his triple-screen
system: only trade in the direction of the EMA trend, and time the entry with the power
oscillators.  We test the literal long/flat (and long/short) version on a daily tape:

- **Long when** the EMA13 is *rising* (trend up) **and** Bear Power is *negative but
  rising* (bears losing ground) — Elder's classic long signal.
- **Flat / short otherwise.**

Two simplifications make this a fair, look-ahead-free daily timing rule:

- Position for day *t+1* is decided from data through the close of day *t* (one `shift`,
  applied once — the documented execution lag).
- We race the rule's **net** return stream against **buy-and-hold**, excess-of-cash to
  excess-of-cash, with HAC *t*, a permutation/label-shuffle placebo, a cost sweep, and a
  head-to-head against the obvious simpler benchmark (a 200-day SMA trend filter and a
  plain EMA13-cross) so the "Elder Ray is better" claim is actually tested.

No look-ahead: all indicators computed on closes/highs/lows up to bar *t*; the position is
applied to the return of bar *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Elder Ray indicator
# ---------------------------------------------------------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    """Standard exponential moving average (adjust=False, the trading convention)."""
    return series.ewm(span=span, adjust=False).mean()


def elder_ray(bars: pd.DataFrame, span: int = 13) -> pd.DataFrame:
    """Bull Power, Bear Power and the EMA13 trend, per Elder (1989).

    Returns a frame with columns ``ema``, ``bull`` (High − EMA), ``bear`` (Low − EMA),
    indexed identically to ``bars``.  The first ``span`` bars carry the EMA's warm-up
    transient (ewm with adjust=False has no NaN region, but the early values are
    untrustworthy and are skipped by the warm-up cut in the backtest).
    """
    e = ema(bars["close"], span)
    return pd.DataFrame(
        {"ema": e, "bull": bars["high"] - e, "bear": bars["low"] - e},
        index=bars.index,
    )


# ---------------------------------------------------------------------------
# Signal generators — the timing rules (position known at close of t)
# ---------------------------------------------------------------------------
def elder_signal(bars: pd.DataFrame, span: int = 13, long_short: bool = False) -> pd.Series:
    """Elder-Ray long/flat (or long/short) target position, known at the close of *t*.

    Long (+1) when the EMA13 trend is *up* (EMA rising vs the prior bar) AND Bear Power is
    *negative but rising* (Bear[t] > Bear[t-1] while Bear[t] < 0) — Elder's classic long
    setup: in an uptrend, enter when the bears' grip is loosening.

    With ``long_short=True`` the symmetric short (−1) fires when the EMA13 trend is *down*
    AND Bull Power is *positive but falling* (Bull[t] < Bull[t-1] while Bull[t] > 0).
    Otherwise the position is flat (0) for the long/short variant, or *held* (carry the
    last long/flat state) for the long/flat variant — Elder's rule is an entry/exit timer,
    so the long/flat book stays long until the trend turns down.

    Returns the *target* position series; the one-day execution lag is applied in
    ``backtest`` (this series is the position decided at close *t*).
    """
    er = elder_ray(bars, span)
    ema_up = er["ema"] > er["ema"].shift(1)
    bear_rising = (er["bear"] > er["bear"].shift(1)) & (er["bear"] < 0)
    bull_falling = (er["bull"] < er["bull"].shift(1)) & (er["bull"] > 0)

    long_entry = ema_up & bear_rising
    long_exit = ~ema_up  # trend turned down → leave the long

    pos = pd.Series(0.0, index=bars.index)
    if long_short:
        short_entry = (~ema_up) & bull_falling
        pos[long_entry] = 1.0
        pos[short_entry] = -1.0
        return pos

    # Long/flat: a stateful timer — go long on entry, exit when the trend turns down,
    # otherwise carry the previous state.
    state = 0.0
    out = np.zeros(len(bars))
    le = long_entry.to_numpy()
    lx = long_exit.to_numpy()
    for i in range(len(bars)):
        if state == 0.0 and le[i]:
            state = 1.0
        elif state == 1.0 and lx[i]:
            state = 0.0
        out[i] = state
    return pd.Series(out, index=bars.index)


def sma_filter_signal(bars: pd.DataFrame, window: int = 200) -> pd.Series:
    """The obvious simpler benchmark: long when Close > its `window`-day SMA, else flat."""
    sma = bars["close"].rolling(window).mean()
    return (bars["close"] > sma).astype(float)


def ema_cross_signal(bars: pd.DataFrame, span: int = 13) -> pd.Series:
    """A second simple benchmark: long when Close > EMA(span) (the Elder EMA, no power)."""
    e = ema(bars["close"], span)
    return (bars["close"] > e).astype(float)


# ---------------------------------------------------------------------------
# Backtest — daily long/flat with one-day execution lag, costs one-way x NAV
# ---------------------------------------------------------------------------
def backtest(
    bars: pd.DataFrame,
    position: pd.Series,
    cost_bps: float = 2.0,
    rf_annual: float = 0.0,
    borrow_bps_annual: float = 50.0,
    warmup: int = 30,
) -> pd.DataFrame:
    """Run a daily long/flat (or long/short) book and return a per-day ledger.

    Convention (one documented execution lag): the target ``position`` is decided at the
    close of *t* and earns the asset return of *t+1* — applied with a single ``shift(1)``.
    Costs are charged one-way × NAV on the change in position
    (``cost_bps * |Δposition|``).  Short days pay daily borrow (``borrow_bps_annual``).

    ``rf_annual`` is the cash rate the *flat* days earn (and the benchmark's excess-of-cash
    baseline); pass the realised rate to make the Sharpe race excess-vs-excess.  Default 0
    keeps the offline synthetic control clean; the real-tape run uses 0 as well and races
    the strategy's excess stream against buy-and-hold's excess stream on the same rf.

    Columns: ``ret_asset`` (asset daily return), ``pos`` (effective position, lagged),
    ``ret_strat_gross``, ``ret_strat_net``, ``ret_bh`` (buy-and-hold), ``in_market``.
    """
    px = bars["close"].astype(float)
    ret = px.pct_change().fillna(0.0)
    rf_daily = rf_annual / TRADING_DAYS
    borrow_daily = borrow_bps_annual * 1e-4 / TRADING_DAYS

    pos_eff = position.shift(1).fillna(0.0)  # the one execution lag
    if warmup > 0:
        pos_eff.iloc[:warmup] = 0.0

    dpos = pos_eff.diff().abs().fillna(pos_eff.abs())
    cost = dpos * cost_bps * 1e-4
    borrow = (pos_eff < 0).astype(float) * (-pos_eff) * borrow_daily

    # Flat capital earns the cash rate; invested capital earns the asset.
    invested = pos_eff * ret
    cash_leg = (1.0 - pos_eff.clip(lower=0.0)) * rf_daily  # idle long-capital in cash
    gross = invested + cash_leg
    net = gross - cost - borrow

    bh = ret.copy()

    return pd.DataFrame(
        {
            "ret_asset": ret,
            "pos": pos_eff,
            "ret_strat_gross": gross,
            "ret_strat_net": net,
            "ret_bh": bh,
            "in_market": (pos_eff != 0).astype(float),
        },
        index=bars.index,
    )


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def hac_tstat(returns: np.ndarray, lags: int | None = None) -> float:
    """Newey-West HAC t-stat that the mean daily return differs from zero."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 10:
        return float("nan")
    mu = r.mean()
    e = r - mu
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def sharpe(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised Sharpe of a daily return stream (assumes returns already excess-of-cash)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def cagr(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate from a daily return stream."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    growth = np.prod(1.0 + r)
    yrs = r.size / periods_per_year
    return float(growth ** (1.0 / yrs) - 1.0) if growth > 0 and yrs > 0 else float("nan")


def max_drawdown(returns: np.ndarray) -> float:
    """Worst peak-to-trough drawdown of the equity curve (a negative number)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def rotation_placebo(
    positions: np.ndarray,
    asset_ret: np.ndarray,
    n_perm: int = 2000,
    seed: int = 422,
) -> tuple[np.ndarray, float, float]:
    """Circular-rotation placebo on the *position-vs-return alignment*.

    The strategy's edge, if real, is in *when* it is invested.  We destroy that alignment
    by circularly rotating the position stream by a random offset against the *fixed* asset
    returns (preserving both series' own autocorrelation), recompute the mean book return,
    and ask how often a random rotation does as well as the realised alignment.  A small p
    means being-invested-on-good-days is genuinely non-random.

    Returns ``(placebo_means, actual_mean, p_value)``.
    """
    pos = np.asarray(positions, dtype=float)
    ret = np.asarray(asset_ret, dtype=float)
    n = pos.size
    if n < 30:
        return np.array([]), float("nan"), float("nan")
    actual = float((pos * ret).mean())
    rng = np.random.default_rng(seed)
    means = np.empty(n_perm)
    for i in range(n_perm):
        shift = int(rng.integers(1, n))
        means[i] = (np.roll(pos, shift) * ret).mean()
    p = (np.sum(means >= actual) + 1) / (n_perm + 1)
    return means, actual, float(p)


def permutation_pvalue(ledger: pd.DataFrame, n_perm: int = 2000, seed: int = 422) -> float:
    """Convenience wrapper: the rotation-placebo p from a backtest ledger."""
    _, _, p = rotation_placebo(
        ledger["pos"].to_numpy(), ledger["ret_asset"].to_numpy(),
        n_perm=n_perm, seed=seed,
    )
    return p


def summarize_book(ledger: pd.DataFrame, rf_annual: float = 0.0) -> dict:
    """Headline stats for a strategy ledger vs buy-and-hold, excess-of-cash.

    Sharpe is computed on the *excess-of-cash* daily stream for both arms so the race is
    excess-vs-excess.  Returns net strategy stats, buy-and-hold stats, the HAC *t* on the
    strategy's net excess return, time-in-market, turnover (one-way), and the permutation p.
    """
    rf_daily = rf_annual / TRADING_DAYS
    strat = ledger["ret_strat_net"].to_numpy()
    bh = ledger["ret_bh"].to_numpy()
    strat_x = strat - rf_daily
    bh_x = bh - rf_daily
    turn = float(ledger["pos"].diff().abs().sum())
    yrs = len(ledger) / TRADING_DAYS

    return {
        "n_days": int(len(ledger)),
        "years": float(yrs),
        "strat_sharpe": sharpe(strat_x),
        "strat_cagr": cagr(strat),
        "strat_mdd": max_drawdown(strat),
        "strat_hac_t": hac_tstat(strat_x),
        "bh_sharpe": sharpe(bh_x),
        "bh_cagr": cagr(bh),
        "bh_mdd": max_drawdown(bh),
        "bh_hac_t": hac_tstat(bh_x),
        "time_in_market": float(ledger["in_market"].mean()),
        "turnover_oneway_per_yr": float(turn / yrs) if yrs > 0 else float("nan"),
        "perm_p": permutation_pvalue(ledger),
        # The decisive race: does the rule's excess Sharpe beat buy-and-hold's?
        "sharpe_delta": sharpe(strat_x) - sharpe(bh_x),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_experiment(
    bars: pd.DataFrame,
    span: int = 13,
    cost_bps: float = 2.0,
    long_short: bool = False,
    rf_annual: float = 0.0,
) -> dict:
    """Build the Elder-Ray book and the two simpler benchmarks; return a comparison dict.

    Returns a dict with ``elder``, ``sma200``, ``emacross`` summary dicts plus the raw
    Elder ledger under ``ledger`` (for plotting equity curves).
    """
    pos = elder_signal(bars, span=span, long_short=long_short)
    led = backtest(bars, pos, cost_bps=cost_bps, rf_annual=rf_annual)
    elder = summarize_book(led, rf_annual=rf_annual)

    sma_led = backtest(bars, sma_filter_signal(bars, 200), cost_bps=cost_bps, rf_annual=rf_annual)
    ema_led = backtest(bars, ema_cross_signal(bars, span), cost_bps=cost_bps, rf_annual=rf_annual)

    return {
        "elder": elder,
        "sma200": summarize_book(sma_led, rf_annual=rf_annual),
        "emacross": summarize_book(ema_led, rf_annual=rf_annual),
        "ledger": led,
    }
