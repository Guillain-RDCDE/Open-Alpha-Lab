"""The Guppy Multiple MA (GMMA) and its honest controls — Study 435.

Daryl Guppy's indicator plots **two ribbons** of exponential moving averages:

- the **short-term ('trader') ribbon** — EMAs of 3, 5, 8, 10, 12, 15 days, said to
  reflect the sentiment of short-term traders;
- the **long-term ('investor') ribbon** — EMAs of 30, 35, 40, 45, 50, 60 days, said
  to reflect the conviction of longer-term investors.

The folk reading: when the short ribbon trades **above** the long ribbon you are in an
uptrend; when the two ribbons are **wide apart and parallel**, the trend has *conviction*
and will continue; when they **compress and tangle**, the trend is exhausting. The desk
turns this into a falsifiable timing rule and races it.

Rules implemented (all long/flat or long/short; signal formed on close of *t*, acted on the
return of *t+1* — one shift, applied once):

- ``ribbon_signal`` — the canonical GMMA cross: long when the *mean* of the short ribbon is
  above the *mean* of the long ribbon (uptrend), flat (or short) otherwise.
- ``conviction_signal`` — Guppy's distinctive claim: long *only when the ribbon separation
  (conviction) is wide* (short ribbon above long AND the normalised spread above its own
  rolling median). This is the "ribbon reveals conviction" test — does gating on width help?
- ``sma_signal`` / ``ema_signal`` — the **obvious simpler benchmarks** (a single 50-day
  SMA / EMA trend filter). If GMMA's twelve lines beat one line, the ribbon earns its keep;
  if not, it is an ornate repackaging of a single moving average.
- ``random_timing_signal`` — the null control: same in-market fraction, random days. Isolates
  *timing skill* from mere *exposure reduction*.

Inference: a HAC (Newey-West) t-stat on the strategy's mean daily excess return, a
return-difference HAC t-stat for Sharpe races (Jobson-Korkie style), a label-shuffle
permutation placebo on the signal, a one-way cost sweep, and the synthetic positive control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

SHORT_RIBBON = (3, 5, 8, 10, 12, 15)
LONG_RIBBON = (30, 35, 40, 45, 50, 60)


# ---------------------------------------------------------------------------
# The GMMA ribbons
# ---------------------------------------------------------------------------
def ribbons(
    close: pd.Series,
    short: tuple[int, ...] = SHORT_RIBBON,
    long: tuple[int, ...] = LONG_RIBBON,
) -> dict[str, pd.DataFrame]:
    """Compute both GMMA ribbons (EMAs). Returns ``{'short': df, 'long': df}``.

    Each frame's columns are the EMA spans; EMAs use ``adjust=False`` (the recursive
    form a charting package draws). The longest span (60) needs warm-up; downstream
    rules require all spans valid (``min_periods`` via the long ribbon).
    """
    short_df = pd.DataFrame(
        {f"ema{n}": close.ewm(span=n, adjust=False, min_periods=n).mean() for n in short},
        index=close.index,
    )
    long_df = pd.DataFrame(
        {f"ema{n}": close.ewm(span=n, adjust=False, min_periods=n).mean() for n in long},
        index=close.index,
    )
    return {"short": short_df, "long": long_df}


def ribbon_means(close: pd.Series, **kw) -> pd.DataFrame:
    """Mean of each ribbon and the normalised spread (conviction proxy).

    Columns:
      ``short_mean`` / ``long_mean`` — the mean EMA of each ribbon;
      ``spread``     — (short_mean − long_mean) / close, a scale-free separation;
      ``above``      — 1 when short_mean > long_mean (uptrend), else 0.
    """
    rb = ribbons(close, **kw)
    sm = rb["short"].mean(axis=1)
    lm = rb["long"].mean(axis=1)
    spread = (sm - lm) / close
    out = pd.DataFrame(
        {"short_mean": sm, "long_mean": lm, "spread": spread,
         "above": (sm > lm).astype(float)},
        index=close.index,
    )
    return out


# ---------------------------------------------------------------------------
# Timing rules (each returns a {0,1} or {-1,0,1} signal, lagged one day)
# ---------------------------------------------------------------------------
def _lag(sig: pd.Series, name: str) -> pd.Series:
    """One execution lag: a signal known at close of t acts on the return t -> t+1."""
    return sig.shift(1).rename(name)


def ribbon_signal(close: pd.Series, allow_short: bool = False, **kw) -> pd.Series:
    """Canonical GMMA cross: long when short ribbon > long ribbon.

    ``allow_short=False`` → long/flat (1 when above, 0 when below).
    ``allow_short=True``  → long/short (+1 above, −1 below).
    Lagged one day (no look-ahead).
    """
    rm = ribbon_means(close, **kw)
    raw = rm["above"]
    if allow_short:
        raw = raw * 2 - 1  # {0,1} -> {-1,+1}
    return _lag(raw, "ribbon")


def conviction_signal(
    close: pd.Series, lookback: int = 60, allow_short: bool = False, **kw
) -> pd.Series:
    """Guppy's 'conviction' rule: long only when the ribbons are *wide* and short>long.

    The distinctive GMMA claim is that a *wide, parallel* separation signals conviction
    and a continuing trend, while a *compressed* ribbon warns of exhaustion. We make this
    falsifiable: hold long only when (a) the short ribbon is above the long ribbon AND
    (b) the normalised spread exceeds its own trailing ``lookback``-day median (i.e. the
    ribbon is wider than usual). Otherwise flat (or short, if ``allow_short``).
    Lagged one day.
    """
    rm = ribbon_means(close, **kw)
    wide = rm["spread"] > rm["spread"].rolling(lookback, min_periods=lookback).median()
    long_ok = (rm["above"] > 0) & wide
    raw = long_ok.astype(float)
    if allow_short:
        # short when below AND widening downward
        narrow_down = (rm["above"] < 1) & (
            rm["spread"] < rm["spread"].rolling(lookback, min_periods=lookback).median()
        )
        raw = long_ok.astype(float) - narrow_down.astype(float)
    return _lag(raw, "conviction")


def sma_signal(close: pd.Series, n: int = 50, allow_short: bool = False) -> pd.Series:
    """The obvious simpler benchmark: a single n-day SMA trend filter. Lagged one day."""
    sma = close.rolling(n, min_periods=n).mean()
    raw = (close > sma).astype(float)
    if allow_short:
        raw = raw * 2 - 1
    return _lag(raw, "sma")


def ema_signal(close: pd.Series, n: int = 50, allow_short: bool = False) -> pd.Series:
    """A single n-day EMA trend filter — the EMA-family simpler benchmark. Lagged one day."""
    ema = close.ewm(span=n, adjust=False, min_periods=n).mean()
    raw = (close > ema).astype(float)
    if allow_short:
        raw = raw * 2 - 1
    return _lag(raw, "ema")


def random_timing_signal(signal: pd.Series, seed: int = 0) -> pd.Series:
    """A null control: same in-market *frequency* as ``signal`` but on random days.

    If the GMMA's Sharpe edge over buy-and-hold were purely reduced exposure, this
    control (same exposure, random days) would match it. If GMMA beats it, the *timing*
    (which days) adds value.
    """
    base = signal.dropna()
    in_market_frac = float((base > 0).mean())
    rng = np.random.default_rng(seed)
    n = len(signal)
    rand = pd.Series(
        (rng.random(n) < in_market_frac).astype(float),
        index=signal.index, name="random",
    )
    rand[signal.isna()] = np.nan
    return rand


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def run_backtest(
    close: pd.Series,
    signal: pd.Series,
    rf_daily: float | pd.Series = 0.0,
    cost_bps: float = 2.0,
    borrow_bps_ann: float = 50.0,
) -> pd.DataFrame:
    """Apply a {-1,0,1} (or {0,1}) signal to daily returns. Costs one-way x NAV.

    Parameters
    ----------
    close : adjusted daily close.
    signal : already-lagged position series (signal at t acts on return t -> t+1).
    rf_daily : daily risk-free / cash return (credited on the flat portion, i.e. on
        1 - |position|). Scalar (e.g. 0.04/252) or a Series.
    cost_bps : one-way transaction cost in bps, charged on |Δposition| each switch.
    borrow_bps_ann : annual borrow cost charged on the *short* exposure (negative pos).

    Returns columns: ``r_asset``, ``r_cash``, ``position``, ``r_strategy``, ``r_bh``,
    ``r_strategy_excess`` (strategy − cash), ``r_bh_excess`` (asset − cash).
    """
    r_asset = (close / close.shift(1) - 1.0).rename("r_asset")
    if isinstance(rf_daily, pd.Series):
        r_cash = rf_daily.reindex(close.index).fillna(0.0)
    else:
        r_cash = pd.Series(float(rf_daily), index=close.index)
    r_cash.name = "r_cash"

    pos = signal.reindex(close.index).fillna(0.0)
    # cash weight = 1 - |position| (long earns asset, short earns -asset, rest in cash)
    cash_w = 1.0 - pos.abs()

    switches = pos.diff().abs().fillna(0.0)
    cost = switches * cost_bps * 1e-4
    borrow = (-pos).clip(lower=0.0) * (borrow_bps_ann * 1e-4 / TRADING_DAYS_PER_YEAR)

    r_strat = (pos * r_asset + cash_w * r_cash - cost - borrow).rename("r_strategy")

    out = pd.concat(
        [r_asset, r_cash, pos.rename("position"), r_strat, r_asset.rename("r_bh")],
        axis=1,
    )
    out["r_strategy_excess"] = out["r_strategy"] - out["r_cash"]
    out["r_bh_excess"] = out["r_bh"] - out["r_cash"]
    return out.dropna()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def _hac_tstat(x: np.ndarray) -> float:
    """Newey-West HAC t-stat on the mean of ``x`` (auto lag = floor(4(n/100)^(2/9)))."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """Headline annualised stats for a daily return series + HAC t-stat on the mean.

    The HAC t-stat is the inference-bar number: it decides whether the mean daily
    (excess) return is distinguishable from zero. Pass an *excess* series for a fair race.
    """
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n == 0:
        return {k: float("nan") for k in
                ("n_days", "cagr", "sharpe", "vol_ann", "max_drawdown", "mean_daily_bps", "tstat")}
    mu = r.mean()
    std = r.std(ddof=1)
    sharpe = float(mu / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    vol_ann = float(std * np.sqrt(periods_per_year))
    cum = (1.0 + r).cumprod()
    years = n / periods_per_year
    cagr = float(cum.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and cum.iloc[-1] > 0 else float("nan")
    dd = float((cum / cum.cummax() - 1.0).min())
    return {
        "n_days": int(n),
        "cagr": cagr,
        "sharpe": sharpe,
        "vol_ann": vol_ann,
        "max_drawdown": dd,
        "mean_daily_bps": float(mu * 1e4),
        "tstat": _hac_tstat(r.to_numpy()),
    }


def sharpe_diff_tstat(r1: pd.Series, r2: pd.Series) -> float:
    """HAC t-stat on the daily return *difference* r1 − r2 (Jobson-Korkie style).

    |t| >= 2 on the difference means arm 1 has a reliably higher risk-adjusted return
    than arm 2 on this tape.
    """
    diff = (pd.Series(r1) - pd.Series(r2)).dropna()
    return _hac_tstat(diff.to_numpy())


def permutation_pvalue(
    close: pd.Series,
    signal_fn,
    rf_daily: float = 0.0,
    cost_bps: float = 0.0,
    n_perm: int = 1000,
    seed: int = 435,
    **sig_kw,
) -> dict:
    """Label-shuffle placebo: how often does a *block-shuffled* signal beat the real one?

    We compute the real strategy's mean excess return, then ``n_perm`` times we randomly
    *circularly rotate* the (lagged) signal — destroying its alignment with returns while
    preserving its autocorrelation/in-market fraction — and recompute. The p-value is the
    fraction of rotated signals whose mean excess return is >= the real one. A real edge
    sits in the right tail (small p); noise gives p ~ 0.5.
    """
    sig = signal_fn(close, **sig_kw).dropna()
    bt = run_backtest(close.loc[sig.index], sig, rf_daily=rf_daily, cost_bps=cost_bps)
    real_mean = bt["r_strategy_excess"].mean()

    rng = np.random.default_rng(seed)
    arr = sig.to_numpy()
    asset = (close.loc[sig.index] / close.loc[sig.index].shift(1) - 1.0).to_numpy()
    n = len(arr)
    ge = 0
    perm_means = np.empty(n_perm)
    for i in range(n_perm):
        shift = int(rng.integers(1, n - 1))
        rotated = np.roll(arr, shift)
        cash_w = 1.0 - np.abs(rotated)
        # excess of cash with flat rf=0 in placebo (rf cancels in excess); keep it simple
        pm = np.nanmean(rotated * asset)  # mean strategy return; cash term = rf*cash_w
        perm_means[i] = pm
        if pm >= real_mean:
            ge += 1
    return {
        "real_mean_bps": float(real_mean * 1e4),
        "perm_mean_bps": float(np.nanmean(perm_means) * 1e4),
        "p_value": float((ge + 1) / (n_perm + 1)),
        "n_perm": n_perm,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_experiment(
    close: pd.Series,
    rf_daily: float | pd.Series = 0.0,
    cost_bps: float = 2.0,
    sma_n: int = 50,
    conviction_lookback: int = 60,
    allow_short: bool = False,
    random_seed: int = 42,
) -> dict:
    """Run all arms and return their summaries + the key race t-stats.

    Arms: buy-and-hold, GMMA ribbon cross, GMMA conviction-gated, single SMA, single EMA,
    and a random-timing control matched to the ribbon-cross in-market fraction. Every
    Sharpe is reported on the *excess-of-cash* series so the race is excess-vs-excess.
    """
    sig_ribbon = ribbon_signal(close, allow_short=allow_short)
    sig_conv = conviction_signal(close, lookback=conviction_lookback, allow_short=allow_short)
    sig_sma = sma_signal(close, n=sma_n, allow_short=allow_short)
    sig_ema = ema_signal(close, n=sma_n, allow_short=allow_short)
    sig_rand = random_timing_signal(sig_ribbon, seed=random_seed)

    bt_ribbon = run_backtest(close, sig_ribbon, rf_daily=rf_daily, cost_bps=cost_bps)
    bt_conv = run_backtest(close, sig_conv, rf_daily=rf_daily, cost_bps=cost_bps)
    bt_sma = run_backtest(close, sig_sma, rf_daily=rf_daily, cost_bps=cost_bps)
    bt_ema = run_backtest(close, sig_ema, rf_daily=rf_daily, cost_bps=cost_bps)
    bt_rand = run_backtest(close, sig_rand, rf_daily=rf_daily, cost_bps=cost_bps)

    out = {
        "bh": summary(bt_ribbon["r_bh_excess"]),
        "ribbon": summary(bt_ribbon["r_strategy_excess"]),
        "conviction": summary(bt_conv["r_strategy_excess"]),
        "sma": summary(bt_sma["r_strategy_excess"]),
        "ema": summary(bt_ema["r_strategy_excess"]),
        "random": summary(bt_rand["r_strategy_excess"]),
        "in_market_frac": float((sig_ribbon.dropna() > 0).mean()),
        "n_switches": int(sig_ribbon.diff().abs().fillna(0).sum()),
        # excess-vs-excess Sharpe races (HAC t on daily return difference)
        "t_ribbon_vs_bh": sharpe_diff_tstat(bt_ribbon["r_strategy_excess"], bt_ribbon["r_bh_excess"]),
        "t_ribbon_vs_rand": sharpe_diff_tstat(bt_ribbon["r_strategy_excess"], bt_rand["r_strategy_excess"]),
        "t_ribbon_vs_sma": sharpe_diff_tstat(bt_ribbon["r_strategy_excess"], bt_sma["r_strategy_excess"]),
        "t_conv_vs_ribbon": sharpe_diff_tstat(bt_conv["r_strategy_excess"], bt_ribbon["r_strategy_excess"]),
    }
    return out
