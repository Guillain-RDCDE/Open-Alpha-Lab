"""Strategy + inference for Study 925 — Front-End Trend.

The rule: read the **trend in the front of the yield curve** and let it pick your
duration. Signal on day ``t`` is the 3-month (63 trading-day) change in the 13-week bill
yield ``^IRX``. Rates **falling** (change < 0) → hold **TLT** (20y+ Treasuries, ~17 years
of duration). Rates **rising or flat** → hold **BIL** (1-3 month bills, no duration).
Binary, no shorting, re-evaluated daily. The signal formed at the close of day ``t`` is
acted on at day ``t+1`` — exactly **one** execution lag, applied in ``rate_trend_signal``
via a single ``.shift(1)`` and nowhere else.

A second, price-only expression of the same idea is offered for robustness:
``shy_momentum_signal`` uses the **12-minus-1-month total return of SHY** (the 1-3y
Treasury ETF) in place of the yield change — a rising short-bond price *is* a falling
front-end yield, so the two are the same bet read off different instruments.

We race four legs, all measured **excess-of-cash** (minus BIL's total return) so the
Sharpe comparison is apples-to-apples:

1. **static_mid** — buy-and-hold **IEF** (7-10y). The honest static-duration benchmark:
   roughly the average duration the switch rule carries, bought once and forgotten.
2. **switch** — the front-end trend rule (TLT when rates fall, BIL when they rise).
3. **random** — the null control: in TLT on *random* days at the **same** in-market
   frequency as the switch rule. If the switch beats random, the *timing* adds value
   beyond mere average exposure to duration. Read it **gross as well as net**: a
   frequency-matched coin flips ~``2p(1-p)N`` times against a trend rule's handful, so
   net-only it is handicapped by churn it never chose (see ``random_control_sweep``).
4. **static_long** — buy-and-hold **TLT**, reported for context (the switch rule's
   instrument, held all the time).
5. **matched blend** — the *turnover-free* exposure control: a constant ``w`` = the rule's
   own in-market fraction in TLT and ``1 - w`` in bills, rebalanced daily
   (``constant_weight_control`` / ``matched_blend_race``). It carries the same average
   duration as the rule with **no regime flips at all**, so unlike the random control it
   is not handicapped by friction — which makes it the honest answer to "is the *timing*
   worth anything?".

Costs are charged one-way x NAV on each switch day. There is **no short leg anywhere**,
so no borrow is paid or assumed; the cost sweep is the friction sensitivity that matters.

Returns are **simple** (arithmetic) so the binary portfolio return is exactly
``signal * r_long + (1 - signal) * r_cash`` and the wealth path compounds correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Signals (each carries exactly one execution lag)
# --------------------------------------------------------------------------- #
def rate_change(yield_pct: pd.Series, lookback: int = 63) -> pd.Series:
    """Trailing ``lookback``-day change in a yield level, in percentage points.

    ``yield_pct`` is a *yield*, not a price (``^IRX`` is the 13-week bill discount yield
    in percent). Negative = the front end has fallen over the window.
    """
    return (yield_pct - yield_pct.shift(lookback)).rename("d_rate")


def rate_trend_signal(yield_pct: pd.Series, lookback: int = 63) -> pd.Series:
    """1 = own duration (front-end yield falling), 0 = sit in bills. One-day lag.

    The comparison uses only data through day ``t``; the single ``.shift(1)`` is the
    study's one and only execution lag (act at ``t+1`` on a signal seen at ``t``).
    """
    d = rate_change(yield_pct, lookback=lookback)
    raw = (d < 0).astype(float)
    raw[d.isna()] = np.nan
    return raw.shift(1).rename("signal")


def shy_momentum_signal(shy_close: pd.Series, lookback: int = 252, gap: int = 21) -> pd.Series:
    """12-minus-1-month total-return momentum of SHY: 1 = own duration, 0 = bills.

    A price-only-free alternative read of the same bet (SHY total return rises when the
    front end falls), with the standard one-month skip. Same single ``.shift(1)`` lag.
    """
    mom = (shy_close.shift(gap) / shy_close.shift(lookback) - 1.0).rename("mom")
    raw = (mom > 0).astype(float)
    raw[mom.isna()] = np.nan
    return raw.shift(1).rename("signal")


def random_signal(signal: pd.Series, seed: int = 0) -> pd.Series:
    """Null control: the same in-market *frequency* as ``signal``, on random days.

    If the switch rule's edge were purely average duration exposure, this
    frequency-matched random control would show it too. If the switch beats random,
    *which* days it owns duration is what adds value.
    """
    in_frac = float(signal.dropna().mean())
    rng = np.random.default_rng(seed)
    n = len(signal)
    rand = pd.Series(
        (rng.random(n) < in_frac).astype(float), index=signal.index, name="random_signal"
    )
    rand[signal.isna()] = np.nan
    return rand


# --------------------------------------------------------------------------- #
# Backtest engine (simple returns; binary duration/cash with switching cost)
# --------------------------------------------------------------------------- #
def run_backtest(
    long_leg: pd.Series,
    cash: pd.Series,
    signal: pd.Series,
    cost_bps: float = 2.0,
) -> pd.DataFrame:
    """Apply a binary duration/cash signal to daily total-return closes.

    Parameters
    ----------
    long_leg, cash:
        Daily total-return **close levels** of the duration instrument (TLT) and the
        cash leg (BIL), aligned on a common index.
    signal:
        A {0, 1} series, **already lagged**: 1 = hold the duration leg, 0 = hold cash.
    cost_bps:
        One-way transaction cost in bps of NAV, charged on every signal flip. 2 bps is
        a fair default for TLT/BIL (both penny-wide, multi-billion-dollar ETFs); the
        sweep in ``timer_stats`` runs 0 to 25 bps.

    Returns a frame with ``r_long``, ``r_cash``, ``signal`` and ``r_strategy``
    (net of switch costs). Simple returns.
    """
    common = long_leg.index.intersection(cash.index)
    lg = long_leg.reindex(common).sort_index()
    ch = cash.reindex(common).sort_index()
    r_long = lg.pct_change().rename("r_long")
    r_cash = ch.pct_change().rename("r_cash")
    sig = signal.reindex(common).astype(float)

    switches = sig.diff().abs().fillna(0.0)
    cost = cost_bps * 1e-4
    r_strat = (sig * r_long + (1.0 - sig) * r_cash - switches * cost).rename("r_strategy")

    return pd.concat([r_long, r_cash, sig.rename("signal"), r_strat], axis=1).dropna()


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * (float(u[l:] @ u[:-l]) / n)
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def _auto_lags(n: int) -> int:
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Performance summary
# --------------------------------------------------------------------------- #
def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Annualised stats for a daily simple-return series.

    Pass an *excess-of-cash* series to get the excess Sharpe. ``cagr`` is the geometric
    growth of the series as given (so on an excess series it is an excess CAGR).
    """
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    mu = r.mean()
    std = r.std(ddof=1)
    sharpe = float(mu / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")

    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    cagr = (float(wealth.iloc[-1] ** (1.0 / years) - 1.0)
            if years > 0 and wealth.iloc[-1] > 0 else float("nan"))
    dd = float((wealth / wealth.cummax() - 1.0).min())

    return {
        "n_days": int(n),
        "cagr": cagr,
        "sharpe": sharpe,
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "max_drawdown": dd,
        "mean_daily_bps": float(mu * 1e4),
        "tstat": newey_west_t(r.to_numpy(), lags=_auto_lags(n)),
    }


def max_drawdown(returns: pd.Series) -> float:
    r = pd.Series(returns).astype(float).dropna()
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def sharpe_diff_tstat(r1: pd.Series, r2: pd.Series) -> float:
    """HAC *t* on the daily return *difference* r1 - r2.

    Both arms are excess-of-cash, so the cash leg cancels in the difference and this is
    the Jobson-Korkie return-difference test in its Newey-West form. |*t*| >= 2 on the
    real tape is the desk's bar for calling an advantage real.
    """
    diff = (r1 - r2).dropna()
    if len(diff) < 3:
        return float("nan")
    return newey_west_t(diff.to_numpy(), lags=_auto_lags(len(diff)))


# --------------------------------------------------------------------------- #
# The excess-vs-excess race (the headline)
# --------------------------------------------------------------------------- #
def compare(
    long_leg: pd.Series,
    mid_leg: pd.Series,
    cash: pd.Series,
    signal: pd.Series,
    cost_bps: float = 2.0,
    random_seed: int = 925,
) -> dict:
    """Race the front-end-trend switch against static IEF, static TLT and random timing.

    ``signal`` must already be lagged (see ``rate_trend_signal``). Every arm is reported
    **excess-of-cash** (minus the cash leg's own total return), so the Sharpe race is
    apples-to-apples and the strategy gets no credit for merely parking in bills while
    bills paid 5%.
    """
    idx = long_leg.index.intersection(mid_leg.index).intersection(cash.index)
    lg, md, ch = long_leg.loc[idx], mid_leg.loc[idx], cash.loc[idx]

    sig = signal.reindex(idx)
    rand = random_signal(sig, seed=random_seed)
    bt = run_backtest(lg, ch, sig, cost_bps=cost_bps)
    bt_rand = run_backtest(lg, ch, rand, cost_bps=cost_bps)

    r_mid = md.pct_change().reindex(bt.index)

    e_switch = (bt["r_strategy"] - bt["r_cash"]).rename("e_switch")
    e_mid = (r_mid - bt["r_cash"]).rename("e_mid").dropna()
    e_long = (bt["r_long"] - bt["r_cash"]).rename("e_long")
    e_rand = (bt_rand["r_strategy"] - bt_rand["r_cash"]).rename("e_rand")

    common = e_switch.index.intersection(e_mid.index)
    e_switch, e_mid = e_switch.loc[common], e_mid.loc[common]
    e_long, e_rand = e_long.loc[common], e_rand.reindex(common)

    in_frac = float(sig.reindex(common).mean())
    n_switches = int(sig.reindex(common).diff().abs().fillna(0.0).sum())

    return {
        "switch": summary(e_switch),
        "static_mid": summary(e_mid),
        "static_long": summary(e_long),
        "random": summary(e_rand),
        "dd_switch_abs": max_drawdown(bt["r_strategy"].loc[common]),
        "dd_mid_abs": max_drawdown(r_mid.loc[common]),
        "dd_long_abs": max_drawdown(bt["r_long"].loc[common]),
        "dd_rand_abs": max_drawdown(bt_rand["r_strategy"].reindex(common)),
        "excess_sharpe_adv": summary(e_switch)["sharpe"] - summary(e_mid)["sharpe"],
        "sharpe_adv_vs_random": summary(e_switch)["sharpe"] - summary(e_rand)["sharpe"],
        "t_diff_vs_mid": sharpe_diff_tstat(e_switch, e_mid),
        "t_diff_vs_long": sharpe_diff_tstat(e_switch, e_long),
        "t_diff_vs_random": sharpe_diff_tstat(e_switch, e_rand),
        "in_market_frac": in_frac,
        "n_switches": n_switches,
        "e_switch": e_switch, "e_mid": e_mid, "e_long": e_long, "e_rand": e_rand,
        "bt": bt,
    }


def compare_rate_rule(
    yield_pct: pd.Series,
    long_leg: pd.Series,
    mid_leg: pd.Series,
    cash: pd.Series,
    lookback: int = 63,
    cost_bps: float = 2.0,
    random_seed: int = 925,
) -> dict:
    """Convenience wrapper: build the ^IRX rate-trend signal, then ``compare``."""
    sig = rate_trend_signal(yield_pct, lookback=lookback)
    return compare(long_leg, mid_leg, cash, sig,
                   cost_bps=cost_bps, random_seed=random_seed)


# --------------------------------------------------------------------------- #
# Bootstrap Sharpe CI (circular block bootstrap on the excess series)
# --------------------------------------------------------------------------- #
def bootstrap_sharpe_ci(
    excess: pd.Series,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 925,
    periods_per_year: int = TRADING_DAYS,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the annualised Sharpe of an excess series.

    Blocks of ``block`` consecutive days preserve vol clustering and the rate-cycle
    persistence that makes daily bond returns anything but i.i.d.
    """
    r = np.asarray(pd.Series(excess).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"sharpe": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": n}
    ann = np.sqrt(periods_per_year)
    sd = r.std(ddof=1)
    point = float(r.mean() / sd * ann) if sd > 0 else float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.full(n_boot, np.nan)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        s = r[idx]
        ssd = s.std(ddof=1)
        if ssd > 0:
            boots[b] = s.mean() / ssd * ann
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((valid < 0).mean()), "n_obs": n,
            "n_boot": int(valid.size), "block": block}


def bootstrap_diff_ci(
    a: pd.Series, b: pd.Series, n_boot: int = 2000, block: int = 21,
    seed: int = 925, periods_per_year: int = TRADING_DAYS, alpha: float = 0.05,
) -> dict:
    """Block-bootstrap CI for the *Sharpe difference* between two aligned excess series.

    Resamples the same block indices from both arms, so the paired structure (and the
    cash leg they share) is preserved.
    """
    df = pd.concat([pd.Series(a).rename("a"), pd.Series(b).rename("b")], axis=1).dropna()
    x, y = df["a"].to_numpy(), df["b"].to_numpy()
    n = x.size
    if n < block + 2:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": n}
    ann = np.sqrt(periods_per_year)

    def _sh(v):
        s = v.std(ddof=1)
        return v.mean() / s * ann if s > 0 else np.nan

    point = float(_sh(x) - _sh(y))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.full(n_boot, np.nan)
    for i in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[i] = _sh(x[idx]) - _sh(y[idx])
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((valid < 0).mean()), "n_obs": n,
            "n_boot": int(valid.size), "block": block}


# --------------------------------------------------------------------------- #
# Robustness: era cut, lookback sweep, cost sweep, calendar years
# --------------------------------------------------------------------------- #
def era_cut(
    yield_pct: pd.Series, long_leg: pd.Series, mid_leg: pd.Series, cash: pd.Series,
    split: str = "2016-01-01", lookback: int = 63, cost_bps: float = 2.0,
) -> dict:
    """Split the sample at ``split`` and re-run the excess-Sharpe race on each half.

    The signal is built on the **full** yield history and then sliced, so the late era
    does not pay a fresh 63-day warmup (that would be a different rule, not a robustness
    check). No look-ahead is introduced: the signal is still formed from past yields only.
    """
    sig_full = rate_trend_signal(yield_pct, lookback=lookback)
    out = {}
    for tag, sl in [("early", slice(None, split)), ("late", slice(split, None))]:
        lg, md, ch = long_leg.loc[sl], mid_leg.loc[sl], cash.loc[sl]
        if len(lg) < 250:
            out[tag] = None
            continue
        cmp = compare(lg, md, ch, sig_full, cost_bps=cost_bps)
        out[tag] = {
            "n_days": cmp["switch"]["n_days"],
            "sharpe_mid": cmp["static_mid"]["sharpe"],
            "sharpe_switch": cmp["switch"]["sharpe"],
            "excess_sharpe_adv": cmp["excess_sharpe_adv"],
            "t_diff_vs_mid": cmp["t_diff_vs_mid"],
            "in_market_frac": cmp["in_market_frac"],
            "dd_mid_abs": cmp["dd_mid_abs"],
            "dd_switch_abs": cmp["dd_switch_abs"],
        }
    return out


def lookback_sweep(
    yield_pct: pd.Series, long_leg: pd.Series, mid_leg: pd.Series, cash: pd.Series,
    lookbacks=(21, 42, 63, 126, 252), cost_bps: float = 2.0,
) -> list[dict]:
    """How much of the result is the arbitrary 63-day window? Sweep it and see."""
    rows = []
    for lb in lookbacks:
        cmp = compare_rate_rule(yield_pct, long_leg, mid_leg, cash,
                                lookback=lb, cost_bps=cost_bps)
        rows.append({
            "lookback": lb,
            "sharpe_switch": cmp["switch"]["sharpe"],
            "excess_sharpe_adv": cmp["excess_sharpe_adv"],
            "t_diff_vs_mid": cmp["t_diff_vs_mid"],
            "in_market_frac": cmp["in_market_frac"],
            "n_switches": cmp["n_switches"],
        })
    return rows


def timer_stats(
    yield_pct: pd.Series, long_leg: pd.Series, mid_leg: pd.Series, cash: pd.Series,
    lookback: int = 63, cost_bps_grid=(0.0, 1.0, 2.0, 5.0, 10.0, 25.0),
) -> list[dict]:
    """Cost sweep. No short leg anywhere, so there is no borrow to pay — only switch
    friction, charged one-way x NAV on each flip."""
    rows = []
    for c in cost_bps_grid:
        cmp = compare_rate_rule(yield_pct, long_leg, mid_leg, cash,
                                lookback=lookback, cost_bps=c)
        rows.append({
            "cost_bps": c,
            "sharpe_switch": cmp["switch"]["sharpe"],
            "excess_sharpe_adv": cmp["excess_sharpe_adv"],
            "t_diff_vs_mid": cmp["t_diff_vs_mid"],
            "cagr_switch": cmp["switch"]["cagr"],
            "dd_switch_abs": cmp["dd_switch_abs"],
            "n_switches": cmp["n_switches"],
        })
    return rows


def calendar_year_table(
    yield_pct: pd.Series, long_leg: pd.Series, mid_leg: pd.Series, cash: pd.Series,
    lookback: int = 63, cost_bps: float = 2.0,
) -> pd.DataFrame:
    """Per-calendar-year *absolute* total return (%) for the switch vs static IEF/TLT.

    Absolute, not excess — the lived experience. Partial years at either end of the
    sample are kept but flagged by their day count.
    """
    sig = rate_trend_signal(yield_pct, lookback=lookback)
    idx = long_leg.index.intersection(mid_leg.index).intersection(cash.index)
    bt = run_backtest(long_leg.loc[idx], cash.loc[idx], sig, cost_bps=cost_bps)
    r_mid = mid_leg.loc[idx].pct_change().reindex(bt.index)
    rows = []
    for y in sorted(set(bt.index.year)):
        m = bt.index.year == y
        rows.append({
            "year": int(y),
            "n_days": int(m.sum()),
            "switch_pct": float((1.0 + bt["r_strategy"][m]).prod() - 1.0) * 100,
            "ief_pct": float((1.0 + r_mid[m].dropna()).prod() - 1.0) * 100,
            "tlt_pct": float((1.0 + bt["r_long"][m]).prod() - 1.0) * 100,
            "in_duration_frac": float(bt["signal"][m].mean()),
        })
    return pd.DataFrame(rows).set_index("year")


def random_control_sweep(
    long_leg: pd.Series, mid_leg: pd.Series, cash: pd.Series, signal: pd.Series,
    n_seeds: int = 30, seed0: int = 925, cost_bps: float = 2.0,
) -> dict:
    """Average the switch-vs-random comparison over many control seeds.

    A *single* random control is one draw: with 4,800 daily coin flips its realised
    Sharpe swings by several tenths, so quoting one seed is exactly the trap this desk
    keeps catching. Sweep the seed and report the mean, the spread and how often the
    HAC *t* clears 2. Run it **gross as well as net** — a frequency-matched random
    control flips roughly ``2p(1-p)N`` times, far more than a persistent trend rule, so
    at any positive cost the random arm is handicapped by turnover it never chose.
    """
    advs, ts = [], []
    for i in range(n_seeds):
        cmp = compare(long_leg, mid_leg, cash, signal,
                      cost_bps=cost_bps, random_seed=seed0 + i)
        advs.append(cmp["sharpe_adv_vs_random"])
        ts.append(cmp["t_diff_vs_random"])
    advs, ts = np.asarray(advs, float), np.asarray(ts, float)
    return {
        "n_seeds": int(n_seeds), "cost_bps": cost_bps,
        "adv_mean": float(advs.mean()), "adv_sd": float(advs.std(ddof=1)),
        "adv_min": float(advs.min()), "adv_max": float(advs.max()),
        "t_mean": float(ts.mean()), "t_min": float(ts.min()), "t_max": float(ts.max()),
        "frac_t_ge_2": float((ts >= 2.0).mean()),
    }


def constant_weight_control(
    long_leg: pd.Series, cash: pd.Series, weight: float, cost_bps: float = 2.0,
) -> dict:
    """The turnover-free exposure control: a constant ``weight`` in the duration leg.

    Holds ``weight`` of NAV in ``long_leg`` and the rest in ``cash``, **rebalanced daily**
    back to that weight. Its turnover is the small drift-correction trade (roughly
    ``2 |w_drift - w|`` of NAV a day), not a regime flip, and it is charged at the same
    ``cost_bps`` the switch rule pays — so the comparison is friction-fair in a way the
    frequency-matched random control is not.

    Returns the excess-of-cash series plus its summary and realised annual turnover.
    """
    common = long_leg.index.intersection(cash.index)
    rl = long_leg.reindex(common).sort_index().pct_change().rename("r_long")
    rc = cash.reindex(common).sort_index().pct_change().rename("r_cash")
    df = pd.concat([rl, rc], axis=1).dropna()
    w = float(weight)
    gross = w * df["r_long"] + (1.0 - w) * df["r_cash"]
    drift = (w * (1.0 + df["r_long"])) / (w * (1.0 + df["r_long"])
                                          + (1.0 - w) * (1.0 + df["r_cash"]))
    turnover = 2.0 * (drift - w).abs()
    net = gross - turnover * cost_bps * 1e-4
    excess = (net - df["r_cash"]).rename("e_blend")
    return {
        "weight": w,
        "excess": excess,
        "turnover_ann": float(turnover.mean() * TRADING_DAYS),
        "summary": summary(excess),
    }


def matched_blend_race(
    long_leg: pd.Series, mid_leg: pd.Series, cash: pd.Series, signal: pd.Series,
    cost_bps: float = 2.0,
) -> dict:
    """Race the switch rule against a constant-weight blend of the **same** exposure.

    The random control answers "did the rule beat a coin?" — but only after handing the
    coin a turnover bill it never asked for. This control removes that confound entirely:
    same average time in duration, zero regime flips, same cost assumption. If the rule's
    *timing* is worth anything, it must beat this.
    """
    cmp = compare(long_leg, mid_leg, cash, signal, cost_bps=cost_bps)
    ctrl = constant_weight_control(long_leg, cash, cmp["in_market_frac"], cost_bps=cost_bps)
    e_b = ctrl["excess"].reindex(cmp["e_switch"].index).dropna()
    e_s = cmp["e_switch"].reindex(e_b.index)
    s_s, s_b = summary(e_s)["sharpe"], summary(e_b)["sharpe"]
    return {
        "weight": ctrl["weight"],
        "cost_bps": cost_bps,
        "sharpe_switch": s_s,
        "sharpe_blend": s_b,
        "adv_vs_blend": s_s - s_b,
        "t_diff_vs_blend": sharpe_diff_tstat(e_s, e_b),
        "turnover_ann_blend": ctrl["turnover_ann"],
        "n_switches": cmp["n_switches"],
        "n_days": int(len(e_b)),
    }


def conditional_day_test(
    yield_pct: pd.Series, long_leg: pd.Series, cash: pd.Series,
    lookback: int = 63, cost_bps: float = 0.0,
) -> dict:
    """The rule's timing, stripped of portfolio construction and of cost.

    Compares the **excess-of-cash daily return of the duration leg** on days the rule
    says "own duration" against days it says "sit in bills". This is the cleanest read
    of whether the front-end trend carries information at all: if it does, the on-day
    mean must beat the off-day mean by more than noise.

    Two *t*s on the spread. Welch's assumes independent draws inside each bucket, which
    daily bond returns under a persistent signal are not, so it is reported only for
    reference; ``spread_hac_t`` is the **HAC** version and is the one to read. It is the
    Newey-West *t* of the series ``e * (s - p) / (p(1-p))``, whose mean is exactly the
    difference in bucket means (equivalently: a dummy regression with HAC errors).
    """
    sig = rate_trend_signal(yield_pct, lookback=lookback)
    bt = run_backtest(long_leg, cash, sig, cost_bps=cost_bps)
    e_long = bt["r_long"] - bt["r_cash"]
    on = e_long[bt["signal"] == 1.0]
    off = e_long[bt["signal"] == 0.0]

    s = bt["signal"]
    p = float(s.mean())
    if 0.0 < p < 1.0:
        contrast = (e_long * (s - p) / (p * (1.0 - p))).to_numpy()
        hac_t = newey_west_t(contrast, lags=_auto_lags(len(contrast)))
    else:
        hac_t = float("nan")

    return {
        "n_on": int(len(on)), "n_off": int(len(off)),
        "on_bp": float(on.mean() * 1e4), "off_bp": float(off.mean() * 1e4),
        "spread_bp": float((on.mean() - off.mean()) * 1e4),
        "t_on": newey_west_t(on.to_numpy(), lags=_auto_lags(len(on))),
        "t_off": newey_west_t(off.to_numpy(), lags=_auto_lags(len(off))),
        "spread_hac_t": hac_t,
        "welch_t": welch_t(on.to_numpy(), off.to_numpy()),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (machinery proof — never supports the stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(frame: pd.DataFrame, lookback: int = 63, cost_bps: float = 2.0) -> dict:
    """Run the excess-Sharpe race on a synthetic (rate, long, mid, cash) tape.

    On a world with planted rate-trend persistence the switch rule should beat the static
    intermediate leg; on the i.i.d.-increment null it should not. Proves the harness is
    unbiased — it never supports a real-tape stamp.
    """
    cmp = compare_rate_rule(frame["rate"], frame["long"], frame["mid"], frame["cash"],
                            lookback=lookback, cost_bps=cost_bps)
    return {
        "excess_sharpe_adv": cmp["excess_sharpe_adv"],
        "sharpe_switch": cmp["switch"]["sharpe"],
        "sharpe_mid": cmp["static_mid"]["sharpe"],
        "t_diff_vs_mid": cmp["t_diff_vs_mid"],
        "sharpe_adv_vs_random": cmp["sharpe_adv_vs_random"],
        "in_market_frac": cmp["in_market_frac"],
        "n_days": cmp["switch"]["n_days"],
    }
