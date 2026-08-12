"""Strategy + inference for Study 912 — Gold + Trend.

The rule: hold the gold ETF (GLD) only when its price is **above** its 200-day simple
moving average; otherwise sit in T-bills (BIL, the cash leg). Re-evaluated daily, binary
(fully in gold or fully in cash), no shorting. The signal formed at the close of day ``t``
is acted on at day ``t+1`` (one-day rebalance lag — no look-ahead).

We race three legs, all measured **excess-of-cash** (minus the BIL total-return leg) so
the comparison is apples-to-apples on risk-adjusted terms:

1. **buy_and_hold** — always long gold. The volatile diversifier with long dead decades.
2. **trend** — the 200-day MA overlay: long gold above the MA, in T-bills below.
3. **random** — the null control: cash on random days at the *same* in-market frequency
   as the trend rule. If trend beats random, the *timing* (which days) adds value beyond
   mere exposure reduction.

The headline questions: does the trend overlay deliver (a) a **better excess-of-cash
Sharpe** and (b) **much shallower drawdowns** than buy-and-hold gold, and does the return
difference clear an HAC *t* >= 2 with a bootstrap Sharpe CI clear of zero, holding across
sub-eras and surviving realistic ETF costs? A rule can slash drawdown (by ducking the dead
decade) while barely moving Sharpe — we call that out honestly.

Returns are **simple** (arithmetic) so the binary in/out portfolio return is exactly
``signal * r_gold + (1 - signal) * r_cash`` and the wealth path compounds correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Indicators & signal
# --------------------------------------------------------------------------- #
def sma(close: pd.Series, n: int) -> pd.Series:
    """Simple moving average, requiring a full ``n``-observation window (no partials)."""
    return close.rolling(n, min_periods=n).mean()


def trend_signal(close: pd.Series, sma_n: int = 200) -> pd.Series:
    """Daily binary trend signal: 1 when close > SMA(n) [hold gold], 0 [go to cash].

    The signal on day ``t`` uses only data through ``t``; it is lagged one day so it is
    acted on at ``t+1`` (the standard tactical-rule no-look-ahead convention). Returns a
    {0, 1} series aligned to ``close.index``, NaN through the warmup.
    """
    above = (close > sma(close, sma_n)).astype(float)
    return above.shift(1).rename("signal")


def random_signal(signal: pd.Series, seed: int = 0) -> pd.Series:
    """Null control: same in-market *frequency* as ``signal`` but on random days.

    If the trend rule's Sharpe/drawdown edge were purely from *reduced exposure*, this
    exposure-matched random control would show the same edge. If trend beats random, the
    *timing* (which days it is out) is what adds value.
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
# Backtest engine (simple returns; binary gold/cash with switching cost)
# --------------------------------------------------------------------------- #
def run_backtest(
    gold: pd.Series,
    cash: pd.Series,
    signal: pd.Series,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Apply a binary gold/cash signal to daily total-return closes.

    Parameters
    ----------
    gold, cash:
        Daily total-return **close levels** (adjusted) of the gold ETF and the cash
        (T-bill) leg, aligned on a common index.
    signal:
        A {0, 1} series (already lagged): 1 = hold gold, 0 = hold cash.
    cost_bps:
        One-way transaction cost in bps, charged each time the signal flips (a switch
        gold->cash or cash->gold). Default 5 bps one-way ≈ generous for a penny-spread
        ETF like GLD.

    Returns a frame with ``r_gold`` (buy-and-hold), ``r_cash``, ``signal``, ``r_strategy``
    (net of switch costs), and ``r_random`` slots filled by the caller. Simple returns.
    """
    common = gold.index.intersection(cash.index)
    g = gold.reindex(common).sort_index()
    c = cash.reindex(common).sort_index()
    r_gold = g.pct_change().rename("r_gold")
    r_cash = c.pct_change().rename("r_cash")
    sig = signal.reindex(common).astype(float)

    switches = sig.diff().abs().fillna(0.0)
    cost = cost_bps * 1e-4
    r_strat = (sig * r_gold + (1.0 - sig) * r_cash - switches * cost).rename("r_strategy")

    out = pd.concat([r_gold, r_cash, sig.rename("signal"), r_strat], axis=1)
    return out.dropna()


# --------------------------------------------------------------------------- #
# Inference primitives (mirror of Study 803)
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
    gamma0 = float(u @ u) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        cov = float(u[l:] @ u[:-l]) / n
        var += 2.0 * w * cov
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Performance summary (simple daily returns)
# --------------------------------------------------------------------------- #
def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Headline annualised stats for a daily simple-return series.

    Sharpe (of the series as given — pass an *excess-of-cash* series for the excess
    Sharpe), annualised vol, CAGR (geometric), max drawdown, and the HAC *t* on the mean.
    """
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    mu = r.mean()
    std = r.std(ddof=1)
    sharpe = float(mu / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    vol_ann = float(std * np.sqrt(periods_per_year))

    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    cagr = float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and wealth.iloc[-1] > 0 else float("nan")
    dd = float((wealth / wealth.cummax() - 1.0).min())

    return {
        "n_days": int(n),
        "cagr": cagr,
        "sharpe": sharpe,
        "vol_ann": vol_ann,
        "max_drawdown": dd,
        "mean_daily_bps": float(mu * 1e4),
        "tstat": newey_west_t(r.to_numpy(), lags=int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))),
    }


def max_drawdown(returns: pd.Series) -> float:
    r = pd.Series(returns).astype(float).dropna()
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def sharpe_diff_tstat(r1: pd.Series, r2: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """HAC *t* on the daily return *difference* r1 − r2 (does arm 1 out-risk-adjust arm 2?).

    Because both arms are measured excess-of-cash, the cash leg cancels in the difference,
    so ``r1 − r2`` is simply strategy-minus-buy-and-hold — the Jobson-Korkie return-diff
    test in its Newey-West form. |*t*| >= 2 = a reliably higher risk-adjusted return here.
    """
    diff = (r1 - r2).dropna()
    if len(diff) < 3:
        return float("nan")
    return newey_west_t(diff.to_numpy(), lags=int(np.floor(4.0 * (len(diff) / 100.0) ** (2.0 / 9.0))))


# --------------------------------------------------------------------------- #
# The excess-vs-excess race (the headline)
# --------------------------------------------------------------------------- #
def compare(
    gold: pd.Series,
    cash: pd.Series,
    sma_n: int = 200,
    cost_bps: float = 5.0,
    random_seed: int = 912,
) -> dict:
    """Race buy-and-hold gold, the 200-day trend overlay, and the random control.

    Every leg is reported **excess-of-cash** (minus the BIL total-return leg) so the
    Sharpe race is apples-to-apples. Returns a dict of per-leg summaries (on the excess
    series), the excess-Sharpe advantage (trend − buy&hold), the HAC *t* on the return
    difference, the in-market fraction, and the switch count.
    """
    sig = trend_signal(gold, sma_n=sma_n)
    rand = random_signal(sig, seed=random_seed)
    bt = run_backtest(gold, cash, sig, cost_bps=cost_bps)
    bt_rand = run_backtest(gold, cash, rand, cost_bps=cost_bps)

    # Excess-of-cash series (cash leg subtracted from every arm).
    e_bh = (bt["r_gold"] - bt["r_cash"]).rename("e_bh")
    e_trend = (bt["r_strategy"] - bt["r_cash"]).rename("e_trend")
    e_rand = (bt_rand["r_strategy"] - bt_rand["r_cash"]).rename("e_rand")

    s_bh = summary(e_bh)
    s_trend = summary(e_trend)
    s_rand = summary(e_rand)

    # Absolute (not excess) drawdown & CAGR are what an investor actually lives through.
    dd_bh = max_drawdown(bt["r_gold"])
    dd_trend = max_drawdown(bt["r_strategy"])
    dd_rand = max_drawdown(bt_rand["r_strategy"])

    in_frac = float(sig.reindex(bt.index).mean())
    n_switches = int(sig.reindex(bt.index).diff().abs().fillna(0.0).sum())

    return {
        "bh": s_bh, "trend": s_trend, "random": s_rand,
        "dd_bh_abs": dd_bh, "dd_trend_abs": dd_trend, "dd_rand_abs": dd_rand,
        "excess_sharpe_adv": s_trend["sharpe"] - s_bh["sharpe"],
        "sharpe_adv_vs_random": s_trend["sharpe"] - s_rand["sharpe"],
        "t_diff_vs_bh": sharpe_diff_tstat(e_trend, e_bh),
        "t_diff_vs_random": sharpe_diff_tstat(e_trend, e_rand),
        "in_market_frac": in_frac,
        "n_switches": n_switches,
        "e_bh": e_bh, "e_trend": e_trend, "e_rand": e_rand,
        "bt": bt,
    }


# --------------------------------------------------------------------------- #
# Bootstrap Sharpe CI (circular block bootstrap on the excess series)
# --------------------------------------------------------------------------- #
def bootstrap_sharpe_ci(
    excess: pd.Series,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 912,
    periods_per_year: int = TRADING_DAYS,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the annualised Sharpe of an excess-return series.

    Blocks of ``block`` consecutive days preserve vol clustering. Returns the point
    Sharpe, the (1-alpha) percentile interval, and the share of resamples with a negative
    Sharpe — a blunt "could this be zero?" read.
    """
    r = np.asarray(pd.Series(excess).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"sharpe": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": n}
    ann = np.sqrt(periods_per_year)
    point = float(r.mean() / r.std(ddof=1) * ann) if r.std(ddof=1) > 0 else float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.full(n_boot, np.nan)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        s = r[idx]
        sd = s.std(ddof=1)
        if sd > 0:
            boots[b] = s.mean() / sd * ann
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "sharpe": point, "ci_low": float(lo), "ci_high": float(hi),
        "frac_negative": float((valid < 0).mean()), "n_obs": n, "n_boot": int(valid.size),
        "block": block,
    }


# --------------------------------------------------------------------------- #
# Era cut & calendar-year table
# --------------------------------------------------------------------------- #
def era_cut(
    gold: pd.Series,
    cash: pd.Series,
    split: str = "2016-01-01",
    sma_n: int = 200,
    cost_bps: float = 5.0,
) -> dict:
    """Split the sample at ``split`` and re-run the excess-Sharpe race on each half.

    A real edge should show a positive excess-Sharpe advantage in *both* halves; a lucky
    one shows up in only one.
    """
    out = {}
    for tag, sl in [("early", slice(None, split)), ("late", slice(split, None))]:
        g = gold.loc[sl]
        c = cash.loc[sl]
        if len(g) < sma_n + 60:
            out[tag] = None
            continue
        cmp = compare(g, c, sma_n=sma_n, cost_bps=cost_bps)
        out[tag] = {
            "n_days": cmp["trend"]["n_days"],
            "sharpe_bh": cmp["bh"]["sharpe"],
            "sharpe_trend": cmp["trend"]["sharpe"],
            "excess_sharpe_adv": cmp["excess_sharpe_adv"],
            "t_diff_vs_bh": cmp["t_diff_vs_bh"],
            "dd_bh_abs": cmp["dd_bh_abs"],
            "dd_trend_abs": cmp["dd_trend_abs"],
        }
    return out


def calendar_year_table(gold: pd.Series, cash: pd.Series,
                        sma_n: int = 200, cost_bps: float = 5.0) -> pd.DataFrame:
    """Per-calendar-year total return (%) for buy-and-hold gold vs the trend overlay.

    Both are *absolute* (not excess) simple-compounded returns — the lived experience,
    where the dead decade shows as red buy-and-hold years the overlay sidesteps.
    """
    sig = trend_signal(gold, sma_n=sma_n)
    bt = run_backtest(gold, cash, sig, cost_bps=cost_bps)
    yr = bt.index.year
    rows = []
    for y in sorted(set(yr)):
        m = bt.index.year == y
        bh = float((1.0 + bt["r_gold"][m]).prod() - 1.0)
        tr = float((1.0 + bt["r_strategy"][m]).prod() - 1.0)
        frac = float(bt["signal"][m].mean())
        rows.append({"year": int(y), "bh_pct": bh * 100, "trend_pct": tr * 100,
                     "in_gold_frac": frac})
    return pd.DataFrame(rows).set_index("year")


# --------------------------------------------------------------------------- #
# The costed timer
# --------------------------------------------------------------------------- #
def timer_stats(gold: pd.Series, cash: pd.Series, sma_n: int = 200,
                cost_bps_grid=(0.0, 1.0, 5.0, 10.0, 25.0)) -> list[dict]:
    """Cost sweep: the excess-Sharpe advantage and drawdown at several one-way costs.

    The trend rule is low-turnover (a handful of switches per year), so friction is
    charged only on the (few) switch days: one-way cost × NAV per flip. No short leg → no
    borrow. Reports gross-to-net so the honest tradability picture is explicit.
    """
    out = []
    for c in cost_bps_grid:
        cmp = compare(gold, cash, sma_n=sma_n, cost_bps=c)
        out.append({
            "cost_bps": c,
            "sharpe_trend": cmp["trend"]["sharpe"],
            "excess_sharpe_adv": cmp["excess_sharpe_adv"],
            "t_diff_vs_bh": cmp["t_diff_vs_bh"],
            "dd_trend_abs": cmp["dd_trend_abs"],
            "cagr_trend": cmp["trend"]["cagr"],
            "n_switches": cmp["n_switches"],
        })
    return out


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof — never supports the stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, sma_n: int = 200, cost_bps: float = 5.0) -> dict:
    """Run the excess-Sharpe race on a synthetic (asset, cash) tape.

    On the planted dead-decade world the trend overlay should cut drawdown and lift the
    excess Sharpe; on the flat-vol null it should do neither. Proves the machinery is
    unbiased — it never supports a real-tape stamp.
    """
    cmp = compare(prices["asset"], prices["cash"], sma_n=sma_n, cost_bps=cost_bps)
    return {
        "excess_sharpe_adv": cmp["excess_sharpe_adv"],
        "sharpe_bh": cmp["bh"]["sharpe"],
        "sharpe_trend": cmp["trend"]["sharpe"],
        "t_diff_vs_bh": cmp["t_diff_vs_bh"],
        "dd_bh_abs": cmp["dd_bh_abs"],
        "dd_trend_abs": cmp["dd_trend_abs"],
        "dd_rand_abs": cmp["dd_rand_abs"],
        "dd_improvement": cmp["dd_trend_abs"] - cmp["dd_bh_abs"],
        "in_market_frac": cmp["in_market_frac"],
        "n_days": cmp["trend"]["n_days"],
    }
