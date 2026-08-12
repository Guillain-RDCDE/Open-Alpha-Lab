"""Strategy + inference for Study 894 — Trend Overlay on 60/40.

The rule under test. Take the balanced book — **60% SPY / 40% IEF** — and lay a 200-day
trend filter over *each leg independently*: hold the equity sleeve when SPY's close is
above its 200-day moving average, else park that 60% in cash (BIL); hold the bond sleeve
when IEF is above its own 200-day MA, else park that 40% in cash. The target weights stay
60/40; only the *in-market vs cash* state of each leg moves. Re-evaluated daily.

Two comparison arms, graded **excess-of-cash vs excess-of-cash** (both minus the BIL
T-bill leg, so a fair race that strips out the level of short rates):

1. **static_6040** — the benchmark: always 60% SPY + 40% IEF, rebalanced to weight
   (idealised, costless).
2. **trend_overlay** — the rule: each leg in its asset when above its 200-day MA, in BIL
   otherwise; charged one-way switching costs on the fraction of NAV that flips, plus an
   optional short-term-gains **tax drag** on the gains realised when a leg is forced to
   cash (a trend overlay's real-world friction that a buy-and-hold book never pays).

No look-ahead: the signal on day *t* uses closes through *t* and is acted on at *t+1*
(one documented ``shift``). The headline question is whether the overlay **cuts drawdown
while keeping most of the return** — a real excess-vs-excess Sharpe advantage with a HAC
*t* >= 2 on the return difference that a bootstrap CI keeps clear of zero and that holds
across eras — or whether it merely trades return for calm (the usual trend-overlay
outcome), or gets eaten by switching costs and tax.

Inference primitives (``newey_west_t`` / ``one_sample_t`` / ``welch_t`` /
``wilson_interval`` / ``sharpe_ci_bootstrap`` / ``timer``) mirror the desk house style.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Indicator + signal
# --------------------------------------------------------------------------- #
def moving_average(close: pd.Series, n: int = 200) -> pd.Series:
    """Simple moving average, ``n`` full observations required (no partial windows)."""
    return close.rolling(n, min_periods=n).mean()


def trend_signal(close: pd.Series, n: int = 200) -> pd.Series:
    """Daily {0,1} trend signal, **lagged one day** (no look-ahead).

    1 when the close is at/above its ``n``-day SMA (stay invested), 0 when below (step to
    cash), **NaN while the MA is still warming up** (fewer than ``n`` observations) so the
    caller can drop that undefined period rather than default it into cash. The signal
    formed at the close of day *t* is acted on at *t+1* (``shift(1)``).
    """
    ma = moving_average(close, n)
    above = (close >= ma).astype(float).where(ma.notna())  # NaN until the MA is defined
    return above.shift(1).rename(f"sig_{close.name}")


# --------------------------------------------------------------------------- #
# Returns + the two arms
# --------------------------------------------------------------------------- #
def simple_returns(px: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns per column (index=date, columns=ticker)."""
    return px.sort_index().pct_change()


def static_6040(
    ret: pd.DataFrame, w_spy: float = 0.60, spy: str = "SPY", ief: str = "IEF"
) -> pd.Series:
    """Benchmark: constant-weight 60/40 daily return (rebalanced to weight, costless)."""
    return (w_spy * ret[spy] + (1.0 - w_spy) * ret[ief]).rename("r_bench")


def trend_overlay(
    px: pd.DataFrame,
    w_spy: float = 0.60,
    ma_n: int = 200,
    cost_bps: float = 3.0,
    tax_rate: float = 0.0,
    spy: str = "SPY",
    ief: str = "IEF",
    cash: str = "BIL",
) -> pd.DataFrame:
    """Run the trend-overlaid 60/40 and the static 60/40 side by side on a price panel.

    Each leg is invested in its asset when the (lagged) 200-day trend signal is 1, else in
    ``cash`` (BIL). Target weights are ``w_spy`` / ``1-w_spy``. Costs are charged only on
    the *incremental* trading the overlay does — the fraction of NAV that flips in or out —
    at ``cost_bps`` one-way; the benchmark is treated as a costless constant-weight book.

    **Tax drag** (``tax_rate`` > 0): when a leg is forced from its asset to cash, the gain
    accrued over that invested spell is *realised* and taxed at ``tax_rate`` (a short-term
    rate for holdings under a year — the trend overlay's turnover keeps most spells short).
    A buy-and-hold 60/40 defers this tax, so it is a genuine overlay-only friction. Only
    positive realised gains are taxed (losses harvest no cash back in this simple model),
    computed in a single O(n) pass over the (rare) switch events — not a per-date loop.

    Returns a DataFrame with columns:
      ``r_bench``      static 60/40 daily return,
      ``r_gross``      overlay before costs/tax,
      ``r_net``        overlay after switching costs and tax drag,
      ``r_cash``       the BIL daily return (for excess-of-cash races),
      ``sig_spy``/``sig_ief``  the applied {0,1} signals,
      ``w_eq``/``w_bd``        the applied equity/bond weights (0 or target),
      ``turnover``     fraction of NAV traded that day.
    """
    ret = simple_returns(px)
    r_spy, r_ief, r_cash = ret[spy], ret[ief], ret[cash]

    sig_spy_raw = trend_signal(px[spy], ma_n).reindex(px.index)
    sig_ief_raw = trend_signal(px[ief], ma_n).reindex(px.index)
    # Fair-race window: start only once BOTH legs' 200-day MAs are defined, so the
    # overlay is never parked in cash-by-default while the static book is invested.
    valid = sig_spy_raw.notna() & sig_ief_raw.notna()
    first = valid.idxmax() if bool(valid.any()) else px.index[-1]
    sig_spy = sig_spy_raw.fillna(0.0)
    sig_ief = sig_ief_raw.fillna(0.0)

    w_bd = 1.0 - w_spy
    # Per-leg realised return: asset when signal=1, cash when signal=0.
    r_eq_leg = sig_spy * r_spy + (1.0 - sig_spy) * r_cash
    r_bd_leg = sig_ief * r_ief + (1.0 - sig_ief) * r_cash
    r_gross = (w_spy * r_eq_leg + w_bd * r_bd_leg).rename("r_gross")

    r_bench = static_6040(ret, w_spy, spy, ief)

    # --- switching costs: NAV fraction that flips, per leg, per day ----------
    d_eq = sig_spy.diff().abs().fillna(0.0)
    d_bd = sig_ief.diff().abs().fillna(0.0)
    turnover = (w_spy * d_eq + w_bd * d_bd).rename("turnover")
    switch_cost = turnover * (cost_bps * 1e-4)

    # --- tax drag on gains realised when a leg exits to cash -----------------
    tax_eq = _exit_tax(sig_spy.to_numpy(), r_spy.to_numpy(), tax_rate) * w_spy
    tax_bd = _exit_tax(sig_ief.to_numpy(), r_ief.to_numpy(), tax_rate) * w_bd
    tax = pd.Series(tax_eq + tax_bd, index=px.index, name="tax")

    r_net = (r_gross - switch_cost - tax).rename("r_net")

    out = pd.concat(
        [r_bench, r_gross, r_net, r_cash.rename("r_cash"),
         sig_spy.rename("sig_spy"), sig_ief.rename("sig_ief"),
         (sig_spy * w_spy).rename("w_eq"), (sig_ief * w_bd).rename("w_bd"),
         turnover],
        axis=1,
    )
    out = out[out.index >= first]   # drop the MA warm-up so both arms race identically
    return out.dropna()


def _exit_tax(sig: np.ndarray, r_asset: np.ndarray, tax_rate: float) -> np.ndarray:
    """Per-day tax charged when the leg exits its asset (single O(n) pass, no date loop).

    Tracks the compounded gain of the asset while the leg is invested (signal==1); on the
    day the leg flips to cash, if that spell's holding-period return is positive, charge
    ``tax_rate * gain`` (as a fraction of the leg's NAV) on that day. NaNs in returns are
    treated as flat. Returns a per-day tax fraction array aligned to ``sig``.
    """
    n = len(sig)
    tax = np.zeros(n)
    if tax_rate <= 0.0:
        return tax
    hpr = 1.0          # compounded gross of the current invested spell
    invested = False
    for i in range(n):
        s = sig[i]
        ri = r_asset[i]
        ri = 0.0 if not np.isfinite(ri) else ri
        if s >= 0.5:                      # invested today
            if not invested:
                invested = True
                hpr = 1.0
            hpr *= (1.0 + ri)
        else:                             # in cash today
            if invested:                  # we just exited -> realise the spell's gain
                gain = hpr - 1.0
                if gain > 0.0:
                    tax[i] = tax_rate * gain
                invested = False
                hpr = 1.0
    return tax


# --------------------------------------------------------------------------- #
# Inference primitives (house style)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) *t* of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
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
# Performance summary + excess-vs-excess race
# --------------------------------------------------------------------------- #
def _max_drawdown(r: np.ndarray) -> float:
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def summary(returns: pd.Series, rf: pd.Series | float = 0.0,
            periods_per_year: int = TRADING_DAYS) -> dict:
    """Headline annualised stats for a daily return series (excess of ``rf`` for Sharpe).

    ``rf`` is a per-day cash return series (or scalar). Sharpe is computed on the
    **excess** return (r - rf); CAGR / vol / max-drawdown on the raw return.
    """
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if isinstance(rf, pd.Series):
        rfr = rf.reindex(r.index).fillna(0.0)
    else:
        rfr = pd.Series(float(rf), index=r.index)
    ex = (r - rfr).to_numpy()
    raw = r.to_numpy()
    sd_ex = ex.std(ddof=1)
    sharpe = float(ex.mean() / sd_ex * np.sqrt(periods_per_year)) if sd_ex > 0 else float("nan")
    vol = float(raw.std(ddof=1) * np.sqrt(periods_per_year))
    cum = np.prod(1.0 + raw)
    years = n / periods_per_year
    cagr = float(cum ** (1.0 / years) - 1.0) if years > 0 and cum > 0 else float("nan")
    return {
        "n_days": int(n),
        "cagr": cagr,
        "sharpe_excess": sharpe,
        "vol_ann": vol,
        "max_drawdown": _max_drawdown(raw),
        "mean_daily_bps": float(raw.mean() * 1e4),
        "t_nw": newey_west_t(ex, lags=10),
    }


def excess_race(bt: pd.DataFrame, which: str = "r_net",
                periods_per_year: int = TRADING_DAYS) -> dict:
    """Excess-of-cash vs excess-of-cash race: overlay (``which``) vs static 60/40.

    Both arms are taken minus the BIL cash leg (``r_cash``). Reports each excess Sharpe,
    the Sharpe **advantage**, and the HAC *t* / one-sample *t* on the daily return
    difference (overlay − benchmark), whose sign is the excess-vs-excess return edge.
    """
    r_cash = bt["r_cash"]
    strat = summary(bt[which], rf=r_cash, periods_per_year=periods_per_year)
    bench = summary(bt["r_bench"], rf=r_cash, periods_per_year=periods_per_year)
    diff = (bt[which] - bt["r_bench"]).to_numpy()
    return {
        "sharpe_strat": strat["sharpe_excess"],
        "sharpe_bench": bench["sharpe_excess"],
        "sharpe_adv": strat["sharpe_excess"] - bench["sharpe_excess"],
        "maxdd_strat": strat["max_drawdown"],
        "maxdd_bench": bench["max_drawdown"],
        "dd_cut": strat["max_drawdown"] - bench["max_drawdown"],  # +ve = shallower DD
        "cagr_strat": strat["cagr"],
        "cagr_bench": bench["cagr"],
        "vol_strat": strat["vol_ann"],
        "vol_bench": bench["vol_ann"],
        "diff_bps": float(np.nanmean(diff) * 1e4),
        "t_nw_diff": newey_west_t(diff, lags=10),
        "t_1s_diff": one_sample_t(diff),
        "n_days": strat["n_days"],
    }


# --------------------------------------------------------------------------- #
# Bootstrap Sharpe / mean CI (stationary block bootstrap)
# --------------------------------------------------------------------------- #
def sharpe_ci_bootstrap(excess: pd.Series, n_boot: int = 2000, block: int = 20,
                        alpha: float = 0.05, seed: int = 894,
                        periods_per_year: int = TRADING_DAYS) -> dict:
    """Circular-block-bootstrap CI for the annualised Sharpe of an excess-return series."""
    x = np.asarray(pd.Series(excess).astype(float).dropna())
    n = len(x)
    if n < block + 2:
        return {"sharpe": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        s = x[idx[:n]]
        sd = s.std(ddof=1)
        stats[b] = s.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else np.nan
    sd0 = x.std(ddof=1)
    sr = x.mean() / sd0 * np.sqrt(periods_per_year) if sd0 > 0 else float("nan")
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe": float(sr), "lo": float(lo), "hi": float(hi), "n_boot": n_boot}


def sharpe_adv_bootstrap(bt: pd.DataFrame, which: str = "r_gross", n_boot: int = 3000,
                         block: int = 20, alpha: float = 0.05, seed: int = 894,
                         periods_per_year: int = TRADING_DAYS) -> dict:
    """Paired circular-block bootstrap CI for the **excess-Sharpe advantage** (overlay − static).

    Both arms are taken excess of the BIL cash leg and resampled with the *same* block
    indices each draw (paired), so the CI reflects the sampling uncertainty of the Sharpe
    *difference*. ``P(adv>0)`` is the share of draws with a positive advantage — the honest
    read on whether the risk-adjusted edge is distinguishable from zero.
    """
    ex_s = (bt[which] - bt["r_cash"]).to_numpy(dtype=float)
    ex_b = (bt["r_bench"] - bt["r_cash"]).to_numpy(dtype=float)
    m = np.isfinite(ex_s) & np.isfinite(ex_b)
    ex_s, ex_b = ex_s[m], ex_b[m]
    n = len(ex_s)
    if n < block + 2:
        return {"adv": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_pos": float("nan")}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    ann = np.sqrt(periods_per_year)
    advs = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        idx = idx[:n]
        s, bb = ex_s[idx], ex_b[idx]
        sds, sdb = s.std(ddof=1), bb.std(ddof=1)
        ss = s.mean() / sds * ann if sds > 0 else np.nan
        sb = bb.mean() / sdb * ann if sdb > 0 else np.nan
        advs[b] = ss - sb
    sds0, sdb0 = ex_s.std(ddof=1), ex_b.std(ddof=1)
    adv0 = (ex_s.mean() / sds0 - ex_b.mean() / sdb0) * ann if sds0 > 0 and sdb0 > 0 else float("nan")
    lo, hi = np.nanpercentile(advs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"adv": float(adv0), "lo": float(lo), "hi": float(hi),
            "p_pos": float(np.nanmean(advs > 0)), "n_boot": n_boot}


def diff_mean_ci_bootstrap(diff: pd.Series, n_boot: int = 2000, block: int = 20,
                           alpha: float = 0.05, seed: int = 894) -> dict:
    """Block-bootstrap CI for the mean of the daily return difference (bps/day)."""
    x = np.asarray(pd.Series(diff).astype(float).dropna())
    n = len(x)
    if n < block + 2:
        return {"mean_bps": float("nan"), "lo_bps": float("nan"), "hi_bps": float("nan")}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[b] = x[idx[:n]].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean_bps": float(x.mean() * 1e4), "lo_bps": float(lo * 1e4),
            "hi_bps": float(hi * 1e4), "n_boot": n_boot}


# --------------------------------------------------------------------------- #
# Calendar-year + drawdown + era tables
# --------------------------------------------------------------------------- #
def calendar_year_table(bt: pd.DataFrame, which: str = "r_net") -> pd.DataFrame:
    """Per-calendar-year total return for the overlay (``which``) and the static book."""
    strat = bt[which]
    bench = bt["r_bench"]
    grp = strat.groupby(strat.index.year)
    rows = {
        "overlay_%": grp.apply(lambda s: (np.prod(1.0 + s) - 1.0) * 100),
        "static_%": bench.groupby(bench.index.year).apply(lambda s: (np.prod(1.0 + s) - 1.0) * 100),
    }
    tbl = pd.DataFrame(rows)
    tbl["diff_pp"] = tbl["overlay_%"] - tbl["static_%"]
    return tbl


def era_cut(bt: pd.DataFrame, split: str = "2017-01-01", which: str = "r_net") -> dict:
    """Excess-vs-excess race on two eras split at ``split`` — is the effect one era's?"""
    lo = bt[bt.index < pd.Timestamp(split)]
    hi = bt[bt.index >= pd.Timestamp(split)]
    return {
        "split": split,
        "early": excess_race(lo, which) if len(lo) > 260 else None,
        "late": excess_race(hi, which) if len(hi) > 260 else None,
    }


# --------------------------------------------------------------------------- #
# The costed timer — gross vs net across a cost grid
# --------------------------------------------------------------------------- #
def timer(px: pd.DataFrame, w_spy: float = 0.60, ma_n: int = 200,
          cost_grid=(0.0, 1.0, 3.0, 5.0), tax_rate: float = 0.0) -> list[dict]:
    """Overlay excess-Sharpe advantage & DD cut vs the static book across a cost grid.

    Each row re-runs the overlay at one one-way ``cost_bps`` (plus the fixed ``tax_rate``)
    and reports the net excess-vs-excess Sharpe advantage, the drawdown cut, and the HAC
    *t* on the net return difference — the honest test of whether the overlay is bankable.
    """
    rows = []
    for cb in cost_grid:
        bt = trend_overlay(px, w_spy=w_spy, ma_n=ma_n, cost_bps=cb, tax_rate=tax_rate)
        rc = excess_race(bt, which="r_net")
        rows.append({
            "cost_bps": cb, "tax_rate": tax_rate,
            "sharpe_adv": rc["sharpe_adv"], "dd_cut": rc["dd_cut"],
            "diff_bps": rc["diff_bps"], "t_nw_diff": rc["t_nw_diff"],
            "cagr_strat": rc["cagr_strat"], "cagr_bench": rc["cagr_bench"],
        })
    return rows


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(px: pd.DataFrame, w_spy: float = 0.60, ma_n: int = 200,
                     cost_bps: float = 3.0) -> dict:
    """Run the headline excess-vs-excess race on a synthetic price panel (no network)."""
    bt = trend_overlay(px, w_spy=w_spy, ma_n=ma_n, cost_bps=cost_bps, tax_rate=0.0)
    rc = excess_race(bt, which="r_net")
    return {
        "sharpe_adv": rc["sharpe_adv"], "dd_cut": rc["dd_cut"],
        "t_nw_diff": rc["t_nw_diff"], "diff_bps": rc["diff_bps"],
        "maxdd_strat": rc["maxdd_strat"], "maxdd_bench": rc["maxdd_bench"],
        "n_days": rc["n_days"],
    }
