"""Strategy + inference for Study 896 — Risk-Parity + Trend.

Two books on the same four sleeves (SPY / TLT / GLD / DBC), rebalanced **monthly**,
with exactly one execution lag (weights and gates are formed on information known at the
close of the prior day):

* **Plain risk-parity.** Each sleeve gets an **inverse-volatility** weight from its
  trailing ``lookback``-day realized vol (normalised to sum to 1); fully invested in the
  four sleeves, held through the month. This is study 68's All-Weather book.

* **Risk-parity + trend.** The SAME inverse-vol risk budget, but each sleeve carries a
  **200-day trend gate**: hold the sleeve only while its price is above its 200-day
  moving average (known at the prior close); otherwise that sleeve's weight sits in
  **cash** (BIL T-bills) for the month. The un-gated sleeves keep their exact plain-RP
  weight — trend never re-levers into the survivors, it only steps a falling sleeve out
  to cash.

The races are **excess-of-cash on both legs** (each book's daily return minus the BIL
T-bill return), so the risk-free convention cancels and the Sharpe comparison is
excess-vs-excess. Because ``sum_i w_i = 1``, plain-RP excess ``= sum_i w_i (r_i - cash)``
and RP+trend excess ``= sum_i w_i * gate_i * (r_i - cash)`` — the gate simply zeroes a
sleeve's excess contribution while it is below trend.

Inference (the desk bar):
  * **excess-vs-excess Sharpe advantage** (RP+trend minus plain RP) with a **paired
    block-bootstrap CI** on the Sharpe difference;
  * **HAC (Newey-West) t** on the daily excess-return difference (the return leg);
  * **max drawdown** on the true (cash-inclusive) NAV, a calendar-year table, and a
    two-era robustness cut;
  * a **costed** version (one-way bps x turnover x NAV per monthly rebalance; the books
    are long-or-cash, so there is no borrow leg);
  * a **shuffled-gate placebo** (is the drawdown relief genuine timing or just holding
    less?) and a seeded **synthetic control** (null must not improve, planted trend
    world must light up).

Distinct from [68-all-weather](../../68-all-weather/) (plain inverse-vol RP, no trend),
[110-faber-timing](../../110-faber-timing/) (a single-asset 10-month SMA timer),
[595-managed-futures-allocation](../../595-managed-futures-allocation/) and
[894-trend-6040](../../894-trend-6040/) (trend on a 60/40, not on a risk-parity budget).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
LOOKBACK = 60          # trailing window for the inverse-vol risk budget (days)
SMA = 200              # trend-gate moving average (days)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns (index=date, columns=ticker)."""
    return prices.sort_index().pct_change()


def inverse_vol_weights(vol: pd.DataFrame) -> pd.DataFrame:
    """Row-wise inverse-volatility (risk-parity) weights, normalised to sum to 1.

    ``vol`` is a frame of trailing realized vols (index=date, columns=sleeve); rows with
    all-NaN vol return all-NaN (burn-in). Vectorised — no per-date loop."""
    iv = 1.0 / vol.replace(0.0, np.nan)
    return iv.div(iv.sum(axis=1), axis=0)


def trend_gate(prices: pd.DataFrame, window: int = SMA) -> pd.DataFrame:
    """1.0 where price is at/above its ``window``-day SMA, else 0.0 (the sleeve is
    de-risked to cash). NaN during the SMA burn-in."""
    sma = prices.rolling(window, min_periods=window).mean()
    gate = (prices >= sma).astype(float)
    return gate.where(sma.notna())


def _month_start_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    """Boolean array: True on the first trading day of each calendar month."""
    mk = idx.year.to_numpy() * 12 + idx.month.to_numpy()
    return np.concatenate([[True], mk[1:] != mk[:-1]])


# --------------------------------------------------------------------------- #
# The backtest (monthly rebalance, one execution lag, vectorised)
# --------------------------------------------------------------------------- #
def backtest(prices: pd.DataFrame, returns: pd.DataFrame, cash: pd.Series,
             sleeves: list[str], use_trend: bool, lookback: int = LOOKBACK,
             sma: int = SMA, cost_bps: float = 0.0) -> dict:
    """Daily total & excess-of-cash returns for one book (plain RP or RP+trend).

    At the first trading day of each month the effective sleeve weight is
    ``inverse_vol(trailing lookback vol)`` (times the 200d trend gate if ``use_trend``),
    both evaluated on information known at the **prior** close (one ``shift(1)`` lag),
    then held through the month. The complement ``1 - sum_i eff_i`` sits in cash.

    Costs: ``cost_bps`` one-way x turnover (sum of \\|Δ effective sleeve weight\\| at each
    rebalance) x NAV, charged on the rebalance day; cash moves are free and there is no
    short/borrow leg. Returns a dict with the daily ``total`` and ``excess`` series, the
    effective-weight frame, average risky exposure and annualised turnover.
    """
    px, ret = prices[sleeves], returns[sleeves]
    idx = ret.index

    vol = ret.rolling(lookback, min_periods=lookback).std(ddof=1).shift(1)
    w = inverse_vol_weights(vol)
    if use_trend:
        gate = trend_gate(px, sma).shift(1)
        eff_full = w * gate
    else:
        eff_full = w

    reb = _month_start_mask(idx)
    eff = eff_full.where(pd.Series(reb, index=idx), axis=0)
    eff = eff.ffill()

    valid = eff.notna().all(axis=1)
    eff = eff[valid]
    r = ret.loc[eff.index]
    c = cash.reindex(eff.index).fillna(0.0)

    risky = (eff * r).sum(axis=1)
    cash_w = 1.0 - eff.sum(axis=1)
    total = risky + cash_w * c

    # turnover / costs: |Δ effective sleeve weight| on rebalance days only
    reb_days = pd.Series(_month_start_mask(eff.index), index=eff.index)
    d_eff = eff.diff().abs().sum(axis=1)
    d_eff.iloc[0] = 0.0                       # entry trade excluded (both books pay it)
    turn = d_eff.where(reb_days, 0.0)
    cost = (cost_bps / 1e4) * turn
    total = total - cost
    excess = total - c

    n_reb = int(reb_days.sum())
    years = len(eff) / TRADING_DAYS
    return {
        "total": total, "excess": excess, "eff": eff, "cash_w": cash_w,
        "avg_risky": float(eff.sum(axis=1).mean()),
        "avg_gate": float((eff > 0).sum(axis=1).mean() / len(sleeves)) if use_trend else 1.0,
        "turnover_ann": float(turn.sum() / years) if years > 0 else float("nan"),
        "n_days": int(len(eff)), "n_reb": n_reb,
    }


# --------------------------------------------------------------------------- #
# Performance summary
# --------------------------------------------------------------------------- #
def max_drawdown(total: pd.Series | np.ndarray) -> float:
    nav = np.cumprod(1.0 + np.asarray(total, float))
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1.0).min())


def sharpe(excess: pd.Series | np.ndarray, ann: int = TRADING_DAYS) -> float:
    x = np.asarray(excess, float)
    x = x[np.isfinite(x)]
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(ann)) if sd > 0 else float("nan")


def perf(total: pd.Series, excess: pd.Series, ann: int = TRADING_DAYS) -> dict:
    """CAGR & vol on the true (cash-inclusive) NAV; Sharpe on the excess-of-cash series;
    max drawdown on the true NAV; terminal wealth multiple."""
    t = np.asarray(total, float); t = t[np.isfinite(t)]
    e = np.asarray(excess, float); e = e[np.isfinite(e)]
    lg = np.log1p(t)
    return {
        "cagr_pct": float(np.exp(lg.mean() * ann) - 1.0) * 100.0,
        "vol_ann_pct": float(t.std(ddof=1) * np.sqrt(ann)) * 100.0,
        "sharpe": sharpe(e, ann),
        "exc_ann_pct": float(e.mean() * ann) * 100.0,
        "maxdd_pct": max_drawdown(t) * 100.0,
        "wealth_mult": float(np.exp(lg.sum())),
        "n": int(len(t)), "years": len(t) / ann,
    }


# --------------------------------------------------------------------------- #
# Inference — HAC, bootstrap, one-sample
# --------------------------------------------------------------------------- #
def nw_lags(n: int) -> int:
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def hac_tstat(x: np.ndarray, lags: int | None = None) -> dict:
    """HAC (Newey-West, Bartlett) t of the mean of a serially-correlated series."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return {"mean": float("nan"), "t": float("nan"), "n": n, "lags": 0}
    if lags is None:
        lags = nw_lags(n)
    mu = x.mean(); e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        lrv += 2.0 * (1.0 - k / (lags + 1.0)) * (float(e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(lrv, 0.0) / n)
    return {"mean": float(mu), "t": float(mu / se) if se > 0 else float("nan"),
            "n": n, "lags": lags}


def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def sharpe_diff_bootstrap(exc_a: pd.Series, exc_b: pd.Series, n_boot: int = 2000,
                          block: int = 21, seed: int = 896,
                          ann: int = TRADING_DAYS) -> dict:
    """Paired circular block-bootstrap CI for the Sharpe **difference** (A − B).

    A and B are the two books' excess-of-cash return series on the SAME days; blocks of
    ``block`` consecutive paired observations are resampled (preserving serial
    dependence and the cross-book pairing) and the Sharpe difference recomputed. Reports
    the observed difference and the 2.5/97.5 percentile CI plus P(diff > 0)."""
    a = np.asarray(exc_a, float); b = np.asarray(exc_b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    obs = sharpe(a, ann) - sharpe(b, ann)
    if n < block + 2:
        return {"obs": obs, "lo": float("nan"), "hi": float("nan"),
                "p_gt0": float("nan"), "n_boot": 0}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        idx = idx[:n]
        sa = a[idx]; sb = b[idx]
        da = sa.std(ddof=1); db = sb.std(ddof=1)
        diffs[i] = (sa.mean() / da if da > 0 else 0.0) * np.sqrt(ann) \
            - (sb.mean() / db if db > 0 else 0.0) * np.sqrt(ann)
    return {"obs": float(obs), "lo": float(np.percentile(diffs, 2.5)),
            "hi": float(np.percentile(diffs, 97.5)),
            "p_gt0": float((diffs > 0).mean()), "n_boot": n_boot}


# --------------------------------------------------------------------------- #
# The head-to-head race
# --------------------------------------------------------------------------- #
def race(prices: pd.DataFrame, returns: pd.DataFrame, cash: pd.Series,
         sleeves: list[str], lookback: int = LOOKBACK, sma: int = SMA,
         cost_bps: float = 0.0, ann: int = TRADING_DAYS) -> dict:
    """Plain RP vs RP+trend on the same post-burn-in sample: perf of both legs, the
    excess-vs-excess Sharpe advantage, and the HAC t on the daily excess-return diff."""
    plain = backtest(prices, returns, cash, sleeves, use_trend=False,
                     lookback=lookback, sma=sma, cost_bps=cost_bps)
    trend = backtest(prices, returns, cash, sleeves, use_trend=True,
                     lookback=lookback, sma=sma, cost_bps=cost_bps)
    common = plain["excess"].index.intersection(trend["excess"].index)
    pe, te = plain["excess"].loc[common], trend["excess"].loc[common]
    pt, tt = plain["total"].loc[common], trend["total"].loc[common]
    pp, tp = perf(pt, pe, ann), perf(tt, te, ann)
    d = (te - pe).to_numpy()
    hd = hac_tstat(d)
    return {
        "plain": pp, "trend": tp,
        "sharpe_adv": tp["sharpe"] - pp["sharpe"],
        "dd_relief_pp": tp["maxdd_pct"] - pp["maxdd_pct"],   # trend − plain (>0 = shallower)
        "ret_diff_ann_pct": hd["mean"] * ann * 100.0,        # excess_trend − excess_plain
        "t_ret_diff": hd["t"],
        "avg_gate_trend": trend["avg_gate"], "avg_risky_trend": trend["avg_risky"],
        "turnover_plain": plain["turnover_ann"], "turnover_trend": trend["turnover_ann"],
        "n_days": len(common), "years": len(common) / ann,
        "plain_excess": pe, "trend_excess": te,
        "plain_total": pt, "trend_total": tt,
    }


# --------------------------------------------------------------------------- #
# Calendar-year table + era cut
# --------------------------------------------------------------------------- #
def calendar_years(r: dict) -> pd.DataFrame:
    """Per-calendar-year total return (%) of plain RP, RP+trend and the gap."""
    pt, tt = r["plain_total"], r["trend_total"]
    def yr(s):
        return s.groupby(s.index.year).apply(lambda x: (1.0 + x).prod() - 1.0) * 100.0
    out = pd.DataFrame({"plain_%": yr(pt), "trend_%": yr(tt)})
    out["gap_pp"] = out["trend_%"] - out["plain_%"]
    return out.round(2)


def era_cut(prices: pd.DataFrame, returns: pd.DataFrame, cash: pd.Series,
            sleeves: list[str], lookback: int = LOOKBACK, sma: int = SMA,
            cost_bps: float = 0.0, ann: int = TRADING_DAYS) -> list[dict]:
    """Split the post-burn-in sample in half by time; race each era."""
    full = race(prices, returns, cash, sleeves, lookback, sma, cost_bps, ann)
    idx = full["plain_excess"].index
    mid = idx[len(idx) // 2]
    out = []
    for name, lo, hi in (("era 1", idx[0], idx[len(idx) // 2 - 1]),
                         ("era 2", mid, idx[-1])):
        m = (prices.index >= lo) & (prices.index <= hi)
        # re-race on the sliced tape (burn-in re-consumed inside each era is avoided by
        # slicing the already-formed excess series instead)
        pe = full["plain_excess"].loc[lo:hi]; te = full["trend_excess"].loc[lo:hi]
        pt = full["plain_total"].loc[lo:hi]; tt = full["trend_total"].loc[lo:hi]
        out.append({
            "era": name, "start": lo.date().isoformat(), "end": hi.date().isoformat(),
            "sharpe_plain": sharpe(pe, ann), "sharpe_trend": sharpe(te, ann),
            "sharpe_adv": sharpe(te, ann) - sharpe(pe, ann),
            "maxdd_plain_pct": max_drawdown(pt) * 100.0,
            "maxdd_trend_pct": max_drawdown(tt) * 100.0,
            "t_ret_diff": hac_tstat((te - pe).to_numpy())["t"],
            "n_days": len(pe),
        })
    return out


# --------------------------------------------------------------------------- #
# Cost sweep
# --------------------------------------------------------------------------- #
def cost_sweep(prices: pd.DataFrame, returns: pd.DataFrame, cash: pd.Series,
               sleeves: list[str], costs=(0.0, 5.0, 10.0, 20.0),
               lookback: int = LOOKBACK, sma: int = SMA,
               ann: int = TRADING_DAYS) -> list[dict]:
    out = []
    for cb in costs:
        r = race(prices, returns, cash, sleeves, lookback, sma, cb, ann)
        out.append({
            "cost_bps": cb,
            "sharpe_plain": r["plain"]["sharpe"], "sharpe_trend": r["trend"]["sharpe"],
            "sharpe_adv": r["sharpe_adv"],
            "maxdd_plain_pct": r["plain"]["maxdd_pct"],
            "maxdd_trend_pct": r["trend"]["maxdd_pct"],
            "cagr_plain_pct": r["plain"]["cagr_pct"], "cagr_trend_pct": r["trend"]["cagr_pct"],
            "t_ret_diff": r["t_ret_diff"],
        })
    return out


# --------------------------------------------------------------------------- #
# Placebo — is the drawdown relief timing, or just holding less?
# --------------------------------------------------------------------------- #
def placebo_shuffle(prices: pd.DataFrame, returns: pd.DataFrame, cash: pd.Series,
                    sleeves: list[str], n_seeds: int = 200, base_seed: int = 896,
                    lookback: int = LOOKBACK, sma: int = SMA,
                    ann: int = TRADING_DAYS) -> dict:
    """Keep the trend gate's on/off *frequency* per sleeve but destroy its **timing**:
    permute each sleeve's monthly gate across rebalance dates. If the real gate's
    drawdown/Sharpe sit inside the shuffled cloud, the relief was mere exposure
    reduction, not timing. Reports observed vs placebo mean and the p-values."""
    obs = race(prices, returns, cash, sleeves, lookback, sma, 0.0, ann)
    px, ret = prices[sleeves], returns[sleeves]
    idx = ret.index
    vol = ret.rolling(lookback, min_periods=lookback).std(ddof=1).shift(1)
    w = inverse_vol_weights(vol)
    gate = trend_gate(px, sma).shift(1)
    reb = _month_start_mask(idx)
    reb_idx = idx[reb]
    # monthly gate values (on rebalance days) per sleeve
    g_reb = gate.reindex(reb_idx)

    adv, dds = [], []
    c_full = cash.reindex(idx).fillna(0.0)
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        g_sh = g_reb.copy()
        for col in sleeves:
            vals = g_sh[col].to_numpy().copy()
            fin = np.isfinite(vals)
            perm = vals[fin].copy()
            rng.shuffle(perm)
            vals[fin] = perm
            g_sh[col] = vals
        g_daily = g_sh.reindex(idx).ffill()
        eff = (w * g_daily)
        eff = eff.where(pd.Series(reb, index=idx), axis=0).ffill()
        valid = eff.notna().all(axis=1)
        eff = eff[valid]
        r_ = ret.loc[eff.index]; c_ = c_full.loc[eff.index]
        total = (eff * r_).sum(axis=1) + (1.0 - eff.sum(axis=1)) * c_
        exc = total - c_
        adv.append(sharpe(exc, ann) - obs["plain"]["sharpe"])
        dds.append(max_drawdown(total) * 100.0)
    adv, dds = np.asarray(adv), np.asarray(dds)
    return {
        "obs_sharpe_adv": obs["sharpe_adv"], "obs_maxdd_pct": obs["trend"]["maxdd_pct"],
        "placebo_mean_adv": float(adv.mean()), "placebo_mean_maxdd_pct": float(dds.mean()),
        "p_sharpe": float((adv >= obs["sharpe_adv"]).mean()),
        "p_dd": float((dds >= obs["trend"]["maxdd_pct"]).mean()),   # as shallow as observed
        "n_seeds": n_seeds, "adv_draws": adv, "dd_draws": dds,
    }


# --------------------------------------------------------------------------- #
# Synthetic control — averaged over seeds (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_check(edge: float, n_seeds: int = 20, n_days: int = 5000,
                    base_seed: int = 896, lookback: int = LOOKBACK, sma: int = SMA,
                    ann: int = TRADING_DAYS) -> dict:
    """Run the full plain-vs-trend race on ``n_seeds`` independent synthetic worlds and
    average the Sharpe advantage, the drawdown relief and the HAC t of the return diff.
    ``edge = 0`` is the null (trend must not help); ``edge > 0`` plants sustained bear
    grinds a 200-day gate should step out of."""
    from . import data as d
    advs, ddr, ts = [], [], []
    for s in range(n_seeds):
        px, ret, cash = d.synthetic_world(edge=edge, seed=base_seed + 1000 * s,
                                          n_days=n_days)
        sleeves = list(ret.columns)
        r = race(px, ret, cash, sleeves, lookback, sma, 0.0, ann)
        advs.append(r["sharpe_adv"]); ddr.append(r["dd_relief_pp"]); ts.append(r["t_ret_diff"])
    advs, ddr, ts = np.asarray(advs), np.asarray(ddr), np.asarray(ts)
    return {
        "mean_sharpe_adv": float(advs.mean()), "sd_sharpe_adv": float(advs.std(ddof=1)),
        "mean_dd_relief_pp": float(ddr.mean()),
        "mean_t_ret_diff": float(ts.mean()),
        "share_adv_pos": float((advs > 0).mean()), "n_seeds": n_seeds,
    }
