"""The Ivy Portfolio engine and its honest controls — Study 655.

Mebane Faber's "Ivy Portfolio" (2009 book, built on the 2007 SSRN paper *A Quantitative
Approach to Tactical Asset Allocation*): endowment-style diversification for retail — 20% each
across US equity, foreign equity, REITs, bonds and commodities — optionally overlaid with a
**10-month simple-moving-average timer** applied *independently to each sleeve*: hold the asset
when its month-end price is above its trailing 10-month average, otherwise park that 20% in
T-bills.

Two questions, kept separate on purpose:

1. **Does the 5-asset equal-weight mix diversify?** (Sharpe / drawdown vs a 60/40 and vs the
   equity sleeve alone — the "endowment style" claim.)
2. **Does the SMA timer add alpha, or does it just cut risk?** (timed vs static, a
   matched-exposure random-timing control, and the honest split between return and Sharpe.)

Conventions
-----------
- Inputs are **monthly simple total returns** (``data.monthly_panel``), one row per rebalance.
- Both arms rebalance to target weights **every month** — the natural cadence given monthly
  data and the monthly SMA re-check; turnover is tracked explicitly (see ``weighted_portfolio``)
  so the cost/turnover comparison between the static and timed arms is apples-to-apples.
- **Execution lag — exactly one, documented.** The SMA signal uses the price through month
  *t-1*'s close; the resulting position earns month *t*'s return (``sma_signal`` applies the
  shift once, at the source — nowhere else in this module re-shifts).
- Costs: one-way bps x NAV, charged against **total absolute weight change** at each rebalance
  (selling out of a sleeve and buying into cash counts twice — once per leg — exactly as the
  weight vector records it).
- Sharpe is **excess-of-BIL** (the T-bill ETF that also plays the timer's cash leg) throughout,
  so every race — static vs 60/40, timed vs static — compares excess-of-cash to excess-of-cash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ASSETS, CASH, SMA_WINDOW

MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# The SMA timing signal
# --------------------------------------------------------------------------- #
def sma_signal(closes: pd.DataFrame, assets: list[str] = ASSETS,
               window: int = SMA_WINDOW) -> pd.DataFrame:
    """Boolean "hold the asset" signal, already shifted — zero look-ahead.

    ``True`` at row *t* means: at month *t-1*'s close, price was above its trailing ``window``-
    month simple moving average (inclusive of month *t-1*), so the sleeve is held **during**
    month *t*. Rows without a full ``window``-month history are ``NaN`` (dropped by the caller).
    """
    px = closes[assets]
    sma = px.rolling(window).mean()
    raw = (px > sma).where(sma.notna())  # NaN comparisons must stay NaN, not silently False
    return raw.shift(1)                  # applied to the FOLLOWING month's return


# --------------------------------------------------------------------------- #
# Weighted portfolio engine — target weights per row, tracked turnover
# --------------------------------------------------------------------------- #
def weighted_portfolio(returns: pd.DataFrame, weights: pd.DataFrame,
                       cost_bps: float = 0.0) -> tuple[pd.Series, pd.Series]:
    """Net monthly return of a portfolio rebalanced to ``weights`` every row.

    ``weights`` gives the TARGET allocation for the row it earns (decided using information
    known before that row started — the caller is responsible for the shift). Between rows, the
    prior target drifts with that row's realised returns before the next rebalance is measured
    against it, so turnover reflects genuine drift + signal changes, not a bookkeeping artefact.
    The very first row is assumed pre-positioned at its own target (no entry cost — a one-time,
    economically negligible convention over an 18-year monthly sample).

    Returns ``(net_return, turnover)`` — turnover is the one-way sum of absolute weight changes
    at that rebalance, so ``turnover * cost_bps`` is the desk's one-way-x-NAV-per-leg cost.
    """
    cols = list(weights.columns)
    idx = returns.index.intersection(weights.index)
    R = returns.loc[idx, cols].to_numpy(dtype=float)
    W = weights.loc[idx, cols].to_numpy(dtype=float)
    n = R.shape[0]
    out = np.empty(n)
    turns = np.empty(n)
    cost = cost_bps * 1e-4
    w_prev = W[0].copy()
    for t in range(n):
        target = W[t]
        turn = float(np.abs(target - w_prev).sum()) if t > 0 else 0.0
        gross = float(target @ R[t])
        out[t] = gross - turn * cost
        turns[t] = turn
        grown = target * (1.0 + R[t])
        s = grown.sum()
        w_prev = grown / s if s > 0 else target
    return pd.Series(out, index=idx, name="net"), pd.Series(turns, index=idx, name="turnover")


def static_weights(index: pd.Index, assets: list[str] = ASSETS, cash: str = CASH) -> pd.DataFrame:
    """Constant 20% each across the five Ivy sleeves, 0% cash, every month."""
    w = pd.DataFrame(0.0, index=index, columns=assets + [cash])
    w[assets] = 1.0 / len(assets)
    return w


def timed_weights(signal: pd.DataFrame, assets: list[str] = ASSETS,
                  cash: str = CASH) -> pd.DataFrame:
    """20% per sleeve when ``signal`` is True, else that 20% moves to ``cash`` (BIL)."""
    idx = signal.dropna(how="any").index
    sig = signal.loc[idx, assets].astype(bool)
    w = pd.DataFrame(0.0, index=idx, columns=assets + [cash])
    per = 1.0 / len(assets)
    w[assets] = np.where(sig.to_numpy(), per, 0.0)
    w[cash] = per * (~sig).sum(axis=1).to_numpy()
    return w


def sixty_forty_weights(index: pd.Index, stock: str = "VTI", bond: str = "AGG",
                        cash: str = CASH) -> pd.DataFrame:
    """Classic 60/40 (VTI/AGG), monthly-rebalanced, for comparison."""
    w = pd.DataFrame(0.0, index=index, columns=[stock, bond, cash])
    w[stock] = 0.60
    w[bond] = 0.40
    return w


# --------------------------------------------------------------------------- #
# Performance stats
# --------------------------------------------------------------------------- #
def perf(r: pd.Series, rf: pd.Series) -> dict:
    """CAGR, ann. vol, Sharpe (EXCESS of BIL), max drawdown."""
    r = pd.Series(r).astype(float).dropna()
    rf = pd.Series(rf).reindex(r.index).fillna(0.0)
    n = len(r)
    if n < 2:
        return {k: float("nan") for k in ("cagr", "vol", "sharpe", "max_dd", "n")}
    eq = (1.0 + r).cumprod()
    yrs = n / MONTHS_PER_YEAR
    cagr = float(eq.iloc[-1] ** (1.0 / yrs) - 1.0)
    vol = float(r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))
    ex = r - rf
    sharpe = (float(ex.mean() / ex.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))
              if ex.std(ddof=1) > 0 else float("nan"))
    dd = float((eq / eq.cummax() - 1.0).min())
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": dd, "n": int(n)}


def hac_tstat(x: pd.Series | np.ndarray) -> float:
    """Newey-West (HAC) t-stat for the mean of a monthly series (Bartlett kernel)."""
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    denom = np.sqrt(va + vb)
    return float((a.mean() - b.mean()) / denom) if denom > 0 else float("nan")


def active_stats(strat: pd.Series, bench: pd.Series) -> dict:
    """Mean active return (bps/mo, %/yr) + HAC t of the monthly active series."""
    act = (pd.Series(strat) - pd.Series(bench)).dropna()
    return {"mean_bps": float(act.mean() * 1e4),
            "ann_pct": float(act.mean() * MONTHS_PER_YEAR * 100.0),
            "hac_t": hac_tstat(act), "n": int(len(act))}


# --------------------------------------------------------------------------- #
# Bootstrap CI for a Sharpe difference (circular block bootstrap)
# --------------------------------------------------------------------------- #
def bootstrap_sharpe_diff(a: pd.Series, b: pd.Series, rf: pd.Series | None = None,
                          block: int = 6, n_boot: int = 2000, seed: int = 655) -> dict:
    """Circular block bootstrap CI for the Sharpe DIFFERENCE (arm ``a`` minus arm ``b``).

    Blocks of ``block`` months preserve short-run serial correlation; ``rf`` (if given) is
    subtracted from both arms first so the resampled Sharpes are already excess-of-cash.
    """
    idx = a.index.intersection(b.index)
    ra = a.reindex(idx).to_numpy(dtype=float)
    rb = b.reindex(idx).to_numpy(dtype=float)
    if rf is not None:
        f = rf.reindex(idx).fillna(0.0).to_numpy(dtype=float)
        ra = ra - f
        rb = rb - f

    def _sharpe(r):
        sd = r.std(ddof=1)
        return float(r.mean() / sd * np.sqrt(MONTHS_PER_YEAR)) if sd > 0 else float("nan")

    n = len(ra)
    point = _sharpe(ra) - _sharpe(rb)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    wins = 0
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        offsets = (starts[:, None] + np.arange(block)[None, :]) % n
        sel = offsets.reshape(-1)[:n]
        sa, sb = _sharpe(ra[sel]), _sharpe(rb[sel])
        diffs[i] = sa - sb
        if sa > sb:
            wins += 1
    lo, hi = float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))
    return {"point": float(point), "ci95": (lo, hi), "frac_a_wins": wins / n_boot,
            "n": n, "n_boot": n_boot}


# --------------------------------------------------------------------------- #
# Random-timing baseline — matched exposure, shuffled timing (>= 20 seeds)
# --------------------------------------------------------------------------- #
def random_timing_baseline(returns: pd.DataFrame, signal: pd.DataFrame,
                           cost_bps: float = 5.0, n_seeds: int = 20,
                           base_seed: int = 655) -> dict:
    """Shuffle each sleeve's in/out timing at random, holding its time-in-market FIXED.

    For every asset column, permute the True/False signal series across time (same count of
    "in" months, random which months) — one permutation per seed — rebuild the timed portfolio
    from the shuffled signal, and compare to the ACTUAL SMA-timed portfolio's stats. Isolates
    whether the 10-month rule's specific *timing* carries information, versus merely holding a
    reduced average exposure (which any random timer matched on exposure would also achieve).
    Averaged over >= 20 seeds (single-seed baselines are banned on this desk).
    """
    assert n_seeds >= 20
    idx = signal.dropna(how="any").index
    sig = signal.loc[idx, ASSETS].astype(bool)
    actual_w = timed_weights(signal)
    actual_net, _ = weighted_portfolio(returns, actual_w, cost_bps=cost_bps)
    actual_perf = perf(actual_net, returns[CASH])

    dd_list, sharpe_list, cagr_list, t_list = [], [], [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        shuf = sig.apply(lambda col: pd.Series(rng.permutation(col.to_numpy()), index=col.index))
        w = pd.DataFrame(0.0, index=idx, columns=ASSETS + [CASH])
        per = 1.0 / len(ASSETS)
        w[ASSETS] = np.where(shuf.to_numpy(), per, 0.0)
        w[CASH] = per * (~shuf).sum(axis=1).to_numpy()
        net, _ = weighted_portfolio(returns, w, cost_bps=cost_bps)
        p = perf(net, returns[CASH])
        dd_list.append(p["max_dd"])
        sharpe_list.append(p["sharpe"])
        cagr_list.append(p["cagr"])
        common = actual_net.index.intersection(net.index)
        t_list.append(welch_t(actual_net.loc[common].to_numpy(), net.loc[common].to_numpy()))

    return {
        "n_seeds": n_seeds,
        "actual": actual_perf,
        "base_dd_mean": float(np.mean(dd_list)),
        "base_sharpe_mean": float(np.mean(sharpe_list)),
        "base_cagr_mean": float(np.mean(cagr_list)),
        "welch_t_mean": float(np.mean(t_list)),
        "dd_beat_share": float(np.mean([actual_perf["max_dd"] > d for d in dd_list])),
        "sharpe_beat_share": float(np.mean([actual_perf["sharpe"] > s for s in sharpe_list])),
    }


# --------------------------------------------------------------------------- #
# Synthetic machinery control
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: pd.DataFrame, cost_bps: float = 5.0) -> dict:
    """Run the static vs timed engine on a synthetic panel; return the active-return HAC t
    and the drawdown reduction (timed vs static)."""
    closes = panel[ASSETS].copy()
    cash_lvl = panel[CASH + "_lvl"]
    rets = pd.DataFrame(index=panel.index)
    rets[ASSETS] = closes.pct_change()
    rets[CASH] = cash_lvl.pct_change()
    rets = rets.dropna(how="all")

    sw = static_weights(rets.index)
    static_net, _ = weighted_portfolio(rets, sw, cost_bps=cost_bps)

    sig = sma_signal(closes)
    tw = timed_weights(sig)
    timed_net, turns = weighted_portfolio(rets, tw, cost_bps=cost_bps)

    common = static_net.index.intersection(timed_net.index)
    a = active_stats(timed_net.loc[common], static_net.loc[common])
    ps = perf(static_net.loc[common], rets.loc[common, CASH])
    pt = perf(timed_net.loc[common], rets.loc[common, CASH])
    return {"active_hac_t": a["hac_t"], "active_ann_pct": a["ann_pct"],
            "static_dd": ps["max_dd"], "timed_dd": pt["max_dd"],
            "static_sharpe": ps["sharpe"], "timed_sharpe": pt["sharpe"],
            "n": a["n"]}
