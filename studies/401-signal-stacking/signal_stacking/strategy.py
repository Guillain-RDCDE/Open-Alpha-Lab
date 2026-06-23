"""The signal-stacking engine and its honest controls — Study 401 (Signal-Stacking).

The viral recipe has four moving parts; this module is each one, named and testable:

1. **The composite z-score (the "ζ-field").** Standardise each signal, weight it, sum:
   ``Z_t = sum_k w_k * zscore(signal_k)_t``. Equal weights are the honest default; a
   *snooped* alternative weights each signal by its **in-sample** Sharpe — the move that
   manufactures a gorgeous backtest. ``composite_score`` builds it.

2. **The position rule.** Standardise the composite and trade its sign past a threshold:
   long when ``Z > thr``, short when ``Z < -thr`` (or flat when ``|Z| <= thr`` — the
   "stays flat when signals disagree" claim, which is just *de-risking*, not crash
   prediction). One execution lag is already in the data; nothing is shifted again here.

3. **The backtest.** ``pos_t * (fwd_ret_t - rf)`` is the excess-of-cash return, with
   one-way turnover charged at ``cost_bps`` per flip. Every rule is raced **excess-vs-
   excess** against always-long, so a cash-heavy rule is judged fairly.

4. **The honest arbiter.** Two corrections the thread never applies:
     * **The √K law.** ``composite_ic`` measures the composite's information coefficient
       against the average single signal's. Stacking *decorrelated* edges raises IC like
       √K; stacking *redundant* ones (or pure noise) does not. This is the whole ballgame.
     * **A permutation test.** ``permutation_pvalue`` shuffles the return relative to the
       signals to destroy any real link, recomputes the composite's Sharpe many times, and
       reports the fraction of shuffled worlds that beat the observed Sharpe — the p-value
       that asks "is the composite better than its own signals, reshuffled into noise?"
     * **An in/out-of-sample split.** ``snoop_split`` *selects* the best signals and their
       weights on the first half and measures them on the held-out half — the gap between
       the two is the snooping tax the headline equity curve hides.

The verdict is a *methodology* one: stacking weak signals creates an edge **only** when the
signals carry genuine, decorrelated information (the positive control confirms the √K
boost); on a pure-noise stack it manufactures nothing real, and the only "edge" it shows is
in-sample selection that evaporates out of sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Composite score (the "ζ-field") and the position rule
# ---------------------------------------------------------------------------
def zscore(s: pd.Series) -> pd.Series:
    """Full-sample z-score of a series (the standardisation the recipe applies)."""
    sd = s.std(ddof=0)
    return (s - s.mean()) / (sd if sd > 0 else 1.0)


def composite_score(
    signals: pd.DataFrame, weights: np.ndarray | None = None
) -> pd.Series:
    """The weighted, standardised composite ``Z_t = sum_k w_k * zscore(signal_k)_t``.

    ``weights`` defaults to equal (the honest blend). Pass Sharpe-derived weights to build
    the snooped version. The result is itself re-standardised to unit std so the position
    threshold has a stable meaning across blends.
    """
    z = signals.apply(zscore, axis=0)
    if weights is None:
        w = np.ones(z.shape[1]) / z.shape[1]
    else:
        w = np.asarray(weights, dtype=float)
        w = w / (np.sum(np.abs(w)) + 1e-12)
    comp = z.to_numpy() @ w
    comp = pd.Series(comp, index=signals.index, name="composite")
    return zscore(comp)


def position_from_score(
    score: pd.Series, thr: float = 0.0, long_short: bool = True
) -> pd.Series:
    """+1 / 0 / -1 position from the (standardised) composite, known at the close of t.

    ``thr`` is a z-score band: trade only when the composite agrees strongly enough.
    ``thr = 0`` with ``long_short=True`` is the pure-sign timing book; ``thr > 0`` carves
    out the "flat when signals disagree" middle the thread credits for avoiding drawdowns.
    """
    on_long = score > thr
    on_short = score < -thr
    if long_short:
        pos = np.where(on_long, 1.0, np.where(on_short, -1.0, 0.0))
    else:
        pos = np.where(on_long, 1.0, 0.0)
    return pd.Series(pos, index=score.index, name="pos")


# ---------------------------------------------------------------------------
# Backtest with one execution lag (in the data) and one-way turnover costs
# ---------------------------------------------------------------------------
def backtest_position(
    pos: pd.Series, fwd_ret: pd.Series, rf_daily: float = 0.0, cost_bps: float = 0.0
) -> pd.Series:
    """Net **excess-of-cash** daily return of a position series.

    ``pos_t * (fwd_ret_t - rf)`` earns the excess return when in the market, 0 in cash, the
    negated excess when short. Costs: one-way turnover × NAV at ``cost_bps`` whenever the
    position changes (each flip is a trade). Excess-vs-excess by construction.
    """
    excess = fwd_ret - rf_daily
    gross = pos * excess
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * (cost_bps * 1e-4)
    return (gross - cost).rename("ret")


def buy_and_hold_excess(fwd_ret: pd.Series, rf_daily: float = 0.0) -> pd.Series:
    """The always-long excess-of-cash return — the benchmark every blend races."""
    return (fwd_ret - rf_daily).rename("bh")


def backtest_composite(
    signals: pd.DataFrame,
    fwd_ret: pd.Series,
    weights: np.ndarray | None = None,
    thr: float = 0.0,
    long_short: bool = True,
    rf_daily: float = 0.0,
    cost_bps: float = 0.0,
) -> pd.Series:
    """Convenience: composite → position → net excess return, in one call."""
    score = composite_score(signals, weights=weights)
    pos = position_from_score(score, thr=thr, long_short=long_short)
    return backtest_position(pos, fwd_ret, rf_daily=rf_daily, cost_bps=cost_bps)


# ---------------------------------------------------------------------------
# Metrics & honest inference (shared desk idiom)
# ---------------------------------------------------------------------------
def sharpe(daily: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised Sharpe of an (already-excess) daily return series."""
    r = daily.dropna()
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")


def hac_tstat(daily: pd.Series) -> float:
    """Newey-West (HAC) *t*-stat that the mean (excess) daily return is non-zero."""
    x = daily.dropna().to_numpy(dtype=float)
    n = x.size
    if n < 3:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        wk = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * wk * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def block_bootstrap_ci(
    daily: pd.Series, block: int = 10, n_boot: int = 2000, seed: int = 401, alpha: float = 0.05
) -> tuple[float, float]:
    """Circular block-bootstrap CI on the *mean* daily return (preserves autocorrelation)."""
    x = daily.dropna().to_numpy(dtype=float)
    n = x.size
    if n < block + 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[b] = x[idx][:n].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


# ---------------------------------------------------------------------------
# The √K law — does stacking actually raise the information coefficient?
# ---------------------------------------------------------------------------
def information_coefficient(signal: pd.Series, fwd_ret: pd.Series) -> float:
    """Correlation of a signal with the return it tries to time (its IC)."""
    a = signal.to_numpy(dtype=float)
    b = fwd_ret.to_numpy(dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    sa, sb = a[m].std(), b[m].std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def composite_ic(
    signals: pd.DataFrame, fwd_ret: pd.Series, weights: np.ndarray | None = None
) -> dict:
    """Compare the composite's IC to the average single signal's — the √K test.

    Returns the mean and best single-signal IC, the equal-weight composite IC, the realised
    "lift" (composite / mean single), and the √K reference (the lift you'd get if the
    signals were perfectly decorrelated). Stacking *helps* only when the realised lift tracks
    √K; redundant or noise signals leave it stuck near 1.
    """
    ics = np.array([information_coefficient(signals[c], fwd_ret) for c in signals.columns])
    comp = composite_score(signals, weights=weights)
    comp_ic = information_coefficient(comp, fwd_ret)
    mean_single = float(np.nanmean(np.abs(ics)))
    best_single = float(np.nanmax(np.abs(ics)))
    lift = float(abs(comp_ic) / mean_single) if mean_single > 1e-9 else float("nan")
    return {
        "single_ics": ics,
        "mean_single_ic": mean_single,
        "best_single_ic": best_single,
        "composite_ic": float(comp_ic),
        "lift": lift,
        "sqrt_k_reference": float(np.sqrt(signals.shape[1])),
        "n_signals": signals.shape[1],
    }


# ---------------------------------------------------------------------------
# The permutation test — is the composite better than its signals, reshuffled?
# ---------------------------------------------------------------------------
def permutation_pvalue(
    signals: pd.DataFrame,
    fwd_ret: pd.Series,
    weights: np.ndarray | None = None,
    thr: float = 0.0,
    long_short: bool = True,
    n_perm: int = 1000,
    seed: int = 401,
) -> dict:
    """Permutation p-value for the composite's Sharpe under "signals predict nothing".

    The composite/position are fixed; we shuffle the *return* vector to break any real
    signal→return link, recompute the strategy Sharpe in each shuffled world, and report the
    fraction that match or beat the observed Sharpe (+1 smoothing). A large p means the
    composite's performance is exactly what reshuffling its own signals into noise produces —
    the stack added nothing real.
    """
    score = composite_score(signals, weights=weights)
    pos = position_from_score(score, thr=thr, long_short=long_short).to_numpy(dtype=float)
    r = fwd_ret.to_numpy(dtype=float)
    m = np.isfinite(pos) & np.isfinite(r)
    pos, r = pos[m], r[m]
    obs = _sharpe_arr(pos * r)

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rp = r[rng.permutation(r.size)]
        if _sharpe_arr(pos * rp) >= obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return {"observed_sharpe": float(obs), "perm_p": float(p), "n_perm": n_perm}


def _sharpe_arr(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


# ---------------------------------------------------------------------------
# The snooping split — select on the first half, pay on the second
# ---------------------------------------------------------------------------
def snoop_split(
    signals: pd.DataFrame,
    fwd_ret: pd.Series,
    top_m: int | None = None,
    thr: float = 0.0,
    long_short: bool = True,
    cost_bps: float = 0.0,
) -> dict:
    """The p-hacker's blend: pick the best signals + Sharpe-weights *in sample*, pay OOS.

    Splits the panel in half. On the **first half**, rank single-signal Sharpes, keep the
    top ``top_m`` (default: all with positive Sharpe), and weight them by that in-sample
    Sharpe (clipped at 0) — the "weighted by historical Sharpe" move from the thread. Then
    report the composite's Sharpe **in sample** (the gorgeous backtest) and **out of sample**
    (what you actually get). The gap is the snooping tax.
    """
    n = len(fwd_ret)
    cut = n // 2
    s_is, s_oos = signals.iloc[:cut], signals.iloc[cut:]
    r_is, r_oos = fwd_ret.iloc[:cut], fwd_ret.iloc[cut:]

    # In-sample single-signal Sharpes (sign-timing each signal alone).
    is_sharpes = {}
    for c in signals.columns:
        pos = position_from_score(zscore(s_is[c]), thr=thr, long_short=long_short)
        is_sharpes[c] = sharpe(backtest_position(pos, r_is, cost_bps=cost_bps))
    ranked = sorted(is_sharpes.items(), key=lambda kv: (kv[1] if np.isfinite(kv[1]) else -1e9),
                    reverse=True)
    if top_m is None:
        chosen = [c for c, sh in ranked if np.isfinite(sh) and sh > 0]
        if not chosen:
            chosen = [ranked[0][0]]
    else:
        chosen = [c for c, _ in ranked[:max(1, top_m)]]

    w = np.array([max(0.0, is_sharpes[c]) for c in chosen])
    if w.sum() <= 0:
        w = np.ones(len(chosen))

    is_ret = backtest_composite(s_is[chosen], r_is, weights=w, thr=thr,
                                long_short=long_short, cost_bps=cost_bps)
    oos_ret = backtest_composite(s_oos[chosen], r_oos, weights=w, thr=thr,
                                 long_short=long_short, cost_bps=cost_bps)
    return {
        "chosen": chosen,
        "n_chosen": len(chosen),
        "is_sharpe": sharpe(is_ret),
        "oos_sharpe": sharpe(oos_ret),
        "is_hac_t": hac_tstat(is_ret),
        "oos_hac_t": hac_tstat(oos_ret),
    }


# ---------------------------------------------------------------------------
# The full experiment — everything the notebooks and results.md quote
# ---------------------------------------------------------------------------
def run_experiment(
    signals: pd.DataFrame,
    fwd_ret: pd.Series,
    thr: float = 0.0,
    long_short: bool = True,
    cost_bps: float = 0.0,
    rf_daily: float = 0.0,
    n_perm: int = 1000,
    seed: int = 401,
) -> dict:
    """Stack the whole panel honestly: equal-weight composite vs best single, √K lift,
    permutation p, and the snooped in/out-of-sample split.

    Returns a bundle: ``composite`` (equal-weight Sharpe + HAC t), ``best_single`` and
    ``mean_single`` Sharpes, ``ic`` (the √K test), ``perm`` (permutation p of the honest
    composite), ``snoop`` (the IS/OOS gap of the snooped blend), and the buy-and-hold Sharpe.
    """
    # Honest equal-weight composite.
    comp_ret = backtest_composite(signals, fwd_ret, weights=None, thr=thr,
                                  long_short=long_short, rf_daily=rf_daily, cost_bps=cost_bps)
    comp_sharpe = sharpe(comp_ret)
    comp_t = hac_tstat(comp_ret)

    # Single-signal sign-timing books (each alone), for the best/mean comparison.
    singles = []
    for c in signals.columns:
        pos = position_from_score(zscore(signals[c]), thr=thr, long_short=long_short)
        singles.append(sharpe(backtest_position(pos, fwd_ret, rf_daily=rf_daily,
                                                cost_bps=cost_bps)))
    singles = np.array(singles, dtype=float)

    ic = composite_ic(signals, fwd_ret)
    perm = permutation_pvalue(signals, fwd_ret, thr=thr, long_short=long_short,
                              n_perm=n_perm, seed=seed)
    snoop = snoop_split(signals, fwd_ret, thr=thr, long_short=long_short, cost_bps=cost_bps)
    bh = buy_and_hold_excess(fwd_ret, rf_daily=rf_daily)

    return {
        "composite": {"sharpe": comp_sharpe, "hac_t": comp_t},
        "best_single": float(np.nanmax(singles)),
        "mean_single": float(np.nanmean(singles)),
        "ic": ic,
        "perm": perm,
        "snoop": snoop,
        "bh_sharpe": sharpe(bh),
        "n_signals": signals.shape[1],
        "cost_bps": cost_bps,
    }
