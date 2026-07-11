"""The Larry Portfolio engine and its honest controls — Study 657.

Larry Swedroe's "Larry Portfolio" (named for him by Bill Schultheis, detailed across
Swedroe's books since the early 2000s): put a SMALL slice of the risk budget into the
highest-expected-return equity factor — small-cap VALUE — and park the rest in safe,
short/intermediate bonds. The pitch: because small-value has a higher expected return per
unit of equity risk than the broad market, a much smaller equity sleeve can deliver
60/40-like portfolio returns while running far less total equity exposure (lower vol, lower
drawdown, lower correlation to a pure-equity crash).

Two questions, kept separate on purpose (the same discipline as sibling study
`97-balancing-act <../97-balancing-act/>`_):

1. **Does 30% small-value / 70% bonds match a 60% market / 40% bonds on RETURN**, while
   running materially less risk (vol, drawdown)? (the headline race + a bootstrap CI on the
   return AND Sharpe differences.)
2. **Has the small-value premium itself decayed** since being popularised — the load-bearing
   assumption the whole portfolio depends on? (an externally-justified era split, HAC t of
   the difference, exactly the pattern used in sibling study 637's era contrast.)

Conventions
-----------
- Inputs are **daily simple total returns** (auto-adjusted closes -> ``pct_change``).
- Both blends rebalance to target weights **annually** (first trading day of the calendar
  year), drifting in between — identical rebalance convention to 97-balancing-act, so the
  60/40 numbers are directly comparable across the two studies.
- Costs: one-way bps x NAV, charged against total absolute weight change at each rebalance.
- Sharpe is **excess-of-SHY** (the T-bill-like ETF that is also the study's cash proxy)
  throughout, so every race compares excess-of-cash to excess-of-cash.
- No execution lag needed: fixed-weight calendar rebalancing carries no signal, hence no
  look-ahead to guard against (same reasoning as 97-balancing-act).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Return helpers
# --------------------------------------------------------------------------- #
def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from a total-return price frame (first row dropped)."""
    return prices.pct_change().dropna()


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


# --------------------------------------------------------------------------- #
# The blend engine — fixed weights, annual rebalance
# --------------------------------------------------------------------------- #
def rebalanced_blend(returns: pd.DataFrame, weights: dict[str, float],
                     rebalance: str = "annual", cost_bps: float = 0.0) -> pd.Series:
    """Daily net return of a fixed-weight blend rebalanced on a calendar schedule.

    Weights drift with the assets between rebalances; on each rebalance date the book is
    reset to ``weights`` and the one-way turnover (sum of absolute weight changes) is charged
    at ``cost_bps``. ``rebalance='annual'`` resets on the first trading day of each calendar
    year.
    """
    cols = list(weights.keys())
    R = returns[cols].to_numpy()
    w_target = np.array([weights[c] for c in cols], dtype=float)
    n = R.shape[0]
    idx = returns.index

    if rebalance == "annual":
        marks = idx.to_series().groupby(idx.year).head(1).index
    elif rebalance == "none":
        marks = idx[:1]
    else:
        raise ValueError(f"unknown rebalance schedule: {rebalance!r}")
    rebal = pd.Index(marks)

    w = w_target.copy()
    out = np.empty(n)
    cost = cost_bps * 1e-4
    for t in range(n):
        if idx[t] in rebal:
            turn = np.abs(w_target - w).sum()
            w = w_target.copy()
        else:
            turn = 0.0
        port_ret = float(w @ R[t]) - turn * cost
        out[t] = port_ret
        w = w * (1.0 + R[t])
        s = w.sum()
        if s != 0:
            w = w / s
    return pd.Series(out, index=idx, name="blend")


def single_asset(returns: pd.DataFrame, ticker: str) -> pd.Series:
    return returns[ticker].rename(ticker)


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def stats(net: pd.Series, rf: pd.Series | None = None) -> dict:
    """CAGR, ann. vol, Sharpe (EXCESS of cash if ``rf`` given), max drawdown."""
    net = net.astype(float)
    equity = (1.0 + net).cumprod()
    n = len(net)
    years = n / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    if rf is not None:
        ex = (net - rf.reindex(net.index).fillna(0.0)).astype(float)
    else:
        ex = net
    sharpe = (float(ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS))
              if ex.std(ddof=1) > 0 else float("nan"))
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe,
            "max_dd": _max_drawdown(equity.to_numpy()), "n": int(n),
            "final": float(equity.iloc[-1])}


# --------------------------------------------------------------------------- #
# Inference: HAC t-stat and circular block bootstrap on a difference series
# --------------------------------------------------------------------------- #
def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) t-stat for the mean of ``x`` (Bartlett kernel)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 5:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    denom = np.sqrt(va + vb)
    return float((a.mean() - b.mean()) / denom) if denom > 0 else float("nan")


def _annualised_sharpe(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def bootstrap_diff(a: pd.Series, b: pd.Series, rf: pd.Series | None = None,
                   metric: str = "sharpe", block: int = 21, n_boot: int = 2000,
                   seed: int = 657) -> dict:
    """Circular block bootstrap CI for the DIFFERENCE (arm ``a`` - arm ``b``) in either the
    annualised Sharpe (``metric='sharpe'``) or the mean daily return (``metric='mean'``).

    Resamples the two aligned (excess, if ``rf`` given) return series JOINTLY in circular
    blocks (preserves short-run serial correlation and the cross-correlation between the two
    arms), recomputes the metric per resample, and returns the point difference with a 95%
    CI and the bootstrap win-rate for arm ``a``.
    """
    idx = a.index.intersection(b.index)
    ra = a.reindex(idx).to_numpy(dtype=float)
    rb = b.reindex(idx).to_numpy(dtype=float)
    if rf is not None:
        f = rf.reindex(idx).fillna(0.0).to_numpy(dtype=float)
        ra = ra - f
        rb = rb - f

    def _metric(r):
        return _annualised_sharpe(r) if metric == "sharpe" else float(r.mean() * TRADING_DAYS)

    n = len(idx)
    point = _metric(ra) - _metric(rb)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    a_wins = 0
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        offsets = (starts[:, None] + np.arange(block)[None, :]) % n
        sel = offsets.reshape(-1)[:n]
        sa, sb = _metric(ra[sel]), _metric(rb[sel])
        diffs[i] = sa - sb
        if sa > sb:
            a_wins += 1
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"point": float(point), "ci95": (float(lo), float(hi)),
            "frac_a_wins": a_wins / n_boot, "n": n, "block": block, "n_boot": n_boot}


# --------------------------------------------------------------------------- #
# The small-value premium's own decay — the load-bearing assumption
# --------------------------------------------------------------------------- #
def premium_series(returns: pd.DataFrame, sv: str = "IJS", mkt: str = "SPY") -> pd.Series:
    """Daily small-value-minus-market spread (the raw premium the whole portfolio leans on)."""
    return (returns[sv] - returns[mkt]).rename("sv_minus_mkt")


def premium_stats(spread: pd.Series) -> dict:
    """Annualised mean spread + HAC t for the whole-sample small-value premium."""
    x = spread.dropna().to_numpy(dtype=float)
    return {"ann_pct": float(x.mean() * TRADING_DAYS * 100.0),
            "hac_t": hac_tstat(x), "n": int(len(x))}


def era_contrast(spread: pd.Series, split: str) -> dict:
    """Small-value premium before vs since ``split``: within-era HAC t's, plus the HAC t of
    the era DIFFERENCE (late minus early), justified split, tested as a difference — the same
    pattern as sibling study 637's era contrast."""
    early = spread[spread.index < split].dropna().to_numpy(dtype=float)
    late = spread[spread.index >= split].dropna().to_numpy(dtype=float)
    return {"n_early": len(early), "n_late": len(late),
            "early_ann_pct": float(early.mean() * TRADING_DAYS * 100.0) if len(early) else float("nan"),
            "late_ann_pct": float(late.mean() * TRADING_DAYS * 100.0) if len(late) else float("nan"),
            "hac_t_early": hac_tstat(early), "hac_t_late": hac_tstat(late),
            "welch_t_diff": welch_t(late, early)}


# --------------------------------------------------------------------------- #
# Synthetic machinery control
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: pd.DataFrame, cost_bps: float = 2.0) -> dict:
    """Run the premium-detection statistic AND the Larry-vs-60/40 engine on a synthetic
    (MKT, SV, BOND) panel.

    The PASS/FAIL machinery check is ``premium_hac_t`` — the same HAC-*t*-on-the-spread
    primitive the third axis applies to the real IJS-SPY tape. Under ``premium=0`` (CAPM-
    neutral synthetic world, ``beta_sv=1.0``) it must NOT fire across >= 10 seeds; a planted
    premium must light it up. The portfolio-level numbers (Larry vs 60/40 CAGR/Sharpe gap)
    are reported alongside for the "how big a premium would close the gap" story — the
    30%-SV/70%-bond blend structurally trails a 60%-market blend even at premium=0 (it runs
    far less equity beta), so that gap is NOT the pass/fail signal.
    """
    rets = panel.pct_change().dropna()
    rf = pd.Series(0.0, index=rets.index)  # no cash leg in the synthetic world

    spread = (rets["SV"] - rets["MKT"]).dropna()
    premium_hac_t = hac_tstat(spread.to_numpy())

    larry = rebalanced_blend(rets, {"SV": 0.30, "BOND": 0.70},
                             rebalance="annual", cost_bps=cost_bps)
    sixty = rebalanced_blend(rets, {"MKT": 0.60, "BOND": 0.40},
                             rebalance="annual", cost_bps=cost_bps)
    s_larry, s_sixty = stats(larry, rf=rf), stats(sixty, rf=rf)
    return {"premium_hac_t": premium_hac_t,
            "premium_ann_pct": float(spread.mean() * TRADING_DAYS * 100.0),
            "cagr_gap": s_larry["cagr"] - s_sixty["cagr"],
            "sharpe_gap": s_larry["sharpe"] - s_sixty["sharpe"],
            "larry_sharpe": s_larry["sharpe"], "sixty_sharpe": s_sixty["sharpe"]}
