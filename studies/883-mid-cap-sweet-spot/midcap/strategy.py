"""Strategy + inference for Study 883 — Mid-Cap Sweet Spot.

The claim: the mid-cap ETF is the "forgotten middle" — a **higher risk-adjusted return
than BOTH large (SPY) and small (IWM)**. We grade it honestly on two axes:

* **Signal** — does mid earn a genuine **excess-of-cash Sharpe advantage over BOTH**
  neighbours? We run an excess-vs-excess Sharpe race (every leg minus BIL cash), a
  Newey-West (HAC) *t* on the pairwise daily *return difference* (mid − large, mid −
  small; the cash leg cancels in a difference, so this reaches back to the full IJH /
  MDY tape), and a **paired circular-block-bootstrap CI** on the Sharpe advantage. The
  edge must clear the bar over BOTH neighbours AND hold across sub-eras.
* **Tradability** — does a costed long-mid / short-neighbour spread survive one-way ETF
  spreads + borrow on the short leg?

Inference primitives (``newey_west_t`` / ``one_sample_t`` / ``welch_t`` /
``wilson_interval``) are local and unit-tested; annualised Sharpe reuses
``quantlab.stats.annualized_sharpe``. Daily equity returns are mildly autocorrelated, so
the plain *t* is banned on the difference series — HAC only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.stats import annualized_sharpe  # noqa: E402

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Return panel
# --------------------------------------------------------------------------- #
def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple total-return series from a wide close frame (index=date)."""
    return prices.sort_index().pct_change()


def excess(ret: pd.DataFrame, asset: str, cash: str) -> pd.Series:
    """Excess-of-cash daily return of ``asset`` (asset minus the cash leg), aligned."""
    return (ret[asset] - ret[cash]).dropna()


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


def newey_west_t(x: np.ndarray, lags: int | None = None) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0.

    ``lags=None`` uses the standard rule of thumb ``floor(4*(n/100)**(2/9))``.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
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


def _max_drawdown(ret: pd.Series) -> float:
    """Max drawdown (%, negative) of a daily simple-return series."""
    r = ret.dropna()
    if len(r) == 0:
        return float("nan")
    curve = (1.0 + r).cumprod()
    return float((curve / curve.cummax() - 1.0).min() * 100.0)


# --------------------------------------------------------------------------- #
# The excess-of-cash Sharpe race
# --------------------------------------------------------------------------- #
def race(ret: pd.DataFrame, assets: list[str], cash: str) -> pd.DataFrame:
    """Per-asset excess-of-cash summary on the common (all-legs-present) window.

    Every leg's Sharpe is computed on its **excess-of-cash** daily series so the race is
    apples-to-apples (excess-vs-excess). Returns a frame indexed by asset with the
    annualised return, vol, excess-of-cash Sharpe, and max drawdown.
    """
    common = ret[assets + [cash]].dropna()
    rows = {}
    for a in assets:
        r = common[a]
        exc = common[a] - common[cash]
        rows[a] = {
            "ann_ret_pct": float(r.mean() * TRADING_DAYS * 100),
            "ann_vol_pct": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100),
            "ex_sharpe": float(annualized_sharpe(exc, periods_per_year=TRADING_DAYS)),
            "max_dd_pct": _max_drawdown(r),
            "n": int(len(common)),
        }
    out = pd.DataFrame(rows).T
    out["n"] = out["n"].astype(int)
    return out


def pairwise_diff(ret: pd.DataFrame, mid: str, other: str,
                  start: str | None = None, end: str | None = None) -> dict:
    """Mid-minus-other daily return difference (cash cancels): mean, HAC & one-sample t.

    Because ``mid_excess − other_excess = mid − other``, the cash leg drops out of the
    difference — so this test runs on the full overlapping mid/other tape (back to 1995
    for MDY/SPY), not just the BIL-anchored window.
    """
    d = (ret[mid] - ret[other]).dropna()
    if start is not None:
        d = d[d.index >= pd.Timestamp(start)]
    if end is not None:
        d = d[d.index < pd.Timestamp(end)]
    x = d.to_numpy()
    return {
        "n": int(len(d)),
        "start": str(d.index.min().date()) if len(d) else None,
        "end": str(d.index.max().date()) if len(d) else None,
        "ann_diff_pct": float(np.nanmean(x) * TRADING_DAYS * 100),
        "t_nw": newey_west_t(x),
        "t_1s": one_sample_t(x),
    }


def sharpe_adv_bootstrap(ret: pd.DataFrame, mid: str, other: str, cash: str,
                         n_boot: int = 2000, block: int = 21, seed: int = 883) -> dict:
    """Paired circular-block-bootstrap CI for mid's excess-Sharpe ADVANTAGE over ``other``.

    Advantage = Sharpe(mid − cash) − Sharpe(other − cash), on the common window. The two
    excess series are resampled **jointly** (same block indices) to preserve their
    cross-correlation, and the Sharpe difference is recomputed on each resample. A 95%
    percentile CI that stays clear of zero is the robust bar; a CI spanning zero means
    the advantage is not statistically distinguishable from none.
    """
    common = ret[[mid, other, cash]].dropna()
    exM = (common[mid] - common[cash]).to_numpy()
    exO = (common[other] - common[cash]).to_numpy()
    n = len(exM)

    def _sr(x):
        sd = x.std(ddof=1)
        return x.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan

    adv = float(_sr(exM) - _sr(exO))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel() % n
        idx = idx[:n]
        diffs[b] = _sr(exM[idx]) - _sr(exO[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "adv": adv, "ci_lo": float(lo), "ci_hi": float(hi),
        "clears_zero": bool(lo > 0 or hi < 0),
        "n": int(n), "n_boot": int(n_boot),
    }


# --------------------------------------------------------------------------- #
# Robustness — eras + calendar years
# --------------------------------------------------------------------------- #
def era_table(ret: pd.DataFrame, mid: str, other: str,
              edges: list[str]) -> pd.DataFrame:
    """Pairwise mid − other annual diff + HAC t across the era boundaries in ``edges``.

    ``edges`` is a sorted list of cut dates; consecutive pairs define each era. Runs on
    the cash-independent difference, so eras predating BIL are fine.
    """
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = pairwise_diff(ret, mid, other, start=lo, end=hi)
        rows.append({"era": f"{lo[:7]}→{hi[:7]}", "n": s["n"],
                     "ann_diff_pct": s["ann_diff_pct"], "t_nw": s["t_nw"]})
    return pd.DataFrame(rows)


def calendar_year_table(ret: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    """Per-calendar-year total return (%) for each asset (vectorised, no per-date loop)."""
    g = (1.0 + ret[assets]).groupby(ret.index.year).prod() - 1.0
    return (g * 100.0).round(2)


# --------------------------------------------------------------------------- #
# Tradability — the costed long-mid / short-neighbour spread
# --------------------------------------------------------------------------- #
def costed_spread(ret: pd.DataFrame, mid: str, other: str,
                  cost_bps_oneway: float = 3.0, borrow_bps_yr: float = 50.0,
                  rebalances_per_year: float = 4.0, er_diff_bps_yr: float = 0.0) -> dict:
    """Long mid / short ``other``, dollar-neutral: gross & net annualised diff + HAC t.

    Gross monthly/daily P&L = (mid − other). Charges: borrow on the short leg
    (``borrow_bps_yr``/yr), a round-trip one-way cost on both legs at each of
    ``rebalances_per_year`` rebalances (2 sides × one-way × turnover), and any expense-
    ratio differential. IJH/MDY vs SPY/IWM are all cheap, tight-spread mega-ETFs, so the
    friction is small — but so is the edge.
    """
    d = (ret[mid] - ret[other]).dropna()
    gross_ann = float(d.mean() * TRADING_DAYS * 100)
    charge_ann = (borrow_bps_yr
                  + 2.0 * cost_bps_oneway * rebalances_per_year
                  + er_diff_bps_yr) / 100.0
    charge_daily = charge_ann / 100.0 / TRADING_DAYS
    net = d - charge_daily
    return {
        "gross_ann_pct": gross_ann,
        "charge_ann_pct": float(charge_ann),
        "net_ann_pct": float(net.mean() * TRADING_DAYS * 100),
        "t_net_nw": newey_west_t(net.to_numpy()),
        "n": int(len(d)),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame) -> dict:
    """Run the excess-Sharpe advantage of ``mid`` over BOTH neighbours on a synthetic world.

    Returns the two advantages and whether mid clears BOTH (the planted-edge signal). On
    the null world both advantages sit at ~0; with a planted edge both are positive.
    """
    ret = daily_returns(prices)
    r = race(ret, ["mid", "large", "small"], "cash")
    adv_large = float(r.loc["mid", "ex_sharpe"] - r.loc["large", "ex_sharpe"])
    adv_small = float(r.loc["mid", "ex_sharpe"] - r.loc["small", "ex_sharpe"])
    d_large = pairwise_diff(ret, "mid", "large")
    d_small = pairwise_diff(ret, "mid", "small")
    return {
        "adv_large": adv_large, "adv_small": adv_small,
        "beats_both": bool(adv_large > 0 and adv_small > 0),
        "t_large": d_large["t_nw"], "t_small": d_small["t_nw"],
        "mid_sharpe": float(r.loc["mid", "ex_sharpe"]),
    }
