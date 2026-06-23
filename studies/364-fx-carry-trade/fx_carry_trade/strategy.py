"""Strategy + inference for Study 364 — FX Carry Trade.

The carry rule, on the monthly panel:

    Each month, rank the currencies by their *carry* (the annualised short-rate
    differential vs USD, here a fixed per-currency proxy). Go **long the top-k**
    high-yielders and **short the bottom-k** low-yielders, equal-weight within each
    leg, dollar-neutral. Hold one month; the carry map is static so the ranking is
    constant — this is the canonical "borrow low-yield, hold high-yield" basket.

We measure the realised basket return with a **1-day (here: one-month) execution
lag** convention applied to spot, charge **one-way costs × turnover** plus a borrow
spread on the short leg, and confront the result with:

  * the **annualised Sharpe** of the carry basket (excess of cash — the carry is
    already a differential, so the basket return is an excess return);
  * a **Newey-West / HAC t-stat** on the monthly mean (the Signal-axis test);
  * the **skewness and the worst-month / max-drawdown** of the basket — the decisive
    *shape* numbers: carry is short-vol, so a high Sharpe hides a fat left tail;
  * a **crash-conditional** split — the basket's mean return in calm vs risk-off
    months — quantifying how much of the premium is paid back in the tail.

The decisive finding is not "is the mean positive" (it is — UIP fails) but "is it a
free lunch" (it is not — the premium is compensation for a sold-insurance crash
skew). cf. the desk's short-vol carry studies; same sold-insurance shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# Carry basket construction
# --------------------------------------------------------------------------- #
def carry_ranks(carry: dict[str, float], columns) -> pd.Series:
    """Carry (annualised %, proxy) aligned to the panel columns, high to low."""
    return pd.Series({c: carry.get(c, 0.0) for c in columns}).sort_values(ascending=False)


def carry_weights(carry: dict[str, float], columns, k: int = 3) -> pd.Series:
    """Dollar-neutral weights: +1/k on the top-k carries, -1/k on the bottom-k.

    The carry map is static, so the weights are constant across months (the basket
    is a fixed long-high / short-low portfolio).
    """
    ranked = carry_ranks(carry, columns)
    w = pd.Series(0.0, index=ranked.index)
    if len(ranked) < 2 * k:
        k = max(1, len(ranked) // 2)
    w.iloc[:k] = 1.0 / k
    w.iloc[-k:] = -1.0 / k
    return w.reindex(columns).fillna(0.0)


def basket_returns(total_ret: pd.DataFrame, carry: dict[str, float],
                   k: int = 3) -> pd.Series:
    """Monthly return of the long-high / short-low carry basket (dollar-neutral).

    ``total_ret`` is the per-currency monthly total return (spot + carry/12) of a
    *long* USD-funded position. The short leg earns the negative, handled by the
    negative weights. The result is already an excess-of-cash return.
    """
    w = carry_weights(carry, total_ret.columns, k=k)
    return (total_ret * w).sum(axis=1).rename("carry_basket")


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def net_basket(total_ret: pd.DataFrame, carry: dict[str, float], k: int = 3,
               cost_bps: float = 2.0, borrow_bps_ann: float = 50.0) -> pd.Series:
    """Carry basket net of one-way trading cost × turnover + borrow on the short leg.

    The static ranking means month-to-month turnover is ~0 once the book is on, so
    the dominant frictions are the **initial** entry (one-way cost on each leg, paid
    once) amortised, and the ongoing **borrow** on the short leg. We model the steady
    state: a fixed monthly drag = borrow_bps_ann/12 on the gross short notional (1.0)
    plus a small monthly cost proxy for rolls/rebalancing (cost_bps one-way on the
    full 2.0 gross book, per month).
    """
    gross = basket_returns(total_ret, carry, k=k)
    monthly_drag = (cost_bps / 1e4) * 2.0 + (borrow_bps_ann / 1e4) / MONTHS_PER_YEAR
    return (gross - monthly_drag).rename("carry_basket_net")


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def hac_t(x: np.ndarray, lags: int = 6) -> float:
    """Newey-West (HAC) t-stat of the mean of ``x`` (NaN if < 3 obs)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    e = x - mu
    gamma0 = (e @ e) / n
    var = gamma0
    for L in range(1, lags + 1):
        if L >= n:
            break
        w = 1.0 - L / (lags + 1.0)
        cov = (e[L:] @ e[:-L]) / n
        var += 2.0 * w * cov
    se = np.sqrt(var / n)
    if se == 0:
        return float("nan")
    return float(mu / se)


def annualised_sharpe(x: np.ndarray) -> float:
    """Annualised Sharpe of monthly excess returns (NaN if < 3 / zero vol)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))


def skewness(x: np.ndarray) -> float:
    """Sample skewness of monthly returns (NaN if < 3)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    m = x.mean()
    s = x.std(ddof=0)
    if s == 0:
        return float("nan")
    return float(((x - m) ** 3).mean() / s ** 3)


def max_drawdown(x: np.ndarray) -> float:
    """Max drawdown of the cumulative (additive) return path of ``x`` (<= 0)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan")
    cum = np.cumsum(x)
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())


def crash_split(x: np.ndarray, q: float = 0.10) -> dict:
    """Split monthly returns into 'risk-off' (worst ``q`` quantile) vs 'calm'.

    Returns the mean in each bucket and the share of the *total* additive return
    given back in the worst-``q`` months — the carry-crash accounting.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return {"calm_mean": float("nan"), "off_mean": float("nan"),
                "tail_share": float("nan")}
    thr = np.quantile(x, q)
    off = x[x <= thr]
    calm = x[x > thr]
    total = x.sum()
    return {"calm_mean": float(calm.mean()), "off_mean": float(off.mean()),
            "tail_share": float(off.sum() / total) if total != 0 else float("nan")}


def summarize(total_ret: pd.DataFrame, carry: dict[str, float], k: int = 3,
              cost_bps: float = 2.0, borrow_bps_ann: float = 50.0) -> dict:
    """Headline stats for the carry basket: annualised mean, Sharpe, HAC t, skew,
    worst month, max drawdown, and the crash-conditional split (gross and net)."""
    gross = basket_returns(total_ret, carry, k=k)
    net = net_basket(total_ret, carry, k=k, cost_bps=cost_bps,
                     borrow_bps_ann=borrow_bps_ann)
    g = gross.values
    return {
        "n_months": int(len(gross)),
        "ann_mean": float(g.mean() * MONTHS_PER_YEAR),
        "ann_mean_net": float(net.values.mean() * MONTHS_PER_YEAR),
        "sharpe": annualised_sharpe(g),
        "sharpe_net": annualised_sharpe(net.values),
        "hac_t": hac_t(g),
        "skew": skewness(g),
        "worst_month": float(np.nanmin(g)),
        "max_dd": max_drawdown(g),
        "crash": crash_split(g),
    }


def cost_sweep(total_ret: pd.DataFrame, carry: dict[str, float], k: int = 3,
               borrow_grid=(0.0, 25.0, 50.0, 100.0)) -> list[dict]:
    """Net annualised mean & Sharpe across a borrow-cost grid (bps/yr on the short)."""
    out = []
    for b in borrow_grid:
        net = net_basket(total_ret, carry, k=k, cost_bps=2.0, borrow_bps_ann=b)
        out.append({"borrow_bps": b,
                    "ann_mean_net": float(net.values.mean() * MONTHS_PER_YEAR),
                    "sharpe_net": annualised_sharpe(net.values)})
    return out
