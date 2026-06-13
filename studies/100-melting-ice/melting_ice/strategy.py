"""The leveraged-ETF simulator and the volatility-decay decomposition — Study 100.

The folklore: *"a 3x ETF (TQQQ, UPRO) is toxic — 'volatility decay' guarantees it bleeds
to zero; never hold it more than a day."* The kernel of truth is the constant-leverage
**daily rebalancing identity**: a fund that resets to k-times exposure every day does
NOT deliver k-times the period return. From daily underlying returns ``r_t`` the levered
NAV compounds as

    L_T = prod_t ( 1 + k*r_t - daily_fee )

while the naive "k-times the period return" benchmark a believer pictures is

    naive_k_T = 1 + k * (prod_t (1 + r_t) - 1).

The gap between them is **path-dependent**, and the log-return identity makes the two
opposing forces explicit (Cheng & Madhavan 2009; Avellaneda & Zhang 2010). To leading
order, with mean daily log-return ``g`` and daily variance ``s2``,

    E[ log(1 + k*r) ]  ≈  k*g  -  (k(k-1)/2) * s2 .          # the variance-drag term

So a 3x sleeve earns ~3x the *drift* but pays a drag of ``3*2/2 = 3`` times the variance.
Decay is **not** a law: it is a race between ``k*g`` (drift, helped by compounding in a
trend) and ``(k(k-1)/2)*s2`` (the drag). The break-even is the variance at which they tie:

    s2*  =  g / ( (k-1)/2 )   ⇔   for k=3,  s2* = g .

Above ``s2*`` the levered log-CAGR falls below k*underlying; below it, it exceeds it.

This module:
  - simulates a levered ETF from the underlying daily returns with an annual expense +
    financing approximation (charged ONCE, as a daily fee — never the bench's
    twice-charged-rate wound),
  - decomposes realised vs naive-k and the drag-vs-drift terms,
  - computes the break-even variance,
  - reports drawdown and a local Newey-West HAC t-stat where relevant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Real-world all-in cost of a 3x ETF, charged ONCE as a daily fee (annual / TRADING_DAYS):
#   ~0.95% expense ratio
# + financing on the 2x BORROWED notional at ~the short rate (averaging ~2%/yr over
#   2009-2026, but >5%/yr in 2023-24) -> ~2x * ~2% ≈ a few %/yr
# + small tracking slippage.
# The implied all-in drag that reconciles the simulator's final NAV with the *real*
# TQQQ/UPRO tape is ~5%/yr (see examples/verify.py, reproduction_error). We use 5.0%/yr
# as the stated default so the simulator reproduces the real fund within tolerance; the
# decomposition and break-even below use raw underlying returns and are fee-INDEPENDENT.
DEFAULT_ANNUAL_FEE = 0.050


# ---------------------------------------------------------------------------
# The constant-leverage simulator
# ---------------------------------------------------------------------------
def simulate_letf(
    close: pd.Series,
    k: float = 3.0,
    annual_fee: float = DEFAULT_ANNUAL_FEE,
) -> pd.Series:
    """Simulate a k-times daily-rebalanced ETF NAV from an underlying close series.

    The fund resets to k-times exposure every day: its daily return is ``k * r_t`` minus
    a daily fee, and the NAV is the running product. ``annual_fee`` (expense + financing
    approximation) is converted to a per-day charge and subtracted ONCE. Returns the NAV
    series normalised to 1.0 at the first date.
    """
    r = close.pct_change().fillna(0.0)
    daily_fee = annual_fee / TRADING_DAYS
    lev_ret = k * r - daily_fee
    lev_ret.iloc[0] = 0.0
    nav = (1.0 + lev_ret).cumprod()
    return nav.rename(f"{k:g}x")


def naive_k_curve(close: pd.Series, k: float = 3.0) -> pd.Series:
    """The believer's mental model: 1 + k*(cumulative underlying return).

    This is *not* a tradable NAV — it is the straight-line 'k times the period return'
    that the 'always decays' folklore implicitly compares against. The whole point of the
    study is that the real levered NAV departs from this line in *both* directions.
    """
    cum = close / close.iloc[0] - 1.0
    return (1.0 + k * cum).rename(f"naive-{k:g}x")


# ---------------------------------------------------------------------------
# The volatility-decay decomposition
# ---------------------------------------------------------------------------
def drag_decomposition(close: pd.Series, k: float = 3.0) -> dict:
    """Split the levered log-CAGR into the drift term and the variance-drag term.

    Using daily log-returns ``x_t = log(1+r_t)`` of the underlying, with mean ``g`` and
    variance ``s2``:

      - levered drift contribution per day  ≈  k * g
      - variance-drag per day               ≈  (k(k-1)/2) * s2
      - leading-order levered log-CAGR/day  ≈  k*g - (k(k-1)/2)*s2

    Returns annualised drift, drag, their net, the realised underlying/levered log-CAGRs,
    the break-even variance ``s2_star`` (annualised vol at break-even too), and whether
    drag overwhelms drift on this tape.
    """
    r = close.pct_change().dropna()
    x = np.log1p(r.to_numpy())
    g = float(x.mean())
    s2 = float(x.var(ddof=1))

    drift_term = k * g
    drag_term = (k * (k - 1.0) / 2.0) * s2
    net_lead = drift_term - drag_term

    # Break-even: variance at which drag == drift  =>  s2* = g / ((k-1)/2).
    s2_star = g / ((k - 1.0) / 2.0) if k != 1.0 else float("inf")

    return {
        "k": k,
        "g_daily": g,
        "s2_daily": s2,
        "drift_ann": drift_term * TRADING_DAYS,
        "drag_ann": drag_term * TRADING_DAYS,
        "net_leading_ann": net_lead * TRADING_DAYS,
        "under_logcagr_ann": g * TRADING_DAYS,
        "lev_logcagr_ann": net_lead * TRADING_DAYS,  # leading-order levered log-CAGR
        "s2_star_daily": s2_star,
        "breakeven_vol_ann": float(np.sqrt(max(s2_star, 0.0)) * np.sqrt(TRADING_DAYS))
        if s2_star > 0 else float("nan"),
        "realized_vol_ann": float(np.sqrt(s2) * np.sqrt(TRADING_DAYS)),
        "drag_dominates": bool(drag_term > drift_term),
    }


# ---------------------------------------------------------------------------
# Headline stats on a NAV / return series
# ---------------------------------------------------------------------------
def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def nav_stats(nav: pd.Series) -> dict:
    """CAGR / vol / Sharpe / max-drawdown / final multiple from a NAV series (base 1.0)."""
    nav = nav.astype(float)
    ret = nav.pct_change().fillna(0.0)
    n = len(nav)
    years = n / TRADING_DAYS
    final = float(nav.iloc[-1])
    cagr = float(final ** (1.0 / years) - 1.0) if years > 0 and final > 0 else float("nan")
    vol = float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = (
        float(ret.mean() / ret.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if ret.std(ddof=1) > 0 else float("nan")
    )
    return {
        "final": final,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": _max_drawdown(nav.to_numpy()),
    }


def realized_vs_naive(close: pd.Series, k: float = 3.0,
                      annual_fee: float = DEFAULT_ANNUAL_FEE) -> dict:
    """Compare the realised levered NAV to the naive-k line over the whole window.

    Returns the two final multiples and the gap (realised − naive). A positive gap means
    compounding-in-a-trend beat the drag (the levered fund did BETTER than 'k times the
    period return'); a negative gap means decay won.
    """
    nav = simulate_letf(close, k=k, annual_fee=annual_fee)
    naive = naive_k_curve(close, k=k)
    realized_final = float(nav.iloc[-1])
    naive_final = float(naive.iloc[-1])
    return {
        "realized_final": realized_final,
        "naive_final": naive_final,
        "gap": realized_final - naive_final,
        "compounding_won": bool(realized_final > naive_final),
        "nav": nav,
        "naive": naive,
    }


# ---------------------------------------------------------------------------
# Real-tape reproduction: how close is the simulator to the actual fund?
# ---------------------------------------------------------------------------
def reproduction_error(under_close: pd.Series, real_letf_close: pd.Series,
                       k: float = 3.0, annual_fee: float = DEFAULT_ANNUAL_FEE) -> dict:
    """How well the simulator reproduces the *real* levered ETF from its underlying.

    Both series are normalised to 1.0 at the first common date; we report the daily
    tracking-return correlation, the RMS daily tracking error (bps), and the final-NAV
    ratio (simulated / real). A correlation near 1 and a small final-ratio error confirm
    the constant-leverage identity — not magic — is what the fund does.
    """
    sim = simulate_letf(under_close, k=k, annual_fee=annual_fee)
    real = (real_letf_close / real_letf_close.iloc[0]).rename("real")
    sim_r = sim.pct_change().dropna()
    real_r = real.pct_change().dropna()
    common = sim_r.index.intersection(real_r.index)
    sr, rr = sim_r.loc[common].to_numpy(), real_r.loc[common].to_numpy()
    corr = float(np.corrcoef(sr, rr)[0, 1])
    rms_te_bps = float(np.sqrt(np.mean((sr - rr) ** 2)) * 1e4)
    return {
        "corr": corr,
        "rms_te_bps": rms_te_bps,
        "sim_final": float(sim.iloc[-1]),
        "real_final": float(real.iloc[-1]),
        "final_ratio": float(sim.iloc[-1] / real.iloc[-1]),
        "sim": sim,
        "real": real,
    }


# ---------------------------------------------------------------------------
# Inference: HAC t-stat on a daily series
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) t-stat for the mean of ``x`` (local, no quantlab dep)."""
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
    for kk in range(1, lags + 1):
        w = 1.0 - kk / (lags + 1.0)
        lrv += 2.0 * w * float(e[kk:] @ e[:-kk]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")
