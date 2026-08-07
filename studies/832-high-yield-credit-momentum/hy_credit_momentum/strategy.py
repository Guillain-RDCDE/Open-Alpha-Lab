"""Strategy + inference for Study 832 — High-Yield Credit Momentum.

The claim: the **trailing total-return trend of duration-hedged high-yield credit**
(HYG in excess of IEF) times equity risk. Positive credit trend → risk-on → hold SPY;
negative → de-risk → hold IEF. We test two things, honestly and separately:

* **Signal (discrimination).** Does the credit trend known at the close of ``t-1``
  *discriminate* the day-``t`` equity-excess return ``xs = r_SPY − r_IEF``? We split
  ``xs`` into risk-on days (trend > 0) and risk-off days (trend ≤ 0) and take a
  Newey-West *t* on the **difference of means** (risk-on − risk-off). A positive,
  robust *t* means the trend genuinely precedes SPY out-earning IEF. (Splitting on
  ``xs`` — not on raw SPY — nets out the unconditional equity risk premium, so a coin
  that is "always risk-on" scores ~0, not a spurious win.)

* **Tradability (the timer vs buy-and-hold).** Build the actual switch: hold SPY when
  the trend known at ``t-1`` is positive, else hold IEF. Charge one-way cost × NAV per
  switch **leg** (a SPY↔IEF switch is 2 legs). Compare the **net** Sharpe / CAGR against
  a 100%-SPY buy-and-hold, with a Newey-West *t* on the daily active return
  (strategy − buy-and-hold).

Distinct from [115-credit-spreads](../../115-credit-spreads/) (credit-spread **level** as
a warning), [795-corporate-bond-momentum](../../795-corporate-bond-momentum/)
(cross-sectional bond momentum), [131-utilities-canary](../../131-utilities-canary/)
(utilities relative strength as the risk canary).

Inference primitives (``one_sample_t`` / ``welch_t`` / ``newey_west_t`` /
``wilson_interval`` / a permutation ``placebo`` / a costed ``timer``) mirror the desk's
canonical study 803.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Returns + the credit-trend signal
# --------------------------------------------------------------------------- #
def daily_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns of each ETF close (index=date, columns=ticker)."""
    return panel.sort_index().pct_change()


def credit_trend(panel: pd.DataFrame, lookback: int = 126) -> pd.Series:
    """Trailing ``lookback``-day total return of **HYG in excess of IEF**.

    ``trend[t] = (HYG_t / HYG_{t-L} − 1) − (IEF_t / IEF_{t-L} − 1)`` — the duration-hedged
    credit trend, positive when high-yield out-earned duration-matched Treasuries over the
    window (risk appetite rising). Value on row ``t`` uses prices through ``t``; the timing
    rules shift by one day so a day-``t`` position uses only information known at ``t-1``.
    """
    p = panel.sort_index()
    hyg = p["HYG"] / p["HYG"].shift(lookback) - 1.0
    ief = p["IEF"] / p["IEF"].shift(lookback) - 1.0
    out = hyg - ief
    out.name = "credit_trend"
    return out


# --------------------------------------------------------------------------- #
# Signal (discrimination) — does the trend precede SPY out-earning IEF?
# --------------------------------------------------------------------------- #
def signal_frame(panel: pd.DataFrame, lookback: int = 126) -> pd.DataFrame:
    """Assemble the day-``t`` frame used by both the signal and the timer.

    Columns: ``trend`` (credit trend known at ``t-1``, one ``shift``), ``risk_on``
    (1 if that trend > 0 else 0), ``r_spy``/``r_ief`` (day-``t`` simple returns), and
    ``xs`` = ``r_spy − r_ief`` (the equity-excess-over-duration the timer tries to time).
    """
    ret = daily_returns(panel)
    trend_lag = credit_trend(panel, lookback).shift(1)   # known at close t-1
    df = pd.DataFrame({
        "trend": trend_lag,
        "risk_on": (trend_lag > 0).astype(float),
        "r_spy": ret["SPY"],
        "r_ief": ret["IEF"],
    })
    df["xs"] = df["r_spy"] - df["r_ief"]
    return df.dropna(subset=["trend", "r_spy", "r_ief"])


def signal_stats(panel: pd.DataFrame, lookback: int = 126, nw_lags: int = 10) -> dict:
    """Discrimination test: mean day-``t`` equity-excess ``xs`` on risk-on vs risk-off
    days, and the Newey-West *t* on the difference (risk-on − risk-off).
    """
    df = signal_frame(panel, lookback)
    on = df.loc[df["risk_on"] == 1, "xs"].to_numpy(dtype=float)
    off = df.loc[df["risk_on"] == 0, "xs"].to_numpy(dtype=float)
    # HAC t on the difference via a TIME-ORDERED regime-contrast series whose mean equals
    # mean(xs|on) − mean(xs|off); a Newey-West t on it correctly widens the SE when the
    # risk-on regime is persistent (few effective independent observations).
    diff_series = _regime_diff_series(df["risk_on"].to_numpy(), df["xs"].to_numpy())
    return {
        "n_days": int(len(df)),
        "n_on": int(on.size),
        "n_off": int(off.size),
        "on_bps": float(np.nanmean(on) * 1e4) if on.size else float("nan"),
        "off_bps": float(np.nanmean(off) * 1e4) if off.size else float("nan"),
        "diff_bps": float((np.nanmean(on) - np.nanmean(off)) * 1e4)
        if on.size and off.size else float("nan"),
        "t_nw": newey_west_t(diff_series, nw_lags),
        "welch_t": welch_t(on, off),
        "on_frac": float(df["risk_on"].mean()),
    }


def _regime_diff_series(flag: np.ndarray, x: np.ndarray) -> np.ndarray:
    """A **time-ordered** contrast series ``g`` with ``mean(g) = mean(x|on) − mean(x|off)``.

    With ``p = P(on)``, ``g[t] = x[t]·1{on}/p − x[t]·1{off}/(1−p)``. Averaging over ``t``
    gives ``mean(x|on) − mean(x|off)`` exactly, and — because ``g`` stays in calendar
    order — a Newey-West *t* on it inherits the regime persistence: long risk-on runs
    inflate the HAC variance and shrink the *t*, which is the honest penalty for a
    slow-moving conditioning signal.
    """
    flag = np.asarray(flag, dtype=float)
    x = np.asarray(x, dtype=float)
    p_on = float(flag.mean())
    if p_on <= 0.0 or p_on >= 1.0:
        return np.array([np.nan])
    return x * flag / p_on - x * (1.0 - flag) / (1.0 - p_on)


# --------------------------------------------------------------------------- #
# Inference primitives (mirror of study 803)
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


def sharpe(daily: np.ndarray) -> float:
    daily = np.asarray(daily, dtype=float)
    daily = daily[~np.isnan(daily)]
    if daily.size < 2:
        return float("nan")
    sd = daily.std(ddof=1)
    return float(daily.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def cagr(daily: np.ndarray) -> float:
    daily = np.asarray(daily, dtype=float)
    daily = daily[~np.isnan(daily)]
    if daily.size == 0:
        return float("nan")
    growth = float(np.prod(1.0 + daily))
    yrs = daily.size / TRADING_DAYS
    return growth ** (1.0 / yrs) - 1.0 if growth > 0 and yrs > 0 else float("nan")


def max_drawdown(daily: np.ndarray) -> float:
    daily = np.asarray(daily, dtype=float)
    daily = daily[~np.isnan(daily)]
    if daily.size == 0:
        return float("nan")
    eq = np.cumprod(1.0 + daily)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


# --------------------------------------------------------------------------- #
# The costed timer vs buy-and-hold
# --------------------------------------------------------------------------- #
def timer_stats(
    panel: pd.DataFrame,
    lookback: int = 126,
    cost_bps: float = 5.0,
    nw_lags: int = 10,
) -> dict:
    """Backtest the SPY↔IEF switch and compare (net) to a 100%-SPY buy-and-hold.

    ``risk_on`` (trend at ``t-1`` > 0) → hold SPY on day ``t``; else hold IEF. A switch
    (position change) turns the whole book over: sell one ETF, buy the other = **2 legs**,
    charged ``cost_bps`` one-way each × NAV on the switch day. Long-only, so no borrow.
    Returns gross/net Sharpe, CAGR, max-DD for the timer and the benchmark, plus a
    Newey-West *t* on the daily **active** return (timer_net − buy-and-hold).
    """
    df = signal_frame(panel, lookback)
    pos = df["risk_on"].to_numpy(dtype=float)          # 1 = SPY, 0 = IEF (day t)
    r_spy = df["r_spy"].to_numpy(dtype=float)
    r_ief = df["r_ief"].to_numpy(dtype=float)

    gross = pos * r_spy + (1.0 - pos) * r_ief
    # turnover: |pos_t - pos_{t-1}|; a full switch (Δ=1) costs 2 legs of one-way cost
    prev = np.concatenate([[0.0], pos[:-1]])            # start flat -> first day entry
    switch = np.abs(pos - prev)                         # 1 on a switch day, else 0
    leg_cost = 2.0 * cost_bps / 1e4
    costs = switch * leg_cost
    net = gross - costs

    bh = r_spy                                          # 100%-SPY buy-and-hold
    active = net - bh

    n_switch = int((switch > 0).sum())
    return {
        "n_days": int(len(df)),
        "n_switches": n_switch,
        "on_frac": float(pos.mean()),
        "gross_sharpe": sharpe(gross),
        "net_sharpe": sharpe(net),
        "bh_sharpe": sharpe(bh),
        "net_cagr": cagr(net),
        "bh_cagr": cagr(bh),
        "net_maxdd": max_drawdown(net),
        "bh_maxdd": max_drawdown(bh),
        "active_bps": float(np.nanmean(active) * 1e4),
        "active_t_nw": newey_west_t(active, nw_lags),
        "cost_drag_bps_yr": float(costs.mean() * TRADING_DAYS * 1e4),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the discrimination a lucky alignment of the risk-on labelling?
# --------------------------------------------------------------------------- #
def placebo_pvalue(
    panel: pd.DataFrame,
    lookback: int = 126,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 832,
) -> dict:
    """Keep the day-``t`` equity-excess series but **circularly shift** the risk-on
    labels by a random offset (breaks the trend→outcome link, preserves each series'
    autocorrelation and the risk-on fraction). p = share of shuffled worlds whose
    (on−off) difference is >= the observed one (right-tail test).
    """
    df = signal_frame(panel, lookback)
    flag = df["risk_on"].to_numpy(dtype=float)
    xs = df["xs"].to_numpy(dtype=float)
    n = len(xs)
    obs = float(xs[flag == 1].mean() - xs[flag == 0].mean()) if n else float("nan")

    diffs = []
    if n > 10:
        for seed in range(n_seeds):
            rng = np.random.default_rng(base_seed + seed)
            for _ in range(n_draws_per_seed):
                k = int(rng.integers(1, n))
                fperm = np.roll(flag, k)
                on = xs[fperm == 1]
                off = xs[fperm == 0]
                if on.size and off.size:
                    diffs.append(on.mean() - off.mean())
    diffs = np.asarray(diffs)
    return {
        "obs_bps": obs * 1e4,
        "placebo_mean_bps": float(diffs.mean() * 1e4) if diffs.size else float("nan"),
        "placebo_sd_bps": float(diffs.std(ddof=1) * 1e4) if diffs.size > 1 else float("nan"),
        "p_value": float((diffs >= obs).mean()) if diffs.size else float("nan"),
        "n_draws": int(diffs.size),
        "draws_bps": diffs * 1e4,
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: pd.DataFrame, lookback: int = 126) -> dict:
    """Run the headline signal + timer stats on a synthetic panel."""
    s = signal_stats(panel, lookback)
    t = timer_stats(panel, lookback, cost_bps=0.0)
    return {
        "diff_bps": s["diff_bps"], "t_nw": s["t_nw"], "welch_t": s["welch_t"],
        "active_t_nw": t["active_t_nw"], "net_sharpe": t["net_sharpe"],
        "bh_sharpe": t["bh_sharpe"], "n_days": s["n_days"],
    }
