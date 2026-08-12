"""Strategy + inference for Study 865 — Credit → Equity Lead-Lag.

The claim: **"credit leads equity"** — duration-hedged high-yield credit (HYG in excess of
IEF) *turns before* stocks, so its trailing 1-4-week return should **predict the next
week's SPY return** (a Granger-style lead-lag). We test two things, honestly and
separately:

* **Signal (predictability).** A weekly **predictive regression** of *next* week's SPY
  return on the trailing ``k``-week duration-hedged HY-excess return known at the close of
  week ``t``. We report the slope, its **Newey-West** *t*, and the R². A positive, robust
  slope means the credit trend genuinely *leads* equity by a week. As a companion read we
  also split next-week SPY by the sign of the credit trend (risk-on vs risk-off) and take a
  Newey-West *t* on the difference of means — the discrimination the timing overlay lives on.

* **Tradability (the overlay vs buy-and-hold).** Build the actual overlay: hold SPY next
  week when the trailing credit trend known at week ``t`` is positive, else de-risk to IEF.
  Charge one-way cost × NAV per switch **leg** (a SPY↔IEF switch is 2 legs). Compare the
  **net** Sharpe / CAGR / max-drawdown against a 100%-SPY buy-and-hold, with a Newey-West
  *t* on the weekly active return (overlay − buy-and-hold).

Distinct from [115-credit-spreads](../../115-credit-spreads/) (credit-spread **level** as a
stress warning), [832-high-yield-credit-momentum](../../832-high-yield-credit-momentum/)
(HY **momentum** graded as its own daily SPY↔IEF timer on the trend *sign*),
[131-utilities-canary](../../131-utilities-canary/) (utilities relative strength as the
risk canary), [379-etf-lead-lag](../../379-etf-lead-lag/) (generic cross-ETF lead-lag).
This study's own axis is the **weekly Granger-style predictive regression of next-week SPY
on the trailing HY-excess return** — a forecast object, not a level, a same-asset momentum,
or a sign-switch.

Inference primitives (``one_sample_t`` / ``welch_t`` / ``newey_west_t`` /
``nw_regression`` / ``wilson_interval`` / a permutation ``placebo`` / a costed ``overlay``)
mirror the desk's canonical study 803.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_WEEKS = 52
WEEK_RULE = "W-FRI"


# --------------------------------------------------------------------------- #
# Weekly returns + the duration-hedged credit trend
# --------------------------------------------------------------------------- #
def weekly_prices(panel: pd.DataFrame) -> pd.DataFrame:
    """Last close of each calendar week (Friday-anchored) for each ETF."""
    return panel.sort_index().resample(WEEK_RULE).last().dropna(how="any")


def weekly_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Weekly simple returns of each ETF (Friday-to-Friday close-to-close)."""
    return weekly_prices(panel).pct_change()


def credit_trend(panel: pd.DataFrame, lookback_wk: int = 4) -> pd.Series:
    """Trailing ``lookback_wk``-week total return of **HYG in excess of IEF**.

    ``trend[t] = (HYG_t / HYG_{t-k} − 1) − (IEF_t / IEF_{t-k} − 1)`` on the *weekly* close
    grid — the duration-hedged credit move over the trailing window, positive when
    high-yield out-earned duration-matched Treasuries (risk appetite rising). The value on
    week ``t`` uses prices through the close of week ``t``; the lead-lag tests align it
    against the *following* week's SPY return, so there is no look-ahead.
    """
    wp = weekly_prices(panel)
    hyg = wp["HYG"] / wp["HYG"].shift(lookback_wk) - 1.0
    ief = wp["IEF"] / wp["IEF"].shift(lookback_wk) - 1.0
    out = hyg - ief
    out.name = "credit_trend"
    return out


def leadlag_frame(panel: pd.DataFrame, lookback_wk: int = 4) -> pd.DataFrame:
    """Assemble the weekly frame used by both the regression and the overlay.

    Columns: ``trend`` (trailing ``k``-week HY-excess through the close of week ``t``),
    ``risk_on`` (1 if that trend > 0 else 0), ``r_spy_next`` (SPY return in week ``t+1``),
    ``r_ief_next`` (IEF return in week ``t+1``). Aligning the trend at ``t`` with the
    ``t+1`` returns is the one-week execution lag: a signal read at Friday close is acted
    on over the following week. Zero look-ahead.
    """
    wr = weekly_returns(panel)
    trend = credit_trend(panel, lookback_wk)
    spy_next = wr["SPY"].shift(-1)
    ief_next = wr["IEF"].shift(-1)
    df = pd.DataFrame({
        "trend": trend,
        "risk_on": (trend > 0).astype(float),
        "r_spy_next": spy_next,
        "r_ief_next": ief_next,
    })
    return df.dropna(subset=["trend", "r_spy_next", "r_ief_next"])


# --------------------------------------------------------------------------- #
# Signal (predictability) — the Granger-style predictive regression
# --------------------------------------------------------------------------- #
def leadlag_regression(panel: pd.DataFrame, lookback_wk: int = 4, nw_lags: int = 6) -> dict:
    """Predictive regression: next-week SPY return on the trailing ``k``-week HY-excess.

    ``r_SPY[t+1] = a + b · trend_k[t] + u``. Reports the slope ``b`` (in units of
    "SPY-return per 1.0 of HY-excess trend"), a scale-free per-σ effect (``b × σ_trend``,
    the SPY move for a one-standard-deviation credit trend), the Newey-West (HAC) *t* on the
    slope, and the R². A positive, robust *t* is the "credit leads equity" claim.
    """
    df = leadlag_frame(panel, lookback_wk)
    x = df["trend"].to_numpy(dtype=float)
    y = df["r_spy_next"].to_numpy(dtype=float)
    beta, t_nw, r2 = nw_regression(x, y, nw_lags)
    return {
        "n_weeks": int(len(df)),
        "lookback_wk": int(lookback_wk),
        "beta": beta,
        "beta_t_nw": t_nw,
        "r2": r2,
        "sd_trend": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
        # SPY move (bps) for a one-σ credit trend:
        "per_sd_bps": float(beta * np.std(x, ddof=1) * 1e4) if len(x) > 1 else float("nan"),
        "corr": float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan"),
    }


def signal_stats(panel: pd.DataFrame, lookback_wk: int = 4, nw_lags: int = 6) -> dict:
    """Companion discrimination test: mean next-week SPY return on risk-on (trend > 0) vs
    risk-off (trend ≤ 0) weeks, and the Newey-West *t* on the difference (risk-on − off).

    Splitting on next-week SPY (not on an excess) is deliberate: this is the exact quantity
    the timing overlay earns, so a positive robust difference is the overlay's edge.
    """
    df = leadlag_frame(panel, lookback_wk)
    on = df.loc[df["risk_on"] == 1, "r_spy_next"].to_numpy(dtype=float)
    off = df.loc[df["risk_on"] == 0, "r_spy_next"].to_numpy(dtype=float)
    diff_series = _regime_diff_series(df["risk_on"].to_numpy(), df["r_spy_next"].to_numpy())
    return {
        "n_weeks": int(len(df)),
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
    inflate the HAC variance and shrink the *t*, the honest penalty for a slow-moving
    conditioning signal.
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


def newey_west_t(x: np.ndarray, lags: int = 6) -> float:
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


def nw_regression(x: np.ndarray, y: np.ndarray, lags: int = 6) -> tuple[float, float, float]:
    """OLS slope of ``y = a + b·x`` with a Newey-West (HAC, Bartlett) *t* on the slope.

    Returns ``(beta, t_nw, r2)``. The HAC standard error uses the score series
    ``g[t] = (x[t] − x̄)·u[t]`` (``u`` = OLS residual): the sandwich variance of ``b`` is
    ``S / (Σ xc²)²`` with ``S`` the Bartlett-weighted long-run variance of ``Σ g``. Pure
    numpy — no statsmodels dependency in the offline path.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    n = len(x)
    if n < 3:
        return (float("nan"), float("nan"), float("nan"))
    xc = x - x.mean()
    sxx = float(xc @ xc)
    if sxx <= 0:
        return (float("nan"), float("nan"), float("nan"))
    beta = float((xc @ (y - y.mean())) / sxx)
    alpha = float(y.mean() - beta * x.mean())
    resid = y - (alpha + beta * x)
    g = xc * resid
    # Bartlett-weighted long-run variance of the score sum (levels, not divided by n)
    S = float(g @ g)
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        S += 2.0 * w * float(g[l:] @ g[:-l])
    var_beta = S / (sxx * sxx)
    t_nw = beta / np.sqrt(var_beta) if var_beta > 0 else float("nan")
    ss_tot = float((y - y.mean()) @ (y - y.mean()))
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return (beta, float(t_nw), float(r2))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def sharpe(weekly: np.ndarray) -> float:
    weekly = np.asarray(weekly, dtype=float)
    weekly = weekly[~np.isnan(weekly)]
    if weekly.size < 2:
        return float("nan")
    sd = weekly.std(ddof=1)
    return float(weekly.mean() / sd * np.sqrt(TRADING_WEEKS)) if sd > 0 else float("nan")


def cagr(weekly: np.ndarray) -> float:
    weekly = np.asarray(weekly, dtype=float)
    weekly = weekly[~np.isnan(weekly)]
    if weekly.size == 0:
        return float("nan")
    growth = float(np.prod(1.0 + weekly))
    yrs = weekly.size / TRADING_WEEKS
    return growth ** (1.0 / yrs) - 1.0 if growth > 0 and yrs > 0 else float("nan")


def max_drawdown(weekly: np.ndarray) -> float:
    weekly = np.asarray(weekly, dtype=float)
    weekly = weekly[~np.isnan(weekly)]
    if weekly.size == 0:
        return float("nan")
    eq = np.cumprod(1.0 + weekly)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


# --------------------------------------------------------------------------- #
# The costed timing overlay vs buy-and-hold
# --------------------------------------------------------------------------- #
def overlay_stats(
    panel: pd.DataFrame,
    lookback_wk: int = 4,
    cost_bps: float = 5.0,
    nw_lags: int = 6,
) -> dict:
    """Backtest the weekly SPY↔IEF overlay and compare (net) to a 100%-SPY buy-and-hold.

    ``risk_on`` (trailing credit trend at week ``t`` > 0) → hold SPY over week ``t+1``;
    else hold IEF. A switch (position change) turns the whole book over: sell one ETF, buy
    the other = **2 legs**, charged ``cost_bps`` one-way each × NAV on the switch week.
    Long-only, so no borrow. Returns gross/net Sharpe, CAGR, max-DD for the overlay and the
    benchmark, plus a Newey-West *t* on the weekly **active** return (overlay_net − B&H).
    """
    df = leadlag_frame(panel, lookback_wk)
    pos = df["risk_on"].to_numpy(dtype=float)          # 1 = SPY, 0 = IEF (week t+1)
    r_spy = df["r_spy_next"].to_numpy(dtype=float)
    r_ief = df["r_ief_next"].to_numpy(dtype=float)

    gross = pos * r_spy + (1.0 - pos) * r_ief
    prev = np.concatenate([[0.0], pos[:-1]])           # start flat -> first week entry
    switch = np.abs(pos - prev)                        # 1 on a switch week, else 0
    leg_cost = 2.0 * cost_bps / 1e4
    costs = switch * leg_cost
    net = gross - costs

    bh = r_spy                                         # 100%-SPY buy-and-hold
    active = net - bh

    return {
        "n_weeks": int(len(df)),
        "n_switches": int((switch > 0).sum()),
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
        "cost_drag_bps_yr": float(costs.mean() * TRADING_WEEKS * 1e4),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the lead-lag a lucky alignment of the risk-on labelling?
# --------------------------------------------------------------------------- #
def placebo_pvalue(
    panel: pd.DataFrame,
    lookback_wk: int = 4,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 865,
) -> dict:
    """Keep the next-week SPY series but **circularly shift** the risk-on labels by a
    random offset (breaks the lead→outcome link, preserves each series' autocorrelation and
    the risk-on fraction). p = share of shuffled worlds whose (on−off) difference is >= the
    observed one (right-tail test).
    """
    df = leadlag_frame(panel, lookback_wk)
    flag = df["risk_on"].to_numpy(dtype=float)
    y = df["r_spy_next"].to_numpy(dtype=float)
    n = len(y)
    obs = float(y[flag == 1].mean() - y[flag == 0].mean()) if n else float("nan")

    diffs = []
    if n > 10:
        for seed in range(n_seeds):
            rng = np.random.default_rng(base_seed + seed)
            for _ in range(n_draws_per_seed):
                k = int(rng.integers(1, n))
                fperm = np.roll(flag, k)
                on = y[fperm == 1]
                off = y[fperm == 0]
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
def synthetic_detect(panel: pd.DataFrame, lookback_wk: int = 4) -> dict:
    """Run the headline lead-lag regression + overlay on a synthetic panel."""
    r = leadlag_regression(panel, lookback_wk)
    s = signal_stats(panel, lookback_wk)
    t = overlay_stats(panel, lookback_wk, cost_bps=0.0)
    return {
        "beta_t_nw": r["beta_t_nw"], "per_sd_bps": r["per_sd_bps"], "r2": r["r2"],
        "diff_bps": s["diff_bps"], "t_nw": s["t_nw"], "welch_t": s["welch_t"],
        "active_t_nw": t["active_t_nw"], "net_sharpe": t["net_sharpe"],
        "bh_sharpe": t["bh_sharpe"], "n_weeks": r["n_weeks"],
    }
