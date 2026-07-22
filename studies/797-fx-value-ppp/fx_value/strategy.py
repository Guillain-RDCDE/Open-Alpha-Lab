"""Strategy + inference for Study 797 — FX Value (PPP real-rate mean reversion).

The claim: **currencies whose real exchange rate is undervalued vs their own long-run
average subsequently appreciate.** Build the real rate ``q_i = S_i x CPI_i / CPI_US``,
measure each currency's deviation of ``log q`` from a long trailing average, and go
**long the cheap (undervalued) / short the rich (overvalued)**, dollar-neutral, monthly.

This is the FX **value** factor — the opposite tilt to **carry** (rank on the rate
differential, sibling 364) and a real *time series* of the PPP gap, not the single
Big-Mac folklore snapshot (sibling 215), nor trend/momentum (147) or the dollar-cycle
level (114).

Measurements:

* **The signal.** ``value_i,t = trailing_mean(log q_i, window) - log q_i,t`` (positive
  => currently cheap vs its own history => a long). Cross-sectionally demeaned each
  month; ranked; the long-short book is the top-minus-bottom tercile (or a rank-weighted
  spread), dollar-neutral.
* **One documented execution lag.** The signal is formed at the **close of month t**
  from FX(t) and CPI known through ``t - CPI_PUB_LAG`` (publication lag baked in), and
  it earns the **month t+1** spot return — one ``shift``, applied once. No look-ahead
  into an unreleased CPI print.
* **Inference.** One-sample *t* and a **Newey-West (HAC)** *t* on the monthly long-short
  return (the decisive real-tape number), a **placebo** (random sign/relabelling of the
  cross-section), and a **Wilson** interval on the monthly hit rate.
* **The costed timer.** Turnover = one-way x NAV of the change in weights each rebalance;
  costs charged both legs; **shorts pay a borrow/financing spread**. One-way cost swept.

The decisive number is the Newey-West *t* of the value long-short on the REAL tape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12


# --------------------------------------------------------------------------- #
# Signal
# --------------------------------------------------------------------------- #
def value_signal(logq: pd.DataFrame, window: int = 60, min_periods: int = 36) -> pd.DataFrame:
    """PPP-value signal: trailing-mean(log q) minus current log q, using ONLY past
    data (the trailing mean is computed on a window *ending at t*, so it is
    point-in-time). Positive => the currency's real rate is below its own long
    average => cheap/undervalued => a long. Cross-sectional level is left in; the
    portfolio step demeans."""
    trail = logq.rolling(window, min_periods=min_periods).mean()
    return (trail - logq)


def spot_returns(fx: pd.DataFrame) -> pd.DataFrame:
    """Simple month-over-month FX spot return, USD per foreign (a positive value =
    the foreign currency appreciated vs USD over the month)."""
    return fx.pct_change()


# --------------------------------------------------------------------------- #
# Portfolio construction — dollar-neutral long cheap / short rich
# --------------------------------------------------------------------------- #
def _rank_weights(row: pd.Series) -> pd.Series:
    """Cross-sectional dollar-neutral weights from a signal row: demeaned rank,
    scaled so the gross book is 1 (sum |w| = 1, sum w = 0). Long the high-signal
    (cheap) currencies, short the low-signal (rich) ones."""
    s = row.dropna()
    if len(s) < 3:
        return pd.Series(0.0, index=row.index)
    r = s.rank()
    r = r - r.mean()
    denom = np.abs(r).sum()
    w = r / denom if denom > 0 else r * 0.0
    return w.reindex(row.index).fillna(0.0)


def weights_panel(signal: pd.DataFrame) -> pd.DataFrame:
    """Row-by-row dollar-neutral rank weights (long cheap, short rich)."""
    return signal.apply(_rank_weights, axis=1)


def portfolio_returns(signal: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    """Monthly long-short value return. Weights are formed at the close of month t
    (from the signal, which already carries the CPI publication lag) and earn the
    month t+1 spot return: ONE shift, applied once here.

    returns a Series indexed by the return month (t+1)."""
    w = weights_panel(signal)
    # signal known at close of t -> earns t+1 return: shift weights forward one month
    w_lag = w.shift(1)
    port = (w_lag * rets).sum(axis=1, min_count=1)
    # only months where at least one weight was active
    active = (w_lag.abs().sum(axis=1) > 0)
    return port[active].dropna()


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


def newey_west_t(x: np.ndarray, lags: int = 6) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0 for a scalar mean —
    a regression of x on a constant, HAC-robust standard error on the intercept."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    u = x - x.mean()
    S = (u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        g = (u[l:] @ u[:-l]) / n
        S += 2.0 * w * g
    se = np.sqrt(S / n)
    return float(x.mean() / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def annualized_sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(MONTHS))


# --------------------------------------------------------------------------- #
# Headline stats on a long-short monthly return series
# --------------------------------------------------------------------------- #
def headline_stats(port: pd.Series, nw_lags: int = 6) -> dict:
    x = port.to_numpy(dtype=float)
    k = int((x > 0).sum())
    lo, hi = wilson_interval(k, len(x))
    ann = float(np.nanmean(x) * MONTHS)
    return {
        "n_months": len(x),
        "mean_bps": float(np.nanmean(x) * 1e4),
        "ann_pct": ann * 100,
        "vol_ann_pct": float(np.nanstd(x, ddof=1) * np.sqrt(MONTHS) * 100),
        "sharpe": annualized_sharpe(x),
        "t_one_sample": one_sample_t(x),
        "t_nw": newey_west_t(x, lags=nw_lags),
        "hit": k, "hit_rate": k / len(x), "hit_lo": lo, "hit_hi": hi,
        "start": str(port.index.min().date()), "end": str(port.index.max().date()),
    }


def placebo_pvalue(signal: pd.DataFrame, rets: pd.DataFrame,
                   n_draws_per_seed: int = 500, n_seeds: int = 20,
                   base_seed: int = 797) -> dict:
    """Random-sign placebo: independently flip each month's cross-sectional weight
    signs at random (holding the gross book and the return tape fixed) and record the
    full-sample mean. p = share of random books whose mean Sharpe >= the observed —
    a right-tail test (the claim is a positive value premium)."""
    obs = annualized_sharpe(portfolio_returns(signal, rets).to_numpy())
    w = weights_panel(signal).shift(1)
    cols = w.columns
    Wl = w.to_numpy()
    R = rets.reindex(w.index)[cols].to_numpy()
    valid = np.isfinite(Wl).all(axis=1) & (np.abs(Wl).sum(axis=1) > 0)
    Wl = Wl[valid]
    R = np.nan_to_num(R[valid])
    n = Wl.shape[0]
    sharpes = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            flip = rng.choice([-1.0, 1.0], size=n)[:, None]
            pr = (Wl * flip * R).sum(axis=1)
            sd = pr.std(ddof=1)
            sharpes.append(pr.mean() / sd * np.sqrt(MONTHS) if sd > 0 else 0.0)
    sharpes = np.asarray(sharpes)
    return {"obs_sharpe": float(obs), "placebo_mean": float(sharpes.mean()),
            "placebo_sd": float(sharpes.std(ddof=1)),
            "p_value": float((sharpes >= obs).mean()),
            "n_draws": len(sharpes), "draws": sharpes}


# --------------------------------------------------------------------------- #
# The costed timer — turnover x cost, shorts pay borrow
# --------------------------------------------------------------------------- #
def timer_stats(signal: pd.DataFrame, rets: pd.DataFrame,
                cost_bps: float = 10.0, borrow_bps_ann: float = 100.0) -> dict:
    """Net long-short value return after (a) rebalancing cost = one-way cost x the
    change in weights each month (turnover, both legs charged as |Δw| x NAV), and
    (b) a borrow/financing spread on the short book (borrow_bps_ann, annualised,
    charged monthly on gross short notional). One documented execution lag (weights
    shifted one month) is already inside ``portfolio_returns``."""
    w = weights_panel(signal).shift(1)
    active = (w.abs().sum(axis=1) > 0)
    w = w[active]
    gross = (w * rets.reindex(w.index)).sum(axis=1, min_count=1)

    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    rebal_cost = turnover * (cost_bps / 1e4)
    short_notional = w.clip(upper=0).abs().sum(axis=1)          # gross short book
    borrow_cost = short_notional * (borrow_bps_ann / 1e4 / MONTHS)

    net = (gross - rebal_cost - borrow_cost).dropna()
    x = net.to_numpy(dtype=float)
    return {"n_months": len(x), "cost_bps": cost_bps, "borrow_bps_ann": borrow_bps_ann,
            "gross_ann_pct": float(gross.reindex(net.index).mean() * MONTHS * 100),
            "net_mean_bps": float(np.nanmean(x) * 1e4),
            "net_ann_pct": float(np.nanmean(x) * MONTHS * 100),
            "net_sharpe": annualized_sharpe(x),
            "net_t_nw": newey_west_t(x),
            "avg_turnover": float(turnover.reindex(net.index).mean()),
            "worst_month_pct": float(np.nanmin(x) * 100)}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(logq: pd.DataFrame, window: int = 60, min_periods: int = 36) -> dict:
    """Run the value sort on a synthetic real-rate panel. Spot returns are the change
    in log real rate (CPI held constant in the synthetic world), so the same signal /
    portfolio machinery applies unchanged."""
    sig = value_signal(logq, window=window, min_periods=min_periods)
    rets = logq.diff()
    port = portfolio_returns(sig, rets)
    h = headline_stats(port)
    return {"n_months": h["n_months"], "sharpe": h["sharpe"],
            "t_nw": h["t_nw"], "ann_pct": h["ann_pct"]}
