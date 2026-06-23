"""Strategy + inference for Study 373 — Dispersion-Trade.

The dispersion trade (on a realized-vol proxy here, since implied vols are not free):

    index variance is a correlation-weighted average of single-name variances. Under a
    one-factor approximation, sigma_index ≈ sqrt(rho) * avg_single_sigma, so for pairwise
    correlation rho < 1 the index vol is *strictly below* the average single-name vol. The
    dispersion gap is ``avg_single_vol − index_vol >= 0`` — mechanically positive.

The pitch: "index vol is reliably cheap" — sell index variance, buy single-name variance,
and harvest the gap. We test it in two parts:

  1. **Sign of the gap (the identity).** Is the realized gap reliably positive? Yes, almost
     always — but that is *subadditivity*, a no-arbitrage identity, not an edge. A signal
     that is positive by construction is `NONE` on the "is it a free lunch" axis.
  2. **Carry of the trade (the edge).** Does *holding* the long-single / short-index book
     earn a reliable positive return after a one-day lag and real costs? We build a daily
     **dispersion-carry P&L** (long single-name variance vs short index variance, struck at
     a trailing reference so we earn the gap above what was already priced in), then measure
     its mean, Sharpe, win-rate, a Welch *t* vs a zero/placebo null, and net it of bid/ask +
     vega costs scaled by turnover.

The decisive findings are statistical and operational: the *gap* is real but mechanical
(NONE on the free-lunch axis), and the *carry* is a thin, fat-tailed, correlation-spike-
exposed stream that does not clear a robust t >= 2 once costs are charged. The convexity
the trade is long is exactly the convexity that detonates in a correlation-1 crash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANNUAL = 252.0
TRADING_DAYS = {1: 21, 3: 63, 6: 126, 12: 252}   # months -> trading days


# --------------------------------------------------------------------------- #
# Carry book — long single-name variance / short index variance
# --------------------------------------------------------------------------- #
def carry_pnl(frame: pd.DataFrame, ref_window: int = 63, lag: int = 1) -> pd.Series:
    """Daily P&L (in variance points, annualised) of the dispersion-carry book.

    The book is long single-name realized variance and short index realized variance,
    *struck* at a trailing ``ref_window``-day reference dispersion (what was already
    "priced in" — the proxy for the implied strike). We earn the amount by which today's
    realized dispersion exceeds that trailing strike:

        gap_var_t   = avg_vol_t**2 - index_vol_t**2          (realized dispersion in var pts)
        strike_t    = mean(gap_var) over the prior ref_window days   (the trailing strike)
        pnl_t       = gap_var_{t}   - strike_{t-1}            (mean-reversion-free carry)

    The signal (strike) is known at ``t-1``; we book the realized gap of ``t`` — a single,
    documented one-day execution lag (the ``lag`` argument shifts the strike). No look-ahead.
    Returned in annualised variance points (vol**2); fat-tailed by nature.
    """
    gap_var = frame["avg_vol"] ** 2 - frame["index_vol"] ** 2
    strike = gap_var.rolling(ref_window).mean().shift(lag)
    pnl = (gap_var - strike).dropna()
    return pnl


def synthetic_carry_pnl(frame: pd.DataFrame, ref_window: int = 63, lag: int = 1) -> pd.Series:
    """Carry P&L for a synthetic basket, with the planted ``_carry`` edge added in.

    Builds the dispersion proxy from the synthetic wide price frame, computes the same
    trailing-strike carry as :func:`carry_pnl`, then adds the deterministic planted carry
    stored in ``frame.attrs['_carry']`` (a constant daily accrual in variance points). With
    ``_carry == 0`` the book is the pure mechanical/realized stream; a large planted carry
    must drive the Welch t above 2.
    """
    from . import data as _data
    w = int(frame.attrs.get("_window", 21))
    prox = _data.dispersion_proxy(frame, window=w)
    base = carry_pnl(prox, ref_window=ref_window, lag=lag)
    carry = float(frame.attrs.get("_carry", 0.0))
    return base + carry


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def welch_t_vs_zero(sample: np.ndarray) -> float:
    """t-stat of mean(sample) against zero (one-sample). NaN if sample < 2."""
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    if se == 0:
        return float("nan")
    return float(x.mean() / se)


def hac_t_vs_zero(sample: np.ndarray, lags: int = 21) -> float:
    """Newey-West (HAC) t-stat of the mean against zero, robust to autocorrelation.

    The carry stream is strongly autocorrelated (overlapping realized-vol windows), so the
    naive t overstates significance. This is the honest Signal-axis statistic.
    """
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < lags + 2:
        return float("nan")
    d = x - x.mean()
    gamma0 = (d @ d) / n
    var = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)               # Bartlett kernel
        cov = (d[k:] @ d[:-k]) / n
        var += 2.0 * w * cov
    se = np.sqrt(var / n)
    if se == 0:
        return float("nan")
    return float(x.mean() / se)


def placebo_pvalue(pnl: np.ndarray, block: int = 21, n_draws: int = 20_000,
                   seed: int = 373) -> dict:
    """Sign-flip / block-permutation placebo for the carry mean.

    The carry is autocorrelated, so an i.i.d. shuffle would destroy the clustering that the
    inference must respect. We instead flip the sign of whole ``block``-day chunks at random
    (a stationary-bootstrap-flavoured null that preserves within-block structure) and ask how
    often the placebo mean matches or beats the observed mean. ``p`` is the share of draws
    with ``placebo_mean >= observed_mean`` — the honest "could random sign luck explain it?"
    """
    x = np.asarray(pnl, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return {"obs_mean": float("nan"), "placebo_mean": float("nan"), "p_value": float("nan")}
    obs = float(x.mean())
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    # precompute block index map
    for i in range(n_draws):
        signs = rng.choice((-1.0, 1.0), size=n_blocks)
        s = np.repeat(signs, block)[:n]
        means[i] = (x * s).mean()
    p = float((means >= obs).mean())
    return {"obs_mean": obs, "placebo_mean": float(means.mean()), "p_value": p}


def summarize_carry(pnl: pd.Series | np.ndarray, hac_lags: int = 21) -> dict:
    """Headline stats for the carry stream: n, mean, annualised Sharpe, win-rate,
    naive t, HAC t, and the block-sign-flip placebo p-value."""
    x = np.asarray(pnl, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {"n": int(len(x)), "mean": float("nan"), "sharpe": float("nan"),
                "win": float("nan"), "t": float("nan"), "t_hac": float("nan"),
                "p_placebo": float("nan")}
    mu, sd = x.mean(), x.std(ddof=1)
    sharpe = float(mu / sd * np.sqrt(ANNUAL)) if sd > 0 else float("nan")
    pl = placebo_pvalue(x)
    return {
        "n": int(len(x)),
        "mean": float(mu),
        "sharpe": sharpe,
        "win": float((x > 0).mean()),
        "t": welch_t_vs_zero(x),
        "t_hac": hac_t_vs_zero(x, lags=hac_lags),
        "p_placebo": pl["p_value"],
    }


def gap_sign_stats(frame: pd.DataFrame) -> dict:
    """How often the realized dispersion gap is positive (the identity), and its mean.

    A near-100% positive share is *expected* — it is subadditivity, not an edge. We report
    it precisely so the front-card can say 'real but mechanical'."""
    g = frame["gap"].dropna().values
    return {
        "n": int(len(g)),
        "mean_gap": float(g.mean()),
        "share_positive": float((g > 0).mean()),
        "mean_index_vol": float(frame["index_vol"].dropna().mean()),
        "mean_avg_vol": float(frame["avg_vol"].dropna().mean()),
        "mean_impl_corr": float(frame["impl_corr"].dropna().mean()),
    }


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def net_of_costs(pnl: pd.Series, vega_cost_var: float = 5e-4,
                 turnover_per_year: float = 12.0) -> dict:
    """Charge bid/ask + vega costs against the carry, scaled by rebalancing turnover.

    Dispersion is run as a *book* of variance swaps re-struck periodically; each re-strike
    pays a vega/bid-ask cost. We model an annual cost = ``vega_cost_var * turnover_per_year``
    in the same annualised variance-point units as the P&L, spread evenly across the year,
    and subtract it from the daily carry. Returns gross/net mean, annualised mean, and the
    net annualised Sharpe — the honest 'after the spread' number.
    """
    x = pnl.dropna().values
    if len(x) < 2:
        return {"n": 0, "gross_mean": float("nan"), "net_mean": float("nan"),
                "gross_ann": float("nan"), "net_ann": float("nan"),
                "net_sharpe": float("nan")}
    daily_cost = vega_cost_var * turnover_per_year / ANNUAL
    net = x - daily_cost
    sd = net.std(ddof=1)
    return {
        "n": int(len(x)),
        "gross_mean": float(x.mean()),
        "net_mean": float(net.mean()),
        "gross_ann": float(x.mean() * ANNUAL),
        "net_ann": float(net.mean() * ANNUAL),
        "net_sharpe": float(net.mean() / sd * np.sqrt(ANNUAL)) if sd > 0 else float("nan"),
    }
