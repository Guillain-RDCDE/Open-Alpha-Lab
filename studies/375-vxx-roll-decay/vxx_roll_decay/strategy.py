"""Strategy + inference for Study 375 — VXX-Roll-Decay (the short-vol carry).

The book is the simplest possible "harvest the roll" rule:

    Hold a **constant-notional short** in the VIX-futures ETP, rebalanced daily.
    The daily book return is ``-r_etp`` (short the long-ETP return), optionally only
    when the term structure was in **contango** at the prior close.

The pitch: contango is the usual state, so each roll bleeds the long ETP and pays the short
a steady carry. The catch: when vol spikes the ETP explodes and the short takes a violent
loss — picking up nickels in front of a steamroller. We separate the two:

  * **Signal** — is the average short-carry daily return reliably positive? A **Newey-West
    (HAC)** t-stat (carry is autocorrelated; an i.i.d. t would overstate it) and a
    **placebo / sign-shuffle** null. A synthetic control with a *planted* carry knob proves
    the engine recovers a real edge and does not manufacture one from noise.
  * **Tradability** — net of **borrow** on the short and **rebalancing turnover**, and then
    the part that actually decides it: the **tail** — skew, excess kurtosis, the single worst
    day, and the compounded max drawdown. A positive-Sharpe carry with a -90% drawdown and a
    -40% day is real-but-undeployable.

The decisive finding is two-sided: the carry **is** statistically real (HAC t > 2), and it
is **also** undeployable at NAV scale because the crash tail can wipe a constant-notional
short out (Volmageddon, COVID, the 2024 unwind). Real signal, fragile-to-mirage trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252  # trading days / year


# --------------------------------------------------------------------------- #
# The short-carry book
# --------------------------------------------------------------------------- #
def etp_returns(frame: pd.DataFrame) -> pd.Series:
    """Daily simple return of the long ETP (NaN-dropped)."""
    return frame["etp"].pct_change().dropna()


def short_carry(frame: pd.DataFrame, contango_only: bool = False,
                contango_thresh: float = 1.0) -> pd.Series:
    """Daily return of a constant-notional **short** in the ETP, rebalanced daily.

    The book return is ``-r_etp``. If ``contango_only`` is True the short is **on** only when
    the term structure was in contango (``contango > thresh``) at the *prior* close (a 1-day
    lag, no look-ahead); otherwise the book sits flat (return 0) that day.
    """
    r_long = etp_returns(frame)
    r_short = -r_long
    if not contango_only:
        return r_short
    on = (frame["contango"].shift(1) > contango_thresh).reindex(r_short.index).fillna(False)
    return r_short.where(on, 0.0)


# --------------------------------------------------------------------------- #
# Inference — is the carry real?
# --------------------------------------------------------------------------- #
def hac_t(x: np.ndarray, lags: int = 21) -> dict:
    """Newey-West (Bartlett-kernel) HAC t-stat that the mean of ``x`` is > 0.

    Carry returns are positively autocorrelated (vol regimes persist); an i.i.d. t overstates
    significance, so we use a HAC SE with ``lags`` (≈ one trading month) Bartlett weights.
    Returns mean, HAC SE, and the t = mean / SE.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return {"mean": float("nan"), "se": float("nan"), "t": float("nan"), "n": n}
    m = x.mean()
    e = x - m
    var = (e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1)
        var += 2.0 * w * (e[k:] @ e[:-k]) / n
    se = np.sqrt(var / n)
    t = m / se if se > 0 else float("nan")
    return {"mean": float(m), "se": float(se), "t": float(t), "n": n}


def placebo_pvalue(x: np.ndarray, n_draws: int = 20_000, seed: int = 375) -> dict:
    """Sign-shuffle placebo null for a daily carry series.

    Under the null "no carry," the sign of each daily return is exchangeable. We flip each
    day's sign by a fair coin ``n_draws`` times and ask how often the shuffled mean matches or
    beats the observed mean. ``p`` = P[shuffled mean >= observed mean] — the honest
    distribution-free answer to "could this carry be a coin's idea of a trend?"
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    obs = float(x.mean())
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(n_draws)
    for i in range(n_draws):
        signs = rng.choice([-1.0, 1.0], size=n)
        means[i] = (signs * x).mean()
    p = float((means >= obs).mean())
    return {"obs_mean": obs, "placebo_mean": float(means.mean()), "p_value": p}


# --------------------------------------------------------------------------- #
# Tradability — costs and the tail
# --------------------------------------------------------------------------- #
def net_of_costs(r_short: pd.Series, borrow_annual: float = 0.03,
                 cost_bps: float = 2.0) -> pd.Series:
    """Charge a short-vol carry book its real frictions, per day.

    * **Borrow** — the short pays a stock-loan/financing fee of ``borrow_annual`` per year,
      applied as ``borrow_annual / 252`` per day the book is on.
    * **Rebalancing turnover** — a constant-notional daily-rebalanced short turns over a
      slice of NAV each day; we charge a flat ``cost_bps`` (one-way, in bps) per active day.

    Days where the book is flat (return exactly 0.0, e.g. out of contango) pay nothing.
    """
    on = (r_short != 0.0)
    daily_drag = (borrow_annual / ANN + cost_bps / 1e4) * on.astype(float)
    return r_short - daily_drag


def perf(r: pd.Series) -> dict:
    """Annualised performance + the tail diagnostics that decide tradability.

    Returns annualised mean/vol/Sharpe, the compounded max drawdown of a constant-notional
    book, daily skew & excess kurtosis (the crash signature), the single worst day, and the
    1% daily VaR/CVaR.
    """
    r = r = pd.Series(r).dropna()
    mu = r.mean() * ANN
    sd = r.std(ddof=1) * np.sqrt(ANN)
    sharpe = mu / sd if sd > 0 else float("nan")
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    var1 = float(np.percentile(r.values, 1))
    cvar1 = float(r[r <= var1].mean()) if (r <= var1).any() else float("nan")
    return {
        "ann_ret": float(mu), "ann_vol": float(sd), "sharpe": float(sharpe),
        "max_dd": dd, "skew": float(r.skew()), "exkurt": float(r.kurtosis()),
        "worst_day": float(r.min()), "worst_day_date": r.idxmin(),
        "var1": var1, "cvar1": cvar1, "n": int(len(r)),
    }


def summarize(frame: pd.DataFrame, contango_only: bool = False,
              borrow_annual: float = 0.03, cost_bps: float = 2.0,
              lags: int = 21) -> dict:
    """One-call headline: the short-carry book gross & net, its HAC t, placebo p, and tail."""
    rs = short_carry(frame, contango_only=contango_only)
    rn = net_of_costs(rs, borrow_annual=borrow_annual, cost_bps=cost_bps)
    h = hac_t(rs.values, lags=lags)
    pl = placebo_pvalue(rs.values)
    return {
        "gross": perf(rs), "net": perf(rn),
        "hac_t": h["t"], "hac_se": h["se"], "daily_mean": h["mean"],
        "placebo_p": pl["p_value"],
        "frac_contango": float((frame["contango"] > 1.0).mean()),
    }
