"""The cross-sectional reversal/momentum engine and its honest controls — Study 251.

The QuantNest-7 claim we test: rank crypto tokens each day by their trailing 21-day
return (one-day skip); **past losers outperform past winners** — a short-term *reversal*,
not the momentum effect of Liu et al. (2022). The paper reports this as factor REV with a
Newey-West *t* of 6.19 and a +151%/yr long-short quintile spread.

We rebuild it on a public daily panel and pit it against momentum, a shuffle control, and
realistic crypto trading costs. Four arms:

1. **reversal** — each day, long the bottom quintile (worst trailing 21-day return),
   short the top quintile (best). Equal-weight within each leg, reformed daily.
2. **momentum** — the mirror: long winners, short losers. If the paper is right, this
   loses where reversal wins.
3. **shuffle** — the null control: the *same* quintile machinery, but the trailing-return
   signal is randomly permuted across tokens each day, destroying any real information
   while preserving turnover and leg structure. If reversal beats shuffle, the *ranking*
   carries cross-sectional information.
4. costs — the long-short book reforms **every day** on a 21-day signal, so turnover is
   near-total daily. At crypto bid-ask spreads (the paper notes >1% per trade for smaller
   tokens) the gross spread is the headline; the net spread is the reality.

No look-ahead: the signal on day *t* uses returns up to *t-1* (with a further one-day
skip); the long-short return is earned over *t-1 -> t*. One execution lag, documented.

Fama-MacBeth: ``fama_macbeth`` runs the paper's own apparatus — a daily cross-sectional
regression of next-day return on the lagged, cross-sectionally standardized signal — and
reports the time-series average slope with a Newey-West *t*, the direct mirror of the
paper's Table 8.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 365   # crypto: no weekends/holidays


# ---------------------------------------------------------------------------
# Returns and the cross-sectional signal
# ---------------------------------------------------------------------------
def daily_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns of each token (close-to-close), aligned to ``panel.index``."""
    return panel.pct_change()


def trailing_signal(panel: pd.DataFrame, lookback: int = 21, skip: int = 1) -> pd.DataFrame:
    """Trailing ``lookback``-day return per token, ending ``skip`` days before *t*.

    The signal at row *t* uses prices up to *t-skip*, i.e. it is already lagged for use in
    forming the day-*t* portfolio that earns the *t-1 -> t* return. A high value = a recent
    winner. Reversal sorts ascending (long the low values); momentum sorts descending.
    """
    shifted = panel.shift(skip)
    return shifted / shifted.shift(lookback) - 1.0


# ---------------------------------------------------------------------------
# Long-short quintile construction
# ---------------------------------------------------------------------------
def _day_weights(sig_row: np.ndarray, leg: str, n_groups: int, m: int) -> np.ndarray | None:
    """Signed equal-weight vector for one day, or ``None`` if too few valid names.

    Longs the bottom ``1/n_groups`` of the ranking and shorts the top (``leg='reversal'``),
    or the reverse (``leg='momentum'``). NaN-signal tokens are dropped from the cross-section.
    """
    valid = np.where(np.isfinite(sig_row))[0]
    n = len(valid)
    if n < n_groups:
        return None
    k = max(1, n // n_groups)
    order = valid[np.argsort(sig_row[valid], kind="stable")]
    low, high = order[:k], order[-k:]
    long_idx, short_idx = (low, high) if leg == "reversal" else (high, low)
    w = np.zeros(m)
    w[long_idx] = 1.0 / len(long_idx)
    w[short_idx] = -1.0 / len(short_idx)
    return w


def long_short_returns(
    panel: pd.DataFrame,
    signal: pd.DataFrame,
    leg: str = "reversal",
    n_groups: int = 5,
    cost_bps: float = 0.0,
) -> pd.Series:
    """Daily long-short return of the quintile spread, reformed every day.

    Parameters
    ----------
    leg:
        ``'reversal'`` longs the *low*-signal quintile (recent losers) and shorts the *high*
        quintile (recent winners). ``'momentum'`` does the reverse. The two are exact
        negatives of each other before costs.
    cost_bps:
        One-way transaction cost in bps, charged on the *traded* fraction of each leg as the
        membership turns over from one day to the next. The legs are equal-weight; turnover
        is measured as the L1 change in per-name weights across the whole book (long + short)
        divided by 2 (one-way). With a 21-day signal reformed daily, turnover runs near 100%
        per day — the engine's whole point.

    Returns a daily net long-short return series (long leg minus short leg, minus costs),
    indexed by ``panel.index`` over the period where the signal is defined.
    """
    if leg not in ("reversal", "momentum"):
        raise ValueError("leg must be 'reversal' or 'momentum'")
    idx = panel.index
    m = panel.shape[1]
    rets_np = daily_returns(panel).to_numpy(dtype=float)
    sig_np = signal.to_numpy(dtype=float)
    cost = cost_bps * 1e-4

    out = np.full(len(idx), np.nan)
    prev_w = np.zeros(m)
    for t in range(len(idx)):
        w = _day_weights(sig_np[t], leg, n_groups, m)
        if w is None:
            prev_w = np.zeros(m)
            continue
        r_t = rets_np[t]
        gross = float(np.nansum(w * np.where(np.isfinite(r_t), r_t, 0.0)))
        turnover = float(np.abs(w - prev_w).sum()) / 2.0   # one-way fraction traded
        out[t] = gross - turnover * cost
        prev_w = w

    return pd.Series(out, index=idx, name=leg).dropna()


def shuffle_signal(signal: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Permute each day's cross-section of signal values across tokens (the null control).

    Preserves the per-day set of signal values (so quintile cut-points and turnover behave
    normally) but severs the link between a token and its own trailing return. A long-short
    built on this should earn ~0 in expectation.
    """
    rng = np.random.default_rng(seed)
    out = signal.copy()
    vals = np.array(out.to_numpy(dtype=float), copy=True)
    for t in range(vals.shape[0]):
        row = vals[t]
        mask = ~np.isnan(row)
        if mask.sum() > 1:
            perm = rng.permutation(row[mask])
            row[mask] = perm
            vals[t] = row
    return pd.DataFrame(vals, index=signal.index, columns=signal.columns)


# ---------------------------------------------------------------------------
# Fama-MacBeth — the paper's own apparatus
# ---------------------------------------------------------------------------
def fama_macbeth(
    panel: pd.DataFrame,
    signal: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """Daily cross-sectional regression of return on the lagged standardized signal.

    Each day we z-score the cross-section of signal values, regress that day's token
    returns on the z-scores (slope = the day's factor premium), and collect the slope
    series. We report its time-series mean and a Newey-West *t* — the direct analogue of
    the paper's Table 8 REV row. A **positive** average slope means high-signal (recent
    winner) tokens earn more (momentum); a **negative** slope means recent losers earn
    more (reversal). The paper reports REV as a *negated* winner signal, so its positive
    REV premium corresponds to a negative slope here.
    """
    rets_np = daily_returns(panel).to_numpy(dtype=float)
    sig_np = signal.to_numpy(dtype=float)
    slopes = []
    for t in range(len(panel.index)):
        x = sig_np[t]
        y = rets_np[t]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 5:
            continue
        xv, yv = x[ok], y[ok]
        sd = xv.std()
        if sd == 0:
            continue
        z = (xv - xv.mean()) / sd
        # OLS slope of y on z (z has unit variance): cov(z,y)/var(z).
        slope = float(np.cov(z, yv, ddof=0)[0, 1] / np.var(z))
        slopes.append(slope)

    s = np.asarray(slopes, dtype=float)
    n = len(s)
    mu = float(s.mean()) if n else float("nan")
    t_nw = _hac_tstat(s) if n else float("nan")
    return {
        "n_days": n,
        "avg_slope_daily": mu,
        "avg_slope_bps": mu * 1e4,
        "tstat_nw": t_nw,
        "reversal_premium_ann": -mu * periods_per_year,   # reversal = negated winner slope
    }


# ---------------------------------------------------------------------------
# Performance summary
# ---------------------------------------------------------------------------
def _hac_tstat(x: np.ndarray) -> float:
    """Newey-West HAC t-stat for the mean of a 1-D series, automatic lag selection."""
    e = np.asarray(x, dtype=float)
    e = e[np.isfinite(e)]
    n = len(e)
    if n < 3:
        return float("nan")
    mu = e.mean()
    d = e - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(d @ d) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(d[k:] @ d[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """Headline annualised statistics for a daily (long-short) return series.

    Sharpe, annualised vol, annualised return (arithmetic mean × periods), CAGR (geometric),
    maximum drawdown of the cumulative long-short P&L, and the HAC *t*-stat for the mean
    daily return (the inference-bar number). 365 days/year (crypto never closes).
    """
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    mu = r.mean()
    std = r.std(ddof=1)
    sharpe = float(mu / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    vol_ann = float(std * np.sqrt(periods_per_year))
    ann_ret = float(mu * periods_per_year)

    cum = (1.0 + r).cumprod()
    years = n / periods_per_year
    cagr = float(cum.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and cum.iloc[-1] > 0 else float("nan")
    dd = float((cum / cum.cummax() - 1.0).min())

    return {
        "n_days": int(n),
        "ann_return": ann_ret,
        "cagr": cagr,
        "sharpe": sharpe,
        "vol_ann": vol_ann,
        "max_drawdown": dd,
        "mean_daily_bps": float(mu * 1e4),
        "tstat": _hac_tstat(r.to_numpy()),
    }


def avg_turnover(signal: pd.DataFrame, leg: str = "reversal", n_groups: int = 5) -> float:
    """Average one-way daily turnover of the long-short book (fraction of NAV traded/day)."""
    m = signal.shape[1]
    sig_np = signal.to_numpy(dtype=float)
    prev_w = np.zeros(m)
    tos = []
    for t in range(len(signal.index)):
        w = _day_weights(sig_np[t], leg, n_groups, m)
        if w is None:
            prev_w = np.zeros(m)
            continue
        tos.append(float(np.abs(w - prev_w).sum()) / 2.0)
        prev_w = w
    return float(np.mean(tos)) if tos else float("nan")


def compare_arms(
    panel: pd.DataFrame,
    lookback: int = 21,
    skip: int = 1,
    n_groups: int = 5,
    cost_bps: float = 0.0,
    shuffle_seed: int = 42,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """Run all arms (reversal, momentum, shuffle control) and summarise each.

    Returns a dict with keys ``'reversal'``, ``'momentum'``, ``'shuffle'`` (each a
    ``summary()`` dict), plus ``'fama_macbeth'`` (the cross-sectional regression result),
    ``'turnover'`` (one-way daily, reversal leg) and the run parameters.
    """
    sig = trailing_signal(panel, lookback=lookback, skip=skip)
    rev = long_short_returns(panel, sig, leg="reversal", n_groups=n_groups, cost_bps=cost_bps)
    mom = long_short_returns(panel, sig, leg="momentum", n_groups=n_groups, cost_bps=cost_bps)
    shuf = long_short_returns(
        panel, shuffle_signal(sig, seed=shuffle_seed),
        leg="reversal", n_groups=n_groups, cost_bps=cost_bps,
    )
    return {
        "reversal": summary(rev, periods_per_year=periods_per_year),
        "momentum": summary(mom, periods_per_year=periods_per_year),
        "shuffle": summary(shuf, periods_per_year=periods_per_year),
        "fama_macbeth": fama_macbeth(panel, sig, periods_per_year=periods_per_year),
        "turnover": avg_turnover(sig, leg="reversal", n_groups=n_groups),
        "lookback": lookback, "skip": skip, "n_groups": n_groups, "cost_bps": cost_bps,
    }
