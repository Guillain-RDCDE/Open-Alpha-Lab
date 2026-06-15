"""The strategy and its honest controls — Study 202 (Fifty-Two-Week-Low).

The folk recipe:
    Every week (or month), rank all S&P 500 stocks by their *proximity to the
    52-week low*.  Buy the quintile closest to their lows (the "beaten down"
    contrarian bet) and hold for the next period.  Sell short the quintile
    nearest their highs (the "momentum" names).  Collect the spread as the
    losers bounce back.

The 52-week-low proximity signal (normalised position in the 52-week range):
    proximity(t) = (close(t) - low_52w(t)) / (high_52w(t) - low_52w(t) + eps)

- proximity ≈ 0  → stock near its 52-week low  (the contrarian long)
- proximity ≈ 1  → stock near its 52-week high (the momentum short / sold)

Three tests share one forward-return engine:

1. **Quintile sort** — rank all stocks by proximity, form equal-weight quintile
   portfolios, measure forward 1/4/13-week returns. The contrarian thesis predicts
   Q1 (near-low) > Q5 (near-high).

2. **Long-only bottom decile** — buy only the 10% most beaten-down names, hold
   for N weeks, vs an equal-weight buy-and-hold of the full basket (the fair
   comparison for a long-only investor).

3. **Random-entry control** — same basket, random-date entries, equal-weight,
   to strip out the equity risk premium. This is the fair coin: if proximity
   adds nothing, the near-low return equals the random return.

No look-ahead: proximity is computed on closes up to *t*; the trade enters at
the *next* bar's open (approximated as the next-day close for simplicity, since
we are testing a weekly rebalancing rule where open/close distinctions matter far
less than selection quality).

Survivorship bias is explicit: the basket is fixed to tickers still trading today;
this inflates long-run returns and is named on the Signal axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Proximity signal
# ---------------------------------------------------------------------------
def proximity_to_52w_low(
    panel: pd.DataFrame,
    window: int = 252,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Proximity of each stock's close to its trailing 52-week low.

    proximity(t) = (close(t) - rolling_low_252(t))
                   / (rolling_high_252(t) - rolling_low_252(t) + eps)

    Returns a DataFrame of the same shape as ``panel``, in [0, 1].  Values near
    0 indicate the stock is near its 52-week low; values near 1 indicate it is
    near its 52-week high.  The first ``window - 1`` rows are NaN (warm-up).

    The ``eps`` prevents division by zero on a stock that has not moved at all
    over the look-back window (e.g. very early data, halted stocks).
    """
    lo = panel.rolling(window, min_periods=window).min()
    hi = panel.rolling(window, min_periods=window).max()
    prox = (panel - lo) / (hi - lo + eps)
    return prox.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Forward-return calculation
# ---------------------------------------------------------------------------
def forward_returns(
    panel: pd.DataFrame,
    hold_days: int = 5,
) -> pd.DataFrame:
    """Forward log-return over ``hold_days`` trading days for each stock.

    fwd_ret(t, ticker) = log(close(t + hold_days) / close(t))

    Computed on closes; the last ``hold_days`` rows are NaN. No look-ahead:
    the return starting from bar t is available only at t + hold_days.
    """
    return panel.apply(lambda s: np.log(s.shift(-hold_days) / s))


# ---------------------------------------------------------------------------
# Quintile portfolio backtest
# ---------------------------------------------------------------------------
def quintile_backtest(
    prox: pd.DataFrame,
    fwd: pd.DataFrame,
    n_quintiles: int = 5,
    rebalance_freq: int = 5,
) -> dict[str, pd.DataFrame]:
    """Cross-sectional quintile sort on proximity, measured every ``rebalance_freq`` days.

    On each rebalancing date, stocks are ranked by their proximity signal (ascending,
    so rank 1 = nearest to 52-week low) and assigned to quintiles 1 through 5.  The
    quintile return on that date is the equal-weight mean of forward returns for
    constituent stocks.

    Returns a dict with keys 'Q1' through 'Q{n_quintiles}' and 'Q1_minus_Q5',
    each a Series of per-rebalancing-date returns (using the signal date as index).
    """
    # Align and drop warm-up NaNs
    dates = prox.index[prox.notna().all(axis=1) & fwd.notna().all(axis=1)]
    dates = dates[::rebalance_freq]   # rebalance every N trading days

    q_returns: dict[str, list] = {f"Q{q}": [] for q in range(1, n_quintiles + 1)}
    q_returns["Q1_minus_Q5"] = []
    index_out = []

    for t in dates:
        px = prox.loc[t]
        fr = fwd.loc[t]
        if px.isna().any() or fr.isna().any():
            continue
        # Rank ascending: rank 1 = smallest proximity = nearest 52-week low
        ranks = px.rank(method="first")
        n = len(ranks)
        q_size = n / n_quintiles
        for q in range(1, n_quintiles + 1):
            lo_rank = (q - 1) * q_size
            hi_rank = q * q_size
            mask = (ranks > lo_rank) & (ranks <= hi_rank)
            if mask.sum() == 0:
                q_returns[f"Q{q}"].append(float("nan"))
            else:
                q_returns[f"Q{q}"].append(float(fr[mask].mean()))
        index_out.append(t)
        q_returns["Q1_minus_Q5"].append(
            q_returns["Q1"][-1] - q_returns["Q5"][-1]
        )

    out = {}
    for k, vals in q_returns.items():
        out[k] = pd.Series(vals, index=pd.DatetimeIndex(index_out, name="date"), name=k)
    return out


# ---------------------------------------------------------------------------
# Long-only bottom-decile vs equal-weight baseline
# ---------------------------------------------------------------------------
def long_bottom_decile(
    prox: pd.DataFrame,
    fwd: pd.DataFrame,
    decile_frac: float = 0.10,
    rebalance_freq: int = 5,
) -> tuple[pd.Series, pd.Series]:
    """Long only the bottom-decile (nearest 52-week low), vs equal-weight baseline.

    Returns (strat_returns, baseline_returns) as aligned Series of per-period
    log-returns at each rebalancing date.

    The equal-weight baseline is the mean forward return of *all* stocks in the
    basket — the benchmark a long-only investor should beat.
    """
    dates = prox.index[prox.notna().all(axis=1) & fwd.notna().all(axis=1)]
    dates = dates[::rebalance_freq]

    strat, baseline = [], []
    index_out = []

    for t in dates:
        px = prox.loc[t]
        fr = fwd.loc[t]
        if px.isna().any() or fr.isna().any():
            continue
        cutoff = px.quantile(decile_frac)
        mask = px <= cutoff
        if mask.sum() == 0:
            continue
        strat.append(float(fr[mask].mean()))
        baseline.append(float(fr.mean()))
        index_out.append(t)

    idx = pd.DatetimeIndex(index_out, name="date")
    return pd.Series(strat, index=idx, name="bottom_decile"), pd.Series(baseline, index=idx, name="baseline")


# ---------------------------------------------------------------------------
# Random-entry control
# ---------------------------------------------------------------------------
def random_directions(n: int, seed: int = 0) -> np.ndarray:
    """A reproducible vector of ±1 — the control arm's coin."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=n)


def random_entry_control(
    panel: pd.DataFrame,
    fwd: pd.DataFrame,
    n_picks: int | None = None,
    rebalance_freq: int = 5,
    seed: int = 202,
) -> pd.Series:
    """Equal-weight mean forward return of a random subset of stocks, as the placebo.

    On each rebalancing date, randomly select ``n_picks`` stocks (or all, if None)
    and return the equal-weight mean forward return.  This is the benchmark that
    strips out the equity premium: if proximity adds nothing, the near-low return
    is not distinguishable from this random draw.
    """
    rng = np.random.default_rng(seed)
    n_cols = panel.shape[1]
    n_picks = n_picks or max(1, n_cols // 5)  # default: pick same size as Q1

    dates = panel.index[
        panel.notna().all(axis=1) & fwd.notna().all(axis=1)
    ]
    dates = dates[::rebalance_freq]

    rets = []
    index_out = []
    tickers = list(panel.columns)
    for t in dates:
        fr = fwd.loc[t]
        if fr.isna().any():
            continue
        chosen = rng.choice(tickers, size=n_picks, replace=False)
        rets.append(float(fr[chosen].mean()))
        index_out.append(t)

    return pd.Series(rets, index=pd.DatetimeIndex(index_out, name="date"), name="random")


# ---------------------------------------------------------------------------
# Summarise a return series with HAC inference
# ---------------------------------------------------------------------------
def summarize(ret: pd.Series, col: str = "value") -> dict:
    """Headline statistics for a return series.

    Returns n, win-rate, mean return (bps/period), annualised return (%), and a
    HAC Newey-West t-stat on the mean — the inference-bar number that decides
    whether the edge is distinguishable from zero.

    ``ret`` is a Series of raw (log) returns per period.
    """
    r = np.asarray(ret.dropna(), dtype=float)
    n = r.size
    if n == 0:
        return {k: float("nan") for k in
                ["n", "win_rate", "mean_bps", "ann_pct", "tstat"]}

    ann_factor = 252.0 / max(ret.index.to_series().diff().median().days, 1)
    out = {
        "n": int(n),
        "win_rate": float((r > 0).mean()),
        "mean_bps": float(r.mean() * 1e4),
        "ann_pct": float(r.mean() * ann_factor * 100.0),
        "tstat": float("nan"),
    }
    if n > 5:
        mu = r.mean()
        e = r - mu
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        lrv = float(e @ e) / n
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        out["tstat"] = float(mu / se) if se > 0 else float("nan")
    return out
