"""The Mispricing-Score strategy -- Stambaugh, Yu & Yuan (2015).

Construction (a faithful, price-only replica of the SYY composite):

  1. Each month, for every stock compute several well-known anomaly signals,
     each oriented so that a HIGHER value = MORE OVERPRICED (worse expected
     return):
       - momentum_12_1   : trailing 12-1 month return (high = glamour winner)
       - max_lottery     : mean of the 5 best daily returns last month (high = lottery)
       - volatility      : trailing 6-month realised vol (high = speculative)
       - asset_growth_px : trailing 12-month price run-up proxy (high = over-extended)
       - dist_52w_high   : closeness to the 52-week high (high = euphoric)
       - st_reversal     : last-month return (high recent = mean-reverts down)
     (These are the price-computable members of the SYY 11-anomaly set; the
     fundamental ones -- accruals, net issuance, profitability, distress --
     need clean per-name financials that yfinance does not deliver reliably,
     so we document that limitation rather than fake it.)

  2. Cross-sectionally RANK each signal to a percentile in [0, 1].
  3. The COMPOSITE mispricing score = the average of the available percentile
     ranks (SYY's "combination" measure). High score = overpriced.
  4. Sort into quintiles. LONG the cheapest quintile (low score), SHORT the
     most overpriced quintile (high score). Equal-weight, dollar-neutral.
  5. Hold one month, enter the close ONE day after the signal date (one
     documented execution lag, no same-bar fill, no look-ahead).

We also break out the long-only and short-only legs, because SYY's headline
claim is that the edge is concentrated in the SHORT (overpriced) leg.

Costs: one-way bps x turnover x NAV on each leg, plus an annual borrow charge
on the short leg. Gross vs net are labelled.

The panel is **survivorship-biased**; results are upper-bound estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Rebalance frequency (month start).
REBAL_FREQ = "MS"

# Quintile fraction for the long/short legs.
QUANTILE = 0.20

# Minimum names to form a sort.
MIN_STOCKS = 20

# Cost model.
ONE_WAY_BPS = 10.0          # 10 bps per side, per leg
BORROW_ANNUAL_BPS = 75.0    # 75 bps/yr borrow on the short leg


# ---------------------------------------------------------------------------
# Anomaly signals (each oriented HIGH = more overpriced)
# ---------------------------------------------------------------------------
def _anomaly_signals(
    prices: pd.DataFrame,
    asof: pd.Timestamp,
    rets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute the six price-only anomaly signals as of ``asof`` (inclusive).

    Returns a (ticker x signal) DataFrame. Every signal uses only data on or
    before ``asof`` -- no look-ahead. NaNs for names without enough history.
    ``rets`` (precomputed pct_change of ``prices``) may be passed to avoid
    recomputation in a backtest loop.
    """
    mask = prices.index <= asof
    px = prices.loc[mask]
    if len(px) < 252:
        return pd.DataFrame()
    if rets is None:
        r = px.pct_change()
    else:
        r = rets.loc[mask]

    # 12-1 momentum: return from 12m ago to 1m ago (skip the most recent month).
    p_now = px.iloc[-1]
    p_1m = px.iloc[-21]
    p_12m = px.iloc[-252]
    mom_12_1 = (p_1m / p_12m) - 1.0

    # MAX lottery: mean of the 5 largest daily returns over the last month.
    last_m = r.iloc[-21:].to_numpy()
    # top-5 per column without full sort
    part = np.partition(np.nan_to_num(last_m, nan=-np.inf), -5, axis=0)[-5:]
    max_lottery = pd.Series(part.mean(axis=0), index=px.columns)

    # Volatility: trailing 6-month (126d) realised daily vol, annualised.
    vol_6m = r.iloc[-126:].std() * np.sqrt(252)

    # Asset-growth proxy: 12-month total price run-up (over-extension).
    asset_growth_px = (p_now / p_12m) - 1.0

    # Distance to 52-week high: 1 - (high - now)/high  -> high = near the top.
    high_52w = px.iloc[-252:].max()
    dist_52w_high = 1.0 - (high_52w - p_now) / high_52w.replace(0, np.nan)

    # Short-term reversal: last-month return (recent winners mean-revert).
    st_reversal = (p_now / p_1m) - 1.0

    sig = pd.DataFrame(
        {
            "momentum_12_1": mom_12_1,
            "max_lottery": max_lottery,
            "volatility": vol_6m,
            "asset_growth_px": asset_growth_px,
            "dist_52w_high": dist_52w_high,
            "st_reversal": st_reversal,
        }
    )
    return sig


def mispricing_score(signals: pd.DataFrame) -> pd.Series:
    """Average the cross-sectional percentile ranks into the composite score.

    Each column is ranked to a percentile in [0, 1] (higher = more overpriced),
    then averaged across the available columns (SYY's combination measure).
    Returns a Series of composite scores in [0, 1]; high = overpriced.
    """
    if signals.empty:
        return pd.Series(dtype=float)
    ranks = signals.rank(pct=True)         # column-wise cross-sectional percentile
    score = ranks.mean(axis=1, skipna=True)
    return score.dropna()


# ---------------------------------------------------------------------------
# Long-short backtest
# ---------------------------------------------------------------------------
def _compound(ser: pd.Series) -> float:
    return float((1.0 + ser.dropna()).prod() - 1.0)


def long_short(
    prices: pd.DataFrame,
    rebal_freq: str = REBAL_FREQ,
    quantile: float = QUANTILE,
    min_stocks: int = MIN_STOCKS,
    score_fn=mispricing_score,
    shuffle_seed: int | None = None,
) -> pd.DataFrame:
    """Monthly composite-mispricing long-short backtest.

    For each rebalance date *t*:
      - build the six anomaly signals on data <= t,
      - average their ranks into the composite score,
      - LONG the bottom-``quantile`` (cheap) names, SHORT the top-``quantile``
        (overpriced) names, equal-weight,
      - enter the close ONE day after *t* and hold until the next rebalance.

    ``shuffle_seed`` (if set) randomly permutes the composite score across names
    each month -- the placebo / label-shuffle null (the sort is then random).

    Returns a DataFrame indexed by rebalance date with columns:
      long_ret, short_ret, ls_ret (long - short), turnover, n.
    """
    rets = prices.pct_change()
    rebal_dates = pd.date_range(prices.index[252], prices.index[-1], freq=rebal_freq)
    rebal_dates = [d for d in rebal_dates if d <= prices.index[-1]]

    rows: list[dict] = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None

    for i, t in enumerate(rebal_dates[:-1]):
        next_t = rebal_dates[i + 1]
        sig = _anomaly_signals(prices, t, rets=rets)
        if sig.empty:
            continue
        score = score_fn(sig)
        score = score.dropna()
        if len(score) < min_stocks:
            continue

        if rng is not None:
            shuffled = rng.permutation(score.values)
            score = pd.Series(shuffled, index=score.index)

        k = max(1, int(round(len(score) * quantile)))
        ordered = score.sort_values()
        long_names = list(ordered.index[:k])         # lowest score = cheapest
        short_names = list(ordered.index[-k:])        # highest score = overpriced

        # Execution lag: enter the close one day AFTER the signal date t.
        future = rets.loc[(rets.index > t) & (rets.index <= next_t)]
        if future.empty:
            continue
        # Drop the first available day's open exposure (one-day lag): hold from
        # the day after t. ``future`` already starts strictly after t; we apply
        # the lag by skipping the first row's contribution.
        if len(future) > 1:
            future = future.iloc[1:]
        if future.empty:
            continue

        long_ret = float(future[long_names].apply(_compound).mean())
        short_ret = float(future[short_names].apply(_compound).mean())
        ls_ret = long_ret - short_ret

        # Turnover: fraction of names changed vs last month, both legs.
        cur_long, cur_short = set(long_names), set(short_names)
        if prev_long or prev_short:
            turn_long = 1.0 - len(cur_long & prev_long) / max(len(cur_long), 1)
            turn_short = 1.0 - len(cur_short & prev_short) / max(len(cur_short), 1)
            turnover = 0.5 * (turn_long + turn_short)
        else:
            turnover = 1.0
        prev_long, prev_short = cur_long, cur_short

        rows.append(
            {
                "date": t,
                "long_ret": long_ret,
                "short_ret": short_ret,
                "ls_ret": ls_ret,
                "turnover": turnover,
                "n": int(len(score)),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------
def apply_costs(
    result: pd.DataFrame,
    one_way_bps: float = ONE_WAY_BPS,
    borrow_annual_bps: float = BORROW_ANNUAL_BPS,
) -> pd.Series:
    """Net the long-short return for trading costs + short borrow.

    Trading cost per month = one_way_bps/1e4 * turnover * (both legs = 2 sides).
    Borrow cost per month = borrow_annual_bps/1e4 / 12 on the short leg.
    Returns a net monthly ls return Series.
    """
    gross = result["ls_ret"]
    trade_cost = (one_way_bps / 1e4) * result["turnover"] * 2.0
    borrow_cost = (borrow_annual_bps / 1e4) / 12.0
    return gross - trade_cost - borrow_cost


# ---------------------------------------------------------------------------
# Summary statistics (annualised, HAC t)
# ---------------------------------------------------------------------------
def summary(monthly_returns: pd.Series, periods_per_year: int = 12) -> dict:
    """Annualised mean/vol/Sharpe, Newey-West HAC t, hit-rate, max DD, n."""
    r = pd.Series(monthly_returns).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n")}

    mean_m = float(r.mean())
    vol_m = float(r.std(ddof=1))
    mean_ann = mean_m * periods_per_year
    vol_ann = vol_m * np.sqrt(periods_per_year)
    sr = mean_ann / vol_ann if vol_ann > 0 else np.nan

    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())

    arr = r.to_numpy()
    mu = arr.mean()
    e = arr - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se) if se > 0 else np.nan

    return {
        "mean": mean_ann,
        "vol": vol_ann,
        "sharpe": sr,
        "tstat": tstat,
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": dd,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_null(
    prices: pd.DataFrame,
    n_shuffles: int = 200,
    seed: int = 535,
    **kwargs,
) -> dict:
    """Shuffle the composite score across names each month; record the
    distribution of long-short mean returns under the null.

    Returns ``{p_value, null_mean, null_std, observed_mean}`` where p_value is
    the fraction of shuffled means that match-or-exceed the observed |mean|.
    """
    obs = long_short(prices, **kwargs)
    if obs.empty:
        return {"p_value": np.nan, "null_mean": np.nan, "null_std": np.nan, "observed_mean": np.nan}
    observed_mean = float(obs["ls_ret"].mean()) * 12.0

    rng = np.random.default_rng(seed)
    null_means: list[float] = []
    for _ in range(n_shuffles):
        s = int(rng.integers(0, 2**31 - 1))
        sh = long_short(prices, shuffle_seed=s, **kwargs)
        if sh.empty:
            continue
        null_means.append(float(sh["ls_ret"].mean()) * 12.0)

    null = np.array(null_means)
    if null.size == 0:
        return {"p_value": np.nan, "null_mean": np.nan, "null_std": np.nan, "observed_mean": observed_mean}
    p_value = float(np.mean(np.abs(null) >= abs(observed_mean)))
    return {
        "p_value": p_value,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        "observed_mean": observed_mean,
    }


# ---------------------------------------------------------------------------
# Synthetic positive control
# ---------------------------------------------------------------------------
def synthetic_control(premiums=(-0.20, -0.10, 0.0, 0.10, 0.20, 0.30), seed: int = 0) -> pd.DataFrame:
    """Sweep a planted mispricing premium through the engine; the composite
    long-short mean should be monotone in the premium (and ~0 at the null).

    Returns a DataFrame: premium, ls_mean_pct, ls_t, n.
    """
    from . import data as _data

    rows = []
    for prem in premiums:
        prices, _spy, _truth = _data.synthetic_panel(
            n_stocks=45, n_days=2200, mispricing_premium=prem, seed=seed
        )
        res = long_short(prices, min_stocks=10, quantile=0.20)
        if res.empty:
            rows.append({"premium": prem, "ls_mean_pct": np.nan, "ls_t": np.nan, "n": 0})
            continue
        s = summary(res["ls_ret"])
        rows.append(
            {"premium": prem, "ls_mean_pct": s["mean"] * 100.0, "ls_t": s["tstat"], "n": s["n"]}
        )
    return pd.DataFrame(rows)
