"""The Firm-Age-Anomaly strategy -- the new-list effect (Fama-French 2004; Jiang-Lee-Zhang 2005).

Construction:
  1. Proxy firm age from each ticker's first available price date.
  2. Each month, compute age_t = (rebal_date - first_price_date) in years for every name
     that already has >= min_history days of price data.
  3. Rank by age; split into terciles. The long leg is the OLD tercile (mature firms);
     the short leg is the YOUNG tercile (recently-listed firms).
  4. Hold equal-weight for one calendar month, measuring each leg's compounded return.
  5. AGE return = old_leg_return - young_leg_return (old-minus-young).

The portfolio is a dollar-neutral long-short. Returns are measured as monthly returns.
Inference uses a Newey-West HAC one-sample t-stat on the monthly AGE series.

Robustness:
  - A label-shuffle placebo: permute the age labels across names, rebuild the long-short,
    and record where the real t-stat sits in the permutation null.
  - Costs: one-way bps x turnover x NAV each rebalance, plus a borrow charge on the short
    (young) leg.

The basket is **survivorship-biased** (names still trading in 2026). Young firms that
failed are absent, biasing the old-minus-young spread *downward* -- the measured premium
is conservative. We name this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Rebalancing frequency (month start)
REBAL_FREQ = "MS"

# Minimum price history (trading days) before a name is eligible -- ~6 months.
MIN_HISTORY = 126

# Tercile fraction for the long/short legs.
TERCILE = 1.0 / 3.0

# Default trading-cost assumptions.
ONE_WAY_BPS = 10.0          # 10 bps one-way transaction cost
BORROW_BPS_ANNUAL = 100.0   # 100 bps/yr stock-borrow on the short (young) leg


# ---------------------------------------------------------------------------
# Firm-age long-short portfolio
# ---------------------------------------------------------------------------
def age_portfolio(
    prices: pd.DataFrame,
    spy_prices: pd.Series,
    first_dates: pd.Series | None = None,
    rebal_freq: str = REBAL_FREQ,
    min_history: int = MIN_HISTORY,
    tercile: float = TERCILE,
    min_stocks: int = 9,
    lag_days: int = 1,
) -> pd.DataFrame:
    """Construct monthly old-minus-young (firm-age) long-short returns.

    For each rebalancing date *t*:
      1. age_i = (t - first_date_i) in years; eligible if the name has >= min_history days
         of price history as of *t*.
      2. Rank by age. Long = oldest tercile, Short = youngest tercile.
      3. Enter at the close ``lag_days`` after *t* (one-day execution lag; no same-bar fill,
         no look-ahead -- the age ranking uses only data available at *t*).
      4. Hold equal-weight to the next rebalance; AGE return = old_ret - young_ret.

    Returns a DataFrame indexed by rebalancing date with columns:
        ``old_ret``   -- equal-weight return of the old (long) leg
        ``young_ret`` -- equal-weight return of the young (short) leg
        ``age_ret``   -- old_ret - young_ret (the long-short)
        ``mkt_ret``   -- SPY return over the same period
        ``avg_age_old``   -- average firm age (yrs) of the long leg
        ``avg_age_young`` -- average firm age (yrs) of the short leg
        ``turnover``  -- two-sided name turnover vs the previous rebalance (0..2)
        ``n``         -- number of eligible names
    """
    if first_dates is None:
        first_dates = pd.Series(
            {t: prices[t].first_valid_index() for t in prices.columns}, name="first_date"
        )

    tickers = [t for t in prices.columns if pd.notna(first_dates.get(t))]
    prices = prices[tickers]
    px = prices.to_numpy()
    idx = prices.index
    cols = list(prices.columns)
    col_pos = {c: j for j, c in enumerate(cols)}
    valid = ~np.isnan(px)
    cum_valid = np.cumsum(valid, axis=0)  # history count up to & incl. each row
    fd_arr = np.array([first_dates[c].to_datetime64() for c in cols])

    mkt_rets = spy_prices.pct_change().dropna()
    mkt_idx = mkt_rets.index
    mkt_vals = mkt_rets.to_numpy()

    rebal_dates = pd.date_range(start=idx[0], end=idx[-1], freq=rebal_freq)
    rebal_dates = [d for d in rebal_dates if idx[0] <= d <= idx[-1]]
    lag = pd.Timedelta(days=lag_days)

    rows: list[dict] = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()

    def _compound_block(block: np.ndarray) -> np.ndarray:
        """Per-column compounded return of a (days x names) return block, NaN-safe."""
        out = np.empty(block.shape[1])
        for jj in range(block.shape[1]):
            col = block[:, jj]
            col = col[~np.isnan(col)]
            out[jj] = (1.0 + col).prod() - 1.0 if col.size else np.nan
        return out

    for i in range(len(rebal_dates) - 1):
        rebal_dt = rebal_dates[i]
        next_rebal = rebal_dates[i + 1]

        # Row position of last bar <= rebal_dt
        rpos = idx.searchsorted(rebal_dt, side="right") - 1
        if rpos < 0:
            continue
        rebal_ts = rebal_dt.to_datetime64()
        listed = fd_arr <= rebal_ts
        enough = cum_valid[rpos] >= min_history
        elig = listed & enough
        if elig.sum() < min_stocks:
            continue

        ages = (rebal_ts - fd_arr).astype("timedelta64[D]").astype(float) / 365.25
        elig_pos = np.where(elig)[0]
        order = elig_pos[np.argsort(ages[elig_pos])]  # youngest first
        k = max(1, int(np.floor(len(order) * tercile)))
        young_pos = order[:k]
        old_pos = order[-k:]
        young_tickers = [cols[j] for j in young_pos]
        old_tickers = [cols[j] for j in old_pos]

        # One-day execution lag: returns realised in (rebal_dt+lag, next_rebal+lag].
        entry_cut = rebal_dt + lag
        end_cut = next_rebal + lag
        # daily returns block for held names over the window
        lo = idx.searchsorted(entry_cut, side="right")
        hi = idx.searchsorted(end_cut, side="right")
        if hi <= lo:
            continue
        # returns over [lo, hi): px[lo..hi]/px[lo-1..hi-1]-1 ; use pct from prices
        seg = px[lo - 1:hi] if lo > 0 else px[lo:hi]
        if seg.shape[0] < 2:
            continue
        rblock = seg[1:] / seg[:-1] - 1.0

        old_comp = _compound_block(rblock[:, old_pos])
        young_comp = _compound_block(rblock[:, young_pos])
        old_ret = float(np.nanmean(old_comp))
        young_ret = float(np.nanmean(young_comp))
        if np.isnan(old_ret) or np.isnan(young_ret):
            continue

        mlo = mkt_idx.searchsorted(entry_cut, side="left")
        mhi = mkt_idx.searchsorted(end_cut, side="right")
        mseg = mkt_vals[mlo:mhi]
        mseg = mseg[~np.isnan(mseg)]
        mkt_ret = float((1.0 + mseg).prod() - 1.0) if mseg.size else np.nan

        age_ret = old_ret - young_ret

        # Turnover vs previous rebalance (fraction of legs that changed, two-sided 0..2).
        cur_long, cur_short = set(old_tickers), set(young_tickers)
        if prev_long or prev_short:
            to_long = len(cur_long.symmetric_difference(prev_long)) / max(len(cur_long | prev_long), 1)
            to_short = len(cur_short.symmetric_difference(prev_short)) / max(len(cur_short | prev_short), 1)
            turnover = to_long + to_short
        else:
            turnover = 2.0  # initial full establishment
        prev_long, prev_short = cur_long, cur_short

        rows.append(
            {
                "date": rebal_dt,
                "old_ret": old_ret,
                "young_ret": young_ret,
                "age_ret": age_ret,
                "mkt_ret": mkt_ret,
                "avg_age_old": float(np.mean(ages[old_pos])),
                "avg_age_young": float(np.mean(ages[young_pos])),
                "turnover": float(turnover),
                "n": int(elig.sum()),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
# Costs: one-way bps x turnover + borrow on the short leg
# ---------------------------------------------------------------------------
def apply_costs(
    result: pd.DataFrame,
    one_way_bps: float = ONE_WAY_BPS,
    borrow_bps_annual: float = BORROW_BPS_ANNUAL,
) -> pd.Series:
    """Net monthly AGE returns after transaction costs and short borrow.

    Transaction cost per rebalance = one_way_bps/1e4 * turnover (turnover already counts
    both legs). Borrow charge = borrow_bps_annual/1e4 / 12 on the short (young) leg each
    month the position is held. Returns the net monthly AGE series.
    """
    tc = (one_way_bps / 1e4) * result["turnover"]
    borrow = (borrow_bps_annual / 1e4) / 12.0
    return result["age_ret"] - tc - borrow


# ---------------------------------------------------------------------------
# Placebo: label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(
    prices: pd.DataFrame,
    spy_prices: pd.Series,
    first_dates: pd.Series | None = None,
    n_perm: int = 200,
    seed: int = 532,
    **kw,
) -> dict:
    """Shuffle the firm-age labels across names and rebuild the long-short.

    Under the null (age carries no return information), permuting which name is "old" vs
    "young" should produce AGE means as large as the real one about as often. We report the
    two-sided permutation p-value: fraction of shuffles whose |mean| >= |real mean|.
    """
    if first_dates is None:
        first_dates = pd.Series(
            {t: prices[t].first_valid_index() for t in prices.columns}, name="first_date"
        )

    real = age_portfolio(prices, spy_prices, first_dates=first_dates, **kw)
    if real.empty:
        return {"real_mean": np.nan, "p_value": np.nan, "n_perm": 0}
    real_mean = float(real["age_ret"].mean())

    rng = np.random.default_rng(seed)
    tickers = list(first_dates.index)
    fd_vals = first_dates.to_numpy()
    hits = 0
    perm_means = []
    for _ in range(n_perm):
        perm = rng.permutation(len(fd_vals))
        shuffled = pd.Series(fd_vals[perm], index=tickers, name="first_date")
        res = age_portfolio(prices, spy_prices, first_dates=shuffled, **kw)
        if res.empty:
            continue
        m = float(res["age_ret"].mean())
        perm_means.append(m)
        if abs(m) >= abs(real_mean):
            hits += 1
    n_eff = len(perm_means)
    p = (hits + 1) / (n_eff + 1) if n_eff > 0 else np.nan
    return {
        "real_mean": real_mean,
        "p_value": float(p),
        "n_perm": n_eff,
        "perm_mean": float(np.mean(perm_means)) if perm_means else np.nan,
        "perm_std": float(np.std(perm_means)) if perm_means else np.nan,
    }


# ---------------------------------------------------------------------------
# Annualised summary statistics
# ---------------------------------------------------------------------------
def summary(monthly_returns: pd.Series, periods_per_year: int = 12) -> dict:
    """Annualised statistics for a monthly return series.

    Returns mean (annualised), vol (annualised), Sharpe, a Newey-West HAC t-stat,
    hit-rate (months > 0), max drawdown, and *n*. The HAC t-stat is the inference-bar
    number (one-sample t that the mean monthly AGE return = 0).
    """
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
