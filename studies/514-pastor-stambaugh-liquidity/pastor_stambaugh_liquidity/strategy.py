"""The Pastor-Stambaugh liquidity-risk engine -- Pastor & Stambaugh (2003).

The idea under test is a *risk loading*, not a *level*:

  1. Build an **aggregate market liquidity** series. We use an Amihud-style monthly
     average illiquidity across the cross-section: for each name and month, the average
     daily |return| / dollar-volume; then average across names. A *rise* in illiquidity
     is a *negative* liquidity shock. We sign-flip so a higher series = MORE liquid, and
     take the monthly first difference (the "innovation" / liquidity shock).

  2. For each stock, estimate a **liquidity beta** gamma_i on a trailing window:
     regress the stock's monthly excess return on the contemporaneous liquidity shock
     (controlling for the market excess return). gamma_i is the loading on the shock.

  3. Each month, rank stocks by their trailing liquidity beta. Pastor-Stambaugh predict
     that **high-gamma** stocks (more exposed to liquidity-shock risk) earn HIGHER
     returns. Go long the high-gamma tertile, short the low-gamma tertile, equal-weight,
     dollar-neutral. Hold one month.

Discipline (METHODOLOGY house rules):
- ONE execution lag: betas estimated through month *t-1*; the sort is public at the
  close of *t-1*; we enter at the *t-1* close and earn month *t*'s return. No same-bar
  fill, no look-ahead (the trailing window never includes the holding month).
- Costs: one-way bps x NAV x turnover, both legs, charged at each rebalance; plus an
  annual borrow on the short leg, pro-rated monthly.
- One-sample / Newey-West HAC t on the monthly long-short series is the inference number.
- Placebo: shuffle the liquidity-beta labels across names and re-sort -- the long-short
  mean should collapse to ~0 (p-value from a permutation null).
- A deterministic synthetic positive control proves the engine recovers a planted premium.

Survivorship: the universe is current large-caps projected backwards -- named on the
SIGNAL axis; results are upper bounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12

# Risk-free proxy (annual, constant -- no FRED dependency)
RF_ANNUAL = 0.03
RF_MONTHLY = RF_ANNUAL / 12.0

# Trailing window (months) over which each name's liquidity beta is estimated
BETA_WINDOW_M = 36

# Default trading cost (one-way, basis points) and short borrow (annual, bps)
COST_BPS = 10.0
BORROW_ANN_BPS = 50.0


# ---------------------------------------------------------------------------
# Aggregate liquidity series
# ---------------------------------------------------------------------------
def aggregate_liquidity(
    close: pd.DataFrame,
    dollar_volume: pd.DataFrame,
) -> pd.Series:
    """Monthly aggregate-liquidity *level* (higher = more liquid), Amihud-style.

    For each name and trading day: illiq = |daily return| / dollar_volume (Amihud 2002).
    Average within each calendar month, then across names -> aggregate monthly
    illiquidity. We sign-flip (multiply by -1) and z-score so the series reads as a
    *liquidity* level (higher = more liquid). The monthly first difference of this series
    is the liquidity *shock* (innovation) used as the risk factor.
    """
    daily_ret = close.pct_change()
    illiq = daily_ret.abs() / dollar_volume.replace(0, np.nan)
    illiq = illiq.replace([np.inf, -np.inf], np.nan)

    # Monthly mean illiquidity per name, then cross-sectional average
    monthly_illiq_name = illiq.resample("ME").mean()
    agg_illiq = monthly_illiq_name.mean(axis=1)

    # Sign-flip to liquidity, z-score for a clean shock series
    liq = -agg_illiq
    liq_z = (liq - liq.mean()) / (liq.std(ddof=1) + 1e-18)
    liq_z.name = "agg_liquidity"
    return liq_z.dropna()


def liquidity_shock(liq_level: pd.Series) -> pd.Series:
    """Liquidity innovation = monthly first difference of the aggregate-liquidity level."""
    return liq_level.diff().dropna().rename("liq_shock")


# ---------------------------------------------------------------------------
# Monthly returns
# ---------------------------------------------------------------------------
def monthly_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Month-end simple returns from a daily close panel."""
    me = close.resample("ME").last()
    return me.pct_change().dropna(how="all")


def monthly_excess(close: pd.DataFrame, spy: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Monthly excess (over RF) stock returns and market excess return."""
    stk = monthly_returns(close) - RF_MONTHLY
    mkt = close_to_monthly(spy) - RF_MONTHLY
    return stk, mkt


def close_to_monthly(spy: pd.Series) -> pd.Series:
    me = spy.resample("ME").last()
    return me.pct_change().dropna().rename("mkt")


# ---------------------------------------------------------------------------
# Liquidity betas (the risk loading)
# ---------------------------------------------------------------------------
def liquidity_betas(
    stk_excess: pd.DataFrame,
    mkt_excess: pd.Series,
    liq_shock: pd.Series,
    window: int = BETA_WINDOW_M,
) -> pd.DataFrame:
    """Rolling liquidity beta gamma_i for each name.

    For each ending month *t*, on the trailing ``window`` months regress::

        r_i = a + b * mkt_excess + gamma * liq_shock + e

    gamma is the loading on the contemporaneous liquidity shock (the Pastor-Stambaugh
    liquidity beta). Returns a (month x ticker) frame of trailing gammas. A month's gamma
    uses ONLY data up to and including that month -- the holding month is always strictly
    after, so there is no look-ahead.
    """
    idx = stk_excess.index.intersection(mkt_excess.index).intersection(liq_shock.index)
    R = stk_excess.reindex(idx)
    M = mkt_excess.reindex(idx).to_numpy()
    L = liq_shock.reindex(idx).to_numpy()

    months = list(idx)
    out = pd.DataFrame(index=idx, columns=stk_excess.columns, dtype=float)

    for end in range(window - 1, len(months)):
        sl = slice(end - window + 1, end + 1)
        m = M[sl]
        l = L[sl]
        X = np.column_stack([np.ones_like(m), m, l])  # (window, 3)
        # Solve once; reuse the pseudo-inverse projection for every name
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            continue
        proj = XtX_inv @ X.T  # (3, window)
        block = R.iloc[sl].to_numpy()  # (window, n_stocks)
        # gamma row = third coefficient for each column; skip columns with NaNs
        for j, col in enumerate(R.columns):
            y = block[:, j]
            if np.isnan(y).any():
                continue
            coef = proj @ y
            out.iat[end, j] = coef[2]
    return out


# ---------------------------------------------------------------------------
# Long-short sort
# ---------------------------------------------------------------------------
def liquidity_sort(
    close: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    spy: pd.Series,
    window: int = BETA_WINDOW_M,
    quantile: float = 1.0 / 3.0,
    cost_bps: float = 0.0,
    borrow_ann_bps: float = 0.0,
    shuffle_seed: int | None = None,
) -> pd.DataFrame:
    """Monthly long high-gamma / short low-gamma return, net of costs and borrow.

    One execution lag: the gamma at month *t-1* (trailing window ending *t-1*) ranks the
    cross-section; we hold over month *t*. Equal-weight within each tertile, dollar
    neutral (long $1 high-gamma, short $1 low-gamma).

    ``cost_bps`` -- one-way trading cost (bps) charged on the turnover of BOTH legs each
    rebalance. ``borrow_ann_bps`` -- annual short-borrow on the short leg, pro-rated
    monthly. ``shuffle_seed`` -- if given, the gamma labels are permuted across names
    each month before ranking (the placebo / label-shuffle null).

    Returns a frame indexed by holding month with columns:
        ``long_ret``, ``short_ret``, ``ls_gross``, ``ls_net``, ``turnover``, ``n``.
    """
    stk_ex, mkt_ex = monthly_excess(close, spy)
    liq = aggregate_liquidity(close, dollar_volume)
    shock = liquidity_shock(liq)
    gammas = liquidity_betas(stk_ex, mkt_ex, shock, window=window)

    stk_ret = monthly_returns(close)  # simple (not excess) for the leg returns
    months = list(gammas.index)

    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None

    rows: list[dict] = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()

    for i in range(len(months) - 1):
        sig_month = months[i]
        hold_month = months[i + 1]

        g = gammas.loc[sig_month].dropna()
        if len(g) < 9:  # need enough names for tertiles
            continue

        if rng is not None:
            vals = g.to_numpy().copy()
            rng.shuffle(vals)
            g = pd.Series(vals, index=g.index)

        n = len(g)
        k = max(1, int(np.floor(n * quantile)))
        ranked = g.sort_values()
        low_names = list(ranked.index[:k])      # low gamma -> SHORT
        high_names = list(ranked.index[-k:])    # high gamma -> LONG

        if hold_month not in stk_ret.index:
            continue
        hold = stk_ret.loc[hold_month]

        long_r = float(hold[high_names].dropna().mean())
        short_r = float(hold[low_names].dropna().mean())
        if np.isnan(long_r) or np.isnan(short_r):
            continue

        ls_gross = long_r - short_r

        # Turnover: fraction of names that changed in each leg (one-way, both legs)
        cur_long, cur_short = set(high_names), set(low_names)
        if prev_long or prev_short:
            turn_long = len(cur_long ^ prev_long) / (2 * max(len(cur_long), 1))
            turn_short = len(cur_short ^ prev_short) / (2 * max(len(cur_short), 1))
            turnover = turn_long + turn_short
        else:
            turnover = 2.0  # initial full establishment of both legs
        prev_long, prev_short = cur_long, cur_short

        monthly_cost = turnover * cost_bps * 1e-4
        monthly_borrow = borrow_ann_bps * 1e-4 / MONTHS_PER_YEAR
        ls_net = ls_gross - monthly_cost - monthly_borrow

        rows.append(
            {
                "date": hold_month,
                "long_ret": long_r,
                "short_ret": short_r,
                "ls_gross": ls_gross,
                "ls_net": ls_net,
                "turnover": turnover,
                "n": n,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def hac_tstat(r: pd.Series) -> float:
    """Newey-West (HAC) one-sample t-stat on the mean of a monthly series."""
    x = pd.Series(r).dropna().to_numpy(dtype=float)
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def summary(r: pd.Series, periods_per_year: int = 12) -> dict:
    """Annualised stats for a monthly return series + HAC t (the inference number)."""
    r = pd.Series(r).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: np.nan for k in
                ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n")}
    mean_ann = float(r.mean()) * periods_per_year
    vol_ann = float(r.std(ddof=1)) * np.sqrt(periods_per_year)
    sr = mean_ann / vol_ann if vol_ann > 0 else np.nan
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    return {
        "mean": mean_ann,
        "vol": vol_ann,
        "sharpe": sr,
        "tstat": hac_tstat(r),
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": dd,
        "n": n,
    }


def placebo_pvalue(
    close: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    spy: pd.Series,
    window: int = BETA_WINDOW_M,
    n_perm: int = 200,
    seed: int = 514,
) -> dict:
    """Permutation null: shuffle the liquidity-beta labels and re-sort ``n_perm`` times.

    Returns the real long-short annualised mean, the permutation mean distribution, and
    a two-sided p-value (fraction of |placebo mean| >= |real mean|). If the real signal
    is genuine, the placebo means cluster around zero and p is small.
    """
    real = liquidity_sort(close, dollar_volume, spy, window=window)
    if real.empty:
        return {"real_mean_ann": float("nan"), "p_value": float("nan"), "placebo": []}
    real_mean = float(real["ls_gross"].mean() * MONTHS_PER_YEAR)

    rng = np.random.default_rng(seed)
    placebo_means: list[float] = []
    for _ in range(n_perm):
        s = int(rng.integers(0, 2**31 - 1))
        pl = liquidity_sort(close, dollar_volume, spy, window=window, shuffle_seed=s)
        if pl.empty:
            continue
        placebo_means.append(float(pl["ls_gross"].mean() * MONTHS_PER_YEAR))

    arr = np.array(placebo_means)
    if arr.size == 0:
        return {"real_mean_ann": real_mean, "p_value": float("nan"), "placebo": []}
    p = float((np.abs(arr) >= abs(real_mean)).mean())
    return {
        "real_mean_ann": real_mean,
        "p_value": p,
        "placebo_mean": float(arr.mean()),
        "placebo_std": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
        "placebo": placebo_means,
    }


# ---------------------------------------------------------------------------
# Synthetic positive control
# ---------------------------------------------------------------------------
def control_sweep(
    premiums: list[float],
    n_stocks: int = 45,
    n_days: int = 2600,
    window: int = 30,
    seed: int = 514,
) -> pd.DataFrame:
    """Plant a known liquidity-risk premium and verify the engine recovers it.

    For each premium, generate a synthetic panel, run the full sort, and record the
    long-short annualised mean and HAC t. A faithful engine is monotone in the premium
    and prints a clear positive t at a strong planted premium. This is a power check
    ONLY -- it never licenses a real-tape stamp.
    """
    from . import data as _data

    rows = []
    for prem in premiums:
        close, dv, spy, _ = _data.synthetic_panel(
            n_stocks=n_stocks, n_days=n_days, liq_premium=prem, seed=seed
        )
        res = liquidity_sort(close, dv, spy, window=window)
        if res.empty:
            rows.append({"premium": prem, "ls_mean_ann": np.nan, "tstat": np.nan, "n": 0})
            continue
        s = summary(res["ls_gross"])
        rows.append({"premium": prem, "ls_mean_ann": s["mean"], "tstat": s["tstat"], "n": s["n"]})
    return pd.DataFrame(rows)
