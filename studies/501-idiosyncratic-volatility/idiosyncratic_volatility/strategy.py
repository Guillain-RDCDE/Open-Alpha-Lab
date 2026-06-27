"""The Idiosyncratic-Volatility engine and its honest controls -- Study 501.

The claim (Ang, Hodrick, Xing & Zhang 2006): stocks with HIGH idiosyncratic volatility --
the residual std after stripping out the market factor (CAPM) -- earn LOWER subsequent
returns. The puzzling part is the *negative* sign: standard theory says only systematic
(non-diversifiable) risk should be priced, so idiosyncratic risk should be irrelevant; AHXZ
find it is priced, and with the *wrong* sign.

Construction (monthly rebalance):
  1. For each stock, regress its trailing 252-day daily excess return on SPY's daily excess
     return (CAPM). The IDIOSYNCRATIC VOL is the annualised std of the OLS residual.
  2. Each month, sort stocks into IVOL quintiles (Q1 = lowest IVOL, Q5 = highest).
  3. Hold a LOW-MINUS-HIGH long-short: long Q1 (low IVOL), short Q5 (high IVOL),
     equal-weight within each leg, dollar-neutral.
  4. Measure the next-month return -- entered one trading day AFTER the signal date (a
     single documented execution lag, no same-bar fill, no look-ahead).

The puzzle predicts a POSITIVE low-minus-high spread (low IVOL beats high IVOL). This is
DISTINCT from Study 330 (total-vol low-vol anomaly): here vol is the CAPM RESIDUAL, with the
market beta removed, so any spread is not just a re-labelled low-beta bet.

Costs: a one-way bps x NAV x turnover charge each rebalance, plus an annual borrow on the
short (high-IVOL) leg. Gross vs net is reported separately.

The panel is **survivorship-biased** (current S&P 500 projected backwards). All results are
upper-bound estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12

# Approximate risk-free rate (annualised), used for CAPM excess returns. No FRED dependency.
RF_ANNUAL = 0.03
RF_DAILY = RF_ANNUAL / 252.0

# Trailing window for the residual-vol estimate (one trading year).
IVOL_WINDOW = 252

# Number of quintiles for the cross-sectional sort.
N_QUANTILES = 5

# Rebalance frequency (month-start).
REBAL_FREQ = "MS"


# ---------------------------------------------------------------------------
# Idiosyncratic volatility (rolling CAPM residual std)
# ---------------------------------------------------------------------------
def rolling_ivol(
    stock_excess: pd.DataFrame,
    mkt_excess: pd.Series,
    window: int = IVOL_WINDOW,
) -> pd.DataFrame:
    """Rolling annualised std of each stock's CAPM residual vs the market.

    For each rolling ``window`` of daily excess returns, the residual variance of the OLS of
    ``r_i`` on ``r_mkt`` is::

        Var(eps_i) = Var(r_i) - beta_i^2 * Var(r_mkt),    beta_i = Cov(r_i, r_mkt) / Var(r_mkt)

    which equals ``Var(r_i) * (1 - rho^2)`` with ``rho`` the rolling correlation. IVOL is the
    annualised residual std ``sqrt(Var(eps_i) * 252)``. Computed vectorised via pandas rolling
    moments (no python loop over stocks); NaN until ``window // 2`` observations exist.

    Returns a (date x ticker) frame of annualised IVOL.
    """
    common = stock_excess.index.intersection(mkt_excess.index)
    sr = stock_excess.reindex(common)
    mr = mkt_excess.reindex(common)

    minp = window // 2
    var_i = sr.rolling(window, min_periods=minp).var()
    var_m = mr.rolling(window, min_periods=minp).var()
    cov_im = sr.rolling(window, min_periods=minp).cov(mr)

    beta = cov_im.div(var_m, axis=0)
    resid_var = var_i.sub(beta.pow(2).mul(var_m, axis=0))
    resid_var = resid_var.clip(lower=0.0)
    return np.sqrt(resid_var * 252.0)


# ---------------------------------------------------------------------------
# The low-minus-high IVOL long-short
# ---------------------------------------------------------------------------
def ivol_long_short(
    prices: pd.DataFrame,
    spy_prices: pd.Series,
    window: int = IVOL_WINDOW,
    n_quantiles: int = N_QUANTILES,
    rebal_freq: str = REBAL_FREQ,
    min_stocks: int = 15,
    cost_bps: float = 0.0,
    borrow_ann_bps: float = 0.0,
) -> pd.DataFrame:
    """Construct the monthly low-minus-high IVOL long-short, gross and net.

    Steps at each rebalance date *t*:
      1. Compute IVOL (rolling CAPM-residual std) up to *t*; take the last snapshot <= *t*.
      2. Sort into ``n_quantiles`` quintiles; Q1 = lowest IVOL, Q_top = highest.
      3. Long Q1 (low IVOL), short Q_top (high IVOL), equal-weight, dollar-neutral.
      4. The holding window is the NEXT month, entered one trading day AFTER *t* (execution
         lag) -- so the signal is strictly public before any fill.
      5. Turnover is the L1 change in target weights vs the prior month; costs = one-way
         ``cost_bps`` x turnover, plus a pro-rated annual borrow on the short leg.

    Returns a DataFrame indexed by rebalance date with columns:
        ``low_ivol_ret``  -- equal-weight next-month return of the low-IVOL leg (Q1)
        ``high_ivol_ret`` -- equal-weight next-month return of the high-IVOL leg (Q_top)
        ``ls_gross``      -- gross low-minus-high spread (long Q1 - short Q_top)
        ``turnover``      -- one-sided turnover of the combined book this rebalance
        ``cost``          -- trading + borrow drag this month
        ``ls_net``        -- net spread (gross - cost)
        ``mkt_ret``       -- SPY return over the same holding window
        ``avg_ivol_low`` / ``avg_ivol_high`` -- mean IVOL of each leg at formation
        ``n``             -- number of ranked stocks
    """
    stock_rets = prices.pct_change().dropna(how="all")
    mkt_rets = spy_prices.pct_change().dropna()
    stock_excess = stock_rets.sub(RF_DAILY)
    mkt_excess = mkt_rets.sub(RF_DAILY)

    ivol = rolling_ivol(stock_excess, mkt_excess, window=window)

    rebal_dates = pd.date_range(
        start=prices.index[min(window, len(prices) - 1)],
        end=prices.index[-1],
        freq=rebal_freq,
    )
    rebal_dates = [d for d in rebal_dates if d <= prices.index[-1]]

    rows: list[dict] = []
    prev_weights: pd.Series | None = None

    for i, rebal_dt in enumerate(rebal_dates[:-1]):
        next_rebal = rebal_dates[i + 1]

        snap = ivol[ivol.index <= rebal_dt]
        if snap.empty:
            continue
        iv = snap.iloc[-1].dropna()
        iv = iv[iv > 0]
        if len(iv) < min_stocks:
            continue

        # Quintile labels 0..n_quantiles-1 by ascending IVOL.
        try:
            labels = pd.qcut(iv.rank(method="first"), n_quantiles, labels=False)
        except ValueError:
            continue
        low_tickers = iv.index[labels == 0]
        high_tickers = iv.index[labels == n_quantiles - 1]
        if len(low_tickers) == 0 or len(high_tickers) == 0:
            continue

        # Target dollar-neutral weights: +1/nL on low leg, -1/nH on high leg.
        weights = pd.Series(0.0, index=iv.index)
        weights[low_tickers] = 1.0 / len(low_tickers)
        weights[high_tickers] = -1.0 / len(high_tickers)

        # Execution lag: the holding window opens one trading day AFTER the signal date.
        future_idx = stock_rets.index[stock_rets.index > rebal_dt]
        if len(future_idx) == 0:
            continue
        entry_dt = future_idx[0]
        hold_mask = (stock_rets.index >= entry_dt) & (stock_rets.index <= next_rebal)
        hold = stock_rets.loc[hold_mask]
        mkt_hold = mkt_rets.loc[(mkt_rets.index >= entry_dt) & (mkt_rets.index <= next_rebal)]
        if hold.empty or mkt_hold.empty:
            continue

        def _compound(ser: pd.Series) -> float:
            return float((1.0 + ser.dropna()).prod() - 1.0)

        low_ret = float(hold[low_tickers].apply(_compound).mean())
        high_ret = float(hold[high_tickers].apply(_compound).mean())
        mkt_ret = _compound(mkt_hold)
        ls_gross = low_ret - high_ret  # low-minus-high (the puzzle predicts > 0)

        # Turnover: L1 change in target weights vs prior month (full re-strike if first).
        if prev_weights is None:
            turnover = float(weights.abs().sum())  # 2.0 for a fully-built book
        else:
            aligned = weights.reindex(prev_weights.index.union(weights.index)).fillna(0.0)
            prev_al = prev_weights.reindex(aligned.index).fillna(0.0)
            turnover = float((aligned - prev_al).abs().sum())
        prev_weights = weights

        # Costs: one-way bps x turnover, plus pro-rated annual borrow on the short leg ($1).
        trade_cost = cost_bps * 1e-4 * turnover
        borrow_cost = borrow_ann_bps * 1e-4 / MONTHS_PER_YEAR
        cost = trade_cost + borrow_cost
        ls_net = ls_gross - cost

        rows.append({
            "date": rebal_dt,
            "low_ivol_ret": low_ret,
            "high_ivol_ret": high_ret,
            "ls_gross": ls_gross,
            "turnover": turnover,
            "cost": cost,
            "ls_net": ls_net,
            "mkt_ret": mkt_ret,
            "avg_ivol_low": float(iv[low_tickers].mean()),
            "avg_ivol_high": float(iv[high_tickers].mean()),
            "n": int(len(iv)),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
# Quintile mean returns (the monotone-decreasing AHXZ table)
# ---------------------------------------------------------------------------
def quintile_means(
    prices: pd.DataFrame,
    spy_prices: pd.Series,
    window: int = IVOL_WINDOW,
    n_quantiles: int = N_QUANTILES,
    rebal_freq: str = REBAL_FREQ,
    min_stocks: int = 15,
) -> pd.Series:
    """Average next-month return of each IVOL quintile (Q1=low ... Q_top=high).

    The AHXZ puzzle says this series is DECREASING in IVOL (low-IVOL quintile out-earns
    high-IVOL quintile). Returns a Series indexed ``Q1..Qn`` of annualised mean returns.
    """
    stock_rets = prices.pct_change().dropna(how="all")
    mkt_rets = spy_prices.pct_change().dropna()
    ivol = rolling_ivol(stock_rets.sub(RF_DAILY), mkt_rets.sub(RF_DAILY), window=window)

    rebal_dates = pd.date_range(
        start=prices.index[min(window, len(prices) - 1)],
        end=prices.index[-1], freq=rebal_freq,
    )
    rebal_dates = [d for d in rebal_dates if d <= prices.index[-1]]

    buckets: dict[int, list[float]] = {q: [] for q in range(n_quantiles)}
    for i, rebal_dt in enumerate(rebal_dates[:-1]):
        next_rebal = rebal_dates[i + 1]
        snap = ivol[ivol.index <= rebal_dt]
        if snap.empty:
            continue
        iv = snap.iloc[-1].dropna()
        iv = iv[iv > 0]
        if len(iv) < min_stocks:
            continue
        try:
            labels = pd.qcut(iv.rank(method="first"), n_quantiles, labels=False)
        except ValueError:
            continue
        future_idx = stock_rets.index[stock_rets.index > rebal_dt]
        if len(future_idx) == 0:
            continue
        entry_dt = future_idx[0]
        hold = stock_rets.loc[(stock_rets.index >= entry_dt) & (stock_rets.index <= next_rebal)]
        if hold.empty:
            continue
        comp = (1.0 + hold).prod() - 1.0
        for q in range(n_quantiles):
            tickers = iv.index[labels == q]
            if len(tickers):
                buckets[q].append(float(comp[tickers].mean()))

    means = {f"Q{q + 1}": (np.mean(v) * MONTHS_PER_YEAR if v else np.nan)
             for q, v in buckets.items()}
    return pd.Series(means)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def hac_tstat(r: pd.Series) -> float:
    """Newey-West (HAC) t-stat on the mean of a monthly return series.

    Lag rule ``floor(4*(n/100)^(2/9))`` -- the desk's shared convention.
    """
    x = pd.Series(r).dropna().to_numpy(dtype=float)
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def summary(monthly_returns: pd.Series) -> dict:
    """Annualised stats for a monthly spread series: mean, vol, Sharpe, HAC t, hit, max-DD, n."""
    r = pd.Series(monthly_returns).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: float("nan") for k in
                ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n")}
    mean_ann = float(r.mean()) * MONTHS_PER_YEAR
    vol_ann = float(r.std(ddof=1)) * np.sqrt(MONTHS_PER_YEAR)
    sr = mean_ann / vol_ann if vol_ann > 0 else float("nan")
    nav = (1.0 + r).cumprod()
    dd = float((nav / nav.cummax() - 1.0).min())
    return {
        "mean": mean_ann,
        "vol": vol_ann,
        "sharpe": sr,
        "tstat": hac_tstat(r),
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": dd,
        "n": int(n),
    }


def placebo_pvalue(
    spread: pd.Series,
    n_shuffle: int = 2000,
    seed: int = 501,
) -> float:
    """Label-shuffle / sign-flip placebo p-value for the spread's mean.

    Under the null of no edge, the sign of each monthly spread is exchangeable. We draw
    ``n_shuffle`` random sign-flips of the demeaned-around-zero series and ask how often the
    permuted |mean| exceeds the observed |mean|. A small p-value means the observed mean is
    hard to manufacture by chance; a large one means it is noise.
    """
    x = pd.Series(spread).dropna().to_numpy(dtype=float)
    n = x.size
    if n < 6:
        return float("nan")
    obs = abs(x.mean())
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n_shuffle, n))
    perm_means = np.abs((flips * x).mean(axis=1))
    return float((perm_means >= obs).mean())
