"""The Betting-Against-Correlation engine and its honest controls -- Study 502.

The claim (Asness, Frazzini, Gormsen & Pedersen 2020, "Betting Against Correlation"): the
Betting-Against-Beta premium is carried by the *correlation* component of beta, not the
volatility component. Since

    beta_i = rho_{i,m} * (sigma_i / sigma_m),

they split BAB into a Betting-Against-Correlation (BAC) leg -- sort on rho, hold vol roughly
constant -- and a Betting-Against-Volatility (BAV) leg, and find BAC carries the premium.

This module builds BAC on a large-cap survivor cross-section:

  1. Each rebalance, estimate every name's trailing 252-day **correlation** to the market and
     its trailing **beta** (for the neutralisation, not the sort).
  2. Rank by correlation; long the low-correlation half, short the high-correlation half.
  3. **Beta-neutralise**: scale the short leg so the book's ex-ante market beta ~ 0. Without
     this the "edge" would just be a hidden short of high-beta market exposure -- exactly the
     conflation AFGP warn against, and the line that separates this from [238 Betting-Against-Beta].
  4. Enter the close **one trading day after** the signal date (one documented execution lag,
     no same-bar fill, no look-ahead). Hold to the next monthly rebalance.

Costs: a one-way ``cost_bps`` x NAV charge x realised turnover on each rebalance, plus an annual
short borrow on the short leg's notional. Gross vs net are reported separately.

Inference: a one-sample / Newey-West HAC t-stat on the monthly book return is the bar number;
a **label-shuffle placebo** (permute the correlation ranks before forming legs) is the null;
a deterministic **synthetic positive control** proves the engine recovers a planted BAC premium
and stays flat on the null.

The panel is **survivorship-biased** (current large-cap names projected backwards). All positive
results are upper-bound estimates -- named on the Signal axis with an opt-in guard note.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Approximate constant risk-free rate (annualised). No FRED dependency; sufficient for ranking
# and for an excess-return Sharpe.
RF_ANNUAL = 0.03
RF_DAILY = RF_ANNUAL / 252

CORR_WINDOW = 252      # trailing window for correlation & beta estimation (1 trading year)
REBAL_FREQ = "MS"      # month-start rebalancing
MONTHS_PER_YEAR = 12
EXEC_LAG_DAYS = 1      # enter the close one day after the signal is public


# ---------------------------------------------------------------------------
# Rolling correlation & beta
# ---------------------------------------------------------------------------
def rolling_corr_beta(
    stock_rets: pd.DataFrame,
    mkt_rets: pd.Series,
    window: int = CORR_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling correlation-to-market and rolling beta for each name.

    corr_{i,t} = Corr(r_i, r_mkt) and beta_{i,t} = Cov(r_i, r_mkt)/Var(r_mkt) on the trailing
    ``window`` days. Returns ``(corr, beta)`` as (date x ticker) frames; NaN until the window
    is half-filled.
    """
    common = stock_rets.index.intersection(mkt_rets.index)
    sr = stock_rets.reindex(common)
    mr = mkt_rets.reindex(common)
    minp = window // 2

    roll_cov = sr.rolling(window, min_periods=minp).cov(mr)
    roll_var = mr.rolling(window, min_periods=minp).var()
    roll_std_s = sr.rolling(window, min_periods=minp).std()
    roll_std_m = mr.rolling(window, min_periods=minp).std()

    beta = roll_cov.div(roll_var, axis=0)
    corr = roll_cov.div(roll_std_s.mul(roll_std_m, axis=0))
    return corr, beta


# ---------------------------------------------------------------------------
# BAC portfolio construction (correlation sort, beta-neutral, one execution lag)
# ---------------------------------------------------------------------------
def bac_portfolio(
    prices: pd.DataFrame,
    spy_prices: pd.Series,
    window: int = CORR_WINDOW,
    rebal_freq: str = REBAL_FREQ,
    min_stocks: int = 16,
    sort_on: str = "corr",
    shuffle_seed: int | None = None,
    exec_lag_days: int = EXEC_LAG_DAYS,
) -> pd.DataFrame:
    """Monthly Betting-Against-Correlation book.

    For each rebalance date *t*:
      1. Snapshot trailing correlation & beta as of *t*.
      2. Rank by ``sort_on`` (``"corr"`` for BAC; ``"beta"`` or ``"vol"`` for the decomposition
         comparisons). Optionally permute the rank labels first (``shuffle_seed`` -> placebo).
      3. Long the low-rank half, short the high-rank half (equal weight within a leg).
      4. Beta-neutralise: scale the short leg by ``avg_beta_long / avg_beta_short`` so the net
         ex-ante market beta ~ 0.
      5. Enter the close ``exec_lag_days`` after *t*; hold to the next rebalance.

    Returns a per-month DataFrame with columns:
        ``long_ret``/``short_ret``  -- equal-weight leg returns (held)
        ``avg_corr_long``/``avg_corr_short`` -- average correlation of each leg at rebalance
        ``avg_beta_long``/``avg_beta_short`` -- average beta of each leg at rebalance
        ``w_short``  -- beta-neutralising weight on the short leg
        ``bac_ret``  -- long_ret - w_short * short_ret (the beta-neutral book, gross)
        ``turnover`` -- name turnover vs the prior rebalance (for the cost charge)
        ``mkt_ret``  -- SPY return over the same hold
        ``n``        -- universe size that month
    """
    stock_rets = prices.pct_change().dropna(how="all")
    mkt_rets = spy_prices.pct_change().dropna()

    stock_excess = stock_rets.sub(RF_DAILY)
    mkt_excess = mkt_rets.sub(RF_DAILY)

    corr, beta = rolling_corr_beta(stock_excess, mkt_excess, window=window)
    signal = {"corr": corr, "beta": beta}.get(sort_on)
    if sort_on == "vol":
        signal = stock_excess.rolling(window, min_periods=window // 2).std()
    if signal is None:
        raise ValueError(f"unknown sort_on={sort_on!r}")

    rebal_dates = pd.date_range(
        start=prices.index[window], end=prices.index[-1], freq=rebal_freq
    )
    rebal_dates = [d for d in rebal_dates if d <= prices.index[-1]]

    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    rows: list[dict] = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()

    for i, rebal_dt in enumerate(rebal_dates[:-1]):
        next_rebal = rebal_dates[i + 1]

        sig_avail = signal[signal.index <= rebal_dt]
        beta_avail = beta[beta.index <= rebal_dt]
        corr_avail = corr[corr.index <= rebal_dt]
        if sig_avail.empty or beta_avail.empty:
            continue
        sig_snap = sig_avail.iloc[-1].dropna()
        beta_snap = beta_avail.iloc[-1]
        corr_snap = corr_avail.iloc[-1]

        sig_snap = sig_snap[np.isfinite(sig_snap)]
        if len(sig_snap) < min_stocks:
            continue

        # Optional label shuffle (placebo): permute the signal across names.
        if rng is not None:
            permuted = rng.permutation(sig_snap.to_numpy())
            sig_snap = pd.Series(permuted, index=sig_snap.index)

        med = sig_snap.median()
        long_tk = sig_snap[sig_snap <= med].index
        short_tk = sig_snap[sig_snap > med].index
        if len(long_tk) < 2 or len(short_tk) < 2:
            continue

        b_long = float(beta_snap.reindex(long_tk).dropna().mean())
        b_short = float(beta_snap.reindex(short_tk).dropna().mean())
        if not np.isfinite(b_long) or not np.isfinite(b_short) or b_short <= 0:
            continue
        w_short = b_long / b_short  # beta-neutralising scale on the short leg

        # Execution lag: enter the close exec_lag_days after the signal date.
        entry = rebal_dt + pd.Timedelta(days=exec_lag_days)
        hold = stock_rets.loc[(stock_rets.index > entry) & (stock_rets.index <= next_rebal)]
        mkt_hold = mkt_rets.loc[(mkt_rets.index > entry) & (mkt_rets.index <= next_rebal)]
        if hold.empty or mkt_hold.empty:
            continue

        def _compound(ser: pd.Series) -> float:
            s = ser.dropna()
            return float((1.0 + s).prod() - 1.0) if len(s) else float("nan")

        long_ret = float(hold[long_tk].apply(_compound).mean())
        short_ret = float(hold[short_tk].apply(_compound).mean())
        mkt_ret = _compound(mkt_hold)
        if not (np.isfinite(long_ret) and np.isfinite(short_ret)):
            continue

        bac_ret = long_ret - w_short * short_ret

        cur_long, cur_short = set(long_tk), set(short_tk)
        if prev_long or prev_short:
            denom = max(len(cur_long) + len(cur_short), 1)
            changed = len(cur_long ^ prev_long) + len(cur_short ^ prev_short)
            turnover = changed / (2.0 * denom)
        else:
            turnover = 1.0
        prev_long, prev_short = cur_long, cur_short

        rows.append({
            "date": rebal_dt,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "avg_corr_long": float(corr_snap.reindex(long_tk).dropna().mean()),
            "avg_corr_short": float(corr_snap.reindex(short_tk).dropna().mean()),
            "avg_beta_long": b_long,
            "avg_beta_short": b_short,
            "w_short": w_short,
            "bac_ret": bac_ret,
            "turnover": turnover,
            "mkt_ret": mkt_ret,
            "n": int(len(sig_snap)),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")


def apply_costs(
    book: pd.DataFrame,
    cost_bps: float = 5.0,
    borrow_ann_bps: float = 50.0,
    ret_col: str = "bac_ret",
) -> pd.Series:
    """Net the gross book for one-way turnover x NAV costs + an annual short borrow.

    Each rebalance trades both legs, so the realised round-trip cost is
    ``2 * cost_bps * turnover`` (the long leg plus the beta-scaled short leg). The short borrow
    is charged on the short notional ``w_short`` at ``borrow_ann_bps`` annualised, pro-rated
    monthly. Returns the net monthly series.
    """
    gross = book[ret_col].astype(float)
    cost = 2.0 * cost_bps * 1e-4 * book["turnover"].astype(float)
    borrow = book["w_short"].astype(float) * borrow_ann_bps * 1e-4 / MONTHS_PER_YEAR
    return (gross - cost - borrow).rename("bac_net")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def hac_tstat(r: pd.Series) -> float:
    """Newey-West (HAC) t-stat on the mean of a monthly return series.

    Lag rule ``floor(4*(n/100)^(2/9))`` -- the desk standard.
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


def block_bootstrap_ci(
    r: pd.Series, block: int = 6, n_boot: int = 2000, alpha: float = 0.05, seed: int = 502
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the annualised mean of a monthly series."""
    x = pd.Series(r).dropna().to_numpy(dtype=float)
    n = x.size
    if n < block + 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[b] = x[idx[:n]].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo * MONTHS_PER_YEAR), float(hi * MONTHS_PER_YEAR))


def summary(monthly_returns: pd.Series) -> dict:
    """Annualised stats for a monthly book: mean, vol, Sharpe, HAC t, hit-rate, max-DD, n."""
    r = pd.Series(monthly_returns).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: float("nan") for k in
                ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n")}
    mean_ann = float(r.mean() * MONTHS_PER_YEAR)
    vol_ann = float(r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))
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
    prices: pd.DataFrame,
    spy_prices: pd.Series,
    n_shuffles: int = 200,
    seed: int = 502,
    **kw,
) -> dict:
    """Label-shuffle placebo: how often does a shuffled-correlation book beat the real one?

    The real BAC book's mean monthly return is compared against the distribution of mean
    returns from ``n_shuffles`` books built on **permuted** correlation ranks (the signal is
    destroyed but the panel, beta-neutralisation, costs and lag are identical). The one-sided
    p-value is the fraction of placebo means >= the real mean. A real signal sits in the right
    tail (small p); noise sits in the bulk.
    """
    real = bac_portfolio(prices, spy_prices, **kw)
    if real.empty:
        return {"real_mean": float("nan"), "p_value": float("nan"), "n_shuffles": 0}
    real_mean = float(real["bac_ret"].mean())

    rng = np.random.default_rng(seed)
    placebo_means = []
    for _ in range(n_shuffles):
        s = int(rng.integers(0, 2**31 - 1))
        pb = bac_portfolio(prices, spy_prices, shuffle_seed=s, **kw)
        if not pb.empty:
            placebo_means.append(float(pb["bac_ret"].mean()))
    placebo_means = np.asarray(placebo_means)
    if placebo_means.size == 0:
        return {"real_mean": real_mean, "p_value": float("nan"), "n_shuffles": 0}
    p = float((placebo_means >= real_mean).mean())
    return {
        "real_mean": real_mean,
        "p_value": p,
        "n_shuffles": int(placebo_means.size),
        "placebo_mean": float(placebo_means.mean()),
        "placebo_std": float(placebo_means.std(ddof=1)) if placebo_means.size > 1 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Synthetic positive control -- seed-robust (averaged over many seeds)
# ---------------------------------------------------------------------------
def synthetic_control_curve(
    premiums=(0.0, 0.02, 0.04, 0.06, 0.08, 0.10),
    n_seeds: int = 20,
    base_seed: int = 502,
) -> pd.DataFrame:
    """Seed-robust positive control: mean HAC t of the BAC book vs the planted premium.

    For each planted ``bac_premium`` we generate ``n_seeds`` independent synthetic panels and
    average the beta-neutral book's HAC t. A faithful engine sits at t~0 on the null
    (premium=0) and climbs through t=2 as the planted premium grows -- and because the t is
    averaged over >=20 seeds, no single lucky RNG draw can manufacture the result (the house
    rule for any synthetic-dependent claim).
    """
    from . import data as _data

    rows = []
    for prem in premiums:
        ts = []
        for k in range(n_seeds):
            stock_rets, mkt, _ = _data.synthetic_panel(
                bac_premium=prem, seed=base_seed + k
            )
            # synthetic_panel emits returns directly; reconstruct price-like frames.
            prices = (1.0 + stock_rets).cumprod()
            spy = (1.0 + mkt).cumprod()
            book = bac_portfolio(prices, spy, min_stocks=8)
            if not book.empty:
                ts.append(hac_tstat(book["bac_ret"]))
        ts = np.asarray([t for t in ts if np.isfinite(t)])
        rows.append({
            "premium": prem,
            "mean_t": float(ts.mean()) if ts.size else float("nan"),
            "std_t": float(ts.std(ddof=1)) if ts.size > 1 else float("nan"),
            "n_seeds": int(ts.size),
        })
    return pd.DataFrame(rows).set_index("premium")
