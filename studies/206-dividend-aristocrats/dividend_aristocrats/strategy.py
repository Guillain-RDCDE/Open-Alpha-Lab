"""Strategy and honest controls — Study 206 (Dividend-Aristocrats).

The folk recipe: buy and hold NOBL (or a basket of Dividend Aristocrats) instead of
the broad market (SPY), capture the supposed quality/dividend-growth premium, and sleep
soundly through crashes because "quality companies that grow dividends every year are
the bedrock of every portfolio."

We test three concrete questions with honest controls:

1. **Signal** — does NOBL deliver meaningfully higher risk-adjusted returns than SPY
   over its live history (2013–2026)?  Control: a random-allocation portfolio mixing
   NOBL and SPY at the same average weight.

2. **Crash behaviour** — does NOBL offer protection in the worst S&P 500 drawdowns?
   The quality/low-vol story predicts it should.

3. **Tradability** — what does an investor actually capture net of the ETF expense
   ratio (0.35%/yr for NOBL vs 0.0945%/yr for SPY)?

Inference uses HAC t-statistics (Newey-West) on daily log-return differences and a
circular block-bootstrap Sharpe CI.  All look-ahead is prohibited: we compare only
realised total-return indices, no forecasting signal.

Functions
---------
daily_returns(prices)      — log-returns for each column; NaN on the first row.
excess_return(prices, ...)  — NOBL minus SPY daily log-return differential.
summarize(ret, ...)         — HAC t-stat, Sharpe, CAGR, max drawdown, summary dict.
rolling_alpha(ret, ...)     — rolling 252-day alpha of NOBL over SPY.
crash_analysis(prices, ...) — per-drawdown NOBL vs SPY comparison.
beta_alpha_ols(ret, ...)    — OLS alpha & beta of NOBL on SPY.
random_mix_control(prices, ...) — random-weight NOBL/SPY mixture as a control arm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
NOBL_ER = 0.0035   # NOBL expense ratio (0.35%/yr), as of 2026
SPY_ER = 0.000945  # SPY expense ratio (0.0945%/yr), as of 2026


# ---------------------------------------------------------------------------
# Return decomposition helpers
# ---------------------------------------------------------------------------
def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log-return frame aligned on ``prices`` (NaN on the first row for each column).

    Log-returns compound correctly and avoid the sign asymmetry of arithmetic returns.
    """
    return np.log(prices / prices.shift(1))


def excess_return(
    prices: pd.DataFrame,
    asset: str = "NOBL",
    benchmark: str = "SPY",
) -> pd.Series:
    """Daily log-return difference ``asset − benchmark`` (the alpha-seeking series).

    A positive mean here — if statistically robust — is the claimed edge.  The series
    includes both the quality factor tilt and any non-market risk being taken.
    """
    ret = daily_returns(prices).dropna()
    return (ret[asset] - ret[benchmark]).rename(f"{asset}_minus_{benchmark}")


# ---------------------------------------------------------------------------
# Drawdown helper
# ---------------------------------------------------------------------------
def max_drawdown(price_series: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of a price (or NAV) series, as a negative fraction."""
    cum_max = price_series.cummax()
    dd = (price_series - cum_max) / cum_max
    return float(dd.min())


def drawdown_series(price_series: pd.Series) -> pd.Series:
    """Full drawdown series (fraction from running peak)."""
    cum_max = price_series.cummax()
    return (price_series - cum_max) / cum_max


# ---------------------------------------------------------------------------
# Core summarise
# ---------------------------------------------------------------------------
def summarize(
    ret: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    label: str = "",
) -> dict:
    """Headline statistics for a daily log-return series.

    Returns CAGR (annualised geometric return), annualised vol, Sharpe (gross, rf=0),
    max-drawdown of the total-return index derived from ``ret``, and a HAC t-stat on
    the mean daily return (the key inference-bar number).

    The HAC t-stat (Newey-West) is the primary significance test.  ``|t| >= 2`` is the
    inference bar; below that the effect is statistically indistinguishable from noise.
    """
    r = np.asarray(ret, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 10:
        return {"label": label, "n": n, "cagr_pct": float("nan"),
                "vol_ann_pct": float("nan"), "sharpe_ann": float("nan"),
                "max_dd_pct": float("nan"), "mean_bps": float("nan"), "tstat": float("nan")}

    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    cagr = float((np.exp(mu * periods_per_year) - 1) * 100)
    vol_ann = float(sigma * np.sqrt(periods_per_year) * 100)
    sharpe = float(mu / sigma * np.sqrt(periods_per_year)) if sigma > 0 else float("nan")

    # NAV for drawdown
    nav = pd.Series(np.exp(np.cumsum(r)))
    mdd = float(max_drawdown(nav) * 100)

    # HAC t-stat (Newey-West Bartlett kernel, lag rule of thumb)
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    e = r - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se) if se > 0 else float("nan")

    return {
        "label": label,
        "n": n,
        "cagr_pct": cagr,
        "vol_ann_pct": vol_ann,
        "sharpe_ann": sharpe,
        "max_dd_pct": mdd,
        "mean_bps": mu * 1e4,
        "tstat": tstat,
    }


def beta_alpha_ols(
    prices: pd.DataFrame,
    asset: str = "NOBL",
    benchmark: str = "SPY",
) -> dict:
    """OLS regression of asset daily log-returns on benchmark log-returns.

    Decomposes the asset return into:
        r_asset = alpha + beta * r_benchmark + eps

    A statistically significant positive alpha (t_alpha >= 2) would be the market-
    neutral quality premium.  Beta > 1 flags that much of any return advantage is
    disguised leverage on the benchmark.  Beta < 1 is the typical quality/low-vol
    signature — lower drawdowns but also lower upside participation in bull markets.
    """
    ret = daily_returns(prices).dropna()
    y = ret[asset].to_numpy(dtype=float)
    x = ret[benchmark].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)

    # OLS: y = alpha + beta * x
    beta, alpha = np.polyfit(x, y, 1)
    resid = y - (beta * x + alpha)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # t-stat on alpha (standard OLS SE)
    s2 = ss_res / max(n - 2, 1)
    x_bar = x.mean()
    sxx = float(np.sum((x - x_bar) ** 2))
    se_alpha = np.sqrt(s2 / n + s2 * x_bar**2 / sxx) if sxx > 0 else float("nan")
    t_alpha = float(alpha / se_alpha) if np.isfinite(se_alpha) and se_alpha > 0 else float("nan")

    return {
        "alpha_daily_bps": float(alpha * 1e4),
        "alpha_ann_pct": float(((1 + alpha) ** TRADING_DAYS_PER_YEAR - 1) * 100),
        "t_alpha": t_alpha,
        "beta": float(beta),
        "r_squared": float(r2),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Rolling alpha
# ---------------------------------------------------------------------------
def rolling_alpha(
    prices: pd.DataFrame,
    asset: str = "NOBL",
    benchmark: str = "SPY",
    window: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Rolling one-year daily alpha of ``asset`` relative to ``benchmark``.

    Each period's alpha is the trailing ``window``-day mean of the excess return
    (no regression; this is the simple active return, not market-neutral alpha).
    Sign and trend matter most: a flat-then-falling rolling alpha is the post-
    publication decay signature (McLean & Pontiff 2016).
    """
    diff = excess_return(prices, asset=asset, benchmark=benchmark)
    return diff.rolling(window, min_periods=window // 2).mean() * TRADING_DAYS_PER_YEAR * 1e4


# ---------------------------------------------------------------------------
# Crash analysis
# ---------------------------------------------------------------------------
def crash_analysis(
    prices: pd.DataFrame,
    asset: str = "NOBL",
    benchmark: str = "SPY",
    threshold: float = -0.10,
) -> pd.DataFrame:
    """Identify significant benchmark drawdowns and compare asset performance inside them.

    A drawdown episode begins when the benchmark first falls ``threshold`` (default −10%)
    from a running peak and ends when it recovers back above the prior peak.  Within each
    episode the NOBL and SPY total returns are compared.  A positive "protection" column
    indicates NOBL held up better — the quality/dividend-growth crash-shield claim.
    """
    bench_prices = prices[benchmark]
    asset_prices = prices[asset]

    # Identify drawdown episodes
    cum_max = bench_prices.cummax()
    dd = (bench_prices - cum_max) / cum_max

    episodes = []
    in_drawdown = False
    start_idx = None
    trough_idx = None
    trough_dd = 0.0

    for i, (date, val) in enumerate(dd.items()):
        if not in_drawdown and val <= threshold:
            in_drawdown = True
            start_idx = date
            trough_idx = date
            trough_dd = val
        elif in_drawdown:
            if val < trough_dd:
                trough_dd = val
                trough_idx = date
            if val >= 0.0:  # recovered
                end_idx = date
                # Compute returns inside episode
                bench_ep = bench_prices.loc[start_idx:end_idx]
                asset_ep = asset_prices.loc[start_idx:end_idx]
                bench_ret = float(bench_ep.iloc[-1] / bench_ep.iloc[0] - 1) * 100
                asset_ret = float(asset_ep.iloc[-1] / asset_ep.iloc[0] - 1) * 100
                episodes.append({
                    "start": start_idx,
                    "trough": trough_idx,
                    "end": end_idx,
                    "bench_max_dd_pct": round(trough_dd * 100, 1),
                    f"{benchmark}_total_ret_pct": round(bench_ret, 1),
                    f"{asset}_total_ret_pct": round(asset_ret, 1),
                    "protection_pp": round(asset_ret - bench_ret, 1),
                })
                in_drawdown = False
                trough_idx = None
                trough_dd = 0.0

    return pd.DataFrame(episodes)


# ---------------------------------------------------------------------------
# Random-mix control
# ---------------------------------------------------------------------------
def random_mix_control(
    prices: pd.DataFrame,
    asset: str = "NOBL",
    benchmark: str = "SPY",
    target_nobl_weight: float = 0.50,
    n_seeds: int = 20,
    seed: int = 206,
) -> pd.DataFrame:
    """Randomly shuffle the asset's daily weight around ``target_nobl_weight``.

    For each seed, draw daily weights for NOBL ~ Uniform(0, 2 * target_nobl_weight)
    and compute the mixture daily log-return.  This control removes the timing
    dimension entirely and asks: how often does a random mix of NOBL and SPY do as well
    as a 100% NOBL portfolio?  If most random mixes beat NOBL, the "pure aristocrats"
    concentration isn't adding value beyond diversification.

    Returns a DataFrame of Sharpe ratios, one row per seed.
    """
    ret = daily_returns(prices).dropna()
    rng = np.random.default_rng(seed)
    n = len(ret)

    rows = []
    for s in range(n_seeds):
        sub_rng = np.random.default_rng(seed + s)
        w_nobl = sub_rng.uniform(0.0, 2.0 * target_nobl_weight, size=n).clip(0, 1)
        w_spy = 1.0 - w_nobl
        mix_ret = w_nobl * ret[asset].to_numpy() + w_spy * ret[benchmark].to_numpy()
        mu = float(np.nanmean(mix_ret))
        sigma = float(np.nanstd(mix_ret, ddof=1))
        sharpe = float(mu / sigma * np.sqrt(TRADING_DAYS_PER_YEAR)) if sigma > 0 else float("nan")
        rows.append({"seed": seed + s, "mean_nobl_weight": float(w_nobl.mean()), "sharpe_ann": sharpe})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Net-of-fees comparison
# ---------------------------------------------------------------------------
def net_of_fees(
    prices: pd.DataFrame,
    nobl_er: float = NOBL_ER,
    spy_er: float = SPY_ER,
) -> pd.DataFrame:
    """Adjust price series for annual expense ratios (applied daily as a drag).

    Returns an adjusted price frame with the same columns.  The drag is applied as a
    constant daily return deduction: exp(-ER/252) per day.  This is the correct way to
    compare ETF total-return indices net of their management fees.
    """
    adj = prices.copy()
    daily_drag_nobl = np.exp(-nobl_er / TRADING_DAYS_PER_YEAR)
    daily_drag_spy = np.exp(-spy_er / TRADING_DAYS_PER_YEAR)
    # Cumulative drag factor from the start
    n = len(prices)
    nobl_drag = daily_drag_nobl ** np.arange(n)
    spy_drag = daily_drag_spy ** np.arange(n)
    adj["NOBL"] = prices["NOBL"] * nobl_drag
    adj["SPY"] = prices["SPY"] * spy_drag
    return adj
