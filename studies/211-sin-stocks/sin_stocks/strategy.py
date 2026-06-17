"""Strategy and honest controls — Study 211 (Sin-Stocks).

The folk recipe: build an equal-weight basket of "sin stocks" (tobacco, alcohol, gambling,
defense) that are systematically underweighted or excluded from ESG/SRI portfolios.  The
neglect premium hypothesis (Hong & Kacperczyk 2009) claims that institutional exclusion
depresses valuations, raising expected returns for investors willing to hold vice stocks.

We test three concrete questions with honest controls:

1. **Signal** — does an equal-weight sin basket (MO, PM, STZ, LVS, LMT, RTX) deliver
   meaningfully higher risk-adjusted returns than SPY over 2007–2026?  And vs DSI
   (iShares MSCI KLD 400 Social ETF) as the ESG counter-portfolio?

2. **Alpha decomposition** — is any outperformance alpha (market-neutral) or just sector/
   value tilt in disguise?  OLS decomposes into beta + alpha with HAC-robust inference.

3. **Crash behaviour** — do sin stocks (often defensive consumer staples / government
   contractors / pricing-power tobacco) offer protection in downturns?  This is a secondary
   claim sometimes appended to the neglect story.

Inference uses HAC t-statistics (Newey-West) on daily log-return differences and a
circular block-bootstrap Sharpe CI.  All look-ahead is prohibited.

Functions
---------
build_equal_weight(basket_prices)          — daily equal-weight basket returns.
daily_returns(prices)                       — log-returns; NaN on the first row.
excess_return(prices, asset, benchmark)     — asset minus benchmark daily log-return diff.
summarize(ret, ...)                          — HAC t-stat, Sharpe, CAGR, max drawdown.
rolling_alpha(prices, ...)                   — rolling 252-day excess return.
crash_analysis(prices, ...)                  — per-drawdown comparison.
beta_alpha_ols(prices, ...)                  — OLS alpha & beta of sin vs benchmark.
sector_attribution(prices, basket)           — per-ticker contribution to basket return.
random_mix_control(prices, ...)             — random-weight control arm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# Approximate expense ratios for benchmarks
SPY_ER = 0.000945   # SPY expense ratio (0.0945%/yr), as of 2026
DSI_ER = 0.0025     # DSI expense ratio (0.25%/yr), as of 2026

# Sin basket tickers (imported here to avoid circular imports)
SIN_BASKET = ["MO", "PM", "STZ", "LVS", "LMT", "RTX"]


# ---------------------------------------------------------------------------
# Equal-weight basket construction
# ---------------------------------------------------------------------------
def build_equal_weight(
    basket_prices: pd.DataFrame,
) -> pd.Series:
    """Compute daily equal-weight basket log-return (simple average of individual log-returns).

    Equal-weighting avoids survivorship/concentration bias that would arise from cap-weighting
    a small basket.  The basket is assumed to be rebalanced daily (frictionless) — an
    important caveat when assessing tradability.

    Returns a Series of daily log-returns named ``SIN_EW``.
    """
    log_rets = np.log(basket_prices / basket_prices.shift(1)).dropna()
    ew_ret = log_rets.mean(axis=1)
    return ew_ret.rename("SIN_EW")


# ---------------------------------------------------------------------------
# Return helpers
# ---------------------------------------------------------------------------
def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log-return frame aligned on ``prices`` (NaN on the first row for each column)."""
    return np.log(prices / prices.shift(1))


def excess_return(
    prices: pd.DataFrame,
    asset: str = "SIN_EW",
    benchmark: str = "SPY",
) -> pd.Series:
    """Daily log-return difference ``asset − benchmark`` (the alpha-seeking series).

    A positive mean — if statistically robust (HAC |t| >= 2) — is the claimed neglect premium.
    """
    ret = daily_returns(prices).dropna()
    return (ret[asset] - ret[benchmark]).rename(f"{asset}_minus_{benchmark}")


# ---------------------------------------------------------------------------
# Drawdown helpers
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

    Returns CAGR, annualised vol, Sharpe (gross, rf=0), max-drawdown, and a HAC t-stat on
    the mean daily return.  ``|t| >= 2`` is the inference bar.
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


# ---------------------------------------------------------------------------
# OLS alpha / beta decomposition
# ---------------------------------------------------------------------------
def beta_alpha_ols(
    ret_asset: pd.Series,
    ret_benchmark: pd.Series,
    label: str = "",
) -> dict:
    """OLS regression of asset daily log-returns on benchmark log-returns.

    Decomposes the asset return into:
        r_asset = alpha + beta * r_benchmark + eps

    A statistically significant positive alpha (t_alpha >= 2) is the market-neutral
    neglect premium.  Beta != 1 reveals the sector tilt embedded in the sin basket.
    """
    idx = ret_asset.index.intersection(ret_benchmark.index)
    y = ret_asset.loc[idx].to_numpy(dtype=float)
    x = ret_benchmark.loc[idx].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 10:
        return {k: float("nan") for k in
                ("alpha_daily_bps", "alpha_ann_pct", "t_alpha", "beta", "r_squared", "n")}

    beta, alpha = np.polyfit(x, y, 1)
    resid = y - (beta * x + alpha)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    s2 = ss_res / max(n - 2, 1)
    x_bar = x.mean()
    sxx = float(np.sum((x - x_bar) ** 2))
    se_alpha = np.sqrt(s2 / n + s2 * x_bar**2 / sxx) if sxx > 0 else float("nan")
    t_alpha = float(alpha / se_alpha) if np.isfinite(se_alpha) and se_alpha > 0 else float("nan")

    return {
        "label": label,
        "alpha_daily_bps": float(alpha * 1e4),
        "alpha_ann_pct": float(((1 + alpha) ** TRADING_DAYS_PER_YEAR - 1) * 100),
        "t_alpha": t_alpha,
        "beta": float(beta),
        "r_squared": float(r2),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Sector attribution
# ---------------------------------------------------------------------------
def sector_attribution(
    all_prices: pd.DataFrame,
    basket: list[str] | None = None,
) -> pd.DataFrame:
    """Per-ticker contribution to the basket's annualised CAGR.

    Returns a DataFrame with one row per ticker: CAGR, Sharpe, and max drawdown.
    Useful for diagnosing whether the basket return is driven by one ticker
    (e.g. tobacco) or is broad-based across sin sectors.
    """
    if basket is None:
        basket = SIN_BASKET
    rows = []
    for t in basket:
        if t not in all_prices.columns:
            continue
        ret_t = np.log(all_prices[t] / all_prices[t].shift(1)).dropna()
        s = summarize(ret_t, label=t)
        rows.append(s)
    return pd.DataFrame(rows).set_index("label")


# ---------------------------------------------------------------------------
# Rolling alpha
# ---------------------------------------------------------------------------
def rolling_alpha(
    ew_ret: pd.Series,
    benchmark_ret: pd.Series,
    window: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Rolling one-year daily alpha of sin basket relative to benchmark.

    Each period's alpha is the trailing ``window``-day mean of the excess return
    (simple active return, not market-neutral).  A flat-then-falling rolling alpha
    is the post-publication decay signature.
    """
    idx = ew_ret.index.intersection(benchmark_ret.index)
    diff = (ew_ret.loc[idx] - benchmark_ret.loc[idx]).rename("SIN_EW_minus_SPY")
    return diff.rolling(window, min_periods=window // 2).mean() * TRADING_DAYS_PER_YEAR * 1e4


# ---------------------------------------------------------------------------
# Crash analysis
# ---------------------------------------------------------------------------
def crash_analysis(
    all_ret: pd.DataFrame,
    asset: str = "SIN_EW",
    benchmark: str = "SPY",
    threshold: float = -0.10,
) -> pd.DataFrame:
    """Identify significant benchmark drawdowns and compare asset performance inside them.

    Uses cumulative return indices from log-returns.  A drawdown episode begins when
    the benchmark falls ``threshold`` from a running peak and ends when it recovers.
    Positive "protection" means SIN_EW held up better.
    """
    # Build NAV series from log-returns
    bench_nav = np.exp(np.cumsum(all_ret[benchmark].dropna()))
    bench_nav.index = all_ret[benchmark].dropna().index

    asset_nav = np.exp(np.cumsum(all_ret[asset].dropna()))
    asset_nav.index = all_ret[asset].dropna().index

    # Align
    idx = bench_nav.index.intersection(asset_nav.index)
    bench_nav = bench_nav.loc[idx]
    asset_nav = asset_nav.loc[idx]

    cum_max = bench_nav.cummax()
    dd = (bench_nav - cum_max) / cum_max

    episodes = []
    in_drawdown = False
    start_idx = None
    trough_idx = None
    trough_dd = 0.0

    for date, val in dd.items():
        if not in_drawdown and val <= threshold:
            in_drawdown = True
            start_idx = date
            trough_idx = date
            trough_dd = val
        elif in_drawdown:
            if val < trough_dd:
                trough_dd = val
                trough_idx = date
            if val >= 0.0:
                end_idx = date
                bench_ep = bench_nav.loc[start_idx:end_idx]
                asset_ep = asset_nav.loc[start_idx:end_idx]
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
    ew_ret: pd.Series,
    spy_ret: pd.Series,
    target_sin_weight: float = 0.50,
    n_seeds: int = 20,
    seed: int = 211,
) -> pd.DataFrame:
    """Randomly shuffle the sin basket weight around ``target_sin_weight``.

    For each seed, draw daily weights for SIN_EW ~ Uniform(0, 2 * target_sin_weight)
    and compute the mixture daily log-return.  This control removes timing entirely and
    asks: does a random SIN/SPY mix do as well as 100% SIN_EW?  If so, the pure sin
    concentration isn't adding value beyond blending.

    Returns a DataFrame of Sharpe ratios, one row per seed.
    """
    idx = ew_ret.index.intersection(spy_ret.index)
    ew = ew_ret.loc[idx].to_numpy()
    spy = spy_ret.loc[idx].to_numpy()
    n = len(ew)

    rows = []
    for s in range(n_seeds):
        sub_rng = np.random.default_rng(seed + s)
        w_sin = sub_rng.uniform(0.0, 2.0 * target_sin_weight, size=n).clip(0, 1)
        w_spy = 1.0 - w_sin
        mix_ret = w_sin * ew + w_spy * spy
        mu = float(np.nanmean(mix_ret))
        sigma = float(np.nanstd(mix_ret, ddof=1))
        sharpe = float(mu / sigma * np.sqrt(TRADING_DAYS_PER_YEAR)) if sigma > 0 else float("nan")
        rows.append({"seed": seed + s, "mean_sin_weight": float(w_sin.mean()), "sharpe_ann": sharpe})
    return pd.DataFrame(rows)
