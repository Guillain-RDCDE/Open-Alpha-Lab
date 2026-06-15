"""Strategy and honest controls — Study 201 (Dividend-Growth).

The claim: stocks that have *consistently raised* their dividends for several consecutive
years (dividend growers) earn superior forward returns vs (a) an equal-weight basket of the
same names and (b) high-yield non-growers in the same universe.

Three arms tested each year:

- **Growers** — names with a consecutive-raises streak >= ``streak_years`` (default 3).
  Equal-weight, rebalanced annually at year-end.  Long only.
- **Equal-weight baseline** — all names in the basket, equal-weight.  This is the
  unconditional return; any grower premium is measured against it.
- **High-yield non-growers** — names *not* in the grower bucket, ranked by their trailing
  dividend yield (proxied here by the dividend-sum / end-of-year price), and sorted to the
  top quintile.  This tests whether the outperformance the literature associates with
  dividends is driven by the *growth* or by the *yield* (the Dogs-of-the-Dow insight).

No look-ahead: the streak for year *y* is built from dividend history through December *y*;
the forward return is the calendar year *y+1* total return — information available only after
year-end rebalance.

Inference uses quantlab HAC t-stats (Newey-West) for the annual return series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal classification
# ---------------------------------------------------------------------------
def classify_growers(
    streaks: pd.DataFrame,
    streak_years: int = 3,
) -> pd.DataFrame:
    """Boolean mask (year x ticker): True if ticker is a dividend grower in that year.

    A ticker is a "grower" in year *y* if its consecutive-raises streak (number of
    consecutive prior years with strictly higher dividends) is >= ``streak_years``.
    This is a backward-looking screen: we need only information available by end-of-year *y*.
    """
    return (streaks >= streak_years).astype(bool)


def classify_high_yield_nongrowers(
    streaks: pd.DataFrame,
    div_yield: pd.DataFrame | None,
    streak_years: int = 3,
    yield_top_q: float = 0.20,
) -> pd.DataFrame:
    """Boolean mask (year x ticker): True if ticker is a high-yield non-grower.

    Non-growers are all tickers NOT in the grower bucket.  Among those, the top-quintile
    (by trailing div yield) in each year form the comparator "Dogs" arm.  If ``div_yield``
    is None (synthetic data without yield), falls back to picking the bottom-streak
    non-growers as an approximate comparator.
    """
    growers = classify_growers(streaks, streak_years)
    not_grower = ~growers
    mask = pd.DataFrame(False, index=streaks.index, columns=streaks.columns)

    for y in streaks.index:
        non_g = not_grower.loc[y]
        candidates = non_g[non_g].index.tolist()
        if not candidates:
            continue
        if div_yield is not None and y in div_yield.index:
            ylds = div_yield.loc[y, candidates].dropna()
            if len(ylds) > 0:
                thresh = ylds.quantile(1.0 - yield_top_q)
                selected = ylds[ylds >= thresh].index
                mask.loc[y, selected] = True
                continue
        # Fallback: pick bottom-streak non-growers (streaks 0–1 — never raised)
        s = streaks.loc[y, candidates].fillna(0)
        thresh = s.quantile(yield_top_q)
        selected = s[s <= thresh].index
        mask.loc[y, selected] = True

    return mask


# ---------------------------------------------------------------------------
# Annual return construction
# ---------------------------------------------------------------------------
def arm_returns(
    fwd_ret: pd.DataFrame,
    mask: pd.DataFrame,
    min_names: int = 3,
) -> pd.Series:
    """Equal-weight average next-year return of the names selected by ``mask`` each year.

    Years with fewer than ``min_names`` in the mask are excluded (to avoid noise-dominated
    one-name portfolios, especially in the yield-nongrower fallback path).
    """
    out = {}
    for y in mask.index:
        sel = mask.columns[mask.loc[y]].tolist()
        if len(sel) < min_names or y not in fwd_ret.index:
            continue
        r = fwd_ret.loc[y, sel].dropna()
        if len(r) < min_names:
            continue
        out[y] = r.mean()
    return pd.Series(out, name="ret").sort_index()


# ---------------------------------------------------------------------------
# Hedge / spread
# ---------------------------------------------------------------------------
def hedge_returns(
    grower_ret: pd.Series,
    baseline_ret: pd.Series,
) -> pd.Series:
    """Year-by-year grower return minus equal-weight baseline return (the alpha spread).

    Aligns on the common years.  A positive mean with a significant HAC t-stat
    would be the evidence needed to claim "growers outperform" vs the baseline.
    """
    aligned = grower_ret.align(baseline_ret, join="inner")
    spread = aligned[0] - aligned[1]
    spread.name = "grower_minus_ew"
    return spread


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summary(returns: pd.Series) -> dict:
    """Headline annual-return statistics with a HAC t-stat (Newey-West).

    Reports mean, vol, Sharpe, HAC t-stat, hit rate (fraction of positive years), max
    drawdown on the compounded equity curve, and n.  The HAC t-stat is the inference-bar
    number: |t| >= 2 is the minimum bar for a real signal; less is noise.
    """
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate",
                                    "max_dd", "n")}
    mean = float(r.mean())
    vol = float(r.std(ddof=1))
    sharpe = float(mean / vol) if vol > 0 else np.nan

    # Newey-West HAC t-stat
    e = r.to_numpy() - mean
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = float(np.sqrt(max(lrv, 0.0) / n))
    tstat = float(mean / se) if se > 0 else np.nan

    eq = (1.0 + r).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())
    return {
        "mean": mean,
        "vol": vol,
        "sharpe": sharpe,
        "tstat": tstat,
        "hit_rate": float((r > 0).mean()),
        "max_dd": max_dd,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Trailing dividend yield (for the real panel)
# ---------------------------------------------------------------------------
def trailing_div_yield(
    div_annual: pd.DataFrame,
    px_year_end: pd.DataFrame,
) -> pd.DataFrame:
    """Trailing-12-month dividend yield at each year-end: div_sum / price.

    Both inputs are (year x ticker) DataFrames.  The yield for year *y* is used to
    classify high-yield non-growers for the year-*y* rebalance (signal available at
    December *y*; forward return is year *y+1*).
    """
    common_tickers = div_annual.columns.intersection(px_year_end.columns)
    common_years = div_annual.index.intersection(px_year_end.index)
    div_aligned = div_annual.loc[common_years, common_tickers]
    px_aligned = px_year_end.loc[common_years, common_tickers]
    yield_df = div_aligned / px_aligned.replace(0, np.nan)
    yield_df.index.name = "year"
    return yield_df
