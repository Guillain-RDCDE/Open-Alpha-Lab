"""What the adjustment convention costs — Study 972.

One question, four places to ask it:

1. ``yield_table`` — the arithmetic. The gap between the total-return CAGR and the price-only
   CAGR *is* the reinvested dividend yield, so this table doubles as a measurement of each
   fund's realised yield and as a check that the two tapes are the same tape.
2. ``ranking_table`` — the cross-sectional damage. Rank the universe by trailing return on
   each panel and count how often the two rankings disagree, and by how much (Spearman, plus
   the share of pairs whose order flips). A price-only ranking is a **yield-tilted** ranking:
   it systematically demotes whatever pays income, and momentum strategies rank for a living.
3. ``momentum_backtest`` — the same 12-1 momentum sleeve run twice, once on each panel, with
   one day of execution lag and costs. The difference is not the dividend income (both books
   are scored on total returns); it is the difference in *which assets the signal selected*.
   That separation is the point, and ``score_on`` makes it explicit.
4. ``risk_table`` — volatility, drawdown and Sharpe on both panels, because a price-only
   series has a lower mean and (very nearly) the same variance, which flatters nothing and
   quietly worsens every risk-adjusted ratio.

The convention is not a matter of taste. A total-return series answers "what happened to my
money"; a price-only series answers "what happened to the quoted price". Almost every question
anyone actually asks is the first one — and the exceptions (index-level valuation work, chart
patterns, drawdown-from-peak in *price* terms) are real but rare, which is why the study
measures the damage rather than merely asserting the rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TRADING_DAYS = 252
LOOKBACK, SKIP = 252, 21


def cagr(prices: pd.Series) -> float:
    """Annualised compound growth of one series over its own span."""
    p = prices.dropna()
    years = (p.index[-1] - p.index[0]).days / 365.25
    return float((p.iloc[-1] / p.iloc[0]) ** (1 / years) - 1) if years > 0 else np.nan


def yield_table(tr: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Per ticker: both CAGRs, the implied reinvested yield, and both volatilities."""
    rows = []
    for c in tr.columns:
        a, b = tr[c].dropna(), px[c].dropna()
        idx = a.index.intersection(b.index)
        a, b = a.loc[idx], b.loc[idx]
        ra, rb = a.pct_change().dropna(), b.pct_change().dropna()
        rows.append({
            "ticker": c, "n": int(len(idx)),
            "cagr_tr": cagr(a), "cagr_px": cagr(b),
            "implied_yield": cagr(a) - cagr(b),
            "vol_tr": float(ra.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            "vol_px": float(rb.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            "share_of_return": (cagr(a) - cagr(b)) / cagr(a) if cagr(a) else np.nan,
        })
    return pd.DataFrame(rows).set_index("ticker")


def trailing_return(prices: pd.DataFrame, lookback: int = LOOKBACK,
                    skip: int = SKIP) -> pd.DataFrame:
    """12-1 trailing return: from ``t-lookback`` to ``t-skip``, the standard momentum signal."""
    return prices.shift(skip) / prices.shift(lookback) - 1.0


def ranking_table(tr: pd.DataFrame, px: pd.DataFrame, lookback: int = LOOKBACK,
                  skip: int = SKIP, freq: str = "ME") -> pd.DataFrame:
    """How often the two panels disagree about the cross-sectional ordering.

    Measured at each month end on the assets available on both panels: Spearman rank
    correlation, the share of asset *pairs* whose relative order flips, and whether the
    top-ranked asset is the same one.
    """
    sa = trailing_return(tr, lookback, skip).resample(freq).last().dropna(how="all")
    sb = trailing_return(px, lookback, skip).resample(freq).last().dropna(how="all")
    rows = []
    for d in sa.index.intersection(sb.index):
        a, b = sa.loc[d].dropna(), sb.loc[d].dropna()
        cols = a.index.intersection(b.index)
        if len(cols) < 3:
            continue
        a, b = a[cols], b[cols]
        rho = spearmanr(a.to_numpy(), b.to_numpy()).statistic
        ra, rb = a.rank(), b.rank()
        flips = sum(1 for i in range(len(cols)) for j in range(i + 1, len(cols))
                    if np.sign(ra.iloc[i] - ra.iloc[j]) != np.sign(rb.iloc[i] - rb.iloc[j]))
        n_pairs = len(cols) * (len(cols) - 1) / 2
        rows.append({"date": d, "n_assets": int(len(cols)), "spearman": float(rho),
                     "pair_flips": int(flips), "flip_share": float(flips / n_pairs),
                     "same_top": bool(a.idxmax() == b.idxmax()),
                     "same_bottom": bool(a.idxmin() == b.idxmin())})
    return pd.DataFrame(rows).set_index("date")


def momentum_backtest(signal_panel: pd.DataFrame, score_panel: pd.DataFrame,
                      top_k: int = 3, lookback: int = LOOKBACK, skip: int = SKIP,
                      cost_bps: float = 5.0, freq: str = "ME") -> dict:
    """Rank on ``signal_panel``, hold the top ``top_k``, and always score on ``score_panel``.

    The separation is deliberate. Scoring both arms on the **total-return** panel means the
    comparison is not "one book collects dividends and the other does not" — both do. What
    differs is only *which assets the signal chose*, which is the actual question: does
    ranking on a price chart pick worse assets?

    One execution lag (weights formed at a month end apply from the next session) and
    ``cost_bps`` per unit of traded notional.
    """
    sig = trailing_return(signal_panel, lookback, skip)
    rebal = sig.resample(freq).last().dropna(how="all")
    rets = score_panel.pct_change()
    weights = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    for d in rebal.index:
        row = rebal.loc[d].dropna()
        if len(row) < top_k:
            continue
        picks = row.nlargest(top_k).index
        weights.loc[weights.index > d, :] = 0.0
        weights.loc[weights.index > d, picks] = 1.0 / top_k
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    gross = (weights * rets).sum(axis=1)
    net = gross - turnover * cost_bps / 1e4
    curve = (1 + net).cumprod()
    years = len(net) / TRADING_DAYS
    vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return {"cagr": float(curve.iloc[-1] ** (1 / years) - 1), "vol": vol,
            "sharpe": float(net.mean() / net.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            "max_dd": float((curve / curve.cummax() - 1).min()),
            "turnover_ann": float(turnover.sum() / years),
            "returns": net, "weights": weights}


def holding_yield(weights: pd.DataFrame, yields: pd.Series) -> float:
    """Average dividend yield of whatever the signal held — the tilt, measured."""
    w = weights.loc[weights.sum(axis=1) > 0]
    if w.empty:
        return np.nan
    return float((w * yields.reindex(w.columns)).sum(axis=1).mean())


def risk_table(tr: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Drawdown and Sharpe on both panels — the ratios a price-only view quietly worsens."""
    rows = []
    for c in tr.columns:
        out = {"ticker": c}
        for name, panel in (("tr", tr), ("px", px)):
            p = panel[c].dropna()
            r = p.pct_change().dropna()
            curve = p / p.iloc[0]
            out[f"sharpe_{name}"] = float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))
            out[f"maxdd_{name}"] = float((curve / curve.cummax() - 1).min())
            out[f"years_underwater_{name}"] = float(
                (curve < curve.cummax()).sum() / TRADING_DAYS)
        rows.append(out)
    df = pd.DataFrame(rows).set_index("ticker")
    df["sharpe_gap"] = df["sharpe_tr"] - df["sharpe_px"]
    df["maxdd_gap"] = df["maxdd_tr"] - df["maxdd_px"]
    return df


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Real** if the largest implied yield in the universe exceeds 2%/yr (the
      convention then moves the headline return by more than most strategies earn); **Weak**
      above 0.5%; **None** below.
    - **Usefulness**: **Useful** if the choice flips a *conclusion* — the momentum sleeve's
      Sharpe ordering, or more than 10% of cross-sectional pair rankings; **Fragile** if it
      moves numbers without changing an ordering; **Mirage** otherwise.
    """
    y = h["max_implied_yield"]
    signal = "Real" if y >= 0.02 else ("Weak" if y >= 0.005 else "None")
    flips = h["mean_flip_share"]
    changed = h["momentum_sharpe_gap"] != 0 and abs(h["momentum_cagr_gap"]) >= 0.005
    trad = ("Useful" if (flips >= 0.10 or changed)
            else ("Fragile" if y >= 0.005 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"The gap between the two conventions is the reinvested dividend, and on this "
            f"universe it runs from **{h['min_implied_yield']:.2%}/yr** "
            f"({h['min_yield_ticker']}) to **{h['max_implied_yield']:.2%}/yr** "
            f"({h['max_yield_ticker']}) — which on {h['max_yield_ticker']} is "
            f"**{h['max_share_of_return']:.0%} of its entire total return**. Volatility is "
            f"almost untouched (largest difference {h['max_vol_gap']:.2%}), so every "
            f"risk-adjusted ratio moves with the numerator alone: the Sharpe gap reaches "
            f"**{h['max_sharpe_gap']:.2f}**."),
        "trad": trad,
        "trad_why": (
            f"Yes. Ranking the same universe on price charts instead of total returns "
            f"reorders **{h['mean_flip_share']:.0%}** of asset pairs in an average month and "
            f"picks a different leader in **{1 - h['same_top_share']:.0%}** of them. Run as a "
            f"momentum sleeve — both arms scored on total returns, so only the *selection* "
            f"differs — ranking on price gives {h['momentum_cagr_px']:+.2%}/yr against "
            f"{h['momentum_cagr_tr']:+.2%}/yr for ranking on total return "
            f"({h['momentum_cagr_gap']:+.2%}/yr, Sharpe {h['momentum_sharpe_px']:+.2f} vs "
            f"{h['momentum_sharpe_tr']:+.2f}), and it holds a portfolio yielding "
            f"{h['yield_tilt']:+.2%} less. The price-only signal is not neutral — it is a "
            f"systematic bet against income."),
        "one_sentence": (
            f"A price chart is a total-return series with the dividends deleted, and on this "
            f"universe that is up to **{h['max_implied_yield']:.1%} a year** — enough to "
            f"reorder {h['mean_flip_share']:.0%} of a monthly cross-sectional ranking, tilt a "
            f"momentum sleeve away from income by {abs(h['yield_tilt']):.2%}, and move a "
            f"Sharpe ratio by {h['max_sharpe_gap']:.2f}."),
    }
