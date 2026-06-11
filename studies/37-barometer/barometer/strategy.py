"""The books — (a) cross-asset macro-momentum and (b) the inflation-hedge tilt.

Two slow, causal, vol-scaled books built on the *change* in the latent macro state:

  (a) **Macro-momentum book** (:func:`macro_momentum_weights`). For each asset, tilt by the recent
      **trend** in the macro drivers it loads on — long the assets that improving macro momentum
      favours, short the ones it hurts. Concretely, score each asset by the sign-weighted, smoothed
      macro momentum it is exposed to (growth-up favours pro-cyclical assets, inflation-up favours real
      assets), vol-scale so each asset contributes equal risk, and **lag** so the book is tradable.

  (b) **Inflation-hedge tilt** (:func:`inflation_tilt_weights`). A simpler, single-theme overlay: when
      inflation momentum is positive, **overweight real assets** (commodities, TIPS, gold) and underweight
      nominal ones; flip when inflation is falling. This is the Neville et al. (2021) "best strategies for
      inflationary times" idea in miniature — and it should pay *specifically* in rising-inflation regimes
      (the beat-7 regime split tests exactly that).

Everything is monthly; we annualise by ``×12`` (mean) and ``×√12`` (vol). All weights are formed from the
macro state *as of month-end* and earn *next* month's return (a one-month lag), so nothing peeks ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ASSETS, ASSET_ORDER

MONTHS_PER_YEAR = 12


def _resolve_spec(assets: dict | None, order: list | None) -> tuple[dict, list]:
    """Pick the asset spec: the synthetic ``ASSETS``/``ASSET_ORDER`` by default, else the one passed in.

    Lets every book run unchanged on the synthetic control (tests) *and* on the real 18-ETF panel, by
    handing the real ``REAL_ASSETS`` / ``REAL_ASSET_ORDER`` spec through ``assets`` / ``order``.
    """
    spec = ASSETS if assets is None else assets
    return spec, list(spec.keys()) if order is None else order


def summary(returns: pd.Series, periods_per_year: int = MONTHS_PER_YEAR) -> dict:
    """Annualised Sharpe, return, vol, CAGR, max-drawdown, skew for a monthly return series."""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "ann_return", "vol_ann", "cagr",
                                    "max_drawdown", "skew", "n_months")}
    mean, std = r.mean(), r.std(ddof=1)
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    years = len(r) / periods_per_year
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and equity.iloc[-1] > 0 else np.nan
    return {"sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
            "ann_return": float(mean * periods_per_year), "vol_ann": float(std * np.sqrt(periods_per_year)),
            "cagr": float(cagr), "max_drawdown": float(dd.min()), "skew": float(r.skew()),
            "n_months": int(len(r))}


def _macro_momentum(macro: pd.DataFrame, smooth: int = 3) -> pd.DataFrame:
    """The smoothed one-month change (trend) in each macro driver — the raw signal both books use."""
    chg = macro.diff()
    if smooth and smooth > 1:
        chg = chg.rolling(smooth, min_periods=1).mean()
    # z-score each driver over an expanding window so growth and inflation contribute comparably,
    # using only past data (expanding) — keeps the signal causal and scale-free.
    mu = chg.expanding(min_periods=6).mean()
    sd = chg.expanding(min_periods=6).std(ddof=0).replace(0.0, np.nan)
    return ((chg - mu) / sd).fillna(0.0)


def macro_momentum_weights(macro: pd.DataFrame, smooth: int = 3, assets: dict | None = None,
                           order: list | None = None) -> pd.DataFrame:
    """Cross-asset macro-momentum weights, dollar-neutral and gross-normalised, lagged one month.

    Each asset's raw tilt is its macro exposure dotted with the *current* macro momentum::

        score_{a,t} = g_beta_a · zΔgrowth_t + i_beta_a · zΔinflation_t

    so an asset is overweighted when the macro trends it benefits from are improving. Scores are
    cross-sectionally demeaned (dollar-neutral), gross-normalised to 1, and lagged one month. Defaults to
    the synthetic ``ASSETS`` spec; pass ``assets``/``order`` (e.g. ``REAL_ASSETS``) for the real ETF book.
    """
    spec, order = _resolve_spec(assets, order)
    mm = _macro_momentum(macro, smooth=smooth)
    g_beta = np.array([spec[a]["g_beta"] for a in order])
    i_beta = np.array([spec[a]["i_beta"] for a in order])
    score = (np.outer(mm["growth"].to_numpy(), g_beta)
             + np.outer(mm["inflation"].to_numpy(), i_beta))
    W = pd.DataFrame(score, index=macro.index, columns=order)
    W = W.sub(W.mean(axis=1), axis=0)                       # dollar-neutral
    gross = W.abs().sum(axis=1).replace(0.0, np.nan)
    W = W.div(gross, axis=0).fillna(0.0)
    return W.shift(1).fillna(0.0)


def inflation_tilt_weights(macro: pd.DataFrame, smooth: int = 3, assets: dict | None = None,
                           order: list | None = None) -> pd.DataFrame:
    """Inflation-hedge weights: overweight real assets when inflation momentum is positive, lagged.

    The single-theme overlay — long real assets (commodities/TIPS/gold), short nominal (equities/bonds)
    scaled by the *sign and size* of inflation momentum, dollar-neutral, gross 1, lagged one month. When
    inflation momentum flips negative the book flips too (short real, long nominal). Defaults to the
    synthetic spec; pass ``assets``/``order`` for the real ETF book.
    """
    spec, order = _resolve_spec(assets, order)
    mm = _macro_momentum(macro, smooth=smooth)
    real = np.array([1.0 if spec[a]["real"] else -1.0 for a in order])
    score = np.outer(mm["inflation"].to_numpy(), real)     # +infl momentum → long real basket
    W = pd.DataFrame(score, index=macro.index, columns=order)
    W = W.sub(W.mean(axis=1), axis=0)
    gross = W.abs().sum(axis=1).replace(0.0, np.nan)
    W = W.div(gross, axis=0).fillna(0.0)
    return W.shift(1).fillna(0.0)


def book_returns(returns: pd.DataFrame, macro: pd.DataFrame, kind: str = "macro_momentum",
                 cost_bps: float = 5.0, smooth: int = 3, assets: dict | None = None,
                 order: list | None = None) -> pd.Series:
    """Net monthly return of a book (``'macro_momentum'`` or ``'inflation'``), minus turnover cost.

    Weights are formed from the macro state and applied to that month's asset returns (the weights are
    already lagged inside the weight functions). Cost is ``cost_bps`` per unit of weight traded. Pass
    ``assets``/``order`` to run on the real ETF panel rather than the synthetic spec.
    """
    _, order = _resolve_spec(assets, order)
    if kind == "macro_momentum":
        w = macro_momentum_weights(macro, smooth=smooth, assets=assets, order=order)
    elif kind == "inflation":
        w = inflation_tilt_weights(macro, smooth=smooth, assets=assets, order=order)
    else:
        raise ValueError(f"unknown book kind {kind!r}")
    w = w.reindex(returns.index).fillna(0.0)[order]
    gross = (w * returns[order].fillna(0.0)).sum(axis=1)
    cost = (cost_bps * 1e-4) * w.diff().abs().sum(axis=1)
    return (gross - cost).rename(kind)


def turnover_ann(macro: pd.DataFrame, kind: str = "macro_momentum", smooth: int = 3,
                 assets: dict | None = None, order: list | None = None) -> float:
    """Average annual one-way turnover (Σ|Δw|·12) — macro signals are slow, so this is low."""
    if kind == "macro_momentum":
        w = macro_momentum_weights(macro, smooth=smooth, assets=assets, order=order)
    else:
        w = inflation_tilt_weights(macro, smooth=smooth, assets=assets, order=order)
    return float(w.diff().abs().sum(axis=1).mean() * MONTHS_PER_YEAR)


def compare(returns: pd.DataFrame, macro: pd.DataFrame, cost_bps: float = 5.0, smooth: int = 3,
            periods_per_year: int = MONTHS_PER_YEAR, assets: dict | None = None,
            order: list | None = None) -> dict:
    """Headline stats for both books plus their (low) turnover."""
    out = {}
    for kind in ("macro_momentum", "inflation"):
        r = book_returns(returns, macro, kind=kind, cost_bps=cost_bps, smooth=smooth,
                         assets=assets, order=order)
        s = summary(r, periods_per_year)
        out[kind] = {**s, "turnover_ann": turnover_ann(macro, kind=kind, smooth=smooth,
                                                       assets=assets, order=order)}
    return out
