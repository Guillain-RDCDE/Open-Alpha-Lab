"""Strategy and analysis layer for Study 271 (Cardboard-Box).

The "Dr. Cardboard" claim: year-over-year growth in box / rail-freight output is
a clean read on the real economy and therefore *forecasts the next year's stock
market*. We test this honestly:

  - **Predictive regression** of next-year S&P return on this year's freight
    growth, with a **Newey-West (HAC)** t-stat. With ~54 annual observations and
    one regressor, this is the right inference: overlapping-free annual data, but
    we still allow one lag of autocorrelation for honesty. REAL requires
    |t_HAC| >= 2 on the real tape.

  - The **coincident confound**: box/rail output is contemporaneous with GDP,
    which is itself driven by the same forces as equities. So even a *coincident*
    correlation (year-Y freight vs year-Y market) is unsurprising. The forecasting
    test (year-Y freight vs year-Y+1 market) is the one that would actually be
    tradable, and it is the one we headline.

  - A **timing strategy**: go long the S&P next year when freight growth was
    positive (economy expanding), flat otherwise. We charge a one-way cost on
    every position change, applied to NAV, and compare gross/net to passive
    buy-and-hold. Price-only returns throughout (labeled).

  - A **positive control** on synthetic data with a planted beta, to confirm the
    regression machinery detects a forecasting signal when one exists.

Survivorship note for the Signal axis: the S&P 500 itself is a survivorship-
managed index (additions/deletions). The freight series is an aggregate, not a
selected basket, so the main survivorship exposure is the index's own composition
drift — named here so it isn't smuggled past the reader.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Newey-West (HAC) OLS t-stat for a single-regressor predictive regression
# ---------------------------------------------------------------------------
def ols_hac(y: np.ndarray, x: np.ndarray, lags: int = 1) -> dict:
    """OLS of ``y`` on ``[1, x]`` with a Newey-West HAC covariance.

    Returns a dict with ``alpha``, ``beta`` (slope on x), ``t_beta`` (HAC t-stat
    on the slope), ``r2``, ``n``, and ``lags``. The HAC adjustment uses a Bartlett
    kernel with ``lags`` lags (default 1 — annual data has little serial
    correlation, but we stay honest).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    y, x = y[mask], x[mask]
    n = len(y)
    if n < 3:
        return {"alpha": np.nan, "beta": np.nan, "t_beta": np.nan,
                "r2": np.nan, "n": int(n), "lags": int(lags)}

    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ (X.T @ y)
    resid = y - X @ coef

    # Newey-West sandwich: X' Omega X with Bartlett weights.
    S = (X * resid[:, None]).T @ (X * resid[:, None])  # lag 0
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        u_t = (X[L:] * resid[L:, None])
        u_tL = (X[:-L] * resid[:-L, None])
        gamma = u_t.T @ u_tL
        S += w * (gamma + gamma.T)
    cov = XtX_inv @ S @ XtX_inv
    se_beta = float(np.sqrt(cov[1, 1]))

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    beta = float(coef[1])
    t_beta = beta / se_beta if se_beta > 0 else np.nan
    return {"alpha": float(coef[0]), "beta": beta, "t_beta": float(t_beta),
            "r2": float(r2), "n": int(n), "lags": int(lags)}


# ---------------------------------------------------------------------------
# Predictive regression on the forecast frame
# ---------------------------------------------------------------------------
def predictive_regression(frame: pd.DataFrame, lags: int = 1) -> dict:
    """Regress next-year S&P return on this year's freight growth, HAC t-stat.

    ``frame`` is the output of ``data.build_forecast_frame`` (columns
    ``predictor`` = freight growth %, ``fwd_return`` = next-year S&P return).
    """
    x = frame["predictor"].to_numpy(dtype=float)
    y = frame["fwd_return"].to_numpy(dtype=float)
    res = ols_hac(y, x, lags=lags)
    # Pearson correlation for context.
    if len(x) >= 3:
        r, p = scipy_stats.pearsonr(x, y)
    else:
        r, p = np.nan, np.nan
    res["pearson_r"] = float(r)
    res["pearson_p"] = float(p)
    return res


def coincident_regression(
    freight: pd.DataFrame, gspc: pd.DataFrame, predictor: str = "box_growth", lags: int = 1
) -> dict:
    """Same-year (coincident) regression: year-Y freight vs year-Y S&P return.

    This is the *non-tradable* benchmark — it shows freight and equities co-move
    contemporaneously (both ride GDP), which is exactly why people mistake it for
    a forecast.
    """
    f = freight[["year", predictor]].rename(columns={predictor: "predictor"})
    g = gspc[["year", "gspc_return"]]
    m = f.merge(g, on="year", how="inner")
    res = ols_hac(m["gspc_return"].to_numpy(float), m["predictor"].to_numpy(float), lags=lags)
    if len(m) >= 3:
        r, p = scipy_stats.pearsonr(m["predictor"], m["gspc_return"])
    else:
        r, p = np.nan, np.nan
    res["pearson_r"] = float(r)
    res["pearson_p"] = float(p)
    return res


# ---------------------------------------------------------------------------
# Timing strategy: long next year iff freight growth > threshold
# ---------------------------------------------------------------------------
def timing_strategy(
    frame: pd.DataFrame,
    threshold: float = 0.0,
    cost_bps: float = 10.0,
) -> dict:
    """Long the S&P next year iff this year's freight growth > ``threshold``.

    A one-way transaction cost of ``cost_bps`` basis points is charged on NAV at
    every change of position (entering or leaving the market). Returns gross and
    net annualised returns, the buy-and-hold benchmark, hit-rate, turnover, and
    the per-year strategy returns.

    All returns are **price-only** (the input ``fwd_return`` is a price return).
    """
    fr = frame.sort_values("ret_year").reset_index(drop=True).copy()
    pos = (fr["predictor"] > threshold).astype(int).to_numpy()
    rets = fr["fwd_return"].to_numpy(dtype=float)

    # Position changes (turnover): compare to previous year's position (start flat).
    prev = np.concatenate([[0], pos[:-1]])
    changes = np.abs(pos - prev)
    turnover = float(changes.mean())  # avg one-way changes per year

    gross = pos * rets
    cost = changes * (cost_bps / 1e4)
    net = gross - cost

    n = len(fr)
    bah_ann = float((1 + rets).prod() ** (1 / n) - 1)
    gross_ann = float((1 + gross).prod() ** (1 / n) - 1)
    net_ann = float((1 + net).prod() ** (1 / n) - 1)

    # Hit-rate of the position vs. realised market direction.
    mkt_up = rets > 0
    hits = np.where(pos == 1, mkt_up, ~mkt_up)
    hit_rate = float(hits.mean())
    uncond_up = float(mkt_up.mean())

    return {
        "n": int(n),
        "threshold": float(threshold),
        "cost_bps": float(cost_bps),
        "turnover": round(turnover, 4),
        "frac_invested": round(float(pos.mean()), 4),
        "bah_ann": round(bah_ann, 4),
        "gross_ann": round(gross_ann, 4),
        "net_ann": round(net_ann, 4),
        "hit_rate": round(hit_rate, 4),
        "uncond_up_rate": round(uncond_up, 4),
        "net_minus_bah": round(net_ann - bah_ann, 4),
        "strat_net_returns": net,
        "bah_returns": rets,
        "years": fr["ret_year"].to_numpy(),
    }


# ---------------------------------------------------------------------------
# Summarize — compact R-dict for notebooks / results.md
# ---------------------------------------------------------------------------
def summarize(
    freight: pd.DataFrame,
    gspc: pd.DataFrame,
    cost_bps: float = 10.0,
    lags: int = 1,
) -> dict:
    """One-stop summary dict for the real freight + ^GSPC tapes.

    Mirrors the ``R = dict(...)`` in build_notebooks.py and docs/results.md.
    """
    from cardboard_box import data as _data

    frame_box = _data.build_forecast_frame(freight, gspc, predictor="box_growth")
    frame_rail = _data.build_forecast_frame(freight, gspc, predictor="rail_growth")

    reg_box = predictive_regression(frame_box, lags=lags)
    reg_rail = predictive_regression(frame_rail, lags=lags)
    coin_box = coincident_regression(freight, gspc, predictor="box_growth", lags=lags)

    strat = timing_strategy(frame_box, threshold=0.0, cost_bps=cost_bps)

    return {
        "n": reg_box["n"],
        "year_start": int(frame_box["ret_year"].min()),
        "year_end": int(frame_box["ret_year"].max()),
        # Predictive (forecast) regression — box
        "box_beta": round(reg_box["beta"], 4),
        "box_t": round(reg_box["t_beta"], 3),
        "box_r2": round(reg_box["r2"], 4),
        "box_pearson_r": round(reg_box["pearson_r"], 4),
        "box_pearson_p": round(reg_box["pearson_p"], 4),
        # Predictive (forecast) regression — rail
        "rail_beta": round(reg_rail["beta"], 4),
        "rail_t": round(reg_rail["t_beta"], 3),
        "rail_r2": round(reg_rail["r2"], 4),
        # Coincident (non-tradable) regression — box
        "coin_box_t": round(coin_box["t_beta"], 3),
        "coin_box_r2": round(coin_box["r2"], 4),
        "coin_box_pearson_r": round(coin_box["pearson_r"], 4),
        # Timing strategy
        "uncond_up_pct": round(strat["uncond_up_rate"] * 100, 1),
        "hit_rate_pct": round(strat["hit_rate"] * 100, 1),
        "turnover": strat["turnover"],
        "frac_invested": strat["frac_invested"],
        "bah_ann_pct": round(strat["bah_ann"] * 100, 2),
        "gross_ann_pct": round(strat["gross_ann"] * 100, 2),
        "net_ann_pct": round(strat["net_ann"] * 100, 2),
        "net_minus_bah_pct": round(strat["net_minus_bah"] * 100, 2),
    }
