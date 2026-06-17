"""Strategy and analysis layer for Study 274 (Fine-Wine).

The pitch for fine wine as an asset class is *diversification*: a low correlation
to equities plus a respectable long-run return. We test that pitch honestly:

  - **Raw correlation** of Liv-ex 100 annual returns with the S&P 500. A low
    number is the headline the wine-merchants quote.
  - **De-smoothed (Geltner) correlation.** Mid-price/appraisal indices are
    *smoothed*: an AR(1) in the reported series mechanically depresses measured
    volatility and correlation. The Geltner (1991/1993) first-order unsmoothing
    recovers an estimate of the underlying return, and almost always *raises*
    the correlation and the volatility. This is the killer adjustment.
  - **Risk/return.** CAGR, annual volatility, Sharpe for wine vs equity, price-
    only and before the very large frictions wine carries (buy/sell spread,
    storage, insurance).
  - **The mean-variance benefit test.** Does adding wine to an equity portfolio
    raise the Sharpe ratio, gross and net of costs, raw and de-smoothed? With a
    HAC (Newey-West) t-stat on the per-year Sharpe-improvement contribution.

The honest reckoning: n is tiny (~22 annual observations), wine prices are
quoted as mid-prices (a one-way spread of ~10–15% plus ~£10–15/case-year of
storage is real), and the index is survivorship-curated. A *measured* low
correlation is an upper bound on the true diversification benefit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# One-way transaction friction for fine wine (merchant/auction spread is large).
# Liv-ex mid-price round-trip costs: ~10% buyer's premium / merchant margin one
# way is a conservative, widely cited figure; storage+insurance ~1%/yr.
DEFAULT_WINE_ONEWAY_COST = 0.10   # one-way, applied on turnover x NAV
DEFAULT_WINE_CARRY = 0.01         # annual storage/insurance drag
DEFAULT_EQ_ONEWAY_COST = 0.0005   # S&P ETF one-way cost


# ---------------------------------------------------------------------------
# De-smoothing (Geltner first-order unsmoothing)
# ---------------------------------------------------------------------------
def estimate_ar1(returns: np.ndarray) -> float:
    """First-order autocorrelation of a return series (the smoothing coefficient)."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 3:
        return 0.0
    r0 = r[:-1] - r[:-1].mean()
    r1 = r[1:] - r[1:].mean()
    denom = np.sqrt((r0 ** 2).sum() * (r1 ** 2).sum())
    if denom == 0:
        return 0.0
    return float((r0 * r1).sum() / denom)


def desmooth(returns: np.ndarray, rho: float | None = None) -> np.ndarray:
    """Geltner first-order unsmoothing: r_true_t = (r_obs_t - rho*r_obs_{t-1})/(1-rho).

    ``rho`` defaults to the estimated first-order autocorrelation. The recovered
    series has higher volatility and (typically) higher correlation with other
    assets. The first observation is returned unchanged (no prior).
    """
    r = np.asarray(returns, dtype=float)
    if rho is None:
        rho = estimate_ar1(r)
    rho = float(np.clip(rho, -0.95, 0.95))
    out = r.copy()
    if abs(1.0 - rho) < 1e-9:
        return out
    out[1:] = (r[1:] - rho * r[:-1]) / (1.0 - rho)
    return out


# ---------------------------------------------------------------------------
# Risk / return primitives
# ---------------------------------------------------------------------------
def cagr(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return float("nan")
    return float(np.prod(1.0 + r) ** (1.0 / len(r)) - 1.0)


def ann_vol(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    return float(np.std(r, ddof=1))


def sharpe(returns: np.ndarray, rf: float = 0.02) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    v = ann_vol(r)
    if not np.isfinite(v) or v == 0:
        return float("nan")
    return float((np.mean(r) - rf) / v)


def hac_tstat(x: np.ndarray, lags: int = 2) -> tuple[float, float]:
    """Newey-West HAC t-stat for H0: mean(x) = 0 (one-sample), with `lags` lags.

    Returns (t, mean). Used on the per-year diversification-benefit contribution.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    mu = x.mean()
    e = x - mu
    gamma0 = (e @ e) / n
    var = gamma0
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1)
        cov = (e[k:] @ e[:-k]) / n
        var += 2.0 * w * cov
    se = np.sqrt(max(var, 1e-18) / n)
    return float(mu / se), float(mu)


# ---------------------------------------------------------------------------
# Correlation, raw and de-smoothed
# ---------------------------------------------------------------------------
def correlation_block(df: pd.DataFrame,
                      wine_col: str = "wine_return",
                      eq_col: str = "sp500_return") -> dict:
    """Raw and de-smoothed correlation between wine and equity annual returns."""
    w = df[wine_col].to_numpy(dtype=float)
    e = df[eq_col].to_numpy(dtype=float)
    raw_corr = float(np.corrcoef(w, e)[0, 1])
    rho = estimate_ar1(w)
    w_ds = desmooth(w, rho)
    ds_corr = float(np.corrcoef(w_ds, e)[0, 1])
    return {
        "n": int(len(w)),
        "raw_corr": round(raw_corr, 4),
        "wine_ar1": round(rho, 4),
        "desmoothed_corr": round(ds_corr, 4),
        "wine_vol_raw": round(ann_vol(w), 4),
        "wine_vol_desmoothed": round(ann_vol(w_ds), 4),
        "eq_vol": round(ann_vol(e), 4),
    }


# ---------------------------------------------------------------------------
# Mean-variance diversification benefit
# ---------------------------------------------------------------------------
def two_asset_frontier(wine: np.ndarray, eq: np.ndarray,
                       rf: float = 0.02,
                       n_grid: int = 101) -> pd.DataFrame:
    """Sharpe of wine/equity blends across a long-only weight grid."""
    wine = np.asarray(wine, dtype=float)
    eq = np.asarray(eq, dtype=float)
    rows = []
    for wgt in np.linspace(0.0, 1.0, n_grid):
        blend = wgt * wine + (1 - wgt) * eq
        rows.append({"wine_weight": round(float(wgt), 4),
                     "cagr": cagr(blend),
                     "vol": ann_vol(blend),
                     "sharpe": sharpe(blend, rf=rf)})
    return pd.DataFrame(rows)


def diversification_benefit(
    df: pd.DataFrame,
    wine_col: str = "wine_return",
    eq_col: str = "sp500_return",
    rf: float = 0.02,
    wine_weight: float = 0.20,
    wine_oneway_cost: float = DEFAULT_WINE_ONEWAY_COST,
    wine_carry: float = DEFAULT_WINE_CARRY,
    eq_oneway_cost: float = DEFAULT_EQ_ONEWAY_COST,
    rebalance_turnover: float = 0.20,
    hac_lags: int = 2,
) -> dict:
    """Does adding ``wine_weight`` to an equity book raise the Sharpe ratio?

    Compares 100% equity vs (wine_weight wine + (1-wine_weight) equity), in
    four lenses:
      - gross / raw   : reported wine returns, no costs
      - net  / raw    : reported wine returns minus carry and rebalancing cost
      - gross / desmoothed : Geltner-unsmoothed wine, no costs (the honest risk)
      - net  / desmoothed  : de-smoothed AND cost-charged (the only number that
        could justify a real allocation)

    Costs: an execution lag of one year is implicit (annual rebalancing at
    year-end). The wine sleeve pays ``wine_carry`` (storage/insurance) every
    year on its NAV plus ``wine_oneway_cost`` x ``rebalance_turnover`` one-way
    on rebalanced notional; the equity sleeve pays ``eq_oneway_cost`` similarly.

    Returns a dict with gross/net Sharpe for the blend vs equity-only, and a HAC
    t-stat on the per-year *Sharpe-improvement contribution* (the demeaned
    blend-minus-equity excess return, the quantity whose positive mean is what a
    diversification benefit means).
    """
    w_raw = df[wine_col].to_numpy(dtype=float)
    eq = df[eq_col].to_numpy(dtype=float)
    rho = estimate_ar1(w_raw)
    w_ds = desmooth(w_raw, rho)

    # Per-year cost drag on the wine sleeve (charged on the sleeve NAV).
    wine_cost = wine_carry + wine_oneway_cost * rebalance_turnover
    eq_cost = eq_oneway_cost * rebalance_turnover

    def _blend(wine_series, net: bool):
        wr = wine_series - (wine_cost if net else 0.0)
        er = eq - (eq_cost if net else 0.0)
        return wine_weight * wr + (1 - wine_weight) * er

    eq_only_gross = eq
    eq_only_net = eq - (eq_cost if True else 0.0)

    out = {
        "wine_weight": wine_weight,
        "n": int(len(eq)),
        "wine_ar1": round(rho, 4),
        "eq_only_sharpe": round(sharpe(eq_only_gross, rf=rf), 4),
        "eq_only_cagr": round(cagr(eq_only_gross), 4),
    }

    for label, series, net in [
        ("gross_raw", w_raw, False),
        ("net_raw", w_raw, True),
        ("gross_desmoothed", w_ds, False),
        ("net_desmoothed", w_ds, True),
    ]:
        blend = _blend(series, net)
        eq_ref = eq_only_net if net else eq_only_gross
        out[f"blend_sharpe_{label}"] = round(sharpe(blend, rf=rf), 4)
        out[f"blend_cagr_{label}"] = round(cagr(blend), 4)
        out[f"sharpe_delta_{label}"] = round(
            sharpe(blend, rf=rf) - sharpe(eq_ref, rf=rf), 4)
        # HAC t on the per-year blend-minus-equity excess return.
        t, mu = hac_tstat(blend - eq_ref, lags=hac_lags)
        out[f"benefit_hac_t_{label}"] = round(t, 3)
        out[f"benefit_mean_{label}"] = round(mu, 4)

    return out


# ---------------------------------------------------------------------------
# Summarize — compact R-dict for notebooks / results.md
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame,
              wine_col: str = "wine_return",
              eq_col: str = "sp500_return",
              rf: float = 0.02,
              wine_weight: float = 0.20) -> dict:
    """One-stop summary dict for the real merged DataFrame (mirrors results.md)."""
    cb = correlation_block(df, wine_col=wine_col, eq_col=eq_col)
    db = diversification_benefit(df, wine_col=wine_col, eq_col=eq_col,
                                 rf=rf, wine_weight=wine_weight)
    w = df[wine_col].to_numpy(dtype=float)
    e = df[eq_col].to_numpy(dtype=float)
    return {
        "n": cb["n"],
        "year_start": int(df["year"].min()),
        "year_end": int(df["year"].max()),
        # risk/return
        "wine_cagr_pct": round(cagr(w) * 100, 1),
        "eq_cagr_pct": round(cagr(e) * 100, 1),
        "wine_vol_pct": round(cb["wine_vol_raw"] * 100, 1),
        "wine_vol_desmoothed_pct": round(cb["wine_vol_desmoothed"] * 100, 1),
        "eq_vol_pct": round(cb["eq_vol"] * 100, 1),
        "wine_sharpe": round(sharpe(w, rf=rf), 3),
        "eq_sharpe": round(sharpe(e, rf=rf), 3),
        # correlation
        "raw_corr": cb["raw_corr"],
        "wine_ar1": cb["wine_ar1"],
        "desmoothed_corr": cb["desmoothed_corr"],
        # diversification benefit
        "eq_only_sharpe": db["eq_only_sharpe"],
        "blend_sharpe_gross_raw": db["blend_sharpe_gross_raw"],
        "blend_sharpe_net_raw": db["blend_sharpe_net_raw"],
        "blend_sharpe_gross_desmoothed": db["blend_sharpe_gross_desmoothed"],
        "blend_sharpe_net_desmoothed": db["blend_sharpe_net_desmoothed"],
        "sharpe_delta_net_desmoothed": db["sharpe_delta_net_desmoothed"],
        "benefit_hac_t_gross_raw": db["benefit_hac_t_gross_raw"],
        "benefit_hac_t_net_desmoothed": db["benefit_hac_t_net_desmoothed"],
    }
