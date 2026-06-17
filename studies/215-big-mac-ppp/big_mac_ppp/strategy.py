"""Strategy and honest controls — Study 215 (Big-Mac-PPP).

The recipe: each July, read the Economist Big Mac mis-valuation (% over/under-priced
vs USD using simple raw-price PPP). Go long the N most under-valued currencies and
short the N most over-valued currencies, entering in August and holding for one year.
Rebalance annually. This is the long-horizon PPP mean-reversion hypothesis.

The honest control: a random-ranking control — the same long-short construction but
with currencies assigned random ranks each year (i.i.d. draw), averaged over many seeds.
If Big Mac PPP tells us something, the PPP sort should beat the coin. If it doesn't,
both earn zero.

No look-ahead: the Big Mac snapshot for year T (published July T) drives the positions
entered in August T; the forward return is measured Aug T to Jul T+1. This is exactly
what a real investor would observe and do.

Cost model: FX spot is liquid; 5-10 bps round-trip for institutional G10 trades.
Annual rebalance means one trade per pair per year — very low turnover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal: Big Mac PPP mis-valuation ranks
# ---------------------------------------------------------------------------
def misval_ranks(misval: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank of Big Mac mis-valuation each year.

    Positive mis_val = over-valued (expensive Big Mac vs US) -> short candidate.
    Rank is 1 = most under-valued (short list bottom), N = most over-valued (long short list top).
    Rank is assigned cross-sectionally within each year, based on currencies with data.

    Returns a DataFrame of the same shape as ``misval`` with ranks.
    """
    return misval.rank(axis=1, method="first")


def random_ranks(misval: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Reproducible i.i.d. random cross-sectional ranks — the control arm."""
    rng = np.random.default_rng(seed)
    n_rows, n_cols = misval.shape
    out = np.zeros_like(misval.values, dtype=float)
    for i in range(n_rows):
        n_valid = int(misval.iloc[i].notna().sum())
        if n_valid == 0:
            out[i] = np.nan
            continue
        perm = rng.permutation(n_valid) + 1
        out[i, :n_valid] = perm
        out[i, n_valid:] = np.nan
    return pd.DataFrame(out, index=misval.index, columns=misval.columns)


# ---------------------------------------------------------------------------
# Portfolio construction (annual, long-short on PPP valuation)
# ---------------------------------------------------------------------------
def build_portfolio(
    misval: pd.DataFrame,
    fx_annual: pd.DataFrame,
    ranks: pd.DataFrame | None = None,
    top_n: int = 3,
    bot_n: int = 3,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Annual long-short portfolio from Big Mac PPP ranks.

    Long the ``bot_n`` most under-valued currencies (lowest rank = cheapest Big Mac),
    short the ``top_n`` most over-valued (highest rank = most expensive Big Mac),
    equal weight within each leg. Enter August of snapshot year, hold 12 months.

    ``cost_bps`` is the one-way transaction cost per leg, applied once at the annual
    rebalance on currencies that change side. At 100% annual turnover: 2 * cost_bps
    per pair.

    Returns a DataFrame with columns:
        year, ret_gross, ret_net, n_long, n_short, turnover
    indexed by snapshot year.
    """
    if ranks is None:
        ranks = misval_ranks(misval)

    rows = []
    prev_weight: dict[str, float] = {}
    common_years = sorted(set(misval.index) & set(fx_annual.index))

    for yr in common_years:
        rank_row = ranks.loc[yr].dropna()
        if len(rank_row) < top_n + bot_n:
            continue

        sorted_ccy = rank_row.sort_values()
        # Under-valued = LOW rank (cheap Big Mac) -> long them (expect appreciation)
        longs = sorted_ccy.head(bot_n).index.tolist()
        # Over-valued = HIGH rank (expensive Big Mac) -> short them (expect depreciation)
        shorts = sorted_ccy.tail(top_n).index.tolist()

        weight: dict[str, float] = {}
        for c in longs:
            weight[c] = 1.0 / bot_n
        for c in shorts:
            weight[c] = -1.0 / top_n

        # Portfolio return = weighted sum of forward FX returns
        ret_gross = 0.0
        for c, w in weight.items():
            if c in fx_annual.columns and not pd.isna(fx_annual.loc[yr, c]):
                ret_gross += w * fx_annual.loc[yr, c]

        # Turnover = fraction of notional that changes side
        all_ccy = set(weight) | set(prev_weight)
        delta = sum(abs(weight.get(c, 0.0) - prev_weight.get(c, 0.0)) for c in all_ccy) / 2.0
        cost = delta * cost_bps * 1e-4
        ret_net = ret_gross - cost

        rows.append({
            "year": yr,
            "ret_gross": ret_gross,
            "ret_net": ret_net,
            "n_long": len(longs),
            "n_short": len(shorts),
            "turnover": delta,
        })
        prev_weight = weight

    df = pd.DataFrame(rows).set_index("year")
    return df


# ---------------------------------------------------------------------------
# Cross-sectional regression: Big Mac mis-val predicts forward FX return?
# ---------------------------------------------------------------------------
def cross_sectional_regression(panel: pd.DataFrame) -> dict:
    """Pool all year-currency observations and regress fwd_log_ret on mis_val_pct.

    A robust PPP reversion signal would show a negative slope: currencies with a
    higher (more positive) mis-valuation tend to fall more vs USD.

    Returns dict with: slope, intercept, t_slope, r_squared, n_obs.
    """
    df = panel.dropna(subset=["mis_val_pct", "fwd_log_ret"])
    n = len(df)
    if n < 5:
        return {"slope": float("nan"), "intercept": float("nan"),
                "t_slope": float("nan"), "r_squared": float("nan"), "n_obs": int(n)}

    x = df["mis_val_pct"].to_numpy()
    y = df["fwd_log_ret"].to_numpy()

    # OLS
    x_dm = x - x.mean()
    slope = float(np.dot(x_dm, y) / np.dot(x_dm, x_dm))
    intercept = float(y.mean() - slope * x.mean())

    # Residuals and t-stat (panel OLS, treat each obs as independent)
    y_hat = slope * x + intercept
    resid = y - y_hat
    ss_res = np.dot(resid, resid)
    ss_tot = np.dot(y - y.mean(), y - y.mean())
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Newey-West style HAC (simple HC0 here since panel residuals are cross-sectionally
    # correlated within year — we use simple OLS se as a lower bound)
    var_slope = (ss_res / (n - 2)) / np.dot(x_dm, x_dm)
    se_slope = float(np.sqrt(max(var_slope, 0.0)))
    t_slope = float(slope / se_slope) if se_slope > 0 else float("nan")

    return {
        "slope": slope,
        "intercept": intercept,
        "t_slope": t_slope,
        "r_squared": float(r_sq),
        "n_obs": int(n),
    }


# ---------------------------------------------------------------------------
# Summary statistics (annual returns)
# ---------------------------------------------------------------------------
def summarize(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline annual statistics for one ledger.

    Returns n_years, annualised mean (%), annualised Sharpe (no sqrt since annual),
    skew, and a simple t-stat on the annual mean (few obs — use t distribution).
    """
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 2:
        return {
            "n_years": int(n),
            "mean_ann_pct": float("nan"),
            "sharpe_ann": float("nan"),
            "skew": float("nan"),
            "tstat": float("nan"),
        }

    mu = r.mean()
    sd = r.std(ddof=1)
    out = {
        "n_years": int(n),
        "mean_ann_pct": float(mu * 100),        # already annual log-return
        "sharpe_ann": float(mu / sd) if sd > 0 else float("nan"),
        "skew": float(pd.Series(r).skew()),
        "tstat": float("nan"),
    }

    # Simple t-stat on the mean (annual obs — too few for Newey-West lags)
    if n >= 4:
        se = sd / np.sqrt(n)
        out["tstat"] = float(mu / se) if se > 0 else float("nan")

    return out


def cost_sweep(
    misval: pd.DataFrame,
    fx_annual: pd.DataFrame,
    ranks: pd.DataFrame | None = None,
    costs_bps: list[float] | None = None,
    top_n: int = 3,
    bot_n: int = 3,
) -> pd.DataFrame:
    """Net annual return statistics across a range of per-leg costs.

    Returns a tidy DataFrame indexed by cost_bps with columns from :func:`summarize`.
    """
    if costs_bps is None:
        costs_bps = [0, 5, 10, 20, 50]
    if ranks is None:
        ranks = misval_ranks(misval)
    rows = {}
    for c in costs_bps:
        led = build_portfolio(misval, fx_annual, ranks=ranks, top_n=top_n, bot_n=bot_n, cost_bps=c)
        rows[c] = summarize(led, "ret_net")
    return pd.DataFrame(rows).T
