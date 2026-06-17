"""Strategy and analysis layer for Study 277 (Trading-Cards).

The claim under test: *graded Pokemon / sports cards are an investable asset
class* -- a store of value with equity-like (or better) returns and useful
diversification. We treat the curated annual card index as the "asset" and pin
it against the S&P 500 with the questions a serious allocator would ask:

  - **Return & risk.** CAGR, annualised vol, and Sharpe of cards vs equities.
    Collectibles are illiquid and pay no yield, so the bar is high: to be
    "investable" they need to clear equities on a risk-adjusted basis AFTER a
    realistic haul of frictions.
  - **The excess-return test.** Regress card returns on equity returns; the
    intercept (alpha) and its HAC (Newey-West) t-stat answer "is there a
    structural edge beyond beta?" A robust HAC t >= 2 on the real tape is the
    bar for a REAL signal.
  - **Diversification.** Correlation and beta to equities -- is there a genuine
    low-correlation benefit, or did cards crash *with* (or worse than) stocks?
  - **Frictions.** Collectibles carry enormous round-trip costs: auction-house
    buyer + seller premiums (often 20% each side), grading fees, the bid-ask of
    an illiquid market, and forgone dividends. We net these out explicitly.
  - **Bootstrap.** A stationary block bootstrap on the paired annual returns to
    put a confidence interval on the alpha and the Sharpe gap, because n is tiny
    and one boom dominates the sample.

The tiny-n / one-boom reckoning: the whole "cards beat stocks" story rides on a
single 2020-2021 mania that round-tripped. Strip those two years and the asset
class is unremarkable. We make that fragility explicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STUDY_SEED = 277


# ---------------------------------------------------------------------------
# Return / risk summary
# ---------------------------------------------------------------------------
def _cagr(returns: np.ndarray) -> float:
    g = np.prod(1.0 + returns)
    n = len(returns)
    if n == 0 or g <= 0:
        return float("nan")
    return float(g ** (1.0 / n) - 1.0)


def _max_drawdown(returns: np.ndarray) -> float:
    """Max drawdown of the cumulative wealth curve (negative number)."""
    wealth = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(wealth)
    dd = wealth / peak - 1.0
    return float(dd.min()) if len(dd) else float("nan")


def perf_stats(returns: np.ndarray, rf: float = 0.0) -> dict:
    """CAGR, annualised vol, Sharpe, max drawdown for an annual-return array."""
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    mu = float(returns.mean()) if n else float("nan")
    vol = float(returns.std(ddof=1)) if n > 1 else float("nan")
    cagr = _cagr(returns)
    sharpe = (mu - rf) / vol if vol and vol > 0 else float("nan")
    return {
        "n": int(n),
        "mean": round(mu, 4),
        "vol": round(vol, 4),
        "cagr": round(cagr, 4),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown": round(_max_drawdown(returns), 4),
    }


# ---------------------------------------------------------------------------
# Newey-West (HAC) alpha of cards over equities
# ---------------------------------------------------------------------------
def hac_alpha(
    card_ret: np.ndarray,
    gspc_ret: np.ndarray,
    lags: int = 2,
) -> dict:
    """Regress card returns on equity returns; return alpha, beta, HAC t-stats.

    Model: ``card_t = alpha + beta * gspc_t + eps_t``. The intercept ``alpha`` is
    the structural excess return of cards over their equity beta. We compute
    Newey-West (HAC) standard errors with ``lags`` lags so the t-stat is robust
    to the serial correlation and heteroskedasticity that plague a tiny annual
    series dominated by one boom.

    Returns a dict with alpha, beta, their HAC t-stats and p-values, R^2, n, corr.
    """
    y = np.asarray(card_ret, dtype=float)
    x = np.asarray(gspc_ret, dtype=float)
    n = len(y)
    X = np.column_stack([np.ones(n), x])  # [1, gspc]
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ (X.T @ y)
    resid = y - X @ beta_hat

    # Newey-West HAC covariance.
    S = np.zeros((2, 2))
    for t in range(n):
        xt = X[t:t + 1].T  # (2,1)
        S += (resid[t] ** 2) * (xt @ xt.T)
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        for t in range(L, n):
            xt = X[t:t + 1].T
            xtl = X[t - L:t - L + 1].T
            gamma = resid[t] * resid[t - L] * (xt @ xtl.T + xtl @ xt.T)
            S += w * gamma
    cov_hac = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov_hac))

    alpha, beta = float(beta_hat[0]), float(beta_hat[1])
    se_alpha, se_beta = float(se[0]), float(se[1])
    t_alpha = alpha / se_alpha if se_alpha > 0 else float("nan")
    t_beta = beta / se_beta if se_beta > 0 else float("nan")

    # Two-sided p via normal approx (tiny n -> conservative reading in docs).
    from scipy import stats as scipy_stats
    p_alpha = float(2 * (1 - scipy_stats.norm.cdf(abs(t_alpha)))) if np.isfinite(t_alpha) else float("nan")
    p_beta = float(2 * (1 - scipy_stats.norm.cdf(abs(t_beta)))) if np.isfinite(t_beta) else float("nan")

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    corr = float(np.corrcoef(y, x)[0, 1]) if n > 1 else float("nan")

    return {
        "n": int(n),
        "alpha": round(alpha, 4),
        "alpha_ann_pct": round(alpha * 100, 2),
        "beta": round(beta, 3),
        "t_alpha": round(float(t_alpha), 3),
        "t_beta": round(float(t_beta), 3),
        "p_alpha": round(p_alpha, 4),
        "p_beta": round(p_beta, 4),
        "r2": round(float(r2), 3),
        "corr": round(corr, 3),
        "lags": int(lags),
    }


# ---------------------------------------------------------------------------
# Frictions: net the collectible round-trip costs out of the card return
# ---------------------------------------------------------------------------
def apply_frictions(
    card_ret: np.ndarray,
    one_way_cost: float = 0.15,
    holding_years: float = 5.0,
) -> np.ndarray:
    """Subtract an *amortised* round-trip friction from each annual card return.

    Collectibles carry brutal frictions: auction-house buyer's + seller's
    premiums (often ~15-20% EACH side), grading fees, and the spread of an
    illiquid market. A buy-and-hold-then-sell allocator pays the round trip
    ``2 * one_way_cost`` once over a typical ``holding_years`` hold, which we
    amortise to an annual drag. This is deliberately *generous* (one round trip
    per multi-year hold, not per year).

    Returns the net annual return array.
    """
    round_trip = 2.0 * one_way_cost
    annual_drag = round_trip / max(holding_years, 1e-9)
    return np.asarray(card_ret, dtype=float) - annual_drag


# ---------------------------------------------------------------------------
# Block bootstrap on the paired returns
# ---------------------------------------------------------------------------
def block_bootstrap_alpha(
    card_ret: np.ndarray,
    gspc_ret: np.ndarray,
    n_boot: int = 5000,
    block: int = 3,
    seed: int = STUDY_SEED,
) -> dict:
    """Stationary block bootstrap CI for the card alpha and the Sharpe gap.

    Resamples the paired (card, gspc) annual returns in blocks of length
    ``block`` (to preserve the short-run autocorrelation of the boom/bust), and
    on each resample recomputes the OLS alpha and the (card Sharpe - equity
    Sharpe). Returns the empirical 2.5/50/97.5 percentiles and the fraction of
    resamples with alpha > 0.
    """
    y = np.asarray(card_ret, dtype=float)
    x = np.asarray(gspc_ret, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)

    alphas = np.empty(n_boot)
    sharpe_gaps = np.empty(n_boot)
    n_blocks = int(np.ceil(n / block))
    for b in range(n_boot):
        idx = []
        for _ in range(n_blocks):
            start = rng.integers(0, n)
            idx.extend([(start + k) % n for k in range(block)])
        idx = np.array(idx[:n])
        yb, xb = y[idx], x[idx]
        X = np.column_stack([np.ones(n), xb])
        try:
            coef = np.linalg.lstsq(X, yb, rcond=None)[0]
            alphas[b] = coef[0]
        except np.linalg.LinAlgError:
            alphas[b] = np.nan
        cs = yb.mean() / yb.std(ddof=1) if yb.std(ddof=1) > 0 else 0.0
        es = xb.mean() / xb.std(ddof=1) if xb.std(ddof=1) > 0 else 0.0
        sharpe_gaps[b] = cs - es

    a = alphas[np.isfinite(alphas)]
    return {
        "alpha_p2.5": round(float(np.percentile(a, 2.5)), 4),
        "alpha_p50": round(float(np.percentile(a, 50)), 4),
        "alpha_p97.5": round(float(np.percentile(a, 97.5)), 4),
        "frac_alpha_pos": round(float((a > 0).mean()), 3),
        "sharpe_gap_p50": round(float(np.percentile(sharpe_gaps, 50)), 3),
        "sharpe_gap_p2.5": round(float(np.percentile(sharpe_gaps, 2.5)), 3),
        "n_boot": int(n_boot),
    }


# ---------------------------------------------------------------------------
# Summarize -- compact R-dict for notebooks / results.md
# ---------------------------------------------------------------------------
def summarize(
    df: pd.DataFrame,
    one_way_cost: float = 0.15,
    holding_years: float = 5.0,
    n_boot: int = 5000,
    seed: int = STUDY_SEED,
) -> dict:
    """One-stop summary dict for the merged (cards vs equities) DataFrame.

    ``df`` must have ``card_return`` and ``gspc_return`` columns. Returns a flat
    dict mirroring the R = dict(...) in build_notebooks.py and docs/results.md.
    """
    card = df["card_return"].to_numpy(dtype=float)
    gspc = df["gspc_return"].to_numpy(dtype=float)
    card_net = apply_frictions(card, one_way_cost=one_way_cost, holding_years=holding_years)

    ps_card = perf_stats(card)
    ps_card_net = perf_stats(card_net)
    ps_gspc = perf_stats(gspc)
    reg = hac_alpha(card, gspc, lags=2)
    reg_net = hac_alpha(card_net, gspc, lags=2)
    boot = block_bootstrap_alpha(card, gspc, n_boot=n_boot, seed=seed)

    # The one-boom fragility: drop 2020 & 2021 if present.
    if "year" in df.columns:
        ex = df[~df["year"].isin([2020, 2021])]
        card_ex = ex["card_return"].to_numpy(dtype=float)
        cagr_ex = perf_stats(card_ex)["cagr"]
    else:
        cagr_ex = float("nan")

    return {
        "n": ps_card["n"],
        # Gross card stats
        "card_cagr_pct": round(ps_card["cagr"] * 100, 1),
        "card_vol_pct": round(ps_card["vol"] * 100, 1),
        "card_sharpe": ps_card["sharpe"],
        "card_maxdd_pct": round(ps_card["max_drawdown"] * 100, 1),
        # Net (after frictions) card stats
        "card_net_cagr_pct": round(ps_card_net["cagr"] * 100, 1),
        "card_net_sharpe": ps_card_net["sharpe"],
        # Equity stats
        "gspc_cagr_pct": round(ps_gspc["cagr"] * 100, 1),
        "gspc_vol_pct": round(ps_gspc["vol"] * 100, 1),
        "gspc_sharpe": ps_gspc["sharpe"],
        "gspc_maxdd_pct": round(ps_gspc["max_drawdown"] * 100, 1),
        # Regression (gross)
        "alpha_ann_pct": reg["alpha_ann_pct"],
        "beta": reg["beta"],
        "t_alpha": reg["t_alpha"],
        "p_alpha": reg["p_alpha"],
        "corr": reg["corr"],
        "r2": reg["r2"],
        # Regression (net)
        "alpha_net_ann_pct": round(reg_net["alpha"] * 100, 2),
        "t_alpha_net": reg_net["t_alpha"],
        # Bootstrap
        "boot_alpha_p2.5": round(boot["alpha_p2.5"] * 100, 2),
        "boot_alpha_p50": round(boot["alpha_p50"] * 100, 2),
        "boot_alpha_p97.5": round(boot["alpha_p97.5"] * 100, 2),
        "boot_frac_alpha_pos": boot["frac_alpha_pos"],
        # Fragility
        "card_cagr_ex_boom_pct": round(cagr_ex * 100, 1),
    }
