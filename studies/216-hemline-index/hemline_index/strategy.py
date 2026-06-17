"""The strategy and its honest controls -- Study 216 (Hemline-Index).

The folk claim: when women's skirts get shorter, the stock market goes up; when
hemlines fall (longer skirts), the market falls.  Sometimes called the
"Hemline Indicator" or "Hemline Theory", attributed to economist George Taylor (1926)
and popularised in the 1970s and 1980s financial media.

We measure this claim three ways:

1. **Contingency table** -- decade-by-decade agreement rate between hemline direction
   (rising=bullish, falling=bearish) and market direction (positive / negative).
   With n = 10 decades this is the entire data universe.

2. **Phi coefficient and chi-square test** -- a formal association measure for a 2x2
   contingency table, with Fisher's exact test (appropriate for tiny n) and the
   chi-square approximation for comparison.

3. **Binary timing backtest** -- simulate a "buy when hemlines are rising, cash when
   falling" strategy over the 10-decade panel and compare to buy-and-hold.
   The timing strategy is evaluated only to show the scale of the edge if the
   signal were real -- not as a viable trading rule.

The honest null: hemline direction and stock-market direction are independent.
No look-ahead is baked in: hemline direction is labelled retrospectively per decade
(you would only know at the END of the decade whether hemlines had risen or fallen,
making any live trading rule impossible).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Contingency table: decade agreement rate
# ---------------------------------------------------------------------------
def agreement_table(df: pd.DataFrame) -> dict:
    """Count how many decades hemline direction matches market direction.

    Returns a dict with the 2x2 contingency counts and the raw hit rate.
    Hemline bullish (+1) = rising skirts.  Market bullish = positive decade return.
    """
    tp = int(((df["hemline_dir"] == 1) & df["market_positive"]).sum())
    fp = int(((df["hemline_dir"] == 1) & ~df["market_positive"]).sum())
    fn = int(((df["hemline_dir"] == -1) & df["market_positive"]).sum())
    tn = int(((df["hemline_dir"] == -1) & ~df["market_positive"]).sum())
    n = len(df)
    hit_rate = (tp + tn) / n if n > 0 else float("nan")
    return {
        "tp": tp,  # rising hem + bull market (claim correct)
        "fp": fp,  # rising hem + bear market (claim wrong)
        "fn": fn,  # falling hem + bull market (claim wrong)
        "tn": tn,  # falling hem + bear market (claim correct)
        "n": n,
        "hit_rate": hit_rate,
        "n_correct": tp + tn,
        "n_wrong": fp + fn,
    }


# ---------------------------------------------------------------------------
# Statistical tests: phi coefficient and Fisher's exact test
# ---------------------------------------------------------------------------
def phi_and_fisher(contingency: dict) -> dict:
    """Phi coefficient and Fisher's exact p-value for the 2x2 hemline-market table.

    The phi coefficient measures association in a 2x2 table; it ranges from -1 to +1.
    Fisher's exact test is the appropriate significance test when n is small (n = 10).
    A chi-square approximation is also returned for comparison (but is unreliable at
    these cell sizes).

    Returns dict with phi, chi2, p_chi2, p_fisher, and direction note.
    """
    tp = contingency["tp"]
    fp = contingency["fp"]
    fn = contingency["fn"]
    tn = contingency["tn"]
    n = contingency["n"]

    # Phi coefficient
    denom = math.sqrt((tp + fp) * (fn + tn) * (tp + fn) * (fp + tn))
    phi = (tp * tn - fp * fn) / denom if denom > 0 else float("nan")

    # Chi-square (unreliable for small n, shown for comparison only)
    expected = np.array([
        [(tp + fp) * (tp + fn) / n, (tp + fp) * (fp + tn) / n],
        [(fn + tn) * (tp + fn) / n, (fn + tn) * (fp + tn) / n],
    ])
    observed = np.array([[tp, fp], [fn, tn]])
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.sum((observed - expected) ** 2 / np.where(expected > 0, expected, np.nan)))
    p_chi2 = 1.0 - _chi2_cdf(chi2, df=1) if np.isfinite(chi2) else float("nan")

    # Fisher's exact test (one-sided: is there a positive association?)
    p_fisher = _fisher_exact_p(tp, fp, fn, tn)

    return {
        "phi": phi,
        "chi2": chi2,
        "p_chi2": p_chi2,
        "p_fisher_one_sided": p_fisher,
        "p_fisher_two_sided": min(1.0, 2.0 * p_fisher),
        "n": n,
        "low_n_warning": n < 20,
    }


# ---------------------------------------------------------------------------
# Decade-by-decade timing backtest
# ---------------------------------------------------------------------------
def timing_backtest(df: pd.DataFrame, col: str = "sp500_decade_pct") -> dict:
    """Simulate the hemline timer vs buy-and-hold over the decade panel.

    Strategy: hold the market (use decade return) when hemline is rising (+1);
    go to cash (0% decade return) when hemline is falling (-1).
    Compound both across all decades and annualise over 10-year periods.

    Returns annualised return of both strategies and the advantage.
    """
    rets_pct = df[col].to_numpy(dtype=float) / 100.0  # fraction
    hemline = df["hemline_dir"].to_numpy(dtype=int)
    n = len(rets_pct)
    n_years = n * 10  # each decade is 10 years

    # Timer: hold when hemline_dir == +1, cash otherwise
    position = (hemline == 1).astype(float)
    strat_r = position * rets_pct
    bah_r = rets_pct.copy()

    strat_cum = float(np.prod(1.0 + strat_r))
    bah_cum = float(np.prod(1.0 + bah_r))

    strat_ann = float(strat_cum ** (1.0 / n_years) - 1.0) * 100.0
    bah_ann = float(bah_cum ** (1.0 / n_years) - 1.0) * 100.0

    return {
        "n_decades": n,
        "n_years": n_years,
        "n_held": int(position.sum()),
        "n_cash": int((position == 0).sum()),
        "strat_cum": strat_cum,
        "bah_cum": bah_cum,
        "strat_ann_pct": strat_ann,
        "bah_ann_pct": bah_ann,
        "ann_advantage_pct": strat_ann - bah_ann,
    }


# ---------------------------------------------------------------------------
# Omnibus summary
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> dict:
    """Headline inference for the hemline-market claim.

    Combines the contingency table, phi coefficient, Fisher's exact test, and
    the timing backtest into a single result dict suitable for notebooks and verify.py.
    """
    ct = agreement_table(df)
    stats = phi_and_fisher(ct)
    bt = timing_backtest(df)

    # Decade-level hit rate breakdown
    correct_decades = list(
        df[
            ((df["hemline_dir"] == 1) & df["market_positive"])
            | ((df["hemline_dir"] == -1) & ~df["market_positive"])
        ].index
        if "decade_label" not in df.columns
        else df[
            ((df["hemline_dir"] == 1) & df["market_positive"])
            | ((df["hemline_dir"] == -1) & ~df["market_positive"])
        ]["decade_label"]
    )

    return {
        # Contingency
        "n": ct["n"],
        "tp": ct["tp"],
        "fp": ct["fp"],
        "fn": ct["fn"],
        "tn": ct["tn"],
        "hit_rate": ct["hit_rate"],
        "n_correct": ct["n_correct"],
        "n_wrong": ct["n_wrong"],
        # Phi + Fisher
        "phi": stats["phi"],
        "chi2": stats["chi2"],
        "p_chi2": stats["p_chi2"],
        "p_fisher_one": stats["p_fisher_one_sided"],
        "p_fisher_two": stats["p_fisher_two_sided"],
        "low_n_warning": stats["low_n_warning"],
        # Timing
        "strat_ann_pct": bt["strat_ann_pct"],
        "bah_ann_pct": bt["bah_ann_pct"],
        "ann_advantage_pct": bt["ann_advantage_pct"],
        "n_held": bt["n_held"],
        "n_cash": bt["n_cash"],
        "correct_decades": correct_decades,
    }


# ---------------------------------------------------------------------------
# Internal helpers: Fisher's exact test, chi2 CDF
# ---------------------------------------------------------------------------
def _fisher_exact_p(tp: int, fp: int, fn: int, tn: int) -> float:
    """One-sided Fisher's exact p-value for a 2x2 table.

    Tests H0: no positive association between hemline direction and market direction.
    Uses the hypergeometric probability of observing tp or more 'successes' given
    the marginal totals.
    """
    r1, r2 = tp + fp, fn + tn
    c1, c2 = tp + fn, fp + tn
    n = r1 + r2
    p = 0.0
    for k in range(tp, min(r1, c1) + 1):
        p += _hypergeo_pmf(k, r1, c1, n)
    return float(np.clip(p, 0.0, 1.0))


def _hypergeo_pmf(k: int, r1: int, c1: int, n: int) -> float:
    """Hypergeometric PMF: P(X = k) for the 2x2 Fisher table."""
    log_p = (
        _log_comb(r1, k)
        + _log_comb(n - r1, c1 - k)
        - _log_comb(n, c1)
    )
    return float(math.exp(log_p)) if log_p > -700 else 0.0


def _log_comb(n: int, k: int) -> float:
    """log C(n, k) using the log-gamma function."""
    if k < 0 or k > n:
        return -math.inf
    return (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    )


def _chi2_cdf(x: float, df: int = 1) -> float:
    """Chi-squared CDF via scipy if available, otherwise a simple approximation."""
    if x <= 0:
        return 0.0
    try:
        from scipy.stats import chi2 as _chi2
        return float(_chi2.cdf(x, df))
    except ImportError:
        # Rough approximation for df=1: chi2(1) CDF ~ erf(sqrt(x/2))
        return math.erf(math.sqrt(x / 2.0)) if df == 1 else 0.0
