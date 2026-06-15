"""The strategy and its honest controls — Study 169 (Fluent-Tickers).

The folk claim (Alter & Oppenheimer, PNAS 2006): stocks with *fluent*,
easily-pronounceable ticker symbols (e.g. "KAR") earn higher returns than stocks
with *disfluent*, hard-to-pronounce tickers (e.g. "RDO").  The proposed mechanism
is **processing-fluency bias**: investors are cognitively drawn to things that feel
easy to process, causing them to bid up fluent-ticker stocks.

We implement this as a long/short portfolio study on the S&P 500 survivor panel
(survivorship bias named on the Signal axis):

- **Signal**: classify each ticker as FLUENT or NON-FLUENT by the heuristic score
  from :mod:`data` (median split).
- **Strategy**: at the start of each calendar year, re-sort the current panel and
  hold the FLUENT quintile long, NON-FLUENT quintile short (equal-weight both legs).
  No rebalancing intra-year (the ticker label is fixed by the symbol string and
  does not change within a ticker's life).
- **Control**: a *random-label* arm — the same long/short structure but with ticker
  labels shuffled randomly (seeded).  This is the relevant baseline: does the
  fluency label add anything beyond picking random halves of the same universe?

The honest multiple-comparisons reckoning: we test ONE portfolio (fluent minus
non-fluent).  We could also test sub-sorts (by industry, market cap, listing year)
but deliberately limit to a single pre-specified test to avoid inflating Type I
errors — and even that one fails.

HAC t-stat on the annual long-short return differentials is the decisive number.
``summarize()`` returns a Newey-West t on the *annual* excess returns of the
long-short spread.

No look-ahead: ticker fluency scores depend only on the ticker string itself (which
is known at the moment of IPO and does not change).  Forward returns are observed
strictly after the portfolio formation date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import classify_tickers, score_tickers


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------
def build_ls_portfolio(
    prices: pd.DataFrame,
    fluent_mask: pd.Series,
    rebalance: str = "YS",
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Long FLUENT, short NON-FLUENT equal-weight portfolio.

    Parameters
    ----------
    prices : DataFrame
        (dates × tickers) closing prices.
    fluent_mask : Series
        Boolean Series indexed by ticker; True = FLUENT.
    rebalance : str
        Pandas offset alias for rebalancing frequency.  ``"AS"`` = annually
        at year-start (``"YS"`` in newer pandas).  The portfolio is *not*
        rebalanced intra-period since ticker fluency labels are fixed.  We keep
        the annual cadence only to re-align equal weights.
    cost_bps : float
        One-way round-trip cost in basis points applied at each rebalance
        to both the long and short legs.

    Returns
    -------
    DataFrame
        Columns: ``ret_fluent`` (equal-weight FLUENT close-to-close),
        ``ret_nonfluent`` (equal-weight NON-FLUENT), ``ret_ls`` (FLUENT minus
        NON-FLUENT), ``ret_ls_net`` (after costs).
    """
    log_rets = np.log(prices / prices.shift(1)).iloc[1:]

    fluent_cols = [t for t in prices.columns if fluent_mask.get(t, False)]
    nonfluent_cols = [t for t in prices.columns if not fluent_mask.get(t, True)]

    if not fluent_cols or not nonfluent_cols:
        raise ValueError("Need at least one ticker in each fluency group.")

    r_fluent = log_rets[fluent_cols].mean(axis=1)
    r_nonfluent = log_rets[nonfluent_cols].mean(axis=1)
    r_ls = r_fluent - r_nonfluent

    # Estimate cost drag: rebalance events hit both legs.
    # We apply a one-time cost at each rebalance date.
    rebal_dates = pd.date_range(prices.index[0], prices.index[-1], freq=rebalance)
    # find actual dates in the index closest to rebal_dates
    cost_series = pd.Series(0.0, index=log_rets.index)
    for rd in rebal_dates:
        # round to next available date
        future = log_rets.index[log_rets.index >= rd]
        if len(future) == 0:
            continue
        actual = future[0]
        cost_series[actual] += cost_bps * 2e-4  # both legs

    r_ls_net = r_ls - cost_series

    return pd.DataFrame(
        {
            "ret_fluent": r_fluent,
            "ret_nonfluent": r_nonfluent,
            "ret_ls": r_ls,
            "ret_ls_net": r_ls_net,
        },
        index=log_rets.index,
    )


def build_random_ls_portfolio(
    prices: pd.DataFrame,
    fluent_mask: pd.Series,
    seed: int = 0,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Random-label control arm: shuffle the fluent/non-fluent labels and rerun.

    Same structure as :func:`build_ls_portfolio` but the label assignment is
    random (seeded for reproducibility).  This is the honest baseline: a
    long/short on *random halves* of the same universe, with the same costs.
    If the fluency signal adds nothing, the two should be indistinguishable.
    """
    rng = np.random.default_rng(seed)
    tickers = list(prices.columns)
    n_fluent = int(fluent_mask.sum())
    shuffled_idx = rng.permutation(len(tickers))
    random_mask = pd.Series(
        {t: i < n_fluent for i, t in zip(shuffled_idx, tickers)},
        name="fluent",
    )
    return build_ls_portfolio(prices, random_mask, cost_bps=cost_bps)


# ---------------------------------------------------------------------------
# Annual aggregation
# ---------------------------------------------------------------------------
def annual_ls_returns(portfolio: pd.DataFrame, col: str = "ret_ls") -> pd.Series:
    """Sum log-returns within each calendar year (no compounding within year).

    Returns a Series indexed by year (int), one observation per full year in the
    panel.  This is the raw material for the HAC t-test: n = number of full years.
    """
    r = portfolio[col].copy()
    r.index = pd.to_datetime(r.index)
    by_year = r.groupby(r.index.year).sum()
    return by_year


# ---------------------------------------------------------------------------
# Inference — Newey-West HAC on annual excess returns
# ---------------------------------------------------------------------------
def summarize(annual_rets: pd.Series, annualize: bool = False) -> dict:
    """Headline statistics for the long-short annual return series.

    The decisive number is the Newey-West HAC t-stat on the mean annual
    long-short return.  Because we aggregate to *annual* observations, the
    autocorrelation structure is naturally mild — but we apply NW nonetheless
    to be conservative.

    Returns n (years), mean (bps/year), win_rate (fraction positive), tstat,
    and a rough Sharpe (mean / std).
    """
    r = annual_rets.dropna().to_numpy(dtype=float)
    n = r.size
    out = {
        "n_years": int(n),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "win_rate": float((r > 0).mean()) if n else float("nan"),
        "sharpe": float(r.mean() / r.std(ddof=1)) if n > 1 and r.std() > 0 else float("nan"),
        "tstat": float("nan"),
    }
    if n > 5:
        mu = r.mean()
        e = r - mu
        lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
        lrv = float(e @ e) / n
        for k in range(1, min(lags + 1, n)):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        out["tstat"] = float(mu / se) if se > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Daily-frequency summarize (for notebooks and verify.py)
# ---------------------------------------------------------------------------
def summarize_daily(daily_rets: pd.Series) -> dict:
    """Quick summary on daily long-short returns (for cost sweep, etc.).

    Returns n, mean_bps (per day), annualised Sharpe (×sqrt(252)), and HAC t.
    """
    r = daily_rets.dropna().to_numpy(dtype=float)
    n = r.size
    out = {
        "n_days": int(n),
        "mean_bps_day": float(r.mean() * 1e4) if n else float("nan"),
        "sharpe_ann": float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if n > 1 and r.std() > 0 else float("nan"),
        "tstat": float("nan"),
    }
    if n > 5:
        mu = r.mean()
        e = r - mu
        lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
        lrv = float(e @ e) / n
        for k in range(1, min(lags + 1, n)):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        out["tstat"] = float(mu / se) if se > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Universe helpers
# ---------------------------------------------------------------------------
def get_sp500_tickers() -> list[str]:
    """Return the S&P 500 constituent list (survivorship-biased snapshot).

    Falls back to a small hard-coded representative set if quantlab.universe
    is unavailable.  This is only used in examples/verify.py; the test suite
    uses the synthetic generator.
    """
    try:
        import sys
        import os

        sys.path.insert(
            0,
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..")
            ),
        )
        from quantlab.universe import sp500_symbols

        return sp500_symbols(allow_survivorship_bias=True)
    except Exception:
        # Fallback: a representative sample of S&P 500 tickers with a mix of
        # fluent (pronounceable) and disfluent (consonant-cluster) names.
        return [
            # Fluent-scoring tickers (contain vowels, easy to say)
            "CAT", "CVX", "KO", "MCD", "VZ", "IBM", "AIG", "AXP", "AMT", "ARE",
            "AVB", "AVGO", "AMAT", "ADBE", "ADI", "AEE", "AEP", "AES", "AFL",
            "AIG", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALK", "ALL", "ALLE",
            # Disfluent-scoring tickers (consonant clusters, hard to say)
            "BRK", "LLY", "DVN", "NVT", "FMC", "GPN", "PNW", "PWR", "SWK",
            "VFC", "VLO", "VNO", "VTR", "WRK", "XYL", "HWM", "BWA", "LKQ",
            "MKC", "MLM", "MMC", "MPC", "MRK", "MRO", "MSI", "MTB", "MTD",
            # Additional mix
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM",
            "BAC", "WMT", "XOM", "PG", "JNJ", "HD", "MA", "UNH", "V",
        ]
