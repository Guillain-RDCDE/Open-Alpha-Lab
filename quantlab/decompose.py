"""Core: exact night/day decomposition of daily OHLC returns.

The whole project rests on one identity. For each trading day ``t`` with an
open/close auction price, the close-to-close return splits *exactly* into an
overnight leg (yesterday's close -> today's open) and an intraday leg
(today's open -> today's close):

    (1 + r_overnight) * (1 + r_intraday) = (1 + r_close_close)

because
    r_overnight  = Open[t]  / Close[t-1] - 1
    r_intraday   = Close[t] / Open[t]    - 1
    r_close_close= Close[t] / Close[t-1] - 1

There is no modelling choice here, no parameter, no fitting. Knuteson's whole
argument is that the *overnight* leg accumulates almost all the long-run gain
while the *intraday* leg is flat-to-negative. This module just measures that,
honestly, so you can reproduce his figures and judge them yourself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _excess_returns(returns, rf, periods_per_year: int) -> pd.Series:
    """Subtract a risk-free rate from a per-period return series.

    ``rf`` is either a scalar *annualised* rate (converted to per-period as
    ``rf / periods_per_year``, a first-order approximation that is exact to
    well under a basis point at realistic rates) or a *per-period* series
    aligned (reindexed) on the returns' index. ``rf=0.0`` is a no-op.
    """
    s = pd.Series(returns).astype(float)
    if np.ndim(rf) == 0:
        rf = float(rf)
        if rf != 0.0:
            s = s - rf / float(periods_per_year)
    else:
        rf_s = pd.Series(rf).astype(float)
        if not s.index.equals(rf_s.index):
            rf_s = rf_s.reindex(s.index)
        s = s - rf_s
    return s


def decompose(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Split daily OHLC into overnight / intraday / close-close returns.

    Parameters
    ----------
    ohlc : DataFrame
        Must contain ``Open`` and ``Close`` columns indexed by date (ascending).
        ``High``/``Low``/``Volume`` are ignored if present. Column matching is
        case-insensitive so Yahoo's ``Open``/``Close`` and lowercase both work.

    Returns
    -------
    DataFrame indexed by date with columns:
        r_overnight, r_intraday, r_close_close   (simple daily returns)
        cum_overnight, cum_intraday, cum_close_close
            cumulative *return* curves, i.e. equity-minus-one
            (cumprod(1+r) - 1), so they start near 0 like Knuteson's plots.

    The first row has NaN overnight/close-close (no prior close) and is dropped.
    """
    df = _normalise_columns(ohlc)
    open_, close = df["Open"], df["Close"]
    prev_close = close.shift(1)

    r_on = open_ / prev_close - 1.0
    r_id = close / open_ - 1.0
    r_cc = close / prev_close - 1.0

    out = pd.DataFrame(
        {
            "r_overnight": r_on,
            "r_intraday": r_id,
            "r_close_close": r_cc,
        }
    ).dropna()

    out["cum_overnight"] = (1.0 + out["r_overnight"]).cumprod() - 1.0
    out["cum_intraday"] = (1.0 + out["r_intraday"]).cumprod() - 1.0
    out["cum_close_close"] = (1.0 + out["r_close_close"]).cumprod() - 1.0
    return out


def summary(
    dec: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    rf: float | pd.Series = 0.0,
) -> pd.DataFrame:
    """Headline stats per leg. Sharpe is the number that actually matters.

    Returns a DataFrame indexed by leg (overnight / intraday / close_close) with:
        cum_return     total compounded return over the sample
        mean_bps_day   average daily return, in basis points (raw, not excess)
        vol_ann        annualised volatility
        sharpe         annualised Sharpe on *excess* returns,
                       mean/std * sqrt(periods)
        n_days         number of observations

    ``rf`` is the risk-free rate used for the Sharpe: a scalar annualised rate
    or a per-period series aligned on the index (default 0, which reproduces
    the historical behaviour exactly). With short rates near 4-5% (2023-26)
    the omission is worth roughly 2 bps/day of excess return — material for a
    thin overnight edge, negligible when rates were ~0.

    Note the ``sqrt(periods)`` annualisation of the Sharpe assumes serially
    uncorrelated returns; see ``analytics.lo_annualization_factor`` for the
    Lo (2002) autocorrelation-adjusted factor.
    """
    legs = {
        "overnight": dec["r_overnight"],
        "intraday": dec["r_intraday"],
        "close_close": dec["r_close_close"],
    }
    rows = {}
    for name, r in legs.items():
        mean = r.mean()
        std = r.std(ddof=1)
        ex = _excess_returns(r, rf, periods_per_year)
        ex_mean, ex_std = ex.mean(), ex.std(ddof=1)
        rows[name] = {
            "cum_return": (1.0 + r).prod() - 1.0,
            "mean_bps_day": mean * 1e4,
            "vol_ann": std * np.sqrt(periods_per_year),
            "sharpe": (ex_mean / ex_std * np.sqrt(periods_per_year)) if ex_std > 0 else np.nan,
            "n_days": int(r.shape[0]),
        }
    return pd.DataFrame(rows).T[
        ["cum_return", "mean_bps_day", "vol_ann", "sharpe", "n_days"]
    ]


def check_identity(dec: pd.DataFrame, atol: float = 1e-9) -> float:
    """Return the max absolute violation of the decomposition identity.

    Should be ~0 (floating-point dust). Used by the test-suite and as a cheap
    data-sanity gate before you trust any figure built on ``dec``.
    """
    lhs = (1.0 + dec["r_overnight"]) * (1.0 + dec["r_intraday"])
    rhs = 1.0 + dec["r_close_close"]
    max_err = float((lhs - rhs).abs().max())
    if not np.isfinite(max_err):
        raise ValueError("Non-finite returns in decomposition; check input prices.")
    return max_err


def _normalise_columns(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Map case-insensitive Open/Close (incl. yfinance MultiIndex) to canonical."""
    df = ohlc.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance single-ticker frames come as ('Open', 'SPY') etc.
        df.columns = df.columns.get_level_values(0)
    lookup = {str(c).lower(): c for c in df.columns}
    missing = [k for k in ("open", "close") if k not in lookup]
    if missing:
        raise KeyError(
            f"OHLC frame missing required column(s) {missing}; got {list(df.columns)}"
        )
    renamed = df.rename(columns={lookup["open"]: "Open", lookup["close"]: "Close"})
    return renamed.sort_index()
