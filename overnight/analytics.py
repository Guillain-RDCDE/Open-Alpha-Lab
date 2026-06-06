"""Research-grade analytics — the tools a sceptical quant reaches for next.

Five questions that decide whether "overnight Sharpe 0.8" means anything:

  1. Is it significant once you stop pretending daily returns are i.i.d.?
     -> ``mean_tstat_hac`` (Newey-West) and ``sharpe_with_se`` (Lo 2002).
  2. Are we even comparing like with like? The overnight window spans ~17.5h on
     weekdays and ~65h over weekends; the intraday window is 6.5h. -> ``calendar_hours``
     and ``time_normalized_summary`` put both legs on a per-hour footing.
  3. Has the edge decayed since it was published? -> ``rolling_sharpe``
     (cf. McLean & Pontiff 2016 on post-publication decay).
  4. How much capital can actually harvest it before market impact eats it?
     -> ``capacity_estimate`` (square-root impact law).

Everything is deterministic and unit-tested. Methods: Newey & West (1987),
Lo (2002), Almgren et al. (2005). See docs/references.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .decompose import TRADING_DAYS_PER_YEAR

# NYSE regular session, in hours past midnight (local exchange time).
_OPEN_H = 9.5
_CLOSE_H = 16.0
_INTRADAY_HOURS = _CLOSE_H - _OPEN_H  # 6.5


# ---------------------------------------------------------------------------
# 1. Honest inference: autocorrelation-robust significance
# ---------------------------------------------------------------------------
def mean_tstat_hac(returns: pd.Series, lags: int | None = None) -> dict:
    """Newey-West (HAC) t-statistic for the sample mean return.

    Daily returns are mildly autocorrelated and heteroskedastic; the naive
    t-stat overstates significance. This uses the Newey-West long-run variance
    with the Bartlett kernel. ``lags=None`` picks the standard rule of thumb
    ``floor(4*(n/100)^(2/9))``.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    mu = r.mean()
    e = r - mu
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n  # gamma_0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)  # Bartlett weight
        gamma_k = float(e[k:] @ e[:-k]) / n
        lrv += 2.0 * w * gamma_k
    se_mean = np.sqrt(max(lrv, 0.0) / n)
    return {
        "mean_bps": mu * 1e4,
        "se_bps": se_mean * 1e4,
        "tstat": mu / se_mean if se_mean > 0 else np.nan,
        "lags": lags,
        "n": n,
    }


def sharpe_with_se(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """Annualised Sharpe with the Lo (2002) i.i.d. standard error and t-stat.

    Lo's delta-method SE for the per-period Sharpe SR is sqrt((1 + SR^2/2)/n).
    The t-stat (SR/SE) is invariant to annualisation. A |t| < ~2 means the
    Sharpe is not distinguishable from zero, however pretty the cumulative chart.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    sd = r.std(ddof=1)
    sr = r.mean() / sd if sd > 0 else np.nan  # per-period
    se = np.sqrt((1.0 + 0.5 * sr**2) / n)
    ann = np.sqrt(periods_per_year)
    return {
        "sharpe_ann": sr * ann,
        "se_ann": se * ann,
        "tstat": sr / se if se > 0 else np.nan,
        "n": n,
    }


# ---------------------------------------------------------------------------
# 2. Apples to apples: normalise by calendar time
# ---------------------------------------------------------------------------
def calendar_hours(index: pd.DatetimeIndex) -> pd.Series:
    """Overnight calendar hours preceding each trading day.

    From the prior close (16:00) to the open (09:30): consecutive days -> 17.5h,
    across a weekend -> 65.5h. Formula: gap_days*24 - intraday_hours.
    The first day has no preceding overnight window (NaN).
    """
    idx = pd.DatetimeIndex(index)
    gap_days = np.empty(len(idx))
    gap_days[0] = np.nan
    gap_days[1:] = (idx[1:] - idx[:-1]).days
    overnight_h = gap_days * 24.0 - _INTRADAY_HOURS
    return pd.Series(overnight_h, index=idx, name="overnight_hours")


def time_normalized_summary(dec: pd.DataFrame) -> pd.DataFrame:
    """Compare the legs *per calendar hour*, not per session.

    The raw "overnight >> intraday" comparison is apples-to-oranges: the
    overnight leg accrues over far more calendar time. Returns mean basis points
    **per hour** for each leg, plus the ratio of session length. If the per-hour
    drifts are similar, much of the "anomaly" is just the clock.
    """
    on_hours = calendar_hours(dec.index)
    mask = on_hours.notna()
    on_bps_per_h = (dec.loc[mask, "r_overnight"] * 1e4 / on_hours[mask]).mean()
    id_bps_per_h = (dec.loc[mask, "r_intraday"] * 1e4 / _INTRADAY_HOURS).mean()
    mean_on_hours = on_hours[mask].mean()
    return pd.DataFrame(
        {
            "mean_bps_per_session": [
                dec.loc[mask, "r_overnight"].mean() * 1e4,
                dec.loc[mask, "r_intraday"].mean() * 1e4,
            ],
            "session_hours": [mean_on_hours, _INTRADAY_HOURS],
            "mean_bps_per_hour": [on_bps_per_h, id_bps_per_h],
        },
        index=["overnight", "intraday"],
    )


# ---------------------------------------------------------------------------
# 3. Has the edge decayed?  (rolling Sharpe / alpha decay)
# ---------------------------------------------------------------------------
def rolling_sharpe(
    returns: pd.Series,
    window: int = TRADING_DAYS_PER_YEAR * 5,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Trailing annualised Sharpe over a rolling window (default 5y).

    A flat-then-falling line is the classic post-publication decay signature
    (McLean & Pontiff 2016): once an edge is documented, it tends to weaken.
    """
    r = pd.Series(returns).astype(float).dropna()
    m = r.rolling(window).mean()
    s = r.rolling(window).std(ddof=1)
    return (m / s * np.sqrt(periods_per_year)).dropna()


# ---------------------------------------------------------------------------
# 4. Capacity: how much money can harvest the edge before impact kills it?
# ---------------------------------------------------------------------------
def capacity_estimate(
    dec: pd.DataFrame,
    ohlc: pd.DataFrame,
    impact_coef: float = 1.0,
    lookback: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """Dollar capacity at which square-root market impact equals the gross edge.

    Square-root law (Almgren et al. 2005): one-way impact (return units)
    ``≈ impact_coef * daily_vol * sqrt(Q / ADV)``, where Q is order size in
    shares and ADV the average daily volume. The strategy trades a round trip
    each night, so set ``2 * impact = mean overnight return`` and solve:

        Q* / ADV = (edge / (2 * impact_coef * daily_vol))^2
        capacity$ ≈ Q* * price

    This is an order-of-magnitude figure (the coefficient is venue-specific) but
    it makes the key point: a firm large enough to *move world markets* could not
    quietly harvest an edge this thin — the impact of trading at scale would
    swamp it. Requires a ``Volume`` column on ``ohlc``.
    """
    edge = dec["r_overnight"].mean()
    daily_vol = dec["r_close_close"].std(ddof=1)
    cols = {str(c).lower(): c for c in ohlc.columns}
    if "volume" not in cols:
        raise KeyError("capacity_estimate needs a 'Volume' column on ohlc.")
    adv = float(ohlc[cols["volume"]].tail(lookback).mean())
    price = float(ohlc[cols["close"]].tail(lookback).mean())

    participation = (edge / (2.0 * impact_coef * daily_vol)) ** 2
    q_shares = participation * adv
    capacity_usd = q_shares * price
    return {
        "edge_bps": edge * 1e4,
        "daily_vol_bps": daily_vol * 1e4,
        "adv_shares": adv,
        "price": price,
        "participation_rate": participation,
        "capacity_shares": q_shares,
        "capacity_usd": capacity_usd,
    }
