"""Research-grade analytics — the tools a sceptical quant reaches for next.

Five questions that decide whether "overnight Sharpe 0.8" means anything:

  1. Is it significant once you stop pretending daily returns are i.i.d.?
     -> ``mean_tstat_hac`` (Newey-West) for the mean, and ``sharpe_with_se``
     for the Sharpe. Note: the default ``sharpe_with_se`` SE is the *i.i.d.*
     delta-method formula (the i.i.d. case of Lo 2002) and understates the
     uncertainty on autocorrelated or fat-tailed returns — pass
     ``method='mertens'`` (skewness/kurtosis-adjusted) or ``method='lo'``
     (autocorrelation-adjusted) for honest errors.
  2. Are we even comparing like with like? The overnight window spans ~17.5h on
     weekdays and ~65h over weekends; the intraday window is 6.5h. -> ``calendar_hours``
     and ``time_normalized_summary`` put both legs on a per-hour footing.
  3. Has the edge decayed since it was published? -> ``rolling_sharpe``
     (cf. McLean & Pontiff 2016 on post-publication decay).
  4. How much capital can actually harvest it before market impact eats it?
     -> ``capacity_estimate`` (square-root impact law).

Everything is deterministic and unit-tested. Methods: Newey & West (1987),
Lo (2002), Mertens (2002), Almgren et al. (2005). See docs/references.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .decompose import TRADING_DAYS_PER_YEAR, _excess_returns

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


def sharpe_with_se(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    method: str = "iid",
    q: int | None = None,
    rf: float | pd.Series = 0.0,
) -> dict:
    """Annualised Sharpe with a delta-method standard error and t-stat.

    The per-period Sharpe SR = mean/std has, under i.i.d. normal returns,
    Var(SR) = (1 + SR^2/2)/n — this is the *i.i.d. case* of Lo (2002), and the
    default here for backward compatibility. Real daily returns are fat-tailed
    and autocorrelated, so the default **understates** the SE; prefer one of:

    - ``method='mertens'`` — Mertens (2002) higher-moment correction,
      Var(SR) = (1 + SR^2/2 - g3*SR + (g4-3)/4 * SR^2)/n, with g3 the sample
      skewness and g4 the (non-excess) kurtosis. Robust to non-normality,
      a good default for daily strategy returns.
    - ``method='lo'`` — Lo (2002) autocorrelation correction: the variance of
      the mean is inflated by the Bartlett-weighted long-run factor
      1 + 2*sum_{k=1..q} (1 - k/(q+1)) rho_k over the first ``q`` estimated
      autocorrelations (``q=None`` uses the Newey-West rule of thumb
      ``floor(4*(n/100)^(2/9))``), giving Var(SR) = (factor + SR^2/2)/n.

    ``rf`` is a scalar annualised risk-free rate or a per-period aligned
    series (default 0 — historical behaviour); at 2023-26 short rates the
    omission flatters a thin daily edge by ~2 bps/day.

    The t-stat (SR/SE) is invariant to annualisation. A |t| < ~2 means the
    Sharpe is not distinguishable from zero, however pretty the cumulative chart.
    """
    r = np.asarray(_excess_returns(returns, rf, periods_per_year), dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    sd = r.std(ddof=1)
    sr = r.mean() / sd if sd > 0 else np.nan  # per-period

    d = r - r.mean()
    used_q = None
    if method == "iid":
        var = (1.0 + 0.5 * sr**2) / n
    elif method == "mertens":
        m2 = float((d**2).mean())
        g3 = float((d**3).mean()) / m2**1.5 if m2 > 0 else 0.0
        g4 = float((d**4).mean()) / m2**2 if m2 > 0 else 3.0
        var = (1.0 + 0.5 * sr**2 - g3 * sr + (g4 - 3.0) / 4.0 * sr**2) / n
    elif method == "lo":
        used_q = int(q) if q is not None else int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        g0 = float(d @ d) / n
        factor = 1.0
        for k in range(1, min(used_q, n - 1) + 1):
            rho_k = (float(d[k:] @ d[:-k]) / n) / g0 if g0 > 0 else 0.0
            factor += 2.0 * (1.0 - k / (used_q + 1.0)) * rho_k
        var = (max(factor, 0.0) + 0.5 * sr**2) / n
    else:
        raise ValueError("method must be 'iid', 'mertens' or 'lo'")

    se = np.sqrt(max(var, 0.0))
    ann = np.sqrt(periods_per_year)
    return {
        "sharpe_ann": sr * ann,
        "se_ann": se * ann,
        "tstat": sr / se if se > 0 else np.nan,
        "n": n,
        "method": method,
        "q": used_q,
    }


def lo_annualization_factor(returns: pd.Series, q: int = TRADING_DAYS_PER_YEAR) -> float:
    """Lo (2002) autocorrelation-adjusted Sharpe annualisation factor.

    The usual ``SR_ann = sqrt(q) * SR`` assumes serially uncorrelated returns.
    Lo's exact factor for aggregating ``q`` periods is

        q / sqrt(q + 2 * sum_{k=1..q-1} (q - k) * rho_k)

    with ``rho_k`` the lag-k return autocorrelation (estimated here, summed up
    to ``min(q-1, n-2)`` lags). Positive autocorrelation makes the factor
    *smaller* than sqrt(q) (the naive rule overstates the annual Sharpe);
    negative autocorrelation makes it larger. For white noise it reduces to
    sqrt(q). This is provided for citation/sensitivity — none of the library
    defaults are changed by it. Returns NaN if the implied long-run variance
    is non-positive (a degenerate estimate on short, strongly negatively
    autocorrelated samples).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 3:
        return float("nan")
    d = r - r.mean()
    g0 = float(d @ d) / n
    if g0 <= 0:
        return float("nan")
    s = 0.0
    for k in range(1, min(q - 1, n - 2) + 1):
        rho_k = (float(d[k:] @ d[:-k]) / n) / g0
        s += (q - k) * rho_k
    denom = q + 2.0 * s
    if denom <= 0:
        return float("nan")
    return float(q / np.sqrt(denom))


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


def capacity_curve(
    dec: pd.DataFrame,
    ohlc: pd.DataFrame,
    sizes_usd=(1e6, 1e7, 1e8, 1e9, 1e10),
    impact_coef: float = 1.0,
    lookback: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Net overnight edge as a function of deployed capital (order size).

    For each notional in ``sizes_usd``, the participation rate is
    ``(size/price)/ADV``; one-way square-root impact is
    ``impact_coef * daily_vol * sqrt(participation)``; the strategy pays it on
    both legs each night. The net edge is the gross overnight return minus that
    round-trip impact. Where net <= 0, the size is uneconomic.

    Returns a DataFrame indexed by ``size_usd`` with the participation rate,
    round-trip impact (bps) and net edge (bps). Makes the scale argument
    concrete: the edge is gone long before a "market-moving" book is reached.
    """
    edge_bps = dec["r_overnight"].mean() * 1e4
    daily_vol_bps = dec["r_close_close"].std(ddof=1) * 1e4
    cols = {str(c).lower(): c for c in ohlc.columns}
    if "volume" not in cols:
        raise KeyError("capacity_curve needs a 'Volume' column on ohlc.")
    adv = float(ohlc[cols["volume"]].tail(lookback).mean())
    price = float(ohlc[cols["close"]].tail(lookback).mean())
    adv_usd = adv * price

    rows = {}
    for size in sizes_usd:
        participation = size / adv_usd
        impact_rt = 2.0 * impact_coef * daily_vol_bps * np.sqrt(participation)
        rows[float(size)] = {
            "participation_rate": participation,
            "impact_roundtrip_bps": impact_rt,
            "net_edge_bps": edge_bps - impact_rt,
        }
    out = pd.DataFrame(rows).T
    out.index.name = "size_usd"
    return out
