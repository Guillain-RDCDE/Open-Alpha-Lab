"""The engine and its honest controls — Study 334 (ARK-Innovation).

Three questions, three tools:

1. **The behaviour gap (the headline).** ``dollar_weighted_return`` computes the
   money-weighted IRR of the fund's actual cashflows (flows in, terminal value out) and
   ``time_weighted_return`` the buy-and-hold CAGR. The *gap* between them is what
   performance-chasing flows cost the average investor — the "buy-the-top" penalty. This
   is an accounting identity on flows + prices; no fitting, no free parameters.

2. **Trend vs mean-reversion (was it a momentum win?).** ``ma_crossover_signal`` and
   ``mean_reversion_signal`` turn the daily price tape into a long/flat overlay; the
   forward returns are scored with a HAC *t* and an excess-of-benchmark Sharpe.

3. **Honest inference.** ``hac_tstat`` (Newey-West), ``block_bootstrap_ci`` (circular
   block bootstrap, which respects volatility clustering), ``net_returns`` (one-way costs
   × NAV, one documented execution lag), and ``excess_sharpe`` (excess-of-cash to
   excess-of-cash when racing a benchmark).

Execution-lag convention: a signal known at the close of day *t* earns the return of day
*t+1* — one ``shift(1)``, applied once, in ``overlay_returns``. Calendar-known flows need
no lag (a month's net subscription is a fact of that month).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12


# ===========================================================================
# 1 · The behaviour gap — time-weighted vs dollar-weighted return
# ===========================================================================
def time_weighted_return(price: pd.Series, periods_per_year: int) -> dict:
    """Buy-and-hold geometric (CAGR) return of a price/NAV path.

    This is what a single dollar invested at the start and held to the end earned — the
    fund's *advertised* return, independent of when other people's money came and went.
    """
    p = price.dropna().to_numpy(dtype=float)
    if p.size < 2 or p[0] <= 0:
        return {"total_return": float("nan"), "cagr": float("nan"), "n_periods": int(p.size)}
    total = p[-1] / p[0] - 1.0
    years = (p.size - 1) / periods_per_year
    cagr = (p[-1] / p[0]) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    return {"total_return": float(total), "cagr": float(cagr), "n_periods": int(p.size)}


def _npv(rate: float, cashflows: np.ndarray) -> float:
    """Net present value of dated cashflows at a per-period ``rate`` (period 0..N-1).

    Discounting in log-space keeps long horizons (180+ periods) from overflowing when a
    bracketing search probes a near-(-1) rate; an out-of-domain ``1 + rate <= 0`` returns
    a large NPV with the sign of the terminal cashflow so the bisector steps away from it.
    """
    one_plus = 1.0 + rate
    if one_plus <= 0.0:
        return float(np.sign(cashflows[-1]) * 1e18)
    t = np.arange(cashflows.size, dtype=float)
    disc = cashflows * np.exp(-t * np.log(one_plus))
    return float(np.sum(disc))


def money_weighted_irr(cashflows: np.ndarray, periods_per_year: int) -> float:
    """The internal rate of return (per *year*) of a stream of dated cashflows.

    ``cashflows`` is signed from the *investor's* point of view: negative when money goes
    into the fund (a subscription), positive when it comes out (a redemption or the
    terminal value). Solved by bisection on the per-period IRR over a sane bracket
    ``(-0.95, 1.0)`` (per-*period* rates outside that are not economically meaningful for
    a monthly fund series), then annualised; returns NaN if no sign change brackets a root.
    """
    cf = np.asarray(cashflows, dtype=float)
    if cf.size < 2 or not (np.any(cf > 0) and np.any(cf < 0)):
        return float("nan")
    lo, hi = -0.95, 1.0
    f_lo, f_hi = _npv(lo, cf), _npv(hi, cf)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or np.sign(f_lo) == np.sign(f_hi):
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = _npv(mid, cf)
        if abs(f_mid) < 1e-10:
            break
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    period_irr = 0.5 * (lo + hi)
    return float((1.0 + period_irr) ** periods_per_year - 1.0)


def cashflows_from_flows(price: pd.Series, flow: pd.Series) -> np.ndarray:
    """Build the investor cashflow stream from a NAV path and net dollar flows.

    Sign convention (investor's view): a net subscription this period is *negative* (cash
    leaves the investor), and the fund's terminal market value is a single positive
    cashflow at the end (everyone redeems). The terminal value is the running invested
    *units* (flows ÷ NAV) marked at the final NAV — the dollars an average holder could
    walk away with. This is the standard money-weighted construction (Dichev 2007).
    """
    p = price.to_numpy(dtype=float)
    f = flow.to_numpy(dtype=float)
    n = min(p.size, f.size)
    p, f = p[:n], f[:n]
    units = np.cumsum(f / p)  # net units bought with each period's dollar flow
    terminal_value = units[-1] * p[-1]
    cf = -f.astype(float).copy()  # subscriptions are negative cashflows to the investor
    cf[-1] += terminal_value  # redeem everything at the end
    return cf


def dollar_weighted_return(price: pd.Series, flow: pd.Series, periods_per_year: int) -> float:
    """Annualised money-weighted (IRR) return the *average dollar* in the fund earned."""
    cf = cashflows_from_flows(price, flow)
    return money_weighted_irr(cf, periods_per_year)


def behaviour_gap(price: pd.Series, flow: pd.Series, periods_per_year: int) -> dict:
    """The headline: time-weighted CAGR, dollar-weighted IRR, and their gap (TWR − DWR).

    A *positive* gap means the buy-and-hold return beat the average investor's realised
    return — the textbook performance-chasing penalty. A gap near zero means flows were
    not return-chasing (the null).
    """
    twr = time_weighted_return(price, periods_per_year)["cagr"]
    dwr = dollar_weighted_return(price, flow, periods_per_year)
    return {
        "twr_cagr": float(twr),
        "dwr_irr": float(dwr),
        "gap": float(twr - dwr) if np.isfinite(twr) and np.isfinite(dwr) else float("nan"),
    }


# ===========================================================================
# 2 · Trend vs mean-reversion overlays
# ===========================================================================
def ma_crossover_signal(close: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """Long/flat trend signal: 1 when fast SMA > slow SMA (a momentum stance), else 0.

    Stamped on the close of bar *t*; the one-day execution lag is applied in
    ``overlay_returns``, not here.
    """
    f = close.rolling(fast, min_periods=fast).mean()
    s = close.rolling(slow, min_periods=slow).mean()
    return (f > s).astype(float).where(s.notna(), 0.0).fillna(0.0)


def ts_momentum_signal(close: pd.Series, lookback: int = 1) -> pd.Series:
    """Long/flat time-series-momentum signal: 1 when the trailing ``lookback``-day return
    is positive (ride the persistence), else 0.

    Unlike the slow MA crossover, this directly harvests short-horizon return
    *persistence* — the cleanest test of "was it a momentum win?". Stamped on the close of
    bar *t*; the one-day execution lag is applied in ``overlay_returns``.
    """
    trailing = close.pct_change().rolling(lookback, min_periods=lookback).sum()
    return (trailing > 0).astype(float).where(trailing.notna(), 0.0).fillna(0.0)


def mean_reversion_signal(close: pd.Series, lookback: int = 5, z: float = 1.0) -> pd.Series:
    """Long/flat mean-reversion signal: 1 when the close is ``z`` std *below* its rolling
    mean (oversold), else 0. The contrarian counterpart to the trend overlay."""
    mu = close.rolling(lookback, min_periods=lookback).mean()
    sd = close.rolling(lookback, min_periods=lookback).std(ddof=0)
    zscore = (close - mu) / sd.replace(0.0, np.nan)
    return (zscore < -z).astype(float).where(zscore.notna(), 0.0).fillna(0.0)


def overlay_returns(
    close: pd.Series,
    signal: pd.Series,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Daily returns of a long/flat overlay driven by ``signal``.

    Execution lag: the signal known at the close of *t* is held for the return of *t+1*
    (one ``shift(1)``, applied once here). Costs are charged one-way × NAV on every change
    in position (turnover = |Δposition|, counted both legs by definition of |Δ|), in bps.

    Returns columns: ``asset_ret`` (buy-and-hold daily return), ``position`` (lagged),
    ``gross``, ``net``.
    """
    asset_ret = close.pct_change().fillna(0.0)
    position = signal.shift(1).fillna(0.0)  # the ONE documented execution lag
    gross = position * asset_ret
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * cost_bps * 1e-4
    net = gross - cost
    return pd.DataFrame(
        {"asset_ret": asset_ret, "position": position, "gross": gross, "net": net}
    )


# ===========================================================================
# 3 · Honest inference
# ===========================================================================
def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) t-stat of the mean of ``x`` against zero.

    No hard dependency on quantlab being importable — the desk's standard automatic
    bandwidth ``floor(4 (n/100)^(2/9))`` is used when ``lags`` is None.
    """
    r = np.asarray(x, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 6:
        return float("nan")
    mu = r.mean()
    e = r - mu
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def block_bootstrap_ci(
    x: np.ndarray,
    stat=np.mean,
    block: int = 21,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 334,
) -> tuple[float, float]:
    """Circular block-bootstrap confidence interval for ``stat`` of ``x``.

    Blocks (default ~one trading month) preserve the autocorrelation/volatility
    clustering that i.i.d. resampling would destroy.
    """
    r = np.asarray(x, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < block + 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        stats[b] = stat(r[idx[:n]])
    lo = float(np.quantile(stats, alpha / 2.0))
    hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
    return (lo, hi)


def sharpe(returns: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised Sharpe ratio of a return series (no risk-free subtraction)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def excess_sharpe(
    strat_ret: np.ndarray,
    bench_ret: np.ndarray,
    rf_daily: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """Excess-of-cash to excess-of-cash comparison when a strategy races a benchmark.

    A long/flat overlay sits in cash part of the time, so comparing its raw Sharpe with a
    fully-invested benchmark's raw Sharpe is the apples-to-oranges trap the desk warns
    about. We subtract the same ``rf_daily`` from both legs and report each excess Sharpe
    plus the difference.
    """
    s = np.asarray(strat_ret, dtype=float) - rf_daily
    b = np.asarray(bench_ret, dtype=float) - rf_daily
    return {
        "strat_excess_sharpe": sharpe(s, periods_per_year),
        "bench_excess_sharpe": sharpe(b, periods_per_year),
        "diff": sharpe(s, periods_per_year) - sharpe(b, periods_per_year),
    }


def summarize_overlay(
    daily: pd.DataFrame,
    col: str = "net",
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """Headline stats for an overlay return series: mean (bps/day), Sharpe, HAC t, and
    fraction of days invested."""
    r = daily[col].to_numpy(dtype=float)
    return {
        "n_days": int(np.isfinite(r).sum()),
        "mean_bps": float(np.nanmean(r) * 1e4),
        "sharpe": sharpe(r, periods_per_year),
        "hac_t": hac_tstat(r),
        "pct_invested": float((daily["position"] > 0).mean()) if "position" in daily else float("nan"),
    }
