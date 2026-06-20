"""The strategy and its honest controls — Study 303 (Uranium-Revival).

The "nuclear renaissance" trade: ride the uranium-miner ETFs (URA, URNM) with a
**trend-following** overlay — be long when the trend is up, in cash when it is down — and
ask whether that beats simply buying and holding the same volatile theme.

The engine is a classic time-series trend rule:

    long when  close > SMA(n)   (canonical n=200)   else   flat (in cash).

A signal known at the close of bar *t* earns the return of bar *t+1* — **one** ``shift``,
applied once, documented here (no calendar-known shortcut applies). Costs are charged
one-way × NAV on every change of position (turnover), so a flip in/out pays once each way.

The decisive question is not "did the strategy make money" (in a boom *everything* made
money) but "did **timing the trend** add anything *over just holding the theme*?" So the
headline statistic is the **excess return of the timed book over buy-and-hold**, and its
HAC (Newey-West) *t*. Because the timed book sits in cash part of the time, the fair race
is **excess-of-cash to excess-of-cash**: we compare the timed book's return *in excess of
the risk-free rate* against buy-and-hold's, so the verdict isn't manufactured by the timed
book dodging risk during drawdowns.

We also expose a **block bootstrap** CI on the timing edge (i.i.d. resampling would
destroy the volatility clustering that dominates a single thematic ETF) and a per-trade
turnover count for the capacity/cost discussion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Indicators & signals
# ---------------------------------------------------------------------------
def sma(close: pd.Series, n: int) -> pd.Series:
    """Simple moving average of period ``n``."""
    return close.rolling(n, min_periods=n).mean()


def trend_signal(bars: pd.DataFrame, n: int = 200) -> pd.Series:
    """Trend-following position: 1.0 when close > SMA(n), else 0.0 (in cash).

    The position is decided at the close of bar *t*. Look-ahead is handled in
    :func:`book_returns` by the single ``shift(1)`` (the position earns *t+1*'s return).
    """
    close = bars["close"]
    pos = (close > sma(close, n)).astype(float)
    return pos.fillna(0.0)


def daily_returns(bars: pd.DataFrame) -> pd.Series:
    """Simple close-to-close daily returns (price/total-return as the tape provides)."""
    return bars["close"].pct_change().fillna(0.0)


# ---------------------------------------------------------------------------
# Books
# ---------------------------------------------------------------------------
def book_returns(
    bars: pd.DataFrame,
    position: pd.Series,
    cost_bps: float = 0.0,
    rf_annual: float = 0.0,
) -> pd.DataFrame:
    """Daily net returns of a timed book vs buy-and-hold, with the cash leg made explicit.

    The position known at the close of *t* (``position``) earns *t+1*'s asset return — a
    single ``shift(1)``. When flat, the book earns the risk-free rate ``rf_annual`` (an
    annual rate, converted to daily). Costs are charged one-way × NAV on the **change** in
    position (``|Δposition|``), so a 0→1→0 round-trip pays ``cost_bps`` twice.

    Returns a frame with columns:
      ``asset``   — the asset's daily return (buy-and-hold leg),
      ``timed``   — the timed book's net daily return (asset when long, rf when flat),
      ``rf``      — the daily risk-free return,
      ``pos``     — the *effective* (lagged) position actually held that day.
    """
    r_asset = daily_returns(bars)
    rf_daily = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0

    held = position.shift(1).fillna(0.0)            # the ONE execution lag
    turnover = held.diff().abs().fillna(held.abs())  # one-way turnover per day
    cost = turnover * cost_bps * 1e-4

    timed = held * r_asset + (1.0 - held) * rf_daily - cost

    return pd.DataFrame(
        {"asset": r_asset, "timed": timed, "rf": rf_daily, "pos": held},
        index=bars.index,
    )


def excess_timing_returns(book: pd.DataFrame) -> pd.Series:
    """The timing edge as a daily return series: (timed − rf) − (asset − rf) = timed − asset.

    Despite the algebra collapsing to ``timed − asset``, framing it as excess-of-cash vs
    excess-of-cash is the honest race: both legs are measured against the same cash
    benchmark, so the timed book gets no free credit for sitting in cash during a
    drawdown beyond the risk-free rate it actually earns there.
    """
    return (book["timed"] - book["rf"]) - (book["asset"] - book["rf"])


# ---------------------------------------------------------------------------
# Honest statistics
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West HAC *t*-stat of the mean of ``x`` (no hard quantlab dependency)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def block_bootstrap_ci(
    x: np.ndarray,
    block: int = 21,
    n_boot: int = 2000,
    seed: int = 303,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the *mean* of ``x`` (annualised, daily input).

    Blocks preserve the volatility clustering an i.i.d. bootstrap would destroy. Returns
    the (lo, hi) bounds of the annualised mean at level ``1 - alpha``.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < block + 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]) % n
        sample = x[idx.ravel()[:n]]
        means[b] = sample.mean()
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (float(lo * TRADING_DAYS_PER_YEAR), float(hi * TRADING_DAYS_PER_YEAR))


# ---------------------------------------------------------------------------
# Performance summary
# ---------------------------------------------------------------------------
def _ann_return(daily: np.ndarray) -> float:
    daily = np.asarray(daily, dtype=float)
    if daily.size == 0:
        return float("nan")
    return float((1.0 + daily).prod() ** (TRADING_DAYS_PER_YEAR / daily.size) - 1.0)


def _ann_sharpe(excess_daily: np.ndarray) -> float:
    excess_daily = np.asarray(excess_daily, dtype=float)
    sd = excess_daily.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return float("nan")
    return float(excess_daily.mean() / sd * np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(daily: np.ndarray) -> float:
    eq = np.cumprod(1.0 + np.asarray(daily, dtype=float))
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def summarize(book: pd.DataFrame) -> dict:
    """Headline statistics: timed vs buy-and-hold, the timing edge, and its inference.

    The decisive numbers:
      * ``timing_t`` — HAC *t* of the daily timing edge (timed − asset). The inference-bar
        statistic: trend-timing is a REAL improvement on buy-and-hold only if |t| ≥ 2.
      * ``timed_sharpe`` / ``bh_sharpe`` — **excess-of-cash** Sharpe of each book (the fair
        race), and ``sharpe_diff``.
      * ``timing_ci`` — block-bootstrap CI on the annualised timing edge.
    """
    timed = book["timed"].to_numpy()
    asset = book["asset"].to_numpy()
    rf = book["rf"].to_numpy()
    edge = excess_timing_returns(book).to_numpy()

    return {
        "n_days": int(len(book)),
        "pct_time_in_market": float(book["pos"].mean() * 100.0),
        "n_round_trips": int(round(book["pos"].diff().abs().fillna(0).sum() / 2.0)),
        "timed_ann": _ann_return(timed),
        "bh_ann": _ann_return(asset),
        "timing_edge_ann": _ann_return(timed) - _ann_return(asset),
        "timed_sharpe": _ann_sharpe(timed - rf),
        "bh_sharpe": _ann_sharpe(asset - rf),
        "sharpe_diff": _ann_sharpe(timed - rf) - _ann_sharpe(asset - rf),
        "timed_maxdd": _max_drawdown(timed),
        "bh_maxdd": _max_drawdown(asset),
        "timing_t": hac_tstat(edge),
        "timing_ci": block_bootstrap_ci(edge),
    }


# ---------------------------------------------------------------------------
# Cost sweep helper
# ---------------------------------------------------------------------------
def cost_sweep(
    bars: pd.DataFrame,
    n: int = 200,
    costs_bps=(0.0, 5.0, 10.0, 20.0, 50.0),
    rf_annual: float = 0.0,
) -> pd.DataFrame:
    """Net timing edge and its HAC *t* across a sweep of one-way costs."""
    pos = trend_signal(bars, n=n)
    rows = []
    for c in costs_bps:
        book = book_returns(bars, pos, cost_bps=c, rf_annual=rf_annual)
        s = summarize(book)
        rows.append(
            {
                "cost_bps": c,
                "timed_ann": s["timed_ann"],
                "edge_ann": s["timing_edge_ann"],
                "timing_t": s["timing_t"],
                "sharpe_diff": s["sharpe_diff"],
            }
        )
    return pd.DataFrame(rows)
