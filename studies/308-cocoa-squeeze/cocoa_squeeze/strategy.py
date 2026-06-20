"""The engine and its honest controls — Study 308 (Cocoa-Squeeze).

After cocoa (CC=F) went parabolic in 2024, two folk reactions compete:

1. **Momentum / "ride the squeeze".** When a market is stretched far above its trend and
   still rising, trend-followers stay long — *the parabola continues*.
2. **Mean-reversion / "fade the blow-off".** When a market is grotesquely stretched above
   its trend, the smart-money trade is to *short the crack* — the parabola reverts.

Both are testable. We define a **stretch** signal — the close's distance above its slow
moving average, in volatility units (a z-scored extension) — and study the *forward*
returns conditional on being deep in a blow-off:

- ``momentum_signal``: long while stretch is high and still rising.
- ``reversion_signal``: short while stretch is extreme (a "fade the parabola" bet).

Look-ahead: the convention is one shift. A signal known at the **close of bar t** earns
the return of bar **t+1** — ``forward_returns`` applies exactly one ``shift``. The stretch
is computed from closes up to and including bar t; the trade is held over t+1.

Costs are charged **one-way × turnover × NAV** (a position flip pays the round trip), and
shorts pay a borrow/financing leg on top. Sharpe is reported on the **strategy's own
returns** (a single asset, no cross-asset race), annualised at 252 days.

The honest control is a **random-sign timer** on the same in-blow-off bars and a
deterministic **synthetic blow-off** with a planted, known truth: the engine must find an
edge when a real parabola is planted, and *not* on a pure random walk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# The stretch / blow-off measure
# ---------------------------------------------------------------------------
def log_returns(close: pd.Series) -> pd.Series:
    """Daily log returns (close-to-close)."""
    return np.log(close).diff()


def stretch_z(close: pd.Series, ma_span: int = 100, vol_span: int = 100) -> pd.Series:
    """Volatility-scaled extension of price above its slow moving average.

    ``stretch = (log P_t - EMA_ma_span(log P)_t) / vol``, where ``vol`` is the EWM standard
    deviation of daily log returns scaled to the MA horizon. A large positive value means
    the price is parabolically far above trend — a blow-off. Stamped on the close of bar t
    (uses only information available at t).
    """
    lp = np.log(close)
    ma = lp.ewm(span=ma_span, adjust=False).mean()
    daily_sigma = log_returns(close).ewm(span=vol_span, adjust=False).std()
    horizon_sigma = daily_sigma * np.sqrt(ma_span)
    return (lp - ma) / horizon_sigma.replace(0.0, np.nan)


# ---------------------------------------------------------------------------
# Signals — the two competing folk reactions
# ---------------------------------------------------------------------------
def momentum_signal(
    close: pd.Series,
    enter_z: float = 1.0,
    ma_span: int = 100,
    vol_span: int = 100,
    rising_lookback: int = 5,
) -> pd.Series:
    """+1 (long) while the market is stretched above trend *and* still rising.

    Fires when ``stretch_z`` exceeds ``enter_z`` and the stretch is higher than it was
    ``rising_lookback`` bars ago (the parabola is still accelerating). Returns a position
    Series in {0, +1} stamped on the close of bar t.
    """
    z = stretch_z(close, ma_span, vol_span)
    rising = z > z.shift(rising_lookback)
    pos = ((z > enter_z) & rising).astype(float)
    return pos.fillna(0.0)


def reversion_signal(
    close: pd.Series,
    enter_z: float = 2.0,
    ma_span: int = 100,
    vol_span: int = 100,
) -> pd.Series:
    """-1 (short) while the market is *extremely* stretched above trend ("fade it").

    Fires when ``stretch_z`` exceeds ``enter_z`` (a grotesque blow-off). Returns a position
    Series in {0, -1} stamped on the close of bar t — the "short the parabola" bet.
    """
    z = stretch_z(close, ma_span, vol_span)
    pos = -(z > enter_z).astype(float)
    return pos.fillna(0.0)


def crack_signal(
    close: pd.Series,
    hi_lookback: int = 250,
    pullback: float = 0.07,
    pullback_window: int = 20,
    hold: int = 20,
) -> pd.Series:
    """-1 (short) for ``hold`` days once a parabola *cracks*: price set a multi-month high
    within ``pullback_window`` days and has since fallen ``pullback`` from its recent max.

    This is the disciplined "fade the blow-off only after it rolls over" version — you do
    not stand in front of a still-rising parabola. Returns a position Series in {0, -1}
    stamped on the close of bar t. Tests whether the *post-peak* crack is tradable.
    """
    recent_max = close.rolling(pullback_window, min_periods=1).max()
    was_at_high = close.shift(1) >= close.shift(1).rolling(hi_lookback, min_periods=1).max() * 0.97
    cracked = (close < recent_max * (1 - pullback)) & was_at_high
    pos = np.zeros(len(close))
    idx = np.where(cracked.to_numpy())[0]
    for i in idx:
        pos[i:min(i + hold, len(close))] = -1.0
    return pd.Series(pos, index=close.index)


def random_sign_positions(template: pd.Series, seed: int = 0) -> pd.Series:
    """A reproducible control: random ±1 on the *same* in-position bars as ``template``.

    Where ``template`` is non-zero (the timer is active), draw a random direction; elsewhere
    flat. Isolates whether the *direction* carries information vs the *timing* alone.
    """
    rng = np.random.default_rng(seed)
    active = template.to_numpy() != 0.0
    out = np.zeros(len(template))
    n = int(active.sum())
    if n:
        out[active] = rng.choice([-1.0, 1.0], size=n)
    return pd.Series(out, index=template.index)


# ---------------------------------------------------------------------------
# Forward-return engine — one execution lag, costs one-way × turnover × NAV
# ---------------------------------------------------------------------------
def book_returns(
    close: pd.Series,
    position: pd.Series,
    cost_bps: float = 0.0,
    borrow_bps_yr: float = 0.0,
) -> pd.Series:
    """Daily net strategy returns from a position series, with ONE execution lag.

    The position known at the close of bar t is held over bar **t+1** (a single
    ``shift(1)``). Costs are charged one-way on turnover: ``cost_bps`` per unit of
    ``|Δposition|`` (a flat→long entry pays once; a +1→−1 flip pays twice). Short exposure
    pays a daily borrow/financing leg of ``borrow_bps_yr / 252`` per unit short.

    Returns a daily net-return Series (simple returns) aligned to ``close.index``.
    """
    simple_ret = close.pct_change()
    held = position.shift(1).fillna(0.0)          # the one documented lag
    gross = held * simple_ret

    turnover = position.diff().abs().fillna(position.abs())
    trade_cost = turnover * (cost_bps * 1e-4)
    borrow_cost = held.clip(upper=0.0).abs() * (borrow_bps_yr * 1e-4 / TRADING_DAYS_PER_YEAR)

    return (gross - trade_cost - borrow_cost).fillna(0.0)


# ---------------------------------------------------------------------------
# Honest statistics — HAC t, block bootstrap CI, Sharpe
# ---------------------------------------------------------------------------
def hac_tstat(returns: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) t-stat of the mean of a daily return series."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 6:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = r.mean()
    e = r - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def block_bootstrap_ci(
    returns: np.ndarray,
    block: int = 20,
    n_boot: int = 2000,
    seed: int = 308,
    stat=np.mean,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Circular block-bootstrap CI of a statistic (default mean daily return).

    Resampling in blocks preserves the autocorrelation/volatility clustering that i.i.d.
    resampling would destroy. Returns the ``(lo, hi)`` percentile interval.
    """
    r = np.asarray(returns, dtype=float)
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
    return (float(np.quantile(stats, alpha / 2)),
            float(np.quantile(stats, 1 - alpha / 2)))


def summarize(returns: pd.Series | np.ndarray) -> dict:
    """Headline daily-return statistics for one strategy book.

    Returns: active-day count, annualised mean (%), annualised Sharpe, HAC t-stat on the
    mean daily return, and the block-bootstrap 95% CI of the mean daily return (bps/day).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n_active = int((r != 0.0).sum())
    n = r.size
    out = {
        "n_days": n,
        "n_active": n_active,
        "ann_return_pct": float(r.mean() * TRADING_DAYS_PER_YEAR * 100) if n else float("nan"),
        "sharpe": float("nan"),
        "tstat": hac_tstat(r),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "ci_lo_bps": float("nan"),
        "ci_hi_bps": float("nan"),
    }
    if n > 1 and r.std(ddof=1) > 0:
        out["sharpe"] = float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    lo, hi = block_bootstrap_ci(r)
    out["ci_lo_bps"] = lo * 1e4 if np.isfinite(lo) else float("nan")
    out["ci_hi_bps"] = hi * 1e4 if np.isfinite(hi) else float("nan")
    return out


# ---------------------------------------------------------------------------
# Event picker — locate the parabolic blow-off window
# ---------------------------------------------------------------------------
def find_blowoff_peak(close: pd.Series, ma_span: int = 100) -> dict:
    """Locate the single most extreme blow-off: the bar of maximum stretch, the prior
    trough, and the forward drawdown from the peak.

    Returns a dict with peak/trough dates, the peak stretch, the run-up multiple from the
    pre-blowoff low, and the worst forward drawdown after the peak.
    """
    z = stretch_z(close, ma_span)
    if z.dropna().empty:
        return {}
    peak_i = int(np.nanargmax(close.to_numpy()))   # the price top (the human "the peak")
    peak_date = close.index[peak_i]
    peak_px = float(close.iloc[peak_i])
    # Pre-peak trough within a 1-year window before the peak.
    lo_start = max(0, peak_i - 252)
    pre = close.iloc[lo_start:peak_i + 1]
    trough_i = int(pre.values.argmin()) + lo_start if len(pre) else peak_i
    trough_px = float(close.iloc[trough_i])
    # Forward drawdown from the peak.
    fwd = close.iloc[peak_i:]
    fwd_dd = float(fwd.min() / peak_px - 1.0) if len(fwd) else float("nan")
    return {
        "peak_date": peak_date,
        "peak_px": peak_px,
        "peak_z": float(z.iloc[peak_i]),
        "trough_date": close.index[trough_i],
        "trough_px": trough_px,
        "runup_mult": peak_px / trough_px if trough_px else float("nan"),
        "fwd_drawdown": fwd_dd,
    }
