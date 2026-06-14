"""The strategy and its honest controls — Study 128 (Keltner-Channel).

The Keltner Channel is built as:
  - Middle band: EMA(20) of close
  - Upper band:  EMA(20) + 2 * ATR(10)
  - Lower band:  EMA(20) - 2 * ATR(10)

Two contradictory folk rules are tested:

1. **Breakout** (momentum): close pierces the *upper* channel → buy long.
   The claim: price is trending and will continue upward.

2. **Reversion**: close pierces the *lower* channel → buy long.
   The claim: price is "too far" from the channel and will snap back.

The contradiction is the point: if the channel encodes meaningful "too far" information
then *one* of these must be wrong — or both are noise.  We implement both and pin each
against two controls:

- **Random-entry control** — random dates on the same tape, same hold length, so the
  baseline reflects the instrument's drift, not the channel signal.  This is the key
  comparison: does the channel pierce *add* anything beyond randomly sampling the tape?

- **Buy-and-hold baseline** — the instrument's unconditional daily return over the same
  window, for context.

Exit regime: a fixed **20-bar time-exit** (marked at close), so both rules are evaluated
on the same horizon and the comparison is clean.  A symmetric ±1 ATR barrier version is
also provided for the quant teardown.

No look-ahead: the channel pierce is detected on the close of day *t*; the trade is
entered at day *t+1*'s open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Keltner Channel indicators
# ---------------------------------------------------------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with ``span`` (the standard ``com = (span-1)/2`` form)."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def atr(bars: pd.DataFrame, n: int = 10) -> pd.Series:
    """Average True Range over ``n`` bars (simple rolling mean of TR, Wilder-style).

    TR = max(high-low, |high-prev_close|, |low-prev_close|).  The first ``n-1`` values
    are NaN.
    """
    prev_c = bars["close"].shift(1)
    tr = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_c).abs(),
            (bars["low"] - prev_c).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def keltner_channel(
    bars: pd.DataFrame,
    ema_span: int = 20,
    atr_n: int = 10,
    n_atr: float = 2.0,
) -> pd.DataFrame:
    """Classic Keltner Channel: EMA(ema_span) +/- n_atr * ATR(atr_n).

    Returns a DataFrame with columns ``mid``, ``upper``, ``lower``, valid from
    bar ``max(ema_span, atr_n)`` onward (earlier rows are NaN).
    """
    mid = ema(bars["close"], span=ema_span)
    a = atr(bars, n=atr_n)
    band = n_atr * a
    return pd.DataFrame(
        {"mid": mid, "upper": mid + band, "lower": mid - band},
        index=bars.index,
    )


# ---------------------------------------------------------------------------
# Entry signals
# ---------------------------------------------------------------------------
def channel_entries(
    bars: pd.DataFrame,
    ema_span: int = 20,
    atr_n: int = 10,
    n_atr: float = 2.0,
    side: str = "breakout",
) -> pd.DataFrame:
    """Bars where the close pierces a Keltner Channel band.

    ``side="breakout"``  → close > upper band (momentum signal, long entry).
    ``side="reversion"`` → close < lower band (mean-reversion signal, long entry).

    The pierce is detected on the *current* close; the trade is always entered at
    the *next* bar's open by :func:`run_trades` — no look-ahead.

    Only the *first* bar of each consecutive outside-band run is kept (the initial
    pierce day), so overlapping entries are suppressed.

    Returns a DataFrame with index = signal date and column ``dir`` (always +1 for
    our long-only tests).
    """
    if side not in ("breakout", "reversion"):
        raise ValueError(f"side must be 'breakout' or 'reversion', got {side!r}")

    kc = keltner_channel(bars, ema_span=ema_span, atr_n=atr_n, n_atr=n_atr)
    close = bars["close"]

    if side == "breakout":
        mask = close > kc["upper"]
    else:
        mask = close < kc["lower"]

    # Keep only the first bar of each consecutive run (the pierce day).
    first_pierce = mask & ~mask.shift(1, fill_value=False)
    idx = close.index[first_pierce & kc["upper"].notna()]
    return pd.DataFrame({"dir": np.ones(len(idx), dtype=int)}, index=idx)


def random_entries(
    bars: pd.DataFrame,
    n: int,
    seed: int = 0,
    warmup: int = 20,
) -> pd.DataFrame:
    """``n`` random entry dates from the tape, respecting the warm-up period.

    The dates are sampled uniformly from bars[warmup:] (after the channel warm-up),
    so the distribution is fair.  Returns a DataFrame with a ``dir`` column.
    """
    rng = np.random.default_rng(seed)
    valid = bars.index[warmup:]
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    chosen = pd.DatetimeIndex(sorted(chosen))
    return pd.DataFrame({"dir": np.ones(len(chosen), dtype=int)}, index=chosen)


# ---------------------------------------------------------------------------
# Forward-return engine
# ---------------------------------------------------------------------------
def run_trades(
    bars: pd.DataFrame,
    entries: pd.DataFrame,
    hold_days: int = 20,
    cost_bps: float = 2.0,
) -> pd.DataFrame:
    """Run fixed-hold forward trades and return a per-trade ledger.

    For each entry date the position is entered at the *next* bar's open and held
    for exactly ``hold_days`` calendar bars (marked out at the ``hold_days``-th
    close, or the last available bar).  Using a fixed hold for both arms makes the
    breakout vs reversion vs random comparison apple-to-apple.

    ``cost_bps`` is a one-way-per-leg round-trip cost charged on the gross return.

    Columns: ``entry_date, entry, exit_date, exit, bars_held, ret_gross, ret_net``.
    """
    close = bars["close"]
    open_ = bars["open"]
    dates = bars.index.tolist()
    pos = {d: i for i, d in enumerate(dates)}
    n_bars = len(bars)

    rows = []
    for sig_date in entries.index:
        i = pos.get(sig_date)
        if i is None or i + 1 >= n_bars:
            continue
        e = i + 1                        # enter at next bar's open
        last = min(e + hold_days - 1, n_bars - 1)
        entry_px = open_.iat[e]
        exit_px = close.iat[last]
        ret_gross = (exit_px - entry_px) / entry_px
        rows.append(
            {
                "entry_date": bars.index[e],
                "entry": entry_px,
                "exit_date": bars.index[last],
                "exit": exit_px,
                "bars_held": last - e + 1,
                "ret_gross": ret_gross,
                "ret_net": ret_gross - cost_bps * 1e-4,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trade-ledger summary
# ---------------------------------------------------------------------------
def summarize(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline per-trade statistics for one ledger.

    Returns trade count, win-rate, mean return (bps/trade), the per-trade Sharpe, the
    P&L skew, and a HAC (Newey-West) t-stat on the mean — the inference-bar number
    that decides whether the edge is distinguishable from zero.
    """
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    out = {
        "n_trades": int(n),
        "win_rate": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "sharpe_per_trade": (
            float(r.mean() / r.std(ddof=1)) if n > 1 and r.std() > 0 else float("nan")
        ),
        "skew": float(pd.Series(r).skew()) if n > 2 else float("nan"),
        "tstat": float("nan"),
    }
    if n > 5:
        # Newey-West HAC t-stat (no hard quantlab dependency in this hot path).
        mu = r.mean()
        e = r - mu
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        lrv = float(e @ e) / n
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        out["tstat"] = float(mu / se) if se > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Buy-and-hold baseline
# ---------------------------------------------------------------------------
def buy_and_hold_mean_bps(bars: pd.DataFrame, hold_days: int = 20) -> float:
    """Mean ``hold_days``-day forward return (bps) for all bars, as a baseline.

    This is the unconditional forward return any randomly-timed long position earns;
    it equals the random-entry control in expectation.
    """
    close = bars["close"]
    fwd = close.shift(-hold_days) / close - 1.0
    return float(fwd.dropna().mean() * 1e4)
