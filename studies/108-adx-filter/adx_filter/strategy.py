"""Strategy and honest controls — Study 108 (ADX-Filter).

The folk rule under test: *"Only trade a trend rule when ADX(14) > 25 — that means
the trend is strong enough to trade.  When ADX ≤ 25 stay flat."*  We decompose this
into a meta-study:

    GATED rule   = enter when trend signal fires AND ADX(14) > 25
    UNGATED rule = enter whenever the trend signal fires (ADX ignored)
    RANDOM entry = identical entry dates as UNGATED, direction drawn by a fair coin

The trend signal itself is a **20/50 MA cross** (simple, widely cited) on daily bars.
We hold for a fixed ``hold_days`` days (default 20) or until end of data.  A
symmetric ±1 ATR(20) barrier variant is also available for the honest signal test.

Three comparisons decide the verdict:

1. UNGATED vs RANDOM — does the MA cross itself have any directional signal at all?
2. GATED vs UNGATED  — does the ADX filter improve over the unfiltered rule?
3. GATED vs RANDOM   — does the gated rule beat a coin? (the direct fair test)

If ADX filtering adds nothing we expect GATED ≈ UNGATED, both ≈ RANDOM, and the
entire three-way gap is noise.

No look-ahead: the signal (ADX + MA cross) is formed on closes up to day *t*; the
trade is entered at *t+1*'s open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def sma(close: pd.Series, n: int) -> pd.Series:
    """Simple moving average with a clean warm-up (NaN for the first n−1 bars)."""
    return close.rolling(n, min_periods=n).mean()


def atr(bars: pd.DataFrame, n: int = 20) -> pd.Series:
    """Wilder-style average true range, in price units."""
    prev_close = bars["close"].shift(1)
    tr = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def adx(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder's Average Directional Index (ADX), period ``n``.

    ADX measures the *strength* of a trend (directional movement) on a 0–100 scale.
    A reading above 25 is the canonical "strong trend" threshold; below 20–25 is
    "weak or ranging".  The indicator is direction-agnostic — it does not say whether
    the trend is up or down.

    Implementation follows the standard Wilder (1978) recipe:
      1. Compute True Range (TR), +DM, −DM.
      2. Smooth with Wilder's exponential (α = 1/n) for ATR, +DI, −DI.
      3. DX = |+DI − −DI| / (+DI + −DI) * 100.
      4. ADX = Wilder-smooth(DX, n).
    """
    high = bars["high"]
    low = bars["low"]
    close = bars["close"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    # True Range
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    # Raw directional movements
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=bars.index, dtype=float)
    minus_dm_s = pd.Series(minus_dm, index=bars.index, dtype=float)

    # Wilder's exponential smoothing: equivalent to ewm with alpha=1/n, adjust=False.
    # We require at least n prior bars for the initial sum (standard behaviour).
    alpha = 1.0 / n
    atr_w = tr.ewm(alpha=alpha, adjust=False, min_periods=n).mean() * n
    plus_di = plus_dm_s.ewm(alpha=alpha, adjust=False, min_periods=n).mean() * n / atr_w * 100
    minus_di = minus_dm_s.ewm(alpha=alpha, adjust=False, min_periods=n).mean() * n / atr_w * 100

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx_val = dx.ewm(alpha=alpha, adjust=False, min_periods=n).mean()
    return adx_val


def ma_cross_entries(
    close: pd.Series,
    fast: int = 20,
    slow: int = 50,
) -> pd.DataFrame:
    """Bars where SMA(fast) crosses SMA(slow).

    ``dir`` = +1 for an up-cross (long), -1 for a down-cross (short/flat).
    The cross is detected on the sign change of (fast − slow) between consecutive
    closes — known at the close of the crossover bar, traded at the next bar's open.
    No look-ahead.
    """
    f = sma(close, fast)
    s = sma(close, slow)
    diff = f - s
    sign = np.sign(diff)
    crossed = sign.ne(sign.shift(1)) & sign.ne(0) & sign.shift(1).notna()
    out = pd.DataFrame({"dir": sign[crossed].astype(int)})
    out.index.name = close.index.name
    return out


def donchian_entries(
    bars: pd.DataFrame,
    n: int = 20,
) -> pd.DataFrame:
    """Donchian channel breakout entries (alternative trend signal).

    Long when today's close sets a new ``n``-day high (bullish breakout);
    short when it sets a new ``n``-day low (bearish breakout).  Lookback is
    over the prior ``n`` bars (no look-ahead: today's close is excluded from
    the prior window).
    """
    close = bars["close"]
    high_n = close.shift(1).rolling(n, min_periods=n).max()
    low_n = close.shift(1).rolling(n, min_periods=n).min()
    bull = close > high_n
    bear = close < low_n
    # Only emit a signal on the *first* bar of each breakout to avoid stacking trades.
    bull_new = bull & ~bull.shift(1).fillna(False)
    bear_new = bear & ~bear.shift(1).fillna(False)
    signal = pd.Series(0, index=close.index, dtype=int)
    signal[bull_new] = 1
    signal[bear_new] = -1
    entries = signal[signal != 0].rename("dir")
    return pd.DataFrame({"dir": entries})


def apply_adx_gate(
    entries: pd.DataFrame,
    adx_series: pd.Series,
    threshold: float = 25.0,
) -> pd.DataFrame:
    """Return only the subset of entries where ADX ≥ ``threshold`` on the signal bar.

    ADX is measured at the signal bar *t* (i.e. known at the close of that bar —
    no look-ahead), then the trade is placed at *t+1*'s open by :func:`run_trades`.
    """
    adx_at_signal = adx_series.reindex(entries.index)
    mask = adx_at_signal >= threshold
    return entries[mask].copy()


def random_directions(n: int, seed: int = 0) -> np.ndarray:
    """A reproducible vector of ±1 — the control arm's coin."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=n)


# ---------------------------------------------------------------------------
# Forward-return engine (hold for N bars)
# ---------------------------------------------------------------------------
def run_trades(
    bars: pd.DataFrame,
    entries: pd.DataFrame,
    hold_days: int = 20,
    cost_bps: float = 1.0,
    directions: np.ndarray | None = None,
) -> pd.DataFrame:
    """Hold-N-day forward-return engine.  Returns a per-trade ledger.

    For each entry bar the trade is entered at the *next* bar's open and closed at
    the close of the ``hold_days``-th bar after entry (or at the last available bar
    if the tape is shorter).

    ``directions`` overrides the entry signs (the random-control arm passes a ±1
    vector aligned to ``entries``).  ``cost_bps`` is a round-trip cost charged as a
    flat hit on the net return.

    Columns: ``entry_ts, dir, entry, exit, bars_held, ret_gross, ret_net``.
    """
    close = bars["close"]
    open_ = bars["open"]
    pos = {ts: i for i, ts in enumerate(bars.index)}

    dirs = (
        np.asarray(directions, dtype=int)
        if directions is not None
        else entries["dir"].to_numpy(dtype=int)
    )

    rows = []
    n_bars = len(bars)
    for sig_ts, d in zip(entries.index, dirs):
        i = pos.get(sig_ts)
        if i is None or i + 1 >= n_bars:
            continue
        e = i + 1  # enter at the next bar's open
        exit_i = min(e + hold_days - 1, n_bars - 1)
        entry_px = open_.iat[e]
        exit_px = close.iat[exit_i]
        ret_gross = d * (exit_px - entry_px) / entry_px
        rows.append(
            {
                "entry_ts": bars.index[e],
                "dir": int(d),
                "entry": entry_px,
                "exit": exit_px,
                "bars_held": exit_i - e + 1,
                "ret_gross": ret_gross,
                "ret_net": ret_gross - cost_bps * 1e-4,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ATR-barrier engine (symmetric ±1 ATR exits)
# ---------------------------------------------------------------------------
def run_trades_barrier(
    bars: pd.DataFrame,
    entries: pd.DataFrame,
    tp_R: float = 1.0,
    sl_R: float = 1.0,
    atr_n: int = 20,
    cost_bps: float = 1.0,
    max_hold: int = 60,
    directions: np.ndarray | None = None,
) -> pd.DataFrame:
    """ATR-barrier trades — the honest symmetric payoff for the signal test.

    Enter at *t+1*'s open; risk unit R = ATR(atr_n) at the signal bar.
    Take-profit at entry ± tp_R*R, stop at entry ∓ sl_R*R.
    When both are hit in the same bar the stop wins (conservative fill).
    Positions never held beyond ``max_hold`` bars (closed at that bar's close).

    Columns: ``entry_ts, dir, entry, exit, exit_reason, bars_held, ret_gross, ret_net``.
    """
    close = bars["close"]
    open_ = bars["open"]
    high = bars["high"]
    low = bars["low"]
    r_unit = atr(bars, atr_n)
    pos = {ts: i for i, ts in enumerate(bars.index)}

    dirs = (
        np.asarray(directions, dtype=int)
        if directions is not None
        else entries["dir"].to_numpy(dtype=int)
    )

    rows = []
    n_bars = len(bars)
    for sig_ts, d in zip(entries.index, dirs):
        i = pos.get(sig_ts)
        if i is None or i + 1 >= n_bars:
            continue
        e = i + 1
        R = r_unit.iat[i]
        if not np.isfinite(R) or R <= 0:
            continue
        entry_px = open_.iat[e]
        tp = entry_px + d * tp_R * R
        sl = entry_px - d * sl_R * R

        exit_px = exit_reason = None
        last = e
        end = min(e + max_hold, n_bars)
        for j in range(e, end):
            hi, loo = high.iat[j], low.iat[j]
            hit_sl = (loo <= sl) if d > 0 else (hi >= sl)
            hit_tp = (hi >= tp) if d > 0 else (loo <= tp)
            if hit_sl:
                exit_px, exit_reason = sl, "sl"
                last = j
                break
            if hit_tp:
                exit_px, exit_reason = tp, "tp"
                last = j
                break
            last = j
        if exit_px is None:
            exit_px, exit_reason = close.iat[last], "eod"

        ret_gross = d * (exit_px - entry_px) / entry_px
        rows.append(
            {
                "entry_ts": bars.index[e],
                "dir": int(d),
                "entry": entry_px,
                "exit": exit_px,
                "exit_reason": exit_reason,
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
    P&L skew, and a HAC Newey-West t-stat on the mean — the inference-bar number that
    decides whether the edge is distinguishable from zero.
    """
    if ledger is None or (hasattr(ledger, "empty") and ledger.empty):
        return {k: float("nan") for k in
                ["n_trades", "win_rate", "mean_bps", "sharpe_per_trade", "skew", "tstat"]}
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
        # Newey-West HAC t-stat — same kernel as Study 72 for apples-to-apples.
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
