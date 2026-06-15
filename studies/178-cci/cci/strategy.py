"""The strategy and its honest controls — Study 178 (CCI).

Donald Lambert's Commodity Channel Index (CCI, 1980) is a normalised oscillator that
measures how far the typical price has strayed from its n-period average, scaled by a
robust dispersion measure (mean absolute deviation):

    TP(t)      = (high + low + close) / 3
    SMA_TP(n)  = simple moving average of TP over the trailing n bars
    MAD_TP(n)  = mean absolute deviation of TP over the trailing n bars
    CCI(t)     = (TP(t) - SMA_TP(n)) / (0.015 * MAD_TP(n))

The 0.015 constant was chosen by Lambert so that ~70–80% of CCI values fall between
−100 and +100 on commodity futures.  The folk rules on equities:

- CCI crosses above +100  → "overbought" → *sell / go short*, expect a pullback.
- CCI crosses below −100  → "oversold"   → *buy*, expect a bounce.

Two complementary framings are tested — both with long-flat (no short side) AND
long-short variants:

1. **Breach framing** — enter on the *first* bar where CCI enters the extreme zone
   (crosses above +100 or below −100).  One entry per episode.
2. **Cross-back framing** — enter when CCI *exits* the extreme zone (crosses back above
   −100 or below +100).  The "confirmed bounce / fade" version.

Both framings use the same honest comparison: the **same entry bars, random direction**
(a fair coin), so the verdict is "does CCI add directional information over a coin?"

No look-ahead: all indicators computed on closes / TP up to bar *t*; trade entered at
*t+1*'s open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CCI indicator
# ---------------------------------------------------------------------------
def cci(bars: pd.DataFrame, period: int = 20) -> pd.Series:
    """Lambert's Commodity Channel Index over a rolling ``period`` window.

    CCI(t) = (TP(t) - SMA_TP(period)) / (0.015 * MAD_TP(period))

    where TP = (high + low + close) / 3, SMA_TP is the rolling mean of TP, and
    MAD_TP is the mean absolute deviation of TP — all computed over the same single
    trailing ``period``-bar window.  Returns a Series indexed identically to ``bars``.

    The 0.015 factor normalises the statistic so that roughly 70–80% of readings on
    commodity futures fall inside [−100, +100].  On equity daily bars the distribution
    is similar; the ±100 thresholds serve as the conventional overbought/oversold triggers.

    NaN is returned for the first ``period - 1`` bars where the rolling window is not yet
    fully populated.

    Implementation note: the SMA and MAD are computed together in a single ``apply`` over
    the rolling ``period``-bar window to avoid the double-rolling artefact that would occur
    if SMA and MAD were each computed with separate ``rolling().mean()`` calls.
    """
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    tp_arr = tp.to_numpy(dtype=float)
    n = len(tp_arr)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = tp_arr[i - period + 1 : i + 1]  # length = period
        mean = window.mean()
        mad = np.abs(window - mean).mean()
        if mad == 0.0:
            result[i] = 0.0
        else:
            result[i] = (tp_arr[i] - mean) / (0.015 * mad)
    return pd.Series(result, index=bars.index, name="cci")


# ---------------------------------------------------------------------------
# Signal generators — two framings
# ---------------------------------------------------------------------------
def breach_entries(
    cci_series: pd.Series,
    oversold: float = -100.0,
    overbought: float = 100.0,
) -> pd.DataFrame:
    """Entry signals when CCI *first enters* an extreme zone (breach / zone-entry).

    A **buy signal** (+1) fires on the first bar where CCI falls below ``oversold``
    (previous bar was at or above the threshold).  A **sell signal** (-1) fires on the
    first bar where CCI rises above ``overbought`` (previous bar was at or below).  One
    entry fires per episode — back-to-back bars deep in the zone do not generate
    repeated signals.

    Signals are known at the close of bar *t*; trades enter at the *next* bar's open.
    """
    prev = cci_series.shift(1)
    enter_oversold = (cci_series < oversold) & (prev >= oversold) & prev.notna()
    enter_overbought = (cci_series > overbought) & (prev <= overbought) & prev.notna()

    signal = pd.Series(0, index=cci_series.index, dtype=int)
    signal[enter_oversold] = 1    # oversold → expect a bounce up
    signal[enter_overbought] = -1  # overbought → expect a pullback down

    entries = signal[signal != 0]
    return pd.DataFrame({"dir": entries.astype(int)})


def crossback_entries(
    cci_series: pd.Series,
    oversold: float = -100.0,
    overbought: float = 100.0,
) -> pd.DataFrame:
    """Entry signals when CCI *exits* an extreme zone (cross-back / confirmed bounce).

    A **buy signal** (+1) fires when CCI crosses back above ``oversold`` (from below
    −100 to above −100) — the confirmed oversold-bounce framing.  A **sell signal** (−1)
    fires when CCI crosses back below ``overbought`` (from above +100 to below +100) —
    the confirmed overbought-fade framing.

    Signals are known at the close of bar *t*; trades enter at the *next* bar's open.
    """
    prev = cci_series.shift(1)
    exit_oversold = (cci_series >= oversold) & (prev < oversold) & prev.notna()
    exit_overbought = (cci_series <= overbought) & (prev > overbought) & prev.notna()

    signal = pd.Series(0, index=cci_series.index, dtype=int)
    signal[exit_oversold] = 1
    signal[exit_overbought] = -1

    entries = signal[signal != 0]
    return pd.DataFrame({"dir": entries.astype(int)})


def random_directions(n: int, seed: int = 0) -> np.ndarray:
    """A reproducible vector of ±1 — the control arm's coin."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=n)


# ---------------------------------------------------------------------------
# Fixed-horizon forward-return backtest
# ---------------------------------------------------------------------------
def run_trades(
    bars: pd.DataFrame,
    entries: pd.DataFrame,
    hold_days: int = 5,
    cost_bps: float = 1.0,
    directions: np.ndarray | None = None,
    long_only: bool = False,
) -> pd.DataFrame:
    """Run fixed-horizon forward-return trades and return a per-trade ledger.

    For each signal bar *t*, the trade is entered at bar *t+1*'s open and exited at the
    close of bar *t + hold_days* (or the last available bar, whichever comes first).
    This is the simplest look-ahead-free measurement: does the price, after an extreme
    CCI condition, move in the predicted direction over the next ``hold_days`` bars?

    ``directions`` overrides the entry sign vector (the random-control arm passes a ±1
    vector aligned to ``entries``).  ``cost_bps`` is a one-way round-trip cost deducted
    from the gross return.  With ``long_only=True`` only buy (+1) signals are executed
    (the long-flat variant).

    Columns: ``entry_ts, dir, entry, exit_price, bars_held, ret_gross, ret_net``.
    """
    open_ = bars["open"]
    close_ = bars["close"]
    idx = bars.index
    n_bars = len(bars)
    pos = {ts: i for i, ts in enumerate(idx)}

    dirs = (
        np.asarray(directions, dtype=int)
        if directions is not None
        else entries["dir"].to_numpy(dtype=int)
    )

    rows = []
    for sig_ts, d in zip(entries.index, dirs):
        if long_only and d < 0:
            continue
        i = pos.get(sig_ts)
        if i is None or i + 1 >= n_bars:
            continue
        e = i + 1  # enter at the next bar's open
        entry_px = open_.iat[e]
        exit_idx = min(e + hold_days - 1, n_bars - 1)
        exit_px = close_.iat[exit_idx]
        n_held = exit_idx - e + 1

        ret_gross = d * (exit_px - entry_px) / entry_px
        rows.append(
            {
                "entry_ts": idx[e],
                "dir": int(d),
                "entry": entry_px,
                "exit_price": exit_px,
                "bars_held": n_held,
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

    Returns trade count, win-rate, mean return (bps/trade), per-trade Sharpe, P&L skew,
    and a HAC Newey-West t-stat on the mean — the inference-bar number that decides
    whether the edge is distinguishable from zero.
    """
    if len(ledger) == 0:
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
        # Newey-West HAC t-stat — no hard dependency on quantlab.
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
