"""The strategy and its honest controls — Study 180 (TRIX).

TRIX (Jack Hutson, 1983) is the 1-day percentage rate-of-change of a *triple-smoothed*
EMA of the close.  The triple smoothing is designed to filter out short cycles and expose
only longer-term trends.  Two common trading signals:

- **Zero-line cross** — TRIX crosses zero from below (buy) or above (sell short).
  Analogous to a MACD zero-line cross but with heavier lag from the triple smoothing.
- **Signal-line cross** — TRIX crosses its own ``signal``-period EMA.  The TRIX line is
  the "fast" line; the EMA of TRIX is the "slow" line, mirroring the MACD/signal mechanic.

The folk claim is that the triple smoothing filters false signals, leaving only "real" trend
changes.  We test whether either signal beats a **random-direction control** on the *same*
entry bars — the one comparison that decides whether TRIX adds directional information.

Three elements share one forward-return backtest engine (:func:`run_trades`):

- **TRIX zero-line cross strategy** — the primary hypothesis.
- **TRIX signal-line cross strategy** — the secondary (common variant) hypothesis.
- **Random-direction control** — same entry bars, ±1 drawn i.i.d., so a coin scores ≈ 0.

No look-ahead: all indicators computed on closes up to bar *t*; trade entered at *t+1*'s
open.  Multiple hypotheses (zero-line vs signal-line, plus parameter variations) are
corrected via the Bonferroni method when testing jointly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# TRIX indicator
# ---------------------------------------------------------------------------
def ema(series: pd.Series, period: int) -> pd.Series:
    """Standard EMA with span ``period``, requiring ``period`` observations to start."""
    return series.ewm(span=period, min_periods=period, adjust=False).mean()


def trix(close: pd.Series, period: int = 15) -> pd.Series:
    """Triple-smoothed EMA rate-of-change — the TRIX oscillator.

    With period ``n``:
        EMA1 = EMA(close, n)
        EMA2 = EMA(EMA1, n)
        EMA3 = EMA(EMA2, n)
        TRIX = 100 * (EMA3 - EMA3.shift(1)) / EMA3.shift(1)

    The triple smoothing attenuates cycles shorter than ~3n bars.  The result is a slow,
    momentum-style oscillator: TRIX > 0 means the triple-smoothed trend is still rising.

    Warmup: the first valid value appears after approximately ``3 * period`` bars because
    three sequential EMAs must each accumulate ``min_periods`` non-NaN inputs.
    """
    e1 = ema(close, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    # Rate-of-change of the triple-smoothed EMA (percentage points).
    trix_val = 100.0 * (e3 - e3.shift(1)) / e3.shift(1)
    trix_val.name = f"trix_{period}"
    return trix_val


def trix_signal(trix_series: pd.Series, signal_period: int = 9) -> pd.Series:
    """EMA(signal_period) of the TRIX line — the signal line for crossover trading."""
    sig = ema(trix_series, signal_period)
    sig.name = f"trix_signal_{signal_period}"
    return sig


# ---------------------------------------------------------------------------
# Signal generators — two framings
# ---------------------------------------------------------------------------
def zero_line_entries(trix_series: pd.Series) -> pd.DataFrame:
    """Entry signals when TRIX crosses the zero line.

    A **buy signal** (+1) fires on the first bar where TRIX crosses from negative to
    positive (TRIX < 0 the previous bar, TRIX >= 0 this bar).  A **sell signal** (-1)
    fires on the first bar where TRIX crosses from positive to negative.

    Signals are known at the close of bar *t*; trades enter at the *next* bar's open.
    This captures the onset of a new positive/negative trend phase in the triple-smoothed
    EMA — one entry per cross, not one per day the indicator is on one side.
    """
    prev = trix_series.shift(1)
    bull = (trix_series >= 0) & (prev < 0) & prev.notna()
    bear = (trix_series < 0) & (prev >= 0) & prev.notna()
    sig = pd.Series(0, index=trix_series.index, dtype=int)
    sig[bull] = 1
    sig[bear] = -1
    entries = sig[sig != 0]
    return pd.DataFrame({"dir": entries.astype(int)})


def signal_line_entries(trix_series: pd.Series, signal_series: pd.Series) -> pd.DataFrame:
    """Entry signals when TRIX crosses its own signal line (EMA of TRIX).

    A **buy signal** (+1) fires when TRIX crosses above the signal line.
    A **sell signal** (-1) fires when TRIX crosses below the signal line.

    This is the MACD-style variant: TRIX is the "fast" line, the EMA of TRIX is the
    "slow" line.  Crossovers here are slightly less lagged than the zero-line cross.
    """
    diff = trix_series - signal_series
    prev_diff = diff.shift(1)
    bull = (diff >= 0) & (prev_diff < 0) & prev_diff.notna()
    bear = (diff < 0) & (prev_diff >= 0) & prev_diff.notna()
    sig = pd.Series(0, index=trix_series.index, dtype=int)
    sig[bull] = 1
    sig[bear] = -1
    entries = sig[sig != 0]
    return pd.DataFrame({"dir": entries.astype(int)})


def random_directions(n: int, seed: int = 0) -> np.ndarray:
    """A reproducible vector of ±1 — the control arm's coin."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=n)


# ---------------------------------------------------------------------------
# Forward-return backtest
# ---------------------------------------------------------------------------
def run_trades(
    bars: pd.DataFrame,
    entries: pd.DataFrame,
    hold_days: int = 20,
    cost_bps: float = 1.0,
    directions: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run fixed-horizon forward-return trades and return a per-trade ledger.

    For each signal bar *t*, the trade is entered at bar *t+1*'s open and exited at the
    close of bar *t + hold_days* (or the last bar, whichever comes first).

    TRIX's triple smoothing introduces heavy lag, so the holding window is deliberately
    set to a medium horizon (``hold_days=20`` trading days, ~one calendar month) to give
    the confirmed trend time to play out.  This is the fairest setting for the hypothesis
    "TRIX identifies real trend changes".

    ``directions`` overrides the entry sign vector (the random-control arm passes a ±1
    vector aligned to ``entries``).  ``cost_bps`` is a one-way-times-two round-trip cost
    deducted from the gross return.

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
