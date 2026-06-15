"""The strategy and its honest controls — Study 182 (Vortex-Indicator).

Etienne Botes and Douglas Siepman (2010) introduced the Vortex Indicator in the January
2010 issue of *Technical Analysis of Stocks & Commodities*.  It computes two directional
movement lines:

    VM+(t) = |high(t) - low(t-1)|       (upward vortex movement)
    VM-(t) = |low(t)  - high(t-1)|      (downward vortex movement)
    TR(t)  = max(|high(t) - low(t)|,
                 |high(t) - close(t-1)|,
                 |low(t)  - close(t-1)|)   (true range)

    VI+(t, n) = sum(VM+[t-n+1 .. t]) / sum(TR[t-n+1 .. t])
    VI-(t, n) = sum(VM-[t-n+1 .. t]) / sum(TR[t-n+1 .. t])

Folk recipe (n = 14):
  - VI+ crosses above VI–  →  bullish trend change → *buy* next open.
  - VI– crosses above VI+  →  bearish trend change → *sell / short* next open.

We test the VI+/VI– crossover long/short signal against a random-direction control on
the *same* entry bars, using a fixed-horizon forward-return backtest.  Two framings:

1. **Crossover** — enter on every VI+ / VI– crossover bar.
2. **Threshold crossover** — enter only when the faster line is meaningfully above the
   slower (|VI+ - VI-| > threshold), to filter noise.

Both use the same honest comparison: **same entries, random direction** (a fair coin).

No look-ahead: all indicators computed on bars up to *t*; trade entered at *t+1*'s open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Vortex Indicator
# ---------------------------------------------------------------------------
def vortex(bars: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Botes-Siepman Vortex Indicator (2010) over a ``period``-bar rolling window.

    Returns a DataFrame with two columns:

    - ``vi_plus``  — upward vortex index VI+(n)
    - ``vi_minus`` — downward vortex index VI-(n)

    The first ``period`` bars are NaN (the rolling sum window requires ``period`` bars
    of vortex movement, each of which itself requires a lagged high/low, so the warm-up
    is effectively ``period + 1`` original bars).

    The ratio formulation means VI+ and VI- are both roughly around 1.0 on a random walk
    and diverge only when directional movement dominates.
    """
    high = bars["high"]
    low = bars["low"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = bars["close"].shift(1)

    # Vortex movements (directional lines)
    vm_plus = (high - prev_low).abs()
    vm_minus = (low - prev_high).abs()

    # True range
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Smoothed ratios — rolling sum over ``period`` bars; min_periods = period
    vm_plus_sum = vm_plus.rolling(period, min_periods=period).sum()
    vm_minus_sum = vm_minus.rolling(period, min_periods=period).sum()
    tr_sum = tr.rolling(period, min_periods=period).sum()

    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum

    return pd.DataFrame(
        {"vi_plus": vi_plus, "vi_minus": vi_minus},
        index=bars.index,
    )


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------
def crossover_entries(
    vi: pd.DataFrame,
    min_gap: float = 0.0,
) -> pd.DataFrame:
    """Entry signals at every VI+/VI– crossover.

    A **buy signal** (+1) fires when VI+ crosses above VI– (previous bar had VI+ ≤ VI–).
    A **sell signal** (-1) fires when VI– crosses above VI+ (previous bar had VI– ≤ VI+).

    ``min_gap`` filters weak crossovers: only fire when the magnitude of the gap
    (|VI+ - VI-|) after the cross exceeds this threshold.  Default 0.0 captures all
    crossovers; a value such as 0.02 filters out very marginal ones.

    Signals are known at the close of bar *t*; trades enter at the *next* bar's open.
    """
    vp = vi["vi_plus"]
    vm = vi["vi_minus"]
    diff = vp - vm
    prev_diff = diff.shift(1)

    # Cross up: vi_plus > vi_minus this bar AND was below-or-equal last bar
    cross_up = (diff > 0) & (prev_diff <= 0) & prev_diff.notna()
    # Cross down: vi_minus > vi_plus this bar AND was below-or-equal last bar
    cross_dn = (diff < 0) & (prev_diff >= 0) & prev_diff.notna()

    if min_gap > 0.0:
        cross_up = cross_up & (diff.abs() > min_gap)
        cross_dn = cross_dn & (diff.abs() > min_gap)

    signal = pd.Series(0, index=vi.index, dtype=int)
    signal[cross_up] = 1
    signal[cross_dn] = -1

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
    hold_days: int = 14,
    cost_bps: float = 1.0,
    directions: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run fixed-horizon forward-return trades and return a per-trade ledger.

    For each signal bar *t*, the trade is entered at bar *t+1*'s open and exited at the
    close of bar *t + hold_days* (or the last available bar, whichever comes first).
    This is the simplest look-ahead-free measurement: does the price, after a Vortex
    crossover, move in the predicted direction over the next ``hold_days`` bars?

    ``hold_days = 14`` matches the VI period, capturing roughly one indicator cycle.

    ``directions`` overrides the entry sign vector (the random-control arm passes a ±1
    vector aligned to ``entries``).  ``cost_bps`` is a one-way round-trip cost deducted
    from the gross return.

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
