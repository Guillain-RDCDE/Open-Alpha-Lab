"""The strategy and its honest controls — Study 107 (Stochastic-Oscillator).

The folk recipe: George Lane's Stochastic Oscillator — compute %K(14) as the position of
the close within the 14-period high-low range, then smooth it to %D(3).  The signal:

- **Oversold buy**: %K crosses *above* %D while both are below 20 → buy next open.
- **Overbought sell**: %K crosses *below* %D while both are above 80 → short next open.

We implement a forward-return backtest and pin it against the only comparison that
decides whether it is anything but a random entry: the **same entry bars with a random
direction** (the coin control).

Two complementary framings share one engine (:func:`run_trades`):

- **mean-reversion** — enter on the zone-filtered cross, hold a fixed window (1, 3, 5
  days), measure the mean forward return vs the random-direction control.  This is the
  honest test of whether %K/%D in an extreme zone predicts the next few days.
- **crossover-only** — ignore the zone filter and trade *every* cross (the looser recipe
  seen in many tutorials).  This higher-frequency variant yields more trades and lets us
  measure whether the zone filter adds anything at all.

No look-ahead: all indicators are computed on closes up to bar *t*; the trade is entered
at bar *t+1*'s open; returns are measured from that open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------
def stochastic(
    bars: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """Compute %K and %D for the Stochastic Oscillator.

    %K(t) = 100 * (close(t) - lowest_low(k_period)) / (highest_high(k_period) - lowest_low(k_period))
    %D(t) = SMA(%K, d_period)

    Both are known at the close of bar *t* and available to trade at the *next* bar's
    open — look-ahead-safe by construction.
    """
    ll = bars["low"].rolling(k_period, min_periods=k_period).min()
    hh = bars["high"].rolling(k_period, min_periods=k_period).max()
    rng = hh - ll
    # Avoid division by zero when the range is flat.
    pct_k = pd.Series(
        np.where(rng > 0, 100.0 * (bars["close"] - ll) / rng, 50.0),
        index=bars.index,
        name="pct_k",
    )
    pct_d = pct_k.rolling(d_period, min_periods=d_period).mean().rename("pct_d")
    return pd.concat([pct_k, pct_d], axis=1)


def crossover_signals(
    pct_k: pd.Series,
    pct_d: pd.Series,
    oversold: float = 20.0,
    overbought: float = 80.0,
    zone_filter: bool = True,
) -> pd.DataFrame:
    """Detect %K/%D crossovers, optionally restricted to oversold/overbought zones.

    A **buy signal** fires when %K crosses from below to above %D.  A **sell signal**
    fires when %K crosses from above to below %D.  With ``zone_filter=True`` buy signals
    are retained only while both %K and %D are below ``oversold`` (< 20) and sell signals
    only while both are above ``overbought`` (> 80) — the classic Lane recipe.

    The cross is detected at bar *t* (known from closes); the trade enters at bar *t+1*.
    Returns a DataFrame with column ``dir`` (``+1`` = buy, ``-1`` = sell).
    """
    diff = pct_k - pct_d
    sign = np.sign(diff)
    prev_sign = sign.shift(1)
    crossed_up = (sign > 0) & (prev_sign < 0) & prev_sign.notna()
    crossed_dn = (sign < 0) & (prev_sign > 0) & prev_sign.notna()

    if zone_filter:
        # Buy only in oversold territory; sell only in overbought.
        buy_zone = (pct_k < oversold) & (pct_d < oversold)
        sell_zone = (pct_k > overbought) & (pct_d > overbought)
        signal = pd.Series(0, index=pct_k.index, dtype=int)
        signal[crossed_up & buy_zone] = 1
        signal[crossed_dn & sell_zone] = -1
    else:
        signal = pd.Series(0, index=pct_k.index, dtype=int)
        signal[crossed_up] = 1
        signal[crossed_dn] = -1

    entries = signal[signal != 0]
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
    hold_days: int = 5,
    cost_bps: float = 1.0,
    directions: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run fixed-horizon forward-return trades and return a per-trade ledger.

    For each signal bar *t*, the trade is entered at bar *t+1*'s open and exited at the
    close of bar *t + hold_days* (or the last bar, whichever comes first).  This is the
    simplest look-ahead-free measurement of mean-reversion: does the price, after an
    extreme %K/%D cross, move in the predicted direction over the next ``hold_days`` bars?

    ``directions`` overrides the entry sign vector (the random-control arm passes a ±1
    vector).  ``cost_bps`` is a one-way-times-two round-trip cost deducted from the gross
    return on the entry-to-exit span.

    Columns: ``entry_ts, dir, entry, exit_price, bars_held, ret_gross, ret_net``.
    """
    open_ = bars["open"]
    close_ = bars["close"]
    idx = bars.index
    n_bars = len(bars)
    # Map each timestamp to its integer position for fast forward scanning.
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
        # Exit at the close of bar e + hold_days - 1, clipped to tape end.
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
    r = ledger[col].to_numpy(dtype=float) if len(ledger) > 0 else np.array([])
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
