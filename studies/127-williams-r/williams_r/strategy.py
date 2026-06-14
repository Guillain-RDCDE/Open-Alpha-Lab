"""The strategy and its honest controls — Study 127 (Williams-R).

Larry Williams' %R is a normalised 14-period stochastic that measures where the close
sits within the high-low range of the past 14 bars:

    %R(t) = -100 * (highest_high(14) - close(t)) / (highest_high(14) - lowest_low(14))

Range: 0 (close at the 14-day high = overbought) to -100 (close at the 14-day low = oversold).

The folk recipe:
  - %R dips below -80 ("oversold") → **buy** next open, expect a bounce.
  - %R rises above -20 ("overbought") → **short** next open, expect a pullback.

Two complementary framings are tested:

1. **Mean-reversion framing** — enter when %R enters the extreme zone (< -80 or > -20),
   hold a fixed window (1, 3, 5, 10 days), measure the mean forward return vs a random-
   direction control on identical entry bars.  This is the direct test of whether extreme
   %R predicts the next few days' direction.

2. **Cross framing** — enter when %R *crosses* back through a threshold (from below -80
   back up through -80 = a reversal signal; from above -20 back down through -20).  This
   is the "confirmed bounce" version seen in many trading tutorials.

Both framings use the same honest comparison: the **same entry bars, random direction**
(a fair coin), so the verdict is "does %R add directional information over a coin?"

No look-ahead: all indicators computed on closes up to bar *t*; trade entered at *t+1*'s
open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Williams %R indicator
# ---------------------------------------------------------------------------
def williams_r(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R oscillator, period-bar look-back.

    %R(t) = -100 * (HH(period) - close(t)) / (HH(period) - LL(period))

    where HH and LL are the highest-high and lowest-low over the trailing ``period``
    bars (inclusive of bar *t*).  Returns a Series in the range [–100, 0]:

    - Values near 0 → close near the top of the recent range → "overbought".
    - Values near –100 → close near the bottom of the recent range → "oversold".

    The convention %R ∈ [−100, 0] keeps the folk thresholds intact: oversold < −80,
    overbought > −20 (equivalently, %R < −80 and %R > −20).
    """
    hh = bars["high"].rolling(period, min_periods=period).max()
    ll = bars["low"].rolling(period, min_periods=period).min()
    rng = hh - ll
    # Williams %R = -100 * (HH - close) / (HH - LL), ranging from 0 (at HH) to -100 (at LL).
    # Interpretation: values near 0 → overbought; values near -100 → oversold.
    raw = -100.0 * (hh - bars["close"]) / rng
    # Where range is zero but hh/ll are valid, substitute -50 (midpoint).
    flat_mask = (rng == 0) & hh.notna()
    raw[flat_mask] = -50.0
    wr = raw.rename("williams_r")
    return wr


# ---------------------------------------------------------------------------
# Signal generators — two framings
# ---------------------------------------------------------------------------
def zone_entries(
    wr: pd.Series,
    oversold: float = -80.0,
    overbought: float = -20.0,
) -> pd.DataFrame:
    """Entry signals when %R *first enters* an extreme zone (mean-reversion framing).

    A **buy signal** (+1) fires on the first bar where %R falls below ``oversold``
    (i.e. the previous bar was *above* the threshold).  A **sell signal** (-1) fires on
    the first bar where %R rises above ``overbought`` (the previous bar was *below* the
    threshold).  This captures the *onset* of an oversold/overbought condition — one
    entry per episode, not one per day inside the zone.

    Signals are known at the close of bar *t*; trades enter at the *next* bar's open.
    """
    prev = wr.shift(1)
    # Buy: %R crosses below oversold (from above to below -80)
    enter_oversold = (wr < oversold) & (prev >= oversold) & prev.notna()
    # Sell: %R crosses above overbought (from below to above -20)
    enter_overbought = (wr > overbought) & (prev <= overbought) & prev.notna()

    signal = pd.Series(0, index=wr.index, dtype=int)
    signal[enter_oversold] = 1    # expect bounce up
    signal[enter_overbought] = -1  # expect pullback down

    entries = signal[signal != 0]
    return pd.DataFrame({"dir": entries.astype(int)})


def cross_entries(
    wr: pd.Series,
    oversold: float = -80.0,
    overbought: float = -20.0,
) -> pd.DataFrame:
    """Entry signals when %R *exits* an extreme zone (cross-back / confirmed bounce).

    A **buy signal** (+1) fires when %R rises back through ``oversold`` (crosses from
    below -80 to above -80) after having been in the oversold zone.  A **sell signal**
    (-1) fires when %R falls back through ``overbought`` (crosses from above -20 to
    below -20) after an overbought episode.  This is the "confirmed reversal" framing.

    Signals are known at the close of bar *t*; trades enter at the *next* bar's open.
    """
    prev = wr.shift(1)
    # Buy: %R crosses back up through oversold threshold (oversold zone exit)
    exit_oversold = (wr >= oversold) & (prev < oversold) & prev.notna()
    # Sell: %R crosses back down through overbought threshold (overbought zone exit)
    exit_overbought = (wr <= overbought) & (prev > overbought) & prev.notna()

    signal = pd.Series(0, index=wr.index, dtype=int)
    signal[exit_oversold] = 1
    signal[exit_overbought] = -1

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
    simplest look-ahead-free measurement: does the price, after an extreme %R condition,
    move in the predicted direction over the next ``hold_days`` bars?

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
