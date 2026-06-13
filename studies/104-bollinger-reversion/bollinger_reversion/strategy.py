"""The strategy and its honest controls — Study 104 (Bollinger-Reversion).

The folk recipe: when the daily close pierces the *lower* Bollinger Band (20-day SMA
minus 2 standard deviations), buy; hold until the close reaches the *middle* band (SMA)
or the *upper* band (SMA plus 2 sigma).  The claim: "price always returns to the bands."

We implement this as a forward-return study pinned against two controls:

1. **Random-entry control** — identical hold-length draws on random calendar dates (same
   instrument, same epoch), so the baseline reflects the instrument's drift, not the
   band-touch.  This is the key comparison: does the band-pierce *add* anything beyond
   randomly sampling the tape?

2. **Breakout-entry control** — the *opposite* folk rule: buy when the close pierces the
   *upper* band (a momentum/breakout signal).  Bollinger himself never endorsed breakouts
   as a signal, but it is widely claimed.  Testing both in one frame exposes the
   contradiction: exactly one of "reversion from the lower band" and "breakout from the
   upper band" can be right — and both can be zero.

Three exit regimes share one engine (:func:`run_trades`):

- **mid-exit** — close the position when price crosses back through the middle band (the
  20-day SMA).  This is the recipe as usually stated.
- **upper-exit** — close when price reaches the upper band.  A longer hold.
- **time-exit** — flat after a fixed number of bars (e.g. 20 days).  A cleaner
  comparison: forces both reversion and breakout to be evaluated on the same horizon.

No look-ahead: the band-pierce is detected on the close of day *t*; the trade is
entered at day *t+1*'s open; exit conditions are checked from *t+1* onward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Bollinger-Band indicators
# ---------------------------------------------------------------------------
def bollinger_bands(
    close: pd.Series, window: int = 20, n_sigma: float = 2.0
) -> pd.DataFrame:
    """Classic Bollinger Bands: SMA ± n_sigma * rolling_std (ddof=1).

    Returns a DataFrame with columns ``mid``, ``upper``, ``lower``, valid from
    bar ``window`` onward (first ``window-1`` rows are NaN).
    """
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=1)
    return pd.DataFrame(
        {"mid": mid, "upper": mid + n_sigma * std, "lower": mid - n_sigma * std},
        index=close.index,
    )


def band_entries(
    close: pd.Series,
    window: int = 20,
    n_sigma: float = 2.0,
    side: str = "reversion",
) -> pd.DataFrame:
    """Bars where the close pierces a Bollinger Band.

    ``side="reversion"`` → close < lower band (the mean-reversion buy).
    ``side="breakout"``  → close > upper band (the momentum buy).

    The pierce is detected on the *current* close, so the trade is always entered
    at the *next* bar's open by :func:`run_trades` — no look-ahead.

    Returns a DataFrame indexed by the signal bar with a ``dir`` column (always +1
    for the long-only reversion / breakout tests we run here).
    """
    if side not in ("reversion", "breakout"):
        raise ValueError(f"side must be 'reversion' or 'breakout', got {side!r}")
    bb = bollinger_bands(close, window=window, n_sigma=n_sigma)
    if side == "reversion":
        mask = close < bb["lower"]
    else:
        mask = close > bb["upper"]
    # Keep only the *first* bar of each consecutive run (the "pierce" day, not all
    # subsequent days while price stays outside the band).
    first_pierce = mask & ~mask.shift(1, fill_value=False)
    idx = close.index[first_pierce]
    return pd.DataFrame({"dir": np.ones(len(idx), dtype=int)}, index=idx)


def random_entries(
    bars: pd.DataFrame, n: int, seed: int = 0, window: int = 20
) -> pd.DataFrame:
    """``n`` random entry dates from the tape, respecting the warm-up period.

    The dates are sampled uniformly from bars[window:] (after the band warm-up),
    so the distribution is fair.  Returns a DataFrame with a ``dir`` column.
    """
    rng = np.random.default_rng(seed)
    valid = bars.index[window:]
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    chosen = pd.DatetimeIndex(sorted(chosen))
    return pd.DataFrame({"dir": np.ones(len(chosen), dtype=int)}, index=chosen)


# ---------------------------------------------------------------------------
# Forward-return engine
# ---------------------------------------------------------------------------
def run_trades(
    bars: pd.DataFrame,
    entries: pd.DataFrame,
    exit_mode: str = "mid",
    bb_window: int = 20,
    bb_sigma: float = 2.0,
    max_hold: int = 60,
    cost_bps: float = 2.0,
) -> pd.DataFrame:
    """Run forward trades and return a per-trade ledger.

    For each entry date the position is entered at the *next* bar's open.  The exit
    condition depends on ``exit_mode``:

    - ``"mid"``   — first close at or above the middle band (20-SMA).
    - ``"upper"`` — first close at or above the upper band.
    - ``"time"``  — after exactly ``max_hold`` bars regardless (marked out at close).

    Positions are always capped at ``max_hold`` bars (time-stop).  A trade that hits
    ``max_hold`` without touching the band is exited at the last close (``"timeout"``).

    ``cost_bps`` is a one-way-per-leg round-trip cost charged on the net return.

    Columns: ``entry_date, entry, exit_date, exit, exit_reason, bars_held,
    ret_gross, ret_net``.
    """
    if exit_mode not in ("mid", "upper", "time"):
        raise ValueError(f"exit_mode must be 'mid', 'upper', or 'time'; got {exit_mode!r}")

    bb = bollinger_bands(bars["close"], window=bb_window, n_sigma=bb_sigma)
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
        e = i + 1                       # enter at next bar's open
        entry_px = open_.iat[e]

        exit_px = exit_reason = None
        last = min(e + max_hold - 1, n_bars - 1)

        for j in range(e, last + 1):
            c = close.iat[j]
            mid = bb["mid"].iat[j]
            upper = bb["upper"].iat[j]

            if exit_mode == "time":
                if j == last:
                    exit_px, exit_reason = c, "timeout"
                    last = j
                    break
            elif exit_mode == "mid":
                if np.isfinite(mid) and c >= mid:
                    exit_px, exit_reason = c, "mid"
                    last = j
                    break
            elif exit_mode == "upper":
                if np.isfinite(upper) and c >= upper:
                    exit_px, exit_reason = c, "upper"
                    last = j
                    break

            if j == last:               # timed out
                exit_px, exit_reason = c, "timeout"

        if exit_px is None:
            exit_px, exit_reason = close.iat[last], "timeout"

        ret_gross = (exit_px - entry_px) / entry_px
        rows.append(
            {
                "entry_date": bars.index[e],
                "entry": entry_px,
                "exit_date": bars.index[last],
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
        # Newey-West HAC t-stat (no hard quantlab dependency).
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
# Annualised-return helper
# ---------------------------------------------------------------------------
def annualised_return(mean_bps: float, trades_per_year: float) -> float:
    """Rough annualised P&L: ``mean_bps/trade * trades/year / 1e4``, as a fraction."""
    return mean_bps * trades_per_year / 1e4
