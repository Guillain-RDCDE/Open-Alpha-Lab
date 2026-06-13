"""The strategy and its honest controls — Study 106 (Supertrend).

The folk recipe: plot the Supertrend indicator (ATR period = 10, multiplier = 3)
on the daily chart.  When price crosses *above* the Supertrend band, go long; when
price crosses *below*, go short.  Wildly popular on TradingView (millions of chart
views), pitched in countless YouTube tutorials as a "reliable trend-following system."

We implement the canonical Supertrend computation and pin it against the one comparison
that settles whether it is anything but a lagging dice roll: the **same flip dates with
a random direction** on a symmetric ATR-barrier backtest.

Two exit regimes share one engine (:func:`run_trades`):

- **symmetric** — take-profit and stop at ±1 ATR(20) from entry. The only
  direction-fair payoff: a coin earns ≈ 0, so the flip must earn more to be real.
- **fixed-day hold** — hold for N calendar days then close. Checks whether the
  post-flip drift is positive at horizons that match Supertrend's intended use.

No look-ahead: the Supertrend state is computed from closes up to bar *t*; the trade
is entered at bar *t+1*'s open; barriers / exits are checked from *t+1* onward.

Relationship to Study 72 (Loaded-Dice, SMA(5/10) on 5-minute bars) and Study 78
(Crossed-Wires, MACD on daily bars): Supertrend is also a lagging ATR-band indicator
that flips direction as price crosses a moving band — same fundamental question.
We compare explicitly to those null results throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Supertrend indicator
# ---------------------------------------------------------------------------
def atr(bars: pd.DataFrame, n: int = 10) -> pd.Series:
    """Wilder-style (RMA) average true range, in price units.

    Wilder's original uses an EMA with alpha=1/n (equivalent to RMA/EWM with
    com=n-1).  This is the convention used by TradingView's Supertrend.
    """
    prev_close = bars["close"].shift(1)
    tr = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # RMA = EWM with com = n-1 (alpha = 1/n, no adjust)
    return tr.ewm(com=n - 1, min_periods=n, adjust=False).mean()


def supertrend(
    bars: pd.DataFrame,
    atr_n: int = 10,
    mult: float = 3.0,
) -> pd.DataFrame:
    """Canonical Supertrend indicator (TradingView-style).

    Returns a DataFrame with columns:
    - ``upper``    — upper ATR band: HL2 + mult * ATR(atr_n).
    - ``lower``    — lower ATR band: HL2 - mult * ATR(atr_n).
    - ``st``       — the active Supertrend line (upper or lower, with locking logic).
    - ``dir``      — current trend direction: +1 = price above ST (bullish), -1 = bearish.

    The locking logic (if the band was upper and price crosses above it, the band
    cannot go up further; if lower and price crosses below, it cannot go down) is
    what distinguishes Supertrend from a plain ATR band — it prevents premature flips.

    The first ``atr_n`` bars will have NaN ``st`` / ``dir`` due to ATR warmup.
    """
    hl2 = (bars["high"] + bars["low"]) / 2.0
    at = atr(bars, atr_n)
    raw_upper = hl2 + mult * at
    raw_lower = hl2 - mult * at
    close = bars["close"]

    n = len(bars)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    st = np.full(n, np.nan)
    direction = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(at.iat[i]):
            continue
        # Compute locked bands
        if i == 0 or np.isnan(upper[i - 1]):
            upper[i] = raw_upper.iat[i]
            lower[i] = raw_lower.iat[i]
        else:
            # Upper band: only lock down if previous close was below it
            upper[i] = (
                raw_upper.iat[i]
                if raw_upper.iat[i] < upper[i - 1] or close.iat[i - 1] > upper[i - 1]
                else upper[i - 1]
            )
            # Lower band: only lock up if previous close was above it
            lower[i] = (
                raw_lower.iat[i]
                if raw_lower.iat[i] > lower[i - 1] or close.iat[i - 1] < lower[i - 1]
                else lower[i - 1]
            )

        # Active line and direction
        if i == 0 or np.isnan(st[i - 1]):
            # Initialise: use upper if price below it, lower if above
            st[i] = upper[i] if close.iat[i] <= upper[i] else lower[i]
            direction[i] = 1 if close.iat[i] > st[i] else -1
        else:
            prev_dir = direction[i - 1]
            if prev_dir == -1:  # was bearish (price below upper band)
                if close.iat[i] > upper[i]:  # flip to bullish
                    st[i] = lower[i]
                    direction[i] = 1
                else:
                    st[i] = upper[i]
                    direction[i] = -1
            else:  # was bullish (price above lower band)
                if close.iat[i] < lower[i]:  # flip to bearish
                    st[i] = upper[i]
                    direction[i] = -1
                else:
                    st[i] = lower[i]
                    direction[i] = 1

    return pd.DataFrame(
        {"upper": upper, "lower": lower, "st": st, "dir": direction},
        index=bars.index,
    )


def flip_entries(
    bars: pd.DataFrame,
    atr_n: int = 10,
    mult: float = 3.0,
) -> pd.DataFrame:
    """Bars where the Supertrend direction flips.

    ``dir`` = +1 for a flip to bullish (long entry), -1 for a flip to bearish
    (short entry).  Detected from the sign change of ``direction`` between
    consecutive bars — known at the *close* of the flip bar, trade taken at the
    next bar's open (no look-ahead).
    """
    st = supertrend(bars, atr_n=atr_n, mult=mult)
    d = st["dir"]
    # Flip: direction changed AND both values are valid
    flipped = d.ne(d.shift(1)) & d.notna() & d.shift(1).notna()
    out = pd.DataFrame({"dir": d[flipped].astype(int)})
    out.index.name = bars.index.name
    return out


def random_directions(n: int, seed: int = 0) -> np.ndarray:
    """A reproducible vector of ±1 — the control arm's coin."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=n)


# ---------------------------------------------------------------------------
# Barrier backtest (symmetric ATR exits)
# ---------------------------------------------------------------------------
def run_trades(
    bars: pd.DataFrame,
    entries: pd.DataFrame,
    tp_R: float = 1.0,
    sl_R: float = 1.0,
    atr_n: int = 20,
    cost_bps: float = 1.0,
    directions: np.ndarray | None = None,
    max_hold: int = 60,
) -> pd.DataFrame:
    """Run ATR-barrier trades and return a per-trade ledger.

    For each entry bar the trade is entered at the *next* bar's open; the risk unit is
    ``R = ATR(atr_n)`` measured at the entry bar.  Take-profit sits ``tp_R * R`` away,
    stop ``sl_R * R`` away.  If neither barrier is hit within ``max_hold`` bars the
    trade is closed at the last available bar's close.  When a single bar straddles
    both barriers the **stop is assumed first** (conservative fill).

    ``directions`` overrides the entry signs (the random-control arm passes a ±1 vector
    aligned to ``entries``).  ``cost_bps`` is a one-way round-trip cost charged on the
    net return.

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
        e = i + 1  # enter at the next bar's open
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
            if hit_sl:  # conservative: stop wins a straddling bar
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
# Fixed-day forward return (alternative exit)
# ---------------------------------------------------------------------------
def run_forward_returns(
    bars: pd.DataFrame,
    entries: pd.DataFrame,
    hold_days: int = 20,
    cost_bps: float = 1.0,
    directions: np.ndarray | None = None,
) -> pd.DataFrame:
    """Hold for exactly ``hold_days`` bars after entry, close at the bar's close.

    A simpler alternative to barrier exits for calendar-horizon analysis.  Used to
    check whether the Supertrend flip predicts a *drifting* forward return over a
    horizon that matches the indicator's typical holding period.
    """
    close = bars["close"]
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
        exit_i = min(e + hold_days - 1, n_bars - 1)
        entry_px = close.iat[e]
        exit_px = close.iat[exit_i]
        ret_gross = d * (exit_px - entry_px) / entry_px
        rows.append(
            {
                "entry_ts": bars.index[e],
                "dir": int(d),
                "entry": entry_px,
                "exit": exit_px,
                "exit_reason": f"d{hold_days}",
                "bars_held": exit_i - e + 1,
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

    Returns trade count, win-rate, mean return (bps/trade), the per-trade Sharpe,
    the P&L skew, and a HAC Newey-West t-stat on the mean — the inference-bar number
    that decides whether the edge is distinguishable from zero.
    """
    if ledger.empty:
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
        # Newey-West HAC t-stat — same kernel as Studies 72 and 78 for apples-to-apples
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
