"""The strategy and its honest controls — Study 129 (Heikin-Ashi).

The folk recipe: on the daily chart, Heikin-Ashi (HA) candles "filter noise" and "let
you ride the trend."  When the HA candle colour flips from red to green go long; when
it flips from green to red go short (or flat).  Millions of retail traders use HA as
their primary entry signal.

We implement it in three arms and pin each against the one comparison that decides
whether it contains any directional information at all: the **same flip dates with a
random direction** on a symmetric ATR-barrier backtest.

Three comparisons:

1. **HA colour flip (the claim)** — enter in the direction of the flip, ±1 ATR(20)
   symmetric barriers.
2. **Raw-candle colour flip (the raw-candle equivalent)** — same protocol but use the
   *raw* OHLC candle colour (close >= open) to detect flips.  Fair baseline: does the
   smoothing add anything over the unsmoothed version?
3. **Random-direction control** — identical flip dates, ±1 direction drawn i.i.d.  The
   coin: if HA beats this it carries some directional information.

No look-ahead: the HA state at bar *t* is computed from OHLC data up to bar *t*; the
trade is entered at bar *t+1*'s open; barriers are checked from *t+1* onward.

CRITICAL — HA is recursive (HA_open[t] depends on HA_open[t-1] and HA_close[t-1]),
so it CANNOT be vectorised from scratch as a simple rolling window.  We compute it
bar-by-bar (or via two-pass pandas with correct initialisation) to avoid look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Heikin-Ashi indicator
# ---------------------------------------------------------------------------
def heikin_ashi(bars: pd.DataFrame) -> pd.DataFrame:
    """Causal Heikin-Ashi OHLC from raw OHLC — no look-ahead.

    The recursive initialisation follows the canonical definition:

        HA_close[t] = (O + H + L + C)[t] / 4
        HA_open[t]  = (HA_open[t-1] + HA_close[t-1]) / 2
        HA_high[t]  = max(H[t], HA_open[t], HA_close[t])
        HA_low[t]   = min(L[t], HA_open[t], HA_close[t])

    Seed: HA_open[0] = bars["open"].iloc[0] (the first raw open).
    HA_close[0] = (O+H+L+C)[0]/4 as usual.

    Returns a DataFrame with columns ``ha_open, ha_high, ha_low, ha_close``
    aligned to ``bars.index``.
    """
    o = bars["open"].to_numpy()
    h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    c = bars["close"].to_numpy()
    n = len(bars)

    ha_c = (o + h + lo + c) / 4.0  # vectorised — no recursion in ha_close
    ha_o = np.empty(n)
    ha_h = np.empty(n)
    ha_l = np.empty(n)

    ha_o[0] = o[0]  # seed
    ha_h[0] = max(h[0], ha_o[0], ha_c[0])
    ha_l[0] = min(lo[0], ha_o[0], ha_c[0])

    for i in range(1, n):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
        ha_h[i] = max(h[i], ha_o[i], ha_c[i])
        ha_l[i] = min(lo[i], ha_o[i], ha_c[i])

    return pd.DataFrame(
        {"ha_open": ha_o, "ha_high": ha_h, "ha_low": ha_l, "ha_close": ha_c},
        index=bars.index,
    )


def ha_colour(ha: pd.DataFrame) -> pd.Series:
    """Per-bar HA colour: +1 (green, ha_close >= ha_open) or -1 (red).

    This is the signal traders use: a run of green bars = uptrend,
    a run of red bars = downtrend.
    """
    return pd.Series(
        np.where(ha["ha_close"] >= ha["ha_open"], 1, -1),
        index=ha.index,
        name="ha_colour",
    )


def raw_colour(bars: pd.DataFrame) -> pd.Series:
    """Per-bar raw-candle colour: +1 (green, close >= open) or -1 (red).

    The unsmoothed equivalent of ha_colour; used as the honest baseline to check
    whether HA smoothing adds anything over the plain raw-candle colour flip.
    """
    return pd.Series(
        np.where(bars["close"] >= bars["open"], 1, -1),
        index=bars.index,
        name="raw_colour",
    )


def colour_flip_entries(colour: pd.Series, warmup: int = 0) -> pd.DataFrame:
    """Bars where the colour sequence flips direction.

    ``dir`` = +1 for a flip to green (long entry), -1 for a flip to red (short
    entry).  Detected from the sign change of ``colour`` between consecutive bars
    — known at the *close* of the flip bar, trade taken at the next bar's open
    (no look-ahead).  Bars within the first ``warmup`` rows are excluded to allow
    the HA recursion to settle.

    Returns a DataFrame with column ``dir`` indexed by the flip bar's timestamp.
    """
    flipped = colour.ne(colour.shift(1)) & colour.notna() & colour.shift(1).notna()
    if warmup > 0:
        flipped.iloc[:warmup] = False
    out = pd.DataFrame({"dir": colour[flipped].astype(int)})
    out.index.name = colour.index.name
    return out


def random_directions(n: int, seed: int = 0) -> np.ndarray:
    """A reproducible vector of ±1 — the control arm's coin."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=n)


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------
def atr(bars: pd.DataFrame, n: int = 20) -> pd.Series:
    """Simple (rolling-mean) average true range, in price units.

    Uses the standard rolling mean (not Wilder EMA) so the warmup is exactly n
    bars, making it straightforward to reason about look-ahead — all values are
    computed from data up to and including bar *t*.
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
    return tr.rolling(n, min_periods=n).mean()


# ---------------------------------------------------------------------------
# Barrier backtest
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

    For each entry bar the trade is entered at the *next* bar's open; the risk
    unit is ``R = ATR(atr_n)`` measured at the entry bar.  Take-profit sits
    ``tp_R * R`` away, stop ``sl_R * R`` away.  If neither barrier is hit within
    ``max_hold`` bars the trade is closed at the last available bar's close.  When
    a single bar straddles both barriers the **stop is assumed first** (conservative
    fill — the same discipline as Studies 72, 106, 78).

    ``directions`` overrides the entry signs (the random-control arm passes a ±1
    vector aligned to ``entries``).  ``cost_bps`` is a one-way-times-two round-trip
    cost charged on the net return.

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
        # Newey-West HAC t-stat — the desk's standard inference tool.
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
