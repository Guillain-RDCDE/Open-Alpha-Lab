"""The strategy and its honest controls — Study 72 (Loaded-Dice).

The folk recipe: on a 5-minute chart, take the SMA(5)/SMA(10) crossover, go *with* the
cross (long an up-cross, short a down-cross), grab a few dollars, repeat all day. We
implement it as a barrier backtest and pin it against the one comparison that decides
whether it is anything but a dice roll: the **same entries with a random direction**.

Three exit regimes share one engine (:func:`run_trades`):

- **symmetric** — take-profit and stop in the *same* unit (R = one ATR at entry). This
  is the only honest way to ask "does the cross point the right way?", because the
  payoff is direction-symmetric: a coin gets ~0.
- **naive fixed-tick** — a small take-profit and a far stop (the "grab $3-4" recipe).
  It manufactures a high win-rate and a left-skewed P&L: the textbook way to *look*
  right while bleeding. We ship it precisely to expose the trap.

No look-ahead: the crossover is confirmed on closes up to bar *t*; the trade is entered
at bar *t+1*'s open; barriers are checked from *t+1* onward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Indicators & signals
# ---------------------------------------------------------------------------
def sma(close: pd.Series, n: int) -> pd.Series:
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


def crossover_entries(close: pd.Series, fast: int = 5, slow: int = 10) -> pd.DataFrame:
    """Bars where SMA(fast) crosses SMA(slow). ``dir`` = +1 up-cross, -1 down-cross.

    The cross is detected from the *sign change* of (fast − slow) between consecutive
    closes, so it is known at the close of the crossover bar — the trade is taken at the
    next bar's open by :func:`run_trades`.
    """
    f = sma(close, fast)
    s = sma(close, slow)
    diff = f - s
    sign = np.sign(diff)
    crossed = sign.ne(sign.shift(1)) & sign.ne(0) & sign.shift(1).notna()
    out = pd.DataFrame({"dir": sign[crossed].astype(int)})
    out.index.name = close.index.name
    return out


def random_directions(n: int, seed: int = 0) -> np.ndarray:
    """A reproducible vector of ±1 — the control arm's coin."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=n)


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
    flat_at_session_end: bool = True,
) -> pd.DataFrame:
    """Run barrier trades and return a per-trade ledger.

    For each entry bar the trade is entered at the *next* bar's open; the risk unit is
    ``R = ATR(atr_n)`` measured at the entry bar. Take-profit sits ``tp_R * R`` away,
    stop ``sl_R * R`` away (set ``sl_R`` huge to approximate "no stop"). The forward
    bars are scanned until a barrier is touched or — with ``flat_at_session_end`` — the
    session's last bar, where the position is marked out at the close (positions never
    held overnight). When a single bar straddles both barriers the **stop is assumed
    first** (the conservative fill).

    ``directions`` overrides the entry signs (the random-control arm passes a ±1 vector
    aligned to ``entries``). ``cost_bps`` is a one-way-times-two round-trip cost charged
    on the net return.

    Columns: ``entry_ts, dir, entry, exit, exit_reason, bars_held, ret_gross, ret_net``.
    """
    close = bars["close"]
    open_ = bars["open"]
    high = bars["high"]
    low = bars["low"]
    r_unit = atr(bars, atr_n)

    # Map each bar to an integer position and to its session (calendar date in ET).
    pos = {ts: i for i, ts in enumerate(bars.index)}
    session = bars.index.normalize()

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
        sess = session[e]

        exit_px = exit_reason = None
        last = e
        for j in range(e, n_bars):
            if session[j] != sess:  # never cross into the next session
                last = j - 1
                exit_px, exit_reason = close.iat[last], "eod"
                break
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
        if exit_px is None:  # ran off the end of the tape inside a session
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
    P&L skew (the tell for the fixed-tick trap), and a HAC t-stat on the mean — the
    inference-bar number that decides whether the edge is distinguishable from zero.
    """
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    out = {
        "n_trades": int(n),
        "win_rate": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "sharpe_per_trade": float(r.mean() / r.std(ddof=1)) if n > 1 and r.std() > 0 else float("nan"),
        "skew": float(pd.Series(r).skew()) if n > 2 else float("nan"),
        "tstat": float("nan"),
    }
    if n > 5:
        # Local Newey-West so the engine has no hard dependency on quantlab being importable.
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
