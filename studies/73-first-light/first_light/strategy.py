"""The strategy and its honest controls — Study 73 (First-Light).

The Opening Range Breakout (ORB): define the opening range as the first N minutes of
the RTH session; go **long** when price breaks *above* the range high, **short** when it
breaks *below* the range low; hold until the stop (range-size away) or the 16:00 close.
Popularised by Zarattini & Aiolfi (2023) who report spectacular back-test returns on
QQQ/TQQQ — but with a methodology that conflates simple trend-following returns with a
genuine opening-range edge.

Three questions drive this study:

1. **Does the ORB direction beat a coin?** Identical entry timestamps, random-direction
   control. The signal is the opening range; the null is a fair die.
2. **Does it survive costs?** At the rule's natural turnover (~1 trade/day), spread +
   commission sweeps quickly turn a thin edge into a loser.
3. **Is the TQQQ leverage a free amplifier?** No — 3× leverage + financing (~50 bps/day
   drag at current rates) means the gross ORB edge must be enormous to survive.

No look-ahead: the opening range is formed from the first ``range_bars`` closes; the
breakout signal is confirmed at the close of the bar that breaches the range; entry is at
the *next* bar's open. Positions are never held overnight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Opening-range signal
# ---------------------------------------------------------------------------
def orb_entries(
    bars: pd.DataFrame,
    range_bars: int = 1,
) -> pd.DataFrame:
    """Detect ORB breakout entries for each session.

    Parameters
    ----------
    bars:
        5-minute OHLCV frame restricted to RTH (09:30–15:55 ET).
    range_bars:
        Number of 5-minute bars that define the opening range.
        1 → the first 5-minute bar only (the "5-minute ORB").
        3 → the first 15-minute opening range.

    Returns
    -------
    DataFrame with columns ``dir`` (+1 long, −1 short) indexed by the bar **whose close**
    confirms the breakout — entry is taken at the *next* bar's open by :func:`run_trades`.
    At most one signal per session is produced (the first breakout of either bound).
    """
    session_date = bars.index.normalize()
    sessions = session_date.unique()
    rows = []
    for day in sessions:
        day_mask = session_date == day
        day_bars = bars[day_mask]
        n = len(day_bars)
        if n <= range_bars:
            continue  # not enough bars to form range + trade
        # Opening range: first `range_bars` bars of this session.
        or_high = day_bars["high"].iloc[:range_bars].max()
        or_low = day_bars["low"].iloc[:range_bars].min()
        or_size = or_high - or_low
        if or_size <= 0:
            continue
        # Scan subsequent bars for the first close that confirms a breakout.
        for j in range(range_bars, n):
            bar = day_bars.iloc[j]
            if bar["close"] > or_high:
                rows.append({"ts": day_bars.index[j], "dir": 1, "or_high": or_high,
                             "or_low": or_low, "or_size": or_size})
                break
            if bar["close"] < or_low:
                rows.append({"ts": day_bars.index[j], "dir": -1, "or_high": or_high,
                             "or_low": or_low, "or_size": or_size})
                break
    if not rows:
        return pd.DataFrame(columns=["dir", "or_high", "or_low", "or_size"])
    df = pd.DataFrame(rows).set_index("ts")
    df.index.name = "ts"
    return df


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
    sl_multiplier: float = 1.0,
    tp_multiplier: float | None = None,
    cost_bps: float = 1.0,
    directions: np.ndarray | None = None,
    financing_bps_per_day: float = 0.0,
) -> pd.DataFrame:
    """Run ORB barrier trades and return a per-trade ledger.

    For each entry the trade is entered at the *next* bar's open. The stop is
    ``sl_multiplier * or_size`` away from entry; the take-profit (if ``tp_multiplier``
    is not None) sits ``tp_multiplier * or_size`` away — otherwise the only exit is
    the stop or end-of-day.  With ``tp_multiplier=None`` the exit at close is the
    primary outcome (this mirrors the "close at 16:00" rule). When a bar straddles both
    barriers the **stop is assumed first** (conservative). Positions are never held
    overnight.

    ``directions`` overrides the entry signs (random-control arm). ``cost_bps`` is the
    round-trip cost. ``financing_bps_per_day`` simulates leverage drag (e.g. TQQQ ≈
    50 bps/day at current 1-month T-bill rates + product cost).

    Columns: ``entry_ts, dir, entry, exit, exit_reason, bars_held, ret_gross, ret_net``.
    """
    close = bars["close"]
    open_ = bars["open"]
    high = bars["high"]
    low = bars["low"]

    pos = {ts: i for i, ts in enumerate(bars.index)}
    session = bars.index.normalize()

    dirs = (
        np.asarray(directions, dtype=int)
        if directions is not None
        else entries["dir"].to_numpy(dtype=int)
    )

    rows = []
    n_bars = len(bars)
    for (sig_ts, row_e), d in zip(entries.iterrows(), dirs):
        i = pos.get(sig_ts)
        if i is None or i + 1 >= n_bars:
            continue
        e = i + 1  # enter at the next bar's open
        or_size = float(row_e.get("or_size", 0.0)) if hasattr(row_e, "get") else 0.0
        if or_size <= 0:
            continue
        entry_px = open_.iat[e]
        sl = entry_px - d * sl_multiplier * or_size
        tp = (entry_px + d * tp_multiplier * or_size) if tp_multiplier is not None else None
        sess = session[e]

        exit_px = exit_reason = None
        last = e
        for j in range(e, n_bars):
            if session[j] != sess:
                last = j - 1
                exit_px, exit_reason = close.iat[last], "eod"
                break
            hi, loo = high.iat[j], low.iat[j]
            hit_sl = (loo <= sl) if d > 0 else (hi >= sl)
            hit_tp = (tp is not None) and ((hi >= tp) if d > 0 else (loo <= tp))
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
        # Financing drag: charged per bar held, scaled to bps/day (78 bars/day).
        bars_held = last - e + 1
        financing = financing_bps_per_day * 1e-4 * bars_held / 78.0
        rows.append(
            {
                "entry_ts": bars.index[e],
                "dir": int(d),
                "entry": entry_px,
                "exit": exit_px,
                "exit_reason": exit_reason,
                "bars_held": bars_held,
                "ret_gross": ret_gross,
                "ret_net": ret_gross - cost_bps * 1e-4 - financing,
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
    r = ledger[col].to_numpy(dtype=float) if len(ledger) else np.array([])
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
