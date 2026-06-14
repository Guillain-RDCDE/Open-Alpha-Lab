"""The strategy and its honest controls — Study 126 (Parabolic-SAR).

The folk recipe: plot the Parabolic SAR (stop-and-reverse) indicator with Wilder's
canonical settings (AF init = 0.02, step = 0.02, max = 0.20) on the daily chart.  Go
*long* when price crosses *above* the SAR (SAR flips from above price to below); go
*short* when price crosses *below* (SAR flips from below to above).  The indicator was
designed by J. Welles Wilder Jr. and introduced in *New Concepts in Technical Trading
Systems* (1978) and remains one of the most widely taught technical indicators on
platforms from TradingView to Investopedia.

We implement the canonical Wilder SAR computation and pin it against the one comparison
that settles whether it is anything but a lagging dice roll: the **same flip dates with
a random direction** on a symmetric ATR-barrier backtest.  If the SAR direction carries
real information, it should beat a coin.  On a range-bound tape the SAR whipsaws
constantly — every false flip is a round-trip loss — so heavy turnover is the first
hazard, and the symmetric test catches both the signal question and the cost question.

No look-ahead: the SAR state is computed from closes, highs, and lows up to bar *t*;
the trade is entered at bar *t+1*'s open; barriers are checked from *t+1* onward.

Relationship to Study 72 (Loaded-Dice, SMA crossover), Study 78 (MACD / Crossed-Wires),
and Study 106 (Supertrend): all are lagging trend-following indicators in the same
family — we compare explicitly to those null results in the notebooks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SAR_AF_INIT, SAR_AF_MAX, SAR_AF_STEP


# ---------------------------------------------------------------------------
# Parabolic SAR indicator
# ---------------------------------------------------------------------------
def parabolic_sar(
    bars: pd.DataFrame,
    af_init: float = SAR_AF_INIT,
    af_step: float = SAR_AF_STEP,
    af_max: float = SAR_AF_MAX,
) -> pd.DataFrame:
    """Wilder's Parabolic SAR (stop-and-reverse), canonical implementation.

    Returns a DataFrame with columns:
    - ``sar``   — the SAR value for each bar (NaN for the first initialisation bar).
    - ``dir``   — current trend direction: +1 = long (SAR below price), -1 = short
                  (SAR above price).
    - ``af``    — current acceleration factor (starts at af_init, steps by af_step
                  each bar the extreme extends, capped at af_max).
    - ``ep``    — the extreme point in the current trend (highest high for long,
                  lowest low for short).

    The algorithm (Wilder 1978, chapter 6):
    1. Initialise the first valid bar: direction is long if close > open, else short.
    2. Each subsequent bar:
       a. Project: SAR_t = SAR_{t-1} + AF * (EP - SAR_{t-1}).
       b. Adjust: for a long, SAR_t = min(SAR_t, low_{t-1}, low_{t-2}); for a short,
          SAR_t = max(SAR_t, high_{t-1}, high_{t-2}).
       c. Check flip: if SAR_t crosses price (SAR_t > low_t for long, SAR_t < high_t
          for short), flip direction, set SAR to the prior extreme point, reset AF.
       d. Update EP: if the trend continues and price sets a new extreme, step AF.

    The first bar always carries NaN (no prior extreme to anchor the SAR).
    """
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    close = bars["close"].to_numpy()
    n = len(bars)

    sar_arr = np.full(n, np.nan)
    dir_arr = np.full(n, np.nan)
    af_arr = np.full(n, np.nan)
    ep_arr = np.full(n, np.nan)

    if n < 2:
        return pd.DataFrame(
            {"sar": sar_arr, "dir": dir_arr, "af": af_arr, "ep": ep_arr},
            index=bars.index,
        )

    # Initialise bar 0 — Wilder: long if close > open (or just use long as default)
    direction = 1 if close[0] >= (bars["open"].iloc[0] if "open" in bars.columns else close[0]) else -1
    af = af_init
    ep = high[0] if direction == 1 else low[0]
    sar = low[0] if direction == 1 else high[0]

    sar_arr[0] = sar
    dir_arr[0] = direction
    af_arr[0] = af
    ep_arr[0] = ep

    for i in range(1, n):
        # Project new SAR
        sar_new = sar + af * (ep - sar)

        # Adjust: SAR cannot be inside the prior two bars on the trend side
        if direction == 1:
            sar_new = min(sar_new, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
        else:
            sar_new = max(sar_new, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])

        # Check for flip
        flipped = (direction == 1 and sar_new > low[i]) or \
                  (direction == -1 and sar_new < high[i])
        if flipped:
            direction = -direction
            sar_new = ep           # new SAR is the prior extreme
            ep = low[i] if direction == -1 else high[i]
            af = af_init
        else:
            # Update EP and AF
            if direction == 1 and high[i] > ep:
                ep = high[i]
                af = min(af + af_step, af_max)
            elif direction == -1 and low[i] < ep:
                ep = low[i]
                af = min(af + af_step, af_max)

        sar = sar_new
        sar_arr[i] = sar
        dir_arr[i] = direction
        af_arr[i] = af
        ep_arr[i] = ep

    return pd.DataFrame(
        {"sar": sar_arr, "dir": dir_arr, "af": af_arr, "ep": ep_arr},
        index=bars.index,
    )


def flip_entries(
    bars: pd.DataFrame,
    af_init: float = SAR_AF_INIT,
    af_step: float = SAR_AF_STEP,
    af_max: float = SAR_AF_MAX,
) -> pd.DataFrame:
    """Bars where the Parabolic SAR direction flips.

    ``dir`` = +1 for a flip to long (long entry), -1 for a flip to short (short entry).
    Detected from the sign change of the SAR direction between consecutive bars — known
    at the *close* of the flip bar, trade taken at the next bar's open (no look-ahead).
    """
    st = parabolic_sar(bars, af_init=af_init, af_step=af_step, af_max=af_max)
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
# ATR (Wilder-style RMA) — risk unit for the symmetric barriers
# ---------------------------------------------------------------------------
def atr(bars: pd.DataFrame, n: int = 20) -> pd.Series:
    """Wilder-style (RMA) average true range, in price units.

    Uses EWM with com = n-1 (alpha = 1/n), the form Wilder originally used and
    TradingView adopts.  This is the risk unit R for the symmetric barrier backtest.
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
    return tr.ewm(com=n - 1, min_periods=n, adjust=False).mean()


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
    trade is closed at the last available bar's close.  When a single bar straddles both
    barriers the **stop is assumed first** (conservative fill).

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
# Trade-ledger summary
# ---------------------------------------------------------------------------
def summarize(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline per-trade statistics for one ledger.

    Returns trade count, win-rate, mean return (bps/trade), the per-trade Sharpe, the
    P&L skew, and a HAC Newey-West t-stat on the mean — the inference-bar number that
    decides whether the edge is distinguishable from zero.
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
        # Newey-West HAC t-stat — same kernel as Studies 72, 78, 106
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
