"""The strategy and its honest controls — Study 87 (Center-Line).

The folk claim: 'price always returns to the session VWAP — fade large deviations and
you'll win.' We implement the VWAP-fade as a barrier backtest and pin it against the one
comparison that decides whether it is anything but a noise trade: the **same entries with
a random direction**.

Three pieces of analysis share one engine (:func:`run_trades`):

- **VWAP-fade (the claim)** — when price deviates from the running session VWAP by more
  than a threshold, go *against* the deviation (short when above VWAP, long when below),
  with a symmetric ±1 ATR exit. If reversion is real, this should beat a coin.
- **Random-direction control** — identical entry bars, coin-flip direction; the comparison
  that strips out any structural edge from the entry timing and isolates whether the VWAP
  direction signal adds value.
- **Threshold sweep** — test mild (0.5 ATR), moderate (1.0 ATR) and aggressive (2.0 ATR)
  deviation thresholds to check whether a sharper entry buys a better edge.

No look-ahead: the running VWAP and ATR are computed on closes and volume up to bar *t*;
the trade is entered at bar *t+1*'s open; barriers are checked from *t+1* onward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def running_vwap(bars: pd.DataFrame) -> pd.Series:
    """Running session VWAP: cumulative (price * volume) / cumulative volume, per session.

    Reset at every new calendar date (new RTH session). Uses the bar's close as its
    representative price, weighted by bar volume. At bar *t* the VWAP includes bar *t*
    itself — no look-ahead because the signal is formed at close *t* and the trade opens
    at *t+1*.
    """
    date = bars.index.normalize()
    pv = bars["close"] * bars["volume"]
    cum_pv = pv.groupby(date).cumsum()
    cum_vol = bars["volume"].groupby(date).cumsum()
    vwap = cum_pv / cum_vol
    vwap.name = "vwap"
    return vwap


def atr(bars: pd.DataFrame, n: int = 20) -> pd.Series:
    """Wilder-style average true range (ATR), in price units."""
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


def vwap_deviation(bars: pd.DataFrame, vwap: pd.Series) -> pd.Series:
    """Normalised deviation of price from session VWAP, in ATR units.

    A positive value means price is *above* VWAP (the fade goes short); negative means
    price is *below* VWAP (the fade goes long). Expressed in ATR units so the threshold
    is scale-invariant.
    """
    r_unit = atr(bars, n=20)
    dev = (bars["close"] - vwap) / r_unit
    dev.name = "vwap_dev_atr"
    return dev


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------
def vwap_fade_entries(
    bars: pd.DataFrame,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """Bars where price deviates from session VWAP by more than ``threshold`` ATRs.

    Entry direction is the *fade*: when price is above VWAP we go short (dir = -1),
    when below we go long (dir = +1). Only bars where |deviation| >= threshold and
    ATR is defined are returned.

    Returns a DataFrame with column ``dir`` (±1) indexed by the signal bar timestamp.
    """
    vwap = running_vwap(bars)
    dev = vwap_deviation(bars, vwap)

    # Signal: |deviation| >= threshold, direction is opposite to deviation sign.
    mask = dev.abs() >= threshold
    direction = -np.sign(dev)  # fade: short when above VWAP, long when below
    direction = direction[mask].astype(int)

    out = pd.DataFrame({"dir": direction})
    out.index.name = bars.index.name
    # Drop bars where ATR was NaN (the first ~20 bars of history).
    out = out.dropna()
    out = out[out["dir"] != 0]
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
    stop ``sl_R * R`` away. The forward bars are scanned until a barrier is touched or —
    with ``flat_at_session_end`` — the session's last bar, where the position is marked
    out at the close (positions never held overnight). When a single bar straddles both
    barriers the **stop is assumed first** (the conservative fill).

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
    P&L skew, and a HAC t-stat on the mean — the inference-bar number that decides
    whether the edge is distinguishable from zero.
    """
    if ledger.empty:
        return {
            "n_trades": 0,
            "win_rate": float("nan"),
            "mean_bps": float("nan"),
            "sharpe_per_trade": float("nan"),
            "skew": float("nan"),
            "tstat": float("nan"),
        }
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
        # Newey-West HAC t-stat — local implementation so tests have no quantlab dep.
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
