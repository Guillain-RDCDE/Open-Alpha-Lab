"""The strategy and its honest controls — Study 183 (Fisher-Transform).

Ehlers (2002) maps the midpoint of the rolling highest-high / lowest-low range into a
Gaussian via the atanh (Fisher) transform:

    midpoint(t) = (highest_high(N) + lowest_low(N)) / 2
    raw(t)      = 2 * (close(t) - lowest_low(N)) / (highest_high(N) - lowest_low(N)) - 1
    x(t)        = clamp(raw(t), -0.999, 0.999)
    Fisher(t)   = 0.5 * ln((1 + x(t)) / (1 - x(t))) = atanh(x(t))
    Trigger(t)  = Fisher(t - 1)

The folk rule: **buy** when Fisher crosses *above* Trigger; **sell/short** when Fisher
crosses *below* Trigger.  The study's central claim is that the transform is *monotone*
in the clamped normalised price ``x``, so Fisher-vs-Trigger crossovers carry no
information beyond what the underlying price extremes already contain.

**Three honest tests, one control arm:**

1. **Fisher/Trigger crossover** — the primary signal as described by Ehlers.
2. **Raw normalised value crossover** — replaces Fisher with the raw ``x``, showing
   that the transform's monotonicity means the crossover timestamps are *identical* to
   those of the Fisher crossover (up to numerical precision at the clamp boundary): the
   transform adds zero signal.
3. **Random-direction control** — the same entry bars, coin-flip direction: the
   baseline that decides whether the signal adds anything over chance.

All three share the same fixed-horizon forward-return engine (:func:`run_trades`).  No
look-ahead: the indicator is built on closes up to bar *t*; the trade enters at *t+1*'s
open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Ehlers Fisher Transform indicator
# ---------------------------------------------------------------------------
def fisher_transform(
    bars: pd.DataFrame,
    period: int = 10,
    clamp: float = 0.999,
) -> pd.DataFrame:
    """Ehlers (2002) Fisher Transform and its trigger line.

    Parameters
    ----------
    bars : DataFrame
        Daily OHLCV with columns ``high``, ``low``, ``close``.
    period : int
        Look-back window for highest-high / lowest-low (default 10, Ehlers' original).
    clamp : float
        Upper bound for ``|x|`` before the atanh (keeps the log finite; default 0.999).

    Returns
    -------
    DataFrame with columns:
      ``fisher``  — the Fisher Transform value at bar *t*.
      ``trigger`` — ``fisher`` lagged by 1 bar (Fisher at *t-1*).
      ``raw_x``   — the raw clamped normalised midpoint, before the transform.
    """
    hh = bars["high"].rolling(period, min_periods=period).max()
    ll = bars["low"].rolling(period, min_periods=period).min()
    rng = hh - ll

    # Normalise close into [-1, 1] via the midpoint of the highest-high / lowest-low.
    # Ehlers' original formula uses the midpoint price, not the close, but the close-
    # based version is more common in practice and easier to reproduce exactly.
    raw = pd.Series(
        np.where(rng > 0, 2.0 * (bars["close"] - ll) / rng - 1.0, 0.0),
        index=bars.index,
    )
    # Clamp to keep the atanh finite.
    x = raw.clip(-clamp, clamp)
    # Fisher transform: atanh(x) = 0.5 * ln((1+x)/(1-x)).
    fisher = 0.5 * np.log((1.0 + x) / (1.0 - x))
    trigger = fisher.shift(1)

    return pd.DataFrame(
        {"fisher": fisher, "trigger": trigger, "raw_x": x},
        index=bars.index,
    )


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------
def crossover_entries(ft: pd.DataFrame) -> pd.DataFrame:
    """Fisher/Trigger crossover signals — the primary Ehlers rule.

    A **buy signal** (+1) fires when Fisher crosses *above* Trigger; a **sell signal**
    (-1) fires when Fisher crosses *below* Trigger.  The cross is detected from the sign
    change of (Fisher − Trigger) between consecutive bars.

    Signals are known at the close of bar *t*; trades enter at bar *t+1*'s open.
    """
    diff = ft["fisher"] - ft["trigger"]
    sign = np.sign(diff)
    # A genuine cross: sign changes and neither side is NaN.
    crossed = sign.ne(sign.shift(1)) & sign.ne(0) & sign.shift(1).notna() & ft["fisher"].notna()
    out = pd.DataFrame({"dir": sign[crossed].astype(int)})
    out.index.name = ft.index.name
    return out


def raw_crossover_entries(ft: pd.DataFrame) -> pd.DataFrame:
    """Raw-x / (x_lagged) crossover — the same signal without the atanh wrapper.

    Because atanh is strictly monotone on (−1, +1), the sign of (Fisher − Trigger) is
    identical to the sign of (raw_x − raw_x_lagged) everywhere the clamp is not active.
    This function makes that explicit: if it produces the same entry bars as
    :func:`crossover_entries`, the transform added nothing.
    """
    raw = ft["raw_x"]
    raw_lag = raw.shift(1)
    diff = raw - raw_lag
    sign = np.sign(diff)
    crossed = sign.ne(sign.shift(1)) & sign.ne(0) & sign.shift(1).notna() & raw.notna()
    out = pd.DataFrame({"dir": sign[crossed].astype(int)})
    out.index.name = ft.index.name
    return out


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
    close of bar *t + hold_days* (or the last bar, whichever comes first).  No look-ahead:
    the signal is computed on data up to and including bar *t*.

    ``directions`` overrides the entry sign vector (the random-control arm passes a ±1
    vector aligned to ``entries``).  ``cost_bps`` is a round-trip cost in basis points
    charged on the gross return.

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
    out: dict = {
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
        # Newey-West HAC t-stat — no hard dependency on quantlab for portability.
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
# Monotonicity verification helper
# ---------------------------------------------------------------------------
def fisher_vs_raw_agreement(ft: pd.DataFrame) -> dict:
    """Verify that the Fisher and raw-x crossovers fire on the same bars.

    Returns the fraction of Fisher entry bars that coincide with a raw-x entry bar, and
    the fraction of raw-x entry bars that coincide with a Fisher entry bar.  Both should
    be 1.0 (or very close, up to the tiny fraction of bars where the clamp is active).
    This is the empirical proof that the transform adds no information.
    """
    f_ent = set(crossover_entries(ft).index)
    r_ent = set(raw_crossover_entries(ft).index)
    if not f_ent:
        return {"fisher_in_raw": float("nan"), "raw_in_fisher": float("nan"),
                "n_fisher": 0, "n_raw": 0}
    return {
        "fisher_in_raw": len(f_ent & r_ent) / len(f_ent),
        "raw_in_fisher": len(r_ent & f_ent) / len(r_ent) if r_ent else float("nan"),
        "n_fisher": len(f_ent),
        "n_raw": len(r_ent),
    }
