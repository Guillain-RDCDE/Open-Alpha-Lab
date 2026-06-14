"""The strategy and its honest controls — Study 125 (Ichimoku-Cloud).

The folk recipe: plot the Ichimoku Kinko Hyo system (Hosoda, 1969) on the daily chart
with standard settings (Tenkan 9, Kijun 26, Senkou B 52, displacement 26). The signal:
go **long** when price is above the cloud (Kumo) *and* Tenkan-sen > Kijun-sen; go
**short** when price is below the cloud and Tenkan < Kijun. Enter on the close of the
signal bar, at the next bar's open.

No look-ahead: the Kumo (cloud) at bar *t* is constructed from Senkou Span A and B
values that were *displaced forward 26 bars*. In practice, to know the cloud bounds
at price bar *t* without look-ahead, we use the Senkou Span values computed from data
up to bar *t - 26* — i.e., we look at the cloud that was "published" 26 bars ago and
is now sitting at the current bar. This is the canonical Ichimoku no-look-ahead
treatment. See ``_ichimoku()`` for the exact implementation.

We pin the signal against the one comparison that settles whether it is anything but a
lagging trend rule: the **same flip dates with a random direction** on a symmetric
ATR-barrier backtest. Two exit regimes share one engine (:func:`run_trades`):

- **symmetric** — take-profit and stop at ±1 ATR(20) from entry. The only
  direction-fair payoff: a coin earns ≈ 0.
- **fixed-day hold** — hold for N calendar days then close. Checks whether the
  post-signal drift is positive at horizons that match Ichimoku's intended use.

Relationship to Study 72 (Loaded-Dice, SMA(5/10) on 5-minute bars), Study 78
(Crossed-Wires, MACD on daily bars), and Study 106 (Supertrend): Ichimoku is yet
another lagging trend system — same fundamental question, expected same answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Standard Ichimoku parameters (Hosoda original / TradingView default)
TENKAN_N = 9
KIJUN_N = 26
SENKOU_B_N = 52
DISPLACEMENT = 26


# ---------------------------------------------------------------------------
# Ichimoku Kinko Hyo indicator
# ---------------------------------------------------------------------------
def _donchian_mid(high: pd.Series, low: pd.Series, n: int) -> pd.Series:
    """Mid-point of the n-bar high/low channel — the Ichimoku line building block."""
    return (high.rolling(n, min_periods=n).max() + low.rolling(n, min_periods=n).min()) / 2.0


def ichimoku(
    bars: pd.DataFrame,
    tenkan_n: int = TENKAN_N,
    kijun_n: int = KIJUN_N,
    senkou_b_n: int = SENKOU_B_N,
    displacement: int = DISPLACEMENT,
) -> pd.DataFrame:
    """Canonical Ichimoku Kinko Hyo indicator.

    Returns a DataFrame aligned to ``bars.index`` with:
    - ``tenkan``    — Tenkan-sen (conversion line): mid of n=9 high/low channel.
    - ``kijun``     — Kijun-sen (base line): mid of n=26 high/low channel.
    - ``senkou_a``  — Senkou Span A: average of Tenkan and Kijun, **no-look-ahead
                      shifted**: the value at index *i* was computed from bars up to
                      *i - displacement* (i.e. bar *i - 26*) and displaced forward.
                      If ``i < displacement`` the value is NaN.
    - ``senkou_b``  — Senkou Span B: mid of n=52 high/low channel, same no-look-ahead
                      displacement. NaN where insufficient history.
    - ``chikou``    — Chikou Span (lagging span): today's close plotted 26 bars back;
                      shown for completeness, NOT used in the signal (it requires
                      comparing today's close to the price 26 bars ago — which is fine
                      for visual reference but introduces survivorship if used naively).
    - ``cloud_top`` — pointwise max(senkou_a, senkou_b); the upper Kumo boundary.
    - ``cloud_bot`` — pointwise min(senkou_a, senkou_b); the lower Kumo boundary.

    All values are aligned to the **current** bar index, with the no-look-ahead
    treatment of Senkou Spans already applied — consumers do not need to shift further.
    """
    high = bars["high"]
    low = bars["low"]
    close = bars["close"]

    tenkan = _donchian_mid(high, low, tenkan_n)
    kijun = _donchian_mid(high, low, kijun_n)

    # Senkou Span A and B: computed from history, then shifted forward by 'displacement'.
    # To get the no-look-ahead value at bar t, we compute A/B at bar (t - displacement)
    # and attribute it to bar t. Equivalently: compute the raw series, then .shift(displacement).
    raw_a = (tenkan + kijun) / 2.0
    raw_b = _donchian_mid(high, low, senkou_b_n)
    senkou_a = raw_a.shift(displacement)
    senkou_b = raw_b.shift(displacement)

    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
    chikou = close.shift(-displacement)  # for display only

    return pd.DataFrame(
        {
            "tenkan": tenkan,
            "kijun": kijun,
            "senkou_a": senkou_a,
            "senkou_b": senkou_b,
            "cloud_top": cloud_top,
            "cloud_bot": cloud_bot,
            "chikou": chikou,
        },
        index=bars.index,
    )


def signal_entries(
    bars: pd.DataFrame,
    tenkan_n: int = TENKAN_N,
    kijun_n: int = KIJUN_N,
    senkou_b_n: int = SENKOU_B_N,
    displacement: int = DISPLACEMENT,
) -> pd.DataFrame:
    """Bars where the Ichimoku composite signal fires or flips.

    The composite signal at bar *t* is:
    - **+1 (bullish)**: close > cloud_top *and* Tenkan > Kijun (TK cross to bull).
    - **-1 (bearish)**: close < cloud_bot *and* Tenkan < Kijun (TK cross to bear).
    - **0 (neutral/inside cloud)**: otherwise.

    An *entry* is any bar where the composite signal *changes* to ±1 from a different
    value (including 0). This means we only enter on transitions — a flip from neutral
    to bullish, from bearish to bullish, etc. — not re-enter on every bar while in a
    trend. The entry is stamped at the bar when the condition first fires; the trade is
    taken at the *next* bar's open (no look-ahead).

    All Ichimoku values are computed with the no-look-ahead treatment baked into
    :func:`ichimoku` — the cloud at bar *t* uses only data up to bar *t - displacement*.
    """
    ichi = ichimoku(bars, tenkan_n=tenkan_n, kijun_n=kijun_n,
                    senkou_b_n=senkou_b_n, displacement=displacement)
    close = bars["close"]

    above_cloud = close > ichi["cloud_top"]
    below_cloud = close < ichi["cloud_bot"]
    tk_bull = ichi["tenkan"] > ichi["kijun"]
    tk_bear = ichi["tenkan"] < ichi["kijun"]

    sig = pd.Series(0, index=bars.index, dtype=int)
    sig[above_cloud & tk_bull] = 1
    sig[below_cloud & tk_bear] = -1

    # Retain only *transitions* (signal change from one state to another ±1 state)
    prev = sig.shift(1)
    changed = sig.ne(prev) & sig.ne(0)
    out = pd.DataFrame({"dir": sig[changed].astype(int)})
    out.index.name = bars.index.name
    return out


def random_directions(n: int, seed: int = 0) -> np.ndarray:
    """A reproducible vector of ±1 — the control arm's coin."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=n)


# ---------------------------------------------------------------------------
# ATR (simple rolling, for barrier sizing)
# ---------------------------------------------------------------------------
def atr(bars: pd.DataFrame, n: int = 20) -> pd.Series:
    """Simple rolling average true range, in price units."""
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
    ``R = ATR(atr_n)`` measured at the entry bar. Take-profit sits ``tp_R * R`` away,
    stop ``sl_R * R`` away. If neither barrier is hit within ``max_hold`` bars the
    trade is closed at the last available bar's close. When a single bar straddles
    both barriers the **stop is assumed first** (conservative fill).

    ``directions`` overrides the entry signs (the random-control arm passes a ±1 vector
    aligned to ``entries``). ``cost_bps`` is a one-way round-trip cost charged on the
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
    cost_bps: float = 0.0,
    directions: np.ndarray | None = None,
) -> pd.DataFrame:
    """Hold for exactly ``hold_days`` bars after entry, close at the bar's close.

    A simple alternative to barrier exits for calendar-horizon analysis. Used to check
    whether the Ichimoku signal predicts a *drifting* forward return over a horizon
    that matches the indicator's typical holding period (multi-week to multi-month).
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
        # Newey-West HAC t-stat — same kernel as Studies 72, 78, 106 for apples-to-apples
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
