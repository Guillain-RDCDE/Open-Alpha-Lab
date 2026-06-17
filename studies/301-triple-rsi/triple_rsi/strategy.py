"""The strategy and its honest controls — Study 301 (Triple-RSI).

The "Triple RSI" recipe (popularised by QuantifiedStrategies.com and similar vendor
backtests, claiming a ~90% win rate on SPY). Four entry conditions, three of them on the
5-period RSI:

1. RSI(5) is below 30 (oversold), **and**
2. RSI(5) printed *down* for the third day in a row, **and**
3. RSI(5) three trading days ago was below 60, **and**
4. the close is above the 200-day moving average.

If all four hold, **buy at the close**. **Sell at the close when RSI(5) crosses above 50.**

We implement it as a per-trade ledger and pin it against the one comparison that decides
whether it is anything more than "oversold usually bounces in a bull market": the **same
entry bars with a random direction** (long or short with equal probability), using the
same exit rule. If the rule knows something, it beats the random control; if not, it
doesn't.

The headline claim is a **90% win rate**, so the study's third axis interrogates exactly
that: a high hit-rate with a fast "take the bounce" exit is the textbook shape of an
*exit-asymmetry illusion* — many tiny wins, rare large losses — not necessarily a
positive expectancy. We measure win-rate, mean, skew and the HAC t-stat side by side.

Look-ahead: the published rule says "buy at the close" of the signal bar, so the
canonical engine enters at the signal bar's close (``entry="close"``). Because that uses
the closing price to both *form* the signal and *fill* the trade, we also expose a
conservative ``entry="next_open"`` variant (enter at the next bar's open) as a robustness
check — if the edge only survives the same-bar-close fill, it is a measurement artefact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def rsi(close: pd.Series, n: int = 5) -> pd.Series:
    """Wilder RSI of period ``n``. Stamped on the *close* bar that completes it."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    # Wilder smoothing = EMA with alpha=1/n.
    avg_up = up.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    avg_dn = dn.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    rs = avg_up / avg_dn.replace(0.0, np.inf)
    return 100.0 - 100.0 / (1.0 + rs)


def sma(close: pd.Series, n: int) -> pd.Series:
    """Simple moving average of period ``n``."""
    return close.rolling(n, min_periods=n).mean()


# ---------------------------------------------------------------------------
# Signal generation — the four conditions
# ---------------------------------------------------------------------------
def triple_rsi_entries(
    bars: pd.DataFrame,
    rsi_n: int = 5,
    rsi_buy: float = 30.0,
    down_days: int = 3,
    was_below: float = 60.0,
    was_below_lag: int = 3,
    trend_sma: int | None = 200,
) -> pd.Series:
    """Bars satisfying the four Triple-RSI entry conditions.

    Returns a boolean Series indexed by ``bars.index``, True on entry-signal bars (the
    signal is stamped on the *close* of bar *t*).

    Parameters
    ----------
    bars :
        Daily OHLCV frame.
    rsi_n :
        RSI period (canonical 5).
    rsi_buy :
        Condition 1 — RSI(``rsi_n``) below this threshold (canonical 30).
    down_days :
        Condition 2 — RSI fell for this many consecutive days, i.e. RSI is down today
        and on each of the prior ``down_days - 1`` bars (canonical 3 → "down for the
        third day in a row").
    was_below :
        Condition 3 — RSI ``was_below_lag`` days ago was below this level (canonical 60).
    was_below_lag :
        The lag for condition 3 (canonical 3 trading days ago).
    trend_sma :
        Condition 4 — require close > SMA(``trend_sma``) (canonical 200). Set to None to
        disable and see the result without the trend filter.
    """
    close = bars["close"]
    r = rsi(close, rsi_n)

    cond_oversold = r < rsi_buy                       # (1)

    down = r.diff() < 0                               # RSI lower than the prior bar
    cond_down = down.copy()                           # (2) down for ``down_days`` in a row
    for k in range(1, down_days):
        cond_down = cond_down & down.shift(k)

    cond_was_below = r.shift(was_below_lag) < was_below  # (3)

    signal = cond_oversold & cond_down & cond_was_below
    if trend_sma is not None:
        signal = signal & (close > sma(close, trend_sma))  # (4)
    return signal.fillna(False)


def random_directions(n: int, seed: int = 0) -> np.ndarray:
    """A reproducible vector of ±1 — the control arm's coin."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=n)


# ---------------------------------------------------------------------------
# Forward-return engine
# ---------------------------------------------------------------------------
def run_trades(
    bars: pd.DataFrame,
    signal: pd.Series,
    rsi_n: int = 5,
    rsi_exit: float = 50.0,
    max_hold: int = 100,
    cost_bps: float = 0.0,
    direction: int = 1,
    entry: str = "close",
    random_dirs: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run Triple-RSI trades from ``signal`` and return a per-trade ledger.

    For each signal bar the trade is entered at that bar's **close** (the published rule)
    when ``entry="close"``, or at the *next* bar's open when ``entry="next_open"`` (the
    conservative look-ahead-free variant). The exit mirrors the recipe: close the trade
    on the first bar whose RSI(``rsi_n``) closes **above** ``rsi_exit`` (canonical 50), or
    after ``max_hold`` trading days, whichever comes first. The last available bar also
    forces a flat.

    ``cost_bps`` is a round-trip cost deducted from the gross return.

    Columns: ``entry_date, dir, entry, exit, exit_reason, days_held, ret_gross, ret_net``.
    """
    close = bars["close"]
    open_ = bars["open"]
    r = rsi(close, rsi_n)

    signal_locs = np.where(signal.to_numpy())[0]
    n_bars = len(bars)

    dirs = (
        np.asarray(random_dirs, dtype=int)
        if random_dirs is not None
        else np.full(len(signal_locs), direction, dtype=int)
    )

    rows = []
    for k, i in enumerate(signal_locs):
        d = dirs[k]
        if entry == "close":
            e = i                       # fill at the signal bar's close
            entry_px = close.iat[e]
            first_exit_scan = e + 1     # exit can only happen on a *later* bar
        elif entry == "next_open":
            e = i + 1                   # fill at the next bar's open
            if e >= n_bars:
                continue
            entry_px = open_.iat[e]
            first_exit_scan = e         # exit can be checked from the entry bar on
        else:
            raise ValueError(f"unknown entry mode {entry!r}")

        exit_px = None
        exit_reason = None
        last = e
        hold_cap = min(first_exit_scan + max_hold, n_bars)
        for j in range(first_exit_scan, hold_cap):
            last = j
            rsi_j = r.iat[j]
            if np.isfinite(rsi_j):
                # Long exits when RSI rises above rsi_exit; short mirrors it.
                exit_cond = (rsi_j > rsi_exit) if d > 0 else (rsi_j < (100.0 - rsi_exit))
                if exit_cond:
                    exit_px = close.iat[j]
                    exit_reason = "rsi_exit"
                    break
        if exit_px is None:
            exit_px = close.iat[last]
            held = last - e
            exit_reason = "max_hold" if held >= max_hold else "eod"

        ret_gross = d * (exit_px - entry_px) / entry_px
        rows.append(
            {
                "entry_date": bars.index[e],
                "dir": int(d),
                "entry": entry_px,
                "exit": exit_px,
                "exit_reason": exit_reason,
                "days_held": max(last - e, 1),
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

    Returns trade count, win-rate, mean return (bps/trade), per-trade Sharpe, skew,
    and a HAC t-stat on the mean — the inference-bar number that decides whether the
    edge is distinguishable from zero. Win-rate and skew are reported side by side so the
    "90% win rate" claim can be read against the *expectancy* it is supposed to imply.
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
            float(r.mean() / r.std(ddof=1))
            if n > 1 and r.std() > 0
            else float("nan")
        ),
        "skew": float(pd.Series(r).skew()) if n > 2 else float("nan"),
        "tstat": float("nan"),
    }
    if n > 5:
        # Newey-West HAC — no hard dependency on quantlab being importable.
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
# Split helper (regime / sub-period robustness)
# ---------------------------------------------------------------------------
def split_ledger(
    ledger: pd.DataFrame,
    split_year: int = 2010,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a ledger into pre- and post-``split_year`` halves by ``entry_date``."""
    cut = pd.Timestamp(f"{split_year}-01-01")
    pre = ledger[ledger["entry_date"] < cut].copy()
    post = ledger[ledger["entry_date"] >= cut].copy()
    return pre, post
