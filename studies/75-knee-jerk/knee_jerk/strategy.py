"""The strategy and its honest controls — Study 75 (Knee-Jerk).

The Connors RSI(2) mean-reversion recipe (from Connors & Alvarez, *Short Term Trading
Strategies That Work*, 2008): buy when the 2-period RSI drops below 10 (extreme
oversold); exit when RSI(2) closes above 60, or after N hold days, whichever comes
first. An optional 200-day SMA filter keeps us long-only in an uptrend.

We implement it as a per-trade ledger and pin it against the one comparison that decides
whether it is anything more than a dice roll: the **same entry bars with a random
direction** (long or short with equal probability), using the same exit rule. If the
rule knows something, it beats the random control; if not, it doesn't.

Two key split checks:
- **Pre/post 2009** — the Connors book was published in 2008/2009. Did the edge decay
  after publication? We split the tape and report each half's HAC t-stat.
- **With/without the 200-SMA trend filter** — does trend context change the verdict?

No look-ahead: the RSI(2) is computed on closes up to and including bar *t*; the trade
is entered at bar *t+1*'s open; exits are checked from *t+1* onward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def rsi(close: pd.Series, n: int = 2) -> pd.Series:
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
# Signal generation
# ---------------------------------------------------------------------------
def rsi2_entries(
    bars: pd.DataFrame,
    rsi_enter: float = 10.0,
    trend_sma: int | None = 200,
) -> pd.Series:
    """Bars where RSI(2) < ``rsi_enter`` (and price is above SMA(``trend_sma``) if set).

    Returns a boolean Series indexed by ``bars.index``, True on entry signal bars.
    The signal is stamped on the *close* of bar *t*; the trade enters at *t+1*'s open.

    Parameters
    ----------
    bars :
        Daily OHLCV frame.
    rsi_enter :
        RSI(2) threshold below which a signal is generated.  Connors' canonical value is 10.
    trend_sma :
        If not None, also require price > SMA(n) on the signal bar (the uptrend filter).
        Set to None to disable and see the unfiltered result.
    """
    close = bars["close"]
    r = rsi(close, 2)
    signal = r < rsi_enter
    if trend_sma is not None:
        ma = sma(close, trend_sma)
        signal = signal & (close > ma)
    return signal


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
    rsi_exit: float = 60.0,
    max_hold: int = 10,
    cost_bps: float = 2.0,
    direction: int = 1,
    random_dirs: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run RSI(2) trades from ``signal`` and return a per-trade ledger.

    For each signal bar the trade is entered at the *next* bar's open (long, or in the
    direction of ``random_dirs`` if provided). The exit rule mirrors Connors: close when
    RSI(2) closes above ``rsi_exit``, or after ``max_hold`` trading days, whichever
    comes first. The last available bar also forces a flat.

    ``cost_bps`` is a round-trip cost deducted from the gross return.

    Columns: ``entry_date, dir, entry, exit, exit_reason, days_held, ret_gross, ret_net``.
    """
    close = bars["close"]
    open_ = bars["open"]
    r2 = rsi(close, 2)

    # Signal bar positions (indices where signal is True).
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
        e = i + 1  # enter at next bar's open
        if e >= n_bars:
            continue
        entry_px = open_.iat[e]

        exit_px = None
        exit_reason = None
        last = e
        for j in range(e, min(e + max_hold, n_bars)):
            last = j
            # Check RSI(2) exit: if d=+1 we exit when RSI > rsi_exit; vice-versa.
            rsi_j = r2.iat[j]
            if np.isfinite(rsi_j):
                exit_cond = (rsi_j > rsi_exit) if d > 0 else (rsi_j < (100.0 - rsi_exit))
                if exit_cond:
                    exit_px = close.iat[j]
                    exit_reason = "rsi_exit"
                    break
        if exit_px is None:
            exit_px = close.iat[last]
            exit_reason = "max_hold" if last - e + 1 >= max_hold else "eod"

        ret_gross = d * (exit_px - entry_px) / entry_px
        rows.append(
            {
                "entry_date": bars.index[e],
                "dir": int(d),
                "entry": entry_px,
                "exit": exit_px,
                "exit_reason": exit_reason,
                "days_held": last - e + 1,
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
    edge is distinguishable from zero.
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
# Publication-decay split
# ---------------------------------------------------------------------------
PUBLICATION_YEAR = 2009  # Connors & Alvarez book published 2008; widely known by 2009.


def split_ledger(
    ledger: pd.DataFrame,
    split_year: int = PUBLICATION_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a ledger into pre- and post-publication halves.

    Returns ``(pre, post)`` where ``pre`` contains trades whose ``entry_date`` is before
    ``split_year-01-01``, and ``post`` contains trades from that date onward.
    """
    cut = pd.Timestamp(f"{split_year}-01-01")
    pre = ledger[ledger["entry_date"] < cut].copy()
    post = ledger[ledger["entry_date"] >= cut].copy()
    return pre, post
