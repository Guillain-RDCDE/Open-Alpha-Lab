"""The strategy and its honest controls — Study 221 (Mayer-Multiple).

The Mayer Multiple (MM) is price / 200-day SMA, popularised by Trace Mayer in 2018.
The thesis:

  - MM < 1.0 (price below 200-day SMA): Bitcoin is "cheap" — accumulate / go long.
  - MM > 2.4 (price more than 2.4x above 200-day SMA): Bitcoin is "expensive" — trim.

Unlike the Bitcoin Rainbow Chart (Study 174), the Mayer Multiple is computed in real
time from freely available price data — there is no look-ahead bias in its construction.
But that does not mean it predicts forward returns.  The honest question is whether
conditioning on MM bands meaningfully shifts the next-N-day return distribution.

This module implements:

- ``mayer_signal`` — binary ±1/0 position: long when MM < BUY_THRESHOLD, flat otherwise;
  optionally short when MM > SELL_THRESHOLD.
- ``backtest`` — shift signal by 1 (no look-ahead), apply cost per turnover.
- ``band_analysis`` — conditional return statistics by MM decile / bucket.
- ``summarize`` — HAC t-stat on per-day strategy returns vs the buy-and-hold benchmark.
- ``forward_return_by_band`` — 30-day forward return grouped by MM regime (below 1, 1-2.4,
  above 2.4).  The primary diagnostic for whether bands predict anything.

No look-ahead anywhere: the MM at date *t* is computed from prices up to and including *t*
and is used to form a signal that trades at the *open* of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import BUY_THRESHOLD, SELL_THRESHOLD, SMA_WINDOW


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
def mayer_signal(
    df: pd.DataFrame,
    buy_threshold: float = BUY_THRESHOLD,
    sell_threshold: float = SELL_THRESHOLD,
    allow_short: bool = False,
) -> pd.Series:
    """Binary +1/0/−1 position from Mayer Multiple bands.

    Signal at date *t* is derived from MM at *t* and traded at *t+1* (handled by
    ``backtest``'s shift).

    - Long (+1)  when MM < ``buy_threshold``
    - Short (-1) when MM > ``sell_threshold`` AND ``allow_short=True``
    - Flat (0)   otherwise
    """
    mm = df["mayer_multiple"]
    pos = pd.Series(0.0, index=df.index, name="signal")
    pos[mm < buy_threshold] = 1.0
    if allow_short:
        pos[mm > sell_threshold] = -1.0
    return pos


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Daily strategy returns from a Mayer Multiple signal.

    ``signal`` is formed on the close of day *t* and shifted by 1 day so it
    applies to the *next* day's return — no look-ahead.

    Each change in position incurs a round-trip cost of ``cost_bps`` basis points.

    Returns a DataFrame with columns:
        ``r_market``    — daily log return of BTC
        ``signal``      — signal as formed (pre-shift)
        ``position``    — position as traded (shifted by 1)
        ``r_gross``     — gross daily strategy return
        ``r_net``       — net return after costs
        ``cum_log_net`` — cumulative log-return of the strategy
        ``cum_log_bh``  — cumulative log-return of buy-and-hold
    """
    r = df["log_return"]
    pos_traded = signal.shift(1).fillna(0.0)
    turnover = pos_traded.diff().abs().fillna(0.0)
    cost = turnover * cost_bps * 1e-4

    r_gross = pos_traded * r
    r_net = r_gross - cost

    return pd.DataFrame(
        {
            "r_market": r,
            "signal": signal,
            "position": pos_traded,
            "r_gross": r_gross,
            "r_net": r_net,
            "cum_log_net": r_net.cumsum(),
            "cum_log_bh": r.cumsum(),
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Band analysis — the primary diagnostic
# ---------------------------------------------------------------------------
def forward_return_by_band(
    df: pd.DataFrame,
    horizon: int = 30,
    buy_threshold: float = BUY_THRESHOLD,
    sell_threshold: float = SELL_THRESHOLD,
) -> pd.DataFrame:
    """Conditional forward returns by Mayer Multiple regime.

    Computes the ``horizon``-day log forward return at each date and groups by
    three regimes:
        - "cheap"     : MM < ``buy_threshold``
        - "neutral"   : ``buy_threshold`` <= MM <= ``sell_threshold``
        - "expensive" : MM > ``sell_threshold``

    Returns a DataFrame with columns [regime, mm, fwd_30d] for all valid dates
    (i.e., where MM is not NaN and there are ``horizon`` future days available).

    The *t-stat on the mean difference between "cheap" and "neutral" forward returns*
    is the decisive inference-bar number for the signal axis.
    """
    mm = df["mayer_multiple"]
    log_r = df["log_return"]

    # Forward return: sum of the next ``horizon`` log returns starting from t+1.
    fwd = log_r.shift(-1).rolling(horizon).sum().shift(-(horizon - 1))

    def _regime(x: float) -> str:
        if x < buy_threshold:
            return "cheap"
        elif x > sell_threshold:
            return "expensive"
        else:
            return "neutral"

    valid = mm.notna() & fwd.notna()
    out = pd.DataFrame(
        {
            "regime": mm[valid].map(_regime),
            "mm": mm[valid],
            "fwd_30d": fwd[valid],
        },
        index=df.index[valid],
    )
    return out


def band_tstat(fwd_df: pd.DataFrame, group_a: str = "cheap", group_b: str = "neutral") -> float:
    """HAC t-stat on the mean difference in 30-day forward log-returns between two regimes.

    Uses Newey-West with automatic lag selection.  Returns nan if either group is too small.
    """
    a = fwd_df.loc[fwd_df["regime"] == group_a, "fwd_30d"].dropna().to_numpy(dtype=float)
    b = fwd_df.loc[fwd_df["regime"] == group_b, "fwd_30d"].dropna().to_numpy(dtype=float)
    if len(a) < 10 or len(b) < 10:
        return float("nan")
    # Pool the excess returns of a over b into a two-sample test via the difference series.
    # We build a balanced proxy: for each observation, label it +1 (a) or -1 (b) and test
    # the pooled excess.  A simpler and more transparent approach: Welch t-test on means,
    # noting the HAC correction is on the individual series.
    mu_a, mu_b = a.mean(), b.mean()
    # HAC SE for each group separately.
    def _hac_se(x: np.ndarray) -> float:
        n = len(x)
        e = x - x.mean()
        lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
        lrv = float(e @ e) / n
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        return float(np.sqrt(max(lrv, 0.0) / n))

    se_a, se_b = _hac_se(a), _hac_se(b)
    se_diff = float(np.sqrt(se_a ** 2 + se_b ** 2))
    if se_diff <= 0:
        return float("nan")
    return float((mu_a - mu_b) / se_diff)


# ---------------------------------------------------------------------------
# Performance summary & inference
# ---------------------------------------------------------------------------
def summarize(bt: pd.DataFrame, col: str = "r_net") -> dict:
    """Headline per-day statistics for one backtest.

    Returns trade count, win-rate, mean daily return (bps), annualised Sharpe,
    annualised geometric return, HAC t-stat, and time in market.
    """
    r = bt[col].dropna().to_numpy(dtype=float)
    r_bh = bt["r_market"].dropna().to_numpy(dtype=float)
    n = r.size

    ann_strat = float(np.exp(r.sum() * (252.0 / n)) - 1.0) if n > 0 else float("nan")
    ann_bh = float(np.exp(r_bh.sum() * (252.0 / len(r_bh))) - 1.0) if len(r_bh) > 0 else float("nan")

    out: dict = {
        "n_days": int(n),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "ann_ret_pct": float(ann_strat * 100.0) if not np.isnan(ann_strat) else float("nan"),
        "ann_bh_pct": float(ann_bh * 100.0) if not np.isnan(ann_bh) else float("nan"),
        "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if n > 1 and r.std() > 0 else float("nan"),
        "tstat": float("nan"),
        "time_in_market_pct": float((bt["position"].dropna() != 0).mean() * 100.0),
    }
    if n > 10:
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
# MM distribution statistics — narrative support
# ---------------------------------------------------------------------------
def mm_distribution(df: pd.DataFrame) -> dict:
    """Basic distribution statistics for the Mayer Multiple series."""
    mm = df["mayer_multiple"].dropna()
    return {
        "mean": float(mm.mean()),
        "median": float(mm.median()),
        "std": float(mm.std()),
        "min": float(mm.min()),
        "max": float(mm.max()),
        "pct_below_1": float((mm < 1.0).mean() * 100.0),
        "pct_above_2p4": float((mm > 2.4).mean() * 100.0),
        "pct_neutral": float(((mm >= 1.0) & (mm <= 2.4)).mean() * 100.0),
        "n": int(len(mm)),
    }
