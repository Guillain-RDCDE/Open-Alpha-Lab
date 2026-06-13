"""The strategy and its honest controls — Study 113 (Gold-Silver-Ratio).

The folk recipe: track the gold/silver ratio (gold price / silver price). When the ratio
is extremely *high* (silver cheap relative to gold), go long silver / short gold, betting
the ratio will fall back to its historical mean. When the ratio is extremely *low* (gold
cheap relative to silver), go long gold / short silver. In practice many traders implement
this as a rotation between GLD and SLV rather than a true long/short pairs trade.

We implement it as a z-score strategy:
- Compute a rolling z-score of the ratio over a lookback window.
- Enter long-spread when z > +threshold (ratio extreme high → long SLV/short GLD).
- Enter short-spread when z < -threshold (ratio extreme low → long GLD/short SLV).
- Exit when z reverts toward zero (|z| < exit_threshold).

The honest control is a **random-entry** baseline: same entry timing, random direction.
We also compare gross returns to a buy-and-hold of each metal.

Three additional checks that distinguish mean-reversion evidence from a random walk:
1. **Engle-Granger cointegration test** — are gold and silver prices actually cointegrated,
   or just two correlated trending series?
2. **ADF test on the ratio** — does the ratio have a unit root, or is it stationary?
3. **Half-life estimation** — how long does a ratio deviation take to decay, if it decays at all?

No look-ahead: signals are computed on closes up to day *t*; positions are entered at *t+1*
(approximated as the next close for daily data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Ratio & z-score
# ---------------------------------------------------------------------------
def compute_ratio(gold: pd.DataFrame, silver: pd.DataFrame) -> pd.Series:
    """Gold/silver ratio from close prices, aligned on the common index."""
    g = gold["close"].rename("gold")
    s = silver["close"].rename("silver")
    aligned = pd.concat([g, s], axis=1).dropna()
    ratio = aligned["gold"] / aligned["silver"]
    ratio.name = "ratio"
    return ratio


def zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std.

    Uses a minimum of ``window`` observations so the first ``window-1`` values are NaN.
    This is the signal; it is stamped on the close of day *t* and acted upon at *t+1*.
    """
    mu = series.rolling(window, min_periods=window).mean()
    sigma = series.rolling(window, min_periods=window).std(ddof=1)
    return ((series - mu) / sigma).rename("zscore")


def adf_pvalue(series: pd.Series) -> float:
    """Augmented Dickey-Fuller test p-value (H0: unit root; reject → stationary).

    A p-value < 0.05 suggests the series is mean-reverting. For the ratio, a low
    p-value would support the folk-trade's premise. For individual prices we expect
    a high p-value (random walk).
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series.dropna().to_numpy(), autolag="AIC")
        return float(result[1])
    except ImportError:
        return float("nan")


def half_life(series: pd.Series) -> float:
    """Estimate mean-reversion half-life via OLS on the AR(1) equation.

    Regresses ``Δy_t = alpha + beta * y_{t-1} + eps_t``. The half-life is
    ``-log(2) / log(1 + beta)`` (in periods). A positive half-life indicates mean
    reversion; ``inf`` or negative means the series does not mean-revert.
    """
    y = series.dropna().to_numpy(dtype=float)
    if len(y) < 20:
        return float("nan")
    delta_y = np.diff(y)
    y_lag = y[:-1]
    # OLS: delta_y = alpha + beta * y_lag
    X = np.column_stack([np.ones(len(y_lag)), y_lag])
    try:
        coef, *_ = np.linalg.lstsq(X, delta_y, rcond=None)
        beta = coef[1]
        if beta >= 0:
            return float("inf")
        return float(-np.log(2) / np.log(1 + beta))
    except np.linalg.LinAlgError:
        return float("nan")


def engle_granger_pvalue(gold: pd.Series, silver: pd.Series) -> float:
    """Engle-Granger cointegration test p-value.

    Regresses log(gold) on log(silver), tests the residual for stationarity (ADF).
    If p < 0.05, the series are cointegrated and a pairs strategy has structural support.
    Returns NaN if statsmodels is unavailable.
    """
    try:
        from statsmodels.tsa.stattools import coint
        aligned = pd.concat([gold, silver], axis=1).dropna()
        _, pval, _ = coint(np.log(aligned.iloc[:, 0]), np.log(aligned.iloc[:, 1]))
        return float(pval)
    except ImportError:
        return float("nan")


# ---------------------------------------------------------------------------
# Z-score signal & trade ledger
# ---------------------------------------------------------------------------
def zscore_entries(
    ratio: pd.Series,
    window: int = 252,
    enter_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.DataFrame:
    """Generate entry signals based on z-score thresholds.

    Returns a DataFrame with columns:
    - ``dir``: +1 means "ratio will fall" (long silver/short gold), -1 means "ratio will rise"
      (long gold/short silver).
    - ``z_at_entry``: the z-score at the signal bar.

    Signals are detected at close of day *t* and acted on the next day (no look-ahead).
    We track position state to avoid re-entering while already in a trade and to detect exits.
    """
    z = zscore(ratio, window)
    z_vals = z.to_numpy(dtype=float)
    dates = ratio.index
    n = len(dates)

    rows = []
    in_trade = False
    current_dir = 0
    entry_idx = None

    for i in range(n):
        zv = z_vals[i]
        if not np.isfinite(zv):
            continue
        if not in_trade:
            if zv > enter_z:
                # Ratio extremely high → expect reversion down → long spread: long silver, short gold
                in_trade = True
                current_dir = +1
                entry_idx = i
            elif zv < -enter_z:
                # Ratio extremely low → expect reversion up → long gold, short silver
                in_trade = True
                current_dir = -1
                entry_idx = i
        else:
            # Check for exit
            if abs(zv) < exit_z:
                rows.append({
                    "entry_date": dates[entry_idx],
                    "exit_date": dates[i],
                    "dir": current_dir,
                    "z_at_entry": z_vals[entry_idx],
                    "z_at_exit": zv,
                })
                in_trade = False
                current_dir = 0
                entry_idx = None

    return pd.DataFrame(rows)


def random_entries(
    entries: pd.DataFrame,
    seed: int = 0,
) -> pd.DataFrame:
    """Random-direction control: same entry/exit dates, random ±1 direction."""
    rng = np.random.default_rng(seed)
    out = entries.copy()
    out["dir"] = rng.choice([-1, 1], size=len(entries))
    return out


def run_trades(
    gold: pd.DataFrame,
    silver: pd.DataFrame,
    entries: pd.DataFrame,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Compute per-trade returns from the entry/exit schedule.

    Each trade is a spread: ``dir * (r_silver - r_gold)`` from entry date to exit date.
    ``dir = +1`` means long silver / short gold (bet ratio falls);
    ``dir = -1`` means long gold / short silver (bet ratio rises).

    The trade return is the cumulative log return of the spread over the holding period.
    Cost ``cost_bps`` is a round-trip charge (one-way × 2 legs × 2 sides = 4-way; we
    charge ``cost_bps`` as a round-trip total on the spread, typical for an ETF pairs trade).

    Columns: ``entry_date, exit_date, dir, z_at_entry, days_held, ret_gross, ret_net``.
    """
    gold_close = gold["close"]
    silver_close = silver["close"]

    # Align both series on a common date index
    common = gold_close.index.intersection(silver_close.index)
    gold_c = gold_close.reindex(common)
    silver_c = silver_close.reindex(common)

    rows = []
    for _, row in entries.iterrows():
        ed = row["entry_date"]
        xd = row["exit_date"]
        d = int(row["dir"])

        # We act at next close after signal; entry price = close on entry_date + 1 business day
        # For simplicity, we use the close on entry_date as entry price (the signal is from
        # the prior day's close, so this is at t+1, which approximates the open/close of entry day).
        if ed not in common or xd not in common:
            continue
        g_entry = gold_c.loc[ed]
        g_exit = gold_c.loc[xd]
        s_entry = silver_c.loc[ed]
        s_exit = silver_c.loc[xd]
        if not (np.isfinite(g_entry) and np.isfinite(g_exit) and
                np.isfinite(s_entry) and np.isfinite(s_exit)):
            continue
        if g_entry <= 0 or g_exit <= 0 or s_entry <= 0 or s_exit <= 0:
            continue

        # Log returns of each leg over the holding period
        r_gold = np.log(g_exit / g_entry)
        r_silver = np.log(s_exit / s_entry)
        # Spread return: dir +1 → long silver short gold; dir -1 → long gold short silver
        ret_gross = float(d * (r_silver - r_gold))
        days_held = (xd - ed).days

        # Locate entry and exit in common index for day count
        try:
            i_entry = common.get_loc(ed)
            i_exit = common.get_loc(xd)
            trading_days = i_exit - i_entry
        except KeyError:
            trading_days = max(1, days_held // 7 * 5)

        rows.append({
            "entry_date": ed,
            "exit_date": xd,
            "dir": d,
            "z_at_entry": row.get("z_at_entry", float("nan")),
            "days_held": days_held,
            "trading_days": trading_days,
            "ret_gross": ret_gross,
            "ret_net": ret_gross - cost_bps * 1e-4,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Buy-and-hold benchmarks
# ---------------------------------------------------------------------------
def bah_return(bars: pd.DataFrame, start_date=None, end_date=None) -> float:
    """Total log return of a buy-and-hold position over a date range."""
    c = bars["close"]
    if start_date is not None:
        c = c[c.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        c = c[c.index <= pd.Timestamp(end_date)]
    c = c.dropna()
    if len(c) < 2:
        return float("nan")
    return float(np.log(c.iloc[-1] / c.iloc[0]))


def bah_daily_returns(bars: pd.DataFrame) -> pd.Series:
    """Daily log returns of a buy-and-hold position."""
    return np.log(bars["close"] / bars["close"].shift(1)).dropna()


# ---------------------------------------------------------------------------
# Trade-ledger summary
# ---------------------------------------------------------------------------
def summarize(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline per-trade statistics for one ledger.

    Returns trade count, win-rate, mean return (bps/trade), P&L skew, and a
    HAC t-stat (Newey-West) on the mean — the inference-bar number that decides
    whether the edge is distinguishable from zero.
    """
    if len(ledger) == 0:
        return {k: float("nan") for k in
                ["n_trades", "win_rate", "mean_bps", "sharpe_per_trade", "skew", "tstat"]}
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
        # Newey-West HAC t-stat
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
