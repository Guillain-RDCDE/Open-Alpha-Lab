"""The strategy and its honest controls — Study 310 (Platinum-Palladium).

The folk recipe: track the platinum/palladium ratio (platinum price / palladium price).
When the ratio is extremely *high* (palladium cheap relative to platinum), go long
palladium / short platinum, betting the ratio falls back. When the ratio is extremely
*low* (platinum cheap relative to palladium), go long platinum / short palladium. Both
metals are autocatalysts, so the relative price "should" anchor — or so the story goes.

We implement it as a z-score strategy:
- Compute a rolling z-score of the ratio over a lookback window.
- Enter long-spread when z > +threshold (ratio extreme high → long PA / short PL).
- Enter short-spread when z < -threshold (ratio extreme low → long PL / short PA).
- Exit when z reverts toward zero (|z| < exit_threshold).

The honest control is a **random-direction** baseline: same entry/exit dates, random
direction. We also benchmark against buy-and-hold of each metal.

Three structural checks distinguish mean-reversion evidence from a random walk:
1. **ADF test on the ratio** — does the ratio have a unit root, or is it stationary?
2. **Engle-Granger cointegration** — are platinum and palladium prices actually
   cointegrated, or just two correlated trending metals?
3. **Half-life estimation** — how long does a ratio deviation take to decay, if at all?

The killer for this pair is *regime change*: the 2018–2022 palladium boom flipped the
ratio below 1.0 for years, so "the mean it reverts to" was itself non-constant. The
structural tests over the full window will reflect that instability.

No look-ahead: signals are computed on closes up to day *t*; the spread is held from the
entry close to the exit close (the daily-data approximation of a next-bar fill).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Ratio & z-score
# ---------------------------------------------------------------------------
def compute_ratio(platinum: pd.DataFrame, palladium: pd.DataFrame) -> pd.Series:
    """Platinum/palladium ratio from close prices, aligned on the common index."""
    p = platinum["close"].rename("platinum")
    d = palladium["close"].rename("palladium")
    aligned = pd.concat([p, d], axis=1).dropna()
    ratio = aligned["platinum"] / aligned["palladium"]
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

    A p-value < 0.05 suggests the series is mean-reverting. For the ratio, a low p-value
    would support the folk-trade's premise; a high p-value says "random walk". Returns
    NaN if statsmodels is unavailable (so the offline core never hard-depends on it).
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
    X = np.column_stack([np.ones(len(y_lag)), y_lag])
    try:
        coef, *_ = np.linalg.lstsq(X, delta_y, rcond=None)
        beta = coef[1]
        if beta >= 0:
            return float("inf")
        return float(-np.log(2) / np.log(1 + beta))
    except np.linalg.LinAlgError:
        return float("nan")


def engle_granger_pvalue(platinum: pd.Series, palladium: pd.Series) -> float:
    """Engle-Granger cointegration test p-value.

    Regresses log(platinum) on log(palladium), tests the residual for stationarity (ADF).
    If p < 0.05, the series are cointegrated and a pairs strategy has structural support.
    Returns NaN if statsmodels is unavailable.
    """
    try:
        from statsmodels.tsa.stattools import coint

        aligned = pd.concat([platinum, palladium], axis=1).dropna()
        aligned = aligned[(aligned.iloc[:, 0] > 0) & (aligned.iloc[:, 1] > 0)]
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
    - ``dir``: +1 means "ratio will fall" (long palladium / short platinum), -1 means
      "ratio will rise" (long platinum / short palladium).
    - ``z_at_entry``: the z-score at the signal bar.

    Signals are detected at close of day *t* and acted on the next day (no look-ahead).
    Position state prevents re-entry while already in a trade and detects exits.
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
                # Ratio extremely high → expect reversion down → long PA / short PL
                in_trade = True
                current_dir = +1
                entry_idx = i
            elif zv < -enter_z:
                # Ratio extremely low → expect reversion up → long PL / short PA
                in_trade = True
                current_dir = -1
                entry_idx = i
        else:
            if abs(zv) < exit_z:
                rows.append(
                    {
                        "entry_date": dates[entry_idx],
                        "exit_date": dates[i],
                        "dir": current_dir,
                        "z_at_entry": z_vals[entry_idx],
                        "z_at_exit": zv,
                    }
                )
                in_trade = False
                current_dir = 0
                entry_idx = None

    return pd.DataFrame(rows)


def random_entries(entries: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Random-direction control: same entry/exit dates, random ±1 direction."""
    rng = np.random.default_rng(seed)
    out = entries.copy()
    out["dir"] = rng.choice([-1, 1], size=len(entries))
    return out


def run_trades(
    platinum: pd.DataFrame,
    palladium: pd.DataFrame,
    entries: pd.DataFrame,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Compute per-trade returns from the entry/exit schedule.

    Each trade is a spread: ``dir * (r_palladium - r_platinum)`` from entry to exit.
    ``dir = +1`` means long palladium / short platinum (bet ratio falls);
    ``dir = -1`` means long platinum / short palladium (bet ratio rises).

    The trade return is the log return of the spread over the holding period. ``cost_bps``
    is a round-trip charge applied once per trade to NAV (a two-leg futures spread; the
    short leg pays the same friction as the long, captured in the single round-trip total).

    Columns: ``entry_date, exit_date, dir, z_at_entry, days_held, trading_days,
    ret_gross, ret_net``.
    """
    plat_close = platinum["close"]
    pall_close = palladium["close"]

    common = plat_close.index.intersection(pall_close.index)
    plat_c = plat_close.reindex(common)
    pall_c = pall_close.reindex(common)

    rows = []
    for _, row in entries.iterrows():
        ed = row["entry_date"]
        xd = row["exit_date"]
        d = int(row["dir"])

        if ed not in common or xd not in common:
            continue
        p_entry = plat_c.loc[ed]
        p_exit = plat_c.loc[xd]
        d_entry = pall_c.loc[ed]
        d_exit = pall_c.loc[xd]
        if not (
            np.isfinite(p_entry)
            and np.isfinite(p_exit)
            and np.isfinite(d_entry)
            and np.isfinite(d_exit)
        ):
            continue
        if p_entry <= 0 or p_exit <= 0 or d_entry <= 0 or d_exit <= 0:
            continue

        r_plat = np.log(p_exit / p_entry)
        r_pall = np.log(d_exit / d_entry)
        # Spread return: dir +1 → long palladium short platinum; -1 → long platinum short palladium
        ret_gross = float(d * (r_pall - r_plat))
        days_held = (xd - ed).days

        try:
            i_entry = common.get_loc(ed)
            i_exit = common.get_loc(xd)
            trading_days = i_exit - i_entry
        except KeyError:
            trading_days = max(1, days_held // 7 * 5)

        rows.append(
            {
                "entry_date": ed,
                "exit_date": xd,
                "dir": d,
                "z_at_entry": row.get("z_at_entry", float("nan")),
                "days_held": days_held,
                "trading_days": trading_days,
                "ret_gross": ret_gross,
                "ret_net": ret_gross - cost_bps * 1e-4,
            }
        )
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

    Returns trade count, win-rate, mean return (bps/trade), per-trade Sharpe, P&L skew,
    and a HAC t-stat (Newey-West) on the mean — the inference-bar number that decides
    whether the edge is distinguishable from zero.
    """
    if len(ledger) == 0:
        return {
            k: float("nan")
            for k in ["n_trades", "win_rate", "mean_bps", "sharpe_per_trade", "skew", "tstat"]
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
        # Newey-West HAC t-stat — no hard dependency on quantlab being importable.
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


def block_bootstrap_ci(
    ledger: pd.DataFrame,
    col: str = "ret_net",
    block: int = 4,
    n_boot: int = 2000,
    seed: int = 310,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the mean per-trade return (in bps).

    Blocks of ``block`` consecutive trades preserve any short-run dependence in the
    trade sequence. Returns ``(lo, hi)`` at the ``1-alpha`` level, in bps. NaN if the
    ledger is too thin.
    """
    if len(ledger) < block + 1:
        return float("nan"), float("nan")
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < block + 1:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]) % n
        sample = r[idx.ravel()][:n]
        means[b] = sample.mean()
    lo = float(np.quantile(means, alpha / 2.0) * 1e4)
    hi = float(np.quantile(means, 1.0 - alpha / 2.0) * 1e4)
    return lo, hi
