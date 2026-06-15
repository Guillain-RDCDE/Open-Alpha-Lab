"""Strategy and honest controls — Study 196 (Long-Term-Reversal).

The De Bondt & Thaler (1985) recipe: sort stocks each month by their trailing
36–60 month total return; long the bottom quintile (past *losers*) and short (or
compare against) the top quintile (past *winners*); hold for 1–3 years. The
hypothesis is long-horizon overreaction: the market over-extrapolates multi-year
trends, so losers mean-revert upward and winners mean-revert downward.

Three honest comparisons settle the matter:

1. **Loser vs Winner** — the classic long-short spread (loser quintile minus
   winner quintile monthly return).
2. **Loser vs Equal-weight market** — the long-only loser excess return, which
   is the more practical benchmark.
3. **Random portfolio control** — a randomly-drawn quintile of the *same size*
   as the loser portfolio, seeded reproducibly, to test whether the loser signal
   carries information beyond pure concentration.

**No look-ahead**: the formation window for month t is the cumulative return
from month t−``formation`` through month t−1 (inclusive). The holding period
starts at month t (positions entered at the next month's open, i.e., the first
trading day of month t). Fundamentals lagged equivalently.

**Survivorship bias**: the real-tape universe is current S&P 500 projected
backwards. All real-tape results are upper bounds; the bias is named on the
Signal axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def trailing_return(
    prices: pd.DataFrame,
    formation: int = 36,
    skip: int = 1,
) -> pd.DataFrame:
    """Trailing ``formation``-month log cumulative return, lagged by ``skip`` months.

    ``skip = 1`` ensures the signal for month t is formed on data ending at
    month t-1 (no look-ahead). The return for ticker j at month t is::

        log(price_{t-1-0} / price_{t-1-formation})

    Tickers with fewer than ``formation`` valid closes return NaN.

    Parameters
    ----------
    prices:
        Month-end price panel (date × ticker).
    formation:
        Formation window in months (default 36 = 3 years).
    skip:
        Number of months of gap between end of formation window and the
        holding period (default 1, following the 1-month skip convention
        of the momentum/reversal literature to avoid microstructure effects).

    Returns
    -------
    DataFrame of trailing log-returns aligned to the holding-period month
    (i.e., signal.loc[t] is the signal *used* to form the portfolio at month t).
    """
    # log return over formation window, ending at t-skip
    log_prices = np.log(prices.replace(0, np.nan))
    # Shift skip months so we avoid look-ahead
    signal = log_prices.shift(skip) - log_prices.shift(skip + formation)
    signal.index.name = "date"
    return signal


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------
def quintile_returns(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    q: float = 0.20,
    hold: int = 1,
    min_stocks: int = 10,
) -> pd.DataFrame:
    """Monthly loser-minus-winner spread plus equal-weight market return.

    For each month t, stocks are sorted by signal.loc[t] into quintiles.
    The bottom ``q`` fraction (lowest trailing return = *losers*) is the
    long portfolio; the top ``q`` fraction (highest trailing return = *winners*)
    is the short leg.

    The forward return is the simple (not log) return from month t to t+hold:
    we use 1-month forward returns here and accumulate them when computing
    multi-month holding stats.

    Returns a DataFrame with columns:
    - ``loser``   : equal-weight mean return of the bottom quintile
    - ``winner``  : equal-weight mean return of the top quintile
    - ``market``  : equal-weight mean return of all ranked stocks
    - ``spread``  : loser − winner
    - ``n_total`` : number of stocks with valid signal and forward return
    - ``n_loser`` : stocks in the loser quintile
    - ``n_winner``: stocks in the winner quintile
    """
    # Monthly forward simple return
    fwd = prices.pct_change(periods=1).shift(-1)  # month t → t+1

    rows: list[dict] = []
    for t in signal.index:
        if t not in fwd.index:
            continue
        s = signal.loc[t].dropna()
        f = fwd.loc[t].dropna()
        common = s.index.intersection(f.index)
        if len(common) < min_stocks:
            continue
        s = s.loc[common]
        f = f.loc[common]

        lo_thresh = s.quantile(q)
        hi_thresh = s.quantile(1.0 - q)
        loser_mask = s <= lo_thresh
        winner_mask = s >= hi_thresh

        loser_ret = float(f[loser_mask].mean())
        winner_ret = float(f[winner_mask].mean())
        market_ret = float(f.mean())

        rows.append(
            {
                "date": t,
                "loser": loser_ret,
                "winner": winner_ret,
                "market": market_ret,
                "spread": loser_ret - winner_ret,
                "n_total": int(len(common)),
                "n_loser": int(loser_mask.sum()),
                "n_winner": int(winner_mask.sum()),
            }
        )
    result = pd.DataFrame(rows).set_index("date")
    result.index = pd.DatetimeIndex(result.index)
    return result


def random_portfolio_returns(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    q: float = 0.20,
    n_draws: int = 300,
    seed: int = 196,
    min_stocks: int = 10,
) -> pd.Series:
    """Null distribution of random-quintile excess returns (loser-quintile size).

    For each month, draw ``n_draws`` random subsets of the same number of stocks
    as the loser portfolio. Returns a Series of (random excess vs market) across
    all (month × draw) combinations — the null distribution for the loser excess.
    """
    rng = np.random.default_rng(seed)
    fwd = prices.pct_change(periods=1).shift(-1)
    excesses: list[float] = []

    for t in signal.index:
        if t not in fwd.index:
            continue
        s = signal.loc[t].dropna()
        f = fwd.loc[t].dropna()
        common = s.index.intersection(f.index)
        if len(common) < min_stocks:
            continue
        s = s.loc[common]
        f = f.loc[common]
        n_loser = max(1, int(round(len(s) * q)))
        mkt = f.to_numpy()
        mkt_mean = mkt.mean()
        all_idx = np.arange(len(mkt))
        for _ in range(n_draws):
            pick = rng.choice(all_idx, size=n_loser, replace=False)
            excesses.append(float(mkt[pick].mean() - mkt_mean))

    return pd.Series(excesses, name="random_excess")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summarize(annual_ret: pd.Series) -> dict:
    """Headline statistics for an annual (or monthly) return series.

    Uses Newey-West HAC standard error for the t-stat — appropriate for
    short/overlapping annual series where even one-lag autocorrelation
    can inflate naive t-stats.
    """
    r = pd.Series(annual_ret).astype(float).dropna()
    n = int(r.size)
    if n < 3:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "n")}

    mu = float(r.mean())
    std = float(r.std(ddof=1))
    sr = mu / std if std > 0 else float("nan")

    # Newey-West HAC t-stat
    e = r.to_numpy() - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se) if se > 0 else float("nan")

    eq = (1.0 + r).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())

    return {
        "mean": float(mu),
        "vol": float(std),
        "sharpe": float(sr),
        "tstat": float(tstat),
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": float(max_dd),
        "n": n,
    }


def annualise_monthly(stats: dict, periods: int = 12) -> dict:
    """Convert monthly stats to annualised equivalents."""
    out = dict(stats)
    if "mean" in out and np.isfinite(out["mean"]):
        out["mean_ann"] = float(out["mean"] * periods)
    if "vol" in out and np.isfinite(out["vol"]):
        out["vol_ann"] = float(out["vol"] * np.sqrt(periods))
    if "mean_ann" in out and "vol_ann" in out and out["vol_ann"] > 0:
        out["sharpe_ann"] = float(out["mean_ann"] / out["vol_ann"])
    return out


def turnover_cost_drag(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    q: float = 0.20,
    one_way_bps: float = 10.0,
    min_stocks: int = 10,
) -> pd.Series:
    """Monthly cost drag from portfolio turnover (assuming full rebalance each month).

    Each month the loser-quintile composition changes; the fraction of the
    portfolio that turns over is the overlap complement. Cost drag is
    turnover × 2 × one-way cost (round-trip).

    Returns a Series of monthly cost drag (in fraction, so 0.001 = 0.1% = 10 bps).
    """
    rows: list[dict] = []
    prev_losers: set = set()

    for t in signal.index:
        s = signal.loc[t].dropna()
        f = prices.loc[t].dropna() if t in prices.index else pd.Series(dtype=float)
        common = s.index.intersection(f.index)
        if len(common) < min_stocks:
            prev_losers = set()
            continue
        s = s.loc[common]
        lo_thresh = s.quantile(q)
        curr_losers = set(s[s <= lo_thresh].index.tolist())
        if prev_losers:
            retained = prev_losers & curr_losers
            turnover = 1.0 - len(retained) / len(curr_losers) if curr_losers else 1.0
        else:
            turnover = 1.0
        drag = turnover * 2.0 * one_way_bps * 1e-4
        rows.append({"date": t, "turnover": turnover, "drag": drag})
        prev_losers = curr_losers

    if not rows:
        return pd.Series(dtype=float, name="drag")
    df = pd.DataFrame(rows).set_index("date")
    return df["drag"].rename("drag")
