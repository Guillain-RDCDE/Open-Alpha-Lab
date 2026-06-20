"""Strategy and honest controls -- Study 329 (One-Month-Reversal).

The Jegadeesh (1990) recipe: each month, rank stocks by *last month's* return; long the
bottom quintile (last month's *losers*) and short (or compare against) the top quintile
(last month's *winners*); hold for one month and rebalance. The hypothesis is short-horizon
overreaction / liquidity provision: the crowd over-pushes one-month moves, and the
laggards mean-revert next month.

Three honest comparisons settle the matter:

1. **Loser vs Winner** -- the classic long-short spread (loser quintile minus winner
   quintile, dollar-neutral).
2. **Loser vs Equal-weight market** -- the long-only loser excess return.
3. **Random portfolio control** -- a random quintile of the *same size*, seeded
   reproducibly, to test whether the loser signal carries information beyond concentration.

**The microstructure trap.** A 1-month reversal measured with last month's *close* both
forming the signal and pricing the entry is the textbook bid-ask-bounce illusion (a stock
that closed on the bid prints a low return and "reverts" upward purely because the next
print lands on the ask). The canonical engine uses ``skip=0`` (Jegadeesh's own choice);
we also expose ``skip=1`` -- skip one month between formation and holding -- as the
microstructure-safe robustness check. If the edge only survives ``skip=0``, it is largely
bid-ask bounce, not a tradable signal.

**No look-ahead**: the signal for holding month t is the return ``skip`` months earlier
(``skip=0`` -> last month's return, known at the close of month t-1; ``skip=1`` -> the
return two months back). The return earned is month t's. One shift, applied once.

**Survivorship bias**: the real-tape universe is the current S&P 500 projected backwards.
All real-tape results are upper bounds; the bias is named on the Signal axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def trailing_return(prices: pd.DataFrame, skip: int = 0) -> pd.DataFrame:
    """One-month formation return, ``skip`` months before the holding month.

    The signal aligned to holding month t is the simple monthly return of month
    ``t - 1 - skip``. With ``skip = 0`` this is last month's return (the Jegadeesh
    convention); ``skip = 1`` leaves a one-month gap to defuse bid-ask bounce.

    Parameters
    ----------
    prices:
        Month-end price panel (date x ticker).
    skip:
        Months of gap between the formation month and the holding month (default 0).

    Returns
    -------
    DataFrame of one-month formation returns aligned to the holding month (i.e.
    ``signal.loc[t]`` is the signal *used* to form the portfolio held in month t).
    Burn-in rows are NaN.
    """
    ret = prices.pct_change()
    signal = ret.shift(1 + skip)  # return of month t-1-skip, aligned to month t
    signal.index.name = "date"
    return signal


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------
def quintile_returns(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    q: float = 0.20,
    min_stocks: int = 20,
) -> pd.DataFrame:
    """Monthly loser-minus-winner spread plus equal-weight market and turnover.

    For each holding month t, stocks are sorted by ``signal.loc[t]`` into quintiles. The
    bottom ``q`` fraction (last month's *losers*) is the long leg; the top ``q`` fraction
    (last month's *winners*) is the short leg. The forward return is month t's simple
    return -- the alignment of ``signal`` already makes ``signal.loc[t]`` a pre-period
    quantity, so the realised return is contemporaneous with the holding month (no extra
    shift; the one lag lives in ``trailing_return``).

    Returns a DataFrame indexed by date with columns:
    ``loser, winner, market, spread, loser_turn, n_total, n_loser, n_winner``.
    ``loser_turn`` is the one-way fraction of the loser leg replaced vs the prior month.
    """
    fwd = prices.pct_change()  # month t's simple return (the holding-month return)

    rows: list[dict] = []
    prev_losers: set = set()
    for t in signal.index:
        if t not in fwd.index:
            continue
        s = signal.loc[t].dropna()
        f = fwd.loc[t].dropna()
        common = s.index.intersection(f.index)
        if len(common) < min_stocks:
            prev_losers = set()
            continue
        s = s.loc[common]
        f = f.loc[common]

        lo_thresh = s.quantile(q)
        hi_thresh = s.quantile(1.0 - q)
        loser_mask = s <= lo_thresh
        winner_mask = s >= hi_thresh

        cur_losers = set(s[loser_mask].index)
        if prev_losers:
            retained = prev_losers & cur_losers
            turn = 1.0 - len(retained) / len(cur_losers) if cur_losers else 1.0
        else:
            turn = 1.0
        prev_losers = cur_losers

        rows.append(
            {
                "date": t,
                "loser": float(f[loser_mask].mean()),
                "winner": float(f[winner_mask].mean()),
                "market": float(f.mean()),
                "spread": float(f[loser_mask].mean() - f[winner_mask].mean()),
                "loser_turn": float(turn),
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
    n_draws: int = 200,
    seed: int = 329,
    min_stocks: int = 20,
) -> pd.Series:
    """Null distribution of random-quintile excess returns (loser-quintile size).

    For each month, draw ``n_draws`` random subsets of the same number of stocks as the
    loser portfolio and record their excess over the equal-weight market. The full
    (month x draw) distribution is the null for the loser excess -- centred at zero by
    construction, so any loser excess beyond it is signal, not concentration.
    """
    rng = np.random.default_rng(seed)
    fwd = prices.pct_change()
    excesses: list[float] = []
    for t in signal.index:
        if t not in fwd.index:
            continue
        s = signal.loc[t].dropna()
        f = fwd.loc[t].dropna()
        common = s.index.intersection(f.index)
        if len(common) < min_stocks:
            continue
        f = f.loc[common]
        n_loser = max(1, int(round(len(common) * q)))
        mkt = f.to_numpy()
        mkt_mean = mkt.mean()
        idx = np.arange(len(mkt))
        for _ in range(n_draws):
            pick = rng.choice(idx, size=n_loser, replace=False)
            excesses.append(float(mkt[pick].mean() - mkt_mean))
    return pd.Series(excesses, name="random_excess")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summarize(monthly_ret: pd.Series) -> dict:
    """Headline statistics for a monthly return series, with a Newey-West HAC t-stat.

    Returns mean, vol, naive Sharpe, HAC t-stat, hit-rate, max drawdown and n.
    """
    r = pd.Series(monthly_ret).astype(float).dropna()
    n = int(r.size)
    if n < 3:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n")}

    mu = float(r.mean())
    std = float(r.std(ddof=1))
    sr = mu / std if std > 0 else float("nan")

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
        "mean": mu, "vol": std, "sharpe": sr, "tstat": tstat,
        "hit_rate": float((r > 0).mean()), "max_drawdown": max_dd, "n": n,
    }


def annualise_monthly(stats: dict, periods: int = 12) -> dict:
    """Convert monthly stats to annualised equivalents."""
    out = dict(stats)
    if np.isfinite(out.get("mean", np.nan)):
        out["mean_ann"] = float(out["mean"] * periods)
    if np.isfinite(out.get("vol", np.nan)):
        out["vol_ann"] = float(out["vol"] * np.sqrt(periods))
    if "mean_ann" in out and out.get("vol_ann", 0) > 0:
        out["sharpe_ann"] = float(out["mean_ann"] / out["vol_ann"])
    return out


def beta_alpha(leg: pd.Series, market: pd.Series, periods: int = 12) -> tuple[float, float]:
    """Market beta and annualised alpha of ``leg`` regressed on ``market`` (both monthly)."""
    common = leg.dropna().index.intersection(market.dropna().index)
    x = market.loc[common].to_numpy()
    y = leg.loc[common].to_numpy()
    if len(x) < 3 or np.var(x) == 0:
        return float("nan"), float("nan")
    beta = float(np.cov(y, x)[0, 1] / np.var(x))
    alpha = float((y.mean() - beta * x.mean()) * periods)
    return beta, alpha


def net_spread(res: pd.DataFrame, one_way_bps: float) -> pd.Series:
    """Spread net of transaction costs at ``one_way_bps`` per side.

    The dollar-neutral book rebalances *both* legs monthly; we charge the loser-leg
    turnover as a proxy for both sides. Cost is counted one-way x NAV across the two legs,
    round-trip -- i.e. ``drag = 2 legs x turnover x 2 (round trip) x one_way_bps``. The
    monthly drag is deducted from the gross spread. Costs are net (labelled net).
    """
    turn = res["loser_turn"].reindex(res.index).fillna(res["loser_turn"].mean())
    drag = 2.0 * turn * 2.0 * one_way_bps * 1e-4
    return (res["spread"] - drag).rename("net_spread")


def break_even_cost(res: pd.DataFrame) -> float:
    """One-way cost (bps) at which the mean net spread hits zero."""
    to = float(res["loser_turn"].mean())
    mu = float(res["spread"].mean())
    if to <= 0:
        return float("nan")
    return mu / (2.0 * to * 2.0) * 1e4
