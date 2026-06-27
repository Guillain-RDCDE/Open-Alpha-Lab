"""Strategy and honest controls -- Study 538 (Industry-Relative-Reversal).

The Hameed-Mian (2015) / Da-Liu-Schaumburg (2014) recipe. Decompose last month's return:

    r[i, t-1] = industry_mean[g(i), t-1] + ( r[i, t-1] - industry_mean[g(i), t-1] )
                \\_____ across-industry ____/   \\________ within-industry (idio) _______/

Form two competing one-month reversal signals from month t-1's return:

- **RAW**: sort on ``-r[i,t-1]`` (the Jegadeesh 1990 / Study 329 signal). Buy last
  month's losers, short last month's winners.
- **INDUSTRY-RELATIVE (IRR)**: sort on ``-( r[i,t-1] - industry_mean )``. Buy the names
  that fell most *relative to their own sector*, short those that rose most relative to
  their sector. The across-industry move is differenced out.

The claim under test: the IRR spread is *larger and cleaner* than the RAW spread, because
the reversal lives in the within-industry component and the raw signal dilutes it with the
non-reversing industry component.

**No look-ahead**: the signal aligned to holding month t is built from month ``t-1``'s
return (known at the close of t-1). The return earned is month t's. One shift, applied
once in ``raw_signal`` / ``industry_relative_signal``. A ``skip`` knob exposes a one-month
gap (bid-ask-bounce robustness).

**Survivorship bias**: the basket is the current S&P 500 projected backwards; all real-
tape results are upper bounds, named on the Signal axis.

The honest null here is a **placebo industry-label shuffle**: randomly permute the sector
tags across tickers and rebuild the IRR signal. If the IRR edge is genuinely *industry*
information, scrambling the industry map should destroy the within-industry advantage; if
it survives a shuffled map, the "industry adjustment" was doing nothing real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def _monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change()


def raw_signal(prices: pd.DataFrame, skip: int = 0) -> pd.DataFrame:
    """Raw one-month formation return aligned to the holding month (Study 329 signal).

    ``signal.loc[t]`` is the simple return of month ``t-1-skip``. With ``skip=0`` this is
    last month's return; ``skip=1`` leaves a one-month gap (bid-ask-bounce-safe).
    """
    ret = _monthly_returns(prices)
    sig = ret.shift(1 + skip)
    sig.index.name = "date"
    return sig


def industry_relative_signal(
    prices: pd.DataFrame, sectors: pd.Series, skip: int = 0
) -> pd.DataFrame:
    """Industry-relative one-month formation return aligned to the holding month.

    For each formation month, subtract the (equal-weight) industry mean return from each
    stock's return, then align to holding month t exactly as ``raw_signal`` does. This is
    the within-industry component on which Hameed-Mian (2015) sort.

    ``sectors`` is a ticker -> sector Series. Tickers without a sector are dropped from the
    demeaning (they cannot be made industry-relative).
    """
    ret = _monthly_returns(prices)
    sec = sectors.reindex(ret.columns)
    valid = sec.dropna().index
    ret = ret[valid]
    sec = sec.loc[valid]

    # industry mean per (month, sector), broadcast back to each ticker
    grp = ret.T.groupby(sec)  # rows=tickers grouped by sector, cols=months
    ind_mean = grp.transform("mean").T  # (month x ticker), each ticker's sector mean
    idio = ret - ind_mean

    sig = idio.shift(1 + skip)
    sig.index.name = "date"
    return sig


# ---------------------------------------------------------------------------
# Portfolio construction -- the long-short quintile engine
# ---------------------------------------------------------------------------
def quintile_returns(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    q: float = 0.20,
    min_stocks: int = 15,
) -> pd.DataFrame:
    """Monthly loser-minus-winner spread (dollar-neutral) plus market and turnover.

    For each holding month t, stocks are sorted by ``signal.loc[t]``. The bottom ``q``
    fraction (last month's losers, by whichever signal) is the long leg; the top ``q``
    fraction (winners) is the short leg. The realised return is month t's simple return
    (the alignment of ``signal`` already made it a pre-period quantity -- no extra shift).

    Returns a DataFrame indexed by date with columns:
    ``loser, winner, market, spread, turn, n_total, n_loser, n_winner``.
    ``turn`` is the one-way fraction of the loser leg replaced vs the prior month.
    """
    fwd = _monthly_returns(prices)
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
                "turn": float(turn),
                "n_total": int(len(common)),
                "n_loser": int(loser_mask.sum()),
                "n_winner": int(winner_mask.sum()),
            }
        )
    result = pd.DataFrame(rows).set_index("date")
    result.index = pd.DatetimeIndex(result.index)
    return result


# ---------------------------------------------------------------------------
# Placebo: industry-label shuffle null
# ---------------------------------------------------------------------------
def shuffle_industry_labels(sectors: pd.Series, seed: int) -> pd.Series:
    """Return a permuted copy of the sector map (same tags, scrambled across tickers)."""
    rng = np.random.default_rng(seed)
    vals = sectors.to_numpy().copy()
    rng.shuffle(vals)
    return pd.Series(vals, index=sectors.index, name="sector")


def placebo_irr_tstats(
    prices: pd.DataFrame,
    sectors: pd.Series,
    n_shuffles: int = 200,
    q: float = 0.20,
    skip: int = 0,
    seed: int = 538,
) -> np.ndarray:
    """Distribution of IRR spread HAC t-stats under shuffled industry labels.

    For each of ``n_shuffles`` random permutations of the sector map, rebuild the IRR
    signal and record the loser-minus-winner spread HAC t-stat. This is the null for the
    *industry adjustment*: if the real IRR t sits in the tail of this distribution, the
    adjustment uses genuine industry structure; if it sits in the bulk, the "industry-
    relative" framing is an empty relabelling.
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_shuffles)
    out = np.empty(n_shuffles)
    for i, sd in enumerate(seeds):
        shuf = shuffle_industry_labels(sectors, int(sd))
        sig = industry_relative_signal(prices, shuf, skip=skip)
        res = quintile_returns(sig, prices, q=q)
        out[i] = summarize(res["spread"])["tstat"]
    return out


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


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------
def net_spread(res: pd.DataFrame, one_way_bps: float, borrow_bps_ann: float = 50.0) -> pd.Series:
    """Spread net of transaction costs at ``one_way_bps`` per side, plus short borrow.

    The dollar-neutral book rebalances both legs monthly; we charge the loser-leg turnover
    as a proxy for both sides, one-way x NAV across the two legs, round-trip:
    ``drag = 2 legs x turnover x 2 (round trip) x one_way_bps``. On top we deduct a flat
    monthly **borrow** on the short (winner) leg (``borrow_bps_ann`` annualised / 12).
    """
    turn = res["turn"].reindex(res.index).fillna(res["turn"].mean())
    drag = 2.0 * turn * 2.0 * one_way_bps * 1e-4
    borrow = (borrow_bps_ann * 1e-4) / 12.0  # short leg only, ~1x NAV
    return (res["spread"] - drag - borrow).rename("net_spread")


def break_even_cost(res: pd.DataFrame) -> float:
    """One-way cost (bps) at which the mean net spread (excl. borrow) hits zero."""
    to = float(res["turn"].mean())
    mu = float(res["spread"].mean())
    if to <= 0:
        return float("nan")
    return mu / (2.0 * to * 2.0) * 1e4
