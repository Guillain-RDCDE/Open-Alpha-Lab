"""Strategy and honest controls -- Study 254 (WSB-Mentions).

The claim under test: *the r/WallStreetBets mention count calls the meme move.*
Operationally -- at each month-end, rank the meme basket by this month's WSB
mention count; long the most-mentioned half (the "hype" leg), short the
least-mentioned half (the "quiet" leg); hold one month and rebalance.  This is
a cross-sectional, within-basket sort, NOT a market-timing rule.

Three honest comparisons settle the matter:

1. **Hype vs Quiet** -- the long-short spread (top-half minus bottom-half next-
   month return), the literal "buy what Reddit is talking about" portfolio.
2. **Hype vs equal-weight basket** -- the long-only top-half excess return, the
   more practical benchmark (you can't easily short illiquid meme names).
3. **Random-portfolio control** -- a randomly-drawn half of the same size, to
   test whether the mention rank carries information beyond luck.

We also run the **reversal** sign (short the loud names, buy the quiet ones),
because the leading folk-finance prior is *"buy the rumor, sell the news"* --
mentions peak at the top, so high-buzz names should *under*-perform next month.

**Endogeneity (named on the Signal axis)**: mentions are *contemporaneous* with
the price move -- Reddit gets loud *because* the stock already ran.  A
contemporaneous proxy used as a forward predictor is mechanically suspect; the
only honest question is whether next-month returns are forecastable at all.

**Survivorship / delisting**: the basket keeps collapsed names (BBBY).  Where a
name has no price, it drops from that month's cross-section -- so the realized
short leg quietly excludes the worst blow-ups, flattering any "short the hype"
result.  This is an *upper bound* on the reversal edge.

**Execution lag**: the signal at month-end t uses only month-t mentions and
month-t price; the return earned is month t -> t+1 (one-month lag).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def mention_signal(mentions: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    """Cross-sectional WSB-mention score per month (higher = louder).

    For each month-end row, the score is the (log) mention count -- the strategy
    ranks names *within* the month, so the raw level is enough.  Using ``log``
    compresses the GME/AMC spikes so the rank is not a one-stock dummy.  No
    look-ahead: the score at month t uses only month-t mentions, and is mapped
    to the month t -> t+1 forward return downstream.

    Returns a DataFrame the same shape as ``mentions``; NaN where the mention
    cell is NaN (never observed).  Zeros stay as low scores (quiet names).
    """
    m = mentions.astype(float)
    if log:
        return np.log1p(m)
    return m


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------
def half_returns(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    q: float = 0.50,
    min_stocks: int = 6,
) -> pd.DataFrame:
    """Monthly loud-half-minus-quiet-half spread plus equal-weight basket return.

    For each month t with at least ``min_stocks`` ranked names, stocks are split
    by the mention signal into a loud (top ``q``) and quiet (bottom ``q``) leg.
    The forward return is the simple return from month t to t+1.

    Columns:
    - ``hype``    : equal-weight mean next-month return of the loud half
    - ``quiet``   : equal-weight mean next-month return of the quiet half
    - ``basket``  : equal-weight mean next-month return of all ranked names
    - ``spread``  : hype - quiet  (the "buy the buzz" long-short)
    - ``n_total`` : number of names with valid signal and forward return
    - ``n_hype``  : names in the loud leg
    - ``n_quiet`` : names in the quiet leg
    """
    fwd = prices.pct_change(periods=1).shift(-1)  # month t -> t+1

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

        hi_thresh = s.quantile(1.0 - q)
        lo_thresh = s.quantile(q)
        hype_mask = s >= hi_thresh
        quiet_mask = s <= lo_thresh

        hype_ret = float(f[hype_mask].mean())
        quiet_ret = float(f[quiet_mask].mean())
        basket_ret = float(f.mean())

        rows.append(
            {
                "date": t,
                "hype": hype_ret,
                "quiet": quiet_ret,
                "basket": basket_ret,
                "spread": hype_ret - quiet_ret,
                "n_total": int(len(common)),
                "n_hype": int(hype_mask.sum()),
                "n_quiet": int(quiet_mask.sum()),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["hype", "quiet", "basket", "spread", "n_total", "n_hype", "n_quiet"]
        )
    result = pd.DataFrame(rows).set_index("date")
    result.index = pd.DatetimeIndex(result.index)
    return result


def random_portfolio_returns(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    q: float = 0.50,
    n_draws: int = 300,
    seed: int = 254,
    min_stocks: int = 6,
) -> pd.Series:
    """Null distribution of random-half excess returns (loud-leg size).

    For each month, draw ``n_draws`` random subsets of the same number of stocks
    as the loud leg.  Returns a Series of (random excess vs basket) across all
    (month x draw) combinations -- centred at zero by construction.
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
        f = f.loc[common]
        n_hype = max(1, int(round(len(common) * q)))
        vals = f.to_numpy()
        mean = vals.mean()
        all_idx = np.arange(len(vals))
        for _ in range(n_draws):
            pick = rng.choice(all_idx, size=n_hype, replace=False)
            excesses.append(float(vals[pick].mean() - mean))

    return pd.Series(excesses, name="random_excess")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summarize(series: pd.Series) -> dict:
    """Headline statistics for a monthly return/spread series (HAC t-stat)."""
    r = pd.Series(series).astype(float).dropna()
    n = int(r.size)
    if n < 3:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "n")} | {"n": n}

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
    q: float = 0.50,
    one_way_bps: float = 30.0,
    min_stocks: int = 6,
) -> pd.Series:
    """Monthly cost drag from loud-leg turnover (full rebalance each month).

    Meme names are illiquid and the short leg pays heavy borrow, so the default
    one-way cost is a conservative 30 bps.  Drag = turnover x 2 x one-way cost.
    Returns a Series of monthly cost drag in fraction (0.001 = 10 bps).
    """
    rows: list[dict] = []
    prev_hype: set = set()

    for t in signal.index:
        s = signal.loc[t].dropna()
        f = prices.loc[t].dropna() if t in prices.index else pd.Series(dtype=float)
        common = s.index.intersection(f.index)
        if len(common) < min_stocks:
            prev_hype = set()
            continue
        s = s.loc[common]
        hi_thresh = s.quantile(1.0 - q)
        curr_hype = set(s[s >= hi_thresh].index.tolist())
        if prev_hype:
            retained = prev_hype & curr_hype
            turnover = 1.0 - len(retained) / len(curr_hype) if curr_hype else 1.0
        else:
            turnover = 1.0
        drag = turnover * 2.0 * one_way_bps * 1e-4
        rows.append({"date": t, "turnover": turnover, "drag": drag})
        prev_hype = curr_hype

    if not rows:
        return pd.Series(dtype=float, name="drag")
    df = pd.DataFrame(rows).set_index("date")
    return df["drag"].rename("drag")
