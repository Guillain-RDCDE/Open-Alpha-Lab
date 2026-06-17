"""Strategy and honest controls -- Study 252 (Search-Trends).

The claim under test: *spikes in Google searches for a stock's name front-run
its return.*  The naive bullish reading is "attention = informed buying =
higher future returns".  Da, Engelberg & Gao (2011) found the opposite at
horizon: a search spike is *uninformed retail price pressure* that pushes the
price up for a couple of weeks and then **reverses** over the following months.

So there are two competing hypotheses, and the sign of the signal decides which:

1. **Front-running (momentum)** -- high attention this month predicts a *high*
   return next month.  (The folk-wisdom "buy what's trending" story.)
2. **Reversal (price pressure)** -- high attention this month predicts a *low*
   return next month.  (The Da-Engelberg-Gao story.)

We build an *attention z-score* (this month's search index vs each stock's own
trailing baseline), sort the basket each month into a high-attention top group
and a low-attention bottom group, and measure the next-month return spread.
The **spread we report is LOW-attention minus HIGH-attention** -- i.e. it is
positive when the *reversal* hypothesis holds (fade the hype).  A momentum
effect would show up as a *negative* spread.

Three honest comparisons settle it:

1. **Low vs High attention** -- the long-low/short-high spread (the headline).
2. **Low/High vs equal-weight basket** -- the practical long-only benchmark.
3. **Random-portfolio control** -- a randomly-drawn group of the same size, to
   test whether the attention sort carries information beyond luck.

**No look-ahead**: the z-score for month-end t uses search data through t only
(trailing window strictly before-and-including t); it predicts the return earned
*in month t+1*.  One-month execution lag.

**Survivorship + proxy bias**: the basket is fixed survivors and the attention
series is a hand-curated proxy.  Real-tape results are upper bounds; named on
the Signal axis.

**Costs**: monthly rebalancing of a concentrated long/short book; the short leg
(hyped meme names) is exactly the hard-to-borrow universe.  One-way cost x NAV
is applied in the cost sweep; shorts pay borrow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def attention_zscore(
    attn: pd.DataFrame,
    window: int = 6,
    min_periods: int = 3,
) -> pd.DataFrame:
    """Standardised attention shock: this month's search index vs its own trailing mean.

    For each ticker, the z-score at month-end t is::

        (attn[t] - rolling_mean(window)[t]) / rolling_std(window)[t]

    using a trailing window that *includes* t (search data is observable at the
    close of month t).  High z = an abnormal spike in search interest this month.

    This standardisation removes the per-ticker base level (a ticker that is
    always heavily searched is not "spiking"); only the *shock* relative to the
    stock's own recent baseline matters -- exactly the Da-Engelberg-Gao
    abnormal-SVI construction.

    Parameters
    ----------
    attn :
        Month-end attention panel (date x ticker), e.g. the 0-100 Trends index.
    window :
        Trailing window length (months) for the baseline mean/std.
    min_periods :
        Minimum observations before a z-score is emitted (burn-in otherwise NaN).

    Returns
    -------
    DataFrame of attention z-scores (same shape as ``attn``).  NaN during the
    burn-in window.
    """
    roll = attn.rolling(window=window, min_periods=min_periods)
    mu = roll.mean()
    sd = roll.std(ddof=0)
    z = (attn - mu) / sd.replace(0.0, np.nan)
    z.index = pd.DatetimeIndex(z.index)
    z.index.name = "date"
    return z


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------
def attention_returns(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    q: float = 0.30,
    min_stocks: int = 6,
) -> pd.DataFrame:
    """Monthly low-attention-minus-high-attention spread plus basket return.

    For each month-end t, stocks are sorted by ``signal.loc[t]`` (attention
    z-score).  The bottom ``q`` fraction (lowest attention) is the LONG leg; the
    top ``q`` fraction (highest attention) is the SHORT leg.  The forward return
    is the simple return from month t to t+1.

    The reported ``spread`` is **low minus high** -- positive when fading the
    hype (the reversal hypothesis) pays.

    Returns a DataFrame with columns:
    - ``high``    : equal-weight mean next-month return of the high-attention group
    - ``low``     : equal-weight mean next-month return of the low-attention group
    - ``market``  : equal-weight mean next-month return of all ranked stocks
    - ``spread``  : low - high  (positive => reversal/fade-the-hype pays)
    - ``n_total`` : number of stocks with valid signal and forward return
    - ``n_high``  : stocks in the high-attention group
    - ``n_low``   : stocks in the low-attention group
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
        high_mask = s >= hi_thresh
        low_mask = s <= lo_thresh

        high_ret = float(f[high_mask].mean())
        low_ret = float(f[low_mask].mean())
        market_ret = float(f.mean())

        rows.append(
            {
                "date": t,
                "high": high_ret,
                "low": low_ret,
                "market": market_ret,
                "spread": low_ret - high_ret,  # low - high: positive => reversal
                "n_total": int(len(common)),
                "n_high": int(high_mask.sum()),
                "n_low": int(low_mask.sum()),
            }
        )
    result = pd.DataFrame(rows).set_index("date")
    result.index = pd.DatetimeIndex(result.index)
    return result


def random_portfolio_returns(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    q: float = 0.30,
    n_draws: int = 200,
    seed: int = 252,
    min_stocks: int = 6,
) -> pd.Series:
    """Null distribution of random-group excess returns (low-attention group size).

    For each month, draw ``n_draws`` random subsets of the same number of stocks
    as the low-attention group.  Returns a Series of (random excess vs market)
    across all (month x draw) combinations.  Should centre at zero.
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
        n_low = max(1, int(round(len(common) * q)))
        mkt = f.to_numpy()
        mkt_mean = mkt.mean()
        all_idx = np.arange(len(mkt))
        for _ in range(n_draws):
            pick = rng.choice(all_idx, size=n_low, replace=False)
            excesses.append(float(mkt[pick].mean() - mkt_mean))

    return pd.Series(excesses, name="random_excess")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summarize(series: pd.Series) -> dict:
    """Headline statistics for a monthly return/spread series (Newey-West HAC t-stat)."""
    r = pd.Series(series).astype(float).dropna()
    n = int(r.size)
    if n < 3:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "n")} | {"n": n}

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
    q: float = 0.30,
    one_way_bps: float = 15.0,
    borrow_bps: float = 8.0,
    min_stocks: int = 6,
) -> pd.Series:
    """Monthly cost drag from rebalancing the long/short attention book.

    Each month both the low-attention (long) and high-attention (short) groups
    change.  Drag is::

        turnover_long  * 2 * one_way_bps          (round-trip on the long leg)
      + turnover_short * 2 * one_way_bps          (round-trip on the short leg)
      + borrow_bps                                (monthly borrow on the short leg)

    Costs are charged on NAV (one-way x NAV).  ``borrow_bps`` is the monthly
    borrow fee on the (typically hard-to-borrow, meme) short leg.

    Returns a Series of monthly cost drag (fraction; 0.001 = 10 bps).
    """
    rows: list[dict] = []
    prev_low: set = set()
    prev_high: set = set()

    for t in signal.index:
        s = signal.loc[t].dropna()
        f = prices.loc[t].dropna() if t in prices.index else pd.Series(dtype=float)
        common = s.index.intersection(f.index)
        if len(common) < min_stocks:
            prev_low = set()
            prev_high = set()
            continue
        s = s.loc[common]
        hi_thresh = s.quantile(1.0 - q)
        lo_thresh = s.quantile(q)
        curr_low = set(s[s <= lo_thresh].index.tolist())
        curr_high = set(s[s >= hi_thresh].index.tolist())

        if prev_low:
            to_low = 1.0 - len(prev_low & curr_low) / len(curr_low) if curr_low else 1.0
        else:
            to_low = 1.0
        if prev_high:
            to_high = 1.0 - len(prev_high & curr_high) / len(curr_high) if curr_high else 1.0
        else:
            to_high = 1.0

        drag = (to_low + to_high) * 2.0 * one_way_bps * 1e-4 + borrow_bps * 1e-4
        rows.append({"date": t, "turnover": 0.5 * (to_low + to_high), "drag": drag})
        prev_low = curr_low
        prev_high = curr_high

    if not rows:
        return pd.Series(dtype=float, name="drag")
    df = pd.DataFrame(rows).set_index("date")
    return df["drag"].rename("drag")
