"""Strategy and honest controls -- Study 253 (Wiki-Views).

The recipe: each month t, measure the *attention surge* per stock as the
month-over-month log change in Wikipedia page-views.  Cross-sectionally rank
stocks by that surge.  The Da-Engelberg-Gao (2011) attention prior says high
retail attention precedes short-horizon *reversal*, so the tradable long-short
is **short the top-surge decile, long the bottom-surge decile**, held one month.

We report the spread both ways and let the sign fall out of the data, but the
headline ``spread`` column is *low-minus-high* (long the attention drought,
short the attention surge) so a positive spread = the reversal story works.

Three honest comparisons settle the matter:

1. **Low-minus-High surge** -- the long-short reversal spread (bottom-surge
   decile minus top-surge decile next-month return).
2. **Low-surge vs equal-weight basket** -- the long-only excess return, the
   more practical benchmark.
3. **Random-portfolio control** -- a randomly-drawn decile of the same size,
   to test whether the surge signal carries information beyond luck.

**No look-ahead**: the surge for month t uses page-views observed during month
t (level at t vs t-1); the forward return is earned during month t+1.  One-month
execution lag: positions formed at month-end close t.

**Survivorship bias**: the basket is a fixed mega-cap set projected backwards;
all real-tape results are upper bounds, named on the Signal axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def attention_surge(
    views: pd.DataFrame,
    smooth: int = 1,
) -> pd.DataFrame:
    """Month-over-month log change in page-views -- the attention-surge signal.

    For each month-end t and ticker, the surge is ``log(views_t / views_{t-k})``
    where ``k = smooth`` (default 1).  A *high* surge = attention spiked this
    month; a *low* (negative) surge = attention faded.

    No look-ahead: the surge at row t uses only page-view levels through t.  The
    forward return that this signal predicts (row t in ``decile_returns``) is the
    return earned during month t+1.

    Returns a DataFrame of surges (same shape as ``views``); the first
    ``smooth`` rows are NaN.
    """
    v = views.astype(float)
    surge = np.log(v / v.shift(smooth))
    surge.index = pd.DatetimeIndex(surge.index)
    return surge


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------
def decile_returns(
    surge: pd.DataFrame,
    prices: pd.DataFrame,
    q: float = 0.20,
    min_stocks: int = 8,
) -> pd.DataFrame:
    """Monthly low-surge-minus-high-surge spread plus equal-weight basket return.

    For each month t, stocks are ranked by ``surge.loc[t]``.  The bottom ``q``
    fraction (attention *drought*) is the long leg; the top ``q`` fraction
    (attention *surge*) is the short leg.  The forward return is the simple
    return from month t to t+1.

    With a ~30-name basket, ``q = 0.20`` gives ~6 names per leg (a 0.10 decile
    would be only ~3 names).

    Returns a DataFrame with columns:
    - ``low``     : equal-weight mean fwd return of the bottom-surge (drought) leg
    - ``high``    : equal-weight mean fwd return of the top-surge leg
    - ``market``  : equal-weight mean fwd return of all ranked stocks
    - ``spread``  : low - high  (positive = reversal story works)
    - ``n_total`` : number of stocks with valid signal and forward return
    - ``n_low``   : stocks in the drought leg
    - ``n_high``  : stocks in the surge leg
    """
    fwd = prices.pct_change(periods=1).shift(-1)  # month t -> t+1

    rows: list[dict] = []
    for t in surge.index:
        if t not in fwd.index:
            continue
        s = surge.loc[t].dropna()
        f = fwd.loc[t].dropna()
        common = s.index.intersection(f.index)
        if len(common) < min_stocks:
            continue
        s = s.loc[common]
        f = f.loc[common]

        hi_thresh = s.quantile(1.0 - q)
        lo_thresh = s.quantile(q)
        high_mask = s >= hi_thresh   # attention surge
        low_mask = s <= lo_thresh    # attention drought

        high_ret = float(f[high_mask].mean())
        low_ret = float(f[low_mask].mean())
        market_ret = float(f.mean())

        rows.append(
            {
                "date": t,
                "low": low_ret,
                "high": high_ret,
                "market": market_ret,
                "spread": low_ret - high_ret,
                "n_total": int(len(common)),
                "n_low": int(low_mask.sum()),
                "n_high": int(high_mask.sum()),
            }
        )
    result = pd.DataFrame(rows).set_index("date")
    result.index = pd.DatetimeIndex(result.index)
    return result


def random_portfolio_returns(
    surge: pd.DataFrame,
    prices: pd.DataFrame,
    q: float = 0.20,
    n_draws: int = 300,
    seed: int = 253,
    min_stocks: int = 8,
) -> pd.Series:
    """Null distribution of random-decile excess returns (drought-leg size).

    For each month, draw ``n_draws`` random subsets of the same number of stocks
    as the drought leg.  Returns a Series of (random excess vs equal-weight
    basket) across all (month x draw) combinations.
    """
    rng = np.random.default_rng(seed)
    fwd = prices.pct_change(periods=1).shift(-1)
    excesses: list[float] = []

    for t in surge.index:
        if t not in fwd.index:
            continue
        s = surge.loc[t].dropna()
        f = fwd.loc[t].dropna()
        common = s.index.intersection(f.index)
        if len(common) < min_stocks:
            continue
        f = f.loc[common]
        n_leg = max(1, int(round(len(common) * q)))
        mkt = f.to_numpy()
        mkt_mean = mkt.mean()
        all_idx = np.arange(len(mkt))
        for _ in range(n_draws):
            pick = rng.choice(all_idx, size=n_leg, replace=False)
            excesses.append(float(mkt[pick].mean() - mkt_mean))

    return pd.Series(excesses, name="random_excess")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summarize(series: pd.Series) -> dict:
    """Headline statistics for a monthly return/spread series (Newey-West HAC t)."""
    r = pd.Series(series).astype(float).dropna()
    n = int(r.size)
    if n < 3:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "n")}

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
    surge: pd.DataFrame,
    prices: pd.DataFrame,
    q: float = 0.20,
    one_way_bps: float = 10.0,
    min_stocks: int = 8,
) -> pd.Series:
    """Monthly cost drag from turnover of BOTH legs (long drought + short surge).

    Each month both legs are fully reconstituted; the fraction that turns over
    is the overlap complement.  Cost drag is ``turnover x 2 x one_way`` per leg,
    summed across the two legs.  The short (surge) leg additionally pays borrow,
    added by the caller via ``borrow_bps``.

    Returns a Series of monthly cost drag (fraction; 0.001 = 10 bps).
    """
    rows: list[dict] = []
    prev_low: set = set()
    prev_high: set = set()

    for t in surge.index:
        s = surge.loc[t].dropna()
        p = prices.loc[t].dropna() if t in prices.index else pd.Series(dtype=float)
        common = s.index.intersection(p.index)
        if len(common) < min_stocks:
            prev_low = set()
            prev_high = set()
            continue
        s = s.loc[common]
        hi_thresh = s.quantile(1.0 - q)
        lo_thresh = s.quantile(q)
        curr_high = set(s[s >= hi_thresh].index.tolist())
        curr_low = set(s[s <= lo_thresh].index.tolist())

        def _to(prev, curr):
            if prev and curr:
                return 1.0 - len(prev & curr) / len(curr)
            return 1.0

        to_low = _to(prev_low, curr_low)
        to_high = _to(prev_high, curr_high)
        # round-trip on each leg
        drag = (to_low + to_high) * 2.0 * one_way_bps * 1e-4
        rows.append({"date": t, "turnover": 0.5 * (to_low + to_high), "drag": drag})
        prev_low, prev_high = curr_low, curr_high

    if not rows:
        return pd.Series(dtype=float, name="drag")
    df = pd.DataFrame(rows).set_index("date")
    return df["drag"].rename("drag")
