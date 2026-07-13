"""Strategy + inference for Study 751 — Fortune-500-Inclusion (event-study abnormal returns).

The claim: **making the Fortune 500** (a prestige/attention shock) lifts the stock, and
**falling off** sheds it — the index-inclusion trade (Study 249) applied to a magazine
ranking. We pin it down with a textbook short-window **event study**:

    For each dated list-reveal, fit a **market model** (stock = α + β·SPY) on a clean
    PRE-event estimation window [−est−gap, −gap], then compute the **abnormal return**
    AR_t = r_stock,t − (α + β·r_SPY,t) on the event window and cumulate it into a **CAR**
    (cumulative abnormal return).

We then ask, with a 1-day execution lag for any tradable variant:

  * Is the **added** CAR different from zero? the **dropped** CAR? (Welch / t vs 0)
  * Is **added − dropped** different from zero? (the believers' core claim)
  * A **placebo** null: random non-event windows on the same names, to size the small-sample
    sampling distribution of a CAR;
  * a **win-rate** (P[CAR > 0]) vs the 50/50 base rate;
  * **one-way costs** applied to a "buy on the reveal, hold the window" variant.

The decisive number is not the sign of any point estimate but the **sample size** and the
**absent mechanism**: unlike an S&P-500 addition (which forces index funds to buy), a
Fortune-500 debut triggers no mechanical demand and reveals no new information (the ranking is
built from already-public revenue). With ~a dozen events per bucket and no demand shock, the
added-minus-dropped CAR is statistically indistinguishable from noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Event-study geometry (trading days, relative to the list reveal = day 0).
EST_DAYS = 120         # length of the market-model estimation window
EST_GAP = 10           # gap between estimation window and event window (no leakage)
DEFAULT_WINDOW = (0, 2)   # CAR window [0, +2] trading days around the reveal


# --------------------------------------------------------------------------- #
# Market-model abnormal returns for a single event
# --------------------------------------------------------------------------- #
def _simple_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change()


def event_car(prices: pd.DataFrame, ticker: str, list_date: pd.Timestamp,
              window: tuple[int, int] = DEFAULT_WINDOW,
              est_days: int = EST_DAYS, est_gap: int = EST_GAP,
              lag: int = 0) -> float:
    """Market-model cumulative abnormal return (CAR) for one list-reveal event.

    Fits ``r_stock = a + b * r_SPY`` on the estimation window [−est_gap−est_days,
    −est_gap] relative to the reveal, then cumulates AR over the event ``window``
    (inclusive, in trading days). ``lag`` shifts the event window forward by ``lag`` days to
    model a one-day execution delay (no look-ahead). Returns NaN if the windows do not fit
    the available history.
    """
    if ticker not in prices.columns or "SPY" not in prices.columns:
        return float("nan")
    rs = _simple_returns(prices[ticker]).dropna()
    rm = _simple_returns(prices["SPY"]).dropna()
    common = rs.index.intersection(rm.index)
    rs, rm = rs.loc[common], rm.loc[common]
    idx = common
    pos = int(np.searchsorted(idx, pd.Timestamp(list_date)))
    if pos >= len(idx):
        return float("nan")
    lo_ev = pos + window[0] + lag
    hi_ev = pos + window[1] + lag
    est_hi = pos - est_gap
    est_lo = est_hi - est_days
    if est_lo < 1 or hi_ev >= len(idx):
        return float("nan")
    mkt_e = rm.values[est_lo:est_hi]
    stk_e = rs.values[est_lo:est_hi]
    if len(mkt_e) < 30:
        return float("nan")
    b, a = np.polyfit(mkt_e, stk_e, 1)
    abn = rs.values[lo_ev:hi_ev + 1] - (a + b * rm.values[lo_ev:hi_ev + 1])
    return float(np.nansum(abn))


def car_panel(prices: pd.DataFrame, events: list[dict],
              window: tuple[int, int] = DEFAULT_WINDOW, lag: int = 0) -> pd.DataFrame:
    """CAR for every event in the table; returns a tidy frame with the ``added`` flag."""
    rows = []
    for e in events:
        car = event_car(prices, e["ticker"], e["list_date"], window=window, lag=lag)
        if not np.isfinite(car):
            continue
        rows.append({"ticker": e["ticker"], "list_date": e["list_date"],
                     "added": bool(e["added"]), "car": car})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Placebo null — random non-event windows on the same names
# --------------------------------------------------------------------------- #
def placebo_car_dist(prices: pd.DataFrame, tickers: list[str], k: int,
                     window: tuple[int, int] = DEFAULT_WINDOW,
                     n_draws: int = 20_000, seed: int = 751) -> np.ndarray:
    """Sampling distribution of a k-event mean CAR under the null (no event effect).

    Repeatedly draws ``k`` random (ticker, date) pairs from the same universe, computes each
    one's market-model CAR over the same window, and records the mean. This is the honest
    small-sample yardstick: "could a dozen random dates on these names have produced this
    CAR?"
    """
    rng = np.random.default_rng(seed)
    avail = [t for t in tickers if t in prices.columns]
    rm = _simple_returns(prices["SPY"]).dropna()
    rets = {}
    for t in avail:
        rs = _simple_returns(prices[t]).dropna()
        common = rs.index.intersection(rm.index)
        rets[t] = (rs.loc[common].values, rm.loc[common].values)
    span_lo = EST_DAYS + EST_GAP + 1
    means = np.empty(n_draws)
    for i in range(n_draws):
        vals = []
        for _ in range(k):
            t = avail[rng.integers(0, len(avail))]
            stk, mkt = rets[t]
            n = len(stk)
            hi = n - window[1] - 2
            if hi <= span_lo:
                continue
            pos = int(rng.integers(span_lo, hi))
            est_hi = pos - EST_GAP
            est_lo = est_hi - EST_DAYS
            b, a = np.polyfit(mkt[est_lo:est_hi], stk[est_lo:est_hi], 1)
            abn = stk[pos + window[0]:pos + window[1] + 1] - \
                (a + b * mkt[pos + window[0]:pos + window[1] + 1])
            vals.append(float(np.nansum(abn)))
        means[i] = np.mean(vals) if vals else np.nan
    return means[np.isfinite(means)]


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def welch_t(sample: np.ndarray, other: np.ndarray | None = None) -> float:
    """Welch t. If ``other`` is None, test ``mean(sample) != 0``; else two-sample."""
    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]
    if len(sample) < 2:
        return float("nan")
    if other is None:
        m, se = sample.mean(), sample.std(ddof=1) / np.sqrt(len(sample))
        return float(m / se) if se > 0 else float("nan")
    other = np.asarray(other, dtype=float)
    other = other[np.isfinite(other)]
    if len(other) < 2:
        return float("nan")
    m1, m0 = sample.mean(), other.mean()
    se = np.sqrt(sample.var(ddof=1) / len(sample) + other.var(ddof=1) / len(other))
    return float((m1 - m0) / se) if se > 0 else float("nan")


def placebo_pvalue(obs_mean: float, null_dist: np.ndarray, two_sided: bool = True) -> float:
    """Empirical p-value of ``obs_mean`` against a placebo ``null_dist``."""
    null_dist = np.asarray(null_dist, dtype=float)
    null_dist = null_dist[np.isfinite(null_dist)]
    if null_dist.size == 0:
        return float("nan")
    if two_sided:
        center = null_dist.mean()
        return float((np.abs(null_dist - center) >= abs(obs_mean - center)).mean())
    return float((null_dist >= obs_mean).mean())


def summarize_bucket(car: np.ndarray) -> dict:
    """n, mean CAR (%), win-rate (P[CAR>0]), and the t of mean vs 0."""
    car = np.asarray(car, dtype=float)
    car = car[np.isfinite(car)]
    if len(car) == 0:
        return {"n": 0, "mean_pct": float("nan"), "win": float("nan"), "t": float("nan")}
    return {
        "n": int(len(car)),
        "mean_pct": float(car.mean() * 100),
        "win": float((car > 0).mean()),
        "t": welch_t(car),
    }


def summarize(panel: pd.DataFrame, prices: pd.DataFrame | None = None,
              tickers: list[str] | None = None,
              window: tuple[int, int] = DEFAULT_WINDOW,
              n_draws: int = 20_000) -> dict:
    """Headline event-study stats: added / dropped / all buckets, the added−dropped
    difference (Welch t), and a placebo p-value for the added bucket if prices given."""
    added = panel.loc[panel["added"], "car"].to_numpy(float)
    dropped = panel.loc[~panel["added"], "car"].to_numpy(float)
    allc = panel["car"].to_numpy(float)
    out = {
        "added": summarize_bucket(added),
        "dropped": summarize_bucket(dropped),
        "all": summarize_bucket(allc),
        "diff_pct": float(np.nanmean(added) - np.nanmean(dropped)) * 100
        if len(added) and len(dropped) else float("nan"),
        "diff_t": welch_t(added, dropped),
    }
    if prices is not None and tickers is not None and len(added):
        null = placebo_car_dist(prices, tickers, k=len(added), window=window,
                                n_draws=n_draws)
        out["added_placebo_p"] = placebo_pvalue(float(np.nanmean(added)), null)
        out["null_mean_pct"] = float(np.nanmean(null) * 100)
    return out


def net_of_costs(car_mean: float, cost_bps: float = 10.0) -> dict:
    """One-way round-trip cost on a 'buy on the reveal, hold the window' trade.

    A short event-window trade is one entry + one exit; 10 bps round-trip covers a liquid
    large-cap desk. Returns gross/net mean CAR. (The point is the same as the rest of the
    study: cost is not the binding constraint — the event count and the missing demand shock
    are.)"""
    c = cost_bps / 1e4
    return {"gross_pct": float(car_mean * 100), "net_pct": float((car_mean - c) * 100),
            "cost_bps": cost_bps}
