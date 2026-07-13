"""Strategy + inference for Study 717 — Person-of-the-Year (cover-curse event study).

The claim: TIME putting a CEO on its Person-of-the-Year cover marks a *peak* — the stock
drifts down afterward (the "magazine cover curse"). We pin it with a long-horizon
**event study**:

    For each mid-December coronation, fit a **market model** (stock = α + β·SPY) on a clean
    PRE-event estimation window [−est−gap, −gap], then compute the **abnormal return**
    AR_t = r_stock,t − (α + β·r_SPY,t) over a LONG post-announcement window and cumulate it
    into a **CAR** — the drift the curse predicts to be negative.

Because you only learn the pick when the cover drops, the window starts at **+1 trading
day** (buy/short at the next close, no look-ahead). We then ask:

  * Is the pooled honoree CAR reliably **negative** over 1 / 3 / 6 / 12 months? (t vs 0)
  * Could four random mid-December windows on the same names look this "cursed"? (placebo)
  * Is the apparent curse just **selection** — were the honorees crowned after a huge
    *prior* run-up, so the post-drift is momentum mean-reversion, not a magazine effect?
    (``prior_runup`` vs post CAR)
  * A **short** trade: short the honoree at +1 day, hold the horizon, pay **borrow** —
    gross AND net, because these are hard-to-borrow meme names.

The decisive number is not the sign of the point estimate (with AMZN'99 and TSLA'21 in the
set it *is* negative) but the **sample size and the confound**: four events, dominated by
two bubble-era icons, cannot distinguish a cover curse from zenith mean-reversion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Event-study geometry (trading days, relative to the announcement = day 0).
EST_DAYS = 120         # length of the market-model estimation window
EST_GAP = 5            # gap between estimation window and event window (no leakage)
DEFAULT_WINDOW = (1, 252)  # CAR window [+1, +252] ≈ the 12-month post-coronation drift


# --------------------------------------------------------------------------- #
# Market-model abnormal returns for a single event
# --------------------------------------------------------------------------- #
def _simple_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change()


def event_car(prices: pd.DataFrame, ticker: str, announce_date: pd.Timestamp,
              window: tuple[int, int] = DEFAULT_WINDOW,
              est_days: int = EST_DAYS, est_gap: int = EST_GAP) -> float:
    """Market-model cumulative abnormal return (CAR) for one coronation.

    Fits ``r_stock = a + b * r_SPY`` on the estimation window [−est_gap−est_days, −est_gap]
    relative to the announcement, then cumulates AR over the event ``window`` (inclusive, in
    trading days). The window starts at +1 by default (you only learn the pick at the cover
    reveal, so you can act no earlier than the next close). Returns NaN if the windows do
    not fit the available history.
    """
    if ticker not in prices.columns or "SPY" not in prices.columns:
        return float("nan")
    rs = _simple_returns(prices[ticker]).dropna()
    rm = _simple_returns(prices["SPY"]).dropna()
    common = rs.index.intersection(rm.index)
    rs, rm = rs.loc[common], rm.loc[common]
    idx = common
    pos = int(np.searchsorted(idx, pd.Timestamp(announce_date)))
    if pos >= len(idx):
        return float("nan")
    lo_ev = pos + window[0]
    hi_ev = pos + window[1]
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


def prior_runup(prices: pd.DataFrame, ticker: str, announce_date: pd.Timestamp,
                lookback: int = 252) -> float:
    """Raw stock return over the ``lookback`` trading days BEFORE the coronation.

    The selection confound: TIME crowns people at their zenith. A large positive prior
    run-up next to a negative post CAR is the fingerprint of momentum mean-reversion, not a
    magazine effect. Returns a simple total-return over [−lookback, 0].
    """
    if ticker not in prices.columns:
        return float("nan")
    s = prices[ticker].dropna()
    pos = int(np.searchsorted(s.index, pd.Timestamp(announce_date)))
    if pos - lookback < 0 or pos >= len(s):
        return float("nan")
    p0 = s.values[pos - lookback]
    p1 = s.values[pos]
    if not (p0 > 0):
        return float("nan")
    return float(p1 / p0 - 1.0)


def car_panel(prices: pd.DataFrame, events: list[dict],
              window: tuple[int, int] = DEFAULT_WINDOW,
              with_runup: bool = False) -> pd.DataFrame:
    """CAR for every honoree; tidy frame with ``direct`` flag (and optional prior run-up)."""
    rows = []
    for e in events:
        car = event_car(prices, e["ticker"], e["announce_date"], window=window)
        if not np.isfinite(car):
            continue
        row = {"ticker": e["ticker"], "announce_date": e["announce_date"],
               "honoree": e.get("honoree", e["ticker"]),
               "direct": bool(e["direct"]), "car": car}
        if with_runup:
            row["runup"] = prior_runup(prices, e["ticker"], e["announce_date"])
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Placebo null — random non-event mid-December windows on the same names
# --------------------------------------------------------------------------- #
def placebo_car_dist(prices: pd.DataFrame, tickers: list[str], k: int,
                     window: tuple[int, int] = DEFAULT_WINDOW,
                     n_draws: int = 20_000, seed: int = 717) -> np.ndarray:
    """Sampling distribution of a k-event mean CAR under the null (no cover effect).

    Repeatedly draws ``k`` random (ticker, date) pairs from the same universe, computes each
    one's market-model CAR over the same long window, and records the mean. The honest
    small-sample yardstick: "could four random dates on these names have produced this CAR?"
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


def placebo_pvalue(obs_mean: float, null_dist: np.ndarray, one_sided: str = "left") -> float:
    """Empirical p-value of ``obs_mean`` against a placebo ``null_dist``.

    The curse is directional (a *decline*), so the natural test is one-sided-left:
    ``p = Pr[random mean <= observed mean]``. Pass ``one_sided='two'`` for the two-sided
    magnitude test.
    """
    null_dist = np.asarray(null_dist, dtype=float)
    null_dist = null_dist[np.isfinite(null_dist)]
    if null_dist.size == 0:
        return float("nan")
    if one_sided == "two":
        center = null_dist.mean()
        return float((np.abs(null_dist - center) >= abs(obs_mean - center)).mean())
    if one_sided == "right":
        return float((null_dist >= obs_mean).mean())
    return float((null_dist <= obs_mean).mean())


def summarize_bucket(car: np.ndarray) -> dict:
    """n, mean CAR (%), curse-rate (P[CAR<0]), and the t of mean vs 0."""
    car = np.asarray(car, dtype=float)
    car = car[np.isfinite(car)]
    if len(car) == 0:
        return {"n": 0, "mean_pct": float("nan"), "curse": float("nan"), "t": float("nan")}
    return {
        "n": int(len(car)),
        "mean_pct": float(car.mean() * 100),
        "curse": float((car < 0).mean()),
        "t": welch_t(car),
    }


def summarize(panel: pd.DataFrame, prices: pd.DataFrame | None = None,
              tickers: list[str] | None = None,
              window: tuple[int, int] = DEFAULT_WINDOW,
              n_draws: int = 20_000) -> dict:
    """Headline event-study stats: pooled + direct/linked buckets, and a placebo p-value
    for the pooled honoree mean if prices are given (one-sided-left = 'cursed')."""
    allc = panel["car"].to_numpy(float)
    direct = panel.loc[panel["direct"], "car"].to_numpy(float)
    linked = panel.loc[~panel["direct"], "car"].to_numpy(float)
    out = {
        "all": summarize_bucket(allc),
        "direct": summarize_bucket(direct),
        "linked": summarize_bucket(linked),
    }
    if prices is not None and tickers is not None and len(allc):
        null = placebo_car_dist(prices, tickers, k=len(allc), window=window,
                                n_draws=n_draws)
        out["placebo_p_left"] = placebo_pvalue(float(np.nanmean(allc)), null, "left")
        out["placebo_p_two"] = placebo_pvalue(float(np.nanmean(allc)), null, "two")
        out["null_mean_pct"] = float(np.nanmean(null) * 100)
    return out


def net_of_costs(car_mean: float, horizon_days: int = 252,
                 borrow_ann: float = 0.05, trade_bps: float = 10.0) -> dict:
    """Borrow-aware economics of SHORTING the honoree for the horizon.

    The curse trade is a short: you profit from a *negative* honoree CAR, so the gross P&L
    is ``-car_mean``. You pay (a) one round-trip of ``trade_bps`` and (b) stock-borrow at
    ``borrow_ann`` per year, prorated over the horizon. Borrow on AMZN'99 / TSLA'21 /
    DJT-class names ran far above 5%/yr at times, so this is a *generous* floor. Returns the
    gross and net short P&L (as a %; positive = the short made money)."""
    gross = -car_mean
    cost = trade_bps / 1e4 + borrow_ann * (horizon_days / 252.0)
    return {"gross_pct": float(gross * 100), "net_pct": float((gross - cost) * 100),
            "borrow_ann_pct": borrow_ann * 100, "trade_bps": trade_bps}
