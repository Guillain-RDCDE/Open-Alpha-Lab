"""Strategy + inference for Study 718 — Forbes-Billionaire-Drift (event-study CARs).

The claim: when a founder is *newly minted* onto the Forbes billionaires list every
spring, **buy the vehicle** — the public company behind the fresh new name still has room
to run. We pin it down with a textbook short-window **event study** around the list's
publication date:

    For each newly-minted-billionaire vehicle, fit a **market model** (stock = α + β·SPY)
    on a clean PRE-list estimation window [−est−gap, −gap], then compute the **abnormal
    return** AR_t = r_stock,t − (α + β·r_SPY,t) on an event window and cumulate it into a
    **CAR** (cumulative abnormal return).

The heart of the teardown is a **reverse-causality** trap. A founder enters the list
*because* their stock already multiplied, so:

  * **PRE-list** window [−63,−1] — a huge, positive abnormal return **by construction**
    (that run-up is *why* they made the list). It is pure selection / look-ahead: you could
    only have "traded" it by knowing the future list. It is the illusion the story trades on.
  * **POST-list** window [+1,+63] — the only **holdable** question: is there abnormal
    return left over *after* the list is public, when you could actually buy the vehicle?
  * **ANNOUNCE** window [0,+2] — the repricing on the list days themselves.

We then judge the POST leg (and the ANNOUNCE leg) with a 1-day execution lag for any
tradable variant, a Welch t vs 0, a placebo / randomization null (random non-event
windows on the same names), a win-rate vs the 50/50 base rate, and one-way costs.

The decisive number is the **post-list** mean CAR and its t: a small, sign-unstable,
insignificant drift is what "buy the fresh billionaire's vehicle" actually delivers once
you strip out the run-up that put them on the list in the first place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Event-study geometry (trading days, relative to the list publication = day 0).
EST_DAYS = 60          # length of the market-model estimation window (young, high-vol names)
EST_GAP = 5            # gap between estimation window and event window (no leakage)

PRE_WINDOW = (-63, -1)      # the mechanical pre-list run-up (selection / look-ahead)
ANNOUNCE_WINDOW = (0, 2)    # the list-day repricing
POST_WINDOW = (1, 63)       # the tradable post-list drift — the headline question
DEFAULT_WINDOW = POST_WINDOW


# --------------------------------------------------------------------------- #
# Market-model abnormal returns for a single event
# --------------------------------------------------------------------------- #
def _simple_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change()


def event_car(prices: pd.DataFrame, ticker: str, list_date: pd.Timestamp,
              window: tuple[int, int] = DEFAULT_WINDOW,
              est_days: int = EST_DAYS, est_gap: int = EST_GAP,
              lag: int = 0, bench: str = "SPY") -> float:
    """Market-model cumulative abnormal return (CAR) for one list event.

    Fits ``r_stock = a + b * r_bench`` on the estimation window [−est_gap−est_days,
    −est_gap] relative to the list date, then cumulates AR over the event ``window``
    (inclusive, in trading days). ``bench`` is the benchmark column (``SPY`` = broad
    market, ``QQQ`` = tech, used to test how much of the "abnormal" return is just the
    growth beta SPY doesn't hedge). ``lag`` shifts the event window forward by ``lag`` days
    to model a one-day execution delay (no look-ahead). Returns NaN if the windows do not
    fit the available history.
    """
    if ticker not in prices.columns or bench not in prices.columns:
        return float("nan")
    rs = _simple_returns(prices[ticker]).dropna()
    rm = _simple_returns(prices[bench]).dropna()
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
    if est_lo < 1 or lo_ev < 0 or hi_ev >= len(idx):
        return float("nan")
    mkt_e = rm.values[est_lo:est_hi]
    stk_e = rs.values[est_lo:est_hi]
    if len(mkt_e) < 40:
        return float("nan")
    b, a = np.polyfit(mkt_e, stk_e, 1)
    abn = rs.values[lo_ev:hi_ev + 1] - (a + b * rm.values[lo_ev:hi_ev + 1])
    return float(np.nansum(abn))


def car_panel(prices: pd.DataFrame, events: list[dict],
              window: tuple[int, int] = DEFAULT_WINDOW, lag: int = 0,
              bench: str = "SPY") -> pd.DataFrame:
    """CAR for every event in the table; returns a tidy frame."""
    rows = []
    for e in events:
        car = event_car(prices, e["ticker"], e["list_date"], window=window, lag=lag,
                        bench=bench)
        if not np.isfinite(car):
            continue
        rows.append({"ticker": e["ticker"], "founder": e["founder"],
                     "list_year": e["list_year"], "list_date": e["list_date"],
                     "car": car})
    return pd.DataFrame(rows)


def raw_excess_car(prices: pd.DataFrame, ticker: str, list_date: pd.Timestamp,
                   window: tuple[int, int] = DEFAULT_WINDOW, bench: str = "SPY") -> float:
    """Plain **excess-over-benchmark** CAR (implicit beta = 1) — no market-model fit.

    The beta-artifact-free cross-check: just sum ``r_stock − r_bench`` over the window. On
    freshly-IPO'd, ultra-high-beta names a short-window market-model beta is unstable, so a
    fitted "abnormal return" can diverge sharply from the plain excess return. If the two
    disagree, the "abnormal drift" is a beta artifact, not alpha."""
    if ticker not in prices.columns or bench not in prices.columns:
        return float("nan")
    rs = _simple_returns(prices[ticker]).dropna()
    rm = _simple_returns(prices[bench]).dropna()
    common = rs.index.intersection(rm.index)
    rs, rm = rs.loc[common], rm.loc[common]
    pos = int(np.searchsorted(common, pd.Timestamp(list_date)))
    lo, hi = pos + window[0], pos + window[1]
    if lo < 0 or hi >= len(common):
        return float("nan")
    return float(np.nansum(rs.values[lo:hi + 1] - rm.values[lo:hi + 1]))


def raw_excess_panel(prices: pd.DataFrame, events: list[dict],
                     window: tuple[int, int] = DEFAULT_WINDOW,
                     bench: str = "SPY") -> np.ndarray:
    """Plain excess-over-benchmark CAR for every event (beta = 1 cross-check)."""
    vals = [raw_excess_car(prices, e["ticker"], e["list_date"], window=window, bench=bench)
            for e in events]
    arr = np.array(vals, dtype=float)
    return arr[np.isfinite(arr)]


# --------------------------------------------------------------------------- #
# Placebo null — random non-event windows on the same names
# --------------------------------------------------------------------------- #
def placebo_car_dist(prices: pd.DataFrame, tickers: list[str], k: int,
                     window: tuple[int, int] = DEFAULT_WINDOW,
                     n_draws: int = 20_000, seed: int = 718) -> np.ndarray:
    """Sampling distribution of a k-event mean CAR under the null (no event effect).

    Repeatedly draws ``k`` random (ticker, date) pairs from the same universe, computes
    each one's market-model CAR over the same window, and records the mean. The honest
    small-sample yardstick: "could two dozen random dates on these same high-flying names
    have produced this CAR?"
    """
    rng = np.random.default_rng(seed)
    avail = [t for t in tickers if t in prices.columns]
    rm = _simple_returns(prices["SPY"]).dropna()
    rets = {}
    for t in avail:
        rs = _simple_returns(prices[t]).dropna()
        common = rs.index.intersection(rm.index)
        rets[t] = (rs.loc[common].values, rm.loc[common].values)
    span_lo = EST_DAYS + EST_GAP + abs(window[0]) + 1
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
            if est_lo < 1 or len(mkt[est_lo:est_hi]) < 40:
                continue
            b, a = np.polyfit(mkt[est_lo:est_hi], stk[est_lo:est_hi], 1)
            lo_ev, hi_ev = pos + window[0], pos + window[1]
            abn = stk[lo_ev:hi_ev + 1] - (a + b * mkt[lo_ev:hi_ev + 1])
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


def summarize(prices: pd.DataFrame, events: list[dict], tickers: list[str] | None = None,
              n_draws: int = 12_000) -> dict:
    """Headline event-study stats across the three windows (pre / announce / post).

    The PRE window exposes the selection illusion (huge & positive by construction), the
    POST window is the tradable question (placebo-tested), the ANNOUNCE window is the
    list-day repricing.
    """
    out = {}
    for name, win in [("pre", PRE_WINDOW), ("announce", ANNOUNCE_WINDOW),
                      ("post", POST_WINDOW)]:
        panel = car_panel(prices, events, window=win)
        car = panel["car"].to_numpy(float)
        out[name] = summarize_bucket(car)
    if tickers is not None:
        post_panel = car_panel(prices, events, window=POST_WINDOW)
        pc = post_panel["car"].to_numpy(float)
        pc = pc[np.isfinite(pc)]
        if len(pc):
            null = placebo_car_dist(prices, tickers, k=len(pc), window=POST_WINDOW,
                                    n_draws=n_draws)
            out["post_placebo_p"] = placebo_pvalue(float(pc.mean()), null)
            out["post_null_mean_pct"] = float(np.nanmean(null) * 100)
    return out


def net_of_costs(car_mean: float, cost_bps: float = 20.0) -> dict:
    """One-way round-trip cost on a 'buy the vehicle, hold the window' trade.

    A single event-window trade is one entry + one exit; 20 bps round-trip is generous for
    a liquid large-cap but conservative for the freshly-IPO'd, wider-spread names here.
    Returns gross/net mean CAR. (Cost is not the binding constraint — the absence of drift
    is — but we charge it honestly all the same.)"""
    c = cost_bps / 1e4
    return {"gross_pct": float(car_mean * 100), "net_pct": float((car_mean - c) * 100),
            "cost_bps": cost_bps}
