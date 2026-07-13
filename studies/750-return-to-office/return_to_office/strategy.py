"""Strategy + inference for Study 750 — Return-to-Office (event study on office REITs).

The claim: when a marquee employer orders workers **back to the office**, the demand for
office space firms up — so office landlords (the REITs) should get a **reaction pop** on
the mandate. We pin it down with a textbook short-window **event study on a sector basket**:

    For each dated RTO-mandate announcement, build the **office-REIT basket** (equal-weight
    daily returns of the surviving pure-office landlords), fit a **market model**
    (basket = α + β·SPY) on a clean PRE-event estimation window, then compute the
    **abnormal return** AR_t = r_basket,t − (α + β·r_SPY,t) on the event window and
    cumulate it into a **CAR** (cumulative abnormal return).

We then ask, with a 1-day execution lag for any tradable variant:

  * Is the **all-events** basket CAR different from zero? (Welch / t vs 0)
  * Is the **strict** (full 5-day) bucket bigger than the **hybrid** bucket? — the
    believers' core claim that a *real* mandate should move offices more (Welch t);
  * A **placebo** null: random non-event windows of the *same basket*, to size the
    small-sample sampling distribution of a basket CAR;
  * a **win-rate** (P[CAR > 0]) vs the 50/50 base rate;
  * one-way **costs** applied to a "buy the basket on the mandate, hold the window" trade;
  * a **VNQ** robustness pass — does office react *beyond* what all REITs did that day?

The decisive number is not the sign of any point estimate but the **sample size** and the
**standard error**: with ~two dozen events on a single, secularly-declining sector, an
announcement-window basket CAR of a fraction of a percent is well inside its own error —
an event-study power problem stacked on top of a sector that trades on **rates and secular
WFH**, not on any one company's memo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Event-study geometry (trading days, relative to the announcement = day 0).
EST_DAYS = 120         # length of the market-model estimation window
EST_GAP = 10           # gap between estimation window and event window (no leakage)
DEFAULT_WINDOW = (0, 2)   # CAR window [0, +2] trading days around the announcement


# --------------------------------------------------------------------------- #
# The office-REIT basket
# --------------------------------------------------------------------------- #
def basket_returns(prices: pd.DataFrame, members: list[str]) -> pd.Series:
    """Equal-weight daily simple return of the office-REIT basket.

    Each day averages the pct-change of whichever members have a price that day (so a REIT
    that lists mid-sample simply joins the average when it appears). Returns a single
    return series aligned to the price index.
    """
    cols = [m for m in members if m in prices.columns]
    if not cols:
        return pd.Series(dtype=float)
    rets = prices[cols].pct_change()
    return rets.mean(axis=1, skipna=True).dropna()


# --------------------------------------------------------------------------- #
# Market-model abnormal returns for a single event (basket vs benchmark)
# --------------------------------------------------------------------------- #
def event_car(basket_ret: pd.Series, bench_ret: pd.Series, when: pd.Timestamp,
              window: tuple[int, int] = DEFAULT_WINDOW,
              est_days: int = EST_DAYS, est_gap: int = EST_GAP,
              lag: int = 0) -> float:
    """Market-model cumulative abnormal return (CAR) of the basket for one RTO event.

    Fits ``r_basket = a + b * r_bench`` on the estimation window [−est_gap−est_days,
    −est_gap] relative to the announcement, then cumulates AR over the event ``window``
    (inclusive, in trading days). ``lag`` shifts the event window forward by ``lag`` days
    to model a one-day execution delay (no look-ahead). Returns NaN if the windows do not
    fit the available history.
    """
    common = basket_ret.index.intersection(bench_ret.index)
    rb = basket_ret.loc[common]
    rm = bench_ret.loc[common]
    idx = common
    pos = int(np.searchsorted(idx, pd.Timestamp(when)))
    if pos >= len(idx):
        return float("nan")
    lo_ev = pos + window[0] + lag
    hi_ev = pos + window[1] + lag
    est_hi = pos - est_gap
    est_lo = est_hi - est_days
    if est_lo < 1 or hi_ev >= len(idx) or lo_ev < 0:
        return float("nan")
    mkt_e = rm.values[est_lo:est_hi]
    bsk_e = rb.values[est_lo:est_hi]
    if len(mkt_e) < 30:
        return float("nan")
    b, a = np.polyfit(mkt_e, bsk_e, 1)
    abn = rb.values[lo_ev:hi_ev + 1] - (a + b * rm.values[lo_ev:hi_ev + 1])
    return float(np.nansum(abn))


def car_panel(prices: pd.DataFrame, events: list[dict], members: list[str],
              benchmark: str = "SPY", window: tuple[int, int] = DEFAULT_WINDOW,
              lag: int = 0) -> pd.DataFrame:
    """Basket CAR for every RTO event; returns a tidy frame with the ``strict`` flag."""
    if benchmark not in prices.columns:
        return pd.DataFrame(columns=["date", "employer", "strict", "car"])
    br = basket_returns(prices, members)
    mr = prices[benchmark].pct_change().dropna()
    rows = []
    for e in events:
        car = event_car(br, mr, e["date"], window=window, lag=lag)
        if not np.isfinite(car):
            continue
        rows.append({"date": e["date"], "employer": e["employer"],
                     "strict": bool(e["strict"]), "car": car})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Placebo null — random non-event windows of the same basket
# --------------------------------------------------------------------------- #
def placebo_car_dist(prices: pd.DataFrame, members: list[str], k: int,
                     benchmark: str = "SPY",
                     window: tuple[int, int] = DEFAULT_WINDOW,
                     n_draws: int = 20_000, seed: int = 750) -> np.ndarray:
    """Sampling distribution of a k-event mean basket CAR under the null (no event effect).

    Repeatedly draws ``k`` random dates, computes the basket's market-model CAR over the
    same window at each, and records the mean. The honest small-sample yardstick: "could a
    couple-dozen random dates on this basket have produced this CAR?"
    """
    rng = np.random.default_rng(seed)
    br = basket_returns(prices, members)
    mr = prices[benchmark].pct_change().dropna()
    common = br.index.intersection(mr.index)
    bsk = br.loc[common].values
    mkt = mr.loc[common].values
    n = len(bsk)
    span_lo = EST_DAYS + EST_GAP + 1
    hi = n - window[1] - 2
    means = np.empty(n_draws)
    if hi <= span_lo:
        return np.array([])
    for i in range(n_draws):
        vals = []
        for _ in range(k):
            pos = int(rng.integers(span_lo, hi))
            est_hi = pos - EST_GAP
            est_lo = est_hi - EST_DAYS
            b, a = np.polyfit(mkt[est_lo:est_hi], bsk[est_lo:est_hi], 1)
            seg = slice(pos + window[0], pos + window[1] + 1)
            abn = bsk[seg] - (a + b * mkt[seg])
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
              members: list[str] | None = None, benchmark: str = "SPY",
              window: tuple[int, int] = DEFAULT_WINDOW,
              n_draws: int = 20_000) -> dict:
    """Headline event-study stats: strict / hybrid / all buckets, the strict−hybrid
    difference (Welch t), and an all-events placebo p-value if prices are given."""
    strict = panel.loc[panel["strict"], "car"].to_numpy(float)
    hybrid = panel.loc[~panel["strict"], "car"].to_numpy(float)
    allc = panel["car"].to_numpy(float)
    out = {
        "strict": summarize_bucket(strict),
        "hybrid": summarize_bucket(hybrid),
        "all": summarize_bucket(allc),
        "diff_pct": (float(np.nanmean(strict) - np.nanmean(hybrid)) * 100
                     if len(strict) and len(hybrid) else float("nan")),
        "diff_t": welch_t(strict, hybrid),
    }
    if prices is not None and members is not None and len(allc):
        null = placebo_car_dist(prices, members, k=len(allc), benchmark=benchmark,
                                window=window, n_draws=n_draws)
        out["all_placebo_p"] = placebo_pvalue(float(np.nanmean(allc)), null)
        out["null_mean_pct"] = float(np.nanmean(null) * 100) if null.size else float("nan")
    return out


def net_of_costs(car_mean: float, cost_bps: float = 10.0) -> dict:
    """One-way round-trip cost on a 'buy the basket on the mandate, hold the window' trade.

    A short event-window basket trade is one entry + one exit; 10 bps round-trip covers an
    institutional REIT desk trading a liquid basket. Returns gross/net mean CAR. As with the
    rest of the study, cost is not the binding constraint — the event count and the sign of
    the reaction are."""
    c = cost_bps / 1e4
    return {"gross_pct": float(car_mean * 100), "net_pct": float((car_mean - c) * 100),
            "cost_bps": cost_bps}
