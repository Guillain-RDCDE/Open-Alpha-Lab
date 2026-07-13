"""Strategy + inference for Study 749 — Layoff-Drift (event-study abnormal returns).

The claim: a mass-layoff announcement is a *bullish* catalyst — the market cheers the
cost discipline (the **"restructuring pop"**) and then the stock keeps drifting up as
margins improve (a **PEAD-style continuation**). The bear framing is the mirror image:
layoffs signal distress, so the stock sags and drifts *down*. We pin both down with a
textbook event study:

    For each dated layoff announcement, fit a **market model** (stock = α + β·SPY) on a
    clean PRE-event estimation window, then compute the **abnormal return**
    AR_t = r_stock,t − (α + β·r_SPY,t). Cumulate it over a short **pop** window
    [+1, +3] (the "restructuring pop") and a longer **drift** window [+4, +63] (the
    PEAD-style continuation). A one-day entry lag applies throughout: you learn the
    announcement intraday and can only act from the next close, so there is no
    look-ahead.

We then ask:

  * Is the **pop** leg different from zero? (Welch t across events + placebo null)
  * Is the **drift** leg different from zero? (Welch t across events **and** a Newey-West
    **HAC t** on the pooled daily abnormal-return series — the honest test of a drift,
    whose daily increments are autocorrelated by construction)
  * Does a believers' **buy-and-hold-the-drift** book survive a one-way cost?

The decisive object is not the sign of any point estimate but whether either leg clears
|t| ≥ 2 on ~two dozen events: with a small, heterogeneous sample the abnormal returns
scatter far more than any mean, and the survivor tape (the worst restructurings that
delisted leave no series) biases the drift *up*, so a survivor-only drift near zero is a
conservative refutation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Event-study geometry (trading days, relative to the announcement = day 0).
EST_DAYS = 120          # length of the market-model estimation window
EST_GAP = 10            # gap between estimation window and event window (no leakage)
LAG = 1                 # one-day execution lag — act from the next close
POP_WIN = (1, 3)        # "restructuring pop" window [+1, +3] trading days
DRIFT_WIN = (4, 63)     # PEAD-style drift window [+4, +63] trading days (~3 months)


# --------------------------------------------------------------------------- #
# Market-model abnormal returns for a single event
# --------------------------------------------------------------------------- #
def _simple_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change()


def event_abnormal(prices: pd.DataFrame, ticker: str, announce_date: pd.Timestamp,
                   pop_win: tuple[int, int] = POP_WIN,
                   drift_win: tuple[int, int] = DRIFT_WIN,
                   est_days: int = EST_DAYS, est_gap: int = EST_GAP,
                   lag: int = LAG) -> dict | None:
    """Market-model abnormal returns around one layoff announcement.

    Fits ``r_stock = a + b * r_SPY`` on the estimation window [−est_gap−est_days,
    −est_gap] relative to the announcement, then cumulates the abnormal return AR over
    the **pop** window and the **drift** window (both inclusive, in trading days, and
    both offset by ``lag`` so entry is the day after the headline). Returns ``None`` if
    the windows do not fit the available history.

    The returned dict carries the two cumulative CARs (``pop``, ``drift``) and the
    per-day abnormal-return arrays over each window (``pop_daily``, ``drift_daily``) so
    the caller can pool daily increments for a HAC t-stat.
    """
    if ticker not in prices.columns or "SPY" not in prices.columns:
        return None
    rs = _simple_returns(prices[ticker]).dropna()
    rm = _simple_returns(prices["SPY"]).dropna()
    common = rs.index.intersection(rm.index)
    rs, rm = rs.loc[common], rm.loc[common]
    idx = common
    pos = int(np.searchsorted(idx, pd.Timestamp(announce_date)))
    if pos <= 0 or pos >= len(idx):
        return None
    # estimation window (clean, pre-event)
    est_hi = pos - est_gap
    est_lo = est_hi - est_days
    # event windows, offset by the execution lag
    pop_lo = pos + pop_win[0] + lag
    pop_hi = pos + pop_win[1] + lag
    drift_lo = pos + drift_win[0] + lag
    drift_hi = pos + drift_win[1] + lag
    if est_lo < 1 or drift_hi >= len(idx):
        return None
    mkt_e = rm.values[est_lo:est_hi]
    stk_e = rs.values[est_lo:est_hi]
    if len(mkt_e) < 30:
        return None
    b, a = np.polyfit(mkt_e, stk_e, 1)

    def abn(lo, hi):
        return rs.values[lo:hi + 1] - (a + b * rm.values[lo:hi + 1])

    pop_daily = abn(pop_lo, pop_hi)
    drift_daily = abn(drift_lo, drift_hi)
    return {"ticker": ticker, "announce_date": idx[pos],
            "pop": float(np.nansum(pop_daily)),
            "drift": float(np.nansum(drift_daily)),
            "pop_daily": pop_daily, "drift_daily": drift_daily}


def car_panel(prices: pd.DataFrame, events: list[dict],
              pop_win: tuple[int, int] = POP_WIN,
              drift_win: tuple[int, int] = DRIFT_WIN, lag: int = LAG) -> pd.DataFrame:
    """Pop and drift CAR for every event in the table; drops unpriced / short-history ones."""
    rows = []
    for e in events:
        w = event_abnormal(prices, e["ticker"], e["announce_date"],
                           pop_win=pop_win, drift_win=drift_win, lag=lag)
        if w is None:
            continue
        rows.append({"ticker": w["ticker"], "announce_date": w["announce_date"],
                     "cut": e.get("cut", 0), "pop": w["pop"], "drift": w["drift"]})
    return pd.DataFrame(rows)


def pooled_daily_drift(prices: pd.DataFrame, events: list[dict],
                       drift_win: tuple[int, int] = DRIFT_WIN, lag: int = LAG) -> np.ndarray:
    """Pool the daily abnormal returns over the drift window across every event.

    The mean of this pool is the average daily drift; its **HAC (Newey-West) t** is the
    honest test of whether a drift exists, because the daily increments within an event
    are autocorrelated (a drift is, by definition, a persistent same-sign push)."""
    out = []
    for e in events:
        w = event_abnormal(prices, e["ticker"], e["announce_date"],
                           drift_win=drift_win, lag=lag)
        if w is None:
            continue
        out.append(w["drift_daily"])
    return np.concatenate(out) if out else np.array([])


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def welch_t(sample: np.ndarray, mu0: float = 0.0) -> float:
    """One-sample t of ``mean(sample) - mu0``. NaN if sample < 2 or zero variance."""
    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    return float((sample.mean() - mu0) / se) if se > 0 else float("nan")


def hac_t(x: np.ndarray, lags: int = 5) -> float:
    """Newey-West (HAC) t-stat that the mean of a series is zero.

    For a drift, the natural object is the pooled daily abnormal return: its mean is the
    average daily drift, and the Newey-West correction with ``lags`` accounts for the
    autocorrelation a genuine drift would induce. Returns t = mean / HAC-SE(mean)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    xc = x - x.mean()
    gamma0 = float(xc @ xc) / n
    s = gamma0
    for k in range(1, min(lags, n - 1) + 1):
        cov = float(xc[k:] @ xc[:-k]) / n
        w = 1.0 - k / (lags + 1)          # Bartlett kernel
        s += 2.0 * w * cov
    if s <= 0:
        return float("nan")
    se = np.sqrt(s / n)                    # HAC SE of the mean
    return float(x.mean() / se) if se > 0 else float("nan")


def placebo_pvalue(prices: pd.DataFrame, tickers: list[str], k: int, leg: str = "pop",
                   pop_win: tuple[int, int] = POP_WIN,
                   drift_win: tuple[int, int] = DRIFT_WIN, lag: int = LAG,
                   n_draws: int = 20_000, seed: int = 749,
                   observed: float | None = None) -> dict:
    """Small-sample placebo null: draw ``k`` random (ticker, date) non-event windows many
    times and ask how often a random draw's mean abnormal ``leg`` return (market-model
    adjusted) matches/beats the real layoff set.

    Returns the observed mean, the placebo mean, and the empirical two-sided p-value
    ``P[|random-draw mean| >= |observed mean|]`` — the honest answer to "could ~two dozen
    random windows on these names have produced this pop (or this drift) by chance?".
    """
    if observed is None or not np.isfinite(observed):
        return {"k": k, "obs": observed, "placebo_mean": float("nan"),
                "p_value": float("nan")}
    win = pop_win if leg == "pop" else drift_win
    span = win[1] - win[0] + 1
    avail = [t for t in tickers if t in prices.columns]
    rm = _simple_returns(prices["SPY"]).dropna()
    rets = {}
    for t in avail:
        rs = _simple_returns(prices[t]).dropna()
        common = rs.index.intersection(rm.index)
        rets[t] = (rs.loc[common].values, rm.loc[common].values)
    span_lo = EST_DAYS + EST_GAP + 1
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for i in range(n_draws):
        vals = []
        for _ in range(k):
            t = avail[rng.integers(0, len(avail))]
            stk, mkt = rets[t]
            n = len(stk)
            hi = n - span - lag - 2
            if hi <= span_lo:
                continue
            pos = int(rng.integers(span_lo, hi))
            est_hi = pos - EST_GAP
            est_lo = est_hi - EST_DAYS
            b, a = np.polyfit(mkt[est_lo:est_hi], stk[est_lo:est_hi], 1)
            lo = pos + win[0] + lag
            hh = pos + win[1] + lag
            abn = stk[lo:hh + 1] - (a + b * mkt[lo:hh + 1])
            vals.append(float(np.nansum(abn)))
        means[i] = np.mean(vals) if vals else np.nan
    means = means[np.isfinite(means)]
    if means.size == 0:
        return {"k": k, "obs": float(observed), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    p = float((np.abs(means) >= abs(observed)).mean())
    return {"k": k, "obs": float(observed), "placebo_mean": float(means.mean()),
            "p_value": p}


def summarize_leg(car: np.ndarray) -> dict:
    """n, mean CAR (%), win-rate (P[CAR>0]), and the Welch t of mean vs 0."""
    car = np.asarray(car, dtype=float)
    car = car[np.isfinite(car)]
    if len(car) == 0:
        return {"n": 0, "mean_pct": float("nan"), "win": float("nan"), "t": float("nan")}
    return {"n": int(len(car)), "mean_pct": float(car.mean() * 100),
            "win": float((car > 0).mean()), "t": welch_t(car)}


def summarize(panel: pd.DataFrame, prices: pd.DataFrame | None = None,
              tickers: list[str] | None = None, n_draws: int = 8000,
              hac_series: np.ndarray | None = None) -> dict:
    """Headline event-study stats: pop / drift legs (Welch t + placebo p), the drift
    HAC t on the pooled daily series, and the cross-sectional win-rates."""
    out = {"n": int(len(panel))}
    for leg in ("pop", "drift"):
        x = panel[leg].to_numpy(float)
        d = summarize_leg(x)
        if prices is not None and tickers is not None and len(x):
            d["p_placebo"] = placebo_pvalue(prices, tickers, k=int(np.isfinite(x).sum()),
                                            leg=leg, observed=float(np.nanmean(x)),
                                            n_draws=n_draws)["p_value"]
        else:
            d["p_placebo"] = float("nan")
        out[leg] = d
    if hac_series is not None and len(hac_series):
        out["drift"]["hac_t"] = hac_t(hac_series)
        out["drift"]["daily_mean_bps"] = float(np.nanmean(hac_series) * 1e4)
    return out


# --------------------------------------------------------------------------- #
# Tradability — buy-and-hold-the-drift book, net of costs
# --------------------------------------------------------------------------- #
def net_of_costs(drift_mean: float, cost_bps: float = 10.0) -> dict:
    """One-way round-trip cost on a 'buy the day after the announcement, hold the drift
    window, then sell' trade — one entry + one exit. 10 bps round-trip covers an
    institutional large-cap desk. Returns gross/net mean drift CAR. (As with the rest of
    the study, cost is not the binding constraint — the absence of a drift is.)"""
    c = cost_bps / 1e4
    return {"gross_pct": float(drift_mean * 100),
            "net_pct": float((drift_mean - c) * 100), "cost_bps": cost_bps}
