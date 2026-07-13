"""Strategy + inference for Study 720 — Super-Bowl-Advertiser (does the big-ad signal pay?).

The claim: a company that runs a **Super Bowl commercial** gets a burst of attention and its
stock **drifts up in the days after the game** — a "big-ad signal" you could buy every
February. We make this falsifiable with an event study:

  * **Abnormal returns.** Around each game we compute returns *in excess of the market* (SPY)
    — the believers' "drift" is supposed to be abnormal, not just beta.
  * **Two legs + a reaction.** A short **drift window** right after the game (the "big-ad
    signal" you'd trade) and a longer **hold window** after that (does it persist). We also
    record the **Monday reaction** itself — the first trading day's excess move, which folds
    in the un-tradable Fri->Mon weekend gap — to separate the instant from the drift.
  * **Inference.** A Welch *t* on the cross-section of event abnormal returns vs zero, a
    **placebo / bootstrap** null sized to the event count (random non-event windows on the
    same tape), and a **win-rate** of the drift vs a 50% base rate.
  * **Execution & costs.** A 1-day entry lag (you learn which ads ran Sunday night, but you
    can only act at Monday's close -> the drift leg starts Tuesday, no weekend-gap look-ahead)
    and one-way costs × turnover on a long-the-advertisers ad-calendar book.

The decisive object is the *cross-section of a few dozen events*: with ~32 events, an abnormal
drift of a few tenths of a percent is well inside its own standard error, and the survivor tape
(the loudest advertisers that went to zero delisted) biases the drift *up*. Same small-sample /
base-rate pathology as the desk's other event studies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Abnormal-return event windows
# --------------------------------------------------------------------------- #
def _excess_log_returns(stock: pd.Series, mkt: pd.Series) -> pd.Series:
    """Daily log return of ``stock`` in excess of the market (SPY), aligned on dates."""
    df = pd.concat([stock, mkt], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    lr = np.log(df.iloc[:, 0]).diff() - np.log(df.iloc[:, 1]).diff()
    return lr.dropna()


def event_window(prices: pd.DataFrame, ticker: str, when: pd.Timestamp,
                 drift: int = 5, hold: int = 20, lag: int = 1) -> dict | None:
    """Abnormal (excess-of-SPY) returns around one Super Bowl ad.

    ``when`` is the Monday after the game (the first tradable day). With a 1-day entry lag,
    the **drift** leg is the cumulative excess return over the ``drift`` trading days starting
    the day after that Monday (i.e. Tuesday onward — the days-after-the-game "big-ad signal"
    you could actually trade, with no weekend-gap look-ahead); the **hold** leg is the
    cumulative excess return over the following ``hold`` trading days. The **Monday reaction**
    is the single-day excess return on ``when`` itself (folds in the un-tradable Fri->Mon
    gap). Returns ``None`` if the ticker isn't priced around ``when`` or the window overruns.
    """
    if ticker not in prices.columns or "SPY" not in prices.columns:
        return None
    ex = _excess_log_returns(prices[ticker], prices["SPY"])
    if ex.empty or pd.isna(when):
        return None
    pos = ex.index.searchsorted(when)
    if pos <= 0 or pos >= len(ex):
        return None
    monday_ret = float(np.expm1(ex.iloc[pos]))     # the reaction day (incl. weekend gap)
    start = pos + lag                              # act the day after the Monday close
    drift_end = start + drift
    hold_end = drift_end + hold
    if hold_end >= len(ex):
        return None
    drift_ret = float(np.expm1(ex.iloc[start:drift_end].sum()))
    hold_ret = float(np.expm1(ex.iloc[drift_end:hold_end].sum()))
    return {"ticker": ticker, "monday": monday_ret, "drift": drift_ret, "hold": hold_ret,
            "date": ex.index[pos]}


def collect_events(bundle: dict, drift: int = 5, hold: int = 20,
                   lag: int = 1) -> pd.DataFrame:
    """Run :func:`event_window` over every row of the event table; drop unpriced ones."""
    prices, events = bundle["prices"], bundle["events"]
    rows = []
    for _, r in events.iterrows():
        w = event_window(prices, r["ticker"], r["snapped"], drift=drift, hold=hold, lag=lag)
        if w is None:
            continue
        w["year"] = r["year"]
        w["label"] = r["label"]
        rows.append(w)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def welch_t(sample: np.ndarray, mu0: float = 0.0) -> float:
    """One-sample t of ``mean(sample) - mu0``. NaN if sample < 2 or zero variance."""
    sample = np.asarray(sample, dtype=float)
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    if se == 0:
        return float("nan")
    return float((sample.mean() - mu0) / se)


def placebo_pvalue(bundle: dict, k: int, leg: str = "drift", drift: int = 5, hold: int = 20,
                   lag: int = 1, n_draws: int = 20_000, seed: int = 720,
                   observed: float | None = None) -> dict:
    """Small-sample placebo null: draw ``k`` random (ticker, date) non-event windows many
    times and ask how often a random draw's mean abnormal ``leg`` return matches/beats the
    real advertiser set.

    Returns the observed mean, the placebo mean, and the empirical two-sided-ish p-value
    ``P[|random-draw mean| >= |observed mean|]`` — the honest answer to "could ~32 random
    windows have produced this drift by chance?".
    """
    prices = bundle["prices"]
    tickers = [c for c in prices.columns if c != "SPY"]
    if observed is None or np.isnan(observed):
        return {"k": k, "obs": observed, "placebo_mean": float("nan"),
                "p_value": float("nan")}
    span = drift if leg == "drift" else hold
    # Build, for every ticker, the array of ALL `span`-day cumulative excess returns (offset
    # to skip the ticker's own drift region for the hold leg), then pool them across tickers
    # into one vector. Sampling from this pool is the placebo draw.
    pool = []
    for t in tickers:
        s = _excess_log_returns(prices[t], prices["SPY"]).values
        if len(s) <= drift + hold + lag + 5:
            continue
        c = np.concatenate([[0.0], np.cumsum(s)])   # c[i+1]-c[i+1-span] = sum of span days
        off = 0 if leg == "drift" else drift
        lo = lag + off
        hi = len(s) - span - 1
        if hi <= lo:
            continue
        starts = np.arange(lo, hi)
        sums = c[starts + span] - c[starts]
        pool.append(np.expm1(sums))
    if not pool:
        return {"k": k, "obs": float(observed), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    pool = np.concatenate(pool)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(pool), size=(n_draws, k))
    means = pool[draws].mean(axis=1)
    p = float((np.abs(means) >= abs(observed)).mean())
    return {"k": k, "obs": float(observed), "placebo_mean": float(means.mean()),
            "p_value": p}


def summarize(events: pd.DataFrame, bundle: dict | None = None, drift: int = 5,
              hold: int = 20, lag: int = 1, placebo: bool = True) -> dict:
    """Headline stats: per-leg mean abnormal return, win-rate, Welch t, placebo p."""
    out = {"n": int(len(events))}
    if len(events) == 0:
        return out
    # the un-tradable Monday reaction, reported separately
    m = events["monday"].values
    out["monday"] = {"mean": float(np.mean(m)), "win": float((m > 0).mean()),
                     "t": welch_t(m)}
    for leg in ("drift", "hold"):
        x = events[leg].values
        d = {
            "mean": float(np.mean(x)),
            "win": float((x > 0).mean()),
            "t": welch_t(x),
        }
        if placebo and bundle is not None:
            d["p_placebo"] = placebo_pvalue(bundle, len(x), leg=leg, drift=drift, hold=hold,
                                            lag=lag, observed=d["mean"])["p_value"]
        else:
            d["p_placebo"] = float("nan")
        out[leg] = d
    return out


# --------------------------------------------------------------------------- #
# Tradability — long-the-advertisers ad-calendar book, net of costs
# --------------------------------------------------------------------------- #
def net_of_costs(events: pd.DataFrame, cost_bps: float = 10.0) -> dict:
    """A long-the-advertisers book that buys the drift, one-way cost in bps per crossing.

    The believers' trade is: every February, buy the basket of Super Bowl advertisers Monday
    and ride the "big-ad signal" for the drift window. That is **two** one-way executions per
    event (in, out). These are large, liquid names, so ``cost_bps`` is a modest *one-way*
    charge per crossing. Returns gross/net mean per-event P&L for the drift leg (and the
    drift+hold hold-through variant).
    """
    if len(events) == 0:
        return {"n": 0, "gross_drift": float("nan"), "net_drift": float("nan"),
                "gross_hold": float("nan"), "net_hold": float("nan")}
    c = cost_bps / 1e4
    drift = events["drift"].values
    hold = events["hold"].values
    combo = drift + hold                          # buy Monday, hold through the whole window
    return {
        "n": int(len(events)),
        "gross_drift": float(drift.mean()), "net_drift": float(drift.mean() - 2 * c),
        "gross_hold": float(combo.mean()), "net_hold": float(combo.mean() - 2 * c),
        "cost_bps": cost_bps,
    }
