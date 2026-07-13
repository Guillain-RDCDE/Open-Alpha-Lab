"""Strategy + inference for Study 722 — Logo-Rebrand (renewal, or a floundering firm?).

The claim comes in two mutually contradictory flavours, and we test both against the tape:

  * **Renewal camp.** A rebrand / new logo marks a turnaround — a fresh identity that
    re-rates the stock. *Buy the rebrand.* Prediction: positive post-reveal **drift**.
  * **Skeptic camp.** A rebrand is what a *floundering* firm does to distract from
    fundamentals — a vanity red flag. *Fade the rebrand.* Prediction: negative drift.

Both are directional bets on the **abnormal drift after a rebrand**, so a single event study
adjudicates them:

  * **Abnormal returns.** Around each rebrand we compute returns *in excess of the market*
    (SPY) — a re-rating is supposed to be abnormal, not just beta.
  * **Two legs.** A short **announce window** right after the reveal (the market's instant
    verdict) and a longer **drift window** after that (the renewal / decline horizon). Renewal
    needs a positive drift; the skeptic needs a negative one; folklore-as-noise gives neither.
  * **Inference.** A Welch *t* on the cross-section of event abnormal returns vs zero, a
    **placebo / bootstrap** null sized to the event count (random non-event dates on the same
    tape), and a **win-rate** vs a 50% base rate.
  * **Execution & costs.** A 1-day entry lag (you act the day *after* the rebrand headline)
    and one-way costs × turnover on a buy-the-rebrand book.

The decisive object is the *cross-section of a couple-dozen events*: with ~24 events, an
abnormal drift of a few percent is well inside its own standard error, and the survivor tape
(the worst-outcome rebrands delisted / went private) biases the drift *up*, against the
skeptic. Same small-sample / base-rate pathology as the desk's other announcement studies
(cf. Study 389 — Name-Change-Effect).
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
                 announce: int = 5, drift: int = 120, lag: int = 1) -> dict | None:
    """Abnormal (excess-of-SPY) returns around one rebrand.

    With a 1-day entry lag, the **announce** leg is the cumulative excess return over the
    ``announce`` trading days starting the day after the reveal; the **drift** leg is the
    cumulative excess return over the following ``drift`` trading days (~6 months at 120).
    Returns ``None`` if the ticker isn't priced around ``when`` (e.g. listed later) or the
    window overruns.
    """
    if ticker not in prices.columns or "SPY" not in prices.columns:
        return None
    ex = _excess_log_returns(prices[ticker], prices["SPY"])
    if ex.empty or pd.isna(when):
        return None
    pos = ex.index.searchsorted(when)
    if pos <= 0 or pos >= len(ex):
        return None
    start = pos + lag                       # act the day after the headline
    ann_end = start + announce
    drift_end = ann_end + drift
    if drift_end >= len(ex):
        return None
    ann_ret = float(np.expm1(ex.iloc[start:ann_end].sum()))
    drift_ret = float(np.expm1(ex.iloc[ann_end:drift_end].sum()))
    return {"ticker": ticker, "announce": ann_ret, "drift": drift_ret,
            "date": ex.index[pos]}


def collect_events(bundle: dict, announce: int = 5, drift: int = 120,
                   lag: int = 1) -> pd.DataFrame:
    """Run :func:`event_window` over every row of the event table; drop unpriced ones."""
    prices, events = bundle["prices"], bundle["events"]
    rows = []
    for _, r in events.iterrows():
        w = event_window(prices, r["ticker"], r["snapped"], announce=announce,
                         drift=drift, lag=lag)
        if w is None:
            continue
        w["kind"] = r["kind"]
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


def placebo_pvalue(bundle: dict, k: int, leg: str = "drift", announce: int = 5,
                   drift: int = 120, lag: int = 1, n_draws: int = 20_000, seed: int = 722,
                   observed: float | None = None) -> dict:
    """Small-sample placebo null: draw ``k`` random (ticker, date) non-event windows many
    times and ask how often a random draw's mean abnormal ``leg`` return matches/beats the
    real rebrand set.

    Returns the observed mean, the placebo mean, and the empirical two-sided-ish p-value
    ``P[|random-draw mean| >= |observed mean|]`` — the honest answer to "could ~24 random
    windows have produced this announce reaction (or this drift) by chance?".
    """
    prices = bundle["prices"]
    tickers = [c for c in prices.columns if c != "SPY"]
    if observed is None or np.isnan(observed):
        return {"k": k, "obs": observed, "placebo_mean": float("nan"),
                "p_value": float("nan")}
    span = announce if leg == "announce" else drift
    # Build, for every ticker, the array of ALL `span`-day cumulative excess returns
    # (windows offset to skip the ticker's own announce region for the drift leg), then pool
    # them across tickers into one vector. Sampling from this pool is the placebo draw.
    pool = []
    for t in tickers:
        s = _excess_log_returns(prices[t], prices["SPY"]).values
        if len(s) <= announce + drift + lag + 5:
            continue
        c = np.concatenate([[0.0], np.cumsum(s)])      # c[i+span]-c[i] = sum of span days
        off = 0 if leg == "announce" else announce
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


def summarize(events: pd.DataFrame, bundle: dict | None = None, announce: int = 5,
              drift: int = 120, lag: int = 1, placebo: bool = True) -> dict:
    """Headline stats: per-leg mean abnormal return, win-rate, Welch t, placebo p."""
    out = {"n": int(len(events))}
    if len(events) == 0:
        return out
    for leg in ("announce", "drift"):
        x = events[leg].values
        d = {
            "mean": float(np.mean(x)),
            "win": float((x > 0).mean()),
            "t": welch_t(x),
        }
        if placebo and bundle is not None:
            d["p_placebo"] = placebo_pvalue(bundle, len(x), leg=leg, announce=announce,
                                            drift=drift, lag=lag, observed=d["mean"])["p_value"]
        else:
            d["p_placebo"] = float("nan")
        out[leg] = d
    return out


# --------------------------------------------------------------------------- #
# Tradability — the "buy the rebrand" (renewal) book, net of costs
# --------------------------------------------------------------------------- #
def net_of_costs(events: pd.DataFrame, cost_bps: float = 10.0) -> dict:
    """A long "buy the rebrand and hold the drift window" book, one-way cost per crossing.

    The renewal camp's trade is: buy on the rebrand headline (day+1), hold the drift window,
    then exit — **two** one-way crossings per event (in, out). These are mostly large-cap,
    liquid names, so ``cost_bps`` is a small one-way charge. We report the announce leg, the
    drift leg, and the full "hold through both" P&L, gross and net. (The skeptic's fade trade
    is just the sign-flip of the same book.)
    """
    if len(events) == 0:
        return {"n": 0, "gross_announce": float("nan"), "net_announce": float("nan"),
                "gross_drift": float("nan"), "net_drift": float("nan"),
                "gross_hold": float("nan"), "net_hold": float("nan")}
    c = cost_bps / 1e4
    ann = events["announce"].values
    drift = events["drift"].values
    hold = ann + drift                          # buy at reveal, hold through both windows
    return {
        "n": int(len(events)),
        "gross_announce": float(ann.mean()), "net_announce": float(ann.mean() - 2 * c),
        "gross_drift": float(drift.mean()), "net_drift": float(drift.mean() - 2 * c),
        "gross_hold": float(hold.mean()), "net_hold": float(hold.mean() - 2 * c),
        "cost_bps": cost_bps,
    }
