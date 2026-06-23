"""Strategy + inference for Study 389 — Name-Change-Effect (theme-chasing rebrands).

The claim: a company that renames toward the hot theme (``.com``, ``blockchain``, ``AI``)
**pops** on the rename and then **gives it back**. We make this falsifiable with an event
study:

  * **Abnormal returns.** Around each rebrand we compute returns *in excess of the market*
    (SPY) — the believers' "pop" is supposed to be abnormal, not just beta.
  * **Two legs.** A short **pop window** right after the rename (the +) and a longer
    **fade window** after that (the −). The story needs *both*: a significant positive pop
    AND a significant negative fade (the give-back).
  * **Inference.** A Welch *t* on the cross-section of event abnormal returns vs zero, a
    **placebo / bootstrap** null sized to the event count (random non-event dates on the
    same tape), and a **win-rate** of the pop vs a 50% base rate.
  * **Execution & costs.** A 1-day entry lag (you act the day *after* the rename headline)
    and one-way costs × turnover on a buy-the-pop / fade-it book.

The decisive object is the *cross-section of a few dozen events*: with ~25 events, an
abnormal pop of a few percent is well inside its own standard error, and the survivor tape
(the worst give-backs delisted) biases the fade *toward* zero. Same small-sample / base-rate
pathology as the desk's other event studies.
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
                 pop: int = 5, fade: int = 60, lag: int = 1) -> dict | None:
    """Abnormal (excess-of-SPY) returns around one rebrand.

    With a 1-day entry lag, the **pop** leg is the cumulative excess return over the
    ``pop`` trading days starting the day after the rename; the **fade** leg is the
    cumulative excess return over the following ``fade`` trading days. Returns ``None`` if
    the ticker isn't priced around ``when`` (e.g. listed later) or the window overruns.
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
    pop_end = start + pop
    fade_end = pop_end + fade
    if fade_end >= len(ex):
        return None
    pop_ret = float(np.expm1(ex.iloc[start:pop_end].sum()))
    fade_ret = float(np.expm1(ex.iloc[pop_end:fade_end].sum()))
    return {"ticker": ticker, "pop": pop_ret, "fade": fade_ret,
            "date": ex.index[pos]}


def collect_events(bundle: dict, pop: int = 5, fade: int = 60,
                   lag: int = 1) -> pd.DataFrame:
    """Run :func:`event_window` over every row of the event table; drop unpriced ones."""
    prices, events = bundle["prices"], bundle["events"]
    rows = []
    for _, r in events.iterrows():
        w = event_window(prices, r["ticker"], r["snapped"], pop=pop, fade=fade, lag=lag)
        if w is None:
            continue
        w["wave"] = r["wave"]
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


def placebo_pvalue(bundle: dict, k: int, leg: str = "pop", pop: int = 5, fade: int = 60,
                   lag: int = 1, n_draws: int = 20_000, seed: int = 389,
                   observed: float | None = None) -> dict:
    """Small-sample placebo null: draw ``k`` random (ticker, date) non-event windows many
    times and ask how often a random draw's mean abnormal ``leg`` return matches/beats the
    real rebrand set.

    Returns the observed mean, the placebo mean, and the empirical two-sided-ish p-value
    ``P[|random-draw mean| >= |observed mean|]`` — the honest answer to "could ~25 random
    windows have produced this pop (or this fade) by chance?".
    """
    prices = bundle["prices"]
    tickers = [c for c in prices.columns if c != "SPY"]
    if observed is None or np.isnan(observed):
        return {"k": k, "obs": observed, "placebo_mean": float("nan"),
                "p_value": float("nan")}
    span = pop if leg == "pop" else fade
    # Build, for every ticker, the array of ALL `span`-day cumulative excess returns
    # (windows offset to skip the ticker's own pop region for the fade leg), then pool
    # them across tickers into one vector. Sampling from this pool is the placebo draw.
    pool = []
    for t in tickers:
        s = _excess_log_returns(prices[t], prices["SPY"]).values
        if len(s) <= pop + fade + lag + 5:
            continue
        c = np.concatenate([[0.0], np.cumsum(s)])      # c[i+1]-c[i+1-span] = sum of span days
        # all valid window sums: indices [lag+off .. len-span-1]
        off = 0 if leg == "pop" else pop
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


def summarize(events: pd.DataFrame, bundle: dict | None = None, pop: int = 5,
              fade: int = 60, lag: int = 1, placebo: bool = True) -> dict:
    """Headline stats: per-leg mean abnormal return, win-rate, Welch t, placebo p."""
    out = {"n": int(len(events))}
    if len(events) == 0:
        return out
    for leg in ("pop", "fade"):
        x = events[leg].values
        d = {
            "mean": float(np.mean(x)),
            "win": float((x > 0).mean()),
            "t": welch_t(x),
        }
        if placebo and bundle is not None:
            d["p_placebo"] = placebo_pvalue(bundle, len(x), leg=leg, pop=pop, fade=fade,
                                            lag=lag, observed=d["mean"])["p_value"]
        else:
            d["p_placebo"] = float("nan")
        out[leg] = d
    return out


# --------------------------------------------------------------------------- #
# Tradability — buy-the-pop / fade-it book, net of costs
# --------------------------------------------------------------------------- #
def net_of_costs(events: pd.DataFrame, cost_bps: float = 30.0) -> dict:
    """A long-the-pop then short-the-fade book, one-way cost in bps charged per leg-entry.

    The believers' trade is: buy on the rename (capture the pop), then flip to short
    (capture the give-back). That is **four** one-way executions per event (in, out, in,
    out), and the names are small-cap rebrands with wide spreads — ``cost_bps`` is a
    *one-way* charge per crossing. Returns gross/net mean per-event P&L for the round trip.
    """
    if len(events) == 0:
        return {"n": 0, "gross_pop": float("nan"), "net_pop": float("nan"),
                "gross_fade": float("nan"), "net_fade": float("nan"),
                "gross_combo": float("nan"), "net_combo": float("nan")}
    c = cost_bps / 1e4
    pop = events["pop"].values
    fade = events["fade"].values
    # combo = long the pop, then SHORT the fade (profit if fade is negative)
    gross_combo = pop - fade
    net_combo = gross_combo - 4 * c          # 4 one-way crossings (in/out, in/out)
    return {
        "n": int(len(events)),
        "gross_pop": float(pop.mean()), "net_pop": float(pop.mean() - 2 * c),
        "gross_fade": float(fade.mean()), "net_fade": float(fade.mean() - 2 * c),
        "gross_combo": float(gross_combo.mean()), "net_combo": float(net_combo.mean()),
        "cost_bps": cost_bps,
    }
