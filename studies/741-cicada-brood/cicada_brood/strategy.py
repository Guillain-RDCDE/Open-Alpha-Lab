"""The event-study engine and its honest controls — Study 741 (Cicada-Brood).

The claim under test, steelmanned as far as a deliberately silly claim can be: a
periodical-cicada brood emergence is a huge, media-saturated, *fixed-calendar* natural
event; a folklore "cicada indicator" holds that emergence springs are special for the
S&P 500 — and, crucially, that you could *front-run* them, because the emergence year has
been known to the calendar since the previous emergence 13 or 17 years earlier. So this
is the one piece of folklore on the desk with **zero look-ahead risk by construction**:
we are allowed to enter at the last close *before* the spring window, because the event's
date was never a surprise. If even a perfectly-foreseeable calendar signal is nothing,
that is the cleanest possible spurious-pattern demonstration.

The machinery, one execution convention documented throughout:

* ``daily_returns`` / ``abnormal_returns`` — a constant-mean market model (Brown & Warner
  1985): the "normal" return is SPY's full-sample mean daily return, so the abnormal
  return is the demeaned series. Demeaning strips the market's ordinary bull drift, so a
  positive spring CAR means "above SPY's own average", not "stocks go up".
* ``spring_window`` — for one year, the anchor (first session on/after May 1), the raw
  total-return over the [anchor-1 -> anchor+K] window (the tradable May-June return), and
  the constant-mean abnormal CAR over [anchor .. anchor+K].
* ``build_event_table`` — one row per emergence year: raw window return, abnormal CAR,
  and the timer legs. Independent, non-overlapping yearly events — the correct unit.
* ``one_sample_t`` — the primary statistic across emergence years (NOT a daily panel);
  ``wilson_interval`` / ``hit_rate`` carry the up-fraction; ``welch_t`` contrasts
  emergence vs non-emergence springs.
* ``random_year_placebo`` — the falsification test: draw the same number of *random*
  years, average their spring window return, repeat over many seeds. Because every draw
  uses the same May-June calendar window, the placebo controls for the ordinary
  seasonal ("sell in May") baseline automatically — it asks only whether *cicada* springs
  differ from *random* springs.
* ``car_path`` — the mean cumulative abnormal-return path across emergence years (the
  anatomy chart).
* ``spring_timer`` / ``summarize_timer`` — the tradable overlay: a pre-schedulable
  calendar trade, long SPY over the emergence-spring window only, one round trip of
  one-way costs charged twice against NAV, vs the unconditional all-year spring baseline.
* ``synthetic_detect`` — the machinery proof on a synthetic tape with a planted bump.

Costs are one-way x NAV per leg; the overlay is long-only (no borrow, no shorting bugs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt


# --------------------------------------------------------------------------- #
# Returns + abnormal returns (constant-mean market model)
# --------------------------------------------------------------------------- #
def daily_returns(close: pd.Series) -> pd.Series:
    """Simple close-to-close daily total returns."""
    return close.pct_change()


def abnormal_returns(ret: pd.Series) -> pd.Series:
    """Abnormal return = daily return minus its own full-sample mean (constant-mean model).

    Demeaning removes the trivial up-drift of equities so a spring CAR is not just
    "stocks go up over two months". The mean is taken over the covered series (NaNs
    excluded)."""
    return ret - ret.mean(skipna=True)


# --------------------------------------------------------------------------- #
# One year's spring window
# --------------------------------------------------------------------------- #
def spring_window(close: pd.Series, ar: pd.Series, year: int, k: int = dt.WINDOW_K
                  ) -> dict | None:
    """The cicada-spring measurements for one calendar ``year``.

    Anchor = first trading session on/after May 1 (``data._anchor_pos``). Returns a dict
    with the anchor/entry/exit dates, the **raw** total-return over [anchor-1 -> anchor+k]
    (the tradable May-June window return), and the **abnormal CAR** = sum of demeaned
    daily returns over [anchor .. anchor+k]. ``None`` if the window runs off the tape.
    """
    anchor = dt._anchor_pos(close, year)
    if anchor is None or anchor - 1 < 0 or anchor + k >= len(close):
        return None
    entry = anchor - 1                      # last April close (pre-window)
    exit_ = anchor + k                      # ~end-June close
    raw_ret = float(close.iat[exit_] / close.iat[entry] - 1.0)
    abn_car = float(ar.iloc[anchor: exit_ + 1].sum())
    return {"year": year,
            "anchor_date": close.index[anchor], "entry_date": close.index[entry],
            "exit_date": close.index[exit_], "raw_ret": raw_ret, "abn_car": abn_car}


def build_event_table(close: pd.Series, ar: pd.Series, years, k: int = dt.WINDOW_K,
                      cost_bps: float = 0.0) -> pd.DataFrame:
    """One row per emergence ``year``: raw window return, abnormal CAR, timer legs.

    ``cost_bps`` is one-way; the timer nets one round trip (2 x cost). Years whose window
    falls off the tape are dropped (not zero-filled). Independent, non-overlapping yearly
    events — the correct unit of inference."""
    rows = []
    for y in years:
        w = spring_window(close, ar, y, k)
        if w is None:
            continue
        w = dict(w)
        w["ret_gross"] = w["raw_ret"]
        w["ret_net"] = w["raw_ret"] - 2.0 * cost_bps * 1e-4
        rows.append(w)
    return pd.DataFrame(rows)


def all_year_windows(close: pd.Series, years, k: int = dt.WINDOW_K) -> pd.Series:
    """Raw spring window return for every year in ``years`` (the baseline / placebo pool).

    Returns a Series indexed by year. Uses the same May-June window as the events, so any
    seasonal effect is common to events and baseline alike."""
    ar = abnormal_returns(daily_returns(close))  # unused for raw, keeps signature simple
    out = {}
    for y in years:
        w = spring_window(close, ar, y, k)
        if w is not None:
            out[y] = w["raw_ret"]
    return pd.Series(out, dtype=float)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> dict:
    """One-sample t of mean(x) vs 0 — the right unit for independent yearly events."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2 or x.std(ddof=1) == 0:
        return {"n": int(n), "mean": float(x.mean()) if n else float("nan"),
                "sd": float("nan"), "t": float("nan")}
    se = x.std(ddof=1) / np.sqrt(n)
    return {"n": int(n), "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "t": float(x.mean() / se)}


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances). NaN if either group has < 2."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def hit_rate(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    k = int((x > 0).sum())
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": int(n), "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}


# --------------------------------------------------------------------------- #
# Random-year placebo — is the mean cicada-spring return in the luck cloud?
# --------------------------------------------------------------------------- #
def random_year_placebo(close: pd.Series, pool_years, n_events: int, k: int = dt.WINDOW_K,
                        n_seeds: int = 20, n_draws_per_seed: int = 1000,
                        base_seed: int = 741, tail: str = "right") -> dict:
    """Draw ``n_events`` random years from ``pool_years`` (with replacement across draws,
    without replacement within a draw), average their spring window return; repeat
    ``n_seeds x n_draws_per_seed`` times. ``tail="right"``: the silly claim predicts a
    positive "cicada bull", so p = share of null means >= observed emergence mean.

    Every draw uses the same May-June window, so the placebo controls the seasonal
    baseline by construction — it isolates whether *cicada* springs beat *random* springs.
    """
    windows = all_year_windows(close, pool_years, k)
    obs_pool = windows.values
    means = []
    npool = len(obs_pool)
    if npool < n_events:
        return {"obs": float("nan"), "placebo_mean": float("nan"),
                "placebo_sd": float("nan"), "p_value": float("nan"), "n_draws": 0}
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            pick = rng.choice(npool, size=n_events, replace=False)
            means.append(float(obs_pool[pick].mean()))
    return {"placebo_mean": float(np.mean(means)),
            "placebo_sd": float(np.std(means, ddof=1)),
            "means": np.asarray(means), "n_draws": len(means)}


def placebo_pvalue(observed: float, placebo_means: np.ndarray, tail: str = "right") -> float:
    """One-sided empirical p-value of ``observed`` within the placebo draws."""
    m = np.asarray(placebo_means, dtype=float)
    if m.size == 0 or not np.isfinite(observed):
        return float("nan")
    if tail == "right":
        return float((m >= observed).mean())
    return float((m <= observed).mean())


# --------------------------------------------------------------------------- #
# Event anatomy — mean cumulative abnormal return by offset
# --------------------------------------------------------------------------- #
def car_path(close: pd.Series, ar: pd.Series, years, k: int = dt.WINDOW_K) -> pd.DataFrame:
    """Mean CAR by offset (0..k from the May-1 anchor) across emergence years, each
    offset's own one-sample t. CAR(0) is the first in-window session's abnormal return;
    the path is re-anchored so CAR at offset 0 starts the cumulation (nothing before the
    window)."""
    paths = []
    for y in years:
        anchor = dt._anchor_pos(close, y)
        if anchor is None or anchor + k >= len(ar):
            continue
        seg = ar.iloc[anchor: anchor + k + 1].to_numpy()
        if np.all(np.isfinite(seg)):
            paths.append(np.cumsum(seg))
    if not paths:
        return pd.DataFrame(columns=["offset", "car", "t"]).set_index("offset")
    arr = np.vstack(paths)
    rows = []
    for j in range(arr.shape[1]):
        col = arr[:, j]
        s = one_sample_t(col)
        rows.append({"offset": j, "car": float(col.mean()), "t": s["t"]})
    return pd.DataFrame(rows).set_index("offset")


# --------------------------------------------------------------------------- #
# The tradable overlay — a pre-schedulable cicada-spring calendar trade
# --------------------------------------------------------------------------- #
def spring_timer(close: pd.Series, years, k: int = dt.WINDOW_K, cost_bps: float = 0.0
                 ) -> pd.DataFrame:
    """Long-only overlay: hold SPY over the emergence-spring window only.

    Entry at the last April close (``anchor-1``), exit at ``anchor+k``. Zero look-ahead by
    construction — the emergence year has been on the calendar since the last emergence
    13/17 years earlier, so entering the session before the window opens is fully
    foreseeable, not a peek. One round trip per year: one-way cost charged twice vs NAV.
    """
    ar = abnormal_returns(daily_returns(close))
    rows = []
    for y in years:
        w = spring_window(close, ar, y, k)
        if w is None:
            continue
        gross = w["raw_ret"]
        rows.append({"year": y, "entry_date": w["entry_date"], "exit_date": w["exit_date"],
                     "ret_gross": float(gross), "ret_net": float(gross - 2.0 * cost_bps * 1e-4)})
    return pd.DataFrame(rows)


def summarize_timer(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline stats for a timer ledger: n, win-rate, mean (bps), one-sample t."""
    if ledger.empty:
        return {"n": 0, "win_rate": float("nan"), "mean_bps": float("nan"), "t": float("nan")}
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    s = one_sample_t(r)
    return {"n": int(r.size), "win_rate": float((r > 0).mean()) if r.size else float("nan"),
            "mean_bps": s["mean"] * 1e4 if r.size else float("nan"), "t": s["t"]}


def unconditional_spring_baseline(close: pd.Series, pool_years, k: int = dt.WINDOW_K) -> float:
    """Mean spring window return (bps) across ALL pool years — the do-nothing-special
    benchmark the cicada overlay must beat to be an edge."""
    w = all_year_windows(close, pool_years, k)
    return float(w.mean() * 1e4) if len(w) else float("nan")


def excess_over_baseline(close: pd.Series, event_years, pool_years, k: int = dt.WINDOW_K,
                         cost_bps: float = 0.0) -> dict:
    """The HONEST tradability statistic: cicada-spring net return minus the every-spring
    baseline, one-sample t of the per-event excess.

    A raw one-sample t of the cicada spring return vs *zero* is dominated by SPY's
    ordinary two-month bull drift (beta) — it is positive for essentially any two-month
    equity window and says nothing about cicadas. The edge, if any, is the **excess over
    the unconditional every-spring baseline** (alpha): does holding SPY specifically in
    cicada springs beat holding it in an average spring? This nets one round trip of
    costs and subtracts the (constant) baseline before the t-test."""
    ar = abnormal_returns(daily_returns(close))
    ev = build_event_table(close, ar, event_years, k, cost_bps)
    base = float(all_year_windows(close, pool_years, k).mean())
    excess = ev["ret_net"].to_numpy() - base
    s = one_sample_t(excess)
    return {"n": s["n"], "excess_bps": s["mean"] * 1e4, "t": s["t"],
            "baseline_bps": base * 1e4}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(close: pd.Series, emergence_years, k: int = dt.WINDOW_K) -> dict:
    """Run the headline abnormal-CAR one-sample t on a synthetic world."""
    ar = abnormal_returns(daily_returns(close))
    ev = build_event_table(close, ar, emergence_years, k)
    return one_sample_t(ev["abn_car"].to_numpy()) if not ev.empty else {"n": 0, "t": float("nan")}
