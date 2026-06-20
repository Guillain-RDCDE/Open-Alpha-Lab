"""The event-study engine and its honest controls — Study 326 (Stablecoin-Depeg).

The thesis under test: a stablecoin **breaking the buck** is a crypto-wide *sell* signal —
when UST or USDC depegs, BTC/ETH should fall (and keep falling), so getting out at the
depeg should beat staying long.

We answer it two ways:

1. **Event study.** For each depeg date, measure the basket's cumulative abnormal return
   (CAR) over a window ``[-pre, +post]`` trading days around the event, where "abnormal" is
   relative to the basket's own mean daily return outside event windows (a constant-mean
   model — the standard, assumption-light event-study benchmark; crypto has no clean factor
   model). We summarise the average CAR across events with a HAC/Newey-West *t* and a
   **circular block bootstrap** CI. The decisive honesty point: the *real* universe of
   market-wide depegs is **n = 2** (UST, USDC). No average over two events clears a *t* ≥ 2
   bar — so the real tape can never stamp Signal REAL here, by construction.

2. **Synthetic control.** Draw many random non-event windows from the same tape and build
   the null distribution of the CAR. A real depeg signal must sit in that distribution's
   tail; if the two real depeg CARs are not unusual against randomly-placed windows, "depeg
   = sell signal" is indistinguishable from "any random bad week."

3. **A tradable rule (Beat 6).** "On a depeg, sell the basket and sit in cash for ``hold``
   days, then buy back." One **documented execution lag**: the depeg is only confirmed
   intraday on day t, so we act at the **close of day t+1** (enter cash for the return of
   t+2 onward). Costs are charged **one-way × NAV** on each switch (out and back = two
   one-way legs). We race the rule's excess return *of cash* against the basket's excess
   *of cash* — but with only two events the "edge" is a two-trade anecdote, not a strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 365  # crypto trades every day; Yahoo daily bars are calendar days


# ---------------------------------------------------------------------------
# Newey-West (HAC) t-stat on a mean — no hard dependency on quantlab
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """HAC (Newey-West) t-stat that the mean of ``x`` differs from zero.

    Falls back to NaN when there are too few observations to estimate it (e.g. the
    two-event real sample) — the whole point being that n = 2 cannot certify anything.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return float("nan")
    mu = x.mean()
    e = x - mu
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(lags, n - 1))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# Event-window extraction
# ---------------------------------------------------------------------------
def _event_locs(index: pd.DatetimeIndex, events: pd.DatetimeIndex) -> list[int]:
    """Integer positions of each event date in ``index`` (nearest on/after the date).

    A depeg date may fall on a non-trading bar for some series; we snap to the first
    bar on or after it. Events with no bar on/after them are dropped.
    """
    locs = []
    arr = index.to_numpy()
    for ev in events:
        pos = np.searchsorted(arr, np.datetime64(ev))
        if pos < len(index):
            locs.append(int(pos))
    return locs


def event_windows(
    bars: pd.DataFrame,
    events: pd.DatetimeIndex,
    pre: int = 5,
    post: int = 10,
) -> pd.DataFrame:
    """Aligned daily returns in ``[-pre, +post]`` trading days around each event.

    Returns a DataFrame indexed by relative day (``-pre … +post``) with one column per
    event (the basket's simple daily ``ret``). Day 0 is the event bar. Windows that run
    off either end of the tape are dropped.
    """
    ret = bars["ret"].to_numpy(dtype=float)
    locs = _event_locs(bars.index, events)
    cols = {}
    rel = np.arange(-pre, post + 1)
    for i, loc in enumerate(locs):
        lo, hi = loc - pre, loc + post
        if lo < 0 or hi >= len(ret):
            continue
        cols[f"ev{i}"] = ret[lo : hi + 1]
    return pd.DataFrame(cols, index=pd.Index(rel, name="rel_day"))


def car(window_returns: pd.Series | np.ndarray, from_day: int, pre: int) -> float:
    """Cumulative return of one event window from relative day ``from_day`` to the end.

    ``from_day = 0`` includes the event day; ``from_day = 1`` is the *forward* CAR (the
    tradable part — the move you could still catch after the depeg is public).
    """
    w = np.asarray(window_returns, dtype=float)
    start = pre + from_day
    seg = w[start:]
    seg = seg[np.isfinite(seg)]
    return float(np.prod(1.0 + seg) - 1.0)


def event_cars(
    windows: pd.DataFrame, from_day: int = 0, pre: int = 5
) -> np.ndarray:
    """The cumulative abnormal return of each event window (one value per event)."""
    return np.array(
        [car(windows[c].to_numpy(), from_day=from_day, pre=pre) for c in windows.columns]
    )


# ---------------------------------------------------------------------------
# Synthetic control — the null distribution of random non-event windows
# ---------------------------------------------------------------------------
def null_car_distribution(
    bars: pd.DataFrame,
    events: pd.DatetimeIndex,
    pre: int = 5,
    post: int = 10,
    from_day: int = 0,
    n_draws: int = 5000,
    exclude_radius: int = 20,
    group_size: int | None = None,
    seed: int = 326,
) -> np.ndarray:
    """Null distribution of the **mean-of-``group_size``-windows** CAR.

    Draws ``n_draws`` groups of ``group_size`` random non-event window centres (default
    ``group_size = len(events)`` — match the apples-to-apples count) and returns each
    group's *mean* CAR. Centres within ``exclude_radius`` of a real event are banned, so the
    null is genuinely "an ordinary set of windows." The observed mean CAR across the real
    depeg events is read against this distribution: a real signal sits in its left tail.

    Comparing the mean-of-n against single-window CARs would be unfair (a single 16-day
    crypto window is wildly noisy); averaging ``group_size`` of them is the like-for-like
    null.
    """
    rng = np.random.default_rng(seed)
    ret = bars["ret"].to_numpy(dtype=float)
    n = len(ret)
    if group_size is None:
        group_size = max(1, len(events))
    locs = set(_event_locs(bars.index, events))
    banned = set()
    for loc in locs:
        for k in range(-exclude_radius, exclude_radius + 1):
            banned.add(loc + k)

    valid_centres = np.array([c for c in range(pre, n - post) if c not in banned])
    if valid_centres.size == 0:
        return np.array([])

    out = np.empty(n_draws)
    for j in range(n_draws):
        centres = rng.choice(valid_centres, size=group_size, replace=True)
        cars = [car(ret[c - pre : c + post + 1], from_day=from_day, pre=pre) for c in centres]
        out[j] = float(np.mean(cars))
    return out


def empirical_pvalue(observed_mean: float, null: np.ndarray) -> float:
    """One-sided left-tail p-value: P(null CAR <= observed mean) (depeg = sell ⇒ left)."""
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(observed_mean):
        return float("nan")
    return float((null <= observed_mean).mean())


# ---------------------------------------------------------------------------
# Circular block bootstrap CI for the mean CAR across events
# ---------------------------------------------------------------------------
def block_bootstrap_ci(
    x: np.ndarray,
    block: int = 1,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 326,
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the mean of ``x``.

    With the two-event real sample ``block`` defaults to 1 (events are independent shocks)
    and the CI is necessarily enormous — that width *is* the finding.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[b] = x[idx][:n].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)


# ---------------------------------------------------------------------------
# Headline event-study summary
# ---------------------------------------------------------------------------
def summarize_event_study(
    bars: pd.DataFrame,
    events: pd.DatetimeIndex,
    pre: int = 5,
    post: int = 10,
    from_day: int = 0,
    n_draws: int = 5000,
    seed: int = 326,
) -> dict:
    """One dict of headline numbers for the event study.

    Keys: ``n_events``, ``mean_car``, ``median_car``, ``tstat`` (HAC; NaN when n<3),
    ``ci_lo`` / ``ci_hi`` (block bootstrap), ``p_value`` (vs synthetic-control null),
    and the per-event CARs (``cars``).
    """
    windows = event_windows(bars, events, pre=pre, post=post)
    cars = event_cars(windows, from_day=from_day, pre=pre)
    null = null_car_distribution(
        bars, events, pre=pre, post=post, from_day=from_day, n_draws=n_draws, seed=seed
    )
    mean_car = float(np.mean(cars)) if cars.size else float("nan")
    lo, hi = block_bootstrap_ci(cars, seed=seed)
    return {
        "n_events": int(cars.size),
        "mean_car": mean_car,
        "median_car": float(np.median(cars)) if cars.size else float("nan"),
        "tstat": hac_tstat(cars),
        "ci_lo": lo,
        "ci_hi": hi,
        "p_value": empirical_pvalue(mean_car, null),
        "cars": cars,
        "windows": windows,
    }


# ---------------------------------------------------------------------------
# The tradable "sell on the depeg" rule (Beat 6)
# ---------------------------------------------------------------------------
def sell_on_depeg(
    bars: pd.DataFrame,
    events: pd.DatetimeIndex,
    hold: int = 10,
    lag: int = 1,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Long the basket always, but go to **cash** for ``hold`` days after each depeg.

    Execution lag: the depeg is confirmed intraday on day t, so we act at the **close of
    day t+``lag``** — the position goes flat for the returns of days t+lag+1 … t+lag+hold,
    then re-enters. Costs are one-way × NAV, charged on each switch (exit + re-entry = two
    one-way legs of ``cost_bps``).

    Returns a daily frame with ``ret`` (basket), ``pos`` (1=long, 0=cash), ``strat_ret``
    (rule net return) and ``cash_ret`` (=0; the cash leg earns nothing here — a
    conservative, single-asset comparison so the race is excess-of-cash vs excess-of-cash).
    """
    ret = bars["ret"].fillna(0.0).to_numpy(dtype=float)
    n = len(ret)
    pos = np.ones(n)
    locs = _event_locs(bars.index, events)
    for loc in locs:
        start = loc + lag + 1  # first day whose return we sit out
        end = min(start + hold, n)
        pos[start:end] = 0.0

    # Turnover: a switch happens whenever pos changes day-to-day; charge one-way each.
    switch = np.abs(np.diff(np.concatenate([[1.0], pos])))  # start from long
    cost = switch * (cost_bps * 1e-4)
    strat_ret = pos * ret - cost

    out = pd.DataFrame(
        {"ret": ret, "pos": pos, "strat_ret": strat_ret, "cash_ret": np.zeros(n)},
        index=bars.index,
    )
    return out


def equity_stats(daily_ret: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """Annualised return, vol and Sharpe (excess-of-zero = excess-of-cash) for a series."""
    r = np.asarray(daily_ret, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std() == 0:
        return {"ann_ret": float("nan"), "ann_vol": float("nan"), "sharpe": float("nan")}
    mu, sd = r.mean(), r.std(ddof=1)
    return {
        "ann_ret": float(mu * periods_per_year),
        "ann_vol": float(sd * np.sqrt(periods_per_year)),
        "sharpe": float(mu / sd * np.sqrt(periods_per_year)),
    }
