"""Strategy and analysis layer for Study 289 (Diwali-Muhurat).

The Muhurat-trading / Diwali omen: holding Indian equities across the Diwali
(Laxmi Pujan) Muhurat session is supposed to bring an auspicious year. We make
that tradeable for a non-Indian investor by buying the iShares MSCI India ETF
(INDA) on the first US session *after* each Diwali and holding it for a fixed
window, then pin the event-window return against:

  - The **unconditional daily-return baseline** (what INDA earns on an average
    day). The killer question: does the Diwali window beat a random window of the
    same length?
  - A **block-permutation test**: place the same number of equal-length windows
    at random start dates 10,000 times and see where the real mean lands.
  - A **Newey-West (HAC) t-test** on the per-event window returns vs the
    unconditional mean, with a lag matched to the holding period (overlap-robust).
  - **One-way costs × NAV** on entry and exit, with the (small) ETF spread, to
    show the net edge after the round-trip you actually pay.

The tiny-n reckoning: there are at most 23 Diwalis in the INDA era (and INDA only
exists from 2012, ~13 events). With daily vol ~1.3%, a 5-day window has ~3% noise,
so to clear |t| ≥ 2 on ~13 events the window would need a structural ~1.7% Diwali
premium. We check whether the observed premium is anywhere near that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Event-window construction
# ---------------------------------------------------------------------------
def event_windows(
    prices: pd.DataFrame,
    diwali: pd.DataFrame,
    hold_days: int = 5,
    lag: int = 1,
    col: str = "INDA",
) -> pd.DataFrame:
    """Compute the cumulative INDA return of the post-Diwali window for each year.

    For each Diwali date d, we find the first trading day on-or-after d, apply a
    ``lag`` of ``lag`` sessions (default 1 = buy at the *next* close, an honest
    one-session execution lag — the event is public so we do not trade on the
    event close itself), and hold ``hold_days`` sessions.

    Returns a DataFrame indexed by ``year`` with columns:
      ``entry_date``, ``exit_date``, ``window_ret`` (cumulative simple return of
      the window), ``n_sessions`` (actual sessions captured).
    Years whose window cannot be fully formed (event outside the price history)
    are dropped.
    """
    px = prices[col].dropna().sort_index()
    daily = px.pct_change()
    dates = px.index.to_numpy()
    n = len(px)

    rows = []
    for _, r in diwali.iterrows():
        d = np.datetime64(pd.Timestamp(r["diwali"]).to_datetime64())
        first = int(np.searchsorted(dates, d, side="left"))
        if first >= n:
            continue
        entry = first + lag  # buy at this close, hold hold_days from here
        exit_ = entry + hold_days
        if entry >= n:
            continue
        exit_ = min(exit_, n)
        # window return = product of daily returns over (entry+1 .. exit)
        seg = daily.iloc[entry + 1 : exit_ + 1] if entry + 1 <= n else daily.iloc[0:0]
        seg = seg.dropna()
        if len(seg) == 0:
            continue
        window_ret = float((1.0 + seg).prod() - 1.0)
        rows.append({
            "year": int(r["year"]),
            "entry_date": px.index[entry],
            "exit_date": px.index[min(exit_, n - 1)],
            "window_ret": window_ret,
            "n_sessions": int(len(seg)),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.set_index("year").sort_index()
    return out


def baseline_window_mean(prices: pd.DataFrame, hold_days: int = 5, col: str = "INDA") -> float:
    """Unconditional mean ``hold_days``-session forward return of INDA (the honest baseline)."""
    px = prices[col].dropna().sort_index()
    fwd = (px.shift(-hold_days) / px - 1.0).dropna()
    return float(fwd.mean())


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _nw_tstat(x: np.ndarray, lag: int) -> float:
    """Newey-West (HAC) t-stat of the mean of x against zero, with given lag."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float("nan")
    mu = x.mean()
    e = x - mu
    gamma0 = (e @ e) / n
    var = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1)
        cov = (e[k:] @ e[:-k]) / n
        var += 2.0 * w * cov
    se = np.sqrt(max(var, 1e-18) / n)
    return float(mu / se)


def event_stats(
    prices: pd.DataFrame,
    diwali: pd.DataFrame,
    hold_days: int = 5,
    lag: int = 1,
    n_permutations: int = 10_000,
    seed: int = 289,
    col: str = "INDA",
) -> dict:
    """Full event-study statistics for the post-Diwali INDA window.

    Compares each event's ``hold_days`` window return against the unconditional
    forward-window mean, with:
      - a plain t-test on the per-event excess window returns,
      - a Newey-West t-stat (lag = hold_days) on the excess returns,
      - a block-permutation test: draw ``n_events`` random window start dates,
        10,000 times, and record the fraction with mean window return >= observed.

    Returns a flat dict of rounded numbers (plus the raw permutation distribution
    under ``perm_means`` for plots).
    """
    ev = event_windows(prices, diwali, hold_days=hold_days, lag=lag, col=col)
    base = baseline_window_mean(prices, hold_days=hold_days, col=col)

    rets = ev["window_ret"].to_numpy(dtype=float)
    n_events = len(rets)
    mean_ev = float(rets.mean()) if n_events else float("nan")
    excess = rets - base

    # Plain t on excess.
    if n_events >= 2:
        sd = rets.std(ddof=1)
        t_plain = float(np.sqrt(n_events) * excess.mean() / (sd + 1e-18))
    else:
        t_plain = float("nan")

    # Newey-West t on excess (overlap within a window is mild; events are far apart).
    t_nw = _nw_tstat(excess, lag=hold_days) if n_events >= 2 else float("nan")

    hit_rate = float((rets > 0).mean()) if n_events else float("nan")

    # Block-permutation test on the forward-window series.
    px = prices[col].dropna().sort_index()
    fwd = (px.shift(-hold_days) / px - 1.0).dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    if len(fwd) > n_events and n_events > 0:
        perm_means = np.empty(n_permutations, dtype=float)
        for i in range(n_permutations):
            idx = rng.choice(len(fwd), size=n_events, replace=False)
            perm_means[i] = fwd[idx].mean()
        perm_p = float((perm_means >= mean_ev).mean())
    else:
        perm_means = np.array([])
        perm_p = float("nan")

    return {
        "n_events": int(n_events),
        "hold_days": int(hold_days),
        "lag": int(lag),
        "mean_event_ret": round(mean_ev, 6),
        "baseline_window_ret": round(float(base), 6),
        "excess_ret": round(float(mean_ev - base), 6),
        "hit_rate": round(hit_rate, 4),
        "t_plain": round(t_plain, 3),
        "t_nw": round(t_nw, 3),
        "perm_p": round(perm_p, 4),
        "perm_means": perm_means,
    }


# ---------------------------------------------------------------------------
# Net-of-cost edge
# ---------------------------------------------------------------------------
def net_edge(
    prices: pd.DataFrame,
    diwali: pd.DataFrame,
    hold_days: int = 5,
    lag: int = 1,
    cost_bps_one_way: float = 5.0,
    col: str = "INDA",
) -> dict:
    """Mean event-window return gross and net of a round-trip cost (entry + exit).

    Costs are one-way × NAV on both legs (a long-only ETF rotation: no borrow).
    ``cost_bps_one_way`` defaults to 5 bps — generous-but-realistic for a liquid
    US ETF like INDA. The round-trip drag is ``2 * cost_bps_one_way``.
    """
    ev = event_windows(prices, diwali, hold_days=hold_days, lag=lag, col=col)
    gross = float(ev["window_ret"].mean()) if len(ev) else float("nan")
    drag = 2.0 * cost_bps_one_way * 1e-4
    net = gross - drag
    base = baseline_window_mean(prices, hold_days=hold_days, col=col)
    return {
        "n_events": int(len(ev)),
        "gross_mean": round(gross, 6),
        "round_trip_cost": round(drag, 6),
        "net_mean": round(net, 6),
        "baseline_window_ret": round(float(base), 6),
        "net_excess": round(float(net - base), 6),
    }


# ---------------------------------------------------------------------------
# Summarize — compact R-dict for notebooks / results.md
# ---------------------------------------------------------------------------
def summarize(
    prices: pd.DataFrame,
    diwali: pd.DataFrame,
    hold_days: int = 5,
    lag: int = 1,
    n_permutations: int = 10_000,
    seed: int = 289,
    col: str = "INDA",
) -> dict:
    """One-stop summary dict for a prices frame + the Diwali table.

    Mirrors the R = dict(...) in build_notebooks.py and docs/results.md. All
    returns are reported in percent for the headline window.
    """
    st = event_stats(prices, diwali, hold_days=hold_days, lag=lag,
                     n_permutations=n_permutations, seed=seed, col=col)
    ne = net_edge(prices, diwali, hold_days=hold_days, lag=lag, col=col)
    return {
        "n_events": st["n_events"],
        "hold_days": hold_days,
        "mean_event_pct": round(st["mean_event_ret"] * 100, 3),
        "baseline_pct": round(st["baseline_window_ret"] * 100, 3),
        "excess_pct": round(st["excess_ret"] * 100, 3),
        "hit_rate_pct": round(st["hit_rate"] * 100, 1),
        "t_plain": st["t_plain"],
        "t_nw": st["t_nw"],
        "perm_p": st["perm_p"],
        "gross_pct": round(ne["gross_mean"] * 100, 3),
        "net_pct": round(ne["net_mean"] * 100, 3),
        "net_excess_pct": round(ne["net_excess"] * 100, 3),
    }
