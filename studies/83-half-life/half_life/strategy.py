"""The strategy and its honest controls — Study 83 (Half-Life).

The folk recipe: buy BTC right after each halving and ride the 'supply shock' rally for
the next 12–18 months. We implement this as a window-return study and pin it against
two fair controls:

- **Random-day control**: identical-length forward windows starting on random days
  (same calendar, same number of windows). Tests whether the halving dates *specifically*
  predict outsized returns, vs the secular BTC bull market doing all the work.
- **Secular-return model**: an OLS fit to the log-price trend over the full sample,
  then each halving window's excess return above the fitted trend. This separates
  'the halving caused this' from 'BTC was going up anyway during that era'.

The inference bar is brutally honest: with n=3 (or 4) event windows, HAC t-stats are
low-powered by construction and no claim can clear |t| >= 2. That is not a bug — it
is the study's main finding.

No look-ahead: the halving date is known at *t* (the block was mined); the trade enters
at *t+1*'s open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import HALVING_DATES


# ---------------------------------------------------------------------------
# Core event-window metrics
# ---------------------------------------------------------------------------
def window_returns(
    bars: pd.DataFrame,
    halvings: list[pd.Timestamp] | None = None,
    window_days: int = 365,
    col: str = "close",
) -> pd.DataFrame:
    """Forward log-return of BTC over ``window_days`` trading days after each halving.

    Entry is at the *next* available trading day's open after the halving date (no
    look-ahead). The log-return is computed on the ``close`` column from the entry bar
    forward to the bar ``window_days`` trading days later (or the tape end).

    Returns a DataFrame with one row per halving event that falls inside the tape's
    date range plus has a complete forward window.

    Columns: ``halving_date, entry_date, entry_px, exit_date, exit_px, log_ret,
              pct_ret, n_days_actual``.
    """
    if halvings is None:
        halvings = HALVING_DATES

    close = bars[col]
    rows = []
    for hd in halvings:
        # Find the first trading day strictly after the halving date
        after = bars.index[bars.index > hd]
        if len(after) == 0:
            continue
        entry_date = after[0]
        entry_pos = bars.index.get_loc(entry_date)
        exit_pos = min(entry_pos + window_days, len(bars) - 1)
        if exit_pos <= entry_pos:
            continue

        entry_px = bars["open"].iloc[entry_pos]
        exit_px = close.iloc[exit_pos]
        log_r = np.log(exit_px / entry_px)
        rows.append(
            {
                "halving_date": hd,
                "entry_date": entry_date,
                "entry_px": entry_px,
                "exit_date": bars.index[exit_pos],
                "exit_px": exit_px,
                "log_ret": log_r,
                "pct_ret": np.expm1(log_r),
                "n_days_actual": exit_pos - entry_pos,
            }
        )
    return pd.DataFrame(rows)


def random_control_returns(
    bars: pd.DataFrame,
    n_events: int,
    window_days: int = 365,
    seed: int = 83,
    n_draws: int = 500,
    col: str = "close",
) -> pd.DataFrame:
    """Monte-Carlo control: draw ``n_draws`` sets of ``n_events`` random starting
    days from the tape, each starting day at least ``window_days`` before the tape
    end, and compute the mean forward log-return for each set.

    This answers: *"how often does a random equally-sized bundle of BTC windows
    produce returns as good as the halving windows?"*

    Returns a Series of length ``n_draws`` where each element is the mean forward
    log-return for one random bundle.
    """
    rng = np.random.default_rng(seed)
    close = bars[col]
    # Eligible starting positions: must have a full window ahead
    eligible = np.arange(0, len(bars) - window_days)
    if len(eligible) < n_events:
        return pd.Series(dtype=float)

    means = []
    for _ in range(n_draws):
        idxs = rng.choice(eligible, size=n_events, replace=False)
        rets = [np.log(close.iloc[i + window_days] / bars["open"].iloc[i]) for i in idxs]
        means.append(float(np.mean(rets)))
    return pd.Series(means, name="random_mean_log_ret")


def secular_excess(
    bars: pd.DataFrame,
    halvings: list[pd.Timestamp] | None = None,
    window_days: int = 365,
) -> pd.DataFrame:
    """Compute each halving window's log-return *minus* the secular BTC trend return
    over the same calendar span.

    The secular trend is an OLS line fit to log(close) vs trading-day index over the
    full tape. For each halving window we compute what the OLS line predicts for the
    same duration and subtract it. Positive excess means the halving added a boost
    above the long-run trend; negative means it underperformed the trend.
    """
    lp = np.log(bars["close"].to_numpy(dtype=float))
    t = np.arange(len(lp), dtype=float)
    # OLS: lp = a + b*t
    b = np.cov(t, lp)[0, 1] / np.var(t)
    daily_trend = b  # log-return per trading day on the trend line

    events = window_returns(bars, halvings=halvings, window_days=window_days)
    events = events.copy()
    events["trend_log_ret"] = events["n_days_actual"] * daily_trend
    events["excess_log_ret"] = events["log_ret"] - events["trend_log_ret"]
    return events


# ---------------------------------------------------------------------------
# Per-period time series for the rolling-window look
# ---------------------------------------------------------------------------
def rolling_window_returns(
    bars: pd.DataFrame,
    window_days: int = 365,
    step_days: int = 30,
    col: str = "close",
) -> pd.DataFrame:
    """Forward log-return for every starting day (stepping ``step_days``).

    Used in the 'how does any window compare?' chart — the halving events are just 4
    points in this distribution.
    """
    close = bars[col]
    rows = []
    i = 0
    while i + window_days < len(bars):
        entry_px = bars["open"].iloc[i]
        exit_px = close.iloc[i + window_days]
        rows.append(
            {
                "start_date": bars.index[i],
                "log_ret": np.log(exit_px / entry_px),
            }
        )
        i += step_days
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trade-ledger summary — HAC t-stat on a tiny sample
# ---------------------------------------------------------------------------
def summarize(returns: np.ndarray | pd.Series, label: str = "") -> dict:
    """Headline statistics for an array of event returns.

    Returns n, mean, std, min, max, and a HAC t-stat (Newey-West). With n <= 4 the
    t-stat is explicitly unreliable; we compute it anyway and flag it.

    This is the inference-bar function: the 'Does the halving predict outsized returns?'
    question is answered by the t-stat — which, at n=3–4, will be nowhere near ±2.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    out = {
        "label": label,
        "n": int(n),
        "mean": float(r.mean()) if n else float("nan"),
        "std": float(r.std(ddof=1)) if n > 1 else float("nan"),
        "min": float(r.min()) if n else float("nan"),
        "max": float(r.max()) if n else float("nan"),
        "tstat": float("nan"),
        "low_power_warning": n < 10,
    }
    if n > 1:
        mu = r.mean()
        e = r - mu
        lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
        lrv = float(e @ e) / n
        for k in range(1, min(lags + 1, n)):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        out["tstat"] = float(mu / se) if se > 0 else float("nan")
    return out


def power_analysis(
    bars: pd.DataFrame,
    halvings: list[pd.Timestamp] | None = None,
    window_days: int = 365,
    seed: int = 83,
    n_draws: int = 2000,
) -> dict:
    """Bootstrap power: what fraction of random bundles of the same n beat the
    observed halving mean? Equivalently, the empirical p-value of the halving
    mean against the random-day null.

    Returns the p-value and the 90th / 95th percentile of the random-bundle means.
    """
    events = window_returns(bars, halvings=halvings, window_days=window_days)
    if events.empty:
        return {"p_value": float("nan"), "n_events": 0}
    obs_mean = events["log_ret"].mean()
    n_ev = len(events)
    ctrl = random_control_returns(bars, n_events=n_ev, window_days=window_days,
                                  seed=seed, n_draws=n_draws)
    p_value = float((ctrl >= obs_mean).mean())
    return {
        "obs_mean_log_ret": float(obs_mean),
        "n_events": int(n_ev),
        "p_value": float(p_value),
        "ctrl_p90": float(ctrl.quantile(0.90)),
        "ctrl_p95": float(ctrl.quantile(0.95)),
        "ctrl_mean": float(ctrl.mean()),
        "ctrl_std": float(ctrl.std()),
    }
