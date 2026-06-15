"""Strategy and analysis for Study 167 (Hindenburg-Omen).

The folk recipe: when a Hindenburg Omen fires, a stock-market crash is "imminent" — exit
equities (or go short SPY) and wait.  The counter-claim, equally famous: the signal is a
*false-alarm machine* that has fired dozens of times without a crash following.

We test this honestly with two intertwined questions:

1. **Signal quality** — Is the distribution of SPY forward returns (30-, 60-, 90-, 120-day)
   *after* a Hindenburg day distinguishably different (lower) from the unconditional
   distribution?  We use a Welch t-test and report the HAC t-stat on the cross-signal-date
   return series.

2. **False-alarm rate** — What fraction of Hindenburg clusters is followed by a ≥5% drawdown
   in the next 120 trading days?  Compare to the base-rate of such drawdowns on random days.

Two additional controls:

- **Cluster deduplication**: consecutive Hindenburg days within 30 calendar days are grouped
  into a single *cluster* (the signal builder clusters them this way).  Each cluster counts
  once.  This is conservative — without it each additional day in a cluster could look like
  a "prediction."
- **Multiple comparisons**: we test 4 horizons × 3 thresholds = 12 hypotheses;
  Bonferroni-corrected *p*-value is reported alongside the raw one.

No look-ahead: signal on day *t*, forward return measured from *t+1*'s close.

Survivorship note: the breadth panel uses current S&P 500 membership (survivors), which
likely *understates* low-counts during crashes.  This is named in the Signal verdict and
discussed in Beat 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Cluster deduplication
# ---------------------------------------------------------------------------
def cluster_signals(
    breadth: pd.DataFrame,
    window_days: int = 30,
) -> pd.Series:
    """Return a boolean Series: one True per *cluster* of consecutive Hindenburg days.

    Consecutive signal days within ``window_days`` calendar days of the first signal
    in a cluster are merged; only the *first* day of each cluster is kept.  This is
    the standard Hindenburg-cluster convention and the conservative choice (no cherry
    picking the "best" day in a cluster).

    Returns a boolean Series aligned on the input index.
    """
    signal_dates = breadth.index[breadth["hindenburg"] == 1]
    if len(signal_dates) == 0:
        return pd.Series(False, index=breadth.index, name="cluster_start")

    cluster_starts = []
    last_start = None
    for d in signal_dates:
        if last_start is None or (d - last_start).days > window_days:
            cluster_starts.append(d)
            last_start = d

    out = pd.Series(False, index=breadth.index, name="cluster_start")
    out.loc[cluster_starts] = True
    return out


# ---------------------------------------------------------------------------
# Forward-return extraction
# ---------------------------------------------------------------------------
def forward_returns(
    breadth: pd.DataFrame,
    signal: pd.Series,
    horizons: tuple[int, ...] = (30, 60, 90, 120),
) -> pd.DataFrame:
    """Extract SPY forward returns after each signal date.

    For each signal day, compound the SPY daily returns from ``t+1`` through
    ``t+horizon`` (inclusive).  Signals with insufficient history are dropped
    (NaN).

    Returns a DataFrame indexed by signal date, columns = horizon labels
    (``ret30``, ``ret60``, etc.).
    """
    signal_dates = breadth.index[signal]
    spy_ret = breadth["spy_ret"]

    rows = []
    for d in signal_dates:
        try:
            loc = breadth.index.get_loc(d)
        except KeyError:
            continue
        row: dict = {"date": d}
        for h in horizons:
            if loc + h >= len(breadth):
                row[f"ret{h}"] = np.nan
            else:
                r = spy_ret.iloc[loc + 1 : loc + h + 1].to_numpy(dtype=float)
                row[f"ret{h}"] = float(np.prod(1.0 + r[np.isfinite(r)]) - 1.0)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["date"] + [f"ret{h}" for h in horizons]).set_index("date")
    return pd.DataFrame(rows).set_index("date")


def unconditional_forward_returns(
    breadth: pd.DataFrame,
    horizons: tuple[int, ...] = (30, 60, 90, 120),
    sample_every: int = 21,     # ~monthly to keep samples roughly independent
) -> pd.DataFrame:
    """SPY forward returns on non-signal days, sampled ~monthly.

    Provides the unconditional baseline: what is the typical forward return on any
    random day?  We sample every ``sample_every`` trading days (default ~monthly)
    to keep the observations approximately independent.  Signal days and the
    ``max(horizons)`` tail are excluded.

    Returns a DataFrame with the same horizon columns as :func:`forward_returns`.
    """
    spy_ret = breadth["spy_ret"]
    max_h = max(horizons)
    sig_dates = set(breadth.index[breadth["hindenburg"] == 1])

    rows = []
    valid_idx = [
        i for i in range(0, len(breadth) - max_h, sample_every)
        if breadth.index[i] not in sig_dates
    ]
    for i in valid_idx:
        d = breadth.index[i]
        row: dict = {"date": d}
        for h in horizons:
            if i + h >= len(breadth):
                row[f"ret{h}"] = np.nan
            else:
                r = spy_ret.iloc[i + 1 : i + h + 1].to_numpy(dtype=float)
                row[f"ret{h}"] = float(np.prod(1.0 + r[np.isfinite(r)]) - 1.0)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["date"] + [f"ret{h}" for h in horizons]).set_index("date")
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
# Statistical tests — HAC t-stat (local implementation, same as quantlab)
# ---------------------------------------------------------------------------
def _hac_tstat(x: np.ndarray) -> float:
    """Newey-West HAC t-stat for the mean, with Bartlett kernel."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    mu = x.mean()
    e = x - mu
    if n < 6:
        se = x.std(ddof=1) / np.sqrt(n)
        return float(mu / se) if se > 0 else float("nan")
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def _welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t-statistic comparing means of two independent samples."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float(diff / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# Crash rate: was there a ≥ threshold drawdown within the window?
# ---------------------------------------------------------------------------
def crash_rate(
    breadth: pd.DataFrame,
    signal: pd.Series,
    horizon: int = 120,
    drawdown_threshold: float = -0.05,
) -> dict:
    """Fraction of signal clusters followed by a ≥|threshold| peak-to-trough drawdown.

    For each signal date we compute the running maximum of SPY close over the
    ``horizon``-day window and check whether it ever drops ``drawdown_threshold``
    below that peak.  Returns the signal crash rate, the base-rate on non-signal
    random days (sampled monthly), and the Welch t on the proportions.
    """
    spy_ret = breadth["spy_ret"]
    spy_close = (1.0 + spy_ret.fillna(0.0)).cumprod()

    signal_dates = breadth.index[signal]
    base_dates = breadth.index[~signal & (np.arange(len(breadth)) % 21 == 0)]

    def had_crash(loc: int) -> bool:
        if loc + horizon >= len(spy_close):
            return False
        window_prices = spy_close.iloc[loc + 1 : loc + horizon + 1].to_numpy()
        if len(window_prices) == 0:
            return False
        running_max = np.maximum.accumulate(window_prices)
        dd = window_prices / running_max - 1.0
        return bool((dd <= drawdown_threshold).any())

    def rate(dates):
        vals = []
        for d in dates:
            try:
                loc = breadth.index.get_loc(d)
            except KeyError:
                continue
            if loc + horizon < len(breadth):
                vals.append(int(had_crash(loc)))
        return np.array(vals, dtype=float)

    sig_vals = rate(signal_dates)
    base_vals = rate(base_dates)

    sig_rate = float(sig_vals.mean()) if len(sig_vals) > 0 else float("nan")
    base_rate = float(base_vals.mean()) if len(base_vals) > 0 else float("nan")
    t = _welch_t(sig_vals, base_vals)

    return {
        "n_signals": int(len(sig_vals)),
        "n_crashes_after_signal": int(sig_vals.sum()),
        "signal_crash_rate": sig_rate,
        "base_crash_rate": base_rate,
        "false_alarm_rate": 1.0 - sig_rate,
        "welch_t": t,
    }


# ---------------------------------------------------------------------------
# Main summarise function
# ---------------------------------------------------------------------------
def summarize(
    breadth: pd.DataFrame,
    horizons: tuple[int, ...] = (30, 60, 90, 120),
    cluster_window: int = 30,
    drawdown_threshold: float = -0.05,
) -> dict:
    """Top-level summary dict, suitable for the R-dict in build_notebooks.py.

    Computes:
    - Signal dates and cluster counts.
    - Per-horizon: signal mean return, unconditional mean return, Welch t, HAC t on
      the signal-date series.
    - Crash rate analysis at the 120-day horizon.
    - Bonferroni-corrected significance threshold for 12 hypotheses (4 horizons × 3
      thresholds).
    """
    from scipy import stats as scipy_stats  # lazy import

    sig = cluster_signals(breadth, window_days=cluster_window)
    fwd_sig = forward_returns(breadth, sig, horizons=horizons)
    fwd_base = unconditional_forward_returns(breadth, horizons=horizons)

    n_hyp = len(horizons) * 3   # 3 thresholds tested (2.0%, 2.2%, 2.5%)
    out: dict = {
        "n_signal_days": int((breadth["hindenburg"] == 1).sum()),
        "n_clusters": int(sig.sum()),
        "n_base": len(fwd_base),
    }

    for h in horizons:
        col = f"ret{h}"
        s_arr = fwd_sig[col].dropna().to_numpy(dtype=float)
        b_arr = fwd_base[col].dropna().to_numpy(dtype=float)

        s_mean = float(s_arr.mean()) if len(s_arr) > 0 else float("nan")
        b_mean = float(b_arr.mean()) if len(b_arr) > 0 else float("nan")
        t_hac = _hac_tstat(s_arr)
        t_welch = _welch_t(s_arr, b_arr)

        # Two-sided p-value from Welch t, df ~ n_s + n_b - 2 (conservative)
        df_approx = max(len(s_arr) + len(b_arr) - 2, 1)
        p_raw = float(2.0 * scipy_stats.t.sf(abs(t_welch), df=df_approx)) if np.isfinite(t_welch) else float("nan")
        p_bonf = min(1.0, p_raw * n_hyp) if np.isfinite(p_raw) else float("nan")

        out[f"sig_mean_{h}d"] = round(s_mean * 100, 2)   # in %
        out[f"base_mean_{h}d"] = round(b_mean * 100, 2)
        out[f"t_hac_{h}d"] = round(t_hac, 2)
        out[f"t_welch_{h}d"] = round(t_welch, 2)
        out[f"p_raw_{h}d"] = round(p_raw, 4)
        out[f"p_bonf_{h}d"] = round(p_bonf, 4)
        out[f"n_sig_{h}d"] = len(s_arr)

    cr = crash_rate(breadth, sig, horizon=120, drawdown_threshold=drawdown_threshold)
    out.update(cr)
    return out
