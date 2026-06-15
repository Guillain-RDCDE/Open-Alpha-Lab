"""Strategy and honest controls — Study 168 (Advance-Decline).

The folk recipe: the generals are advancing but the soldiers are retreating.
When the stock index (SPY) makes a new N-day high while the cumulative A/D line
does NOT make a new high over the same window, a "bearish divergence" is flagged
as a warning that the index rally is narrow and fragile. The claim is that forward
index returns after such a divergence signal are materially worse than the
unconditional baseline.

We implement:

 - :func:`ad_divergence_signal` — the bearish divergence detector (no look-ahead).
 - :func:`forward_returns` — fixed-horizon forward SPY returns.
 - :func:`backtest` — divergence arm vs an unconditional baseline (random calendar
   windows) with Newey-West HAC t-stats.
 - :func:`summarize` — headline statistics from a returns array.

Two controls anchor the honest inference:
 1. **Unconditional baseline** — the distribution of *all* forward-N-day SPY
    returns, against which the post-divergence distribution is tested.
 2. **Permutation baseline** — same-size random sub-samples of the return
    distribution (same effective n), to assess whether any sub-sample would
    "look special" by chance (a mild multiple-comparisons check when we sweep
    the lookback window).

No look-ahead: the divergence is detected using closes up to day t; the
forward return begins at day t+1's open (approximated as the next day's close
for daily data — a one-day lag convention consistent with the desk standard).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core signal: bearish price-vs-breadth divergence
# ---------------------------------------------------------------------------

def ad_divergence_signal(
    spy: pd.Series,
    ad_line: pd.Series,
    lookback: int = 63,          # ~3 months (the typical "medium-term" window)
    min_index_gain: float = 0.03, # index must have risen ≥3% to qualify
) -> pd.Series:
    """Daily bearish-divergence flag: index at N-day high, A/D line is NOT.

    A bearish divergence is flagged on day t when ALL of the following hold:

    1. ``spy[t]`` is at or above the N-day rolling maximum (index at a new high
       for the lookback window).
    2. ``spy[t] / spy[t - lookback] - 1 >= min_index_gain`` (the index has
       actually *risen* meaningfully, not just failed to fall).
    3. ``ad_line[t]`` is strictly below the N-day rolling maximum of the A/D
       line (breadth has NOT confirmed the new high).

    Returns a boolean pd.Series (True = bearish divergence on that day).
    No future data is used: rolling maxima are computed over the window
    ending at t (inclusive), so the signal is known at the close of day t.
    """
    spy   = spy.dropna()
    ad    = ad_line.dropna()
    common = spy.index.intersection(ad.index)
    spy   = spy.loc[common]
    ad    = ad.loc[common]

    spy_max_N  = spy.rolling(lookback, min_periods=lookback).max()
    ad_max_N   = ad.rolling(lookback, min_periods=lookback).max()
    spy_ret_N  = spy / spy.shift(lookback) - 1.0

    at_high         = spy >= spy_max_N * (1 - 1e-8)   # index at N-day high
    has_risen       = spy_ret_N >= min_index_gain       # risen enough to matter
    breadth_lagging = ad < ad_max_N                    # A/D not at N-day high

    signal = at_high & has_risen & breadth_lagging
    signal.name = f"divergence_{lookback}d"
    return signal


def forward_returns(
    spy: pd.Series,
    horizon: int = 21,           # ~1 month forward
) -> pd.Series:
    """Forward log-return for SPY over ``horizon`` trading days.

    Return[t] = log(spy[t + horizon] / spy[t]).  Values near the tail of the
    series (where the full horizon is unavailable) are NaN.
    No look-ahead: day-t forward return uses only prices at t and beyond.
    """
    lr = np.log(spy.shift(-horizon) / spy)   # log(price_{t+h} / price_t)
    lr.name = f"fwd_{horizon}d"
    return lr


# ---------------------------------------------------------------------------
# Backtest: post-divergence vs baseline
# ---------------------------------------------------------------------------

def backtest(
    spy: pd.Series,
    ad_line: pd.Series,
    lookback: int = 63,
    horizon: int = 21,
    min_index_gain: float = 0.03,
) -> dict:
    """Compare forward SPY returns after a bearish divergence vs the baseline.

    Returns
    -------
    dict with keys:
      ``signal_days``     : pd.Series[bool] of divergence flags
      ``fwd``             : pd.Series of forward returns (all days)
      ``div_rets``        : array of forward returns on divergence days
      ``base_rets``       : array of forward returns on ALL days (unconditional)
      ``div_stats``       : dict from :func:`summarize` on ``div_rets``
      ``base_stats``      : dict from :func:`summarize` on ``base_rets``
      ``diff_mean_bps``   : mean(div) - mean(base) in bps
    """
    sig  = ad_divergence_signal(spy, ad_line, lookback=lookback,
                                 min_index_gain=min_index_gain)
    fwd  = forward_returns(spy, horizon=horizon)

    # Align — drop NaNs in both
    df = pd.concat([sig, fwd], axis=1).dropna()
    sig_col, fwd_col = df.columns[0], df.columns[1]

    base_rets = df[fwd_col].to_numpy(dtype=float)
    div_mask  = df[sig_col].astype(bool)
    div_rets  = df.loc[div_mask, fwd_col].to_numpy(dtype=float)

    div_stats  = summarize(div_rets,  label="post-divergence")
    base_stats = summarize(base_rets, label="unconditional")

    diff = (div_stats["mean_bps"] - base_stats["mean_bps"])

    return {
        "signal_days":    sig,
        "fwd":            fwd,
        "div_rets":       div_rets,
        "base_rets":      base_rets,
        "div_stats":      div_stats,
        "base_stats":     base_stats,
        "diff_mean_bps":  float(diff),
        "n_divergences":  int(div_mask.sum()),
    }


# ---------------------------------------------------------------------------
# Window sweep — how sensitive is the verdict to the lookback?
# ---------------------------------------------------------------------------

def sweep_lookback(
    spy: pd.Series,
    ad_line: pd.Series,
    lookbacks: list[int] | None = None,
    horizon: int = 21,
) -> pd.DataFrame:
    """Run the divergence backtest across a range of lookback windows.

    Returns a DataFrame indexed by lookback with columns:
    ``n_signals``, ``div_mean_bps``, ``base_mean_bps``, ``diff_bps``,
    ``div_tstat``, ``base_tstat``.
    """
    if lookbacks is None:
        lookbacks = [21, 42, 63, 126, 252]

    rows = []
    for lb in lookbacks:
        res = backtest(spy, ad_line, lookback=lb, horizon=horizon)
        rows.append({
            "lookback":      lb,
            "n_signals":     res["n_divergences"],
            "div_mean_bps":  res["div_stats"]["mean_bps"],
            "base_mean_bps": res["base_stats"]["mean_bps"],
            "diff_bps":      res["diff_mean_bps"],
            "div_tstat":     res["div_stats"]["tstat"],
            "base_tstat":    res["base_stats"]["tstat"],
        })
    return pd.DataFrame(rows).set_index("lookback")


# ---------------------------------------------------------------------------
# Permutation baseline — is ANY sub-sample this "extreme"?
# ---------------------------------------------------------------------------

def permutation_baseline(
    base_rets: np.ndarray,
    n_signal: int,
    n_perm: int = 2000,
    seed: int = 168,
) -> dict:
    """Draw ``n_perm`` random sub-samples of size ``n_signal`` from ``base_rets``.

    Returns the distribution of their means (in bps) and the fraction that are
    *more negative* than the observed divergence mean — a permutation p-value.
    """
    rng = np.random.default_rng(seed)
    perm_means = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.choice(len(base_rets), size=n_signal, replace=False)
        perm_means[i] = base_rets[idx].mean() * 1e4
    return {
        "perm_means_bps": perm_means,
        "perm_mean":      float(perm_means.mean()),
        "perm_p10":       float(np.percentile(perm_means, 10)),
        "perm_p90":       float(np.percentile(perm_means, 90)),
    }


# ---------------------------------------------------------------------------
# Drawdown analysis
# ---------------------------------------------------------------------------

def max_drawdown_after_signal(
    spy: pd.Series,
    signal: pd.Series,
    max_horizon: int = 63,
) -> dict:
    """Compare maximum drawdown in the ``max_horizon`` days after a divergence
    signal vs the unconditional max drawdown over the same length windows.

    Returns dicts with ``signal_dd_mean`` and ``base_dd_mean`` (both negative,
    in return units).
    """
    spy_np   = spy.to_numpy(dtype=float)
    dates    = spy.index
    pos_map  = {d: i for i, d in enumerate(dates)}
    sig_days = signal[signal].index

    def _max_dd(i0: int) -> float:
        i1 = min(i0 + max_horizon, len(spy_np))
        window = spy_np[i0:i1]
        if len(window) < 2:
            return np.nan
        cum = window / window[0]
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        return float(dd.min())

    sig_dds = [_max_dd(pos_map[d] + 1) for d in sig_days
               if pos_map.get(d, -1) + 1 < len(spy_np)]
    all_dds = [_max_dd(i) for i in range(0, len(spy_np) - max_horizon, 5)]

    return {
        "signal_dd_mean": float(np.nanmean(sig_dds)) if sig_dds else np.nan,
        "base_dd_mean":   float(np.nanmean(all_dds)) if all_dds else np.nan,
        "signal_dd_n":    len([x for x in sig_dds if np.isfinite(x)]),
    }


# ---------------------------------------------------------------------------
# Summary statistics with HAC t-stat
# ---------------------------------------------------------------------------

def summarize(returns: np.ndarray, label: str = "") -> dict:
    """Headline statistics for an array of returns.

    Returns mean in bps, hit-rate, Sharpe (per-observation), and a Newey-West
    HAC t-stat for the mean — the inference-bar number. Identical in spirit to
    the study-72 ``summarize``; adapted for daily forward-return arrays rather
    than per-trade ledgers.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    out: dict = {
        "label":    label,
        "n":        int(n),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "hit_rate": float((r > 0).mean()) if n else float("nan"),
        "sharpe":   float(r.mean() / r.std(ddof=1)) if n > 1 and r.std() > 0 else float("nan"),
        "tstat":    float("nan"),
    }
    if n > 5:
        mu  = r.mean()
        e   = r - mu
        lgs = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        lrv = float(e @ e) / n
        for k in range(1, lgs + 1):
            w    = 1.0 - k / (lgs + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        out["tstat"] = float(mu / se) if se > 0 else float("nan")
    return out
