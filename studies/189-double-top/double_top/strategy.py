"""Pattern detection and honest controls — Study 189 (Double-Top / Double-Bottom).

The double-top (M) is two price peaks at a similar level, separated by a trough;
the double-bottom (W) is the mirror.  The *confirmation* signal is:

- **Double-top confirmed** at bar *t* when the price breaks below the intervening
  trough (the "neckline").  The claimed direction is *bearish* (−1): sell short,
  expecting price to fall further.
- **Double-bottom confirmed** at bar *t* when the price breaks above the
  intervening peak (the "neckline").  The claimed direction is *bullish* (+1):
  buy, expecting price to rise further.

Detection algorithm (see :func:`detect_double_patterns`):

1. Find local peaks and troughs in the close series using ``scipy.signal.find_peaks``
   with a minimum distance between peaks (``min_bars``) and a minimum prominence
   (``prominence_frac`` of the trailing 20-day ATR).
2. For each pair of consecutive peaks that are "similar" in height
   (within ``tolerance`` of each other), identify the trough between them and
   record the pattern when the close breaks *below* that trough.
3. Mirror logic for double-bottom (consecutive troughs, break above the peak).

The honest control is a **random-day placebo**: for each confirmed signal, draw a
random bar from the same tape (with no look-ahead exclusion), assign the same
claimed direction, and measure the same forward return.  The pattern adds value only
when its mean forward return *exceeds* the unconditional baseline; a Newey-West HAC
t-stat on the *excess* is the decisive number.

Multiple comparisons: two pattern types × two horizons (1-day, 5-day, 20-day) = 6
t-stats.  At α = 5% we expect 0.3 spurious signals; we apply a Bonferroni correction
(threshold |t| ≥ 3.0 for six tests) and note what survives.

No look-ahead: patterns are detected using only closes up to and including bar *t*;
all forward returns begin at bar *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Helpers — peak/trough detection
# ---------------------------------------------------------------------------
def _atr(bars: pd.DataFrame, n: int = 20) -> pd.Series:
    """Wilder-style Average True Range (rolling window, price units)."""
    prev_close = bars["close"].shift(1)
    tr = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _find_local_peaks(
    series: np.ndarray,
    min_bars: int = 10,
    prominence: float = 0.0,
) -> np.ndarray:
    """Indices of local maxima in ``series`` with minimum separation ``min_bars``."""
    peaks, _ = find_peaks(series, distance=min_bars, prominence=prominence)
    return peaks


def _find_local_troughs(
    series: np.ndarray,
    min_bars: int = 10,
    prominence: float = 0.0,
) -> np.ndarray:
    """Indices of local minima (= peaks of the negated series)."""
    troughs, _ = find_peaks(-series, distance=min_bars, prominence=prominence)
    return troughs


# ---------------------------------------------------------------------------
# Pattern detection — double-top and double-bottom
# ---------------------------------------------------------------------------
PATTERN_NAMES = ["double_top", "double_bottom"]

PATTERN_DIR = {
    "double_top": -1,    # bearish: price expected to fall after confirmation
    "double_bottom": +1, # bullish: price expected to rise after confirmation
}


def detect_double_patterns(
    bars: pd.DataFrame,
    min_bars: int = 10,
    tolerance: float = 0.04,
    lookback: int = 120,
    atr_prominence_mult: float = 0.5,
) -> pd.DataFrame:
    """Detect confirmed double-top and double-bottom patterns on daily bars.

    For each bar *t*, we scan the trailing ``lookback`` bars for a pair of peaks
    (double-top) or troughs (double-bottom) that are within ``tolerance`` of each
    other.  Confirmation happens when:

    - **Double-top**: the close at bar *t* breaks *below* the trough that lies
      between the two peaks (the neckline), and this break has not already been
      signalled in a previous bar (no duplicate signals within ``min_bars``).
    - **Double-bottom**: the close at bar *t* breaks *above* the peak that lies
      between the two troughs.

    Parameters
    ----------
    bars :
        Daily OHLCV frame (timezone-naïve, indexed by date).
    min_bars :
        Minimum number of bars between consecutive peaks / troughs (and between
        consecutive signals of the same type).
    tolerance :
        Maximum relative height difference between the two peaks (or troughs) for
        them to be considered "similar": ``|h2 - h1| / h1 ≤ tolerance``.
    lookback :
        Number of bars of history to scan when searching for the prior peak pair.
        Patterns older than this window are ignored.
    atr_prominence_mult :
        Minimum peak prominence as a multiple of the trailing 20-day ATR.  A low
        value (e.g. 0.5 ATR) keeps small wiggles while still filtering micro-noise.

    Returns
    -------
    pd.DataFrame
        Boolean columns ``double_top`` and ``double_bottom``, indexed by date.
        ``True`` at row *t* means the pattern is confirmed at the *close* of bar
        *t*; forward returns begin at *t+1*.
    """
    close = bars["close"].to_numpy(dtype=float)
    atr_vals = _atr(bars).to_numpy(dtype=float)
    n = len(close)

    dt_signal = np.zeros(n, dtype=bool)
    db_signal = np.zeros(n, dtype=bool)

    last_dt = -min_bars  # track last signal to avoid duplicates
    last_db = -min_bars

    for t in range(lookback, n):
        window = close[t - lookback : t + 1]  # up to and including t
        atr_t = atr_vals[t] if np.isfinite(atr_vals[t]) else 0.0
        prom = max(atr_t * atr_prominence_mult, 1e-8)

        # --- double-top detection ---
        if t - last_dt >= min_bars:
            peaks = _find_local_peaks(window, min_bars=min_bars, prominence=prom)
            if len(peaks) >= 2:
                p1_idx, p2_idx = peaks[-2], peaks[-1]
                h1, h2 = window[p1_idx], window[p2_idx]
                # Two peaks of similar height
                if abs(h2 - h1) / (h1 + 1e-8) <= tolerance:
                    # Trough between the two peaks
                    between = window[p1_idx : p2_idx + 1]
                    neckline = between.min()
                    # Confirmation: current close breaks below neckline
                    if close[t] < neckline:
                        dt_signal[t] = True
                        last_dt = t

        # --- double-bottom detection ---
        if t - last_db >= min_bars:
            troughs = _find_local_troughs(window, min_bars=min_bars, prominence=prom)
            if len(troughs) >= 2:
                tr1_idx, tr2_idx = troughs[-2], troughs[-1]
                h1, h2 = window[tr1_idx], window[tr2_idx]
                if abs(h2 - h1) / (h1 + 1e-8) <= tolerance:
                    between = window[tr1_idx : tr2_idx + 1]
                    neckline = between.max()
                    if close[t] > neckline:
                        db_signal[t] = True
                        last_db = t

    return pd.DataFrame(
        {"double_top": dt_signal, "double_bottom": db_signal},
        index=bars.index,
    )


# ---------------------------------------------------------------------------
# Forward return engine (close-to-close)
# ---------------------------------------------------------------------------
def forward_returns(
    bars: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 20)
) -> pd.DataFrame:
    """Log close-to-close returns at each horizon (in trading days).

    The *h*-day return at row *t* is ``ln(close[t+h] / close[t])``.  A pattern
    confirmed at bar *t* uses these returns (which start at *t+1* implicitly
    since the return measures *t → t+h* and the position is opened at *t+1*).
    No look-ahead: pattern confirmed at close of *t*, return begins at open of
    *t+1* (approximated here as the close-to-close return from *t*).
    """
    close = bars["close"]
    out = {f"fwd_{h}": np.log(close.shift(-h) / close) for h in horizons}
    return pd.DataFrame(out, index=bars.index)


# ---------------------------------------------------------------------------
# Main study runner — signal vs random-day placebo
# ---------------------------------------------------------------------------
def run_pattern_study(
    bars: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 20),
    cost_bps: float = 1.0,
    seed: int = 189,
    **detect_kwargs,
) -> dict[str, pd.DataFrame]:
    """Run the double-pattern study on one tape, returning per-pattern ledgers.

    For each pattern name and each horizon, builds:
    - **signal arm**: forward returns on confirmed pattern days, in the claimed
      direction, net of a round-trip ``cost_bps``.
    - **random arm**: the same number of randomly selected days from the same
      tape, assigned the same claimed direction (unconditional base-rate placebo).

    Returns a dict keyed by pattern name; each value is a DataFrame with columns
    ``entry_date, dir, fwd_h, ret_signal, ret_random`` (one row per signal bar).

    No look-ahead: ``detect_double_patterns`` uses only closes up to bar *t*;
    forward returns start at *t+1*.
    """
    patterns = detect_double_patterns(bars, **detect_kwargs)
    fwd = forward_returns(bars, horizons)
    rng = np.random.default_rng(seed)
    results: dict[str, pd.DataFrame] = {}

    for name in PATTERN_NAMES:
        mask = patterns[name].astype(bool)
        if not mask.any():
            continue
        signal_idx = bars.index[mask]
        claimed_dir = PATTERN_DIR[name]

        # Align forward returns to signal bars.
        sig_fwd = fwd.loc[signal_idx].dropna(how="all")
        valid_idx = sig_fwd.index
        n_valid = len(valid_idx)
        if n_valid == 0:
            continue

        # Random-day control: draw n_valid random bars (exclude the last max(h) rows).
        max_h = max(horizons)
        eligible = bars.index[:-max_h]
        rand_pos = rng.choice(len(eligible), size=n_valid, replace=False)
        rand_dates = eligible[np.sort(rand_pos)]
        rand_fwd = fwd.loc[rand_dates].dropna(how="all")
        n_rand = len(rand_fwd)

        rows = []
        for sig_date, rand_date in zip(valid_idx[:n_rand], rand_fwd.index[:n_rand]):
            row: dict = {"entry_date": sig_date, "dir": claimed_dir}
            for h in horizons:
                col = f"fwd_{h}"
                if col in sig_fwd.columns and col in rand_fwd.columns:
                    rs = claimed_dir * float(sig_fwd.at[sig_date, col]) - cost_bps * 1e-4
                    rr = claimed_dir * float(rand_fwd.at[rand_date, col]) - cost_bps * 1e-4
                    row[f"ret_signal_{h}d"] = rs
                    row[f"ret_random_{h}d"] = rr
            rows.append(row)

        if rows:
            results[name] = pd.DataFrame(rows)

    return results


# ---------------------------------------------------------------------------
# Summarise one ledger — HAC t-stat on the mean excess return
# ---------------------------------------------------------------------------
def summarize_pattern(ledger: pd.DataFrame, horizon: int = 5) -> dict:
    """Headline statistics for one pattern at one horizon.

    Returns the signal count, win-rate, mean signal return (bps), mean random
    baseline return (bps), excess (signal − random), and a Newey-West HAC
    t-stat on the *excess* — the decisive number for the inference bar.

    A |t| ≥ 2 on the excess is required for Signal = REAL; after Bonferroni
    correction (6 tests) the threshold is |t| ≥ 3.
    """
    sig_col = f"ret_signal_{horizon}d"
    rand_col = f"ret_random_{horizon}d"
    if sig_col not in ledger.columns or rand_col not in ledger.columns:
        return {}

    rs = ledger[sig_col].to_numpy(dtype=float)
    rr = ledger[rand_col].to_numpy(dtype=float)
    valid = np.isfinite(rs) & np.isfinite(rr)
    rs, rr = rs[valid], rr[valid]
    n = len(rs)
    if n == 0:
        return {}

    excess = rs - rr
    out = {
        "n": int(n),
        "win_rate": float((rs > 0).mean()),
        "mean_bps": float(rs.mean() * 1e4),
        "baseline_bps": float(rr.mean() * 1e4),
        "excess_bps": float(excess.mean() * 1e4),
        "tstat_signal": float("nan"),
        "tstat_excess": float("nan"),
    }
    if n > 5:
        for arr, key in [(rs, "tstat_signal"), (excess, "tstat_excess")]:
            mu = arr.mean()
            e = arr - mu
            lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
            lrv = float(e @ e) / n
            for k in range(1, lags + 1):
                w = 1.0 - k / (lags + 1.0)
                lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
            se = np.sqrt(max(lrv, 0.0) / n)
            out[key] = float(mu / se) if se > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Cross-ticker pooled study helper
# ---------------------------------------------------------------------------
def run_pooled_study(
    tickers: list[str],
    fetch: bool = False,
    horizons: tuple[int, ...] = (1, 5, 20),
    cost_bps: float = 1.0,
    cache_dir: str | None = None,
    seed: int = 189,
    **detect_kwargs,
) -> dict[str, pd.DataFrame]:
    """Pool double-pattern study results across a list of tickers.

    Calls :func:`run_pattern_study` for each ticker and concatenates the
    per-ticker ledgers.  Returns a dict keyed by pattern name.
    """
    from .data import fetch_daily, DEFAULT_CACHE

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE

    pooled: dict[str, list[pd.DataFrame]] = {name: [] for name in PATTERN_NAMES}
    for ticker in tickers:
        bars = fetch_daily(ticker, fetch=fetch, cache_dir=cache_dir)
        study = run_pattern_study(bars, horizons=horizons, cost_bps=cost_bps,
                                  seed=seed, **detect_kwargs)
        for name, df in study.items():
            pooled[name].append(df)

    return {
        name: pd.concat(frames, ignore_index=True)
        for name, frames in pooled.items()
        if frames
    }
