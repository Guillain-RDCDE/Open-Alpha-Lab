"""The strategy and its honest controls — Study 187 (Three-Soldiers).

Two famous three-bar candlestick *continuation* patterns are detected on daily
OHLCV bars and measured against the only comparison that decides whether they
carry directional information: a **random-day control** with the same direction
bias as the unconditional base-rate of the tape.

Patterns detected (confirmed at the close of bar *t*; forward return starts at
bar *t+1*):

- **Three White Soldiers** (3WS) — three consecutive strongly bullish candles,
  each closing near its high, each opening within the prior bar's body, and
  each making a higher close.  Claimed: a powerful continuation / resumption
  signal to the upside after a consolidation.

- **Three Black Crows** (3BC) — the mirror image: three consecutive strongly
  bearish candles, each closing near its low, each opening within the prior
  bar's body, and each making a lower close.  Claimed: a continuation / trend
  resumption signal to the downside.

The honest test uses **1-day, 5-day, and 10-day forward returns** (close-to-
close log returns starting at *t+1*).  A **random-day control arm** selects the
same number of random days and assigns the same claimed direction (+1 for 3WS,
−1 for 3BC), providing the unconditional baseline.  A pattern adds real signal
only if its mean forward return *exceeds* the unconditional baseline with a HAC
t-stat |t| ≥ 2.

Multiple-comparisons caveat: two patterns × three horizons = six t-stats.  We
apply a Bonferroni correction (α_adj = 0.05/6 ≈ 0.008, |t|_adj ≈ 2.65) and
report which, if any, survive it.

No look-ahead: patterns are identified on data available at the *close* of bar
*t*; forward returns begin at bar *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pattern detectors — all confirmed at the close of bar t
# ---------------------------------------------------------------------------
def _body(bars: pd.DataFrame) -> pd.Series:
    """Signed body: close − open."""
    return bars["close"] - bars["open"]


def _body_abs(bars: pd.DataFrame) -> pd.Series:
    """Absolute body size."""
    return (bars["close"] - bars["open"]).abs()


def _range(bars: pd.DataFrame) -> pd.Series:
    """Full intrabar range: high − low."""
    return bars["high"] - bars["low"]


def _upper_shadow(bars: pd.DataFrame) -> pd.Series:
    """Upper shadow: high − max(open, close)."""
    return bars["high"] - bars[["open", "close"]].max(axis=1)


def _lower_shadow(bars: pd.DataFrame) -> pd.Series:
    """Lower shadow: min(open, close) − low."""
    return bars[["open", "close"]].min(axis=1) - bars["low"]


def three_white_soldiers(
    bars: pd.DataFrame,
    body_min_frac: float = 0.5,
    shadow_max_frac: float = 0.30,
) -> pd.Series:
    """Boolean Series: bar *t* completes a Three White Soldiers pattern.

    Three White Soldiers criteria (evaluated at bar *t*, looking back at bars
    t-2, t-1, and t):

    1. All three bars are bullish (close > open).
    2. The three closes strictly ascend: close[t-2] < close[t-1] < close[t].
    3. The three opens also ascend: open[t-2] < open[t-1] < open[t]
       (the "marching" structure that distinguishes soldiers from a spike).
    4. Each body is substantial: body ≥ ``body_min_frac`` × total range (the
       bars are strong, not doji-like).
    5. Each upper shadow is small: upper shadow ≤ ``shadow_max_frac`` × total
       range (each bar closes near its high, confirming buyer conviction).

    Bulkowski (2008) and Morris (2006) define the pattern in terms of three
    consecutive long white (bullish) candles with ascending closes.  The strict
    "open within prior body" condition from older texts is omitted here because
    it dramatically reduces signal count on modern gapped-open markets without
    improving predictive accuracy (see Lim & Loh 2014).

    ``body_min_frac`` controls how strong each candle must be; ``shadow_max_frac``
    controls how close each close must be to the high.
    """
    body = _body(bars)            # close - open (positive for bullish bars)
    rng = _range(bars)             # high - low
    up_shd = _upper_shadow(bars)   # high - max(open, close)

    # Single-bar quality conditions.
    bullish = body > 0
    strong_body = body >= body_min_frac * rng.clip(lower=1e-8)
    small_upper = up_shd <= shadow_max_frac * rng.clip(lower=1e-8)
    bar_quality = bullish & strong_body & small_upper

    # Ascending closes and opens across all three bars.
    asc_close_t = bars["close"] > bars["close"].shift(1)
    asc_close_t1 = bars["close"].shift(1) > bars["close"].shift(2)
    asc_open_t = bars["open"] > bars["open"].shift(1)
    asc_open_t1 = bars["open"].shift(1) > bars["open"].shift(2)

    result = (
        bar_quality
        & bar_quality.shift(1).fillna(False)
        & bar_quality.shift(2).fillna(False)
        & asc_close_t
        & asc_close_t1
        & asc_open_t
        & asc_open_t1
    )
    return result.fillna(False)


def three_black_crows(
    bars: pd.DataFrame,
    body_min_frac: float = 0.5,
    shadow_max_frac: float = 0.30,
) -> pd.Series:
    """Boolean Series: bar *t* completes a Three Black Crows pattern.

    Three Black Crows criteria — the exact mirror of Three White Soldiers
    (evaluated at bar *t*, looking back at bars t-2, t-1, and t):

    1. All three bars are bearish (close < open).
    2. The three closes strictly descend: close[t-2] > close[t-1] > close[t].
    3. The three opens also descend: open[t-2] > open[t-1] > open[t]
       (the descending staircase structure).
    4. Each body is substantial (body ≥ ``body_min_frac`` × range).
    5. Each lower shadow is small (lower shadow ≤ ``shadow_max_frac`` × range) —
       each bar closes near its low, confirming seller conviction.
    """
    body = _body(bars)    # negative for bearish bars
    body_abs = body.abs()
    rng = _range(bars)
    lo_shd = _lower_shadow(bars)

    bearish = body < 0
    strong_body = body_abs >= body_min_frac * rng.clip(lower=1e-8)
    small_lower = lo_shd <= shadow_max_frac * rng.clip(lower=1e-8)
    bar_quality = bearish & strong_body & small_lower

    desc_close_t = bars["close"] < bars["close"].shift(1)
    desc_close_t1 = bars["close"].shift(1) < bars["close"].shift(2)
    desc_open_t = bars["open"] < bars["open"].shift(1)
    desc_open_t1 = bars["open"].shift(1) < bars["open"].shift(2)

    result = (
        bar_quality
        & bar_quality.shift(1).fillna(False)
        & bar_quality.shift(2).fillna(False)
        & desc_close_t
        & desc_close_t1
        & desc_open_t
        & desc_open_t1
    )
    return result.fillna(False)


# ---------------------------------------------------------------------------
# Pattern name registry
# ---------------------------------------------------------------------------
PATTERN_NAMES = ["three_white_soldiers", "three_black_crows"]

# Claimed direction: +1 = pattern predicts continuation upward,
#                   -1 = pattern predicts continuation downward.
PATTERN_DIR = {
    "three_white_soldiers": +1,
    "three_black_crows": -1,
}


def detect_patterns(bars: pd.DataFrame) -> pd.DataFrame:
    """Detect Three White Soldiers and Three Black Crows on ``bars``.

    Returns a boolean DataFrame with one column per pattern, indexed as ``bars``.
    A ``True`` at row *t* means the pattern is confirmed at the *close* of bar
    *t* — forward returns start at bar *t+1*.  Requires at least 3 bars.
    """
    return pd.DataFrame(
        {
            "three_white_soldiers": three_white_soldiers(bars),
            "three_black_crows": three_black_crows(bars),
        },
        index=bars.index,
    )


# ---------------------------------------------------------------------------
# Forward return engine
# ---------------------------------------------------------------------------
def forward_returns(
    bars: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 10)
) -> pd.DataFrame:
    """Log close-to-close returns at the given ``horizons`` (in trading days).

    The *h*-day return at row *t* is ``ln(close[t+h] / close[t])``.  A pattern
    at bar *t* exploits ``fwd_h`` — the return from bar *t* to bar *t+h*, which
    *starts* at bar *t+1* — so there is no look-ahead here.
    """
    close = bars["close"]
    return pd.DataFrame(
        {f"fwd_{h}": np.log(close.shift(-h) / close) for h in horizons},
        index=bars.index,
    )


# ---------------------------------------------------------------------------
# Pattern vs random-day control study
# ---------------------------------------------------------------------------
def run_pattern_study(
    bars: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10),
    cost_bps: float = 1.0,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Run Three White Soldiers / Three Black Crows against a random-day baseline.

    For each pattern and each horizon, compute:

    - **signal arm**: forward returns on pattern-triggered days, in the claimed
      direction, net of ``cost_bps`` round-trip.
    - **random arm**: the same number of randomly selected days (same tape, same
      claimed direction), providing the unconditional baseline.

    Returns a dict keyed by pattern name, each value a DataFrame with columns
    ``entry_date, dir, ret_signal_{h}d, ret_random_{h}d`` for each horizon.

    No look-ahead: ``detect_patterns`` uses only data at the *close* of bar *t*;
    forward returns start at *t+1*.
    """
    patterns = detect_patterns(bars)
    fwd = forward_returns(bars, horizons)
    rng = np.random.default_rng(seed)
    max_h = max(horizons)

    results: dict[str, pd.DataFrame] = {}
    for name in PATTERN_NAMES:
        mask = patterns[name].fillna(False).astype(bool)
        if not mask.any():
            continue
        signal_idx = bars.index[mask]
        claimed_dir = PATTERN_DIR[name]
        n = len(signal_idx)

        # Align forward returns to signal bars; drop rows where future data is NaN.
        sig_fwd = fwd.loc[signal_idx].dropna(how="all")
        valid_idx = sig_fwd.index
        n_valid = len(valid_idx)
        if n_valid == 0:
            continue

        # Random-day control: draw n_valid random bars from eligible dates
        # (exclude the last max_h rows, as their forward return doesn't exist).
        eligible = bars.index[:-max_h]
        rand_pos = np.sort(rng.choice(len(eligible), size=n_valid, replace=False))
        rand_dates = eligible[rand_pos]
        rand_fwd = fwd.loc[rand_dates].dropna(how="all")
        n_rand = len(rand_fwd)

        rows = []
        for i, (sig_date, rand_date) in enumerate(
            zip(valid_idx[:n_rand], rand_fwd.index[:n_rand])
        ):
            row: dict = {"entry_date": sig_date, "dir": claimed_dir}
            for h in horizons:
                col = f"fwd_{h}"
                if col in sig_fwd.columns and col in rand_fwd.columns:
                    sig_r = sig_fwd.at[sig_date, col] if sig_date in sig_fwd.index else float("nan")
                    rnd_r = rand_fwd.at[rand_date, col] if rand_date in rand_fwd.index else float("nan")
                    row[f"ret_signal_{h}d"] = (
                        claimed_dir * float(sig_r) - cost_bps * 1e-4
                        if np.isfinite(sig_r) else float("nan")
                    )
                    row[f"ret_random_{h}d"] = (
                        claimed_dir * float(rnd_r) - cost_bps * 1e-4
                        if np.isfinite(rnd_r) else float("nan")
                    )
            rows.append(row)

        if rows:
            results[name] = pd.DataFrame(rows)

    return results


# ---------------------------------------------------------------------------
# Summary statistics — HAC t-stat on excess return over the baseline
# ---------------------------------------------------------------------------
def summarize_pattern(ledger: pd.DataFrame, horizon: int = 1) -> dict:
    """Headline per-pattern statistics for one horizon.

    Computes the mean signal return, mean random-baseline return, their
    difference (the excess return), and a Newey–West HAC t-stat on the
    *excess* return — the decisive number for the inference bar.

    Returns trade count, win-rate, mean signal (bps/trade), baseline mean,
    excess mean, HAC t-stat on excess, and HAC t-stat on the raw signal.
    A |t| ≥ 2 on the *excess* is required for Signal=REAL; the Bonferroni
    threshold is |t| ≥ 2.65 (α_adj = 0.05/6, two patterns × three horizons).
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
        "baseline_mean_bps": float(rr.mean() * 1e4),
        "excess_bps": float(excess.mean() * 1e4),
        "tstat_excess": float("nan"),
        "tstat_signal": float("nan"),
    }
    if n > 5:
        for arr, key in [(excess, "tstat_excess"), (rs, "tstat_signal")]:
            mu = arr.mean()
            e = arr - mu
            lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
            lrv = float(e @ e) / n
            for k in range(1, lags + 1):
                w = 1.0 - k / (lags + 1.0)
                lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
            se = np.sqrt(max(lrv, 0.0) / n)
            out[key] = float(mu / se) if se > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Cross-ticker pooled study
# ---------------------------------------------------------------------------
def run_pooled_study(
    tickers: list[str],
    fetch: bool = False,
    horizons: tuple[int, ...] = (1, 5, 10),
    cost_bps: float = 1.0,
    cache_dir: str | None = None,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Pool pattern study results across a list of tickers.

    Calls :func:`run_pattern_study` for each ticker and concatenates the
    per-ticker ledgers.  Returns a dict keyed by pattern name.
    """
    from .data import fetch_daily, DEFAULT_CACHE

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE

    pooled: dict[str, list[pd.DataFrame]] = {name: [] for name in PATTERN_NAMES}
    for t in tickers:
        bars = fetch_daily(t, fetch=fetch, cache_dir=cache_dir)
        study = run_pattern_study(bars, horizons=horizons, cost_bps=cost_bps, seed=seed)
        for name, df in study.items():
            pooled[name].append(df)

    return {
        name: pd.concat(frames, ignore_index=True)
        for name, frames in pooled.items()
        if frames
    }
