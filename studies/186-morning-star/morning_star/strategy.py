"""The strategy and its honest controls — Study 186 (Morning-Star).

Two three-candle reversal patterns are detected programmatically on daily OHLCV
bars and pinned against the only comparison that decides whether they read the tape
or the weather: **random-day entries with the same direction bias** (the
unconditional base-rate of up-days in the sample).

Patterns detected (all confirmed at the close of bar *t*; forward return begins at
bar *t+1*):

- **Morning star** — a three-candle bullish reversal after a downtrend.
  Candle 1: large bearish body.
  Candle 2 (the "star"): small body gapping below candle-1's low body, indecision.
  Candle 3: large bullish body closing into candle-1's body.
  Claimed: signals the end of a downtrend; buy the next open.

- **Evening star** — the bearish mirror image of the morning star.
  Candle 1: large bullish body.
  Candle 2 (the "star"): small body gapping above candle-1's high body, indecision.
  Candle 3: large bearish body closing into candle-1's body.
  Claimed: signals the end of an uptrend; sell/short the next open.

Both patterns share the same three-bar detection logic with tunable thresholds,
making it trivial to audit look-ahead or tighten/loosen the parameters.

The study's honest test uses **1-day and 5-day forward returns** (not a barrier
backtest — patterns are one-shot daily signals, not intrabar entries).  A
**random-day control arm** selects the same number of random days from the same
tape and assigns the same direction (+1 or −1 according to the pattern's claim),
providing the unconditional baseline.  A pattern adds value only if its mean
forward return *exceeds* the unconditional baseline; a HAC t-stat on the *excess*
return is the decisive number.

Multiple-comparisons caveat: two patterns × two horizons = four t-stats.  At the
usual α=5% level, one spurious significant result is expected every twenty
independent experiments.  We apply a Bonferroni correction (|t| ≥ 2.50 for
k=4 tests) and note which patterns survive it.

No look-ahead: patterns are identified on data available at the *close* of bar *t*;
forward returns begin at bar *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Body / shadow helpers — all confirmed at the close of bar t
# ---------------------------------------------------------------------------
def _body_size(bars: pd.DataFrame) -> pd.Series:
    """Absolute body size: |close − open|."""
    return (bars["close"] - bars["open"]).abs()


def _body_top(bars: pd.DataFrame) -> pd.Series:
    """Top of the real body: max(open, close)."""
    return bars[["open", "close"]].max(axis=1)


def _body_bottom(bars: pd.DataFrame) -> pd.Series:
    """Bottom of the real body: min(open, close)."""
    return bars[["open", "close"]].min(axis=1)


def _range(bars: pd.DataFrame) -> pd.Series:
    """True intrabar range (high − low)."""
    return bars["high"] - bars["low"]


def _is_bullish(bars: pd.DataFrame) -> pd.Series:
    """True where close > open (bullish candle)."""
    return bars["close"] > bars["open"]


def _is_bearish(bars: pd.DataFrame) -> pd.Series:
    """True where close < open (bearish candle)."""
    return bars["close"] < bars["open"]


# ---------------------------------------------------------------------------
# Morning-star detector (bullish reversal)
# ---------------------------------------------------------------------------
def morning_star(
    bars: pd.DataFrame,
    body_large_frac: float = 0.50,
    body_star_frac: float = 0.40,
    penetration: float = 0.30,
    prior_down_days: int = 2,
    star_gap_frac: float = 0.50,
) -> pd.Series:
    """Boolean Series: bar *t* completes a morning-star pattern.

    Three-candle definition (candles at t-2, t-1, t):

    1. **Candle t-2**: large bearish body (body ≥ ``body_large_frac`` × daily range).
    2. **Candle t-1** (the star): small body (body ≤ ``body_star_frac`` × candle t-2's
       body); star's body_top is at most ``star_gap_frac`` × body_t2 *above* the t-2
       body_bottom.  This allows a partial overlap: a true gap (star top ≤ t-2 bottom)
       is the textbook ideal, but on liquid US equities with small overnight gaps an
       "almost-gap" — where the star sits mostly below the prior body — is the practical
       version of the pattern.  Setting ``star_gap_frac=0`` enforces the strict gap.
    3. **Candle t**: bullish body that closes at least ``penetration`` of the way into
       candle t-2's body.  ``penetration=0.30`` means at least 30% closure — the
       conservative default; the textbook often uses 50% but empirical usage varies.
    4. **Prior context**: the close at t-2 is below the close ``prior_down_days`` days
       before t-2, confirming a downtrend context before the pattern.

    Returns a Boolean Series aligned to ``bars.index``; ``False`` where any lookback
    bar is missing (the first ``prior_down_days + 2`` bars are always False).
    """
    c = bars["close"]
    o = bars["open"]

    body_t = _body_size(bars)
    range_t = _range(bars)

    body_t2 = body_t.shift(2)
    range_t2 = range_t.shift(2)
    close_t2 = c.shift(2)
    body_top_t2 = _body_top(bars).shift(2)
    body_bot_t2 = _body_bottom(bars).shift(2)
    bearish_t2 = _is_bearish(bars).shift(2)

    body_t1 = body_t.shift(1)
    body_top_t1 = _body_top(bars).shift(1)

    bullish_t0 = _is_bullish(bars)

    # --- Condition 1: candle t-2 is large and bearish ---
    c1 = bearish_t2 & (body_t2 >= body_large_frac * range_t2.clip(lower=1e-12))

    # --- Condition 2: candle t-1 is the "star" — small body, at or near below t-2 body ---
    # Star body ≤ body_star_frac × candle t-2's body.
    # Star body_top ≤ body_bot_t2 + star_gap_frac * body_t2 (allows partial overlap).
    c2 = (body_t1 <= body_star_frac * body_t2.clip(lower=1e-12)) & (
        body_top_t1 <= body_bot_t2 + star_gap_frac * body_t2
    )

    # --- Condition 3: candle t is bullish and closes into t-2's body ---
    # close[t] >= body_bot_t2 + penetration * body_t2
    penetration_level = body_bot_t2 + penetration * body_t2
    c3 = bullish_t0 & (c >= penetration_level)

    # --- Condition 4: prior downtrend context ---
    prior_context_close = c.shift(2 + prior_down_days)
    c4 = close_t2 < prior_context_close

    result = c1 & c2 & c3 & c4
    return result.fillna(False)


# ---------------------------------------------------------------------------
# Evening-star detector (bearish reversal)
# ---------------------------------------------------------------------------
def evening_star(
    bars: pd.DataFrame,
    body_large_frac: float = 0.50,
    body_star_frac: float = 0.40,
    penetration: float = 0.30,
    prior_up_days: int = 2,
    star_gap_frac: float = 0.50,
) -> pd.Series:
    """Boolean Series: bar *t* completes an evening-star pattern.

    Mirror of :func:`morning_star` — a three-candle bearish reversal after an uptrend:

    1. **Candle t-2**: large bullish body (body ≥ ``body_large_frac`` × daily range).
    2. **Candle t-1** (the star): small body (body ≤ ``body_star_frac`` × t-2's body);
       star's body_bottom is at most ``star_gap_frac`` × body_t2 *below* the t-2
       body_top.  Allows a partial overlap (same relaxation as the morning-star, for
       the same practical reason: liquid US equities rarely gap cleanly).
    3. **Candle t**: bearish body closing at least ``penetration`` into candle t-2's body.
    4. **Prior context**: close at t-2 exceeds close at ``t-2-prior_up_days`` (uptrend).

    The same tunable parameters apply as in :func:`morning_star`.
    """
    c = bars["close"]
    o = bars["open"]

    body_t = _body_size(bars)
    range_t = _range(bars)

    body_t2 = body_t.shift(2)
    range_t2 = range_t.shift(2)
    close_t2 = c.shift(2)
    body_top_t2 = _body_top(bars).shift(2)
    body_bot_t2 = _body_bottom(bars).shift(2)
    bullish_t2 = _is_bullish(bars).shift(2)

    body_t1 = body_t.shift(1)
    body_bot_t1 = _body_bottom(bars).shift(1)

    bearish_t0 = _is_bearish(bars)

    # --- Condition 1: candle t-2 is large and bullish ---
    c1 = bullish_t2 & (body_t2 >= body_large_frac * range_t2.clip(lower=1e-12))

    # --- Condition 2: candle t-1 is the star — small body, at or near above t-2's body ---
    # Star body_bottom ≥ body_top_t2 - star_gap_frac * body_t2 (allows partial overlap).
    c2 = (body_t1 <= body_star_frac * body_t2.clip(lower=1e-12)) & (
        body_bot_t1 >= body_top_t2 - star_gap_frac * body_t2
    )

    # --- Condition 3: candle t is bearish, closes into t-2's body ---
    # close[t] <= body_top_t2 - penetration * body_t2
    penetration_level = body_top_t2 - penetration * body_t2
    c3 = bearish_t0 & (c <= penetration_level)

    # --- Condition 4: prior uptrend context ---
    prior_context_close = c.shift(2 + prior_up_days)
    c4 = close_t2 > prior_context_close

    result = c1 & c2 & c3 & c4
    return result.fillna(False)


# ---------------------------------------------------------------------------
# Unified pattern detection
# ---------------------------------------------------------------------------
PATTERN_NAMES = ["morning_star", "evening_star"]

# Claimed direction: +1 = pattern predicts next day up, -1 = predicts next day down.
PATTERN_DIR = {
    "morning_star": +1,
    "evening_star": -1,
}


def detect_patterns(bars: pd.DataFrame) -> pd.DataFrame:
    """Detect both three-candle reversal patterns on ``bars``.

    Returns a boolean indicator frame with columns ``morning_star`` and
    ``evening_star``.  A ``True`` at row *t* means the pattern is confirmed at
    the close of bar *t* — the claimed forward return begins at bar *t+1*.

    All inputs are strictly on data available at close of bar *t*; no
    look-ahead is possible since the three-candle windows look backward.
    """
    ms = morning_star(bars)
    es = evening_star(bars)
    return pd.DataFrame(
        {"morning_star": ms, "evening_star": es},
        index=bars.index,
    )


# ---------------------------------------------------------------------------
# Forward return engine
# ---------------------------------------------------------------------------
def forward_returns(bars: pd.DataFrame, horizons: tuple[int, ...] = (1, 5)) -> pd.DataFrame:
    """Log close-to-close returns at the given ``horizons`` (in trading days).

    The *h*-day return at row *t* is the return from bar *t* to bar *t+h*:
    ``ln(close[t+h] / close[t])``.  A pattern at row *t* exploits ``fwd_1`` and
    ``fwd_5`` (the forward returns beginning at *t+1*, i.e. with the pattern bar
    already in the past — no look-ahead).
    """
    close = bars["close"]
    out = {}
    for h in horizons:
        out[f"fwd_{h}"] = np.log(close.shift(-h) / close)
    return pd.DataFrame(out, index=bars.index)


# ---------------------------------------------------------------------------
# Pattern-vs-baseline study
# ---------------------------------------------------------------------------
def run_pattern_study(
    bars: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5),
    cost_bps: float = 1.0,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Run the full pattern vs random-baseline study on one tape.

    For each pattern and each horizon, compute:
    - ``signal`` arm: forward returns on pattern-triggered days, in the claimed direction.
    - ``random`` arm: the same number of randomly selected days (same tape, same direction
      as claimed by the pattern), for comparison.

    Returns a dict keyed by ``pattern_name``, each value a DataFrame with columns
    ``entry_date, dir, ret_signal_<h>d, ret_random_<h>d`` (one row per signal bar).

    No look-ahead: ``detect_patterns`` uses only data up to and including bar *t*;
    forward returns start at *t+1* (which is ``fwd_h`` stored at row *t*).
    """
    patterns = detect_patterns(bars)
    fwd = forward_returns(bars, horizons)
    rng = np.random.default_rng(seed)

    results = {}
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

        # Random-day control: draw n_valid random bars (excluding last max(horizons) rows
        # so the forward return exists), assign claimed direction.
        max_h = max(horizons)
        eligible = bars.index[:-max_h]
        n_eligible = len(eligible)
        if n_eligible < n_valid:
            # Not enough eligible bars — draw with replacement.
            rand_idx = rng.choice(n_eligible, size=n_valid, replace=True)
        else:
            rand_idx = rng.choice(n_eligible, size=n_valid, replace=False)
        rand_dates = eligible[np.sort(rand_idx)]
        rand_fwd = fwd.loc[rand_dates].dropna(how="all")
        n_rand = len(rand_fwd)

        rows = []
        for i, (sig_date, rand_date) in enumerate(
            zip(valid_idx[:n_rand], rand_fwd.index)
        ):
            row = {"entry_date": sig_date, "dir": claimed_dir}
            for h in horizons:
                col = f"fwd_{h}"
                if col in sig_fwd.columns:
                    row[f"ret_signal_{h}d"] = (
                        claimed_dir * float(sig_fwd.at[sig_date, col]) - cost_bps * 1e-4
                    )
                    row[f"ret_random_{h}d"] = (
                        claimed_dir * float(rand_fwd.at[rand_date, col]) - cost_bps * 1e-4
                    )
            rows.append(row)

        if rows:
            results[name] = pd.DataFrame(rows)
    return results


# ---------------------------------------------------------------------------
# Summarise one study ledger — HAC t-stat on the mean excess return
# ---------------------------------------------------------------------------
def summarize_pattern(ledger: pd.DataFrame, horizon: int = 1) -> dict:
    """Headline per-pattern statistics.

    Computes the mean signal return, mean random-baseline return, their
    difference (the excess return), and a Newey-West HAC t-stat on the
    *excess* return — the decisive number for the inference bar.

    Returns trade count, win-rate, mean return (bps/trade), baseline mean,
    excess mean, and the HAC t-stat on the excess.  A |t| ≥ 2 on the excess
    is required for Signal=REAL; with Bonferroni for 4 tests the threshold is
    |t| ≥ 2.50.
    """
    sig_col = f"ret_signal_{horizon}d"
    rand_col = f"ret_random_{horizon}d"
    if sig_col not in ledger.columns or rand_col not in ledger.columns:
        return {}

    rs = ledger[sig_col].to_numpy(dtype=float)
    rr = ledger[rand_col].to_numpy(dtype=float)
    rs = rs[np.isfinite(rs)]
    rr = rr[np.isfinite(rr)]
    n = min(len(rs), len(rr))
    if n == 0:
        return {}

    rs, rr = rs[:n], rr[:n]
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
        for arr, key in [
            (excess, "tstat_excess"),
            (rs, "tstat_signal"),
        ]:
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
    horizons: tuple[int, ...] = (1, 5),
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


def random_directions(n: int, seed: int = 0) -> np.ndarray:
    """A reproducible vector of ±1 — the control arm's coin."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=n)
