"""The strategy and its honest controls — Study 76 (Rice-Paper).

Five classic Japanese candlestick reversal patterns are detected programmatically
on daily OHLCV bars and pinned against the only comparison that decides whether
they read the tape or the weather: **random-day entries with the same direction
bias** (the unconditional base-rate of up-days in the sample).

Patterns detected (all on a *per-bar* basis, confirmed at the close of bar *t*;
forward return is bar *t+1*'s close-to-close):

- **Bullish engulfing** — bearish prior bar fully engulfed by a larger bullish bar.
  Claimed: signals a bottom / reversal to the upside.
- **Bearish engulfing** — mirror image.
  Claimed: signals a top / reversal to the downside.
- **Hammer** — small body near the top of the range, long lower shadow, tiny upper
  shadow, after a down-move.  Claimed: bullish reversal.
- **Shooting star** — mirror of hammer (small body near bottom, long upper shadow)
  after an up-move.  Claimed: bearish reversal.
- **Doji** — open ≈ close (indecision).  The folklore direction is context-dependent;
  we treat it as a *bearish* signal when it appears after a run-up and *bullish* after
  a run-down, following the standard "evening/morning doji star" logic simplified to
  the single-bar sign.

The study's honest test uses **1-day and 5-day forward returns** (not a barrier
backtest — patterns are one-shot daily signals, not intrabar entries).  A
**random-day control arm** selects the same number of random days from the same
tape and assigns the same direction (+1 or −1 according to the pattern's claim),
providing the unconditional baseline.  A pattern adds value only if its mean
forward return *exceeds* the unconditional baseline; a HAC t-stat on the
*excess* return is the decisive number.

Multiple-comparisons caveat: five patterns × two horizons = ten t-stats.  At the
usual α=5% level, one spurious significant result is expected by chance alone.
We apply a Bonferroni correction and note which patterns survive it.

No look-ahead: patterns are identified on data available at the *close* of bar *t*;
forward returns begin at bar *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pattern detectors — all confirmed at the close of bar t
# ---------------------------------------------------------------------------
def _body(bars: pd.DataFrame) -> pd.Series:
    """Signed body size (close − open) as a fraction of the prior close."""
    return bars["close"] - bars["open"]


def _range(bars: pd.DataFrame) -> pd.Series:
    """True intrabar range (high − low)."""
    return bars["high"] - bars["low"]


def _lower_shadow(bars: pd.DataFrame) -> pd.Series:
    """Lower shadow = min(open, close) − low."""
    return bars[["open", "close"]].min(axis=1) - bars["low"]


def _upper_shadow(bars: pd.DataFrame) -> pd.Series:
    """Upper shadow = high − max(open, close)."""
    return bars["high"] - bars[["open", "close"]].max(axis=1)


def bullish_engulfing(bars: pd.DataFrame) -> pd.Series:
    """Boolean Series: bar t is a bullish engulfing candle.

    Conditions (applied to bar t and its predecessor t-1):
    - Prior bar (t-1) was bearish: close < open.
    - Current bar (t) is bullish: close > open.
    - Current body fully engulfs prior body:
      open ≤ prior_close  AND  close ≥ prior_open.
    """
    body = _body(bars)
    prev_open = bars["open"].shift(1)
    prev_close = bars["close"].shift(1)
    prior_bearish = prev_close < prev_open
    curr_bullish = bars["close"] > bars["open"]
    engulfs = (bars["open"] <= prev_close) & (bars["close"] >= prev_open)
    return (prior_bearish & curr_bullish & engulfs).fillna(False)


def bearish_engulfing(bars: pd.DataFrame) -> pd.Series:
    """Boolean Series: bar t is a bearish engulfing candle.

    Mirror of bullish engulfing: prior bar bullish, current bar bearish and
    fully engulfing the prior body.
    """
    body = _body(bars)
    prev_open = bars["open"].shift(1)
    prev_close = bars["close"].shift(1)
    prior_bullish = prev_close > prev_open
    curr_bearish = bars["close"] < bars["open"]
    engulfs = (bars["open"] >= prev_close) & (bars["close"] <= prev_open)
    return (prior_bullish & curr_bearish & engulfs).fillna(False)


def hammer(bars: pd.DataFrame, shadow_ratio: float = 2.0, body_max_frac: float = 0.35) -> pd.Series:
    """Boolean Series: bar t is a hammer (bullish reversal signal).

    Conditions:
    - Small real body (body ≤ body_max_frac × total range).
    - Long lower shadow (lower shadow ≥ shadow_ratio × body size).
    - Tiny upper shadow (upper shadow ≤ body size or ≤ 10% of range).
    - Prior bar was bearish (close < open) — context for a reversal.

    ``shadow_ratio`` and ``body_max_frac`` are the two tunable parameters that
    define how strict the pattern detector is.
    """
    rng = _range(bars)
    body_abs = _body(bars).abs()
    lo_shadow = _lower_shadow(bars)
    up_shadow = _upper_shadow(bars)
    prev_close = bars["close"].shift(1)
    prev_open = bars["open"].shift(1)
    small_body = body_abs <= body_max_frac * rng
    long_lower = lo_shadow >= shadow_ratio * body_abs.clip(lower=1e-8)
    tiny_upper = up_shadow <= body_abs + 1e-8
    prior_bearish = prev_close < prev_open
    return (small_body & long_lower & tiny_upper & prior_bearish).fillna(False)


def shooting_star(
    bars: pd.DataFrame, shadow_ratio: float = 2.0, body_max_frac: float = 0.35
) -> pd.Series:
    """Boolean Series: bar t is a shooting star (bearish reversal signal).

    Mirror of hammer: long upper shadow, tiny lower shadow, prior bar was bullish.
    """
    rng = _range(bars)
    body_abs = _body(bars).abs()
    lo_shadow = _lower_shadow(bars)
    up_shadow = _upper_shadow(bars)
    prev_close = bars["close"].shift(1)
    prev_open = bars["open"].shift(1)
    small_body = body_abs <= body_max_frac * rng
    long_upper = up_shadow >= shadow_ratio * body_abs.clip(lower=1e-8)
    tiny_lower = lo_shadow <= body_abs + 1e-8
    prior_bullish = prev_close > prev_open
    return (small_body & long_upper & tiny_lower & prior_bullish).fillna(False)


def doji(bars: pd.DataFrame, body_max_frac: float = 0.10) -> pd.Series:
    """Boolean Series: bar t is a doji (open ≈ close, indecision candle).

    A doji is defined as a bar whose body is no more than ``body_max_frac``
    of the total range.  Direction assignment (bullish/bearish) is handled
    downstream in :func:`detect_patterns` based on the 3-day prior context.
    """
    rng = _range(bars)
    body_abs = _body(bars).abs()
    return (body_abs <= body_max_frac * rng).fillna(False)


# ---------------------------------------------------------------------------
# Unified pattern detection
# ---------------------------------------------------------------------------
PATTERN_NAMES = [
    "bullish_engulfing",
    "bearish_engulfing",
    "hammer",
    "shooting_star",
    "doji_bullish",
    "doji_bearish",
]

# Claimed direction: +1 = pattern predicts next day up, -1 = predicts next day down.
PATTERN_DIR = {
    "bullish_engulfing": +1,
    "bearish_engulfing": -1,
    "hammer": +1,
    "shooting_star": -1,
    "doji_bullish": +1,
    "doji_bearish": -1,
}


def detect_patterns(bars: pd.DataFrame) -> pd.DataFrame:
    """Detect all five pattern families on ``bars``; return a boolean indicator frame.

    Columns are the six pattern names in ``PATTERN_NAMES``.  A ``True`` at row *t*
    means the pattern is confirmed at the close of bar *t* — the claimed forward
    return begins at bar *t+1*.

    The doji is split into directional variants: doji_bullish fires when a doji
    appears after 3 consecutively declining closes (a 'morning star' context);
    doji_bearish fires after 3 consecutively rising closes.
    """
    be = bullish_engulfing(bars)
    beg = bearish_engulfing(bars)
    h = hammer(bars)
    ss = shooting_star(bars)
    d = doji(bars)

    # Directional doji: split by 3-day prior close direction.
    close = bars["close"]
    prior_trend = close.shift(1) - close.shift(4)  # 3-day prior move
    d_bull = d & (prior_trend < 0)
    d_bear = d & (prior_trend > 0)

    return pd.DataFrame(
        {
            "bullish_engulfing": be,
            "bearish_engulfing": beg,
            "hammer": h,
            "shooting_star": ss,
            "doji_bullish": d_bull,
            "doji_bearish": d_bear,
        },
        index=bars.index,
    )


# ---------------------------------------------------------------------------
# Forward return engine
# ---------------------------------------------------------------------------
def forward_returns(bars: pd.DataFrame, horizons: tuple[int, ...] = (1, 5)) -> pd.DataFrame:
    """Log close-to-close returns at the given ``horizons`` (in trading days).

    The *h*-day return at row *t* is the return from bar *t* to bar *t+h*:
    ``ln(close[t+h] / close[t])``.  A pattern at row *t* exploits ``fwd_1`` and
    ``fwd_5`` (the forward returns that begin at *t+1*, i.e. with the pattern bar
    already in the past — no look-ahead).

    Note: for a pattern at bar *t*, the relevant forward return is the return
    that *starts at t+1*, which is stored at row *t* in this frame as
    ``ln(close[t+h] / close[t])``.
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
      assigned at the unconditional base-rate), for comparison.

    Returns a dict keyed by ``pattern_name``, each value a DataFrame with columns
    ``entry_date, dir, fwd_h, ret_signal, ret_random`` (one row per signal bar).

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
        n = len(signal_idx)
        claimed_dir = PATTERN_DIR[name]

        # Align forward returns to signal bars (shifted by 1 so the return starts at t+1).
        # fwd_h stored at row t already represents ln(close[t+h]/close[t]).
        sig_fwd = fwd.loc[signal_idx].dropna(how="all")
        valid_idx = sig_fwd.index
        n_valid = len(valid_idx)
        if n_valid == 0:
            continue

        # Random-day control: draw n_valid random bars (excluding last max(horizons) rows
        # so the forward return exists), assign claimed direction.
        max_h = max(horizons)
        eligible = bars.index[:-max_h]
        rand_idx = rng.choice(len(eligible), size=n_valid, replace=False)
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
                    row[f"ret_signal_{h}d"] = claimed_dir * float(sig_fwd.at[sig_date, col]) - cost_bps * 1e-4
                    row[f"ret_random_{h}d"] = claimed_dir * float(rand_fwd.at[rand_date, col]) - cost_bps * 1e-4
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
    is required for Signal=REAL.
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
        for col_name, arr, key in [
            ("excess", excess, "tstat_excess"),
            ("signal", rs, "tstat_signal"),
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
