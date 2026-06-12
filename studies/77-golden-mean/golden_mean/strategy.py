"""The strategy and its honest controls — Study 77 (Golden-Mean).

The folk recipe: identify a significant swing (a local peak-to-trough move of at
least some minimum size), compute the Fibonacci retracement levels at 38.2%, 50%,
and 61.8% of that swing, and expect price to *bounce* (reverse) when it touches
one of those levels. The levels are said to represent "universal mathematical
harmonics" that markets allegedly respect.

We test this against the only comparison that decides whether it is numerology or
signal: the same set of entries with **randomly-shifted "placebo" levels** — control
levels spaced by the same swing size but at non-Fibonacci positions. Only the
*Fibonacci specificity* — the fact that *these ratios* vs *other ratios* — can
separate the two arms.

The key design constraint is *speed*: with ~6000 swings per ticker × thousands of
forward bars, naively scanning every (swing, bar, level) triple is O(10^9) and
impractical. The engine instead:

1. Restricts level-checking to a **forward window** starting immediately after the
   swing identification bar and limited to ``max_fwd_days`` days (default 40).
2. Checks touches **vectorised** using numpy broadcasting rather than Python loops.
3. Deduplicates: if the same level is touched multiple times within a window, only
   the *first* touch is kept (one entry per approach).

No look-ahead: swings are identified from data strictly up to bar *t*; the
forward window starts at bar *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FIB_RATIOS, FIB_KEY


# ---------------------------------------------------------------------------
# Level computation helpers
# ---------------------------------------------------------------------------
def fib_levels(peak: float, trough: float, ratios=FIB_KEY) -> dict[float, float]:
    """Compute Fibonacci retracement price levels for a given swing.

    For a swing from ``peak`` to ``trough``, the retracement levels are:
    ``trough + ratio * (peak - trough)`` for each ratio. Returns a dict
    mapping each ratio to its price level.
    """
    span = peak - trough
    return {r: trough + r * span for r in ratios}


def placebo_levels(
    peak: float,
    trough: float,
    seed: int = 42,
    n_ratios: int = 3,
) -> dict[float, float]:
    """Randomly-shifted 'placebo' levels — the control arm.

    Generates ``n_ratios`` levels uniformly distributed in [0.10, 0.90] of the
    swing range, seeded deterministically. The seed is swing-specific to avoid
    all placebo sets collapsing to the same relative positions. These are NOT
    Fibonacci ratios; they represent the null hypothesis that *any* level
    within the swing range would show the same bounce rate as Fibonacci levels.
    """
    rng = np.random.default_rng(seed)
    span = peak - trough
    ratios = sorted(rng.uniform(0.10, 0.90, n_ratios))
    return {r: trough + r * span for r in ratios}


def round_number_levels(
    close: float,
    n_levels: int = 5,
    step: float | None = None,
) -> list[float]:
    """Round-number support/resistance levels near the current price.

    Returns the nearest ``n_levels`` round numbers both above and below
    ``close``. The step is auto-scaled if not provided.
    """
    if step is None:
        if close < 50:
            step = 1.0
        elif close < 200:
            step = 5.0
        elif close < 500:
            step = 10.0
        elif close < 2000:
            step = 50.0
        else:
            step = 100.0

    base = round(close / step) * step
    levels = [base + i * step for i in range(-n_levels, n_levels + 1)]
    return [lv for lv in levels if lv > 0]


# ---------------------------------------------------------------------------
# Swing detection — rolling peak/trough approach
# ---------------------------------------------------------------------------
def find_swings(
    bars: pd.DataFrame,
    lookback: int = 60,
    min_swing_pct: float = 0.05,
    step: int = 5,
) -> pd.DataFrame:
    """Identify significant local peak-to-trough swings.

    For each bar *t* (sampled every ``step`` bars for speed), looks back
    ``lookback`` bars to find the highest high (peak) and lowest low (trough).
    A swing is recorded only when the peak-to-trough move is at least
    ``min_swing_pct`` (e.g. 0.05 = 5%). Returns a DataFrame of swings with
    columns ``peak``, ``trough``, ``swing_size``, and ``direction``
    (+1 = price approaching from above, i.e. peak is the recent extreme;
    -1 = price approaching from below, i.e. trough is the recent extreme).

    Indexed by the bar at which each swing is identified. No look-ahead.
    """
    highs = bars["high"].to_numpy()
    lows = bars["low"].to_numpy()
    n = len(bars)

    records = []
    for t in range(lookback, n, step):
        peak = float(highs[t - lookback: t + 1].max())
        trough = float(lows[t - lookback: t + 1].min())
        swing_size = (peak - trough) / peak if peak > 0 else 0.0
        if swing_size < min_swing_pct:
            continue
        peak_idx = int(highs[t - lookback: t + 1].argmax())
        trough_idx = int(lows[t - lookback: t + 1].argmin())
        # +1 = peak is more recent than trough → price declining toward trough
        # -1 = trough is more recent than peak → price rallying toward peak
        direction = +1 if peak_idx > trough_idx else -1
        records.append({
            "date": bars.index[t],
            "bar_i": t,
            "peak": peak,
            "trough": trough,
            "swing_size": swing_size,
            "direction": direction,
        })

    if not records:
        return pd.DataFrame(columns=["date", "bar_i", "peak", "trough",
                                     "swing_size", "direction"])
    return pd.DataFrame(records).set_index("date")


# ---------------------------------------------------------------------------
# Vectorised bounce engine — fast enough for 25 years of daily data
# ---------------------------------------------------------------------------
def run_bounces(
    bars: pd.DataFrame,
    swings: pd.DataFrame,
    fwd_bars: int = 5,
    tolerance_pct: float = 0.005,
    use_placebo: bool = False,
    cost_bps: float = 1.0,
    ratios=None,
    max_fwd_days: int = 40,
    max_swings: int | None = None,
) -> pd.DataFrame:
    """For each swing, find the first level touch in the forward window.

    For each swing in ``swings``, computes the Fibonacci (or placebo) levels,
    then scans the next ``max_fwd_days`` bars to find the *first* bar where
    price enters the tolerance band around any level. The forward return is
    measured from that touch bar over the next ``fwd_bars`` days.

    ``use_placebo=True`` substitutes randomly-placed placebo levels (the control).
    ``cost_bps`` is subtracted from each bounce return. A touch requires the bar's
    high-low range to contain the level OR the close to be within
    ``tolerance_pct * level`` of it. One entry per (swing, level) approach.

    ``max_swings`` caps the number of swings processed (for speed in notebooks).
    Returns a per-touch ledger.
    """
    if ratios is None:
        ratios = FIB_KEY

    close_arr = bars["close"].to_numpy()
    high_arr = bars["high"].to_numpy()
    low_arr = bars["low"].to_numpy()
    n = len(bars)
    pos = {ts: i for i, ts in enumerate(bars.index)}

    swing_list = swings
    if max_swings is not None and len(swing_list) > max_swings:
        # Sample evenly to cover the full date range
        idx = np.round(np.linspace(0, len(swing_list) - 1, max_swings)).astype(int)
        swing_list = swing_list.iloc[idx]

    rows = []
    for swing_date, sw in swing_list.iterrows():
        pk, tr, d = float(sw["peak"]), float(sw["trough"]), int(sw["direction"])
        span = pk - tr
        if span <= 0:
            continue
        t0 = int(sw["bar_i"])
        if t0 + fwd_bars + 1 >= n:
            continue

        seed = int(abs(hash((round(pk, 4), round(tr, 4)))) % (2 ** 31))
        if use_placebo:
            levels = placebo_levels(pk, tr, seed=seed, n_ratios=len(ratios))
        else:
            levels = fib_levels(pk, tr, ratios=ratios)

        # Forward scan window: t0+1 to t0+max_fwd_days (capped at end of tape)
        win_end = min(t0 + max_fwd_days, n - fwd_bars - 1)
        if win_end <= t0:
            continue
        w_hi = high_arr[t0 + 1: win_end + 1]
        w_lo = low_arr[t0 + 1: win_end + 1]
        w_cl = close_arr[t0 + 1: win_end + 1]

        for ratio, lv in levels.items():
            tol = tolerance_pct * lv
            # Touch: bar's high-low range brackets the level, or close within tol
            touched = ((w_lo <= lv) & (lv <= w_hi)) | (np.abs(w_cl - lv) <= tol)
            first_touch = int(np.argmax(touched)) if touched.any() else -1
            if first_touch < 0:
                continue

            bar_i = t0 + 1 + first_touch
            if bar_i + fwd_bars >= n:
                continue
            entry_px = close_arr[bar_i]
            exit_px = close_arr[bar_i + fwd_bars]
            ret_gross = d * (exit_px - entry_px) / entry_px
            rows.append({
                "swing_date": swing_date,
                "touch_date": bars.index[bar_i],
                "ratio": ratio,
                "level_price": lv,
                "direction": d,
                "entry": entry_px,
                "exit": exit_px,
                "ret_gross": float(ret_gross),
                "ret_net": float(ret_gross) - cost_bps * 1e-4,
                "bounced": ret_gross > 0,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["swing_date", "touch_date", "ratio", "level_price", "direction",
                 "entry", "exit", "ret_gross", "ret_net", "bounced"]
    )


# ---------------------------------------------------------------------------
# Round-number bounce engine — vectorised
# ---------------------------------------------------------------------------
def run_round_number_bounces(
    bars: pd.DataFrame,
    fwd_bars: int = 5,
    tolerance_pct: float = 0.005,
    cost_bps: float = 1.0,
    use_placebo: bool = False,
) -> pd.DataFrame:
    """Find round-number level touches and measure forward bounce returns.

    Identifies round-number levels near the current price at each bar. For each
    bar's high-low range that brackets a round number (or close within tolerance),
    records the forward ``fwd_bars``-day return in the direction away from the
    level (the expected bounce direction).

    For the control arm (``use_placebo=True``), off-round levels (midpoints
    between adjacent round numbers) are used instead.
    """
    close = bars["close"].to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    n = len(bars)
    rows = []

    for i in range(1, n - fwd_bars):
        cl = float(close[i])
        hi_bar = float(high[i])
        lo_bar = float(low[i])

        lvls = round_number_levels(cl, n_levels=3)
        if use_placebo:
            step = (lvls[1] - lvls[0]) if len(lvls) >= 2 else 5.0
            lvls = [lv + 0.5 * step for lv in lvls]

        for lv in lvls:
            tol = tolerance_pct * lv
            touched = (lo_bar <= lv <= hi_bar) or (abs(cl - lv) <= tol)
            if not touched:
                continue
            prev_cl = float(close[i - 1])
            d = +1 if prev_cl < lv else -1
            entry_px = cl
            exit_px = float(close[i + fwd_bars])
            ret_gross = d * (exit_px - entry_px) / entry_px
            rows.append({
                "touch_date": bars.index[i],
                "level_price": lv,
                "direction": d,
                "entry": entry_px,
                "exit": exit_px,
                "ret_gross": float(ret_gross),
                "ret_net": float(ret_gross) - cost_bps * 1e-4,
                "bounced": ret_gross > 0,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["touch_date", "level_price", "direction", "entry", "exit",
                 "ret_gross", "ret_net", "bounced"]
    )


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summarize(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline per-touch statistics for one ledger.

    Returns touch count, bounce-rate, mean forward return (bps/touch), skew,
    and a HAC t-stat on the mean — the inference-bar number that decides whether
    any Fibonacci specificity is distinguishable from zero.
    """
    if ledger.empty or col not in ledger.columns:
        return {
            "n_touches": 0,
            "bounce_rate": float("nan"),
            "mean_bps": float("nan"),
            "tstat": float("nan"),
            "skew": float("nan"),
        }
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    out = {
        "n_touches": int(n),
        "bounce_rate": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "skew": float(pd.Series(r).skew()) if n > 2 else float("nan"),
        "tstat": float("nan"),
    }
    if n > 5:
        mu = r.mean()
        e = r - mu
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        lrv = float(e @ e) / n
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        out["tstat"] = float(mu / se) if se > 0 else float("nan")
    return out
