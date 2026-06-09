"""Koroush AK's three "optimal environment" filters, mechanized — and the test of whether they pay.

The claim is that breakouts only follow through when (1) price *grinds* into the level in a slow
staircase rather than a vertical spike, (2) volume *builds* into the move, and (3) the trend is
*clean* — few moving-average crossovers. Each is a discretionary chart judgment; we turn each into a
single number read off the approach window, so a trade can be graded and the filtered subset
compared against the whole:

    * :func:`grind_score` — staircase vs spike. The fraction of the run-up *not* concentrated in its
      single biggest bar: ``1 − max_bar_gain / total_gain``. Near 1 = gains spread across many bars
      (a grind); near 0 = one explosive candle did all the work (a spike). Koroush wants the grind.
    * :func:`volume_slope` — is participation building? The OLS slope of volume across the approach
      window, normalized by mean volume. Positive = rising into the break (what he wants); flat or
      negative = no conviction.
    * :func:`crossover_count` — chop vs trend. The number of times price crosses its 30-bar smoothed
      moving average (SMMA, Koroush's named indicator) over the window. Few crossings = a directional
      trend (good); many = indecision (bad). Returned *negated* so "higher is better" holds for all
      three filters uniformly.

The honest test (run in the notebooks): on the **null tape** these scores are functions of noise, so
conditioning on them must *not* lift the win rate above the coin-flip baseline — it can only shrink
the sample (the selection illusion that makes a thinner equity curve look "cleaner"). On a tape where
continuation is genuinely *gated on a grind* (``data.synthetic_intraday(cont_requires_grind=True)``)
the grind score *should* recover real edge — proof the filters can work when there is something to
read, and therefore that their failure on the real claim is a finding, not a rigged test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def smma(series: pd.Series, length: int = 30) -> pd.Series:
    """Wilder's smoothed moving average — an EMA with ``alpha = 1/length`` (Koroush's "30 SMMA")."""
    return series.ewm(alpha=1.0 / length, adjust=False).mean()


def grind_score(bars: pd.DataFrame, idx: int, window: int = 20) -> float:
    """Staircase-vs-spike score in [0,1] for the approach ending at ``idx``. Higher = grindier."""
    lo = max(0, idx - window + 1)
    close = bars["Close"].iloc[lo: idx + 1].to_numpy()
    if len(close) < 3:
        return float("nan")
    steps = np.diff(close)
    up = steps[steps > 0].sum()
    if up <= 0:
        return 0.0
    biggest = steps.max()
    return float(np.clip(1.0 - biggest / up, 0.0, 1.0))


def volume_slope(bars: pd.DataFrame, idx: int, window: int = 20) -> float:
    """Normalized OLS slope of volume over the approach window. Positive = building participation."""
    lo = max(0, idx - window + 1)
    v = bars["Volume"].iloc[lo: idx + 1].to_numpy()
    if len(v) < 3 or v.mean() == 0:
        return float("nan")
    x = np.arange(len(v))
    slope = np.polyfit(x, v, 1)[0]
    return float(slope / v.mean())


def crossover_count(bars: pd.DataFrame, idx: int, window: int = 20, length: int = 30) -> float:
    """*Negated* count of price×SMMA(30) crossings over the window — higher (fewer crossings) = cleaner."""
    lo = max(0, idx - window + 1 - length)        # warm up the SMMA before the window
    seg = bars["Close"].iloc[lo: idx + 1]
    if len(seg) < length + 3:
        return float("nan")
    line = smma(seg, length)
    sign = np.sign((seg - line).to_numpy())
    sign = sign[-window:]
    crossings = int((np.diff(sign) != 0).sum())
    return float(-crossings)


def annotate(trades: pd.DataFrame, bars: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Attach the three filter scores to each trade, read off the bars *before* entry (no look-ahead)."""
    if trades.empty:
        out = trades.copy()
        for c in ("grind", "vol_slope", "clean"):
            out[c] = pd.Series(dtype="float64")
        return out
    out = trades.copy()
    idx = out["entry_idx"].astype(int)
    out["grind"] = [grind_score(bars, i, window) for i in idx]
    out["vol_slope"] = [volume_slope(bars, i, window) for i in idx]
    out["clean"] = [crossover_count(bars, i, window) for i in idx]
    return out


def passes(annotated: pd.DataFrame, quantile: float = 0.5) -> pd.Series:
    """Boolean: a trade passes all three filters if each score is in the *better* half of its own
    distribution (above the ``quantile`` cut). A relative cut keeps the A-grade subset comparable in
    spirit to "only take the cleanest setups" without hand-picked thresholds.
    """
    if annotated.empty:
        return pd.Series(dtype=bool)
    ok = pd.Series(True, index=annotated.index)
    for c in ("grind", "vol_slope", "clean"):
        cut = annotated[c].quantile(quantile)
        ok &= annotated[c] >= cut
    return ok.rename("passes")


def filter_lift(trades: pd.DataFrame, bars: pd.DataFrame, window: int = 20,
                quantile: float = 0.5) -> dict:
    """Does the A-grade (all-filters-pass) subset beat the field? Win rates, sizes, and the lift.

    The headline is ``win_rate_filtered − win_rate_all`` together with the *shrinkage* in trade count.
    A lift indistinguishable from zero while ``n`` collapses is the signature of a decorative filter:
    it makes the survivors *look* selective without adding expectancy.
    """
    ann = annotate(trades, bars, window)
    resolved = ann[ann["outcome"] != 0]
    if resolved.empty:
        return {"n_all": 0, "n_filtered": 0, "win_rate_all": float("nan"),
                "win_rate_filtered": float("nan"), "lift": float("nan"), "kept_frac": float("nan")}
    ok = passes(resolved, quantile)
    sub = resolved[ok]
    wr_all = float((resolved["outcome"] == 1).mean())
    wr_sub = float((sub["outcome"] == 1).mean()) if len(sub) else float("nan")
    return {
        "n_all": int(len(resolved)),
        "n_filtered": int(len(sub)),
        "kept_frac": float(len(sub) / len(resolved)),
        "win_rate_all": wr_all,
        "win_rate_filtered": wr_sub,
        "lift": float(wr_sub - wr_all),
    }
