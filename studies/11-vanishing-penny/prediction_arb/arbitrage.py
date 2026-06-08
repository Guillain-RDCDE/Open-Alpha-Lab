"""From a gap series to **episodes**, and from episodes to a **half-life**.

An arbitrage *episode* is one life of the free penny: a shock opens the gap above a
threshold, the gap then bleeds back toward zero as arbitrageurs close it, and the episode
ends when it falls back under threshold. The single number the whole study turns on is the
**half-life** — how many minutes until the mispricing has shrunk by half — estimated two
deliberately different ways so the verdict doesn't lean on one functional form:

    * :func:`time_to_half` — assumption-light: per episode, count the steps from the peak
      until the gap is half its peak size; take the **median** across episodes. For a pure
      exponential this *is* the half-life, with no curve fit.
    * :func:`fit_half_life` — a pooled **log-linear** decay fit through the post-peak points
      of every episode at once: ``log(g/g_peak) = -(ln 2 / H)·Δt``. Cross-checks the median.

Agreement between the two is the tell that the decay is genuinely exponential (an
arbitraged gap closing at a rate proportional to its size); disagreement would flag a
fat tail of slow-closers the median hides — which the survival curve in :mod:`robustness`
then shows directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Episode:
    """One arbitrage opening: open → peak → close, with the decay points kept for fitting."""
    market: str
    t_open: pd.Timestamp
    t_peak: pd.Timestamp
    peak: float            # signed gap at the peak (the guaranteed $/$1 you could lock)
    sign: int              # +1 = buy-both (p_yes+p_no<1), -1 = sell-both
    duration: int          # steps from open until the gap falls back under threshold
    time_to_half: float    # steps from peak until |g| <= |peak|/2 (nan if it closes first)
    values: np.ndarray     # the signed gap over the episode, for the decay fit


def detect_episodes(
    series: pd.Series,
    open_threshold: float = 0.03,
    close_threshold: float | None = None,
    name: str | None = None,
) -> list[Episode]:
    """Carve one market's gap series into episodes (contiguous runs above threshold).

    An episode is a maximal run where ``|g| >= close_threshold`` that *peaks* at or above
    ``open_threshold`` — so a gap has to genuinely open (clear ``open_threshold``) to count,
    but we follow its full decay down to ``close_threshold`` (default: same as open) before
    calling it closed. ``time_to_half`` is measured from the peak; it is ``nan`` when the
    gap decays back under threshold before reaching half its peak (a peak only marginally
    above threshold), so those marginal episodes don't bias the half-life downward.
    """
    close_threshold = open_threshold if close_threshold is None else close_threshold
    g = series.to_numpy(dtype=float)
    absg = np.abs(g)
    n = len(g)
    out: list[Episode] = []
    i = 0
    while i < n:
        if absg[i] < close_threshold:
            i += 1
            continue
        j = i
        while j < n and absg[j] >= close_threshold:
            j += 1
        seg, absseg = g[i:j], absg[i:j]
        pk = int(np.argmax(absseg))
        peak = float(seg[pk])
        if abs(peak) >= open_threshold:
            half = abs(peak) / 2.0
            tth = np.nan
            for k in range(pk, len(absseg)):
                if absseg[k] <= half:
                    tth = float(k - pk)
                    break
            out.append(Episode(
                market=name if name is not None else (series.name or "?"),
                t_open=series.index[i], t_peak=series.index[i + pk],
                peak=peak, sign=int(np.sign(peak)), duration=j - i,
                time_to_half=tth, values=seg.copy(),
            ))
        i = j
    return out


def detect_all(gap: pd.DataFrame, open_threshold: float = 0.03,
               close_threshold: float | None = None) -> list[Episode]:
    """Every episode across every column of a gap panel."""
    eps: list[Episode] = []
    for col in gap.columns:
        eps.extend(detect_episodes(gap[col].dropna(), open_threshold,
                                   close_threshold, name=col))
    return eps


def time_to_half(episodes: list[Episode]) -> float:
    """The assumption-light half-life: median steps-to-half across episodes (drops nans)."""
    vals = np.array([e.time_to_half for e in episodes], dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else float("nan")


def fit_half_life(episodes: list[Episode], min_points: int = 3) -> float:
    """Pooled log-linear half-life: regress ``log(|g|/|g_peak|)`` on Δt-since-peak, no intercept.

    Each episode contributes its post-peak decay, centred so it passes through the origin
    at the peak; the common slope ``b`` is ``log(½)/H`` per step, so ``H = -ln 2 / b``.
    Returns ``nan`` if there aren't enough decaying points or the pooled slope is non-negative
    (no decay to speak of).
    """
    xs, ys = [], []
    for e in episodes:
        absv = np.abs(e.values)
        pk = int(np.argmax(absv))
        post = absv[pk:]
        peakv = absv[pk]
        mask = post > 1e-6
        if mask.sum() < min_points or peakv <= 0:
            continue
        dt = np.arange(pk, len(absv))[mask] - pk
        logc = np.log(post[mask]) - np.log(peakv)
        xs.append(dt.astype(float))
        ys.append(logc)
    if not xs:
        return float("nan")
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    denom = float(np.dot(x, x))
    if denom <= 0:
        return float("nan")
    b = float(np.dot(x, y) / denom)
    if b >= 0:
        return float("nan")
    return float(-np.log(2.0) / b)


def summary(episodes: list[Episode]) -> dict:
    """Headline episode statistics: count, the two half-lives, median peak penny, duration."""
    if not episodes:
        return {"n_episodes": 0}
    peaks = np.array([abs(e.peak) for e in episodes])
    durs = np.array([e.duration for e in episodes], dtype=float)
    n_floored = sum(not np.isfinite(e.time_to_half) for e in episodes)
    return {
        "n_episodes": len(episodes),
        "half_life_median_min": time_to_half(episodes),
        "half_life_fit_min": fit_half_life(episodes),
        # fraction of episodes too short to even *observe* a halving at this sampling — the
        # gap is back under threshold by the next sample. On a tape this is the headline:
        # a high frac_below_floor means the true half-life is below the sampling interval.
        "frac_below_floor": float(n_floored / len(episodes)),
        "median_peak_penny": float(np.median(peaks)),
        "median_duration_min": float(np.median(durs)),
        "frac_buy_both": float(np.mean([e.sign > 0 for e in episodes])),
    }
