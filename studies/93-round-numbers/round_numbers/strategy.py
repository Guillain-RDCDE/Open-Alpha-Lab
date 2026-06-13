"""The two tests and their honest controls — Study 93 (Round-Numbers).

The folk claim has two limbs, and the desk's job is to keep them apart:

- **(a) Clustering.** *"Prices are magnetised by round numbers — they pile up at and near
  whole-dollar / round levels."* This is a static property of the price distribution; we
  test it with a chi-square on the distance from each close to its nearest round level
  (:func:`clustering_test`). This limb is well documented (Harris 1991) and likely **real**.
- **(b) Tradability.** *"Fade an approach to a round number and collect the bounce."* This
  is a *forecasting* claim and a much taller order. We implement the fade as a fixed-horizon
  event study (:func:`fade_returns`): when the close sits within a small band of a round
  level, take the side that bets on a reversal away from the level, hold ``horizon`` days,
  and record the signed return **demeaned by the tape's own drift** (so the test measures
  reversal, not the market's trend). The edge is then pinned against a **random-level
  control** (:func:`random_level_grid`) that fades the *same number of times* near *random*
  thresholds — exactly the desk's "beats a coin?" control (cf. the matched-coin in Study 91
  and the random-direction control in Study 87). If the round levels carry no information,
  the real fade should not beat the random-level fade.

The execution convention is the desk's: proximity to a round level is known at the close of
day *t*, so the fade is *entered* at *t+1* and earns the return from *t+1* onward — one lag,
applied once, baked into :func:`fade_returns` (it enters at ``close[i+1]``). Transaction
costs are charged per round-trip in :func:`net_of_costs`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# (a) Clustering — distance to the nearest round level
# ---------------------------------------------------------------------------
def distance_to_round(close: np.ndarray, grid: float) -> np.ndarray:
    """Unsigned distance from each price to its nearest round level, as a fraction of grid.

    For ``grid = 1`` (whole dollars) this is ``min(frac, 1-frac)`` of the price's
    fractional part, in ``[0, 0.5]``; for a general grid it is the same on the ``price/grid``
    scale. Round-number magnetism shows up as **excess mass near 0**.
    """
    u = np.asarray(close, dtype=float) / grid
    frac = u - np.floor(u)
    return np.minimum(frac, 1.0 - frac)


def clustering_test(close: np.ndarray, grid: float, bins: int = 10) -> dict:
    """Chi-square test that the distance-to-nearest-round-level is uniform.

    Bins the (folded) distance into ``bins`` equal cells on ``[0, 0.5]`` and runs a
    chi-square against the uniform expectation. Returns the statistic, p-value, the bin
    counts, and the empirical fraction of closes within 5% of a round level (uniform
    expectation: ``0.10`` for the default 10 bins). A large statistic / tiny p-value with
    excess mass in the first bin is the clustering signature.
    """
    dist = distance_to_round(close, grid)
    counts, _ = np.histogram(dist, bins=bins, range=(0.0, 0.5))
    chi2, p = stats.chisquare(counts)
    near = float((dist < 0.05).mean())
    return {
        "chi2": float(chi2),
        "p": float(p),
        "counts": counts.tolist(),
        "n": int(dist.size),
        "frac_near": near,
        "expected_near": 1.0 / bins,
    }


# ---------------------------------------------------------------------------
# (b) Tradability — the round-number fade as a fixed-horizon event study
# ---------------------------------------------------------------------------
def round_grid(close: np.ndarray, grid: float) -> np.ndarray:
    """The nearest round level (multiple of ``grid``) for each price — the real thresholds."""
    return np.round(np.asarray(close, dtype=float) / grid) * grid


def random_level_grid(close: np.ndarray, grid: float, seed: int = 0) -> np.ndarray:
    """A *random-level* threshold grid: the same spacing, shifted by a random phase.

    The control fades a grid of levels spaced exactly like the round numbers but offset by a
    uniformly random phase in ``[0, grid)`` — so it triggers a similar number of trades at
    *arbitrary* price levels rather than the psychologically special round ones. If the round
    levels carry no forecasting information, fading them should be no better than fading these.
    """
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, grid)
    c = np.asarray(close, dtype=float)
    return np.round((c - phase) / grid) * grid + phase


def fade_returns(
    close: pd.Series | np.ndarray,
    grid: float,
    levels: np.ndarray | None = None,
    band: float = 0.004,
    horizon: int = 1,
) -> np.ndarray:
    """Signed, drift-demeaned returns of fading proximity to a level grid.

    On each day ``i`` whose close is within ``band`` (fraction of price) of its nearest level
    in ``levels`` (defaulting to the round grid), take the **reversal side**: short if price is
    *below* the level (betting it stalls and falls back), long if *above* (betting it bounces
    up). Enter at ``close[i+1]`` (one-day lag, no peeking), hold ``horizon`` days, and record
    the signed return **minus the tape's mean drift over the same horizon** — so a positive
    mean is a genuine reversal edge, not the market's trend leaking in.
    """
    c = np.asarray(close.to_numpy() if isinstance(close, pd.Series) else close, dtype=float)
    n = c.size
    if levels is None:
        levels = round_grid(c, grid)
    mu = float(np.diff(np.log(c)).mean()) if n > 2 else 0.0  # mean daily log-drift
    out: list[float] = []
    for i in range(1, n - horizon - 1):
        d = (c[i] - levels[i]) / c[i]
        if abs(d) < band:
            side = -1.0 if d < 0 else 1.0
            raw = (c[i + 1 + horizon] - c[i + 1]) / c[i + 1]
            out.append(side * (raw - mu * horizon))
    return np.asarray(out, dtype=float)


def net_of_costs(trade_returns: np.ndarray, cost_bps: float = 5.0) -> np.ndarray:
    """Charge a round-trip cost on each fade (one-way ``cost_bps`` per leg, two legs)."""
    return np.asarray(trade_returns, dtype=float) - 2.0 * cost_bps * 1e-4


# ---------------------------------------------------------------------------
# Pooling across the basket + the random-level control
# ---------------------------------------------------------------------------
def pooled_fade(
    names: list[tuple[str, float, pd.Series]],
    band: float = 0.004,
    horizon: int = 1,
) -> np.ndarray:
    """Pool the real round-number fade trades across every (ticker, grid, close) in ``names``."""
    chunks = [fade_returns(close, grid, band=band, horizon=horizon) for _, grid, close in names]
    return np.concatenate(chunks) if chunks else np.array([])


def random_level_pool_means(
    names: list[tuple[str, float, pd.Series]],
    n_seeds: int = 200,
    band: float = 0.004,
    horizon: int = 1,
) -> np.ndarray:
    """Per-seed pooled mean trade return when the same basket fades *random* levels.

    For each seed, every name fades a phase-shifted (random) grid of the same spacing; the
    trades are pooled and the mean recorded. The resulting distribution of means is the null
    the real round-number fade must beat to claim the round levels carry information.
    """
    means = np.empty(n_seeds)
    for s in range(n_seeds):
        chunks = [
            fade_returns(close, grid, levels=random_level_grid(close.to_numpy(), grid, seed=s),
                         band=band, horizon=horizon)
            for _, grid, close in names
        ]
        pool = np.concatenate(chunks) if chunks else np.array([0.0])
        means[s] = float(pool.mean())
    return means


# ---------------------------------------------------------------------------
# Inference: HAC t-stat on a return series
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) t-stat for the mean of ``x`` (local, no quantlab dep)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 5:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")
