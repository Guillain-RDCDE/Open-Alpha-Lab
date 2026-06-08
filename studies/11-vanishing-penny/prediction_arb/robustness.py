"""The honesty layer — does the half-life hold up, and what's left for a human?

Four checks, each aimed at a way the headline number could mislead:

    * :func:`survival_curve` — the full distribution behind the median: the fraction of
      episodes still open after *t* minutes. A short half-life with a fat slow tail is a
      different animal from a uniformly fast one, and the median alone hides which.
    * :func:`bootstrap_half_life` — is the half-life distinguishable from a much slower
      one, or an artefact of a handful of episodes? A percentile CI by resampling episodes.
    * :func:`resolution_sweep` — the study's load-bearing caveat made into a number: re-run
      detection on a deliberately **coarsened** tape and watch the episode count collapse
      and the half-life inflate. What a minute tape (let alone a human) literally *cannot
      see*.
    * :func:`retail_capture` — the tradability bottom line: of the guaranteed penny at the
      peak, what fraction is still on the table once you react ``latency`` minutes later?
      Exponential decay makes this brutal, and it is the whole beat-6 answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .arbitrage import Episode, detect_all, time_to_half


def survival_curve(episodes: list[Episode], grid: np.ndarray | None = None) -> pd.DataFrame:
    """Fraction of episodes still open after *t* minutes — the distribution behind the median.

    Uses each episode's ``duration`` (steps from open until the gap falls back under
    threshold). Monotone non-increasing by construction; ``frac_open`` starts at 1.0.
    """
    durs = np.array([e.duration for e in episodes], dtype=float)
    if grid is None:
        hi = int(np.percentile(durs, 99)) + 1 if durs.size else 1
        grid = np.arange(0, max(hi, 2))
    frac = np.array([(durs > t).mean() if durs.size else 0.0 for t in grid])
    return pd.DataFrame({"t_min": grid, "frac_open": frac}).set_index("t_min")


def bootstrap_half_life(
    episodes: list[Episode],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Percentile CI for the median half-life by resampling episodes with replacement."""
    finite = [e for e in episodes if np.isfinite(e.time_to_half)]
    n = len(finite)
    point = time_to_half(episodes)
    if n == 0:
        return {"half_life_min": point, "ci_low": float("nan"),
                "ci_high": float("nan"), "n_episodes": 0, "n_boot": n_boot}
    tths = np.array([e.time_to_half for e in finite], dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.array([np.median(tths[rng.integers(0, n, n)]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "half_life_min": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_episodes": n,
        "n_boot": n_boot,
    }


def resolution_sweep(
    gap: pd.DataFrame,
    fidelities: tuple[int, ...] = (1, 2, 5, 15, 30, 60),
    open_threshold: float = 0.03,
) -> pd.DataFrame:
    """Re-detect on a coarsened tape — the half-life you *measure* vs the sampling you have.

    For each ``fidelity`` k (minutes per sample) we keep every k-th observation, re-run
    detection, and report the episode count and the measured half-life **re-expressed in
    minutes** (steps × k). As k grows the fast episodes vanish between samples: the count
    falls and the surviving half-life inflates — the precise sense in which the real
    2-second race is invisible to a minute tape, and a fortiori to a person.
    """
    rows = []
    base_n = None
    for k in fidelities:
        coarse = gap.iloc[::k]
        eps = detect_all(coarse, open_threshold=open_threshold)
        n = len(eps)
        if base_n is None:
            base_n = n
        durs = np.array([e.duration for e in eps], dtype=float)
        rows.append({
            "fidelity_min": k,
            "n_episodes": n,
            "frac_episodes_seen": float(n / base_n) if base_n else float("nan"),
            "median_duration_min": float(np.median(durs)) * k if n else float("nan"),
            "half_life_min": time_to_half(eps) * k,
        })
    return pd.DataFrame(rows).set_index("fidelity_min")


def retail_capture(half_life_min: float, latency_min: float) -> float:
    """Fraction of the peak penny still on the table after reacting ``latency_min`` later.

    Pure exponential decay: ``capture = ½^(latency / half_life)``. With a 6-minute
    half-life, a 60-second reaction already leaves ~89%, but a realistic human loop
    (spot it, log in, size both legs, submit two CLOB orders that confirm a block apart)
    is minutes, and the paper's *fast* wallets act in ~30 ms — the gap between those two
    latencies is the whole edge.
    """
    if half_life_min <= 0:
        return float("nan")
    return float(0.5 ** (latency_min / half_life_min))


def retail_capture_table(
    half_life_min: float,
    latencies_min: tuple[float, ...] = (0.0005, 0.5, 1, 2, 5, 10, 30),
) -> pd.DataFrame:
    """Capture fraction across a ladder of reaction latencies (0.5 ms bot → 30 min human)."""
    rows = [{"latency_min": lat,
             "capture_frac": retail_capture(half_life_min, lat)}
            for lat in latencies_min]
    return pd.DataFrame(rows).set_index("latency_min")
