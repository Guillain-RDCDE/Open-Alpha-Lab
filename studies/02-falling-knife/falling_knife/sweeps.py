"""Parameter sweeps: is -3% special, or did we just pick a round number?

The project is named after a -3% threshold, but a single threshold proves
nothing. If the (non-)edge is real it should vary *smoothly* and sensibly as we
move the knob; if a lone value lights up while its neighbours are flat, that value
is data-mined, not meaningful.

Two sweeps:
    * :func:`threshold_sweep` — vary the drop size (-2% ... -5%) for a given
      trigger and read the conditional excess-vs-random at each horizon.
    * :func:`window_sweep` — vary N for the window-based triggers (T3 drawdown
      look-back, T4 cumulative horizon).

Both report the *excess over a random day* and its permutation p-value, never the
absolute return — same yardstick as :mod:`falling_knife.benchmark`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import benchmark, triggers


def threshold_sweep(
    ohlc: pd.DataFrame,
    returns: pd.DataFrame,
    trigger_fn=triggers.close_to_close,
    thresholds=(-0.02, -0.03, -0.04, -0.05, -0.07),
    horizons=(1, 5, 10, 20),
    cooldown: int = 20,
    n_iter: int = 1500,
    seed: int = 0,
) -> dict:
    """Excess return and p-value across drop thresholds x horizons.

    ``trigger_fn`` must accept ``(returns, threshold)`` (T1/T2, or a lambda that
    fixes the window for T3/T4). Returns a dict of DataFrames indexed by threshold:
        ``excess``    — mean conditional minus unconditional forward return
        ``p_greater`` — permutation prob a random basket does at least as well
        ``n_events``  — fresh events at that threshold
    """
    exc, pval, nev = {}, {}, {}
    for thr in thresholds:
        raw = trigger_fn(returns, thr)
        events = triggers.first_crossings(raw, cooldown=cooldown)
        bench = benchmark.conditional_vs_unconditional(
            ohlc, events, horizons=horizons, n_iter=n_iter, seed=seed
        )
        key = f"{thr:.0%}"
        exc[key] = {h: (bench.loc[h, "excess"] if h in bench.index else np.nan) for h in horizons}
        pval[key] = {h: (bench.loc[h, "p_greater"] if h in bench.index else np.nan) for h in horizons}
        nev[key] = int(bench["n_events"].iloc[0]) if len(bench) else 0

    excess_df = pd.DataFrame(exc).T
    excess_df.index.name = "threshold"
    pval_df = pd.DataFrame(pval).T
    pval_df.index.name = "threshold"
    counts = pd.Series(nev, name="n_events")
    counts.index.name = "threshold"
    return {"excess": excess_df, "p_greater": pval_df, "n_events": counts}


def window_sweep(
    ohlc: pd.DataFrame,
    returns: pd.DataFrame,
    kind: str = "drawdown",
    windows=(5, 10, 20, 50, 100),
    threshold: float = -0.03,
    horizon: int = 5,
    cooldown: int = 20,
    n_iter: int = 1500,
    seed: int = 0,
) -> pd.DataFrame:
    """Excess vs random at one horizon as the trigger's look-back N varies.

    ``kind`` is ``'drawdown'`` (T3 rolling-high window) or ``'cumulative'``
    (T4 N-day return). Returns a DataFrame indexed by N with ``excess,
    p_greater, n_events``.
    """
    rows = []
    for w in windows:
        if kind == "drawdown":
            raw = triggers.drawdown(returns, threshold, window=w)
        elif kind == "cumulative":
            raw = triggers.cumulative(returns, threshold, n_days=w)
        else:
            raise ValueError("kind must be 'drawdown' or 'cumulative'")
        events = triggers.first_crossings(raw, cooldown=cooldown)
        bench = benchmark.conditional_vs_unconditional(
            ohlc, events, horizons=(horizon,), n_iter=n_iter, seed=seed
        )
        if horizon in bench.index:
            r = bench.loc[horizon]
            rows.append({"N": w, "excess": r["excess"], "p_greater": r["p_greater"],
                         "n_events": int(r["n_events"])})
        else:
            rows.append({"N": w, "excess": np.nan, "p_greater": np.nan, "n_events": 0})
    return pd.DataFrame(rows).set_index("N")
