"""Data layer for Study 351 — BTC 5-minute Polymarket momentum.

Two sources, both offline-friendly:

* **Real tape.** 1-minute BTCUSDT bars (Binance public klines, no key) cached under
  ``_cache/btc_1m_45d.csv``. We fold them into the Polymarket object: non-overlapping
  **5-minute windows** aligned to the wall clock (minute % 5 == 0), the exact unit a
  ``btc-updown-5m-<unix>`` market resolves on (UP if close >= open). Each window keeps the
  open, the price at *2 minutes left* (the strategy's entry snapshot), at *1 minute left*,
  and the final close.
* **Synthetic.** A deterministic geometric-Brownian window generator (fixed seed) used as
  the positive control: with a known per-minute volatility the continuation probability has
  a closed form, ``Phi(move / (sigma * sqrt(minutes_left)))``, so we can check the engine
  measures what theory predicts — and that the *fair* binary price equals that probability,
  which is the whole tradability argument.

Plus a tiny loader for the live Polymarket capture (``_cache/live_log.csv``) produced by the
read-only paper-trade scaffold (real CLOB asks vs realised outcomes; zero key, zero money).

Pure numpy + stdlib; never touches the network.
"""

from __future__ import annotations

import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "btc_1m_45d.csv")
LIVE_CACHE = os.path.join(HERE, "..", "_cache", "live_log.csv")
MIN_MS = 60_000


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def load_bars(path: str = DEFAULT_CACHE) -> dict[int, tuple[float, float, float, float]]:
    """Return ``{open_time_ms: (open, high, low, close)}`` from the cached klines CSV."""
    bars: dict[int, tuple[float, float, float, float]] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            t = int(row["open_time"])
            bars[t] = (float(row["open"]), float(row["high"]),
                       float(row["low"]), float(row["close"]))
    return bars


def build_windows(bars: dict[int, tuple[float, float, float, float]]) -> np.ndarray:
    """Fold 1-minute bars into 5-minute Polymarket windows.

    Returns an ``(n, 4)`` array of columns ``[open, px@2min-left, px@1min-left, close]``.
    A window starts on a minute divisible by 5 and needs all five contiguous bars.
    """
    rows = []
    for t in sorted(bars):
        if (t // MIN_MS) % 5 != 0:
            continue
        seq = [bars.get(t + i * MIN_MS) for i in range(5)]
        if any(b is None for b in seq):
            continue
        o = seq[0][0]          # window open
        c2 = seq[2][3]         # close at 3 min elapsed  -> 2 minutes left
        c3 = seq[3][3]         # close at 4 min elapsed  -> 1 minute left
        cf = seq[4][3]         # final close (resolution price)
        rows.append((o, c2, c3, cf))
    return np.asarray(rows, dtype=float)


def load_real(path: str = DEFAULT_CACHE) -> np.ndarray:
    """Convenience: bars -> windows in one call."""
    return build_windows(load_bars(path))


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_windows(n: int = 12_000, sigma_per_min: float = 45.0,
                      drift_per_min: float = 0.0, start_px: float = 64_000.0,
                      seed: int = 351) -> np.ndarray:
    """Deterministic 5-minute windows from an arithmetic Brownian path.

    ``sigma_per_min`` is the per-minute std of the BTC move in *dollars*. With no drift the
    continuation probability at a $D snapshot with ``k`` minutes left is exactly
    ``Phi(D / (sigma_per_min * sqrt(k)))`` — see :func:`strategy.theoretical_continuation`.
    Returns the same ``[open, px@2left, px@1left, close]`` layout as :func:`build_windows`.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift_per_min, sigma_per_min, size=(n, 5))
    path = start_px + np.cumsum(steps, axis=1)          # prices after each of the 5 minutes
    o = np.full(n, start_px)
    c2 = path[:, 2]      # after 3 minutes -> 2 left
    c3 = path[:, 3]      # after 4 minutes -> 1 left
    cf = path[:, 4]      # final
    return np.column_stack([o, c2, c3, cf])


# --------------------------------------------------------------------------- #
# Live Polymarket capture
# --------------------------------------------------------------------------- #
def load_live(path: str = LIVE_CACHE) -> list[dict]:
    """Resolved rows from the live paper-trade log (favored ask vs realised outcome)."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("favored_win") not in ("True", "False"):
                continue
            out.append({
                "abs_move": float(r["abs_move"]),
                "signal70": r["signal70"] == "True",
                "favored_ask": float(r["favored_ask"]),
                "favored_win": r["favored_win"] == "True",
            })
    return out
