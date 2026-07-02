"""Data layer for Study 574 (Penny-Beat) — the earnings-surprise discontinuity.

The raw material of a penny-beat test is a panel of, per firm-quarter:

- the **consensus EPS forecast** (the mean/median analyst estimate just before the report), and
- the **actual reported EPS**,

so that the *surprise* ``s = actual - consensus`` (in dollars, or in cents) can be histogrammed.
The penny-beat tell is a **discontinuity** in that histogram: a deficit of observations just below
zero and an excess spike at exactly ``s = +$0.01`` — firms that managed a small miss into a small
beat (Burgstahler & Dichev 1997; Degeorge, Patel & Zeckhauser 1999). That consensus-vs-actual
panel is a licensed I/B/E/S product; a no-key retail stack cannot reach a survivorship-free,
point-in-time version of it. So this study is **synthetic-only** and says so on the SIGNAL axis.

One schema, one generator:

- ``synthetic_panel`` — a deterministic, offline generator of a firm-quarter panel with columns
  ``surprise`` (rounded to the nearest cent, the observable), ``penny_beat`` (a 0/1 flag = did the
  firm land exactly on +$0.01), and ``forward_ret`` (the subsequent return). Two knobs plant the
  effect:

    * ``spike`` — how much mass is *moved* from just-below-zero into the +$0.01 bin (the
      earnings-management discontinuity). ``spike = 0`` gives a smooth, unmanaged surprise
      distribution (the null for the *discontinuity*).
    * ``penny_penalty`` — the extra forward-return drag on the penny-beaters (the trading claim).
      ``penny_penalty = 0`` is the null for the *return* test.

  With both knobs on, the study's positive control is one bottle; with both zero, the joint null.
  Tests never touch the network.

There is deliberately **no** real ``fetch_*`` here: the free real data does not exist for this
test, which is the whole point of the synthetic-only classification (see ``__init__`` and the
SIGNAL-axis caveat in ``docs/results.md``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

# The bin (in cents) that a "penny beat" lands in: exactly +1 cent above consensus.
PENNY_CENTS = 1

# Distribution knobs used everywhere so the synthetic world is one fixed object.
DEFAULT_N = 6000
DEFAULT_SURPRISE_SD_CENTS = 6.0   # sd of the underlying (unmanaged) surprise, in cents


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    spike: float           # fraction of just-below-zero mass moved into the +1c bin (>=0)
    penny_penalty: float   # extra forward-return drag on penny-beaters (<=0 is the claim)

    @property
    def has_discontinuity(self) -> bool:
        return self.spike > 0.0

    @property
    def has_penalty(self) -> bool:
        return self.penny_penalty != 0.0


def synthetic_panel(
    n: int = DEFAULT_N,
    spike: float = 0.55,
    penny_penalty: float = -0.05,
    surprise_sd_cents: float = DEFAULT_SURPRISE_SD_CENTS,
    idio_vol: float = 0.14,
    seed: int = 574,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible firm-quarter panel with a tunable penny-beat discontinuity + return penalty.

    Construction:

    1. Draw a *latent* (unmanaged) surprise in cents from a smooth Normal(0, ``surprise_sd_cents``)
       and round to the nearest cent — this is what an *unmanaged* population would report.
    2. **Earnings management (the ``spike`` knob).** A fraction ``spike`` of the firms whose latent
       surprise landed just below zero (in the small-miss bins −1c, −2c, −3c) are "managed": their
       reported surprise is pushed *up* to exactly +1c. This carves the tell-tale deficit below zero
       and the excess spike at +1c. ``spike = 0`` leaves the distribution smooth (no discontinuity).
    3. **Forward returns.** A baseline post-earnings drift that rises gently with the (honest)
       surprise — bigger beats drift up (a mild PEAD-like slope) — plus idiosyncratic noise. The
       *managed penny-beaters* additionally carry ``penny_penalty`` (a drag): having borrowed from
       the future, they underperform. ``penny_penalty = 0`` removes the drag (return null).

    The returned frame has one row per firm-quarter with the columns the engine consumes:
    ``surprise`` (cents, integer-valued float), ``managed`` (0/1 truth flag — hidden from the
    engine's return test except as the label it must *infer* from the +1c bin), ``penny_beat``
    (0/1: reported surprise == +1c) and ``forward_ret``.

    Returns ``(panel, truth)``.
    """
    rng = np.random.default_rng(seed)

    # 1. latent unmanaged surprise, in cents (rounded)
    latent = rng.normal(0.0, surprise_sd_cents, size=n)
    surprise = np.round(latent).astype(int)

    # 2. earnings management: move a fraction of small-misses up to exactly +1c
    small_miss = np.isin(surprise, [-1, -2, -3])
    managed = np.zeros(n, dtype=int)
    if spike > 0.0:
        draw = rng.random(n)
        pick = small_miss & (draw < spike)
        surprise = surprise.copy()
        surprise[pick] = PENNY_CENTS
        managed[pick] = 1

    penny_beat = (surprise == PENNY_CENTS).astype(int)

    # 3. forward returns: a mild slope in the honest surprise + idio, minus the penalty on managed
    pead_slope = 0.004  # per cent of surprise (gentle, capped below)
    capped = np.clip(surprise, -8, 8)
    drift = 0.02 + pead_slope * capped
    idio = idio_vol * rng.standard_normal(n)
    forward_ret = drift + idio + penny_penalty * managed

    panel = pd.DataFrame(
        {
            "surprise": surprise.astype(float),
            "managed": managed,
            "penny_beat": penny_beat,
            "forward_ret": forward_ret,
        }
    )
    panel.index.name = "firm_quarter"
    return panel, WorldTruth(spike=float(spike), penny_penalty=float(penny_penalty))


def surprise_histogram(panel: pd.DataFrame, lo: int = -12, hi: int = 12) -> pd.Series:
    """Counts of firm-quarters by integer surprise (cents), over the ``[lo, hi]`` window.

    The object the discontinuity test reads: a smooth histogram is the null; a deficit just below
    zero with an excess at +1c is the earnings-management fingerprint.
    """
    bins = np.arange(lo, hi + 1)
    counts = panel["surprise"].value_counts().reindex(bins, fill_value=0).sort_index()
    counts.index = counts.index.astype(int)
    counts.index.name = "surprise_cents"
    return counts.rename("count")


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, dict):
        blob = json.dumps(obj, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
