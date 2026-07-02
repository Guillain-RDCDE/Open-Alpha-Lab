"""Data layer for Study 566 (Earnings-Call-Tone) — the linguistic-PEAD event panel.

The unit of observation is an **earnings call**: a firm reports, holds its conference call, and the
transcript carries a *net tone* — the (positive − negative) sentiment-word share under a finance
sentiment lexicon (Loughran-McDonald 2011). The outcome is the firm's **post-call drift**: the
cumulative abnormal return (CAR) over the trading days *after* the call, i.e. the market-adjusted
return the tone is claimed to forecast.

Two tapes, one schema (columns: ``ticker``, ``quarter``, ``net_tone``, ``surprise``, ``car``):

- ``synthetic_panel`` — a deterministic, offline generator (fixed seed = 566). A single knob,
  ``tone_beta``, plants the linguistic-drift link: post-call CAR loads on the *residual* tone (the
  part of tone orthogonal to the numeric surprise), so ``tone_beta = 0`` is the null (tone tells you
  nothing beyond the number) and ``tone_beta > 0`` is the planted anomaly (upbeat tone → positive
  drift). This is the study's positive control and null in one bottle; tests never touch the network.

- ``fetch_panel`` — the **real-tape stub.** Scored earnings-call transcripts joined to event-time
  abnormal returns are a paid-vendor / hand-scored-academic product, not something a no-key retail
  stack can reach. So this returns the cached panel if a reader has dropped one into ``_cache/`` in
  the study schema, and otherwise an **empty frame** — never a fabricated tape. Because there is no
  free real feed, the Signal axis is capped at ``WEAK`` (a ``REAL`` stamp needs a robust *t* ≥ 2 on
  a real tape) and the limitation is named openly, like the desk's lego-returns / whisky-cask
  synthetic-only studies.

Honesty baked into the generator:

- **Tone is entangled with the number.** Real transcripts sound upbeat *because* the quarter beat;
  the naive tone→CAR slope therefore double-counts the numeric surprise (PEAD). The generator makes
  ``net_tone`` correlate with ``surprise`` on purpose, so the strategy must *control for surprise*
  to isolate the linguistic edge — the study's central honesty check.
- **Execution lag.** The CAR is measured strictly *after* the call date; the signal (tone) is known
  at the call, the position is entered the next session, no future data enters the tone score.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

# A fixed illustrative universe of large-cap names for the synthetic panel's ticker labels only
# (the numbers are generated, not fetched — the labels just make the panel readable).
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AVGO", "ORCL",
    "JPM", "BAC", "WFC", "GS", "XOM", "CVX", "JNJ", "PFE", "MRK", "ABBV",
    "PG", "KO", "PEP", "WMT", "COST", "HD", "MCD", "DIS", "CAT", "GE",
    "BA", "HON", "T", "VZ", "CMCSA", "NFLX", "CRM", "ADBE", "INTC", "AMD",
    "QCOM", "TXN",
]


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    tone_beta: float  # CAR loading on *residual* tone (0 = null; >0 = the linguistic drift)

    @property
    def has_anomaly(self) -> bool:
        return self.tone_beta != 0.0


# ---------------------------------------------------------------------------
# Synthetic event panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 40,
    n_quarters: int = 12,
    tone_beta: float = 0.020,
    surprise_beta: float = 0.030,
    car_idio: float = 0.045,
    seed: int = 566,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible earnings-call event panel with a tunable *linguistic-drift* anomaly.

    For each firm-quarter we draw a standardised numeric ``surprise`` (SUE-like) and a ``net_tone``
    that is *entangled* with it — upbeat calls follow good quarters — plus an idiosyncratic tone
    component (the manager's residual optimism, the part a linguistic edge would live in)::

        net_tone   = tone_load * surprise + tone_resid
        car        = market_drift
                     + surprise_beta * surprise            # the numeric PEAD (Study 363)
                     + tone_beta     * tone_resid          # the LINGUISTIC drift (the claim)
                     + idio

    Because the post-call CAR loads on the *residual* tone (the part orthogonal to the number), the
    honest test must control for ``surprise``: the naive tone→CAR slope is inflated by PEAD, and only
    the surprise-controlled slope isolates the words. ``tone_beta = 0`` is the null (tone adds nothing
    beyond the number); ``tone_beta > 0`` plants the anomaly. One row per firm-quarter with columns
    ``ticker``, ``quarter``, ``net_tone``, ``surprise``, ``car`` — the schema ``fetch_panel`` would
    produce from real transcripts.

    Returns ``(panel, truth)``.
    """
    rng = np.random.default_rng(seed)
    tickers = (UNIVERSE * ((n_firms // len(UNIVERSE)) + 1))[:n_firms]

    rows = []
    for q in range(n_quarters):
        surprise = rng.standard_normal(n_firms)                    # standardised numeric surprise
        tone_resid = rng.standard_normal(n_firms)                  # residual optimism (edge lives here)
        # Tone is entangled with the surprise (upbeat calls follow beats) plus residual optimism:
        net_tone = 0.6 * surprise + tone_resid
        idio = car_idio * rng.standard_normal(n_firms)
        market_drift = 0.002
        car = (
            market_drift
            + surprise_beta * surprise
            + tone_beta * tone_resid
            + idio
        )
        for j in range(n_firms):
            rows.append(
                {
                    "ticker": tickers[j],
                    "quarter": f"Q{q + 1:02d}",
                    "net_tone": float(net_tone[j]),
                    "surprise": float(surprise[j]),
                    "car": float(car[j]),
                }
            )
    panel = pd.DataFrame(rows)
    return panel, WorldTruth(tone_beta=tone_beta)


# ---------------------------------------------------------------------------
# Real tape — a cache-first STUB (no free feed exists; empty by design)
# ---------------------------------------------------------------------------
def _panel_cache() -> str:
    return os.path.join(DEFAULT_CACHE, "earnings_call_panel.parquet")


def fetch_panel(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Real earnings-call tone → post-call CAR panel, cache-first — **stub**.

    Scored transcripts joined to event-time abnormal returns are a paid-vendor / hand-scored
    product; there is no free, no-key feed to download. So this returns:

    - the cached panel if a reader has placed one at ``_cache/earnings_call_panel.parquet`` in the
      study schema (``ticker``, ``quarter``, ``net_tone``, ``surprise``, ``car``), or
    - an **empty frame** otherwise (``fetch=True`` cannot manufacture data that does not exist).

    Never fabricates a tape. The absence of a free real feed is why the Signal axis is capped at
    ``WEAK`` and named on the SIGNAL axis.
    """
    path = _panel_cache()
    if os.path.exists(path):
        df = pd.read_parquet(path)
        needed = {"ticker", "quarter", "net_tone", "surprise", "car"}
        if needed.issubset(df.columns):
            return df
    # No free feed to fetch; return empty regardless of `fetch` (documented stub).
    _ = fetch
    return pd.DataFrame(columns=["ticker", "quarter", "net_tone", "surprise", "car"])


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        num = obj.select_dtypes(include=[np.number])
        arr = np.ascontiguousarray(num.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, dict):
        blob = json.dumps(obj, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
