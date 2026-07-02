"""Data layer for Study 552 (App-Store-Rankings) — app-ranking momentum as a fundamental nowcast.

The claim (alt-data desk folklore, and the serious version behind it): a consumer-tech company's
**App Store download rank** is a real-time proxy for demand — units shipped, subscriptions taken,
engagement won. So when a public company's app *climbs* the charts (rank number falls toward #1),
that improvement should *nowcast* the fundamentals and, if the market is slow to price it, predict
the stock's **forward return** cross-sectionally. This is the broad cousin of the single-name
[294 Coinbase-Rank](../../294-coinbase-rank/) omen: instead of one contrarian top signal on BTC,
we sort a *cross-section* of consumer-tech names by their ranking *improvement* and ask whether the
rank-momentum leaders out-earn the laggards.

## Data availability — why this study is synthetic-only

There is **no free, survivorship-clean, point-in-time App Store ranking panel** a retail stack can
reach. The daily "top charts" are:

- **Ephemeral** — Apple publishes only a live snapshot; there is no official historical rank API.
- **Vendor-gated** — the usable history (App Annie / data.ai, Sensor Tower, Apptopia) is expensive,
  licensed, and itself modelled/estimated, not a clean tape.
- **Mapping-noisy** — an *app* is not a *ticker*: many top apps are private (or subsidiaries), and a
  public parent's revenue is only partly the app (Uber ride-hail vs Eats, Meta's app family, etc.).

So — like the desk's [273 lego-returns](../../273-lego-returns/), [275 whisky-cask](../../275-whisky-cask/)
and [276 sneaker-resale](../../276-sneaker-resale/) collectible studies — the honest move is to build
a **deterministic synthetic panel** with a single planted-effect knob, prove the engine catches the
effect when present and stays flat at the null, and state plainly on the SIGNAL axis that **no real
tape was tested**. A synthetic-only study can never earn `REAL` (that needs a robust t >= 2 on a
real tape); it is capped at `WEAK`/`NONE`.

## The two entry points, one schema

- ``synthetic_panel`` — a reproducible cross-section (fixed seed = 552). One knob,
  ``rank_alpha``, plants the effect: forward return loads on the *ranking-improvement* signal with
  coefficient ``rank_alpha`` (``>0`` = the claim is true; ``=0`` = the null). Returned frame has one
  row per name: ``rank_improve`` (the observable signal), ``fwd_ret`` (the forward return), and the
  latent ``true_demand`` (for diagnostics only — the strategy never sees it).
- ``fetch_panel`` — present for interface parity, but it **cannot** return real data (no free
  source). With ``fetch=False`` (default) it returns an empty frame; with ``fetch=True`` it raises,
  documenting the data-availability wall rather than silently faking a tape.

No look-ahead: the ranking-improvement signal is a *trailing* change (last month's rank minus this
month's rank), fully known at the observation date; the position is entered at that date and held
over the *forward* month. One documented execution lag; no future data enters the signal.
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

SEED = 552
TRADING_DAYS = 252

# A representative universe of consumer-tech names whose fortunes are (partly) app-driven — the
# kind of cross-section an alt-data desk would score. Labels only; the panel is synthetic, so these
# are illustrative tags, not a claim about any real ticker.
UNIVERSE = [
    "SOCIAL_A", "SOCIAL_B", "STREAM_A", "STREAM_B", "RIDE_A", "RIDE_B",
    "FOOD_A", "FOOD_B", "GAME_A", "GAME_B", "FINTECH_A", "FINTECH_B",
    "ECOMM_A", "ECOMM_B", "DATING_A", "DATING_B", "TRAVEL_A", "TRAVEL_B",
    "MUSIC_A", "MUSIC_B", "SHOP_A", "SHOP_B", "CRYPTO_A", "CRYPTO_B",
    "HEALTH_A", "HEALTH_B", "EDU_A", "EDU_B", "NEWS_A", "NEWS_B",
]


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    rank_alpha: float  # forward-return loading on the ranking-improvement signal (>0 = claim true)

    @property
    def has_effect(self) -> bool:
        return self.rank_alpha != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_names: int = 30,
    rank_alpha: float = 0.15,
    signal_noise: float = 1.0,
    idio_vol: float = 1.5,
    seed: int = SEED,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible consumer-tech cross-section with a tunable ranking-momentum effect.

    Model of one observation month:

    - Each name has a latent **demand shock** ``true_demand`` ~ N(0, 1) this month (the real change
      in units/subscriptions the app is capturing).
    - The **observable signal** is the *ranking improvement* — how much the app climbed the charts —
      which is a *noisy read* on demand::

          rank_improve = true_demand + signal_noise * noise     (then z-scored across the panel)

      ``signal_noise`` controls how blurry the alt-data is (App Store rank is discrete, capped at
      #1, and only partly maps to the public parent's revenue — so the read is noisy).
    - The **forward return** loads on *true demand* (fundamentals eventually show up in price), plus
      idiosyncratic noise::

          fwd_ret = market_drift + rank_alpha * true_demand + idio_vol * eps

    With ``rank_alpha > 0`` the names whose apps climbed (high ``rank_improve``, a noisy proxy for
    high demand) tend to earn more — the claim. With ``rank_alpha = 0`` forward return is unrelated
    to demand and thus to the ranking signal — the null. The strategy sees only ``rank_improve`` and
    ``fwd_ret``; ``true_demand`` is returned for diagnostics but never used by the engine.

    Returns ``(panel, truth)``.
    """
    rng = np.random.default_rng(seed)
    n = int(n_names)

    true_demand = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    raw_signal = true_demand + signal_noise * noise
    # z-score the observable signal across the cross-section (what a desk would rank on)
    rank_improve = (raw_signal - raw_signal.mean()) / (raw_signal.std(ddof=1) or 1.0)

    market_drift = 0.01  # ~1%/month market backdrop, common to all names
    idio = idio_vol * rng.standard_normal(n)
    fwd_ret = market_drift + rank_alpha * true_demand + idio

    tickers = [UNIVERSE[j % len(UNIVERSE)] if n <= len(UNIVERSE) else f"N{j:03d}"
               for j in range(n)]
    panel = pd.DataFrame(
        {
            "rank_improve": rank_improve,
            "fwd_ret": fwd_ret,
            "true_demand": true_demand,
        },
        index=pd.Index(tickers[:n], name="ticker"),
    )
    return panel, WorldTruth(rank_alpha=rank_alpha)


def synthetic_history(
    n_names: int = 30,
    n_months: int = 48,
    rank_alpha: float = 0.15,
    signal_noise: float = 1.0,
    idio_vol: float = 1.5,
    seed: int = SEED,
) -> pd.DataFrame:
    """A stacked panel of ``n_months`` monthly cross-sections (long format).

    Draws one independent cross-section per month (a fresh seed offset per month, deterministic in
    ``seed``) so the pooled information coefficient and its month-by-month distribution can be
    measured. Columns: ``month``, ``ticker``, ``rank_improve``, ``fwd_ret``, ``true_demand``.
    """
    frames = []
    for m in range(int(n_months)):
        pan, _ = synthetic_panel(
            n_names=n_names, rank_alpha=rank_alpha, signal_noise=signal_noise,
            idio_vol=idio_vol, seed=seed + 1000 * (m + 1),
        )
        pan = pan.reset_index()
        pan.insert(0, "month", m)
        frames.append(pan)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Real tape — does not exist for free; documented, not faked
# ---------------------------------------------------------------------------
def fetch_panel(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Interface parity with cache-first studies — but there is **no free real tape**.

    A survivorship-clean, point-in-time App Store ranking panel joined to public tickers is not
    reachable from a no-key retail stack (Apple publishes no historical rank API; the usable history
    is licensed/estimated). So:

    - ``fetch=False`` (default): return an empty frame (cache-only convention).
    - ``fetch=True``: raise ``NotImplementedError`` — we refuse to silently fabricate a "real" tape.

    This is the data-availability wall stated on the SIGNAL axis; the study is synthetic-only.
    """
    if fetch:
        raise NotImplementedError(
            "No free, point-in-time, survivorship-clean App Store ranking panel exists for a "
            "no-key retail stack (Apple has no historical rank API; vendor history is licensed and "
            "modelled). Study 552 is synthetic-only by design — see data.py docstring."
        )
    return pd.DataFrame()


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        num = obj.select_dtypes(include=[np.number])
        arr = np.ascontiguousarray(num.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, dict):
        blob = json.dumps(obj, sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
