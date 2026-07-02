"""Data layer for Study 586 (Liquidation-Cascade) — the crypto capitulation-bottom folklore.

The claim under test: a large **forced-liquidation** event (a cascade of margin-called longs
dumped by the exchange) marks a *capitulation bottom* — the mechanical selling overshoots and the
price then **bounces**. The tradable read is "buy after the liquidation spike."

What a real test would need is a per-day **aggregate forced-liquidation USD** series (the kind
Coinglass / Amberdata sell for Binance/Bybit/OKX). That series does **not** exist on a free,
no-key retail stack with usable history, so this study is **synthetic-only** — stated openly on the
SIGNAL axis, and the reason the study is capped at `WEAK`/`NONE` (a `REAL` stamp needs a robust
*t* ≥ 2 on a *real* tape, which we cannot build here).

Two entry points, one schema — a daily frame indexed by date with columns:

- ``ret``          : daily log-ish simple return of the (synthetic) BTC-style price
- ``liq``          : aggregate forced-liquidation flow that day (USD-ish, ≥ 0; heavy right tail)
- ``price``        : the cumulated price level (for plotting)

``synthetic_series`` is the deterministic offline core. A single knob, ``bounce_alpha``, plants the
folklore: after a large-liquidation day, the next ``H`` days carry an extra drift of
``bounce_alpha`` per unit of (standardised) liquidation shock. ``bounce_alpha > 0`` = the
capitulation-bounce is real; ``= 0`` = the null (liquidations carry no forward information);
``< 0`` = liquidations predict *more* bleeding (momentum / "falling knife").

``fetch_series`` is a cache-first stub: there is no free real liquidation tape, so with
``fetch=False`` (the default) it returns an empty frame, and it never touches the network in the
reproducible core. It exists so the schema and the "point-this-at-a-real-feed" seam are explicit.
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

TRADING_DAYS = 365  # crypto trades every day


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic series."""

    bounce_alpha: float  # forward drift per unit of standardised liquidation shock

    @property
    def has_bounce(self) -> bool:
        return self.bounce_alpha != 0.0


# ---------------------------------------------------------------------------
# Synthetic series — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_series(
    n_days: int = 1500,
    bounce_alpha: float = 0.0,
    horizon: int = 5,
    base_vol: float = 0.035,
    seed: int = 586,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible daily BTC-style tape with an explicit forced-liquidation channel.

    The generative story, day by day:

    1. A base return ``eps ~ N(0, base_vol)`` with mild volatility clustering (a slow-moving vol
       state), so drawdowns cluster the way crypto really does.
    2. **Forced liquidations** are driven by *down* moves: a big negative return margin-calls
       leveraged longs, so ``liq`` spikes when the day is red and vol is high. ``liq`` has a heavy
       right tail (most days quiet, occasional cascades) — modelled as an exponential scaled by the
       size of the down move and the vol state.
    3. The **planted effect**: after a day with a large (standardised) liquidation shock ``z``, the
       next ``horizon`` days receive an extra drift ``bounce_alpha * z / horizon``. With
       ``bounce_alpha > 0`` a liquidation spike is followed by a *bounce* (the folklore);
       ``= 0`` is the null; ``< 0`` is a falling-knife continuation.

    Crucially, the liquidation spike itself lands with the *down* day (contemporaneous), so a naive
    "liquidations => price fell" correlation is mechanical and says nothing about the *forward*
    bounce — which is exactly why the event study looks only at returns *after* the event day.

    Returns ``(frame, truth)`` where ``frame`` has columns ``ret``, ``liq``, ``price``.
    """
    rng = np.random.default_rng(seed)

    # Slow-moving volatility state (clustering), mean-reverting around 1.0.
    vol_state = np.ones(n_days)
    v = 1.0
    for i in range(n_days):
        v = 0.94 * v + 0.06 * 1.0 + 0.10 * rng.standard_normal()
        v = float(np.clip(v, 0.4, 3.0))
        vol_state[i] = v

    eps = base_vol * vol_state * rng.standard_normal(n_days)

    # Forced liquidation flow: spikes on big DOWN days scaled by the vol state. Heavy right tail.
    down = np.clip(-eps, 0.0, None)                       # only red days margin-call longs
    intensity = 1.0 + 60.0 * down * vol_state             # bigger red + higher vol => more liq
    liq = rng.exponential(scale=intensity)                # >= 0, heavy tail
    liq = liq * 1e6                                        # USD-ish scale (cosmetic)

    # Standardised liquidation shock z (for the planted forward effect).
    lz = np.log1p(liq)
    z = (lz - lz.mean()) / (lz.std(ddof=1) + 1e-12)

    # Plant the forward bounce: a large-liq day at t adds drift to days t+1 .. t+horizon.
    ret = eps.copy()
    if bounce_alpha != 0.0:
        add = bounce_alpha * z / max(horizon, 1)
        for lag in range(1, horizon + 1):
            ret[lag:] += add[:-lag] if lag < n_days else 0.0

    price = 10000.0 * np.cumprod(1.0 + ret)
    dates = pd.date_range("2019-01-01", periods=n_days, freq="D")
    frame = pd.DataFrame({"ret": ret, "liq": liq, "price": price}, index=dates)
    frame.index.name = "date"
    return frame, WorldTruth(bounce_alpha=bounce_alpha)


# ---------------------------------------------------------------------------
# Real tape — a cache-first stub (no free liquidation feed exists)
# ---------------------------------------------------------------------------
def _cache_path() -> str:
    return os.path.join(DEFAULT_CACHE, "liquidation_series.parquet")


def fetch_series(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Cache-first real liquidation tape — a documented stub.

    A genuine test needs a per-day aggregate forced-liquidation USD series (Coinglass / Amberdata /
    an exchange futures API). None is reachable on a free, no-key stack with usable history, so:

    - ``fetch=False`` (default): return the cached parquet if a reader has staged one under
      ``_cache/liquidation_series.parquet`` with the columns ``ret``, ``liq`` (``price`` optional),
      else an **empty** frame. The reproducible core never needs the network.
    - ``fetch=True``: there is nothing free to download, so we raise a clear ``NotImplementedError``
      pointing at the paid feeds — the honest seam, not a silent fake.

    This is why the study is synthetic-only and capped at `WEAK`/`NONE`.
    """
    path = _cache_path()
    if os.path.exists(path):
        df = pd.read_parquet(path)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    if not fetch:
        return pd.DataFrame()
    raise NotImplementedError(
        "No free forced-liquidation tape exists. Stage a per-day frame with columns "
        "['ret','liq'] (from Coinglass/Amberdata/an exchange futures API) at "
        f"{path} and re-run with fetch=False. This study is synthetic-only by design."
    )


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
