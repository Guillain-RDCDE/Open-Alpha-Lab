"""Data access for the half-life study — the **arb-gap series** and where it comes from.

One object runs the whole study: the **gap**

    g(t) = 1 - (p_yes(t) + p_no(t))

for a binary market. On Polymarket the YES and NO outcomes trade in *separate* CLOB
order books, so their prices need not sum to $1 — and when ``p_yes + p_no < 1`` you can
buy *both* sides for less than the $1 they are jointly guaranteed to pay at resolution.
That is the "market-rebalancing" arbitrage the paper (Saguillo et al., 2025) priced at
~$40M. ``g > 0`` is exactly that free penny; ``g < 0`` is the mirror trade (sell both).
This module produces ``g`` two ways, the desk's standing split:

    * :func:`synthetic_markets` — fully **offline**. Arbitrage episodes open at random
      and then **decay exponentially with a half-life we choose**, so the estimator in
      :mod:`arbitrage` has a ground truth to recover and the tests can assert on it. No
      network ever runs in CI.
    * :func:`fetch_prices_history` + :func:`build_gap_panel` — read real Polymarket CLOB
      ``prices-history`` for a manifest of resolved markets, cache each token's series to
      parquet, and assemble the gap panel. **Cache-only by construction**: a token with no
      cached parquet is skipped, never a silent network stall mid-run.

The resolution decision, named up front as a house rule. The CLOB ``prices-history``
endpoint returns a **sampled** series whose finest ``fidelity`` is ~1 minute, and its
order-book snapshots stopped updating around 2026-02-20 (so depth, hence capacity, is only
reconstructable before then). A minute tape **cannot resolve the 2-second block** on which
the real race is run. We therefore measure a **coarse upper bound** on the half-life and
say so loudly — which, far from weakening the verdict, *is* the verdict: if even our tape
can't see the close, a human refreshing a browser certainly can't trade it. The synthetic
half-life is set in the same minute units so the two layers speak the same clock.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# CLOB price history base; the synthetic clock matches its finest fidelity (~1 min).
CLOB_PRICES_HISTORY = "https://clob.polymarket.com/prices-history"
DEFAULT_FIDELITY_MIN = 1          # minutes per sample — the finest the endpoint serves
ORDERBOOK_FROZEN_ASOF = "2026-02-20"   # snapshots stopped here; depth only before this

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(_HERE, "..", "_cache")


# --------------------------------------------------------------------------- #
# Synthetic markets — offline, with a baked-in half-life to recover
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GroundTruth:
    """What the synthetic generator baked in, so a test can check the estimator."""
    half_life_min: float       # the true decay half-life, in minutes (= sample steps)
    open_threshold: float      # the gap above which an episode is 'open'
    n_markets: int


def synthetic_markets(
    n_markets: int = 48,
    n_steps: int = 6000,
    half_life_min: float = 6.0,
    open_threshold: float = 0.03,
    shock_rate: float = 0.006,
    shock_scale: float = 0.05,
    obs_noise: float = 0.0015,
    two_sided: bool = True,
    start: str = "2024-04-01",
    seed: int = 0,
) -> tuple[pd.DataFrame, GroundTruth]:
    """A toy book of prediction markets whose arbitrage gaps **decay with a known half-life**.

    Each market sits at no-arb (``g ≈ 0``) until a **shock** opens a gap of size
    ``g0 ~ open_threshold + Exp(shock_scale)`` — a fresh news print, a fat market order,
    a lagging leg. From there the gap **decays exponentially** as arbitrageurs close it:

        g(t+1) = g(t) · ½^(1/half_life_min)              (between shocks)

    so the mispricing loses half its size every ``half_life_min`` steps — *exactly* the
    quantity :func:`arbitrage.time_to_half` and :func:`arbitrage.fit_half_life` exist to
    recover. Shocks arrive as a Bernoulli(``shock_rate``) process; with ``two_sided`` the
    sign is random (``g > 0`` buy-both, ``g < 0`` sell-both). A little observation noise
    (``obs_noise``) is layered on so the series isn't a clean exponential the estimator
    could cheat off of.

    Returns ``(gap, truth)`` — ``gap`` is a (steps × markets) DataFrame of ``g(t)`` on a
    1-minute index, ``truth`` carries the baked-in half-life. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n_steps, freq="1min", name="ts")
    decay = 0.5 ** (1.0 / half_life_min)

    cols: dict[str, np.ndarray] = {}
    for m in range(n_markets):
        g = np.zeros(n_steps)
        cur = 0.0
        shocks = rng.random(n_steps) < shock_rate
        mags = open_threshold + rng.exponential(shock_scale, n_steps)
        signs = rng.choice([-1.0, 1.0], n_steps) if two_sided else np.ones(n_steps)
        for t in range(n_steps):
            if shocks[t]:
                cur = signs[t] * mags[t]      # a fresh dislocation opens the gap
            else:
                cur *= decay                  # arbitrageurs bleed it back toward zero
            g[t] = cur
        g = g + obs_noise * rng.standard_normal(n_steps)
        cols[f"MKT{m:02d}"] = g

    gap = pd.DataFrame(cols, index=idx)
    truth = GroundTruth(half_life_min=half_life_min, open_threshold=open_threshold,
                        n_markets=n_markets)
    return gap, truth


# --------------------------------------------------------------------------- #
# Real markets — Polymarket CLOB prices-history, cached per token, gap assembled
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MarketRef:
    """One binary market: a label, its two CLOB token ids (YES, NO), and its active window.

    ``start_ts``/``end_ts`` (unix seconds) bound the market's *active* life, discovered from
    its own price history, so the fetcher pulls minute data only where the market traded
    rather than day-chunking across dead calendar — they make the real run both cheaper and
    reproducible. ``None`` means "let the caller decide the window".
    """
    slug: str
    yes_token: str
    no_token: str
    start_ts: int | None = None
    end_ts: int | None = None


def load_manifest(path: str) -> list[MarketRef]:
    """Read a JSON manifest of resolved markets ``[{slug, yes_token, no_token, ...}, ...]``.

    The manifest is the study's *input of record* — built once from the Gamma API
    (``gamma-api.polymarket.com/markets``, which carries each market's ``clobTokenIds``)
    and committed, so the real run is reproducible without re-discovering markets live.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return [MarketRef(slug=m["slug"], yes_token=str(m["yes_token"]),
                      no_token=str(m["no_token"]),
                      start_ts=m.get("start_ts"), end_ts=m.get("end_ts")) for m in raw]


def _cache_path(token: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"clob_{token}.parquet")


_CHUNK_SECONDS = 86_400          # the minute endpoint caps a fidelity=1 window at ~1 day


def _parse_history(hist: list) -> pd.DataFrame:
    df = pd.DataFrame(hist)
    if df.empty:
        return pd.DataFrame({"price": []}, index=pd.DatetimeIndex([], name="ts"))
    return (df.assign(ts=pd.to_datetime(df["t"], unit="s"))
              .set_index("ts")[["p"]]
              .rename(columns={"p": "price"})
              .sort_index())


def fetch_prices_history(
    token: str,
    start_ts: int,
    end_ts: int,
    fidelity_min: int = DEFAULT_FIDELITY_MIN,
    cache_dir: str = DEFAULT_CACHE,
    session=None,
) -> pd.DataFrame:
    """Fetch (and cache) one token's CLOB price history as a ``ts``-indexed ``price`` frame.

    Hits ``clob.polymarket.com/prices-history`` and writes one parquet per token to
    ``cache_dir``; a subsequent call returns the cache. The endpoint answers
    ``{"history": [{"t": <unix>, "p": <price>}, ...]}`` but **caps a ``fidelity=1`` window
    at ~1 day**, so a longer span is pulled in daily chunks and concatenated — that day
    loop is the only reason the minute tape is reachable at all. Network lives **only**
    here, and only on a cache miss; every downstream function reads the cache. ``session``
    is an optional ``requests.Session`` (injected so tests never import requests).
    """
    path = _cache_path(token, cache_dir)
    if os.path.exists(path):
        return pd.read_parquet(path)

    sess = session
    if sess is None:
        import requests  # lazy: the offline core (and CI) never imports it
        sess = requests.Session()
    span = max(0, end_ts - start_ts)
    step = _CHUNK_SECONDS if fidelity_min <= 1 else span + 1
    parts = []
    lo = start_ts
    while lo < end_ts:
        hi = min(lo + step, end_ts)
        resp = sess.get(
            CLOB_PRICES_HISTORY,
            params={"market": token, "startTs": lo, "endTs": hi, "fidelity": fidelity_min},
            timeout=30,
        )
        resp.raise_for_status()
        parts.append(_parse_history(resp.json().get("history", [])))
        lo = hi
    if span == 0:                                          # single-shot (e.g. test stubs)
        resp = sess.get(
            CLOB_PRICES_HISTORY,
            params={"market": token, "startTs": start_ts, "endTs": end_ts,
                    "fidelity": fidelity_min},
            timeout=30,
        )
        resp.raise_for_status()
        parts.append(_parse_history(resp.json().get("history", [])))

    df = (pd.concat(parts) if parts else _parse_history([]))
    df = df[~df.index.duplicated(keep="last")].sort_index()
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(path)
    return df


def cached_tokens(cache_dir: str = DEFAULT_CACHE) -> set[str]:
    """Every token id with a cached ``prices-history`` parquet (the markets we *can* price)."""
    if not os.path.isdir(cache_dir):
        return set()
    return {f[len("clob_"):-len(".parquet")]
            for f in os.listdir(cache_dir)
            if f.startswith("clob_") and f.endswith(".parquet")}


def build_gap_panel(
    manifest: list[MarketRef],
    cache_dir: str = DEFAULT_CACHE,
    fidelity_min: int = DEFAULT_FIDELITY_MIN,
) -> pd.DataFrame:
    """Assemble the real gap panel ``g = 1 - (p_yes + p_no)``, one column per market.

    **Cache-only**: a market is included only if *both* of its token parquets are present
    (a half-cached market is skipped, never half-fetched on the fly). The two legs are
    aligned on the union minute grid and forward-filled within the overlap (a quoted price
    persists until it next prints), then the gap is differenced from $1. NaNs outside the
    overlap are dropped per market by the consumers, never forward-filled across the join.
    """
    have = cached_tokens(cache_dir)
    cols: dict[str, pd.Series] = {}
    for ref in manifest:
        if ref.yes_token not in have or ref.no_token not in have:
            continue
        yes = pd.read_parquet(_cache_path(ref.yes_token, cache_dir))["price"]
        no = pd.read_parquet(_cache_path(ref.no_token, cache_dir))["price"]
        grid = yes.index.union(no.index)
        y = yes.reindex(grid).ffill()
        n = no.reindex(grid).ffill()
        cols[ref.slug] = (1.0 - (y + n)).rename(ref.slug)
    if not cols:
        return pd.DataFrame()
    gap = pd.DataFrame(cols).sort_index()
    gap.index.name = "ts"
    return gap
