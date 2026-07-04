"""Data layer for Study 636 — Exchange Listing Pop (the "Coinbase effect").

Two worlds, one shape (a wide daily USD close panel + an event table of Coinbase
listing days):

* **Real tape.** Daily closes from **Binance USDT spot klines** (public, keyless) for
  every event coin plus BTC (the market column). The coins trade on Binance long
  *before* Coinbase lists them — that is the whole point: Binance carries the price
  history around the Coinbase event. Binance keeps serving klines for pairs it later
  delisted, which softens (but does not remove) survivorship.

* **Event table.** ``EVENTS`` below is **hardcoded** (desk convention for event
  tables). Day 0 for each coin = the date of the **first daily candle of its USD
  product on Coinbase Exchange** (``GET api.exchange.coinbase.com/products/{id}/
  candles?granularity=86400``, fetched once on 2026-07-03) — the venue's own record
  of when trading actually started, no hand-curated blog dates. Construction:
  Coinbase-USD products ∩ Binance-USDT pairs, first CB candle ≥ 2018-01-01, at least
  60 days of Binance history *before* the listing (so the event has a pre-history),
  listing at least 35 days old (so the +30 window is complete), and — the ticker-
  collision killer — the Coinbase first-candle close must be within [0.6, 1.6] of the
  Binance close on the same day (unrelated tokens that share a ticker across venues
  fail this; the per-row third field is that venue price ratio, its source check).
  Stablecoins / wrapped assets are excluded. Five assets whose first Coinbase book
  was **USDC-quoted** in the 2018-20 era (BAT, ZEC, CVC, DNT, MANA) are excluded —
  for them the first USD-product candle post-dates the true listing, so the USD date
  would mis-date the event (checked against the -USDC products on 2026-07-03).

* **Announcement subset.** ``ANNOUNCEMENTS`` — a small, documented table of Coinbase
  *announcement* dates (blog posts) vs actual first-trading days, per-row source URL.
  In the Coinbase-Pro era the pattern is announce day T (transfers open), trading
  ~T+2; since 2022-04 the "listing roadmap" policy front-loads the news by weeks.
  Used on the third axis (announcement vs listing timestamps).

* **Synthetic.** ``synthetic_world`` — a deterministic daily panel with a plantable
  **pop** (abnormal return spread over days −2..0) and **fade** (spread over days
  +1..+20) around staggered pseudo-listings, plus the null (``pop = fade = 0``). The
  positive / null control for the event-study machinery. ~5 years of days — far
  below the 250-year ns-Timestamp ceiling.

Cache-first: ``fetch_panel(fetch=True)`` touches the network ONCE and writes
``_cache/elp_prices.parquet``; everything else reads the cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
PRICES_CACHE = os.path.join(CACHE_DIR, "elp_prices.parquet")

UA = {"User-Agent": "OpenAlphaLab research desk guillain@poulpe.us"}

# --------------------------------------------------------------------------- #
# The event table (hardcoded; see module docstring for the construction rule).
# Source per row: first Coinbase Exchange USD daily candle (api.exchange.coinbase.com,
# fetched 2026-07-03); third field = CB first-candle close / Binance same-day close
# (the venue-agreement check that killed cross-venue ticker collisions).
# --------------------------------------------------------------------------- #
EVENTS = [
    # (coin, Coinbase listing day = first CB Exchange USD daily candle, venue price ratio CB/Binance on day 0)
    ("XRP", "2019-02-26", 1.007),
    ("XLM", "2019-03-14", 0.98),
    ("EOS", "2019-04-09", 1.055),
    ("LINK", "2019-06-27", 0.625),
    ("DASH", "2019-09-17", 0.946),
    ("ATOM", "2020-01-14", 0.894),
    ("OMG", "2020-05-19", 0.763),
    ("BAND", "2020-08-11", 0.884),
    ("LRC", "2020-09-15", 0.965),
    ("REN", "2020-10-06", 0.853),
    ("SNX", "2020-12-15", 1.035),
    ("BNT", "2020-12-15", 1.271),
    ("AAVE", "2020-12-15", 1.029),
    ("SUSHI", "2021-03-11", 0.817),
    ("MATIC", "2021-03-11", 0.999),
    ("SKL", "2021-03-11", 0.957),
    ("ADA", "2021-03-18", 1.025),
    ("CRV", "2021-03-25", 0.765),
    ("ANKR", "2021-03-25", 0.639),
    ("OGN", "2021-04-09", 1.154),
    ("ENJ", "2021-04-09", 1.101),
    ("NKN", "2021-04-09", 1.357),
    ("1INCH", "2021-04-09", 1.079),
    ("TRB", "2021-05-06", 0.889),
    ("RLC", "2021-05-06", 0.881),
    ("DOGE", "2021-06-03", 1.081),
    ("DOT", "2021-06-16", 1.075),
    ("SOL", "2021-06-17", 1.115),
    ("CHZ", "2021-06-17", 1.148),
    ("FET", "2021-07-27", 1.01),
    ("AXS", "2021-08-12", 0.976),
    ("ORN", "2021-08-12", 1.003),
    ("IOTX", "2021-08-12", 1.209),
    ("TRU", "2021-08-12", 1.049),
    ("COTI", "2021-08-26", 0.98),
    ("YFII", "2021-09-01", 1.12),
    ("ZEN", "2021-09-16", 1.071),
    ("AVAX", "2021-09-30", 0.966),
    ("BADGER", "2021-10-14", 1.218),
    ("PERP", "2021-10-19", 0.998),
    ("ARPA", "2021-10-19", 1.155),
    ("POLS", "2021-12-07", 1.158),
    ("IDEX", "2021-12-07", 1.217),
    ("SUPER", "2021-12-07", 1.195),
    ("MDT", "2021-12-09", 1.073),
    ("BLZ", "2021-12-09", 1.061),
    ("STX", "2022-01-20", 1.398),
    ("DIA", "2022-01-25", 1.117),
    ("UNFI", "2022-01-25", 1.248),
    ("FIDA", "2022-02-01", 1.09),
    ("RNDR", "2022-02-03", 0.905),
    ("ALICE", "2022-02-24", 0.972),
    ("HIGH", "2022-03-08", 0.985),
    ("ERN", "2022-03-08", 1.075),
    ("MINA", "2022-04-06", 1.002),
    ("ROSE", "2022-04-26", 0.989),
    ("FLOW", "2022-05-19", 1.047),
    ("SAND", "2022-05-26", 1.049),
    ("KSM", "2022-06-02", 1.068),
    ("C98", "2022-06-07", 0.959),
    ("POND", "2022-06-16", 0.953),
    ("DAR", "2022-06-16", 1.06),
    ("ATA", "2022-06-16", 1.121),
    ("FIS", "2022-06-16", 1.09),
    ("DREP", "2022-06-23", 1.06),
    ("GNO", "2022-07-21", 1.027),
    ("MTL", "2022-07-21", 1.05),
    ("RARE", "2022-08-02", 1.079),
    ("LOKA", "2022-08-04", 1.062),
    ("CELR", "2022-08-16", 1.192),
    ("OOKI", "2022-08-30", 1.054),
    ("NEAR", "2022-09-01", 1.041),
    ("CVX", "2022-09-08", 0.933),
    ("OCEAN", "2022-09-08", 0.948),
    ("INJ", "2022-09-20", 0.998),
    ("PUNDIX", "2022-09-20", 1.052),
    ("HBAR", "2022-10-13", 1.08),
    ("ILV", "2022-10-13", 1.069),
    ("LDO", "2022-11-17", 1.026),
    ("QI", "2022-11-23", 1.222),
    ("PYR", "2022-11-29", 0.972),
    ("LIT", "2022-12-06", 1.042),
    ("GHST", "2022-12-06", 1.012),
    ("EGLD", "2022-12-07", 0.977),
    ("ANT", "2022-12-08", 0.92),
    ("KAVA", "2023-01-19", 0.97),
    ("T", "2023-01-26", 1.174),
    ("AUDIO", "2023-01-26", 1.046),
    ("VOXEL", "2023-02-09", 1.092),
    ("TVK", "2023-04-18", 1.186),
    ("MULTI", "2023-05-04", 1.023),
    ("OSMO", "2023-05-25", 1.042),
    ("VTHO", "2023-09-13", 1.199),
    ("VET", "2023-09-13", 0.928),
    ("ARKM", "2024-04-02", 1.137),
    ("ZK", "2024-09-25", 0.87),
    ("IO", "2024-10-09", 0.932),
    ("PEPE", "2024-11-13", 0.895),
    ("WIF", "2024-11-13", 1.073),
    ("FLOKI", "2024-11-21", 1.056),
    ("TURBO", "2024-12-11", 1.165),
    ("PNUT", "2025-01-14", 1.033),
    ("ETHFI", "2025-02-06", 0.96),
    ("PYTH", "2025-02-20", 1.018),
    ("TAO", "2025-02-20", 1.078),
    ("REZ", "2025-03-06", 1.212),
    ("COOKIE", "2025-03-11", 1.247),
    ("PENDLE", "2025-03-27", 1.148),
    ("ALT", "2025-03-27", 1.193),
    ("RSR", "2025-04-22", 1.053),
    ("WLD", "2025-04-30", 1.099),
    ("ENA", "2025-06-05", 0.884),
    ("CAKE", "2025-06-12", 1.066),
    ("S", "2025-06-24", 1.164),
    ("W", "2025-07-02", 1.118),
    ("BIO", "2025-07-31", 1.085),
    ("WCT", "2025-08-14", 0.996),
    ("AWE", "2025-09-04", 1.029),
    ("KMNO", "2025-09-11", 1.08),
    ("LAYER", "2025-09-11", 0.977),
    ("BNB", "2025-10-22", 0.967),
    ("TON", "2025-11-18", 1.145),
    ("XPL", "2025-12-02", 0.984),
    ("HYPER", "2025-12-10", 1.075),
    ("PLUME", "2025-12-10", 1.101),
    ("RAY", "2026-01-14", 1.063),
    ("WAL", "2026-02-11", 0.92),
    ("SIGN", "2026-04-21", 1.023),
    ("VIRTUAL", "2026-04-29", 0.987),
]

# Documented announcement -> first-trade gaps (third axis). Sources: the Coinbase
# blog posts themselves + contemporaneous press.
ANNOUNCEMENTS = [
    # (coin, announce date, first trading day, source)
    ("MATIC", "2021-03-09", "2021-03-11",
     "coinbase.com/blog/polygon-matic-skale-network-skl-and-sushiswap-sushi-are-launching-on-coinbase-pro"),
    ("ADA", "2021-03-16", "2021-03-18",
     "blog.coinbase.com/cardano-ada-is-launching-on-coinbase-pro-694b1cb8c778"),
    ("SOL", "2021-05-20", "2021-06-17",
     "blog.coinbase.com/solana-sol-is-launching-on-coinbase-pro-734ccb2511b2 (launch delayed by technical issues)"),
    ("DOGE", "2021-06-01", "2021-06-03",
     "blog.coinbase.com/dogecoin-doge-is-launching-on-coinbase-pro-1d73bf66dd9d"),
]

# 2022-04-28: Coinbase switches to a public "listing roadmap" that front-loads the
# news by weeks (coinbase.com/blog/increasing-transparency-for-new-asset-listings-
# on-coinbase, after the DOJ listing-front-running case). Era split used in the study.
ROADMAP_ERA = pd.Timestamp("2022-04-28")


def events_frame() -> pd.DataFrame:
    """The hardcoded event table as a DataFrame (coin, day0, venue_ratio)."""
    df = pd.DataFrame(EVENTS, columns=["coin", "day0", "venue_ratio"])
    df["day0"] = pd.to_datetime(df["day0"])
    return df.sort_values("day0").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Real tape (cache-first)
# --------------------------------------------------------------------------- #
def _get_json(url: str, retries: int = 4):
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:                                  # noqa: BLE001
            last = e
            time.sleep(1.5)
    raise RuntimeError(f"fetch failed: {url}: {last}")


def _binance_daily(symbol: str) -> pd.Series | None:
    """Full daily close history for one Binance spot pair (paginated)."""
    out, start = [], 0
    while True:
        k = _get_json(f"https://api.binance.com/api/v3/klines?symbol={symbol}"
                      f"&interval=1d&startTime={start}&limit=1000")
        time.sleep(0.08)
        if not k:
            break
        out += k
        if len(k) < 1000:
            break
        start = k[-1][0] + 86_400_000
    if not out:
        return None
    idx = pd.to_datetime([r[0] for r in out], unit="ms").normalize()
    s = pd.Series([float(r[4]) for r in out], index=idx, name=symbol)
    return s[~s.index.duplicated()]


def fetch_panel(fetch: bool = False, cache_path: str = PRICES_CACHE) -> pd.DataFrame:
    """Wide daily close panel (event coins + BTC); cache-only unless ``fetch=True``."""
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached panel at {cache_path}. Run fetch_panel(fetch=True) once.")
        px = pd.read_parquet(cache_path)
        px.index = pd.to_datetime(px.index)
        return px.sort_index()
    coins = [c for c, _, _ in EVENTS] + ["BTC"]
    cols = {}
    for coin in coins:
        s = _binance_daily(coin + "USDT")
        if s is not None and len(s) > 90:
            cols[coin] = s
    px = pd.DataFrame(cols).sort_index()
    # keep only complete UTC days
    now = pd.Timestamp.utcnow().tz_localize(None).normalize()
    px = px[px.index <= now - pd.Timedelta(days=1)]
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    px.to_parquet(cache_path)
    return px


def have_real(cache_path: str = PRICES_CACHE) -> bool:
    return os.path.exists(cache_path)


def load_real(cache_path: str = PRICES_CACHE) -> pd.DataFrame:
    return fetch_panel(fetch=False, cache_path=cache_path)


def fingerprint(px: pd.DataFrame) -> str:
    """Short content hash of the close panel, for the as-of stamp."""
    arr = np.ascontiguousarray(np.nan_to_num(px.to_numpy(dtype=float)))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic control — plantable pop + fade
# --------------------------------------------------------------------------- #
def synthetic_world(n_coins: int = 60, n_days: int = 1800, pop: float = 0.0,
                    fade: float = 0.0, seed: int = 636,
                    mkt_vol: float = 0.04, idio_vol: float = 0.06
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic daily close panel + staggered pseudo-listing events.

    Every coin is ``1.0 × market + idiosyncratic noise``. If ``pop`` != 0 an abnormal
    log return of ``pop`` (total) is planted evenly over event days −2..0 (the
    announcement-to-listing pop); if ``fade`` != 0 an abnormal −``fade`` (total) is
    planted evenly over days +1..+20 (the give-back). With ``pop = fade = 0`` the
    event windows are statistically indistinguishable from any other window — the
    machinery must NOT manufacture significance. ~5 years of days.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n_days, freq="D")
    mkt = rng.normal(0.001, mkt_vol, n_days)
    cols = {"BTC": 100.0 * np.exp(np.cumsum(mkt))}
    events = []
    lo, hi = 250, n_days - 60
    day0_pos = np.linspace(lo, hi, n_coins).round().astype(int)
    for j in range(n_coins):
        name = f"C{j:02d}"
        r = mkt + rng.normal(0.0, idio_vol, n_days)
        p0 = int(day0_pos[j])
        if pop != 0.0:
            r[p0 - 2: p0 + 1] += pop / 3.0
        if fade != 0.0:
            r[p0 + 1: p0 + 21] -= fade / 20.0
        cols[name] = 1.0 * np.exp(np.cumsum(r))
        events.append(dict(coin=name, day0=idx[p0], venue_ratio=1.0))
    px = pd.DataFrame(cols, index=idx)
    return px, pd.DataFrame(events)
