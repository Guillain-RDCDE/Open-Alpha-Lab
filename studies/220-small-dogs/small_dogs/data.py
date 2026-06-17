"""Data layer for Study 220 (Small Dogs of the Dow).

Reuses the point-in-time Dow membership timeline and the real price/dividend
panel from Study 88 (Dogs-of-the-Dow). The Small Dogs strategy adds a second
filter on top of the Dogs: of the ten highest-yield Dow members, take the five
with the *lowest absolute share price* at the rebalance date. The Foolish Four
is the further subset: the four highest-priced of those five (i.e. skip the
#1 lowest-priced name and take the next four).

The canonical O'Higgins-Downes source (1991) and the Motley Fool popularisation
of the Foolish Four (mid-1990s) share the same data-mining backstory: the
Foolish Four was discovered in a back-optimisation exercise on the same data
window used to test it — a canonical example of in-sample "discovery" later
found to vanish out-of-sample (see Domian & Louton 1998; McQueen, Shields &
Thorley 1997).

Membership, loading and cache strategy are identical to Study 88. We re-export
the Study 88 helpers directly so the two studies stay in sync: any fix to the
point-in-time timeline in Study 88 is automatically picked up here.

Cache note: this study writes its own yfinance cache to
``studies/220-small-dogs/_cache/`` (a separate directory from Study 88's cache)
to avoid collisions when both studies run in parallel. The study uses the
Study 88 cache *first* if it is present (read-only), falling back to its own.
"""

from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Study 88 cache (read-only fallback — if it exists, skip the network fetch).
DOGS88_CACHE = os.path.abspath(
    os.path.join(HERE, "..", "..", "88-dogs-of-the-dow", "_cache")
)

# Re-export the point-in-time membership helpers from Study 88 so any timeline
# corrections there propagate automatically.
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "88-dogs-of-the-dow")))

try:
    from dogs_of_the_dow.data import (  # type: ignore[import]
        members_asof,
        all_tickers_in_window,
        TICKER_ALIAS,
        UNRECOVERABLE,
        _cache_path,
        synthetic_panel as _base_synthetic_panel,
        fingerprint,
    )
    _dogs88_available = True
except ImportError:
    _dogs88_available = False
    # Stub fallbacks so the module still imports for tests/CI.
    def members_asof(date):  # type: ignore[misc]
        return []
    def all_tickers_in_window(**kw):  # type: ignore[misc]
        return []
    TICKER_ALIAS = {}
    UNRECOVERABLE = ()


# ---------------------------------------------------------------------------
# Real loader — re-uses Study 88 cache files where present
# ---------------------------------------------------------------------------
def load_panel(
    start: str = "1999-12-01",
    end: str | None = None,
    cache_dir: str | None = None,
    use_cache: bool = True,
) -> dict:
    """Load the Dow panel (total-return prices + dividends) for the Small Dogs engine.

    Cache strategy: prefer the Study 88 cache (``88-dogs-of-the-dow/_cache/``) if it
    exists, then fall back to this study's own ``_cache/`` directory. Network fetches are
    only triggered on a full cache miss. The returned dict is identical to Study 88's
    ``load_panel`` return shape: ``prices``, ``divs``, ``bench``, ``tickers``.
    """
    local_cache = cache_dir or STUDY_CACHE
    os.makedirs(local_cache, exist_ok=True)
    tickers = all_tickers_in_window(end=end)

    prices = {}
    divs = {}
    for t in tickers:
        px, dv = _load_one(t, local_cache, use_cache)
        if px is not None and len(px):
            prices[t] = px
        if dv is not None and len(dv):
            divs[t] = dv

    price_df = pd.DataFrame(prices).sort_index()
    div_df = pd.DataFrame(divs).sort_index()
    price_df = price_df.loc[start:end]
    bench = _load_bench(local_cache, use_cache).loc[start:end]
    return {"prices": price_df, "divs": div_df, "bench": bench, "tickers": tickers}


def _load_one(ticker: str, local_cache: str, use_cache: bool):
    """Total-return close + dividend for one ticker.

    Checks Study 88 cache first (read-only), then this study's local cache, then network.
    """
    px_name = f"{ticker}_px.parquet"
    dv_name = f"{ticker}_div.parquet"

    # 1. Study 88 cache (read-only).
    for src in (DOGS88_CACHE, local_cache):
        px_path = os.path.join(src, px_name)
        dv_path = os.path.join(src, dv_name)
        if use_cache and os.path.exists(px_path) and os.path.exists(dv_path):
            px = pd.read_parquet(px_path)["close"]
            dv = pd.read_parquet(dv_path)["div"]
            return px, dv

    # 2. Network fetch.
    try:
        import yfinance as yf
    except ImportError:
        return None, None

    fetch_symbol = TICKER_ALIAS.get(ticker, ticker)
    tk = yf.Ticker(fetch_symbol)
    raw = tk.history(period="max", auto_adjust=True)
    if raw is None or raw.empty:
        return None, None
    px = raw["Close"].rename("close")
    px.index = pd.to_datetime(px.index).tz_localize(None)
    dv = tk.dividends.rename("div")
    if dv is None or len(dv) == 0:
        dv = pd.Series(dtype=float, name="div")
    dv.index = pd.to_datetime(dv.index).tz_localize(None)
    local_px = os.path.join(local_cache, px_name)
    local_dv = os.path.join(local_cache, dv_name)
    if use_cache:
        px.to_frame().to_parquet(local_px)
        dv.to_frame().to_parquet(local_dv)
    return px, dv


def _load_bench(local_cache: str, use_cache: bool) -> pd.DataFrame:
    """DIA total-return benchmark — reuses Study 88 cache if present."""
    bench_name = "DIA_bench.parquet"
    for src in (DOGS88_CACHE, local_cache):
        path = os.path.join(src, bench_name)
        if use_cache and os.path.exists(path):
            return pd.read_parquet(path)

    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame(columns=["close"])

    raw = yf.Ticker("DIA").history(period="max", auto_adjust=True)
    close = raw["Close"].rename("close")
    close.index = pd.to_datetime(close.index).tz_localize(None)
    out = close.to_frame()
    local_path = os.path.join(local_cache, bench_name)
    if use_cache:
        out.to_parquet(local_path)
    return out


# ---------------------------------------------------------------------------
# Synthetic panel — extended for Small Dogs / Foolish Four control
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_names: int = 30,
    n_years: int = 25,
    planted: bool = True,
    excess_small: float = 0.04,
    seed: int = 220,
) -> dict:
    """Deterministic annual synthetic panel for Small Dogs / Foolish Four controls.

    Builds ``n_names`` stocks over ``n_years`` calendar years. Each name has an annual
    total return, a trailing dividend yield, and an absolute share price.

    When ``planted=True``:

    - The top-10-yield names earn ``excess_small / 2`` extra return (Dogs-level planted
      edge), AND the 5 lowest-priced of those 10 earn a further ``excess_small / 2`` extra
      return — so the Small Dogs earn the full ``excess_small`` vs the index and roughly
      ``excess_small / 2`` vs the Dogs. The Foolish Four (names 2-5 of the 5 by price) earn
      ``excess_small * 0.35`` extra vs the index.

    When ``planted=False``: all return noise is independent of both yield and price.

    Returns a dict with:
    - ``yields`` — (years x names) DataFrame
    - ``prices_abs`` — (years x names) absolute share price at rebalance
    - ``fwd_returns`` — (years x names) next-year total return
    - ``truth`` — metadata block
    """
    rng = np.random.default_rng(seed)
    years = list(range(2000, 2000 + n_years))
    names = [f"N{i:02d}" for i in range(n_names)]

    base_ret = 0.08
    vol = 0.16

    # Each stock gets a persistent "price level" that drifts; we just use the raw level for
    # ranking within each year.
    raw_prices = rng.uniform(10.0, 200.0, size=(n_years, n_names))
    yields = rng.uniform(0.005, 0.06, size=(n_years, n_names))
    noise = rng.normal(0.0, vol, size=(n_years, n_names))
    fwd = base_ret + noise

    if planted:
        for y in range(n_years):
            yield_order = np.argsort(-yields[y])
            dogs_idx = yield_order[:10]
            fwd[y, dogs_idx] += excess_small / 2  # Dogs edge

            price_order_within_dogs = np.argsort(raw_prices[y, dogs_idx])
            small_dogs_idx = dogs_idx[price_order_within_dogs[:5]]
            fwd[y, small_dogs_idx] += excess_small / 2  # Small Dogs extra edge

            # Foolish Four = names 2-5 (skip the cheapest; index 1:5 by price within dogs)
            foolish_four_idx = dogs_idx[price_order_within_dogs[1:5]]
            fwd[y, foolish_four_idx] += excess_small * 0.35 / 4  # tiny planted edge

    px_df = pd.DataFrame(raw_prices, index=years, columns=names)
    yld_df = pd.DataFrame(yields, index=years, columns=names)
    fwd_df = pd.DataFrame(fwd, index=years, columns=names)
    truth = {
        "planted": planted,
        "excess_small": excess_small if planted else 0.0,
        "n_names": n_names,
        "n_years": n_years,
        "seed": seed,
    }
    return {"yields": yld_df, "prices_abs": px_df, "fwd_returns": fwd_df, "truth": truth}


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------
def fingerprint(obj) -> str:
    """Short content fingerprint (SHA-1, first 12 hex chars) of a tape or panel dict."""
    h = hashlib.sha1()
    if isinstance(obj, dict):
        for key in ("bench", "prices"):
            if key in obj and obj[key] is not None:
                arr = np.ascontiguousarray(
                    np.nan_to_num(obj[key].to_numpy(dtype=float))
                )
                h.update(arr.tobytes())
    elif isinstance(obj, pd.DataFrame):
        h.update(
            np.ascontiguousarray(np.nan_to_num(obj.to_numpy(dtype=float))).tobytes()
        )
    else:
        h.update(
            np.ascontiguousarray(np.nan_to_num(np.asarray(obj, dtype=float))).tobytes()
        )
    return h.hexdigest()[:12]
