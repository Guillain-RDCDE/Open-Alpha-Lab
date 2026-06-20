"""Data layer for Study 337 (Covered-Call-ETF).

Two tapes, both shaped as monthly frames carrying, per fund, a **price return** and a
**distribution yield** so that total return = price + distribution can be decomposed (the
study's whole point is that the headline distribution is mostly your own NAV handed back).

- ``synthetic_buywrite`` — a *deterministic, offline* generator that mechanically replicates a
  buy-write: hold the index, sell a one-month call struck ``cap`` above spot, collect a
  ``premium`` each month. The premium is paid out as a **distribution** while the **price**
  return is the index capped at ``cap``. A ``cap = +inf, premium = 0`` setting is the null
  (the fund *is* the index); a tight ``cap`` with a fat ``premium`` is the positive control —
  a big distribution yield sitting on top of an eroding price (the income illusion in a bottle).
  The synthetic is a *machinery proof*, never market evidence (see METHODOLOGY → inference bar).

- ``fetch_panel`` — the real Yahoo! monthly tape for the buy-write ETFs (JEPI, JEPQ, QYLD,
  XYLD, RYLD) and the SPY/QQQ benchmarks, **cache-first** (parquet under the shared ``_cache``)
  so the test-suite and the reproducible core never touch the network. ``auto_adjust=True``
  gives **total return** (distributions reinvested) — that is the only fair yardstick.
  ``fetch_price_and_dist`` additionally pulls split-only price + the dividend stream so the
  distribution yield can be separated from price return on the *real* funds too.

No look-ahead lives here; the racing discipline (one execution lag, excess-vs-excess) is in
``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")

# The buy-write ETFs we race, plus the benchmarks they implicitly track.
BUYWRITE = ["JEPI", "JEPQ", "QYLD", "XYLD", "RYLD"]
BENCH = ["SPY", "QQQ"]
TICKERS = BUYWRITE + BENCH

MONTHS_PER_YEAR = 12


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline buy-write replicator
# ---------------------------------------------------------------------------
def synthetic_buywrite(
    n_years: int = 12,
    cap: float = 0.025,
    premium: float = 0.015,
    mkt_mu: float = 0.009,
    mkt_vol: float = 0.15,
    start: str = "2014-01-31",
    seed: int = 337,
) -> tuple[pd.DataFrame, dict]:
    """A monthly buy-write world with the income/price split made explicit.

    Each month the index draws a log return ``g`` (drift ``mkt_mu``, monthly vol
    ``mkt_vol/sqrt(12)``). The buy-write:

      - **price return** = ``min(g, cap)`` — the index, but every up-move above the strike
        ``cap`` is shaved off (the call is exercised against you). Downside is uncapped.
      - **distribution yield** = ``premium`` — a constant payout (the call premium), the
        thing the marketing calls "income".

    Its **total return** is ``min(g, cap) + premium``. Setting ``cap = +inf, premium = 0``
    makes the buy-write identical to the index (the null). A tight ``cap`` with a fat
    ``premium`` plants the **income illusion**: a large, steady distribution yield on top of
    a price line that bleeds because the up-capture is gone.

    Returns ``(panel, truth)``. ``panel`` is a monthly frame with columns
    ``index_tr, bw_tr, bw_price, bw_dist`` (all simple monthly returns); ``truth`` records the
    planted parameters and the implied annual distribution yield.
    """
    rng = np.random.default_rng(seed)
    n = n_years * MONTHS_PER_YEAR
    # Decorative label only — period_range, never date_range(periods=BIG) (CI overflow guard).
    idx = pd.period_range(start=start[:7], periods=n, freq="M").to_timestamp(how="end")
    idx = idx.normalize()
    idx.name = "date"

    g = mkt_mu + (mkt_vol / np.sqrt(MONTHS_PER_YEAR)) * rng.standard_normal(n)
    index_tr = np.expm1(g)                      # the index's simple total return
    bw_price = np.expm1(np.minimum(g, cap))     # capped up-moves, uncapped down
    bw_dist = np.full(n, premium)               # the steady "income"
    bw_tr = bw_price + bw_dist                  # buy-write total return

    panel = pd.DataFrame(
        {"index_tr": index_tr, "bw_tr": bw_tr, "bw_price": bw_price, "bw_dist": bw_dist},
        index=idx,
    )
    truth = {
        "cap": cap,
        "premium": premium,
        "ann_dist_yield": (1.0 + premium) ** MONTHS_PER_YEAR - 1.0,
        "mkt_mu": mkt_mu,
        "mkt_vol": mkt_vol,
        "n_years": n_years,
        "seed": seed,
        "is_capped": np.isfinite(cap),
    }
    return panel, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo monthly, cache-first
# ---------------------------------------------------------------------------
def _panel_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "covered_call_etf_tr_panel.parquet")


def _split_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "covered_call_etf_price_dist.parquet")


def fetch_panel(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Monthly **total-return** (auto-adjusted) returns for the buy-write ETFs + benchmarks.

    Cache-first: reads the shared parquet if present; only touches the network on an explicit
    ``fetch=True`` (then caches). Empty frame on a cache miss with ``fetch=False`` so callers
    can fall back to the synthetic tape gracefully (the notebooks banner which tape they show).
    """
    path = _panel_path(cache_dir)
    if os.path.exists(path):
        return pd.read_parquet(path)
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf  # lazy: only on an explicit network call

    px = yf.download(TICKERS, period="max", interval="1mo",
                     auto_adjust=True, progress=False)["Close"]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    ret = px.resample("ME").last().pct_change().dropna(how="all")
    ret.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    ret.to_parquet(path)
    return ret


def fetch_price_and_dist(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Split-only **price** return and **distribution yield**, per fund, monthly.

    This is what lets us separate, on the *real* funds, how much of the total return is the
    NAV moving (price) vs cash handed out (distributions) — the income-illusion decomposition.
    Columns are a MultiIndex ``(ticker, {price, dist})``. Cache-first; empty on a cache miss.
    """
    path = _split_path(cache_dir)
    if os.path.exists(path):
        return pd.read_parquet(path)
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf

    frames = {}
    for t in TICKERS:
        tk = yf.Ticker(t)
        # Split-only adjusted price (NOT auto_adjust, so dividends are NOT folded in).
        hist = tk.history(period="max", interval="1mo", auto_adjust=False)
        if hist.empty:
            continue
        hist.index = pd.DatetimeIndex(hist.index).tz_localize(None)
        px = hist["Close"].resample("ME").last()
        divs = hist.get("Dividends", pd.Series(dtype=float)).resample("ME").sum()
        price_ret = px.pct_change()
        dist_yield = (divs / px.shift(1)).reindex(price_ret.index).fillna(0.0)
        frames[(t, "price")] = price_ret
        frames[(t, "dist")] = dist_yield
    out = pd.DataFrame(frames).dropna(how="all")
    out.index.name = "date"
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["ticker", "leg"])
    os.makedirs(cache_dir, exist_ok=True)
    out.to_parquet(path)
    return out


def fingerprint(panel: pd.DataFrame) -> str:
    """A short content fingerprint of a returns panel, for the as-of stamp."""
    arr = np.ascontiguousarray(panel.to_numpy(dtype=float, na_value=0.0))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
