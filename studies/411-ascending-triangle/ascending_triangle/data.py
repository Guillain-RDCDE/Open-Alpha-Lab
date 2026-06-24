"""Data layer for Study 411 — Ascending Triangle (the classic bullish-continuation figure).

Two tapes, one shape (a tz-naive daily OHLC frame indexed by date):

* **Real tape.** Daily *auto-adjusted* (split + dividend) OHLC for SPY plus a fixed basket of
  large, long-listed US large-caps (yfinance, no key). Cache-first into ``_cache/`` as one wide
  parquet per field, so the offline core and notebooks never touch the network once the cache
  exists. This is a **survivors** basket (all still trading) — survivorship is named on the
  Signal axis: a fixed surviving-names panel cannot include names that delisted after a failed
  breakout, which tilts post-breakout forward returns mildly *up* (i.e. *for* the figure).

* **Synthetic.** A deterministic, fixed-seed generator (:func:`synthetic_panel`) that injects a
  *planted* ascending-triangle shape (flat top + rising lows) before a chosen number of breakouts
  and then plants an extra forward drift after the breakout proportional to a knob ``edge``
  (0 = null, >0 = real). It is the positive control: with ``edge = 0`` the post-breakout edge
  must NOT manufacture significance; with a large ``edge`` the detector + inference must light up.

The ascending triangle, like every chart figure, is partly in the eye of the beholder, so we
test the closest **mechanical** definition we can write down — a roughly flat resistance (a band
of swing highs at the same level), a rising lower trendline (swing lows trending up into the
apex), and a confirmed close above the resistance — and we say so loudly. ``yfinance`` is
imported lazily — only on a cache miss.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# A transparent, fixed basket of large, long-listed US large-caps with deep, clean daily
# histories on yfinance. SPY is the index proxy the README hook compares against. The basket is
# *survivors* (all still trading) — named on the Signal axis.
BASKET = [
    "SPY", "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "IBM",
    "CVX", "PFE", "MRK", "INTC", "CSCO", "HD", "MCD", "DIS", "BA", "CAT",
    "MMM", "HON", "UNH", "ORCL", "PEP", "ABT", "TXN", "COST", "LOW", "GS",
]


# --------------------------------------------------------------------------- #
# Real tape — yfinance daily OHLC, cache-first
# --------------------------------------------------------------------------- #
def _field_path(field: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"triangle_{field}.parquet")


def have_real(cache_dir: str = DEFAULT_CACHE) -> bool:
    return all(os.path.exists(_field_path(f, cache_dir))
               for f in ("open", "high", "low", "close"))


def fetch_panel(start: str = "2005-01-01", end: str | None = None,
                cache_dir: str = DEFAULT_CACHE) -> dict[str, pd.DataFrame]:
    """Download the basket's daily OHLC and cache one wide parquet per field.

    Network-only; used once to build the cache. Auto-adjusted (split + dividend) closes give a
    clean, total-return-adjusted tape — we label it as such. Retries a couple of times with a
    small backoff if yfinance rate-limits, then caches so re-runs are offline.
    """
    import time

    import yfinance as yf

    raw = None
    for attempt in range(3):
        try:
            raw = yf.download(BASKET, start=start, end=end, auto_adjust=True,
                              progress=False, group_by="column")
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            pass
        time.sleep(2.0 * (attempt + 1))
    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance returned no data for the ascending-triangle basket")

    os.makedirs(cache_dir, exist_ok=True)
    fields = {}
    for field in ("Open", "High", "Low", "Close"):
        wide = raw[field].copy()
        wide = wide.dropna(how="all")
        keep = [c for c in wide.columns if wide[c].notna().mean() >= 0.60]
        wide = wide[keep]
        if wide.index.tz is not None:
            wide.index = wide.index.tz_localize(None)
        wide.index = pd.DatetimeIndex(wide.index, name="date")
        wide.to_parquet(_field_path(field.lower(), cache_dir))
        fields[field.lower()] = wide
    return fields


def load_real(cache_dir: str = DEFAULT_CACHE) -> dict[str, pd.DataFrame]:
    """Cache-first load of the daily OHLC panel as a dict of wide frames.

    Returns ``{"open": df, "high": df, "low": df, "close": df}`` (index = date, columns =
    tickers). On a cache miss, calls :func:`fetch_panel` once (the only network touch).
    """
    if not have_real(cache_dir):
        fetch_panel(cache_dir=cache_dir)
    out = {}
    for field in ("open", "high", "low", "close"):
        df = pd.read_parquet(_field_path(field, cache_dir)).sort_index()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = pd.DatetimeIndex(df.index, name="date")
        out[field] = df
    return out


def ohlc_for(panel: dict[str, pd.DataFrame], ticker: str) -> pd.DataFrame:
    """Slice one ticker's OHLC out of the wide panel into a tidy per-name frame."""
    cols = {}
    for f in ("open", "high", "low", "close"):
        if ticker in panel[f].columns:
            cols[f] = panel[f][ticker]
    df = pd.DataFrame(cols).dropna()
    df.index = pd.DatetimeIndex(df.index, name="date")
    return df


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_names: int = 20, n_days: int = 2600, edge: float = 0.0,
                    seed: int = 411, daily_vol: float = 0.011,
                    n_planted: int = 8, hold: int = 40
                    ) -> tuple[dict[str, pd.DataFrame], dict]:
    """Deterministic OHLC panel with PLANTED ascending-triangle shapes and a forward-edge knob.

    Each name is a daily random walk. For each name we carve ``n_planted`` clean ascending
    triangles into the close path: a horizontal **resistance** ceiling the price taps several
    times while the **lows rise** toward it (a converging wedge with a flat top), then a
    **breakout** bar that pushes through the ceiling. If ``edge`` != 0 the ``hold`` sessions
    *after* each planted breakout get an extra daily drift of ``edge / hold`` — a real
    post-breakout continuation, exactly what the folklore claims. With ``edge = 0`` the path
    after a (still-shaped) breakout is pure noise: the detector must NOT manufacture significance.

    Returns ``({"open","high","low","close"}, truth)`` in the same shape as :func:`load_real`.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2008-01-02", periods=n_days)
    names = [f"N{i:02d}" for i in range(n_names)]

    o, h, lo, c = {}, {}, {}, {}
    planted_breakouts = 0
    tri_w = 70          # triangle width in bars (base -> apex)
    n_taps = 4          # times the price taps the flat resistance
    rise = 0.16         # how far the lows rise across the triangle (fraction)

    for name in names:
        ret = rng.normal(0.0002, daily_vol, size=n_days)
        close = 100.0 * np.exp(np.cumsum(ret))
        slots = np.linspace(120, n_days - tri_w - hold - 20, n_planted).astype(int)
        for s in slots:
            base = close[s]
            ceiling = base * 1.12                 # the flat resistance level
            # rising-low floor: starts ~rise below the ceiling, climbs toward it
            floor0 = ceiling * (1.0 - rise)
            for t in range(tri_w):
                frac = t / (tri_w - 1)
                floor = floor0 + (ceiling - floor0) * frac
                # a slow saw-tooth that bounces between the rising floor and the flat ceiling
                phase = (t * n_taps) / (tri_w - 1)
                osc = 0.5 * (1.0 + np.cos(2.0 * np.pi * phase))   # 1 at ceiling taps, 0 at floor
                close[s + t] = floor + (ceiling - floor) * osc
            # breakout bar: a clean push above the ceiling
            bo = s + tri_w
            close[bo] = ceiling * 1.03
            planted_breakouts += 1
            # continuation drift (or noise) after the breakout, then resume the walk.
            # With edge=0 the post-breakout path carries the SAME drift as the base walk
            # (0.0002/day), so the excess-over-base-rate is a clean null; edge>0 adds a real
            # continuation on top.
            after = rng.normal(0.0002, daily_vol, size=hold)
            if edge != 0.0:
                after += edge / hold
            close[bo + 1:bo + 1 + hold] = close[bo] * np.exp(np.cumsum(after))
            # re-anchor the random walk after the planted block so levels stay sane
            tail = bo + 1 + hold
            if tail < n_days:
                tail_ret = rng.normal(0.0002, daily_vol, size=n_days - tail)
                close[tail:] = close[tail - 1] * np.exp(np.cumsum(tail_ret))

        open_ = np.empty_like(close)
        open_[0] = close[0]
        open_[1:] = close[:-1]
        wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
        hi = np.maximum(open_, close) + wick
        low = np.minimum(open_, close) - wick
        o[name], h[name], lo[name], c[name] = open_, hi, low, close

    panel = {
        "open": pd.DataFrame(o, index=idx),
        "high": pd.DataFrame(h, index=idx),
        "low": pd.DataFrame(lo, index=idx),
        "close": pd.DataFrame(c, index=idx),
    }
    for f in panel:
        panel[f].index = pd.DatetimeIndex(panel[f].index, name="date")
    truth = {"edge": edge, "n_names": n_names, "n_days": n_days,
             "n_planted_total": planted_breakouts, "hold": hold, "seed": seed}
    return panel, truth


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #
def fingerprint(panel: dict[str, pd.DataFrame]) -> str:
    """A short content fingerprint of the panel (concatenated close columns)."""
    closes = panel["close"]
    arr = np.ascontiguousarray(np.nan_to_num(closes.to_numpy(dtype=float)).ravel())
    cols = "|".join(map(str, closes.columns))
    rows = f"{closes.index.min()}|{closes.index.max()}|{len(closes)}|{cols}"
    h = hashlib.sha1(rows.encode("utf-8") + arr.tobytes())
    return h.hexdigest()[:12]
