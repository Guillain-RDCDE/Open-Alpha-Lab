"""Data layer for Study 410 — Cup & Handle (William O'Neil's chart figure).

Two tapes, one shape (a tz-naive daily OHLC frame indexed by date):

* **Real tape.** Daily *split-adjusted* closes (and OHLC) for SPY plus a fixed basket of
  large, long-listed US large-caps (yfinance, no key). Cache-first into ``_cache/`` as one
  wide parquet per field, so the offline core and notebooks never touch the network once the
  cache exists. This is a **survivors** basket (all still trading) — survivorship is named on
  the Signal axis: a fixed surviving-names panel cannot include names that delisted, which
  tilts post-breakout forward returns mildly *up*.

* **Synthetic.** A deterministic, fixed-seed generator (:func:`synthetic_panel`) that injects
  a *planted* cup-with-handle shape before a chosen fraction of breakouts and then plants an
  extra forward drift after the breakout proportional to a knob ``edge`` (0 = null, >0 = real).
  It is the positive control: with ``edge = 0`` the post-breakout edge must NOT manufacture
  significance; with a large ``edge`` the detector + inference must light up.

The cup-with-handle is the *most subjective* of the classic figures, so we test the closest
**mechanical** definition (swing-pivot cup + a shallow handle + a confirmed breakout above the
left-rim resistance) and say so loudly. ``yfinance`` is imported lazily — only on a cache miss.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# A transparent, fixed basket of large, long-listed US large-caps with deep, clean daily
# histories on yfinance. SPY is the index proxy the README hook compares against. The basket
# is *survivors* (all still trading) — named on the Signal axis.
BASKET = [
    "SPY", "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "IBM",
    "CVX", "PFE", "MRK", "INTC", "CSCO", "HD", "MCD", "DIS", "BA", "CAT",
    "MMM", "HON", "UNH", "ORCL", "PEP", "ABT", "TXN", "COST", "LOW", "GS",
]


# --------------------------------------------------------------------------- #
# Real tape — yfinance daily OHLC, cache-first
# --------------------------------------------------------------------------- #
def _field_path(field: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"cuph_{field}.parquet")


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
        raise RuntimeError("yfinance returned no data for the cup-and-handle basket")

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
                    seed: int = 410, daily_vol: float = 0.013,
                    n_planted: int = 6, hold: int = 40
                    ) -> tuple[dict[str, pd.DataFrame], dict]:
    """Deterministic OHLC panel with PLANTED cup-with-handle shapes and a forward-edge knob.

    Each name is a daily random walk. For each name we carve ``n_planted`` clean
    cup-with-handle figures into the close path (a smooth ~U dip and recovery to the left rim,
    a shallow handle pullback, then a breakout bar above the rim). If ``edge`` != 0 the ``hold``
    sessions *after* each planted breakout get an extra daily drift of ``edge / hold`` — a real
    post-breakout continuation, the exact thing O'Neil claims. With ``edge = 0`` the path after
    a (still-shaped) breakout is pure noise: the detector must NOT manufacture significance.

    Returns ``({"open","high","low","close"}, truth)`` in the same shape as :func:`load_real`.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2008-01-02", periods=n_days)
    names = [f"N{i:02d}" for i in range(n_names)]

    o = {}
    h = {}
    lo = {}
    c = {}
    planted_breakouts = 0
    cup_w = 80          # cup width in bars
    handle_w = 12       # handle width in bars
    depth = 0.18        # cup depth (fraction)

    for name in names:
        ret = rng.normal(0.0002, daily_vol, size=n_days)
        # space the planted figures out, leaving room for the hold window
        slots = np.linspace(120, n_days - cup_w - handle_w - hold - 20,
                            n_planted).astype(int)
        for s in slots:
            # build a smooth additive bump that draws the cup: down then back to the rim
            t = np.arange(cup_w)
            cup = -depth * (np.sin(np.pi * t / (cup_w - 1)) ** 2)   # U-shaped dip, 0 at ends
            # write the cup as an additive log-level adjustment via its first difference
            cup_ret = np.diff(np.concatenate([[0.0], cup]))
            ret[s:s + cup_w] += cup_ret
            # handle: a shallow pullback (~5%) then recovery
            hbase = s + cup_w
            hand = -0.05 * (np.sin(np.pi * np.arange(handle_w) / (handle_w - 1)) ** 2)
            hand_ret = np.diff(np.concatenate([[0.0], hand]))
            ret[hbase:hbase + handle_w] += hand_ret
            # breakout bar: a clean push above the left rim
            bo = hbase + handle_w
            ret[bo] += 0.03
            planted_breakouts += 1
            if edge != 0.0:
                ret[bo + 1:bo + 1 + hold] += edge / hold

        close = 100.0 * np.exp(np.cumsum(ret))
        open_ = np.empty_like(close)
        open_[0] = 100.0
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
