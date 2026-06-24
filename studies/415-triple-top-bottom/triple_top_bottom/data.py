"""Data layer for Study 415 — Triple Top & Bottom (the classic three-tap reversal figure).

Two tapes, one shape (a tz-naive daily OHLC frame indexed by date):

* **Real tape.** Daily *auto-adjusted* (split + dividend) OHLC for SPY plus a fixed basket of
  large, long-listed US large-caps (yfinance, no key). Cache-first into ``_cache/`` as one wide
  parquet per field, so the offline core and notebooks never touch the network once the cache
  exists. This is a **survivors** basket (all still trading) — survivorship is named on the
  Signal axis: a fixed surviving-names panel cannot include names that delisted after a failed
  breakout, which tilts post-breakout forward returns mildly *up* (i.e. *for* the bullish
  triple-bottom, *against* the bearish triple-top).

* **Synthetic.** A deterministic, fixed-seed generator (:func:`synthetic_panel`) that injects a
  *planted* triple-bottom shape (three swing lows at one support level, then a break above the
  neckline) before a chosen number of breakouts and then plants an extra forward drift after the
  breakout proportional to a knob ``edge`` (0 = null, >0 = real). It is the positive control:
  with ``edge = 0`` the post-breakout edge must NOT manufacture significance; with a large
  ``edge`` the detector + inference must light up.

The triple top/bottom, like every chart figure, is partly in the eye of the beholder — *"three
failures at one level"* is exactly the kind of description three chartists will draw three ways.
So we test the closest **mechanical** definition we can write down — three swing highs (or lows)
clustered at one price within a tolerance, separated by meaningful pullbacks, then a confirmed
close *through* the neckline — and we say so loudly. ``yfinance`` is imported lazily, only on a
cache miss.
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
    return os.path.join(cache_dir, f"triple_{field}.parquet")


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
        raise RuntimeError("yfinance returned no data for the triple-top/bottom basket")

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
                    seed: int = 415, daily_vol: float = 0.011,
                    n_planted: int = 8, hold: int = 40
                    ) -> tuple[dict[str, pd.DataFrame], dict]:
    """Deterministic OHLC panel with PLANTED triple-bottom shapes and a forward-edge knob.

    Each name is a daily random walk. For each name we carve ``n_planted`` clean **triple
    bottoms** into the close path: a horizontal **support** floor the price taps three times
    (three lows at the same level) with rallies up to a **neckline** in between, then a
    **breakout** bar that closes above the neckline (the confirmation). If ``edge`` != 0 the
    ``hold`` sessions *after* each planted breakout get an extra daily drift of ``edge / hold`` —
    a real post-breakout continuation, exactly what the folklore claims for the bullish reversal.
    With ``edge = 0`` the path after a (still-shaped) breakout carries only the base walk drift,
    so the excess-over-base-rate is a clean null: the detector must NOT manufacture significance.

    Returns ``({"open","high","low","close"}, truth)`` in the same shape as :func:`load_real`.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2008-01-02", periods=n_days)
    names = [f"N{i:02d}" for i in range(n_names)]

    o, h, lo, c = {}, {}, {}, {}
    planted_breakouts = 0
    fig_w = 80          # figure width in bars (first low -> breakout)
    depth = 0.12        # how far the neckline sits above the support (fraction)

    for name in names:
        ret = rng.normal(0.0002, daily_vol, size=n_days)
        close = 100.0 * np.exp(np.cumsum(ret))
        slots = np.linspace(120, n_days - fig_w - hold - 20, n_planted).astype(int)
        for s in slots:
            support = close[s]                       # the flat support level (three taps)
            neckline = support * (1.0 + depth)       # the rally ceiling between the lows
            # three valleys at `support` separated by two rallies to `neckline`
            # build a smooth W-ish path: lows near s, s+fig_w/3, s+2*fig_w/3
            for t in range(fig_w):
                frac = t / (fig_w - 1)
                # three troughs -> cos with 3 full cycles dipping to support at the troughs
                phase = frac * 3.0
                osc = 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))   # 0 at troughs, 1 at peaks
                close[s + t] = support + (neckline - support) * osc
            # breakout bar: a clean close above the neckline
            bo = s + fig_w
            close[bo] = neckline * 1.03
            planted_breakouts += 1
            # continuation drift (or noise) after the breakout, then resume the walk.
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
