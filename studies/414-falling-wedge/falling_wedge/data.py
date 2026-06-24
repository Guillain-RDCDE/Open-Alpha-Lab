"""Data layer for Study 414 — Falling Wedge (the classic bullish-reversal figure).

Two tapes, one shape (a tz-naive daily OHLC frame indexed by date):

* **Real tape.** Daily *auto-adjusted* (split + dividend) OHLC for SPY plus a fixed basket of
  large, long-listed US large-caps (yfinance, no key). Cache-first into ``_cache/`` as one wide
  parquet per field, so the offline core and notebooks never touch the network once the cache
  exists. This is a **survivors** basket (all still trading) — survivorship is named on the
  Signal axis: a fixed surviving-names panel cannot include names that delisted after a failed
  wedge break, which tilts post-breakout forward returns mildly *up* (i.e. *for* the figure).

* **Synthetic.** A deterministic, fixed-seed generator (:func:`synthetic_panel`) that carves a
  *planted* falling-wedge shape (two converging, downward-sloping trendlines) into the close path
  before a chosen number of upside breakouts and then plants an extra forward drift after the
  breakout proportional to a knob ``edge`` (0 = null, >0 = real). It is the positive control:
  with ``edge = 0`` the post-breakout edge must NOT manufacture significance; with a large
  ``edge`` the detector + inference must light up.

A falling wedge, like every chart figure, is partly in the eye of the beholder, so we test the
closest **mechanical** definition we can write down — two downward-sloping trendlines (through the
swing highs and the swing lows) that **converge** as the figure narrows, with the highs falling
faster than the lows, followed by a confirmed close above the upper (resistance) line — and we say
so loudly. ``yfinance`` is imported lazily — only on a cache miss.
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
    return os.path.join(cache_dir, f"wedge_{field}.parquet")


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
        raise RuntimeError("yfinance returned no data for the falling-wedge basket")

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
                    seed: int = 414, daily_vol: float = 0.011,
                    n_planted: int = 8, hold: int = 40
                    ) -> tuple[dict[str, pd.DataFrame], dict]:
    """Deterministic OHLC panel with PLANTED falling-wedge shapes and a forward-edge knob.

    Each name is a daily random walk. For each name we carve ``n_planted`` clean falling wedges
    into the close path: two **downward-sloping trendlines** that **converge** — an upper line
    (through the swing highs) falling faster than a lower line (through the swing lows), so the
    range narrows toward the apex — then a **breakout** bar that pushes back **above** the upper
    line. If ``edge`` != 0 the ``hold`` sessions *after* each planted breakout get an extra daily
    drift of ``edge / hold`` — a real post-breakout continuation, exactly what the folklore
    claims. With ``edge = 0`` the path after a (still-shaped) breakout carries only the base-rate
    drift: the detector must NOT manufacture significance.

    Returns ``({"open","high","low","close"}, truth)`` in the same shape as :func:`load_real`.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2008-01-02", periods=n_days)
    names = [f"N{i:02d}" for i in range(n_names)]

    o, h, lo, c = {}, {}, {}, {}
    planted_breakouts = 0
    wedge_w = 70         # wedge width in bars (base -> apex)
    n_osc = 4            # number of swing-high taps across the wedge
    top_drop = 0.18      # how far the upper line falls across the wedge (fraction)
    bot_drop = 0.06      # how far the lower line falls (smaller -> the lines converge)
    gap = 20             # quiet bars on each side of a wedge (isolate the figure)
    # one self-contained slot = gap + wedge + breakout + hold + gap
    slot_len = gap + wedge_w + 1 + hold + gap

    for name in names:
        close = np.empty(n_days, dtype=float)
        # a very quiet backbone so the planted wedge is the only structure the detector sees;
        # the backbone drifts at the same base rate as the post-breakout null path.
        backbone = rng.normal(0.00015, daily_vol * 0.35, size=n_days)
        close[:] = 100.0 * np.exp(np.cumsum(backbone))
        # deterministic, non-overlapping slots
        n_slots = min(n_planted, (n_days - 40) // slot_len)
        for k in range(n_slots):
            s = 30 + k * slot_len + gap          # wedge start
            base = close[s - 1]
            top0 = base                          # upper line start (highest high)
            bot0 = base * (1.0 - 0.12)           # lower line start, below the top
            for t in range(wedge_w):
                frac = t / (wedge_w - 1)
                top = top0 * (1.0 - top_drop * frac)     # upper line falls fast
                bot = bot0 * (1.0 - bot_drop * frac)     # lower line falls slow -> converge
                # a saw-tooth that bounces between the converging lines
                phase = (t * n_osc) / (wedge_w - 1)
                osc = 0.5 * (1.0 + np.cos(2.0 * np.pi * phase))   # 1 at top, 0 at bottom
                close[s + t] = bot + (top - bot) * osc
            # breakout bar: a clean close just above the (now-low) upper line — not a spike,
            # so the post-breakout forward window measures the planted continuation, not a snap-back.
            bo = s + wedge_w
            top_apex = top0 * (1.0 - top_drop)           # upper line value at the apex
            close[bo] = top_apex * 1.01
            planted_breakouts += 1
            # continuation drift (or noise) over the hold window after the breakout.
            after = rng.normal(0.00015, daily_vol * 0.35, size=hold)
            if edge != 0.0:
                after += edge / hold
            end = min(bo + 1 + hold, n_days)
            close[bo + 1:end] = close[bo] * np.exp(np.cumsum(after[:end - bo - 1]))
            # re-anchor the quiet backbone after the block onto the post-block level so the next
            # isolated slot starts clean (slots never overlap by construction).
            tail = end
            if tail < n_days:
                tail_ret = rng.normal(0.00015, daily_vol * 0.35, size=n_days - tail)
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
