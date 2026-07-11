"""Data layer for Study 696 — Double-Bottom (the classic W-shaped bullish reversal figure).

Two tapes, one shape (a tz-naive daily OHLC frame indexed by date, wide panel keyed by ticker):

* **Real tape.** Daily *auto-adjusted* (split + dividend) OHLC for SPY plus a fixed basket of
  large, long-listed US large-caps (yfinance, no key) — the **same 30-name survivors basket**
  used by sibling swing-pattern studies 415 (triple top/bottom) and 695 (inverse head-and-
  shoulders), so the tape and its survivorship footprint are directly comparable. Cache-first
  into ``_cache/`` as one wide parquet per OHLC field, so the offline core and notebooks never
  touch the network once the cache exists. Survivorship is named on the Signal axis: a fixed
  surviving-names panel cannot include names that delisted after a failed breakout, which tilts
  post-breakout forward returns mildly *up* — *for* this bullish figure.

* **Synthetic.** A deterministic, fixed-seed generator (:func:`synthetic_panel`) that carves a
  *planted* double-bottom shape (two troughs at one support level separated by a rally to a
  "neckline" peak, then a confirmed close above that peak) into each name's path before a chosen
  number of breakouts, then adds an extra forward drift after the breakout proportional to a
  knob ``edge`` (0 = null, >0 = a real continuation). It is the positive control: with
  ``edge = 0`` the post-breakout excess must NOT manufacture significance; with a large ``edge``
  the detector + inference must light up.

The double bottom, like every chart figure, is partly in the eye of the beholder — "two troughs
at a similar level" is exactly the kind of description three chartists draw three ways. So we
test the closest **mechanical** definition we can write down (see ``strategy.py``) and say so
loudly. ``yfinance`` is imported lazily, only on a cache miss.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# The SAME fixed, transparent 30-name large-cap survivors basket used by sibling swing-pattern
# studies 415 (triple top/bottom) and 695 (inverse head-and-shoulders) — long, clean daily
# histories on yfinance, chosen for sector spread. SPY is the index proxy. Survivorship (all
# still trading in 2026) is named on the Signal axis.
BASKET = [
    "SPY", "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "IBM",
    "CVX", "PFE", "MRK", "INTC", "CSCO", "HD", "MCD", "DIS", "BA", "CAT",
    "MMM", "HON", "UNH", "ORCL", "PEP", "ABT", "TXN", "COST", "LOW", "GS",
]

START = "2005-01-01"
AS_OF = "2026-06-30"        # last complete calendar month at publication (2026-07-10)


# --------------------------------------------------------------------------- #
# Real tape — yfinance daily OHLC, cache-first
# --------------------------------------------------------------------------- #
def _field_path(field: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"db_{field}.parquet")


def have_real(cache_dir: str = DEFAULT_CACHE) -> bool:
    return all(os.path.exists(_field_path(f, cache_dir))
               for f in ("open", "high", "low", "close"))


def fetch_panel(start: str = START, end: str | None = None,
                cache_dir: str = DEFAULT_CACHE) -> dict[str, pd.DataFrame]:
    """Download the basket's daily OHLC and cache one wide parquet per field. Network; once."""
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
        raise RuntimeError("yfinance returned no data for the double-bottom basket")

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


def load_real(cache_dir: str = DEFAULT_CACHE, asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    """Cache-first load of the daily OHLC panel, sliced to [START, asof]."""
    if not have_real(cache_dir):
        fetch_panel(cache_dir=cache_dir)
    out = {}
    hi = pd.Timestamp(asof)
    for field in ("open", "high", "low", "close"):
        df = pd.read_parquet(_field_path(field, cache_dir)).sort_index()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = pd.DatetimeIndex(df.index, name="date")
        out[field] = df.loc[df.index <= hi]
    return out


def close_for(panel: dict[str, pd.DataFrame], ticker: str) -> pd.Series:
    """One ticker's clean close series out of the wide panel."""
    return panel["close"][ticker].dropna()


# --------------------------------------------------------------------------- #
# Synthetic positive control — planted double-bottom + a tunable continuation edge
# --------------------------------------------------------------------------- #
def _carve_double_bottom(close: np.ndarray, start: int, base: float, depth: float,
                         width: int) -> None:
    """Overwrite ``close[start:start+width]`` with a clean W-shaped double-bottom.

    Five anchor points spread over ``width`` bars: the level before the pattern, the first
    trough, the middle rally back to the "neckline" (the intervening peak), the second trough
    (roughly as deep as the first — the tolerance the detector requires), and the confirmed
    close above the neckline. Linearly interpolated between anchors — smooth, deterministic,
    recognisable by the detector.
    """
    w = width
    anchors = [
        (0,          base),
        (w // 4,     base * (1.0 - depth)),        # trough 1
        (2 * w // 4, base),                          # neckline (the middle peak)
        (3 * w // 4, base * (1.0 - 0.95 * depth)),  # trough 2 (near-equal depth)
        (w - 1,      base * 1.03),                   # confirmed neckline break
    ]
    xs = np.array([a[0] for a in anchors])
    ys = np.array([a[1] for a in anchors])
    t = np.arange(w)
    close[start:start + w] = np.interp(t, xs, ys)


def synthetic_panel(n_names: int = 20, n_days: int = 2600, edge: float = 0.0,
                    seed: int = 696, daily_vol: float = 0.011,
                    n_planted: int = 8, hold: int = 40, depth: float = 0.14,
                    ) -> tuple[dict[str, pd.DataFrame], dict]:
    """Deterministic OHLC panel with PLANTED double-bottom shapes and a forward-edge knob.

    Each name is a daily random walk. ``n_planted`` clean double-bottom shapes are carved into
    the close path (see :func:`_carve_double_bottom`), each ending on a confirmed
    neckline-break bar. If ``edge`` != 0 the ``hold`` sessions *after* each planted breakout get
    an extra daily drift of ``edge / hold`` — a real post-breakout continuation, exactly what the
    folklore claims. With ``edge = 0`` the path after a (still-shaped) breakout carries only the
    base walk drift, so the excess-over-base-rate is a clean null: the detector must NOT
    manufacture significance.

    Returns ``({"open","high","low","close"}, truth)`` in the same shape as :func:`load_real`.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2008-01-02", periods=n_days)
    names = [f"N{i:02d}" for i in range(n_names)]

    o, h, lo, c = {}, {}, {}, {}
    planted_breakouts = 0
    fig_w = 70          # figure width in bars (first trough -> confirmed break)

    for name in names:
        ret = rng.normal(0.0002, daily_vol, size=n_days)
        close = 100.0 * np.exp(np.cumsum(ret))
        slots = np.linspace(120, n_days - fig_w - hold - 20, n_planted).astype(int)
        for s in slots:
            base = close[s]
            _carve_double_bottom(close, s, base, depth, fig_w)
            bo = s + fig_w - 1                     # the confirmed-break bar
            planted_breakouts += 1
            after = rng.normal(0.0002, daily_vol, size=hold)
            if edge != 0.0:
                after += edge / hold
            close[bo + 1:bo + 1 + hold] = close[bo] * np.exp(np.cumsum(after))
            tail = bo + 1 + hold
            if tail < n_days:
                tail_ret = rng.normal(0.0002, daily_vol, size=n_days - tail)
                close[tail:] = close[tail - 1] * np.exp(np.cumsum(tail_ret))

        open_ = np.empty_like(close)
        open_[0] = close[0]
        open_[1:] = close[:-1]
        wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
        hi_ = np.maximum(open_, close) + wick
        low_ = np.minimum(open_, close) - wick
        o[name], h[name], lo[name], c[name] = open_, hi_, low_, close

    panel = {
        "open": pd.DataFrame(o, index=idx),
        "high": pd.DataFrame(h, index=idx),
        "low": pd.DataFrame(lo, index=idx),
        "close": pd.DataFrame(c, index=idx),
    }
    for f in panel:
        panel[f].index = pd.DatetimeIndex(panel[f].index, name="date")
    truth = {"edge": edge, "n_names": n_names, "n_days": n_days,
             "n_planted_total": planted_breakouts, "hold": hold, "depth": depth, "seed": seed}
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
    hsh = hashlib.sha1(rows.encode("utf-8") + arr.tobytes())
    return hsh.hexdigest()[:12]
