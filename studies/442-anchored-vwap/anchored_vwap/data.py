"""Data layer for Study 442 — Anchored VWAP.

Two sources, both offline-friendly once cached:

* **Real tape.** Intraday **5-minute** bars (OHLCV) for a small basket of very liquid US
  ETFs/large-caps via yfinance (no key). Yahoo serves roughly **60 calendar days** of 5-minute
  history for US equities, so the real-tape window is deliberately short — stated loudly as a
  power/capacity caveat. Bars are cached as one parquet per ticker under ``_cache/`` (a tz-aware
  ``America/New_York`` index, columns ``open/high/low/close/volume``). From the bars we build,
  per session, the **anchored VWAP** line (anchored to the 09:30 session open) and detect the
  bars where price **touches/crosses** that line.

* **Synthetic (the offline CORE).** A deterministic, fixed-seed intraday session generator with
  a planted **magnet** knob ``magnet``. The price is a within-session random walk; when
  ``magnet > 0`` a mean-reverting pull toward the running AVWAP line is injected (the "magnet"
  the folklore claims). ``magnet = 0`` is the null: AVWAP is just an average of a random walk
  and carries **no** predictive structure — a touch is followed by a coin flip. A large
  ``magnet`` must make the post-touch reversion light up. This is the positive control that
  proves the harness *can* detect a magnet, so a flat real-tape result is a real "no", not a
  dead detector.

Pure numpy + pandas + stdlib for the offline path. ``fetch_bars`` (network) is only used once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

NY_TZ = "America/New_York"

# A small, transparent basket of the most liquid US tape — tight spreads, deep books, the
# friendliest possible ground for an intraday level effect. SPY/QQQ are the index ETFs; the
# three large-caps add single-name behaviour. Liquid by construction: if an AVWAP magnet does
# not survive costs *here*, it survives nowhere.
BASKET = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]

# yfinance 5-minute bars cap at ~60 calendar days for US equities.
INTERVAL = "5m"
PERIOD = "60d"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str = DEFAULT_CACHE) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_5m.parquet")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns, lowercase, tz-convert to New York."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].copy()
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    df.index = idx.tz_convert(NY_TZ)
    df.index.name = "ts"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # regular trading hours only
    return df.between_time("09:30", "16:00")


def fetch_bars(ticker: str, period: str = PERIOD, interval: str = INTERVAL,
               cache_dir: str = DEFAULT_CACHE, retries: int = 3) -> pd.DataFrame:
    """Download intraday bars for ``ticker`` via yfinance and cache a parquet.

    Network-only; used once to build the cache. Never imported by the offline notebook cells.
    Retries a couple of times with a short backoff on a rate-limit / empty pull.
    """
    import time

    import yfinance as yf

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(ticker, period=period, interval=interval,
                              auto_adjust=False, progress=False)
            if raw is not None and not raw.empty:
                bars = _normalize(raw)
                os.makedirs(cache_dir, exist_ok=True)
                bars.to_parquet(_cache_path(ticker, cache_dir))
                return bars
        except Exception as e:  # pragma: no cover - network path
            last_err = e
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"yfinance returned no {interval} bars for {ticker}: {last_err}")


def fetch_basket(tickers: list[str] | None = None, cache_dir: str = DEFAULT_CACHE) -> dict:
    """Fetch + cache the whole basket. Returns {ticker: bars}. Network-only, run once."""
    tickers = tickers or BASKET
    out = {}
    for tk in tickers:
        out[tk] = fetch_bars(tk, cache_dir=cache_dir)
    return out


def have_real(tickers: list[str] | None = None, cache_dir: str = DEFAULT_CACHE) -> bool:
    tickers = tickers or BASKET
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_bars(ticker: str, cache_dir: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load one ticker's cached 5-minute bars (tz-aware NY index)."""
    df = pd.read_parquet(_cache_path(ticker, cache_dir))
    if df.index.tz is None:
        df.index = df.index.tz_localize(NY_TZ)
    return df.sort_index()


def load_real(tickers: list[str] | None = None,
              cache_dir: str = DEFAULT_CACHE) -> dict[str, pd.DataFrame]:
    """Cached basket -> {ticker: bars}. Offline; raises only via load on a true cache miss."""
    tickers = tickers or BASKET
    return {tk: load_bars(tk, cache_dir) for tk in tickers
            if os.path.exists(_cache_path(tk, cache_dir))}


def drop_partial_sessions(bars: pd.DataFrame, min_bars: int = 78) -> pd.DataFrame:
    """Keep only *complete* RTH sessions (>= ``min_bars`` 5-minute bars = full 09:30-16:00 day).

    A stamped run never includes a partial session (the in-progress final day). 78 bars is a
    full US regular-trading-hours day at 5-minute resolution.
    """
    sess = (bars.index.tz_convert(NY_TZ) if bars.index.tz is not None
            else bars.index).normalize()
    counts = pd.Series(1, index=bars.index).groupby(sess).transform("size")
    return bars[counts >= min_bars]


def load_real_complete(tickers: list[str] | None = None,
                       cache_dir: str = DEFAULT_CACHE) -> dict[str, pd.DataFrame]:
    """``load_real`` with partial (in-progress) sessions dropped — the stamped-run loader."""
    return {tk: drop_partial_sessions(b)
            for tk, b in load_real(tickers, cache_dir).items()}


def fingerprint(bars_by_ticker: dict[str, pd.DataFrame]) -> str:
    """A short content fingerprint of the loaded basket (close columns), for the as-of stamp."""
    h = hashlib.sha1()
    for tk in sorted(bars_by_ticker):
        h.update(tk.encode())
        h.update(np.ascontiguousarray(
            bars_by_ticker[tk]["close"].to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic positive control — an intraday tape with a planted AVWAP magnet
# --------------------------------------------------------------------------- #
def synthetic_panel(n_sessions: int = 60, bars_per_session: int = 78, magnet: float = 0.0,
                    seed: int = 442, sig_bar: float = 0.0009,
                    vol_base: float = 1.0e5) -> tuple[pd.DataFrame, dict]:
    """Deterministic intraday OHLCV panel with a PLANTED AVWAP-magnet knob.

    Each session is a within-session random walk of ``bars_per_session`` 5-minute bars
    (78 bars ≈ a US RTH day). The anchored VWAP is the running volume-weighted mean price from
    the session open. The ``magnet`` knob adds a mean-reverting pull of the *next* bar's return
    toward the current AVWAP line:

        r_t = drift + magnet * (avwap_{t-1} - price_{t-1}) / price_{t-1} + e_t,
        e_t ~ N(0, sig_bar).

    With ``magnet = 0`` the AVWAP is just the running average of a random walk and carries **no**
    forecast power: a touch/cross of the line is followed by a coin flip (the null). With
    ``magnet > 0`` price is genuinely pulled back to the line, so after price stretches away and
    then touches the line again, the subsequent move reverts — the "magnet" the folklore claims.

    Returns ``(panel, truth)`` where ``panel`` is a tz-aware 5-minute OHLCV frame across the
    sessions (same shape as a real ``load_bars`` frame) and ``truth`` records the planted knob.
    """
    rng = np.random.default_rng(seed)
    drift = 0.0
    rows_ts, o, h, l, c, v = [], [], [], [], [], []

    day0 = pd.Timestamp("2024-01-02 09:30", tz=NY_TZ)
    sess_starts = pd.bdate_range(day0.normalize(), periods=n_sessions)
    for d in sess_starts:
        start = pd.Timestamp(f"{d.date()} 09:30", tz=NY_TZ)
        ts = start + pd.to_timedelta(np.arange(bars_per_session) * 5, unit="m")
        price = 100.0
        cum_pv = 0.0
        cum_v = 0.0
        avwap_prev = price
        for k in range(bars_per_session):
            # AVWAP magnet: a restoring pull of the next bar's return toward the running line.
            # magnet=0 -> AVWAP is the running mean of a random walk, no forecast power (the
            # null). magnet>0 -> price is genuinely pulled back to the line, so a touch is
            # followed by reversion. The one-sample / HAC t on the post-touch reaction is the
            # clean detector this knob is built to light up.
            pull = 0.0 if k == 0 else magnet * (avwap_prev - price) / price
            ret = drift + pull + rng.normal(0.0, sig_bar)
            new_price = price * (1.0 + ret)
            op = price
            cl = new_price
            hi = max(op, cl) * (1.0 + abs(rng.normal(0.0, sig_bar * 0.5)))
            lo = min(op, cl) * (1.0 - abs(rng.normal(0.0, sig_bar * 0.5)))
            vol = vol_base * (1.0 + abs(rng.normal(0.0, 0.3)))
            typ = (hi + lo + cl) / 3.0
            cum_pv += typ * vol
            cum_v += vol
            avwap_prev = cum_pv / cum_v
            rows_ts.append(ts[k]); o.append(op); h.append(hi); l.append(lo)
            c.append(cl); v.append(vol)
            price = new_price

    panel = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v},
                         index=pd.DatetimeIndex(rows_ts, name="ts"))
    truth = {"magnet": magnet, "n_sessions": n_sessions,
             "bars_per_session": bars_per_session, "seed": seed}
    return panel, truth
