"""Data layer for Study 684 — Inverted Hammer.

Two tapes, one shape (a tz-naive daily OHLC frame indexed by date):

* **Real tape.** Daily OHLC for a fixed basket of ~26 liquid US large-caps + SPY
  (yfinance, no key), cache-first into ``_cache/`` so the reproducible core and the
  notebooks run offline once warmed. This is the same **survivors** basket used by the
  sibling candlestick studies (403-hammer-hanging-man, 404-shooting-star): every name
  still trades in 2026. Survivorship is named on the Signal axis: for a **bullish**
  reversal claim (buy after a downtrend) the bias points *toward* finding a bounce —
  survivors are, definitionally, names that recovered from their dips rather than
  delisting through them — so a null/weak result here is conservative and a positive
  result is suspect-by-survivorship. Reasoned about explicitly, not buried.

* **Synthetic.** ``synthetic_panel`` — a deterministic, fixed-seed generator with a
  **planted-edge knob**. ``edge = 0`` is a pure random walk (no bounce after a long
  upper wick); ``edge > 0`` plants a genuine post-inverted-hammer bounce so the harness
  can prove it *can* detect a real floor when one exists. It is the positive control,
  never market evidence.

The **inverted hammer** is a one-bar shape: a small real body sitting at the **bottom**
of the session range with a long **upper** shadow (and little/no lower shadow),
appearing after a **downtrend**. The folklore: buyers tested higher intraday and,
even though the close gave most of it back, the mere fact they showed up signals the
sellers are running out of room — a bullish reversal (buy). The *same geometry* after
an **uptrend** is the bearish "shooting star" (sibling study 404) — the two folk names
are the identical candle, split only by prior trend, exactly as hammer/hanging-man
(sibling study 403) split the mirror-image lower-wick shape. This study isolates the
inverted-hammer (bullish, post-downtrend) side and asks whether it beats the base rate.

No look-ahead is baked in here: the pattern is read on bar *t*; the forward return is
measured starting from bar *t+1*'s close (one documented execution lag) in ``strategy``.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# The same transparent, fixed basket of liquid US large-caps + SPY used by the sibling
# candlestick studies (403, 404) — keeps the panels directly comparable. Survivors (all
# still trade in 2026): for a BULLISH reversal claim the tilt points TOWARD a bounce
# (these names recovered from their dips), so a null/weak bullish result is conservative
# — named on the Signal axis.
BASKET = [
    "SPY", "AAPL", "MSFT", "JNJ", "PG", "KO", "JPM", "WMT", "XOM", "CVX",
    "PFE", "MRK", "INTC", "CSCO", "HD", "MCD", "DIS", "BA", "CAT", "MMM",
    "HON", "UNH", "ORCL", "PEP", "COST", "GS",
]


# --------------------------------------------------------------------------- #
# Real tape — yfinance daily OHLC, cache-first
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"ih_{safe}_1d.parquet")


def fetch_one(ticker: str, period: str = "max", cache_dir: str = DEFAULT_CACHE,
              fetch: bool = False, retries: int = 3) -> pd.DataFrame:
    """Daily OHLC for one ticker; cache-first, network only on miss (or ``fetch=True``).

    Uses **un-adjusted** OHLC (``auto_adjust=False``) because the candle *shape* (the wick
    geometry) must come from the price levels actually printed that session — an adjusted
    series rescales the whole bar and can distort the body/wick ratios the pattern depends
    on. The forward return in ``strategy`` is computed on the same un-adjusted closes; it is
    therefore **price-only** (no dividends), labelled as such everywhere.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch and os.path.exists(path):
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only on a real network call

        last_err = None
        bars = None
        for attempt in range(retries):
            try:
                raw = yf.download(ticker, period=period, interval="1d",
                                  auto_adjust=False, progress=False, threads=False)
                if raw is not None and not raw.empty:
                    bars = raw
                    break
            except Exception as e:  # pragma: no cover - network path
                last_err = e
            time.sleep(1.5 * (attempt + 1))
        if bars is None or bars.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}: {last_err}")
        if isinstance(bars.columns, pd.MultiIndex):
            bars.columns = bars.columns.get_level_values(0)
        bars = bars.rename(columns=str.lower)[["open", "high", "low", "close"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index = pd.DatetimeIndex(bars.index, name="date")
    return bars.dropna()


def have_real(cache_dir: str = DEFAULT_CACHE, basket: list[str] | None = None) -> bool:
    basket = basket or BASKET
    return all(os.path.exists(_cache_path(t, cache_dir)) for t in basket)


def load_real(cache_dir: str = DEFAULT_CACHE, basket: list[str] | None = None,
              fetch: bool = False) -> dict[str, pd.DataFrame]:
    """Cache-first map ``{ticker -> daily OHLC frame}`` for the whole basket."""
    basket = basket or BASKET
    return {t: fetch_one(t, cache_dir=cache_dir, fetch=fetch) for t in basket}


# --------------------------------------------------------------------------- #
# Synthetic positive control — deterministic, planted-edge knob
# --------------------------------------------------------------------------- #
def synthetic_panel(n_names: int = 12, n_days: int = 4000, edge: float = 0.0,
                    daily_vol: float = 0.012, ih_rate: float = 0.04,
                    seed: int = 684) -> tuple[dict[str, pd.DataFrame], dict]:
    """A deterministic OHLC panel with a PLANTED post-inverted-hammer BOUNCE knob.

    A fraction ``ih_rate`` of sessions are drawn as **inverted-hammer-shaped** bars (small
    body at the bottom of the range, long upper wick). When ``edge > 0`` the day *after* an
    inverted hammer gets an extra *positive* drift of ``edge`` (a genuine bounce off the
    exhausted low); when ``edge = 0`` the next day is pure noise — so the post-pattern
    forward return must NOT reach significance however the noise falls. This is the
    small-effect lesson on data where we know the truth.

    Returns ``({ticker -> OHLC frame}, truth)`` where ``truth`` records the planted params.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2008-01-02", periods=n_days)
    panel: dict[str, pd.DataFrame] = {}

    # the trade enters at close[t+1] after a pattern on t and is held H days, so the
    # planted bounce must land on the HELD returns (positions t+2 .. t+1+H). H here
    # matches the detector's shortest horizon so the day-1 forward return picks it up.
    H = 1
    for k in range(n_names):
        is_ih = rng.random(n_days) < ih_rate
        ret = rng.normal(0.0002, daily_vol, size=n_days)
        if edge != 0.0:
            for i in np.flatnonzero(is_ih):
                lo = i + 2                         # first held return (entry at close[t+1])
                hi = min(i + 2 + H, n_days)        # H held returns
                if lo < n_days:
                    ret[lo:hi] += edge / H        # planted BOUNCE over the held window
        close = 100.0 * np.cumprod(1.0 + ret)
        open_ = np.empty(n_days)
        open_[0] = 100.0
        open_[1:] = close[:-1]
        high = np.empty(n_days)
        low = np.empty(n_days)
        for i in range(n_days):
            o, c = open_[i], close[i]
            if is_ih[i]:
                body_top = max(o, c)
                body_bot = min(o, c)
                wick = daily_vol * (2.5 + rng.random() * 2.0) * o
                high[i] = body_top + wick                       # long UPPER wick
                low[i] = body_bot - daily_vol * 0.1 * o * rng.random()
            else:
                high[i] = max(o, c) + abs(rng.normal(0.0, daily_vol * 0.5)) * o
                low[i] = min(o, c) - abs(rng.normal(0.0, daily_vol * 0.5)) * o
        panel[f"N{k:02d}"] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close}, index=idx
        )

    truth = {"edge": edge, "ih_rate": ih_rate, "n_names": n_names,
             "n_days": n_days, "seed": seed}
    return panel, truth


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #
def fingerprint(panel: dict[str, pd.DataFrame]) -> str:
    """A short content fingerprint over the basket's close columns, for the as-of stamp."""
    h = hashlib.sha1()
    for t in sorted(panel):
        h.update(t.encode())
        h.update(np.ascontiguousarray(panel[t]["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
