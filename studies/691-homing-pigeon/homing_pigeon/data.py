"""Data layer for Study 691 — Homing Pigeon.

Two tapes, one shape (a tz-naive daily OHLC frame indexed by date):

* **Real tape.** Daily OHLC for a fixed basket of ~26 liquid US large-caps + SPY
  (yfinance, no key), cache-first into ``_cache/`` so the reproducible core and the
  notebooks run offline once warmed. This is the **same survivors basket** used by the
  sibling candlestick studies (403-hammer-hanging-man, 406-harami-pattern,
  684-inverted-hammer): every name still trades in 2026. Survivorship is named on the
  Signal axis: for a **bullish** reversal claim (buy after a downtrend) the bias points
  *toward* finding a bounce — survivors are, definitionally, names that recovered from
  their dips rather than delisting through them — so a null/weak result here is
  conservative and a positive result is suspect-by-survivorship. Reasoned about
  explicitly, not buried.

* **Synthetic.** ``synthetic_panel`` — a deterministic, fixed-seed generator with a
  **planted-edge knob**. With ``edge = 0`` any post-pigeon move is pure luck (the
  detector must NOT manufacture significance); with ``edge > 0`` the day after a
  *naturally occurring* homing-pigeon shape gets a genuine extra upward drift, so the
  harness can prove it *can* bank a real floor when one exists. It is the positive
  control, never market evidence.

The **homing pigeon** is a two-bar bullish reversal: a large **down** (bearish) candle
followed by a smaller **down** candle whose real body sits entirely *inside* the prior
one, appearing after a downtrend. Unlike the [harami](../../406-harami-pattern/) (whose
two bodies have *opposite* colours), both homing-pigeon bodies are the *same* colour —
the folklore reads the shrinking down-day as sellers running out of conviction, like a
pigeon "returning home" to safety. It is a rarer, more specific cousin of the harami and
of the [inverted hammer](../../684-inverted-hammer/)'s "long upper wick" floor story.

No look-ahead is baked in here: the pattern is read on bars *t-1, t*; the forward return
is measured starting from bar *t+1*'s close (one documented execution lag) in
``strategy``.
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
# candlestick studies (403, 406, 684) — keeps the panels directly comparable. Survivors
# (all still trade in 2026): for a BULLISH reversal claim the tilt points TOWARD a bounce
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
    return os.path.join(cache_dir, f"hp_{safe}_1d.parquet")


def fetch_one(ticker: str, period: str = "max", cache_dir: str = DEFAULT_CACHE,
              fetch: bool = False, retries: int = 3) -> pd.DataFrame:
    """Daily OHLC for one ticker; cache-first, network only on miss (or ``fetch=True``).

    Uses **un-adjusted** OHLC (``auto_adjust=False``) because the candle *shape* (the two
    real bodies, one nested in the other) must come from the price levels actually printed
    that session — an adjusted series rescales the whole bar and can distort the body
    ratios the pattern depends on. The forward return in ``strategy`` is computed on the
    same un-adjusted closes; it is therefore **price-only** (no dividends), labelled as
    such everywhere.
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
def synthetic_panel(n_names: int = 30, n_days: int = 2200, edge: float = 0.0,
                    seed: int = 691, daily_vol: float = 0.013,
                    start: str = "2010-01-04") -> tuple[dict[str, pd.DataFrame], dict]:
    """Deterministic OHLC panel with a PLANTED post-homing-pigeon bounce (knob ``edge``).

    Each name is a random walk in log-price (open = prior close, no gaps). We build
    proper OHLC bars and, after each bar, test whether the *just-formed* pair is a
    textbook **homing pigeon** (see ``strategy.is_homing_pigeon``: prior bar down with a
    large body, current bar also down with a smaller body sitting fully inside the prior
    one). If ``edge != 0`` we add an *extra* upward drift of ``edge`` spread over the
    **next** bar's return — i.e. a real day-after bounce in the direction the pattern
    predicts. Shape only (no trend filter baked in here — that split is applied by
    ``strategy.conditional_returns`` on the *real* tape); this control only asks "can the
    engine detect and bank a planted reversal after this exact two-bar shape?".

    - ``edge = 0`` -> pure random walk: any post-pattern move is luck, the detector must
      NOT manufacture significance however the noise falls.
    - ``edge > 0`` -> a genuine planted bounce the detector must light up on.

    Returns ``(panel dict, truth dict)`` — the panel matches ``load_real`` in shape.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)
    panel: dict[str, pd.DataFrame] = {}
    n_planted = 0
    # A small overnight GAP (open != prior close) is essential here: with gapless opens
    # (open[t] == close[t-1] exactly), a same-colour "current body inside prior body"
    # shape is mathematically impossible (the current close would have to sit on both
    # sides of the prior close at once). Real tapes gap; this control must too, or the
    # detector can never fire and the whole positive control is vacuous.
    gap_vol = daily_vol * 0.35
    for k in range(n_names):
        eps = rng.normal(0.0002, daily_vol, n_days)
        gaps = rng.normal(0.0, gap_vol, n_days)
        close = np.empty(n_days)
        open_ = np.empty(n_days)
        hi = np.empty(n_days)
        lo = np.empty(n_days)
        wick = np.abs(rng.normal(0.0, daily_vol * 0.4, n_days))
        prev_close = 100.0
        # A homing pigeon confirmed at the close of bar i is entered at close[i+1] (one
        # execution lag) and held to close[i+1+H]; with H=1 the planted bounce must land
        # on bar i+2's RETURN (the move from the entry price to the exit price), not on
        # bar i+1 (that would just reprice the entry and, if anything, shrink the edge).
        boost_targets: set[int] = set()
        for i in range(n_days):
            boosted = i in boost_targets
            r = eps[i] + (edge if boosted else 0.0)
            if boosted:
                n_planted += 1
                boost_targets.discard(i)
            o = prev_close * np.exp(gaps[i])
            c = o * np.exp(r)
            close[i] = c
            open_[i] = o
            w = wick[i] * max(o, c)
            hi[i] = max(o, c) + w
            lo[i] = min(o, c) - w
            # detect a homing pigeon on the pair (i-1, i) -> schedule the boost at i+2
            if i >= 1:
                po0, pc0 = open_[i - 1], close[i - 1]
                po1, pc1 = o, c
                body0 = po0 - pc0     # positive iff day0 is a down day
                body1 = po1 - pc1     # positive iff day1 is a down day
                if body0 > 0 and 0 < body1 < body0:
                    lo0, hi0 = pc0, po0
                    lo1, hi1 = pc1, po1
                    if lo1 >= lo0 and hi1 <= hi0 and (i + 2) < n_days:
                        if edge != 0.0:
                            boost_targets.add(i + 2)
                        else:
                            n_planted += 1  # edge=0: still count occurrences, no boost
            prev_close = c
        panel[f"N{k:02d}"] = pd.DataFrame(
            {"open": open_, "high": hi, "low": lo, "close": close},
            index=pd.DatetimeIndex(cal, name="date"),
        )
    truth = {"edge": edge, "n_names": n_names, "n_days": n_days, "seed": seed,
             "n_planted_days": int(n_planted)}
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
