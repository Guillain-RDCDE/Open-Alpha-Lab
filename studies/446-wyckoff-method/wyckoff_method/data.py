"""Data layer for Study 446 — Wyckoff Method.

Two tapes, one shape (a tz-naive daily OHLCV frame indexed by date). Wyckoff is a
*price-AND-volume* method, so unlike most TA studies we keep the volume column — the
Spring and Upthrust events are only "real" Wyckoff signals when volume confirms.

* ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  ``edge = 0`` is a pure random walk inside random ranges (a fair coin), so the
  Spring/Upthrust detector must NOT manufacture significance. ``edge > 0`` plants exactly
  the structure Wyckoff's theory needs: an accumulation range whose **Spring** (a false
  break below support on low volume) is followed by an *extra* upward drift (the "markup"),
  and a distribution range whose **Upthrust** (false break above resistance) is followed by
  an *extra* downward drift (the "markdown"). With the planted edge the detector must light
  up — the positive control that proves the harness can detect the effect it plants.

* ``load_real`` — real daily OHLCV for SPY + a basket of liquid indices/ETFs (yfinance,
  no key), cache-first into ``_cache/`` so the reproducible core and notebooks run offline
  once cached. Daily bars give decades of ranges — far more power than an intraday study.

Wyckoff's method (accumulation/distribution "phases", the Composite Operator, the nine
buying/selling tests) is **irreducibly subjective** — the "right" phase labelling is only
clear in hindsight and two analysts annotate the same chart differently. There is no single
falsifiable rule. The tightest mechanical version proponents accept is a **trading-range**
detector (ZigZag pivots bracket a sideways range) plus the two canonical reversal events —
the **Spring** (accumulation) and the **Upthrust** (distribution) — with **volume**
confirmation. We encode exactly that in ``strategy.py`` and test the one forward-looking
prediction it makes: a Spring precedes markup, an Upthrust precedes markdown.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: an event is
detected on the close of day *t*, the trade entered at *t+1*.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# SPY = the headline tape; the rest are deep-history broad indices/ETFs Wyckoff practitioners
# draw ranges on (the "relevant indices" the brief asks for). All survivors — but these are
# *price index / ETF* tapes, not single-name baskets, so there is no cross-sectional
# survivorship sort here.
TICKERS = ("SPY", "QQQ", "DIA", "IWM", "GLD", "XLE", "EEM", "TLT")


# --------------------------------------------------------------------------- #
# Real tape — Yahoo daily, cache-first
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"wy_{safe}_1d.parquet")


def load_real(ticker: str = "SPY", period: str = "max",
              cache_dir: str = DEFAULT_CACHE, fetch: bool = False,
              retries: int = 3) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-first (network only on a cache miss or ``fetch``).

    Returns a tz-naive daily OHLCV frame (open/high/low/close/volume) indexed by date. The
    cache is a parquet under ``_cache/``; once present, every re-run is offline. On a cache
    miss (or ``fetch=True``) it pulls from yfinance with a small retry/backoff and writes the
    parquet. Volume is kept because Wyckoff is a price-AND-volume method.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path) and not fetch:
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = None
        for attempt in range(retries):
            try:
                raw = yf.download(ticker, period=period, interval="1d",
                                  auto_adjust=True, progress=False)
                if raw is not None and not raw.empty:
                    break
            except Exception:
                raw = None
            time.sleep(1.5 * (attempt + 1))
        if raw is None or raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index = pd.DatetimeIndex(bars.index, name="date")
    return bars.dropna().sort_index()


def have_real(ticker: str = "SPY", cache_dir: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(_cache_path(ticker, cache_dir))


def load_basket(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE,
                fetch: bool = False) -> dict[str, pd.DataFrame]:
    """Load every cached ticker we can (cache-first). Returns {ticker: OHLCV frame}."""
    out = {}
    for tk in tickers:
        if fetch or have_real(tk, cache_dir):
            try:
                out[tk] = load_real(tk, cache_dir=cache_dir, fetch=fetch)
            except Exception:
                continue
    return out


# --------------------------------------------------------------------------- #
# Synthetic positive control — deterministic, planted-edge knob
# --------------------------------------------------------------------------- #
def synthetic_panel(n_days: int = 9000, edge: float = 0.0, seed: int = 446,
                    daily_vol: float = 0.011, start: str = "1992-01-03"
                    ) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a planted Wyckoff Spring/Upthrust edge knob.

    The price is a low-vol random walk in log space with synthetic log-normal volume. The
    planted Wyckoff structure is added **only when ``edge != 0``** — so at ``edge = 0`` the
    tape is a pure (quiet) random walk and the detector has nothing real to find. Each
    planted structure is a textbook Wyckoff cycle:

    * a sideways **trading range** — four alternating ~9% legs (up/down/up/down) so the 5%
      ZigZag stamps the four pivots the range detector needs, bracketing a support and a
      resistance;
    * the **test event** — a one-bar poke *against* the eventual move (a **Spring** below
      support for accumulation, an **Upthrust** above resistance for distribution) on *low*
      volume (the Wyckoff "no supply / no demand" tell), then a partial snap-back into the
      range;
    * the **markup / markdown** leg — an extra drift of ``edge`` (in the accumulation/
      distribution direction) over the next ~20 bars on *rising* volume — exactly the move
      the theory predicts the Spring/Upthrust foretells.

    - ``edge = 0`` → quiet random walk, no planted events; the detector must NOT find an edge.
    - ``edge > 0`` → planted markup/markdown the Spring/Upthrust rule can harvest (the
      positive control: it must light up).

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)
    # low background noise so the planted structure dominates when present
    eps = rng.normal(0.0, daily_vol * 0.4, n_days)
    # baseline volume: log-normal, so a trailing rolling average is meaningful
    base_vol = np.exp(rng.normal(15.0, 0.30, n_days))

    osc_amp, osc_leg, markup_len, poke = 0.09, 10, 20, 0.05
    n_planted = 0
    if edge != 0.0:
        i = 80
        while i < n_days - 150:
            sign = 1.0 if rng.random() < 0.5 else -1.0  # +1 accumulation, -1 distribution
            t = i
            # 1) four alternating legs (>5% each) so the 5% ZigZag marks the range pivots
            for leg_sign in (+1, -1, +1, -1):
                eps[t:t + osc_leg] += leg_sign * osc_amp / osc_leg
                t += osc_leg
            # 2) the test event: a poke against the eventual move on LOW volume, then a
            #    partial snap-back into the range (Spring if sign>0, Upthrust if sign<0)
            eps[t] += -sign * poke
            eps[t + 1] += sign * poke * 0.6
            base_vol[t] *= 0.40
            # 3) the markup/markdown leg: extra drift in the `sign` direction, rising volume
            m0 = t + 2
            eps[m0:m0 + markup_len] += sign * edge / markup_len
            base_vol[m0:m0 + markup_len] *= 1.6
            n_planted += 1
            i = m0 + markup_len + int(rng.integers(40, 70))

    close = 100.0 * np.exp(np.cumsum(eps))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": base_vol},
        index=pd.DatetimeIndex(cal, name="date"),
    )
    truth = {"edge": edge, "n_days": n_days, "seed": seed, "daily_vol": daily_vol,
             "n_planted": n_planted}
    return bars, truth


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #
def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
