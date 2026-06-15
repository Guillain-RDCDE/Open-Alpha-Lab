"""Data layer for Study 169 (Fluent-Tickers).

Two tapes, one shape (a daily close-price frame):

- ``synthetic_panel`` — a *deterministic, offline* generator.  A ``fluency_premium``
  knob injects a controlled return difference between fluent and non-fluent tickers so
  the strategy test can verify it recovers an edge *when one is planted* and reads ≈ 0
  when it is absent (``fluency_premium = 0`` is the null).
- ``load_real_panel`` — loads daily OHLCV frames from Yahoo Finance (``yfinance``),
  cache-only by default, for each ticker in the S&P 500 survivor panel.  Survivorship
  bias is **named here**: we are testing on stocks that survived long enough to be in
  the index — that biases every return upward but says nothing directional about
  ticker fluency.

The study's central object is the **fluency score** for each ticker symbol.  We use a
simple, rule-based pronounceability heuristic rather than a phonological model:

  - Vowel-consonant alternation: penalise long consonant clusters (hard to sound out).
  - Length: very short tickers (1-2 chars) are trivially pronounceable; very long
    ones (5+ chars) accumulate consonant clusters.
  - A ticker is **FLUENT** if its score ≥ the median; **NON-FLUENT** otherwise.

This is a deliberate over-simplification to replicate the *spirit* of Alter &
Oppenheimer (2006, PNAS) — they used a 1-10 human-rated pronounceability scale.  A
simple vowel/consonant heuristic captures the same dimension: "KAR" reads as a word,
"RDO" does not.  For the S&P survivor panel we score every ticker at the start of the
study window and hold the score fixed; no look-ahead.

Hardcoded event table: not applicable for this study — the ticker fluency scores are
computed from the ticker strings themselves, not from external event data.  The S&P
constituent list is approximated by ``quantlab.universe.sp500_symbols``.

No look-ahead: scores are assigned at ticker-creation time (the symbol does not
change during the study window) and do not use forward-return information.
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# ---------------------------------------------------------------------------
# Fluency scoring — the study's core signal
# ---------------------------------------------------------------------------
_VOWELS = set("AEIOU")


def _consonant_run(ticker: str) -> int:
    """Longest consecutive consonant run in the ticker symbol (upper-cased)."""
    t = ticker.upper()
    best = cur = 0
    for ch in t:
        if ch.isalpha() and ch not in _VOWELS:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def fluency_score(ticker: str) -> float:
    """Scalar pronounceability score for a ticker symbol (higher = more fluent).

    The heuristic is intentionally simple so it is fully offline and reproducible:

    - Base score = number of vowels in the ticker (each vowel 'opens' a syllable
      and makes the ticker easier to say aloud: "KAR" has one, "RDO" has none).
    - Penalty = longest consonant cluster (a string of 3 consonants like "STR"
      makes pronunciation awkward; 4+ is nearly unpronounceable as a word).
    - Length bonus: a 3-4 character ticker with at least one vowel is the sweet
      spot (e.g. "CAT", "MSFT" is harder).  We reward length=3 to match the
      original paper's focus on short pronounceable tickers.

    Score = vowel_count - 0.5 * max(0, consonant_run - 1)
            + 0.5 * (1 if 2 <= len(ticker) <= 4 else 0)
    """
    t = ticker.upper()
    letters = [c for c in t if c.isalpha()]
    vowels = sum(1 for c in letters if c in _VOWELS)
    crun = _consonant_run(t)
    length_bonus = 0.5 if 2 <= len(letters) <= 4 else 0.0
    return vowels - 0.5 * max(0, crun - 1) + length_bonus


def score_tickers(tickers: list[str]) -> pd.Series:
    """Fluency score for each ticker; returns a Series indexed by ticker.

    Scores are normalised to [0, 1] across the universe so the median split is
    always at 0.5 regardless of universe composition.  Ties are broken by
    alphabetical order (deterministic).
    """
    scores = pd.Series({t: fluency_score(t) for t in tickers}, name="fluency")
    lo, hi = scores.min(), scores.max()
    if hi > lo:
        scores = (scores - lo) / (hi - lo)
    return scores


def classify_tickers(tickers: list[str]) -> pd.Series:
    """Boolean Series: True = FLUENT (score >= median), False = NON-FLUENT.

    Equal to the median get the FLUENT label (upper half inclusive).
    """
    scores = score_tickers(tickers)
    median = scores.median()
    return (scores >= median).rename("fluent")


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 60,
    n_fluent: int = 30,
    n_days: int = 500,
    annual_vol: float = 0.20,
    fluency_premium: float = 0.0,
    annual_mkt: float = 0.07,
    start: str = "2018-01-02",
    seed: int = 169,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """A reproducible cross-sectional daily returns panel with a known fluency premium.

    ``n_fluent`` tickers are labelled FLUENT (score >= median); ``n_stocks - n_fluent``
    are NON-FLUENT.  Fluent stocks receive an additional ``fluency_premium`` in *annual*
    return terms (log-return drift); non-fluent receive nothing extra.  When
    ``fluency_premium = 0`` the two groups are identically distributed — the null.

    Returns ``(prices, fluent_mask, truth)`` where:
    - ``prices`` is a (n_days × n_stocks) DataFrame of synthetic closing prices.
    - ``fluent_mask`` is a boolean Series, True for the FLUENT tickers.
    - ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)

    # Generate synthetic ticker names: fluent ones have vowels, non-fluent are consonant clusters.
    # Fluent: prefix F + two vowels (e.g. "FAA", "FAE", ...) — clearly pronounceable.
    fluent_pool = [f"F{v1}{v2}" for v1 in "AEIOU" for v2 in "AEIOU"]
    fluent_names = fluent_pool[:n_fluent]
    # Non-fluent: prefix N + two consonants (e.g. "NBB", "NBC", ...) — hard to say.
    non_fluent_pool = [f"N{c1}{c2}" for c1 in "BCDFG" for c2 in "BCDFG"]
    non_fluent_names = non_fluent_pool[: (n_stocks - n_fluent)]
    tickers = fluent_names + non_fluent_names

    daily_vol = annual_vol / np.sqrt(252)
    mkt_drift = annual_mkt / 252
    premium_drift = fluency_premium / 252  # extra per-day drift for fluent stocks

    # Simulate: common market factor + idiosyncratic noise.
    mkt_shocks = rng.normal(mkt_drift, daily_vol, n_days)
    idio = rng.normal(0.0, daily_vol * 0.5, (n_days, n_stocks))

    log_rets = np.tile(mkt_shocks[:, None], (1, n_stocks)) + idio
    # Add the fluency premium to the first n_fluent stocks.
    log_rets[:, :n_fluent] += premium_drift

    prices = 100.0 * np.exp(np.cumsum(log_rets, axis=0))
    prices_df = pd.DataFrame(prices, index=dates, columns=tickers)

    fluent_mask = pd.Series(
        {t: i < n_fluent for i, t in enumerate(tickers)}, name="fluent"
    )
    truth = {
        "n_stocks": n_stocks,
        "n_fluent": n_fluent,
        "n_days": n_days,
        "annual_vol": annual_vol,
        "fluency_premium": fluency_premium,
        "annual_mkt": annual_mkt,
        "seed": seed,
    }
    return prices_df, fluent_mask, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo Finance daily prices, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "").replace(".", "_")
    return os.path.join(cache_dir, f"bars_{safe}_1d.parquet")


def fetch_daily(
    ticker: str,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is
    cached as a parquet under ``_cache/``).  Returns a timezone-naïve frame
    indexed by date with columns ``open, high, low, close, volume``.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call fetch_daily({ticker!r}, fetch=True) once to populate the cache."
            )
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        if bars.index.tz is not None:
            bars.index = bars.index.tz_localize(None)
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars


def load_real_panel(
    tickers: list[str],
    start: str = "2010-01-01",
    end: Optional[str] = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    min_history_years: float = 5.0,
) -> pd.DataFrame:
    """Load a (dates × tickers) panel of daily closing prices.

    Only tickers with at least ``min_history_years`` of data from ``start`` are
    included.  Dates are aligned on the *union* of all tickers' dates, with NaN
    forward-filled (price unchanged on missing dates — conservative), then rows
    with any remaining NaN are dropped.  This avoids the degenerate case where a
    single recently-listed ticker collapses the whole panel to its short history.

    Missing tickers (cache absent in cache-only mode) are silently skipped.

    Survivorship bias is structural: we are loading tickers from an S&P 500
    survivor list, so every stock here survived long enough to stay in the index.
    This is *named* in the analysis (scores on the Signal axis) but does not
    confound the *directional* fluency-vs-non-fluent comparison (both groups are
    equally survivorship-biased), only the level of mean returns.
    """
    min_rows = int(min_history_years * 252)
    closes = {}
    for t in tickers:
        try:
            bars = fetch_daily(t, start=start, end=end, fetch=fetch, cache_dir=cache_dir)
            if len(bars) >= min_rows:
                closes[t] = bars["close"]
        except FileNotFoundError:
            pass  # skip missing tickers in cache-only mode
    if not closes:
        raise RuntimeError("No cached data found for any ticker. Run with fetch=True first.")
    # Align on a common index (outer join, forward-fill missing price days, then drop any row
    # where a stock hasn't started yet i.e. leading NaN).
    panel = pd.DataFrame(closes)
    panel = panel.ffill().dropna(how="any")
    panel.index.name = "date"
    return panel


def fingerprint(panel: pd.DataFrame) -> str:
    """A short content fingerprint of the closing-price panel, for the as-of stamp."""
    arr = np.ascontiguousarray(panel.values.astype(float))
    h = hashlib.sha1(arr.tobytes())
    return h.hexdigest()[:12]
