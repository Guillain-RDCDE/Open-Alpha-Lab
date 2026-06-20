"""Data layer for Study 336 (Inverse-Cramer).

Three things live here:

1. ``CALLS`` — a small, **hardcoded, curated** table of notable on-air Jim Cramer
   calls (ticker, date, stated direction: ``+1`` = "buy/bull", ``-1`` = "sell/avoid/bear").
   This table is the study's original sin and we say so loudly: it is the kind of list a
   meme account assembles, *hand-picked because the outcomes are memorable* (the famous Bear
   Stearns "don't be silly" moment, the Coinbase / SVB / Meta calls, etc.). A curated list
   of "his worst moments" is a selected sample, and selection on the outcome is exactly the
   thing that manufactures an "edge" out of nothing. The third verdict axis is about this.

2. ``synthetic_calls`` — a *deterministic, offline* generator. It builds N price paths and
   a list of (path, day, stated_direction) calls. A single knob, ``anti_alpha``, controls
   the only thing an "inverse the pundit" rule can possibly harvest: a tendency for the
   stated direction to be *wrong* over the forward window. ``anti_alpha=0`` is the null —
   the pundit is a coin, fading him is a coin — so a test can assert the rule beats random
   *only* when we plant a genuine anti-signal, and not otherwise. ``selection_bias`` plants
   the *other* failure mode: a list curated so the visible calls look bad even though the
   pundit is, on average, a coin.

3. ``load_real`` — a cache-first loader of the real daily tapes for the tickers in
   ``CALLS`` via the shared :mod:`quantlab.data` parquet cache, with a graceful fallback so
   the offline core and the tests never need the network.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the call is known
*after* the show airs, so the fade is entered at the next session and held forward).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# The curated on-air call table — the study's deliberate selection bias
# ---------------------------------------------------------------------------
# direction: +1 = Cramer was BULLISH (buy / "it's fine" / endorsement),
#            -1 = Cramer was BEARISH (sell / avoid / warning).
# "Fading" means trading the OPPOSITE of `direction`.
# Dates are the on/around-air dates of widely-circulated calls. They are curated
# for *memorability*, which is precisely the bias we interrogate — do NOT read
# this as a representative sample of his recommendations.
CALLS: list[dict] = [
    {"ticker": "BSC",  "date": "2008-03-11", "direction": +1, "note": "'Bear Stearns is fine!' (days before collapse)"},
    {"ticker": "AAPL", "date": "2012-09-21", "direction": +1, "note": "Apple endorsement near a local top"},
    {"ticker": "FB",   "date": "2018-07-25", "direction": +1, "note": "Meta/Facebook before the -19% guidance gap"},
    {"ticker": "GE",   "date": "2017-05-15", "direction": +1, "note": "General Electric defended on the way down"},
    {"ticker": "WFC",  "date": "2016-09-12", "direction": +1, "note": "Wells Fargo backed before the scandal cascade"},
    {"ticker": "NFLX", "date": "2021-11-08", "direction": +1, "note": "Netflix bull near the 2021 peak"},
    {"ticker": "COIN", "date": "2021-11-10", "direction": +1, "note": "Coinbase endorsed at the crypto top"},
    {"ticker": "META", "date": "2022-02-02", "direction": +1, "note": "Meta defended before the -26% print"},
    {"ticker": "NVDA", "date": "2022-08-24", "direction": -1, "note": "Cooled on Nvidia before a major run"},
    {"ticker": "PYPL", "date": "2021-07-26", "direction": +1, "note": "PayPal bull near the peak"},
    {"ticker": "ROKU", "date": "2021-07-26", "direction": +1, "note": "Roku endorsement near the peak"},
    {"ticker": "SQ",   "date": "2021-08-02", "direction": +1, "note": "Block/Square bull near the peak"},
    {"ticker": "ZM",   "date": "2020-10-19", "direction": +1, "note": "Zoom bull near the pandemic peak"},
    {"ticker": "PTON", "date": "2021-01-11", "direction": +1, "note": "Peloton endorsement near the peak"},
    {"ticker": "ARKK", "date": "2021-02-08", "direction": +1, "note": "Innovation-ETF enthusiasm near the peak"},
    {"ticker": "BA",   "date": "2019-03-11", "direction": +1, "note": "Boeing defended into the MAX grounding"},
    {"ticker": "T",    "date": "2018-06-15", "direction": +1, "note": "AT&T bull post-merger drift"},
    {"ticker": "INTC", "date": "2021-01-22", "direction": +1, "note": "Intel bull before relative underperformance"},
    {"ticker": "DIS",  "date": "2021-12-10", "direction": +1, "note": "Disney bull near the peak"},
    {"ticker": "SBUX", "date": "2021-07-27", "direction": +1, "note": "Starbucks bull near the peak"},
    {"ticker": "MSFT", "date": "2019-04-25", "direction": +1, "note": "Microsoft bull (a call that AGED WELL — anti-anchor)"},
    {"ticker": "AMD",  "date": "2018-08-30", "direction": +1, "note": "AMD bull (aged well — anti-anchor)"},
]


def calls_table() -> pd.DataFrame:
    """The curated call table as a tidy DataFrame (date parsed, sorted)."""
    df = pd.DataFrame(CALLS).copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_calls(
    n_paths: int = 40,
    n_days: int = 400,
    n_calls: int = 200,
    anti_alpha: float = 0.0,
    selection_bias: float = 0.0,
    daily_vol: float = 0.02,
    fwd: int = 21,
    start: str = "2010-01-04",
    seed: int = 336,
) -> tuple[dict, pd.DataFrame, dict]:
    """A reproducible universe of price paths plus a list of pundit calls.

    Each path is a geometric random walk (a martingale by default). A list of
    ``n_calls`` calls is sampled: each is ``(path, day, direction)`` where
    ``direction`` is the pundit's *stated* view.

    Two independent knobs control the two ways a fade can look good:

    - ``anti_alpha`` — the **genuine anti-signal**. With probability rising in
      ``anti_alpha`` the realised forward return over the next ``fwd`` days is
      tilted *against* the stated direction (the pundit is systematically wrong).
      ``anti_alpha=0`` → the pundit is a coin and fading him is a coin (the NULL).
    - ``selection_bias`` — the **curation artefact**. The pundit is still a coin,
      but the *published* call list is filtered to over-represent calls that went
      against him ex-post. This manufactures a fade "edge" in the visible sample
      with **no** real predictability. It is the bias the third axis is about.

    Returns ``(prices, calls, truth)``:
      - ``prices``  : ``{path_id: pd.Series}`` of close prices (tz-naive daily).
      - ``calls``   : DataFrame ``[path, call_idx, direction, fwd_ret]``.
      - ``truth``   : the planted parameters.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days)

    # Build the price paths (independent geometric random walks).
    prices: dict[int, pd.Series] = {}
    for p in range(n_paths):
        log_ret = rng.normal(0.0, daily_vol, n_days)
        close = 100.0 * np.exp(np.cumsum(log_ret))
        prices[p] = pd.Series(close, index=sessions, name=f"P{p:02d}")

    # Sample candidate calls. We over-sample, then (optionally) curate down.
    over = int(np.ceil(n_calls / max(1.0 - 0.5 * selection_bias, 0.4))) + n_calls
    rows = []
    for _ in range(over):
        p = int(rng.integers(0, n_paths))
        # Leave room for the forward window.
        i = int(rng.integers(0, n_days - fwd - 1))
        stated = int(rng.choice([-1, 1]))
        s = prices[p]
        fwd_ret = float(s.iloc[i + fwd] / s.iloc[i] - 1.0)

        # Plant the genuine anti-signal: with prob ~anti_alpha, flip the realised
        # forward move to oppose the stated direction (pundit is wrong).
        if anti_alpha > 0.0 and rng.random() < anti_alpha:
            fwd_ret = -abs(fwd_ret) * stated  # forward move opposes the call
        rows.append({"path": p, "call_idx": i, "direction": stated, "fwd_ret": fwd_ret})

    cand = pd.DataFrame(rows)

    if selection_bias > 0.0:
        # Curate: keep calls that went AGAINST the pundit with higher probability.
        # A call "looks bad" when direction * fwd_ret < 0 (he said up, it went down).
        against = (cand["direction"] * cand["fwd_ret"]) < 0
        keep_prob = np.where(against, 1.0, 1.0 - selection_bias)
        keep = rng.random(len(cand)) < keep_prob
        cand = cand[keep]

    calls = cand.head(n_calls).reset_index(drop=True)

    truth = {
        "anti_alpha": anti_alpha,
        "selection_bias": selection_bias,
        "n_paths": n_paths,
        "n_days": n_days,
        "n_calls": len(calls),
        "fwd": fwd,
        "daily_vol": daily_vol,
        "seed": seed,
    }
    return prices, calls, truth


# ---------------------------------------------------------------------------
# Real tape — cache-first via quantlab.data, graceful fallback
# ---------------------------------------------------------------------------
def load_real(
    tickers: list[str] | None = None,
    start: str = "2007-01-01",
    end: str | None = None,
    mode: str = "total_return",
    fetch: bool = False,
) -> dict[str, pd.Series]:
    """Cache-first daily close series for the call tickers via :mod:`quantlab.data`.

    Returns ``{ticker: close Series}`` (tz-naive). With ``fetch=False`` (default)
    only the shared parquet cache is read; a missing cache or an unavailable
    ``quantlab`` raises so callers can fall back to the synthetic tape. Many of
    the call tickers no longer trade under that symbol (BSC, FB) — those are
    simply skipped, which is itself part of the survivorship caveat.

    ``mode='total_return'`` because a multi-week forward hold should include
    dividends; the label travels with every number that uses it.
    """
    if tickers is None:
        tickers = sorted({c["ticker"] for c in CALLS})

    from quantlab import data as qdata  # raises if quantlab unimportable

    out: dict[str, pd.Series] = {}
    for t in tickers:
        cache_path = qdata.CACHE_DIR / f"{t}_{mode}.parquet"
        # Cache-only when fetch=False: skip any ticker not already cached (no network).
        if not fetch and not cache_path.exists():
            continue
        try:
            # use_cache=True so a fetch=True miss writes the parquet for next time.
            df = qdata.fetch(t, start=start, end=end, mode=mode, use_cache=True)
        except Exception:
            # Delisted symbol or network blocked.
            continue
        if df is None or len(df) == 0:
            continue
        close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
        close = close.copy()
        if getattr(close.index, "tz", None) is not None:
            close.index = close.index.tz_localize(None)
        out[t] = close
    if not out:
        raise FileNotFoundError(
            "No cached real tapes for any call ticker. "
            "Run load_real(fetch=True) once (needs network) to populate the cache."
        )
    return out


def fingerprint_calls(df: pd.DataFrame) -> str:
    """A short content fingerprint of a call/forward-return frame."""
    cols = [c for c in ("direction", "fwd_ret") if c in df.columns]
    arr = np.ascontiguousarray(df[cols].to_numpy(dtype=float))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
