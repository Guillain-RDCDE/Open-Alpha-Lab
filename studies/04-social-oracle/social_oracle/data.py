"""Data access: a *feed* of mentions and a *panel* of prices — two objects, kept
straight, because the discipline of this study is in the join between them.

    * the **feed** — rows of ``(timestamp, ticker[, score])``: the influencer said
      ``$SYMBOL`` at a moment. This is the signal. It is messy, third-party,
      survivorship-soaked data (you only ever see the calls someone bothered to
      archive) — the opposite of a clean Yahoo series, and we say so loudly rather
      than pretend otherwise.
    * the **panel** — per-ticker daily OHLC for the names the feed mentions, each
      frame carrying the name's own return ``r_cc`` *and* the market return
      ``r_mkt`` aligned to its dates. Everything downstream measures the name's
      **abnormal** return ``r_cc - r_mkt`` — the name minus the tape it floats on —
      so a green path that was just "the whole small-cap complex ripped" never
      counts as edge.

Two ways in:

    * :func:`synthetic_panel` — fully offline. Builds a toy universe with a *baked-in
      pump-and-fade* (mentions land after a run-up, give a small same-day pop, then
      bleed back) so the whole pipeline — and the signature it's meant to detect —
      runs end-to-end before you ever touch the network.
    * :func:`load_feed` + :func:`build_panel` — bring your own CSV of mentions and
      (optionally) fetch the prices from Yahoo!. This is how a real teardown runs;
      it degrades gracefully to the cache offline.

Survivorship and selection are not footnotes here — see
:func:`social_oracle.mentions.to_events`, which returns the *coverage* counts so a
study can state exactly how many mentions it dropped, and why. (House rule: no
silent caps.)
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
import pandas as pd

AdjustMode = Literal["split_only", "total_return", "raw"]

# A broad benchmark for the "abnormal" in abnormal return. SPY (S&P 500 ETF) is the
# default tape; for a small-cap-heavy feed IWM (Russell 2000) is the honest one, and
# callers can pass their own market return series instead.
DEFAULT_MARKET = "SPY"

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "_cache")

OHLC_COLS = ["Open", "High", "Low", "Close"]


# --------------------------------------------------------------------------- #
# Per-name price fetch + cache (mirrors the desk's Study 02/03 data layer)
# --------------------------------------------------------------------------- #

def _cache_path(ticker: str, mode: AdjustMode) -> str:
    safe = ticker.replace("^", "_").replace("/", "_")
    return os.path.join(_CACHE_DIR, f"{safe}__{mode}.parquet")


def fetch(
    ticker: str,
    *,
    mode: AdjustMode = "split_only",
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return a daily OHLC(V) ``DataFrame`` for ``ticker``, indexed by date.

    Columns: ``Open, High, Low, Close`` (always) plus ``Volume`` when available —
    we keep volume because micro-cap **capacity** is the whole beat-6 question here.
    Downloads cache as parquet under ``_cache/`` so reruns are offline.
    """
    path = _cache_path(ticker, mode)
    if use_cache and os.path.exists(path):
        return _slice(pd.read_parquet(path), start, end)

    df = _download(ticker, mode)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    df.to_parquet(path)
    return _slice(df, start, end)


def _download(ticker: str, mode: AdjustMode) -> pd.DataFrame:
    import yfinance as yf  # deferred: keep the package importable offline

    raw = yf.download(ticker, period="max", auto_adjust=False, actions=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(
            f"No data returned for {ticker!r}. Micro-caps get delisted and renamed — "
            f"a dead symbol returning nothing is itself a survivorship signal."
        )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    base = raw[[c for c in OHLC_COLS]].copy()
    base.columns = OHLC_COLS
    if mode == "total_return" and "Adj Close" in raw.columns:
        base = base.mul(raw["Adj Close"] / raw["Close"], axis=0)
    # split_only / raw: micro-caps rarely pay dividends; we keep prints close to raw.
    if "Volume" in raw.columns:
        base["Volume"] = raw["Volume"]
    base.index.name = "Date"
    return base.dropna(subset=["Open", "Close"]).sort_index()


def _slice(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df.copy()


# --------------------------------------------------------------------------- #
# The feed (signal) and the panel (prices + aligned market)
# --------------------------------------------------------------------------- #

def load_feed(path: str) -> pd.DataFrame:
    """Read a mentions CSV into the canonical feed frame.

    Accepts any CSV with a timestamp column (``timestamp`` / ``date`` / ``created_at``)
    and a ticker column (``ticker`` / ``symbol`` / ``cashtag``); an optional numeric
    ``score`` / ``sentiment`` column is carried through. Tickers are upper-cased and
    a leading ``$`` is stripped. Returns columns ``timestamp, ticker[, score]``,
    sorted by time. This is the format the open-source "Serenity skill" scrapers
    emit, give or take a column name.
    """
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in lower:
                return lower[n]
        return None

    ts_col = pick("timestamp", "date", "created_at", "time")
    tk_col = pick("ticker", "symbol", "cashtag", "tag")
    if ts_col is None or tk_col is None:
        raise ValueError(
            "Feed CSV needs a timestamp column (timestamp/date/created_at) and a "
            f"ticker column (ticker/symbol/cashtag). Got: {list(df.columns)}"
        )
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df[ts_col]),
        "ticker": df[tk_col].astype(str).str.strip().str.upper().str.lstrip("$"),
    })
    sc_col = pick("score", "sentiment", "rating")
    if sc_col is not None:
        out["score"] = pd.to_numeric(df[sc_col], errors="coerce")
    return out.dropna(subset=["timestamp", "ticker"]).sort_values("timestamp").reset_index(drop=True)


def daily_returns(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Attach the close-to-close return ``r_cc`` the rest of the package consumes."""
    out = ohlc.copy()
    out["r_cc"] = out["Close"] / out["Close"].shift(1) - 1.0
    return out


def _equal_weight_market(frames: dict[str, pd.DataFrame]) -> pd.Series:
    """A fallback 'tape': the equal-weight mean daily return across all names.

    Used when no external benchmark is supplied, so ``abnormal = name - peers`` is
    always defined offline. A real teardown should pass a true broad index instead.
    """
    rets = pd.DataFrame({t: f["r_cc"] for t, f in frames.items()})
    return rets.mean(axis=1).rename("r_mkt")


def build_panel(
    prices: dict[str, pd.DataFrame],
    market_ret: pd.Series | None = None,
    clip_daily: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Turn raw per-ticker OHLC into the consumable panel.

    Each frame gets ``r_cc`` (its own return) and ``r_mkt`` (the market return
    reindexed onto its dates), so ``r_cc - r_mkt`` — the **abnormal return** every
    downstream module measures — needs no further alignment. If ``market_ret`` is
    ``None``, an equal-weight average of the panel stands in for the tape.

    ``clip_daily`` winsorizes daily returns to ``±clip_daily`` (e.g. ``1.0`` = ±100%).
    Micro-cap price data is filthy: reverse-split artefacts and bad prints produce
    physically-impossible daily "returns" (a penny stock's split day reading as
    +6,000,000%) that, summed into a cumulative-return pool, swamp every real number.
    A generous clip (±100%/day keeps every genuine meme move — a true +100% close is
    preserved) removes only the impossible. Stated as a decision, not hidden — house
    rule. ``None`` (default) leaves returns untouched (used by the clean synthetic).
    """
    frames = {t: daily_returns(df) for t, df in prices.items()}
    if clip_daily is not None:
        for f in frames.values():
            f["r_cc"] = f["r_cc"].clip(-clip_daily, clip_daily)
    if market_ret is None:
        market_ret = _equal_weight_market(frames)
    for f in frames.values():
        f["r_mkt"] = market_ret.reindex(f.index).fillna(0.0)
    return frames


# --------------------------------------------------------------------------- #
# Synthetic universe — offline, with a baked-in pump-and-fade to detect
# --------------------------------------------------------------------------- #

def synthetic_panel(
    n_tickers: int = 40,
    n_days: int = 500,
    n_mentions: int = 240,
    pop: float = 0.015,
    fade: float = -0.025,
    runup: float = 0.05,
    seed: int = 0,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """A toy universe whose mentions carry a realistic *pump-and-fade* fingerprint.

    Mechanism, deliberately mild so the *tests* of significance still have to work:
    a fraction of names get random "hype" days; on the days *before* a hype day the
    name drifts up (``runup`` — attention follows performance, so you see the move
    before the tweet), on the hype day it pops (``pop``), and over the following ~10
    sessions it bleeds back (``fade``). The market is the equal-weight tape, so the
    pop/fade show up as *abnormal* return, not just beta.

    Returns ``(panel, feed)`` ready for :func:`build_panel` is already applied:
    ``panel`` is the consumable dict (``r_cc`` + ``r_mkt`` attached), ``feed`` is the
    canonical ``timestamp, ticker, score`` frame. Numbers are random — the point is
    that the *method* (and the signature it hunts) runs end-to-end with no network.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    tickers = [f"SYM{i:02d}" for i in range(n_tickers)]

    # idiosyncratic drift + vol per name (small-cap-like: noisy)
    raw_ret = {t: 0.0002 + 0.03 * rng.standard_normal(n_days) for t in tickers}

    # place mentions: pick (ticker, day) pairs, away from the edges
    lo, hi = 12, n_days - 16
    mentions = []
    for _ in range(n_mentions):
        t = tickers[rng.integers(0, n_tickers)]
        d = int(rng.integers(lo, hi))
        mentions.append((t, d))
        r = raw_ret[t]
        r[d - 3:d] += runup / 3.0          # the run-up the tweet is chasing
        r[d] += pop                         # the same-day pop
        r[d + 1:d + 11] += fade / 10.0      # the slow fade afterwards

    prices = {}
    for t in tickers:
        r = raw_ret[t]
        close = 5.0 * np.cumprod(1.0 + r)   # micro-cap-ish price level
        prev = np.concatenate([[5.0], close[:-1]])
        open_ = prev * (1 + 0.004 * rng.standard_normal(n_days))
        high = np.maximum(open_, close) * (1 + np.abs(0.006 * rng.standard_normal(n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(0.006 * rng.standard_normal(n_days)))
        vol = rng.integers(50_000, 500_000, n_days).astype(float)
        prices[t] = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx
        )

    feed = pd.DataFrame({
        "timestamp": [idx[d] for _, d in mentions],
        "ticker": [t for t, _ in mentions],
        "score": rng.integers(50, 100, len(mentions)),
    }).sort_values("timestamp").reset_index(drop=True)

    return build_panel(prices), feed


def market_frame(
    ticker: str = DEFAULT_MARKET,
    *,
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.Series:
    """Fetch a broad-market return series (default SPY) to define abnormal returns."""
    mkt = daily_returns(fetch(ticker, start=start, end=end, use_cache=use_cache))
    return mkt["r_cc"].rename("r_mkt")
