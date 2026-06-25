"""Data layer for Study 493 (New-Highs-New-Lows breadth).

This study needs **market breadth**, not a single tape. Breadth is a cross-sectional
statistic: how many members of a basket are individually making new 52-week highs versus
new 52-week lows. The folklore (the "NH-NL line") is that the *net new highs* breadth line
**leads** the index — it tops/bottoms first, and breadth divergences forecast the tape.

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator of a **basket** with a
  **planted-edge knob**. We build a market factor plus idiosyncratic member paths; with
  ``edge > 0`` the *next* day's market return is pulled up in proportion to how many members
  are at fresh 52-week highs (a genuine "breadth leads price" effect the NH-NL rule can bank);
  with ``edge = 0`` breadth is pure noise versus next-day returns and the breadth-thrust entry
  is a fair coin. This is the positive control — a harness that cannot bank the planted
  breadth-lead proves nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a cached
  parquet if present and only touches the network on an explicit cache miss (with a short
  back-off + retry), then caches the parquet so re-runs are offline.

**Breadth proxy caveat.** True exchange breadth (the NYSE new-highs/new-lows that Investor's
Business Daily and the classic Hindenburg-Omen literature use) counts *thousands* of listed
issues. We cannot fetch that offline, so we proxy breadth with a small basket of liquid ETFs
spanning the broad tape, big-cap tech, small caps, large caps and gold. This is a **coarse
proxy** and it *caps* the test: a 5-to-10-name basket has far less cross-sectional resolution
than a real advance/decline universe. We state this everywhere. The Signal test is unchanged:
does the breadth signal beat **random-day** entries on SPY?

No look-ahead is baked in here — that discipline lives in ``strategy.py``: 52-week extremes use
only trailing data, breadth is read on the close of *t*, and the trade is entered at *t+1*.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# The instrument we trade / forecast (the "index"). Breadth is supposed to LEAD this tape.
INDEX_TICKER = "SPY"

# The 5 cached liquid ETFs — reused from the desk's standard basket so the study runs offline.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]

# The *preferred* breadth basket: liquid US sector ETFs + SPY. We attempt to fetch+cache these
# once (network), then fall back to DEFAULT_TICKERS when offline (the CI / gate case).
SECTOR_BASKET = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]


# --------------------------------------------------------------------------- #
# Synthetic basket — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    n_members: int = 10,
    annual_vol: float = 0.16,
    lookback: int = 252,
    start: str = "2010-01-04",
    seed: int = 493,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """A reproducible daily basket with a *known* amount of "breadth-leads-price" effect.

    Each member is a market factor (shared) plus idiosyncratic noise. We track, online and
    look-ahead-free, how many members sit at a fresh ``lookback``-day high. With ``edge > 0``
    the *next* day's market return gets an upward pull proportional to that breadth fraction
    (minus 0.5), so a breadth-thrust entry banks a real lead; with ``edge = 0`` breadth and
    next-day returns are independent and the entry is a fair coin.

    Returns ``(panel, truth)`` where ``panel`` maps ticker -> OHLC frame; ``panel['SPY']`` is
    the index (the equal-weight basket level). ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    members = [f"M{j:02d}" for j in range(n_members)]
    log_p = np.full(n_members, np.log(100.0))   # member log-prices
    closes = {m: np.empty(n_days) for m in members}
    index_level = np.empty(n_days)

    # A slow *mean-reverting* market regime makes breadth breathe between broad-strength (many
    # new highs) and broad-weakness, so the new-high fraction genuinely oscillates rather than
    # running away. Members share this regime plus idiosyncratic noise; breadth is then a real
    # cross-sectional statistic of the panel.
    #
    # The INDEX is its OWN independent random walk — crucially decoupled from the regime that
    # drives member breadth, so that at edge = 0 breadth carries *no* forward information about
    # the index (a true fair coin; no regime-momentum leak, no unstable feedback loop). The
    # planted edge adds a bounded, exogenous forward drift to the index proportional to today's
    # breadth fraction — the genuine, look-ahead-free "breadth leads price" effect the thrust
    # rule must bank.
    log_idx = np.log(100.0)
    hist: list[np.ndarray] = []
    state = 0.0
    phi = 0.97                                   # regime persistence drives member breadth only
    extra = 0.0                                  # cumulative exogenous breadth-lead drift

    for i in range(n_days):
        state = phi * state + rng.normal(0.0, daily_vol * 0.6)
        mkt = state * 0.18 + rng.normal(0.0, daily_vol * 0.5)
        idio = rng.normal(0.0, daily_vol * 0.85, n_members)
        log_p = log_p + mkt + idio
        for j, m in enumerate(members):
            closes[m][i] = np.exp(log_p[j])

        # online new-high fraction over trailing `lookback` (uses data up to and incl. i)
        hist.append(log_p.copy())
        if len(hist) > lookback:
            hist.pop(0)
        H = np.vstack(hist)                                  # (<=lookback, n_members)
        at_high = (log_p >= H.max(axis=0) - 1e-12).mean()
        at_low = (log_p <= H.min(axis=0) + 1e-12).mean()
        net = at_high - at_low                                # breadth fraction in [-1, 1]

        # the index: independent random walk + (edge>0) exogenous breadth-lead forward drift
        extra += edge * net * daily_vol * 0.7
        log_idx += rng.normal(0.0, daily_vol)
        index_level[i] = np.exp(log_idx + extra)

    def _ohlc(close: np.ndarray) -> pd.DataFrame:
        open_ = np.empty_like(close); open_[0] = close[0]; open_[1:] = close[:-1]
        wick = np.abs(rng.normal(0.0, daily_vol * 0.4, close.size)) * close
        hi = np.maximum(open_, close) + wick
        lo = np.minimum(open_, close) - wick
        return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close},
                            index=pd.DatetimeIndex(sessions, name="date"))

    panel = {m: _ohlc(closes[m]) for m in members}
    panel["SPY"] = _ohlc(index_level)                        # the index = basket level
    truth = {"edge": edge, "annual_vol": annual_vol, "lookback": lookback,
             "n_members": n_members, "n_days": n_days, "seed": seed}
    return panel, truth


# --------------------------------------------------------------------------- #
# Real tape — Yahoo daily, cache-first
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_1d.parquet")


def load_real(
    ticker: str = "SPY",
    start: str = "2005-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    allow_fetch: bool = True,
) -> pd.DataFrame:
    """Real daily OHLC for ``ticker``; **cache-first** (network only on a cache miss).

    Reads a cached parquet if present. Otherwise — and only if ``allow_fetch`` — downloads
    from yfinance (with a couple of retries + back-off on rate limits) and caches the parquet,
    so every subsequent call is fully offline.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path):
        bars = pd.read_parquet(path)
    elif allow_fetch:
        bars = _download(ticker, start, end)
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)
    else:
        raise FileNotFoundError(
            f"No cached daily tape for {ticker} at {path}. "
            f"Call load_real({ticker!r}) once (network) to populate the cache."
        )

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars[["open", "high", "low", "close"]]


def _download(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf  # lazy: only on a real cache miss

    last_err = None
    for attempt in range(3):
        try:
            raw = yf.download(ticker, start=start, end=end, interval="1d",
                              auto_adjust=True, progress=False)
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                bars = raw.rename(columns=str.lower)[["open", "high", "low", "close"]]
                bars.index.name = "date"
                return bars
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"yfinance returned no daily bars for {ticker}: {last_err}")


def load_basket(
    tickers: list[str] | None = None,
    cache_dir: str = DEFAULT_CACHE,
    allow_fetch: bool = False,
    asof: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load the breadth basket cache-first; fall back to the cached 5-ETF set when offline.

    Tries ``SECTOR_BASKET`` (the preferred breadth proxy) if ``allow_fetch`` and a member is
    cached; any ticker missing from the cache and not fetchable is skipped. When no sector ETF
    is available (the offline / CI case) this returns the 5 cached liquid ETFs, the coarse
    proxy the study is built and frozen against.
    """
    want = tickers or (SECTOR_BASKET if allow_fetch else DEFAULT_TICKERS)
    out: dict[str, pd.DataFrame] = {}
    for t in want:
        try:
            b = load_real(t, cache_dir=cache_dir, allow_fetch=allow_fetch)
        except Exception:  # noqa: BLE001 — skip un-cached, un-fetchable members
            continue
        if asof is not None:
            b = b[b.index <= asof]
        out[t] = b
    if not out:  # last-resort: the default 5
        for t in DEFAULT_TICKERS:
            b = load_real(t, cache_dir=cache_dir, allow_fetch=False)
            if asof is not None:
                b = b[b.index <= asof]
            out[t] = b
    return out


def have_real(tickers: list[str] | None = None, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every cached parquet for ``tickers`` is present (offline-safe check)."""
    tickers = tickers or DEFAULT_TICKERS
    return all(os.path.exists(_cache_path(t, cache_dir)) for t in tickers)


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
