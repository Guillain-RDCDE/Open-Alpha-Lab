"""Data layer for Study 316 (Bank-Failure).

Three things live here, all sharing one tz-naive daily shape (a Series of closes):

- ``EVENTS`` — a hardcoded, dated table of headline bank failures / forced rescues
  (Lehman, Bear Stearns, IndyMac, WaMu, Wachovia, SVB, Signature, First Republic,
  Credit Suisse, Continental Illinois ... ). Each row is a *public-knowledge* date:
  the day the failure was known to the market (announcement / FDIC seizure / forced
  sale). That is the day a trader could act, so the event window is anchored there
  with **no extra lag** (a calendar-known date needs no shift; see METHODOLOGY).

- ``synthetic_tape`` — a *deterministic, offline* generator. It builds a daily price
  series (geometric random walk) and, around a set of planted "event" dates, injects
  a known cumulative *post-event drift* (``drift`` knob). ``drift=0`` is a pure null:
  the windows around the planted dates are statistically indistinguishable from
  windows around random dates, so a test can assert the engine finds nothing. A
  positive ``drift`` is the positive control the harness must recover.

- ``load_real`` — cache-first loader for the real Yahoo! daily tape (``yfinance``),
  optionally routed through the shared ``quantlab.data`` cache. Cache-only by default
  so the test-suite and the reproducible core never touch the network; a graceful
  fallback raises a clear ``FileNotFoundError`` that callers turn into a banner.

Adjustment mode: real closes are pulled ``auto_adjust=True`` (total-return-ish,
dividends + splits folded in). We measure *cumulative abnormal-ish returns* on this
adjusted series and label it as such — never as price-only.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# The event table — public-knowledge bank-failure dates
# ---------------------------------------------------------------------------
# (date the failure/rescue was KNOWN to the market, label, region).
# Dates are the announcement / FDIC-seizure / forced-sale day — actionable dates.
EVENTS: list[tuple[str, str, str]] = [
    ("1984-05-17", "Continental Illinois (FDIC rescue announced)", "US"),
    ("2007-09-14", "Northern Rock (bank run / BoE support)", "EU"),
    ("2008-03-16", "Bear Stearns (JPMorgan forced sale)", "US"),
    ("2008-07-11", "IndyMac (FDIC seizure)", "US"),
    ("2008-09-15", "Lehman Brothers (Chapter 11)", "US"),
    ("2008-09-25", "Washington Mutual (FDIC seizure)", "US"),
    ("2008-09-29", "Wachovia (forced sale to Citi/Wells)", "US"),
    ("2011-10-31", "MF Global (bankruptcy)", "US"),
    ("2023-03-10", "Silicon Valley Bank (FDIC seizure)", "US"),
    ("2023-03-12", "Signature Bank (FDIC seizure)", "US"),
    ("2023-03-19", "Credit Suisse (forced UBS takeover)", "EU"),
    ("2023-05-01", "First Republic (FDIC seizure / JPM)", "US"),
]


def events(as_of: str | None = None) -> pd.DataFrame:
    """The event table as a DataFrame; optionally truncated to dates <= ``as_of``.

    Columns: ``date`` (tz-naive Timestamp), ``label``, ``region``.
    """
    df = pd.DataFrame(EVENTS, columns=["date", "label", "region"])
    df["date"] = pd.to_datetime(df["date"])
    if as_of is not None:
        df = df[df["date"] <= pd.Timestamp(as_of)].reset_index(drop=True)
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_tape(
    n_days: int = 6000,
    drift: float = 0.0,
    daily_vol: float = 0.011,
    n_events: int = 12,
    post_window: int = 20,
    start: str = "2000-01-03",
    seed: int = 316,
) -> tuple[pd.Series, pd.DataFrame, dict]:
    """A reproducible daily close series with a *known* post-event drift planted.

    A geometric random walk of ``n_days`` business days. ``n_events`` "event" dates
    are spread evenly through the interior of the sample; on each of the
    ``post_window`` trading days *after* an event we add ``drift / post_window`` to
    the log return, so the cumulative planted post-event move is ~``drift`` (in log
    terms) per event.

    - ``drift = 0``    → a pure null: post-event windows look like any other window.
    - ``drift > 0``    → the "buy the blood" thesis is true (rebound) — the positive
                          control the engine must recover.
    - ``drift < 0``    → the "warning" thesis is true (further decline).

    Returns ``(close, ev, truth)`` where ``close`` is a Series indexed by date,
    ``ev`` is an event table (same schema as :func:`events`), and ``truth`` records
    the planted parameters.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days)
    log_ret = rng.normal(0.0, daily_vol, n_days)

    # Plant events at evenly-spaced interior positions.
    lo, hi = post_window + 50, n_days - post_window - 5
    ev_locs = np.linspace(lo, hi, n_events).round().astype(int)
    per_day = drift / max(post_window, 1)
    for loc in ev_locs:
        log_ret[loc + 1 : loc + 1 + post_window] += per_day

    close = pd.Series(
        100.0 * np.exp(np.cumsum(log_ret)),
        index=pd.DatetimeIndex(sessions, name="date"),
        name="close",
    )
    ev = pd.DataFrame(
        {
            "date": sessions[ev_locs],
            "label": [f"synthetic_event_{i}" for i in range(n_events)],
            "region": ["SYN"] * n_events,
        }
    )
    truth = {
        "drift": drift,
        "daily_vol": daily_vol,
        "n_events": n_events,
        "post_window": post_window,
        "n_days": n_days,
        "seed": seed,
        "event_locs": ev_locs.tolist(),
    }
    return close, ev, truth


def random_dates(
    close: pd.Series,
    n: int,
    margin: int = 30,
    seed: int = 0,
) -> pd.DatetimeIndex:
    """``n`` placebo dates drawn from the interior of ``close``'s index (seeded).

    The synthetic control / placebo arm: re-run the same event-window machinery on
    these random calendar dates. Margins keep the windows inside the sample.
    """
    rng = np.random.default_rng(seed)
    idx = close.index
    lo, hi = margin, len(idx) - margin
    locs = rng.choice(np.arange(lo, hi), size=n, replace=False)
    return pd.DatetimeIndex(sorted(idx[locs]))


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-first
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"close_{safe}_daily.parquet")


def load_real(
    ticker: str = "SPY",
    start: str = "1990-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    use_quantlab: bool = True,
) -> pd.Series:
    """Real daily adjusted close for ``ticker``; cache-first, network only on miss.

    Resolution order:

    1. The study-local parquet under ``_cache/`` (written by a prior ``fetch=True``).
    2. The shared ``quantlab.data`` cache, if importable and it has the ticker
       (``use_quantlab=True``) — a graceful best-effort.
    3. If ``fetch=True``: go to the network via ``yfinance`` and cache the result.

    Otherwise raise ``FileNotFoundError`` so the caller can banner an offline run.
    Series is tz-naive, named ``close``; adjusted (auto_adjust=True) — total-return-ish.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path):
        s = pd.read_parquet(path)["close"]
        s.index = pd.DatetimeIndex(s.index)
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        s.index.name = "date"
        return s.astype(float).rename("close")

    if use_quantlab:
        try:  # best-effort: the shared engine may have a cached panel
            from quantlab import data as qdata  # type: ignore

            getter = getattr(qdata, "load_prices", None) or getattr(
                qdata, "get_close", None
            )
            if getter is not None:
                s = getter(ticker)
                s = pd.Series(s).astype(float)
                s.index = pd.DatetimeIndex(s.index)
                if s.index.tz is not None:
                    s.index = s.index.tz_localize(None)
                return s.rename("close")
        except Exception:
            pass  # fall through to the explicit fetch / raise

    if not fetch:
        raise FileNotFoundError(
            f"No cached daily tape for {ticker} at {path} (and no quantlab cache). "
            f"Call load_real({ticker!r}, fetch=True) once to populate the cache."
        )

    import yfinance as yf  # lazy: only on an explicit network call

    raw = yf.download(ticker, start=start, interval="1d", auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    s = raw["Close"].rename("close").astype(float)
    s.index = pd.DatetimeIndex(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    s.to_frame().to_parquet(path)
    return s


def fingerprint(s: pd.Series) -> str:
    """A short content fingerprint of a close series, for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(s.to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]
