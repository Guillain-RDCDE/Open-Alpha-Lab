"""Data layer for Study 318 (Election-Volatility).

Three things live here:

1. ``ELECTION_TABLE`` — a *hardcoded* table of US presidential **election days** (the
   Tuesday after the first Monday of November, every fourth year). Election day is a
   pure calendar fact, so there is no look-ahead in knowing it in advance: a vol seller
   can position for it weeks out. Each row carries the calendar election date; the
   event-window code snaps it to the first NYSE session on or after it (markets are open
   on election day itself in the modern era, but the snap keeps the code robust).

2. ``synthetic_daily`` — a *deterministic, offline* daily index-like tape with a planted
   "election volatility" knob. ``vol_bump`` scales how much daily volatility rises in a
   window *before* each election (the realized-vol spike), and ``vrp`` plants a gap
   between an implied-vol proxy and the realized vol that follows (the variance-risk
   premium a vol seller would harvest). ``vol_bump=0, vrp=0`` is the null: elections are
   ordinary days and selling vol into them earns nothing. The positive control lets the
   tests assert the machinery recovers a planted spike / premium only when present.

3. ``load_real`` — a cache-first loader for the real ^GSPC (price, for realized vol) and
   ^VIX (implied vol) daily tapes, via this study's ``_cache`` parquet, the shared
   ``quantlab`` cache, or ``yfinance`` on an explicit cache miss — with a graceful
   fallback so the offline core and tests never touch the network.

One execution-lag convention runs through the study's tradable overlay (``strategy.py``):
a vol position is opened at the **close** of the entry session and earns the realized
move of the *next* sessions — a single ``shift``, applied once. Election day is known in
advance (a calendar rule), so the *entry timing* itself needs no lag.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))
# The repo-root shared cache (gitignored), where quantlab parks ^GSPC / ^VIX parquets.
SHARED_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# The US presidential election-day table (a pure calendar fact).
# ---------------------------------------------------------------------------
# Election day = the Tuesday after the first Monday of November, every 4 years. We list
# the modern set our tapes can actually cover (^GSPC back to 1928; ^VIX back to 1990).
_ELECTIONS: list[tuple[str, str]] = [
    ("1928-11-06", "Hoover def. Smith"),
    ("1932-11-08", "FDR def. Hoover"),
    ("1936-11-03", "FDR def. Landon"),
    ("1940-11-05", "FDR def. Willkie"),
    ("1944-11-07", "FDR def. Dewey"),
    ("1948-11-02", "Truman def. Dewey"),
    ("1952-11-04", "Eisenhower def. Stevenson"),
    ("1956-11-06", "Eisenhower def. Stevenson"),
    ("1960-11-08", "Kennedy def. Nixon"),
    ("1964-11-03", "Johnson def. Goldwater"),
    ("1968-11-05", "Nixon def. Humphrey"),
    ("1972-11-07", "Nixon def. McGovern"),
    ("1976-11-02", "Carter def. Ford"),
    ("1980-11-04", "Reagan def. Carter"),
    ("1984-11-06", "Reagan def. Mondale"),
    ("1988-11-08", "Bush def. Dukakis"),
    ("1992-11-03", "Clinton def. Bush"),
    ("1996-11-05", "Clinton def. Dole"),
    ("2000-11-07", "Bush v. Gore (contested)"),
    ("2004-11-02", "Bush def. Kerry"),
    ("2008-11-04", "Obama def. McCain"),
    ("2012-11-06", "Obama def. Romney"),
    ("2016-11-08", "Trump def. Clinton"),
    ("2020-11-03", "Biden def. Trump"),
    ("2024-11-05", "Trump def. Harris"),
]


def election_table(min_year: int | None = None) -> pd.DataFrame:
    """The election-day table as a frame: ``election_date`` (Timestamp) + ``label``.

    ``min_year`` optionally trims to elections on or after that year (e.g. 1990 for the
    VIX era). The election date is a calendar fact, known years ahead — no look-ahead.
    """
    df = pd.DataFrame(_ELECTIONS, columns=["election_date", "label"])
    df["election_date"] = pd.to_datetime(df["election_date"])
    if min_year is not None:
        df = df[df["election_date"].dt.year >= min_year]
    return df.sort_values("election_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 6000,
    vol_bump: float = 0.0,
    vrp: float = 0.0,
    n_elections: int = 16,
    spacing: int = 1008,  # ~4 trading years between elections
    base_vol: float = 0.010,
    pre_window: int = 10,
    start: str = "1995-01-03",
    seed: int = 318,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict]:
    """A reproducible daily tape with a *known* pre-election vol spike and VRP planted.

    The price is a random walk in log returns whose conditional volatility is
    ``base_vol`` everywhere except in the ``pre_window`` sessions *before* each planted
    election, where it is multiplied by ``(1 + vol_bump)``. An implied-vol proxy
    (``vix``) is built as the trailing realized vol plus a constant premium and, crucially,
    an *extra* ``vrp`` of implied vol layered onto the pre-election window — modelling the
    fact that the option market prices election risk *in advance*. The variance-risk
    premium a vol seller harvests is implied-minus-subsequent-realized; with ``vrp > 0``
    the implied bump exceeds the realized bump, so a short-vol seller is paid.

    - ``vol_bump = 0, vrp = 0`` → the null: elections are ordinary; nothing to harvest.
    - ``vol_bump > 0``          → a planted realized-vol spike the event study must find.
    - ``vrp > 0``               → implied over-prices realized; short-vol carry is paid.

    Returns ``(bars, vix, elections, truth)`` with the same shapes the real loaders
    return, so the same analysis code runs on synthetic and real tapes unchanged.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days)

    # Plant elections on a regular ~4-year grid, away from the edges.
    margin = max(pre_window + 25, 30)
    locs = []
    loc = margin
    for _ in range(n_elections):
        if loc >= n_days - margin:
            break
        locs.append(loc)
        loc += spacing
    locs = np.array(locs, dtype=int)

    # Per-session conditional vol: base everywhere, bumped in the pre-election window.
    cond_vol = np.full(n_days, base_vol, dtype=float)
    for e in locs:
        cond_vol[max(e - pre_window, 0):e] *= (1.0 + vol_bump)

    log_ret = rng.normal(0.0, 1.0, n_days) * cond_vol
    close = 100.0 * np.exp(np.cumsum(log_ret))

    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1] * np.exp(rng.normal(0.0, base_vol * 0.3, n_days - 1))
    wick = np.abs(rng.normal(0.0, base_vol * 0.5, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(40_000_000, 120_000_000, n_days).astype(float)
    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(sessions, name="date"),
    )

    # Implied-vol proxy (annualised %, VIX-like): trailing realized vol + constant
    # premium, with an EXTRA vrp bump in the pre-election window (priced in advance).
    realized_ann = (
        pd.Series(log_ret, index=bars.index)
        .rolling(21, min_periods=5).std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0
    )
    realized_ann = realized_ann.bfill()
    implied = realized_ann + 4.0  # a constant baseline VRP, in vol points
    vrp_bump = np.zeros(n_days)
    for e in locs:
        vrp_bump[max(e - pre_window, 0):e] = vrp
    vix = (implied + pd.Series(vrp_bump, index=bars.index)).rename("vix")

    elections = pd.DataFrame(
        {
            "election_date": bars.index[locs],
            "label": [f"planted election {i + 1}" for i in range(len(locs))],
        }
    )
    truth = {
        "vol_bump": vol_bump,
        "vrp": vrp,
        "pre_window": pre_window,
        "n_elections": int(len(locs)),
        "base_vol": base_vol,
        "n_days": n_days,
        "seed": seed,
    }
    return bars, vix, elections, truth


# ---------------------------------------------------------------------------
# Real tapes — cache-first, graceful fallback
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_daily.parquet")


def _shared_candidates(ticker: str):
    """Filenames the shared quantlab cache might use for ``ticker``."""
    safe = ticker  # shared cache keeps the ^ prefix
    return [
        f"{safe}_split_only.parquet",
        f"{safe}_raw.parquet",
        f"{safe}_total_return.parquet",
    ]


def load_real(
    ticker: str = "^GSPC",
    start: str = "1990-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-first, network only on ``fetch=True``.

    Resolution order:
      1. this study's own ``_cache/bars_<ticker>_daily.parquet`` (gitignored);
      2. the shared ``quantlab`` cache (``<ticker>_split_only|raw|total_return.parquet``);
      3. (only with ``fetch=True``) the network via ``yfinance``, then cached locally.

    Raises ``FileNotFoundError`` when nothing is cached and ``fetch`` is False, so the
    offline core and tests never silently reach for the network. ^GSPC and ^VIX are
    price/level indices (no dividends) — labelled price-only, not total-return.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path):
        return _clean(pd.read_parquet(path))

    # 2) shared cache, best-effort. Prefer quantlab.data's CACHE_DIR when it resolves to
    #    a real directory, else fall back to the repo-root ``_cache`` next to quantlab/.
    shared_dirs = [SHARED_CACHE]
    try:
        import quantlab.data as qd  # noqa

        shared_dirs.insert(0, os.path.abspath(str(qd.CACHE_DIR)))
    except Exception:
        pass
    for sdir in shared_dirs:
        for fname in _shared_candidates(ticker):
            shared = os.path.join(sdir, fname)
            if os.path.exists(shared):
                return _clean(_normalise_cols(pd.read_parquet(shared)))

    if not fetch:
        raise FileNotFoundError(
            f"No cached daily tape for {ticker} at {path}. "
            f"Call load_real({ticker!r}, fetch=True) once to populate the cache."
        )

    import yfinance as yf  # lazy: only when we actually go to the network

    raw = yf.download(ticker, start=start, interval="1d", auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    bars = _normalise_cols(raw)
    os.makedirs(cache_dir, exist_ok=True)
    bars.to_parquet(path)
    return _clean(bars)


def _normalise_cols(raw: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in raw.columns}
    keep = [cols[c] for c in ("open", "high", "low", "close", "volume") if c in cols]
    bars = raw[keep].copy()
    bars.columns = [c.lower() for c in bars.columns]
    bars.index.name = "date"
    return bars


def _clean(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars.sort_index()


def load_vix(fetch: bool = False, cache_dir: str = DEFAULT_CACHE) -> pd.Series:
    """The ^VIX implied-vol level as a Series (the ``close`` column), cache-first."""
    bars = load_real("^VIX", fetch=fetch, cache_dir=cache_dir)
    return bars["close"].rename("vix")


def fingerprint(obj) -> str:
    """A short content fingerprint of a tape's close column (or a Series)."""
    arr = obj["close"].to_numpy() if isinstance(obj, pd.DataFrame) else obj.to_numpy()
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
