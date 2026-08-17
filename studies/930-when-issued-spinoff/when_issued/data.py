"""Data layer for Study 930 — the when-issued window of a US spin-off.

Two tapes, one shape (a date-indexed daily total-return close frame plus an event table):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for every parent, every child, the benchmark
  (SPY) and the cash leg (BIL). ``fetch`` touches the network and writes parquet into
  the **shared** desk cache ``studies/_cache`` under the house convention
  ``prices_<TICKER>_1d.parquet`` with a single ``close`` column (retry up to 4x).
  ``load_prices`` reads that cache **offline** and never imports yfinance. The whole
  test-suite runs with NO cache present (synthetic-only), so CI is green on a fresh
  checkout.

- ``synthetic_daily`` / ``synthetic_panel`` — a *deterministic, offline* generator. It
  builds a full daily world (benchmark, cash, one parent and one child per event) with
  a **planted** announcement-to-distribution parent run-up and a planted child
  regular-way drift, and returns it in exactly the schema the real loader produces, so
  the estimator is exercised by the identical code path. The ``signal_strength`` knob
  scales both planted effects: at ``signal_strength=0`` the world is a clean null and
  the estimator must stay quiet.

**The event table is a non-tape input.** ``SPINOFF_EVENTS`` is hand-compiled from the
public record (Form 10 / 8-K filings and the financial press). Announcement dates, the
first regular-way session and the distribution ratios are all **ASSUMPTIONS** carried
into the study, not things measured off the tape:

- ``announce`` — the date the separation was publicly announced. Announcements land
  pre-market, intraday or after the close, so the parent leg is entered at the close of
  the **next** session. ``examples/verify.py`` sweeps this entry by +/- 5 sessions.
- ``regular_way`` — the first session the child trades **regular way** (post-distribution).
  Yahoo's child series sometimes begins earlier, during the when-issued period; anchoring
  on an explicit date rather than "the child's first print" is what keeps the two regimes
  apart. The loader snaps to the first available child session on or after this date.
- ``ratio`` — child shares received per parent share. Used **only** by the combined
  parent+child leg, and swept there (x0.5 / x1 / x2 / equal-weight).

**Survivorship — two channels, both named.** First, the table is *curated*: liquid, widely
covered US spin-offs, so obscure and micro-cap spins are absent by construction. Second,
Yahoo no longer serves a price series for a child that has since been acquired and
delisted, which silently removes those events: ``DELISTED_EVENTS`` records the four this
study lost that way (WK Kellogg, ZimVie, Aaron's, Victoria's Secret) so the hole is
visible rather than invisible. Both channels are named on the Signal axis, not buried here.

No look-ahead is baked in here — the one execution lag lives in ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The SHARED desk cache (studies/_cache), not a study-local one.
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# Study-wide as-of: the last COMPLETE calendar month at build time.
AS_OF = "2026-06-30"

BENCHMARK = "SPY"
CASH = "BIL"

# The data stamp is taken from this date forward — see ``fingerprint``. It sits a year
# before the earliest window in the table (first distribution 2012-05-01).
FP_START = "2011-01-01"

# Horizons, in regular-way sessions after the first regular-way close.
CHILD_HORIZONS = (5, 10, 21)

# --------------------------------------------------------------------------- #
# The hardcoded event table (ASSUMPTION — see the module docstring)
# --------------------------------------------------------------------------- #
# parent      : the ticker that keeps trading (post-spin "stub")
# child       : the ticker distributed to parent holders
# announce    : public announcement of the intended separation (YYYY-MM-DD)
# regular_way : first session of regular-way trading in the child (YYYY-MM-DD)
# ratio       : child shares per parent share (combined-leg weighting only)
SPINOFF_EVENTS: list[dict] = [
    {"parent": "COP",  "child": "PSX",  "announce": "2011-07-14", "regular_way": "2012-05-01", "ratio": 0.5,    "name": "Phillips 66 / ConocoPhillips"},
    {"parent": "ABT",  "child": "ABBV", "announce": "2011-10-19", "regular_way": "2013-01-02", "ratio": 1.0,    "name": "AbbVie / Abbott"},
    {"parent": "EBAY", "child": "PYPL", "announce": "2014-09-30", "regular_way": "2015-07-20", "ratio": 1.0,    "name": "PayPal / eBay"},
    {"parent": "HPQ",  "child": "HPE",  "announce": "2014-10-06", "regular_way": "2015-11-02", "ratio": 1.0,    "name": "HP Enterprise / HP Inc"},
    {"parent": "JCI",  "child": "ADNT", "announce": "2015-07-24", "regular_way": "2016-11-01", "ratio": 0.1,    "name": "Adient / Johnson Controls"},
    {"parent": "MET",  "child": "BHF",  "announce": "2016-01-12", "regular_way": "2017-08-07", "ratio": 0.0909, "name": "Brighthouse / MetLife"},
    {"parent": "MRK",  "child": "OGN",  "announce": "2020-02-05", "regular_way": "2021-06-03", "ratio": 0.1,    "name": "Organon / Merck"},
    {"parent": "IP",   "child": "SLVM", "announce": "2020-08-03", "regular_way": "2021-10-04", "ratio": 0.0909, "name": "Sylvamo / International Paper"},
    {"parent": "IBM",  "child": "KD",   "announce": "2020-10-08", "regular_way": "2021-11-04", "ratio": 0.2,    "name": "Kyndryl / IBM"},
    {"parent": "XPO",  "child": "GXO",  "announce": "2020-12-02", "regular_way": "2021-08-02", "ratio": 1.0,    "name": "GXO Logistics / XPO"},
    {"parent": "EXC",  "child": "CEG",  "announce": "2021-02-24", "regular_way": "2022-02-02", "ratio": 0.3333, "name": "Constellation Energy / Exelon"},
    {"parent": "BDX",  "child": "EMBC", "announce": "2021-05-06", "regular_way": "2022-04-01", "ratio": 0.2,    "name": "Embecta / Becton Dickinson"},
    {"parent": "GE",   "child": "GEHC", "announce": "2021-11-09", "regular_way": "2023-01-04", "ratio": 0.3333, "name": "GE HealthCare / GE"},
    {"parent": "GE",   "child": "GEV",  "announce": "2021-11-09", "regular_way": "2024-04-02", "ratio": 0.25,   "name": "GE Vernova / GE"},
    {"parent": "CXT",  "child": "CR",   "announce": "2022-03-30", "regular_way": "2023-04-04", "ratio": 1.0,    "name": "Crane Company / Crane NXT"},
    {"parent": "XPO",  "child": "RXO",  "announce": "2022-03-08", "regular_way": "2022-11-01", "ratio": 1.0,    "name": "RXO / XPO"},
    {"parent": "FBIN", "child": "MBC",  "announce": "2022-05-04", "regular_way": "2022-12-15", "ratio": 1.0,    "name": "MasterBrand / Fortune Brands"},
    {"parent": "ARMK", "child": "VSTS", "announce": "2022-05-10", "regular_way": "2023-10-02", "ratio": 0.5,    "name": "Vestis / Aramark"},
    {"parent": "MMM",  "child": "SOLV", "announce": "2022-07-26", "regular_way": "2024-04-01", "ratio": 0.25,   "name": "Solventum / 3M"},
    {"parent": "LH",   "child": "FTRE", "announce": "2022-08-09", "regular_way": "2023-07-03", "ratio": 1.0,    "name": "Fortrea / Labcorp"},
    {"parent": "VYX",  "child": "NATL", "announce": "2022-09-15", "regular_way": "2023-10-17", "ratio": 0.5,    "name": "NCR Atleos / NCR Voyix"},
    {"parent": "DHR",  "child": "VLTO", "announce": "2022-09-14", "regular_way": "2023-10-02", "ratio": 0.3333, "name": "Veralto / Danaher"},
    {"parent": "BWA",  "child": "PHIN", "announce": "2022-12-06", "regular_way": "2023-07-03", "ratio": 0.2,    "name": "Phinia / BorgWarner"},
    {"parent": "J",    "child": "AMTM", "announce": "2023-11-20", "regular_way": "2024-09-30", "ratio": 0.2325, "name": "Amentum / Jacobs"},
    {"parent": "WDC",  "child": "SNDK", "announce": "2023-10-30", "regular_way": "2025-02-24", "ratio": 0.3333, "name": "SanDisk / Western Digital"},
    {"parent": "LEN",  "child": "MRP",  "announce": "2024-09-19", "regular_way": "2025-02-10", "ratio": 0.5,    "name": "Millrose / Lennar"},
]

# Events that belong in the table on the merits but whose CHILD no longer has a Yahoo
# price series, because the child was acquired and delisted. They are recorded here, not
# silently dropped, because their absence is a survivorship channel in its own right —
# and note it removes *outcomes of both signs*: Kellanova's WK Kellogg and ZimVie were
# taken out at a premium, Aaron's and Victoria's Secret were not.
DELISTED_EVENTS: list[dict] = [
    {"parent": "PRG",  "child": "AAN",  "announce": "2020-06-04", "regular_way": "2020-12-01", "ratio": 0.5,    "name": "Aaron's / PROG Holdings"},
    {"parent": "BBWI", "child": "VSCO", "announce": "2021-05-11", "regular_way": "2021-08-03", "ratio": 0.3333, "name": "Victoria's Secret / L Brands"},
    {"parent": "ZBH",  "child": "ZIMV", "announce": "2021-02-08", "regular_way": "2022-03-02", "ratio": 0.1,    "name": "ZimVie / Zimmer Biomet"},
    {"parent": "K",    "child": "KLG",  "announce": "2022-06-21", "regular_way": "2023-10-02", "ratio": 0.25,   "name": "WK Kellogg / Kellanova"},
]


def event_tickers(events: list[dict] | None = None) -> tuple[str, ...]:
    """Every ticker the study needs: parents, children, benchmark and cash."""
    ev = SPINOFF_EVENTS if events is None else events
    names = sorted({e["parent"] for e in ev} | {e["child"] for e in ev})
    return tuple(names + [BENCHMARK, CASH])


TICKERS = event_tickers()


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return, shared cache, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2010-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
    skip_cached: bool = True,
) -> list[str]:
    """Download daily total-return closes and cache each as parquet in the shared cache.

    Network-only; run once to populate the cache. ``auto_adjust=True`` so ``close`` is
    split- and dividend-adjusted total return. Returns the list of tickers that could
    **not** be retrieved (empty on a clean run) rather than raising, so one dead ticker
    does not abort a 50-symbol pull.

    Note on the parent series: Yahoo's adjustment factors handle splits and cash
    dividends, but treat a *spin-off distribution* inconsistently. This study never
    computes a parent return that **crosses** the distribution date, precisely to keep
    that ambiguity out of the numbers.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    os.makedirs(cache_dir, exist_ok=True)
    missing: list[str] = []
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if skip_cached and os.path.exists(path):
            continue
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(
                    tk, start=start, end=end, interval="1d",
                    auto_adjust=True, progress=False,
                )
                if raw is not None and len(raw) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        if raw is None or len(raw) == 0:
            missing.append(tk)
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        df = raw[["close"]].copy()
        df.index = pd.to_datetime(df.index)
        try:
            df.index = df.index.tz_localize(None)
        except (TypeError, AttributeError):
            pass
        df.index.name = "date"
        df = df.dropna(subset=["close"])
        df.to_parquet(path)
    return missing


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every ticker's parquet is present in the shared cache (offline-testable)."""
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_prices(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
) -> pd.DataFrame:
    """Read cached daily total-return closes OFFLINE into one aligned close frame.

    One column per ticker, sliced to ``asof`` so the sample never creeps. Raises
    ``FileNotFoundError`` if any ticker is missing — the offline core and the test-suite
    never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call when_issued.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        try:
            s.index = s.index.tz_localize(None)
        except (TypeError, AttributeError):
            pass
        cols[tk] = s[~s.index.duplicated(keep="last")]
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df = df[df.index <= pd.Timestamp(asof)]
    return df


def fingerprint(prices: pd.DataFrame, since: str = FP_START) -> str:
    """Short content fingerprint of a price frame over the study window, for the data stamp.

    Taken over ``since`` .. as-of rather than over the whole frame **on purpose**: the desk
    cache is shared, and another study may re-fetch a common ticker (SPY, BIL) with a
    longer or shorter pre-history. That pre-history changes nothing here — no window in
    this study starts before 2012 — but it would move a whole-frame hash and make the
    stamp unreproducible for reasons that have nothing to do with this study's data.
    """
    px = prices[prices.index >= pd.Timestamp(since)] if since else prices
    arr = np.ascontiguousarray(px.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted when-issued window)
# --------------------------------------------------------------------------- #
def synthetic_daily(
    n_events: int = 25,
    signal_strength: float = 1.0,
    parent_runup_ann: float = 0.12,     # planted parent alpha vs benchmark, annualised
    child_alpha_bps: float = 25.0,      # planted child alpha per day, first 21 sessions
    child_window: int = 21,
    bench_drift_ann: float = 0.08,
    bench_vol_ann: float = 0.16,
    idio_vol_ann: float = 0.28,
    beta: float = 1.0,
    cash_rate_ann: float = 0.02,
    start: str = "2011-01-03",
    n_days: int = 3600,
    seed: int = 930,
) -> tuple[pd.DataFrame, list[dict], dict]:
    """A synthetic world of ``n_events`` spin-offs, in the real loader's exact schema.

    Each event gets a parent series (a benchmark-beta series with a planted *excess*
    drift between its announcement and its distribution) and a child series (a
    benchmark-beta series with a planted excess drift over its first ``child_window``
    regular-way sessions, and none thereafter). The frame also carries ``SPY`` (the
    benchmark) and ``BIL`` (the cash accrual leg).

    ``signal_strength`` scales **both** planted effects linearly: at ``1.0`` the run-up
    and the drift are fully present, at ``0.0`` the world is a clean null (parent and
    child are pure beta plus idiosyncratic noise) and the estimator must stay quiet.

    Returns ``(prices, events, truth)`` where ``events`` uses the same keys as
    ``SPINOFF_EVENTS`` so ``strategy.event_panel`` runs unchanged on it.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=int(n_days), name="date")
    n = len(dates)

    b_mu = bench_drift_ann / TRADING_DAYS_PER_YEAR
    b_sd = bench_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    bench_r = rng.normal(b_mu, b_sd, n)
    bench = 100.0 * np.cumprod(1.0 + bench_r)

    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cash = np.cumprod(np.full(n, cash_daily))

    idio_sd = idio_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    runup_daily = signal_strength * parent_runup_ann / TRADING_DAYS_PER_YEAR
    child_daily = signal_strength * child_alpha_bps * 1e-4

    # Space the events out so their windows do not all collide.
    first_rw = 400
    last_rw = n - child_window - 40
    step = max(30, (last_rw - first_rw) // max(1, n_events))
    cols = {BENCHMARK: bench, CASH: cash}
    events: list[dict] = []
    for k in range(n_events):
        rw_i = first_rw + k * step
        if rw_i >= last_rw:
            break
        # Announcement 6-18 months before the distribution.
        lead = int(rng.integers(126, 378))
        ann_i = max(1, rw_i - lead)

        p_alpha = np.zeros(n)
        p_alpha[ann_i + 1: rw_i] = runup_daily
        parent_r = beta * bench_r + rng.normal(0.0, idio_sd, n) + p_alpha
        parent = 100.0 * np.cumprod(1.0 + parent_r)

        c_alpha = np.zeros(n)
        c_alpha[rw_i + 1: rw_i + 1 + child_window] = child_daily
        child_r = beta * bench_r + rng.normal(0.0, idio_sd, n) + c_alpha
        child = 100.0 * np.cumprod(1.0 + child_r)
        # The child only exists from its regular-way date onward.
        child[:rw_i] = np.nan

        pt, ct = f"SYNP{k:02d}", f"SYNC{k:02d}"
        cols[pt] = parent
        cols[ct] = child
        events.append({
            "parent": pt, "child": ct,
            "announce": str(dates[ann_i].date()),
            "regular_way": str(dates[rw_i].date()),
            "ratio": 0.5,
            "name": f"{ct} / {pt}",
        })

    prices = pd.DataFrame(cols, index=pd.DatetimeIndex(dates, name="date"))
    truth = {
        "n_events": len(events),
        "signal_strength": signal_strength,
        "parent_runup_ann": parent_runup_ann,
        "child_alpha_bps": child_alpha_bps,
        "child_window": child_window,
        "planted_child_alpha_21d": float(child_daily * child_window),
        "planted_parent_runup_daily": float(runup_daily),
        "n_days": int(n),
        "seed": seed,
    }
    return prices, events, truth


def synthetic_panel(
    n_events: int = 25,
    signal_strength: float = 1.0,
    seed: int = 930,
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Convenience wrapper: the per-event alpha panel from a synthetic world.

    Thin shim over ``synthetic_daily`` + ``strategy.event_panel`` so a caller who only
    wants the event table does not have to wire the two together. Imported lazily to
    keep ``data`` free of a strategy dependency at module scope.
    """
    from . import strategy as st  # local import: avoids a circular import at load time

    prices, events, truth = synthetic_daily(
        n_events=n_events, signal_strength=signal_strength, seed=seed, **kwargs
    )
    panel = st.event_panel(prices, events, bench=BENCHMARK, cost_bps=0.0)
    return panel, truth
