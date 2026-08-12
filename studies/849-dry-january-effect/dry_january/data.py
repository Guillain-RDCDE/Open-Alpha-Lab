"""Data layer for Study 849 — Dry January / Veganuary.

The claim under test: the two January cultural waves — **Dry January** (abstain from
alcohol for the month) and **Veganuary** (eat plant-based for the month) — shift consumer
demand hard enough, and predictably enough, to move the relevant stocks in January (and
leave a February "hangover" as the abstainers fall off the wagon). The folklore's two
halves:

* **Alcohol names** (``BUD`` AB InBev, ``STZ`` Constellation, ``TAP`` Molson Coors,
  ``DEO`` Diageo, ``SAM`` Boston Beer) should *under*-perform in January as Dry-January
  demand evaporates — and maybe *bounce* in February.
* **Plant-based** (``BYND`` Beyond Meat) should *out*-perform in January on the Veganuary
  demand pulse.
* **Consumer staples** (``XLP``) is the boring sector control — a defensive basket with no
  particular January story, so its January abnormal return should be a flat ≈ 0.

Because the calendar is **fixed and known decades ahead** — January is always January — a
"is the January abnormal return different from the other eleven months?" test is
zero-look-ahead by construction. If a real demand shift leaked into prices it would show as
a January seasonal in the *abnormal* return (name − ``SPY``); a semi-strong-efficient market
prices a calendar everyone can read, so the desk's prior is nothing.

Two ingredients, one monthly schema:

* **Real tape — daily adjusted closes (yfinance).** ``fetch()`` pulls the eight tickers
  (``auto_adjust=True``, total-return) into this study's OWN ``_cache/`` as one parquet
  (retry up to 4×). ``have_real()`` / ``load_panel()`` read the cached parquet OFFLINE (no
  yfinance import). ``AS_OF = "2026-06-30"`` — the last complete calendar month; the partial
  current month is dropped.

* **Synthetic world — the positive control.** ``synthetic_panel`` builds a deterministic,
  seeded daily (name, ``SPY``) tape in which each **January** carries an EXTRA ``edge`` of
  cumulative abnormal return, spread across the month's business days. ``edge = 0`` is the
  null (no January seasonal); ``edge > 0`` plants exactly a January abnormal-return effect
  that the monthly-seasonality detector must recover — and must NOT manufacture from the
  null.

The offline path is pure numpy + pandas + stdlib. Business-day index kept far below the
pandas ns-timestamp horizon (rule: daily ``bdate_range`` with n well under 10k).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
CACHE_PATH = os.path.join(CACHE_DIR, "dry_january_daily.parquet")

# The three tested groups + the market benchmark.
ALCOHOL = ["BUD", "STZ", "TAP", "DEO", "SAM"]   # Dry-January demand names
PLANT = ["BYND"]                                # Veganuary demand name
STAPLES = ["XLP"]                               # boring sector control
BENCH = "SPY"                                   # market benchmark for the abnormal return
ALL_TICKERS = ALCOHOL + PLANT + STAPLES + [BENCH]

START = "1999-01-01"        # XLP inception is Dec-1998; SPY/alcohol names go back further
AS_OF = "2026-06-30"        # last complete calendar month at publication

__all__ = [
    "ALCOHOL", "PLANT", "STAPLES", "BENCH", "ALL_TICKERS",
    "START", "AS_OF", "CACHE_DIR", "CACHE_PATH",
    "fetch", "have_real", "load_panel", "load_closes", "synthetic_panel", "fingerprint",
]


# --------------------------------------------------------------------------- #
# Real tape — yfinance daily adjusted closes, cache-first
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str = "2026-07-01", retries: int = 4) -> pd.DataFrame:
    """Download the eight daily adjusted (total-return) closes; cache one parquet.

    Network; runs once to build the cache. Retries up to ``retries`` times with a short
    linear backoff on any transient/rate-limit failure. Writes a single wide parquet with a
    tz-naive daily index and one column per ticker (adjusted close).
    """
    import yfinance as yf  # lazy — never imported on the offline path

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                ALL_TICKERS, start=start, end=end, interval="1d",
                auto_adjust=True, progress=False, threads=False,
            )
            if raw is None or raw.empty:
                raise RuntimeError("yfinance returned no bars")
            closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            closes = closes.copy()
            closes.columns = [str(c) for c in closes.columns]
            closes.index.name = "date"
            if closes.index.tz is not None:
                closes.index = closes.index.tz_localize(None)
            # Keep any ticker with at least SOME history; require SPY as the benchmark.
            if BENCH not in closes.columns or closes[BENCH].dropna().empty:
                raise RuntimeError("benchmark SPY missing from the pull")
            if len(closes) < 500:
                raise RuntimeError(f"suspiciously short tape ({len(closes)} rows)")
            os.makedirs(CACHE_DIR, exist_ok=True)
            closes.to_parquet(CACHE_PATH)
            return closes
        except Exception as exc:  # noqa: BLE001 — retry on any transient failure
            last_err = exc
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"yfinance fetch failed after {retries} attempts: {last_err}")


def have_real() -> bool:
    return os.path.exists(CACHE_PATH)


def load_panel(asof: str = AS_OF) -> pd.DataFrame:
    """Cached wide daily close frame, sliced to ``<= asof``. OFFLINE (reads parquet)."""
    if not have_real():
        raise FileNotFoundError(
            f"No cached tape at {CACHE_PATH}. Call dry_january.data.fetch() once."
        )
    df = pd.read_parquet(CACHE_PATH)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[df.index <= pd.Timestamp(asof)].sort_index()
    return df


def load_closes(asof: str = AS_OF) -> dict[str, pd.Series]:
    """Cached ``{ticker: adjusted-close Series}`` (offline), each dropna'd and ``<= asof``."""
    df = load_panel(asof)
    out: dict[str, pd.Series] = {}
    for t in ALL_TICKERS:
        if t in df.columns:
            s = df[t].dropna()
            if not s.empty:
                out[t] = s
    return out


# --------------------------------------------------------------------------- #
# Synthetic world — a planted January abnormal-return seasonal (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 849,
    n_years: int = 25,
    start: str = "2000-01-03",
    rho: float = 0.55,
    name_vol: float = 0.013,
    bench_vol: float = 0.010,
) -> dict[str, pd.Series]:
    """Deterministic seeded daily ``{name, SPY}`` tape with a TUNABLE January seasonal.

    Both series share a common market factor (correlation ≈ ``rho``, like a consumer name
    vs SPY) plus idiosyncratic noise. In every **January**, the ``NAME`` series earns an
    EXTRA ``edge`` of cumulative return, spread evenly across that January's business days,
    so the monthly abnormal return (``NAME − SPY``) has a planted January mean ≈ ``edge``
    while the other months are null. ``edge = 0`` is the null world (no seasonal); the
    monthly-seasonality detector must stay quiet on it and recover ``edge > 0`` monotonically.

    Business-day integer span kept far below the pandas ns-timestamp horizon
    (``n_years × 252`` days from 2000 ≈ year 2025). Returns real daily-close ``Series`` so
    the same monthly pipeline that reads the real tape can consume it unchanged.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * 252)
    idx = pd.bdate_range(start=start, periods=n_days)

    common = rng.normal(0.0, 1.0, n_days)
    idio_n = rng.normal(0.0, 1.0, n_days)
    idio_b = rng.normal(0.0, 1.0, n_days)
    name = name_vol * (rho * common + np.sqrt(1.0 - rho ** 2) * idio_n)
    bench = bench_vol * (rho * common + np.sqrt(1.0 - rho ** 2) * idio_b)

    # Spread the planted January edge across each January's business days.
    is_jan = idx.month == 1
    jan_counts = pd.Series(is_jan.astype(float), index=idx).groupby(idx.year).transform("sum")
    per_day = np.where(is_jan, edge / np.where(jan_counts.to_numpy() > 0, jan_counts.to_numpy(), 1.0), 0.0)
    name = name + per_day

    name_close = 100.0 * np.cumprod(1.0 + name)
    bench_close = 100.0 * np.cumprod(1.0 + bench)
    return {
        "NAME": pd.Series(name_close, index=idx, name="NAME"),
        BENCH: pd.Series(bench_close, index=idx, name=BENCH),
    }


# --------------------------------------------------------------------------- #
# Content fingerprint (for the as-of stamp)
# --------------------------------------------------------------------------- #
def fingerprint(panel: dict[str, pd.Series] | pd.DataFrame) -> str:
    """Short content fingerprint of the concatenated close matrix (the as-of stamp)."""
    if isinstance(panel, pd.DataFrame):
        arr = np.ascontiguousarray(panel.fillna(0.0).to_numpy(dtype=float))
    else:
        cols = [panel[t].to_numpy(dtype=float) for t in sorted(panel)]
        flat = np.concatenate([np.nan_to_num(c) for c in cols]) if cols else np.zeros(1)
        arr = np.ascontiguousarray(flat)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
