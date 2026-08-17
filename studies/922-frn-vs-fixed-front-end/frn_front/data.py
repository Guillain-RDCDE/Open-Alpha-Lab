"""Data layer for Study 922 — Floating-Rate Front End.

Two tapes, one shape (a date-indexed daily frame):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the two Treasury floating-rate-note ETFs
  (**USFR**, **TFLO**), the T-bill fund (**BIL**), and the 1-3 year fixed-coupon
  Treasury fund (**SHY**) — plus **^IRX**, the 13-week T-bill discount rate, which is
  a *price-only yield index*, not an investable total return. ``fetch`` touches the
  network and caches parquet under the shared ``studies/_cache`` (retry up to 4x);
  ``load_prices`` reads that cache **offline** and never imports yfinance. The whole
  test-suite runs with NO cache present (synthetic-only), so CI is green on a fresh
  checkout where ``_cache/`` is git-ignored and absent.

- ``synthetic_panel`` / ``synthetic_daily`` — a *deterministic, offline* generator. A
  short-rate path with a hike / plateau / cut cycle drives three funds built from
  first principles: a zero-duration floater, a lagging bill ladder, and a fixed note
  whose **effective duration is the ``signal_strength`` knob**. The rate cycle is
  always present (so the regime classifier always has rising and falling days to
  label), but at ``signal_strength=0`` the "fixed" leg carries no duration at all —
  there is nothing for the rate direction to do, and the estimator must return ~0 (the
  null). At ``signal_strength=1`` a full 1.85-year duration is planted and the
  floater-minus-fixed gap must swing with the direction of rates. Seed is fixed →
  tests are deterministic.

**Non-tape inputs (declared as PROXY / ASSUMPTION, and swept in `strategy.py`):**

1. ``^IRX`` is quoted as an annualised *discount* rate on an act/360 basis. We convert
   it to a daily cash accrual as ``(irx_{t-1} / 100) / basis`` with ``basis=252`` —
   a bond-equivalent approximation, not the exact bill convention. ``cash_leg`` takes
   the basis as an argument so the whole race can be re-run on 360 and on a
   BIL-as-cash alternative; the conclusions do not move.
2. The **rate-cycle calendar** (``CYCLE_WINDOWS``) is a hardcoded reading of Fed
   history, not something derived from the tape. The mechanical alternative — the sign
   of the 63-day change in ``^IRX`` (``strategy.irx_regime``) — is reported alongside
   it, with the window and threshold swept.
3. Expense ratios (USFR 0.15%, TFLO 0.15%, BIL 0.1356%, SHY 0.15% at build time) are
   quoted as *context only*: they are already inside the total-return closes and are
   never added to the arithmetic.

No look-ahead is baked in here. Exactly one execution lag lives in ``strategy.py``:
every ``^IRX``-derived object (the cash accrual rate and the regime label) is formed
from data through day ``t`` and applied to day ``t+1``.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# USFR/TFLO = Treasury floating-rate-note ETFs; BIL = 1-3 month T-bills;
# SHY = 1-3 year fixed-coupon Treasuries; ^IRX = 13-week bill discount rate (yield index).
TICKERS = ("USFR", "TFLO", "BIL", "SHY", "^IRX")

# Funds carrying an investable total return (^IRX does not).
FUNDS = ("USFR", "TFLO", "BIL", "SHY")

# Study-wide as-of: the last COMPLETE calendar month at build time (drop the partial
# current month so the sample never creeps between reruns).
AS_OF = "2026-06-30"

# Context only — already inside the total-return closes, never added to the arithmetic.
EXPENSE_RATIOS = {"USFR": 0.0015, "TFLO": 0.0015, "BIL": 0.001356, "SHY": 0.0015}


def _col(ticker: str) -> str:
    """Column name for a ticker (``^IRX`` -> ``IRX``)."""
    return ticker.replace("^", "").replace("=", "").replace("/", "")


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"prices_{_col(ticker)}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2010-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` makes the
    fund columns split- and distribution-adjusted **total return** — essential here,
    because these vehicles pay out essentially all of their return as monthly income and
    a price-only comparison would be meaningless. ``^IRX`` is a yield index: its "close"
    is a rate in percent, not a tradable price, and it is labelled as such everywhere.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    out: dict[str, pd.DataFrame] = {}
    os.makedirs(cache_dir, exist_ok=True)
    for tk in tickers:
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
            raise RuntimeError(f"yfinance returned no data for {tk}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        df = raw[["close"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.dropna(subset=["close"])
        df.to_parquet(_cache_path(tk, cache_dir))
        out[tk] = df
    return out


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every ticker's parquet is present in the cache (offline-testable)."""
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_prices(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
) -> pd.DataFrame:
    """Read cached daily closes OFFLINE into one aligned frame, sliced to ``asof``.

    Columns are the ticker names with ``^`` stripped (so ``^IRX`` arrives as ``IRX``).
    Rows are the intersection of all series' trading days, so every comparison is run on
    an identical calendar. Raises ``FileNotFoundError`` if any ticker is missing — the
    offline core and the test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call frn_front.data.fetch() once to populate the cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[_col(tk)] = s
    df = pd.DataFrame(cols).sort_index().dropna()
    df.index.name = "date"
    df = df[df.index <= pd.Timestamp(asof)]
    return df


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted rate cycle)
# --------------------------------------------------------------------------- #
PHASE_FRACTIONS = (0.45, 0.13, 0.09, 0.33)   # flat, hike, plateau, cut


def _rate_path(n_days: int, base: float, amplitude: float) -> np.ndarray:
    """A deterministic hike / plateau / cut short-rate path, in decimal annual terms.

    Four phases in the proportions of the real 2014-2026 cycle (:data:`PHASE_FRACTIONS`):
    a long flat stretch at ``base``, a *fast* linear hike of ``amplitude``, a short
    plateau at the top, and a slower linear cut back to ``base``. The steepness of the
    hike matters — it is what makes duration hurt faster than carry heals — and an
    equal-quarters path would understate it.
    """
    n_flat, n_hike, n_top = (int(round(f * n_days)) for f in PHASE_FRACTIONS[:3])
    n_cut = n_days - n_flat - n_hike - n_top
    return np.concatenate([
        np.full(n_flat, base),
        base + np.linspace(0.0, amplitude, n_hike),
        np.full(n_top, base + amplitude),
        base + amplitude - np.linspace(0.0, amplitude, n_cut),
    ])


def synthetic_panel(
    n_years: int = 12,
    base_rate: float = 0.02,          # short rate at the start / end of the cycle
    amplitude: float = 0.05,          # size of the planted hiking cycle (5 pp)
    discount_margin: float = 0.0012,  # FRN spread over the bill rate (12 bp)
    term_premium: float = 0.0035,     # 1-3y fixed yield pickup over the bill rate
    duration_fixed: float = 1.85,     # effective duration of the fixed 1-3y leg, years
    bill_lag_days: int = 45,          # rolldown lag of a 1-3m bill ladder
    noise_bps: float = 1.5,           # daily idiosyncratic close noise, bps
    signal_strength: float = 1.0,     # 0 = fixed leg has no duration (null), 1 = full
    start: str = "2012-01-02",
    seed: int = 922,
) -> tuple[pd.DataFrame, dict]:
    """A daily front-end panel: a floater, a lagging bill ladder and a fixed 1-3y note.

    Construction, from first principles rather than from a fitted model:

    - ``frn``    accrues ``(rate_t + discount_margin) / 252`` — it resets with the short
      rate, so it carries (essentially) no duration.
    - ``bills``  accrues the *trailing average* of the short rate over ``bill_lag_days``
      — a ladder of already-bought bills reprices only as they mature and roll.
    - ``fixed``  accrues ``(rate_t + term_premium) / 252`` **and** takes a price hit of
      ``-duration_eff * d(yield)`` — the duration that pays in a cutting cycle and
      punishes in a hiking one.
    - ``IRX``    is the short rate in percent, mirroring the real ``^IRX`` column.

    ``signal_strength`` scales the fixed leg's duration: ``duration_eff =
    signal_strength * duration_fixed``. The rate cycle itself is *always* present, so
    the regime classifier always has rising and falling days to label — but at
    ``signal_strength=0`` the "fixed" leg is a floater in disguise and the direction of
    rates cannot flip the ranking, so the regime estimator must return ~0 (the null).
    At 1 the full 1.85-year duration is planted. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: bdate_range with n <= 10000 daily bars stays well inside pandas' ns range.
    dates = pd.bdate_range(start=start, periods=n_days)

    rate = _rate_path(n_days, base_rate, float(amplitude))
    duration_eff = float(signal_strength) * float(duration_fixed)

    bill_rate = pd.Series(rate).rolling(bill_lag_days, min_periods=1).mean().to_numpy()
    fixed_yield = rate + term_premium
    d_yield = np.diff(fixed_yield, prepend=fixed_yield[0])

    noise = noise_bps * 1e-4
    r_frn = (rate + discount_margin) / TRADING_DAYS_PER_YEAR + rng.normal(0, noise, n_days)
    r_bills = bill_rate / TRADING_DAYS_PER_YEAR + rng.normal(0, noise * 0.3, n_days)
    r_fixed = (fixed_yield / TRADING_DAYS_PER_YEAR - duration_eff * d_yield
               + rng.normal(0, noise * 2.0, n_days))

    prices = pd.DataFrame(
        {
            "frn": 100.0 * np.cumprod(1.0 + r_frn),
            "bills": 100.0 * np.cumprod(1.0 + r_bills),
            "fixed": 100.0 * np.cumprod(1.0 + r_fixed),
            "IRX": rate * 100.0,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "signal_strength": float(signal_strength),
        "base_rate": base_rate,
        "amplitude": amplitude,
        "discount_margin": discount_margin,
        "term_premium": term_premium,
        "duration_fixed": duration_fixed,
        "duration_eff": duration_eff,
        "bill_lag_days": bill_lag_days,
        "n_years": n_years,
        "n_days": n_days,
        "seed": seed,
        "rate_min": float(rate.min()),
        "rate_max": float(rate.max()),
    }
    return prices, truth


def synthetic_daily(**kwargs) -> tuple[pd.DataFrame, dict]:
    """Two-column convenience view of :func:`synthetic_panel` (``asset`` = FRN, ``cash``).

    ``asset`` is the floater and ``cash`` the bill ladder, for the handful of tests and
    demos that only need one pair rather than the full front-end panel.
    """
    prices, truth = synthetic_panel(**kwargs)
    out = prices[["frn", "bills"]].rename(columns={"frn": "asset", "bills": "cash"})
    return out, truth
