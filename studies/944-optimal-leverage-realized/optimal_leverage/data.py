"""Data layer for Study 944 — How Much Leverage (the realised growth-optimal multiple).

Two tapes, one shape (a date-indexed daily frame of total-return closes plus a short
rate):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the risky asset (**SPY**) and the cash ETF
  (**BIL**), plus **^IRX**, the 13-week Treasury-bill discount rate in percent (a *level*,
  not a price). ``fetch`` touches the network and writes parquet into the **shared**
  ``studies/_cache`` (retry up to 4x); ``load_prices`` reads that cache **offline** and
  never imports yfinance. The whole test-suite runs with NO cache present (synthetic
  only), so CI is green on a fresh checkout.

- ``synthetic_daily`` — a *deterministic, offline* generator with a **known** Kelly
  multiple. Daily log returns are drawn i.i.d. Student-t (scaled to a target vol) around
  an excess drift; the planted growth-optimal leverage is ``mu_excess / vol**2``. The
  ``signal_strength`` knob scales the excess drift: at ``signal_strength=0`` the asset
  earns exactly cash, so the growth-optimal multiple collapses to **zero leverage** (the
  null — the estimator must not manufacture a reason to lever); at ``signal_strength=1``
  the planted optimum sits at ``kelly_true`` (2.0 by default), which the sweep must
  recover. Seed is fixed, so tests are deterministic.

The question: a constant-leverage, daily-reset position in the equity index has a
concave growth curve in the multiple ``L`` — geometric growth rises with ``L`` while the
variance drag ``L**2 * sigma**2 / 2`` eventually swamps it. Theory (Kelly / Merton) puts
the peak at ``L* = mu_excess / sigma**2``. This study maps the *realised* peak on the
real, financed, costed tape and asks the only question that matters for a leverage user:
**is that peak a stable number you could have known in advance?**

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (an estimate
formed on data through day ``t`` sizes the position at day ``t+1``).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The SHARED desk cache (studies/_cache), not a per-study one.
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# SPY = the risky asset; BIL = the tradable cash ETF (financing cross-check, 2007+);
# ^IRX = the 13-week bill discount rate, the financing base over the whole window.
TICKERS = ("SPY", "BIL", "^IRX")

# Study-wide as-of: the last COMPLETE calendar month at build time. The partial current
# month is dropped so the sample never creeps between reruns.
AS_OF = "2026-06-30"

# --------------------------------------------------------------------------- #
# Non-tape ASSUMPTIONS (PROXIES). Both are swept in strategy.py; neither is a
# measured quantity, and the README labels them as such.
# --------------------------------------------------------------------------- #
# Financing spread over the bill rate paid on the borrowed sleeve, in bps/yr. 50 bps is
# a mid estimate for institutional-quality leverage (E-mini roll cheapness / a box
# spread); retail margin runs far wider, which is why the sweep goes to 200.
DEFAULT_SPREAD_BPS = 50.0
# One-way cost in bps of the *notional traded* by the daily reset. 1 bp is generous for
# an S&P future; the sweep runs 0 -> 5.
DEFAULT_COST_BPS = 1.0


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2000-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily closes for ``tickers`` and cache each as parquet in the shared cache.

    ``auto_adjust=True`` so SPY's and BIL's ``close`` column is the split- and
    dividend-adjusted **total return** — mandatory here, because the whole study is a
    compounding exercise and a price-only SPY would understate the growth curve by the
    dividend yield at every leverage. ^IRX carries no dividend; its ``close`` is a *rate
    level in percent*, and is treated as such everywhere downstream.
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
    """True iff every ticker's parquet is present in the shared cache (offline-testable)."""
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_prices(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
) -> pd.DataFrame:
    """Read the cached daily closes OFFLINE into one aligned frame.

    Columns are named without the ``^`` (``SPY``, ``BIL``, ``IRX``). Sliced to ``asof``
    so the sample never creeps. Raises ``FileNotFoundError`` if any ticker is missing —
    the offline core and the test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached data for {tk} at {path}. "
                f"Call optimal_leverage.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk.replace("^", "")] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df = df[df.index <= pd.Timestamp(asof)]
    return df


def cash_rate_daily(irx_pct: pd.Series) -> pd.Series:
    """Convert the ^IRX quote (13-week bill discount rate, in percent) to a daily rate.

    Money-market convention: an annualised rate on an **act/360** basis, accrued over the
    *calendar* days between consecutive observations — so a Monday bar earns three days
    of carry, exactly as the weekend equity return spans three calendar days. The level
    is carried forward within the gap (the previous close's rate is the rate you actually
    financed at). The first bar is dropped by the caller with the first return.

    ^IRX is a *discount* quote rather than a bond-equivalent yield; over the sample the
    two differ by a few bps, well inside the 0-200 bps spread sweep that this study runs
    anyway. ``strategy.financing_crosscheck`` compares this construction against BIL's
    realised total return, which is the tradable cash leg.
    """
    r = pd.Series(irx_pct).astype(float).sort_index()
    days = pd.Series(r.index, index=r.index).diff().dt.days.astype(float)
    return (r.shift(1) / 100.0) * days / 360.0


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a PLANTED Kelly multiple)
# --------------------------------------------------------------------------- #
def synthetic_daily(
    n_years: int = 40,
    kelly_true: float = 2.0,        # the planted growth-optimal leverage
    vol_ann: float = 0.16,          # annualised vol of the risky asset
    cash_rate_ann: float = 0.03,    # flat cash/financing base
    df_t: float = 5.0,              # Student-t degrees of freedom (fat tails)
    signal_strength: float = 1.0,   # 0 = zero excess drift (null), 1 = full planted drift
    start: str = "1986-01-02",
    seed: int = 944,
) -> tuple[pd.DataFrame, dict]:
    """An i.i.d. daily tape whose growth-optimal leverage is known by construction.

    The excess arithmetic drift is set to ``signal_strength * kelly_true * vol_ann**2``,
    so the continuous-time optimum is ``L* = mu_excess / vol**2 = signal_strength *
    kelly_true``:

    - ``signal_strength = 1`` → the growth curve peaks at ``kelly_true`` (2.0 by default);
      the realised sweep must land there (fat tails pull it slightly *down*, never up).
    - ``signal_strength = 0`` → the asset earns exactly cash; every unit of leverage is
      pure variance drag, so the optimum is **zero** and the sweep must sit on its floor.

    Returns ``(prices, truth)`` where ``prices`` carries ``asset`` (a total-return index)
    and ``cash`` (a cash accrual index), and ``truth`` records the planted parameters.
    Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # Business-day index built with numpy (pandas' bdate_range is ~600x slower here) and
    # OOB-safe by construction: n_days is capped so the last date stays far inside the
    # ns Timestamp horizon on pandas 2.x / Python 3.10.
    if not 0 < n_days <= 80_000:
        raise ValueError("n_years must give 1..80,000 business days (ns-Timestamp safe)")
    dates = pd.DatetimeIndex(
        np.busday_offset(np.datetime64(start, "D"), np.arange(n_days), roll="forward")
    )

    mu_excess_ann = signal_strength * kelly_true * vol_ann ** 2
    sd_d = vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    mu_d = mu_excess_ann / TRADING_DAYS_PER_YEAR + cash_rate_ann / TRADING_DAYS_PER_YEAR

    # Student-t scaled to unit variance, so vol_ann is honoured despite the fat tails.
    raw = rng.standard_t(df_t, n_days)
    raw = raw / np.sqrt(df_t / (df_t - 2.0))
    simple = mu_d + sd_d * raw

    asset = 100.0 * np.cumprod(1.0 + simple)
    cash_d = cash_rate_ann / TRADING_DAYS_PER_YEAR
    cash_idx = 100.0 * np.cumprod(np.full(n_days, 1.0 + cash_d))

    prices = pd.DataFrame(
        {"asset": asset, "cash": cash_idx},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "kelly_true": float(signal_strength * kelly_true),
        "kelly_planted": float(kelly_true),
        "signal_strength": float(signal_strength),
        "vol_ann": float(vol_ann),
        "mu_excess_ann": float(mu_excess_ann),
        "cash_rate_ann": float(cash_rate_ann),
        "df_t": float(df_t),
        "n_years": int(n_years),
        "n_days": int(n_days),
        "seed": int(seed),
    }
    return prices, truth
