"""Data layer for Study 941 — Short Both Legs (short TQQQ *and* SQQQ together).

Two tapes, one shape (a date-indexed daily total-return close frame):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the two daily-reset leveraged Nasdaq-100
  funds (TQQQ, the 3x long; SQQQ, the −3x inverse), the unlevered index ETF (QQQ,
  the beta benchmark) and a cash proxy (BIL, the 1-3 month T-bill ETF). ``fetch``
  touches the network and writes parquet into the **shared** ``studies/_cache``
  (retry up to 4x); ``load_prices`` reads that cache **offline** and never imports
  yfinance. The whole test-suite runs with NO cache present (synthetic only), so CI
  is green on a fresh checkout.

- ``synthetic_daily`` / ``synthetic_panel`` — a *deterministic, offline* generator. It
  simulates an underlying index and then **constructs** the two daily-reset levered
  funds from it exactly as a real one is built (L times the daily underlying return,
  the cash carry on the financed notional, less an all-in fee load, plus demeaned
  tracking noise), alongside a cash accrual leg. Volatility is **high in both worlds**,
  so the textbook compounding decay is always present; what ``signal_strength`` scales
  is the funds' **fee load**. At ``signal_strength=1`` an equal-dollar short of both
  legs must recover the average fee load (the planted effect); at ``signal_strength=0``
  the funds are free and the harvest must vanish **even though the decay is unchanged**
  (the null). Seed is fixed → tests are deterministic.

The claim under test: because a daily-reset levered fund loses ``~0.5 * L * (L-1) *
sigma^2`` per year to rebalancing, shorting the 3x long **and** the −3x inverse in
equal dollars nets out the index exposure and leaves the decay of both — a supposedly
market-neutral harvest. Three things decide whether that is true: whether the decay is
collectable at all by a *constant-dollar* short, **borrow cost** (these funds are not
free to borrow), and the **short-gamma tail** the rebalance schedule creates. The first
is settled by construction in the synthetic control; the other two are swept.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the
rebalance decided at the close of day ``t`` is executed for day ``t+1``).
"""

from __future__ import annotations

import functools
import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The SHARED desk cache (studies/_cache), not a study-local one.
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# TQQQ = 3x long Nasdaq-100, SQQQ = -3x inverse Nasdaq-100 (both ProShares, daily reset),
# QQQ = the unlevered index ETF (beta benchmark), BIL = 1-3M T-bill (cash proxy).
TICKERS = ("TQQQ", "SQQQ", "QQQ", "BIL")

# Study-wide as-of: the last COMPLETE calendar month at build time (the partial current
# month is dropped so the sample never creeps between reruns).
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2005-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` makes the
    ``close`` column split- and distribution-adjusted total return. That matters here:
    TQQQ and SQQQ both pay distributions, and the cash leg's own yield is the whole
    point of the excess-of-cash race (a short book sits on a large cash balance).
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
    """Read cached daily total-return closes OFFLINE into one aligned close frame.

    Returns a frame indexed by date with one column per ticker (the adjusted close),
    sliced to ``asof`` so the sample never creeps. Raises ``FileNotFoundError`` if any
    ticker is missing — the offline core and the test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call short_pair.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df = df[df.index <= pd.Timestamp(asof)]
    return df


@functools.lru_cache(maxsize=8)
def _business_index(start: str, n_days: int) -> pd.DatetimeIndex:
    """Cached business-day index (building one is surprisingly costly, and the synthetic
    panels ask for the same one many times over)."""
    return pd.bdate_range(start=start, periods=n_days)


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted decay to harvest)
# --------------------------------------------------------------------------- #
def build_levered(
    r_under: np.ndarray,
    leverage: float,
    r_cash_daily: np.ndarray | float,
    drag_ann: float,
    tracking: np.ndarray | None = None,
) -> np.ndarray:
    """Construct a daily-reset levered fund's NAV path from underlying daily returns.

    This is the actual mechanic of TQQQ/SQQQ, not an approximation of it: each day the
    fund earns ``leverage * r_under`` on its NAV and carries ``(1 - leverage) * r_cash``
    (the long fund finances ``L-1`` times NAV; the inverse fund holds ``1-L`` times NAV in
    cash), less an all-in ``drag_ann`` (expense ratio plus the financing spread on the
    borrowed / lent notional), plus optional daily tracking noise. The daily reset is what
    creates the volatility decay in the *compounded* path.
    """
    carry = (1.0 - leverage) * np.asarray(r_cash_daily)
    r_fund = leverage * r_under + carry - drag_ann / TRADING_DAYS_PER_YEAR
    if tracking is not None:
        r_fund = r_fund + tracking
    # A daily-reset fund cannot go below zero: floor the daily return at -99%.
    r_fund = np.maximum(r_fund, -0.99)
    return 100.0 * np.cumprod(1.0 + r_fund)


def synthetic_daily(
    n_years: int = 16,
    drift_ann: float = 0.10,           # annualised underlying drift
    vol_ann: float = 0.22,             # annualised underlying vol (HIGH in both worlds)
    vol_of_vol: float = 0.35,          # lognormal vol clustering, so the tail is not Gaussian
    signal_strength: float = 1.0,      # scales the funds' fee load: 0 = free funds (null)
    expense_ann: float = 0.0095,       # each fund's expense ratio at signal_strength = 1
    financing_spread_ann: float = 0.0050,   # spread over cash on the borrowed / lent notional
    tracking_bps: float = 6.0,         # daily idiosyncratic tracking noise per fund
    start: str = "2010-02-11",
    seed: int = 941,
    cash_rate_ann: float = 0.02,
) -> tuple[pd.DataFrame, dict]:
    """A daily tape of (underlying, 3x long fund, −3x inverse fund, cash).

    The volatility is **high in both worlds** — so both funds always suffer the textbook
    ``0.5 * L * (L - 1) * sigma^2`` compounding decay. What the ``signal_strength`` knob
    scales is the funds' **fee load** (expense ratio + financing spread):

    - ``signal_strength = 1`` → the funds charge a realistic all-in load, and an
      equal-dollar short of both must harvest (roughly) the *average* of the two loads.
    - ``signal_strength = 0`` → cost-free funds. The volatility decay is still enormous,
      but a constant-dollar short of both must return ~0 excess-of-cash. This is the null,
      and it is the study's whole point: the decay is a *geometric* artefact that a
      constant-dollar short does not collect — only the fee load is collectable.

    Returns ``(prices, truth)`` where ``prices`` has columns ``under``, ``lev_long``,
    ``lev_short`` and ``cash``, and ``truth`` records the planted parameters, the
    theoretical decay of each leg, and the harvest the machinery *should* recover.
    Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = n_years * TRADING_DAYS_PER_YEAR
    # OOB-safe: a business-day index of n <= 10,000 bars stays well inside pandas' ns range.
    dates = _business_index(start, n_days)

    d = drift_ann / TRADING_DAYS_PER_YEAR
    s = vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)

    # Lognormal vol clustering: a persistent stochastic scale on top of the base vol, so
    # the levered legs see fat tails and clustered decay rather than clean Gaussian noise.
    z = rng.normal(0.0, 1.0, n_days)
    eps = rng.normal(0.0, vol_of_vol * 0.1, n_days)
    logv = np.zeros(n_days)
    for i in range(1, n_days):
        logv[i] = 0.97 * logv[i - 1] + eps[i]
    scale = np.exp(logv - 0.5 * logv.var())
    r_under = d + s * scale * z

    expense_eff = signal_strength * expense_ann
    spread_eff = signal_strength * financing_spread_ann
    drag_long = expense_eff + 2.0 * spread_eff      # 3x long finances 2x NAV
    drag_short = expense_eff + 4.0 * spread_eff     # -3x inverse lends/borrows 4x NAV

    r_cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    tr = tracking_bps * 1e-4
    # Tracking noise is demeaned by construction: it is a *disturbance*, not a second
    # planted effect, so the only thing the harvest can recover is the fee load.
    noise_long = rng.normal(0.0, tr, n_days)
    noise_short = rng.normal(0.0, tr, n_days)
    noise_long -= noise_long.mean()
    noise_short -= noise_short.mean()
    under = 100.0 * np.cumprod(1.0 + r_under)
    lev_long = build_levered(r_under, 3.0, r_cash_daily, drag_long, noise_long)
    lev_short = build_levered(r_under, -3.0, r_cash_daily, drag_short, noise_short)
    cash_idx = 100.0 * np.cumprod(np.full(n_days, 1.0 + r_cash_daily))

    prices = pd.DataFrame(
        {"under": under, "lev_long": lev_long, "lev_short": lev_short, "cash": cash_idx},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "signal_strength": signal_strength,
        "drift_ann": drift_ann,
        "vol_ann": vol_ann,
        "n_years": n_years,
        "n_days": n_days,
        "seed": seed,
        "cash_rate_ann": cash_rate_ann,
        "drag_long_ann": float(drag_long),
        "drag_short_ann": float(drag_short),
        # What an equal-dollar, daily-rebalanced short of both legs should recover.
        "expected_harvest_ann": float(0.5 * (drag_long + drag_short)),
        # Textbook decay of a daily-reset L-times fund: 0.5 * L * (L-1) * sigma^2 per year.
        # Present at FULL size in both worlds — and uncollectable in both.
        "decay_long_ann": float(0.5 * 3.0 * 2.0 * vol_ann ** 2),
        "decay_short_ann": float(0.5 * 3.0 * 4.0 * vol_ann ** 2),
        "realised_vol_under": float(np.std(r_under, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "tracking_bps": tracking_bps,
    }
    return prices, truth


def synthetic_panel(
    n_seeds: int = 6,
    signal_strength: float = 1.0,
    seed0: int = 941,
    **kwargs,
) -> list[tuple[pd.DataFrame, dict]]:
    """A small panel of independent synthetic worlds — one per seed (offline, deterministic).

    Used by the tests and the notebooks to check that the planted harvest is recovered
    across seeds and that the null stays quiet, rather than reading a single lucky path.
    """
    return [
        synthetic_daily(signal_strength=signal_strength, seed=seed0 + i, **kwargs)
        for i in range(n_seeds)
    ]
