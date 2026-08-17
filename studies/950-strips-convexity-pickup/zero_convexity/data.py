"""Data layer for Study 950 — Zero-Coupon Convexity (STRIPS funds vs a duration-matched mix).

Two tapes, one shape (a date-indexed daily frame of total-return closes plus one yield level):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the two zero-coupon Treasury funds
  (**EDV**, Vanguard Extended Duration Treasury, 20-30y STRIPS; **ZROZ**, PIMCO 25+ Year
  Zero-Coupon), the coupon long bond fund (**TLT**, 20+ Year Treasury) and the cash leg
  (**BIL**, 1-3 month T-bills) — plus the **30-year constant-maturity Treasury yield**
  (``^TYX``), which is a *yield level in percentage points*, not a price. ``fetch``
  touches the network and caches parquet under the shared ``studies/_cache`` (retry up to
  4x); ``load_prices`` reads that cache **offline** and never imports yfinance. The
  ``^`` is stripped from cache filenames and from the returned column names, so the
  30-year yield arrives as the column ``TYX``.

- ``synthetic_panel`` — a *deterministic, offline* generator. A fat-tailed daily yield
  path drives two bond-like total-return series through the textbook second-order
  expansion ``r = -D*dy + 0.5*C*dy**2 + carry``, plus a cash accrual leg. The
  ``signal_strength`` knob controls the **convexity gap per unit of duration**: at
  ``signal_strength=0`` the long-duration leg has exactly the convexity a duration-matched
  mix of the short leg already has (the null — no asymmetry to find); at
  ``signal_strength=1`` it carries a genuine convexity pickup that a large-move regression
  must recover, paid for with a planted carry give-up.

Why the mixture of price and yield tapes: the race is *duration-matched*, and the match is
solved from the realised beta of each fund's excess return to **the same rate factor** —
the daily change in the 30-year yield. Without the yield tape there is no shared factor to
match on, only fund-on-fund regressions that quietly bake in whatever curve exposure the
funds happen to differ by.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the hedge ratio
is estimated on data through the last session of month *m* and traded for all of month
*m+1*).
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

# EDV / ZROZ = zero-coupon (STRIPS) Treasury funds; TLT = the coupon long bond;
# BIL = the 1-3 month T-bill cash leg; ^TYX = the 30-year constant-maturity yield.
TICKERS = ("EDV", "ZROZ", "TLT", "BIL", "^TYX")

# Study-wide as-of: the last COMPLETE calendar month at build time (the partial current
# month is dropped so the sample never creeps between reruns).
AS_OF = "2026-06-30"


def column_name(ticker: str) -> str:
    """Column name a ticker gets in the loaded frame (``^TYX`` -> ``TYX``)."""
    return ticker.replace("^", "").replace("=", "").replace("/", "")


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"prices_{column_name(ticker)}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2002-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` so the
    ETF ``close`` columns are split- and distribution-adjusted **total return** — which
    matters enormously here: EDV and ZROZ distribute the accreted coupon of the STRIPS
    they hold, so their *price-only* series drifts down relative to total return. The
    ``^TYX`` series is a yield **level** in percentage points and is passed through
    unchanged (auto_adjust is a no-op on an index of yields).
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
    """Read the cached daily tapes OFFLINE into one aligned frame.

    Returns a frame indexed by date with one column per ticker (``^`` stripped): the
    adjusted total-return close for the ETFs, the yield level in percentage points for
    ``TYX``. Sliced to ``asof`` so the sample never creeps. Raises ``FileNotFoundError``
    if any tape is missing — the offline core and the whole test-suite never touch the
    network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached series for {tk} at {path}. "
                f"Call zero_convexity.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[column_name(tk)] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df = df[df.index <= pd.Timestamp(asof)]
    return df


def fingerprint(frame: pd.DataFrame) -> str:
    """Short content fingerprint of a frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(frame.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted convexity pickup)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_years: int = 18,
    dur_long: float = 24.0,        # duration of the zero-coupon leg (years)
    dur_short: float = 16.5,       # duration of the coupon long-bond leg (years)
    convexity_short: float = 330.0,  # convexity of the coupon leg (years^2)
    convexity_gap: float = 1.50,   # extra convexity-per-duration of the zero leg at ss=1
    carry_giveup_bp_mo: float = 30.0,  # monthly carry the zero leg gives up at ss=1 (bp)
    dy_daily_bp: float = 5.5,      # daily sd of the 30y yield change (basis points)
    dy_tail_df: float = 4.0,       # Student-t d.o.f. of the yield shocks (fat tails)
    mean_reversion: float = 1.0 / 750.0,  # daily OU pull keeping the level plausible
    idio_bp: float = 6.0,          # daily idiosyncratic return noise on the zero leg (bp)
    signal_strength: float = 1.0,  # 0 = null (no convexity gap), 1 = full pickup
    start: str = "2008-01-02",
    seed: int = 950,
    cash_rate_ann: float = 0.02,
    y0: float = 4.0,               # starting 30y yield level (percentage points)
) -> tuple[pd.DataFrame, dict]:
    """A daily two-bond tape driven by one fat-tailed yield factor.

    Each leg is priced through the textbook second-order expansion in the yield change
    ``dy`` (expressed in decimals, i.e. 1 pp = 0.01)::

        r = -D * dy + 0.5 * C * dy**2 + carry

    with the **zero-coupon leg** carrying ``D = dur_long`` and a convexity of
    ``convexity_short * (dur_long / dur_short) * (1 + convexity_gap * signal_strength)``:

    - ``signal_strength = 0`` → the long leg's convexity-per-unit-duration equals the
      short leg's, so a duration-matched mix of the short leg has *exactly* the same
      convexity. There is no asymmetry to find (the null — the squared-move regression
      must stay quiet).
    - ``signal_strength = 1`` → the long leg carries a genuine convexity pickup, paid for
      with ``carry_giveup_bp_mo`` basis points of monthly carry. A correct harness must
      recover a **positive** loading on the squared yield move *and* a **negative**
      intercept.

    Returns ``(panel, truth)`` where ``panel`` has columns ``zero``, ``coupon`` (both
    total-return close levels), ``cash`` (a cash accrual index) and ``yield_pp`` (the
    30-year yield level in percentage points), plus ``truth`` recording the planted
    parameters. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: bdate_range with a modest count stays well inside pandas' ns horizon.
    dates = pd.bdate_range(start=start, periods=n_days)

    conv_long = (
        convexity_short * (dur_long / dur_short) * (1.0 + convexity_gap * signal_strength)
    )
    carry_giveup_daily = (carry_giveup_bp_mo * 1e-4 / 21.0) * signal_strength

    # Fat-tailed yield shocks (Student-t rescaled to the target daily sd), in decimals,
    # fed through a slow Ornstein-Uhlenbeck pull so the level stays in a plausible band
    # over two decades instead of random-walking off to a negative yield.
    t_raw = rng.standard_t(dy_tail_df, n_days)
    t_raw = t_raw / np.sqrt(dy_tail_df / (dy_tail_df - 2.0))
    shocks = t_raw * (dy_daily_bp * 1e-4)
    level = np.empty(n_days)
    x = 0.0                                    # deviation of the yield from y0, in decimals
    for i in range(n_days):
        x = (1.0 - mean_reversion) * x + shocks[i]
        level[i] = x
    dy = np.diff(level, prepend=0.0)

    # Idiosyncratic *return* noise on the zero leg (20s-vs-30s curve reshaping, fund
    # tracking): the residual left after the 30-year factor is accounted for.
    idio = rng.normal(0.0, idio_bp * 1e-4, n_days)

    carry_short_daily = 0.02 / TRADING_DAYS_PER_YEAR  # a flat term premium for both legs
    r_short = -dur_short * dy + 0.5 * convexity_short * dy ** 2 + carry_short_daily
    r_long = (
        -dur_long * dy
        + 0.5 * conv_long * dy ** 2
        + carry_short_daily * (dur_long / dur_short)
        - carry_giveup_daily
        + idio
    )

    # The two legs above are *excess* returns; the published tapes are total returns, so
    # add the same cash accrual the cash leg earns. (Skip this and the cash rate would be
    # double-counted the moment strategy.py subtracts it again.)
    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    r_cash_daily = cash_daily - 1.0
    zero = 100.0 * np.cumprod(1.0 + r_cash_daily + r_long)
    coupon = 100.0 * np.cumprod(1.0 + r_cash_daily + r_short)
    cash = np.cumprod(np.full(n_days, cash_daily))
    yield_pp = y0 + level * 100.0

    panel = pd.DataFrame(
        {"zero": zero, "coupon": coupon, "cash": cash, "yield_pp": yield_pp},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "signal_strength": signal_strength,
        "dur_long": dur_long,
        "dur_short": dur_short,
        "convexity_short": convexity_short,
        "convexity_long": float(conv_long),
        "convexity_per_dur_short": convexity_short / dur_short,
        "convexity_per_dur_long": float(conv_long / dur_long),
        "convexity_gap": convexity_gap,
        "carry_giveup_bp_mo": carry_giveup_bp_mo * signal_strength,
        "duration_ratio": dur_long / dur_short,
        "n_days": n_days,
        "n_years": n_years,
        "seed": seed,
        "cash_rate_ann": cash_rate_ann,
    }
    return panel, truth


def synthetic_daily(**kwargs) -> tuple[pd.DataFrame, dict]:
    """Alias of :func:`synthetic_panel` (the desk's single-tape naming convention)."""
    return synthetic_panel(**kwargs)
