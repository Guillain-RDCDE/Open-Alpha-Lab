"""Data layer for Study 953 — Replicating the Convert.

Two tapes, one shape (a date-indexed daily total-return close frame):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the two convertible-bond ETFs (CWB, the
  $4bn Bloomberg US Convertible Liquid Bond tracker, and ICVT, its investment-grade
  cousin), the three replication building blocks (SPY = large-cap equity, QQQ =
  the growth/tech tilt convertible issuers actually have, LQD = investment-grade
  credit) and the cash leg (BIL, 1-3 month T-bills). ``fetch`` touches the network
  and caches parquet under the **shared** ``studies/_cache`` (retry up to 4x);
  ``load_prices`` reads that cache **offline** and never imports ``yfinance``.
  The whole test-suite runs with NO cache present (synthetic-only), so CI is green
  on a fresh checkout.

- ``synthetic_panel`` — a *deterministic, offline* generator. Three correlated
  factor tapes (equity, growth, credit), a cash accrual leg, and a "fund" built as
  a known static mix of them plus an optional planted **alpha** and planted
  **convexity** (a call-like kicker that pays extra in big up months and cushions
  down months) plus idiosyncratic noise. The ``signal_strength`` knob scales *both*
  planted effects to zero: at ``signal_strength=0`` the fund is *exactly* a static
  mix plus noise (the null — the harness must find no alpha and no convexity); at
  ``signal_strength=1`` both effects are real and must be recovered. Seed is fixed
  so tests are deterministic.

Total return vs price only: every series here is **total return** (``auto_adjust=True``),
which is the only honest basis for a coupon-bearing wrapper like a convertible fund —
CWB's distribution yield has run 1-3%/yr and a price-only comparison would understate
it by roughly that much. CWB's and ICVT's total returns are already **net of the fund's
own expense ratio** (0.40% and 0.20% respectively, per the issuers) because the ratio
is deducted from NAV daily; the replication legs are gross of *their* fees, so the
replica is given a fee handicap in ``strategy`` and the handicap is swept.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the
replication weights are fitted on the in-sample window only and applied strictly from
the first trading day *after* it.
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

# The funds under test and the replication kit.
FUNDS = ("CWB", "ICVT")
FACTORS = ("SPY", "QQQ", "LQD")
CASH = "BIL"
TICKERS = ("CWB", "ICVT", "SPY", "QQQ", "LQD", "BIL")

# Study-wide as-of: the last COMPLETE calendar month at build time (drop the partial
# current month so the sample never creeps between reruns).
AS_OF = "2026-06-30"

# The in-sample / out-of-sample boundary for CWB (roughly the first half of its life).
# Weights are fitted through IS_END and held fixed for everything after it.
IS_END = "2017-12-31"

# PROXY / ASSUMPTION (not tape): published expense ratios, used only to *label* the
# fund's fee drag and to set the replica's own fee handicap. Swept in strategy.py.
FUND_FEE_ANN = {"CWB": 0.0040, "ICVT": 0.0020}
REPLICA_FEE_ANN = 0.0010  # a blended ~10 bps/yr for a SPY/QQQ/LQD sleeve; swept


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2002-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` so the
    ``close`` column is split- and distribution-adjusted total return — mandatory here,
    since both the convertible funds and LQD pay meaningful coupons.
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
                f"Call convert_repl.data.fetch() once to populate the cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df = df[df.index <= pd.Timestamp(asof)]
    return df


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (planted alpha + convexity)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_years: int = 16,
    w_true: tuple[float, float, float] = (0.30, 0.20, 0.35),
    alpha_ann: float = 0.020,          # planted annual alpha over the static mix
    convexity: float = 0.25,           # planted delta ratchet (the convertible's convexity)
    idio_vol_ann: float = 0.030,       # fund-specific noise
    vol_ann: tuple[float, float, float] = (0.17, 0.22, 0.08),
    drift_ann: tuple[float, float, float] = (0.09, 0.12, 0.03),
    corr_eq_growth: float = 0.90,
    corr_eq_credit: float = 0.25,
    signal_strength: float = 1.0,      # 0 = pure static mix + noise (the null)
    start: str = "2009-04-16",
    seed: int = 953,
    cash_rate_ann: float = 0.015,
) -> tuple[pd.DataFrame, dict]:
    """A daily panel of three factor tapes, a cash leg, and a convertible-like fund.

    The fund is built as ``w_true . factors`` (the rest in cash) **plus** two planted
    effects, both scaled by ``signal_strength``:

    - an annual **alpha** of ``alpha_ann`` spread evenly across days;
    - a **delta ratchet** of size ``convexity``: the fund's equity exposure rises after
      the equity leg has rallied and falls after it has sold off (a squashed function of
      the trailing 21-day move, lagged one day so the generator itself never peeks).
      That is exactly what a real convertible does — its delta rises with the stock —
      and against a *static* mix it produces a genuine monthly smile: the fund beats its
      replica in big up months *and* loses less in big down months.

    At ``signal_strength = 0`` both vanish and the fund is *exactly* a static mix plus
    idiosyncratic noise: the null on which the harness must report no alpha and no
    convexity. Returns ``(prices, truth)`` where ``prices`` holds total-return close
    levels for ``eq``, ``growth``, ``credit``, ``cash`` and ``fund``. Deterministic.
    """
    rng = np.random.default_rng(seed)
    n_days = n_years * TRADING_DAYS_PER_YEAR
    # OOB-safe: bdate_range with n <= 10000 daily bars stays well inside pandas' ns range.
    dates = pd.bdate_range(start=start, periods=n_days)

    sig = np.array([v / np.sqrt(TRADING_DAYS_PER_YEAR) for v in vol_ann])
    mu = np.array([v / TRADING_DAYS_PER_YEAR for v in drift_ann])
    corr = np.array([
        [1.0, corr_eq_growth, corr_eq_credit],
        [corr_eq_growth, 1.0, corr_eq_credit * 0.8],
        [corr_eq_credit, corr_eq_credit * 0.8, 1.0],
    ])
    cov = corr * np.outer(sig, sig)
    chol = np.linalg.cholesky(cov)
    shocks = rng.standard_normal((n_days, 3)) @ chol.T
    fac = shocks + mu  # daily simple returns of the three factors

    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    r_cash = np.full(n_days, cash_daily)

    w = np.asarray(w_true, dtype=float)
    ex_fac = fac - r_cash[:, None]
    base_ex = ex_fac @ w                       # the static-mix excess return

    eq_ex = ex_fac[:, 0]
    # Delta ratchet: a squashed trailing-21-day equity move, lagged one day.
    trail = pd.Series(eq_ex).rolling(21, min_periods=21).sum().shift(1).fillna(0.0)
    sig_month = sig[0] * np.sqrt(21.0)
    delta_extra = convexity * np.tanh(trail.to_numpy() / (2.0 * sig_month))
    kicker = delta_extra * eq_ex
    alpha_d = alpha_ann / TRADING_DAYS_PER_YEAR
    idio = rng.normal(0.0, idio_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR), n_days)

    fund_ex = base_ex + signal_strength * (alpha_d + kicker) + idio
    r_fund = fund_ex + r_cash

    def level(r):
        return 100.0 * np.cumprod(1.0 + r)

    prices = pd.DataFrame(
        {
            "eq": level(fac[:, 0]),
            "growth": level(fac[:, 1]),
            "credit": level(fac[:, 2]),
            "cash": level(r_cash),
            "fund": level(r_fund),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "signal_strength": signal_strength,
        "w_true": tuple(float(x) for x in w),
        "alpha_ann": alpha_ann * signal_strength,
        "convexity": convexity * signal_strength,
        "idio_vol_ann": idio_vol_ann,
        "n_years": n_years,
        "n_days": n_days,
        "seed": seed,
        "cash_rate_ann": cash_rate_ann,
        "factors": ("eq", "growth", "credit"),
    }
    return prices, truth


def synthetic_daily(**kwargs):
    """Alias kept for symmetry with the single-asset studies: see ``synthetic_panel``."""
    return synthetic_panel(**kwargs)
