"""Data layer for Study 954 — High Yield in Disguise.

Two tapes, one shape (a date-indexed daily total-return close frame):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the high-yield credit funds (HYG, JNK,
  USHY), the two replication legs (SPY for equity, IEF for 7-10y Treasury duration),
  a cash proxy (BIL, the 1-3 month T-bill ETF) and three alternative Treasury
  maturities (SHY, IEI, TLT) used only to sweep the duration-leg choice.
  ``fetch`` touches the network and
  caches parquet under the **shared** ``studies/_cache`` (retry up to 4x);
  ``load_prices`` reads that cache **offline** and never imports yfinance. The whole
  test-suite runs with NO cache present (synthetic-only), so CI is green on a fresh
  checkout where ``_cache/`` is git-ignored and absent.

- ``synthetic_panel`` / ``synthetic_daily`` — a *deterministic, offline* generator. It
  builds an equity tape, a duration tape, a cash accrual leg, and a "credit fund" that
  is *by construction* ``w_true * equity + (1 - w_true) * duration`` plus an
  idiosyncratic credit shock. The ``signal_strength`` knob plants a **drag** on that
  idiosyncratic leg: at ``signal_strength = 0`` the fund is exactly its replication in
  expectation (the null — the harness must stay quiet); at ``signal_strength = 1`` the
  fund carries a real, detectable annual give-up the harness must find. Seed is fixed
  → tests are deterministic.

Everything here is **total return** (``auto_adjust=True``), never price-only: high-yield
funds distribute 5-8% a year, so a price-only tape would be a different (and wrong)
question entirely.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the blend
weight used on day ``t`` is fitted on returns ending at the *previous* month-end).
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

# HYG/JNK/USHY = the high-yield funds under test; SPY = the equity leg; IEF = the
# 7-10y Treasury duration leg; BIL = the 1-3M T-bill cash proxy every leg is raced
# excess-of.
HY_TICKERS = ("HYG", "JNK", "USHY")
LEG_TICKERS = ("SPY", "IEF", "BIL")
# Alternative Treasury maturities for the duration leg. Choosing IEF (7-10y) is a design
# choice, not tape — a high-yield fund's own duration is nearer 3-4 years — so the race is
# re-run on SHY (1-3y), IEI (3-7y) and TLT (20y+) too. Treasuries ONLY: a leg containing
# corporate credit (AGG, BND, LQD) would smuggle the thing under test into the benchmark.
DUR_LEG_TICKERS = ("SHY", "IEI", "TLT")
TICKERS = HY_TICKERS + LEG_TICKERS + DUR_LEG_TICKERS

# Study-wide as-of: the last COMPLETE calendar month at build time (drop the partial
# current month so the sample never creeps between reruns).
AS_OF = "2026-06-30"

# --------------------------------------------------------------------------- #
# PROXIES / ASSUMPTIONS (non-tape inputs — labelled here, swept in strategy.py)
# --------------------------------------------------------------------------- #
# Published net expense ratios, %/yr, as advertised by the issuers at build time. These
# are NOT tape — they are quoted facts used only to *decompose* the measured gap in the
# narrative. Every headline number is computed from total-return prices in which these
# fees are already deducted, so nothing double-counts them.
EXPENSE_RATIO_PCT = {"HYG": 0.49, "JNK": 0.40, "USHY": 0.08, "SPY": 0.0945, "IEF": 0.15}


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2004-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` makes the
    ``close`` column split- and distribution-adjusted total return — indispensable here,
    since a high-yield fund pays most of its return out as coupon income.
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
                f"Call hy_replication.data.fetch() once to populate the cache."
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
# Synthetic tape — the deterministic offline core (a planted replication gap)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_years: int = 18,
    w_true: float = 0.45,              # the planted equity share of the credit fund
    mu_equity: float = 0.09,           # annualised equity total return
    vol_equity: float = 0.18,
    mu_duration: float = 0.035,        # annualised Treasury total return
    vol_duration: float = 0.07,
    rho_eq_dur: float = -0.20,         # equity/duration correlation
    resid_vol: float = 0.06,           # annualised idiosyncratic credit-shock vol
    resid_drag: float = 0.030,         # annualised give-up planted at signal_strength=1
    signal_strength: float = 1.0,      # 0 = fund IS its replication (null), 1 = full drag
    start: str = "2008-01-02",
    seed: int = 954,
    cash_rate_ann: float = 0.02,
) -> tuple[pd.DataFrame, dict]:
    """A daily four-leg tape: a credit fund, its two replication legs, and cash.

    The credit fund is built *from* the legs::

        r_fund = w_true * r_equity + (1 - w_true) * r_duration + credit_shock

    where ``credit_shock`` always has annualised vol ``resid_vol`` — the fund is never a
    pure blend, exactly as on the real tape — and ``signal_strength`` sets only what that
    extra risk is *paid*:

    - ``signal_strength = 0`` → the shock earns its **fair premium**, the mean that leaves
      the fund's excess-of-cash Sharpe exactly equal to the replication's. This is the
      null: extra risk, fairly compensated, so the vol-matched race must come out level.
    - ``signal_strength = 1`` → the shock instead carries a ``resid_drag`` annual give-up,
      i.e. uncompensated risk; the harness *must* hand the replication the higher Sharpe.

    The fitted weight must recover ``w_true`` in both worlds — that separates "we measured
    the blend correctly" from "the blend won".

    Returns ``(prices, truth)`` where ``prices`` carries the total-return close levels
    ``fund``, ``equity``, ``duration`` and ``cash``. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: bdate_range with n <= 10000 daily bars stays well inside pandas' ns range.
    dates = pd.bdate_range(start=start, periods=n_days)

    sq = np.sqrt(TRADING_DAYS_PER_YEAR)
    z = rng.normal(size=(n_days, 3))
    eq = z[:, 0]
    du = rho_eq_dur * z[:, 0] + np.sqrt(max(1.0 - rho_eq_dur ** 2, 0.0)) * z[:, 1]

    r_equity = mu_equity / TRADING_DAYS_PER_YEAR + (vol_equity / sq) * eq
    r_duration = mu_duration / TRADING_DAYS_PER_YEAR + (vol_duration / sq) * du

    # The "fair premium": the annual mean on the idiosyncratic shock that leaves the
    # fund's excess-of-cash Sharpe exactly equal to the replication's, given that the
    # shock inflates the fund's volatility. Derived from the blend's annualised moments.
    w = w_true
    vol_blend = np.sqrt(
        (w * vol_equity) ** 2 + ((1 - w) * vol_duration) ** 2
        + 2 * w * (1 - w) * rho_eq_dur * vol_equity * vol_duration
    )
    mu_blend_excess = w * mu_equity + (1 - w) * mu_duration - cash_rate_ann
    vol_fund = np.sqrt(vol_blend ** 2 + resid_vol ** 2)
    fair_premium = mu_blend_excess * (vol_fund / vol_blend - 1.0) if vol_blend > 0 else 0.0

    shock_mean_ann = signal_strength * (-resid_drag) + (1.0 - signal_strength) * fair_premium
    shock = shock_mean_ann / TRADING_DAYS_PER_YEAR + (resid_vol / sq) * z[:, 2]
    r_fund = w_true * r_equity + (1.0 - w_true) * r_duration + shock

    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    r_cash = np.full(n_days, cash_daily)

    prices = pd.DataFrame(
        {
            "fund": 100.0 * np.cumprod(1.0 + r_fund),
            "equity": 100.0 * np.cumprod(1.0 + r_equity),
            "duration": 100.0 * np.cumprod(1.0 + r_duration),
            "cash": 100.0 * np.cumprod(1.0 + r_cash),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "w_true": w_true,
        "signal_strength": signal_strength,
        "resid_drag": resid_drag,
        "resid_vol": resid_vol,
        "fair_premium_ann": float(fair_premium),
        "shock_mean_ann": float(shock_mean_ann),
        "planted_drag_ann": signal_strength * resid_drag,
        "vol_blend_ann": float(vol_blend),
        "vol_fund_ann": float(vol_fund),
        "mu_equity": mu_equity, "vol_equity": vol_equity,
        "mu_duration": mu_duration, "vol_duration": vol_duration,
        "rho_eq_dur": rho_eq_dur,
        "cash_rate_ann": cash_rate_ann,
        "n_years": n_years, "n_days": n_days, "seed": seed,
    }
    return prices, truth


def synthetic_daily(**kwargs) -> tuple[pd.DataFrame, dict]:
    """Alias of :func:`synthetic_panel` (the desk's usual single-tape entry point)."""
    return synthetic_panel(**kwargs)
