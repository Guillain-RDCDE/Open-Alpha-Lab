"""Data layer for Study 917 — Stale NAV (US-session spillover into country ETFs).

Two tapes, one shape (a date-indexed daily total-return close frame):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the US benchmark (SPY), five US-listed
  single-country ETFs whose underlying markets are **closed** during the US session
  (EWJ Japan, EWG Germany, FXI China/HK, EWA Australia, EWU United Kingdom), and two
  cash legs: ``^IRX`` (the 13-week T-bill discount rate, our long-sample cash **proxy**)
  and BIL (the tradable 1-3M T-bill ETF, available only from 2007). ``fetch`` touches the
  network and caches parquet under the **shared** ``studies/_cache`` (retry up to 4×);
  ``load_prices`` reads that cache **offline** and never imports yfinance. The whole
  test-suite runs with NO cache present (synthetic-only), so CI is green on a fresh
  checkout.

- ``synthetic_panel`` / ``synthetic_daily`` — a *deterministic, offline* generator. A US
  benchmark tape plus ``n_funds`` country tapes that load on the *contemporaneous* US
  move and, with probability set by ``stale_beta``, absorb a further slice of **yesterday's**
  US move today — the planted stale-price catch-up. The ``signal_strength`` knob scales that
  planted catch-up to zero (the null), where the lagged regression must find nothing.
  Seeds are fixed → tests are deterministic.

The claim under test: Tokyo, Frankfurt and Hong Kong are shut while Wall Street trades, so
a country ETF's price is supposed to be *stale* with respect to the US session that has
just happened — and should therefore "catch up" the next day. That is the textbook
stale-price / fair-value-pricing story from the mutual-fund market-timing literature.
Whether it survives on **US-listed, continuously traded ETFs** in 2026 is the question.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the signal is
formed from data through day ``t`` and acted on for the day ``t+1`` return).
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

# The five single-country ETFs: their home markets are all shut during the US session.
FUNDS = ("EWJ", "EWG", "FXI", "EWA", "EWU")
BENCHMARK = "SPY"

# Everything the study pulls. ``^IRX`` is the long-sample cash PROXY (13-week T-bill
# discount rate, quoted in percent); BIL is the tradable cash ETF used as a cross-check
# from its 2007 inception.
TICKERS = (BENCHMARK,) + FUNDS + ("BIL", "^IRX")

# Study-wide as-of: the last COMPLETE calendar month at build time (drop the partial
# current month so the sample never creeps between reruns).
AS_OF = "2026-06-30"


def safe_name(ticker: str) -> str:
    """Cache-file-safe ticker name: ``^IRX`` -> ``IRX`` (the desk-wide convention)."""
    return ticker.replace("=", "").replace("^", "").replace("/", "")


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"prices_{safe_name(ticker)}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "1993-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` so the
    ``close`` column is split- and dividend-adjusted total return — essential here,
    because country ETFs pay large, lumpy dividends and a price-only tape would put a
    spurious sawtooth into exactly the daily returns we are regressing.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    out = {}
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
        out[safe_name(tk)] = df
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

    Columns are the *safe* ticker names (``^IRX`` -> ``IRX``). Sliced to ``asof`` so the
    sample never creeps. Raises ``FileNotFoundError`` if any ticker is missing — the
    offline core and the test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call stale_nav.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[safe_name(tk)] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df = df[df.index <= pd.Timestamp(asof)]
    return df


def cash_returns_from_irx(irx: pd.Series) -> pd.Series:
    """Daily cash return implied by the ``^IRX`` 13-week T-bill discount rate.

    ``^IRX`` is quoted in percent (e.g. 5.20 = 5.20%/yr). We compound it daily and lag it
    one day, so the cash credited over day ``t`` uses the rate quoted at the close of
    ``t-1`` — known in advance, no look-ahead. This is a **PROXY** for a tradable cash
    leg (it is an index level, not an investable fund); ``examples/verify.py`` re-runs the
    headline on BIL's actual total return over BIL's shorter 2007+ window as a check.
    """
    r = ((1.0 + irx.astype(float) / 100.0) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0)
    return r.shift(1).rename("cash")


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted stale-price catch-up)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_years: int = 20,
    n_funds: int = 5,
    beta_contemp: float = 0.55,     # same-day loading of a country fund on the US session
    stale_beta: float = 0.25,       # planted CATCH-UP: yesterday's US move, absorbed today
    vol_us_ann: float = 0.17,
    vol_local_ann: float = 0.14,    # idiosyncratic (home-market) vol on top of the US load
    drift_ann: float = 0.06,
    signal_strength: float = 1.0,   # 0 = no catch-up at all (the null), 1 = full stale_beta
    start: str = "2000-01-03",
    seed: int = 917,
    cash_rate_ann: float = 0.02,
) -> tuple:
    """A US benchmark tape plus ``n_funds`` country-ETF tapes with a planted catch-up.

    Each fund's daily return is

        ``r_f[t] = drift + beta_contemp * r_us[t] + ss * stale_beta * r_us[t-1] + eps_f[t]``

    so at ``signal_strength = 1`` yesterday's US session genuinely leaks into today's
    country return (the stale-price catch-up the study looks for), and at
    ``signal_strength = 0`` it does not (the null — the lagged regression must find
    nothing, and the top-decile rule must not print an edge).

    Returns ``(prices, truth)`` where ``prices`` has columns ``SPY``, ``FUND1..FUNDk`` and
    ``cash`` (a cash accrual index), and ``truth`` records the planted parameters.
    Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: bdate_range with n <= 10000 daily bars stays well inside pandas' ns range.
    dates = pd.bdate_range(start=start, periods=n_days)

    s_us = vol_us_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    s_loc = vol_local_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    mu = drift_ann / TRADING_DAYS_PER_YEAR
    stale_eff = float(signal_strength) * stale_beta

    r_us = rng.normal(mu, s_us, n_days)
    r_us_lag = np.concatenate([[0.0], r_us[:-1]])

    cols = {BENCHMARK: 100.0 * np.cumprod(1.0 + r_us)}
    for k in range(n_funds):
        eps = rng.normal(0.0, s_loc, n_days)
        r_f = mu + beta_contemp * r_us + stale_eff * r_us_lag + eps
        cols[f"FUND{k + 1}"] = 100.0 * np.cumprod(1.0 + r_f)

    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cols["cash"] = np.cumprod(np.full(n_days, cash_daily))

    prices = pd.DataFrame(cols, index=pd.DatetimeIndex(dates, name="date"))
    truth = {
        "signal_strength": float(signal_strength),
        "stale_beta": float(stale_beta),
        "stale_beta_effective": stale_eff,
        "beta_contemp": float(beta_contemp),
        "vol_us_ann": vol_us_ann,
        "vol_local_ann": vol_local_ann,
        "drift_ann": drift_ann,
        "n_funds": int(n_funds),
        "n_years": int(n_years),
        "n_days": n_days,
        "seed": int(seed),
        "cash_rate_ann": cash_rate_ann,
        "funds": [f"FUND{k + 1}" for k in range(n_funds)],
    }
    return prices, truth


def synthetic_daily(**kwargs) -> tuple:
    """One-fund convenience wrapper around :func:`synthetic_panel` (same knobs)."""
    kwargs.setdefault("n_funds", 1)
    return synthetic_panel(**kwargs)
