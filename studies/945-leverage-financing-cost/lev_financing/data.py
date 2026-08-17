"""Data layer for Study 945 — The Hidden Financing (the borrow rate inside a leveraged ETF).

Two tapes, one shape (a date-indexed daily frame of closes):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the two leveraged wrappers (SSO = 2x S&P 500,
  UPRO = 3x S&P 500), the unlevered benchmark (SPY), the cash leg (BIL, the 1-3 month
  T-bill ETF) and two *rate/price* series that are **not** total-return objects and are
  labelled as such everywhere: ``^IRX`` (the 13-week T-bill discount rate, quoted in
  percent per annum) and ``^GSPC`` (the S&P 500 **price** index, used only as a
  cross-check that shows what happens if you regress on the wrong benchmark).
  ``fetch`` touches the network and caches parquet in the **shared** ``studies/_cache``
  (retry up to 4x); ``load_prices`` reads that cache **offline** and never imports
  yfinance. The whole test-suite runs with NO cache present (synthetic-only), so CI is
  green on a fresh checkout.

- ``synthetic_panel`` — a *deterministic, offline* generator. It builds an index price
  path, a cash/benchmark rate path, and two **synthetic leveraged funds** assembled from
  a *known* financing rate and a *known* expense ratio, plus tracking noise. The
  ``signal_strength`` knob scales the planted financing **spread** over the benchmark
  rate: at ``signal_strength=1`` the wrappers borrow at ``rate + 75 bp`` (the effect the
  estimator must recover); at ``signal_strength=0`` they borrow at exactly the benchmark
  rate with a zero expense ratio, so the estimator must report a spread of ~0 (the null).

The question the study asks: a 2x fund quietly borrows one dollar for every dollar of
your NAV, and a 3x fund borrows two. **At what rate?** The fund never quotes it. But it
is recoverable: regress the fund's daily total return on the benchmark's, and the
intercept is the whole per-day drag — expense ratio plus financing. Strip the published
expense ratio and what is left, annualised and divided by the borrowed dollars, is the
implied financing rate. Compare it with ``^IRX`` and you have the spread the wrapper
charges over T-bills; compare that with a margin desk's quote and you know whether the
wrapper is a cheap loan or an expensive one.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the daily
reset is set from the close of day ``t`` and earns day ``t+1``'s return).
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

# SSO/UPRO = the two survivors of the ProShares S&P 500 leveraged line; SPY = the
# unlevered total-return benchmark; BIL = the investable cash leg; ^IRX = the quoted
# 13-week bill rate; ^GSPC = the S&P 500 PRICE index (wrong-benchmark cross-check).
#
# SURVIVORSHIP (named, not fixed): SSO and UPRO are the wrappers that lived. The US
# leveraged-ETF cohort lost members over this window, and a fund that closed is more
# likely to have been the expensive or badly tracked one, so an estimate built from the
# survivors is a FLOOR for the class rather than its average.
#
# NOT TWO INDEPENDENT DRAWS: both are ProShares funds, sharing an issuer, a swap desk and
# a counterparty list. Their agreement on the implied rate rules out a mis-specified
# estimator; it is a consistency check, not independent confirmation of the level.
#
# ^IRX BASIS (PROXY): Yahoo quotes ^IRX on a bank-DISCOUNT basis. The bond-equivalent
# yield an investor would actually earn is slightly higher (1.396% discount -> ~1.420% BEY
# over this window), so every "spread over ^IRX" reported here is overstated by roughly
# 2-3 bp. Small against a ~68 bp mark-up, and it cuts against the study's own conclusion.
TICKERS = ("SSO", "UPRO", "SPY", "BIL", "IRX", "GSPC")

# The levered wrappers under test and their stated leverage.
FUNDS = {"SSO": 2, "UPRO": 3}

# --- PROXIES / ASSUMPTIONS (non-tape inputs; every one is swept in strategy.py) ------
# Prospectus gross expense ratios, % per annum, as published by ProShares in 2026.
# These are ASSUMPTIONS, not tape: they are read off a document, they have drifted a few
# basis points over the sample, and the headline is swept across a +/- 15 bp band.
EXPENSE_RATIO = {"SSO": 0.89, "UPRO": 0.91}
# SPY's own expense ratio, % per annum. The benchmark leg is SPY's total return, which
# already runs this much *below* the index's total return, so it is added back when the
# drag is expressed against the index rather than against SPY. ASSUMPTION.
SPY_EXPENSE_RATIO = 0.0945
# Indicative broker margin spreads over the benchmark rate, % per annum (ASSUMPTION —
# public rate cards, tier- and balance-dependent, and they move). Used only for the
# "wrapper vs margin account" read, and always shown as a swept grid.
MARGIN_SPREADS = {
    "prime-broker tier": 0.75,
    "low-cost retail": 1.50,
    "mainstream discount broker": 4.00,
    "full-service retail": 6.00,
}

# Study-wide as-of: the last COMPLETE calendar month at build time (drop the partial
# current month so the sample never creeps between reruns).
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily closes, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2006-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily closes for ``tickers`` and cache each as parquet in the shared cache.

    Network-only; run once to populate the cache. ``auto_adjust=True`` so the ``close``
    column is split- and dividend-adjusted **total return** for the ETFs — essential
    here, because a leveraged fund distributes part of its collateral income and its
    drag would otherwise be over-stated by whatever it paid out. ``^IRX`` and ``^GSPC``
    are *not* total-return objects (a quoted rate and a price index respectively); the
    same loader stores them, and every use of them is labelled.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    out: dict[str, pd.DataFrame] = {}
    os.makedirs(cache_dir, exist_ok=True)
    for tk in tickers:
        symbol = tk if tk in ("SSO", "UPRO", "SPY", "BIL") else "^" + tk
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(
                    symbol, start=start, end=end, interval="1d",
                    auto_adjust=True, progress=False,
                )
                if raw is not None and len(raw) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"yfinance returned no data for {symbol}")
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
    """Read cached daily closes OFFLINE into one aligned frame, sliced to ``asof``.

    One column per ticker. ETF columns are adjusted total-return closes; ``IRX`` is a
    quoted annual rate in percent and ``GSPC`` a price index — neither is a wealth
    series, and neither is ever compounded as one. Raises ``FileNotFoundError`` if any
    ticker is missing: the offline core and the test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call lev_financing.data.fetch() once to populate the shared cache."
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
# Synthetic tape — the deterministic offline core (a planted financing spread)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_years: int = 24,
    leverages=(2, 3),
    drift: float = 0.09,               # annualised index drift
    vol: float = 0.17,                 # annualised index vol
    rate_low: float = 0.0015,          # benchmark rate in the ZIRP half (0.15%)
    rate_high: float = 0.0500,         # benchmark rate in the hiking half (5.00%)
    planted_spread: float = 0.0075,    # financing spread over the benchmark rate (75 bp)
    expense_ratio: float = 0.0090,     # fund expense ratio, 0.90%
    tracking_bps: float = 5.0,         # daily tracking noise, bps
    signal_strength: float = 1.0,      # 0 = null (no spread, no fee), 1 = full effect
    start: str = "2002-01-02",
    seed: int = 945,
) -> tuple[pd.DataFrame, dict]:
    """A daily index path plus synthetic leveraged funds built from a KNOWN financing rate.

    The index follows a lognormal random walk. The benchmark rate steps from ``rate_low``
    to ``rate_high`` half-way through the sample, so the generator contains a rate cycle
    the rolling estimator has to track. Each synthetic fund's daily total return is

        r_fund = L * r_index - (L - 1) * (rate + spread) / 252 - fee / 252 + noise

    which is exactly the accounting the estimator inverts. The ``signal_strength`` knob
    scales *both* the planted spread and the fee toward zero:

    - ``signal_strength = 1`` -> the wrappers borrow at ``rate + 75 bp`` and charge 0.90%;
      the estimator must recover both.
    - ``signal_strength = 0`` -> they borrow at exactly the benchmark rate and charge
      nothing; the estimator must report an implied spread of ~0 (the null).

    Returns ``(frame, truth)``. ``frame`` carries ``index`` (a price level), ``cash`` (a
    cash accrual index), ``rate`` (the benchmark rate in **percent per annum**, mirroring
    ``^IRX``'s units) and one column per leverage (``fund_2``, ``fund_3``) holding the
    synthetic fund's total-return level. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: a few thousand business days stays far inside pandas' ns Timestamp range.
    dates = pd.DatetimeIndex(pd.bdate_range(start=start, periods=n_days), name="date")

    spread = planted_spread * signal_strength
    fee = expense_ratio * signal_strength

    rate = np.where(np.arange(n_days) < n_days // 2, rate_low, rate_high)
    d = drift / TRADING_DAYS_PER_YEAR
    s = vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    r_idx = rng.normal(d, s, n_days)
    index = 100.0 * np.cumprod(1.0 + r_idx)

    cash = 100.0 * np.cumprod(1.0 + rate / TRADING_DAYS_PER_YEAR)

    frame = {
        "index": index,
        "cash": cash,
        "rate": rate * 100.0,  # percent per annum, like ^IRX
    }
    for L in leverages:
        noise = rng.normal(0.0, tracking_bps * 1e-4, n_days)
        r_fund = (
            L * r_idx
            - (L - 1) * (rate + spread) / TRADING_DAYS_PER_YEAR
            - fee / TRADING_DAYS_PER_YEAR
            + noise
        )
        frame[f"fund_{L}"] = 100.0 * np.cumprod(1.0 + r_fund)

    prices = pd.DataFrame(frame, index=dates)
    truth = {
        "signal_strength": signal_strength,
        "planted_spread_pct": spread * 100.0,
        "expense_ratio_pct": fee * 100.0,
        "rate_low_pct": rate_low * 100.0,
        "rate_high_pct": rate_high * 100.0,
        "mean_rate_pct": float(rate.mean() * 100.0),
        "leverages": tuple(leverages),
        "n_days": n_days,
        "n_years": n_years,
        "seed": seed,
        "tracking_bps": tracking_bps,
    }
    return prices, truth


def synthetic_daily(**kwargs) -> tuple[pd.DataFrame, dict]:
    """Single-fund convenience wrapper around :func:`synthetic_panel` (2x only)."""
    kwargs.setdefault("leverages", (2,))
    return synthetic_panel(**kwargs)
