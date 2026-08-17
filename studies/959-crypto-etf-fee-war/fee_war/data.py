"""Data layer for Study 959 — Crypto Fee War (realised tracking difference of spot BTC ETFs).

Two tapes, one shape (a date-indexed daily close frame):

- ``fetch`` / ``load_prices`` — daily closes from Yahoo! Finance (``yfinance``,
  ``auto_adjust=True``) for the ten US spot-bitcoin ETFs that all began trading on
  **2024-01-11**, the spot benchmark ``BTC-USD``, a cash proxy (``BIL``) and one later
  launch (Grayscale's Mini Trust, ticker ``BTC``, from 2024-07-31) used as an
  out-of-cohort cross-check. ``fetch`` touches the network and caches parquet under the
  shared ``studies/_cache`` (retry up to 4x); ``load_prices`` reads that cache
  **offline** and never imports yfinance. The whole test-suite runs with NO cache
  present (synthetic only), so CI is green on a fresh checkout.

  **Price-only vs total-return.** ``auto_adjust=True`` gives split- and
  distribution-adjusted closes, but none of these funds has ever paid a distribution:
  for the spot-BTC wrappers the adjusted close *is* the price close, and the two labels
  coincide. ``BTC-USD`` is a **price-only** spot quote — bitcoin pays no coupon, and
  none of these US wrappers stakes or lends its coin, so a price benchmark is the right
  one. ``BIL`` is genuine total return (it distributes monthly) and is used only for the
  excess-of-cash Sharpe race.

- ``synthetic_panel`` — a *deterministic, offline* generator. A planted world of ``n_funds``
  wrappers, each shaved by its own fee, quoted against a benchmark that is observed at a
  **different clock** from the funds. The ``signal_strength`` knob collapses the planted
  fee dispersion: at ``signal_strength=0`` every fund charges the cohort-average fee, so
  the fee-rank test has nothing to find (the null); at ``signal_strength=1`` the planted
  dispersion is the real one. Seed is fixed -> tests are deterministic.

The tape's central awkwardness, which the synthetic generator reproduces on purpose:
bitcoin trades 24/7 while the ETFs are struck at the 16:00 New York close. Every
fund-versus-``BTC-USD`` difference therefore carries a large, mean-zero **clock stub**
(~135 bp per day of it), while every fund-versus-fund difference carries almost none
(~8-9 bp). A fee of 20 bp/yr is 0.08 bp/day. That ratio is the whole measurement problem,
and it decides which estimator can see a fee and which cannot.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the wrapper
choice formed on data through day ``t`` is acted on at day ``t+1``).
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
MONTHS_PER_YEAR = 12

# --------------------------------------------------------------------------- #
# The cohort
# --------------------------------------------------------------------------- #
# The ten US spot-bitcoin ETFs that all started trading on 2024-01-11 (the SEC approved
# the batch together, so the cohort shares one inception date and one common window).
COHORT = ("IBIT", "FBTC", "ARKB", "BITB", "HODL", "BRRR", "BTCO", "EZBC", "BTCW", "GBTC")

BENCH = "BTC-USD"      # spot bitcoin, 24/7 price-only
CASH = "BIL"           # 1-3 month T-bill total return, for the excess-of-cash race
MINI = "BTC"           # Grayscale Bitcoin Mini Trust — launched 2024-07-31, out of cohort

TICKERS = COHORT + (BENCH, CASH, MINI)

# An eleventh January-2024 line, Hashdex's DEFI, is **deliberately excluded**: it was a
# bitcoin *futures* fund that only converted to spot on 2024-03-27, so its first quarter
# is not a spot wrapper and it cannot share the cohort's common window.
EXCLUDED = {"DEFI": "futures fund until its 2024-03-27 spot conversion — not a spot wrapper on the common window"}

# --------------------------------------------------------------------------- #
# The fee schedule — an ASSUMPTION, not tape data
# --------------------------------------------------------------------------- #
# Published sponsor fees in basis points per year, recorded from the funds' fee schedules
# at build time. THIS IS A PROXY / ASSUMPTION: it is not measured on the tape, it carries
# whatever hindsight the current fee sheet carries, and a sponsor can change it. It enters
# ONLY the fee-rank test; the headline cheapest-minus-priciest spread is pure tape and does
# not use it. `strategy.rank_test` is therefore run against both this schedule and the
# waiver-blended variant below, and against rank order alone.
FEE_BPS = {
    "IBIT": 25.0,   # iShares Bitcoin Trust
    "FBTC": 25.0,   # Fidelity Wise Origin Bitcoin Fund
    "ARKB": 21.0,   # ARK 21Shares Bitcoin ETF
    "BITB": 20.0,   # Bitwise Bitcoin ETF
    "HODL": 20.0,   # VanEck Bitcoin ETF
    "BRRR": 25.0,   # CoinShares (ex-Valkyrie) Bitcoin Fund
    "BTCO": 25.0,   # Invesco Galaxy Bitcoin ETF
    "EZBC": 19.0,   # Franklin Bitcoin ETF  <- the fee-cheapest of the cohort
    "BTCW": 25.0,   # WisdomTree Bitcoin Fund
    "GBTC": 150.0,  # Grayscale Bitcoin Trust  <- the fee-priciest of the cohort
    "BTC": 15.0,    # Grayscale Bitcoin Mini Trust (out of cohort, cross-check only)
}

# Launch fee waivers: (fee charged during the waiver, last day of the waiver). Also an
# ASSUMPTION — several waivers were capped by an AUM threshold that the tape cannot see,
# so the stated end date is the *scheduled* one and may overstate how long the waiver ran
# (IBIT's 12 bp introductory fee was capped at the first $5bn, crossed within weeks).
WAIVER = {
    "IBIT": (12.0, "2025-01-10"),
    "FBTC": (0.0, "2024-07-31"),
    "ARKB": (0.0, "2024-07-11"),
    "BITB": (0.0, "2024-07-11"),
    "HODL": (0.0, "2025-03-31"),
    "BRRR": (0.0, "2024-04-11"),
    "BTCO": (0.0, "2024-07-11"),
    "EZBC": (0.0, "2024-08-02"),
    "BTCW": (0.0, "2024-07-11"),
    "GBTC": (150.0, "2024-01-11"),   # never waived a basis point
}

# Study-wide as-of: the last COMPLETE calendar month at build time. Partial months are
# dropped so the monthly estimator never eats a stub and the sample never creeps.
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily closes, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2023-12-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily adjusted closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` so the
    ``close`` column is split- and distribution-adjusted. None of the wrappers
    distributes, so for them adjusted == price; ``BIL`` genuinely compounds its coupon.
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
    """Read cached daily closes OFFLINE into one aligned close frame.

    Returns a frame indexed by date with one column per ticker, sliced to ``asof`` so the
    sample never creeps. Raises ``FileNotFoundError`` if any ticker is missing — the
    offline core and the test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call fee_war.data.fetch() once to populate the cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df[df.index <= pd.Timestamp(asof)]


def cohort_frame(prices: pd.DataFrame, cohort=COHORT, bench: str = BENCH) -> pd.DataFrame:
    """The cohort + benchmark on their common window (every column present, no gaps)."""
    return prices[list(cohort) + [bench]].dropna()


def blended_fee_bps(ticker: str, start, end) -> float:
    """The waiver-blended average fee over ``[start, end]``, in bp/yr — an ASSUMPTION.

    Time-weights the waiver fee before the scheduled waiver end and the headline fee
    afterwards. Where a waiver was capped by an AUM threshold the fund actually crossed
    early, this **understates** what the fund charged; that is why the rank test is also
    run on the unblended headline fee and on rank order alone.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    total = max((end - start).days, 1)
    if ticker not in WAIVER:
        return float(FEE_BPS[ticker])
    wfee, wend = WAIVER[ticker]
    waived = max(min(pd.Timestamp(wend), end) - start, pd.Timedelta(0)).days
    return float((waived * wfee + (total - waived) * FEE_BPS[ticker]) / total)


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted fee ladder)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_years: float = 2.5,
    fees_bps=(19.0, 20.0, 20.0, 21.0, 25.0, 25.0, 25.0, 25.0, 25.0, 150.0),
    bench_vol: float = 0.50,        # annualised vol of the benchmark asset
    bench_drift: float = 0.05,      # annualised drift (irrelevant to a tracking difference)
    clock_stub_bp: float = 135.0,   # daily sd of the 24/7-vs-16:00 clock mismatch, in bp
    premium_innov_bp: float = 5.0,  # daily innovation of each fund's own premium/discount
    premium_phi: float = 0.80,      # AR(1) persistence of that premium
    signal_strength: float = 1.0,   # 0 = every fund charges the cohort-average fee (null)
    start: str = "2024-01-11",
    seed: int = 959,
    cash_rate_ann: float = 0.045,
) -> tuple[pd.DataFrame, dict]:
    """A planted cohort of wrappers on one asset, each shaved by its own fee.

    Construction, mirroring the real measurement problem exactly:

    - a benchmark log price random walk (the *true* asset);
    - each fund's log price = benchmark - fee_i * t + an AR(1) premium/discount;
    - the **observed** benchmark carries an iid mean-zero **clock stub** of
      ``clock_stub_bp`` per day (bitcoin's 24/7 tape read against a 16:00 ETF strike),
      which is invisible in any fund-minus-fund difference and enormous in any
      fund-minus-benchmark one.

    The ``signal_strength`` knob shrinks the planted fee dispersion toward its mean:

    - ``signal_strength = 1`` -> the full planted ladder; the fee rank *is* the tracking
      difference rank, and the estimators must recover it.
    - ``signal_strength = 0`` -> every fund charges the same average fee; there is no
      cross-sectional signal, so the rank test and the pair spread must stay quiet.

    Returns ``(prices, truth)``. ``prices`` has one column per fund (``F00``..), a
    ``bench`` column (the *observed*, clock-stubbed benchmark) and a ``cash`` accrual leg.
    Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(round(n_years * TRADING_DAYS_PER_YEAR))
    # OOB-safe: a few thousand business days stays well inside pandas' ns Timestamp range.
    dates = pd.bdate_range(start=start, periods=n_days)

    fees = np.asarray(fees_bps, dtype=float)
    fees_eff = (1.0 - signal_strength) * fees.mean() + signal_strength * fees
    n_funds = fees_eff.size

    dt = 1.0 / TRADING_DAYS_PER_YEAR
    d_bench = (bench_drift - 0.5 * bench_vol ** 2) * dt
    s_bench = bench_vol * np.sqrt(dt)
    true_log = np.cumsum(rng.normal(d_bench, s_bench, n_days)) + np.log(50000.0)

    # Each fund: benchmark, minus its fee accrued daily, plus its own AR(1) premium.
    t_years = np.arange(n_days, dtype=float) * dt
    cols = {}
    prem_sd = premium_innov_bp * 1e-4
    for i in range(n_funds):
        eps = rng.normal(0.0, prem_sd, n_days)
        prem = np.empty(n_days)
        prem[0] = eps[0]
        for k in range(1, n_days):
            prem[k] = premium_phi * prem[k - 1] + eps[k]
        cols[f"F{i:02d}"] = np.exp(true_log - fees_eff[i] * 1e-4 * t_years + prem)

    # The observed benchmark: the truth plus an iid clock stub (mean zero, big).
    stub = rng.normal(0.0, clock_stub_bp * 1e-4, n_days)
    cols["bench"] = np.exp(true_log + stub)

    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cols["cash"] = np.cumprod(np.full(n_days, cash_daily))

    prices = pd.DataFrame(cols, index=pd.DatetimeIndex(dates, name="date"))
    truth = {
        "fees_bps": fees.tolist(),
        "fees_eff_bps": fees_eff.tolist(),
        "fund_cols": [f"F{i:02d}" for i in range(n_funds)],
        "signal_strength": signal_strength,
        "n_funds": int(n_funds),
        "n_days": int(n_days),
        "n_years": float(n_years),
        "bench_vol": bench_vol,
        "clock_stub_bp": clock_stub_bp,
        "premium_innov_bp": premium_innov_bp,
        "premium_phi": premium_phi,
        "seed": seed,
        "cash_rate_ann": cash_rate_ann,
        "fee_spread_bps": float(fees_eff.max() - fees_eff.min()),
    }
    return prices, truth


def synthetic_daily(**kwargs) -> tuple[pd.DataFrame, dict]:
    """Two-fund convenience wrapper over :func:`synthetic_panel` (cheap vs dear)."""
    kwargs.setdefault("fees_bps", (20.0, 150.0))
    return synthetic_panel(**kwargs)
