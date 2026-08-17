"""Data layer for Study 913 — Tracking-Difference Persistence.

Two tapes, one shape (a date-indexed daily total-return close frame):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for two index-fund families: the S&P 500 trackers
  (SPY, IVV, VOO plus the NAV-priced index mutual funds VFIAX, FXAIX, SWPPX) and the
  Nasdaq-100 pair (QQQ, QQQM), with BIL as the cash leg and ^GSPC as a price-only index
  proxy. ``fetch`` touches the network and caches parquet in the **shared** study cache
  (``studies/_cache/prices_<TICKER>_1d.parquet``, retry up to 4x); ``load_prices`` reads
  that cache **offline** and never imports yfinance. The whole test-suite runs with NO
  cache present (synthetic-only), so CI is green on a fresh checkout.

- ``synthetic_panel`` — a *deterministic, offline* generator: a panel of funds tracking a
  common index, each with a **persistent** fee drag and a **transient** tracking-noise
  term. The ``signal_strength`` knob sets how large the cross-sectional fee spread is
  relative to the noise: at ``signal_strength=1`` last year's tracking-difference rank is
  genuinely informative about next year's; at ``signal_strength=0`` every fund carries the
  same fee and the rank is pure noise (the null). Seed is fixed → tests are deterministic.

**SPLG is not in the panel.** The requested tape included SPLG (the 2 bp SPDR Portfolio
S&P 500 ETF), but at build time Yahoo! Finance returns a single stale bar for it
(2026-07-17) and refuses any historical range — a broken source record, not a short
history. Rather than silently substitute a different fund under the same name, SPLG is
declared missing (``UNAVAILABLE``) and named in the README and in ``docs/results.md``.
The three index mutual funds stand in as the low-fee end of the S&P 500 ladder; they are
**NAV-priced**, which is labelled everywhere it matters (no bid-ask spread, no
premium/discount, and no intraday switching).

**Measurement caveat (a PROXY, and the study's central limitation).** A fund's true
tracking difference is measured NAV-to-NAV against the index with dividends reinvested on
the ex-date. What we can measure from a public tape is the calendar-year total return of
the adjusted close, which folds in dividend-*timing* conventions and the closing print.
That adds a transient error term of roughly 10 bp/yr standard deviation — larger than the
fee spread inside the ETF trio itself. The study measures that noise floor explicitly
rather than pretending it away.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the ranking
formed from calendar year ``y`` is acted on one trading day after that year's last close).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The SHARED study cache (studies/_cache), not a per-study one.
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# The two fund families. The first entry of each family is its **leader** (the oldest and
# largest vehicle), used as the family's index proxy where a single reference is needed.
SP500_ETFS = ("SPY", "IVV", "VOO")
SP500_FUNDS = ("SPY", "IVV", "VOO", "VFIAX", "FXAIX", "SWPPX")
NDX_ETFS = ("QQQ", "QQQM")

# Cash leg (excess-of-cash races) and a price-only index proxy (labelled as such).
CASH = "BIL"
INDEX_PROXY = "GSPC"          # ^GSPC — PRICE-ONLY, no dividends. See ``price_only_note``.

TICKERS = tuple(dict.fromkeys(SP500_FUNDS + NDX_ETFS + (CASH, INDEX_PROXY)))

# Requested but unobtainable from the source at build time (see the module docstring).
UNAVAILABLE = ("SPLG",)

# --------------------------------------------------------------------------- #
# NON-TAPE INPUT — expense ratios are an ASSUMPTION, not a measurement, and they carry
# HINDSIGHT.
# Annual net expense ratio in basis points, as published by the issuer at build time
# (2026-06). These are the *current* ratios; several were cut inside the sample (SPLG
# 0.03 -> 0.02, VOO 0.04 -> 0.03, IVV 0.07 -> 0.03, SWPPX 0.09 -> 0.02 in 2017, FXAIX
# 0.015 only since 2019). Applying them to the whole window means the ``cheapest`` rule
# picks its fund with a fee sheet the early years did not have: the *direction* (SPY and
# QQQ have always been the dearest of their families) is safe, but *which* low-fee rung ends
# up lowest is chosen ex post. Two defences, both in the reported run: the study sweeps how
# far the assumed spread can be wrong, and ``strategy.per_fund_gap_vs_leader`` re-asks the
# money question with no fee sheet at all (every non-flagship fund, and their equal-weight
# blend, against the flagship).
# --------------------------------------------------------------------------- #
EXPENSE_RATIO_BPS = {
    "SPY": 9.45,     # SPDR S&P 500 ETF Trust (a UIT — cannot lend or reinvest dividends)
    "IVV": 3.0,      # iShares Core S&P 500
    "VOO": 3.0,      # Vanguard S&P 500 ETF
    "SPLG": 2.0,     # SPDR Portfolio S&P 500 (listed for completeness; no usable tape)
    "VFIAX": 4.0,    # Vanguard 500 Index Admiral (mutual fund, NAV-priced)
    "FXAIX": 1.5,    # Fidelity 500 Index (mutual fund, NAV-priced)
    "SWPPX": 2.0,    # Schwab S&P 500 Index (mutual fund, NAV-priced)
    "QQQ": 20.0,     # Invesco QQQ Trust (a UIT)
    "QQQM": 15.0,    # Invesco Nasdaq-100 ETF
}

# Study-wide as-of: the last COMPLETE calendar month at build time. Calendar-year work
# additionally drops any partial year (see ``strategy.annual_returns``).
AS_OF = "2026-06-30"


def price_only_note() -> str:
    """One-line reminder of what ^GSPC can and cannot say about tracking difference."""
    return (
        "^GSPC is a PRICE-ONLY index: a fund's total return minus ^GSPC's price return is "
        "approximately the dividend yield (+150 to +200 bp/yr), not a tracking difference. "
        "The *level* of TD is therefore not identified from a public tape; only "
        "cross-sectional differences between funds tracking the same index are."
    )


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "1999-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` makes the
    ``close`` column split- and dividend-adjusted total return — which is the whole point
    here: a tracking difference *is* a total-return difference, and comparing price-only
    closes of funds with different distribution calendars would manufacture the effect.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    out: dict[str, pd.DataFrame] = {}
    os.makedirs(cache_dir, exist_ok=True)
    for tk in tickers:
        symbol = "^GSPC" if tk == "GSPC" else tk
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
        df.index = pd.to_datetime(df.index).tz_localize(None)
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
                f"Call td_persist.data.fetch() once to populate the shared cache."
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
# Synthetic tape — the deterministic offline core (a planted fee ladder)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_funds: int = 4,
    n_years: int = 15,
    fee_spread_bps: float = 30.0,     # top-to-bottom annual fee spread when ss = 1
    noise_bps: float = 8.0,           # sd of the transient annual tracking error
    index_vol: float = 0.16,          # annualised vol of the common index leg
    index_drift: float = 0.09,        # annualised drift of the common index leg
    signal_strength: float = 1.0,     # 0 = every fund charges the same fee (the null)
    start: str = "2006-01-02",
    seed: int = 913,
    cash_rate_ann: float = 0.02,
) -> tuple[pd.DataFrame, dict]:
    """A daily panel of index funds sharing one index, differing by fee and noise.

    Each fund's daily return is ``index_return - fee/252 + daily tracking noise``, where
    the fee is **constant across the whole sample** (persistent, and therefore rankable)
    and the noise is i.i.d. within the year (transient, and therefore not). The
    ``signal_strength`` knob scales the fee spread:

    - ``signal_strength = 1`` → a ``fee_spread_bps`` ladder; last year's tracking-difference
      rank genuinely predicts next year's, because it is the fee ladder showing through.
    - ``signal_strength = 0`` → every fund charges the identical mid fee; the annual rank is
      pure tracking noise and *must not* persist (the null).

    Returns ``(prices, truth)`` where ``prices`` has one column per fund (``fund_0`` …,
    ordered cheapest first) plus ``cash``, and ``truth`` records the planted fees.
    Deterministic given ``seed``. The index is generated but not returned as a column: the
    study never needs it, since tracking differences are taken cross-sectionally.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: a few thousand business days stays far inside pandas' ns Timestamp range.
    dates = pd.bdate_range(start=start, periods=n_days)

    mid_fee = 10.0
    half = 0.5 * fee_spread_bps * float(signal_strength)
    fees_bps = np.linspace(mid_fee - half, mid_fee + half, n_funds)  # cheapest first

    idx_daily = rng.normal(
        index_drift / TRADING_DAYS_PER_YEAR,
        index_vol / np.sqrt(TRADING_DAYS_PER_YEAR),
        n_days,
    )
    noise_daily_sd = (noise_bps * 1e-4) / np.sqrt(TRADING_DAYS_PER_YEAR)

    cols = {}
    for j in range(n_funds):
        drag = (fees_bps[j] * 1e-4) / TRADING_DAYS_PER_YEAR
        r = idx_daily - drag + rng.normal(0.0, noise_daily_sd, n_days)
        cols[f"fund_{j}"] = 100.0 * np.cumprod(1.0 + r)

    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cols["cash"] = np.cumprod(np.full(n_days, cash_daily))

    prices = pd.DataFrame(cols, index=pd.DatetimeIndex(dates, name="date"))
    truth = {
        "signal_strength": float(signal_strength),
        "n_funds": int(n_funds),
        "n_years": int(n_years),
        "n_days": int(n_days),
        "seed": int(seed),
        "fees_bps": {f"fund_{j}": float(fees_bps[j]) for j in range(n_funds)},
        "fee_spread_bps_eff": float(fees_bps[-1] - fees_bps[0]),
        "noise_bps": float(noise_bps),
        "cash_rate_ann": float(cash_rate_ann),
        "cheapest": "fund_0",
    }
    return prices, truth


def synthetic_expense_ratios(truth: dict) -> dict:
    """The planted fee ladder in the same shape as ``EXPENSE_RATIO_BPS`` (for the rules)."""
    return dict(truth["fees_bps"])
