"""Data layer for Study 942 — The Inverse Tax.

Two tapes, one shape (a date-indexed daily frame of closes):

- ``fetch`` / ``load_prices`` — the real tape from Yahoo! Finance (``yfinance``), read from
  the **shared** cache at ``studies/_cache``:

  * ``prices_<TICKER>_1d.parquet`` — daily **total-return** closes (``auto_adjust=True``)
    for the inverse funds (SH = −1× S&P 500, PSQ = −1× Nasdaq-100, SDS = −2× S&P 500),
    their underlyings (SPY, QQQ), the tradable cash leg (BIL) and the 13-week bill
    **rate level** (``^IRX``, quoted in percent, cached as ``prices_IRX_1d.parquet``).
  * ``praw_<TICKER>_1d.parquet`` — daily **price-only** closes (``auto_adjust=False``) for
    SPY and QQQ. These exist because the distinction matters here and nowhere else on the
    desk: a share-short **owes the dividends**, whereas the ProShares inverse funds track
    the **price** index, so their holder never pays them. Everything is labelled
    price-only vs total-return at the point of use.

  ``fetch`` touches the network and populates the cache (retry up to 4×); ``load_prices``
  reads that cache **offline** and never imports yfinance. The whole test-suite runs with
  NO cache present (synthetic-only), so CI is green on a fresh checkout.

- ``synthetic_daily`` / ``synthetic_panel`` — *deterministic, offline* generators. They
  build an index tape, a bill-rate path, and a **simulated inverse fund** whose daily
  return is the constant-leverage identity plus a *planted* structural tax. The
  ``signal_strength`` knob scales that planted tax: at ``signal_strength=1`` the fund
  really is the worse short by a known amount the harness must recover; at
  ``signal_strength=0`` the fund is a costless replicate and the harness must stay quiet.

No look-ahead is baked in here — the single execution-lag convention lives in
``strategy.prepare`` (the exposure set at the close of day ``t-1`` earns day ``t``'s
return; the bill rate financed at is the previous close's ``^IRX`` level).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The SHARED desk cache, not a study-local one.
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# Total-return tickers. ^IRX is a rate *level* in percent, not a price.
TICKERS = ("SH", "PSQ", "SDS", "SPY", "QQQ", "BIL", "^IRX")
# Price-only (auto_adjust=False) closes — only where the dividend distinction bites.
PRICE_ONLY = ("SPY", "QQQ")

# The three inverse funds under test: (underlying, target daily multiple).
FUNDS = {"SH": ("SPY", -1.0), "PSQ": ("QQQ", -1.0), "SDS": ("SPY", -2.0)}

# PROXY / ASSUMPTION (not tape): prospectus gross expense ratios, %/yr. Used only in the
# accounting decomposition, never in the headline gap, and reported as a stated input.
EXPENSE_RATIO_PCT = {"SH": 0.89, "PSQ": 0.95, "SDS": 0.89}

# Study-wide as-of: the last COMPLETE calendar month at build time.
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape — shared cache, offline by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str, prefix: str = "prices") -> str:
    """Shared-cache filename convention: '^' and '=' are stripped from the ticker."""
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"{prefix}_{safe}_1d.parquet")


def _download(tk: str, start: str, end, auto_adjust: bool, retries: int) -> pd.DataFrame:
    import yfinance as yf  # lazy: only when we actually go to the network

    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(tk, start=start, end=end, interval="1d",
                              auto_adjust=auto_adjust, progress=False)
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
    return df.dropna(subset=["close"])


def fetch(
    tickers=TICKERS,
    price_only=PRICE_ONLY,
    start: str = "2004-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download the tape and populate the shared cache. Network-only; run once.

    Total-return closes (``auto_adjust=True``) go to ``prices_*``; the price-only closes
    of SPY/QQQ (``auto_adjust=False``) go to ``praw_*``. ^IRX has no dividends, so its
    ``prices_IRX_1d.parquet`` is simply the rate level in percent.
    """
    out: dict[str, pd.DataFrame] = {}
    os.makedirs(cache_dir, exist_ok=True)
    for tk in tickers:
        df = _download(tk, start, end, True, retries)
        df.to_parquet(_cache_path(tk, cache_dir, "prices"))
        out[tk] = df
    for tk in price_only:
        df = _download(tk, start, end, False, retries)
        df.to_parquet(_cache_path(tk, cache_dir, "praw"))
        out[tk + "_px"] = df
    return out


def have_real(tickers=TICKERS, price_only=PRICE_ONLY, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every parquet this study needs is present in the shared cache."""
    ok = all(os.path.exists(_cache_path(tk, cache_dir, "prices")) for tk in tickers)
    return ok and all(os.path.exists(_cache_path(tk, cache_dir, "praw")) for tk in price_only)


def load_prices(
    tickers=TICKERS,
    price_only=PRICE_ONLY,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
) -> pd.DataFrame:
    """Read the cached closes OFFLINE into one aligned frame, sliced to ``asof``.

    Columns: one per total-return ticker (``^IRX`` becomes ``IRX``) plus ``SPY_px`` /
    ``QQQ_px`` for the price-only closes. Raises ``FileNotFoundError`` on a cache miss —
    the offline core and the test-suite never touch the network.
    """
    cols: dict[str, pd.Series] = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir, "prices")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call inverse_tax.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk.replace("^", "")] = s
    for tk in price_only:
        path = _cache_path(tk, cache_dir, "praw")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached price-only closes for {tk} at {path}. "
                f"Call inverse_tax.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk + "_px"] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df[df.index <= pd.Timestamp(asof)]


def cash_rate_daily(irx_pct: pd.Series) -> pd.Series:
    """Convert the ^IRX quote (13-week bill discount rate, percent) to a per-bar rate.

    Money-market convention: an annualised rate on an **act/360** basis accrued over the
    *calendar* days between consecutive observations, so a Monday bar earns three days of
    carry exactly as the weekend equity return spans three calendar days. The **previous**
    close's level is the rate actually financed at — this is the study's single execution
    lag applied to the rate leg.

    ^IRX is a *discount* quote rather than a bond-equivalent yield; the two differ by a few
    bps over this sample, far inside the 0-100 bps borrow sweep and the 0-2× rebate-credit
    sweep the study runs anyway. ``strategy.financing_crosscheck`` compares this
    construction against BIL's realised total return, the tradable cash leg.
    """
    r = pd.Series(irx_pct).astype(float).sort_index()
    days = pd.Series(r.index, index=r.index).diff().dt.days.astype(float)
    return (r.shift(1) / 100.0) * days / 360.0


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a frame, for the as-of data stamp.

    Taken over daily **returns**, not levels. ``auto_adjust=True`` closes are rescaled by a
    constant factor every time the provider books a new distribution, which changes every
    historical level while leaving every return — and therefore every number in this study
    — untouched. Hashing levels would make the stamp churn on refetches that change
    nothing; hashing returns makes it reproducible.
    """
    arr = np.ascontiguousarray(prices.pct_change().to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.round(arr, 10)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a PLANTED structural tax)
# --------------------------------------------------------------------------- #
def synthetic_daily(
    n_years: int = 25,
    k: float = -1.0,                  # the fund's target daily multiple
    tax_ann_full: float = 0.03,       # the planted structural tax, %/yr as a fraction
    div_yield_ann: float = 0.0,       # index dividend yield (TR minus price return)
    passthrough: float = 1.0,         # units of the bill rate the fund credits on NAV
    vol_ann: float = 0.18,            # annualised index vol
    drift_ann: float = 0.08,          # annualised index total-return drift
    rate_ann: float = 0.03,           # flat bill rate level
    tracking_bps: float = 4.0,        # daily idiosyncratic tracking noise of the fund
    signal_strength: float = 1.0,     # 0 = costless replicate (null), 1 = full planted tax
    start: str = "2006-01-03",
    seed: int = 942,
) -> tuple[pd.DataFrame, dict]:
    """A daily tape carrying an index, a bill rate and a *simulated* inverse fund.

    The fund's daily return is the constant-leverage identity applied to the **price**
    return of the index, plus the financing it credits on NAV, minus the planted tax:

    ``r_fund = k * r_price + passthrough * |k| * rf - tax/252 + noise``

    The honest direct-short replicate the study races it against is, at a rebate credit of
    ``c`` and zero borrow, ``k * r_total + c * |k| * rf``. Since
    ``r_total = r_price + div/252``, the expected annualised gap is

    ``(passthrough - c) * |k| * rate_ann + |k| * div_yield_ann - tax``

    which ``truth["expected_gap_ann"]`` records at ``c = 1``. ``signal_strength`` scales
    the planted tax only: at 0 the fund is a costless replicate (with the default zero
    dividend yield and unit passthrough the expected gap is exactly 0 — the null), at 1 it
    really is the worse short by ``tax_ann_full``. Deterministic given ``seed``.

    Returns ``(prices, truth)`` where ``prices`` has columns ``index_tr`` (total-return
    close), ``index_px`` (price-only close), ``fund`` (the inverse fund's NAV),
    ``cash`` (a bill accrual index, the BIL stand-in) and ``rate_pct`` (the ^IRX-style
    annualised level in percent).
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: bdate_range with a modest period count stays well inside pandas' ns range.
    dates = pd.bdate_range(start=start, periods=n_days)

    tax = float(signal_strength) * float(tax_ann_full)
    mu = drift_ann / TRADING_DAYS_PER_YEAR
    sd = vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)

    r_tr = rng.normal(mu, sd, n_days)
    r_px = r_tr - div_yield_ann / TRADING_DAYS_PER_YEAR

    # act/360 accrual over calendar days, so the fund and the replicate finance identically.
    days = np.empty(n_days)
    days[0] = 1.0
    days[1:] = np.diff(dates.values).astype("timedelta64[D]").astype(float)
    rf = (rate_ann / 360.0) * days

    noise = rng.normal(0.0, tracking_bps * 1e-4, n_days)
    r_fund = k * r_px + passthrough * abs(k) * rf - tax / TRADING_DAYS_PER_YEAR + noise

    prices = pd.DataFrame(
        {
            "index_tr": 100.0 * np.cumprod(1.0 + r_tr),
            "index_px": 100.0 * np.cumprod(1.0 + r_px),
            "fund": 100.0 * np.cumprod(1.0 + r_fund),
            "cash": 100.0 * np.cumprod(1.0 + rf),
            "rate_pct": np.full(n_days, rate_ann * 100.0),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    expected_gap = (
        (passthrough - 1.0) * abs(k) * rate_ann + abs(k) * div_yield_ann - tax
    )
    truth = {
        "k": k, "tax_ann": tax, "tax_ann_full": tax_ann_full,
        "div_yield_ann": div_yield_ann, "passthrough": passthrough,
        "vol_ann": vol_ann, "drift_ann": drift_ann, "rate_ann": rate_ann,
        "tracking_bps": tracking_bps, "signal_strength": float(signal_strength),
        "n_years": n_years, "n_days": n_days, "seed": seed,
        "expected_gap_ann": float(expected_gap),
    }
    return prices, truth


def synthetic_panel(
    seeds=(942, 943, 944, 945, 946, 947),
    signal_strength: float = 1.0,
    **kwargs,
) -> list[tuple[pd.DataFrame, dict]]:
    """A deterministic panel of synthetic tapes — one per seed, same planted parameters.

    Used to check that the detector's sampling distribution is centred where the planted
    truth says it should be (and centred on zero under the null), rather than reading a
    single lucky draw.
    """
    return [synthetic_daily(signal_strength=signal_strength, seed=s, **kwargs) for s in seeds]
