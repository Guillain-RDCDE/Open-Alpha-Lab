"""Data layer for Study 935 — Value Averaging (Edleson) vs Dollar-Cost Averaging.

Two tapes, one shape (a date-indexed daily total-return close frame):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the equity sleeve (SPY), the cash leg
  (BIL, the 1-3 month T-bill ETF) and two cross-check sleeves (IEF, a bond
  variant; QQQ, a higher-vol variant). ``fetch`` touches the network and writes
  parquet into the **shared** ``studies/_cache`` (retry up to 4x); ``load_prices``
  reads that cache **offline** and never imports yfinance. The whole test-suite
  runs with NO cache present (synthetic only), so CI is green on a fresh checkout.

- ``synthetic_daily`` / ``synthetic_panel`` — *deterministic, offline* generators.
  The price is a random walk plus a **transitory, mean-reverting** component whose
  amplitude is the ``signal_strength`` knob. Value averaging is, mechanically, a
  contrarian rule: it buys more after the price falls below its value path and
  sells after it rises above. So a mean-reverting tape is exactly the world where
  VA *must* beat DCA (``signal_strength = 1``), and a pure random walk is the null
  where it must not (``signal_strength = 0``). Seeds are fixed → tests are
  deterministic.

Nothing here knows about the trading rule; the one-day execution lag and the cash
buffer live in ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The desk-wide shared cache (studies/_cache), not a per-study one.
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# SPY = the equity sleeve a saver drips into; BIL = the cash leg the buffer earns;
# IEF / QQQ = the two cross-check sleeves (lower and higher volatility).
TICKERS = ("SPY", "BIL", "IEF", "QQQ")

# Study-wide as-of: the last COMPLETE calendar month at build time, so the sample
# never creeps between reruns.
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "1993-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` makes
    the ``close`` column split- and dividend-adjusted total return, which matters on
    both legs here: the equity sleeve's dividends are a third of its long-run return,
    and the cash buffer's whole contribution *is* BIL's yield.
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

    Returns a frame indexed by date with one column per ticker, sliced to ``asof``
    so the sample never creeps. Raises ``FileNotFoundError`` if any ticker is
    missing — the offline core and the test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call value_avg.data.fetch() once to populate the shared cache."
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


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last trading day of each calendar month present in ``index``.

    These are the decision dates a monthly savings plan actually follows. The trade
    itself happens on the *next* trading day (the one execution lag), which is
    applied in ``strategy.py``, not here.
    """
    idx = pd.DatetimeIndex(index).sort_values()
    s = pd.Series(idx, index=idx)
    last = s.groupby([idx.year, idx.month]).max()
    return pd.DatetimeIndex(sorted(last.values))


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted mean-reverting wobble)
# --------------------------------------------------------------------------- #
def synthetic_daily(
    n_years: int = 30,
    drift_ann: float = 0.08,           # annualised drift of the permanent component
    vol_ann: float = 0.16,             # annualised vol of the permanent component
    swing_ann: float = 0.28,           # sd of the transitory (mean-reverting) wobble
    half_life_days: float = 150.0,     # half-life of that wobble
    signal_strength: float = 1.0,      # 0 = pure random walk (null), 1 = full wobble
    start: str = "1994-01-03",
    seed: int = 935,
    cash_rate_ann: float = 0.03,       # cash yield credited to the buffer
) -> tuple[pd.DataFrame, dict]:
    """A daily equity-like total-return tape with a tunable mean-reverting component.

    The log price is ``permanent_t + signal_strength * transitory_t`` where the
    permanent part is a drifting random walk and the transitory part is a
    zero-mean Ornstein-Uhlenbeck wobble with the given half-life and unconditional
    sd. Value averaging is a contrarian rule, so:

    - ``signal_strength = 1`` → a real, exploitable wobble; VA *should* finish
      richer than DCA on a like-for-like committed-capital basis.
    - ``signal_strength = 0`` → a pure random walk; VA has nothing to lean against
      and must show no reliable edge (the null).

    Returns ``(prices, truth)`` with columns ``asset`` (the equity-like close) and
    ``cash`` (a cash accrual index). Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: bdate_range with n <= 10000 daily bars stays well inside pandas' ns range.
    dates = pd.bdate_range(start=start, periods=n_days)

    mu = drift_ann / TRADING_DAYS_PER_YEAR
    sd = vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    perm = np.cumsum(rng.normal(mu, sd, n_days))

    # OU wobble: x_t = phi x_{t-1} + eps, unconditional sd = swing_ann.
    phi = float(0.5 ** (1.0 / max(half_life_days, 1.0)))
    eps_sd = swing_ann * np.sqrt(max(1.0 - phi * phi, 1e-12))
    shocks = rng.normal(0.0, eps_sd, n_days)
    trans = np.empty(n_days)
    x = 0.0
    for i in range(n_days):
        x = phi * x + shocks[i]
        trans[i] = x

    log_px = perm + float(signal_strength) * trans
    asset = 100.0 * np.exp(log_px - log_px[0])
    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cash_idx = np.cumprod(np.full(n_days, cash_daily))

    prices = pd.DataFrame(
        {"asset": asset, "cash": cash_idx},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "signal_strength": float(signal_strength),
        "drift_ann": drift_ann,
        "vol_ann": vol_ann,
        "swing_ann": swing_ann,
        "swing_eff": float(signal_strength) * swing_ann,
        "half_life_days": half_life_days,
        "phi": phi,
        "n_years": n_years,
        "n_days": n_days,
        "seed": seed,
        "cash_rate_ann": cash_rate_ann,
    }
    return prices, truth


def synthetic_panel(
    n_paths: int = 8,
    signal_strength: float = 0.0,
    base_seed: int = 935,
    **kwargs,
) -> list[tuple[pd.DataFrame, dict]]:
    """``n_paths`` independent synthetic tapes at one ``signal_strength``.

    Used for the null sweep: the VA-minus-DCA gap must be centred on zero across
    seeds when there is no wobble to lean against.
    """
    return [
        synthetic_daily(signal_strength=signal_strength, seed=base_seed + k, **kwargs)
        for k in range(n_paths)
    ]
