"""Data layer for Study 984 — A Dollar Off.

This study cannot use the desk's usual total-return tape, because dividend adjustment is
exactly the thing under measurement. An ``auto_adjust=True`` close has already had the ex-day
drop removed by construction; measuring the drop in it would return 1.000 every time and mean
nothing.

So this module keeps its own cache (prefix ``exday_``, never colliding with the desk's
``prices_`` parquets) holding two columns per ticker:

- ``close`` — the **raw, split-adjusted but dividend-unadjusted** close, i.e. what actually
  printed.
- ``dividend`` — the cash amount that went ex on that date, zero on every other date, from
  Yahoo's corporate-actions feed.

Both come from one ``yfinance`` call with ``auto_adjust=False, actions=True``. ``fetch`` is the
only thing here that touches the network; ``load_bars`` reads the cache offline.

A caveat that belongs in the data layer rather than the results: Yahoo's dividend feed is
**not** a research-grade corporate-actions database. It occasionally misses a special dividend,
occasionally dates one to the pay date rather than the ex-date, and its split back-adjustment
of historical dividends can be off by a factor for very old records. The study restricts itself
to twelve mega-cap payers since 2005 precisely because that is the region where the feed is most
reliable, and ``sanity_report`` prints the checks that were run on it.
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

# Large, liquid, reliable payers across four decades of dividend policy, plus the
# market itself, which every ex-day return has to be corrected for.
PAYERS = ("KO", "JNJ", "PG", "XOM", "T", "VZ", "MO", "PFE", "IBM", "CVX", "MMM", "MCD")
MARKET = "SPY"
TICKERS = PAYERS + (MARKET,)

AS_OF = "2026-06-30"
START = "2005-01-01"

# Elton & Gruber's 1970 estimate, kept here so the synthetic generator can plant it.
ELTON_GRUBER_DEFAULT = 0.778


def _cache_path(ticker: str, cache_dir: str) -> str:
    """Own prefix: these bars are deliberately NOT dividend-adjusted, unlike every
    ``prices_`` parquet on the desk, and confusing the two would silently invert this
    study's result."""
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"exday_{safe}_1d.parquet")


def fetch(tickers=TICKERS, start: str = START, end: str | None = None,
          cache_dir: str = DEFAULT_CACHE, retries: int = 4) -> dict[str, pd.DataFrame]:
    """Download raw closes plus the corporate-actions dividend column (network-only)."""
    import yfinance as yf  # lazy: only when we actually go to the network

    out: dict[str, pd.DataFrame] = {}
    os.makedirs(cache_dir, exist_ok=True)
    for tk in tickers:
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(tk, start=start, end=end, interval="1d",
                                  auto_adjust=False, actions=True, progress=False)
                if raw is not None and len(raw) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"yfinance returned no data for {tk}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=lambda c: str(c).lower().replace(" ", "_"))
        df = pd.DataFrame(index=pd.to_datetime(raw.index))
        df["close"] = pd.to_numeric(raw["close"], errors="coerce")
        df["dividend"] = pd.to_numeric(raw.get("dividends", 0.0), errors="coerce").fillna(0.0)
        df.index.name = "date"
        df = df.dropna(subset=["close"])
        df.to_parquet(_cache_path(tk, cache_dir))
        out[tk] = df
    return out


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every ticker's parquet is present in the shared cache (offline-testable)."""
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_bars(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE,
              asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    """Read the cached raw-close + dividend frames OFFLINE, one per ticker."""
    out = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached bars for {tk} at {path}. "
                f"Call exday.data.fetch() once to populate the shared cache."
            )
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        out[tk] = df[df.index <= pd.Timestamp(asof)].sort_index()
    return out


def load_prices(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE,
                asof: str = AS_OF) -> pd.DataFrame:
    """The raw closes as one aligned panel.

    Named ``load_prices`` for consistency with every other study on the desk, but note the
    difference: these closes are **not** dividend-adjusted. Anything that treats them as a
    total-return series will understate every payer's return by its yield.
    """
    bars = load_bars(tickers, cache_dir, asof)
    df = pd.DataFrame({tk: b["close"] for tk, b in bars.items()}).sort_index()
    df.index.name = "date"
    return df


def load_dividends(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE,
                   asof: str = AS_OF) -> pd.DataFrame:
    """The dividend column as one aligned panel; zero on non-ex dates."""
    bars = load_bars(tickers, cache_dir, asof)
    df = pd.DataFrame({tk: b["dividend"] for tk, b in bars.items()}).sort_index().fillna(0.0)
    df.index.name = "date"
    return df


def sanity_report(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE,
                  asof: str = AS_OF) -> pd.DataFrame:
    """What the corporate-actions feed actually contains, per ticker.

    Printed before any result, because a dividend feed that misses events or mis-dates them
    produces a drop ratio that is wrong in a direction nobody can guess.
    """
    bars = load_bars(tickers, cache_dir, asof)
    rows = []
    for tk, b in bars.items():
        d = b["dividend"]
        ex = d[d > 0]
        yields = (ex / b["close"].reindex(ex.index)).dropna()
        gaps = ex.index.to_series().diff().dt.days.dropna()
        rows.append({
            "ticker": tk,
            "sessions": len(b),
            "ex_days": int(len(ex)),
            "first_ex": str(ex.index[0].date()) if len(ex) else "-",
            "last_ex": str(ex.index[-1].date()) if len(ex) else "-",
            "median_yield": float(yields.median()) if len(yields) else np.nan,
            "max_yield": float(yields.max()) if len(yields) else np.nan,
            "median_gap_days": float(gaps.median()) if len(gaps) else np.nan,
            "suspicious_gaps": int(((gaps < 45) | (gaps > 200)).sum()) if len(gaps) else 0,
        })
    return pd.DataFrame(rows).set_index("ticker")


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


def synthetic_panel(n: int = 2500, n_tickers: int = 8, drop_fraction: float = 1.0,
                    quarterly_yield: float = 0.007, daily_vol: float = 0.012,
                    market_beta: float = 1.0, seed: int = 984) -> dict:
    """A tape with dividends whose ex-day drop is a KNOWN fraction of the dividend.

    ``drop_fraction`` is the truth the estimators are trying to recover: at 1.0 the price falls
    by exactly the dividend, at 0.78 by Elton and Gruber's famous number, at 0.0 not at all.

    The realism that matters is the ratio of the two scales: a quarterly dividend is about
    ``quarterly_yield`` of the price, while a single day's move is about ``daily_vol``. With the
    defaults that is 0.7% of signal against 1.2% of noise **per event** — which is why this
    generator doubles as the study's power analysis.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n)
    mkt_rets = rng.normal(0.0003, 0.009, n)
    market = pd.Series(300 * np.exp(np.cumsum(mkt_rets)), index=idx)
    bars = {}
    for k in range(n_tickers):
        ex_positions = set(range(20 + k * 3, n, 63))
        price = np.zeros(n)
        div = np.zeros(n)
        p = 50.0 + 10 * k
        idio = rng.normal(0, np.sqrt(max(daily_vol ** 2 - (market_beta * 0.009) ** 2, 1e-8)), n)
        for t in range(n):
            r = market_beta * mkt_rets[t] + idio[t]
            p *= (1 + r)
            if t in ex_positions:
                d = quarterly_yield * p
                p -= drop_fraction * d
                div[t] = d
            price[t] = p
        bars[f"SIM{k}"] = pd.DataFrame({"close": price, "dividend": div}, index=idx)
    bars["MKT"] = pd.DataFrame({"close": market.to_numpy(), "dividend": np.zeros(n)}, index=idx)
    return bars
