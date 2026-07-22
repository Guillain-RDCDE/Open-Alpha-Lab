"""Data layer for Study 802 — Stock-Split-Modern.

The question: does the classic post-split *drift* still show up in the **post-2020
mega-cap era** (Tesla, Nvidia, Amazon, Alphabet), or is it just a handful of
high-variance single names riding a bull market?

Three ingredients, all offline once cached:

* **Real split calendar.** ``yfinance`` ``.splits`` (the "Stock Splits" action series)
  on a basket of large-cap US names with split activity since ~2010. We keep only
  **forward** splits with ratio ≥ 1.5 (reverse splits are Study 250's turf).

* **Real prices + a market benchmark.** Split-and-dividend-adjusted daily closes
  (``auto_adjust=True`` — total return) for every name **and for SPY**, so every
  post-event return can be made **market-adjusted** (subtract the market's move over
  the identical window). Without this de-trending, "the stock went up after its split"
  is indistinguishable from "*everything* went up 2020-2024".

* **Synthetic world** — a deterministic, seeded generator that builds raw stock and
  market price paths plus synthetic split dates, with a TUNABLE planted **abnormal**
  post-split drift (knob ``planted_bps``, extra stock-only daily return in the
  post-event window). ``planted_bps = 0`` is the null: market-adjusted CARs are
  centred on zero and the detector must NOT manufacture significance. The synthetic
  tape flows through the *same* ``strategy.build_event_panel`` as the real tape, so the
  control exercises the market-adjustment machinery end to end.

**Honest caveats, stated up front (and again in the notebooks):**

1. **Effective date, not announcement date.** ``yfinance`` reports the *ex-split*
   (effective) date, which trails the *announcement* by ~3-6 weeks. The original
   Ikenberry-Rankine-Stice (1996) drift is measured from *announcement*; we test a
   strictly weaker, post-effective window (same caveat as sibling Study 142).
2. **Survivorship / selection on the Signal axis.** This basket is *today's* winners
   that happened to split. Picking Tesla/Nvidia/Amazon/Alphabet because they are now
   giants selects names that already went up — a bias that points *for* the drift. The
   market-adjustment removes the market-wide bull run but NOT the name-selection bias;
   we name it loudly rather than launder it.
3. **Tiny N, huge idiosyncratic variance.** The modern mega-cap era carries a
   single-digit event count; a couple of high-beta names dominate the mean. This is the
   whole reason the honest verdict is likely "noise, not signal".

Pure numpy + pandas on the offline path. ``fetch()`` (network) runs once to build the
cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))

PRICES_CACHE = os.path.join(CACHE_DIR, "ssm_prices.parquet")   # wide: cols=tickers, +SPY
SPLITS_CACHE = os.path.join(CACHE_DIR, "ssm_splits.parquet")   # long: ticker,date,ratio

START = "2010-01-01"        # earliest split effective date we consider
AS_OF = "2026-06-30"        # last complete calendar month at publication
FETCH_END = "2026-07-01"
ERA_SPLIT = "2020-01-01"    # pre / post the mega-cap split wave (TSLA/AAPL 2020)
MARKET = "SPY"              # total-return market benchmark for abnormal returns

# The brief's named modern mega-caps.
MEGA_CAPS = ["TSLA", "NVDA", "AMZN", "GOOGL", "AAPL"]

# A broad large-cap basket with real forward-split activity since ~2010. This is a
# CURRENT-MEMBERSHIP list of today's large-caps — a survivorship/selection choice we
# name on the Signal axis (see module docstring, caveat 2).
LARGECAP_BASKET = [
    "AAPL", "TSLA", "NVDA", "AMZN", "GOOGL", "GOOG", "AVGO", "AMD",
    "NFLX", "SHOP", "PANW", "FTNT", "DXCM", "ISRG", "LRCX", "CSX",
    "WMT", "CMG", "MELI", "V", "MA", "NKE", "SBUX", "CTAS", "MNST",
    "TSCO", "ODFL", "UNP", "HD", "MSFT",
]


# --------------------------------------------------------------------------- #
# Real tape — network fetch, cached once
# --------------------------------------------------------------------------- #
def fetch(tickers: list[str] | None = None, start: str = "2005-01-01",
          end: str = FETCH_END, min_ratio: float = 1.5) -> None:
    """Download the split calendar + total-return daily closes for the basket and the
    market benchmark; cache both as parquet. Network; run once."""
    import yfinance as yf

    if tickers is None:
        tickers = LARGECAP_BASKET
    os.makedirs(CACHE_DIR, exist_ok=True)

    price_cols: dict[str, pd.Series] = {}
    split_rows: list[dict] = []
    all_names = list(dict.fromkeys(tickers + [MARKET]))

    for tkr in all_names:
        raw = yf.download(tkr, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if raw.empty or "Close" not in raw.columns:
            continue
        s = raw["Close"].copy()
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        price_cols[tkr] = s

        if tkr == MARKET:
            continue
        sp = yf.Ticker(tkr).splits
        if isinstance(sp, pd.Series) and not sp.empty:
            for dt, ratio in sp.items():
                d = pd.Timestamp(dt).tz_localize(None).normalize()
                if float(ratio) >= min_ratio:
                    split_rows.append({"ticker": tkr, "date": d, "ratio": float(ratio)})

    prices = pd.DataFrame(price_cols).sort_index()
    prices.index.name = "date"
    prices.to_parquet(PRICES_CACHE)

    splits = pd.DataFrame(split_rows, columns=["ticker", "date", "ratio"])
    splits = splits.sort_values(["date", "ticker"]).reset_index(drop=True)
    splits.to_parquet(SPLITS_CACHE)


def have_real() -> bool:
    return os.path.exists(PRICES_CACHE) and os.path.exists(SPLITS_CACHE)


def load_prices() -> pd.DataFrame:
    """Cached wide total-return price panel (columns = tickers incl. SPY)."""
    df = pd.read_parquet(PRICES_CACHE)
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


def load_splits(start: str = START, asof: str = AS_OF,
                min_ratio: float = 1.5) -> pd.DataFrame:
    """Cached forward-split calendar, sliced to [start, asof] and ratio ≥ min_ratio."""
    sp = pd.read_parquet(SPLITS_CACHE)
    sp["date"] = pd.to_datetime(sp["date"])
    m = (sp["date"] >= pd.Timestamp(start)) & (sp["date"] <= pd.Timestamp(asof)) \
        & (sp["ratio"] >= min_ratio)
    return sp.loc[m].sort_values(["date", "ticker"]).reset_index(drop=True)


def fingerprint_splits(splits: pd.DataFrame) -> str:
    """Short content hash of the event calendar (ticker+date+ratio)."""
    h = hashlib.sha256()
    for _, r in splits.sort_values(["date", "ticker"]).iterrows():
        h.update(str(r["ticker"]).encode())
        h.update(str(pd.Timestamp(r["date"]).date()).encode())
        h.update(f"{float(r['ratio']):.4f}".encode())
        h.update(b"|")
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic world — planted ABNORMAL post-split drift (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(planted_bps: float = 0.0, seed: int = 802,
                    n_stocks: int = 24, splits_per_stock: int = 2,
                    n_days: int = 2600, beta: float = 1.0,
                    mkt_vol: float = 0.010, idio_vol: float = 0.020,
                    mkt_drift: float = 0.30 / 252,
                    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Deterministic raw stock + market price paths with synthetic split dates and a
    TUNABLE planted **abnormal** post-split drift.

    Each stock return = ``beta * market_return + idiosyncratic`` (default ``beta = 1``
    so that simple market-adjustment leaves a *clean* null — a beta ≠ 1 would leave a
    residual beta tilt that biases the mean, which is a feature of the real estimator
    but muddies a positive control). For each planted split, ``planted_bps`` of extra
    *stock-only* daily return is added across the 252 trading days after the ex-date —
    so it survives market-adjustment. At ``planted_bps = 0`` the market-adjusted CARs
    are centred on zero: the detector must NOT fire.

    Returns ``(prices_wide, market_series, splits_df)`` with exactly the shape the
    real loaders produce, so ``strategy.build_event_panel`` processes it unchanged.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2011-01-03", periods=n_days, name="date")
    n = len(idx)
    extra = planted_bps * 1e-4

    mkt_ret = rng.normal(mkt_drift, mkt_vol, n)
    mkt_close = 100.0 * np.exp(np.cumsum(mkt_ret))
    market = pd.Series(mkt_close, index=idx, name=MARKET)

    price_cols: dict[str, pd.Series] = {}
    split_rows: list[dict] = []
    horizon = 252
    for si in range(n_stocks):
        stock_ret = beta * mkt_ret + rng.normal(0.0, idio_vol, n)
        # plant events on this stock
        ev_positions = sorted(rng.choice(np.arange(260, n - horizon - 5),
                                         size=splits_per_stock, replace=False))
        for pos in ev_positions:
            stock_ret[pos + 1: pos + 1 + horizon] += extra
            split_rows.append({"ticker": f"SYN{si:02d}",
                               "date": idx[pos], "ratio": float(rng.choice([2.0, 3.0, 4.0, 10.0]))})
        price_cols[f"SYN{si:02d}"] = pd.Series(100.0 * np.exp(np.cumsum(stock_ret)), index=idx)

    prices = pd.DataFrame(price_cols)
    prices[MARKET] = market
    prices.index.name = "date"
    splits = pd.DataFrame(split_rows, columns=["ticker", "date", "ratio"]) \
        .sort_values(["date", "ticker"]).reset_index(drop=True)
    return prices, market, splits
