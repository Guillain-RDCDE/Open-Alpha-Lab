"""Fetch real order flow from Binance and build the trades.parquet the empirical leg needs.

Downloads one day of **futures (USD-M)** aggregated trades and best-bid/offer (`bookTicker`)
from the free, no-key public archive at https://data.binance.vision/, joins each trade to the
prevailing order-book mid by an as-of merge on the exchange timestamp, and writes the three
columns [`price`, `qty`, `mid`] to ``_cache/trades.parquet``. Then run
``examples/verify_real.py`` to fit the arrival kernel on it.

    python examples/fetch_binance.py            # default: TRXUSDT, 2024-01-15
    python examples/fetch_binance.py ETHUSDT 2024-03-01

Futures (not spot) because only the futures archive publishes daily `bookTicker` — the best
bid/ask we need to define the mid at each trade. A modestly-liquid symbol keeps the book file
small (TRXUSDT/day ~ 12 MB) while still giving millions of fills.
"""

import os
import sys
import urllib.request as u

import pandas as pd

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(_STUDY, "_cache")
BASE = "https://data.binance.vision/data/futures/um/daily"


def _download(kind: str, sym: str, day: str) -> str:
    os.makedirs(CACHE, exist_ok=True)
    dst = os.path.join(CACHE, f"{sym}-{kind}-{day}.zip")
    if not os.path.exists(dst):
        url = f"{BASE}/{kind}/{sym}/{sym}-{kind}-{day}.zip"
        print("downloading", url)
        u.urlretrieve(url, dst)
    print(f"  {kind}: {os.path.getsize(dst) / 1e6:.1f} MB")
    return dst


def build_trades(sym: str = "TRXUSDT", day: str = "2024-01-15") -> pd.DataFrame:
    """Download (if needed) and join one day of futures trades to the live book mid.

    Returns a frame with [`price`, `qty`, `mid`] — every executed trade tagged with the
    prevailing best-bid/offer midpoint at (or just before) its exchange timestamp.
    """
    agg_zip = _download("aggTrades", sym, day)
    book_zip = _download("bookTicker", sym, day)

    trades = pd.read_csv(agg_zip, compression="zip",
                         usecols=["price", "quantity", "transact_time"])
    trades = trades.rename(columns={"quantity": "qty", "transact_time": "t"}).sort_values("t")

    book = pd.read_csv(book_zip, compression="zip",
                       usecols=["best_bid_price", "best_ask_price", "transaction_time"])
    book["mid"] = 0.5 * (book["best_bid_price"] + book["best_ask_price"])
    book = book.rename(columns={"transaction_time": "t"}).sort_values("t")[["t", "mid"]]

    # As-of join: the prevailing best-bid/offer mid at (or just before) each trade.
    merged = pd.merge_asof(trades, book, on="t", direction="backward").dropna(subset=["mid"])
    return merged[["price", "qty", "mid"]].reset_index(drop=True)


def main(sym: str = "TRXUSDT", day: str = "2024-01-15"):
    print(f"building trades.parquet from Binance futures {sym} {day}")
    out = build_trades(sym, day)
    dst = os.path.join(CACHE, "trades.parquet")
    out.to_parquet(dst)
    print(f"wrote {dst}: {len(out):,} trades joined to a live mid")
    print(out.head().to_string())


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args)
