"""One-shot script to fetch and cache all price data for Study 211."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sin_stocks import data

CACHE = os.path.join(os.path.dirname(__file__), "_cache")

for t in data.ALL_TICKERS:
    print(f"Fetching {t}...")
    df = data.fetch_prices(t, start=data.START_DATE, fetch=True, cache_dir=CACHE)
    print(f"  {t}: {df.index[0].date()} -> {df.index[-1].date()} n={len(df)}")
print("Done.")
