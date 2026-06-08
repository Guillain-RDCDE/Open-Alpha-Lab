# Feed provenance — wsb_mentions.csv

Derived by [`examples/build_wsb_feed.py`](../examples/build_wsb_feed.py) from
**[youyanggu/yolostocks-data](https://github.com/youyanggu/yolostocks-data)** —
daily r/WallStreetBets mention counts, top-100 tickers, **CC-BY-4.0**.

Event = a *viral surge*: a session with count >= **100** mentions AND
>= **4.0x** its trailing **60**-session median, debounced to one event per
name per **21** calendar days. Non-equity tickers (crypto, slang) dropped.

Result: 1,705 events on 224 tickers, 2021-01-11 to 2025-12-29.

The raw source CSVs are cached under `../_rawfeed/` (CC-BY-4.0, © youyanggu). This derived feed inherits that licence.
