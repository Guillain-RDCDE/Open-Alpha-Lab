"""Build a real, reproducible mention feed from public WallStreetBets data.

A single-guru feed (e.g. @aleabitoreddit / "Serenity") can't be redistributed or
reproduced — it lives behind X's auth wall. So the worked **real** dataset for this
study is the next best thing, and arguably the purer test of the same phenomenon:
the *crowd* as the oracle. We use youyanggu/yolostocks-data — daily r/WallStreetBets
mention counts for the top-100 tickers, **CC-BY-4.0**, a stable public archive — and
define an **event** as the day the crowd suddenly fixates on a name:

    a *viral surge* = a session whose mention count is both
      (a) absolutely loud:      count >= FLOOR              (genuinely viral, not noise)
      (b) a jump vs its own base: count >= MULT x trailing-WIN-day median

debounced to one event per name per COOLDOWN calendar days, so a multi-day frenzy is
one episode, not thirty. These thresholds are *decisions, stated in the open* (house
rule); `robustness.py` (clustering bootstrap, name jackknife) and a sensitivity
sweep are what keep them honest.

    python examples/build_wsb_feed.py        # -> _data/wsb_mentions.csv

Output is the canonical feed (`timestamp, ticker, score`) the rest of the study
consumes via `data.load_feed`. Source attribution and the exact parameters are
written alongside in _data/PROVENANCE.md.

Source: https://github.com/youyanggu/yolostocks-data  (CC-BY-4.0)
"""

from __future__ import annotations

import os
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
RAW_DIR = os.path.join(STUDY, "_rawfeed")
DATA_DIR = os.path.join(STUDY, "_data")
YEARS = (2021, 2022, 2023, 2024, 2025)
RAW_BASE = "https://raw.githubusercontent.com/youyanggu/yolostocks-data/main"

# Event definition — stated openly so it can be challenged.
FLOOR = 100        # a session needs >= 100 mentions to count as "viral"
MULT = 4.0         # and >= 4x its own trailing median (a genuine jump, not drift)
WIN = 60           # trailing window (sessions) for the baseline median
COOLDOWN = 21      # calendar days: one frenzy = one event

# Non-equities that ride the WSB board but aren't stock picks (different calendar,
# no Yahoo equity series). Dropped up front; everything else the price join filters.
NON_EQUITY = {
    "BTC", "ETH", "DOGE", "SHIB", "ADA", "XRP", "SOL", "LTC", "USD", "USDT",
    "CRYPTO", "WSB", "YOLO", "CEO", "FD", "FDA", "EV", "AI", "IMO", "DD", "ATH",
    "IPO", "USA", "ALL", "FOR", "ARE", "ANY", "NOW", "ONE", "OUT", "CAN", "GO",
}


def _load_raw() -> pd.DataFrame:
    os.makedirs(RAW_DIR, exist_ok=True)
    frames = []
    for y in YEARS:
        path = os.path.join(RAW_DIR, f"wsb_{y}.csv")
        if not os.path.exists(path):
            urllib.request.urlretrieve(f"{RAW_BASE}/{y}/wallstreetbets_{y}.csv", path)
        df = pd.read_csv(path)
        datecols = [c for c in df.columns if c not in ("ticker", "overall_rank", "total")]
        m = df.melt(id_vars="ticker", value_vars=datecols, var_name="date", value_name="count")
        m["date"] = pd.to_datetime(m["date"], format="%m/%d/%y")
        frames.append(m)
    long = pd.concat(frames, ignore_index=True).dropna()
    return long[long["count"] > 0].sort_values(["ticker", "date"])


def build_feed() -> pd.DataFrame:
    long = _load_raw()
    rows = []
    for tk, g in long.groupby("ticker"):
        if tk in NON_EQUITY:
            continue
        base = g["count"].rolling(WIN, min_periods=10).median()
        hot = (g["count"] >= FLOOR) & (g["count"] >= MULT * base.fillna(1e9))
        last = None
        for d, c in zip(g["date"][hot], g["count"][hot]):
            if last is None or (d - last).days >= COOLDOWN:
                rows.append({"timestamp": d, "ticker": tk, "score": int(c)})
                last = d
    feed = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return feed


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    feed = build_feed()
    out = os.path.join(DATA_DIR, "wsb_mentions.csv")
    feed.to_csv(out, index=False)
    print(f"wrote {out}: {len(feed):,} events, {feed['ticker'].nunique()} tickers, "
          f"{feed['timestamp'].min().date()} -> {feed['timestamp'].max().date()}")

    prov = os.path.join(DATA_DIR, "PROVENANCE.md")
    with open(prov, "w", encoding="utf-8") as fh:
        fh.write(
            "# Feed provenance — wsb_mentions.csv\n\n"
            "Derived by [`examples/build_wsb_feed.py`](../examples/build_wsb_feed.py) from\n"
            "**[youyanggu/yolostocks-data](https://github.com/youyanggu/yolostocks-data)** —\n"
            "daily r/WallStreetBets mention counts, top-100 tickers, **CC-BY-4.0**.\n\n"
            f"Event = a *viral surge*: a session with count >= **{FLOOR}** mentions AND\n"
            f">= **{MULT}x** its trailing **{WIN}**-session median, debounced to one event per\n"
            f"name per **{COOLDOWN}** calendar days. Non-equity tickers (crypto, slang) dropped.\n\n"
            f"Result: {len(feed):,} events on {feed['ticker'].nunique()} tickers, "
            f"{feed['timestamp'].min().date()} to {feed['timestamp'].max().date()}.\n\n"
            "The raw source CSVs are cached under `../_rawfeed/` (CC-BY-4.0, "
            "© youyanggu). This derived feed inherits that licence.\n"
        )
    print(f"wrote {prov}")


if __name__ == "__main__":
    main()
