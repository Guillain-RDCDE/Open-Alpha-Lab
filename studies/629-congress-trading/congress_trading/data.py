"""Data layer for Study 629 — Congress Trading.

Two sources, both offline-friendly once cached:

* **Real tape.** The **Senate Stock Watcher** dataset (timothycarambat/senate-stock-watcher-data
  on GitHub) — every Periodic Transaction Report (PTR) scraped from the Senate's own
  efdsearch.senate.gov, one JSON file per **disclosure day** (the day the filing hit the public
  record). We parse every per-day report, keep **stock purchases with a resolvable ticker**, and
  stamp each with its **disclosure date** — the only date a member of the public could have
  traded on. Coverage: disclosures **2014-01 → 2021-03** (the scraper stopped updating in March
  2021); usable stock purchases start 2015. Prices: yfinance daily **auto-adjusted
  (total-return)** closes for every purchased ticker still resolvable on Yahoo, plus **SPY** as
  the market benchmark. Events cached as ``senate_purchases.csv``; prices as ``px_panel.parquet``.

* **Synthetic.** A deterministic, fixed-seed world: a market factor plus idiosyncratic stocks,
  with disclosure events at random dates and a controllable **planted post-disclosure drift**
  (knob ``edge_daily``, added to the bought stock's return for the holding window after each
  event). ``edge_daily = 0`` is the null — the machinery must NOT manufacture significance.

Known coverage caveats (named in results.md and on the Signal axis):

* **Survivorship in the price panel.** Tickers that were delisted/renamed since (FEYE, DISCA,
  CTL, ...) do not resolve on Yahoo today and drop out of the panel. Senators' picks that later
  died are under-represented — this biases the measured buy-alpha **up**, and the verdict must
  survive that tilt.
* **One-senator concentration.** David Perdue alone is ~39% of all disclosed purchases; splits
  and a Perdue-excluded aggregate are reported.
* **The window is 2015–2021** — post-STOCK-Act, pre-2021; it cannot speak to Ziobrowski's
  1993–1998 sample, only to the modern disclosed tape.
"""

from __future__ import annotations

import io
import json
import os
import re
import tarfile
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
EVENTS_CACHE = os.path.join(CACHE_DIR, "senate_purchases.csv")
PRICES_CACHE = os.path.join(CACHE_DIR, "px_panel.parquet")

REPO_TARBALL = ("https://github.com/timothycarambat/senate-stock-watcher-data/"
                "archive/refs/heads/master.tar.gz")

# ---------------------------------------------------------------------------
# Senator -> party map, hardcoded from public record (keys are the raw names as
# they appear in the Senate Stock Watcher filings). Party as of the senator's
# trading window in this sample. Source: US Senate biographical directory
# (bioguide.congress.gov). Angus King is an Independent who caucuses with the
# Democrats; for the two-way split he is grouped with D (stated in results.md).
# ---------------------------------------------------------------------------
PARTY = {
    "A. Mitchell McConnell, Jr.": "R",
    "Angus S King, Jr.": "D",          # Independent, caucuses with D
    "Christopher A Coons": "D",
    "David A Perdue , Jr": "R",
    "Gary C Peters": "D",
    "James M Inhofe": "R",
    "Jerry Moran,": "R",
    "John Cornyn": "R",
    "John F Reed": "D",
    "John Hoeven": "R",
    "John N Kennedy": "R",
    "John W Hickenlooper": "D",
    "Joseph Manchin, III": "D",
    "Kelly Loeffler": "R",
    "Pat Roberts": "R",
    "Patrick J Toomey": "R",
    "Patty Murray": "D",
    "Rafael E Cruz": "R",
    "Ron L Wyden": "D",
    "Sheldon Whitehouse": "D",
    "Shelley M Capito": "R",
    "Susan M Collins": "R",
    "Tammy Duckworth": "D",
    "Thomas R Carper": "D",
    "William Cassidy": "R",
    "William F Hagerty, IV": "R",
}

# The "famous names" — the meme subset for the third axis: the senators whose
# trading drew a DOJ/SEC insider-trading probe and saturation media coverage in
# 2020 (the COVID-trading scandal): Richard Burr, Kelly Loeffler, James Inhofe,
# Dianne Feinstein (probe reported by NYT/ProPublica, March–May 2020), plus
# David Perdue, whose trading (e.g. Cardlytics) drew its own DOJ investigation
# (NYT, 2020-06-16). Burr and Feinstein have no disclosed stock PURCHASES with
# tickers in this sample, so the famous subset on this tape is:
FAMOUS = ["David A Perdue , Jr", "Kelly Loeffler", "James M Inhofe"]

TICK_RE = re.compile(r">([A-Za-z.\-]{1,6})</a>")
PLAIN_RE = re.compile(r"^[A-Z.\-]{1,6}$")


# --------------------------------------------------------------------------- #
# Real tape — events
# --------------------------------------------------------------------------- #
def fetch_events(events_path: str = EVENTS_CACHE) -> pd.DataFrame:
    """Download the Senate Stock Watcher repo tarball ONCE, parse every per-day
    transaction report (the filename carries the DISCLOSURE date), keep stock
    purchases with a resolvable ticker, and cache the event table.

    Network-only; used once to build the cache.
    """
    req = urllib.request.Request(REPO_TARBALL, headers={"User-Agent": "OpenAlphaLab research desk"})
    with urllib.request.urlopen(req, timeout=120) as r:
        blob = r.read()
    rows = []
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for m in tar.getmembers():
            mm = re.search(r"data/transaction_report_for_(\d\d)_(\d\d)_(\d{4})\.json$", m.name)
            if not mm:
                continue
            disc = f"{mm.group(3)}-{mm.group(1)}-{mm.group(2)}"
            filings = json.load(tar.extractfile(m))
            for f in filings:
                name = (f.get("first_name", "") + " " + f.get("last_name", "")).strip()
                for t in f.get("transactions", []):
                    raw = (t.get("ticker") or "").strip()
                    hit = TICK_RE.search(raw)
                    tick = hit.group(1).upper() if hit else (raw if PLAIN_RE.match(raw) else None)
                    if (t.get("type") != "Purchase" or t.get("asset_type") != "Stock"
                            or not tick or tick == "--"):
                        continue
                    rows.append({"senator": name, "party": PARTY.get(name, "?"),
                                 "famous": name in FAMOUS,
                                 "disclosure_date": disc,
                                 "tx_date": t.get("transaction_date"),
                                 "ticker": tick.replace(".", "-"),  # yahoo convention
                                 "amount": t.get("amount"), "owner": t.get("owner")})
    ev = pd.DataFrame(rows)
    ev["disclosure_date"] = pd.to_datetime(ev["disclosure_date"])
    ev["tx_date"] = pd.to_datetime(ev["tx_date"], format="%m/%d/%Y", errors="coerce")
    ev = ev.sort_values(["disclosure_date", "senator", "ticker"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(events_path), exist_ok=True)
    ev.to_csv(events_path, index=False)
    return ev


def fetch_prices(events_path: str = EVENTS_CACHE, prices_path: str = PRICES_CACHE,
                 start: str = "2013-06-01", end: str = "2022-12-31",
                 chunk: int = 80, retries: int = 3) -> pd.DataFrame:
    """Download yfinance auto-adjusted closes for every purchased ticker + SPY, cache wide.

    Network-only; used once. Tickers that no longer resolve on Yahoo (delisted/renamed)
    drop out — the survivorship this creates is named in results.md.
    """
    import yfinance as yf

    ev = load_events(events_path)
    tickers = sorted(ev["ticker"].unique().tolist())
    frames = []
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(batch, start=start, end=end, auto_adjust=True,
                                  progress=False)["Close"]
                if raw is not None and len(raw) > 0:
                    break
            except Exception:
                time.sleep(3.0)
        if raw is None:
            continue
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(batch[0])
        frames.append(raw)
        time.sleep(1.0)
    px = pd.concat(frames, axis=1)
    px = px.loc[:, px.notna().sum() >= 60]          # need a usable history
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]
    px["SPY"] = spy
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    os.makedirs(os.path.dirname(prices_path), exist_ok=True)
    px.to_parquet(prices_path)
    return px


def have_real(events_path: str = EVENTS_CACHE, prices_path: str = PRICES_CACHE) -> bool:
    return os.path.exists(events_path) and os.path.exists(prices_path)


def load_events(path: str = EVENTS_CACHE) -> pd.DataFrame:
    ev = pd.read_csv(path, parse_dates=["disclosure_date", "tx_date"])
    ev["famous"] = ev["famous"].astype(bool)
    return ev


# Data-integrity guard: Yahoo recycles tickers, so a symbol bought in 2015 can carry a
# DIFFERENT instrument's garbage prices today (e.g. "ITC": ITC Holdings was acquired in 2016;
# the symbol now shows a +672,745% one-day "return" followed by -99.99%). No genuine equity
# gains +300% in one session on this tape (the max real move is GME's +134%), so any column
# with a daily return above +300% is a recycled/corrupt series and is dropped. Deterministic
# on the cache; the count is reported in results.md.
MAX_DAILY_RET = 3.0


def load_prices(path: str = PRICES_CACHE) -> tuple[pd.DataFrame, pd.Series]:
    """Wide adjusted-close panel (stocks) + the SPY benchmark series (integrity-filtered)."""
    px = pd.read_parquet(path).sort_index()
    spy = px["SPY"]
    px = px.drop(columns=["SPY"])
    daily_max = px.pct_change().max()
    bad = daily_max[daily_max > MAX_DAILY_RET].index
    if len(bad):
        px = px.drop(columns=list(bad))
    return px, spy


def load_real() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Convenience: (stock price panel, SPY series, purchase-event table)."""
    px, spy = load_prices()
    return px, spy, load_events()


# --------------------------------------------------------------------------- #
# Synthetic control
# --------------------------------------------------------------------------- #
def synthetic_world(n_stocks: int = 60, n_days: int = 1800, n_events: int = 700,
                    edge_daily: float = 0.0, hold: int = 126,
                    seed: int = 629) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Deterministic market + stocks + disclosure events with a PLANTED post-disclosure drift.

    Stocks load 1.0 on a common market factor plus idiosyncratic noise. Each event picks a
    stock and a disclosure day; if ``edge_daily`` != 0, that stock earns an EXTRA
    ``edge_daily`` per day for the ``hold`` trading days after the event's entry (next day).
    With ``edge_daily = 0`` the events are pure noise: no abnormal-return machinery may
    reach significance. Returns (stock price panel, benchmark price series, event table) in
    the same shape as :func:`load_real`. Business-day index well under the 250-year trap.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-02", periods=n_days)
    mkt_ret = rng.normal(0.0004, 0.010, size=n_days)
    names = [f"S{i:03d}" for i in range(n_stocks)]
    idio = rng.normal(0.0, 0.018, size=(n_days, n_stocks))
    ret = idio + mkt_ret[:, None]

    ev_stock = rng.integers(0, n_stocks, size=n_events)
    ev_day = rng.integers(30, n_days - hold - 30, size=n_events)
    if edge_daily != 0.0:
        for s, d0 in zip(ev_stock, ev_day):
            ret[d0 + 1:d0 + 1 + hold, s] += edge_daily   # entry = next day after disclosure

    # cumprod of SIMPLE returns (exp(cumsum) would gift the noisier stocks a mechanical
    # convexity premium over the benchmark and mis-center the null away from zero)
    px = pd.DataFrame(100.0 * np.cumprod(1.0 + ret, axis=0), index=dates, columns=names)
    spy = pd.Series(100.0 * np.cumprod(1.0 + mkt_ret), index=dates, name="SPY")
    events = pd.DataFrame({
        "senator": [f"Senator {chr(65 + (i % 20))}" for i in range(n_events)],
        "party": ["D" if i % 2 == 0 else "R" for i in range(n_events)],
        "famous": [False] * n_events,
        "disclosure_date": dates[ev_day],
        "tx_date": dates[np.maximum(ev_day - 20, 0)],
        "ticker": [names[s] for s in ev_stock],
        "amount": ["$1,001 - $15,000"] * n_events,
        "owner": ["Self"] * n_events,
    }).sort_values("disclosure_date").reset_index(drop=True)
    return px, spy, events
