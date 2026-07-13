"""Data layer for Study 720 — Super-Bowl-Advertiser (does buying a Super Bowl ad pay?).

Two sources, both offline-friendly:

* **Real tape.** A hardcoded, transparent table of ~32 real *listed* Super Bowl advertisers
  as **advertiser-year events** — a company that ran a national commercial in a given Super
  Bowl (2015-2024). For each event we record the *currently-traded ticker*, the Super Bowl
  **game date** (a Sunday), and a short label; from yfinance daily adjusted closes we then
  measure the event-window drift after the game (the "big-ad signal") vs the market (SPY).
  Cached under ``_cache/superbowl_prices.csv`` (a wide CSV, one column per ticker + SPY).

  The table is the honest part of this study: every row is a real, documented, publicly
  listed advertiser, and the *survivorship* is named loudly — the loudest Super Bowl
  advertisers in history **went to zero** (Pets.com, Computer.com, Kozmo.com — the dot-com
  class of Super Bowl 2000) and leave **no** yfinance series, so the surviving tape is biased
  *toward* the advertisers that did not collapse. That bias works **for** the believers'
  "advertising pays" claim, so a survivor-only drift near zero is a conservative refutation.

* **Synthetic.** A deterministic, fixed-seed generator that plants a controllable
  post-game "drift" into otherwise-random event windows. The positive control: with the
  ``edge`` knob at 0 the inference must NOT manufacture a significant drift out of a few
  dozen events; with a large planted edge it must light up.

Pure numpy + pandas + stdlib for the offline path. ``fetch_prices`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "superbowl_prices.csv")

# ---------------------------------------------------------------------------- #
# The hardcoded Super Bowl advertiser table — real, listed advertiser-year events.
#
# Each row: (ticker, game_date, label)
#   * ticker    — the symbol that carries the advertiser's price on yfinance today.
#   * game_date — the Super Bowl Sunday the commercial aired (a non-trading day; the
#                 engine snaps to the next trading day = the Monday after).
#   * label     — the ad / brand, for the charts.
#
# Sources: Super Bowl ad round-ups (AdAge, USA Today Ad Meter, company press) — every row is
# a company that ran a national in-game commercial that year and was publicly listed at the
# time. The game dates are exact Super Bowl Sundays; the engine works in trading-day offsets
# from the following Monday, so the believers' "buy Monday, ride the buzz" trade is what we
# actually measure (a one-day entry lag, no weekend-gap look-ahead).
# ---------------------------------------------------------------------------- #
ADVERTISERS = [
    # ---- 2015 (SB XLIX, Feb 1) ----------------------------------------------
    ("WIX",   "2015-02-01", "Wix.com — first Super Bowl ad"),
    # ---- 2016 (SB 50, Feb 7) ------------------------------------------------
    ("WIX",   "2016-02-07", "Wix.com — Kung Fu Panda"),
    # ---- 2017 (SB LI, Feb 5) ------------------------------------------------
    ("WIX",   "2017-02-05", "Wix.com — Jason Statham / Gal Gadot"),
    # ---- 2019 (SB LIII, Feb 3) ----------------------------------------------
    ("BUD",   "2019-02-03", "Bud Light — 'corn syrup' medieval ad"),
    ("INTU",  "2019-02-03", "TurboTax — RoboChild"),
    # ---- 2020 (SB LIV, Feb 2) -----------------------------------------------
    ("MSFT",  "2020-02-02", "Microsoft — Katie Sowers"),
    ("GOOGL", "2020-02-02", "Google Pixel — 'Loretta'"),
    ("KHC",   "2020-02-02", "Planters — death of Mr. Peanut"),
    ("AMZN",  "2020-02-02", "Amazon Alexa — Ellen & Portia"),
    ("DIS",   "2020-02-02", "Disney+ — launch push"),
    # ---- 2021 (SB LV, Feb 7) ------------------------------------------------
    ("TMUS",  "2021-02-07", "T-Mobile — Gronk / Adam Levine"),
    ("GM",    "2021-02-07", "GM EV — Will Ferrell 'Norway'"),
    ("RKT",   "2021-02-07", "Rocket Mortgage — 'certain is better'"),
    ("DKNG",  "2021-02-07", "DraftKings — sportsbook push"),
    ("VZ",    "2021-02-07", "Verizon 5G — 'the 5G phone'"),
    # ---- 2022 (SB LVI, Feb 13) ----------------------------------------------
    ("COIN",  "2022-02-13", "Coinbase — bouncing QR code"),
    ("SOFI",  "2022-02-13", "SoFi — 'get your money right'"),
    ("META",  "2022-02-13", "Meta Quest — 'Old Friends, New Fun'"),
    ("CRM",   "2022-02-13", "Salesforce — McConaughey 'not the metaverse'"),
    ("UBER",  "2022-02-13", "Uber Eats — 'don't eat' bit"),
    ("STLA",  "2022-02-13", "Jeep — Springsteen 'The Middle'"),
    ("PEP",   "2022-02-13", "Pepsi — Zero Sugar / halftime-show sponsor"),
    ("TM",    "2022-02-13", "Toyota — 'Brothers' (Bonham)"),
    # ---- 2023 (SB LVII, Feb 12) ---------------------------------------------
    ("PDD",   "2023-02-12", "Temu — 'shop like a billionaire'"),
    ("BKNG",  "2023-02-12", "Booking.com — Melissa McCarthy"),
    ("COIN",  "2023-02-12", "Coinbase — return ad"),
    ("DKNG",  "2023-02-12", "DraftKings — Kevin Hart"),
    ("BUD",   "2023-02-12", "Budweiser — 'Six Degrees of Bud'"),
    # ---- 2024 (SB LVIII, Feb 11) --------------------------------------------
    ("ELF",   "2024-02-11", "e.l.f. Beauty — first-ever ad ('Judge Beauty')"),
    ("PDD",   "2024-02-11", "Temu — repeat billionaire ads"),
    ("ULTA",  "2024-02-11", "Ulta Beauty — first-ever ad"),
    ("BUD",   "2024-02-11", "Bud Light — 'Easy to Drink' comeback"),
]

# Famous Super Bowl advertisers that DELISTED / went to zero (no yfinance series).
# Named for the survivorship caveat: the survivor tape is biased FOR "advertising pays",
# because the loudest advertisers that torched their capital on a Super Bowl spot and then
# collapsed left the tape entirely. The dot-com class of Super Bowl 2000 is the archetype.
DELISTED = [
    "Pets.com — SB 2000 sock puppet; delisted Nov 2000 (~9 months later)",
    "Computer.com — SB 2000; spent ~half its capital on the ad, gone within a year",
    "Kozmo.com — SB 2000; shut down 2001",
    "LifeMinders.com — SB 2000; acquired/gone by 2001",
    "Epidemic Marketing — SB 2000; defunct",
    "OurBeginning.com — SB 2000; defunct",
    "Netpliance / iOpener — SB 2000; product & firm gone",
    "Just for Feet — SB 1999; bankrupt within a year (accounting fraud)",
    "HotJobs.com — SB 1999; absorbed, brand retired",
    "Pixelon — SB 1999-era launch spend; founder was a fugitive, firm imploded",
    "Squarespace (SQSP) — a Super Bowl regular (Zendaya '22, Adam Driver '23, "
    "Scorsese '24) taken PRIVATE by Permira (Oct 2024); the ad buyer left the public tape",
]

TICKERS = sorted({t for t, *_ in ADVERTISERS})


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_prices(start: str = "2013-06-01", end: str | None = None,
                 path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download the advertiser tickers + a market benchmark (SPY) via yfinance and cache.

    Network-only; used once to build ``_cache/superbowl_prices.csv``. Never imported by the
    offline notebook cells. Keeps every column with any usable history.
    """
    import yfinance as yf

    tickers = sorted(set(TICKERS) | {"SPY"})
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached wide adjusted-close frame (index = date, columns = tickers + SPY)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def event_table(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """The advertiser table joined to whether the cached tape covers the game window.

    Returns one row per advertiser-year with the game Sunday snapped forward to the next
    available trading day (the Monday after — the first day you could act on the ad), or NaT
    if the ticker isn't priced around the event (e.g. it listed after that game). Offline if
    the cache exists.
    """
    prices = load_prices(path) if os.path.exists(path) else None
    rows = []
    for tkr, dt, label in ADVERTISERS:
        game = pd.Timestamp(dt)
        snapped = pd.NaT
        if prices is not None and tkr in prices.columns:
            s = prices[tkr].dropna()
            if len(s):
                # first trading day on/after the Sunday game = the Monday after
                after = s.index[(s.index >= game) & (s.index <= game + pd.Timedelta(days=6))]
                if len(after):
                    snapped = after[0]
        rows.append({"ticker": tkr, "game": game, "snapped": snapped,
                     "year": game.year, "label": label})
    return pd.DataFrame(rows)


def load_real(path: str = DEFAULT_CACHE) -> dict:
    """Convenience bundle for the strategy layer: prices + the snapped event table."""
    return {"prices": load_prices(path), "events": event_table(path)}


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_ads(n_events: int = 32, edge: float = 0.0, seed: int = 726,
                  n_days: int = 1_500, mu_daily: float = 0.0004,
                  sig_daily: float = 0.022, drift_days: int = 5,
                  hold_days: int = 20) -> dict:
    """Deterministic event windows with a planted post-game "drift" of size ``edge``.

    For each of ``n_events`` synthetic stocks we draw a random walk (drift ``mu_daily``,
    vol ``sig_daily``), place a Super Bowl at the midpoint, and — when ``edge`` != 0 — add a
    cumulative **+edge** abnormal return spread over ``drift_days`` trading days right after
    the game (the "big-ad signal"). ``edge = 0`` plants nothing: the inference must then find
    no significant drift out of a handful of events, however the noise falls. A large ``edge``
    (e.g. 0.10 = a 10% post-game drift) must light up the drift leg.

    Returns a bundle shaped like :func:`load_real`: a wide price frame (one column per
    synthetic ticker plus ``SPY``) and an event table with the game date per ticker. Uses
    ``pd.period_range`` -> timestamp to stay clear of datetime overflow.
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2005-01-03", periods=n_days, freq="D").to_timestamp()
    cols, events = {}, []
    cols["SPY"] = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.009, n_days)))
    mid = n_days // 2
    for k in range(n_events):
        r = rng.normal(mu_daily, sig_daily, n_days)
        e0 = mid + int(rng.integers(-n_days // 6, n_days // 6))   # jittered game day
        if edge != 0.0:
            # drift spread over the drift_days trading days after the game
            r[e0 + 1:e0 + 1 + drift_days] += edge / drift_days
        price = 60.0 * np.exp(np.cumsum(r))
        tkr = f"AD{k:02d}"
        cols[tkr] = price
        events.append({"ticker": tkr, "game": idx[e0], "snapped": idx[e0],
                       "year": 2000 + k, "label": f"synthetic advertiser {k}"})
    prices = pd.DataFrame(cols, index=idx)
    return {"prices": prices, "events": pd.DataFrame(events)}


def fingerprint(events: pd.DataFrame | None = None) -> str:
    """Short content fingerprint of the advertiser table (ticker + game date), for as-of stamps."""
    if events is None:
        pairs = [(t, d) for t, d, _ in ADVERTISERS]
    else:
        pairs = [(r["ticker"], str(pd.Timestamp(r["game"]).date()))
                 for _, r in events.iterrows()]
    blob = "|".join(f"{t}@{d}" for t, d in pairs).encode()
    return hashlib.sha1(blob).hexdigest()[:12]
