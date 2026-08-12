"""Data layer for Study 845 — Stadium Naming-Rights Curse.

Three ingredients, all offline-friendly once cached:

* **The naming-rights deal table, hardcoded.** ``DEALS`` is a curated table of major,
  well-documented US/European stadium & arena naming-rights deals in which the sponsor
  was (or later became) a **listed** company — announcement date, venue, sponsor and
  the sponsor's ticker. No free, machine-readable "naming-rights deal index" exists, so
  — exactly like the sibling studies that hand-build an event calendar
  ([160-skyscraper-curse](../../160-skyscraper-curse/)'s ``SKYSCRAPER_EVENTS``,
  [707-plane-crash-effect](../../707-plane-crash-effect/)'s ``DISASTERS``) — this is a
  hand-built table cross-referenced against each venue's public naming-rights record.
  Every row carries a ``tradable`` flag: the flagship cautionary tales — **Enron Field
  (1999, Enron went bankrupt in 2001), the FTX Arena (2021, FTX collapsed in 2022),
  Crypto.com Arena (2021) and the MCI Center (WorldCom)** — are all sponsors that were
  either private (crypto exchanges) or delisted into bankruptcy, so they have **no
  usable price tape** for a forward-return study and are marked ``tradable=False``.
  They stay in the table (they *are* the folklore) but are named honestly as
  untestable, not silently dropped: the whole question of this study is whether the
  "curse" is anything beyond those two or three vivid, cherry-picked blow-ups.

* **Real tape.** Daily total-return adjusted closes for SPY (the market benchmark) and
  each tradable sponsor's ticker, from yfinance (no key), cached as CSV under the
  study's own ``_cache/``. A sponsor whose ticker lacks enough history to cover its
  deal's forward window is dropped from that window's test (tracked, named honestly,
  never zero-filled).

* **Synthetic world.** A deterministic, seeded set of "sponsor" random-walk tapes plus
  a common "SPY" tape, with a TUNABLE planted post-deal drift (``edge`` in total
  buy-and-hold-return units) applied to each sponsor over the window after its deal.
  ``edge = 0`` is the null world — sponsors are statistically identical to the market
  after the deal, and the event-study machinery must NOT manufacture a curse from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

SPY_CACHE = os.path.join(CACHE_DIR, "snc_spy.csv")

START = "1997-01-01"
AS_OF = "2026-06-30"        # last complete calendar month at publication

# --------------------------------------------------------------------------- #
# Hardcoded table of stadium / arena naming-rights deals.
# Each row: (announce_date, venue, sponsor, ticker, tradable, note).
#   * announce_date — the widely-reported date the naming-rights deal was announced
#     (public record; cross-referenced against each venue's naming-rights history — see
#     docs/references.md). The event-study code snaps this to the first NYSE session
#     on/after that date (a weekend/overseas announcement rolls forward to the next
#     open) — the study's single documented execution lag (see strategy.py).
#   * ticker — the sponsor's listed ticker (US-listed common or ADR) where one exists.
#   * tradable — False for sponsors with no usable price tape at/after the deal:
#     Enron (delisted into 2001 bankruptcy), MCI/WorldCom (2002 bankruptcy), and the
#     private crypto exchanges FTX & Crypto.com. These are the folklore's poster
#     children and are kept in the table but excluded from the forward-return test,
#     named honestly.
# --------------------------------------------------------------------------- #
DEALS: list[tuple[str, str, str, str, bool, str]] = [
    # --- the famous cautionary tales — NOT tradable (private or delisted-into-bankruptcy)
    ("1999-04-07", "Enron Field (Houston Astros)", "Enron", "ENE", False,
     "sponsor bankrupt 2001-12; delisted — the origin myth of the curse"),
    ("1997-12-01", "MCI Center (Washington)", "MCI/WorldCom", "WCOM", False,
     "WorldCom accounting fraud, bankrupt 2002-07; delisted"),
    ("2021-03-26", "FTX Arena (Miami Heat)", "FTX", "", False,
     "private crypto exchange; collapsed 2022-11 — the modern curse anecdote"),
    ("2021-11-17", "Crypto.com Arena (LA, ex-Staples)", "Crypto.com", "", False,
     "private crypto exchange; deep 2022 crypto-winter layoffs"),
    ("2019-09-15", "SoFi Stadium (LA Rams)", "SoFi", "", False,
     "private at announcement (IPO'd via SPAC 2021-06 as SOFI) — no tape at the deal"),

    # --- tradable, listed sponsors: announcement date + ticker
    ("1998-08-19", "PNC Park (Pittsburgh Pirates)", "PNC Financial", "PNC", True, ""),
    ("1998-11-16", "FedExField (Washington)", "FedEx", "FDX", True, ""),
    ("1998-11-30", "Comerica Park (Detroit Tigers)", "Comerica", "CMA", True, ""),
    ("2001-08-16", "Ford Field (Detroit Lions)", "Ford", "F", True, ""),
    ("2002-06-05", "Minute Maid Park (Houston Astros)", "Coca-Cola", "KO", True,
     "renamed the ex-Enron Field after Enron's collapse"),
    ("2002-06-06", "Lincoln Financial Field (Philadelphia)", "Lincoln National", "LNC", True, ""),
    ("2003-07-21", "Toyota Center (Houston Rockets)", "Toyota", "TM", True, ""),
    ("2004-06-02", "Bank of America Stadium (Carolina)", "Bank of America", "BAC", True, ""),
    ("2006-10-16", "Honda Center (Anaheim)", "Honda", "HMC", True, ""),
    ("2006-11-13", "Citi Field (NY Mets)", "Citigroup", "C", True, ""),
    ("2007-01-18", "Barclays Center (Brooklyn)", "Barclays", "BCS", True, ""),
    ("2007-10-25", "Prudential Center (Newark)", "Prudential Financial", "PRU", True, ""),
    ("2008-01-11", "Progressive Field (Cleveland)", "Progressive", "PGR", True, ""),
    ("2010-07-13", "Wells Fargo Center (Philadelphia)", "Wells Fargo", "WFC", True,
     "renamed from Wachovia Center after the 2008 acquisition"),
    ("2011-08-23", "MetLife Stadium (NY/NJ)", "MetLife", "MET", True, ""),
    ("2015-08-24", "Mercedes-Benz Stadium (Atlanta)", "Mercedes-Benz Group (ex-Daimler)", "MBGYY", True,
     "US ADR of the ex-Daimler parent (history from 2010)"),
    ("2016-01-27", "Chase Center (Golden State)", "JPMorgan Chase", "JPM", True, ""),
    ("2017-08-09", "Capital One Arena (Washington)", "Capital One", "COF", True,
     "renamed from Verizon Center"),
    ("2017-08-29", "Scotiabank Arena (Toronto)", "Bank of Nova Scotia", "BNS", True,
     "renamed from Air Canada Centre"),
    ("2018-07-26", "Fiserv Forum (Milwaukee Bucks)", "Fiserv", "FISV", True,
     "traded as FISV at the deal; renamed ticker FI in 2023 — Yahoo keeps history under FISV"),
    ("2018-12-19", "T-Mobile Park (Seattle Mariners)", "T-Mobile US", "TMUS", True, ""),
    ("2019-08-05", "Allegiant Stadium (Las Vegas Raiders)", "Allegiant Travel", "ALGT", True, ""),
    ("2019-09-17", "Truist Park (Atlanta Braves)", "Truist Financial", "TFC", True, ""),
    ("2020-06-25", "Climate Pledge Arena (Seattle)", "Amazon", "AMZN", True, ""),
    ("2020-10-22", "Ball Arena (Denver)", "Ball Corp", "BALL", True,
     "renamed from Pepsi Center"),
    ("2021-07-15", "Caesars Superdome (New Orleans)", "Caesars Entertainment", "CZR", True, ""),
    ("2021-10-21", "Paycom Center (Oklahoma City)", "Paycom", "PAYC", True, ""),
    ("2022-05-04", "Snapdragon Stadium (San Diego)", "Qualcomm", "QCOM", True, ""),
    ("2023-06-20", "Intuit Dome (LA Clippers)", "Intuit", "INTU", True, ""),
]


def deal_table() -> pd.DataFrame:
    """The curated table as a frame, sorted by announcement date."""
    df = pd.DataFrame(DEALS, columns=["date", "venue", "sponsor", "ticker", "tradable", "note"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def tradable_deals() -> pd.DataFrame:
    """Only the rows with a usable listed sponsor tape (``tradable == True``)."""
    return deal_table().query("tradable").reset_index(drop=True)


def tradable_tickers() -> list[str]:
    return sorted(tradable_deals()["ticker"].unique().tolist())


def _ticker_cache(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"snc_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str = "2026-07-01", retries: int = 4) -> None:
    """Download SPY + every tradable sponsor ticker's total-return closes; cache them.

    Network; run once. ``auto_adjust=True`` so closes fold in splits and dividends
    (total-return), and the forward returns below are plain buy-and-hold price ratios
    on the cached close. Each ticker is retried up to ``retries`` times; a ticker that
    still returns nothing is skipped (its deals are named honestly as no-coverage, not
    zero-filled).
    """
    import time

    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)

    def _dl(sym: str) -> pd.DataFrame | None:
        for k in range(retries):
            try:
                df = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
                if df is not None and len(df) > 0:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    return df[["Close"]].dropna()
            except Exception:
                pass
            time.sleep(1.5 * (k + 1))
        return None

    spy = _dl("SPY")
    if spy is not None:
        spy.to_csv(SPY_CACHE)

    for t in tradable_tickers():
        df = _dl(t)
        if df is not None:
            df.to_csv(_ticker_cache(t))
        else:
            print(f"  (no data for {t} — skipped)")


def have_real() -> bool:
    """True iff the SPY cache plus at least a dozen sponsor caches are present."""
    if not os.path.exists(SPY_CACHE):
        return False
    have = sum(os.path.exists(_ticker_cache(t)) for t in tradable_tickers())
    return have >= 12


def load_spy(asof: str = AS_OF) -> pd.Series:
    spy = pd.read_csv(SPY_CACHE, index_col=0, parse_dates=True).sort_index()["Close"]
    return spy.loc[(spy.index >= START) & (spy.index <= asof)]


def load_prices(asof: str = AS_OF) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Cached ``(spy_close, {ticker: close})`` series, sliced to ``[START, asof]``.

    Only tickers whose cache file exists are returned; a missing ticker's deals are
    simply absent from the forward-return test (named honestly, not zero-filled).
    """
    spy = load_spy(asof)
    prices: dict[str, pd.Series] = {}
    for t in tradable_tickers():
        p = _ticker_cache(t)
        if not os.path.exists(p):
            continue
        s = pd.read_csv(p, index_col=0, parse_dates=True).sort_index()["Close"]
        prices[t] = s.loc[(s.index >= START) & (s.index <= asof)]
    return spy, prices


# --------------------------------------------------------------------------- #
# Reproducibility fingerprint
# --------------------------------------------------------------------------- #
def fingerprint(s: pd.Series) -> str:
    """Short, stable content hash of a price series (12 hex)."""
    arr = np.ascontiguousarray(s.to_numpy(dtype=float))
    h = hashlib.sha1()
    for ts in s.index:
        h.update(str(pd.Timestamp(ts).date()).encode())
    h.update(np.round(arr, 6).tobytes())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic world — a planted post-deal underperformance drift (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(edge: float = 0.0, seed: int = 845,
                    n_events: int = 28, n_days: int = 8000,
                    window: int = 252, daily_vol: float = 0.018,
                    mkt_vol: float = 0.011, start: str = "2000-01-03",
                    ) -> tuple[pd.Series, dict[str, pd.Series], list[tuple[str, pd.Timestamp]]]:
    """A reproducible set of "sponsor" tapes + a common "SPY" tape with a planted curse.

    A common market log-return series (i.i.d. normal, std ``mkt_vol``, mild up-drift)
    drives a market tape; each of ``n_events`` "sponsor" names is that market plus its
    own idiosyncratic noise (std ``daily_vol``). Each sponsor gets exactly one "deal"
    date well away from the edges; over the ``window`` sessions *after* the deal the
    sponsor's returns are docked ``edge / window`` per day — a clean, mechanical
    post-deal underperformance of total size ``edge`` (in buy-and-hold-return units).
    ``edge = 0`` is the null world: sponsors are statistically indistinguishable from
    the market after the deal, and the cross-event BHAR test must NOT reach
    significance.

    Business-day index, span ~32 years — far below the pandas ns-timestamp trap.
    Returns ``(spy_close, {ticker: sponsor_close}, [(ticker, deal_date), ...])``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    mkt_ret = rng.normal(0.0003, mkt_vol, n_days)          # common market factor + drift
    spy_close = pd.Series(100.0 * np.exp(np.cumsum(mkt_ret)), index=idx)

    margin = window + 40
    pool = np.arange(margin, n_days - margin)
    locs = np.sort(rng.choice(pool, size=min(n_events, pool.size), replace=False))

    per_day = edge / max(window, 1)
    idio_drift = -0.5 * daily_vol ** 2   # martingale correction so E[BHAR]=0 at edge=0
    prices: dict[str, pd.Series] = {}
    events: list[tuple[str, pd.Timestamp]] = []
    for i, loc in enumerate(locs):
        ticker = f"SYN{i:02d}"
        # market + idiosyncratic (mean-zero in ratio terms via the drift correction)
        ret = mkt_ret + rng.normal(idio_drift, daily_vol, n_days)
        ret[loc + 1: loc + 1 + window] += per_day   # planted post-deal drift (edge<0 = curse)
        prices[ticker] = pd.Series(100.0 * np.exp(np.cumsum(ret)), index=idx)
        events.append((ticker, idx[loc]))
    return spy_close, prices, events
