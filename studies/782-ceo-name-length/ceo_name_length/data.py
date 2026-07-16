"""Data layer for Study 782 — CEO-Name-Length.

The claim under test: **the number of letters in a company's CEO surname predicts its
stock's return.** This is a deliberately absurd cross-sectional "characteristic" — a
sanity-check on the whole factor-zoo enterprise. If you sort a universe of large caps by
a label that *cannot* carry economic information (how long the boss's family name is) and
still find a long/short spread with a real *t*-stat, that is a data-snooping alarm, not a
discovery. The honest prior is a flat, boring null.

Shape B (cross-sectional characteristic sort): each name in a fixed large-cap universe
gets a static characteristic (surname length), the universe is sorted into terciles, and
we hold a dollar-neutral **long longest-surname / short shortest-surname** book, rebalanced
monthly. The unit of inference is the monthly long/short return.

Four ingredients:

* **The universe + CEO surnames, hardcoded.** 40 large-cap US names with the surname of the
  person who is CEO **as of the as-of date (2026-06)**. This is a deliberate SIMPLIFICATION,
  and it is disclosed loudly: the characteristic is a **static end-of-sample snapshot** —
  we do NOT track CEO turnover through history. That is acceptable *only* because the
  characteristic is economically inert by construction (it is a null-by-design test); it
  would be a look-ahead sin for any characteristic you actually believed in. Surnames are
  real, publicly-verifiable facts (company proxy statements / newsrooms), not a proxy tape.

* **The tradable instruments (yfinance).** The 40 tickers, plus ``SPY`` (S&P 500,
  total-return) as a market reference — the long/short book is dollar-neutral so it is
  market-neutral by construction; SPY is used to sanity-check that neutrality and as the
  named benchmark tape.

* **The characteristic.** ``len(surname)`` — a small integer (2..11 here). Standardised
  cross-sectionally before sorting.

* **Synthetic world.** A deterministic, seeded cross-sectional world with a TUNABLE planted
  linear characteristic->return slope. ``bump = 0`` is the null world; the sort/t machinery
  must not manufacture a spread from it, and must recover a planted slope monotonically.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build
the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

AS_OF = "2026-06-30"     # last complete month at publication
BENCHMARK = "SPY"        # S&P 500 total-return reference (book is dollar-neutral)
START = "2015-01-01"     # sample start for the monthly long/short book

# --------------------------------------------------------------------------- #
# The universe + CEO surname, hardcoded. Surname is the person who is CEO AS OF 2026-06
# (a STATIC end-of-sample snapshot — CEO turnover through history is NOT tracked; this is
# disclosed and is only defensible because the characteristic is inert by design). Surnames
# are real, publicly-verifiable facts from company proxies / newsrooms. `n` = len(surname).
# --------------------------------------------------------------------------- #
UNIVERSE = [
    # ticker, ceo_surname
    ("AAPL", "Cook"),        # Tim Cook
    ("MSFT", "Nadella"),     # Satya Nadella
    ("GOOGL", "Pichai"),     # Sundar Pichai
    ("AMZN", "Jassy"),       # Andy Jassy
    ("META", "Zuckerberg"),  # Mark Zuckerberg
    ("NVDA", "Huang"),       # Jensen Huang
    ("TSLA", "Musk"),        # Elon Musk
    ("JPM", "Dimon"),        # Jamie Dimon
    ("V", "McInerney"),      # Ryan McInerney
    ("WMT", "McMillon"),     # Doug McMillon
    ("JNJ", "Duato"),        # Joaquin Duato
    ("PG", "Moeller"),       # Jon Moeller
    ("HD", "Decker"),        # Ted Decker
    ("KO", "Quincey"),       # James Quincey
    ("PEP", "Laguarta"),     # Ramon Laguarta
    ("CRM", "Benioff"),      # Marc Benioff
    ("ORCL", "Catz"),        # Safra Catz
    ("ADBE", "Narayen"),     # Shantanu Narayen
    ("CSCO", "Robbins"),     # Chuck Robbins
    ("AMD", "Su"),           # Lisa Su
    ("QCOM", "Amon"),        # Cristiano Amon
    ("TXN", "Ilan"),         # Haviv Ilan
    ("IBM", "Krishna"),      # Arvind Krishna
    ("MA", "Miebach"),       # Michael Miebach
    ("BAC", "Moynihan"),     # Brian Moynihan
    ("WFC", "Scharf"),       # Charlie Scharf
    ("GS", "Solomon"),       # David Solomon
    ("C", "Fraser"),         # Jane Fraser
    ("XOM", "Woods"),        # Darren Woods
    ("CVX", "Wirth"),        # Mike Wirth
    ("MRK", "Davis"),        # Robert Davis
    ("PFE", "Bourla"),       # Albert Bourla
    ("MCD", "Kempczinski"),  # Chris Kempczinski
    ("CAT", "Umpleby"),      # Jim Umpleby
    ("GE", "Culp"),          # Larry Culp
    ("ABT", "Ford"),         # Robert Ford
    ("TMO", "Casper"),       # Marc Casper
    ("LLY", "Ricks"),        # David Ricks
    ("ACN", "Sweet"),        # Julie Sweet
    ("CMCSA", "Roberts"),    # Brian Roberts
]


def tickers() -> list[str]:
    return [t for t, _ in UNIVERSE]


def all_tickers() -> list[str]:
    return tickers() + [BENCHMARK]


def surname_len(surname: str) -> int:
    """Letters in the surname (letters only, so an apostrophe/space wouldn't count)."""
    return sum(ch.isalpha() for ch in surname)


def characteristics() -> pd.Series:
    """{ticker: surname length} — the static cross-sectional characteristic."""
    return pd.Series({t: surname_len(s) for t, s in UNIVERSE}, dtype=float)


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"ceo_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str = "2026-07-01", retries: int = 4) -> None:
    """Download adjusted (total-return) daily closes for the universe + SPY; cache them.

    Retries with linear backoff — Yahoo rate-limits transient bursts, so a first empty
    frame is usually cured by a short wait rather than a real "no such ticker".
    """
    import time

    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in all_tickers():
        last_err = None
        for attempt in range(retries):
            try:
                d = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                d = d[["Close"]].dropna()
                if len(d) > 0:
                    d.to_csv(_cache_path(t))
                    break
                last_err = f"empty frame for {t}"
            except Exception as e:  # noqa: BLE001 -- transient network/rate-limit
                last_err = str(e)
            time.sleep(2.0 * (attempt + 1))
        else:
            raise RuntimeError(f"fetch failed for {t} after {retries} tries: {last_err}")


def have_real() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in all_tickers())


def load_real(asof: str = AS_OF) -> dict[str, pd.Series]:
    """Cached {ticker: adjusted-close Series}, each sliced to <= asof."""
    out = {}
    for t in all_tickers():
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True).sort_index()
        s = df["Close"]
        out[t] = s[s.index <= pd.Timestamp(asof)]
    return out


def monthly_returns(prices: dict[str, pd.Series]) -> pd.DataFrame:
    """Month-end simple returns for every universe ticker (SPY kept separately by caller).

    Aligns all names on a common month-end grid and drops months where any name is missing
    (so the cross-section is balanced for the sort)."""
    cols = {}
    for t in tickers():
        m = prices[t].resample("ME").last()
        cols[t] = m.pct_change()
    df = pd.DataFrame(cols).dropna(how="any")
    return df


# --------------------------------------------------------------------------- #
# Synthetic world -- planted linear characteristic->return slope
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, seed: int = 798, n_assets: int = 40,
                    n_months: int = 138) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Deterministic cross-sectional world with a planted characteristic->return slope.

    Each asset gets an integer "surname length" characteristic (2..11). Monthly returns are
    a one-factor market model (asset beta x market) + idiosyncratic noise, PLUS a planted
    term ``bump * z_i`` added to every month of asset i, where ``z_i`` is the standardised
    characteristic. ``bump = 0`` is the null world (characteristic is inert); a positive
    ``bump`` makes longer surnames earn more, which the long(top)/short(bottom) sort must
    recover monotonically.

    Returns (monthly_returns[n_months x n_assets], characteristic Series, market vector).
    """
    rng = np.random.default_rng(seed)
    chars = rng.integers(2, 12, n_assets).astype(float)
    z = (chars - chars.mean()) / chars.std(ddof=0)
    mkt = rng.normal(0.006, 0.043, n_months)
    betas = rng.uniform(0.7, 1.3, n_assets)
    idio = rng.normal(0.0, 0.060, (n_months, n_assets))
    rets = mkt[:, None] * betas[None, :] + idio + bump * z[None, :]
    names = [f"A{i:02d}" for i in range(n_assets)]
    df = pd.DataFrame(rets, columns=names)
    chars_s = pd.Series(chars, index=names)
    return df, chars_s, mkt
