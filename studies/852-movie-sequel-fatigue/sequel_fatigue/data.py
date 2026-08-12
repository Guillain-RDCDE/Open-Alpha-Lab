"""Data layer for Study 852 — Movie-Sequel Fatigue.

The claim under test — **"franchise fatigue."** As a movie franchise grinds out sequel
after sequel, each new entry is supposed to open weaker than the last, and the reflex
corollary on the stock is that the *studio* should react worse to sequel N than it did to
sequel N-1 — and a run of tired sequels (a "down sequence") should hang over the next
entry's reception. We put that on the stand as a cross-event study of studio
opening-weekend abnormal returns, indexed by **sequel number** within a franchise.

Why this is a clean, zero-look-ahead calendar test (same construction as the sibling
[771-box-office-bomb](../../771-box-office-bomb/)):

* **The reveal is scheduled and public.** A wide theatrical release opens on a known
  Friday; the *weekend* box-office estimate is reported publicly on the following Sunday
  and the actuals on Monday. So the first trading session at which "how it opened" is
  common knowledge is the **Monday after the opening weekend**. We hardcode each film's
  real wide-release opening Friday (a verifiable fact) and anchor the studio-reaction
  window on the first trading session on/after the following Monday. The base of the
  reaction window is the opening-Friday close (before any weekend number exists), so a
  "did the studio move on the opening" measurement is look-ahead-free by construction.

Three ingredients, all offline-friendly once cached:

* **The franchise calendar, hardcoded.** ``EVENTS`` is a curated table of **46 franchise
  entries across 14 sub-franchise lines** — Marvel (Avengers, Guardians, Thor, Captain
  America, Ant-Man, Doctor Strange, Black Panther), the Star Wars sequel trilogy, Pirates
  of the Caribbean, Frozen (Disney / ``DIS``); Fast & Furious 5-10, Jurassic World,
  Despicable Me (Comcast-NBCUniversal / ``CMCSA``); Transformers (Paramount / ``PARA``).
  Each row carries the **franchise**, the **sequel number** (the film's ordinal within its
  franchise line — a public fact), the **real wide-release opening date**, and the
  **distributing studio's ticker** at that release. Dates cross-checked against Box Office
  Mojo / studio press releases. NOTE on early Marvel: Phase-1 films (*Iron Man* 1-2,
  *Thor*, *Captain America: The First Avenger*) were **Paramount**-distributed before
  Disney bought the distribution rights in 2010, so the Disney-era Thor/Cap entries keep
  their *true* franchise ordinal (Winter Soldier = Cap #2) while the pre-Disney #1 simply
  is not in the ``DIS`` reaction test — named, not silently renumbered.

* **Real tape.** Daily total-return (``auto_adjust=True``) closes for the three studio
  tickers (``DIS``, ``CMCSA``, ``PARA``) and the market benchmark ``SPY``, from yfinance
  (no key), cached as CSV under this study's own ``_cache/``. **Ticker-continuity, named
  honestly (like 707's carrier-start dates):** ``PARA`` (Paramount Global) has continuous
  adjusted history only from the ViacomCBS→Paramount-Global security dating **2021-02**, so
  the three pre-2021 Transformers entries (*Age of Extinction* 2014, *The Last Knight*
  2017, *Bumblebee* 2018) fall outside ``PARA`` coverage and are **dropped** from the
  reaction test — not back-filled — leaving only *Rise of the Beasts* (2023). ``DIS`` and
  ``CMCSA`` cover the whole table (back to 2003).

* **Synthetic world.** A deterministic, seeded price world (studio tape + benchmark tape)
  with a TUNABLE planted "fatigue" edge: at each synthetic entry's opening window the
  studio takes an abnormal drift that **declines with sequel number** (slope ``-edge``),
  optionally persistent (a down entry drags the next). ``edge = 0`` is the null world —
  the studio reaction carries no sequel-number information — and the cross-event
  slope/fatigue detector must NOT manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

START = "2003-01-01"
AS_OF = "2026-06-30"          # last complete calendar month at publication
BENCHMARK = "SPY"             # S&P 500 total-return proxy
STUDIO_TICKERS = ("DIS", "CMCSA", "PARA")
ALL_TICKERS = STUDIO_TICKERS + (BENCHMARK,)


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"sf_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# The hardcoded franchise calendar: (franchise, title, sequel_number, opening_date,
# studio_ticker). ``sequel_number`` is the film's ordinal within its franchise LINE (a
# public fact); ``studio_ticker`` is the DISTRIBUTOR whose P&L the film hit at release.
# Sources: Box Office Mojo, studio press releases, contemporaneous trade coverage. The
# weekend box-office is public by the following Sunday/Monday, so anchoring the reaction
# on the first session on/after the Monday-after-opening is calendar-known, zero
# look-ahead (see strategy.py). Disney-era Marvel keeps the TRUE ordinal even where the
# franchise's #1 was Paramount-distributed (Thor #1 2011, Cap #1 2011) and thus absent
# from the DIS reaction test.
# --------------------------------------------------------------------------- #
EVENTS: list[tuple[str, str, int, str, str]] = [
    # --- Marvel / Avengers (Disney) ---
    ("Avengers", "The Avengers", 1, "2012-05-04", "DIS"),
    ("Avengers", "Avengers: Age of Ultron", 2, "2015-05-01", "DIS"),
    ("Avengers", "Avengers: Infinity War", 3, "2018-04-27", "DIS"),
    ("Avengers", "Avengers: Endgame", 4, "2019-04-26", "DIS"),
    # --- Guardians of the Galaxy (Disney) ---
    ("Guardians", "Guardians of the Galaxy", 1, "2014-08-01", "DIS"),
    ("Guardians", "Guardians of the Galaxy Vol. 2", 2, "2017-05-05", "DIS"),
    ("Guardians", "Guardians of the Galaxy Vol. 3", 3, "2023-05-05", "DIS"),
    # --- Thor (Disney era; Thor #1 2011 was Paramount, absent) ---
    ("Thor", "Thor: The Dark World", 2, "2013-11-08", "DIS"),
    ("Thor", "Thor: Ragnarok", 3, "2017-11-03", "DIS"),
    ("Thor", "Thor: Love and Thunder", 4, "2022-07-08", "DIS"),
    # --- Captain America (Disney era; Cap #1 2011 was Paramount, absent) ---
    ("Captain America", "Captain America: The Winter Soldier", 2, "2014-04-04", "DIS"),
    ("Captain America", "Captain America: Civil War", 3, "2016-05-06", "DIS"),
    ("Captain America", "Captain America: Brave New World", 4, "2025-02-14", "DIS"),
    # --- Ant-Man (Disney) ---
    ("Ant-Man", "Ant-Man", 1, "2015-07-17", "DIS"),
    ("Ant-Man", "Ant-Man and the Wasp", 2, "2018-07-06", "DIS"),
    ("Ant-Man", "Ant-Man and the Wasp: Quantumania", 3, "2023-02-17", "DIS"),
    # --- Doctor Strange (Disney) ---
    ("Doctor Strange", "Doctor Strange", 1, "2016-11-04", "DIS"),
    ("Doctor Strange", "Doctor Strange in the Multiverse of Madness", 2, "2022-05-06", "DIS"),
    # --- Black Panther (Disney) ---
    ("Black Panther", "Black Panther", 1, "2018-02-16", "DIS"),
    ("Black Panther", "Black Panther: Wakanda Forever", 2, "2022-11-11", "DIS"),
    # --- Star Wars sequel trilogy (Disney) ---
    ("Star Wars", "Star Wars: The Force Awakens", 1, "2015-12-18", "DIS"),
    ("Star Wars", "Star Wars: The Last Jedi", 2, "2017-12-15", "DIS"),
    ("Star Wars", "Star Wars: The Rise of Skywalker", 3, "2019-12-20", "DIS"),
    # --- Pirates of the Caribbean (Disney) ---
    ("Pirates", "Pirates of the Caribbean: The Curse of the Black Pearl", 1, "2003-07-09", "DIS"),
    ("Pirates", "Pirates of the Caribbean: Dead Man's Chest", 2, "2006-07-07", "DIS"),
    ("Pirates", "Pirates of the Caribbean: At World's End", 3, "2007-05-25", "DIS"),
    ("Pirates", "Pirates of the Caribbean: On Stranger Tides", 4, "2011-05-20", "DIS"),
    ("Pirates", "Pirates of the Caribbean: Dead Men Tell No Tales", 5, "2017-05-26", "DIS"),
    # --- Frozen (Disney) ---
    ("Frozen", "Frozen", 1, "2013-11-27", "DIS"),
    ("Frozen", "Frozen II", 2, "2019-11-22", "DIS"),
    # --- Fast & Furious 5-10 (Comcast / Universal) ---
    ("Fast & Furious", "Fast Five", 5, "2011-04-29", "CMCSA"),
    ("Fast & Furious", "Fast & Furious 6", 6, "2013-05-24", "CMCSA"),
    ("Fast & Furious", "Furious 7", 7, "2015-04-03", "CMCSA"),
    ("Fast & Furious", "The Fate of the Furious", 8, "2017-04-14", "CMCSA"),
    ("Fast & Furious", "F9", 9, "2021-06-25", "CMCSA"),
    ("Fast & Furious", "Fast X", 10, "2023-05-19", "CMCSA"),
    # --- Jurassic World (Comcast / Universal) ---
    ("Jurassic World", "Jurassic World", 1, "2015-06-12", "CMCSA"),
    ("Jurassic World", "Jurassic World: Fallen Kingdom", 2, "2018-06-22", "CMCSA"),
    ("Jurassic World", "Jurassic World Dominion", 3, "2022-06-10", "CMCSA"),
    # --- Despicable Me (Comcast / Universal / Illumination) ---
    ("Despicable Me", "Despicable Me 2", 2, "2013-07-03", "CMCSA"),
    ("Despicable Me", "Despicable Me 3", 3, "2017-06-30", "CMCSA"),
    ("Despicable Me", "Despicable Me 4", 4, "2024-07-03", "CMCSA"),
    # --- Transformers (Paramount; PARA continuous only from 2021 -> pre-2021 drop) ---
    ("Transformers", "Transformers: Age of Extinction", 4, "2014-06-27", "PARA"),
    ("Transformers", "Transformers: The Last Knight", 5, "2017-06-21", "PARA"),
    ("Transformers", "Bumblebee", 6, "2018-12-21", "PARA"),
    ("Transformers", "Transformers: Rise of the Beasts", 7, "2023-06-09", "PARA"),
]


def events_frame() -> pd.DataFrame:
    """The curated calendar as a frame: franchise, title, seq, opening (Timestamp), ticker."""
    df = pd.DataFrame(EVENTS, columns=["franchise", "title", "seq", "opening", "ticker"])
    df["opening"] = pd.to_datetime(df["opening"])
    return df.sort_values(["franchise", "seq"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str = "2026-07-01", retries: int = 4) -> None:
    """Download total-return daily closes for the studio tickers + SPY; cache them.

    Retries with linear backoff — Yahoo rate-limits transient bursts, so a first empty
    frame is usually cured by a short wait, not a real "no such ticker".
    """
    import time

    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in ALL_TICKERS:
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
    return all(os.path.exists(_cache_path(t)) for t in ALL_TICKERS)


def load_real(asof: str = AS_OF) -> dict[str, pd.Series]:
    """Cached {ticker: total-return-close Series}, each sliced to [START, asof]."""
    out: dict[str, pd.Series] = {}
    for t in ALL_TICKERS:
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True).sort_index()
        s = df["Close"]
        out[t] = s[(s.index >= pd.Timestamp(START)) & (s.index <= pd.Timestamp(asof))]
    return out


# --------------------------------------------------------------------------- #
# Synthetic world — planted fatigue edge (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(edge: float = 0.0, persist: float = 0.0, seed: int = 852,
                    n_franchises: int = 14, entries_per: int = 4,
                    spacing: int = 130, n_days: int = 6000,
                    car_k: int = 3, daily_vol: float = 0.013,
                    shock_sd: float = 0.0, start: str = "2005-01-03",
                    ) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Deterministic paired (studio, benchmark) price world with a planted fatigue edge.

    Both tapes are correlated (rho ~ 0.5, a single large-cap vs SPY) log-return random
    walks. ``n_franchises`` synthetic franchises each release ``entries_per`` entries
    (sequel numbers ``1..entries_per``), spaced ``spacing`` business days apart. On each
    entry's opening window (the anchor session through ``anchor + car_k``) the studio
    tape takes an EXTRA abnormal drift

        drift = -edge * (seq - mean_seq)  +  persist * (previous entry's planted drift)

    so — with ``edge > 0`` — later sequels react MORE negatively (declining CAR with
    sequel number, the "fatigue" slope) and a down entry drags the next (``persist``).
    ``edge = persist = 0`` is the null world: the studio reaction carries no
    sequel-number information. The detector (a cross-event slope on the reaction CAR)
    must NOT reach significance on the null.

    Business-day index, span ~24 years (n_days below the pandas ns-timestamp horizon).
    Returns (studio_close, benchmark_close, events_df[franchise, seq, opening]).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    rho = 0.5
    common = rng.normal(0.0, daily_vol, n_days)
    idio_a = rng.normal(0.0, daily_vol, n_days)
    idio_b = rng.normal(0.0, daily_vol * 0.7, n_days)
    a = rho * common + np.sqrt(1 - rho ** 2) * idio_a       # studio log-returns
    b = rho * common + np.sqrt(1 - rho ** 2) * idio_b       # benchmark log-returns

    mean_seq = (1 + entries_per) / 2.0
    rows = []
    margin = 60
    pos = margin
    for f in range(n_franchises):
        prev_extra = 0.0          # the persistent, edge-independent part of the reaction
        for seq in range(1, entries_per + 1):
            if pos + car_k + 5 >= n_days:
                break
            # persistent reaction component (an AR(1) on a fresh per-entry shock): with
            # persist>0 and shock_sd>0 a down entry drags the next -> the H2 "fatigue
            # sequence" effect, independent of the H1 sequel-number slope.
            extra = persist * prev_extra + rng.normal(0.0, shock_sd)
            drift = -edge * (seq - mean_seq) + extra
            # spread the opening reaction across the anchor..anchor+car_k window
            per_day = drift / max(car_k + 1, 1)
            for kk in range(car_k + 1):
                a[pos + kk] += per_day
            # opening = a calendar Timestamp 3 days before the anchor session idx[pos];
            # build_event_cars snaps (opening + 3 days) forward to exactly idx[pos].
            rows.append({"franchise": f"SYN{f:02d}", "seq": seq,
                         "opening": idx[pos] - pd.Timedelta(days=3)})
            prev_extra = extra
            pos += spacing
        pos += spacing // 2   # gap between franchises

    close_a = pd.Series(100.0 * np.exp(np.cumsum(a)), index=idx)
    close_b = pd.Series(100.0 * np.exp(np.cumsum(b)), index=idx)
    ev = pd.DataFrame(rows)
    return close_a, close_b, ev
