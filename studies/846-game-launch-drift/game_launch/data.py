"""Data layer for Study 846 — Blockbuster Game-Launch Drift.

The claim under test. Gaming/finance folklore says a marquee AAA launch is a catalyst you
can trade: **"buy the hype into the launch"** and either ride the momentum or **"sell the
news"** afterward. We test the *finance* transplant: does the **publisher's** stock earn
an abnormal return around a blockbuster game's ship date and over the ~20-session drift
window that follows? The desk's prior is firmly **None/Weak** — a heavily-marketed,
scheduled product ship is exactly the catalyst a semi-strong-efficient market has already
priced, one title is a small slice of a large multi-franchise publisher's revenue, and
with only ~30 events the test has low power.

Five ingredients:

* **The launch calendar, hardcoded.** ~37 marquee AAA releases (2013→2024) mapped to the
  publisher whose stock they most plausibly move: **TTWO** (Rockstar/2K — GTA V, RDR2,
  Borderlands 3…), **EA** (Battlefield, Star Wars, Apex, Anthem…), **NTDOY** (Nintendo —
  Zelda BotW/TotK, Odyssey, Smash…), **UBSFY** (Ubisoft — Assassin's Creed, Far Cry,
  Watch Dogs…) and **ATVI** (Activision/Blizzard — CoD, Diablo IV, Overwatch). Each
  **launch date** is the rock-solid public fact we anchor on (standard-edition US street
  dates, cross-checked against each title's Wikipedia release box / publisher press
  releases).

* **The tradable instruments (yfinance).** ``TTWO``, ``EA``, ``NTDOY`` (Nintendo US ADR)
  and ``UBSFY`` (Ubisoft US ADR), each benchmarked against ``SPY`` (S&P 500 total return)
  so we measure the publisher's *abnormal* return, not the market's drift. Each event is
  anchored to *its own* publisher.

* **ATVI is delisted.** Microsoft closed its acquisition of Activision Blizzard on
  2023-10-13, so ``ATVI`` no longer trades and Yahoo! serves **no** price history for it.
  The five ATVI-published launches (Overwatch, CoD WWII, CoD MWII, Diablo IV) are kept in
  the calendar for the record but resolve to **excluded** in the event funnel (no tape) —
  an honest, auditable gap, not a silent drop. (Cyberpunk 2077 and Elden Ring, named in
  the folklore, publish to CD Projekt / Bandai Namco, which are not in this ticker set, so
  they are out of scope.)

* **No fundamental proxy needed.** "Buy the hype around the launch" is a pure price-path
  claim: the ship date *is* the event.

* **Synthetic world.** A deterministic, seeded paired (publisher, benchmark) log-return
  world with a TUNABLE planted "launch drift" on a synthetic calendar. ``drift = 0`` is the
  null world; the one-sample-t / Newey-West machinery must not manufacture significance.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build
the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

AS_OF = "2026-06-30"                          # last complete month at publication
PUBLISHERS = ["TTWO", "EA", "NTDOY", "UBSFY"]  # still-listed, fetchable via yfinance
DELISTED = ["ATVI"]                          # Activision Blizzard — no tape post 2023-10-13
BENCHMARK = "SPY"                            # S&P 500 total-return proxy

# --------------------------------------------------------------------------- #
# The launch calendar, hardcoded: (launch_date, publisher, title). The launch date is the
# anchor (an unambiguous public fact — the standard-edition US street date). ~37 marquee
# AAA releases, 2013->2024, mapped to the publisher whose stock they most plausibly move.
# ATVI titles are kept for the record but have NO tradable tape (ATVI delisted 2023-10-13
# after the Microsoft buyout) and resolve to EXCLUDED in the event funnel.
# Sources: each title's Wikipedia release box; publisher press releases.
# --------------------------------------------------------------------------- #
EVENTS = [
    # launch_date, publisher, title
    ("2013-09-17", "TTWO",  "Grand Theft Auto V"),
    ("2014-11-18", "TTWO",  "Grand Theft Auto V (next-gen)"),
    ("2015-04-14", "TTWO",  "Grand Theft Auto V (PC)"),
    ("2016-10-07", "TTWO",  "Mafia III"),
    ("2016-10-21", "TTWO",  "Civilization VI"),
    ("2018-10-26", "TTWO",  "Red Dead Redemption 2"),
    ("2019-09-13", "TTWO",  "Borderlands 3"),
    ("2016-10-21", "EA",    "Battlefield 1"),
    ("2017-03-21", "EA",    "Mass Effect: Andromeda"),
    ("2017-11-17", "EA",    "Star Wars Battlefront II"),
    ("2018-11-20", "EA",    "Battlefield V"),
    ("2019-02-04", "EA",    "Apex Legends"),
    ("2019-02-22", "EA",    "Anthem"),
    ("2019-11-15", "EA",    "Star Wars Jedi: Fallen Order"),
    ("2021-11-19", "EA",    "Battlefield 2042"),
    ("2023-04-28", "EA",    "Star Wars Jedi: Survivor"),
    ("2017-03-03", "NTDOY", "The Legend of Zelda: Breath of the Wild"),
    ("2017-10-27", "NTDOY", "Super Mario Odyssey"),
    ("2018-12-07", "NTDOY", "Super Smash Bros. Ultimate"),
    ("2019-11-15", "NTDOY", "Pokemon Sword/Shield"),
    ("2020-03-20", "NTDOY", "Animal Crossing: New Horizons"),
    ("2021-10-08", "NTDOY", "Metroid Dread"),
    ("2022-11-18", "NTDOY", "Pokemon Scarlet/Violet"),
    ("2023-05-12", "NTDOY", "The Legend of Zelda: Tears of the Kingdom"),
    ("2023-10-20", "NTDOY", "Super Mario Bros. Wonder"),
    ("2016-11-15", "UBSFY", "Watch Dogs 2"),
    ("2017-10-27", "UBSFY", "Assassin's Creed Origins"),
    ("2018-10-05", "UBSFY", "Assassin's Creed Odyssey"),
    ("2019-03-15", "UBSFY", "Tom Clancy's The Division 2"),
    ("2020-11-10", "UBSFY", "Assassin's Creed Valhalla"),
    ("2021-10-07", "UBSFY", "Far Cry 6"),
    ("2023-10-05", "UBSFY", "Assassin's Creed Mirage"),
    ("2024-08-30", "UBSFY", "Star Wars Outlaws"),
    ("2016-05-24", "ATVI",  "Overwatch"),
    ("2017-11-03", "ATVI",  "Call of Duty: WWII"),
    ("2022-10-28", "ATVI",  "Call of Duty: Modern Warfare II"),
    ("2023-06-06", "ATVI",  "Diablo IV"),
]


def all_tickers() -> list[str]:
    """Tickers with a real, fetchable tape (fetch/load handle these). ATVI is excluded —
    it has no post-buyout history."""
    return PUBLISHERS + [BENCHMARK]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"gamelaunch_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2013-01-01", end: str = "2026-07-01", retries: int = 4) -> None:
    """Download adjusted (total-return) daily closes for the four listed publishers + SPY;
    cache them. ATVI is intentionally NOT fetched (delisted; no tape).

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
    """Cached {ticker: adjusted-close Series}, each sliced to <= asof. ATVI is absent by
    construction (no tape)."""
    out = {}
    for t in all_tickers():
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True).sort_index()
        s = df["Close"]
        out[t] = s[s.index <= pd.Timestamp(asof)]
    return out


# --------------------------------------------------------------------------- #
# Synthetic world -- planted launch drift (null at drift = 0)
# --------------------------------------------------------------------------- #
def synthetic_world(drift: float = 0.0, seed: int = 846,
                    n_events: int = 32, n_days: int = 5200, spacing: int = 150,
                    ) -> tuple[pd.Series, pd.Series, list[int]]:
    """Deterministic paired (publisher, benchmark) log-return world with a planted launch
    drift.

    Both series are correlated (rho ~ 0.5, like a single mid-beta software name vs SPY)
    zero-mean noise; on each synthetic "launch day" (every ``spacing``-th business day) the
    publisher gets an EXTRA ``drift`` log-return spread across the launch day and the four
    sessions after it -- a planted post-launch drift. ``drift = 0`` is the null world; the
    detector must not manufacture significance from it.

    Business-day integer index (positions 0..n_days). Returns
    (pub_logret, bench_logret, launch_positions).
    """
    rng = np.random.default_rng(seed)
    rho = 0.5
    common = rng.normal(0.0, 0.010, n_days)
    idio_a = rng.normal(0.0, 0.015, n_days)
    idio_b = rng.normal(0.0, 0.008, n_days)
    a = rho * common + np.sqrt(1 - rho**2) * idio_a
    b = rho * common + np.sqrt(1 - rho**2) * idio_b

    key_pos = list(range(spacing, n_days - 60, spacing))[:n_events]
    for p in key_pos:
        # spread the planted drift over the launch day and the four sessions after it
        for off in (0, 1, 2, 3, 4):
            if p + off < n_days:
                a[p + off] += drift / 5.0
    idx = pd.RangeIndex(n_days)
    return pd.Series(a, index=idx), pd.Series(b, index=idx), key_pos
