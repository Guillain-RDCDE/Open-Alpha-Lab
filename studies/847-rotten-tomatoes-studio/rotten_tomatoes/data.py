"""Data layer for Study 847 — Rotten-Tomatoes -> Studio.

The claim under test: **a big film's critic reception moves its distributing studio's
stock.** The folklore version is loud in both directions — "a rotten-scored flop tanks
the studio" and "a fresh-scored hit pops it." We steelman it as a clean, tier-conditioned
event study: is the studio's abnormal return around a film's release *systematically
different* when the film is critically fresh vs. critically rotten?

Three ingredients, all offline-friendly once cached:

* **The film table, hardcoded.** ``FILMS`` is a curated table of **~40 major wide
  releases, 2022 -> 2025**, each with (title, studio ticker, real opening/premiere date,
  a coarse public critic *tier* and an approximate public Rotten-Tomatoes score). No
  free, machine-readable "major release + RT score + distributor ticker" panel exists, so
  — exactly like the sibling studies that hand-build an event calendar
  (``771-box-office-bomb``'s ``EVENTS``, ``707-plane-crash-effect``'s ``DISASTERS``) —
  this is a hand-built table of releases any moviegoer would recognise, cross-referenced
  against Rotten Tomatoes (the Tomatometer), Box Office Mojo (opening date + distributor)
  and contemporaneous trade coverage. **Only clearly-fresh (>= 75) or clearly-rotten
  (< 50) films are kept** — mixed-reception 50-74 titles are deliberately excluded so the
  fresh-vs-rotten contrast is unambiguous. The analysis uses **only the tier bucket**;
  the stored score is the approximate public consensus at publication, for reference.

* **Real tape.** Daily total-return closes for the six distributing studios — Disney
  (DIS), Warner Bros. Discovery (WBD), Paramount (PARA), Comcast/Universal (CMCSA),
  Netflix (NFLX) and Sony (SONY) — plus SPY as the market benchmark, all from yfinance
  (no key), cached as CSV under the study's own ``_cache/``. **Ticker-existence is named
  honestly:** WBD only began trading 2022-04-11 (the WarnerMedia-Discovery merger) and
  PARA is the post-2022-02 Paramount Global re-brand, so the film table's WBD/PARA titles
  are all dated *after* those tickers existed. **Studio != film label:** each studio is a
  huge conglomerate (parks, ISPs, streaming, TV, games) for which any one film is a tiny
  revenue sliver — that is precisely the null the desk expects.

* **Synthetic world.** A deterministic, seeded paired (studio, benchmark) log-return
  world with a TUNABLE planted *tier drift*: on rotten pseudo-events the studio takes an
  extra negative post-release drift, on fresh ones an extra positive one, scaled by
  ``edge``. ``edge = 0`` is the null world — the tier-conditioned detector must NOT
  manufacture a fresh-minus-rotten gap from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

AS_OF = "2026-06-30"      # last complete calendar month at publication
BENCHMARK = "SPY"        # S&P 500 total-return proxy
STUDIOS = ("DIS", "WBD", "PARA", "CMCSA", "NFLX", "SONY")

FRESH_MIN = 75           # Tomatometer >= 75 -> "fresh" tier
ROTTEN_MAX = 50          # Tomatometer <  50 -> "rotten" tier (50-74 excluded as "mixed")

# --------------------------------------------------------------------------- #
# The curated film table, hardcoded.
#
#   (title, studio_ticker, opening_date, tier, approx_rt)
#
# * ``opening_date`` is the real US wide-release Friday (Box Office Mojo / studio
#   releases); for the Netflix (NFLX) rows it is the streaming-premiere date — Netflix
#   has no theatrical "opening weekend", so its event is the day the film DROPS and its
#   reception is fully public (named honestly; see references.md). The event-study code
#   snaps this date to the first NYSE session on/after it (a Friday theatrical open and a
#   Wednesday/holiday open alike roll to the first tradable session) — that snap is the
#   study's single documented execution lag (see strategy.py).
# * ``tier`` is "fresh" (Tomatometer >= 75) or "rotten" (< 50); mixed 50-74 titles are
#   intentionally absent. ``approx_rt`` is the approximate public Tomatometer at
#   publication, stored for reference ONLY — the analysis reads the tier bucket, never the
#   number. Coverage window chosen so every ticker (incl. WBD from 2022-04-11 and the
#   post-2022-02 Paramount Global PARA) actually traded on each film's date.
# * Sources: Rotten Tomatoes (Tomatometer), Box Office Mojo (opening date + distributor),
#   contemporaneous trade press (Variety / THR / Deadline). Public record.
# --------------------------------------------------------------------------- #
FILMS: list[tuple[str, str, str, str, int]] = [
    # --- Disney (DIS) ---
    ("Black Panther: Wakanda Forever", "DIS",  "2022-11-11", "fresh",  84),
    ("Inside Out 2",                   "DIS",  "2024-06-14", "fresh",  91),
    ("Deadpool & Wolverine",           "DIS",  "2024-07-26", "fresh",  78),
    ("Ant-Man and the Wasp: Quantumania", "DIS", "2023-02-17", "rotten", 46),
    ("Wish",                           "DIS",  "2023-11-22", "rotten", 48),
    ("Snow White",                     "DIS",  "2025-03-21", "rotten", 39),
    # --- Warner Bros. Discovery (WBD, trades from 2022-04-11) ---
    ("Barbie",                         "WBD",  "2023-07-21", "fresh",  88),
    ("Dune: Part Two",                 "WBD",  "2024-03-01", "fresh",  92),
    ("Wonka",                          "WBD",  "2023-12-15", "fresh",  82),
    ("Sinners",                        "WBD",  "2025-04-18", "fresh",  97),
    ("Black Adam",                     "WBD",  "2022-10-21", "rotten", 39),
    ("Aquaman and the Lost Kingdom",   "WBD",  "2023-12-22", "rotten", 33),
    ("Joker: Folie a Deux",            "WBD",  "2024-10-04", "rotten", 32),
    ("A Minecraft Movie",              "WBD",  "2025-04-04", "rotten", 47),
    # --- Paramount (PARA, post-2022-02 Paramount Global) ---
    ("Top Gun: Maverick",              "PARA", "2022-05-27", "fresh",  96),
    ("Dungeons & Dragons: Honor Among Thieves", "PARA", "2023-03-31", "fresh", 91),
    ("Mission: Impossible - Dead Reckoning Part One", "PARA", "2023-07-12", "fresh", 96),
    ("Smile",                          "PARA", "2022-09-30", "fresh",  79),
    ("Bob Marley: One Love",           "PARA", "2024-02-14", "rotten", 43),
    ("IF",                             "PARA", "2024-05-17", "rotten", 48),
    # --- Comcast / Universal (CMCSA) ---
    ("Nope",                           "CMCSA", "2022-07-22", "fresh",  83),
    ("Puss in Boots: The Last Wish",   "CMCSA", "2022-12-21", "fresh",  95),
    ("M3GAN",                          "CMCSA", "2023-01-06", "fresh",  93),
    ("Oppenheimer",                    "CMCSA", "2023-07-21", "fresh",  93),
    ("Wicked",                         "CMCSA", "2024-11-22", "fresh",  88),
    ("Jurassic World Dominion",        "CMCSA", "2022-06-10", "rotten", 29),
    ("Halloween Ends",                 "CMCSA", "2022-10-14", "rotten", 39),
    ("Five Nights at Freddy's",        "CMCSA", "2023-10-27", "rotten", 33),
    # --- Sony (SONY) ---
    ("Spider-Man: Across the Spider-Verse", "SONY", "2023-06-02", "fresh", 95),
    ("28 Years Later",                 "SONY", "2025-06-20", "fresh",  89),
    ("Morbius",                        "SONY", "2022-04-01", "rotten", 15),
    ("Madame Web",                     "SONY", "2024-02-14", "rotten", 11),
    ("Kraven the Hunter",              "SONY", "2024-12-13", "rotten", 15),
    ("Venom: The Last Dance",          "SONY", "2024-10-25", "rotten", 39),
    # --- Netflix (NFLX, streaming-premiere date; see references.md) ---
    ("Glass Onion: A Knives Out Mystery", "NFLX", "2022-12-23", "fresh", 92),
    ("Society of the Snow",            "NFLX", "2024-01-04", "fresh",  90),
    ("Carry-On",                       "NFLX", "2024-12-13", "fresh",  92),
    ("The Gray Man",                   "NFLX", "2022-07-22", "rotten", 46),
    ("Rebel Moon - Part One",          "NFLX", "2023-12-22", "rotten", 22),
    ("The Electric State",             "NFLX", "2025-03-14", "rotten", 16),
]


def film_table() -> pd.DataFrame:
    """The curated table as a frame: ``title``, ``studio``, ``date`` (Timestamp),
    ``tier`` ("fresh"/"rotten"), ``rt`` (approx Tomatometer)."""
    df = pd.DataFrame(FILMS, columns=["title", "studio", "date", "tier", "rt"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"rt_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def all_tickers() -> list[str]:
    return [BENCHMARK, *STUDIOS]


def fetch(start: str = "2021-06-01", end: str = "2026-07-01", retries: int = 4) -> None:
    """Download total-return (``auto_adjust``) daily closes for SPY + the six studios; cache.

    Network; run once. Retries with linear backoff — Yahoo rate-limits transient bursts, so
    a first empty frame is usually cured by a short wait rather than a real "no such ticker".
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
    """Cached {ticker: adjusted-close Series}, each sliced to <= asof (drops the partial
    current month per the desk's as-of discipline)."""
    out = {}
    for t in all_tickers():
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True).sort_index()
        s = df["Close"]
        out[t] = s[s.index <= pd.Timestamp(asof)]
    return out


# --------------------------------------------------------------------------- #
# Synthetic world — planted tier drift (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(edge: float = 0.0, seed: int = 847, n_per_tier: int = 20,
                    n_days: int = 4000, spacing: int = 45, post: int = 6,
                    ) -> tuple[pd.Series, pd.Series, list[tuple[int, str]]]:
    """Deterministic paired (studio, benchmark) log-return world with a planted tier drift.

    Both series share a common market factor (rho ~ 0.5, like a single large-cap name vs
    SPY) plus idiosyncratic noise. Pseudo-events are laid on a regular grid and alternately
    labelled "fresh"/"rotten"; on the ``post`` sessions AFTER each event the studio takes an
    extra drift of ``+edge`` (fresh) or ``-edge`` (rotten) per day. ``edge = 0`` is the null
    world — fresh and rotten events are statistically identical, and the tier-conditioned
    detector must NOT reach significance.

    Integer RangeIndex (no timestamp horizon risk). Returns
    (studio_logret, bench_logret, [(event_pos, tier), ...]).
    """
    rng = np.random.default_rng(seed)
    rho = 0.5
    common = rng.normal(0.0, 0.010, n_days)
    idio_s = rng.normal(0.0, 0.014, n_days)
    idio_b = rng.normal(0.0, 0.007, n_days)
    studio = rho * common + np.sqrt(1 - rho**2) * idio_s
    bench = rho * common + np.sqrt(1 - rho**2) * idio_b

    n_events = 2 * n_per_tier
    positions = list(range(spacing, n_days - post - 2, spacing))[:n_events]
    events: list[tuple[int, str]] = []
    for i, p in enumerate(positions):
        tier = "fresh" if i % 2 == 0 else "rotten"
        sign = 1.0 if tier == "fresh" else -1.0
        for k in range(1, post + 1):          # planted drift over the following-week window
            if p + k < n_days:
                studio[p + k] += sign * edge
        events.append((p, tier))

    idx = pd.RangeIndex(n_days)
    return pd.Series(studio, index=idx), pd.Series(bench, index=idx), events
