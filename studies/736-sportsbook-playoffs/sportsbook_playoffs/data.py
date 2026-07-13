"""Data layer for Study 736 — Sportsbook-Playoffs.

The claim under test: sportsbook / iGaming stocks — DraftKings (DKNG) and the wider
betting basket — **rally into** the two biggest US betting seasons, NFL playoffs
(January) and March Madness (mid-March), as anticipation of a wall of betting handle
lifts the stocks in the weeks *before* the games start.

Four ingredients, all offline-friendly once cached:

* **The event calendar, hardcoded.** ``EVENTS`` is the first-game date of each NFL
  Wild-Card weekend and the men's March-Madness Round-of-64, **2021 -> 2026** — the
  window in which DraftKings has actually traded as an operating company (its SPAC
  merger closed 2020-04-24, so the first betting season it saw as a public stock is
  January 2021). Both calendars are **fixed and public months in advance** (the NFL
  regular-season schedule and the NCAA tournament dates are set long before the season
  ends), so a "buy N days before the first game" rule has **zero look-ahead** — this is
  a calendar-known rule, like a turn-of-month window, and needs no execution lag. Source:
  Pro-Football-Reference / NFL.com playoff schedules and NCAA.com tournament brackets
  (cross-checked against contemporary reporting for the exact first-game date).

* **Real tape.** Daily total-return closes (yfinance ``auto_adjust=True``) for DKNG plus
  a 5-name US sportsbook / casino-iGaming basket — DraftKings (DKNG), Penn Entertainment
  (PENN), Caesars (CZR), MGM Resorts (MGM), Rush Street Interactive (RSI) — and BETZ, the
  Roundhill Sports Betting & iGaming ETF, as a single packaged "pure-play" vehicle. SPY
  is the market benchmark for the beta-adjusted robustness check. **Named honestly:**
  DKNG's cached history reaches back to 2019-07-25, but that early tape is the *Diamond
  Eagle (DEAC) SPAC trust* — a ~$10 cash shell, not the operating DraftKings — so the
  study slices DKNG from the **2020-04-24 merger close** (``DKNG_START``); every basket
  member and BETZ already trade before the first event (Jan 2021), so all 12 events have
  full basket coverage.

* **A betting-handle seasonality PROXY — clearly labelled, not real tape.**
  ``HANDLE_SEASONALITY`` is a small hardcoded 12-month shape of US commercial
  sports-betting handle, normalised to a mean of 1.0, approximating the well-documented
  seasonal pattern (a big Sep->Jan NFL hump, a March bump, a summer trough) reported in
  American Gaming Association / state-regulator monthly handle releases. It is used
  **only** to motivate *why* one might expect the rally (the betting activity really is
  seasonal); it is an illustrative approximation, never priced, never traded, and never
  presented under a real-tape banner.

* **Synthetic world.** A deterministic, seeded random-walk tape with a TUNABLE planted
  pre-event "run-up" bump on a scheduled event calendar. ``bump = 0`` is the null world —
  run-up windows statistically identical to the rest; the one-sample-t machinery must NOT
  manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

# Instruments
DKNG = "DKNG"
BASKET_TICKERS = ("DKNG", "PENN", "CZR", "MGM", "RSI")   # US sportsbook / casino-iGaming
ETF = "BETZ"                                              # Roundhill Sports Betting & iGaming
BENCH = "SPY"                                             # market benchmark (beta-adj check)
ALL_TICKERS = tuple(dict.fromkeys(BASKET_TICKERS + (ETF, BENCH)))

DKNG_START = "2020-04-24"   # SPAC-merger close: first day DKNG is the operating DraftKings
START = "2019-01-01"
AS_OF = "2026-06-30"        # last complete month at publication (2026-07-13)

# --------------------------------------------------------------------------- #
# Hardcoded event calendar: first-game dates of the two flagship US betting seasons.
# NFL = Wild-Card weekend opening game (Saturday); NCAA = men's Round-of-64 (Thursday,
# the first full-bracket day). Both are fixed & public months ahead -> a "buy N sessions
# before" rule has zero look-ahead. 2021->2026, the DKNG-as-operating-company era.
# --------------------------------------------------------------------------- #
EVENTS: list[tuple[str, str, str]] = [
    # season_start_date, family, label
    ("2021-01-09", "NFL",  "NFL Wild-Card weekend (2020 season)"),
    ("2021-03-19", "NCAA", "March Madness Round of 64 (2021)"),
    ("2022-01-15", "NFL",  "NFL Wild-Card weekend (2021 season)"),
    ("2022-03-17", "NCAA", "March Madness Round of 64 (2022)"),
    ("2023-01-14", "NFL",  "NFL Wild-Card weekend (2022 season)"),
    ("2023-03-16", "NCAA", "March Madness Round of 64 (2023)"),
    ("2024-01-13", "NFL",  "NFL Wild-Card weekend (2023 season)"),
    ("2024-03-21", "NCAA", "March Madness Round of 64 (2024)"),
    ("2025-01-11", "NFL",  "NFL Wild-Card weekend (2024 season)"),
    ("2025-03-20", "NCAA", "March Madness Round of 64 (2025)"),
    ("2026-01-10", "NFL",  "NFL Wild-Card weekend (2025 season)"),
    ("2026-03-19", "NCAA", "March Madness Round of 64 (2026)"),
]


def event_table() -> pd.DataFrame:
    """The event calendar as a frame: ``date`` (Timestamp), ``family``, ``label``."""
    df = pd.DataFrame(EVENTS, columns=["date", "family", "label"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# LABELLED PROXY — US commercial sports-betting handle, monthly seasonal shape.
# NOT real tape, never traded: an illustrative 12-month multiplier (mean 1.0)
# approximating the AGA / state-regulator seasonal pattern (NFL season Sep->Jan is the
# handle peak, a March-Madness bump, a summer trough). Used only to show the *premise*
# of the claim (betting activity is genuinely seasonal) in Beat 2.
# --------------------------------------------------------------------------- #
HANDLE_SEASONALITY: dict[int, float] = {
    1: 1.45,   # January — NFL playoffs + championship run-in, the annual peak
    2: 1.25,   # February — Super Bowl month
    3: 1.20,   # March — March Madness bump
    4: 0.80,   # April — post-tournament trough begins
    5: 0.72,   # May
    6: 0.68,   # June — summer low (only baseball / niche)
    7: 0.66,   # July — annual trough
    8: 0.80,   # August — NFL preseason ramp
    9: 1.15,   # September — NFL kickoff
    10: 1.30,  # October — NFL + MLB playoffs + NBA/NHL tip-off
    11: 1.35,  # November — full NFL/NBA/NHL/CFB slate
    12: 1.34,  # December — NFL stretch run + CFB bowls
}


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"sbp_{ticker.lower()}.csv")


def fetch(start: str = START, end: str = "2026-07-01") -> None:
    """Download total-return daily closes for every ticker; cache them (network, once).

    ``auto_adjust=True`` so closes fold in splits and dividends (total-return, not
    price-only) — the event-study returns below are then plain ``pct_change()``.
    """
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in ALL_TICKERS:
        d = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d[["Close"]].dropna().to_csv(_cache_path(t))


def have_real() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in ALL_TICKERS)


def load_real(asof: str = AS_OF) -> dict[str, pd.Series]:
    """Cached {ticker: total-return close Series}, each sliced to <= asof.

    DKNG is additionally floored at ``DKNG_START`` (the SPAC-merger close) so the flat
    ~$10 DEAC cash-shell tape never enters the operating-company return series.
    """
    out: dict[str, pd.Series] = {}
    for t in ALL_TICKERS:
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True).sort_index()
        s = df["Close"]
        s = s[s.index <= pd.Timestamp(asof)]
        if t == DKNG:
            s = s[s.index >= pd.Timestamp(DKNG_START)]
        out[t] = s
    return out


# --------------------------------------------------------------------------- #
# Synthetic world — planted pre-event run-up bump (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, seed: int = 736,
                    n_days: int = 1500, n_events: int = 12,
                    daily_vol: float = 0.035, run_up: int = 10,
                    start: str = "2020-06-01",
                    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """A reproducible daily "DKNG-like" random-walk tape with a TUNABLE planted run-up.

    A random walk in log returns (i.i.d. normal, std ``daily_vol`` — betting stocks are
    genuinely ~3-4% daily vol) on which ``n_events`` scheduled "season start" dates are
    sprinkled (well away from the edges). Over the ``run_up`` sessions *before* each event
    the tape gets an extra ``bump / run_up`` log-return per day — a clean, mechanical
    rally-into-the-season, the shape the folklore claims. ``bump = 0`` is the null world:
    run-up windows are statistically identical to every other window, and the event-study
    detector must NOT reach significance.

    Business-day index, span ~6 years — far below the pandas ns-timestamp trap. Returns
    (frame with a ``Close`` column, event-date DatetimeIndex).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    log_ret = rng.normal(0.0, daily_vol, n_days)

    margin = 40
    pool = np.arange(margin, n_days - margin)
    locs = np.sort(rng.choice(pool, size=min(n_events, pool.size), replace=False))

    per_day = bump / max(run_up, 1)
    for loc in locs:
        for k in range(1, run_up + 1):        # the planted rally over the run-up window
            if loc - k >= 0:
                log_ret[loc - k] += per_day

    close = pd.Series(100.0 * np.exp(np.cumsum(log_ret)), index=idx)
    return pd.DataFrame({"Close": close}), idx[locs]
