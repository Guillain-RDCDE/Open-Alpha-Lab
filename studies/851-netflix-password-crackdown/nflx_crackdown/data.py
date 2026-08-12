"""Data layer for Study 851 — Netflix Password Crackdown.

The claim under test, steelmanned: Netflix's 2023 **paid-sharing ("password
crackdown")** was widely feared to spike churn, yet it became an **upside surprise** —
a "scary policy that worked". A single-name news-reaction event study measures NFLX's
**abnormal returns** (vs SPY / QQQ) around the public rollout dates and the earnings
that confirmed the subscriber gains. This is honest by construction: **N = 5 events**,
so the test has almost no statistical power — a case study, not a factor.

Three ingredients, all offline-friendly once cached.

* **The crackdown calendar, hardcoded from public record.** ``EVENTS`` is a curated
  table of the **five** market-facing dates of the paid-sharing story (2022-04 first
  flag → 2023-10 confirmation), exactly like the sibling studies that hand-build a
  small event calendar (``707-plane-crash-effect``'s ``DISASTERS``,
  ``299-keynote-drift``'s keynote table). Each row stores the **first NYSE session that
  could trade the news** (earnings print *after* the close, so the reaction session is
  the *next* morning — the study's single, documented execution lag), the underlying
  announcement date, and a one-line public-record note with its source. No look-ahead:
  the news is public before that session's close by construction.

* **Real tape.** Daily total-return adjusted closes for **NFLX** and two benchmarks —
  **SPY** (broad market) and **QQQ** (tech-heavy, the more apt comparison for a
  mega-cap streaming name) — from yfinance (no key), cached as CSV under this study's
  own ``_cache/``. ``auto_adjust=True`` folds in splits/dividends (total-return).

* **Synthetic world.** A deterministic, seeded market-plus-name tape with a TUNABLE
  planted event-day jump (``edge`` in daily-return units) on scheduled pseudo-event
  dates. ``edge = 0`` is the null world — event days statistically identical to every
  other day — and the market-model event-study machinery must NOT manufacture
  significance from it. ``edge > 0`` plants a clean, detectable jump the detector must
  recover.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

TICKERS = ("NFLX", "SPY", "QQQ")
CACHE = {t: os.path.join(CACHE_DIR, f"npc_{t.lower()}.csv") for t in TICKERS}

START = "2015-01-01"        # long history: estimation windows + a deep placebo pool
AS_OF = "2026-06-30"        # last complete calendar month at publication

__all__ = [
    "TICKERS", "START", "AS_OF", "CACHE_DIR",
    "EVENTS", "event_table",
    "fetch", "have_real", "load_real", "synthetic_world",
]

# --------------------------------------------------------------------------- #
# Hardcoded crackdown calendar — public record.
#
# Each row: (react_date, label, announce_date, note). ``react_date`` is the first NYSE
# session that could trade the news; ``announce_date`` is the underlying public
# announcement. Netflix reports quarterly results *after* the US close, so an earnings
# reaction lands on the NEXT session — that snap is the study's single documented
# execution lag (there is zero look-ahead: the print is public before the react
# session's close). Intraday policy announcements (the LatAm test, the US rollout email)
# land same-day. Sources are the Netflix quarterly shareholder letters / press releases
# and same-day reporting; all dates are matters of public record.
# --------------------------------------------------------------------------- #
EVENTS: list[tuple[str, str, str, str]] = [
    ("2022-04-20", "Q1'22 letter first flags paid-sharing (amid a sub miss)",
     "2022-04-19", "Q1'22 shareholder letter (after close 2022-04-19) says Netflix will "
     "start charging for account sharing; reported a first subscriber decline in a "
     "decade — the stock fell ~35% the next session. Source: Netflix Q1'22 letter."),
    ("2022-08-22", "LatAm 'add-a-home' paid-sharing test announced",
     "2022-08-22", "Netflix begins testing an 'add a home' paid-sharing charge in "
     "Argentina, Dominican Republic, El Salvador, Guatemala & Honduras (announced "
     "2022-08-22). Source: Netflix Help Center / same-day press."),
    ("2023-05-23", "Broad US paid-sharing rollout announced",
     "2023-05-23", "Netflix emails US members that account sharing outside a household "
     "now costs an extra $7.99/mo — the broad US 'password crackdown' rollout "
     "(2023-05-23). Source: Netflix newsroom, 2023-05-23."),
    ("2023-07-20", "Q2'23 letter confirms +5.9M subs (amid a revenue miss)",
     "2023-07-19", "Q2'23 shareholder letter (after close 2023-07-19) reports +5.9M "
     "paid net adds as the crackdown lands, but revenue slightly missed — the stock "
     "fell ~8% the next session. Source: Netflix Q2'23 letter."),
    ("2023-10-19", "Q3'23 letter: +8.8M subs, the crackdown pays off",
     "2023-10-18", "Q3'23 shareholder letter (after close 2023-10-18) reports +8.8M "
     "paid net adds — the largest quarterly gain since 2020 — plus a US price rise; "
     "the stock rose ~16% the next session. Source: Netflix Q3'23 letter."),
]


def event_table() -> pd.DataFrame:
    """The curated calendar as a frame: ``date`` (react session Timestamp), ``label``,
    ``announce`` (Timestamp), ``note``."""
    df = pd.DataFrame(EVENTS, columns=["date", "label", "announce", "note"])
    df["date"] = pd.to_datetime(df["date"])
    df["announce"] = pd.to_datetime(df["announce"])
    return df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str = "2026-07-01", retries: int = 4) -> None:
    """Download NFLX + SPY + QQQ total-return closes; cache them as CSV.

    Network; run once. ``auto_adjust=True`` so closes fold in splits/dividends
    (total-return). Retries up to ``retries`` times per ticker on a transient failure.
    """
    import time

    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in TICKERS:
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                df = yf.download(t, start=start, end=end, auto_adjust=True,
                                 progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                out = df[["Close"]].dropna()
                if out.empty:
                    raise RuntimeError(f"empty frame for {t}")
                out.to_csv(CACHE[t])
                break
            except Exception as e:  # pragma: no cover - network path
                last_err = e
                time.sleep(2.0 * (attempt + 1))
        else:  # pragma: no cover - network path
            raise RuntimeError(f"failed to fetch {t}: {last_err}")


def have_real() -> bool:
    return all(os.path.exists(p) for p in CACHE.values())


def load_real(asof: str = AS_OF) -> dict[str, pd.Series]:
    """Cached ``{ticker: close_series}`` sliced to ``[START, asof]``. OFFLINE."""
    out: dict[str, pd.Series] = {}
    for t, path in CACHE.items():
        s = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()["Close"]
        out[t] = s.loc[(s.index >= START) & (s.index <= asof)]
    return out


# --------------------------------------------------------------------------- #
# Synthetic world — planted event-day jump (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(
    edge: float = 0.0,
    seed: int = 851,
    n_days: int = 2600,
    n_events: int = 30,
    mkt_vol: float = 0.010,
    idio_vol: float = 0.018,
    beta: float = 1.25,
    drift: float = 0.08 / 252,
    start: str = "2015-01-05",
) -> tuple[pd.Series, pd.Series, pd.DatetimeIndex]:
    """A reproducible daily (asset, market) tape with a TUNABLE planted event jump.

    The market is an i.i.d. normal random walk (std ``mkt_vol``). The asset is a genuine
    one-factor name: ``r_asset = drift + beta * r_mkt + idio`` (idiosyncratic std
    ``idio_vol``), so a market-model event study has real betas to estimate. On each of
    ``n_events`` scheduled pseudo-event dates (spread well away from the edges) the
    asset takes an extra ``edge`` return bump on the event session only — a clean,
    one-day abnormal jump, the shape the "upside surprise" claim implies. ``edge = 0``
    is the null: event sessions are statistically identical to every other session, and
    the detector must NOT reach significance.

    Business-day index, span ~10 years — far below the pandas ns-timestamp horizon.
    Returns ``(asset_close, market_close, event_date_index)``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)

    r_mkt = rng.normal(0.0, mkt_vol, n_days)
    idio = rng.normal(0.0, idio_vol, n_days)
    r_asset = drift + beta * r_mkt + idio

    margin = 200
    pool = np.arange(margin, n_days - margin)
    locs = np.sort(rng.choice(pool, size=min(n_events, pool.size), replace=False))
    for loc in locs:
        r_asset[loc] += edge                      # the planted one-day abnormal jump

    asset = pd.Series(100.0 * np.cumprod(1.0 + r_asset), index=idx, name="asset")
    mkt = pd.Series(100.0 * np.cumprod(1.0 + r_mkt), index=idx, name="mkt")
    return asset, mkt, idx[locs]
