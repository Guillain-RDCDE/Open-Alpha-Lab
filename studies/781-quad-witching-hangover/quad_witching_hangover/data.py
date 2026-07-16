"""Data layer for Study 781 — Quad-Witching-Hangover.

The claim under test: **the week *after* quad-witching underperforms.** "Quadruple
witching" is the quarterly event when four classes of derivatives expire at once — stock
index futures, stock index options, single-stock options, and single-stock futures — on
the **third Friday of March, June, September and December**. Dealers unwind large hedges
into that close; the folklore says the market is left "hungover" the following week —
depressed, drifting, prone to a give-back — after the expiration-day churn.

This is a pure **calendar event-study** (shape A). The event is a fixture of the
exchange calendar known *years* in advance (it is literally "third Friday of the quarter's
last month"), so a "sit out / short the week after quad-witching" rule is zero-look-ahead
by construction. We test whether SPY's own forward return over the 5 sessions **after**
the quad-witching Friday close is reliably below its ordinary weekly drift.

Four ingredients:

* **The quad-witching calendar, hardcoded.** Every quarterly quad-witching Friday from
  2005->2025 (84 events), each the third Friday of Mar/Jun/Sep/Dec. These are deterministic
  exchange-calendar facts (US options/futures expire the third Friday), cross-checked
  against the CBOE/CME expiration calendars — no estimation, no look-ahead.

* **The tradable instrument (yfinance).** ``SPY`` — the S&P 500 ETF, adjusted
  (total-return) daily closes. The claim is about the index's *own* forward path, so this is
  a **self-benchmarked** single-tape study: there is no cross-sectional benchmark leg. The
  "is it abnormal?" question is answered by a random-window placebo against SPY's own
  history (SPY drifts up ~+0.03%/session, so "underperform" means *below that drift*, which
  the placebo cloud captures directly).

* **No fundamental proxy needed.** "The week after quad-witching" is a pure price-path,
  calendar-anchored claim: the expiration Friday *is* the event.

* **Synthetic world.** A deterministic, seeded single-name log-return world (positive drift
  like SPY) with a TUNABLE planted "post-event hangover dip" on a synthetic quarterly
  calendar. ``dip = 0`` is the null world; the one-sample-t machinery must not manufacture
  significance from it, and a planted dip must be recovered monotonically.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

AS_OF = "2026-06-30"     # last complete month at publication
INSTRUMENT = "SPY"       # S&P 500 ETF, total-return (self-benchmarked; no separate leg)

# --------------------------------------------------------------------------- #
# The quad-witching calendar, hardcoded: (date, year, quarter-month). Quadruple witching
# falls on the third Friday of March, June, September and December, when stock-index
# futures, stock-index options, single-stock options and single-stock futures all expire
# together. These are deterministic exchange-calendar facts (US listed options/futures
# expire the third Friday), so "the week after quad-witching" is known years ahead and is
# zero-look-ahead by construction. Source: CBOE / CME quarterly expiration calendars.
# --------------------------------------------------------------------------- #
EVENTS = [
    ("2005-03-18", 2005, "Mar"), ("2005-06-17", 2005, "Jun"),
    ("2005-09-16", 2005, "Sep"), ("2005-12-16", 2005, "Dec"),
    ("2006-03-17", 2006, "Mar"), ("2006-06-16", 2006, "Jun"),
    ("2006-09-15", 2006, "Sep"), ("2006-12-15", 2006, "Dec"),
    ("2007-03-16", 2007, "Mar"), ("2007-06-15", 2007, "Jun"),
    ("2007-09-21", 2007, "Sep"), ("2007-12-21", 2007, "Dec"),
    ("2008-03-21", 2008, "Mar"), ("2008-06-20", 2008, "Jun"),
    ("2008-09-19", 2008, "Sep"), ("2008-12-19", 2008, "Dec"),
    ("2009-03-20", 2009, "Mar"), ("2009-06-19", 2009, "Jun"),
    ("2009-09-18", 2009, "Sep"), ("2009-12-18", 2009, "Dec"),
    ("2010-03-19", 2010, "Mar"), ("2010-06-18", 2010, "Jun"),
    ("2010-09-17", 2010, "Sep"), ("2010-12-17", 2010, "Dec"),
    ("2011-03-18", 2011, "Mar"), ("2011-06-17", 2011, "Jun"),
    ("2011-09-16", 2011, "Sep"), ("2011-12-16", 2011, "Dec"),
    ("2012-03-16", 2012, "Mar"), ("2012-06-15", 2012, "Jun"),
    ("2012-09-21", 2012, "Sep"), ("2012-12-21", 2012, "Dec"),
    ("2013-03-15", 2013, "Mar"), ("2013-06-21", 2013, "Jun"),
    ("2013-09-20", 2013, "Sep"), ("2013-12-20", 2013, "Dec"),
    ("2014-03-21", 2014, "Mar"), ("2014-06-20", 2014, "Jun"),
    ("2014-09-19", 2014, "Sep"), ("2014-12-19", 2014, "Dec"),
    ("2015-03-20", 2015, "Mar"), ("2015-06-19", 2015, "Jun"),
    ("2015-09-18", 2015, "Sep"), ("2015-12-18", 2015, "Dec"),
    ("2016-03-18", 2016, "Mar"), ("2016-06-17", 2016, "Jun"),
    ("2016-09-16", 2016, "Sep"), ("2016-12-16", 2016, "Dec"),
    ("2017-03-17", 2017, "Mar"), ("2017-06-16", 2017, "Jun"),
    ("2017-09-15", 2017, "Sep"), ("2017-12-15", 2017, "Dec"),
    ("2018-03-16", 2018, "Mar"), ("2018-06-15", 2018, "Jun"),
    ("2018-09-21", 2018, "Sep"), ("2018-12-21", 2018, "Dec"),
    ("2019-03-15", 2019, "Mar"), ("2019-06-21", 2019, "Jun"),
    ("2019-09-20", 2019, "Sep"), ("2019-12-20", 2019, "Dec"),
    ("2020-03-20", 2020, "Mar"), ("2020-06-19", 2020, "Jun"),
    ("2020-09-18", 2020, "Sep"), ("2020-12-18", 2020, "Dec"),
    ("2021-03-19", 2021, "Mar"), ("2021-06-18", 2021, "Jun"),
    ("2021-09-17", 2021, "Sep"), ("2021-12-17", 2021, "Dec"),
    ("2022-03-18", 2022, "Mar"), ("2022-06-17", 2022, "Jun"),
    ("2022-09-16", 2022, "Sep"), ("2022-12-16", 2022, "Dec"),
    ("2023-03-17", 2023, "Mar"), ("2023-06-16", 2023, "Jun"),
    ("2023-09-15", 2023, "Sep"), ("2023-12-15", 2023, "Dec"),
    ("2024-03-15", 2024, "Mar"), ("2024-06-21", 2024, "Jun"),
    ("2024-09-20", 2024, "Sep"), ("2024-12-20", 2024, "Dec"),
    ("2025-03-21", 2025, "Mar"), ("2025-06-20", 2025, "Jun"),
    ("2025-09-19", 2025, "Sep"), ("2025-12-19", 2025, "Dec"),
]


def all_tickers() -> list[str]:
    return [INSTRUMENT]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"qwh_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2004-06-01", end: str = "2026-07-01", retries: int = 4) -> None:
    """Download adjusted (total-return) daily closes for SPY; cache them.

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


# --------------------------------------------------------------------------- #
# Synthetic world -- single drifting name with a planted post-event "hangover" dip
# --------------------------------------------------------------------------- #
def synthetic_world(dip: float = 0.0, seed: int = 793, n_events: int = 84,
                    n_days: int = 5300, spacing: int = 63, mu: float = 0.0003,
                    ) -> tuple[pd.Series, list[int]]:
    """Deterministic single-name log-return world (positive drift, like SPY) with a planted
    post-event hangover.

    Returns are ``mu`` drift plus zero-mean noise; on the trading day just after each
    synthetic "quad-witching day" (every ``spacing``-th business day, ~quarterly) the asset
    gets an EXTRA ``-dip`` log-return — a planted post-event hangover that shows up in the
    post-event window. ``dip = 0`` is the null world (pure drift + noise).

    Business-day integer index (positions 0..n_days). Returns (asset_logret, event_positions).
    """
    rng = np.random.default_rng(seed)
    a = mu + rng.normal(0.0, 0.011, n_days)

    ev_pos = list(range(spacing, n_days - 130, spacing))[:n_events]
    for p in ev_pos:
        if p + 1 < n_days:
            a[p + 1] -= dip     # planted hangover: lands in the post-event window

    idx = pd.RangeIndex(n_days)
    return pd.Series(a, index=idx), ev_pos
