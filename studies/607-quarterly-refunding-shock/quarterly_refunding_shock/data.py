"""Data layer for Study 607 — Quarterly Refunding Shock.

The claim: the U.S. Treasury's **Quarterly Refunding Announcement** (QRA) — the
Wednesday-morning statement in which Treasury reveals its coupon-auction sizes for the
quarter — moves the **long end** of the curve. The Aug/Nov 2023 episodes supposedly turned
QRA day into a first-tier macro event. This module supplies three things:

* **The hardcoded QRA calendar, 2000 → 2026 (106 announcements).** Derived ONCE from the
  official TreasuryDirect auction-query API (``TA_WS``): the mid-quarter refunding
  securities (the 10-Year note auctioned in Feb/May/Aug/Nov — plus, in 2000–2002 quarters
  with no new 10Y, the refunding 5-Year/10Y-reopening pair) are *officially announced via
  the Quarterly Refunding Statement*, so their ``announcementDate`` **is** the QRA date.
  Source: ``https://www.treasurydirect.gov/TA_WS/securities/search?type=Note&format=json``
  (fetched 2026-07-03; derivation preserved in :func:`derive_qra_dates_from_treasurydirect`).
  Sanity: all 106 dates are Wednesdays, and the storied ones match the Treasury press
  archive (2023-08-02, 2023-11-01, 2024-01-31, 2024-10-30, 2001-10-31 = the 30Y-suspension
  statement, 2000-02-02 = the buyback-era statement).

* **The FOMC statement calendar** (hardcoded, same table as desk studies 517/602, source:
  federalreserve.gov historical FOMC calendars, scheduled meetings only). Crucial here:
  QRA Wednesdays and FOMC Wednesdays cluster in the SAME early-Feb/May/Aug/Nov weeks —
  about a quarter of all QRA days are also FOMC statement days (including 2023-11-01!),
  so the honest day-0 test must separate the two.

* **The real tape** (cache-first): ^TNX (CBOE 10-Year yield index; Yahoo serves it already in percent) and
  TLT (iShares 20+Y ETF, total-return via auto_adjust) from yfinance. And a deterministic
  **synthetic world** with a TUNABLE planted event effect (an event-day volatility
  multiplier and/or a signed event-day mean) plus the null (multiplier 1, mean 0), so the
  detection machinery is proven before it touches the tape.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
TNX_CACHE = os.path.join(CACHE_DIR, "qrs_tnx.csv")
TYX_CACHE = os.path.join(CACHE_DIR, "qrs_tyx.csv")
TLT_CACHE = os.path.join(CACHE_DIR, "qrs_tlt.csv")

# The frozen as-of for every headline number (last complete day before the run date;
# June 2026 is the last complete month). Bump deliberately, never by accident.
AS_OF = "2026-06-30"

# --------------------------------------------------------------------------- #
# The QRA calendar, 2000 -> 2026 — 106 Quarterly Refunding Announcement dates.
#
# HARDCODED from the official TreasuryDirect TA_WS auction records (fetched once,
# 2026-07-03): the announcementDate of the mid-quarter refunding 10-Year note
# (Feb/May/Aug/Nov auctions; for 2000-05/2000-11/2001-05/2001-11/2002-05 — quarters
# with no NEW 10Y — the refunding 5-Year + 10Y-reopening, announced on the same
# statement day). The refunding statement is released at 08:30 ET, so the day-0
# close-to-close yield change fully contains the reaction. All dates are Wednesdays.
# Source API: https://www.treasurydirect.gov/TA_WS/securities/search?type=Note&format=json
# Statement archive: https://home.treasury.gov/policy-issues/financing-the-government/quarterly-refunding
# --------------------------------------------------------------------------- #
QRA_DATES = [
    # 2000
    "2000-02-02", "2000-05-03", "2000-08-02", "2000-11-01",
    # 2001  (2001-10-31 = the famous 30Y-suspension statement)
    "2001-01-31", "2001-05-02", "2001-08-01", "2001-10-31",
    # 2002
    "2002-01-30", "2002-05-01", "2002-07-31", "2002-10-30",
    # 2003
    "2003-02-05", "2003-04-30", "2003-07-30", "2003-11-05",
    # 2004
    "2004-02-04", "2004-05-05", "2004-08-04", "2004-11-03",
    # 2005
    "2005-02-02", "2005-05-04", "2005-08-03", "2005-11-02",
    # 2006
    "2006-02-01", "2006-05-03", "2006-08-02", "2006-11-01",
    # 2007
    "2007-01-31", "2007-05-02", "2007-08-01", "2007-10-31",
    # 2008
    "2008-01-30", "2008-04-30", "2008-07-30", "2008-11-05",
    # 2009
    "2009-02-04", "2009-04-29", "2009-08-05", "2009-11-04",
    # 2010
    "2010-02-03", "2010-05-05", "2010-08-04", "2010-11-03",
    # 2011
    "2011-02-02", "2011-05-04", "2011-08-03", "2011-11-02",
    # 2012
    "2012-02-01", "2012-05-02", "2012-08-01", "2012-10-31",
    # 2013
    "2013-02-06", "2013-05-01", "2013-07-31", "2013-11-06",
    # 2014
    "2014-02-05", "2014-04-30", "2014-08-06", "2014-11-05",
    # 2015
    "2015-02-04", "2015-05-06", "2015-08-05", "2015-11-04",
    # 2016
    "2016-02-03", "2016-05-04", "2016-08-03", "2016-11-02",
    # 2017
    "2017-02-01", "2017-05-03", "2017-08-02", "2017-11-01",
    # 2018
    "2018-01-31", "2018-05-02", "2018-08-01", "2018-10-31",
    # 2019
    "2019-01-30", "2019-05-01", "2019-07-31", "2019-10-30",
    # 2020
    "2020-02-05", "2020-05-06", "2020-08-05", "2020-11-04",
    # 2021
    "2021-02-03", "2021-05-05", "2021-08-04", "2021-11-03",
    # 2022
    "2022-02-02", "2022-05-04", "2022-08-03", "2022-11-02",
    # 2023  (2023-08-02 and 2023-11-01 = the episodes that made "QRA" famous)
    "2023-02-01", "2023-05-03", "2023-08-02", "2023-11-01",
    # 2024
    "2024-01-31", "2024-05-01", "2024-07-31", "2024-10-30",
    # 2025
    "2025-02-05", "2025-04-30", "2025-07-30", "2025-11-05",
    # 2026 (through the as-of)
    "2026-02-04", "2026-05-06",
]

# --------------------------------------------------------------------------- #
# FOMC scheduled statement days, 2000 -> 2026-06 (needed to decontaminate QRA day-0:
# both calendars gravitate to the same early-quarter Wednesdays). Same hardcoded table
# as desk studies 517-pre-fomc-drift / 602-macro-announcement-premium, trimmed to 2000+.
# Source: Federal Reserve historical FOMC calendars
# (federalreserve.gov/monetarypolicy/fomc_historical_year.htm). Scheduled meetings only.
# --------------------------------------------------------------------------- #
FOMC_DATES = [
    # 2000
    "2000-02-02", "2000-03-21", "2000-05-16", "2000-06-28", "2000-08-22",
    "2000-10-03", "2000-11-15", "2000-12-19",
    # 2001
    "2001-01-31", "2001-03-20", "2001-05-15", "2001-06-27", "2001-08-21",
    "2001-10-02", "2001-11-06", "2001-12-11",
    # 2002
    "2002-01-30", "2002-03-19", "2002-05-07", "2002-06-26", "2002-08-13",
    "2002-09-24", "2002-11-06", "2002-12-10",
    # 2003
    "2003-01-29", "2003-03-18", "2003-05-06", "2003-06-25", "2003-08-12",
    "2003-09-16", "2003-10-28", "2003-12-09",
    # 2004
    "2004-01-28", "2004-03-16", "2004-05-04", "2004-06-30", "2004-08-10",
    "2004-09-21", "2004-11-10", "2004-12-14",
    # 2005
    "2005-02-02", "2005-03-22", "2005-05-03", "2005-06-30", "2005-08-09",
    "2005-09-20", "2005-11-01", "2005-12-13",
    # 2006
    "2006-01-31", "2006-03-28", "2006-05-10", "2006-06-29", "2006-08-08",
    "2006-09-20", "2006-10-25", "2006-12-12",
    # 2007
    "2007-01-31", "2007-03-21", "2007-05-09", "2007-06-28", "2007-08-07",
    "2007-09-18", "2007-10-31", "2007-12-11",
    # 2008
    "2008-01-30", "2008-03-18", "2008-04-30", "2008-06-25", "2008-08-05",
    "2008-09-16", "2008-10-29", "2008-12-16",
    # 2009
    "2009-01-28", "2009-03-18", "2009-04-29", "2009-06-24", "2009-08-12",
    "2009-09-23", "2009-11-04", "2009-12-16",
    # 2010
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-06-23", "2010-08-10",
    "2010-09-21", "2010-11-03", "2010-12-14",
    # 2011
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22", "2011-08-09",
    "2011-09-21", "2011-11-02", "2011-12-13",
    # 2012
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20", "2012-08-01",
    "2012-09-13", "2012-10-24", "2012-12-12",
    # 2013
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19", "2013-07-31",
    "2013-09-18", "2013-10-30", "2013-12-18",
    # 2014
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18", "2014-07-30",
    "2014-09-17", "2014-10-29", "2014-12-17",
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17", "2015-07-29",
    "2015-09-17", "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27",
    "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26",
    "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01",
    "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31",
    "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020
    "2020-01-29", "2020-03-18", "2020-04-29", "2020-06-10", "2020-07-29",
    "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28",
    "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
    "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
    "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 (scheduled, through the as-of)
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
]

# --------------------------------------------------------------------------- #
# The 2023 narrative — the two borrowing-estimate surprises + the two statements that
# made "QRA" a household word on macro desks. Hardcoded from the Treasury press archive
# (home.treasury.gov/news/press-releases; marketable-borrowing estimates land the Monday
# 15:00 ET before the Wednesday 08:30 ET refunding statement).
# --------------------------------------------------------------------------- #
EPISODES_2023 = [
    ("2023-07-31", "borrowing estimate",
     "Q3-2023 marketable borrowing raised to $1,007B vs $733B projected in May "
     "(+$274B upside surprise)"),
    ("2023-08-02", "QRA statement",
     "coupon auction sizes raised across the curve — the first increase since early 2021 "
     "(confound: Fitch stripped the US AAA after the close on Aug 1)"),
    ("2023-10-30", "borrowing estimate",
     "Q4-2023 borrowing set at $776B — $76B BELOW the $852B flagged in July "
     "(downside surprise)"),
    ("2023-11-01", "QRA statement",
     "long-end size increases slowed vs. expectations "
     "(confound: FOMC statement the same day at 14:00 ET)"),
]


# --------------------------------------------------------------------------- #
# Real tape (network once; cache-first forever after)
# --------------------------------------------------------------------------- #
def fetch_tape(tnx_path: str = TNX_CACHE, tyx_path: str = TYX_CACHE,
               tlt_path: str = TLT_CACHE, retries: int = 3) -> None:
    """Download ^TNX + ^TYX + TLT once and cache them as CSV. Never called offline.

    ^TNX/^TYX are the CBOE 10-Year/30-Year Treasury yield indices; Yahoo serves them
    **already in percent** (e.g. 4.37 = 4.37%), stored under columns ``y10``/``y30``.
    TLT is stored auto-adjusted (**total-return**) under column ``tlt``.
    """
    import yfinance as yf

    os.makedirs(os.path.dirname(tnx_path), exist_ok=True)

    def _grab(ticker, start):
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)["Close"]
                if raw is not None and len(raw) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        s = raw.squeeze() if hasattr(raw, "squeeze") else raw
        s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
        return s.dropna()

    tnx = _grab("^TNX", "1999-11-01")                   # Yahoo ^TNX = yield in %
    tnx.rename("y10").to_frame().to_csv(tnx_path)
    tyx = _grab("^TYX", "1999-11-01")                   # Yahoo ^TYX = 30Y yield in %
    tyx.rename("y30").to_frame().to_csv(tyx_path)
    tlt = _grab("TLT", "2002-07-01")
    tlt.rename("tlt").to_frame().to_csv(tlt_path)


def have_real(tnx_path: str = TNX_CACHE, tyx_path: str = TYX_CACHE,
              tlt_path: str = TLT_CACHE) -> bool:
    return (os.path.exists(tnx_path) and os.path.exists(tyx_path)
            and os.path.exists(tlt_path))


def load_tnx(path: str = TNX_CACHE, as_of: str | None = AS_OF) -> pd.Series:
    """Cached 10-Year yield in percent (daily close), sliced to the frozen as-of."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if as_of:
        df = df[df.index <= pd.Timestamp(as_of)]
    return df["y10"].dropna()


def load_tyx(path: str = TYX_CACHE, as_of: str | None = AS_OF) -> pd.Series:
    """Cached 30-Year yield in percent (daily close), sliced to the frozen as-of."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if as_of:
        df = df[df.index <= pd.Timestamp(as_of)]
    return df["y30"].dropna()


def load_tlt(path: str = TLT_CACHE, as_of: str | None = AS_OF) -> pd.Series:
    """Cached TLT total-return close, sliced to the frozen as-of."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if as_of:
        df = df[df.index <= pd.Timestamp(as_of)]
    return df["tlt"].dropna()


def load_real() -> tuple[pd.Series, pd.Series, pd.Series]:
    return load_tnx(), load_tyx(), load_tlt()


# --------------------------------------------------------------------------- #
# The (optional, network) derivation of the QRA table — kept for provenance.
# --------------------------------------------------------------------------- #
def derive_qra_dates_from_treasurydirect() -> list[str]:  # pragma: no cover (network)
    """Re-derive QRA_DATES from the TreasuryDirect TA_WS API (network; provenance only).

    The refunding 10-Year note auctioned in each Feb/May/Aug/Nov is announced via the
    Quarterly Refunding Statement, so its ``announcementDate`` is the QRA date. For the
    five 2000-2002 quarters with no NEW 10Y (May/Nov refundings then reopened the 10Y),
    the refunding 5-Year / 10Y-reopening pair carries the same statement day.
    """
    import json
    import urllib.request

    url = "https://www.treasurydirect.gov/TA_WS/securities/search?type=Note&format=json"
    req = urllib.request.Request(
        url, headers={"User-Agent": "OpenAlphaLab research desk guillain@poulpe.us"})
    with urllib.request.urlopen(req, timeout=120) as r:
        notes = json.loads(r.read().decode())
    out = {}
    for s in notes:
        term = s.get("securityTerm")
        auc = (s.get("auctionDate") or "")[:10]
        ann = (s.get("announcementDate") or "")[:10]
        if not auc or not ann or int(auc[:4]) < 2000:
            continue
        y, m = int(auc[:4]), int(auc[5:7])
        if m not in (2, 5, 8, 11):
            continue
        key = (y, m)
        rank = {"10-Year": 0, "9-Year 9-Month": 1, "5-Year": 2}.get(term)
        if rank is None:
            continue
        cur = out.get(key)
        if cur is None or (rank, auc) < cur[0]:
            out[key] = ((rank, auc), ann)
    return [ann for _k, (_r, ann) in sorted(out.items())]


# --------------------------------------------------------------------------- #
# Synthetic world — the machinery proof (deterministic, offline, tunable)
# --------------------------------------------------------------------------- #
def synthetic_world(n_days: int = 6800, vol_mult: float = 1.0, mean_bp: float = 0.0,
                    base_sigma_bp: float = 6.0, seed: int = 607) -> tuple[pd.Series, list[str]]:
    """A deterministic daily 10Y-yield world with quarterly pseudo-QRA events.

    Daily yield changes are i.i.d. Normal(0, ``base_sigma_bp``); every 63rd trading day
    is an "announcement day" whose change is drawn Normal(``mean_bp``,
    ``base_sigma_bp * vol_mult``). Knobs:

    * ``vol_mult`` — the PLANTED event-day volatility multiplier (1.0 = the null: event
      days are indistinguishable, and the detector must NOT fire);
    * ``mean_bp``  — a planted signed event-day mean (0.0 = the null for the signed test).

    Returns (yield level series in percent, list of ISO event dates). The ~27-year
    bdate_range span stays far inside the pandas ns-Timestamp limit.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n_days)
    dy = rng.normal(0.0, base_sigma_bp, n_days)
    ev = np.arange(40, n_days - 5, 63)
    dy[ev] = rng.normal(mean_bp, base_sigma_bp * vol_mult, len(ev))
    y = 4.0 + np.cumsum(dy) / 100.0
    lvl = pd.Series(y, index=idx, name="y10")
    return lvl, [d.strftime("%Y-%m-%d") for d in idx[ev]]
