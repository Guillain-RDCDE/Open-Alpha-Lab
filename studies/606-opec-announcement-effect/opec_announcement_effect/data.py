"""Data layer for Study 606 — OPEC Announcement Effect.

Two sources, both offline-friendly once cached:

* **Real tape.** Daily OHLC for WTI front-month futures (``CL=F``), Brent front-month
  futures (``BZ=F``) and the USO ETF (yfinance, no key), cached per ticker under the
  study's ``_cache/``. Plus a **hardcoded table of OPEC / OPEC+ ministerial decision
  dates 2000-2026** (below) — fetched once from the public record and frozen here so the
  study is deterministic and offline.

* **Synthetic.** A deterministic, fixed-seed world with meetings every ~3 months and a
  TUNABLE planted meeting-day effect: a volatility multiple (``vol_mult``) and a signed
  decision-day drift (``drift``). ``vol_mult=1, drift=0`` is the null — the detectors
  must NOT fire; ``vol_mult=2`` (the folklore's "vol doubles") must light up.

Pure numpy + pandas + stdlib on the offline path; ``fetch_prices`` (network) is used once
to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

TICKERS = {"CL=F": "opec_cl.csv", "BZ=F": "opec_bz.csv", "USO": "opec_uso.csv"}

# --------------------------------------------------------------------------- #
# The OPEC / OPEC+ ministerial decision calendar, 2000-2026 (107 meetings)
# --------------------------------------------------------------------------- #
# Scope rule (documented, fixed BEFORE looking at returns):
#   * every OPEC Conference meeting (Ordinary + Extraordinary) 2000-2016,
#   * the two 2001/2003/2006 ministerial "consultative" meetings that set production,
#   * every OPEC and non-OPEC Ministerial Meeting (ONOMM, the OPEC+ decision body)
#     2016-2026 — the 1st through the 41st.
#   EXCLUDED: JMMC monitoring committees and the post-2023 "eight-country" (V8)
#   subgroup calls — they are not the full ministerial decision body. Excluding them is
#   conservative for the vol claim (some, e.g. the 3 Apr 2023 surprise, moved oil hard).
# Date convention: the meeting's DECISION day (the OPEC press-release date). Several
# OPEC+ meetings fall on weekends — the analysis maps every date to the first tradable
# session at-or-after it, per asset (strategy.map_event_days).
# Sources (fetched 2026-07-03, then frozen here):
#   * OPEC press-release archive, https://www.opec.org/press-releases.html (per-meeting
#     pages, e.g. /pr-detail/28-05-dec-2024.html for the 38th ONOMM), and the legacy
#     archive www.opec.org/opec_web/en/press_room/ (e.g. 1050.htm = 129th Conference,
#     1009.htm = 145th, 1006.htm = 146th).
#   * OPEC Annual Statistical Bulletin (meeting chronology), e.g.
#     https://www.opec.org/assets/assetdb/asb-2025.pdf
#   * Wikipedia world-oil-market chronologies (2000-2016) for cross-checks.
# Spot-verified against the press releases: 113th (17 Jan 2001), 114th (16-17 Mar 2001),
# 129th (10 Feb 2004), 131st (3 Jun 2004), 132nd (15 Sep 2004), 135th (16 Mar 2005),
# 136th (15 Jun 2005), 144th (15 Mar 2007), 145th (11 Sep 2007), 146th (5 Dec 2007),
# 17th/19th/20th/22nd/23rd ONOMM (2021), 38th (5 Dec 2024), 39th (28 May 2025),
# 40th (30 Nov 2025), 41st (7 Jun 2026).
OPEC_MEETINGS: list[tuple[str, str]] = [
    # ---- OPEC Conference era ------------------------------------------------
    ("2000-03-28", "109th (EO) Conference - raise quotas ~+1.7 mb/d"),
    ("2000-06-21", "110th (EO) Conference - raise quotas ~+0.7 mb/d"),
    ("2000-09-11", "111th Conference - raise quotas ~+0.8 mb/d"),
    ("2000-11-13", "112th (EO) Conference - quotas unchanged"),
    ("2001-01-17", "113th (EO) Conference - cut 1.5 mb/d"),
    ("2001-03-16", "114th Conference - cut 1.0 mb/d"),
    ("2001-06-05", "115th Conference - quotas unchanged"),
    ("2001-07-03", "116th (EO) Conference - quotas unchanged"),
    ("2001-07-25", "ministerial consultation - cut 1.0 mb/d (eff. 1 Sep)"),
    ("2001-09-26", "117th Conference - quotas unchanged"),
    ("2001-11-14", "118th (EO) Conference - conditional cut 1.5 mb/d"),
    ("2002-03-15", "119th Conference - quotas unchanged"),
    ("2002-06-26", "120th Conference - quotas unchanged"),
    ("2002-09-19", "121st Conference - quotas unchanged"),
    ("2002-12-12", "122nd Conference - ceiling re-based"),
    ("2003-01-12", "123rd (EO) Conference - ceiling +1.5 mb/d"),
    ("2003-03-11", "124th Conference - quotas unchanged"),
    ("2003-04-24", "125th (EO) Conference - ceiling raised, supply trimmed"),
    ("2003-06-11", "126th Conference - quotas unchanged"),
    ("2003-07-31", "consultative meeting - quotas unchanged"),
    ("2003-09-24", "127th Conference - cut 0.9 mb/d"),
    ("2003-12-04", "128th Conference - quotas unchanged"),
    ("2004-02-10", "129th (EO) Conference - cut 1.0 mb/d from April"),
    ("2004-03-31", "130th Conference - April cut confirmed"),
    ("2004-06-03", "131st (EO) Conference - raise ceiling 2.0 mb/d"),
    ("2004-09-15", "132nd Conference - raise ceiling 1.0 mb/d"),
    ("2004-12-10", "133rd (EO) Conference - trim over-production 1.0 mb/d"),
    ("2005-01-30", "134th (EO) Conference - quotas unchanged"),
    ("2005-03-16", "135th Conference - raise ceiling 0.5 mb/d"),
    ("2005-06-15", "136th Conference - raise ceiling 0.5 mb/d"),
    ("2005-09-20", "137th Conference - spare capacity offered"),
    ("2005-12-12", "138th (EO) Conference - quotas unchanged"),
    ("2006-01-31", "139th (EO) Conference - quotas unchanged"),
    ("2006-03-08", "140th Conference - quotas unchanged"),
    ("2006-06-01", "141st Conference (Caracas) - quotas unchanged"),
    ("2006-09-11", "142nd Conference - quotas unchanged"),
    ("2006-10-20", "consultative meeting (Doha) - cut 1.2 mb/d"),
    ("2006-12-14", "143rd (EO) Conference (Abuja) - cut 0.5 mb/d"),
    ("2007-03-15", "144th Conference - quotas unchanged"),
    ("2007-09-11", "145th Conference - raise 0.5 mb/d"),
    ("2007-12-05", "146th (EO) Conference (Abu Dhabi) - unchanged"),
    ("2008-02-01", "147th (EO) Conference - quotas unchanged"),
    ("2008-03-05", "148th Conference - quotas unchanged"),
    ("2008-09-10", "149th Conference - return to quotas (~-0.5 mb/d)"),
    ("2008-10-24", "150th (EO) Conference - cut 1.5 mb/d"),
    ("2008-12-17", "151st (EO) Conference (Oran) - cut 2.2 mb/d"),
    ("2009-03-15", "152nd Conference - quotas unchanged"),
    ("2009-05-28", "153rd Conference - quotas unchanged"),
    ("2009-09-10", "154th Conference - quotas unchanged"),
    ("2009-12-22", "155th Conference (Luanda) - quotas unchanged"),
    ("2010-03-17", "156th Conference - quotas unchanged"),
    ("2010-10-14", "157th Conference - quotas unchanged"),
    ("2010-12-11", "158th Conference (Quito) - quotas unchanged"),
    ("2011-06-08", "159th Conference - NO agreement"),
    ("2011-12-14", "160th Conference - ceiling 30 mb/d"),
    ("2012-06-14", "161st Conference - ceiling unchanged"),
    ("2012-12-12", "162nd Conference - ceiling unchanged"),
    ("2013-05-31", "163rd Conference - ceiling unchanged"),
    ("2013-12-04", "164th Conference - ceiling unchanged"),
    ("2014-06-11", "165th Conference - ceiling unchanged"),
    ("2014-11-27", "166th Conference - NO cut (the price-war meeting)"),
    ("2015-06-05", "167th Conference - ceiling unchanged"),
    ("2015-12-04", "168th Conference - ceiling dropped"),
    ("2016-06-02", "169th Conference - no ceiling agreement"),
    ("2016-09-28", "170th (EO) Conference (Algiers Accord) - target cut"),
    ("2016-11-30", "171st Conference (Vienna Agreement) - cut 1.2 mb/d"),
    # ---- OPEC+ (ONOMM) era --------------------------------------------------
    ("2016-12-10", "1st ONOMM - non-OPEC joins (-0.558 mb/d)"),
    ("2017-05-25", "172nd Conference + 2nd ONOMM - extend cuts 9 months"),
    ("2017-11-30", "173rd Conference + 3rd ONOMM - extend through 2018"),
    ("2018-06-22", "174th Conference (+4th ONOMM Jun 23) - restore ~1 mb/d"),
    ("2018-12-07", "175th Conference + 5th ONOMM - cut 1.2 mb/d"),
    ("2019-07-01", "176th Conference (+6th ONOMM Jul 2) - extend 9 months"),
    ("2019-12-05", "177th Conference (+7th ONOMM Dec 6) - deepen cut 0.5 mb/d"),
    ("2020-03-06", "178th (EO) Conference + 8th ONOMM - talks COLLAPSE (price war)"),
    ("2020-04-09", "9th (EO) ONOMM - tentative 9.7 mb/d cut"),
    ("2020-04-12", "10th (EO) ONOMM - historic 9.7 mb/d cut finalised"),
    ("2020-06-06", "11th ONOMM - deepest cut extended through July"),
    ("2020-12-03", "12th ONOMM - ease +0.5 mb/d from January 2021"),
    ("2021-01-05", "13th ONOMM - Saudi surprise voluntary -1 mb/d"),
    ("2021-03-04", "14th ONOMM - surprise hold"),
    ("2021-04-01", "15th ONOMM - gradual increases May-July"),
    ("2021-04-28", "16th ONOMM - plan confirmed"),
    ("2021-06-01", "17th ONOMM - plan confirmed"),
    ("2021-07-02", "18th ONOMM - UAE standoff (abandoned 5 Jul)"),
    ("2021-07-18", "19th ONOMM - +0.4 mb/d per month deal"),
    ("2021-09-01", "20th ONOMM - +0.4 mb/d confirmed"),
    ("2021-10-04", "21st ONOMM - +0.4 mb/d confirmed"),
    ("2021-11-04", "22nd ONOMM - +0.4 mb/d confirmed"),
    ("2021-12-02", "23rd ONOMM - +0.4 mb/d confirmed"),
    ("2022-01-04", "24th ONOMM - +0.4 mb/d confirmed"),
    ("2022-02-02", "25th ONOMM - +0.4 mb/d confirmed"),
    ("2022-03-02", "26th ONOMM - +0.4 mb/d confirmed"),
    ("2022-03-31", "27th ONOMM - +0.432 mb/d for May"),
    ("2022-05-05", "28th ONOMM - +0.432 mb/d for June"),
    ("2022-06-02", "29th ONOMM - +0.648 mb/d for July"),
    ("2022-06-30", "30th ONOMM - +0.648 mb/d for August"),
    ("2022-08-03", "31st ONOMM - +0.1 mb/d for September"),
    ("2022-09-05", "32nd ONOMM - -0.1 mb/d for October"),
    ("2022-10-05", "33rd ONOMM - cut 2.0 mb/d"),
    ("2022-12-04", "34th ONOMM - targets unchanged"),
    ("2023-06-04", "35th ONOMM - Saudi extra voluntary -1 mb/d"),
    ("2023-11-30", "36th ONOMM - 2.2 mb/d voluntary cuts"),
    ("2024-06-02", "37th ONOMM (Riyadh) - cuts extended, taper sketched"),
    ("2024-12-05", "38th ONOMM - taper delayed again"),
    ("2025-05-28", "39th ONOMM - group targets reaffirmed"),
    ("2025-11-30", "40th ONOMM - targets reaffirmed for 2026"),
    ("2026-06-07", "41st ONOMM - quotas reaffirmed through Dec 2026"),
]


def meeting_dates() -> pd.DatetimeIndex:
    """The frozen meeting calendar as a sorted DatetimeIndex (107 decision days)."""
    return pd.DatetimeIndex(sorted(pd.Timestamp(d) for d, _ in OPEC_MEETINGS))


# --------------------------------------------------------------------------- #
# Real tape (cache-first)
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, TICKERS[ticker])


def fetch_prices(start: str = "2000-01-01", end: str | None = None,
                 retries: int = 3) -> dict[str, pd.DataFrame]:
    """Download daily OHLC for CL=F / BZ=F / USO and cache one CSV per ticker.

    Network-only; used once to build the cache. Never imported by offline cells.
    ``auto_adjust=True`` (splits/dividends folded in — only matters for USO).
    """
    import yfinance as yf

    out = {}
    os.makedirs(CACHE_DIR, exist_ok=True)
    for tk in TICKERS:
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(tk, start=start, end=end, auto_adjust=True,
                                  progress=False)
                if raw is not None and len(raw) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        if raw is None or len(raw) == 0:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        px = raw[["Open", "High", "Low", "Close"]].dropna(how="any").copy()
        px.index = pd.DatetimeIndex(px.index).tz_localize(None).normalize()
        px.to_csv(_cache_path(tk))
        out[tk] = px
    return out


def have_real() -> bool:
    return all(os.path.exists(_cache_path(tk)) for tk in TICKERS)


def load_prices(ticker: str = "CL=F", asof: str | None = None) -> pd.DataFrame:
    """Cached daily OHLC frame for one ticker, optionally sliced to ``asof``."""
    px = pd.read_csv(_cache_path(ticker), index_col=0, parse_dates=True).sort_index()
    if asof is not None:
        px = px[px.index <= pd.Timestamp(asof)]
    return px


def load_real(asof: str | None = None) -> dict[str, pd.DataFrame]:
    """Convenience: {ticker: OHLC frame} for the three cached tapes."""
    return {tk: load_prices(tk, asof=asof) for tk in TICKERS}


# --------------------------------------------------------------------------- #
# Synthetic world (the machinery control)
# --------------------------------------------------------------------------- #
def synthetic_world(n_days: int = 6300, vol_mult: float = 1.0, drift: float = 0.0,
                    meet_every: int = 63, seed: int = 606,
                    sig_daily: float = 0.022) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Deterministic daily OHLC tape with meetings every ``meet_every`` sessions and a
    PLANTED meeting-day effect.

    On meeting days the daily return is drawn with standard deviation
    ``sig_daily * vol_mult`` and mean ``drift`` (all other days: mean 0, sd ``sig_daily``).
    The intraday range scales with the same day's vol. ``vol_mult=1, drift=0`` is the
    null — the vol and drift detectors must NOT fire; ``vol_mult=2`` is the folklore's
    "vol doubles" planted at full strength.

    Returns ``(ohlc_frame, meeting_dates)``. The index is a plain business-day range
    starting 2000-01-03 (~25 years for the default size — comfortably inside the
    pandas ns-Timestamp span).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n_days)
    is_meet = np.zeros(n_days, dtype=bool)
    is_meet[meet_every // 2::meet_every] = True

    sig = np.where(is_meet, sig_daily * vol_mult, sig_daily)
    mu = np.where(is_meet, drift, 0.0)
    ret = rng.normal(mu, sig)
    close = 100.0 * np.cumprod(1.0 + ret)
    prev = np.concatenate([[100.0], close[:-1]])
    span = np.abs(rng.normal(0.0, 0.5 * sig)) + 0.25 * sig
    high = np.maximum(prev, close) * (1.0 + span)
    low = np.minimum(prev, close) * (1.0 - span)
    px = pd.DataFrame({"Open": prev, "High": high, "Low": low, "Close": close},
                      index=idx)
    return px, pd.DatetimeIndex(idx[is_meet])
