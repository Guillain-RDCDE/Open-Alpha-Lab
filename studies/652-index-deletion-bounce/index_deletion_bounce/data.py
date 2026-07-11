"""Data layer for Study 652 — Index-Deletion-Bounce.

Three ingredients, all offline-friendly once cached:

* **The hardcoded deletion calendar.** 70 real S&P 500 **deletions** 2012-12-11 -> 2025-09-22,
  each with an announcement date and an effective date. Every one of these deletions was coded
  by S&P Dow Jones Indices as a **"market capitalization change"** — i.e. the company *shrank*
  out of the index (the classic Chen-Noronha-Singal 2004 "deletion" mechanism) — not an
  acquisition, spin-off or bankruptcy-driven removal (those tickers usually vanish outright and
  there is nothing left to hold "long the deleted name" into). Source: the "Selected changes to
  the list of S&P 500 components" table maintained on Wikipedia's *List of S&P 500 companies*
  page, itself sourced row-by-row to S&P Dow Jones Indices' own index-news announcement PDFs
  (spglobal.com/spdji/en/documents/index-news/announcements/...) — the effective date is the
  table's "Effective Date" column, the announce date is the **date printed on the cited S&P
  press release** (both parsed and hardcoded below; this module makes no network call to build
  the table itself).

* **Real tape.** Daily OHLC for each deleted ticker (yfinance, no key) around its own event
  window, plus daily SPY OHLC as the market benchmark for the whole span. Cached as CSV under
  the study's own ``_cache/``.

  **Named up front — this is the single biggest honesty issue in this study:** a large minority
  of "market-cap-change" deletions are, on inspection, distress names that later went bankrupt,
  got taken private or got acquired themselves — years after the S&P 500 deletion this study is
  about. When that later, *unrelated* corporate death happens, Yahoo Finance frequently drops
  **all** historical data for the ticker, including the untouched, perfectly historical days
  around our 2012-2025 event. Of the 70 hardcoded deletions, **22 have no usable tape left on
  Yahoo at all** (``FTR``, ``CHK``, ``WIN``, ``MNK``, ``JDSU``, ``BTU``, ``X``, ``WPX``, ``DNR``,
  ``LM``, ``RRD``, ``BIG``, ``APOL``, ``SRCL``, ``FL``, ``ADS``, ``HBI``, ``GPS``, ``FBHS``,
  ``DISH``, ``CMA``, ``PDCO`` at the time this study was built) — verified by direct query to
  Yahoo's chart endpoint, not just a `yfinance` retry. This is a **real, directional**
  survivorship bias: the events that live on the tape are disproportionately the ones whose
  companies did *not* keep declining into oblivion, which tilts the surviving sample *toward*
  finding a rebound. It is reported on the Signal axis, not buried — see docs/references.md and
  the notebooks.

* **Synthetic world.** A deterministic, seeded mean-reverting "distressed stock vs market"
  process with a TUNABLE planted dump-into-effective-date + reversal-after knob: on the
  synthetic pseudo-effective day the excess return gets an extra downward push (the forced
  selling), and for ``rebound_days`` afterward an extra upward drift (the reversal). Both knobs
  at 0 reproduce the null world.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build the
cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
SPY_CACHE = os.path.join(CACHE_DIR, "idb_spy.csv")

AS_OF = "2026-06-30"          # last complete calendar month at publication (2026-07-10)
ERA_SPLIT = "2019-01-01"      # within-sample split: first-half vs second-half deletions

# --------------------------------------------------------------------------- #
# Hardcoded deletion calendar: 70 real S&P 500 deletions, 2012 -> 2025, all coded by S&P
# Dow Jones Indices as "Market capitalization change" (distress deletions, not M&A/spin-offs —
# those tickers simply disappear and there is nothing to hold "long the deleted name" into).
# Columns: (ticker, name, announce date, effective date). Source: Wikipedia "List of S&P 500
# companies" -> "Selected changes to the list of S&P 500 components", cross-referenced to the
# cited S&P Dow Jones Indices index-news announcement PDFs for the announce date.
# --------------------------------------------------------------------------- #
DELETIONS = [
    ("RRD", "RR Donnelley", "2012-12-05", "2012-12-11"),
    ("BIG", "Big Lots", "2013-02-07", "2013-02-15"),
    ("APOL", "Apollo Education Group", "2013-06-20", "2013-07-01"),
    ("AMD", "Advanced Micro Devices", "2013-09-11", "2013-09-20"),
    ("ANF", "Abercrombie & Fitch", "2013-12-11", "2013-12-23"),
    ("JDSU", "JDS Uniphase", "2013-12-11", "2013-12-23"),
    ("WPX", "WPX Energy", "2014-03-14", "2014-03-21"),
    ("CLF", "Cliffs Natural Resources", "2014-04-02", "2014-04-02"),
    ("X", "United States Steel", "2014-06-27", "2014-07-02"),
    ("BTU", "Peabody Energy", "2014-09-12", "2014-09-20"),
    ("JBL", "Jabil Circuit", "2014-10-29", "2014-11-05"),
    ("DNR", "Denbury Resources", "2015-03-13", "2015-03-23"),
    ("WIN", "Windstream Communications", "2015-03-30", "2015-04-07"),
    ("LM", "Legg Mason", "2016-11-29", "2016-12-02"),
    ("OI", "Owens-Illinois", "2016-11-29", "2016-12-02"),
    ("FSLR", "First Solar", "2017-03-10", "2017-03-20"),
    ("FTR", "Frontier Communications", "2017-03-10", "2017-03-20"),
    ("R", "Ryder System", "2017-06-09", "2017-06-19"),
    ("TDC", "Teradata", "2017-06-09", "2017-06-19"),
    ("BBBY", "Bed Bath & Beyond", "2017-07-19", "2017-07-26"),
    ("MNK", "Mallinckrodt", "2017-07-19", "2017-07-26"),
    ("CHK", "Chesapeake Energy", "2018-03-09", "2018-03-19"),
    ("PDCO", "Patterson Companies", "2018-03-09", "2018-03-19"),
    ("AYI", "Acuity Brands", "2018-06-08", "2018-06-18"),
    ("RRC", "Range Resources", "2018-06-08", "2018-06-18"),
    ("SRCL", "Stericycle", "2018-11-26", "2018-12-03"),
    ("FL", "Foot Locker", "2019-08-01", "2019-08-09"),
    ("NKTR", "Nektar Therapeutics", "2019-09-20", "2019-10-03"),
    ("AMG", "Affiliated Managers Group", "2019-12-13", "2019-12-23"),
    ("MAC", "Macerich", "2019-12-13", "2019-12-23"),
    ("M", "Macy's", "2020-03-31", "2020-04-06"),
    ("CPRI", "Capri Holdings", "2020-05-06", "2020-05-12"),
    ("HP", "Helmerich & Payne", "2020-05-18", "2020-05-22"),
    ("ADS", "Alliance Data Systems", "2020-06-12", "2020-06-22"),
    ("HOG", "Harley-Davidson", "2020-06-12", "2020-06-22"),
    ("COTY", "Coty", "2020-09-04", "2020-09-21"),
    ("HRB", "H&R Block", "2020-09-04", "2020-09-21"),
    ("FLS", "Flowserve", "2021-03-12", "2021-03-22"),
    ("SLG", "SL Green Realty", "2021-03-12", "2021-03-22"),
    ("NOV", "NOV Inc.", "2021-09-03", "2021-09-20"),
    ("PRGO", "Perrigo", "2021-09-03", "2021-09-20"),
    ("HBI", "Hanesbrands", "2021-12-03", "2021-12-20"),
    ("LEG", "Leggett & Platt", "2021-12-03", "2021-12-20"),
    ("GPS", "Gap", "2022-01-26", "2022-02-03"),
    ("IPGP", "IPG Photonics", "2022-06-03", "2022-06-21"),
    ("UA", "Under Armour (Class C)", "2022-06-03", "2022-06-21"),
    ("PENN", "Penn Entertainment", "2022-09-02", "2022-09-19"),
    ("PVH", "PVH", "2022-09-02", "2022-09-19"),
    ("FBHS", "Fortune Brands Home & Security", "2022-12-12", "2022-12-19"),
    ("VNO", "Vornado Realty Trust", "2022-12-28", "2023-01-05"),
    ("LUMN", "Lumen Technologies", "2023-03-03", "2023-03-20"),
    ("DISH", "Dish Network", "2023-06-02", "2023-06-20"),
    ("DXC", "DXC Technology", "2023-09-28", "2023-10-03"),
    ("OGN", "Organon & Co.", "2023-10-13", "2023-10-18"),
    ("ALK", "Alaska Air Group", "2023-12-01", "2023-12-18"),
    ("SEDG", "SolarEdge", "2023-12-01", "2023-12-18"),
    ("WHR", "Whirlpool Corporation", "2024-03-01", "2024-03-18"),
    ("ZION", "Zions Bancorporation", "2024-03-01", "2024-03-18"),
    ("VFC", "VF Corporation", "2024-03-27", "2024-04-03"),
    ("XRAY", "Dentsply Sirona", "2024-03-27", "2024-04-03"),
    ("CMA", "Comerica", "2024-06-07", "2024-06-24"),
    ("ILMN", "Illumina, Inc.", "2024-06-07", "2024-06-24"),
    ("AAL", "American Airlines Group", "2024-09-06", "2024-09-23"),
    ("BIO", "Bio-Rad Laboratories", "2024-09-06", "2024-09-23"),
    ("BBWI", "Bath & Body Works, Inc.", "2024-09-24", "2024-10-01"),
    ("QRVO", "Qorvo", "2024-12-06", "2024-12-23"),
    ("BWA", "BorgWarner", "2025-03-07", "2025-03-24"),
    ("CE", "Celanese", "2025-03-07", "2025-03-24"),
    ("CZR", "Caesars Entertainment", "2025-09-05", "2025-09-22"),
    ("ENPH", "Enphase Energy", "2025-09-05", "2025-09-22"),
]

# Tickers verified (direct Yahoo chart-endpoint query, not just a yfinance retry) to have NO
# historical tape left at all — the company had a later, unrelated corporate death (bankruptcy,
# take-private, acquisition) that caused Yahoo to drop its entire history. Named here so the
# gap is a documented fact, not a silent hole.
KNOWN_NO_TAPE = frozenset({
    "RRD", "BIG", "APOL", "JDSU", "WPX", "X", "BTU", "DNR", "WIN", "LM", "FTR", "MNK", "CHK",
    "PDCO", "SRCL", "FL", "ADS", "HBI", "GPS", "FBHS", "DISH", "CMA",
})


def deletions_frame() -> pd.DataFrame:
    """The 70-row deletion calendar as a DataFrame, sorted by effective date."""
    df = pd.DataFrame(DELETIONS, columns=["ticker", "name", "announce", "effective"])
    df["announce"] = pd.to_datetime(df["announce"])
    df["effective"] = pd.to_datetime(df["effective"])
    return df.sort_values("effective").reset_index(drop=True)


def _ticker_cache(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"idb_{ticker.replace('.', '_')}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(pre_days: int = 130, post_days: int = 200) -> None:
    """Download SPY (full span) and each deletion ticker's own event window; cache as CSV.

    Each ticker is fetched over ``[effective - pre_days, effective + post_days]`` calendar
    days — comfortably covering the [-5..+40] trading-day event window plus the long-timer
    hold. Network; once. Tickers with no data (see ``KNOWN_NO_TAPE``) are skipped silently
    after one attempt — the gap is handled honestly downstream, not hidden.
    """
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)

    spy = yf.download("SPY", start="2012-01-01", end="2026-07-01",
                       auto_adjust=True, progress=False, threads=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy[["Open", "High", "Low", "Close"]].dropna(how="all").to_csv(SPY_CACHE)

    dl = deletions_frame()
    for _, row in dl.iterrows():
        t = row["ticker"]
        path = _ticker_cache(t)
        if os.path.exists(path):
            continue
        start = (row["effective"] - pd.Timedelta(days=pre_days)).strftime("%Y-%m-%d")
        end = (row["effective"] + pd.Timedelta(days=post_days)).strftime("%Y-%m-%d")
        try:
            px = yf.download(t, start=start, end=end, auto_adjust=True,
                              progress=False, threads=False)
            if isinstance(px.columns, pd.MultiIndex):
                px.columns = px.columns.get_level_values(0)
            px = px[["Open", "High", "Low", "Close"]].dropna(how="all")
            if len(px) >= 60:
                px.to_csv(path)
        except Exception:
            pass  # no tape for this ticker; KNOWN_NO_TAPE documents the honest count


def have_real() -> bool:
    return os.path.exists(SPY_CACHE)


def load_real() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[str]]:
    """Cached per-ticker frames (dict), the SPY frame, and the list of tickers with NO cache.

    The dict only contains tickers whose fetch actually produced usable rows; every ticker in
    ``DELETIONS`` not present in the dict is genuinely missing (delisted off Yahoo) and is
    returned in the third element for honest reporting.
    """
    spy = pd.read_csv(SPY_CACHE, index_col=0, parse_dates=True).sort_index()
    tapes: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for ticker, *_ in DELETIONS:
        path = _ticker_cache(ticker)
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
            if len(df) >= 60:
                tapes[ticker] = df
                continue
        missing.append(ticker)
    return tapes, spy, missing


# --------------------------------------------------------------------------- #
# Synthetic world — planted dump-then-rebound (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_event(seed: int, dump: float = 0.0, rebound: float = 0.0,
                     n_pre: int = 40, n_post: int = 90, rebound_days: int = 40,
                     sigma: float = 0.02) -> pd.DataFrame:
    """One synthetic "distressed stock vs market" excess-return path around a pseudo-event.

    Day index runs ``-n_pre .. +n_post``; the pseudo-effective day is offset 0. Daily excess
    (stock-minus-market) log returns are iid noise, EXCEPT: the [-5..0] window carries an
    extra ``-dump/6`` drift per day (the forced-selling dump into the effective date) and the
    (0, rebound_days] window carries an extra ``+rebound/rebound_days`` drift per day (the
    reversal). Both knobs at 0 reproduce the null world — no dump, no rebound, pure noise.
    """
    rng = np.random.default_rng(seed)
    offsets = np.arange(-n_pre, n_post + 1)
    ar = rng.normal(0.0, sigma, size=len(offsets))
    dump_mask = (offsets >= -5) & (offsets <= 0)
    ar[dump_mask] += -dump / 6.0
    reb_mask = (offsets > 0) & (offsets <= rebound_days)
    ar[reb_mask] += rebound / rebound_days
    return pd.DataFrame({"offset": offsets, "ar": ar}).set_index("offset")


def synthetic_panel(n_events: int = 60, seed: int = 652,
                     dump: float = 0.0, rebound: float = 0.0) -> list[pd.DataFrame]:
    """``n_events`` independent synthetic event paths sharing one (dump, rebound) world."""
    return [synthetic_event(seed=seed * 1000 + i, dump=dump, rebound=rebound)
            for i in range(n_events)]
