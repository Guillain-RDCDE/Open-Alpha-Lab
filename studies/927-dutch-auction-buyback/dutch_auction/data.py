"""Data layer for Study 927 — Dutch Auction (issuer self-tender event study).

Three inputs, one shape (date-indexed daily total-return closes plus an event table):

- ``EVENTS`` — a **hardcoded, publicly-verifiable** table of 180 issuer *modified Dutch
  auction* self-tender offers, 2010-01-01 → 2025-12-31. It is not hand-remembered: it is
  the machine-readable output of one EDGAR full-text-search query, and every row carries
  its **SEC accession number** so any reader can pull the original filing. The query,
  reproduced verbatim in ``EDGAR_QUERY``, is

      https://efts.sec.gov/LATEST/search-index
          ?q="modified Dutch auction"&forms=SC TO-I&startdt=...&enddt=...

  Filings are clustered per registrant (a >150-day gap starts a new offer) and the
  **earliest SC TO-I filing date of each cluster** is taken as the event date. That date
  is the offer's *commencement* — the day the tender documents become public — so it is
  point-in-time clean by construction. Some issuers pre-announce with an SC TO-C or a
  press release a few days earlier; the study measures the (−5, −1) run-up explicitly and
  sweeps the event date ±5 days rather than pretending the two coincide.

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for every event ticker plus **SPY** (the market
  benchmark the abnormal return is measured against) and **BIL** (the cash leg for the
  excess-of-cash Sharpe race). ``fetch`` touches the network and caches parquet in the
  **shared** ``studies/_cache`` (retry up to 4×); ``load_prices`` reads that cache
  **offline** and never imports yfinance.

- ``load_dollar_volume`` — an optional second cache (``dvol_<TK>_1d.parquet``, raw
  close × raw volume) used only for the *liquidity* screen. Absent cache ⇒ the screen is
  skipped and reported as such.

- ``synthetic_daily`` / ``synthetic_panel`` — *deterministic, offline* generators. The
  panel plants an announcement-day jump and a post-announcement drift whose sizes scale
  with ``signal_strength``; at ``signal_strength=0`` the world is a pure null and the
  event-study machinery must stay quiet. Seeds are fixed → tests are deterministic.

**Two named biases.** (1) *Survivorship / ticker recovery*: the CIK→ticker map comes from
SEC's current ``company_tickers.json``, so issuers that were acquired, went private or
were liquidated carry no ticker and drop out. (2) *Price tape*: yfinance serves a ticker's
history under its **current** symbol, so pre-rename history is retained but issuers whose
symbol was recycled after a bankruptcy are excluded by hand (``EXCLUDED_TICKERS``).
Both are named on the Signal axis in the README.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the event is
public at the close of day ``t``; the tradable leg is entered at the close of ``t+1``).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The SHARED desk cache (studies/_cache), not a per-study one.
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# Study-wide as-of: the last COMPLETE calendar month at build time.
AS_OF = "2026-06-30"

# The exact, re-runnable EDGAR full-text-search query behind EVENTS.
EDGAR_QUERY = (
    'https://efts.sec.gov/LATEST/search-index?q=%22modified+Dutch+auction%22'
    "&forms=SC+TO-I&dateRange=custom&startdt=YYYY-01-01&enddt=YYYY-12-31"
)

# Ticker symbols recycled by a *different* company after a bankruptcy, so the yfinance
# tape under that symbol is not the issuer that ran the tender. Excluded by hand.
EXCLUDED_TICKERS = ("CMLS",)

# The market benchmark (abnormal return = stock − SPY) and the cash leg (BIL).
MARKET = "SPY"
CASH = "BIL"

# --------------------------------------------------------------------------- #
# The event table: (ticker, SC TO-I commencement date, SEC accession, registrant)
# Machine-generated from the EDGAR query above; every row is verifiable at
# https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany  or by accession number.
# --------------------------------------------------------------------------- #
EVENTS = (
    ("ATLC", "2010-01-28", "0001188112-10-000152", "CompuCredit Holdings Corp"),
    ("CACC", "2010-06-18", "0000950123-10-059096", "CREDIT ACCEPTANCE CORP"),
    ("FIS", "2010-07-06", "0000950123-10-063756", "Fidelity National Information Services"),
    ("CASY", "2010-07-29", "0001193125-10-170510", "CASEYS GENERAL STORES INC"),
    ("TCX", "2010-09-17", "0001047469-10-008140", "TUCOWS INC /PA/"),
    ("SCHL", "2010-09-28", "0001193125-10-218104", "SCHOLASTIC CORP"),
    ("SLGN", "2010-10-08", "0001193125-10-226619", "SILGAN HOLDINGS INC"),
    ("CXDO", "2010-11-03", "0001354488-10-003328", "IMERGENT INC"),
    ("TREE", "2010-11-18", "0001047469-10-009890", "Tree.com, Inc."),
    ("WTM", "2011-08-18", "0001047469-11-007509", "WHITE MOUNTAINS INSURANCE GROUP LTD"),
    ("AMGN", "2011-11-08", "0001193125-11-300937", "AMGEN INC"),
    ("NATH", "2011-12-08", "0000921895-11-002301", "NATHANS FAMOUS INC"),
    ("TCX", "2011-12-20", "0001437749-11-009657", "TUCOWS INC /PA/"),
    ("MHH", "2012-02-07", "0001193125-12-043294", "Mastech Holdings, Inc."),
    ("HCKT", "2012-02-22", "0001193125-12-072750", "HACKETT GROUP, INC."),
    ("WW", "2012-02-23", "0001193125-12-074628", "WEIGHT WATCHERS INTERNATIONAL INC"),
    ("HR", "2012-06-06", "0001193125-12-261383", "HEALTHCARE TRUST OF AMERICA, INC."),
    ("GBLI", "2012-06-11", "0001193125-12-266305", "Global Indemnity plc"),
    ("VXRT", "2012-07-02", "0001193125-12-291057", "NABI BIOPHARMACEUTICALS"),
    ("CLS", "2012-10-29", "0001047469-12-009837", "CELESTICA INC"),
    ("SLGN", "2012-11-19", "0001193125-12-474983", "SILGAN HOLDINGS INC"),
    ("VICR", "2012-11-26", "0001193125-12-480124", "VICOR CORP"),
    ("NSP", "2012-11-26", "0001193125-12-480193", "INSPERITY, INC."),
    ("TCX", "2012-11-29", "0001437749-12-012294", "TUCOWS INC /PA/"),
    ("EHC", "2013-02-20", "0001193125-13-066145", "HEALTHSOUTH CORP"),
    ("FORR", "2013-04-03", "0001193125-13-140509", "FORRESTER RESEARCH, INC."),
    ("HLIT", "2013-04-26", "0001193125-13-175411", "HARMONIC INC"),
    ("TULP", "2013-07-18", "0001047469-13-007633", "INSIGNIA SYSTEMS INC/MN"),
    ("HAL", "2013-07-26", "0001193125-13-304868", "HALLIBURTON CO"),
    ("HCKT", "2013-08-28", "0001193125-13-349873", "HACKETT GROUP, INC."),
    ("LFVN", "2013-09-24", "0000849146-13-000023", "Lifevantage Corp"),
    ("WEN", "2014-01-14", "0000930413-14-000147", "Wendy's Co"),
    ("HELE", "2014-02-10", "0001047469-14-000714", "HELEN OF TROY LTD"),
    ("CPF", "2014-02-21", "0001047469-14-001124", "CENTRAL PACIFIC FINANCIAL CORP"),
    ("G", "2014-03-06", "0001193125-14-085460", "Genpact LTD"),
    ("IVT", "2014-03-14", "0001193125-14-099424", "Inland American Real Estate Trust, Inc"),
    ("FSK", "2014-04-16", "0001193125-14-144997", "FS Investment CORP"),
    ("SEB", "2014-05-12", "0001047469-14-004816", "SEABOARD CORP /DE/"),
    ("ATLC", "2014-06-23", "0001464343-14-000020", "Atlanticus Holdings Corp"),
    ("GSIT", "2014-07-09", "0001047469-14-006089", "GSI TECHNOLOGY INC"),
    ("DMC", "2014-11-03", "0001193125-14-392952", "FRESH DEL MONTE PRODUCE INC"),
    ("TCX", "2014-12-08", "0001174947-14-000548", "TUCOWS INC /PA/"),
    ("XHR", "2015-02-04", "0001193125-15-033216", "Xenia Hotels & Resorts, Inc."),
    ("SLGN", "2015-02-09", "0001193125-15-038366", "SILGAN HOLDINGS INC"),
    ("OMCC", "2015-02-10", "0001193125-15-040803", "NICHOLAS FINANCIAL INC"),
    ("FNF", "2015-02-23", "0001193125-15-057309", "Fidelity National Financial, Inc."),
    ("MBUU", "2015-04-10", "0001193125-15-125150", "Malibu Boats, Inc."),
    ("CLS", "2015-04-27", "0001047469-15-003939", "CELESTICA INC"),
    ("AIR", "2015-04-27", "0001047469-15-003956", "AAR CORP"),
    ("APLE", "2015-05-18", "0001185185-15-001346", "Apple Hospitality REIT, Inc."),
    ("CPRT", "2015-06-04", "0001193125-15-212656", "COPART INC"),
    ("AOSL", "2015-06-08", "0001174947-15-001014", "ALPHA & OMEGA SEMICONDUCTOR Ltd"),
    ("WEN", "2015-07-01", "0000950142-15-001567", "Wendy's Co"),
    ("NBHC", "2015-07-06", "0001047469-15-005969", "National Bank Holdings Corp"),
    ("ZARE", "2015-07-07", "0001193125-15-246871", "Dividend Capital Diversified Property "),
    ("MSI", "2015-08-07", "0001193125-15-281965", "Motorola Solutions, Inc."),
    ("HRB", "2015-09-02", "0001193125-15-310062", "H&R BLOCK INC"),
    ("NATH", "2015-09-18", "0000921895-15-002111", "NATHANS FAMOUS INC"),
    ("INVA", "2015-10-30", "0001047469-15-008165", "THERAVANCE INC"),
    ("VYX", "2015-11-13", "0001193125-15-376338", "NCR CORP"),
    ("WLFC", "2015-11-17", "0001047469-15-008718", "WILLIS LEASE FINANCE CORP"),
    ("NSP", "2015-12-08", "0001193125-15-397182", "INSPERITY, INC."),
    ("FC", "2015-12-14", "0001047469-15-009182", "FRANKLIN COVEY CO"),
    ("SCHL", "2016-01-21", "0001144204-16-076395", "SCHOLASTIC CORP"),
    ("PAMT", "2016-02-18", "0001174947-16-002100", "PAM TRANSPORTATION SERVICES INC"),
    ("CRAI", "2016-02-22", "0001047469-16-010267", "CRA INTERNATIONAL, INC."),
    ("III", "2016-03-10", "0001047469-16-010960", "Information Services Group Inc."),
    ("ZION", "2016-04-25", "0001193125-16-554192", "ZIONS BANCORPORATION /UT/"),
    ("MNST", "2016-05-10", "0001047469-16-013044", "Monster Beverage Corp"),
    ("VRTS", "2016-05-10", "0001193125-16-584866", "VIRTUS INVESTMENT PARTNERS, INC."),
    ("MSM", "2016-07-07", "0001144204-16-111880", "MSC INDUSTRIAL DIRECT CO INC"),
    ("SLGN", "2016-10-17", "0001193125-16-739187", "SILGAN HOLDINGS INC"),
    ("RGP", "2016-10-18", "0001193125-16-740261", "RESOURCES CONNECTION INC"),
    ("IVT", "2016-10-27", "0001193125-16-748797", "InvenTrust Properties Corp."),
    ("ZARE", "2016-11-09", "0001327978-16-000046", "Dividend Capital Diversified Property "),
    ("GECC", "2017-04-06", "0001193125-17-112091", "Great Elm Capital Corp."),
    ("ARQ", "2017-05-08", "0001515156-17-000045", "Advanced Emissions Solutions, Inc."),
    ("LWAY", "2017-06-26", "0001683168-17-001655", "LIFEWAY FOODS INC"),
    ("WTM", "2017-08-17", "0001047469-17-005289", "WHITE MOUNTAINS INSURANCE GROUP LTD"),
    ("HLF", "2017-08-21", "0001193125-17-262794", "HERBALIFE LTD."),
    ("PAMT", "2017-10-10", "0001571049-17-008149", "PAM TRANSPORTATION SERVICES INC"),
    ("CHDN", "2018-01-10", "0001193125-18-007791", "CHURCHILL DOWNS Inc"),
    ("AMGN", "2018-02-05", "0001193125-18-031432", "AMGEN INC"),
    ("SCKT", "2018-03-12", "0000944075-18-000006", "SOCKET MOBILE, INC."),
    ("WTM", "2018-04-10", "0001047469-18-002667", "WHITE MOUNTAINS INSURANCE GROUP LTD"),
    ("HLF", "2018-04-18", "0001193125-18-120295", "HERBALIFE LTD."),
    ("WHR", "2018-04-26", "0001193125-18-134842", "WHIRLPOOL CORP /DE/"),
    ("ABBV", "2018-05-01", "0001047469-18-003374", "AbbVie Inc."),
    ("PAMT", "2018-05-08", "0001174947-18-000750", "PAM TRANSPORTATION SERVICES INC"),
    ("CLAR", "2018-05-08", "0001144204-18-025866", "Clarus Corp"),
    ("MTEX", "2018-05-18", "0001056358-18-000064", "MANNATECH INC"),
    ("QCOM", "2018-07-31", "0001047469-18-005308", "QUALCOMM INC/DE"),
    ("BBDC", "2018-08-07", "0001193125-18-240456", "Barings BDC, Inc."),
    ("IVT", "2018-08-15", "0001193125-18-249268", "InvenTrust Properties Corp."),
    ("RAMP", "2018-11-13", "0001193125-18-324979", "LiveRamp Holdings, Inc."),
    ("BPYPM", "2019-02-11", "0001104659-19-007111", "Brookfield Property Partners L.P."),
    ("JCI", "2019-05-03", "0001193125-19-136950", "Johnson Controls International plc"),
    ("PAMT", "2019-05-13", "0001174947-19-000670", "PAM TRANSPORTATION SERVICES INC"),
    ("BMNM", "2019-05-29", "0001140361-19-009976", "BIMINI CAPITAL MANAGEMENT, INC."),
    ("LYB", "2019-06-10", "0001410578-19-000308", "LyondellBasell Industries N.V."),
    ("GME", "2019-06-11", "0001193125-19-170214", "GameStop Corp."),
    ("BALY", "2019-06-25", "0001144204-19-032320", "Twin River Worldwide Holdings, Inc."),
    ("BFH", "2019-07-19", "0001193125-19-197208", "ALLIANCE DATA SYSTEMS CORP"),
    ("DVA", "2019-07-22", "0001193125-19-198449", "DAVITA INC."),
    ("ULH", "2019-08-05", "0001174947-19-000954", "UNIVERSAL LOGISTICS HOLDINGS, INC."),
    ("LBTYA", "2019-08-12", "0001193125-19-218521", "Liberty Global plc"),
    ("NSLR", "2019-10-21", "0001104659-19-055263", "SUTTER ROCK CAPITAL CORP."),
    ("SVVC", "2019-12-17", "0001140361-19-022723", "Firsthand Technology Value Fund, Inc."),
    ("XBIT", "2020-01-14", "0001387131-20-000238", "XBiotech Inc."),
    ("MGM", "2020-02-13", "0001193125-20-034386", "MGM Resorts International"),
    ("MTEX", "2020-05-29", "0001056358-20-000049", "MANNATECH INC"),
    ("HLF", "2020-07-13", "0001193125-20-191599", "HERBALIFE NUTRITION LTD."),
    ("MSTR", "2020-08-11", "0001193125-20-216338", "MICROSTRATEGY Inc"),
    ("NOMD", "2020-08-11", "0001193125-20-215627", "Nomad Foods Ltd"),
    ("DVA", "2020-08-17", "0001193125-20-221812", "DAVITA INC."),
    ("AMCX", "2020-09-16", "0001193125-20-246637", "AMC Networks Inc."),
    ("HTH", "2020-09-23", "0001104659-20-107535", "Hilltop Holdings Inc."),
    ("NXDT", "2020-10-30", "0001193125-20-282692", "NexPoint Strategic Opportunities Fund"),
    ("MVBF", "2020-11-17", "0001104659-20-126136", "MVB FINANCIAL CORP"),
    ("AMTB", "2020-11-20", "0001193125-20-298747", "Amerant Bancorp Inc."),
    ("OPTU", "2020-11-23", "0001104659-20-128206", "Altice USA, Inc."),
    ("WHLR", "2020-12-23", "0001213900-20-044397", "Wheeler Real Estate Investment Trust, "),
    ("SLM", "2021-02-02", "0001193125-21-025391", "SLM Corp"),
    ("SENEA", "2021-02-08", "0001104659-21-013394", "Seneca Foods Corp"),
    ("WAFD", "2021-02-09", "0001193125-21-033405", "WASHINGTON FEDERAL INC"),
    ("CRAI", "2021-03-08", "0001104659-21-033019", "CRA INTERNATIONAL, INC."),
    ("MPC", "2021-05-17", "0001193125-21-162819", "Marathon Petroleum Corp"),
    ("EBMT", "2021-05-21", "0001140361-21-018267", "Eagle Bancorp Montana, Inc."),
    ("BMNM", "2021-05-27", "0001104659-21-073232", "BIMINI CAPITAL MANAGEMENT, INC."),
    ("BOX", "2021-06-02", "0001193125-21-179185", "BOX INC"),
    ("PAMT", "2021-07-27", "0001104659-21-096179", "PAM TRANSPORTATION SERVICES INC"),
    ("SLAB", "2021-08-03", "0001193125-21-233985", "SILICON LABORATORIES INC."),
    ("CVLG", "2021-08-09", "0001008886-21-000080", "COVENANT LOGISTICS GROUP, INC."),
    ("ISBA", "2021-09-02", "0000842517-21-000181", "ISABELLA BANK Corp"),
    ("SFES", "2021-10-04", "0001104659-21-122315", "SAFEGUARD SCIENTIFICS INC"),
    ("IVT", "2021-10-12", "0001193125-21-296266", "InvenTrust Properties Corp."),
    ("ECPG", "2021-11-04", "0001193125-21-319365", "ENCORE CAPITAL GROUP INC"),
    ("PRG", "2021-11-04", "0001193125-21-319242", "PROG Holdings, Inc."),
    ("CORT", "2021-11-08", "0001193125-21-322396", "CORCEPT THERAPEUTICS INC"),
    ("RRR", "2021-11-10", "0001193125-21-325421", "Red Rock Resorts, Inc."),
    ("TNET", "2022-02-17", "0001104659-22-024466", "TRINET GROUP, INC."),
    ("HTH", "2022-05-02", "0001104659-22-053972", "Hilltop Holdings Inc."),
    ("CMLS", "2022-05-06", "0001193125-22-143656", "CUMULUS MEDIA INC"),
    ("IMO", "2022-05-06", "0001193125-22-143506", "IMPERIAL OIL LTD"),
    ("ULH", "2022-05-13", "0001193125-22-149503", "UNIVERSAL LOGISTICS HOLDINGS, INC."),
    ("ASTL", "2022-06-21", "0001193125-22-177083", "Algoma Steel Group Inc."),
    ("BALY", "2022-06-24", "0001104659-22-074102", "Bally's Corp"),
    ("OPXS", "2022-08-18", "0001493152-22-023412", "Optex Systems Holdings Inc"),
    ("WTM", "2022-08-22", "0001104659-22-093064", "WHITE MOUNTAINS INSURANCE GROUP LTD"),
    ("NSLR", "2022-09-07", "0001104659-22-097955", "SURO CAPITAL CORP."),
    ("TBPH", "2022-09-28", "0001104659-22-103472", "Theravance Biopharma, Inc."),
    ("SCHL", "2022-10-25", "0001193125-22-268161", "SCHOLASTIC CORP"),
    ("WHLR", "2022-11-01", "0001213900-22-067963", "Wheeler Real Estate Investment Trust, "),
    ("TNET", "2022-11-07", "0001104659-22-115099", "TRINET GROUP, INC."),
    ("TFIN", "2022-11-07", "0001193125-22-278496", "Triumph Bancorp, Inc."),
    ("HCKT", "2022-11-09", "0001193125-22-280761", "HACKETT GROUP, INC."),
    ("IDCC", "2023-01-23", "0001104659-23-005597", "InterDigital, Inc."),
    ("CORT", "2023-03-06", "0001193125-23-060717", "CORCEPT THERAPEUTICS INC"),
    ("NSLR", "2023-04-19", "0001104659-23-047032", "SURO CAPITAL CORP."),
    ("VVV", "2023-05-11", "0001193125-23-141428", "VALVOLINE INC"),
    ("CMLS", "2023-05-12", "0001193125-23-142789", "CUMULUS MEDIA INC"),
    ("MSIF", "2023-05-15", "0001535778-23-000037", "MSC INCOME FUND, INC."),
    ("XBIT", "2023-05-17", "0001193125-23-146632", "XBiotech Inc."),
    ("AIRT", "2023-05-18", "0001437749-23-015119", "AIR T INC"),
    ("OPY", "2023-05-31", "0001104659-23-066256", "OPPENHEIMER HOLDINGS INC"),
    ("DBI", "2023-06-08", "0001193125-23-163260", "Designer Brands Inc."),
    ("IMO", "2023-11-03", "0000049938-23-000084", "IMPERIAL OIL LTD"),
    ("OSPN", "2023-11-13", "0001839882-23-030392", "OneSpan Inc."),
    ("NSLR", "2024-02-20", "0001104659-24-025256", "SURO CAPITAL CORP."),
    ("CNNE", "2024-03-01", "0001193125-24-054315", "Cannae Holdings, Inc."),
    ("GPRK", "2024-03-20", "0000950103-24-004046", "GeoPark Ltd"),
    ("PAMT", "2024-04-24", "0001174947-24-000584", "PAM TRANSPORTATION SERVICES INC"),
    ("MNST", "2024-05-08", "0001104659-24-058430", "Monster Beverage Corp"),
    ("INCY", "2024-05-13", "0001104659-24-060326", "INCYTE CORP"),
    ("COKE", "2024-05-20", "0001193125-24-142266", "Coca-Cola Consolidated, Inc."),
    ("WEX", "2025-02-26", "0001140361-25-006098", "WEX Inc."),
    ("PAMT", "2025-04-03", "0001174947-25-000508", "PAMT CORP"),
    ("IZEA", "2025-05-16", "0001495231-25-000107", "IZEA Worldwide, Inc."),
    ("HCKT", "2025-11-05", "0001193125-25-267274", "HACKETT GROUP, INC."),
    ("WTM", "2025-11-21", "0001104659-25-114719", "WHITE MOUNTAINS INSURANCE GROUP LTD"),
)


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return, cache-only by default
# --------------------------------------------------------------------------- #
def event_tickers() -> tuple:
    """Distinct issuer tickers in ``EVENTS`` (excluding the recycled-symbol names)."""
    return tuple(sorted({t for t, _, _, _ in EVENTS if t not in EXCLUDED_TICKERS}))


def all_tickers() -> tuple:
    """Every ticker the study needs: the issuers plus SPY (market) and BIL (cash)."""
    return tuple(sorted(set(event_tickers()) | {MARKET, CASH}))


def _cache_path(ticker: str, cache_dir: str, prefix: str = "prices") -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"{prefix}_{safe}_1d.parquet")


def fetch(
    tickers=None,
    start: str = "2008-01-01",
    end: str = "2026-07-01",
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
    with_volume: bool = True,
) -> list:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` so ``close``
    is split- and dividend-adjusted total return — mandatory here, because several event
    issuers paid large special dividends inside the measurement windows. With
    ``with_volume`` a second, *unadjusted* pull caches raw close × raw volume as
    ``dvol_<TK>_1d.parquet`` for the liquidity screen only.

    Returns the list of tickers that failed (delisted / no tape) rather than raising, so
    one dead symbol cannot abort a 130-name pull.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    if tickers is None:
        tickers = all_tickers()
    os.makedirs(cache_dir, exist_ok=True)
    failed = []
    for tk in tickers:
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(tk, start=start, end=end, interval="1d",
                                  auto_adjust=True, progress=False, threads=False)
                if raw is not None and len(raw) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        if raw is None or len(raw) == 0:
            failed.append(tk)
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        df = raw[["close"]].copy()
        df.index = pd.to_datetime(df.index)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "date"
        df = df.dropna(subset=["close"])
        df.to_parquet(_cache_path(tk, cache_dir))

        if with_volume and tk not in (MARKET, CASH):
            vraw = None
            for _ in range(retries):
                try:
                    vraw = yf.download(tk, start=start, end=end, interval="1d",
                                       auto_adjust=False, progress=False, threads=False)
                    if vraw is not None and len(vraw) > 0:
                        break
                except Exception:
                    time.sleep(2.0)
            if vraw is None or len(vraw) == 0:
                continue
            if isinstance(vraw.columns, pd.MultiIndex):
                vraw.columns = vraw.columns.get_level_values(0)
            vraw = vraw.rename(columns=str.lower)
            if "close" not in vraw.columns or "volume" not in vraw.columns:
                continue
            dv = pd.DataFrame({"dollar_volume": (vraw["close"] * vraw["volume"]).astype(float)})
            dv.index = pd.to_datetime(vraw.index)
            if getattr(dv.index, "tz", None) is not None:
                dv.index = dv.index.tz_localize(None)
            dv.index.name = "date"
            dv.dropna().to_parquet(_cache_path(tk, cache_dir, prefix="dvol"))
    return failed


def have_real(cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff the benchmark legs and at least 50 issuer tapes are cached."""
    if not all(os.path.exists(_cache_path(t, cache_dir)) for t in (MARKET, CASH)):
        return False
    have = sum(os.path.exists(_cache_path(t, cache_dir)) for t in event_tickers())
    return have >= 50


def _read_close(path: str, asof: str) -> pd.Series:
    s = pd.read_parquet(path)["close"]
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    s = s.sort_index()
    return s[s.index <= pd.Timestamp(asof)]


def load_prices(
    tickers=None,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
    require_market: bool = True,
) -> dict:
    """Read cached daily total-return closes OFFLINE into ``{ticker: Series}``.

    Ragged by design (each issuer has its own listing history), sliced to ``asof`` so the
    sample never creeps. Missing issuer tapes are silently skipped — a 130-name panel is
    always incomplete — but a missing **SPY** or **BIL** raises, because the abnormal
    return and the cash race are not defined without them. Never touches the network.
    """
    if tickers is None:
        tickers = all_tickers()
    out = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            if require_market and tk in (MARKET, CASH):
                raise FileNotFoundError(
                    f"No cached prices for the {tk} leg at {path}. "
                    f"Call dutch_auction.data.fetch() once to populate the shared cache."
                )
            continue
        s = _read_close(path, asof)
        if len(s) > 0:
            out[tk] = s
    if require_market and MARKET not in out:
        raise FileNotFoundError(f"No cached prices for {MARKET}.")
    return out


def load_dollar_volume(tickers=None, cache_dir: str = DEFAULT_CACHE,
                       asof: str = AS_OF) -> dict:
    """Read the optional ``dvol_<TK>_1d.parquet`` cache (raw close × raw volume).

    Used **only** for the liquidity screen. Approximate across splits (yfinance does not
    split-adjust volume), so it is an order-of-magnitude filter, not a microstructure
    measure. Returns ``{}`` when the cache is absent, and the screen is then reported as
    not applied rather than silently assumed.
    """
    if tickers is None:
        tickers = event_tickers()
    out = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir, prefix="dvol")
        if not os.path.exists(path):
            continue
        s = pd.read_parquet(path)["dollar_volume"]
        s.index = pd.to_datetime(s.index)
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
        s = s.sort_index()
        out[tk] = s[s.index <= pd.Timestamp(asof)]
    return out


FP_START = "2009-01-01"


def study_fingerprint(prices: dict, start: str = FP_START, asof: str = AS_OF) -> str:
    """Content hash of exactly what this study consumes: **returns**, ``start`` → ``asof``.

    Two independent sources of false drift are designed out, because the desk cache is
    shared and neighbouring studies re-fetch it:

    1. *Window.* A neighbour re-pulling SPY from 1993 instead of 2008 must not move this
       stamp, so the hash is taken on the study's own date slice, not the whole file.
    2. *Re-adjustment.* ``auto_adjust=True`` back-adjusts the entire price history every
       time a new dividend is paid, which multiplies every historical level by a constant
       and leaves every single return — and therefore every number in this study —
       untouched. Hashing *levels* would stamp a change that does not exist; hashing the
       daily simple returns (rounded to 1e-9, far coarser than the 1e-16 float noise a
       rescale introduces and far finer than anything the study reports) stamps the tape
       the study actually reads.

    The stamp therefore moves when a price *return* moves and stays put when only a
    neighbour's fetch parameters or a fresh dividend do.
    """
    h = hashlib.sha1()
    lo, hi = pd.Timestamp(start), pd.Timestamp(asof)
    for k in sorted(prices):
        s = prices[k]
        s = s[(s.index >= lo) & (s.index <= hi)]
        h.update(k.encode())
        h.update(str(len(s)).encode())
        r = np.diff(np.asarray(s, dtype=float)) / np.asarray(s, dtype=float)[:-1] \
            if len(s) > 1 else np.zeros(0)
        arr = np.round(r, 9)
        h.update(np.ascontiguousarray(np.nan_to_num(arr, nan=0.0)).tobytes())
    return h.hexdigest()[:12]


def fingerprint(obj) -> str:
    """Short content fingerprint of a price frame / dict of series, for the data stamp."""
    if isinstance(obj, dict):
        h = hashlib.sha1()
        for k in sorted(obj):
            h.update(k.encode())
            arr = np.ascontiguousarray(np.asarray(obj[k], dtype=float))
            h.update(np.nan_to_num(arr, nan=0.0).tobytes())
        return h.hexdigest()[:12]
    arr = np.ascontiguousarray(pd.DataFrame(obj).to_numpy(dtype=float))
    return hashlib.sha1(np.nan_to_num(arr, nan=0.0).tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tapes — the deterministic offline core
# --------------------------------------------------------------------------- #
def synthetic_daily(
    n_days: int = 1200,
    drift_ann: float = 0.06,
    vol_ann: float = 0.32,
    start: str = "2012-01-02",
    seed: int = 927,
) -> pd.DataFrame:
    """One single-name daily total-return tape plus a market leg and a cash leg.

    A market factor with beta 1.0 and an idiosyncratic sleeve — the minimum structure an
    abnormal-return (stock − market) event study needs. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    d = drift_ann / TRADING_DAYS_PER_YEAR
    s = vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    mkt = rng.normal(0.07 / TRADING_DAYS_PER_YEAR, 0.16 / np.sqrt(TRADING_DAYS_PER_YEAR), n_days)
    idio = rng.normal(d, s, n_days)
    stock = 50.0 * np.exp(np.cumsum(mkt + idio))
    market = 100.0 * np.exp(np.cumsum(mkt))
    cash = np.cumprod(np.full(n_days, (1.0 + 0.02) ** (1.0 / TRADING_DAYS_PER_YEAR)))
    return pd.DataFrame({"stock": stock, "market": market, "cash": cash},
                        index=pd.DatetimeIndex(dates, name="date"))


def synthetic_panel(
    n_events: int = 145,
    n_days: int = 1400,
    signal_strength: float = 1.0,
    jump_bps: float = 470.0,          # planted announcement-day abnormal return
    drift_bps: float = 300.0,         # planted post-announcement abnormal drift (6m)
    vol_ann: float = 0.34,
    start: str = "2012-01-02",
    seed: int = 927,
) -> tuple:
    """A panel of synthetic issuers with a *planted* self-tender event in each.

    Every name gets an independent idiosyncratic tape on top of one shared market factor
    (so the panel has the same cross-sectional correlation problem as the real one). On
    the planted event day the stock takes an extra ``signal_strength × jump_bps`` return,
    and over the following 126 trading days it accrues ``signal_strength × drift_bps``
    of abnormal drift, spread evenly.

    - ``signal_strength = 1`` → a real announcement pop *and* a real post-event drift; the
      event study must recover both.
    - ``signal_strength = 0`` → the pure null; the machinery must stay quiet.

    Returns ``(panel, events, truth)`` where ``panel`` is ``{name: close Series}`` with a
    ``"MKT"`` market leg and a ``"CASH"`` leg, ``events`` is a list of
    ``(name, date_str, accession, label)`` 4-tuples matching the ``EVENTS`` shape, and
    ``truth`` records the planted parameters. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    s = vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    mkt_r = rng.normal(0.07 / TRADING_DAYS_PER_YEAR,
                       0.16 / np.sqrt(TRADING_DAYS_PER_YEAR), n_days)

    panel = {"MKT": pd.Series(100.0 * np.exp(np.cumsum(mkt_r)),
                              index=pd.DatetimeIndex(dates, name="date"))}
    panel["CASH"] = pd.Series(
        np.cumprod(np.full(n_days, (1.0 + 0.02) ** (1.0 / TRADING_DAYS_PER_YEAR))),
        index=pd.DatetimeIndex(dates, name="date"))

    lo, hi = 300, n_days - 200          # room for the pre-window and the 6m post-window
    events = []
    jump = signal_strength * jump_bps * 1e-4
    drift_per_day = signal_strength * drift_bps * 1e-4 / 126.0
    for i in range(n_events):
        name = f"SYN{i:03d}"
        idio = rng.normal(0.0, s, n_days)
        p = int(rng.integers(lo, hi))
        idio[p] += jump
        idio[p + 20: p + 20 + 126] += drift_per_day
        panel[name] = pd.Series(50.0 * np.exp(np.cumsum(mkt_r + idio)),
                                index=pd.DatetimeIndex(dates, name="date"))
        events.append((name, str(dates[p].date()), f"SYN-{i:04d}", f"Synthetic Issuer {i}"))

    # The plant is applied to LOG returns, but the event study measures a simple
    # buy-and-hold abnormal return, so the recoverable quantity is not the planted log
    # drift: E[e^S − 1] carries a Jensen term σ²/2. Over 126 sessions of idiosyncratic
    # vol that term is large, and quoting the log plant as the target would make an
    # unbiased harness look badly biased. Both numbers are therefore reported.
    sig_6m = vol_ann * np.sqrt(126.0 / TRADING_DAYS_PER_YEAR)
    log_drift_6m = float(drift_per_day * 126.0)
    truth = {
        "n_events": n_events, "n_days": n_days, "signal_strength": signal_strength,
        "jump_bps": jump_bps, "drift_bps": drift_bps, "seed": seed,
        "planted_jump": float(jump),
        "planted_drift_6m": log_drift_6m,
        "planted_drift_6m_log": log_drift_6m,
        "expected_simple_drift_6m": float(np.exp(log_drift_6m + 0.5 * sig_6m ** 2) - 1.0),
        "vol_ann": vol_ann,
    }
    return panel, events, truth
