"""Data layer for Study 745 — Corporate-Jet-Index (governance-red-flag long/short).

Two sources, both offline-friendly:

* **Real tape.** A hardcoded, transparent table of ~24 large-caps (``JET_FIRMS``: ticker,
  heavy/low-perk flag, the year the CEO personal-aircraft perk became public, and a short
  citable note), plus **monthly** total-return closes for each name and SPY (yfinance,
  ``auto_adjust=True`` → dividends reinvested, so this is a *total-return* tape), cached
  under ``_cache/`` as one parquet per ticker. True governance-perk panels (ISS, proxy
  databases) are not free, so the dated, labelled table is the transparent stand-in —
  every input is a public price and a public, citable proxy-statement/press red flag.

* **Synthetic.** A deterministic, fixed-seed generator that builds a monthly long/short
  panel with a *plantable* governance-discount edge (``alpha_bps_month``). It is the
  positive control: with the edge set to zero the HAC inference must NOT manufacture
  significance out of ~10 years of monthly data; with a real planted discount it must
  light up.

**Survivorship, named up front.** The archetypal jet abusers (Tyco/Kozlowski,
WorldCom/Ebbers, Enron, Chesapeake/McClendon) delisted to ~zero and cannot enter a
yfinance tape — see ``DELISTED_ABUSERS``. A survivor-only heavy basket is therefore
biased *against* Yermack's underperformance claim (the worst flyers are missing) — a
caveat that travels on the Signal axis.

Pure numpy + pandas + stdlib for the offline path. ``fetch_prices`` (network) is only
used once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# --------------------------------------------------------------------------- #
# Hardcoded corporate-jet perk table.
# Columns: ticker, heavy (True = documented CEO personal-aircraft red flag;
#          False = matched low-perk / frugal-reputation peer),
#          public_year (the year the perk became a public red flag — SEC proxy
#          disclosure, press exposé, or well-documented reputation), note.
# Sources: company DEF 14A proxy statements (the "All Other Compensation" /
# perquisites tables that itemise personal aircraft use), SEC enforcement, and
# contemporaneous financial-press coverage (WSJ / Reuters / Bloomberg / NYT /
# Forbes). "Heavy" = a CEO whose personal use of the corporate aircraft is a
# documented, itemised perk or a notorious governance talking-point; "Low" =
# a large-cap peer with a frugal reputation or no material personal-aircraft
# disclosure. The heavy/low call is the believers' own framing (Yermack's
# "flights of fancy") and is, of course, somewhat subjective at the margin — we
# say so on the Signal axis. Large, long-listed survivors only, so the monthly
# history is clean; the famous abusers that DELISTED are listed separately below.
# --------------------------------------------------------------------------- #
_RAW_FIRMS = [
    # (ticker, heavy, public_year, note)
    # --- HEAVY: documented CEO personal-aircraft red flag ------------------- #
    ("ORCL", True, 2006, "Larry Ellison — personal aircraft itemised in proxies (Yermack-era)"),
    ("OXY",  True, 2009, "Ray Irani — outsized perks incl. personal aircraft use"),
    ("ANF",  True, 2010, "Michael Jeffries — notorious personal-use aircraft rules (SEC proxy)"),
    ("GE",   True, 2010, "corporate-jet perks (Welch legacy; Immelt 'empty backup jet' 2017)"),
    ("DIS",  True, 2011, "Eisner/Iger-era personal aircraft required & disclosed"),
    ("LVS",  True, 2011, "Sheldon Adelson — personal use of Las Vegas Sands aircraft"),
    ("WYNN", True, 2011, "Steve Wynn — personal aircraft use disclosed in proxies"),
    ("CMCSA",True, 2012, "Comcast — Roberts family corporate-aircraft perquisites"),
    ("IEP",  True, 2013, "Carl Icahn — personal aircraft, Icahn Enterprises"),
    ("META", True, 2015, "Mark Zuckerberg — private-aircraft security/personal allowance"),
    ("GOOGL",True, 2015, "Alphabet — exec aircraft (founders' fleet, Moffett Field)"),
    ("TSLA", True, 2018, "Elon Musk — heavily tracked personal jet use; security perk"),
    # --- LOW: matched frugal-reputation / no material disclosure peers ------ #
    ("BRK-B",False, 2006, "Warren Buffett — famed frugality (flew commercial for years)"),
    ("COST", False, 2006, "Costco — frugal culture, no material personal-aircraft perk"),
    ("WMT",  False, 2006, "Walmart — Walton frugality ethos"),
    ("HD",   False, 2006, "Home Depot — no notable personal-aircraft red flag"),
    ("TGT",  False, 2006, "Target — matched big-box retailer, low-perk"),
    ("LOW",  False, 2006, "Lowe's — matched retailer, no material aircraft perk"),
    ("CSCO", False, 2006, "Cisco — large-cap tech peer, low-perk reputation"),
    ("INTC", False, 2006, "Intel — large-cap tech peer, low-perk reputation"),
    ("PG",   False, 2006, "Procter & Gamble — no material personal-aircraft red flag"),
    ("JNJ",  False, 2006, "Johnson & Johnson — no material personal-aircraft red flag"),
    ("MMM",  False, 2006, "3M — industrial peer, low-perk reputation"),
    ("TXN",  False, 2006, "Texas Instruments — famed capital discipline / frugality"),
]

# The famous jet abusers who DELISTED to ~zero — they CANNOT enter a yfinance tape,
# which is exactly why a survivor-only heavy basket is biased AGAINST the claim.
# (Named on the Signal axis; not priced.)
DELISTED_ABUSERS = [
    ("Tyco / Dennis Kozlowski", "2002", "looting incl. jets/parties; stock collapsed, broken up"),
    ("WorldCom / Bernie Ebbers", "2002", "accounting fraud; jets among perks; bankrupt, delisted"),
    ("Enron / Ken Lay & Skilling", "2001", "corporate fleet; bankrupt, delisted to zero"),
    ("Chesapeake Energy / Aubrey McClendon", "2012", "lavish perks incl. jets; 2020 bankruptcy"),
    ("Adelphia / John Rigas", "2002", "personal jet paid by shareholders; fraud, delisted"),
]

# De-duplicate by ticker and build the canonical table.
_seen: set = set()
JET_FIRMS: list[dict] = []
for _t, _h, _y, _n in _RAW_FIRMS:
    if _t in _seen:
        continue
    _seen.add(_t)
    JET_FIRMS.append({"ticker": _t, "heavy": bool(_h), "public_year": int(_y), "note": _n})

TICKERS = sorted({r["ticker"] for r in JET_FIRMS})
HEAVY = [r["ticker"] for r in JET_FIRMS if r["heavy"]]
LOW = [r["ticker"] for r in JET_FIRMS if not r["heavy"]]


# --------------------------------------------------------------------------- #
# Real tape (network) — one parquet per ticker, plus SPY (monthly total return)
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str = DEFAULT_CACHE) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "").replace("-", "")
    return os.path.join(cache_dir, f"prices_745_{safe}_1mo.parquet")


def fetch_prices(start: str = "2006-01-01", end: str | None = None,
                 cache_dir: str = DEFAULT_CACHE) -> None:
    """Download monthly total-return closes for every firm + SPY and cache parquet.

    Network-only; used once to build ``_cache/``. Never imported by the offline notebook
    cells. ``auto_adjust=True`` reinvests dividends and applies splits, so ``close`` is a
    **total-return** level (labelled as such everywhere). One parquet per ticker (column
    ``close``, monthly index ``date``).
    """
    import yfinance as yf

    os.makedirs(cache_dir, exist_ok=True)
    for ticker in TICKERS + ["SPY"]:
        raw = yf.download(ticker, start=start, end=end, interval="1mo",
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        if raw.empty or "close" not in raw.columns:
            continue
        out = raw[["close"]].copy()
        out.index = pd.DatetimeIndex(out.index).tz_localize(None)
        out.index.name = "date"
        out.to_parquet(_cache_path(ticker, cache_dir))


def have_real(cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff SPY and at least most firm tickers are cached."""
    if not os.path.exists(_cache_path("SPY", cache_dir)):
        return False
    have = sum(os.path.exists(_cache_path(t, cache_dir)) for t in TICKERS)
    return have >= max(1, int(0.8 * len(TICKERS)))


def load_prices(cache_dir: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load cached monthly closes into a wide frame (index = month, columns = names + SPY)."""
    series = {}
    for ticker in TICKERS + ["SPY"]:
        p = _cache_path(ticker, cache_dir)
        if not os.path.exists(p):
            continue
        s = pd.read_parquet(p)["close"]
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        series[ticker] = s
    df = pd.DataFrame(series).sort_index()
    # month-end normalise so all names align on one monthly grid
    df.index = df.index.to_period("M").to_timestamp("M")
    df = df[~df.index.duplicated(keep="last")]
    return df


def load_real(cache_dir: str = DEFAULT_CACHE) -> tuple[pd.DataFrame, list[dict]]:
    """Convenience: cached wide monthly-price frame + the firm table (only priced names)."""
    prices = load_prices(cache_dir)
    firms = [f for f in JET_FIRMS if f["ticker"] in prices.columns]
    return prices, firms


# --------------------------------------------------------------------------- #
# Synthetic positive control — deterministic, no network
# --------------------------------------------------------------------------- #
def synthetic_panel(n_months: int = 186, alpha_bps_month: float = 0.0,
                    seed: int = 745, n_heavy: int = 12, n_low: int = 12,
                    mkt_vol: float = 0.043, idio_vol: float = 0.075,
                    beta_heavy: float = 1.0, beta_low: float = 1.0) -> dict:
    """Deterministic monthly long/short panel with a plantable governance-discount edge.

    Each name has ``r = beta*mkt + eps``; the HEAVY basket additionally carries a monthly
    abnormal drift of ``alpha_bps_month`` bp (Yermack's discount is *negative* — a drag on
    the flyers). The long/short book is **low − heavy** each month, so a negative planted
    heavy alpha shows up as a *positive* long/short mean. The two baskets share the same
    market beta by default, so under the null (``alpha_bps_month = 0``) the excess-of-market
    long/short is centered on zero and the HAC inference must NOT manufacture significance;
    with a large planted discount it must light up. (On the *real* tape the baskets do NOT
    share a beta — that gap is the study's central confound, told separately.)

    Returns a dict with:
      ``ls``      — monthly long/short (low − heavy) excess-of-market return series
      ``heavy``, ``low`` — monthly basket excess-of-market returns
      ``mkt``     — monthly market return
      ``truth``   — the planted parameters.
    """
    rng = np.random.default_rng(seed)
    jump = alpha_bps_month * 1e-4
    mkt = rng.normal(0.006, mkt_vol, n_months)

    def basket(n, beta, drift):
        rets = np.empty((n, n_months))
        for j in range(n):
            rets[j] = beta * mkt + rng.normal(0.0, idio_vol, n_months) + drift
        return rets.mean(axis=0)

    heavy = basket(n_heavy, beta_heavy, jump)     # planted discount lives here
    low = basket(n_low, beta_low, 0.0)
    heavy_x = heavy - mkt
    low_x = low - mkt
    ls = low_x - heavy_x
    return {
        "ls": ls, "heavy": heavy_x, "low": low_x, "mkt": mkt,
        "truth": {"n_months": n_months, "alpha_bps_month": alpha_bps_month,
                  "seed": seed, "n_heavy": n_heavy, "n_low": n_low},
    }


def fingerprint(firms: list[dict]) -> str:
    """Short content fingerprint of the perk table (ticker+heavy+year), for as-of stamps."""
    payload = "|".join(f"{f['ticker']}:{int(f['heavy'])}:{f['public_year']}"
                       for f in sorted(firms, key=lambda r: r["ticker"]))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
