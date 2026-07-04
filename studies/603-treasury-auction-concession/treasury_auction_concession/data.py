"""Data layer for Study 603 — Treasury Auction Concession.

Two sources, both offline-friendly once cached:

* **Real tape.**
    1. **Auction records** — every 10-Year Note and 30-Year Bond auction since 1979 from the
       official U.S. Treasury FiscalData API (``/v1/accounting/od/auctions_query``, the same
       feed that backs TreasuryDirect's auction-query page): auction date, offering amount,
       reopening flag, CUSIP, bid-to-cover, stop-out yield. Cached long
       (``tac_auctions.csv``).
    2. **Daily yields** — CBOE 10Y (``^TNX``) and 30Y (``^TYX``) yield indices from yfinance
       (served in percent). Cached wide (``tac_yields.csv``).
    3. **TLT + cash** — TLT total-return (auto-adjusted) closes and the 13-week T-bill index
       ``^IRX`` (percent, our cash rate) for the tradability leg. Cached wide
       (``tac_tlt.csv``).

* **Synthetic.** A deterministic, fixed-seed world: a daily yield random walk with a
  **planted auction concession** — a controllable pre-auction cheapening (knob
  ``concession_bps``, spread over the 5 trading days into each synthetic auction) and a
  post-auction richening (knob ``richen_bps``) — plus a matching duration-priced bond fund.
  ``concession_bps = richen_bps = 0`` is the null: the detector must NOT fire on it.

Pure numpy + pandas + stdlib for the offline path. The ``fetch_*`` functions (network) are
used once to build the cache and are never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import json
import os
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
AUCTIONS_CACHE = os.path.join(CACHE_DIR, "tac_auctions.csv")
YIELDS_CACHE = os.path.join(CACHE_DIR, "tac_yields.csv")
TLT_CACHE = os.path.join(CACHE_DIR, "tac_tlt.csv")

# Frozen as-of for the headline run — the sample never creeps as new sessions arrive.
AS_OF = "2026-06-30"

FISCALDATA_BASE = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
                   "/v1/accounting/od/auctions_query")
_UA = {"User-Agent": "OpenAlphaLab research desk guillain@poulpe.us"}


# --------------------------------------------------------------------------- #
# Real tape — auction records (FiscalData / TreasuryDirect)
# --------------------------------------------------------------------------- #
def fetch_auctions(path: str = AUCTIONS_CACHE) -> pd.DataFrame:
    """Download every 10Y Note + 30Y Bond auction (1979 ->) and cache one row per auction.

    Network-only; used once to build the cache. Columns: ``auction_date``, ``tenor`` (10Y /
    30Y), ``offering_amt`` ($), ``reopening`` (bool), ``cusip``, ``bid_to_cover``,
    ``high_yield`` (stop-out, %). ``original_security_term`` keeps reopenings (a 9-Year
    10-Month reopening IS a 10Y auction — same supply event).
    """
    fields = ("fields=auction_date,security_type,original_security_term,offering_amt,"
              "reopening,cusip,bid_to_cover_ratio,high_yield")
    frames = []
    for term, styp, tenor in [("10-Year", "Note", "10Y"), ("30-Year", "Bond", "30Y")]:
        url = (f"{FISCALDATA_BASE}?{fields}"
               f"&filter=original_security_term:eq:{term},security_type:eq:{styp}"
               f"&page[size]=10000&sort=auction_date")
        req = urllib.request.Request(url, headers=_UA)
        d = json.load(urllib.request.urlopen(req, timeout=120))
        df = pd.DataFrame(d["data"])
        df["tenor"] = tenor
        frames.append(df)
    au = pd.concat(frames, ignore_index=True)
    au["auction_date"] = pd.to_datetime(au["auction_date"])
    for c in ("offering_amt", "bid_to_cover_ratio", "high_yield"):
        au[c] = pd.to_numeric(au[c].replace("null", np.nan), errors="coerce")
    au["reopening"] = au["reopening"].astype(str).str.strip().eq("Yes")
    au = (au.rename(columns={"bid_to_cover_ratio": "bid_to_cover"})
            [["auction_date", "tenor", "offering_amt", "reopening", "cusip",
              "bid_to_cover", "high_yield"]]
            .dropna(subset=["auction_date", "offering_amt"])
            .drop_duplicates(["tenor", "auction_date", "cusip"])
            .sort_values(["tenor", "auction_date"]).reset_index(drop=True))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    au.to_csv(path, index=False)
    return au


def fetch_tape(yields_path: str = YIELDS_CACHE, tlt_path: str = TLT_CACHE) -> None:
    """Download ^TNX/^TYX (percent) and TLT/^IRX; cache wide CSVs. Network-only, run once."""
    import yfinance as yf

    tnx = yf.download("^TNX", period="max", auto_adjust=True, progress=False)["Close"]
    tyx = yf.download("^TYX", period="max", auto_adjust=True, progress=False)["Close"]
    ylds = pd.concat([tnx, tyx], axis=1)
    ylds.columns = ["y10", "y30"]
    ylds = ylds.dropna(how="all").sort_index()   # Yahoo serves ^TNX/^TYX in percent already
    os.makedirs(os.path.dirname(yields_path), exist_ok=True)
    ylds.to_csv(yields_path)

    tlt = yf.download("TLT", period="max", auto_adjust=True, progress=False)["Close"]
    irx = yf.download("^IRX", period="max", auto_adjust=True, progress=False)["Close"]
    fr = pd.concat([tlt, irx], axis=1)
    fr.columns = ["tlt", "irx"]                            # irx = 13w bill discount rate, %
    fr = fr.dropna(subset=["tlt"]).sort_index()
    fr.to_csv(tlt_path)


def have_real(auctions_path: str = AUCTIONS_CACHE, yields_path: str = YIELDS_CACHE,
              tlt_path: str = TLT_CACHE) -> bool:
    return all(os.path.exists(p) for p in (auctions_path, yields_path, tlt_path))


def load_auctions(path: str = AUCTIONS_CACHE, as_of: str | None = AS_OF) -> pd.DataFrame:
    """Cached auction list (one row per auction), sliced to the frozen as-of."""
    au = pd.read_csv(path, parse_dates=["auction_date"])
    if as_of is not None:
        au = au[au["auction_date"] <= pd.Timestamp(as_of)]
    return au.sort_values(["tenor", "auction_date"]).reset_index(drop=True)


def load_yields(path: str = YIELDS_CACHE, as_of: str | None = AS_OF) -> pd.DataFrame:
    """Cached daily 10Y/30Y yields in percent (columns y10, y30), sliced to as-of."""
    y = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if as_of is not None:
        y = y[y.index <= pd.Timestamp(as_of)]
    return y


def load_tlt(path: str = TLT_CACHE, as_of: str | None = AS_OF) -> pd.DataFrame:
    """Cached TLT total-return closes + ^IRX cash rate (columns tlt, irx), sliced to as-of."""
    fr = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if as_of is not None:
        fr = fr[fr.index <= pd.Timestamp(as_of)]
    return fr


def load_real() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience: (auctions, yields, tlt+cash), all as-of-sliced."""
    return load_auctions(), load_yields(), load_tlt()


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_world(n_years: int = 30, concession_bps: float = 0.0,
                    richen_bps: float = 0.0, seed: int = 603,
                    sig_daily_bps: float = 6.0, duration: float = 17.0,
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministic daily yield world with a PLANTED auction concession.

    Business-day yield path = pure random walk (sigma ``sig_daily_bps`` bps/day) **plus**, if
    planted, ``concession_bps`` of extra upward drift spread evenly over the 5 trading days
    ending on each synthetic auction day, and ``richen_bps`` of downward drift over the 5
    days after. Synthetic auctions fire every 21 trading days (a monthly cycle, like the
    real 10Y), with offering sizes that grow over time plus noise. A bond-fund price is
    priced off the yield with fixed ``duration``, and a flat 2% cash rate fills the ``irx``
    column. Span is ~``n_years`` (well under the 250-year ns-Timestamp trap).

    Returns ``(auctions, yields, tlt)`` in the exact shapes of :func:`load_real` — the
    detector code cannot tell it apart from the real tape. ``concession_bps = 0`` and
    ``richen_bps = 0`` is the null world where the detector must stay quiet.
    """
    rng = np.random.default_rng(seed)
    n_days = int(252 * n_years)
    idx = pd.bdate_range("1990-01-01", periods=n_days)

    dy = rng.normal(0.0, sig_daily_bps, size=n_days)      # bps/day
    auction_locs = np.arange(30, n_days - 10, 21)          # every 21 trading days
    for a in auction_locs:
        if concession_bps != 0.0:
            dy[a - 4:a + 1] += concession_bps / 5.0        # 5 changes ending ON auction day
        if richen_bps != 0.0:
            dy[a + 1:a + 6] -= richen_bps / 5.0
    y = 5.0 + np.cumsum(dy) / 100.0                        # percent
    y = np.clip(y, 0.2, None)
    ylds = pd.DataFrame({"y10": y, "y30": y + 0.3}, index=idx)

    # offering sizes: secular growth + noise (so the size split has something to chew on)
    size = 10e9 * (1.0 + np.arange(len(auction_locs)) / len(auction_locs) * 3.0)
    size = size * np.exp(rng.normal(0.0, 0.15, size=len(auction_locs)))
    au = pd.DataFrame({
        "auction_date": idx[auction_locs],
        "tenor": ["10Y"] * len(auction_locs),
        "offering_amt": size,
        "reopening": [(i % 3 != 0) for i in range(len(auction_locs))],
        "cusip": [f"SYN{i:05d}" for i in range(len(auction_locs))],
        "bid_to_cover": 2.5, "high_yield": np.nan,
    })

    ret = -duration * (dy / 1e4)                           # duration-priced fund return
    px = 100.0 * np.exp(np.cumsum(np.log1p(ret)))
    tlt = pd.DataFrame({"tlt": px, "irx": 2.0}, index=idx)
    return au, ylds, tlt
