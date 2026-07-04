"""Data layer for Study 627 — 13F Cloning (Berkshire Hathaway).

Two sources, both offline-friendly once cached:

* **Real tape, leg 1 — the filings.** Every original 13F-HR filed by Berkshire Hathaway
  (CIK 0001067983) on EDGAR since the XML information-table era began (period 2013-06-30,
  filed 2013-08-15). For each filing we aggregate the information table by CUSIP (Berkshire
  reports the same CUSIP under several managers), rank by reported market value and keep the
  **top 15** (the clone uses the top 10; 11–15 are kept for robustness). Amendments (13F-HR/A)
  are excluded — the clone only ever sees what was public on the original filing date. Cached
  long as ``brk_13f_top.csv``. Note the 45-day disclosure lag is **inside the filing date**:
  a filing dated 2025-08-14 describes the 2025-06-30 portfolio.

* **Real tape, leg 2 — prices.** yfinance daily **auto-adjusted (total-return)** closes for
  every ticker that ever appears in a top-10 slot and is still quotable, plus **SPY** (the
  benchmark), **BRK-B** (the third-axis race) and **^IRX** (13-week T-bill yield, the
  risk-free leg for excess-vs-excess Sharpe). Cached wide as ``clone_prices.csv``.

* **Synthetic world.** A deterministic monthly market with a manager whose picks carry a
  **tunable planted alpha** (knob ``alpha_annual``) disclosed through quarterly "filings"
  with a 45-day lag — exactly the clone's information set. With ``alpha_annual = 0`` the
  clone-vs-market inference must NOT manufacture significance; with a planted 5%/yr it must
  light up. Machinery proof only, never market evidence.

CUSIP -> ticker is a **hardcoded** 30-entry map (every CUSIP that ever reached Berkshire's
top 10 over 2013Q2–2026Q1; verified against the parsed filings). Two names are unpriceable on
yfinance because they were **acquired and delisted** — DIRECTV (bought by AT&T, 2015) and
Activision Blizzard (bought by Microsoft, 2023). Their slots (13 of 520 top-10 slots, 2.5%;
1.5–3.3% of top-10 value weight in the affected quarters) are dropped and the remaining
weights renormalised — a *mild survivorship gap named on the Signal axis* (both names were
pinned merger targets at the time, so the omission is small either way).

Pure numpy + pandas + stdlib for the offline path. The ``fetch_*`` functions (network) are
used once to build the cache and are never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
HOLDINGS_CACHE = os.path.join(CACHE_DIR, "brk_13f_top.csv")
PRICES_CACHE = os.path.join(CACHE_DIR, "clone_prices.csv")

BRK_CIK = "0001067983"
UA = {"User-Agent": "OpenAlphaLab research desk guillain@poulpe.us"}

# Every headline number is computed on the tape sliced to this date (last complete month).
AS_OF = "2026-06-30"

# --------------------------------------------------------------------------- #
# CUSIP -> ticker (hardcoded; every CUSIP that ever reached Berkshire's top 10
# in an original 13F-HR, periods 2013-06-30 .. 2026-03-31, parsed from EDGAR).
# ``None`` = acquired & delisted, no yfinance tape (slot dropped + renormalised).
# --------------------------------------------------------------------------- #
CUSIP_TO_TICKER = {
    "037833100": "AAPL",   # APPLE INC
    "02079K305": "GOOGL",  # ALPHABET INC (class A)
    "025816109": "AXP",    # AMERICAN EXPRESS CO
    "060505104": "BAC",    # BANK OF AMERICA CORP
    "064058100": "BK",     # BANK OF NEW YORK MELLON
    "16119P108": "CHTR",   # CHARTER COMMUNICATIONS
    "166764100": "CVX",    # CHEVRON
    "H1467J104": "CB",     # CHUBB LTD
    "172967424": "C",      # CITIGROUP
    "191216100": "KO",     # COCA-COLA
    "23918K108": "DVA",    # DAVITA
    "247361702": "DAL",    # DELTA AIR LINES
    "30231G102": "XOM",    # EXXON MOBIL
    "37045V100": "GM",     # GENERAL MOTORS
    "38141G104": "GS",     # GOLDMAN SACHS
    "40434L105": "HPQ",    # HP INC
    "459200101": "IBM",    # IBM
    "46625H100": "JPM",    # JPMORGAN CHASE
    "500754106": "KHC",    # KRAFT HEINZ
    "615369105": "MCO",    # MOODY'S
    "674599105": "OXY",    # OCCIDENTAL PETROLEUM
    "718546104": "PSX",    # PHILLIPS 66
    "742718109": "PG",     # PROCTER & GAMBLE
    "874039100": "TSM",    # TAIWAN SEMICONDUCTOR (ADR)
    "902973304": "USB",    # US BANCORP
    "92343V104": "VZ",     # VERIZON
    "931142103": "WMT",    # WALMART
    "949746101": "WFC",    # WELLS FARGO
    "00507V109": None,     # ACTIVISION BLIZZARD — acquired by Microsoft 2023-10, delisted
    "25490A309": None,     # DIRECTV — acquired by AT&T 2015-07, delisted
}

TICKERS = sorted(t for t in CUSIP_TO_TICKER.values() if t)
AUX = ["SPY", "BRK-B", "^IRX"]


# --------------------------------------------------------------------------- #
# Fetchers (network; one-shot cache builders)
# --------------------------------------------------------------------------- #
def _get(url: str, retries: int = 4) -> str:
    import urllib.request
    req = urllib.request.Request(url, headers=UA)
    for k in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            time.sleep(2.0 + 2.0 * k)
    raise RuntimeError(f"failed to fetch {url}")


def fetch_13f(path: str = HOLDINGS_CACHE, top_keep: int = 15) -> pd.DataFrame:
    """Fetch + parse every original Berkshire 13F-HR since 2013 and cache the top holdings.

    Network-only; run once. Aggregates the XML information table by CUSIP (summing across
    Berkshire's reporting managers), excludes put/call lines (Berkshire reports none anyway)
    and 13F-HR/A amendments, keeps the earliest original filing per period, and writes one
    row per (period, rank<=top_keep): period, filing_date, rank, cusip, issuer, value,
    total_value. Values are as reported ($k pre-2023, $ after — only within-filing ranks and
    weights are ever used, so the unit change is harmless).
    """
    import json
    import re
    import xml.etree.ElementTree as ET

    d = json.loads(_get(f"https://data.sec.gov/submissions/CIK{BRK_CIK}.json"))
    filings = []

    def harvest(rec):
        for i in range(len(rec["form"])):
            if rec["form"][i] == "13F-HR":
                filings.append((rec["reportDate"][i], rec["filingDate"][i],
                                rec["accessionNumber"][i]))

    harvest(d["filings"]["recent"])
    for f in d["filings"].get("files", []):
        # older pages carry the same keys ("form", "filingDate", ...) at top level
        harvest(json.loads(_get("https://data.sec.gov/submissions/" + f["name"])))

    filings = [f for f in filings if f[1] >= "2013-01-01"]
    byper = {}
    for per, fdate, acc in sorted(filings, key=lambda x: (x[0], x[1])):
        byper.setdefault(per, (per, fdate, acc))

    rows = []
    for per, fdate, acc in sorted(byper.values()):
        accn = acc.replace("-", "")
        base = f"https://www.sec.gov/Archives/edgar/data/{int(BRK_CIK)}/{accn}"
        idx = json.loads(_get(base + "/index.json"))
        info = [it["name"] for it in idx["directory"]["item"]
                if it["name"].lower().endswith(".xml")
                and "primary_doc" not in it["name"].lower()]
        if not info:            # pre-XML era (2012Q4, 2013Q1) — no information table
            continue
        xml = re.sub(r'xmlns="[^"]+"', "", _get(base + "/" + info[0]), count=1)
        agg: dict[str, list] = {}
        for it in ET.fromstring(xml).iter():
            if not it.tag.endswith("infoTable"):
                continue
            g = {c.tag.split("}")[-1]: c for c in it}
            put = g.get("putCall")
            if put is not None and put.text and put.text.strip():
                continue
            cusip = g["cusip"].text.strip().upper()
            a = agg.setdefault(cusip, [g["nameOfIssuer"].text.strip(), 0.0])
            a[1] += float(g["value"].text.strip())
        top = sorted(agg.items(), key=lambda kv: -kv[1][1])[:top_keep]
        total = sum(v for _, (_, v) in agg.items())
        for rank, (cusip, (name, val)) in enumerate(top, 1):
            rows.append({"period": per, "filing_date": fdate, "rank": rank,
                         "cusip": cusip, "issuer": name, "value": val,
                         "total_value": total})
        time.sleep(0.3)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def fetch_prices(path: str = PRICES_CACHE, start: str = "2013-01-01",
                 end: str = "2026-07-01", retries: int = 3) -> pd.DataFrame:
    """Download the clone universe + SPY + BRK-B + ^IRX (yfinance, total-return) and cache."""
    import yfinance as yf

    cols = TICKERS + AUX
    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(cols, start=start, end=end, auto_adjust=True,
                              progress=False)["Close"]
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            time.sleep(2.0)
    raw = raw.dropna(how="all")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


# --------------------------------------------------------------------------- #
# Cache-first loaders
# --------------------------------------------------------------------------- #
def have_real(holdings_path: str = HOLDINGS_CACHE, prices_path: str = PRICES_CACHE) -> bool:
    return os.path.exists(holdings_path) and os.path.exists(prices_path)


def load_holdings(path: str = HOLDINGS_CACHE) -> pd.DataFrame:
    """Long top-15 holdings frame: period, filing_date, rank, cusip, issuer, value, total_value."""
    if not os.path.exists(path):
        fetch_13f(path)
    df = pd.read_csv(path, dtype={"cusip": str})
    df["period"] = pd.to_datetime(df["period"])
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df.sort_values(["filing_date", "rank"]).reset_index(drop=True)


def load_prices(path: str = PRICES_CACHE, asof: str | None = AS_OF) -> pd.DataFrame:
    """Wide total-return close frame (clone names + SPY + BRK-B + ^IRX), sliced to as-of."""
    if not os.path.exists(path):
        fetch_prices(path)
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if asof is not None:
        px = px[px.index <= pd.Timestamp(asof)]
    return px


def load_real(asof: str | None = AS_OF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: (holdings, prices) — cache-first, as-of'd."""
    return load_holdings(), load_prices(asof=asof)


def daily_rf(prices: pd.DataFrame) -> pd.Series:
    """Daily risk-free rate from ^IRX (13-week T-bill discount yield, % annualised)."""
    irx = prices["^IRX"].ffill().fillna(0.0)
    return irx / 100.0 / 252.0


# --------------------------------------------------------------------------- #
# Synthetic world (deterministic; planted, tunable manager alpha + a null)
# --------------------------------------------------------------------------- #
def synthetic_world(n_names: int = 30, n_months: int = 240, alpha_annual: float = 0.0,
                    seed: int = 627) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic monthly world with a manager whose picks carry a PLANTED alpha.

    30 stocks load on one market factor (betas ~ U[0.8, 1.2], idiosyncratic noise). A
    "manager" holds the 10 names with the highest slowly-mean-reverting latent conviction
    score; every quarter-end the portfolio is disclosed in a "13F" **filed 45 days later**
    — the clone only ever trades on the filing date, exactly like the real construction.
    While a name is held by the manager it earns an extra ``alpha_annual / 12`` per month
    (persistent skill). With ``alpha_annual = 0`` the manager has NO skill and the clone's
    active return vs the market must not reach significance; with a planted 5%/yr it must.

    Returns (prices, holdings) in exactly the real tape's shape: a wide "daily" price frame
    whose rows are month-ends (with SPY = the market index, BRK-B = the manager's own
    portfolio, ^IRX = 0), and a long top-15 holdings frame with period / filing_date /
    rank / cusip / value. The CUSIPs are the synthetic tickers themselves, so
    ``strategy.build_clone`` runs unchanged with ``cusip_map=None``.

    The date span is 20 years (well under the 250-year pandas ns-Timestamp ceiling).
    """
    rng = np.random.default_rng(seed)
    pidx = pd.period_range("2000-01", periods=n_months, freq="M")
    stamps = pidx.to_timestamp(how="end").normalize()
    names = [f"S{i:02d}" for i in range(n_names)]

    betas = rng.uniform(0.8, 1.2, size=n_names)
    mkt = rng.normal(0.007, 0.042, size=n_months)
    idio = rng.normal(0.0, 0.055, size=(n_months, n_names))

    # slowly mean-reverting conviction scores -> persistent, rotating top-10
    score = rng.normal(0.0, 1.0, size=n_names)
    held = np.zeros((n_months, n_names), dtype=bool)
    scores = np.zeros((n_months, n_names))
    for t in range(n_months):
        score = 0.97 * score + rng.normal(0.0, np.sqrt(1 - 0.97**2), size=n_names)
        scores[t] = score
        held[t, np.argsort(-score)[:10]] = True

    a_m = alpha_annual / 12.0
    rets = mkt[:, None] * betas[None, :] + idio + a_m * held

    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(np.log1p(rets), axis=0)),
                          index=stamps, columns=names)
    prices["SPY"] = 100.0 * np.exp(np.cumsum(np.log1p(mkt)))
    mgr = (rets * held / 10.0).sum(axis=1)           # manager's own EW book
    prices["BRK-B"] = 100.0 * np.exp(np.cumsum(np.log1p(mgr)))
    prices["^IRX"] = 0.0

    rows = []
    for t in range(2, n_months, 3):                   # quarter-ends: month 3, 6, 9, ...
        per = stamps[t]
        fdate = per + pd.Timedelta(days=45)           # the 45-day disclosure lag
        order = np.argsort(-scores[t])[:15]
        for rank, j in enumerate(order, 1):
            rows.append({"period": per, "filing_date": fdate, "rank": rank,
                         "cusip": names[j], "issuer": names[j],
                         "value": float(np.exp(scores[t, j]) * 1e6),
                         "total_value": 0.0})
    holdings = pd.DataFrame(rows)
    return prices, holdings
