"""Data layer for Study 861 — Debt-Maturity Rollover Risk.

The claim under test: **firms funded with a high share of SHORT-TERM debt carry rollover
risk** — a large slice of their borrowings matures within a year and has to be refinanced into
whatever rates and credit conditions prevail. The story says these firms **under-earn**
(a distress-style discount that the market mis-prices), and that the penalty **bites hardest
when rates are rising / credit is tightening** — hence an era cut at 2022, the Fed hiking cycle.

The signal is the **short-term-debt share** of total debt, point-in-time on the filing date:

    st_share = (DebtCurrent + LongTermDebtCurrent)
               ------------------------------------------------------
               (DebtCurrent + LongTermDebtCurrent + LongTermDebtNoncurrent)

where the three legs are the us-gaap XBRL concepts of the same name: ``DebtCurrent`` (short-term
borrowings + commercial paper), ``LongTermDebtCurrent`` (the *current maturities* of long-term
debt — the piece rolling over next), and ``LongTermDebtNoncurrent`` (everything due beyond a
year). A ``st_share`` near 1 is a firm living on the short end; near 0 is a firm that has termed
everything out. A balance-sheet-scaled variant ``st_debt_assets = (DebtCurrent +
LongTermDebtCurrent)/Assets`` is carried as a robustness cut (the *level* of the maturity wall
relative to size, not just its share of the debt stack).

Two sources, both offline-friendly once cached:

* **Real tape.**
  - Daily adjusted closes for a fixed basket of ~50 large, debt-carrying US filers (yfinance,
    no key).
  - Per name, the full 10-Q/10-K history of ``DebtCurrent``, ``LongTermDebtCurrent`` and
    ``LongTermDebtNoncurrent`` from EDGAR's XBRL ``companyconcept`` API (``data.sec.gov``), plus
    ``Assets`` for the scaled variant. We anchor on the **noncurrent long-term debt** series
    (every leveraged filer reports it — it is the denominator base) and, at each of its period
    ends, as-of match the two short-term legs (a missing leg is a genuine zero: the firm reported
    no such balance that quarter). Each event is one (ticker, filing date) row carrying the
    signal; forward returns are measured strictly *after* the filing date (no look-ahead).

* **Synthetic.** A deterministic, fixed-seed generator that produces a price + signal panel in
  which forward returns carry a **planted** component: with knob ``edge`` > 0, high-``st_share``
  names drift DOWN and low-``st_share`` names drift UP — exactly the claimed rollover
  underperformance. It is the positive control: with ``edge = 0`` the long-short must NOT
  manufacture significance; with a large planted ``edge`` the claim portfolio (long low-share /
  short high-share) must light up. No network.

Honest about coverage: this is a **thin, uneven** panel. Not every large filer tags the maturity
split cleanly — some report only ``DebtCurrent`` (no current-maturities line), some only the
long-term split (no separate short borrowings), and the tagging start dates differ by name. The
usable cross-section is only reasonably wide in the 2010s+, and it is a first-class caveat of the
study, not a footnote.

Pure numpy + pandas + stdlib for the offline path. ``fetch_panel`` (network) is used once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "rollover_prices.csv")
EVENTS_CACHE = os.path.join(CACHE_DIR, "rollover_events.csv")

UA = "open-alpha-lab research guillain@poulpe.us"

AS_OF = "2026-06-30"        # last complete calendar month at publication
ERA_SPLIT = "2022-01-01"    # pre/post the 2022 Fed hiking cycle (rate-rising sub-period)

# A transparent, fixed basket of large, debt-carrying US filers that report a maturity split on
# EDGAR — industrials, consumer staples/discretionary, telecom, energy, utilities, healthcare and
# levered tech. This is a *survivors* basket (all still trading). Survivorship is named on the
# Signal axis: a fixed surviving-names basket cannot include firms whose rollover wall actually
# sank them (the exact left tail the rollover story is about), which biases a long-low/short-high
# claim portfolio conservatively — the worst high-share names are missing. See docs/references.md.
BASKET = [
    "BA", "CAT", "DE", "MMM", "LMT", "UPS", "FDX", "PG", "KO", "PEP",
    "WMT", "MCD", "SBUX", "NKE", "HD", "LOW", "TGT", "MO", "PM", "T",
    "VZ", "CMCSA", "DIS", "XOM", "CVX", "COP", "KMI", "OKE", "NEE", "DUK",
    "SO", "D", "JNJ", "PFE", "MRK", "ABBV", "AMGN", "BMY", "GILD", "ORCL",
    "IBM", "CSCO", "QCOM", "AVGO", "F", "GM", "HON", "RTX", "UNP", "VZ",
]
BASKET = sorted(set(BASKET))

# SEC CIK (10-digit, zero-padded), resolved once from the SEC ticker map and frozen so the
# offline path never needs the network. XOM uses the pre-reorganisation CIK 0000034088 (the deep
# history lives there, not under the 2024 holding-company CIK).
CIK = {
    "ABBV": "0001551152", "AMGN": "0000318154", "AVGO": "0001730168", "BA": "0000012927",
    "BMY": "0000014272", "CAT": "0000018230", "CMCSA": "0001166691", "COP": "0001163165",
    "CSCO": "0000858877", "CVX": "0000093410", "D": "0000715957", "DE": "0000315189",
    "DIS": "0001744489", "DUK": "0001326160", "F": "0000037996", "FDX": "0001048911",
    "GILD": "0000882095", "GM": "0001467858", "HD": "0000354950", "HON": "0000773840",
    "IBM": "0000051143", "JNJ": "0000200406", "KMI": "0001506307", "KO": "0000021344",
    "LMT": "0000936468", "LOW": "0000060667", "MCD": "0000063908", "MMM": "0000066740",
    "MO": "0000764180", "MRK": "0000310158", "NEE": "0000753308", "NKE": "0000320187",
    "OKE": "0001039684", "ORCL": "0001341439", "PEP": "0000077476", "PFE": "0000078003",
    "PG": "0000080424", "PM": "0001413329", "QCOM": "0000804328", "RTX": "0000101829",
    "SBUX": "0000829224", "SO": "0000092122", "T": "0000732717", "TGT": "0000027419",
    "UNP": "0000100885", "UPS": "0001090727", "VZ": "0000732712", "WMT": "0000104169",
    "XOM": "0000034088",
}

# The three maturity legs. Each is looked up as an *instant* (point-in-time) balance-sheet fact.
DEBT_CURRENT_CONCEPTS = ("DebtCurrent",)                 # short-term borrowings + commercial paper
LTD_CURRENT_CONCEPTS = ("LongTermDebtCurrent",)          # current maturities of long-term debt
LTD_NONCURRENT_CONCEPTS = ("LongTermDebtNoncurrent",)    # long-term debt due beyond a year
ASSET_CONCEPTS = ("Assets",)


# --------------------------------------------------------------------------- #
# EDGAR helpers (network; cache builders only)
# --------------------------------------------------------------------------- #
def _edgar_concept(cik: str, concept: str, retries: int = 3) -> dict | None:
    url = (f"https://data.sec.gov/api/xbrl/companyconcept/"
           f"CIK{cik}/us-gaap/{concept}.json")
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:
            time.sleep(0.5 * (k + 1))
    return None


def _instant_series(cik: str, concepts: tuple[str, ...]) -> pd.DataFrame:
    """Best (longest-history) instant balance-sheet series for one CIK across ``concepts``.

    Instant (point-in-time) us-gaap concepts report one value AT a period end. We keep every
    10-Q/10-K observation, de-duplicate on the period end (keeping the EARLIEST filing that
    disclosed it — first public disclosure, no restatement look-ahead), and pick the concept that
    yields the most distinct quarter ends. Columns: end, filed, val.
    """
    best = pd.DataFrame(columns=["end", "filed", "val"])
    for concept in concepts:
        d = _edgar_concept(cik, concept)
        if d is None:
            continue
        rows = []
        for u in d.get("units", {}).get("USD", []):
            if u.get("form") not in ("10-Q", "10-K", "10-K/A", "10-Q/A"):
                continue
            if not (u.get("end") and u.get("filed") and u.get("val") is not None):
                continue
            rows.append({"end": u["end"], "filed": u["filed"], "val": float(u["val"])})
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["end"] = pd.to_datetime(df["end"])
        df["filed"] = pd.to_datetime(df["filed"])
        df = (df.sort_values("filed").drop_duplicates(subset=["end"], keep="first")
                .sort_values("end").reset_index(drop=True))
        if len(df) > len(best):
            best = df
    return best


# --------------------------------------------------------------------------- #
# Signal construction — short-term-debt share (point-in-time)
# --------------------------------------------------------------------------- #
def _asof_value(series: pd.DataFrame, end: pd.Timestamp, tol: int = 20) -> float | None:
    """Value in ``series`` whose end matches ``end`` within ``tol`` days (same quarter)."""
    if series.empty:
        return None
    gap = (series["end"] - end).abs().dt.days
    if gap.min() > tol:
        return None
    return float(series.loc[gap.idxmin(), "val"])


def build_signal(debt_cur: pd.DataFrame, ltd_cur: pd.DataFrame,
                 ltd_noncur: pd.DataFrame, assets: pd.DataFrame) -> pd.DataFrame:
    """One row per debt quarter carrying the short-term-debt-share signals.

    We anchor on the **noncurrent long-term debt** series (the denominator base, reported by
    every leveraged filer) and take its filing date as the point-in-time stamp. At each period
    end E (filed F, noncurrent balance NC) we as-of match the two short-term legs — DebtCurrent
    (DC) and LongTermDebtCurrent (LC); a missing leg is treated as a **genuine zero** (the firm
    disclosed no such balance that quarter). Then:

      * total_debt   = DC + LC + NC
      * ``st_share``       = (DC + LC) / total_debt          — the primary signal
      * ``st_debt_assets`` = (DC + LC) / Assets(E)           — the balance-sheet-scaled variant

    Rows with a non-positive or degenerate total (no debt, or a share outside [0, 1]) are dropped.
    """
    rows = []
    nc = ltd_noncur.sort_values("end").reset_index(drop=True)
    for _, r in nc.iterrows():
        end, filed, ncv = r["end"], r["filed"], r["val"]
        if ncv is None or not np.isfinite(ncv) or ncv < 0:
            continue
        dc = _asof_value(debt_cur, end) or 0.0
        lc = _asof_value(ltd_cur, end) or 0.0
        dc = max(dc, 0.0)
        lc = max(lc, 0.0)
        total = dc + lc + ncv
        if total <= 0:
            continue
        st_share = (dc + lc) / total
        if not (0.0 <= st_share <= 1.0):
            continue
        a = _asof_value(assets, end)
        st_debt_assets = ((dc + lc) / a) if (a and a > 0) else np.nan
        rows.append({"end": end, "filed": filed, "st_share": st_share,
                     "st_debt_assets": st_debt_assets, "total_debt": total,
                     "debt_current": dc, "ltd_current": lc, "ltd_noncurrent": ncv})
    cols = ["end", "filed", "st_share", "st_debt_assets", "total_debt",
            "debt_current", "ltd_current", "ltd_noncurrent"]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = "2006-01-01", end: str | None = None,
                prices_path: str = PRICES_CACHE, events_path: str = EVENTS_CACHE
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download basket prices + EDGAR maturity legs, build signals, cache.

    Network-only; used once to build the cache. Writes a wide adjusted-close CSV and a long event
    CSV (one row per ticker × debt quarter). Never imported by the offline notebook cells.
    """
    import yfinance as yf

    raw = yf.download(BASKET, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    keep = [c for c in raw.columns if raw[c].notna().mean() >= 0.10]
    prices = raw[keep].copy()
    os.makedirs(os.path.dirname(prices_path), exist_ok=True)
    prices.to_csv(prices_path)

    rows = []
    for tk in BASKET:
        cik = CIK.get(tk)
        if cik is None or tk not in prices.columns:
            continue
        ltd_noncur = _instant_series(cik, LTD_NONCURRENT_CONCEPTS)
        if len(ltd_noncur) < 8:
            continue
        debt_cur = _instant_series(cik, DEBT_CURRENT_CONCEPTS)
        ltd_cur = _instant_series(cik, LTD_CURRENT_CONCEPTS)
        assets = _instant_series(cik, ASSET_CONCEPTS)
        # a firm with no short-term leg at all carries no rollover signal — skip
        if debt_cur.empty and ltd_cur.empty:
            continue
        sig = build_signal(debt_cur, ltd_cur, ltd_noncur, assets)
        for _, s in sig.iterrows():
            rows.append({"ticker": tk, **s.to_dict()})
        time.sleep(0.15)
    events = pd.DataFrame(rows).dropna(subset=["filed", "st_share"])
    events = events.sort_values(["ticker", "filed"]).reset_index(drop=True)
    events.to_csv(events_path, index=False)
    return prices, events


def have_real(prices_path: str = PRICES_CACHE, events_path: str = EVENTS_CACHE) -> bool:
    return os.path.exists(prices_path) and os.path.exists(events_path)


def load_prices(path: str = PRICES_CACHE) -> pd.DataFrame:
    """Wide adjusted-close frame (index = date, columns = tickers), sliced to AS_OF."""
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return px.loc[px.index <= pd.Timestamp(AS_OF)]


def load_events(path: str = EVENTS_CACHE) -> pd.DataFrame:
    """Long event frame: ticker, end, filed, st_share, st_debt_assets, total_debt, legs
    (filings on/before AS_OF)."""
    ev = pd.read_csv(path, parse_dates=["end", "filed"])
    ev = ev[ev["filed"] <= pd.Timestamp(AS_OF)]
    return ev.sort_values(["ticker", "filed"]).reset_index(drop=True)


def load_real() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached prices + events in one call."""
    return load_prices(), load_events()


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_names: int = 30, n_quarters: int = 40, edge: float = 0.0,
                    seed: int = 861, sig_daily: float = 0.018
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic price panel + short-term-debt-share events with a PLANTED return knob.

    Each name has a daily random-walk price and a slowly-drifting short-term-debt share in
    [0, 1]. Quarterly filing dates (~63 trading days apart) each carry the share known that day.
    If ``edge`` != 0, the ~63 sessions of forward return following each filing get an *extra*
    daily drift of ``edge * (0.5 - share) / 63`` — high-share (rollover-risk) names drift DOWN,
    low-share names drift UP, exactly the claimed underperformance. With ``edge = 0`` the forward
    path is pure noise: the claim long-short must NOT reach significance however the noise falls.

    Returns (prices wide frame, event table) in the same shape as ``load_real``.
    """
    rng = np.random.default_rng(seed)
    H = 63
    n_days = n_quarters * H + 200
    idx = pd.bdate_range("2008-01-02", periods=n_days)
    names = [f"N{i:02d}" for i in range(n_names)]

    price_cols = {}
    ev_rows = []
    for name in names:
        ret = rng.normal(0.0003, sig_daily, size=n_days)
        base = float(rng.uniform(0.10, 0.70))          # this name's typical short-term share
        first = int(rng.integers(80, 140))
        q = first
        while q < n_days - H - 5:
            share = float(np.clip(base + rng.normal(0.0, 0.08), 0.02, 0.98))
            if edge != 0.0:
                ret[q + 1:q + 1 + H] += edge * (0.5 - share) / H
            ev_rows.append({"ticker": name, "_pos": q, "st_share": share})
            q += H + int(rng.integers(-4, 5))
        price_cols[name] = 100.0 * np.exp(np.cumsum(ret))

    prices = pd.DataFrame(price_cols, index=idx)
    ev = pd.DataFrame(ev_rows)
    ev["filed"] = [idx[p] for p in ev["_pos"]]
    ev["end"] = ev["filed"] - pd.Timedelta(days=40)
    ev["st_debt_assets"] = ev["st_share"] * 0.2
    ev["total_debt"] = np.nan
    ev["debt_current"] = np.nan
    ev["ltd_current"] = np.nan
    ev["ltd_noncurrent"] = np.nan
    ev = ev[["ticker", "end", "filed", "st_share", "st_debt_assets", "total_debt",
             "debt_current", "ltd_current", "ltd_noncurrent"]]
    return prices, ev.sort_values(["ticker", "filed"]).reset_index(drop=True)
