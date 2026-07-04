"""Data layer for Study 631 — Restatement Shock (Item 4.02).

Two sources, both offline-friendly once cached:

* **Real tape.** 8-K filings whose item list contains **Item 4.02** ("Non-Reliance on
  Previously Issued Financial Statements or a Related Audit Report") — the corporate
  *confession* event: "do not rely on our financials." Events come from **EDGAR full-text
  search** (``efts.sec.gov``), sampled quarter by quarter from 2004-Q4 (the item was created
  by the August 2004 8-K overhaul) to the present, deduplicated per filing and per
  (CIK, date). Tickers come from the FTS ``display_names`` field when present, else from the
  SEC's current ``company_tickers.json`` CIK→ticker map. Daily adjusted closes + volumes for
  the matched tickers and for SPY (the market benchmark) come from yfinance. Cached as
  ``events.csv`` + ``px_close.parquet`` + ``px_dvol.parquet`` + ``spy.csv``.

  **Deads-missing bias, named.** The CIK→ticker map is *current* and yfinance carries no
  delisted history, so firms that were acquired or — crucially — went **bankrupt** after their
  restatement drop out of the panel. Post-event bankruptcies are among the *worst* drift
  outcomes, so this bias **understates** any negative post-confession drift. It is named on
  the Signal axis of every document in this study.

* **Synthetic.** A deterministic, fixed-seed generator producing a market + event-stock panel
  with a controllable **planted day-0 shock** and a **planted post-event drift** (both
  tunable). It is the machinery control: with zero planted drift the drift detector must NOT
  manufacture significance; with a planted drift it must light up.

Pure numpy + pandas + stdlib for the offline path. The ``fetch_*`` functions (network) are
used once to build the cache and are never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
EVENTS_CACHE = os.path.join(CACHE_DIR, "events.csv")
CLOSE_CACHE = os.path.join(CACHE_DIR, "px_close.parquet")
DVOL_CACHE = os.path.join(CACHE_DIR, "px_dvol.parquet")
SPY_CACHE = os.path.join(CACHE_DIR, "spy.csv")

UA = {"User-Agent": "OpenAlphaLab research desk guillain@poulpe.us"}
FTS_URL = ("https://efts.sec.gov/LATEST/search-index?q=%22Item+4.02%22&forms=8-K"
           "&startdt={start}&enddt={end}&from={frm}")
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

# Item 4.02 exists since the 2004-08-23 8-K overhaul; cap keeps the study reproducible-sized.
FTS_START = "2004-09-01"
EVENT_CAP = 1500
PAGES_PER_QUARTER = 4          # 4 pages x 10 hits, relevance-sorted, per calendar quarter

_TICKER_RE = re.compile(r"\(([A-Z][A-Z0-9.\-]{0,9})\)\s+\(CIK")


def _get_json(url: str, retries: int = 3, pause: float = 0.8):
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            time.sleep(pause * (k + 1))
    return None


# --------------------------------------------------------------------------- #
# Real tape — events (EDGAR full-text search)
# --------------------------------------------------------------------------- #
def fetch_events(path: str = EVENTS_CACHE, cap: int = EVENT_CAP,
                 end: str = "2026-06-30", verbose: bool = True) -> pd.DataFrame:
    """Sample Item 4.02 8-K filings from EDGAR FTS, quarter by quarter. Network-only.

    Per quarter we pull up to ``PAGES_PER_QUARTER`` relevance-sorted pages for the query
    ``"Item 4.02"`` restricted to 8-K root forms, then keep only hits whose structured
    ``items`` list actually contains ``4.02`` (dropping filings that merely *mention* the
    phrase). Dedup per filing (adsh) and per (CIK, file_date). If the total exceeds ``cap``
    the sample is thinned deterministically, stratified by quarter (seeded RNG), so the time
    distribution is preserved.
    """
    quarters = pd.period_range(FTS_START, end, freq="Q")
    rows, seen_adsh, seen_ck = [], set(), set()
    for q in quarters:
        q0 = max(pd.Timestamp(FTS_START), q.start_time).date()
        q1 = min(pd.Timestamp(end), q.end_time).date()
        for page in range(PAGES_PER_QUARTER):
            js = _get_json(FTS_URL.format(start=q0, end=q1, frm=page * 10))
            time.sleep(0.15)
            if not js or "hits" not in js:
                break
            hits = js["hits"]["hits"]
            for h in hits:
                s = h.get("_source", {})
                if "4.02" not in (s.get("items") or []):
                    continue
                adsh = s.get("adsh", "")
                ciks = s.get("ciks") or [""]
                cik = ciks[0].lstrip("0") or "0"
                fdate = s.get("file_date", "")
                if not adsh or adsh in seen_adsh or (cik, fdate) in seen_ck:
                    continue
                seen_adsh.add(adsh)
                seen_ck.add((cik, fdate))
                name = (s.get("display_names") or [""])[0]
                m = _TICKER_RE.search(name)
                rows.append({"adsh": adsh, "cik": cik, "file_date": fdate,
                             "form": s.get("form", ""), "name": name,
                             "ticker_fts": (m.group(1) if m else "")})
            if len(hits) < 10:
                break
        if verbose:
            print(f"  {q}: cumulative events = {len(rows)}")
    ev = pd.DataFrame(rows)
    ev["file_date"] = pd.to_datetime(ev["file_date"])
    ev = ev.sort_values("file_date").reset_index(drop=True)

    # fill tickers from the SEC's current CIK->ticker map (dead firms won't map — named bias)
    tmap = _get_json(TICKER_MAP_URL) or {}
    cik2tk = {str(v["cik_str"]): v["ticker"] for v in tmap.values()}
    ev["ticker"] = ev.apply(
        lambda r: r["ticker_fts"] or cik2tk.get(str(r["cik"]), ""), axis=1)

    if len(ev) > cap:  # deterministic, quarter-stratified thinning
        rng = np.random.default_rng(631)
        ev["q"] = ev["file_date"].dt.to_period("Q").astype(str)
        keep_frac = cap / len(ev)
        parts = []
        for _q, g in ev.groupby("q", sort=True):
            k = max(1, int(round(keep_frac * len(g))))
            idx = rng.choice(len(g), size=min(k, len(g)), replace=False)
            parts.append(g.iloc[np.sort(idx)])
        ev = (pd.concat(parts).drop(columns=["q"])
              .sort_values("file_date").reset_index(drop=True))
    os.makedirs(CACHE_DIR, exist_ok=True)
    ev.to_csv(path, index=False)
    if verbose:
        print(f"saved {len(ev)} events -> {path} "
              f"({(ev['ticker'] != '').mean() * 100:.0f}% with a ticker)")
    return ev


def fetch_prices(events: pd.DataFrame | None = None,
                 close_path: str = CLOSE_CACHE, dvol_path: str = DVOL_CACHE,
                 spy_path: str = SPY_CACHE, start: str = "2004-01-01",
                 chunk: int = 80, verbose: bool = True) -> None:
    """Download daily adjusted closes + dollar volume for event tickers and SPY. Network-only.

    Delisted names simply return no data — they are dropped and counted (the deads-missing
    bias, named everywhere). Dollar volume = adjusted close x share volume (a ranking proxy
    for size / limits-to-arbitrage; only its cross-sectional ORDER is used).
    """
    import yfinance as yf

    if events is None:
        events = load_events()
    tks = sorted({t.replace(".", "-") for t in events["ticker"] if t})
    closes, dvols = [], []
    for i in range(0, len(tks), chunk):
        part = tks[i:i + chunk]
        for attempt in range(3):
            try:
                raw = yf.download(part, start=start, auto_adjust=True,
                                  progress=False, group_by="column", threads=True)
                break
            except Exception:
                time.sleep(3.0)
                raw = None
        if raw is None or len(raw) == 0:
            continue
        c = raw["Close"] if "Close" in raw else raw
        v = raw["Volume"] if "Volume" in raw else None
        keep = [t for t in c.columns if c[t].notna().sum() >= 120]
        closes.append(c[keep])
        if v is not None:
            dvols.append((c[keep] * v[keep].reindex(columns=keep)))
        if verbose:
            print(f"  chunk {i // chunk + 1}: {len(keep)}/{len(part)} tickers with data")
        time.sleep(1.0)
    close = pd.concat(closes, axis=1).sort_index()
    close = close.loc[:, ~close.columns.duplicated()]
    dvol = pd.concat(dvols, axis=1).sort_index()
    dvol = dvol.loc[:, ~dvol.columns.duplicated()]
    close.index.name = "date"
    dvol.index.name = "date"
    os.makedirs(CACHE_DIR, exist_ok=True)
    close.to_parquet(close_path)
    dvol.to_parquet(dvol_path)

    spy = yf.download("SPY", start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]
    spy.rename("SPY").to_csv(spy_path)
    if verbose:
        print(f"saved {close.shape[1]} price series + SPY -> _cache/")


def have_real() -> bool:
    return all(os.path.exists(p) for p in
               (EVENTS_CACHE, CLOSE_CACHE, DVOL_CACHE, SPY_CACHE))


def load_events(path: str = EVENTS_CACHE) -> pd.DataFrame:
    ev = pd.read_csv(path, parse_dates=["file_date"])
    ev["ticker"] = ev["ticker"].fillna("").astype(str)
    return ev


def load_real() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Cached (events, close, dollar-volume, spy) — the offline real tape."""
    ev = load_events()
    close = pd.read_parquet(CLOSE_CACHE)
    dvol = pd.read_parquet(DVOL_CACHE)
    spy = pd.read_csv(SPY_CACHE, index_col=0, parse_dates=True).iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    dvol.index = pd.to_datetime(dvol.index)
    return ev, close, dvol, spy


# --------------------------------------------------------------------------- #
# Synthetic control — planted day-0 shock + planted post-event drift
# --------------------------------------------------------------------------- #
def synthetic_market(n_events: int = 600, n_days: int = 3000,
                     day0_shock: float = -0.08, drift: float = 0.0,
                     chronic: float = 0.0, drift_days: int = 63, seed: int = 631,
                     sig_idio: float = 0.028) -> tuple[pd.DataFrame, pd.DataFrame,
                                                       pd.DataFrame, pd.Series]:
    """Deterministic synthetic world in the exact shape of ``load_real()``.

    One market factor (the benchmark), ``n_events`` unit-beta stocks with idiosyncratic
    noise. Each stock has one event day; on it the stock return gets ``day0_shock`` extra;
    over the following ``drift_days`` (starting two days later — mirroring the real drift
    window) it gets ``drift`` extra, spread evenly. Two knobs, two failure modes:

    * ``drift`` — a TRUE event-specific underreaction. The raw drift detector and the
      paired chronic-decay placebo must BOTH light up.
    * ``chronic`` — an annualised alpha drag applied to event stocks on EVERY day (the
      melting-ice-cube confound: the kind of stock that has events also bleeds all the
      time). The raw drift detector false-fires; the paired placebo must stay quiet.

    ``drift = chronic = 0`` is the null: nothing may fire. The date span is ~12 years of
    business days (far below the 250-year ns-Timestamp ceiling).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n_days)
    mkt_ret = rng.normal(4e-4, 0.010, size=n_days)
    spy = pd.Series(100.0 * np.exp(np.cumsum(mkt_ret)), index=idx, name="SPY")

    cols, ev_rows = {}, []
    event_pos = rng.integers(300, n_days - drift_days - 10, size=n_events)
    for j in range(n_events):
        t0 = int(event_pos[j])
        r = mkt_ret + rng.normal(0.0, sig_idio, size=n_days) + chronic / 252.0
        r[t0] += day0_shock
        if drift != 0.0:
            r[t0 + 2: t0 + 2 + drift_days] += drift / drift_days
        tk = f"S{j:04d}"
        cols[tk] = 20.0 * np.exp(np.cumsum(r))
        ev_rows.append({"adsh": f"syn-{j:04d}", "cik": str(j), "name": tk,
                        "form": "8-K", "ticker": tk,
                        "file_date": idx[t0]})
    close = pd.DataFrame(cols, index=idx)
    dvol = close * 1e5          # flat share volume — size split unused in the control
    ev = pd.DataFrame(ev_rows)
    return ev, close, dvol, spy
