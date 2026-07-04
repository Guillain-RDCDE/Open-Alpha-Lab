"""Data layer for Study 608 — Friday News Dump.

Two sources, both offline-friendly once cached:

* **Real tape.** A panel of *negative* 8-K filings harvested once from the EDGAR
  full-text search API (``efts.sec.gov``), 2004-08-23 (the day the expanded 8-K item
  regime — Items 2.06, 4.02, 5.02 — took effect) through 2026-06-30:

  - ``nonreliance``  — Item 4.02 (non-reliance on previously issued financial
    statements, i.e. "our accounts were wrong"),
  - ``impairment``   — Item 2.06 (material impairments),
  - ``ceo_exit``     — Item 5.02 CEO resignations (two resignation phrases),

  plus a **control class** ``earnings`` (Item 2.02, results of operations — the
  ordinary scheduled-news baseline for the *timing* comparison). For every filing we
  fetch the EDGAR **acceptance timestamp** (Eastern time, to the second — the moment
  the news became public), map CIK -> ticker via the SEC's ``company_tickers.json``,
  and pull daily adjusted closes (yfinance) for the mapped names + SPY.

  Sampling rule (documented, fixed BEFORE looking at returns): per class and per
  calendar quarter we keep the first ``CAP_BAD`` (=12) eligible hits the FTS API
  returns (``CAP_CTRL`` =8 for the control) — eligible = pure ``8-K`` (amendments
  excluded) carrying the class's item code. The cap keeps the panel tractable
  (~3,000 filings) and is orthogonal to the weekday under test. NO SILENT CAPS:
  heavy quarters (the 2005-2007 restatement wave runs 150-450 non-reliance filings a
  quarter) are therefore *sampled*, not exhausted.

* **Synthetic.** A deterministic, fixed-seed event panel with a TUNABLE planted
  Friday effect on both margins: ``drift_fri`` / ``drift_other`` (the post-event
  abnormal drift, the DellaVigna-Pollet margin) and ``p_fri`` (the propensity to
  file on Friday, the "hiding" margin). All knobs at the null (drift 0, p_fri=0.2)
  must NOT fire the detectors.

Pure stdlib + numpy + pandas on the offline path; the ``fetch_*`` functions
(network) build the cache once and are never imported by the notebooks' offline
cells. SEC requests carry the declared User-Agent and stay under the 10-req/s limit.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

EVENTS_CSV = os.path.join(CACHE_DIR, "events_raw.csv")
ACCEPT_CSV = os.path.join(CACHE_DIR, "acceptance.csv")
TICKERS_CSV = os.path.join(CACHE_DIR, "cik_tickers.csv")
PRICES_PQ = os.path.join(CACHE_DIR, "prices.parquet")

UA = {"User-Agent": "OpenAlphaLab research desk guillain@poulpe.us"}

START, END = "2004-08-23", "2026-06-30"   # expanded 8-K regime -> as-of
CAP_BAD, CAP_CTRL = 12, 8                 # per class x quarter (see module docstring)

# class -> (required 8-K item code, [FTS phrase queries])
BAD_CLASSES: dict[str, tuple[str, list[str]]] = {
    "nonreliance": ("4.02", ['"non-reliance on previously issued financial statements"']),
    "impairment": ("2.06", ['"Material Impairments"']),
    "ceo_exit": ("5.02", ['"resigned as Chief Executive Officer"',
                          '"resignation as Chief Executive Officer"']),
}
CTRL_CLASS: dict[str, tuple[str, list[str]]] = {
    "earnings": ("2.02", ['"Results of Operations and Financial Condition"']),
}


# --------------------------------------------------------------------------- #
# Small HTTP helper (retry on the FTS API's transient 500s)
# --------------------------------------------------------------------------- #
def _get(url: str, retries: int = 5, pause: float = 0.15) -> bytes | None:
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read()
            time.sleep(pause)
            return body
        except Exception:
            time.sleep(1.0 + 2.0 * k)
    return None


# --------------------------------------------------------------------------- #
# 1) Event harvest — EDGAR full-text search
# --------------------------------------------------------------------------- #
def _quarters(start: str = START, end: str = END) -> list[tuple[str, str]]:
    out = []
    for p in pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="Q"):
        a = max(p.start_time, pd.Timestamp(start)).strftime("%Y-%m-%d")
        b = min(p.end_time.normalize(), pd.Timestamp(end)).strftime("%Y-%m-%d")
        out.append((a, b))
    return out


def _fts_page(phrase: str, a: str, b: str, frm: int) -> list[dict]:
    q = urllib.parse.quote(phrase)
    url = (f"https://efts.sec.gov/LATEST/search-index?q={q}&forms=8-K"
           f"&startdt={a}&enddt={b}" + (f"&from={frm}" if frm else ""))
    body = _get(url)
    if body is None:
        return []
    try:
        return json.loads(body)["hits"]["hits"]
    except Exception:
        return []


def fetch_events(verbose: bool = True) -> pd.DataFrame:
    """Harvest the capped, stratified filing panel once and cache it as CSV."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    rows, seen = [], set()
    all_classes = {**BAD_CLASSES, **CTRL_CLASS}
    for a, b in _quarters():
        for cls, (item, phrases) in all_classes.items():
            cap = CAP_CTRL if cls in CTRL_CLASS else CAP_BAD
            got = 0
            for phrase in phrases:
                frm = 0
                while got < cap and frm <= 200:
                    hits = _fts_page(phrase, a, b, frm)
                    if not hits:
                        break
                    for h in hits:
                        s = h["_source"]
                        adsh = s["adsh"]
                        if (s.get("form") != "8-K" or item not in (s.get("items") or [])
                                or adsh in seen):
                            continue
                        seen.add(adsh)
                        rows.append(dict(cls=cls, adsh=adsh, cik=int(s["ciks"][0]),
                                         file_date=s["file_date"],
                                         name=(s.get("display_names") or [""])[0]))
                        got += 1
                        if got >= cap:
                            break
                    if len(hits) < 100:      # last page for this slice
                        break
                    frm += 100
                if got >= cap:
                    break
        if verbose:
            print(f"  {a}..{b}: cumulative {len(rows)} filings", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(EVENTS_CSV, index=False)
    return df


# --------------------------------------------------------------------------- #
# 2) Acceptance timestamps — the second the news became public
# --------------------------------------------------------------------------- #
_ACCEPT_RE = re.compile(r"Accepted[^0-9]*(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})")


def fetch_acceptance(verbose: bool = True) -> pd.DataFrame:
    """Fetch each filing's EDGAR acceptance datetime (ET). Incremental / resumable."""
    ev = pd.read_csv(EVENTS_CSV)
    done: dict[str, tuple[str, str]] = {}
    if os.path.exists(ACCEPT_CSV):
        old = pd.read_csv(ACCEPT_CSV)
        done = {r.adsh: (r.accept_date, r.accept_time) for r in old.itertuples()}
    todo = ev[~ev["adsh"].isin(done)][["adsh", "cik"]].drop_duplicates("adsh")
    for i, r in enumerate(todo.itertuples()):
        nod = r.adsh.replace("-", "")
        url = (f"https://www.sec.gov/Archives/edgar/data/{r.cik}/{nod}/"
               f"{r.adsh}-index.htm")
        body = _get(url, retries=3)
        m = _ACCEPT_RE.search(body.decode("utf-8", "replace")) if body else None
        if m:
            done[r.adsh] = (m.group(1), m.group(2))
        if verbose and (i + 1) % 200 == 0:
            print(f"  acceptance {i + 1}/{len(todo)}", flush=True)
        if (i + 1) % 200 == 0:      # checkpoint (resume-safe)
            _save_accept(done)
    _save_accept(done)
    return pd.read_csv(ACCEPT_CSV)


def _save_accept(done: dict) -> None:
    pd.DataFrame([(k, v[0], v[1]) for k, v in done.items()],
                 columns=["adsh", "accept_date", "accept_time"]).to_csv(
        ACCEPT_CSV, index=False)


# --------------------------------------------------------------------------- #
# 3) CIK -> ticker (current SEC map — SURVIVORSHIP lives here, named in results)
# --------------------------------------------------------------------------- #
def fetch_ticker_map() -> pd.DataFrame:
    body = _get("https://www.sec.gov/files/company_tickers.json")
    j = json.loads(body)
    rows, seen = [], set()
    for v in j.values():            # ordered by SEC's own ranking; first per CIK
        cik = int(v["cik_str"])
        if cik in seen:
            continue
        seen.add(cik)
        rows.append(dict(cik=cik, ticker=str(v["ticker"]).upper()))
    df = pd.DataFrame(rows)
    df.to_csv(TICKERS_CSV, index=False)
    return df


# --------------------------------------------------------------------------- #
# 4) Prices — daily adjusted closes for mapped tickers + SPY (yfinance)
# --------------------------------------------------------------------------- #
def fetch_prices(chunk: int = 50, verbose: bool = True) -> pd.DataFrame:
    import yfinance as yf

    panel = load_panel(require_prices=False)
    tickers = sorted(set(panel["ticker"].dropna())) + ["SPY"]
    frames = []
    for i in range(0, len(tickers), chunk):
        grp = tickers[i:i + chunk]
        try:
            raw = yf.download(grp, start="2004-01-01", end="2026-07-01",
                              auto_adjust=True, progress=False, group_by="column",
                              threads=True)
        except Exception:
            continue
        if raw is None or len(raw) == 0:
            continue
        px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else \
            raw[["Close"]].rename(columns={"Close": grp[0]})
        frames.append(px)
        if verbose:
            print(f"  prices {min(i + chunk, len(tickers))}/{len(tickers)}", flush=True)
    wide = pd.concat(frames, axis=1)
    wide = wide.loc[:, ~wide.columns.duplicated()]
    wide = wide.dropna(axis=1, how="all")
    wide.index = pd.DatetimeIndex(wide.index).tz_localize(None).normalize()
    wide = wide.sort_index()
    wide.to_parquet(PRICES_PQ)
    return wide


# --------------------------------------------------------------------------- #
# Offline loaders (cache-first; what verify.py and the notebooks use)
# --------------------------------------------------------------------------- #
def have_real() -> bool:
    return all(os.path.exists(p) for p in (EVENTS_CSV, ACCEPT_CSV, PRICES_PQ))


def load_events() -> pd.DataFrame:
    ev = pd.read_csv(EVENTS_CSV)
    acc = pd.read_csv(ACCEPT_CSV)
    df = ev.merge(acc, on="adsh", how="inner")
    df["accept_ts"] = pd.to_datetime(df["accept_date"] + " " + df["accept_time"])
    df["weekday"] = df["accept_ts"].dt.dayofweek          # 0=Mon .. 4=Fri
    df["after_close"] = df["accept_time"] >= "16:00:00"
    df["friday"] = df["weekday"] == 4
    df["friday_pm"] = df["friday"] & df["after_close"]
    return df.sort_values("accept_ts").reset_index(drop=True)


def load_panel(require_prices: bool = True) -> pd.DataFrame:
    """The negative-news panel (bad classes only), CIK-mapped to tickers."""
    df = load_events()
    df = df[df["cls"].isin(BAD_CLASSES)].copy()
    tk = pd.read_csv(TICKERS_CSV)
    df = df.merge(tk, on="cik", how="left")
    if require_prices and os.path.exists(PRICES_PQ):
        cols = set(load_prices().columns)
        df = df[df["ticker"].isin(cols)]
    return df.reset_index(drop=True)


def load_control() -> pd.DataFrame:
    df = load_events()
    return df[df["cls"] == "earnings"].reset_index(drop=True)


def load_prices(asof: str | None = END) -> pd.DataFrame:
    px = pd.read_parquet(PRICES_PQ).sort_index()
    if asof is not None:
        px = px[px.index <= pd.Timestamp(asof)]
    return px


# --------------------------------------------------------------------------- #
# Synthetic world (the machinery control)
# --------------------------------------------------------------------------- #
def synthetic_world(n_events: int = 1500, drift_fri: float = 0.0,
                    drift_other: float = 0.0, p_fri: float = 0.20,
                    react: float = -0.03, idio_vol: float = 0.035,
                    p_pm: float = 0.35, seed: int = 608,
                    ) -> tuple[pd.DataFrame, np.ndarray]:
    """Deterministic synthetic event panel + abnormal-return matrix.

    Each event gets a weekday (Friday with probability ``p_fri`` — the planted
    "hiding" propensity; other weekdays equally likely), an after-close flag
    (prob ``p_pm``), a day-0 abnormal reaction of mean ``react``, and a planted
    post-event drift over days +1..+10 totalling ``drift_fri`` (Friday events) or
    ``drift_other`` (Mon-Thu events), buried in ``idio_vol`` daily noise. Nulls:
    ``drift_fri=drift_other=0`` (no drift gap) and ``p_fri=0.2`` (no hiding).
    Dates span 2005-2025 business days (well inside the ns-Timestamp range).

    Returns ``(panel, armat)`` shaped exactly like the real pipeline's output:
    ``panel`` with friday/friday_pm/accept_ts columns, ``armat`` (n x 11) of
    daily abnormal returns day 0..+10.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2005-01-03", "2025-12-31")
    wd = cal.dayofweek.to_numpy()
    p = np.where(wd == 4, p_fri / max((wd == 4).mean(), 1e-9),
                 (1 - p_fri) / max((wd != 4).mean(), 1e-9))
    p = p / p.sum()
    days = rng.choice(len(cal), size=n_events, p=p)
    friday = wd[days] == 4
    pm = rng.random(n_events) < p_pm
    armat = rng.normal(0.0, idio_vol, size=(n_events, 11))
    armat[:, 0] += react + rng.normal(0.0, idio_vol, n_events)
    per_day = np.where(friday, drift_fri, drift_other) / 10.0
    armat[:, 1:] += per_day[:, None]
    panel = pd.DataFrame(dict(
        accept_ts=cal[days] + pd.Timedelta(hours=17),
        weekday=wd[days], friday=friday, friday_pm=friday & pm,
        after_close=pm, cls="synthetic"))
    return panel, armat
