"""Data layer for Study 630 — Late Filers (Form NT).

The claim under test: a company that cannot file its 10-K / 10-Q on time must file a
**Form 12b-25 "Notification of Late Filing"** (EDGAR form types ``NT 10-K`` / ``NT 10-Q``),
and that public admission is a **sell signal** — the stock underperforms the market over the
following weeks.

Three sources, all offline-friendly once cached:

* **NT filing events (EDGAR).** The quarterly *form-type master index*
  (``edgar/full-index/<year>/QTR<q>/master.idx``) lists every filing with CIK, form type and
  date. We keep exact-form ``NT 10-K`` and ``NT 10-Q`` rows (amendments ``NT 10-K/A`` excluded),
  2002Q1 -> 2026Q2, cached long (``nt_filings.csv``). Fetched once; ~10 MB of extracted rows
  from ~1.3 GB of index files, so never re-fetch casually.

* **CIK -> ticker map (EDGAR).** ``company_tickers.json`` — the SEC's registry of *currently*
  registered companies with exchange tickers. This is the study's named **survivorship gate**:
  an NT filer that later delisted (the worst offenders — that's often *why* they were late)
  has no ticker here and drops out of the panel. The bias UNDERSTATES the sell signal.

* **Prices (yfinance).** Daily auto-adjusted (total-return) closes for the event tickers +
  ``SPY`` (the market benchmark), cached wide (``prices.parquet`` with a CSV fallback).

* **Synthetic world.** A deterministic, fixed-seed daily panel (market factor + idiosyncratic
  noise) with planted NT-style events and a TUNABLE post-event drift (knob ``edge_bps_day``,
  bps/day over the measured 60-day window) plus the ``edge_bps_day = 0`` null. The positive
  control for the whole CAR / calendar-time machinery; never cited as market evidence.

Panel caps (deterministic, seeded, stated in the open): at most ``MAX_TICKERS`` distinct
tickers and ``MAX_EVENTS`` events — the desk's "no silent caps" rule.
"""

from __future__ import annotations

import io
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
NT_CACHE = os.path.join(CACHE_DIR, "nt_filings.csv")
TICKERS_CACHE = os.path.join(CACHE_DIR, "cik_tickers.csv")
PRICES_PARQUET = os.path.join(CACHE_DIR, "prices.parquet")
PRICES_CSV = os.path.join(CACHE_DIR, "prices.csv")

UA = {"User-Agent": "OpenAlphaLab research desk guillain@poulpe.us"}

# --- panel design constants (stated, not silent) --------------------------------------- #
IDX_START_YEAR = 2002          # index fetch start (2 extra years feed the repeat-NT lookback)
EVENT_START = "2004-01-01"     # first event admitted to the panel
EVENT_END = "2026-03-31"       # last event: leaves a complete +60-trading-day window by June 2026
AS_OF = "2026-06-30"           # price tape sliced here; June 2026 is the last complete month
MAX_TICKERS = 500              # deterministic seeded cap on distinct tickers (yfinance load)
MAX_EVENTS = 2000              # deterministic seeded cap on events
SEED = 630
REPEAT_LOOKBACK_D = 455        # a prior NT within ~15 months => "repeat offender" event


# --------------------------------------------------------------------------------------- #
# EDGAR: NT filings via the quarterly form-type master index
# --------------------------------------------------------------------------------------- #
def _get(url: str, timeout: int = 90, retries: int = 3) -> bytes:
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=timeout).read()
        except Exception as exc:                                   # pragma: no cover
            last = exc
            time.sleep(2.0)
    raise RuntimeError(f"fetch failed: {url} ({last})")


def fetch_nt_filings(start_year: int = IDX_START_YEAR, end_yq: tuple[int, int] = (2026, 2),
                     path: str = NT_CACHE, pause: float = 0.15) -> pd.DataFrame:
    """Download the quarterly master indexes once and cache the exact NT 10-K / NT 10-Q rows.

    Network-heavy (one ~5-25 MB index per quarter); used ONCE to build the cache and never
    imported by the offline notebook cells.
    """
    rows = []
    for year in range(start_year, end_yq[0] + 1):
        for q in range(1, 5):
            if (year, q) > end_yq:
                break
            url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/master.idx"
            text = _get(url).decode("latin-1", errors="replace")
            for line in text.splitlines():
                if "|NT 10-K|" in line or "|NT 10-Q|" in line:
                    cik, name, form, dt, _fn = line.split("|")
                    rows.append({"cik": int(cik), "company": name.strip(),
                                 "form": form.strip(), "date_filed": dt.strip()})
            time.sleep(pause)
    df = (pd.DataFrame(rows)
          .drop_duplicates(["cik", "form", "date_filed"])
          .sort_values(["date_filed", "cik"]).reset_index(drop=True))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def fetch_cik_tickers(path: str = TICKERS_CACHE) -> pd.DataFrame:
    """SEC ``company_tickers.json`` -> long (cik, ticker) frame, cached.

    Currently-registered companies only — the named survivorship gate.
    Keeps one (the first-listed, i.e. primary) ticker per CIK.
    """
    d = json.loads(_get("https://www.sec.gov/files/company_tickers.json"))
    df = pd.DataFrame(list(d.values())).rename(columns={"cik_str": "cik"})
    df = df.drop_duplicates("cik", keep="first")[["cik", "ticker", "title"]]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_nt_filings(path: str = NT_CACHE) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date_filed"])
    return df.sort_values(["date_filed", "cik"]).reset_index(drop=True)


def load_cik_tickers(path: str = TICKERS_CACHE) -> pd.DataFrame:
    return pd.read_csv(path)


# --------------------------------------------------------------------------------------- #
# Event panel: NT filings x ticker map, repeat flag, deterministic caps
# --------------------------------------------------------------------------------------- #
def build_events(nt: pd.DataFrame | None = None, tick: pd.DataFrame | None = None,
                 start: str = EVENT_START, end: str = EVENT_END,
                 max_tickers: int = MAX_TICKERS, max_events: int = MAX_EVENTS,
                 seed: int = SEED) -> pd.DataFrame:
    """The event panel: one row per (cik, filing date), ticker-mapped, repeat-flagged, capped.

    * Same-day NT 10-K + NT 10-Q by one CIK collapse to a single event (form = the 10-K one).
    * ``repeat`` — the same CIK filed ANY earlier NT within the prior ``REPEAT_LOOKBACK_D``
      days (~15 months): the "second consecutive NT" third axis.
    * Caps are seeded and deterministic: sample tickers first (so each kept name carries its
      full event history), then sample events.
    """
    nt = load_nt_filings() if nt is None else nt
    tick = load_cik_tickers() if tick is None else tick

    full = nt.copy()                             # full history feeds the repeat lookback
    full["date_filed"] = pd.to_datetime(full["date_filed"])

    # one event per (cik, date); prefer the NT 10-K label when both forms land the same day
    ev = (full.sort_values(["cik", "date_filed", "form"])   # "NT 10-K" sorts before "NT 10-Q"
          .drop_duplicates(["cik", "date_filed"], keep="first"))

    # repeat flag from the FULL (uncapped, unmapped) NT history — per-CIK previous NT date
    prev = ev.groupby("cik")["date_filed"].shift(1)
    ev = ev.assign(prev_nt=prev)
    ev["repeat"] = (ev["date_filed"] - ev["prev_nt"]).dt.days.le(REPEAT_LOOKBACK_D).fillna(False)

    ev = ev[(ev["date_filed"] >= start) & (ev["date_filed"] <= end)]

    # survivorship gate: currently-registered tickers only (named on the Signal axis)
    ev = ev.merge(tick[["cik", "ticker"]], on="cik", how="inner")
    ev["ticker"] = ev["ticker"].str.upper()     # SEC and yfinance share the dash convention

    rng = np.random.default_rng(seed)
    ticks = np.sort(ev["ticker"].unique())
    if len(ticks) > max_tickers:
        keep = rng.choice(ticks, size=max_tickers, replace=False)
        ev = ev[ev["ticker"].isin(set(keep))]
    ev = ev.sort_values(["date_filed", "cik"]).reset_index(drop=True)
    if len(ev) > max_events:
        idx = np.sort(rng.choice(len(ev), size=max_events, replace=False))
        ev = ev.iloc[idx].reset_index(drop=True)
    return ev[["cik", "ticker", "company", "form", "date_filed", "repeat"]]


# --------------------------------------------------------------------------------------- #
# Prices (yfinance) — event tickers + SPY, cached wide
# --------------------------------------------------------------------------------------- #
def fetch_prices(tickers: list[str], start: str = "2003-06-01", end: str = "2026-07-01",
                 batch: int = 80, retries: int = 3) -> pd.DataFrame:
    """Daily auto-adjusted closes for ``tickers`` + SPY; cached wide. Network, run once."""
    import yfinance as yf

    universe = sorted(set(map(str.upper, tickers)) | {"SPY"})
    frames = []
    for i in range(0, len(universe), batch):
        chunk = universe[i:i + batch]
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(chunk, start=start, end=end, auto_adjust=True,
                                  progress=False)["Close"]
                if raw is not None and len(raw) > 0:
                    break
            except Exception:                                     # pragma: no cover
                time.sleep(3.0)
        if raw is None or len(raw) == 0:
            continue
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(chunk[0])
        frames.append(raw)
    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.dropna(how="all", axis=0).dropna(how="all", axis=1)
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    save_prices(prices)
    return prices


def save_prices(prices: pd.DataFrame) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        prices.to_parquet(PRICES_PARQUET)
    except Exception:                                             # pragma: no cover
        prices.to_csv(PRICES_CSV)


def load_prices(as_of: str | None = AS_OF) -> pd.DataFrame:
    if os.path.exists(PRICES_PARQUET):
        px = pd.read_parquet(PRICES_PARQUET)
    else:
        px = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
    px = px.sort_index()
    if as_of is not None:
        px = px[px.index <= pd.Timestamp(as_of)]
    return px


def have_real() -> bool:
    return (os.path.exists(NT_CACHE) and os.path.exists(TICKERS_CACHE)
            and (os.path.exists(PRICES_PARQUET) or os.path.exists(PRICES_CSV)))


def load_real() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached (events, prices) — the offline entry point used everywhere."""
    events = build_events()
    prices = load_prices()
    events = events[events["ticker"].isin(prices.columns)].reset_index(drop=True)
    return events, prices


# --------------------------------------------------------------------------------------- #
# Synthetic world — planted post-NT drift + null
# --------------------------------------------------------------------------------------- #
def synthetic_world(n_names: int = 150, n_days: int = 3000, n_events: int = 400,
                    edge_bps_day: float = 0.0, seed: int = SEED,
                    window: int = 60, lag: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic daily panel + planted NT-style events with a TUNABLE post-event drift.

    Each name is beta-1 on a common market factor plus idiosyncratic noise. ``n_events``
    (name, date) pairs are planted; if ``edge_bps_day`` != 0 the name's return is shifted by
    ``edge_bps_day`` bps on each of the ``window`` trading days the event study measures
    (days ``d0+lag+1 .. d0+lag+window``). With ``edge_bps_day = 0`` the machinery must NOT
    manufacture significance. Business-day index, span well under the 250-year ns-Timestamp
    ceiling. Returns (events, prices) shaped exactly like :func:`load_real` — prices include
    a ``SPY`` column (the market factor).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2005-01-03", periods=n_days)          # ~12 years of weekdays
    names = [f"N{i:03d}" for i in range(n_names)]

    mkt = rng.normal(0.0003, 0.010, size=n_days)
    idio = rng.normal(0.0, 0.025, size=(n_days, n_names))
    rets = mkt[:, None] + idio

    # plant events: uniform over names/dates with a full window ahead
    last_ok = n_days - (lag + window + 2)
    ev_names = rng.integers(0, n_names, size=n_events)
    ev_pos = rng.integers(252, last_ok, size=n_events)
    if edge_bps_day != 0.0:
        e = edge_bps_day / 1e4
        for j, p in zip(ev_names, ev_pos):
            rets[p + lag + 1: p + lag + 1 + window, j] += e

    # cumprod(1 + r): pct_change recovers the draws exactly (exp(cumsum) would inject a
    # sigma^2/2 Jensen drift into the market-adjusted returns and corrupt the null)
    px = pd.DataFrame(100.0 * np.cumprod(1.0 + rets, axis=0), index=dates, columns=names)
    px["SPY"] = 100.0 * np.cumprod(1.0 + mkt)

    events = pd.DataFrame({
        "cik": ev_names,
        "ticker": [names[j] for j in ev_names],
        "company": [names[j] for j in ev_names],
        "form": "NT 10-K",
        "date_filed": dates[ev_pos],
        "repeat": False,
    }).sort_values("date_filed").reset_index(drop=True)
    return events, px
