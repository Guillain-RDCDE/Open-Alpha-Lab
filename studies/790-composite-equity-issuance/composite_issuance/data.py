"""Data layer for Study 790 — Composite Equity Issuance (Daniel-Titman 2006).

Two sources, both offline-friendly once the cache exists:

* **Real tape.** A fixed ~40-name large-cap **survivor** basket. For each name we pull:

  - **Point-in-time shares outstanding** from SEC EDGAR XBRL
    (``us-gaap:CommonStockSharesOutstanding``, with ``dei:EntityCommonStockSharesOutstanding``
    as the fallback concept), via the generic ``companyconcept`` endpoint. Each XBRL fact
    carries a **filed** date, so we build a strictly point-in-time year-end share series: the
    share count as-of each December 31 is the latest fact *filed on or before* that date — no
    look-ahead into a filing that had not yet hit EDGAR.
  - Daily **raw** close and **adjusted** (split- & dividend-adjusted, total-return) close from
    yfinance (no key). The raw close × the point-in-time share count is the **market cap**; the
    adjusted-close ratio is the **total-return** leg.

  We resample everything to a December-year-end grid and cache three wide CSVs under this
  study's own ``_cache/`` (``price_raw.csv``, ``price_adj.csv``, ``shares.csv``).

* **Synthetic.** A deterministic, fixed-seed cross-sectional panel with a **planted composite
  issuance edge** (low-issuance names earn ``edge`` more per year than high-issuance names). It
  is the positive control: ``edge=0`` must NOT manufacture a long-short t≥2 from ~40 names over
  a handful of years; a large planted ``edge`` must light the sort up.

The composite issuance measure itself (Daniel-Titman 2006):

    iota_{i,t} = log( ME_{i,t} / ME_{i,t-5} ) - r_{i}(t-5, t)

where ``ME = shares x raw_price`` is market cap and ``r`` is the 5-year cumulative log
**total** return (adjusted-close ratio). Splits cancel between the two legs; what remains is
net equity issuance in log terms (net of the dividend contribution). This is a **5-year log
composite** measure, distinct from the 1-year pure share-count change of Study 519.

Network is touched ONLY by ``fetch_real`` (run once to build the cache). The offline core
(``load_real``, ``synthetic_panel``) never imports yfinance or urllib.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
PRICE_RAW_CSV = os.path.join(CACHE_DIR, "price_raw.csv")
PRICE_ADJ_CSV = os.path.join(CACHE_DIR, "price_adj.csv")
SHARES_CSV = os.path.join(CACHE_DIR, "shares.csv")

START = "2008-01-01"   # XBRL CommonStockSharesOutstanding reaches ~2009-2010 for large caps
AS_OF = "2025-12-31"   # last complete calendar year at publication (2026-07)
LOOKBACK = 5           # the 5-year composite window (Daniel-Titman)
ERA_SPLIT = 2020       # formation-year era split for the sub-period contrast

SEC_UA = {"User-Agent": "OpenAlphaLab research guillain@poulpe.us"}

# A fixed, transparent large-cap SURVIVOR basket (~40 names) spanning sectors. All are firms
# still trading in 2026 with a long XBRL shares history on EDGAR and deep price history on
# yfinance — a deliberate survivorship choice, named on the Signal axis. The point is the *axis*
# (composite issuance vs return), not the exact roster.
BASKET = [
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "ORCL", "CSCO", "INTC", "IBM",
    "QCOM", "TXN", "ADBE", "CRM", "AMD", "XOM", "CVX", "COP", "JPM", "BAC",
    "WFC", "GS", "JNJ", "PFE", "MRK", "ABBV", "UNH", "WMT", "HD", "MCD",
    "KO", "PEP", "PG", "COST", "NKE", "DIS", "VZ", "T", "CAT", "BA",
]


# --------------------------------------------------------------------------- #
# Real tape (network) — builds the cache; never imported by the offline core
# --------------------------------------------------------------------------- #
def _get(url, timeout=30, headers=None, tries=4):
    last = None
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers or SEC_UA)
            return urllib.request.urlopen(req, timeout=timeout).read()  # noqa: S310
        except Exception as exc:  # pragma: no cover - network flakiness
            last = exc
            time.sleep(1.5)
    raise last


def _ticker_cik_map() -> dict:
    """Yahoo ticker -> zero-padded 10-digit CIK, from SEC's company_tickers.json."""
    cmap = json.loads(_get("https://www.sec.gov/files/company_tickers.json"))
    return {v["ticker"]: str(v["cik_str"]).zfill(10) for v in cmap.values()}


def _edgar_shares_facts(cik: str) -> list[tuple]:
    """All (period_end, filed, value) share-count facts for one CIK.

    Primary concept ``us-gaap:CommonStockSharesOutstanding``; if it yields nothing, fall back to
    ``dei:EntityCommonStockSharesOutstanding`` (the cover-page share count). Every XBRL fact
    carries its ``filed`` date — the raw material for a strictly point-in-time series.
    """
    specs = [("us-gaap", "CommonStockSharesOutstanding"),
             ("dei", "EntityCommonStockSharesOutstanding")]
    for taxo, concept in specs:
        url = (f"https://data.sec.gov/api/xbrl/companyconcept/"
               f"CIK{cik}/{taxo}/{concept}.json")
        try:
            j = json.loads(_get(url, tries=3))
        except Exception:
            continue
        recs = []
        for unit_key, facts in j.get("units", {}).items():
            for f in facts:
                end, filed, val = f.get("end"), f.get("filed"), f.get("val")
                if end and filed and val is not None:
                    recs.append((pd.Timestamp(end), pd.Timestamp(filed), float(val)))
        if recs:
            return recs
    return []


def _point_in_time_shares(recs: list[tuple], year_ends: pd.DatetimeIndex) -> pd.Series:
    """Year-end point-in-time share series: for each December 31, the value from the most
    recently *filed* fact known by that date (filed <= year-end). No look-ahead."""
    if not recs:
        return pd.Series(index=year_ends, dtype=float)
    df = pd.DataFrame(recs, columns=["end", "filed", "val"]).sort_values("filed")
    out = {}
    for ye in year_ends:
        known = df[df["filed"] <= ye]
        # among facts already filed, take the one with the latest period_end
        out[ye] = float(known.sort_values("end")["val"].iloc[-1]) if len(known) else np.nan
    return pd.Series(out, index=year_ends, dtype=float)


def _yf_raw_adj(tickers, start=START, end="2026-01-01", retries=3):
    """Wide year-end (raw close, adjusted close) frames for ``tickers`` (yfinance)."""
    import yfinance as yf

    last = None
    for attempt in range(retries):
        try:
            h = yf.download(list(tickers), start=start, end=end, auto_adjust=False,
                            progress=False)
            break
        except Exception as exc:  # pragma: no cover
            last = exc
            time.sleep(1.5 * (attempt + 1))
    else:
        raise RuntimeError(f"price download failed: {last}")
    # h.columns: MultiIndex (field, ticker)
    raw = h["Close"] if "Close" in h.columns.get_level_values(0) else h[["Close"]]
    adj = h["Adj Close"] if "Adj Close" in h.columns.get_level_values(0) else raw
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    adj.index = pd.DatetimeIndex(adj.index).tz_localize(None)
    raw_ye = raw.resample("YE").last()
    adj_ye = adj.resample("YE").last()
    return raw_ye, adj_ye


def fetch_real(tickers=None, start=START) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download the basket, build December-year-end (raw close, adjusted close, point-in-time
    shares) frames, and cache all three under ``_cache/``. Network-only; run once.

    A ticker for which EDGAR returns no share facts is dropped from all three frames so the
    sort sees an aligned panel.
    """
    if tickers is None:
        tickers = BASKET
    raw_ye, adj_ye = _yf_raw_adj(tickers, start=start)
    year_ends = raw_ye.index

    t2c = _ticker_cik_map()
    shares = {}
    for tk in tickers:
        cik = t2c.get(tk)
        if cik is None:
            print(f"  ! no CIK for {tk}")
            continue
        recs = _edgar_shares_facts(cik)
        pit = _point_in_time_shares(recs, year_ends)
        if pit.notna().sum() >= LOOKBACK + 2:   # need enough history for a 5y window + forward
            shares[tk] = pit
        else:
            print(f"  ! thin EDGAR shares for {tk} ({int(pit.notna().sum())} year-ends)")
        time.sleep(0.12)  # be polite to SEC
    sh_ye = pd.DataFrame(shares)

    keep = [c for c in raw_ye.columns if c in sh_ye.columns]
    raw_ye, adj_ye, sh_ye = raw_ye[keep], adj_ye[keep], sh_ye[keep]

    os.makedirs(CACHE_DIR, exist_ok=True)
    raw_ye.to_csv(PRICE_RAW_CSV)
    adj_ye.to_csv(PRICE_ADJ_CSV)
    sh_ye.to_csv(SHARES_CSV)
    print(f"cached {len(keep)} names, {len(raw_ye)} year-ends "
          f"({raw_ye.index.min().year}->{raw_ye.index.max().year})")
    return raw_ye, adj_ye, sh_ye


# --------------------------------------------------------------------------- #
# Offline core
# --------------------------------------------------------------------------- #
def have_real() -> bool:
    return (os.path.exists(PRICE_RAW_CSV) and os.path.exists(PRICE_ADJ_CSV)
            and os.path.exists(SHARES_CSV))


def load_real() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the cached year-end (raw close, adjusted close, point-in-time shares) frames."""
    raw = pd.read_csv(PRICE_RAW_CSV, index_col=0, parse_dates=True).sort_index()
    adj = pd.read_csv(PRICE_ADJ_CSV, index_col=0, parse_dates=True).sort_index()
    sh = pd.read_csv(SHARES_CSV, index_col=0, parse_dates=True).sort_index()
    cols = [c for c in raw.columns if c in adj.columns and c in sh.columns]
    return raw[cols], adj[cols], sh[cols]


def drop_partial_last_year(raw, adj, sh, asof: pd.Timestamp | None = None):
    """House rule: a stamped run carries no partial year. Drop the final year-end row if it is
    the in-progress current year (its December bar has not fully occurred yet)."""
    if asof is None:
        asof = pd.Timestamp(AS_OF)
    if len(raw) == 0:
        return raw, adj, sh
    if raw.index[-1].year > asof.year:
        return raw.iloc[:-1], adj.iloc[:-1], sh.iloc[:-1]
    return raw, adj, sh


def fingerprint(raw, adj, sh) -> str:
    """Short content fingerprint of the (raw, adj, shares) panel for the as-of stamp."""
    h = hashlib.sha1()
    for df in (raw, adj, sh):
        h.update(np.ascontiguousarray(df.fillna(0.0).to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_names: int = 40, n_years: int = 12, edge: float = 0.0,
                    seed: int = 790, idio_vol: float = 0.28,
                    mkt_vol: float = 0.16, mkt_drift: float = 0.08,
                    lookback: int = LOOKBACK
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministic cross-sectional panel with a **planted composite-issuance edge**.

    Each name has a persistent equity-issuance propensity (chronic diluters vs chronic
    buyback-ers) plus yearly noise; its share count compounds that issuance. The annual
    **return** of a name = market + idiosyncratic noise **minus** ``edge x (its 5-year composite
    issuance rank, centred)`` — so when ``edge>0``, names that issued a lot of equity over the
    trailing 5 years earn less and net buyback-ers earn more, exactly the Daniel-Titman tilt.
    With ``edge=0`` composite issuance is unrelated to return: the long-short sort must NOT
    manufacture a t>=2 from ~40 names over a handful of years, however the draws fall.

    Returns ``(raw_px, adj_px, shares)`` in the wide year-end shape the strategy layer consumes
    (columns ``N00..``). Here raw==adj (a dividend-free synthetic world), so the composite
    measure reduces to exactly the log share-count growth — clean for the control. A decorative
    year-end index is built with ``pd.period_range`` (never a huge ``date_range`` span).
    """
    rng = np.random.default_rng(seed)
    cols = [f"N{j:02d}" for j in range(n_names)]
    idx = pd.period_range("2010", periods=n_years + 1, freq="Y").to_timestamp(how="end").normalize()
    idx = pd.DatetimeIndex(idx, name="date")

    base_issue = rng.normal(0.0, 0.05, size=n_names)   # persistent per-year dilution propensity
    shares = np.empty((n_years + 1, n_names))
    shares[0] = 1.0e9 * np.exp(rng.normal(0.0, 0.2, size=n_names))
    issue_hist = np.empty((n_years, n_names))
    for y in range(n_years):
        issue = base_issue + rng.normal(0.0, 0.02, size=n_names)
        issue_hist[y] = issue
        shares[y + 1] = shares[y] * (1.0 + issue)

    px = np.empty((n_years + 1, n_names))
    px[0] = 100.0
    for y in range(n_years):
        # 5-year composite issuance = log share-count growth over the trailing `lookback` years
        if y >= lookback:
            comp = np.log(shares[y] / shares[y - lookback])
        else:
            comp = np.log(shares[y] / shares[0] + 1e-12)
        z = (comp - comp.mean()) / (comp.std() + 1e-12)
        mkt = mkt_drift + mkt_vol * rng.standard_normal()
        idio = idio_vol * rng.standard_normal(n_names)
        rets = mkt + idio - edge * z            # high composite issuers earn less
        px[y + 1] = px[y] * (1.0 + rets)

    raw = pd.DataFrame(px, index=idx, columns=cols)
    shares_df = pd.DataFrame(shares, index=idx, columns=cols)
    return raw, raw.copy(), shares_df
