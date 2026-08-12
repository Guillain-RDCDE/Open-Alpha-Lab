"""Data layer for Study 853 — Days-Sales-Outstanding Buildup.

The claim under test: **rising Days Sales Outstanding (DSO) is a red flag.** DSO is how many
days of sales are tied up in unpaid customer invoices —

    DSO = AccountsReceivableNetCurrent / (Revenues / 365)

When receivables grow *faster* than sales, DSO climbs, and the Abarbanell-Bushee fundamental-
signal literature reads that as trouble: channel-stuffing (shipping goods customers didn't pull,
booked on credit), aggressive revenue recognition, or deteriorating collections. The prediction
is a **negative** one — high DSO *buildup* precedes *weaker* future earnings and returns — so the
tradeable expression is **long the low-DSO-change names, short the high-DSO-change names**.

Two sources, both offline-friendly once cached:

* **Real tape.**
  - Daily adjusted closes for a fixed basket of ~48 large US filers with real trade receivables
    (tech-hardware, industrials, healthcare, consumer, materials — financials excluded, they have
    no DSO) from yfinance (no key).
  - Per name, the full history of the **current net accounts-receivable** balance from EDGAR's
    XBRL ``companyconcept`` API (``data.sec.gov``): us-gaap ``AccountsReceivableNetCurrent`` with a
    fallback to ``ReceivablesNetCurrent``; plus quarterly **Revenues** (fallback
    ``RevenueFromContractWithCustomerExcludingAssessedTax``). For every figure we keep the period
    end and the **filing date** of the 10-Q/10-K that disclosed it — the day the number went
    public.
  From these we build, per name, **DSO** (receivables ÷ annualised daily revenue) and its
  **year-over-year change in days** (this quarter's DSO minus the same quarter a year ago) — the
  buildup signal — plus a unit-free **percentage-change** robustness variant. Each event is one
  (ticker, filing date) row carrying the signal; forward returns are measured strictly *after* the
  filing is public (no look-ahead).

* **Synthetic.** A deterministic, fixed-seed generator producing a price + signal panel in which
  forward returns carry a **planted** component proportional to *minus* the DSO buildup (knob
  ``edge``): high-buildup names drift down, low-buildup names drift up — exactly the claimed red
  flag. It is the positive control: with ``edge = 0`` the long-short must NOT manufacture
  significance; with a large planted ``edge`` it must light up. No network.

Honest about coverage: this is a **thin, uneven** panel. Deep-history names (AAPL, IBM, CAT, JNJ,
PG…) go back to ~2009 on EDGAR XBRL, but per-name revenue tagging changed with ASC 606 (2018) and
some names switch concepts mid-history, so the matched (AR, revenue) cross-section is smaller than
the raw name count. That is a first-class caveat of the study, not a footnote.

Pure numpy + pandas + stdlib for the offline path. ``fetch_panel`` (network) is used once to build
the cache and is never imported by the notebooks' offline cells.
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
PRICES_CACHE = os.path.join(CACHE_DIR, "dso_prices.csv")
EVENTS_CACHE = os.path.join(CACHE_DIR, "dso_events.csv")

UA = "OpenAlphaLab research contact@example.com"

AS_OF = "2026-06-30"        # last complete calendar month at publication
ERA_SPLIT = "2018-01-01"    # pre/post the ASC-606 revenue-tagging break

DAYS_Q = 365.0 / 4.0        # ≈91.25 — annualise one quarter of revenue to a daily run-rate

# A transparent, fixed basket of large US filers that carry genuine trade receivables and report
# both a current-AR balance and quarterly revenue on EDGAR. Financials/REITs are excluded (DSO is
# undefined for them). This is a *survivors* basket (all still trading) — survivorship is named on
# the Signal axis (see docs/references.md): a fixed surviving-names roster cannot include firms
# that were acquired or delisted. For a long-low-change/short-high-change signal both legs are
# survivors, so the first-order tilt partly cancels, but it can only be argued away, not ignored.
BASKET = [
    # tech / hardware / semis
    "AAPL", "MSFT", "CSCO", "INTC", "IBM", "ORCL", "TXN", "QCOM", "HPQ", "ADI",
    "MU", "AMAT", "KLAC",
    # industrials
    "CAT", "DE", "HON", "MMM", "EMR", "ETN", "ITW", "PH", "ROK", "GD", "LMT", "DOV",
    # healthcare / medtech
    "JNJ", "PFE", "MRK", "ABT", "MDT", "TMO", "BAX", "BDX", "SYK",
    # consumer / retail
    "PG", "KO", "PEP", "CL", "KMB", "NKE", "HD", "LOW", "WMT", "TGT", "COST",
    # materials
    "DD", "PPG",
]

# us-gaap concepts, in order of preference (longest per-name history wins).
AR_CONCEPTS = (
    "AccountsReceivableNetCurrent",
    "ReceivablesNetCurrent",
)
REV_CONCEPTS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)


# --------------------------------------------------------------------------- #
# EDGAR helpers (network; cache builders only)
# --------------------------------------------------------------------------- #
def _http_json(url: str, retries: int = 3) -> dict | None:
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:
            time.sleep(0.5 * (k + 1))
    return None


def _ticker_cik_map() -> dict[str, str]:
    """Resolve ticker → 10-digit zero-padded CIK from the SEC company-ticker map (network)."""
    d = _http_json("https://www.sec.gov/files/company_tickers.json")
    out: dict[str, str] = {}
    if d is None:
        return out
    for _, row in d.items():
        tk = str(row.get("ticker", "")).upper()
        cik = row.get("cik_str")
        if tk and cik is not None and tk not in out:
            out[tk] = f"{int(cik):010d}"
    return out


def _edgar_concept(cik: str, concept: str) -> dict | None:
    url = (f"https://data.sec.gov/api/xbrl/companyconcept/"
           f"CIK{cik}/us-gaap/{concept}.json")
    return _http_json(url)


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


def _quarterly_flow_series(cik: str, concepts: tuple[str, ...]) -> pd.DataFrame:
    """Best (longest-history) *quarterly* flow series (revenue) for one CIK.

    Flow (duration) concepts report a value over a start..end span. We keep only ~one-quarter
    spans (60-100 days), de-duplicate on the period end (earliest filing wins). Columns: end,
    filed, val.
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
            if not (u.get("end") and u.get("start") and u.get("filed")
                    and u.get("val") is not None):
                continue
            span = (pd.Timestamp(u["end"]) - pd.Timestamp(u["start"])).days
            if not (60 <= span <= 100):
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
# Signal construction — YoY change in DSO (point-in-time)
# --------------------------------------------------------------------------- #
def _yoy_match(series: pd.DataFrame, end: pd.Timestamp,
               lo: int = 300, hi: int = 430) -> float | None:
    """Value in ``series`` whose end is ~1 year before ``end`` (gap in [lo, hi] days)."""
    if series.empty:
        return None
    gap = (end - series["end"]).dt.days
    m = (gap >= lo) & (gap <= hi)
    if not m.any():
        return None
    sub = series.loc[m]
    k = (sub["end"].map(lambda e: abs((end - e).days - 365))).idxmin()
    return float(series.loc[k, "val"])


def _asof_value(series: pd.DataFrame, end: pd.Timestamp, tol: int = 20) -> float | None:
    """Value in ``series`` whose end matches ``end`` within ``tol`` days (same quarter)."""
    if series.empty:
        return None
    gap = (series["end"] - end).abs().dt.days
    if gap.min() > tol:
        return None
    return float(series.loc[gap.idxmin(), "val"])


def _dso(ar: float, rev_q: float) -> float | None:
    """Days sales outstanding from a balance-sheet AR level and ONE quarter of revenue.

    DSO = AR / (Revenues/365). With quarterly revenue, the daily run-rate is rev_q / DAYS_Q
    (annualise the quarter), so DSO = AR * DAYS_Q / rev_q. Any constant annualisation factor
    cancels in the same-fiscal-quarter YoY *change*, which also nets out seasonality.
    """
    if rev_q is None or rev_q <= 0 or ar is None or ar < 0:
        return None
    return ar * DAYS_Q / rev_q


COLS = ["end", "filed", "ar", "rev", "dso", "dso_yoy_chg", "dso_yoy_pct",
        "dso_score", "dso_pct_score", "sales_yoy", "next_rev_yoy"]


def build_signal(ar: pd.DataFrame, rev: pd.DataFrame) -> pd.DataFrame:
    """One row per receivables quarter carrying the DSO signals and forward-sales fields.

    For each AR observation (end E, filed F, balance A):
      * ``dso``        = A / (Rev(E)/365)               — the level, in days
      * ``dso_yoy_chg`` = DSO(E) − DSO(E−1yr)           — the buildup, in days (PRIMARY)
      * ``dso_yoy_pct`` = DSO(E)/DSO(E−1yr) − 1         — unit-free buildup (robustness)
      * ``dso_score``   = −dso_yoy_chg                  — signed so HIGH score = LOW buildup =
        the ATTRACTIVE (long) side per the claim; the whole 798-style machinery then goes long
        the top score / short the bottom with no special-casing, and a WRONG-SIGN result shows
        up as a *negative* long-short.
      * ``dso_pct_score`` = −dso_yoy_pct
      * ``sales_yoy``    = Rev(E)/Rev(E−1yr) − 1        — contemporaneous sales growth
      * ``next_rev_yoy`` = Rev(E+1q)/Rev(E+1q−1yr) − 1  — FUTURE sales growth (the channel-
        stuffing mechanism check: does DSO buildup precede a sales *deceleration*?)
    Rows without a valid YoY DSO change are dropped.
    """
    rows = []
    a = ar.sort_values("end").reset_index(drop=True)
    for _, r in a.iterrows():
        end, filed, ar_val = r["end"], r["filed"], r["val"]
        rev_now = _asof_value(rev, end)
        dso_now = _dso(ar_val, rev_now)
        if dso_now is None:
            continue
        ar_prior = _yoy_match(ar, end)
        rev_prior = _yoy_match(rev, end)
        dso_prior = _dso(ar_prior, rev_prior)
        if dso_prior is None or dso_prior <= 0:
            continue
        chg = dso_now - dso_prior
        pct = dso_now / dso_prior - 1.0
        sales_yoy = (rev_now / rev_prior - 1.0) if (rev_now and rev_prior and rev_prior > 0) else np.nan
        nxt = _asof_value(rev, end + pd.Timedelta(days=91), tol=25)
        nxt_prior = _yoy_match(rev, end + pd.Timedelta(days=91))
        next_rev_yoy = (nxt / nxt_prior - 1.0) if (nxt and nxt_prior and nxt_prior > 0) else np.nan
        rows.append({"end": end, "filed": filed, "ar": ar_val, "rev": rev_now,
                     "dso": dso_now, "dso_yoy_chg": chg, "dso_yoy_pct": pct,
                     "dso_score": -chg, "dso_pct_score": -pct,
                     "sales_yoy": sales_yoy, "next_rev_yoy": next_rev_yoy})
    return pd.DataFrame(rows, columns=COLS)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = "2007-01-01", end: str | None = None,
                prices_path: str = PRICES_CACHE, events_path: str = EVENTS_CACHE
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download basket prices + EDGAR receivables / revenue, build DSO signals, cache.

    Network-only; used once to build the cache. Writes a wide adjusted-close CSV and a long event
    CSV (one row per ticker × receivables quarter). Never imported by the offline notebook cells.
    """
    import yfinance as yf

    cik_map = _ticker_cik_map()

    raw = yf.download(BASKET, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    keep = [c for c in raw.columns if raw[c].notna().mean() >= 0.10]
    prices = raw[keep].copy()
    os.makedirs(os.path.dirname(prices_path), exist_ok=True)
    prices.to_csv(prices_path)

    rows = []
    for tk in BASKET:
        cik = cik_map.get(tk)
        if cik is None or tk not in prices.columns:
            continue
        ar = _instant_series(cik, AR_CONCEPTS)
        if len(ar) < 8:
            continue
        rev = _quarterly_flow_series(cik, REV_CONCEPTS)
        if len(rev) < 8:
            continue
        sig = build_signal(ar, rev)
        for _, s in sig.iterrows():
            rows.append({"ticker": tk, **s.to_dict()})
        time.sleep(0.15)
    events = pd.DataFrame(rows).dropna(subset=["filed", "dso_yoy_chg"])
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
    """Long event frame (filings on/before AS_OF)."""
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
                    seed: int = 853, sig_daily: float = 0.020
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic price panel + DSO-buildup events with a PLANTED (negative) return knob.

    Each name has a daily random-walk price. Quarterly filing dates (~63 trading days apart) each
    get a DSO-change signal drawn N(0, 8) days. If ``edge`` != 0, the ~63 sessions of forward
    return following each filing get an *extra* daily drift of ``-edge * chg / (8 * 63)`` — so
    HIGH DSO buildup drifts DOWN and LOW buildup drifts UP, exactly the claimed red flag. With
    ``edge = 0`` the forward path is pure noise: the long-short must NOT reach significance however
    the noise falls.

    Returns (prices wide frame, event table) in the same shape as ``load_real``.
    """
    rng = np.random.default_rng(seed)
    H = 63
    n_days = n_quarters * H + 200
    idx = pd.bdate_range("2010-01-04", periods=n_days)   # n_days well under the 10k daily horizon
    names = [f"N{i:02d}" for i in range(n_names)]
    sig_days = 8.0                                        # sd of the DSO-change signal, in days

    price_cols = {}
    ev_rows = []
    for name in names:
        ret = rng.normal(0.0004, sig_daily, size=n_days)
        first = int(rng.integers(80, 140))
        q = first
        while q < n_days - H - 5:
            chg = rng.normal(0.0, sig_days)
            if edge != 0.0:
                ret[q + 1:q + 1 + H] += -edge * chg / (sig_days * H)
            ev_rows.append({"ticker": name, "_pos": q, "dso_yoy_chg": float(chg)})
            q += H + int(rng.integers(-4, 5))
        price_cols[name] = 100.0 * np.exp(np.cumsum(ret))

    prices = pd.DataFrame(price_cols, index=idx)
    ev = pd.DataFrame(ev_rows)
    ev["filed"] = [idx[p] for p in ev["_pos"]]
    ev["end"] = ev["filed"] - pd.Timedelta(days=40)
    ev["ar"] = np.nan
    ev["rev"] = np.nan
    ev["dso"] = 45.0 + ev["dso_yoy_chg"]
    ev["dso_yoy_pct"] = ev["dso_yoy_chg"] / 45.0
    ev["dso_score"] = -ev["dso_yoy_chg"]
    ev["dso_pct_score"] = -ev["dso_yoy_pct"]
    ev["sales_yoy"] = np.nan
    ev["next_rev_yoy"] = np.nan
    ev = ev[["ticker", *COLS]]
    return prices, ev.sort_values(["ticker", "filed"]).reset_index(drop=True)
