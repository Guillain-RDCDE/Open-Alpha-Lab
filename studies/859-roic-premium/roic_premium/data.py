"""Data layer for Study 859 — Return-on-Invested-Capital Premium.

The claim under test: **ROIC — return on invested capital — is a cleaner "quality" gauge
than ROE.** Where ROE = earnings / book equity is contaminated by leverage (a firm can lift
ROE just by borrowing), ROIC divides *unlevered* operating profit by *all* the capital put to
work, so it is meant to isolate how well the business itself turns capital into profit:

    NOPAT           = OperatingIncomeLoss * (1 - tax_rate)      (net operating profit after tax)
    InvestedCapital = total debt + book equity - cash          (the capital actually deployed)
    ROIC            = NOPAT / InvestedCapital

The quality-factor pitch: sort names on ROIC **level** (and on its year-over-year **change**),
go long high-ROIC / short low-ROIC, and you should earn a forward spread — a purer version of
the ROE / gross-profitability quality premium. The study's decisive question is not just "does
ROIC sort returns?" but "**does ROIC add anything over plain ROE and gross profitability on the
same panel?**"

Two sources, both offline-friendly once cached:

* **Real tape.**
  - Daily adjusted (total-return) closes for a fixed basket of ~44 large US **non-financial**
    filers (yfinance, no key). ROIC / invested capital is economically meaningful for operating
    companies; it is not defined for banks/insurers, so the basket deliberately excludes them.
  - Per name, from EDGAR's XBRL ``companyconcept`` API (``data.sec.gov``): the trailing-twelve-
    month flow ``OperatingIncomeLoss`` (NOPAT numerator) and ``NetIncomeLoss`` (ROE numerator)
    and ``GrossProfit`` (gross-profitability numerator), plus the point-in-time balance-sheet
    instants ``StockholdersEquity``, ``LongTermDebtNoncurrent`` (total-debt proxy),
    ``CashAndCashEquivalentsAtCarryingValue`` and ``Assets``. For each figure we keep the period
    end and the **filing date** of the 10-Q/10-K that disclosed it — the date the number became
    public. Each event is one (ticker, filing) row carrying ROIC level, ROIC change, ROE and
    gross profitability; the forward return is measured strictly *after* the filing date (no
    look-ahead).

* **Synthetic.** A deterministic, fixed-seed generator producing a price + signal panel in
  which forward returns carry a **planted** component proportional to the ROIC signal (knob
  ``edge``). It is the positive control: with ``edge = 0`` the long-short must NOT manufacture
  significance; with a large planted ``edge`` it must light up. No network.

**A note on the tax rate.** We use a flat ``TAX_RATE`` (21%, the post-2017 US statutory rate) to
turn operating income into NOPAT. Because a *common* scalar ``(1 - tax_rate)`` multiplies every
name's NOPAT identically, it **does not change the cross-sectional ROIC ranking at all** — it
only rescales the reported ROIC magnitude. The sort (and hence every long-short result) is
invariant to this choice; only the printed ROIC levels move. We therefore do not chase a
per-firm effective rate.

Honest about coverage: this is a **thin, uneven** panel. XBRL fundamentals only begin ~2009-2010;
several names re-registered under new CIKs after mergers/reorgs (XOM 2024, DIS 2019, LIN 2018),
truncating their machine-readable history. The cross-section is therefore small early and only
gets reasonably wide after ~2012. That is a first-class caveat of the study, not a footnote.
And the basket is **current survivors** — survivorship is named on the Signal axis.

Pure numpy + pandas + stdlib for the offline path. ``fetch_panel`` (network) is used once to
build the cache and is never imported by the notebooks' offline cells.
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
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "roic_prices.csv")
EVENTS_CACHE = os.path.join(CACHE_DIR, "roic_events.csv")

UA = "open-alpha-lab research guillain@poulpe.us"

AS_OF = "2026-06-30"        # last complete calendar month at publication
ERA_SPLIT = "2018-01-01"    # pre/post the 2017 Tax-Cut & Jobs Act (statutory 35% -> 21%)

TAX_RATE = 0.21             # flat NOPAT tax haircut; a common scalar -> invariant to the sort

# A transparent, fixed basket of large US NON-FINANCIAL filers for which ROIC / invested capital
# is economically defined (banks/insurers excluded by design). Current survivors — survivorship
# is named on the Signal axis.
BASKET = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ORCL", "CSCO", "INTC", "IBM",
    "QCOM", "TXN", "ADBE", "CRM", "WMT", "HD", "MCD", "NKE", "SBUX", "PG",
    "KO", "PEP", "COST", "TGT", "LOW", "JNJ", "PFE", "MRK", "ABT", "TMO",
    "LLY", "AMGN", "CAT", "DE", "HON", "MMM", "LMT", "UPS", "XOM", "CVX",
    "LIN", "DIS", "VZ", "T",
]

# SEC CIK (10-digit, zero-padded), resolved once from the SEC ticker map and frozen so the
# offline path never needs the network.
CIK = {
    "AAPL": "0000320193", "MSFT": "0000789019", "GOOGL": "0001652044", "AMZN": "0001018724",
    "META": "0001326801", "NVDA": "0001045810", "ORCL": "0001341439", "CSCO": "0000858877",
    "INTC": "0000050863", "IBM": "0000051143", "QCOM": "0000804328", "TXN": "0000097476",
    "ADBE": "0000796343", "CRM": "0001108524", "WMT": "0000104169", "HD": "0000354950",
    "MCD": "0000063908", "NKE": "0000320187", "SBUX": "0000829224", "PG": "0000080424",
    "KO": "0000021344", "PEP": "0000077476", "COST": "0000909832", "TGT": "0000027419",
    "LOW": "0000060667", "JNJ": "0000200406", "PFE": "0000078003", "MRK": "0000310158",
    "ABT": "0000001800", "TMO": "0000097745", "LLY": "0000059478", "AMGN": "0000318154",
    "CAT": "0000018230", "DE": "0000315189", "HON": "0000773840", "MMM": "0000066740",
    "LMT": "0000936468", "UPS": "0001090727", "XOM": "0002115436", "CVX": "0000093410",
    "LIN": "0001707925", "DIS": "0001744489", "VZ": "0000732712", "T": "0000732717",
}

# Flow (duration) concepts, in order of preference (longest per-name history wins).
OIL_CONCEPTS = ("OperatingIncomeLoss",)
NI_CONCEPTS = ("NetIncomeLoss", "ProfitLoss")
GP_CONCEPTS = ("GrossProfit",)

# Instant (point-in-time) balance-sheet concepts.
EQUITY_CONCEPTS = (
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
)
DEBT_CONCEPTS = (
    "LongTermDebtNoncurrent",
    "LongTermDebt",
    "LongTermDebtAndCapitalLeaseObligations",
)
CASH_CONCEPTS = (
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
)
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

    Instant us-gaap concepts report one value AT a period end. We keep every 10-Q/10-K
    observation, de-duplicate on the period end (keeping the EARLIEST filing that disclosed it —
    first public disclosure, no restatement look-ahead), and pick the concept that yields the
    most distinct quarter ends. Columns: end, filed, val.
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
    """Best (longest-history) *single-quarter* flow series (e.g. operating income) for one CIK.

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


def ttm(qflow: pd.DataFrame, window_days: int = 400,
        span_lo: int = 250, span_hi: int = 330) -> pd.DataFrame:
    """Trailing-twelve-month sum of a single-quarter flow series.

    For each quarter end E we take the four most recent quarterly values whose ends fall in
    ``(E - window_days, E]`` and require them to be **four consecutive quarters** — the span
    between the earliest and latest of the four ends must sit in ``[span_lo, span_hi]`` days
    (three ~91-day gaps ≈ 273 days). This rejects both a partial year (< 4 quarters) and a
    year with a missing quarter (which would stretch the span past ``span_hi``). The TTM figure
    is stamped with the filing date of quarter E (the 10-Q/10-K that closed the year). Columns:
    end, filed, val.
    """
    if qflow.empty:
        return pd.DataFrame(columns=["end", "filed", "val"])
    q = qflow.sort_values("end").reset_index(drop=True)
    rows = []
    for _, r in q.iterrows():
        E = r["end"]
        w = q[(q["end"] <= E) & (q["end"] > E - pd.Timedelta(days=window_days))]
        if len(w) < 4:
            continue
        w4 = w.sort_values("end").iloc[-4:]
        span = (w4["end"].iloc[-1] - w4["end"].iloc[0]).days
        if not (span_lo <= span <= span_hi):
            continue
        rows.append({"end": E, "filed": r["filed"], "val": float(w4["val"].sum())})
    return pd.DataFrame(rows, columns=["end", "filed", "val"])


# --------------------------------------------------------------------------- #
# Signal construction — ROIC level, ROIC change, ROE, gross profitability
# --------------------------------------------------------------------------- #
def _asof_value(series: pd.DataFrame, end: pd.Timestamp, tol: int = 25) -> float | None:
    """Value in ``series`` whose end matches ``end`` within ``tol`` days (same quarter)."""
    if series is None or series.empty:
        return None
    gap = (series["end"] - end).abs().dt.days
    if gap.min() > tol:
        return None
    return float(series.loc[gap.idxmin(), "val"])


def _yoy_value(series: pd.DataFrame, end: pd.Timestamp,
               lo: int = 300, hi: int = 430) -> float | None:
    """Value in ``series`` whose end is ~1 year before ``end`` (gap in [lo, hi] days)."""
    if series is None or series.empty:
        return None
    gap = (end - series["end"]).dt.days
    m = (gap >= lo) & (gap <= hi)
    if not m.any():
        return None
    sub = series.loc[m]
    k = (sub["end"].map(lambda e: abs((end - e).days - 365))).idxmin()
    return float(series.loc[k, "val"])


def build_signal(oil_ttm: pd.DataFrame, ni_ttm: pd.DataFrame, gp_ttm: pd.DataFrame,
                 equity: pd.DataFrame, debt: pd.DataFrame, cash: pd.DataFrame,
                 assets: pd.DataFrame, tax_rate: float = TAX_RATE) -> pd.DataFrame:
    """One row per balance-sheet quarter carrying ROIC and the contrast signals.

    Anchored on the equity instant series (the balance-sheet quarters). For each equity
    observation (end E, filed F, book equity Q):

      * InvestedCapital = debt(E) + Q - cash(E)     (require > 0)
      * NOPAT           = OperatingIncomeLoss_TTM(E) * (1 - tax_rate)
      * ``roic``        = NOPAT / InvestedCapital                       — the primary level signal
      * ``roic_chg``    = roic(E) - roic(E - 1yr)                       — the change variant
      * ``roe``         = NetIncomeLoss_TTM(E) / Q                      — the ROE contrast (200)
      * ``gp``          = GrossProfit_TTM(E) / Assets(E)                — the gross-profit contrast (122)

    Rows without a valid ROIC (missing operating income, missing debt/cash, non-positive
    invested capital) are dropped. ``roic_chg`` is NaN where no ~1-year-prior ROIC exists.
    """
    rows = []
    eq = equity.sort_values("end").reset_index(drop=True)
    for _, r in eq.iterrows():
        E, F, q = r["end"], r["filed"], r["val"]
        d = _asof_value(debt, E)
        c = _asof_value(cash, E)
        oi = _asof_value(oil_ttm, E)
        if d is None or c is None or oi is None:
            continue
        ic = d + q - c
        if ic <= 0:
            continue
        nopat = oi * (1.0 - tax_rate)
        roic = nopat / ic
        ni = _asof_value(ni_ttm, E)
        roe = (ni / q) if (ni is not None and q > 0) else np.nan
        gpv = _asof_value(gp_ttm, E)
        a = _asof_value(assets, E)
        gp = (gpv / a) if (gpv is not None and a and a > 0) else np.nan
        rows.append({"end": E, "filed": F, "invested_capital": ic, "roic": roic,
                     "roe": roe, "gp": gp})
    cols = ["end", "filed", "invested_capital", "roic", "roic_chg", "roe", "gp"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows).sort_values("end").reset_index(drop=True)
    roic_ser = df[["end"]].copy()
    roic_ser["val"] = df["roic"].to_numpy()
    chg = []
    for _, r in df.iterrows():
        prior = _yoy_value(roic_ser, r["end"])
        chg.append(r["roic"] - prior if prior is not None else np.nan)
    df["roic_chg"] = chg
    return df[cols]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = "2009-01-01", end: str | None = None,
                prices_path: str = PRICES_CACHE, events_path: str = EVENTS_CACHE
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download basket prices + EDGAR fundamentals, build ROIC signals, cache.

    Network-only; used once to build the cache. Writes a wide adjusted-close CSV and a long
    event CSV (one row per ticker × balance-sheet quarter). Never imported by the offline
    notebook cells.
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
        oil = ttm(_quarterly_flow_series(cik, OIL_CONCEPTS))
        if len(oil) < 4:
            time.sleep(0.15)
            continue
        ni = ttm(_quarterly_flow_series(cik, NI_CONCEPTS))
        gp = ttm(_quarterly_flow_series(cik, GP_CONCEPTS))
        equity = _instant_series(cik, EQUITY_CONCEPTS)
        debt = _instant_series(cik, DEBT_CONCEPTS)
        cash = _instant_series(cik, CASH_CONCEPTS)
        assets = _instant_series(cik, ASSET_CONCEPTS)
        sig = build_signal(oil, ni, gp, equity, debt, cash, assets)
        for _, s in sig.iterrows():
            rows.append({"ticker": tk, **s.to_dict()})
        time.sleep(0.15)
    events = pd.DataFrame(rows).dropna(subset=["filed", "roic"])
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
    """Long event frame: ticker, end, filed, invested_capital, roic, roic_chg, roe, gp
    (filings on/before AS_OF)."""
    ev = pd.read_csv(path, parse_dates=["end", "filed"])
    ev = ev[ev["filed"] <= pd.Timestamp(AS_OF)]
    return ev.sort_values(["ticker", "filed"]).reset_index(drop=True)


def load_real() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached prices + events in one call."""
    return load_prices(), load_events()


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint for any DataFrame, for the as-of stamp."""
    arr = df.to_numpy(dtype=float, na_value=0.0)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_names: int = 30, n_quarters: int = 44, edge: float = 0.0,
                    seed: int = 859, sig_daily: float = 0.018
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic price panel + ROIC events with a PLANTED forward-return knob.

    Each name has a daily random-walk price. Quarterly filing dates (~63 trading days apart)
    each get a ROIC level signal drawn N(0.12, 0.08) (quality varies across firms). If ``edge``
    != 0, the ~63 sessions of forward return following each filing get an *extra* daily drift of
    ``edge * (roic - mean_roic) / 63`` — high-ROIC names drift up, low ones drift down, exactly
    the claimed premium. With ``edge = 0`` the forward path is pure noise: the long-short must
    NOT reach significance however the noise falls.

    The ``roe`` column is a leverage-contaminated cousin of ROIC (correlated but noisier) and
    ``gp`` an independent quality proxy, so the "does ROIC add anything?" contrast has something
    to chew on. ``roic_chg`` is the quarter-on-quarter change in the planted ROIC.

    Returns (prices wide frame, event table) in the same shape as ``load_real``.
    """
    rng = np.random.default_rng(seed)
    H = 63
    n_days = n_quarters * H + 200
    idx = pd.bdate_range("2010-01-04", periods=n_days)
    names = [f"N{i:02d}" for i in range(n_names)]
    mu_roic = 0.12

    price_cols = {}
    ev_rows = []
    for name in names:
        ret = rng.normal(0.0003, sig_daily, size=n_days)
        first = int(rng.integers(80, 140))
        q = first
        prev_roic = None
        while q < n_days - H - 5:
            roic = float(rng.normal(mu_roic, 0.08))
            if edge != 0.0:
                z = (roic - mu_roic) / 0.08          # standardise so the planted power matches
                ret[q + 1:q + 1 + H] += edge * z / H
            chg = (roic - prev_roic) if prev_roic is not None else np.nan
            prev_roic = roic
            ev_rows.append({"ticker": name, "_pos": q, "roic": roic, "roic_chg": chg,
                            "roe": roic * 1.3 + float(rng.normal(0, 0.05)),
                            "gp": float(rng.normal(0.30, 0.10))})
            q += H + int(rng.integers(-4, 5))
        price_cols[name] = 100.0 * np.exp(np.cumsum(ret))

    prices = pd.DataFrame(price_cols, index=idx)
    ev = pd.DataFrame(ev_rows)
    ev["filed"] = [idx[p] for p in ev["_pos"]]
    ev["end"] = ev["filed"] - pd.Timedelta(days=40)
    ev["invested_capital"] = 1.0e9
    ev = ev[["ticker", "end", "filed", "invested_capital", "roic", "roic_chg", "roe", "gp"]]
    return prices, ev.sort_values(["ticker", "filed"]).reset_index(drop=True)
