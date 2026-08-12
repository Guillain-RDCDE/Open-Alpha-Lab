"""Data layer for Study 860 — Accounting Conservatism (C-Score).

The claim under test (Basu 1997, *"The conservatism principle and the asymmetric timeliness of
earnings"*; Penman & Zhang 2002, *"Accounting Conservatism, the Quality of Earnings, and Stock
Returns"*): **conservative** accounting recognises bad news fast and defers good news, so it
systematically **understates** net operating assets and buries **hidden reserves** on the
balance sheet (allowances, valuation reserves, LIFO/inventory reserves). Those reserves later
unwind into earnings. Penman-Zhang build a *C-score* = estimated reserves ÷ net operating
assets and argue the level (and its change) carries information about the quality of earnings
and, in the strong form, about **forward stock returns** — high-conservatism firms hold a
cushion of un-booked value the market may under-price.

We operationalise a deliberately **simplified, coarse** C-score from the reserve/allowance
accounts that firms actually tag in XBRL:

    reserves = AllowanceForDoubtfulAccountsReceivableCurrent
             + InventoryValuationReserves
             + DeferredTaxAssetsValuationAllowance          (whichever are disclosed)

    cscore     = reserves / Assets            (the coarse, always-available workhorse)
    cscore_noa = reserves / NetOperatingAssets (the Penman-Zhang denominator; thinner)

with NOA = Assets − Cash − (Liabilities − Debt). The signal is the *level* of reserve intensity,
known only on the 10-Q/10-K **filing date** (no look-ahead). The trading claim: sort names on the
C-score, go **long high-conservatism / short low-conservatism**, and see whether the forward
return spread is real.

Two sources, both offline-friendly once cached:

* **Real tape.**
  - Daily adjusted closes for a fixed basket of ~42 deep-history US **non-financial** large
    filers (yfinance, no key). Non-financial by design: a bank's "reserve" is the loan-loss
    allowance — a different animal — and the reserve-intensity idea is written for operating firms.
  - Per name, from EDGAR's XBRL ``companyconcept`` API (``data.sec.gov``): the instant reserve /
    allowance balances above, plus **Assets**, **Cash**, **Liabilities**, **Debt** (for NOA) and
    quarterly **NetIncomeLoss** (for ROA and the Basu asymmetry axis). Each figure keeps its
    period end and the **filing date** of the 10-Q/10-K that disclosed it.

* **Synthetic.** A deterministic, fixed-seed generator producing a price + C-score panel in
  which forward returns carry a **planted** component proportional to the conservatism signal
  (knob ``edge``). It is the positive control: with ``edge = 0`` the long-short must NOT
  manufacture significance; with a large planted ``edge`` it must light up. No network.

Honest about coverage: this is a **thin, uneven** panel. Reserve/allowance accounts are
irregularly tagged — many firms disclose only the allowance for doubtful accounts, and only in
some years — so the reserve total is a *floor* on true estimated reserves and the cross-section
is modest. That coarseness is a first-class caveat of the study, not a footnote.

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
PRICES_CACHE = os.path.join(CACHE_DIR, "cs_prices.csv")
EVENTS_CACHE = os.path.join(CACHE_DIR, "cs_events.csv")

UA = "OpenAlphaLab research contact@example.com"

AS_OF = "2026-06-30"        # last complete calendar month at publication
ERA_SPLIT = "2016-01-01"    # pre/post split for the era robustness cut

# A transparent, fixed basket of ~42 deep-history US NON-financial large filers that carry
# tagged reserve/allowance accounts on EDGAR. Financials (banks, insurers) are excluded on
# purpose: their "reserves" are loan-loss / claims allowances, a different construct. This is a
# *survivors* basket (all still trading). Survivorship is named on the Signal axis: a fixed
# surviving-names roster cannot include operating firms that were acquired or failed. Both legs
# of a long-high-conservatism/short-low-conservatism book are drawn from the same survivor pool,
# so the first-order equity-survivorship tilt partly cancels — but it can only be argued about,
# not ignored (see docs/references.md).
BASKET = [
    "AAPL", "MSFT", "INTC", "CSCO", "ORCL", "IBM", "TXN", "QCOM", "HPQ", "ADI",
    "JNJ", "PFE", "MRK", "ABT", "MDT", "BMY", "AMGN", "GILD", "BAX", "SYK",
    "PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "COST", "TGT", "LOW",
    "CAT", "DE", "GE", "HON", "MMM", "EMR", "ITW", "DOV", "PH", "ROK",
    "XOM", "CVX",
]

# us-gaap concepts, in order of preference per figure. Reserves / allowances / Assets / Cash /
# Liabilities / Debt are instant balance-sheet facts; NetIncomeLoss is a quarterly flow.
ADA_CONCEPTS = (
    "AllowanceForDoubtfulAccountsReceivableCurrent",
    "AllowanceForDoubtfulAccountsReceivable",
)
INVRES_CONCEPTS = ("InventoryValuationReserves",)
DTVA_CONCEPTS = ("DeferredTaxAssetsValuationAllowance",)
ASSET_CONCEPTS = ("Assets",)
CASH_CONCEPTS = (
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
)
LIAB_CONCEPTS = ("Liabilities",)
DEBT_CONCEPTS = ("LongTermDebtNoncurrent", "LongTermDebt")
NI_CONCEPTS = ("NetIncomeLoss", "ProfitLoss")


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


def _ticker_cik_map() -> dict:
    """Resolve {TICKER: 10-digit CIK} once from the SEC ticker map (network)."""
    req = urllib.request.Request("https://www.sec.gov/files/company_tickers.json",
                                 headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    out = {}
    for row in d.values():
        out[row["ticker"].upper()] = f"{int(row['cik_str']):010d}"
    return out


def _instant_series(cik: str, concepts: tuple[str, ...]) -> pd.DataFrame:
    """Best (longest-history) instant balance-sheet series for one CIK across ``concepts``.

    Instant us-gaap facts report one value AT a period end. Keep every 10-Q/10-K observation,
    de-duplicate on the period end (keeping the EARLIEST filing that disclosed it — first public
    disclosure, no restatement look-ahead), pick the concept with the most quarter ends.
    Columns: end, filed, val.
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
    """Best (longest-history) *quarterly* (3-month) flow series for one CIK.

    Income-statement us-gaap facts are reported **year-to-date** in 10-Qs (3/6/9-month
    cumulative) and annually in the 10-K, so pure 90-day facts are sparse. We reconstruct
    quarterly flows by **differencing the cumulative YTD chain**: facts sharing a fiscal-year
    ``start`` are cumulative, so sorted by ``end`` the quarterly value at point *i* is
    ``val[i] - val[i-1]`` (and ``val[0]`` for the first, a native 3-month span). We keep only
    points whose implied quarter span is ~one quarter (75-100 days). De-duplicated on the period
    end (earliest filing wins). Columns: end, filed, val.
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
            if not (75 <= span <= 400):
                continue
            rows.append({"start": u["start"], "end": u["end"],
                         "filed": u["filed"], "val": float(u["val"])})
        if not rows:
            continue
        df = pd.DataFrame(rows)
        for c in ("start", "end", "filed"):
            df[c] = pd.to_datetime(df[c])
        df = (df.sort_values("filed").drop_duplicates(subset=["start", "end"], keep="first")
                .sort_values("end").reset_index(drop=True))
        q_rows = []
        for _, g in df.groupby("start"):
            g = g.sort_values("end").reset_index(drop=True)
            prev_val, prev_end = 0.0, g["start"].iloc[0] - pd.Timedelta(days=1)
            for _, r in g.iterrows():
                q_span = (r["end"] - prev_end).days
                q_val = r["val"] - prev_val
                if 75 <= q_span <= 100:
                    q_rows.append({"end": r["end"], "filed": r["filed"], "val": q_val})
                prev_val, prev_end = r["val"], r["end"]
        if not q_rows:
            continue
        qdf = pd.DataFrame(q_rows)
        qdf = (qdf.sort_values("filed").drop_duplicates(subset=["end"], keep="first")
                  .sort_values("end").reset_index(drop=True))
        if len(qdf) > len(best):
            best = qdf
    return best


# --------------------------------------------------------------------------- #
# Signal construction — reserve-intensity C-score (point-in-time)
# --------------------------------------------------------------------------- #
def _asof_value(series: pd.DataFrame, end: pd.Timestamp, tol: int = 20) -> float | None:
    """Instant value whose end matches ``end`` within ``tol`` days (same quarter)."""
    if series is None or series.empty:
        return None
    gap = (series["end"] - end).abs().dt.days
    if gap.min() > tol:
        return None
    return float(series.loc[gap.idxmin(), "val"])


def build_signal(ada: pd.DataFrame, invres: pd.DataFrame, dtva: pd.DataFrame,
                 assets: pd.DataFrame, cash: pd.DataFrame, liab: pd.DataFrame,
                 debt: pd.DataFrame, ni: pd.DataFrame,
                 prices: pd.Series | None = None) -> pd.DataFrame:
    """One row per reserve-disclosing quarter carrying the C-score signals + mechanism fields.

    Driven off the union of reserve-account period ends (a quarter counts if it discloses at
    least one tagged reserve). For each such period end E (filed on F):
      * ``reserves`` = ADA + InvReserve + DTVA  (whichever are disclosed at E; treat missing as 0
        so long as at least one component is present)
      * ``cscore``     = reserves / Assets(E)                — the coarse ranking signal
      * ``noa``        = Assets − Cash − (Liabilities − Debt) — Penman-Zhang denominator
      * ``cscore_noa`` = reserves / NOA(E)                   — the cleaner-but-thinner variant
      * ``roa``        = NI_q(E) / Assets(E)                 — quarterly earnings scale (Basu axis)
      * ``ret_contemp``= stock return over the fiscal quarter ending ~E (Basu news proxy)
    Rows without a positive Assets denominator (or with no reserve component at all) are dropped.
    The filing date F is the reserve disclosure's filing — the signal is known then, never at E.
    """
    cols = ["end", "filed", "reserves", "assets", "noa", "cscore", "cscore_noa",
            "roa", "ret_contemp"]
    # union of reserve-account period ends (earliest filing per end wins)
    parts = [df for df in (ada, invres, dtva) if df is not None and not df.empty]
    if not parts:
        return pd.DataFrame(columns=cols)
    ends = (pd.concat([p[["end", "filed"]] for p in parts])
            .sort_values("filed").drop_duplicates(subset=["end"], keep="first")
            .sort_values("end").reset_index(drop=True))

    px = None if prices is None else prices.dropna()
    rows = []
    for _, e in ends.iterrows():
        end = e["end"]
        comps = [_asof_value(df, end) for df in (ada, invres, dtva)]
        present = [c for c in comps if c is not None]
        if not present:
            continue
        reserves = float(np.nansum([c if c is not None else np.nan for c in comps]))
        a = _asof_value(assets, end)
        if a is None or a <= 0:
            continue
        c_ = _asof_value(cash, end)
        l_ = _asof_value(liab, end)
        d_ = _asof_value(debt, end)
        if None not in (c_, l_, d_):
            noa = a - c_ - (l_ - d_)
        else:
            noa = np.nan
        cscore = reserves / a
        cscore_noa = (reserves / noa) if (np.isfinite(noa) and noa > 0) else np.nan
        ni_q = _asof_value(ni, end)
        roa = (ni_q / a) if ni_q is not None else np.nan
        # contemporaneous fiscal-quarter stock return (news proxy for Basu asymmetry)
        ret_contemp = np.nan
        if px is not None and len(px) > 5:
            idx = px.index
            p1 = idx.searchsorted(pd.Timestamp(end), side="right") - 1
            p0 = idx.searchsorted(pd.Timestamp(end) - pd.Timedelta(days=95), side="right") - 1
            if 0 <= p0 < p1 < len(px):
                ret_contemp = float(px.iloc[p1] / px.iloc[p0] - 1.0)
        rows.append({"end": end, "filed": e["filed"], "reserves": reserves, "assets": a,
                     "noa": noa, "cscore": cscore, "cscore_noa": cscore_noa,
                     "roa": roa, "ret_contemp": ret_contemp})
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = "2007-01-01", end: str | None = None,
                prices_path: str = PRICES_CACHE, events_path: str = EVENTS_CACHE
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download basket prices + EDGAR reserves/assets/cash/liab/debt/NI, build C-scores, cache.

    Network-only; used once to build the cache. Writes a wide adjusted-close CSV and a long
    event CSV (one row per ticker × reserve-disclosing quarter). Never imported by the offline
    notebook cells.
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
        ada = _instant_series(cik, ADA_CONCEPTS)
        invres = _instant_series(cik, INVRES_CONCEPTS)
        dtva = _instant_series(cik, DTVA_CONCEPTS)
        if len(ada) + len(invres) + len(dtva) < 8:
            continue
        assets = _instant_series(cik, ASSET_CONCEPTS)
        cash = _instant_series(cik, CASH_CONCEPTS)
        liab = _instant_series(cik, LIAB_CONCEPTS)
        debt = _instant_series(cik, DEBT_CONCEPTS)
        ni = _quarterly_flow_series(cik, NI_CONCEPTS)
        sig = build_signal(ada, invres, dtva, assets, cash, liab, debt, ni,
                           prices[tk] if tk in prices.columns else None)
        for _, s in sig.iterrows():
            rows.append({"ticker": tk, **s.to_dict()})
        time.sleep(0.15)
    events = pd.DataFrame(rows).dropna(subset=["filed", "cscore"])
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
    """Long event frame: ticker, end, filed, reserves, assets, noa, cscore, cscore_noa,
    roa, ret_contemp (filings on/before AS_OF)."""
    ev = pd.read_csv(path, parse_dates=["end", "filed"])
    ev = ev[ev["filed"] <= pd.Timestamp(AS_OF)]
    return ev.sort_values(["ticker", "filed"]).reset_index(drop=True)


def load_real() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached prices + events in one call."""
    return load_prices(), load_events()


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_names: int = 30, n_quarters: int = 44, edge: float = 0.0,
                    seed: int = 860, sig_daily: float = 0.020
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic price panel + C-score events with a PLANTED return knob.

    Each name has a daily random-walk price and a persistent latent *conservatism* level (higher
    = more hidden reserves = higher C-score). At each quarterly filing (~63 trading days apart)
    the name emits a C-score = latent + small noise. If ``edge`` != 0, the ~63 sessions of
    forward return following each filing get an *extra* daily drift of ``edge * cscore / 63`` —
    high-conservatism names drift up, low ones drift down, exactly the Penman-Zhang claim. With
    ``edge = 0`` the forward path is pure noise: the long-short must NOT reach significance.

    Returns (prices wide frame, event table) in the same schema as ``load_real``: ``cscore`` is
    the sortable signal. Business-day index kept well below the ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    H = 63
    n_days = n_quarters * H + 200
    idx = pd.bdate_range("2010-01-04", periods=n_days)
    names = [f"N{i:02d}" for i in range(n_names)]

    price_cols = {}
    ev_rows = []
    for name in names:
        latent = float(rng.normal(0.04, 0.02))     # persistent reserve-intensity level (~4%)
        ret = rng.normal(0.0004, sig_daily, size=n_days)
        first = int(rng.integers(80, 140))
        q = first
        while q < n_days - H - 5:
            cscore = latent + rng.normal(0.0, 0.005)
            if edge != 0.0:
                # standardise the reserve-ratio deviation (~2% scale) so ``edge`` plays the same
                # role as in the sibling studies, independent of the tiny absolute C-score level
                z = (cscore - 0.04) / 0.02
                ret[q + 1:q + 1 + H] += edge * z / H
            ev_rows.append({"ticker": name, "_pos": q, "cscore": float(cscore)})
            q += H + int(rng.integers(-4, 5))
        price_cols[name] = 100.0 * np.exp(np.cumsum(ret))

    prices = pd.DataFrame(price_cols, index=idx)
    ev = pd.DataFrame(ev_rows)
    ev["filed"] = [idx[p] for p in ev["_pos"]]
    ev["end"] = ev["filed"] - pd.Timedelta(days=40)
    ev["reserves"] = np.nan
    ev["assets"] = np.nan
    ev["noa"] = np.nan
    ev["cscore_noa"] = ev["cscore"]
    ev["roa"] = np.nan
    ev["ret_contemp"] = np.nan
    ev = ev[["ticker", "end", "filed", "reserves", "assets", "noa", "cscore",
             "cscore_noa", "roa", "ret_contemp"]]
    return prices, ev.sort_values(["ticker", "filed"]).reset_index(drop=True)
