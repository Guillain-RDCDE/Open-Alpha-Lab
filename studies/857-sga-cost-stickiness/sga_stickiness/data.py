"""Data layer for Study 857 — SG&A Cost Stickiness.

The claim under test — **Anderson, Banker & Janakiraman (2003).** Selling, general &
administrative (SG&A) costs are *sticky*: they rise more when sales rise than they fall when
sales fall. Managers, reluctant to fire salespeople or scrap a marketing programme they may
need again next quarter, delay cutting discretionary overhead into a sales decline. ABJ's
canonical regression on annual data,

    Δlog(SG&A) = β₀ + β₁·Δlog(Sales) + β₂·(Decrease · Δlog(Sales)) + ε,

finds β₁ ≈ +0.55 (SG&A rises 0.55% per 1% sales gain) and β₁+β₂ ≈ +0.35 (it falls only 0.35%
per 1% sales *loss*) — so **β₂ < 0 is the signature of stickiness**. We define a firm's
**stickiness = −β₂** (bigger ⇒ stickier ⇒ costs cling harder into a downturn). The trading
hypothesis: a firm whose SG&A stays sticky into a sales decline is running *weaker operating
discipline*, so **sticky-cost firms should under-earn**. We therefore sort on the mirror
signal ``disc = β₂`` (= −stickiness, "cost discipline"; higher ⇒ leaner) so that, if the claim
holds, a long-top / short-bottom spread on ``disc`` is **positive**.

Two sources, both offline-friendly once cached:

* **Real tape.**
  - Daily adjusted closes for a fixed basket of ~33 large US filers that report
    ``SellingGeneralAndAdministrativeExpense`` on EDGAR (yfinance, no key).
  - Per name, the full **quarterly** history of SG&A and **Revenues** from EDGAR's XBRL
    ``companyconcept`` API (``data.sec.gov``), plus ``NetIncomeLoss`` and ``Assets`` for the
    operating-discipline (profitability) axis. Each quarter carries the **filing date** of the
    10-Q/10-K that disclosed it — the day the number became public.
  From the SG&A and revenue series we form, per quarter, the **year-over-year log change** in
  SG&A and in sales and a **sales-decrease dummy**. Then, at *each filing date*, we re-estimate
  the ABJ regression on an **expanding window of only the observations already public** and
  read off β₂ → stickiness. Each event is one (ticker, filing date) row carrying the
  point-in-time stickiness; forward returns / forward profitability are measured strictly
  *after* the filing (no look-ahead).

* **Synthetic.** A deterministic, fixed-seed generator that manufactures quarterly SG&A + sales
  *levels* obeying a per-firm true β₂, runs them through the **same** estimator, and plants a
  forward-return component proportional to the firm's discipline (knob ``edge``). It is the
  positive control: with ``edge = 0`` the long-short must NOT manufacture significance; with a
  large planted ``edge`` it must light up. No network.

Honest about coverage: this is a **thin, uneven, cyclical-tilted** panel. Firm-level β₂ is only
*identified* once a name has actually lived through several year-over-year sales **declines** in
its public window, so steady growers (COST, MCD…) drop out or arrive late, and the estimable
cross-section is dominated by cyclicals (semis, industrials, autos). The point-in-time
expanding-window signal only becomes available ~4-5 years into each name's XBRL history (≈2015).
That is a first-class caveat of the study, not a footnote.

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
PRICES_CACHE = os.path.join(CACHE_DIR, "sga_prices.csv")
EVENTS_CACHE = os.path.join(CACHE_DIR, "sga_events.csv")

UA = "OpenAlphaLab research contact@example.com"

AS_OF = "2026-06-30"        # last complete calendar month at publication
ERA_SPLIT = "2020-07-01"    # pre/post the COVID sales-shock split

# Estimation knobs for the firm-level ABJ regression (point-in-time expanding window).
MIN_OBS = 20                # minimum year-over-year observations before an estimate is trusted
MIN_DEC = 4                 # minimum sales-DECLINE quarters in the window (β₂ must be identified)

# A transparent, fixed basket of large US filers that report a current
# SellingGeneralAndAdministrativeExpense line on EDGAR with deep quarterly history. Chosen
# across retail / staples / industrials / tech-hardware / health / materials / autos to span
# both expansions and the genuine sales *declines* that identify cost stickiness. This is a
# *survivors* basket (all still trading): survivorship is named on the Signal axis — it cannot
# include firms that were acquired or failed, and (worse for a discipline signal) the very
# firms whose lack of cost discipline killed them are exactly the ones missing. The bias runs
# *against* finding that sticky firms under-earn, so it can be argued about, not ignored — see
# docs/references.md.
BASKET = [
    "WMT", "TGT", "COST", "HD", "LOW", "NKE", "MCD", "PEP", "PG", "CL",
    "GIS", "MMM", "HON", "CAT", "DE", "EMR", "GE", "RTX", "AAPL", "HPQ",
    "DELL", "IBM", "TXN", "INTC", "QCOM", "JNJ", "PFE", "MRK", "MDT", "DD",
    "DOW", "LYB", "GM",
]

# SEC CIK (10-digit, zero-padded), resolved once from the SEC ticker map and frozen here so the
# offline path never needs the network.
CIK = {
    "WMT": "0000104169", "TGT": "0000027419", "COST": "0000909832", "HD": "0000354950",
    "LOW": "0000060667", "NKE": "0000320187", "MCD": "0000063908", "PEP": "0000077476",
    "PG": "0000080424", "CL": "0000021665", "GIS": "0000040704", "MMM": "0000066740",
    "HON": "0000773840", "CAT": "0000018230", "DE": "0000315189", "EMR": "0000032604",
    "GE": "0000040545", "RTX": "0000101829", "AAPL": "0000320193", "HPQ": "0000047217",
    "DELL": "0001571996", "IBM": "0000051143", "TXN": "0000097476", "INTC": "0000050863",
    "QCOM": "0000804328", "JNJ": "0000200406", "PFE": "0000078003", "MRK": "0000310158",
    "MDT": "0001613103", "DD": "0001666700", "DOW": "0001751788", "LYB": "0001489393",
    "GM": "0001467858",
}

SGA_CONCEPTS = ("SellingGeneralAndAdministrativeExpense",)
REV_CONCEPTS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)
NI_CONCEPTS = ("NetIncomeLoss",)
ASSET_CONCEPTS = ("Assets",)

EVENT_COLS = [
    "ticker", "end", "filed", "beta1", "beta2", "stickiness", "disc",
    "n_obs", "n_dec", "roa_now", "roa_fwd", "roa_chg",
]


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


def _quarterly_flow_series(cik: str, concepts: tuple[str, ...]) -> pd.DataFrame:
    """Best (longest-history) *quarterly* flow series for one CIK across ``concepts``.

    Flow (duration) us-gaap concepts report a value over a start..end span. We keep only
    ~one-quarter spans (60-100 days), de-duplicate on the period end (keeping the EARLIEST
    filing that disclosed it — first public disclosure, no restatement look-ahead), and pick the
    concept that yields the most distinct quarter ends. Columns: end, filed, val.
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


def _instant_series(cik: str, concepts: tuple[str, ...]) -> pd.DataFrame:
    """Best (longest-history) *instant* balance-sheet series (e.g. Assets) for one CIK.

    Instant concepts report one value AT a period end. Columns: end, filed, val.
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
# Signal construction — firm-level cost stickiness (point-in-time)
# --------------------------------------------------------------------------- #
def yoy_changes(sga: pd.DataFrame, rev: pd.DataFrame,
                lo: int = 300, hi: int = 430) -> pd.DataFrame:
    """Per-quarter year-over-year log changes in SG&A and sales + a sales-decrease dummy.

    For each quarter end E that has a match ~1 year earlier (gap in [lo, hi] days) in *both*
    the SG&A and the revenue series, emit one row:
      * ``dlog_sga`` = log(SG&A_E / SG&A_{E-1yr})
      * ``dlog_rev`` = log(Rev_E  / Rev_{E-1yr})
      * ``dec``      = 1 if Rev_E < Rev_{E-1yr} else 0   (the ABJ decrease indicator)
      * ``filed``    = filing date of the *current* quarter (when the change becomes public)
    Rows needing a non-positive level (log undefined) are dropped. Vectorised YoY matching.
    """
    if sga.empty or rev.empty:
        return pd.DataFrame(columns=["end", "filed", "dlog_sga", "dlog_rev", "dec"])
    s = sga.sort_values("end").reset_index(drop=True)
    r = rev.sort_values("end").reset_index(drop=True)
    s_day = s["end"].to_numpy().astype("datetime64[D]").astype("int64")
    s_val = s["val"].to_numpy(dtype=float)
    r_day = r["end"].to_numpy().astype("datetime64[D]").astype("int64")
    r_val = r["val"].to_numpy(dtype=float)

    def _yoy(day_arr: np.ndarray, val_arr: np.ndarray, end_day: int) -> float:
        gap = end_day - day_arr
        m = (gap >= lo) & (gap <= hi)
        if not m.any():
            return float("nan")
        idx = np.flatnonzero(m)
        k = idx[np.argmin(np.abs(gap[idx] - 365))]
        return float(val_arr[k])

    def _asof(day_arr: np.ndarray, val_arr: np.ndarray, end_day: int, tol: int = 20) -> float:
        gap = np.abs(day_arr - end_day)
        j = int(np.argmin(gap))
        return float(val_arr[j]) if gap[j] <= tol else float("nan")

    rows = []
    for i in range(len(s)):
        end_day = int(s_day[i])
        sga_now = float(s_val[i])
        sga_prior = _yoy(s_day, s_val, end_day)
        rev_now = _asof(r_day, r_val, end_day)
        rev_prior = _yoy(r_day, r_val, end_day)
        if not (np.isfinite(sga_prior) and np.isfinite(rev_now) and np.isfinite(rev_prior)):
            continue
        if sga_now <= 0 or sga_prior <= 0 or rev_now <= 0 or rev_prior <= 0:
            continue
        rows.append({
            "end": s.loc[i, "end"], "filed": s.loc[i, "filed"],
            "dlog_sga": float(np.log(sga_now / sga_prior)),
            "dlog_rev": float(np.log(rev_now / rev_prior)),
            "dec": int(rev_now < rev_prior),
        })
    return pd.DataFrame(rows, columns=["end", "filed", "dlog_sga", "dlog_rev", "dec"])


def estimate_stickiness(window: pd.DataFrame) -> tuple[float, float, int, int]:
    """OLS of the ABJ asymmetric model on a set of YoY observations.

    ``dlog_sga = β₀ + β₁·dlog_rev + β₂·(dec·dlog_rev) + ε``. Returns (β₁, β₂, n, n_dec).
    Returns NaNs when the window is too small OR the decrease interaction is not identified
    (too few decline quarters, or their sales moves carry no variation).
    """
    n = len(window)
    n_dec = int(window["dec"].sum()) if n else 0
    if n < MIN_OBS or n_dec < MIN_DEC:
        return (float("nan"), float("nan"), n, n_dec)
    x = window["dlog_rev"].to_numpy(dtype=float)
    d = window["dec"].to_numpy(dtype=float)
    y = window["dlog_sga"].to_numpy(dtype=float)
    xd = d * x
    # the interaction column must carry variation to identify β₂
    if np.ptp(xd) < 1e-9:
        return (float("nan"), float("nan"), n, n_dec)
    X = np.column_stack([np.ones(n), x, xd])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return (float("nan"), float("nan"), n, n_dec)
    return (float(beta[1]), float(beta[2]), n, n_dec)


def build_stickiness_events(sga: pd.DataFrame, rev: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time expanding-window stickiness, one row per public quarter.

    For each quarter (ordered by filing date F), estimate the ABJ regression on **only the YoY
    observations already public at F** (an expanding window), read β₂ → stickiness = −β₂ and the
    tradeable discipline signal ``disc`` = β₂ (higher ⇒ leaner). Rows before the estimate is
    identified (too few obs / declines) are dropped. Columns: end, filed, beta1, beta2,
    stickiness, disc, n_obs, n_dec.
    """
    yc = yoy_changes(sga, rev)
    if yc.empty:
        return pd.DataFrame(columns=["end", "filed", "beta1", "beta2",
                                     "stickiness", "disc", "n_obs", "n_dec"])
    yc = yc.sort_values("filed").reset_index(drop=True)
    rows = []
    for i in range(len(yc)):
        end, filed = yc.loc[i, "end"], yc.loc[i, "filed"]
        window = yc[yc["filed"] <= filed]
        b1, b2, n, n_dec = estimate_stickiness(window)
        if not np.isfinite(b2):
            continue
        rows.append({"end": end, "filed": filed, "beta1": b1, "beta2": b2,
                     "stickiness": -b2, "disc": b2, "n_obs": n, "n_dec": n_dec})
    return pd.DataFrame(rows, columns=["end", "filed", "beta1", "beta2",
                                       "stickiness", "disc", "n_obs", "n_dec"])


def _roa_series(ni: pd.DataFrame, assets: pd.DataFrame) -> pd.DataFrame:
    """Trailing-four-quarter ROA per quarter end: TTM NetIncome / Assets. Columns: end, roa."""
    if ni.empty or assets.empty:
        return pd.DataFrame(columns=["end", "roa"])
    n = ni.sort_values("end").reset_index(drop=True)
    ttm = n["val"].rolling(4).sum()
    out = []
    a = assets.sort_values("end").reset_index(drop=True)
    for i in range(len(n)):
        if not np.isfinite(ttm.iloc[i]):
            continue
        end = n.loc[i, "end"]
        gap = (a["end"] - end).abs().dt.days
        if gap.min() > 20:
            continue
        av = float(a.loc[gap.idxmin(), "val"])
        if av > 0:
            out.append({"end": end, "roa": float(ttm.iloc[i]) / av})
    return pd.DataFrame(out, columns=["end", "roa"])


def attach_profitability(events: pd.DataFrame, ni: pd.DataFrame, assets: pd.DataFrame,
                         fwd_quarters: int = 4) -> pd.DataFrame:
    """Add roa_now, roa_fwd (~``fwd_quarters`` ahead) and roa_chg to a stickiness-event frame.

    The operating-discipline outcome: does today's stickiness predict *future* ROA? roa_fwd is
    the trailing-four-quarter ROA whose period end is ~``fwd_quarters`` quarters after the event
    end (gap in a tolerance band); roa_chg = roa_fwd − roa_now.
    """
    ev = events.copy()
    for c in ("roa_now", "roa_fwd", "roa_chg"):
        ev[c] = np.nan
    roa = _roa_series(ni, assets)
    if roa.empty or ev.empty:
        return ev
    lo = 90 * fwd_quarters - 45
    hi = 90 * fwd_quarters + 55
    for i in ev.index:
        end = ev.loc[i, "end"]
        gnow = (roa["end"] - end).abs().dt.days
        if gnow.min() <= 20:
            ev.loc[i, "roa_now"] = float(roa.loc[gnow.idxmin(), "roa"])
        fgap = (roa["end"] - end).dt.days
        mf = (fgap >= lo) & (fgap <= hi)
        if mf.any():
            sub = roa.loc[mf]
            k = (sub["end"].map(lambda e: abs((e - end).days - 90 * fwd_quarters))).idxmin()
            ev.loc[i, "roa_fwd"] = float(roa.loc[k, "roa"])
    ev["roa_chg"] = ev["roa_fwd"] - ev["roa_now"]
    return ev


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = "2005-01-01", end: str | None = None,
                prices_path: str = PRICES_CACHE, events_path: str = EVENTS_CACHE
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download basket prices + EDGAR SG&A / revenue / net income / assets, build events, cache.

    Network-only; used once to build the cache. Writes a wide adjusted-close CSV and a long
    event CSV (one row per ticker × public quarter carrying the point-in-time stickiness).
    Never imported by the offline notebook cells.
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
        sga = _quarterly_flow_series(cik, SGA_CONCEPTS)
        rev = _quarterly_flow_series(cik, REV_CONCEPTS)
        if len(sga) < MIN_OBS or len(rev) < MIN_OBS:
            time.sleep(0.15)
            continue
        ev = build_stickiness_events(sga, rev)
        if ev.empty:
            time.sleep(0.15)
            continue
        ni = _quarterly_flow_series(cik, NI_CONCEPTS)
        assets = _instant_series(cik, ASSET_CONCEPTS)
        ev = attach_profitability(ev, ni, assets)
        ev.insert(0, "ticker", tk)
        rows.append(ev)
        time.sleep(0.15)
    events = (pd.concat(rows, ignore_index=True) if rows
              else pd.DataFrame(columns=EVENT_COLS))
    events = events.dropna(subset=["filed", "disc"]).sort_values(["ticker", "filed"])
    events = events.reset_index(drop=True)
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
def _synthetic_firm(rng: np.random.Generator, beta2_true: float, n_quarters: int,
                    beta1_true: float = 0.60, sga_noise: float = 0.03
                    ) -> pd.DataFrame:
    """Quarterly SG&A + revenue *levels* for one firm obeying a true β₂ (for the estimator).

    Revenue log-level follows a random walk with occasional contractions so that year-over-year
    declines actually occur (β₂ must be identifiable). SG&A log-level is built so that the YoY
    log change satisfies the ABJ model with the given (β₁, β₂). Returns a frame the *same
    estimator* consumes: columns end, filed, sga_val, rev_val.
    """
    # revenue log path with negative-drift bursts to guarantee YoY declines
    steps = rng.normal(0.010, 0.045, n_quarters)
    burst = rng.random(n_quarters) < 0.22
    steps[burst] -= rng.uniform(0.06, 0.16, burst.sum())
    r = np.cumsum(steps) + 6.0                       # log-revenue level
    s = np.empty(n_quarters)
    s[:4] = r[:4] - 0.9                              # seed 4 quarters of log-SG&A
    for t in range(4, n_quarters):
        g = r[t] - r[t - 4]                          # YoY log change in revenue
        dec = 1.0 if g < 0 else 0.0
        h = beta1_true * g + beta2_true * dec * g + rng.normal(0.0, sga_noise)
        s[t] = s[t - 4] + h
    ends = pd.period_range("2010Q1", periods=n_quarters, freq="Q")
    end_ts = pd.to_datetime([p.end_time.normalize() for p in ends])
    filed = end_ts + pd.Timedelta(days=40)
    return pd.DataFrame({"end": end_ts, "filed": filed,
                         "sga_val": np.exp(s) * 1e6, "rev_val": np.exp(r) * 1e6})


def synthetic_panel(n_names: int = 24, n_quarters: int = 52, edge: float = 0.0,
                    seed: int = 857, sig_daily: float = 0.018
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic price panel + stickiness events, planted forward-return knob, no network.

    Each firm gets a true β₂ (⇒ true stickiness / discipline). We manufacture its SG&A + sales
    levels, run them through the **real** ``build_stickiness_events`` estimator (so the whole
    pipeline is exercised), and build a daily price whose post-filing forward window carries an
    extra drift ``edge · disc_true / H`` — high-discipline (lean, β₂≈0) names drift up, sticky
    (β₂≪0) names drift down: exactly "sticky firms under-earn". With ``edge = 0`` the forward
    path is pure noise, so the long-short must NOT reach significance.

    Returns (prices wide frame, event table) in the same shape as ``load_real``.
    """
    rng = np.random.default_rng(seed)
    H = 63
    n_days = n_quarters * H + 260
    idx = pd.bdate_range("2010-01-04", periods=n_days)
    names = [f"N{i:02d}" for i in range(n_names)]

    price_cols = {}
    ev_frames = []
    for name in names:
        beta2_true = float(rng.uniform(-0.45, 0.05))        # more negative ⇒ stickier
        disc_true = beta2_true                              # discipline = β₂ (= −stickiness)
        fund = _synthetic_firm(rng, beta2_true, n_quarters)
        ev = build_stickiness_events(
            fund.rename(columns={"sga_val": "val"})[["end", "filed", "val"]],
            fund.rename(columns={"rev_val": "val"})[["end", "filed", "val"]],
        )
        # daily price with a planted post-filing drift proportional to true discipline
        ret = rng.normal(0.0003, sig_daily, size=n_days)
        for f in ev["filed"]:
            pos = idx.searchsorted(pd.Timestamp(f), side="left")
            if edge != 0.0 and 0 < pos < n_days - H - 2:
                ret[pos + 1:pos + 1 + H] += edge * disc_true / H
        price_cols[name] = 100.0 * np.exp(np.cumsum(ret))
        if not ev.empty:
            ev.insert(0, "ticker", name)
            for c in ("roa_now", "roa_fwd", "roa_chg"):
                ev[c] = np.nan
            ev_frames.append(ev)

    prices = pd.DataFrame(price_cols, index=idx)
    events = (pd.concat(ev_frames, ignore_index=True) if ev_frames
              else pd.DataFrame(columns=EVENT_COLS))
    events = events.sort_values(["ticker", "filed"]).reset_index(drop=True)
    return prices, events
