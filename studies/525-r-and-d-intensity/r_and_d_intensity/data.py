"""Data layer for Study 525 — R-And-D-Intensity (Chan-Lakonishok-Sougiannis 2001).

The believers' version (and the academic headline): firms with high **R&D-to-market-
capitalisation** (R&D / ME) earn higher subsequent returns, because the market under-prices the
intangible capital those expensed R&D dollars build. Chan, Lakonishok & Sougiannis (2001) show
the predictability is concentrated in R&D scaled by **market value** (a cheapness / mispricing
signal), and is far weaker when R&D is scaled by **sales** (a pure spending-intensity signal).
This study replicates that exact contrast on a clean survivor basket.

Two tapes, one shape (an annual **fundamentals panel** of R&D + revenue + shares, a monthly
**return panel** + **price panel** over the same names, plus the SPY benchmark):

* **Real tape.** A fixed basket of long-listed US large-caps spanning the R&D spectrum — the
  R&D-heavy (tech / semis / pharma / biotech) *and* the R&D-light (banks, staples, energy,
  retail, telco). For each name we pull annual ``ResearchAndDevelopmentExpense``, revenue and
  shares-outstanding from **SEC EDGAR companyfacts** (public, no key). The two intensity signals
  are formed at each month from data *known* at that month (a one-year reporting lag on
  fundamentals, the *contemporaneous* price for the market-cap denominator):

      rd_me    = R&D(FY Y-1) / market_cap(month)      # the CLS mispricing signal
      rd_sales = R&D(FY Y-1) / revenue(FY Y-1)         # the spending-intensity signal (study 400)

  Monthly **total returns** and **prices** come from yfinance. All cached as parquet under
  ``_cache/``; the offline core never touches the network unless ``fetch=True``.

  The basket is chosen **by sector, not by returns** — explicitly *not* "the firms that won" — so
  the study is not the look-ahead-selection trap of a thematic basket. The residual bias is
  **survivorship** (a current-membership basket; names that delisted are absent), named on the
  Signal axis with its direction reasoned about in writing.

* **Synthetic.** A deterministic, fixed-seed generator with a single planted knob, ``edge``: the
  *true* per-period premium earned per unit of (cross-sectionally de-meaned) R&D/ME. At
  ``edge = 0`` the signal carries **no** information and any long-high/short-low spread is luck —
  the test must NOT manufacture significance. At ``edge > 0`` the high-R&D/ME book genuinely
  out-earns and the harness must recover it. This is the null and the positive control in one.

No network import lives at module top; ``fetch_*`` is called once to build the cache.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

MONTHS_PER_YEAR = 12

# A transparent, fixed basket of long-listed US large-caps spanning the R&D-intensity spectrum.
# Chosen for long history and SECTOR spread (NOT by realised return) so the study isolates the
# intensity tilt, not winner-selection. CIKs are pinned so the EDGAR pull is deterministic.
#   high-R&D end : semis, software, pharma, biotech, hardware
#   low-R&D end  : banks, staples, energy, retail, telco, industrials-lite
BASKET: dict[str, int] = {
    # --- R&D-heavy (high reported R&D) ---
    "AAPL": 320193, "MSFT": 789019, "INTC": 50863, "CSCO": 858877, "ORCL": 1341439,
    "TXN": 97476, "QCOM": 804328, "AMGN": 318154, "GILD": 882095, "PFE": 78003,
    "MRK": 310158, "BMY": 14272, "LLY": 59478, "ADBE": 796343, "IBM": 51143,
    "HON": 773840, "EMR": 32604, "DE": 315189, "MMM": 66740, "GE": 40545,
    # --- R&D-light (low/zero reported R&D) ---
    "JPM": 19617, "BAC": 70858, "WFC": 72971, "C": 831001, "USB": 36104,
    "GS": 886982, "AXP": 4962, "KO": 21344, "PG": 80424, "PEP": 77476,
    "WMT": 104169, "COST": 909832, "HD": 354950, "LOW": 60667, "MCD": 63908,
    "XOM": 34088, "CVX": 93410, "T": 732717, "VZ": 732712, "DUK": 1326160,
}

BENCH = "SPY"      # the market (S&P 500 total-return proxy)

_EDGAR_HEADERS = {"User-Agent": "OpenAlphaLab research guillain@poulpe.us"}
_REV_CONCEPTS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
]
_RD_CONCEPTS = [
    "ResearchAndDevelopmentExpense",
    "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    "ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost",
]
_SHARE_CONCEPTS = [
    "CommonStockSharesOutstanding",
    "EntityCommonStockSharesOutstanding",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
]


# --------------------------------------------------------------------------- #
# Real tape — EDGAR companyfacts (R&D + revenue + shares) and yfinance (price/return)
# --------------------------------------------------------------------------- #
def _companyfacts(cik: int) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    req = urllib.request.Request(url, headers=_EDGAR_HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _annual_from_facts(facts: dict, concepts: list[str], unit: str = "USD") -> dict[int, float]:
    """Map fiscal-year-end year -> full-year USD value for the first matching us-gaap concept.

    Keeps only 10-K, full-period (``fp == 'FY'``) facts whose start->end span is a full year
    (330-400 days), so quarterly and partial-period facts never leak into an annual number.
    A later concept fills only years an earlier concept did not cover.
    """
    g = facts.get("facts", {}).get("us-gaap", {})
    dei = facts.get("facts", {}).get("dei", {})
    out: dict[int, float] = {}
    for c in concepts:
        node = g.get(c) or dei.get(c)
        if not node:
            continue
        for u in node.get("units", {}).get(unit, []):
            if not str(u.get("form", "")).startswith("10-K"):
                continue
            e = u.get("end")
            if not e:
                continue
            ed = dt.date.fromisoformat(e)
            if "start" in u:
                s = dt.date.fromisoformat(u["start"])
                if not (330 <= (ed - s).days <= 400):
                    continue
            out.setdefault(ed.year, float(u["val"]))
    return out


def fetch_fundamentals(basket: dict[str, int] | None = None,
                       cache_dir: str = DEFAULT_CACHE, pause: float = 0.25) -> dict:
    """Build the annual fundamentals panels (rows = fiscal year, cols = ticker) from EDGAR.

    Network-only; used once to build the cache. Three panels are written:
      * ``rd.parquet``     — annual R&D expense (USD); missing floored to 0 at use-time.
      * ``rev.parquet``    — annual revenue (USD).
      * ``shares.parquet`` — shares outstanding at fiscal-year end (for market cap).
    """
    basket = basket or BASKET
    rd_cols, rev_cols, sh_cols = {}, {}, {}
    for tk, cik in basket.items():
        try:
            f = _companyfacts(cik)
        except Exception:
            time.sleep(pause)
            continue
        rev = _annual_from_facts(f, _REV_CONCEPTS)
        rd = _annual_from_facts(f, _RD_CONCEPTS)
        sh = _annual_from_facts(f, _SHARE_CONCEPTS, unit="shares")
        if rev:
            rev_cols[tk] = pd.Series(rev)
        if rd:
            rd_cols[tk] = pd.Series(rd)
        if sh:
            sh_cols[tk] = pd.Series(sh)
        time.sleep(pause)
    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    for name, cols in [("rd", rd_cols), ("rev", rev_cols), ("shares", sh_cols)]:
        panel = pd.DataFrame(cols).sort_index()
        panel.index.name = "fy"
        panel.to_parquet(os.path.join(cache_dir, f"{name}.parquet"))
        out[name] = panel
    return out


def fetch_prices(basket: dict[str, int] | None = None, start: str = "2005-01-01",
                 cache_dir: str = DEFAULT_CACHE) -> dict:
    """Monthly total returns + raw close prices (yfinance auto-adjusted) for the basket + SPY.

    Network-only; used once. Writes ``returns.parquet`` (monthly total return) and
    ``prices.parquet`` (monthly auto-adjusted close, for the market-cap denominator). Drops the
    in-progress final month (a stamped run never holds a partial bar)."""
    import yfinance as yf

    basket = basket or BASKET
    tickers = list(basket.keys()) + [BENCH]
    raw = None
    for attempt in range(4):
        try:
            raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
            if raw is not None and not raw.empty:
                break
        except Exception:
            time.sleep(2.0 * (attempt + 1))
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no price data after retries")
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    monthly = raw.resample("ME").last()
    rets = monthly.pct_change().dropna(how="all")
    cutoff = pd.Timestamp.today().normalize().to_period("M").to_timestamp()
    rets = rets[rets.index < cutoff]
    prices = monthly[monthly.index < cutoff]
    os.makedirs(cache_dir, exist_ok=True)
    rets.to_parquet(os.path.join(cache_dir, "returns.parquet"))
    prices.to_parquet(os.path.join(cache_dir, "prices.parquet"))
    return {"returns": rets, "prices": prices}


def have_real(cache_dir: str = DEFAULT_CACHE) -> bool:
    need = ["rd.parquet", "rev.parquet", "shares.parquet", "returns.parquet", "prices.parquet"]
    return all(os.path.exists(os.path.join(cache_dir, f)) for f in need)


def load_real(cache_dir: str = DEFAULT_CACHE,
              allow_survivorship_bias: bool = False) -> dict:
    """Load the cached real tape as a dict of panels.

    Keys: ``rd`` ``rev`` ``shares`` (annual, rows = fiscal year), ``returns`` ``prices`` (monthly,
    cols = ticker), ``spy`` (monthly SPY total return aligned to ``returns``).

    **Survivorship — named on the Signal axis.** The basket is *current* membership projected
    back: names that delisted are absent. The bias is mild and its *direction is reasoned about
    in writing* — a surviving large-cap basket runs slightly bullish, and since the long leg
    (R&D-heavy tech/pharma) and the short leg (R&D-light banks/staples/energy) are *both*
    survivors, the survivorship tilt is largely common to both legs of the long/short, not a
    clean push on the spread. Opt in with ``allow_survivorship_bias=True``."""
    if not allow_survivorship_bias:
        raise PermissionError(
            "load_real() returns a CURRENT-membership large-cap basket projected back to 2005 "
            "(survivorship: delisted names are absent). Pass allow_survivorship_bias=True to opt "
            "in, and carry the caveat onto the Signal axis."
        )
    rd = pd.read_parquet(os.path.join(cache_dir, "rd.parquet"))
    rev = pd.read_parquet(os.path.join(cache_dir, "rev.parquet"))
    shares = pd.read_parquet(os.path.join(cache_dir, "shares.parquet"))
    rets = pd.read_parquet(os.path.join(cache_dir, "returns.parquet"))
    prices = pd.read_parquet(os.path.join(cache_dir, "prices.parquet"))
    if BENCH in rets.columns:
        spy = rets[BENCH].copy()
        rets = rets[[c for c in rets.columns if c != BENCH]]
    else:                                              # pragma: no cover - cache always has SPY
        spy = pd.Series(0.0, index=rets.index, name=BENCH)
    if BENCH in prices.columns:
        prices = prices[[c for c in prices.columns if c != BENCH]]
    spy.name = "spy"
    return {"rd": rd, "rev": rev, "shares": shares, "returns": rets, "prices": prices, "spy": spy}


# --------------------------------------------------------------------------- #
# Signal construction — R&D/ME (CLS) and R&D/sales (intensity), with a reporting lag
# --------------------------------------------------------------------------- #
def known_annual(panel: pd.DataFrame, year: int, report_lag: int = 1) -> pd.Series:
    """The most-recent annual value *known* at the start of ``year`` (fiscal year <= year-lag)."""
    avail = panel[panel.index <= year - report_lag]
    if avail.empty:
        return pd.Series(dtype=float)
    return avail.ffill().iloc[-1]


def build_signals(real: dict, report_lag: int = 1) -> dict:
    """Build the two monthly cross-sectional signal panels (rows = month, cols = ticker).

    * ``rd_me``    — R&D(FY Y-1) / market_cap(month).  market_cap = price(month) * shares(FY Y-1).
                     The Chan-Lakonishok-Sougiannis mispricing signal (R&D scaled by market value).
    * ``rd_sales`` — R&D(FY Y-1) / revenue(FY Y-1).  The pure spending-intensity signal (study 400).

    R&D missing for a name in a year is floored to 0 (a bank genuinely spends ~0 on R&D), so it
    populates the LOW end of both signals. The market-cap denominator uses the *contemporaneous*
    price (known at the month) but lagged shares (a fundamental) — no look-ahead.
    """
    rd, rev, shares, prices = real["rd"], real["rev"], real["shares"], real["prices"]
    months = prices.index
    cols = list(prices.columns)
    rd_me = pd.DataFrame(index=months, columns=cols, dtype=float)
    rd_sales = pd.DataFrame(index=months, columns=cols, dtype=float)
    for ts in months:
        y = ts.year
        rd_k = known_annual(rd, y, report_lag)
        rev_k = known_annual(rev, y, report_lag)
        sh_k = known_annual(shares, y, report_lag)
        for c in cols:
            r = float(rd_k.get(c, np.nan)) if c in rd_k.index else np.nan
            r = 0.0 if (np.isnan(r) if isinstance(r, float) else False) else r
            v = rev_k.get(c, np.nan) if c in rev_k.index else np.nan
            s = sh_k.get(c, np.nan) if c in sh_k.index else np.nan
            px = prices.at[ts, c]
            mcap = px * s if (pd.notna(px) and pd.notna(s) and s > 0) else np.nan
            if pd.notna(mcap) and mcap > 0:
                rd_me.at[ts, c] = r / mcap
            if pd.notna(v) and v > 0:
                rd_sales.at[ts, c] = max(r, 0.0) / v
    return {"rd_me": rd_me, "rd_sales": rd_sales}


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_years: int = 18, n_assets: int = 40, edge: float = 0.0,
                    market_vol: float = 0.045, idio_vol: float = 0.075,
                    seed: int = 525) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """A deterministic monthly panel whose names carry a *known* R&D/ME premium.

    Each name has a **persistent** R&D/ME score ``x_i`` (high for the first half of the field, low
    for the second — a stable cheap-intangibles-vs-expensive split), drifting mildly month to
    month. The monthly return is::

        r_it = beta_i * mkt_t + (edge / 12) * sign(x_i - median_x) + eps_it

    so ``edge`` is the *true annual* long-high-minus-short-low premium: a high-R&D/ME name earns
    ``+edge/2`` per year and a low one ``-edge/2``, so the long-short spread's expected value is
    exactly ``edge``/yr. The market betas are uniform (= 1), so at ``edge = 0`` the two legs have
    identical market exposure and the long-short carries **no spurious beta spread** — a clean null.

    Returns ``(signal, returns, bench, truth)``: a monthly ``signal`` panel (rd_me proxy), a
    monthly ``returns`` panel, the monthly equal-weight ``bench``, and the planted-parameter dict.
    """
    rng = np.random.default_rng(seed)
    n_months = n_years * MONTHS_PER_YEAR
    half = n_assets // 2

    base_x = np.concatenate([
        rng.uniform(0.10, 0.40, half),                # high R&D/ME (cheap-intangibles) names
        rng.uniform(0.00, 0.05, n_assets - half),     # low R&D/ME names
    ])
    sig = np.clip(base_x[None, :] + rng.normal(0.0, 0.01, size=(n_months, n_assets)), 0.0, 2.0)

    leg_sign = np.where(np.arange(n_assets) < half, +1.0, -1.0)
    premium = leg_sign * (edge / 2.0) / MONTHS_PER_YEAR
    betas = np.ones(n_assets)

    mkt = rng.normal(0.007, market_vol, n_months)
    eps = rng.normal(0.0, idio_vol, size=(n_months, n_assets))
    # Centre each name's idio series to EXACTLY zero time-mean so edge=0 is exactly flat (no seed
    # gets a spurious cross-sectional alpha by luck) — the planted premium is the ONLY source of LS.
    eps -= eps.mean(axis=0, keepdims=True)
    raw = mkt[:, None] * betas[None, :] + premium[None, :] + eps

    # period_range (NOT date_range) for the decorative monthly index — avoids ns overflow on CI.
    midx = pd.period_range("2007-01", periods=n_months, freq="M").to_timestamp()
    midx = pd.DatetimeIndex(midx, name="date")
    cols = [f"N{i:02d}" for i in range(n_assets)]
    returns = pd.DataFrame(raw, index=midx, columns=cols)
    signal = pd.DataFrame(sig, index=midx, columns=cols)
    bench = pd.Series(raw.mean(axis=1), index=midx, name="bench")

    truth = {
        "n_years": n_years, "n_assets": n_assets, "edge": edge,
        "market_vol": market_vol, "idio_vol": idio_vol, "seed": seed,
        "high_cols": cols[: n_assets // 2], "low_cols": cols[n_assets // 2:],
    }
    return signal, returns, bench, truth


def fingerprint(*panels: pd.DataFrame) -> str:
    """A short content fingerprint of the panels, for the as-of stamp."""
    h = hashlib.sha1()
    for p in panels:
        a = np.ascontiguousarray(np.nan_to_num(p.to_numpy(dtype=float), nan=-9.99).ravel())
        h.update(a.tobytes())
    return h.hexdigest()[:12]
