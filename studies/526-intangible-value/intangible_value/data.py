"""Data layer for Study 526 — Intangible-Value (Lev-Srivastava intangibles-adjusted book-to-market).

The believers' version (and the academic headline): plain **book-to-market** (B/M) value
investing has decayed because GAAP *expenses* intangible-building outlays (R&D, a share of SG&A).
Reported book equity therefore understates the true capital of intangible-heavy firms, polluting
the value sort. Lev & Srivastava (2019), Eisfeldt & Papanikolaou (2013) and Peters & Taylor (2017)
propose capitalising those outlays into an **intangible-adjusted book**, which should sharpen the
value sort. This study builds that adjustment on a clean survivor basket and races the
intangible-adjusted-B/M long-short against the *plain* B/M long-short.

Two tapes, one shape (an annual **fundamentals panel** of book equity + R&D + SG&A + shares, a
monthly **return panel** + **price panel** over the same names, plus the SPY benchmark):

* **Real tape.** A fixed basket of long-listed US large-caps spanning the intangible spectrum — the
  intangible-heavy (tech / software / pharma / consumer-brand) *and* the intangible-light (banks,
  energy, utilities, capital-goods). For each name we pull annual ``StockholdersEquity``,
  ``ResearchAndDevelopmentExpense``, ``SellingGeneralAndAdministrativeExpense`` and
  shares-outstanding from **SEC EDGAR companyfacts** (public, no key). Two B/M signals are formed at
  each month from data *known* at that month (a one-year reporting lag on fundamentals, the
  *contemporaneous* price for the market-cap denominator):

      bm_plain   = book_equity(FY Y-1) / market_cap(month)
      bm_intan   = (book_equity + KnowledgeCapital + OrganisationCapital)(FY Y-1) / market_cap(month)

  where ``KnowledgeCapital`` is a perpetual-inventory capitalisation of past R&D (5-yr amortisation)
  and ``OrganisationCapital`` capitalises 30% of past SG&A (3-yr amortisation), following the
  Eisfeldt-Papanikolaou / Peters-Taylor conventions. Monthly **total returns** and **prices** come
  from yfinance. All cached as parquet under ``_cache/``; the offline core never touches the network
  unless ``fetch=True``.

  The basket is chosen **by sector, not by returns** — explicitly *not* "the firms that won" — so
  the study is not the look-ahead-selection trap of a thematic basket. The residual bias is
  **survivorship** (a current-membership basket; names that delisted are absent), named on the
  Signal axis with its direction reasoned about in writing.

* **Synthetic.** A deterministic, fixed-seed generator with a single planted knob, ``edge``: the
  *true* per-period premium earned per unit of (cross-sectionally de-meaned) adjusted B/M. At
  ``edge = 0`` the signal carries **no** information and any long-cheap/short-expensive spread is
  luck — the test must NOT manufacture significance. At ``edge > 0`` the high-adjusted-B/M (cheap)
  book genuinely out-earns and the harness must recover it. This is the null and the positive
  control in one.

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

# Capitalisation conventions (Eisfeldt-Papanikolaou 2013 / Peters-Taylor 2017 / Lev-Radhakrishnan).
RD_AMORT_YEARS = 5          # knowledge capital: R&D amortised straight-line over 5 years
SGA_CAP_FRACTION = 0.30     # 30% of SG&A is investment in organisation capital
SGA_AMORT_YEARS = 3         # organisation capital amortised over 3 years

# A transparent, fixed basket of long-listed US large-caps spanning the intangible-intensity
# spectrum. Chosen for long history and SECTOR spread (NOT by realised return) so the study isolates
# the value/intangible tilt, not winner-selection. CIKs are pinned so the EDGAR pull is
# deterministic. (Same survivor basket family as study 525.)
#   intangible-heavy : software, semis, pharma, biotech, consumer-brand
#   intangible-light : banks, energy, utilities, capital-goods, telco, retail
BASKET: dict[str, int] = {
    # --- intangible-heavy (large reported R&D + brand/SG&A) ---
    "AAPL": 320193, "MSFT": 789019, "INTC": 50863, "CSCO": 858877, "ORCL": 1341439,
    "TXN": 97476, "QCOM": 804328, "AMGN": 318154, "GILD": 882095, "PFE": 78003,
    "MRK": 310158, "BMY": 14272, "LLY": 59478, "ADBE": 796343, "IBM": 51143,
    "KO": 21344, "PG": 80424, "PEP": 77476, "MCD": 63908, "DIS": 1744489,
    # --- intangible-light (low/zero reported R&D, asset-heavy) ---
    "JPM": 19617, "BAC": 70858, "WFC": 72971, "C": 831001, "USB": 36104,
    "GS": 886982, "AXP": 4962, "XOM": 34088, "CVX": 93410, "COP": 1163165,
    "WMT": 104169, "COST": 909832, "HD": 354950, "LOW": 60667, "TGT": 27419,
    "T": 732717, "VZ": 732712, "DUK": 1326160, "SO": 92122, "NEE": 753308,
}

BENCH = "SPY"      # the market (S&P 500 total-return proxy)

_EDGAR_HEADERS = {"User-Agent": "OpenAlphaLab research guillain@poulpe.us"}
_EQUITY_CONCEPTS = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "CommonStockholdersEquity",
]
_RD_CONCEPTS = [
    "ResearchAndDevelopmentExpense",
    "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    "ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost",
]
_SGA_CONCEPTS = [
    "SellingGeneralAndAdministrativeExpense",
    "GeneralAndAdministrativeExpense",
    "SellingGeneralAndAdministrativeExpenses",
]
_SHARE_CONCEPTS = [
    "CommonStockSharesOutstanding",
    "EntityCommonStockSharesOutstanding",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
]


# --------------------------------------------------------------------------- #
# Real tape — EDGAR companyfacts (equity + R&D + SG&A + shares), yfinance (price/return)
# --------------------------------------------------------------------------- #
def _companyfacts(cik: int) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    req = urllib.request.Request(url, headers=_EDGAR_HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _annual_from_facts(facts: dict, concepts: list[str], unit: str = "USD",
                       flow: bool = True) -> dict[int, float]:
    """Map fiscal-year-end year -> USD value for the first matching us-gaap/dei concept.

    Keeps only 10-K facts. For a *flow* concept (income-statement item: R&D, SG&A) we require a
    full-year start->end span (330-400 days) so no quarterly fact leaks in. For a *stock* concept
    (balance-sheet item: equity, shares) there is no ``start``; we keep the fiscal-year-end value.
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
            if flow:
                if "start" not in u:
                    continue
                s = dt.date.fromisoformat(u["start"])
                if not (330 <= (ed - s).days <= 400):
                    continue
            out.setdefault(ed.year, float(u["val"]))
    return out


def fetch_fundamentals(basket: dict[str, int] | None = None,
                       cache_dir: str = DEFAULT_CACHE, pause: float = 0.25) -> dict:
    """Build the annual fundamentals panels (rows = fiscal year, cols = ticker) from EDGAR.

    Network-only; used once to build the cache. Four panels are written:
      * ``equity.parquet`` — common stockholders' equity (USD), the plain *book*.
      * ``rd.parquet``     — annual R&D expense (USD); missing floored to 0 at use-time.
      * ``sga.parquet``    — annual SG&A expense (USD); missing floored to 0 at use-time.
      * ``shares.parquet`` — shares outstanding at fiscal-year end (for market cap).
    """
    basket = basket or BASKET
    eq_cols, rd_cols, sga_cols, sh_cols = {}, {}, {}, {}
    for tk, cik in basket.items():
        try:
            f = _companyfacts(cik)
        except Exception:
            time.sleep(pause)
            continue
        eq = _annual_from_facts(f, _EQUITY_CONCEPTS, flow=False)
        rd = _annual_from_facts(f, _RD_CONCEPTS, flow=True)
        sga = _annual_from_facts(f, _SGA_CONCEPTS, flow=True)
        sh = _annual_from_facts(f, _SHARE_CONCEPTS, unit="shares", flow=False)
        if eq:
            eq_cols[tk] = pd.Series(eq)
        if rd:
            rd_cols[tk] = pd.Series(rd)
        if sga:
            sga_cols[tk] = pd.Series(sga)
        if sh:
            sh_cols[tk] = pd.Series(sh)
        time.sleep(pause)
    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    for name, cols in [("equity", eq_cols), ("rd", rd_cols), ("sga", sga_cols), ("shares", sh_cols)]:
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
    need = ["equity.parquet", "rd.parquet", "sga.parquet", "shares.parquet",
            "returns.parquet", "prices.parquet"]
    return all(os.path.exists(os.path.join(cache_dir, f)) for f in need)


def load_real(cache_dir: str = DEFAULT_CACHE,
              allow_survivorship_bias: bool = False) -> dict:
    """Load the cached real tape as a dict of panels.

    Keys: ``equity`` ``rd`` ``sga`` ``shares`` (annual, rows = fiscal year), ``returns`` ``prices``
    (monthly, cols = ticker), ``spy`` (monthly SPY total return aligned to ``returns``).

    **Survivorship — named on the Signal axis.** The basket is *current* membership projected back:
    names that delisted are absent. The bias is mild and its *direction is reasoned about in
    writing* — a surviving large-cap basket runs slightly bullish, and since the long (cheap) leg
    and the short (expensive) leg are *both* drawn from survivors, the survivorship tilt is largely
    common to both legs of the long/short, not a clean push on the spread. Opt in with
    ``allow_survivorship_bias=True``."""
    if not allow_survivorship_bias:
        raise PermissionError(
            "load_real() returns a CURRENT-membership large-cap basket projected back to 2005 "
            "(survivorship: delisted names are absent). Pass allow_survivorship_bias=True to opt "
            "in, and carry the caveat onto the Signal axis."
        )
    equity = pd.read_parquet(os.path.join(cache_dir, "equity.parquet"))
    rd = pd.read_parquet(os.path.join(cache_dir, "rd.parquet"))
    sga = pd.read_parquet(os.path.join(cache_dir, "sga.parquet"))
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
    return {"equity": equity, "rd": rd, "sga": sga, "shares": shares,
            "returns": rets, "prices": prices, "spy": spy}


# --------------------------------------------------------------------------- #
# Intangible capitalisation — perpetual inventory (no look-ahead: only past flows)
# --------------------------------------------------------------------------- #
def capitalise_flow(flow: pd.Series, amort_years: int) -> pd.Series:
    """Perpetual-inventory stock from an annual expense flow, straight-line amortisation.

    The stock at fiscal year ``Y`` capitalises the current and previous ``amort_years - 1`` years'
    flows, each amortised straight-line:

        stock_Y = sum_{k=0..amort_years-1} flow_{Y-k} * (amort_years - k) / amort_years

    So this year's full flow counts, last year's at (L-1)/L, etc. Uses only flows *up to* year Y —
    no look-ahead. Missing/NaN flows are treated as 0 (a bank genuinely spends ~0 on R&D). Returns a
    Series indexed by the same fiscal years as ``flow``.
    """
    f = flow.astype(float).sort_index()
    years = f.index
    f0 = f.fillna(0.0)
    out = {}
    L = amort_years
    for y in years:
        s = 0.0
        for k in range(L):
            yk = y - k
            if yk in f0.index:
                s += float(f0.loc[yk]) * (L - k) / L
        out[y] = s
    return pd.Series(out).sort_index()


def intangible_capital(rd: pd.DataFrame, sga: pd.DataFrame) -> pd.DataFrame:
    """Per-name intangible capital stock (knowledge + organisation), annual panel.

    ``knowledge_capital`` = capitalise R&D over ``RD_AMORT_YEARS`` years.
    ``organisation_capital`` = capitalise ``SGA_CAP_FRACTION`` of SG&A over ``SGA_AMORT_YEARS``.
    Returns a (fiscal-year × ticker) panel = knowledge + organisation. Names with no R&D and no SG&A
    contribute 0 — they sit at the intangible-light end (book unchanged by the adjustment).
    """
    cols = sorted(set(rd.columns) | set(sga.columns))
    years = sorted(set(rd.index) | set(sga.index))
    out = pd.DataFrame(index=pd.Index(years, name="fy"), columns=cols, dtype=float)
    for c in cols:
        rd_c = rd[c] if c in rd.columns else pd.Series(dtype=float)
        sga_c = sga[c] if c in sga.columns else pd.Series(dtype=float)
        rd_c = rd_c.reindex(years)
        sga_c = sga_c.reindex(years)
        know = capitalise_flow(rd_c, RD_AMORT_YEARS)
        org = capitalise_flow(sga_c * SGA_CAP_FRACTION, SGA_AMORT_YEARS)
        out[c] = (know.reindex(years).fillna(0.0) + org.reindex(years).fillna(0.0)).values
    return out


# --------------------------------------------------------------------------- #
# Signal construction — plain B/M and intangible-adjusted B/M, with a reporting lag
# --------------------------------------------------------------------------- #
def known_annual(panel: pd.DataFrame, year: int, report_lag: int = 1) -> pd.Series:
    """The most-recent annual value *known* at the start of ``year`` (fiscal year <= year-lag)."""
    avail = panel[panel.index <= year - report_lag]
    if avail.empty:
        return pd.Series(dtype=float)
    return avail.ffill().iloc[-1]


def build_signals(real: dict, report_lag: int = 1) -> dict:
    """Build the two monthly cross-sectional B/M signal panels (rows = month, cols = ticker).

    * ``bm_plain`` — book_equity(FY Y-1) / market_cap(month).  The textbook value signal.
    * ``bm_intan`` — (book_equity + intangible_capital)(FY Y-1) / market_cap(month).  The
                     Lev-Srivastava intangibles-adjusted value signal.

    market_cap = price(month) * shares(FY Y-1). The market-cap denominator uses the *contemporaneous*
    price (known at the month) but lagged shares/fundamentals — no look-ahead. Names whose adjusted
    book turns non-positive are dropped from that month's adjusted signal (a negative-equity firm is
    not a value name in the usual sense).
    """
    equity, rd, sga, shares, prices = (real["equity"], real["rd"], real["sga"],
                                       real["shares"], real["prices"])
    intan = intangible_capital(rd, sga)
    months = prices.index
    cols = list(prices.columns)
    bm_plain = pd.DataFrame(index=months, columns=cols, dtype=float)
    bm_intan = pd.DataFrame(index=months, columns=cols, dtype=float)
    for ts in months:
        y = ts.year
        eq_k = known_annual(equity, y, report_lag)
        in_k = known_annual(intan, y, report_lag)
        sh_k = known_annual(shares, y, report_lag)
        for c in cols:
            be = eq_k.get(c, np.nan) if c in eq_k.index else np.nan
            ic = in_k.get(c, np.nan) if c in in_k.index else np.nan
            ic = 0.0 if (isinstance(ic, float) and np.isnan(ic)) else ic
            s = sh_k.get(c, np.nan) if c in sh_k.index else np.nan
            px = prices.at[ts, c]
            mcap = px * s if (pd.notna(px) and pd.notna(s) and s > 0) else np.nan
            if pd.notna(mcap) and mcap > 0 and pd.notna(be) and be > 0:
                bm_plain.at[ts, c] = be / mcap
            adj_book = (be + ic) if pd.notna(be) else np.nan
            if pd.notna(mcap) and mcap > 0 and pd.notna(adj_book) and adj_book > 0:
                bm_intan.at[ts, c] = adj_book / mcap
    return {"bm_plain": bm_plain, "bm_intan": bm_intan, "intan_capital": intan}


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_years: int = 18, n_assets: int = 40, edge: float = 0.0,
                    market_vol: float = 0.045, idio_vol: float = 0.075,
                    seed: int = 526) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """A deterministic monthly panel whose names carry a *known* value (high-B/M) premium.

    Each name has a **persistent** adjusted-B/M score ``x_i`` (high for the first half of the field
    — the *cheap* value names — low for the second — the *expensive* growth names — a stable
    value-vs-growth split), drifting mildly month to month. The monthly return is::

        r_it = beta_i * mkt_t + (edge / 12) * sign(x_i - median_x) + eps_it

    so ``edge`` is the *true annual* long-cheap-minus-short-expensive premium: a high-B/M name earns
    ``+edge/2`` per year and a low one ``-edge/2``, so the long-short spread's expected value is
    exactly ``edge``/yr. The market betas are uniform (= 1), so at ``edge = 0`` the two legs have
    identical market exposure and the long-short carries **no spurious beta spread** — a clean null.

    Returns ``(signal, returns, bench, truth)``: a monthly ``signal`` panel (adjusted-B/M proxy), a
    monthly ``returns`` panel, the monthly equal-weight ``bench``, and the planted-parameter dict.
    """
    rng = np.random.default_rng(seed)
    n_months = n_years * MONTHS_PER_YEAR
    half = n_assets // 2

    base_x = np.concatenate([
        rng.uniform(0.60, 1.40, half),                # high B/M (cheap value) names
        rng.uniform(0.10, 0.40, n_assets - half),     # low B/M (expensive growth) names
    ])
    sig = np.clip(base_x[None, :] + rng.normal(0.0, 0.02, size=(n_months, n_assets)), 0.0, 5.0)

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
        "cheap_cols": cols[: n_assets // 2], "expensive_cols": cols[n_assets // 2:],
    }
    return signal, returns, bench, truth


def fingerprint(*panels: pd.DataFrame) -> str:
    """A short content fingerprint of the panels, for the as-of stamp."""
    h = hashlib.sha1()
    for p in panels:
        a = np.ascontiguousarray(np.nan_to_num(p.to_numpy(dtype=float), nan=-9.99).ravel())
        h.update(a.tobytes())
    return h.hexdigest()[:12]
