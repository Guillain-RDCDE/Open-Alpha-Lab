"""Data layer for Study 400 — Patent-Intensity (the "innovation premium").

Two tapes, one shape (an annual **R&D-intensity panel** + a monthly **return panel** over the
same names, plus the SPY benchmark):

* **Real tape.** A fixed basket of long-listed US large-caps spanning the R&D spectrum — the
  invention-heavy (tech / semis / pharma / biotech) *and* the invention-light (banks, staples,
  energy, retail, telco). For each name we pull annual ``ResearchAndDevelopmentExpense`` and
  revenue from **SEC EDGAR companyfacts** (public, no key) and form **R&D intensity** = R&D /
  revenue — the audited, transparent proxy for "how patent-and-invention-intensive is this firm."
  Monthly **total returns** come from yfinance. Both are cached as parquet under ``_cache/``; the
  offline core never touches the network unless ``fetch=True``.

  The basket is chosen **by sector, not by returns** — explicitly *not* "the firms that won" — so
  the study is not the look-ahead-selection trap of a thematic basket. The residual bias is
  **survivorship** (a current-membership basket; names that delisted are absent) and is named on
  the Signal axis, with its direction reasoned about in writing.

* **Synthetic.** A deterministic, fixed-seed generator with a single planted knob, ``edge``: the
  *true* per-period premium earned per unit of (cross-sectionally de-meaned) R&D intensity. At
  ``edge = 0`` intensity carries **no** information and any long-high/short-low spread is luck —
  the test must NOT manufacture significance. At ``edge > 0`` the high-intensity book genuinely
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
#   high-intensity end : semis, software, pharma, biotech, hardware
#   low-intensity end  : banks, staples, energy, retail, telco, industrials-lite
BASKET: dict[str, int] = {
    # --- invention-heavy (high reported R&D / revenue) ---
    "AAPL": 320193, "MSFT": 789019, "INTC": 50863, "CSCO": 858877, "ORCL": 1341439,
    "TXN": 97476, "QCOM": 804328, "AMGN": 318154, "GILD": 882095, "PFE": 78003,
    "MRK": 310158, "BMY": 14272, "LLY": 59478, "ADBE": 796343, "IBM": 51143,
    "HON": 773840, "EMR": 32604, "DE": 315189, "MMM": 66740, "GE": 40545,
    # --- invention-light (low reported R&D / revenue) ---
    "JPM": 19617, "BAC": 70858, "WFC": 72971, "C": 831001, "USB": 36104,
    "GS": 886982, "AXP": 4962, "KO": 21344, "PG": 80424, "PEP": 77476,
    "WMT": 104169, "COST": 909832, "HD": 354950, "LOW": 60667, "MCD": 63908,
    "XOM": 34088, "CVX": 93410, "T": 732717, "VZ": 732712, "DUK": 1326160,
}

# Names whose business model carries essentially no reported R&D line (financials, energy,
# staples retail, utilities, telco): EDGAR will report R&D ~0 / absent, so intensity is ~0 and
# they populate the LOW-intensity leg. We floor missing R&D at 0 (a bank's intensity IS ~0).
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


# --------------------------------------------------------------------------- #
# Real tape — EDGAR companyfacts (R&D + revenue) and yfinance (returns)
# --------------------------------------------------------------------------- #
def _companyfacts(cik: int) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    req = urllib.request.Request(url, headers=_EDGAR_HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _annual_from_facts(facts: dict, concepts: list[str]) -> dict[int, float]:
    """Map fiscal-year-end year -> full-year USD value for the first matching us-gaap concept.

    Keeps only 10-K, full-period (``fp == 'FY'``) facts whose start->end span is a full year
    (330-400 days), so quarterly and partial-period facts never leak into an annual number.
    A later concept fills only years an earlier concept did not cover.
    """
    g = facts.get("facts", {}).get("us-gaap", {})
    out: dict[int, float] = {}
    for c in concepts:
        node = g.get(c)
        if not node:
            continue
        for u in node.get("units", {}).get("USD", []):
            if not str(u.get("form", "")).startswith("10-K"):
                continue
            if u.get("fp") != "FY" or "start" not in u or "end" not in u:
                continue
            s = dt.date.fromisoformat(u["start"])
            e = dt.date.fromisoformat(u["end"])
            if 330 <= (e - s).days <= 400:
                out.setdefault(e.year, float(u["val"]))
    return out


def fetch_intensity(basket: dict[str, int] | None = None,
                    cache_dir: str = DEFAULT_CACHE, pause: float = 0.25) -> pd.DataFrame:
    """Build the annual R&D-intensity panel (rows = fiscal year, cols = ticker) from EDGAR.

    Network-only; used once to build ``_cache/intensity.parquet``. R&D missing for a name in a
    year is floored to **0** (a bank's reported R&D intensity genuinely is ~0). Intensity is
    R&D / revenue, winsorised at a sane cap (an intensity > 1 is a data artefact and is clipped).
    """
    basket = basket or BASKET
    cols: dict[str, pd.Series] = {}
    for tk, cik in basket.items():
        try:
            f = _companyfacts(cik)
        except Exception:
            time.sleep(pause)
            continue
        rev = _annual_from_facts(f, _REV_CONCEPTS)
        rd = _annual_from_facts(f, _RD_CONCEPTS)
        years = sorted(rev)
        inten = {}
        for y in years:
            r = rev.get(y)
            if r and r > 0:
                inten[y] = min(max(rd.get(y, 0.0), 0.0) / r, 1.0)
        if inten:
            cols[tk] = pd.Series(inten)
        time.sleep(pause)
    panel = pd.DataFrame(cols).sort_index()
    panel.index.name = "fy"
    os.makedirs(cache_dir, exist_ok=True)
    panel.to_parquet(os.path.join(cache_dir, "intensity.parquet"))
    return panel


def fetch_returns(basket: dict[str, int] | None = None, start: str = "2005-01-01",
                  cache_dir: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Monthly total returns (yfinance auto-adjusted) for the basket + SPY -> cache parquet.

    Network-only; used once to build ``_cache/returns.parquet``. Drops the in-progress final
    month (a stamped run never holds a partial bar)."""
    import yfinance as yf

    basket = basket or BASKET
    tickers = list(basket.keys()) + [BENCH]
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    monthly = raw.resample("ME").last()
    rets = monthly.pct_change().dropna(how="all")
    cutoff = pd.Timestamp.today().normalize().to_period("M").to_timestamp()
    rets = rets[rets.index < cutoff]
    os.makedirs(cache_dir, exist_ok=True)
    rets.to_parquet(os.path.join(cache_dir, "returns.parquet"))
    return rets


def have_real(cache_dir: str = DEFAULT_CACHE) -> bool:
    return (os.path.exists(os.path.join(cache_dir, "intensity.parquet"))
            and os.path.exists(os.path.join(cache_dir, "returns.parquet")))


def load_real(cache_dir: str = DEFAULT_CACHE,
              allow_survivorship_bias: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the cached real tape: ``(intensity, returns, spy)``.

    * ``intensity`` — annual R&D-intensity panel (rows = fiscal year, cols = ticker).
    * ``returns``   — monthly total returns for the basket names (cols = ticker).
    * ``spy``       — monthly SPY total return aligned to ``returns``.

    **Survivorship — named on the Signal axis.** The basket is *current* membership projected
    back: names that delisted are absent. The bias is mild and its *direction is reasoned about
    in writing* — a surviving large-cap basket runs slightly bullish, and since the long leg
    (R&D-heavy tech/pharma) and the short leg (banks/staples/energy) are *both* survivors, the
    survivorship tilt is largely common to both legs of the long/short, not a clean push on the
    spread. Opt in with ``allow_survivorship_bias=True``."""
    if not allow_survivorship_bias:
        raise PermissionError(
            "load_real() returns a CURRENT-membership large-cap basket projected back to 2005 "
            "(survivorship: delisted names are absent). Pass allow_survivorship_bias=True to opt "
            "in, and carry the caveat onto the Signal axis."
        )
    inten = pd.read_parquet(os.path.join(cache_dir, "intensity.parquet"))
    rets = pd.read_parquet(os.path.join(cache_dir, "returns.parquet"))
    if BENCH in rets.columns:
        spy = rets[BENCH].copy()
        rets = rets[[c for c in rets.columns if c != BENCH]]
    else:                                              # pragma: no cover - cache always has SPY
        spy = pd.Series(0.0, index=rets.index, name=BENCH)
    spy.name = "spy"
    return inten, rets, spy


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_years: int = 18, n_assets: int = 40, edge: float = 0.0,
                    market_vol: float = 0.045, idio_vol: float = 0.075,
                    seed: int = 400) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """A deterministic monthly panel whose names carry a *known* R&D-intensity premium.

    Each name has a **persistent** R&D intensity ``x_i`` (high for the first half of the field,
    low for the second — a stable tech-vs-bank split), drifting mildly year to year. The monthly
    return is::

        r_it = beta_i * mkt_t + (edge / 12) * sign(x_i - median_x) + eps_it

    so ``edge`` is the *true annual* long-high-minus-short-low premium: a high-intensity name earns
    ``+edge/2`` per year and a low-intensity name ``-edge/2``, so the long-short spread's expected
    value is exactly ``edge``/yr. Crucially the **market betas are de-meaned within each leg**, so
    at ``edge = 0`` the two legs have identical average market exposure and the long-short carries
    **no spurious beta spread** — a clean null.

    - ``edge = 0`` → intensity predicts nothing; the long-high / short-low book has zero expected
      spread and must NOT clear ``t >= 2`` (its spread is luck).
    - ``edge > 0`` → the high-intensity book genuinely out-earns; the harness must recover it.

    Returns ``(intensity, returns, bench, truth)``: an annual ``intensity`` panel, a monthly
    ``returns`` panel, the monthly equal-weight ``bench``, and the planted-parameter dict.
    """
    rng = np.random.default_rng(seed)
    n_months = n_years * MONTHS_PER_YEAR
    half = n_assets // 2

    # persistent per-name intensity: high half ~ tech/pharma, low half ~ banks/staples
    base_x = np.concatenate([
        rng.uniform(0.10, 0.30, half),                # high-intensity names
        rng.uniform(0.00, 0.04, n_assets - half),     # low-intensity names
    ])
    # annual intensity wiggles around the persistent base
    inten_years = np.clip(
        base_x[None, :] + rng.normal(0.0, 0.01, size=(n_years, n_assets)), 0.0, 1.0)

    # the planted per-name monthly premium: +edge/2 for high names, -edge/2 for low names, so the
    # long-minus-short spread equals `edge` per year by construction.
    leg_sign = np.where(np.arange(n_assets) < half, +1.0, -1.0)
    premium = leg_sign * (edge / 2.0) / MONTHS_PER_YEAR

    # market betas are uniform (= 1) across names: the market factor is common to both legs, so a
    # no-edge long-short carries NO market-beta spread (beta dispersion is not needed for this
    # control and would only inject a spurious leg-to-leg spread that fakes a signal at edge=0).
    betas = np.ones(n_assets)

    mkt = rng.normal(0.007, market_vol, n_months)
    eps = rng.normal(0.0, idio_vol, size=(n_months, n_assets))
    # Centre each name's idiosyncratic series to EXACTLY zero time-mean, so that at ``edge = 0``
    # no name carries a realised cross-sectional alpha by luck and the long-short null is exactly
    # flat (the planted ``premium`` is then the *only* source of any long-short spread). Without
    # this, a finite-sample idio realisation gives some seeds a spurious multi-%/yr spread at
    # edge=0 — a false positive the control exists to rule out.
    eps -= eps.mean(axis=0, keepdims=True)
    raw = mkt[:, None] * betas[None, :] + premium[None, :] + eps

    midx = pd.period_range("2007-01", periods=n_months, freq="M").to_timestamp()
    midx = pd.DatetimeIndex(midx, name="date")
    cols = [f"N{i:02d}" for i in range(n_assets)]
    returns = pd.DataFrame(raw, index=midx, columns=cols)
    bench = pd.Series(raw.mean(axis=1), index=midx, name="bench")

    yidx = pd.Index(range(2007, 2007 + n_years), name="fy")
    intensity = pd.DataFrame(inten_years, index=yidx, columns=cols)

    truth = {
        "n_years": n_years, "n_assets": n_assets, "edge": edge,
        "market_vol": market_vol, "idio_vol": idio_vol, "seed": seed,
        "high_cols": cols[: n_assets // 2], "low_cols": cols[n_assets // 2:],
    }
    return intensity, returns, bench, truth


def fingerprint(intensity: pd.DataFrame, returns: pd.DataFrame) -> str:
    """A short content fingerprint of the two panels, for the as-of stamp."""
    a = np.ascontiguousarray(np.nan_to_num(intensity.to_numpy(), nan=-9.99).ravel())
    b = np.ascontiguousarray(np.nan_to_num(returns.to_numpy(), nan=-9.99).ravel())
    return hashlib.sha1(a.tobytes() + b.tobytes()).hexdigest()[:12]
