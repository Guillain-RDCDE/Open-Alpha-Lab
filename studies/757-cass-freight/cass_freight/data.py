"""Data layer for Study 757 — Cass-Freight (a hardcoded Cass proxy + real SPY/IYT).

Three sources, all offline-friendly once the caches exist:

* **The Cass Freight Index (hardcoded, cited, APPROXIMATE).** The real Cass Freight
  Index — *Shipments* component — is published monthly by Cass Information Systems in a
  press release (and mirrored on FRED as ``FRGSHPUSM649NCIS``), but it is **not** freely
  API-available and its full monthly history is behind Cass's reporting. So we hardcode a
  small **annual** path of the shipments level (base Jan-1990 ≈ 1.00, the Cass
  convention), reconstructed from public Cass commentary, and interpolate it to a monthly
  **LABELLED PROXY** with a fixed deterministic seasonal shape. The *shape* is the
  load-bearing fact — the dot-com freight softness (2001–02), the GFC collapse (2008–09),
  the 2015–16 industrial/freight recession, the 2019 slowdown, the 2020 COVID air-pocket,
  and the long 2022–24 freight recession. It is a PROXY for the real tape, never the tape,
  and the study conditions on its **year-over-year growth**, which the exact levels barely
  move. Sources in ``docs/references.md``.

* **Real equity tapes (yfinance).** Month-end adjusted closes for **SPY** (the broad
  market, from 1993) and **IYT** (the iShares U.S. Transportation ETF — the freight-
  sensitive sector, from 2004), cached under ``_cache/``. On a cache miss with network we
  fetch via yfinance; the offline notebook cells never import yfinance.

* **Synthetic positive control.** A deterministic, fixed-seed generator producing a
  freight-like YoY series and an SPY-like price with a PLANTED forward edge knob. With
  ``edge=0`` the inference must NOT manufacture significance; a large planted edge must
  light it up. It is the machinery proof.

Pure numpy + pandas + stdlib for the offline path.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
SPY_CACHE = os.path.join(CACHE_DIR, "spy_monthly.csv")
IYT_CACHE = os.path.join(CACHE_DIR, "iyt_monthly.csv")

# The publication + execution lag from a freight *reference* month to the first month a
# position built on it can earn a return. Cass publishes the reference-month index in the
# middle of the FOLLOWING month, so the print for month t is public by the close of t+1
# (one month of publication delay); a position taken at that close earns the return of t+2
# (one more month of execution). Hence 2 — documented once, applied once. No look-ahead.
DEFAULT_LAG = 2

# --------------------------------------------------------------------------- #
# The Cass Freight Index — hardcoded, cited, APPROXIMATE (a proxy, not a feed)
# --------------------------------------------------------------------------- #
# Annual AVERAGE level of the Cass Freight Index — Shipments component (base Jan-1990 ≈
# 1.00, the Cass convention). Reconstructed from public Cass reporting and clearly labelled
# approximate. The year-over-year *sign pattern* — expansions vs the four contractions
# (2001–02, 2008–09, 2015–16, 2019, 2020, 2022–24) — is the load-bearing shape; the exact
# decimals are not. See docs/references.md.
_CASS_ANNUAL = {
    1999: 1.06, 2000: 1.10, 2001: 1.03, 2002: 1.01, 2003: 1.04,
    2004: 1.11, 2005: 1.15, 2006: 1.16, 2007: 1.14, 2008: 1.09,
    2009: 0.90, 2010: 1.01, 2011: 1.07, 2012: 1.09, 2013: 1.10,
    2014: 1.16, 2015: 1.14, 2016: 1.09, 2017: 1.14, 2018: 1.20,
    2019: 1.14, 2020: 1.06, 2021: 1.16, 2022: 1.17, 2023: 1.09,
    2024: 1.04, 2025: 1.03, 2026: 1.03,
}

# A fixed, deterministic freight seasonal (multiplicative, mean 1.0): soft winter, a
# summer build, a fall peak, a December fade — the well-known shipping calendar. YoY growth
# over 12 months cancels it exactly, so it decorates the level chart without touching the
# signal the study conditions on.
_SEASONAL = np.array([0.96, 0.95, 0.99, 1.00, 1.01, 1.02,
                      1.02, 1.03, 1.04, 1.03, 1.00, 0.95])


def cass_index() -> pd.Series:
    """Monthly Cass Freight Index — Shipments PROXY (month-end), base Jan-1990 ≈ 1.00.

    Annual anchors (``_CASS_ANNUAL``) are placed at each mid-year (June 30) and linearly
    interpolated to month-ends, then multiplied by the fixed seasonal factor. This is an
    explicit, cited, APPROXIMATE stand-in for the real Cass series — never the live tape.
    """
    anchors = pd.Series(
        {pd.Timestamp(f"{y}-06-30"): v for y, v in _CASS_ANNUAL.items()}
    ).sort_index()
    # month-end grid spanning the anchors
    grid = pd.date_range(anchors.index.min(), anchors.index.max(), freq="ME")
    lvl = anchors.reindex(anchors.index.union(grid)).interpolate("time").reindex(grid)
    seas = np.array([_SEASONAL[d.month - 1] for d in lvl.index])
    out = (lvl.values * seas)
    s = pd.Series(out, index=grid, name="cass")
    return s.dropna()


# --------------------------------------------------------------------------- #
# Real equity tapes (yfinance, cached, offline-friendly)
# --------------------------------------------------------------------------- #
def fetch_spy(path: str = SPY_CACHE, start: str = "1993-01-01",
              end: str | None = None) -> pd.Series:
    """Download SPY month-end adjusted closes and cache them (network; once)."""
    return _fetch_one("SPY", path, start, end)


def fetch_iyt(path: str = IYT_CACHE, start: str = "2003-01-01",
              end: str | None = None) -> pd.Series:
    """Download IYT (iShares Transportation) month-end adjusted closes and cache (once)."""
    return _fetch_one("IYT", path, start, end)


def _fetch_one(ticker: str, path: str, start: str, end: str | None) -> pd.Series:
    import yfinance as yf  # lazy: the offline core never needs it

    raw = yf.download(ticker, start=start, end=end, interval="1mo",
                      auto_adjust=True, progress=False)["Close"]
    s = (raw.iloc[:, 0] if isinstance(raw, pd.DataFrame) else raw).dropna()
    s.index = s.index + pd.offsets.MonthEnd(0)  # stamp on the month-end
    s.name = ticker.lower()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s.to_csv(path)
    return s


def have_real(spy: str = SPY_CACHE, iyt: str = IYT_CACHE) -> bool:
    return os.path.exists(spy) and os.path.exists(iyt)


def _load_one(path: str, name: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    s = df.iloc[:, 0].astype(float)
    s.index = s.index + pd.offsets.MonthEnd(0)
    s.name = name
    return s.dropna()


def load_spy(path: str = SPY_CACHE) -> pd.Series:
    """Cached SPY month-end adjusted close as a Series named 'spy'."""
    return _load_one(path, "spy")


def load_iyt(path: str = IYT_CACHE) -> pd.Series:
    """Cached IYT month-end adjusted close as a Series named 'iyt'."""
    return _load_one(path, "iyt")


def build_real(asof: str | None = "2026-06-30") -> pd.DataFrame:
    """Monthly frame with columns ``cass`` (proxy), ``spy`` and ``iyt`` (real), aligned.

    Each freight value is stamped on the month-end of its **reference** month; the strategy
    layer applies :data:`DEFAULT_LAG` months of publication+execution delay, so this frame
    carries no look-ahead by itself. ``iyt`` is NaN before the ETF's 2004 inception (the
    join keeps every month SPY+Cass overlap and lets IYT tests use the shorter window). A
    partial trailing month is dropped by pinning ``asof`` (never in the future).
    """
    cass = cass_index()
    spy = load_spy()
    iyt = load_iyt()
    out = pd.DataFrame({"cass": cass}).join(pd.DataFrame({"spy": spy}), how="inner")
    out = out.join(pd.DataFrame({"iyt": iyt}), how="left")
    if asof is not None:
        out = out.loc[out.index <= pd.Timestamp(asof)]
    return out.dropna(subset=["cass", "spy"])


# --------------------------------------------------------------------------- #
# Synthetic positive control (deterministic, fixed seed, no network)
# --------------------------------------------------------------------------- #
def synthetic(n_months: int = 312, edge: float = 0.0, seed: int = 757,
              mu_monthly: float = 0.006, sig_monthly: float = 0.043,
              phi: float = 0.92) -> pd.DataFrame:
    """Deterministic freight-YoY-like series + SPY-like price with a PLANTED forward edge.

    ``cass_yoy`` is a persistent AR(1) (persistence ``phi`` — the slow, autocorrelated
    swing a real freight cycle carries) with ~unit variance. The SPY-like monthly return
    has drift ``mu_monthly`` and vol ``sig_monthly``; if ``edge`` != 0 an *extra* return of
    ``edge * cass_yoy_t`` is injected into month ``t+DEFAULT_LAG`` — so a high-freight
    reference month genuinely predicts a higher return once the publication+execution lag
    has passed.

    ``edge = 0`` is the null: freight carries no forward information and the inference must
    NOT manufacture significance. A large ``edge`` must light it up. The frame exposes the
    YoY directly as ``cass`` so the strategy's ``expanding`` test (freight growth > 0) reads
    it straight; ``build_real`` instead derives YoY from the level. The date index is a
    decorative monthly label via ``period_range`` (no OutOfBounds on long spans).
    """
    rng = np.random.default_rng(seed)
    yoy = np.empty(n_months)
    yoy[0] = rng.normal()
    for t in range(1, n_months):
        yoy[t] = phi * yoy[t - 1] + rng.normal(0, np.sqrt(1 - phi ** 2))

    ret = rng.normal(mu_monthly, sig_monthly, size=n_months)
    if edge != 0.0:
        ret[DEFAULT_LAG:] += edge * yoy[:-DEFAULT_LAG]

    price = 100.0 * np.exp(np.cumsum(ret))
    idx = pd.period_range("2000-01", periods=n_months, freq="M").to_timestamp(how="end")
    # 'cass_is_yoy' marks that the 'cass' column already IS the YoY growth (not a level)
    df = pd.DataFrame({"cass": yoy, "spy": price, "iyt": price}, index=idx)
    df.attrs["cass_is_yoy"] = True
    return df
