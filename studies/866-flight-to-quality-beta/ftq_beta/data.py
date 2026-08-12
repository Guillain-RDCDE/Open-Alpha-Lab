"""Data layer for Study 866 — Flight-to-Quality Beta.

The claim under test: some stocks are *true defensives* — they rise when long
Treasuries rise on **risk-off** days. For each name we measure a **flight-to-quality
beta** (``beta_ftq``): its beta to the **TLT** long-Treasury daily return, estimated
**only on down-SPY days**. A high FTQ beta is a stock that reliably co-moves with the
safe-haven bid when the market falls — a good crash hedge. The CAPM-of-insurance
prediction is two-sided: such names should (a) earn a **lower** average return (you pay
an insurance premium for the hedge) yet (b) deliver **real crash protection** (a
smaller loss on the worst market days). We test both.

Three ingredients, all offline-friendly once cached.

* **Real tape — a liquid US cross-section.** Daily OHLC for a fixed list of ~50 liquid
  US large-caps (``UNIVERSE`` below), pulled with yfinance through the
  ``quantlab.universe`` **survivorship guard** (``download_panel(...,
  allow_survivorship_bias=True)``). ``auto_adjust=True`` (total-return prices). The
  panel parquet is cached under this study's OWN ``_cache/`` (we point
  ``quantlab.universe``'s cache there via ``OVERNIGHT_CACHE`` *before* importing it).

* **Real tape — the risk-off conditioners.** Daily total-return closes for **TLT**
  (iShares 20+ Year Treasury Bond ETF, the flight-to-quality asset) and **SPY** (the
  market proxy whose down days define "risk-off"). Cached under ``_cache/`` as
  ``market_tlt_spy.parquet``; ``fetch_market()`` (network) builds it, ``load_market()``
  reads it offline.

  **Survivorship — named on the Signal axis.** ``UNIVERSE`` is a *current*
  membership list of names that are liquid mega-caps *today*; feeding it to a
  backward-looking panel omits the delisted / de-rated names and biases any
  cross-sectional result. The guard forces the opt-in; the caveat travels with every
  published number.

* **Synthetic world — the positive control.** A deterministic, seeded panel
  (``synthetic_panel``) with a TUNABLE knob ``edge``: it manufactures a market series,
  a TLT series, and a cross-section of names each carrying a persistent latent
  "flight-to-quality loading" ``c_i`` that both (a) governs how strongly the name loads
  on TLT on down-market days (so a trailing FTQ-beta sort proxies ``c_i``) and (b) —
  only when ``edge > 0`` — **depresses** its forward mean return (the insurance
  premium). ``edge = 0`` is the null world: FTQ betas still vary across names but carry
  **no** information about forward returns, and the sort must find nothing. ``edge > 0``
  plants the "pay-for-the-hedge" relation.

The offline path is pure numpy + pandas + stdlib. ``fetch()`` / ``fetch_market()``
(network) run once to build the caches and are never imported by the notebooks' offline
cells; ``load_panel()`` / ``load_market()`` read the cached parquets directly (no
yfinance import).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
# Point quantlab.universe's cache at THIS study's _cache/ before importing it.
os.environ.setdefault("OVERNIGHT_CACHE", CACHE_DIR)

from quantlab.universe import (  # noqa: E402  (after the env var is set)
    SurvivorshipBiasError,
    download_panel,
    panel_cache_path,
)

START = "2010-01-01"        # panel start (matches quantlab.universe default)
AS_OF = "2026-06-30"        # last complete calendar month at publication

MARKET_CACHE = os.path.join(CACHE_DIR, "market_tlt_spy.parquet")

# A fixed list of ~50 liquid US large-caps — *current* membership, a survivor set.
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "HD", "BAC", "XOM", "CVX", "KO", "PEP", "ABBV",
    "COST", "MRK", "PFE", "CSCO", "ORCL", "ADBE", "CRM", "NKE", "DIS", "MCD",
    "TXN", "INTC", "QCOM", "AMD", "IBM", "GE", "CAT", "BA", "MMM", "HON",
    "UNH", "T", "VZ", "WFC", "GS", "MS", "C", "AXP", "LMT", "UPS",
]

__all__ = [
    "SurvivorshipBiasError",
    "UNIVERSE", "START", "AS_OF", "CACHE_DIR", "MARKET_CACHE",
    "fetch", "fetch_market", "have_real", "have_market",
    "load_panel", "load_market", "synthetic_panel",
]


# --------------------------------------------------------------------------- #
# Real tape — equity cross-section
# --------------------------------------------------------------------------- #
def fetch(start: str = START) -> None:
    """Download the cross-section panel through the survivorship guard; cache it."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    download_panel(
        UNIVERSE, start=start, use_cache=True, allow_survivorship_bias=True,
    )


def have_real() -> bool:
    return os.path.exists(panel_cache_path(UNIVERSE, START))


def load_panel(start: str = START, asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    """Cached panel as ``{ticker: DataFrame[Open, High, Low, Close]}``, sliced to
    ``[start, asof]``. Reads the parquet directly — OFFLINE, no yfinance import."""
    cache = panel_cache_path(UNIVERSE, start)
    raw = pd.read_parquet(cache)
    lo, hi = pd.Timestamp(start), pd.Timestamp(asof)
    panel: dict[str, pd.DataFrame] = {}
    for s in UNIVERSE:
        if s not in raw.columns.get_level_values(0):
            continue
        df = raw[s][["Open", "High", "Low", "Close"]].dropna()
        df = df[(df.index >= lo) & (df.index <= hi)]
        if not df.empty:
            panel[s] = df
    return panel


# --------------------------------------------------------------------------- #
# Real tape — TLT (flight-to-quality asset) + SPY (risk-off conditioner)
# --------------------------------------------------------------------------- #
def fetch_market(start: str = START) -> None:
    """Download TLT + SPY total-return closes with yfinance; cache to ``_cache/``.

    Network-only; never imported by the offline notebook cells. Kept separate from the
    equity ``fetch()`` because these two series are the study's risk-off conditioners,
    not part of the shared 50-name panel.
    """
    import yfinance as yf  # local import — only on the network path

    os.makedirs(CACHE_DIR, exist_ok=True)
    d = yf.download(
        ["TLT", "SPY"], start=start, end="2026-07-01",
        auto_adjust=True, progress=False,
    )
    close = d["Close"][["TLT", "SPY"]].dropna()
    close.to_parquet(MARKET_CACHE)


def have_market() -> bool:
    return os.path.exists(MARKET_CACHE)


def load_market(start: str = START, asof: str = AS_OF) -> pd.DataFrame:
    """Cached ``DataFrame[TLT, SPY]`` daily total-return closes, sliced to
    ``[start, asof]``. OFFLINE — reads the parquet directly."""
    close = pd.read_parquet(MARKET_CACHE)
    lo, hi = pd.Timestamp(start), pd.Timestamp(asof)
    close = close[(close.index >= lo) & (close.index <= hi)]
    return close[["TLT", "SPY"]].dropna().sort_index()


# --------------------------------------------------------------------------- #
# Synthetic world — planted FTQ-beta -> low-return relation (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 866,
    n_assets: int = 40,
    n_days: int = 3000,
    start: str = "2010-01-04",
    daily_vol: float = 0.012,
    drift: float = 0.06 / 252,
    mkt_vol: float = 0.010,
    tlt_vol: float = 0.008,
    ftq_gain: float = 1.2,
    mkt_beta: float = 1.0,
) -> dict:
    """Deterministic seeded world for the FTQ-beta control.

    Returns a dict ``{"panel": {ticker: OHLC df}, "market": Series(SPY-proxy returns),
    "tlt": Series(TLT-proxy returns)}`` sharing one business-day index.

    Construction. A market shock ``m[t] ~ N(0, mkt_vol)`` and a Treasury shock
    ``b[t]`` that carries a **flight-to-quality bid on risk-off days**: when ``m[t] <
    0`` bonds rally against the sell-off (``b`` gets ``-flight * m`` plus noise), else
    ``b`` is idiosyncratic noise. Each asset ``i`` has a persistent latent FTQ loading
    ``c_i[t]`` (an AR(1)); on **down-market days** the name loads on the Treasury shock
    with strength ``ftq_gain * c_i`` (so a high ``c_i`` name rallies with bonds in
    sell-offs — a high realized FTQ beta), and always loads ``mkt_beta`` on the market.
    Only when ``edge > 0`` does a high loading **depress** the forward mean return
    (``- edge * c_i`` — the insurance premium). ``edge = 0`` is the null: FTQ betas
    vary but predict nothing. Business-day index; span well below the pandas
    ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)

    m = rng.normal(0.0, mkt_vol, n_days)                 # market (SPY proxy) returns
    dn = m < 0.0                                         # risk-off days
    flight = 0.9                                          # strength of the safe-haven bid
    noise_b = rng.normal(0.0, tlt_vol, n_days)          # mean-zero Treasury innovation
    # TLT return = baseline + idiosyncratic innovation + a down-day flight-to-quality
    # rally (bonds bid up when the market sells off).
    b = 0.0004 + noise_b + np.where(dn, -flight * m, 0.0)

    market = pd.Series(m, index=idx, name="market")
    tlt = pd.Series(b, index=idx, name="tlt")

    innov_sd = np.sqrt(1.0 - 0.98 ** 2)
    panel: dict[str, pd.DataFrame] = {}
    for i in range(n_assets):
        c = np.empty(n_days)
        c[0] = rng.normal(0.0, 1.0)
        eps = rng.normal(0.0, innov_sd, n_days)
        for t in range(1, n_days):
            c[t] = 0.98 * c[t - 1] + eps[t]

        idio = rng.normal(0.0, daily_vol, n_days)
        # On down-market days the name loads on the *mean-zero* Treasury innovation with
        # strength ftq_gain*c_i — that is exactly its flight-to-quality beta, and being
        # mean-zero it adds NO return bias (so edge=0 is a clean null). It always loads
        # mkt_beta on the market. edge>0 taxes the forward mean (the insurance premium).
        ftq_load = np.where(dn, ftq_gain * c * noise_b, 0.0)
        r = drift - edge * c + mkt_beta * m + ftq_load + idio

        close = 100.0 * np.cumprod(1.0 + r)
        prev_close = np.concatenate([[100.0], close[:-1]])
        open_ = prev_close * (1.0 + rng.normal(0.0, daily_vol / 3, n_days))
        hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
        lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
        panel[f"SYN{i:02d}"] = pd.DataFrame(
            {"Open": open_, "High": hi, "Low": lo, "Close": close}, index=idx
        )
    return {"panel": panel, "market": market, "tlt": tlt}
