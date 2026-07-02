"""Data layer for Study 572 (Capex-Cycle) — the capex-*growth* cousin of asset growth.

The claim: a firm in a **capex-spending binge** — one that is *ramping* its capital
expenditure relative to its asset base — earns lower future returns than a firm *harvesting*
cash (letting capex intensity fall). This is the **capex-specific** slice of the
investment/asset-growth anomaly: not the *level* of investment (Titman-Wei-Xie's CapEx/Assets,
[Study 523](../../523-investment-to-assets/)) nor total-asset growth (Cooper-Gulen-Schill,
[Study 244](../../244-asset-growth/)), but the **change** in capex intensity — the phase of the
capex *cycle* a firm is in.

The signal, precisely:

    capex_intensity_t = |CapEx_t| / TotalAssets_{t-1}
    capex_cycle       = capex_intensity_t - capex_intensity_{t-1}   (a *binge* if > 0)

A positive ``capex_cycle`` is a firm accelerating its spending (over-investment risk); a negative
one is a firm harvesting. The over-investment hypothesis predicts *high capex_cycle -> low future
return* (a negative cross-sectional IC / a positive long-low/short-high hedge).

Two tapes, one schema:

- ``synthetic_panel`` — deterministic, offline, seeded. A single knob ``binge_alpha`` plants the
  effect: the bigger a firm's true capex binge, the *lower* its forward return
  (``binge_alpha < 0`` reproduces the anomaly; ``= 0`` is the null). Positive control + null in
  one bottle. Tests never touch the network.
- ``fetch_panel`` inputs (``fetch_prices`` + ``fetch_cashflow``) — the real yfinance tape (daily
  adjusted prices + a per-name **cash-flow snapshot** of Capital Expenditure and Total Assets),
  cache-first into this study's own ``_cache/`` so the reproducible core never needs the network.

**Data-availability limitation (named on the SIGNAL axis).** yfinance exposes only ~4-5 *annual*
statements per name. Computing a capex *cycle* costs two of those (a change of a ratio that each
need a prior-year asset base), so the real panel is a **snapshot cross-section**: one capex_cycle
value per name scored at the latest fiscal year, one forward window. That is enough for a
cross-sectional IC + placebo on a survivor basket, but it is *not* a deep point-in-time panel — a
true replication needs Compustat / EDGAR history (as [Study 523](../../523-investment-to-assets/)
uses for the capex *level*). This is why the study is capped below REAL.

Survivorship: the basket is names *still large-cap in 2026*, so the aggressive over-investors that
*failed* (the heart of the real effect) are absent by construction — biasing the tape *against*
the anomaly. Stated openly on the SIGNAL axis.

No look-ahead: capex_cycle is scored from the *last two reported* fiscal years (fully public at
the scoring close) and the position is entered at that close and held over the *forward* window.
No future data enters the signal. The same-close fill is a documented one-lag convention.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

# A fixed large-cap survivor basket that spans capex regimes: capex-heavy fabs / hyperscalers /
# energy majors / utilities / industrials (the high-intensity end) and capital-light software /
# staples / financials (the low-intensity end). Names still trading in 2026 (survivorship, named).
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AVGO", "ORCL", "CSCO", "INTC",  # tech
    "TSM", "AMD", "TXN", "QCOM", "MU",                                                # semis (capex)
    "XOM", "CVX", "COP", "SLB",                                                       # energy (capex)
    "JPM", "BAC", "WFC", "GS",                                                        # banks (light)
    "JNJ", "PFE", "MRK", "ABBV",                                                      # pharma
    "PG", "KO", "PEP", "WMT", "COST", "MCD", "HD",                                    # staples/cons
    "CAT", "GE", "BA", "HON", "UNP", "UPS",                                           # industrials
    "T", "VZ",                                                                        # telecom (capex)
    "SO", "DUK", "NEE", "D",                                                          # utilities (capex)
    "DIS", "NKE", "SBUX",                                                             # consumer
]
_seen: set[str] = set()
UNIVERSE = [t for t in UNIVERSE if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]

TRADING_DAYS = 252


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    binge_alpha: float  # alpha per unit of capex binge (negative = the over-investment anomaly)

    @property
    def has_effect(self) -> bool:
        return self.binge_alpha != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 48,
    binge_alpha: float = -0.30,
    idio_vol: float = 0.05,
    seed: int = 572,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible cross-section with a tunable capex-*cycle* effect.

    Each firm gets a latent *capex binge* ``b`` (how much it is ramping capex intensity), drawn
    symmetric around zero (some firms bingeing, some harvesting). From ``b`` we derive the
    *observable* the strategy actually sees — ``capex_cycle`` (a noisy read of ``b``) — plus a
    yearly forward return:

        forward_ret = market_drift + binge_alpha * (b - b_bar) + idio

    With ``binge_alpha < 0`` the biggest bingers earn the *lowest* return (the anomaly);
    ``binge_alpha = 0`` is the null. The frame carries the schema the strategy consumes:
    ``capex_cycle``, ``forward_ret`` (and the latent ``true_b`` for diagnostics) — exactly the
    columns ``build_panel`` produces from real data.

    Returns ``(panel, truth)``.
    """
    rng = np.random.default_rng(seed)
    b = 0.06 * rng.standard_normal(n_stocks)              # latent capex binge (change in intensity)
    b_bar = b.mean()

    # Observable capex_cycle: the latent binge read with noise (rank isn't perfect):
    capex_cycle = b + 0.02 * rng.standard_normal(n_stocks)

    drift = 0.09
    idio = idio_vol * rng.standard_normal(n_stocks)
    forward_ret = drift + binge_alpha * (b - b_bar) + idio

    tickers = [f"S{j:03d}" for j in range(n_stocks)]
    panel = pd.DataFrame(
        {
            "capex_cycle": capex_cycle,
            "forward_ret": forward_ret,
            "true_b": b,
        },
        index=pd.Index(tickers, name="ticker"),
    )
    return panel, WorldTruth(binge_alpha=binge_alpha)


# ---------------------------------------------------------------------------
# Real tape — yfinance daily prices + a cash-flow snapshot, cache-first
# ---------------------------------------------------------------------------
def _price_cache() -> str:
    return os.path.join(DEFAULT_CACHE, "capex_prices.parquet")


def _cf_cache() -> str:
    return os.path.join(DEFAULT_CACHE, "capex_cashflow.json")


def _retry(fn, tries: int = 3, pause: float = 1.0):
    """Tiny retry wrapper for yfinance flakiness."""
    last = None
    for _ in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(pause)
    raise last  # type: ignore[misc]


def fetch_prices(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "2018-01-01",
    end: str = "2026-06-27",
) -> pd.DataFrame:
    """Daily adjusted-close prices for the basket + SPY, cache-first.

    Cache-only by default (``fetch=False``): returns the cached parquet if present, else an empty
    frame. Network is touched only on an explicit ``fetch=True`` (then cached).
    """
    path = _price_cache()
    if os.path.exists(path):
        px = pd.read_parquet(path)
        if px.index.tz is not None:
            px.index = px.index.tz_localize(None)
        return px
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf  # lazy

    tickers = list(UNIVERSE) + ["SPY"]
    raw = _retry(
        lambda: yf.download(
            tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True
        )["Close"]
    )
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    cov = raw.notna().mean()
    raw = raw.loc[:, cov >= 0.5]
    raw = raw.dropna(how="all")
    os.makedirs(cache_dir, exist_ok=True)
    raw.to_parquet(path)
    return raw


def fetch_cashflow(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> dict:
    """A per-name cash-flow + assets snapshot for the capex-cycle legs, cache-first.

    Returns ``{ticker: {"years": [YYYY, ...], "capex": [...], "assets": [...]}}`` where lists are
    ordered **oldest -> newest** annual fiscal-year values of |Capital Expenditure| and
    Total Assets. yfinance exposes only a shallow (~4-5yr) annual history, so this is a snapshot
    (the *cash-flow-snapshot* limitation, named on the SIGNAL axis).

    Cache-only by default; ``fetch=True`` pulls per name (slow, flaky) and caches the JSON.
    """
    path = _cf_cache()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if not fetch:
        return {}

    import yfinance as yf  # lazy

    out: dict[str, dict] = {}
    for tk in UNIVERSE:
        try:
            t = _retry(lambda tk=tk: yf.Ticker(tk))
            cf = t.cashflow
            bs = t.balance_sheet
            if cf is None or getattr(cf, "empty", True) or bs is None or getattr(bs, "empty", True):
                continue
            capex_row = _row(cf, ["Capital Expenditure", "Purchase Of PPE",
                                  "Net PPE Purchase And Sale"])
            asset_row = _row(bs, ["Total Assets"])
            if capex_row is None or asset_row is None:
                continue
            # Align on the columns (fiscal-year timestamps) present in BOTH statements.
            cols = [c for c in cf.columns if c in bs.columns]
            cols = sorted(cols)  # oldest -> newest
            years, capex, assets = [], [], []
            for c in cols:
                cx = capex_row.get(c)
                ta = asset_row.get(c)
                if cx is None or ta is None or pd.isna(cx) or pd.isna(ta) or ta == 0:
                    continue
                years.append(int(pd.Timestamp(c).year))
                capex.append(abs(float(cx)))   # capex is reported negative; use magnitude
                assets.append(float(ta))
            if len(years) < 3:                 # need >=3 yrs to build a change-of-ratio + 1 lag
                continue
            out[tk] = {"years": years, "capex": capex, "assets": assets}
        except Exception:  # noqa: BLE001
            continue
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def _row(df, names):
    """The most-recent-first row (as a {col: value} dict) for the first matching name."""
    if df is None or getattr(df, "empty", True):
        return None
    for nm in names:
        if nm in df.index:
            return df.loc[nm].to_dict()
    return None


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, dict):
        blob = json.dumps(obj, sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
