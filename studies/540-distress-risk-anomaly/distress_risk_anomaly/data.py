"""Data layer for Study 540 (Distress-Risk-Anomaly) — the CHS failure-probability sort.

The Campbell-Hilscher-Szilagyi (2008) distress measure blends *profitability* (net income over
assets), *leverage* (total liabilities over assets) and *equity volatility* (and, in the full
spec, market cap, cash, the price level and excess return). The puzzle is that the firms this
measure flags as **most likely to fail** go on to earn the **lowest** returns — the inverse of a
risk premium.

This study works with what a no-key retail stack can actually reach:

- **Daily adjusted prices** for a fixed ~40-name large-cap survivor basket plus SPY — fully
  historical, so trailing realised **equity volatility** and **forward returns** are honest.
- **A fundamentals snapshot per name** (total liabilities / total assets and net income / total
  assets) from ``yfinance`` ``.balance_sheet`` / ``.financials`` / ``.info`` — the *leverage* and
  *profitability* legs of the CHS measure. yfinance exposes only a shallow history of statements,
  so the accounting legs are a recent snapshot (the limitation is named on the SIGNAL axis).

Two tapes, one schema:

- ``synthetic_panel`` — a deterministic, offline generator. A single knob, ``distress_alpha``,
  plants the *anomaly*: the higher a firm's true failure probability, the **lower** its alpha
  (``distress_alpha < 0`` reproduces the puzzle; ``= 0`` is the null). The study's positive
  control and null in one bottle. Tests never touch the network.
- ``fetch_panel`` — the real yfinance tape (daily prices + a fundamentals snapshot), cache-first
  into this study's own ``_cache/`` so the reproducible core never needs the network.

Survivorship: the basket is names *still trading in 2026*, so the failures that drive the real
CHS effect (firms that actually went bankrupt) are absent by construction. This biases the real
tape *against* finding the puzzle — stated openly on the SIGNAL axis.

No look-ahead: distress is scored from the *trailing* volatility window (ending at the split
close) and the *last reported* statements, so the signal is fully public at that close; the
position is entered at that same close and held over the *forward* window. No future data enters
the signal. The same-close fill is a documented convention (one ~2-year hold; a one-bar-later
entry moves the headline ~1pt and changes no stamp — see ``strategy.py``).
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
SHARED_CACHE = os.path.join(REPO_ROOT, "_cache")

# A fixed ~40-name large-cap survivor basket spanning leverage/profitability regimes:
# mega-cap tech (low leverage), banks & utilities (high leverage), staples, industrials,
# telecoms & media (the higher-distress end of a blue-chip universe).
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AVGO", "ORCL",  # tech
    "JPM", "BAC", "C", "WFC", "GS", "MS",                            # banks (high leverage)
    "XOM", "CVX", "COP",                                             # energy
    "JNJ", "PFE", "MRK", "ABBV",                                     # pharma
    "PG", "KO", "PEP", "WMT", "COST",                                # staples
    "HD", "MCD", "DIS",                                              # consumer
    "CAT", "GE", "BA", "HON",                                        # industrials
    "T", "VZ", "CMCSA",                                              # telecom/media (levered)
    "SO", "DUK", "NEE",                                              # utilities (high leverage)
]
_seen: set[str] = set()
UNIVERSE = [t for t in UNIVERSE if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]

TRADING_DAYS = 252


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    distress_alpha: float  # alpha per unit of true failure prob (negative = the puzzle)

    @property
    def has_anomaly(self) -> bool:
        return self.distress_alpha != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 40,
    distress_alpha: float = -0.12,
    idio_vol: float = 0.06,
    seed: int = 540,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible cross-section with a tunable distress *anomaly*.

    Each firm gets a latent failure probability ``p`` (a CHS-style distress score) drawn from a
    Beta distribution (most firms safe, a right tail of distressed names). From ``p`` we derive
    the *observable* legs the strategy actually sees — high leverage, low profitability and high
    equity volatility — plus a yearly forward return:

        forward_ret = market_drift + distress_alpha * (p - p_bar) + idio

    With ``distress_alpha < 0`` the most-distressed firms earn the *lowest* return (the puzzle);
    ``distress_alpha = 0`` is the null (return is unrelated to distress). The returned frame has
    one row per firm with the columns the strategy consumes: ``leverage``, ``roa``, ``equity_vol``
    and the realised ``forward_ret`` — exactly the schema ``fetch_panel`` produces from real data.

    Returns ``(panel, truth)``.
    """
    rng = np.random.default_rng(seed)
    p = rng.beta(1.6, 6.0, size=n_stocks)            # failure prob, right-skewed
    p_bar = p.mean()

    # Observable CHS legs, monotone in the latent p (with noise so the rank isn't perfect):
    leverage = np.clip(0.35 + 0.55 * p + 0.05 * rng.standard_normal(n_stocks), 0.05, 0.98)
    roa = 0.10 - 0.25 * p + 0.02 * rng.standard_normal(n_stocks)        # profitability falls with p
    equity_vol = np.clip(0.18 + 0.9 * p + 0.03 * rng.standard_normal(n_stocks), 0.08, 1.2)

    drift = 0.09
    idio = idio_vol * rng.standard_normal(n_stocks)
    forward_ret = drift + distress_alpha * (p - p_bar) + idio

    tickers = [f"S{j:03d}" for j in range(n_stocks)]
    panel = pd.DataFrame(
        {
            "leverage": leverage,
            "roa": roa,
            "equity_vol": equity_vol,
            "forward_ret": forward_ret,
            "true_p": p,
        },
        index=pd.Index(tickers, name="ticker"),
    )
    return panel, WorldTruth(distress_alpha=distress_alpha)


# ---------------------------------------------------------------------------
# Real tape — yfinance daily prices + a fundamentals snapshot, cache-first
# ---------------------------------------------------------------------------
def _price_cache() -> str:
    return os.path.join(DEFAULT_CACHE, "distress_prices.parquet")


def _fund_cache() -> str:
    return os.path.join(DEFAULT_CACHE, "distress_fundamentals.json")


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
    start: str = "2014-01-01",
    end: str = "2026-06-26",
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


def fetch_fundamentals(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> dict:
    """A per-name fundamentals snapshot — leverage and ROA — cache-first.

    Returns ``{ticker: {"leverage": tl/ta, "roa": ni/ta}}`` for the legs of the CHS measure that
    need accounting data. yfinance exposes only a shallow statement history, so this is the most
    recent annual snapshot (the *accounting-snapshot* limitation, named on the SIGNAL axis).

    Cache-only by default; ``fetch=True`` pulls per name (slow, flaky) and caches the JSON.
    """
    path = _fund_cache()
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
            bs = t.balance_sheet
            fin = t.financials
            info = t.info or {}
            ta = _first_row(bs, ["Total Assets"])
            tl = _first_row(bs, ["Total Liabilities Net Minority Interest",
                                 "Total Liabilities"])
            ni = _first_row(fin, ["Net Income", "Net Income Common Stockholders"])
            if ta is None or ta == 0:
                ta = info.get("totalAssets")
            lev = (tl / ta) if (tl is not None and ta) else info.get("debtToEquity")
            roa = (ni / ta) if (ni is not None and ta) else info.get("returnOnAssets")
            if lev is None or roa is None:
                continue
            # debtToEquity from .info is in %, convert to a ratio-ish leverage if used as fallback
            if tl is None and lev is not None and lev > 5:
                lev = lev / 100.0
            out[tk] = {"leverage": float(lev), "roa": float(roa)}
        except Exception:  # noqa: BLE001
            continue
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def _first_row(df, names):
    """First non-null value from the most recent column for any of ``names`` in a yf statement."""
    if df is None or getattr(df, "empty", True):
        return None
    for nm in names:
        if nm in df.index:
            row = df.loc[nm].dropna()
            if not row.empty:
                return float(row.iloc[0])
    return None


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, dict):
        blob = json.dumps(obj, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
