"""Data layer for Study 796 — Corporate-Bond-Low-Risk (Betting-Against-Beta in bonds).

The claim (Frazzini & Pedersen 2014, *Betting Against Beta*) is a LOW-RISK anomaly: the
lowest-risk assets earn the highest *risk-adjusted* returns, so a portfolio that levers up
the safe leg and de-levers the risky leg (a "BAB" spread) earns a positive premium. In
bonds this maps onto: shorter-duration / higher-grade / lower-volatility exposures deliver
a higher Sharpe than long-duration / junk / high-volatility ones. We test it on the same
credit + Treasury bond-ETF universe as Study 795, ranking on trailing *volatility*.

Two tapes, one schema — a (date x ticker) daily adjusted-close (total-return) frame:

* ``synthetic_panel`` — a *deterministic, offline* generator. A tunable ``lowrisk_strength``
  knob tilts each asset's SHARPE inversely to its volatility, so the low-vol names earn more
  per unit of risk and a vol-scaled low-minus-high (BAB) book earns a premium.
  ``lowrisk_strength = 0`` is the null: every asset is priced on the SAME Sharpe line
  (return proportional to risk), so a risk-scaled low-minus-high book earns nothing.
  Tests and the reproducible core never touch the network.

* ``fetch_panel`` — real daily total-return prices from yfinance, cached to the study's
  OWN ``_cache/`` dir. Cache-first: returns the cached parquet if present, else an empty
  frame (so notebooks can banner "synthetic tape" on CI) unless ``fetch=True``.

**Survivorship (named on the Signal axis).** The basket is credit/Treasury ETFs still
trading in 2026, chosen ex post and projected backwards. This is milder than a single-name
credit sort — a broad ETF that becomes volatile simply gets *shorted/de-levered*, it does
not "delist" mid-sample — but the vehicle list itself was picked with hindsight (no ETF
that closed is here), so any low-risk premium measured here is treated as an upper bound.
The panel loads through :func:`fetch_panel`, whose docstring repeats the caveat, and the
README/notebooks carry it on the Signal axis. For the BAB direction the bias is subtle:
a survivor set that dropped blown-up high-vol ETFs would *understate* the risky-leg's
drawdowns and thus *understate* the low-risk premium — so a null here is not manufactured.

No look-ahead lives here: the trailing-volatility signal and the single execution lag
(rank on the month-*t* close, earn month *t+1*) live in ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

# Credit + Treasury bond-ETF universe (identical to Study 795, so the low-risk leg and the
# momentum leg are graded on the same tape). Deliberately spans the RISK ladder — from
# 1-3y Treasuries (very low vol) to 20y+ Treasuries and high-yield/EM credit (high vol) —
# so a volatility sort has real dispersion to rank on:
#   Low-risk pole:   SHY (1-3y UST), VCSH (short IG corp)
#   Mid:             LQD (IG corp), IEF (7-10y UST), BKLN (senior loans, floating)
#   High-risk pole:  TLT (20y+ UST, long duration), VCLT (long IG corp),
#                    HYG/JNK/ANGL (high yield), EMB (EM USD sovereign)
UNIVERSE = [
    "SHY", "VCSH", "LQD", "IEF", "BKLN", "VCLT", "TLT", "HYG", "JNK", "ANGL", "EMB",
]
_seen: set[str] = set()
UNIVERSE = [t for t in UNIVERSE if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]

# Human-readable labels + a coarse risk-pole tag (for the notebooks).
SLEEVE = {
    "SHY": "1-3y Treasury", "VCSH": "Short IG corp", "LQD": "IG corporate",
    "IEF": "7-10y Treasury", "BKLN": "Bank loans", "VCLT": "Long IG corp",
    "TLT": "20y+ Treasury", "HYG": "High yield", "JNK": "High yield",
    "ANGL": "Fallen angels", "EMB": "EM sovereign",
}

TRADING_DAYS = 252
START = "2007-01-01"
# Last complete calendar month at publication (built 2026-07-22); partial July dropped.
AS_OF = "2026-06-30"


@dataclass(frozen=True)
class WorldTruth:
    """The planted low-risk (BAB) strength for a synthetic panel."""

    lowrisk_strength: float  # per-unit-vol Sharpe tilt (0 = single Sharpe line = null)

    @property
    def has_lowrisk(self) -> bool:
        return self.lowrisk_strength != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline positive control
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_assets: int = 11,
    n_days: int = 3000,
    lowrisk_strength: float = 0.0,
    base_sharpe: float = 0.30,
    vol_lo: float = 0.02,
    vol_hi: float = 0.16,
    seed: int = 796,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible daily total-return panel with a tunable low-risk (BAB) tilt.

    Each asset ``i`` is assigned a fixed annual volatility ``sigma_i`` spread linearly from
    ``vol_lo`` (the "SHY" pole) to ``vol_hi`` (the "TLT/HYG" pole). Its annual **Sharpe** is

        s_i = base_sharpe + lowrisk_strength * (sigma_bar - sigma_i) / sigma_bar,

    so at ``lowrisk_strength = 0`` **every asset sits on the same Sharpe line** (expected
    return proportional to risk — the CAPM-like null in which a risk-scaled long-safe /
    short-risky book earns nothing). As ``lowrisk_strength`` grows, the low-vol names earn a
    *higher* Sharpe than the high-vol names — exactly the anomaly a vol-scaled BAB book
    harvests. The annual drift is ``mu_i = s_i * sigma_i``; daily drift ``mu_i / 252`` and
    daily vol ``sigma_i / sqrt(252)``. A small common "rates" factor makes the bonds co-move.

    Returns ``(prices, truth)`` with ``prices`` a (date x ticker) total-return frame. The
    business-day index spans a few years (never a huge nanosecond span), safely under the
    pandas ns-Timestamp overflow wall.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2011-01-03", periods=n_days)
    tickers = [f"B{j:02d}" for j in range(n_assets)]

    sigma = np.linspace(vol_lo, vol_hi, n_assets)          # annual vol per asset
    sigma_bar = float(sigma.mean())
    sharpe = base_sharpe + lowrisk_strength * (sigma_bar - sigma) / sigma_bar
    mu = sharpe * sigma                                    # annual drift per asset

    daily_mu = mu / TRADING_DAYS
    daily_sd = sigma / np.sqrt(TRADING_DAYS)

    # A small shared "rates/credit" factor (bonds co-move) plus idiosyncratic risk sized so
    # each asset's *total* daily vol matches daily_sd.
    common_sd_frac = 0.35
    common = rng.normal(0.0, 1.0, size=n_days)
    idio = rng.normal(0.0, 1.0, size=(n_days, n_assets))
    comp_common = common_sd_frac * daily_sd[None, :] * common[:, None]
    comp_idio = np.sqrt(max(1.0 - common_sd_frac**2, 0.0)) * daily_sd[None, :] * idio
    rets = daily_mu[None, :] + comp_common + comp_idio

    prices = pd.DataFrame(rets, index=dates, columns=tickers)
    prices = (1.0 + prices).cumprod() * 100.0
    prices.index.name = "date"
    return prices, WorldTruth(lowrisk_strength)


# ---------------------------------------------------------------------------
# Real panel — yfinance, study-local cache
# ---------------------------------------------------------------------------
def cache_path(cache_dir: str = DEFAULT_CACHE) -> str:
    """Path to the cached real-panel total-return parquet."""
    return os.path.join(cache_dir, "bondlowrisk_prices.parquet")


def have_real(cache_dir: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(cache_path(cache_dir))


def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = START,
    end: str = "2026-07-01",
    retries: int = 3,
    min_coverage: float = 0.60,
) -> pd.DataFrame:
    """Daily total-return (auto-adjusted close) prices for the bond-ETF basket, cache-first.

    Cache-only by default (``fetch=False``): returns the cached parquet if present, else an
    **empty** DataFrame (CI/offline path -> notebooks fall back to frozen numbers). Network
    is touched only on ``fetch=True`` (retried against yfinance flakiness), then the result
    is cached as adjusted close (``auto_adjust=True``, dividends/coupons folded in). The
    caller computes returns via ``prices.pct_change()``.

    Survivorship: this is the current-membership basket projected backwards — a
    hindsight-chosen vehicle list, named on the Signal axis. Names with < ``min_coverage``
    non-null history over the window are dropped (some ETFs listed after 2007).
    """
    p = cache_path(cache_dir)
    if os.path.exists(p):
        df = pd.read_parquet(p)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf  # lazy: only on an explicit network pull

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                list(UNIVERSE), start=start, end=end,
                auto_adjust=True, progress=False, threads=True,
            )["Close"]
            if raw.empty:
                raise RuntimeError("empty yfinance frame")
            raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
            coverage = raw.notna().mean()
            raw = raw.loc[:, coverage >= min_coverage]
            raw = raw.reindex(columns=[c for c in UNIVERSE if c in raw.columns])
            raw = raw.ffill().dropna(how="all")
            raw.index.name = "date"
            os.makedirs(cache_dir, exist_ok=True)
            raw.to_parquet(p)
            return raw
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"yfinance fetch failed after {retries} tries: {last_err}")


def load_real(cache_dir: str = DEFAULT_CACHE, asof: str = AS_OF) -> pd.DataFrame:
    """Cached real panel, sliced to ``<= asof`` (empty frame on a cache miss)."""
    df = fetch_panel(cache_dir=cache_dir, fetch=False)
    if df.empty:
        return df
    return df[df.index <= pd.Timestamp(asof)].copy()


def drop_partial_last_month(prices: pd.DataFrame, asof_day: int = 25) -> pd.DataFrame:
    """Drop trailing in-progress-month rows (house rule: no partial month in a stamp)."""
    if prices.empty:
        return prices
    last = prices.index[-1]
    today = pd.Timestamp.today().normalize()
    if last.year == today.year and last.month == today.month and today.day < asof_day:
        keep = ~((prices.index.year == today.year) & (prices.index.month == today.month))
        return prices.loc[keep]
    return prices


def fingerprint(df: pd.DataFrame | pd.Series) -> str:
    """A short content fingerprint of a panel (for the as-of stamp in docs/results.md)."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    arr = np.ascontiguousarray(df.fillna(0.0).to_numpy(dtype=float))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
