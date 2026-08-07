"""Data layer for Study 826 — Treasury Duration BAB.

The claim under test (Frazzini & Pedersen 2014, *"Betting Against Beta"*): low-beta
assets earn higher **risk-adjusted** returns than high-beta assets, because
leverage-constrained investors over-bid high-beta names. A **BAB** book — long the
low-beta legs levered to unit beta, short the high-beta legs, beta-neutral — should
therefore earn a positive alpha. Frazzini & Pedersen show the effect across many asset
classes, including **US Treasuries sorted by maturity**: a short-duration-vs-long-duration
BAB inside the government-bond curve.

We rebuild that Treasury-curve version from five liquid iShares Treasury ETFs that ladder
the curve by maturity:

    SHY  (1-3y)  <  IEI  (3-7y)  <  IEF  (7-10y)  <  TLH  (10-20y)  <  TLT  (20y+)

Duration rises monotonically SHY → TLT, and so does each ETF's **beta to a duration
factor** (the equal-weight mean of the five daily total-return series). SHY is the
lowest-beta (short-duration) leg; TLT the highest-beta (long-duration) leg. The BAB book
longs the low-beta legs (levered up to unit factor beta) and shorts the high-beta legs
(levered down to unit factor beta), so the net factor exposure is ~zero and any surviving
return is the low-risk alpha.

Two tapes, one schema (a dict ``{ticker: DataFrame[Close]}`` of daily total-return closes):

* ``synthetic_panel`` — a deterministic, offline generator. Five assets with a **fixed,
  monotone spread of factor betas** plus an alpha that (only when ``edge > 0``) is
  **negatively related to beta** — low-beta assets carry a positive alpha, high-beta a
  negative one, exactly the Frazzini-Pedersen pattern. ``edge = 0`` is the null: betas
  still spread out but carry no alpha, and the BAB book must find nothing.
* ``fetch`` — the real yfinance tape (SHY, IEI, IEF, TLH, TLT daily closes,
  ``auto_adjust=True`` total-return), cache-first into this study's own ``_cache/`` so the
  reproducible core never needs the network.

No look-ahead: each ETF's beta is estimated on a trailing window ending at the close of
day ``t-1`` (one ``shift``); the BAB book is held on day ``t``.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Five iShares Treasury ETFs laddering the curve by maturity (short -> long duration).
TICKERS = ["SHY", "IEI", "IEF", "TLH", "TLT"]

START = "2010-01-04"        # TLH/IEI both trade from 2007; 2010 gives a clean common start
AS_OF = "2026-06-30"        # last complete calendar month at publication (drop partial month)

CACHE_PATH = os.path.join(CACHE_DIR, "treasury_etf_closes.parquet")

__all__ = [
    "TICKERS", "START", "AS_OF", "CACHE_DIR", "CACHE_PATH",
    "fetch", "have_real", "load_panel", "load_series", "synthetic_panel",
    "fingerprint",
]


# --------------------------------------------------------------------------- #
# Real tape — Yahoo daily total-return closes, cache-first
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str | None = None, retries: int = 4,
          sleep_s: float = 2.0) -> pd.DataFrame:
    """Download the five Treasury-ETF total-return close series; cache the parquet.

    Retries up to ``retries`` times with a short sleep on a transient yfinance failure.
    ``auto_adjust=True`` gives dividend-reinvested (total-return) closes. The frame is a
    tz-naive daily DatetimeIndex with one column per ticker; cached under this study's
    own ``_cache/``.
    """
    import yfinance as yf  # lazy — never imported by the offline core

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                TICKERS, start=start, end=end, interval="1d",
                auto_adjust=True, progress=False, threads=True,
            )
            if raw is None or raw.empty:
                raise RuntimeError("yfinance returned no daily bars")
            closes = (
                raw["Close"].copy()
                if isinstance(raw.columns, pd.MultiIndex)
                else raw[["Close"]].copy()
            )
            closes = closes[[c for c in TICKERS if c in closes.columns]]
            if closes.shape[1] < len(TICKERS):
                raise RuntimeError(f"missing tickers: got {list(closes.columns)}")
            closes.index.name = "date"
            if closes.index.tz is not None:
                closes.index = closes.index.tz_localize(None)
            closes = closes.dropna(how="any")
            if closes.empty:
                raise RuntimeError("no fully-overlapping rows across the five ETFs")
            os.makedirs(CACHE_DIR, exist_ok=True)
            closes.to_parquet(CACHE_PATH)
            return closes
        except Exception as e:  # noqa: BLE001 — retry any transient failure
            last_err = e
            if attempt < retries - 1:
                time.sleep(sleep_s)
    raise RuntimeError(f"yfinance fetch failed after {retries} attempts: {last_err}")


def have_real() -> bool:
    return os.path.exists(CACHE_PATH)


def load_series(start: str = START, asof: str = AS_OF) -> pd.DataFrame:
    """Cached daily total-return closes as a DataFrame (index=date, columns=tickers),
    sliced to ``[start, asof]``. Reads the parquet directly — OFFLINE, no yfinance."""
    if not have_real():
        raise FileNotFoundError(
            f"No cached tape at {CACHE_PATH}. Call fetch() once to populate."
        )
    df = pd.read_parquet(CACHE_PATH)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[[c for c in TICKERS if c in df.columns]].sort_index()
    lo, hi = pd.Timestamp(start), pd.Timestamp(asof)
    df = df[(df.index >= lo) & (df.index <= hi)].dropna(how="any")
    return df


def load_panel(start: str = START, asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    """Cached panel as ``{ticker: DataFrame[Close]}``, sliced to ``[start, asof]``."""
    closes = load_series(start, asof)
    return {c: closes[[c]].rename(columns={c: "Close"}) for c in closes.columns}


def fingerprint(df: pd.DataFrame) -> str:
    """Short content fingerprint of the close panel, for the as-of stamp."""
    arr = np.ascontiguousarray(df.fillna(0.0).to_numpy(dtype=float))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic world — planted low-beta alpha (the positive control)
# --------------------------------------------------------------------------- #
# Fixed, monotone factor betas mimicking the SHY->TLT duration ladder.
SYN_BETAS = np.array([0.20, 0.55, 1.00, 1.45, 1.90])


def synthetic_panel(
    edge: float = 0.0,
    seed: int = 826,
    n_days: int = 2200,
    start: str = "2010-01-04",
    factor_vol: float = 0.006,
    idio_vol: float = 0.0020,
    drift: float = 0.0,
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded five-asset panel with a TUNABLE planted low-beta alpha.

    A single latent **duration factor** ``f_t`` (mean-zero) drives all five assets with a
    fixed, monotone spread of loadings :data:`SYN_BETAS` (0.20 → 1.90), mimicking the
    SHY → TLT beta ladder. Each asset ``i`` also carries a constant daily alpha

        alpha_i = edge * (beta_bar - beta_i)

    so — **only when ``edge > 0``** — the low-beta assets earn a positive alpha and the
    high-beta assets a negative one (the Frazzini-Pedersen low-risk pattern). The daily
    return is

        r[i,t] = drift + alpha_i + beta_i * f_t + idio_i,t

    ``edge = 0`` is the null: betas still spread out, but every asset has the same
    (zero) alpha, so the beta-neutral BAB book has nothing to earn. ``drift`` defaults to
    ``0`` so the series are **excess of the risk-free rate** — the framing the BAB formula
    ``(r-rf)/beta`` assumes; a non-zero common drift would be asymmetrically levered by the
    ``1/beta`` scaling and is exactly the artefact ``rf`` is meant to cancel. Business-day
    index, well below the pandas ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    betas = SYN_BETAS
    beta_bar = float(betas.mean())
    alpha = edge * (beta_bar - betas)          # low beta -> +alpha when edge>0

    f = rng.normal(0.0, factor_vol, n_days)    # the duration factor
    panel: dict[str, pd.DataFrame] = {}
    for i, tkr in enumerate(TICKERS):
        idio = rng.normal(0.0, idio_vol, n_days)
        r = drift + alpha[i] + betas[i] * f + idio
        close = 100.0 * np.cumprod(1.0 + r)
        panel[tkr] = pd.DataFrame({"Close": close}, index=idx)
    return panel
