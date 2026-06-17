"""Data layer for Study 218 (SPAC-Performance).

Two tapes, one protocol:

- ``synthetic_daily`` — a *deterministic, offline* generator. An alpha knob plants a
  known excess return so tests can assert the engine recovers it when present and reads
  near-zero when absent. ``alpha_ann=0`` is the fair null (the de-SPAC basket matches
  the market after beta-scaling). ``alpha_ann<0`` replicates the hypothesis that SPACs
  structurally underperform after merger. This is the study's null in a bottle.

- ``fetch_daily`` — the real Yahoo! daily adjusted-close tape, cache-only by default so
  the test suite and the reproducible core never touch the network.
  SPAK (the Defiance Next Gen SPAC Derived ETF) launched 2020-10-01 and was delisted
  2022-09-01; the de-SPAC basket (9 names: LCID, RIVN, OPEN, PSFE, CLOV, SKLZ, DKNG,
  SPCE, QS) covers 2021-11-10 onward (RIVN IPO date as the common start).

Survivorship note: the basket consists of SPACs that survived to listing. Delisted
de-SPACs (NKLA, OTRK, GNUS, HYLIION, PTRA…) are absent, biasing the basket
*upward* relative to the true SPAC cohort. The real results are therefore conservative:
the true underperformance is even larger.

No look-ahead: returns are close_{t}/close_{t-1} - 1 computed on arrival.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# SPAK: Defiance Next Gen SPAC Derived ETF (launched 2020-10-01, delisted 2022-09-01)
SPAK_START = "2020-10-01"
SPAK_END = "2022-09-03"  # last available day

# De-SPAC basket: post-merger names; RIVN IPO date sets the common start
BASKET_START = "2021-11-10"
DE_SPAC_TICKERS = ["LCID", "RIVN", "OPEN", "PSFE", "CLOV", "SKLZ", "DKNG", "SPCE", "QS"]

# All tickers that need fetching
ALL_TICKERS = ["SPAK", "SPY"] + DE_SPAC_TICKERS


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 500,
    alpha_ann: float = 0.0,
    beta: float = 1.5,
    market_vol_ann: float = 0.18,
    idio_vol_ann: float = 0.45,
    start: str = "2020-01-02",
    seed: int = 218,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily price frame mimicking a de-SPAC basket vs SPY.

    The market (SPY-like) follows a random walk with annual vol ``market_vol_ann``.
    The SPAC fund (basket-like) is generated as:

        r_spac_t = alpha_daily + beta * r_market_t + eps_t

    where ``alpha_daily = (1 + alpha_ann)^(1/252) - 1`` and ``eps_t`` is i.i.d.
    normal of annual vol ``idio_vol_ann / sqrt(252)``. High idio_vol (45%/yr) reflects
    the extreme individual-name volatility of de-SPACs (LCID, RIVN etc. have moved
    ±20% on single days). Default beta=1.5 mirrors the empirical OLS estimate from the
    real basket (beta ~2.0 but corrected for the survivorship-biased basket drift).
    ``alpha_ann=0`` is a fair null; ``alpha_ann<0`` replicates the structural-dilution
    hypothesis.

    Returns ``(prices, truth)`` where ``prices`` is a DataFrame with columns
    ``['spac', 'spy']`` and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    sigma_mkt = market_vol_ann / np.sqrt(252)
    mu_mkt = 0.07 / 252
    r_mkt = rng.normal(mu_mkt, sigma_mkt, n_days)

    alpha_daily = (1.0 + alpha_ann) ** (1.0 / 252) - 1.0
    sigma_idio = idio_vol_ann / np.sqrt(252)
    r_spac = alpha_daily + beta * r_mkt + rng.normal(0, sigma_idio, n_days)

    spy_px = 100.0 * np.cumprod(1.0 + r_mkt)
    spac_px = 100.0 * np.cumprod(1.0 + r_spac)

    prices = pd.DataFrame({"spac": spac_px, "spy": spy_px}, index=idx)
    prices.index.name = "Date"

    truth = {
        "alpha_ann": alpha_ann,
        "alpha_daily": alpha_daily,
        "beta": beta,
        "market_vol_ann": market_vol_ann,
        "idio_vol_ann": idio_vol_ann,
        "n_days": n_days,
        "seed": seed,
    }
    return prices, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "").lower()
    return os.path.join(cache_dir, f"{safe}_daily.parquet")


def fetch_daily(
    ticker: str,
    start: str = SPAK_START,
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily adjusted-close OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as
    a parquet under ``_cache/``). ``auto_adjust=True`` ensures total-return pricing
    (dividends reinvested). SPAK had an expense ratio of ~0.45%; SPY has ~0.09% ER
    — both are already baked into adjusted Yahoo prices.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call fetch_daily({ticker!r}, fetch=True) once to populate the cache."
            )
        df = pd.read_parquet(path)
    else:
        import yfinance as yf

        raw = yf.download(
            ticker, start=start, end=end, interval="1d",
            auto_adjust=True, progress=False,
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.rename(columns=str.lower)
        df.index.name = "Date"
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(path)

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def load_spak_vs_spy(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Load aligned daily close prices for SPAK and SPY over SPAK's live window.

    SPAK (Defiance Next Gen SPAC ETF, ER ~0.45%) launched 2020-10-01 and was delisted
    2022-09-01, giving 484 trading days (~1.92 years) for the ETF-level comparison.
    Returns a DataFrame with columns ['SPAK', 'SPY'], aligned on common trading days.
    """
    spak_df = fetch_daily("SPAK", fetch=fetch, cache_dir=cache_dir)
    spy_df = fetch_daily("SPY", fetch=fetch, cache_dir=cache_dir)

    spak_close = _get_close(spak_df).rename("SPAK")
    spy_close = _get_close(spy_df).rename("SPY")

    prices = pd.concat([spak_close, spy_close], axis=1).dropna()
    prices.index.name = "Date"
    return prices


def load_basket_vs_spy(
    tickers: list[str] | None = None,
    start: str = BASKET_START,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Load an equal-weight de-SPAC basket and SPY from ``start`` onward.

    Returns a DataFrame with columns ['BASKET', 'SPY'] of daily equal-weight
    arithmetic returns, rebased to price-100 at the common start.

    Survivorship warning: delisted names (NKLA etc.) are absent; the basket
    represents *surviving* de-SPACs only, biasing results upward (less negative
    than the true cohort).
    """
    if tickers is None:
        tickers = DE_SPAC_TICKERS

    spy_df = fetch_daily("SPY", fetch=fetch, cache_dir=cache_dir)
    spy_close = _get_close(spy_df).rename("SPY")

    frames = {}
    for t in tickers:
        df = fetch_daily(t, fetch=fetch, cache_dir=cache_dir)
        frames[t] = _get_close(df)

    basket_prices = pd.DataFrame(frames)
    # Common window: all basket names present + SPY
    common = basket_prices.dropna().index.intersection(spy_close.dropna().index)
    common = common[common >= pd.Timestamp(start)]

    basket_sub = basket_prices.loc[common]
    spy_sub = spy_close.loc[common]

    # Equal-weight daily returns, then rebase to 100
    basket_rets = basket_sub.pct_change()
    basket_ew_ret = basket_rets.mean(axis=1)

    spy_rets = spy_sub.pct_change()

    # Combine as price series starting at 100
    basket_px = (100.0 * (1 + basket_ew_ret).cumprod()).rename("BASKET")
    spy_px = (100.0 * (1 + spy_rets).cumprod()).rename("SPY")

    prices = pd.concat([basket_px, spy_px], axis=1).dropna()
    prices.index.name = "Date"
    return prices


def _get_close(df: pd.DataFrame) -> pd.Series:
    col = "close" if "close" in df.columns else "Close"
    return df[col]


def fingerprint(series: pd.Series) -> str:
    """A short content fingerprint of a price series, for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(series.to_numpy()).tobytes())
    return h.hexdigest()[:12]
