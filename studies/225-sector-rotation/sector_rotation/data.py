"""Data layer for Study 225 (Sector-Rotation).

Two tapes, one shape (a monthly returns panel of sector ETFs):

- ``synthetic_panel`` — a *deterministic, offline* generator. Each sector carries a
  slowly-mean-reverting relative-drift ``alpha_i`` plus a common market factor and
  idiosyncratic noise. With ``mom_strength > 0`` the drift persists long enough that
  the previous momentum window is a useful predictor — the structure
  the momentum signal is trying to harvest. ``mom_strength = 0`` is the
  **null**: only noise and the market factor, so momentum sorts earn nothing. Seeded and
  deterministic.
- ``fetch_etfs`` — the real Yahoo! total-return adjusted closes for the 11 SPDR sector
  ETFs plus SPY, **cache-only** by default (``fetch=False`` raises on a cache miss so
  tests never touch the network; ``fetch=True`` hits Yahoo and caches parquet).

Data note: the SPDR sector ETFs launched in December 1998. SPY launched in January 1993
but we only use it as a benchmark. Effective data window starts 1999-01.

No look-ahead baked in here — that discipline lives in ``strategy.py`` (the momentum
score is formed on closes up to *t*; the position is entered at *t+1*,
the next rebalance).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache")

TRADING_MONTHS_PER_YEAR = 12

# ---------------------------------------------------------------------------
# The 11 SPDR sector ETFs + SPY benchmark (Yahoo tickers, total-return-adjusted).
# All 11 sector SPDRs launched in December 1998 (XLB, XLE, XLF, XLI, XLK, XLP,
# XLU, XLV, XLY, XLC launched 2018 replacing old telecom, XLRE launched 2015).
# We use the original 9 that go back to 1998/1999, plus XLRE/XLC with shorter history.
# ---------------------------------------------------------------------------
SECTOR_ETFS: dict[str, str] = {
    "XLK": "Technology",
    "XLV": "Health_Care",
    "XLF": "Financials",
    "XLY": "Consumer_Discretionary",
    "XLI": "Industrials",
    "XLP": "Consumer_Staples",
    "XLE": "Energy",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real_Estate",
    "XLC": "Communication_Services",
}

# SPY is the benchmark; not in the rotation universe
BENCHMARK_TICKER = "SPY"
TICKERS = list(SECTOR_ETFS.keys())


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_months: int = 240,
    n_sectors: int = 9,
    mom_strength: float = 0.04,
    phi: float = 0.92,
    market_vol: float = 0.04,
    idio_vol: float = 0.035,
    seed: int = 225,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible monthly sector-return panel with a tunable momentum premium.

    Each sector ``i`` has a persistent relative-drift component ``alpha_{i,t}``::

        alpha_{i,t} = phi * alpha_{i,t-1} + sqrt(1 - phi^2) * mom_strength * nu_{i,t}
        r_{i,t} = r_market[t] + alpha_{i,t} + idio_vol * eps_{i,t}

    With ``phi`` near 1 the drift is sticky, so a sector that outperformed last year
    tends to outperform next year — the cross-sectional momentum the signal hunts.
    ``mom_strength = 0`` is the null: ``alpha = 0`` for all, and momentum sorts earn ~0.

    Returns ``(panel, truth)`` where ``truth`` records the planted parameters.
    The panel index is a monthly ``PeriodIndex``; columns are ``SEC00..SEC{n_sectors-1}``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start="1999-01", periods=n_months, freq="M")

    # Market factor (positive drift — equities go up)
    market_ret = rng.normal(0.008, market_vol, n_months)

    # Persistent relative drifts per sector
    innov_scale = mom_strength * np.sqrt(1.0 - phi**2)
    alpha = np.zeros((n_months, n_sectors))
    nu = rng.standard_normal((n_months, n_sectors))
    for t in range(1, n_months):
        alpha[t] = phi * alpha[t - 1] + innov_scale * nu[t]

    eps = rng.standard_normal((n_months, n_sectors))
    rets = market_ret[:, None] + alpha + idio_vol * eps

    cols = [f"SEC{i:02d}" for i in range(n_sectors)]
    panel = pd.DataFrame(rets, index=idx, columns=cols)
    truth = {
        "n_months": n_months,
        "n_sectors": n_sectors,
        "mom_strength": mom_strength,
        "phi": phi,
        "market_vol": market_vol,
        "idio_vol": idio_vol,
        "seed": seed,
        "has_momentum": mom_strength != 0.0,
    }
    return panel, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo sector ETFs, cache to study-local _cache/
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "sector_rotation_monthly.parquet")


def fetch_etfs(
    tickers: list[str] | None = None,
    benchmark: str = BENCHMARK_TICKER,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Monthly total-return (split/dividend-adjusted) returns for sector ETFs and SPY.

    **Cache-only** by default (``fetch=False``): returns the cached parquet if present,
    raises :exc:`FileNotFoundError` if not, so offline paths and tests never touch the
    network. With ``fetch=True``, downloads all tickers from Yahoo, resamples to
    month-end closes, computes simple returns, and writes the parquet.

    Returns
    -------
    panel : pd.DataFrame
        Monthly returns for the 11 sector ETFs (rows = months, cols = tickers).
        Indexed by ``pandas.PeriodIndex`` (monthly).
    spy : pd.Series
        Monthly returns for SPY benchmark, same index alignment.
    """
    if tickers is None:
        tickers = TICKERS
    path = _cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached sector-ETF tape at {path}. "
                f"Call fetch_etfs(fetch=True) once to populate the cache."
            )
        df = pd.read_parquet(path)
        # Restore Period index if stored as timestamp
        if not isinstance(df.index, pd.PeriodIndex):
            df.index = pd.PeriodIndex(df.index, freq="M")
        spy = df.pop("SPY") if "SPY" in df.columns else pd.Series(dtype=float)
        return df, spy

    import yfinance as yf  # lazy: only when we go to the network

    all_tickers = tickers + ([benchmark] if benchmark not in tickers else [])
    closes: dict[str, pd.Series] = {}
    for tk in all_tickers:
        try:
            raw = yf.download(
                tk, period="max", interval="1d", auto_adjust=True, progress=False
            )
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            s = raw["Close"].dropna()
            s.index = pd.DatetimeIndex(s.index).tz_localize(None)
            closes[tk] = s.astype(float)
        except Exception:
            continue

    if not closes:
        raise RuntimeError("yfinance returned no data for any sector ETF ticker")

    prices = pd.DataFrame(closes).sort_index()
    # Resample to month-end and compute simple returns
    monthly = prices.resample("ME").last()
    rets = monthly.pct_change().dropna(how="all")
    rets.index = pd.PeriodIndex(rets.index, freq="M")
    rets.index.name = "date"

    os.makedirs(cache_dir, exist_ok=True)
    # Store as timestamp for parquet compatibility
    rets_ts = rets.copy()
    rets_ts.index = rets_ts.index.to_timestamp()
    rets_ts.to_parquet(path)

    spy = rets.pop("SPY") if "SPY" in rets.columns else pd.Series(dtype=float)
    # Only keep the sector tickers (not benchmark)
    panel = rets[[t for t in tickers if t in rets.columns]]
    return panel, spy


def fingerprint(panel: pd.DataFrame) -> str:
    """A short content fingerprint of the monthly return panel (first column)."""
    first = panel.iloc[:, 0].dropna().to_numpy()
    h = hashlib.sha1(np.ascontiguousarray(first).tobytes())
    return h.hexdigest()[:12]
