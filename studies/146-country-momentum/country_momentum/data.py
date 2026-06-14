"""Data layer for Study 146 (Country-Momentum).

Two tapes, one shape (a monthly returns panel of country ETFs):

- ``synthetic_panel`` — a *deterministic, offline* generator. Each country carries a
  slowly-mean-reverting relative-drift ``alpha_i`` plus a common world factor and
  idiosyncratic noise. With ``mom_strength > 0`` the drift persists long enough that
  the previous 11-month return is a useful predictor of the next month — the structure
  the 12-1 momentum signal is trying to harvest. ``mom_strength = 0`` is the
  **null**: only noise and the world factor, so momentum sorts earn nothing. Seeded and
  deterministic.
- ``fetch_etfs`` — the real Yahoo! total-return adjusted closes for a 23-ETF country
  basket, **cache-only** by default (``fetch=False`` raises on a cache miss so tests
  never touch the network; ``fetch=True`` hits Yahoo and caches parquet).

Data note: country ETFs began launching in 1996 (EWJ, EWG, EWU, EWA, EWC, EWH); most
of the emerging-market ones appeared in 1999-2003. We require at least 10 years of
monthly history to enter the investable universe at each rebalance, so effective
coverage starts around 2008-2010 depending on the country.

No look-ahead baked in here — that discipline lives in ``strategy.py`` (the
12-1 momentum score is formed on closes up to *t*; the position is entered at *t+1*,
the next rebalance).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
STUDY_CACHE = os.path.join(HERE, "..", "_cache")

TRADING_MONTHS_PER_YEAR = 12

# ---------------------------------------------------------------------------
# The 23-ETF country basket (Yahoo tickers, total-return-adjusted closes).
# These span developed and emerging markets; most go back to 1996-2003.
# ---------------------------------------------------------------------------
COUNTRY_ETFS: dict[str, str] = {
    # Developed — early vintage (1996-2000)
    "EWJ": "Japan",
    "EWG": "Germany",
    "EWU": "United_Kingdom",
    "EWA": "Australia",
    "EWC": "Canada",
    "EWH": "Hong_Kong",
    "EWL": "Switzerland",
    "EWP": "Spain",
    "EWQ": "France",
    "EWI": "Italy",
    "EWD": "Sweden",
    "EWN": "Netherlands",
    "EWO": "Austria",
    "EWS": "Singapore",
    "EWK": "Belgium",
    # Emerging — launched 1999-2003
    "EWZ": "Brazil",
    "EWW": "Mexico",
    "EWY": "South_Korea",
    "EWT": "Taiwan",
    "EIS": "Israel",
    "EWM": "Malaysia",
    "THD": "Thailand",
    "EZA": "South_Africa",
}

TICKERS = list(COUNTRY_ETFS.keys())


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_months: int = 240,
    n_countries: int = 16,
    mom_strength: float = 0.04,
    phi: float = 0.94,
    world_vol: float = 0.05,
    idio_vol: float = 0.04,
    seed: int = 146,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible monthly country-return panel with a tunable momentum premium.

    Each country ``i`` has a persistent relative-drift component ``alpha_{i,t}``::

        alpha_{i,t} = phi * alpha_{i,t-1} + sqrt(1 - phi^2) * mom_strength * nu_{i,t}
        r_{i,t} = r_world[t] + alpha_{i,t} + idio_vol * eps_{i,t}

    With ``phi`` near 1 the drift is sticky, so a country that outperformed last year
    tends to outperform next year — the cross-sectional momentum the 12-1 signal hunts.
    ``mom_strength = 0`` is the null: ``alpha = 0`` for all, and momentum sorts earn ≈ 0.

    Returns ``(panel, truth)`` where ``truth`` records the planted parameters.
    The panel index is a monthly ``PeriodIndex``; columns are ``CTR00..CTR{n_countries-1}``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start="2005-01", periods=n_months, freq="M")

    # World factor
    world_ret = rng.normal(0.007, world_vol, n_months)

    # Persistent relative drifts per country
    innov_scale = mom_strength * np.sqrt(1.0 - phi**2)
    alpha = np.zeros((n_months, n_countries))
    nu = rng.standard_normal((n_months, n_countries))
    for t in range(1, n_months):
        alpha[t] = phi * alpha[t - 1] + innov_scale * nu[t]

    eps = rng.standard_normal((n_months, n_countries))
    rets = world_ret[:, None] + alpha + idio_vol * eps

    cols = [f"CTR{i:02d}" for i in range(n_countries)]
    panel = pd.DataFrame(rets, index=idx, columns=cols)
    truth = {
        "n_months": n_months,
        "n_countries": n_countries,
        "mom_strength": mom_strength,
        "phi": phi,
        "world_vol": world_vol,
        "idio_vol": idio_vol,
        "seed": seed,
        "has_momentum": mom_strength != 0.0,
    }
    return panel, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo country ETFs, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "country_momentum_monthly.parquet")


def fetch_etfs(
    tickers: list[str] | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    min_years: float = 8.0,
) -> pd.DataFrame:
    """Monthly total-return (split/dividend-adjusted) returns for the country ETF basket.

    **Cache-only** by default (``fetch=False``): returns the cached parquet if present,
    raises :exc:`FileNotFoundError` if not, so offline paths and tests never touch the
    network. With ``fetch=True``, downloads all tickers from Yahoo, resamples to
    month-end closes, computes log-return differences, and writes the parquet.

    Only tickers with at least ``min_years`` of monthly history are retained. The
    returned frame is indexed by ``pandas.Period`` (monthly) with ticker columns.
    """
    if tickers is None:
        tickers = TICKERS
    path = _cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached country-ETF tape at {path}. "
                f"Call fetch_etfs(fetch=True) once to populate the cache."
            )
        df = pd.read_parquet(path)
        # Restore Period index if stored as timestamp
        if not isinstance(df.index, pd.PeriodIndex):
            df.index = pd.PeriodIndex(df.index, freq="M")
        return df

    import yfinance as yf  # lazy: only when we go to the network

    closes: dict[str, pd.Series] = {}
    for tk in tickers:
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
            years = (s.index[-1] - s.index[0]).days / 365.25
            if years >= min_years:
                closes[tk] = s.astype(float)
        except Exception:
            continue

    if not closes:
        raise RuntimeError("yfinance returned no data for any country ETF ticker")

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
    return rets


def fingerprint(panel: pd.DataFrame) -> str:
    """A short content fingerprint of the monthly return panel (first column), for the as-of stamp."""
    first = panel.iloc[:, 0].dropna().to_numpy()
    h = hashlib.sha1(np.ascontiguousarray(first).tobytes())
    return h.hexdigest()[:12]
