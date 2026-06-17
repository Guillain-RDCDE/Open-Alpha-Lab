"""Data layer for Study 223 (Same-Month Seasonality).

Two tapes, one shape (a month x ticker panel of monthly adjusted closes):

- ``synthetic_panel`` -- a *deterministic, offline* generator.  A ``premium``
  knob plants the exact same-calendar-month seasonality that Heston & Sadka
  (2008) claim: stocks that have historically outperformed in a given calendar
  month continue to outperform in that same calendar month by a known amount.
  ``premium = 0`` is the null: the same-month historical rank carries no
  forward information.

- ``fetch_monthly`` -- the real Yahoo! monthly close panel for a large-cap
  universe (via yfinance), cache-only by default so the test-suite and the
  reproducible core never touch the network.  The cache lives under
  ``_cache/sms_monthly.parquet`` (a date x ticker frame of adjusted monthly
  closes, rows = month-end dates).

**Survivorship-bias caveat**: the ticker list is a fixed basket of large-caps
that survived to 2026.  Positive results from the real tape are upper bounds --
the true live effect is weaker.  This is named on the Signal axis.

No look-ahead: at month-end t, the signal predicts the return earned in month
t+1 (calendar month m+1).  The signal is the average of all past returns in
calendar month m+1, using data strictly before t (ending at t-11 months or
earlier).  The minimum history required is ``min_years`` prior occurrences of
the target calendar month.  Positions are formed at month-end close and held
through the next month (one-month execution lag).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(_HERE, "..", "_cache")

MONTHLY_CACHE = os.path.join(DEFAULT_CACHE, "sms_monthly.parquet")

# Fixed large-cap basket (survivorship-biased, current members projected backwards)
LARGE_CAP_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B",
    "UNH", "JNJ", "JPM", "V", "PG", "MA", "HD", "XOM", "CVX", "MRK",
    "ABBV", "PEP", "KO", "AVGO", "COST", "WMT", "LLY", "TMO", "MCD",
    "CSCO", "ACN", "ABT", "CRM", "DHR", "NEE", "TXN", "NFLX", "QCOM",
    "AMD", "ORCL", "INTC", "CMCSA", "ADBE", "PFE", "T", "DIS", "RTX",
    "HON", "BAC", "GE", "CAT", "GS", "MS", "IBM", "AMGN", "GILD", "MDLZ",
    "AXP", "BLK", "MMM", "DE", "SHW", "CI", "SPGI", "MO", "USB", "SO",
    "DUK", "EMR", "AON", "TGT", "LOW", "F", "GM", "FDX", "UPS", "LMT",
    "NOC", "BA", "GD", "WFC", "C", "AIG", "MET", "PRU", "AFL", "TRV",
]


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 100,
    n_months: int = 240,
    premium: float = 0.02,
    base_ret: float = 0.008,
    ret_vol: float = 0.06,
    min_years: int = 5,
    seed: int = 223,
) -> tuple[pd.DataFrame, dict]:
    """A firm x month panel with a known same-calendar-month seasonality effect.

    Each stock j is assigned a fixed calendar-month profile ``alpha_j_m`` drawn
    from N(0, premium).  The monthly return is::

        base_ret + alpha_j[month(t)] + noise_t

    where ``alpha_j[m]`` is stock j's idiosyncratic monthly seasonal for
    calendar month m, normalized to sum to zero across calendar months.
    When ``premium = 0``, all alphas are zero and same-month ranking carries
    no signal.

    The planted premium is detectable by the Heston-Sadka strategy: sort
    stocks each month by their average same-calendar-month past return
    (formation window = ``min_years`` prior occurrences); long top decile,
    short bottom decile.

    Returns ``(price_df, truth)`` where:
    - ``price_df`` is a (month x firm) frame of cumulative prices.
    - ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range("2000-01-31", periods=n_months, freq="ME")
    firms = [f"F{j:03d}" for j in range(n_firms)]

    # Each firm gets a 12-vector of monthly alphas drawn from N(0, premium).
    # We de-mean each firm's profile so the total expected return is base_ret.
    firm_month_alpha = rng.normal(0.0, premium, (n_firms, 12))
    firm_month_alpha -= firm_month_alpha.mean(axis=1, keepdims=True)  # zero mean

    # Build return matrix
    month_idx = np.array([d.month - 1 for d in months])  # 0-based calendar month
    noise = rng.normal(0.0, ret_vol, (n_months, n_firms))

    ret_mat = np.zeros((n_months, n_firms))
    for t in range(n_months):
        m = month_idx[t]
        ret_mat[t] = base_ret + firm_month_alpha[:, m] + noise[t]

    # Cumulative prices from 100
    log_ret = np.log1p(ret_mat)
    cum_log = np.cumsum(log_ret, axis=0)
    price_arr = 100.0 * np.exp(cum_log)
    price_df = pd.DataFrame(price_arr, index=months, columns=firms)

    truth = {
        "premium": premium,
        "base_ret": base_ret,
        "ret_vol": ret_vol,
        "n_firms": n_firms,
        "n_months": n_months,
        "min_years": min_years,
        "seed": seed,
    }
    return price_df, truth


# ---------------------------------------------------------------------------
# Real tape -- Yahoo monthly closes, cache-only by default
# ---------------------------------------------------------------------------
def fetch_monthly(
    fetch: bool = False,
    cache_path: str = MONTHLY_CACHE,
) -> pd.DataFrame:
    """Monthly adjusted close panel for large-cap names; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is
    cached as a parquet under ``_cache/sms_monthly.parquet``). Columns are
    tickers, index is month-end dates (pd.DatetimeIndex at month-end frequency).
    Returns an empty DataFrame if the cache is absent and ``fetch=False`` is set
    -- callers must guard with ``os.path.exists(cache_path)`` before calling.

    **Survivorship bias**: the ticker universe is a fixed large-cap basket
    projected backwards.  All positive results are upper bounds.
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached monthly panel at {cache_path}. "
                "Call fetch_monthly(fetch=True) once to populate the cache."
            )
        prices = pd.read_parquet(cache_path)
        return prices

    # ------------------------------------------------------------------
    # Live fetch: yfinance monthly closes for the large-cap basket
    # ------------------------------------------------------------------
    import yfinance as yf  # lazy import: network only on fetch=True

    tickers = LARGE_CAP_TICKERS

    raw = yf.download(
        tickers,
        start="1993-01-01",
        interval="1mo",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no monthly data")

    # Extract 'Close' level from MultiIndex columns
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            closes = raw["Close"]
        else:
            closes = raw.xs("Close", axis=1, level=0)
    else:
        closes = raw

    # Month-end alignment
    closes.index = pd.to_datetime(closes.index)
    closes.index = closes.index + pd.offsets.MonthEnd(0)
    closes = closes.sort_index()

    # Drop tickers with fewer than 84 months (7 years) of data
    min_obs = 84
    closes = closes.loc[:, closes.count() >= min_obs]

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    closes.to_parquet(cache_path)
    print(f"Cached monthly panel: {closes.shape[0]} months x {closes.shape[1]} tickers -> {cache_path}")
    return closes


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a prices frame (last row), for the as-of stamp."""
    arr = df.iloc[-1].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
