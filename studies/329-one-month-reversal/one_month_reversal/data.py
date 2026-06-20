"""Data layer for Study 329 (One-Month-Reversal).

The Jegadeesh (1990) short-horizon cross-sectional reversal: rank stocks each month by
*last month's* return; the bottom (losers) are claimed to beat the top (winners) next
month. Distinct from:

- **Study 196 (Long-Term-Reversal)** -- De Bondt-Thaler, 36-60 *month* formation.
  329 is the **1-month** horizon, the opposite end of the autocorrelation spectrum.
- **Study 33 (Slingshot)** -- a *daily*-rebalanced, dollar-neutral, fade-against-peers
  book. 329 is the **monthly-rebalanced quintile** specification of Jegadeesh's Table I.
- **Study 32 (Rip-Tide)** -- intraday/short contrarian on *futures*, not a stock cross-section.

Two tapes, one shape (a month x ticker panel of monthly closes):

- ``synthetic_panel`` -- a *deterministic, offline* generator. A ``reversal`` knob plants
  exactly the short-horizon cross-sectional anti-persistence Jegadeesh claims: last
  month's biggest losers earn the most next month, by a known amount. ``reversal = 0`` is
  the null -- last month's rank carries no information. This is the study's null in a bottle.

- ``fetch_monthly`` -- the real Yahoo! monthly close panel for the S&P 500 universe, via
  the cache-first ``load_real()``. Cache-only by default so the test-suite and the
  reproducible core never touch the network. The headline run reuses the **shared S&P 500
  monthly panel** built by Study 196 (``../../196-long-term-reversal/_cache/ltr_monthly.parquet``)
  when present, falling back to this study's own ``_cache/omr_monthly.parquet``.

**Survivorship-bias caveat**: the ticker list is the *current* S&P 500 membership
projected backwards (via ``quantlab.universe.sp500_symbols``). Every firm survived to
2026, so positive real-tape results are upper bounds. This is named on the Signal axis.

No look-ahead is baked in here -- that discipline lives in ``strategy.py``. The signal
for the holding month t is last month's return (known at the close of month t-1), and the
return earned is that of month t. A ``skip`` knob in ``strategy.trailing_return`` exposes
the microstructure-safe 1-month-gap variant.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))

# This study's own cache target (populated by examples/verify.py --fetch).
MONTHLY_CACHE = os.path.join(DEFAULT_CACHE, "omr_monthly.parquet")

# The shared S&P 500 monthly panel built by Study 196 -- reused cache-first so the
# headline run does not duplicate a 1.5 MB download.
SHARED_LTR_CACHE = os.path.abspath(
    os.path.join(REPO_ROOT, "studies", "196-long-term-reversal", "_cache", "ltr_monthly.parquet")
)

# Last fully-complete month pinned for the headline run (partial months are dropped).
AS_OF_LAST_MONTH = "2026-05-31"


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 150,
    n_months: int = 300,
    reversal: float = 0.05,
    base_ret: float = 0.008,
    ret_vol: float = 0.07,
    seed: int = 329,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """A firm x month panel with a known *one-month* cross-sectional reversal.

    Each month, firms are ranked by *last month's* log-return (the signal). The next
    month's return is::

        base_ret + reversal * z(-last_month_return) + noise

    where ``z(.)`` normalises the negated last-month return to zero mean / unit variance
    cross-sectionally. Negating means last month's biggest *losers* (lowest return) get
    the highest z-score and the highest expected return -- the Jegadeesh (1990) thesis.

    - ``reversal = 0``  -> the null: last month's rank carries no information; the
      loser-minus-winner spread is zero in expectation.
    - ``reversal > 0``  -> a one-month reversal the loser quintile can harvest.
    - ``reversal < 0``  -> short-horizon momentum, working *against* the reversal rule.

    Returns ``(price_df, fwd_ret_df, truth)`` where:
    - ``price_df`` is a (month x firm) frame of cumulative prices (base 100).
    - ``fwd_ret_df`` is a (month x firm) frame of next-month log-returns, aligned so
      ``fwd_ret_df.iloc[t]`` is the return earned in calendar month t+1.
    - ``truth`` records the planted parameters.

    A ``pd.period_range`` is used for the decorative monthly label (then converted to
    timestamps for the last complete month) so we never risk a ``date_range(periods=BIG)``
    timestamp overflow.
    """
    rng = np.random.default_rng(seed)
    # period_range -> month-end timestamps, well under any ns overflow horizon.
    months = pd.period_range("2000-01", periods=n_months, freq="M").to_timestamp(how="end").normalize()
    firms = [f"F{j:03d}" for j in range(n_firms)]

    # Phase 1: plain random-walk log-returns.
    noise = rng.normal(0.0, ret_vol, (n_months, n_firms))
    log_ret_base = base_ret + noise  # (n_months, n_firms)

    # Phase 2: from month 1 on, add a reversal premium driven by month t-1's return.
    #   extra[t, j] = reversal * z(-r[t-1, j])
    # rebuilt iteratively so the planted premium propagates into prices and into the
    # derived forward returns the strategy infers from prices.pct_change.
    log_ret = np.zeros_like(log_ret_base)
    log_ret[0] = log_ret_base[0]
    for t in range(1, n_months):
        prev = log_ret[t - 1]
        neg = -prev
        z = (neg - neg.mean()) / (neg.std() + 1e-9)
        log_ret[t] = log_ret_base[t] + reversal * z

    cum_log = np.cumsum(log_ret, axis=0)
    price_arr = 100.0 * np.exp(cum_log)
    price_df = pd.DataFrame(price_arr, index=months, columns=firms)

    # fwd_ret_df.iloc[t] = log return earned in month t+1.
    fwd_raw = np.full((n_months, n_firms), np.nan)
    fwd_raw[:-1] = log_ret[1:]
    fwd_ret_df = pd.DataFrame(fwd_raw, index=months, columns=firms)

    truth = {
        "reversal": reversal,
        "base_ret": base_ret,
        "ret_vol": ret_vol,
        "n_firms": n_firms,
        "n_months": n_months,
        "seed": seed,
    }
    return price_df, fwd_ret_df, truth


# ---------------------------------------------------------------------------
# Real tape -- cache-first S&P 500 monthly closes
# ---------------------------------------------------------------------------
def load_real(cache_path: str | None = None) -> pd.DataFrame:
    """Cache-first monthly S&P 500 close panel; never touches the network.

    Resolution order (graceful fallback):
      1. an explicit ``cache_path`` if given and present,
      2. this study's own ``_cache/omr_monthly.parquet``,
      3. the shared Study-196 panel ``ltr_monthly.parquet``.

    Raises ``FileNotFoundError`` if none exist (the synthetic core covers offline CI).
    The returned frame is trimmed to the last fully-complete month (``AS_OF_LAST_MONTH``)
    so the headline run never includes a partial month.

    **Survivorship bias**: the universe is the current S&P 500 projected backwards. All
    positive results are upper bounds.
    """
    candidates = [c for c in (cache_path, MONTHLY_CACHE, SHARED_LTR_CACHE) if c]
    for path in candidates:
        if os.path.exists(path):
            prices = pd.read_parquet(path)
            prices = prices.loc[:AS_OF_LAST_MONTH]
            if prices.index.tz is not None:
                prices.index = prices.index.tz_localize(None)
            return prices
    raise FileNotFoundError(
        "No cached monthly panel found. Tried: "
        + ", ".join(candidates)
        + ". Run examples/verify.py --fetch once, or rely on the synthetic core."
    )


def fetch_monthly(
    fetch: bool = False,
    cache_path: str = MONTHLY_CACHE,
) -> pd.DataFrame:
    """Monthly adjusted-close panel for S&P 500 names; cache-only unless ``fetch=True``.

    With ``fetch=False`` this is a thin wrapper over :func:`load_real` (cache-first,
    offline). With ``fetch=True`` it goes to ``yfinance`` for the current S&P 500 basket
    (via ``quantlab.universe.sp500_symbols(allow_survivorship_bias=True)``) and caches the
    result. Columns are tickers; index is month-end dates.

    **Survivorship bias**: the universe is the current S&P 500 projected backwards.
    """
    if not fetch:
        return load_real(cache_path=cache_path)

    import yfinance as yf  # lazy: network only on fetch=True

    try:
        from quantlab.universe import sp500_symbols

        tickers = sp500_symbols(allow_survivorship_bias=True)
    except Exception:
        tickers = [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B",
            "UNH", "JNJ", "JPM", "V", "PG", "MA", "HD", "XOM", "CVX", "MRK",
            "ABBV", "PEP", "KO", "AVGO", "COST", "WMT", "LLY", "TMO", "MCD",
            "CSCO", "ACN", "ABT", "CRM", "DHR", "NEE", "TXN", "NFLX", "QCOM",
            "AMD", "ORCL", "INTC", "CMCSA", "ADBE", "PFE", "T", "DIS", "RTX",
            "HON", "BAC", "GE", "CAT", "GS",
        ]

    raw = yf.download(
        tickers, start="1990-01-01", interval="1mo",
        auto_adjust=True, progress=False, group_by="column",
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no monthly data")
    if isinstance(raw.columns, pd.MultiIndex):
        closes = (
            raw["Close"] if "Close" in raw.columns.get_level_values(0)
            else raw.xs("Close", axis=1, level=0)
        )
    else:
        closes = raw
    closes.index = pd.to_datetime(closes.index) + pd.offsets.MonthEnd(0)
    closes = closes.sort_index()
    closes = closes.loc[:, closes.count() >= 60]  # >=60 months of history

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    closes.to_parquet(cache_path)
    print(f"Cached monthly panel: {closes.shape[0]} months x {closes.shape[1]} tickers -> {cache_path}")
    return closes.loc[:AS_OF_LAST_MONTH]


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a prices frame (last row), for the as-of stamp."""
    arr = df.iloc[-1].dropna().to_numpy(dtype=float)
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]
