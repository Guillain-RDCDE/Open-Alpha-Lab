"""Data layer for Study 196 (Long-Term-Reversal).

Two tapes, one shape (a month × ticker panel of cumulative past returns and
forward holding-period returns):

- ``synthetic_panel`` — a *deterministic, offline* generator. A ``reversal``
  knob plants the exact cross-sectional mean-reversion that De Bondt & Thaler
  (1985) claim: past long-horizon losers earn more than past winners by a known
  amount. ``reversal = 0`` is the null: the trailing rank carries no information
  and the bottom-quintile (losers) should not beat the top-quintile (winners).
  This is the study's null in a bottle.

- ``fetch_monthly`` — the real Yahoo! monthly close panel for the S&P 500
  universe (via yfinance), cache-only by default so the test-suite and the
  reproducible core never touch the network. The cache lives under
  ``_cache/ltr_monthly.parquet`` (a date × ticker frame of adjusted monthly
  closes, rows = month-end dates).

**Survivorship-bias caveat**: the ticker list is the *current* S&P 500
membership projected backwards (via ``quantlab.universe.sp500_symbols``).
Every firm in the panel survived to 2026. Positive results from the real tape
are upper bounds — the true live effect is weaker. This is named on the Signal
axis.

No look-ahead: the formation window for month t uses data strictly up to
month t-1 (returns t−36 to t−1 for 36-month formation, or t−60 to t−1 for
60-month). The first holding month is t. Positions are formed at month-end and
held for ``hold_months`` calendar months before rebalancing.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(_HERE, "..", "_cache")
SHARED_CACHE = os.path.join(REPO_ROOT, "_cache")

MONTHLY_CACHE = os.path.join(DEFAULT_CACHE, "ltr_monthly.parquet")


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 150,
    n_months: int = 300,
    reversal: float = 0.04,
    base_ret: float = 0.008,
    ret_vol: float = 0.07,
    formation: int = 36,
    hold: int = 12,
    seed: int = 196,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """A firm × month panel with a known long-term-reversal effect.

    Each month, firms are ranked by their trailing ``formation``-month cumulative
    log-return (the *signal*). The next ``hold`` months' returns are::

        base_ret + reversal * z(−trailing_rank) + noise

    where ``z(·)`` normalises the reversed rank to zero mean / unit variance
    cross-sectionally. Negating the rank means the biggest *losers* (lowest
    trailing return) get the highest z-score and the highest expected forward
    return — that is the De Bondt-Thaler overreaction thesis.

    ``reversal = 0`` is the null: the loser minus winner spread should be
    zero in expectation.

    Returns ``(price_df, fwd_ret_df, truth)`` where:
    - ``price_df`` is a (month × firm) frame of cumulative prices (log-return
      cumulated from a base of 100).
    - ``fwd_ret_df`` is a (month × firm) frame of next-month log-returns,
      aligned so ``fwd_ret_df.iloc[t]`` is the return in calendar month t+1.
    - ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range("2000-01-31", periods=n_months, freq="ME")
    firms = [f"F{j:03d}" for j in range(n_firms)]

    # Two-phase construction so the planted reversal is visible in BOTH the price
    # panel AND the derived forward returns (which is what strategy.quintile_returns
    # uses from prices.pct_change).
    #
    # Phase 1: draw a plain random-walk price path (n_months × n_firms).
    noise = rng.normal(0.0, ret_vol, (n_months, n_firms))
    log_ret_base = base_ret + noise  # shape (n_months, n_firms)
    cum_log = np.cumsum(log_ret_base, axis=0)

    # Phase 2: for months t >= formation, add a reversal premium that is a function
    # of the trailing-formation log-return ending at t-1.  The planted premium is:
    #   extra_ret[t, j] = reversal * z(-trail[t-1, j])
    # where trail[t-1, j] = cum_log[t-1, j] - cum_log[t-1-formation, j]
    # and z(·) is the cross-sectional z-score of the NEGATED trailing return
    # (losers = negative trail → high z → higher expected next-month return).
    #
    # We rebuild cum_log iteratively so the planted premium propagates into prices.
    extra = np.zeros((n_months, n_firms))
    cum_log2 = np.zeros_like(cum_log)
    cum_log2[:formation] = cum_log[:formation]
    for t in range(formation, n_months):
        trail = cum_log2[t - 1] - cum_log2[t - 1 - formation]
        neg_trail = -trail
        mu_neg = neg_trail.mean()
        sd_neg = neg_trail.std() + 1e-9
        z = (neg_trail - mu_neg) / sd_neg
        extra[t] = reversal * z
        cum_log2[t] = cum_log2[t - 1] + log_ret_base[t] + extra[t]

    price_arr = 100.0 * np.exp(cum_log2)
    price_df = pd.DataFrame(price_arr, index=months, columns=firms)

    # fwd_ret_df: for each month t the log-return earned in the *next* month.
    # This mirrors what strategy.quintile_returns infers from prices.pct_change.
    log_ret2 = np.diff(cum_log2, axis=0, prepend=cum_log2[[0]])
    log_ret2[0] = np.nan  # no return before first month
    fwd_raw = np.full((n_months, n_firms), np.nan)
    fwd_raw[:-1] = log_ret2[1:]  # fwd_ret_df.iloc[t] = log_ret at t+1
    fwd_ret_df = pd.DataFrame(fwd_raw, index=months, columns=firms)

    truth = {
        "reversal": reversal,
        "base_ret": base_ret,
        "ret_vol": ret_vol,
        "n_firms": n_firms,
        "n_months": n_months,
        "formation": formation,
        "hold": hold,
        "seed": seed,
    }
    return price_df, fwd_ret_df, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo monthly closes, cache-only by default
# ---------------------------------------------------------------------------
def fetch_monthly(
    fetch: bool = False,
    cache_path: str = MONTHLY_CACHE,
) -> pd.DataFrame:
    """Monthly adjusted close panel for S&P 500 names; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is
    cached as a parquet under ``_cache/ltr_monthly.parquet``). Columns are
    tickers, index is month-end dates (pd.DatetimeIndex at month-end frequency).
    Returns an empty DataFrame if the cache is absent and ``fetch=False`` is set
    — callers must guard with ``os.path.exists(cache_path)`` before calling.

    **Survivorship bias**: the ticker universe is the current S&P 500 projected
    backwards. All positive results are upper bounds.
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
    # Live fetch: yfinance monthly closes for the current S&P 500 basket
    # ------------------------------------------------------------------
    import yfinance as yf  # lazy import: network only on fetch=True

    try:
        from quantlab.universe import sp500_symbols
        tickers = sp500_symbols(allow_survivorship_bias=True)
    except Exception:
        # Fallback: a representative basket of large caps
        tickers = [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B",
            "UNH", "JNJ", "JPM", "V", "PG", "MA", "HD", "XOM", "CVX", "MRK",
            "ABBV", "PEP", "KO", "AVGO", "COST", "WMT", "LLY", "TMO", "MCD",
            "CSCO", "ACN", "ABT", "CRM", "DHR", "NEE", "TXN", "NFLX", "QCOM",
            "AMD", "ORCL", "INTC", "CMCSA", "ADBE", "PFE", "T", "DIS", "RTX",
            "HON", "BAC", "GE", "CAT", "GS",
        ]

    raw = yf.download(
        tickers,
        start="1990-01-01",
        interval="1mo",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no monthly data")

    # Extract 'Close' level from MultiIndex columns
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", axis=1, level=0)
    else:
        closes = raw

    # Month-end alignment
    closes.index = pd.to_datetime(closes.index)
    closes.index = closes.index + pd.offsets.MonthEnd(0)
    closes = closes.sort_index()

    # Drop tickers with fewer than 60 months of data
    min_obs = 60
    closes = closes.loc[:, closes.count() >= min_obs]

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    closes.to_parquet(cache_path)
    print(f"Cached monthly panel: {closes.shape[0]} months × {closes.shape[1]} tickers -> {cache_path}")
    return closes


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a prices frame (last row), for the as-of stamp."""
    arr = df.iloc[-1].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
