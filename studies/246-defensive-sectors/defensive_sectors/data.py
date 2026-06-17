"""Data layer for Study 246 (Defensive-Sectors Leadership).

Three tickers, one shape (a tz-naive daily frame):

- ``synthetic_daily`` -- a *deterministic, offline* generator.  A
  ``defensive_signal`` knob controls how much predictive power the combined
  XLP+XLU/SPY relative-strength slope carries about forward SPY returns.
  ``defensive_signal=0`` is the null: relative strength carries no information.
  ``defensive_signal<0`` plants the folk claim -- rising defensive RS predicts
  lower forward SPY returns.  ``defensive_signal>0`` is the contra-claim.

- ``fetch_daily`` -- the real Yahoo! daily closes for ``XLP``, ``XLU``, and
  ``SPY``, cache-only by default so the test-suite and reproducible core never
  touch the network.  XLP (Consumer Staples SPDR) launched 1998-12-22, giving
  ~26 years of daily history.  XLU (Utilities SPDR) same inception date.

No look-ahead is baked in here.  Closes of day *t* only form signals that trade
at the close of day *t+1* or later (see ``strategy.py``).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Real-tape tickers.
TICKERS = ["XLP", "XLU", "SPY"]


# ---------------------------------------------------------------------------
# Synthetic tape -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 6500,
    defensive_signal: float = 0.0,
    spy_vol: float = 0.01,
    xlp_beta: float = 0.50,
    xlu_beta: float = 0.40,
    start: str = "1999-01-04",
    seed: int = 246,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily tape with a known amount of defensive predictive power.

    The synthetic tape mimics three daily series -- XLP, XLU, and SPY:

    - ``SPY_ret_t = eps_spy_t`` is i.i.d. normal of standard deviation ``spy_vol``.
    - ``XLP_ret_t = xlp_beta * SPY_ret_t + eps_xlp_t`` (staples beta < 1).
    - ``XLU_ret_t = xlu_beta * SPY_ret_t + eps_xlu_t`` (utilities beta even lower).
    - The combined defensive RS is the average of log(XLP/SPY) and log(XLU/SPY):
      ``combined_rs = 0.5*(log(XLP/SPY) + log(XLU/SPY))``.
    - ``combined_rs_mom`` is the rolling 20-day change in combined RS.
    - Forward SPY returns are then generated as::

          spy_ret_{t+1} = defensive_signal * rs_mom_rank_t + eps_t

      ``defensive_signal=0`` makes SPY a pure random walk; ``defensive_signal<0``
      plants the folk claim (rising defensive RS -> bad forward SPY returns).

    Returns ``(df, truth)`` where ``df`` has columns
    ``[XLP_close, XLU_close, SPY_close, XLP_ret, XLU_ret, SPY_ret,
      combined_rs, rs_mom, rs_mom_rank]``
    indexed by a tz-naive ``pd.DatetimeIndex`` of business days, and ``truth``
    records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    # SPY: pure random walk
    eps_spy = rng.normal(0.0, spy_vol, n_days)

    # XLP: consumer staples (beta ~0.5) + idiosyncratic
    xlp_idio_vol = spy_vol * 0.55
    eps_xlp = rng.normal(0.0, xlp_idio_vol, n_days)
    xlp_ret = xlp_beta * eps_spy + eps_xlp

    # XLU: utilities (beta ~0.4) + idiosyncratic
    xlu_idio_vol = spy_vol * 0.60
    eps_xlu = rng.normal(0.0, xlu_idio_vol, n_days)
    xlu_ret = xlu_beta * eps_spy + eps_xlu

    # Prices from cumulative log returns
    spy_close = 100.0 * np.exp(np.cumsum(eps_spy))
    xlp_close = 25.0 * np.exp(np.cumsum(xlp_ret))
    xlu_close = 30.0 * np.exp(np.cumsum(xlu_ret))

    # Combined defensive relative strength (average of both sector log ratios)
    rs_xlp = np.log(xlp_close) - np.log(spy_close)
    rs_xlu = np.log(xlu_close) - np.log(spy_close)
    combined_rs = 0.5 * (rs_xlp + rs_xlu)

    # RS momentum: 20-day change
    combined_rs_s = pd.Series(combined_rs, index=idx)
    rs_mom = combined_rs_s.diff(20)

    # Rolling percentile rank of RS momentum (252-day lookback, strictly OOS)
    rs_mom_rank = rs_mom.rolling(252, min_periods=63).rank(pct=True)

    # Forward SPY returns with optional planted signal
    spy_ret_final = np.empty(n_days)
    spy_ret_final[0] = eps_spy[0]
    for i in range(1, n_days):
        lagged_rank = rs_mom_rank.iloc[i - 1]
        if np.isnan(lagged_rank):
            spy_ret_final[i] = eps_spy[i]
        else:
            # defensive_signal < 0: high defensive RS rank -> lower SPY ret
            spy_ret_final[i] = defensive_signal * (lagged_rank - 0.5) + eps_spy[i]

    spy_close_final = 100.0 * np.exp(np.cumsum(spy_ret_final))

    df = pd.DataFrame(
        {
            "XLP_close": xlp_close,
            "XLU_close": xlu_close,
            "SPY_close": spy_close_final,
            "XLP_ret": xlp_ret,
            "XLU_ret": xlu_ret,
            "SPY_ret": spy_ret_final,
            "combined_rs": combined_rs_s.to_numpy(),
            "rs_mom": rs_mom.to_numpy(),
            "rs_mom_rank": rs_mom_rank.to_numpy(),
        },
        index=idx,
    )
    truth = {
        "defensive_signal": defensive_signal,
        "spy_vol": spy_vol,
        "xlp_beta": xlp_beta,
        "xlu_beta": xlu_beta,
        "n_days": n_days,
        "seed": seed,
        "start": start,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape -- Yahoo daily closes, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    """Canonical parquet path for the combined XLP/XLU/SPY daily cache."""
    return os.path.join(cache_dir, "daily_xlp_xlu_spy.parquet")


def fetch_daily(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    start: str = "1998-12-22",
) -> pd.DataFrame:
    """Real daily XLP, XLU, and SPY data; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (the result is cached as a
    parquet under ``_cache/``).  With ``fetch=False`` (the default) a
    ``FileNotFoundError`` is raised if the cache is absent -- consistent with the desk
    convention that tests and the reproducible core must never hit the network.

    The returned frame has columns
    ``[XLP_close, XLU_close, SPY_close, XLP_ret, XLU_ret, SPY_ret,
      combined_rs, rs_mom, rs_mom_rank]``
    with a tz-naive ``DatetimeIndex`` named ``date``.

    - ``combined_rs = 0.5*(log(XLP/SPY) + log(XLU/SPY))`` is the average of both
      sector relative-strength log ratios.
    - ``rs_mom = combined_rs.diff(20)`` is the 20-day momentum (positive means
      defensive sectors have been outperforming SPY -- the "risk-off alert").
    - ``rs_mom_rank`` is the rolling 252-day out-of-sample percentile rank of rs_mom.
    """
    path = _cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape at {path}. "
                f"Call fetch_daily(fetch=True) once to populate the cache."
            )
        df = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            TICKERS,
            start=start,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            raise RuntimeError("yfinance returned no daily bars")

        # Multi-ticker download returns a MultiIndex -- flatten to Close only.
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"].copy()
        else:
            closes = raw[["Close"]].copy()

        closes.columns = [str(c) for c in closes.columns]
        closes = closes.rename(
            columns={"XLP": "XLP_close", "XLU": "XLU_close", "SPY": "SPY_close"}
        )
        closes.index.name = "date"

        closes = closes[["XLP_close", "XLU_close", "SPY_close"]].dropna()

        # Log returns (auto_adjust handles dividends and splits)
        closes["XLP_ret"] = np.log(closes["XLP_close"]).diff()
        closes["XLU_ret"] = np.log(closes["XLU_close"]).diff()
        closes["SPY_ret"] = np.log(closes["SPY_close"]).diff()

        # Combined defensive relative strength
        rs_xlp = np.log(closes["XLP_close"]) - np.log(closes["SPY_close"])
        rs_xlu = np.log(closes["XLU_close"]) - np.log(closes["SPY_close"])
        closes["combined_rs"] = 0.5 * (rs_xlp + rs_xlu)

        # RS momentum: 20-day change
        closes["rs_mom"] = closes["combined_rs"].diff(20)

        # Rolling percentile rank (252-day, out-of-sample)
        closes["rs_mom_rank"] = closes["rs_mom"].rolling(252, min_periods=63).rank(pct=True)

        closes = closes.dropna(subset=["XLP_ret", "XLU_ret", "SPY_ret"])

        if closes.index.tz is not None:
            closes.index = closes.index.tz_localize(None)

        os.makedirs(cache_dir, exist_ok=True)
        closes.to_parquet(path)
        df = closes

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a daily tape (SPY_close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(df["SPY_close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
