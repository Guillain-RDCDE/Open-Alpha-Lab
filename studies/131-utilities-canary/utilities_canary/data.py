"""Data layer for Study 131 (Utilities-Canary).

Two tapes, one shape (a tz-naive daily frame):

- ``synthetic_daily`` — a *deterministic, offline* generator.  A ``canary_signal``
  knob controls how much predictive power the XLU/SPY relative-strength slope carries
  about forward SPY returns.  ``canary_signal=0`` is the null: relative strength
  carries no information and the timing signal is a fair coin.  ``canary_signal<0``
  plants the folk claim — rising XLU/SPY RS (defensive outperformance) predicts lower
  forward SPY returns.  ``canary_signal>0`` is the contra-claim.  This lets tests
  verify the machinery can detect a planted effect.

- ``fetch_daily`` — the real Yahoo! daily closes for ``XLU`` and ``SPY``, cache-only
  by default so the test-suite and the reproducible core never touch the network.
  XLU started trading on 1998-12-22, giving ~26 years of daily history — enough
  to be informative.

No look-ahead is baked in here.  XLU and SPY measured at the close of day *t* are only
used to form signals that trade at the close of day *t+1* or later (see ``strategy.py``).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Real-tape tickers.
TICKERS = ["XLU", "SPY"]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 6500,
    canary_signal: float = 0.0,
    spy_vol: float = 0.01,
    xlu_beta: float = 0.40,
    start: str = "1999-01-04",
    seed: int = 131,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily tape with a known amount of canary predictive power.

    The synthetic tape mimics two daily series — XLU and SPY — with the following
    structure:

    - ``SPY_ret_t = eps_spy_t`` is i.i.d. normal of standard deviation ``spy_vol``,
      a pure random walk by default.
    - ``XLU_ret_t = xlu_beta * SPY_ret_t + eps_xlu_t`` gives XLU its characteristic
      defensive beta (< 1) relative to SPY, with idiosyncratic vol matching observed
      patterns.
    - ``RS_t = log(XLU_close_t / SPY_close_t)`` is the relative-strength log ratio.
    - ``RS_mom_t`` is the rolling 20-day change in RS (momentum of relative strength).
    - Forward SPY returns are then generated as::

          spy_ret_{t+1} = canary_signal * rs_mom_rank_t + eps_t

      where ``rs_mom_rank`` is the rolling percentile (out-of-sample) of RS momentum.
      ``canary_signal=0`` makes SPY a pure random walk; ``canary_signal<0`` plants
      the folk claim (rising defensive RS → bad forward SPY returns).

    Returns ``(df, truth)`` where ``df`` has columns
    ``[XLU_close, SPY_close, XLU_ret, SPY_ret, rs, rs_mom, rs_mom_rank]``
    indexed by a tz-naive ``pd.DatetimeIndex`` of business days, and ``truth``
    records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    # SPY: pure random walk (log returns)
    eps_spy = rng.normal(0.0, spy_vol, n_days)

    # XLU: defensive (beta < 1) + idiosyncratic noise
    xlu_idio_vol = spy_vol * 0.60
    eps_xlu = rng.normal(0.0, xlu_idio_vol, n_days)
    xlu_ret = xlu_beta * eps_spy + eps_xlu

    # Prices from cumulative log returns
    spy_close = 100.0 * np.exp(np.cumsum(eps_spy))
    xlu_close = 30.0 * np.exp(np.cumsum(xlu_ret))

    # Relative strength (log ratio)
    rs = np.log(xlu_close) - np.log(spy_close)

    # RS momentum: 20-day change in the log ratio
    rs_s = pd.Series(rs, index=idx)
    rs_mom = rs_s.diff(20)

    # Rolling percentile rank of RS momentum (252-day lookback, strictly out-of-sample)
    rs_mom_rank = rs_mom.rolling(252, min_periods=63).rank(pct=True)

    # Forward SPY returns with optional planted signal
    # (when canary_signal != 0, rising RS momentum → lower future SPY returns)
    spy_ret_final = np.empty(n_days)
    spy_ret_final[0] = eps_spy[0]
    for i in range(1, n_days):
        lagged_rank = rs_mom_rank.iloc[i - 1]
        if np.isnan(lagged_rank):
            spy_ret_final[i] = eps_spy[i]
        else:
            # canary_signal < 0: high RS rank → lower SPY ret (the folk claim)
            spy_ret_final[i] = canary_signal * (lagged_rank - 0.5) + eps_spy[i]

    spy_close_final = 100.0 * np.exp(np.cumsum(spy_ret_final))

    df = pd.DataFrame(
        {
            "XLU_close": xlu_close,
            "SPY_close": spy_close_final,
            "XLU_ret": xlu_ret,
            "SPY_ret": spy_ret_final,
            "rs": rs,
            "rs_mom": rs_mom.to_numpy(),
            "rs_mom_rank": rs_mom_rank.to_numpy(),
        },
        index=idx,
    )
    truth = {
        "canary_signal": canary_signal,
        "spy_vol": spy_vol,
        "xlu_beta": xlu_beta,
        "n_days": n_days,
        "seed": seed,
        "start": start,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily closes, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    """Canonical parquet path for the combined XLU/SPY daily cache."""
    return os.path.join(cache_dir, "daily_xlu_spy.parquet")


def fetch_daily(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    start: str = "1998-12-22",
) -> pd.DataFrame:
    """Real daily XLU and SPY data; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (the result is cached as a
    parquet under ``_cache/``).  With ``fetch=False`` (the default) a
    ``FileNotFoundError`` is raised if the cache is absent — consistent with the desk
    convention that tests and the reproducible core must never hit the network.

    The returned frame has columns
    ``[XLU_close, SPY_close, XLU_ret, SPY_ret, rs, rs_mom, rs_mom_rank]`` with a
    tz-naive ``DatetimeIndex`` named ``date``.

    - ``rs = log(XLU_close / SPY_close)`` is the raw relative-strength log ratio.
    - ``rs_mom = rs.diff(20)`` is the 20-day momentum of relative strength (positive
      means XLU has been outperforming SPY over the past 20 trading days — the
      "canary signal").
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

        # Multi-ticker download returns a MultiIndex — flatten to Close only.
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"].copy()
        else:
            closes = raw[["Close"]].copy()

        closes.columns = [str(c) for c in closes.columns]
        closes = closes.rename(columns={"XLU": "XLU_close", "SPY": "SPY_close"})
        closes.index.name = "date"

        closes = closes[["XLU_close", "SPY_close"]].dropna()

        # Log returns (auto_adjust handles dividends and splits)
        closes["XLU_ret"] = np.log(closes["XLU_close"]).diff()
        closes["SPY_ret"] = np.log(closes["SPY_close"]).diff()

        # Relative strength
        closes["rs"] = np.log(closes["XLU_close"]) - np.log(closes["SPY_close"])

        # RS momentum: 20-day change
        closes["rs_mom"] = closes["rs"].diff(20)

        # Rolling percentile rank (252-day, out-of-sample)
        closes["rs_mom_rank"] = closes["rs_mom"].rolling(252, min_periods=63).rank(pct=True)

        closes = closes.dropna(subset=["XLU_ret", "SPY_ret"])

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
