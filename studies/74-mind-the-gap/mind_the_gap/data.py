"""Data layer for Study 74 (Mind-the-Gap).

Two tapes, one shape — a per-session DataFrame of daily OHLC with a gap column:

- ``synthetic_daily`` — a *deterministic, offline* generator.  A mean-reversion knob
  (``gap_fill_prob``) controls how likely it is that a session's opening gap closes
  intraday.  ``gap_fill_prob=0.5`` is a fair coin (gaps fill by chance alone);
  higher values plant a real gap-fill tendency the strategy can exploit.  The null
  setting ``gap_fill_prob=0.5`` is the study's null in a bottle — any test that finds
  an edge on this tape is a false positive.
- ``fetch_daily`` — real daily OHLCV bars from Yahoo! Finance, cache-only by default
  (``fetch=False`` raises if no cache; ``fetch=True`` hits yfinance and caches a
  parquet).  Daily bars reach back years, giving the gap-fill study real statistical
  power — unlike sub-hourly bars which are capped at ~60 days.

No look-ahead is baked in here.  Gaps are computed from prior-close vs today-open
(fully known at the open); fill status is checked against today's intraday range
(high/low) — which is only known at the close, so a live strategy reads the fill
after the fact.  The strategy module computes the binary outcome and forms signals
from it.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Conventional gap-size buckets (in percent of prior close).
GAP_SMALL = 0.25   # |gap| < 0.25% — "mechanical" fill zone, essentially microstructure
GAP_MED   = 1.00   # 0.25% ≤ |gap| < 1.00% — typical intraday gap
GAP_LARGE = 3.00   # |gap| ≥ 1.00% — "news-gap" territory; fills are rarer


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 500,
    gap_fill_prob: float = 0.5,
    gap_vol_pct: float = 0.8,
    intraday_range_pct: float = 1.2,
    drift_pct: float = 0.03,
    seed: int = 74,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a controllable gap-fill rate.

    Each session is generated as follows:
    - **Overnight gap** drawn from N(0, ``gap_vol_pct`` %) — the open minus the
      prior close, as a fraction.  ``gap_vol_pct=0.8`` corresponds to a ~0.8%
      typical overnight move, roughly matching SPY over the last decade.
    - **Intraday range** (high−low) is lognormal around ``intraday_range_pct``% and
      is independent of the gap.  Whether the open falls between the day's high and
      low is irrelevant — what matters is whether the prior close is bracketed.
    - **Fill flag** — the gap is "filled" when the day's range includes the prior
      close.  Here we plant a known ``gap_fill_prob``: with probability
      ``gap_fill_prob`` the range is widened so it necessarily includes the prior
      close; otherwise it is left as-is (the coin-flip baseline).

    ``gap_fill_prob=0.5`` is a martingale null — no forecastable structure.  Higher
    values plant a real gap-fill tendency.  ``gap_fill_prob=0`` disables any widening
    (but random range still fills sometimes).

    Returns ``(daily, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2022-01-03", periods=n_days, name="date")

    # --- build price series -----------------------------------------------
    overnight_ret = rng.normal(drift_pct * 1e-2, gap_vol_pct * 1e-2, n_days)
    intraday_vol = rng.lognormal(np.log(intraday_range_pct * 1e-2), 0.3, n_days)
    close_within_range = rng.uniform(0.1, 0.9, n_days)  # where the close sits in range

    price = 400.0
    rows = []
    prev_close = price
    for i in range(n_days):
        open_px = prev_close * (1.0 + overnight_ret[i])
        r = intraday_vol[i] * open_px            # raw range (in price)

        # Decide fill: plant the gap_fill_prob knob.
        # The gap is filled when the prior close is within [low, high].
        # We arrange the range symmetrically around the open initially,
        # then check; if the fill knob fires we extend the range to cover prior close.
        low_raw  = open_px - 0.5 * r
        high_raw = open_px + 0.5 * r
        force_fill = rng.random() < gap_fill_prob
        if force_fill:
            # extend the range to cover prev_close (gap fills)
            lo = min(low_raw, prev_close * 0.9995)
            hi = max(high_raw, prev_close * 1.0005)
        else:
            lo, hi = low_raw, high_raw

        close_px = lo + close_within_range[i] * (hi - lo)
        rows.append((open_px, hi, lo, close_px, float(rng.integers(1_000_000, 50_000_000))))
        prev_close = close_px

    daily = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)

    # Compute gap statistics on the synthetic frame.
    daily["gap_pct"]  = (daily["open"] - daily["close"].shift(1)) / daily["close"].shift(1) * 100.0
    daily["gap_fill"] = (
        ((daily["gap_pct"] > 0) & (daily["low"]  <= daily["close"].shift(1))) |
        ((daily["gap_pct"] < 0) & (daily["high"] >= daily["close"].shift(1)))
    )
    daily = daily.dropna(subset=["gap_pct"])

    truth = {
        "gap_fill_prob": gap_fill_prob,
        "gap_vol_pct": gap_vol_pct,
        "intraday_range_pct": intraday_range_pct,
        "n_days": n_days,
        "seed": seed,
    }
    return daily, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily bars, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_1d.parquet")


def fetch_daily(
    ticker: str,
    period: str = "10y",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True``.  Daily bars from Yahoo
    are split-and-dividend-adjusted (``auto_adjust=True``).  Returns a frame indexed
    by date (tz-naive, calendar dates), columns open/high/low/close/volume.

    Adds two derived columns ready for gap analysis:
    - ``gap_pct`` : (open − prev_close) / prev_close × 100  (percent, signed)
    - ``gap_fill``: bool — True if the intraday range includes the prior close
      (i.e. the day's low ≤ prev_close ≤ day's high).
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call fetch_daily({ticker!r}, fetch=True) once to populate the cache."
            )
        daily = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy — only when we hit the network

        raw = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        daily = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        daily.index.name = "date"
        if daily.index.tz is not None:
            daily.index = daily.index.tz_localize(None)
        os.makedirs(cache_dir, exist_ok=True)
        daily.to_parquet(path)

    # (Re-)compute gap columns so they're always present after a cache hit.
    daily = daily.copy()
    prev_close = daily["close"].shift(1)
    daily["gap_pct"]  = (daily["open"] - prev_close) / prev_close * 100.0
    daily["gap_fill"] = (
        ((daily["gap_pct"] > 0) & (daily["low"]  <= prev_close)) |
        ((daily["gap_pct"] < 0) & (daily["high"] >= prev_close))
    )
    return daily.dropna(subset=["gap_pct"])


def fingerprint(daily: pd.DataFrame) -> str:
    """A short content fingerprint of a daily tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(daily["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
