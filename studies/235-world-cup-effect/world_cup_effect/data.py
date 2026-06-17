"""Data layer for Study 235 (World-Cup-Effect).

Two components, both fully offline for the core:

- ``WORLD_CUP_WINDOWS`` — a hardcoded table of every FIFA World Cup tournament
  (1950–2022), recording the start and end dates of the tournament, the host country,
  and the winner. World Cups occur every 4 years. The 1942 and 1946 editions were
  cancelled due to WWII. We cover 1950–2022 (19 editions) for which we have
  continuous equity market data.

  The "World Cup Effect" (after Edmans, Garcia & Norli 2007) posits that
  national sentiment / attention during the World Cup suppresses equity returns in
  host countries or globally. We focus on the global (S&P 500) version here.
  The tournament window is approximately 4–5 weeks (group stage through final).

- ``synthetic_daily`` — a deterministic, offline daily-return generator with an
  optional planted "World Cup window" effect knob. signal_bps = 0 is the null.

- ``fetch_sp500_daily`` — real S&P 500 data via yfinance, cached locally.

Sources:
  - Edmans, A., Garcia, D. & Norli, O. (2007). "Sports Sentiment and Stock Returns."
    Journal of Finance, 62(4), 1967–1998.
  - FIFA: https://www.fifa.com/tournaments/mens/worldcup
  - Wikipedia: FIFA World Cup
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))

# ---------------------------------------------------------------------------
# The hardcoded World Cup windows table — the study's deterministic, offline core
# ---------------------------------------------------------------------------
# Columns:
#   edition     : year (int) — the year in which the tournament was played
#   start_date  : string 'YYYY-MM-DD' — first day of the tournament
#   end_date    : string 'YYYY-MM-DD' — day of the final (last day of tournament)
#   host        : country name (str)
#   winner      : winning country name (str)
#
# Sources: FIFA official records, Wikipedia "FIFA World Cup"
# Note: We use the date the first match was played through the day of the final.
# As-of: 2026-06-16. Last edition covered: 2022 (Qatar).

WORLD_CUP_WINDOWS: list[dict] = [
    # 1950: Brazil (first post-WWII WC; no final match — round-robin final stage)
    {"edition": 1950, "start_date": "1950-06-24", "end_date": "1950-07-16",
     "host": "Brazil", "winner": "Uruguay"},
    # 1954: Switzerland
    {"edition": 1954, "start_date": "1954-06-16", "end_date": "1954-07-04",
     "host": "Switzerland", "winner": "West Germany"},
    # 1958: Sweden
    {"edition": 1958, "start_date": "1958-06-08", "end_date": "1958-06-29",
     "host": "Sweden", "winner": "Brazil"},
    # 1962: Chile
    {"edition": 1962, "start_date": "1962-05-30", "end_date": "1962-06-17",
     "host": "Chile", "winner": "Brazil"},
    # 1966: England
    {"edition": 1966, "start_date": "1966-07-11", "end_date": "1966-07-30",
     "host": "England", "winner": "England"},
    # 1970: Mexico
    {"edition": 1970, "start_date": "1970-05-31", "end_date": "1970-06-21",
     "host": "Mexico", "winner": "Brazil"},
    # 1974: West Germany
    {"edition": 1974, "start_date": "1974-06-13", "end_date": "1974-07-07",
     "host": "West Germany", "winner": "West Germany"},
    # 1978: Argentina
    {"edition": 1978, "start_date": "1978-06-01", "end_date": "1978-06-25",
     "host": "Argentina", "winner": "Argentina"},
    # 1982: Spain
    {"edition": 1982, "start_date": "1982-06-13", "end_date": "1982-07-11",
     "host": "Spain", "winner": "Italy"},
    # 1986: Mexico
    {"edition": 1986, "start_date": "1986-05-31", "end_date": "1986-06-29",
     "host": "Mexico", "winner": "Argentina"},
    # 1990: Italy
    {"edition": 1990, "start_date": "1990-06-08", "end_date": "1990-07-08",
     "host": "Italy", "winner": "West Germany"},
    # 1994: United States
    {"edition": 1994, "start_date": "1994-06-17", "end_date": "1994-07-17",
     "host": "United States", "winner": "Brazil"},
    # 1998: France
    {"edition": 1998, "start_date": "1998-06-10", "end_date": "1998-07-12",
     "host": "France", "winner": "France"},
    # 2002: South Korea / Japan (co-hosted; first Asian WC)
    {"edition": 2002, "start_date": "2002-05-31", "end_date": "2002-06-30",
     "host": "South Korea/Japan", "winner": "Brazil"},
    # 2006: Germany
    {"edition": 2006, "start_date": "2006-06-09", "end_date": "2006-07-09",
     "host": "Germany", "winner": "Italy"},
    # 2010: South Africa (first African WC)
    {"edition": 2010, "start_date": "2010-06-11", "end_date": "2010-07-11",
     "host": "South Africa", "winner": "Spain"},
    # 2014: Brazil
    {"edition": 2014, "start_date": "2014-06-12", "end_date": "2014-07-13",
     "host": "Brazil", "winner": "Germany"},
    # 2018: Russia
    {"edition": 2018, "start_date": "2018-06-14", "end_date": "2018-07-15",
     "host": "Russia", "winner": "France"},
    # 2022: Qatar (first Winter WC — November/December due to heat)
    {"edition": 2022, "start_date": "2022-11-20", "end_date": "2022-12-18",
     "host": "Qatar", "winner": "Argentina"},
]

WORLD_CUP_DF: pd.DataFrame = pd.DataFrame(WORLD_CUP_WINDOWS)
WORLD_CUP_DF["start_date"] = pd.to_datetime(WORLD_CUP_DF["start_date"])
WORLD_CUP_DF["end_date"] = pd.to_datetime(WORLD_CUP_DF["end_date"])


def worldcup_table() -> pd.DataFrame:
    """Return a clean copy of the hardcoded World Cup windows table.

    Columns: ``edition`` (int), ``start_date`` (Timestamp), ``end_date`` (Timestamp),
    ``host`` (str), ``winner`` (str).
    """
    df = WORLD_CUP_DF.copy()
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df


# ---------------------------------------------------------------------------
# Control windows — matched windows outside World Cup years for comparison
# ---------------------------------------------------------------------------
def control_windows(wc_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Generate matched-length control windows 2 years before each World Cup.

    For each World Cup window (start, end), we take a window of the same length
    exactly 2 years prior (same calendar month, same duration). This gives a
    fair "placebo" — same season, same duration, no tournament.

    Returns a DataFrame with columns: ``edition``, ``ctrl_start``, ``ctrl_end``,
    ``duration_days`` (matching the actual WC window).
    """
    if wc_df is None:
        wc_df = worldcup_table()

    rows = []
    for _, r in wc_df.iterrows():
        duration = (r["end_date"] - r["start_date"]).days
        ctrl_start = r["start_date"] - pd.DateOffset(years=2)
        ctrl_end = ctrl_start + pd.Timedelta(days=duration)
        rows.append({
            "edition": r["edition"],
            "ctrl_start": ctrl_start,
            "ctrl_end": ctrl_end,
            "duration_days": duration,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Synthetic daily return generator — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 5000,
    signal_bps: float = 0.0,
    base_daily_return: float = 0.0003,
    vol_daily: float = 0.01,
    wc_fraction: float = 0.15,
    seed: int = 235,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily-return series with an optional planted World Cup effect.

    Each day is drawn i.i.d. from N(base_daily_return, vol_daily^2). Days flagged
    as 'in_wc' (selected randomly with probability ``wc_fraction``) receive an
    extra drift of ``signal_bps`` bps. ``signal_bps = 0`` is the null — World Cup
    windows are uncorrelated with returns.

    Returns ``(df, truth)`` where ``df`` has columns ``date`` (DatetimeIndex),
    ``return`` (daily simple return), ``in_wc`` (bool), and ``truth`` records
    the planted parameters.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1950-01-01", periods=n_days, freq="B")
    in_wc = rng.random(n_days) < wc_fraction
    drift = np.where(in_wc, base_daily_return + signal_bps * 1e-4, base_daily_return)
    rets = drift + rng.normal(0.0, vol_daily, n_days)

    df = pd.DataFrame({
        "date": dates,
        "return": rets,
        "in_wc": in_wc,
    })
    truth = {
        "n_days": n_days,
        "signal_bps": signal_bps,
        "base_daily_return": base_daily_return,
        "vol_daily": vol_daily,
        "wc_fraction": wc_fraction,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real data — S&P 500 daily returns via yfinance (study-local cache)
# ---------------------------------------------------------------------------
def fetch_sp500_daily(
    cache_dir: str = DEFAULT_CACHE,
    start: str = "1950-01-01",
    end: str = "2023-12-31",
) -> pd.DataFrame:
    """Fetch S&P 500 daily price returns (^GSPC) from yfinance, cache locally.

    Cache file: ``studies/235-world-cup-effect/_cache/sp500_daily.parquet``.
    If the cache exists, reads from it. Otherwise fetches from yfinance and
    caches to the study-local directory (NOT the shared repo _cache/).

    Returns a DataFrame with a DatetimeIndex and column ``sp500_return``
    (daily log return) plus ``sp500_price`` (adjusted close).

    Raises ``FileNotFoundError`` if cache absent and yfinance unavailable.
    """
    cache_path = os.path.join(cache_dir, "sp500_daily.parquet")
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df

    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("yfinance is required: pip install yfinance") from exc

    ticker = yf.Ticker("^GSPC")
    hist = ticker.history(start=start, end=end, interval="1d")
    if hist.empty:
        raise FileNotFoundError(
            "yfinance returned no data for ^GSPC. "
            "Check network connectivity or use the cached parquet."
        )

    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    hist = hist[["Close"]].rename(columns={"Close": "sp500_price"})
    hist["sp500_return"] = hist["sp500_price"].pct_change()
    hist = hist.dropna()

    os.makedirs(cache_dir, exist_ok=True)
    hist.to_parquet(cache_path)
    return hist


def tag_wc_windows(
    price_df: pd.DataFrame,
    wc_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Tag each trading day as inside or outside a World Cup window.

    Parameters
    ----------
    price_df : DataFrame with DatetimeIndex and column ``sp500_return``.
    wc_df    : World Cup windows DataFrame (from ``worldcup_table()``).
               If None, uses the full hardcoded table.

    Returns the same DataFrame with additional columns:
      ``in_wc`` (bool), ``edition`` (int or NaN — the World Cup year if in_wc),
      ``ctrl_window`` (bool — True if in the matched 2-year-prior control window).
    """
    if wc_df is None:
        wc_df = worldcup_table()

    df = price_df.copy()
    df["in_wc"] = False
    df["edition"] = np.nan
    df["ctrl_window"] = False

    ctrl_df = control_windows(wc_df)

    for _, row in wc_df.iterrows():
        mask = (df.index >= row["start_date"]) & (df.index <= row["end_date"])
        df.loc[mask, "in_wc"] = True
        df.loc[mask, "edition"] = row["edition"]

    for _, row in ctrl_df.iterrows():
        mask = (df.index >= row["ctrl_start"]) & (df.index <= row["ctrl_end"])
        df.loc[mask, "ctrl_window"] = True

    return df


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of the sp500_return column, for the as-of stamp."""
    vals = df["sp500_return"].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]
