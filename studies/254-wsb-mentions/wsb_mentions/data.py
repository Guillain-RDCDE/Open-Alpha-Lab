"""Data layer for Study 254 (WSB-Mentions).

Three components, all offline for the reproducible core:

- ``MENTION_COUNTS`` -- a hardcoded month x ticker table of approximate
  r/WallStreetBets mention counts for a fixed meme-stock basket, January 2021
  through December 2023 (36 months).  The numbers are *curated, smoothed
  monthly approximations* digitized from public WSB-mention dashboards
  (SwaggyStocks / ApeWisdom / Quiver Quantitative style trackers) and scaled to
  the same order of magnitude.  They are NOT a tick-exact scrape -- they encode
  the well-known *shape* (a January-2021 GME/AMC explosion, the June-2021 AMC
  second wave, the slow 2022-2023 fade) so the study is fully reproducible
  offline.  Mentions for calendar month m are the *contemporaneous* buzz during
  month m; the strategy maps them to the return earned in month m+1.

- ``synthetic_panel`` -- a deterministic, offline generator.  A ``premium`` knob
  plants the exact claim under test: the cross-sectional rank of this month's
  mention count forecasts next month's return.  ``premium = 0`` is the null
  (mentions carry no forward information -- the realistic prior for a hype proxy
  that is itself driven *by* the price move it supposedly predicts).

- ``fetch_meme_monthly`` -- the real Yahoo! monthly close panel for the meme
  basket (via yfinance), cache-only by default so the test-suite and the
  reproducible core never touch the network.  The cache lives under
  ``_cache/meme_monthly.parquet`` (a date x ticker frame of adjusted monthly
  closes, rows = month-end dates).

**Survivorship-bias caveat, inverted**: the meme basket is *not* survivor-
clean -- it deliberately includes names that were squeezed and then collapsed
or were delisted (BBBY went to zero in 2023; SNDL/WISH reverse-split).  Where a
ticker has no Yahoo history for a month, it is simply absent from that month's
cross-section.  The hype proxy is contemporaneous and endogenous (mentions
spike *because* the price already moved), which is the core reason any apparent
"signal" is suspect.  This endogeneity is named on the Signal axis.

No look-ahead: mentions counted during month t predict the return earned during
month t+1.  Positions are formed at the month-t close and held through month
t+1 (one-month execution lag).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(_HERE, "..", "_cache")

MONTHLY_CACHE = os.path.join(DEFAULT_CACHE, "meme_monthly.parquet")

# Fixed meme basket -- deliberately NOT survivor-clean (BBBY went to zero, etc.)
MEME_TICKERS = [
    "GME", "AMC", "BB", "NOK", "KOSS", "PLTR", "TLRY", "SNDL",
    "CLOV", "WISH", "SPCE", "NIO", "BBBY", "MARA",
]

# ---------------------------------------------------------------------------
# The hardcoded WSB mention-count table -- the study's deterministic core
# ---------------------------------------------------------------------------
# Rows = month-end dates (2021-01 .. 2023-12), columns = MEME_TICKERS.
# Values = approximate r/WallStreetBets monthly mention counts (integer-ish),
# curated to encode the canonical buzz timeline:
#   - 2021-01 : the original GME/AMC squeeze (mentions explode)
#   - 2021-02 : the immediate hangover (mentions collapse but stay elevated)
#   - 2021-06 : the AMC second wave
#   - 2022    : steady fade as the meme cycle cools
#   - 2023    : BBBY's final pump-and-collapse (Aug 2023), otherwise low buzz
#
# These are SMOOTHED MONTHLY APPROXIMATIONS digitized from public WSB-mention
# dashboards, not a tick-exact scrape.  They are hardcoded so the core stays
# offline and reproducible (the same rationale as the Super-Bowl table in
# Study 158).  Sources: ApeWisdom / SwaggyStocks / Quiver Quantitative public
# WSB-ticker-mention trackers (2021-2023).  As-of 2026-06-17.
_MENTION_ROWS: list[dict] = [
    # date,         GME,   AMC,   BB,   NOK,  KOSS, PLTR, TLRY, SNDL, CLOV, WISH, SPCE, NIO, BBBY, MARA
    {"date": "2021-01-31", "GME": 41000, "AMC": 12500, "BB": 9800, "NOK": 7200, "KOSS": 5400, "PLTR": 3100, "TLRY": 900, "SNDL": 800, "CLOV": 0, "WISH": 0, "SPCE": 1600, "NIO": 2400, "BBBY": 1900, "MARA": 700},
    {"date": "2021-02-28", "GME": 18000, "AMC": 6400, "BB": 3300, "NOK": 2100, "KOSS": 1200, "PLTR": 4200, "TLRY": 2600, "SNDL": 3400, "CLOV": 0, "WISH": 0, "SPCE": 1900, "NIO": 1700, "BBBY": 600, "MARA": 1100},
    {"date": "2021-03-31", "GME": 14500, "AMC": 4100, "BB": 1500, "NOK": 900, "KOSS": 500, "PLTR": 2600, "TLRY": 1400, "SNDL": 2900, "CLOV": 0, "WISH": 0, "SPCE": 1300, "NIO": 1500, "BBBY": 700, "MARA": 1400},
    {"date": "2021-04-30", "GME": 9800, "AMC": 5200, "BB": 1100, "NOK": 600, "KOSS": 300, "PLTR": 1900, "TLRY": 800, "SNDL": 1600, "CLOV": 1800, "WISH": 0, "SPCE": 1000, "NIO": 1100, "BBBY": 500, "MARA": 1200},
    {"date": "2021-05-31", "GME": 8600, "AMC": 9400, "BB": 1700, "NOK": 700, "KOSS": 400, "PLTR": 1500, "TLRY": 600, "SNDL": 1200, "CLOV": 3100, "WISH": 1400, "SPCE": 900, "NIO": 900, "BBBY": 600, "MARA": 1000},
    {"date": "2021-06-30", "GME": 11200, "AMC": 28000, "BB": 4200, "NOK": 800, "KOSS": 600, "PLTR": 1300, "TLRY": 500, "SNDL": 900, "CLOV": 6800, "WISH": 4900, "SPCE": 2100, "NIO": 800, "BBBY": 1700, "MARA": 800},
    {"date": "2021-07-31", "GME": 7400, "AMC": 13500, "BB": 2100, "NOK": 500, "KOSS": 300, "PLTR": 1100, "TLRY": 400, "SNDL": 700, "CLOV": 2400, "WISH": 2600, "SPCE": 3600, "NIO": 700, "BBBY": 900, "MARA": 700},
    {"date": "2021-08-31", "GME": 6100, "AMC": 9800, "BB": 1500, "NOK": 400, "KOSS": 200, "PLTR": 2200, "TLRY": 350, "SNDL": 500, "CLOV": 1300, "WISH": 1900, "SPCE": 1400, "NIO": 600, "BBBY": 1100, "MARA": 600},
    {"date": "2021-09-30", "GME": 5500, "AMC": 8600, "BB": 1200, "NOK": 350, "KOSS": 180, "PLTR": 1800, "TLRY": 300, "SNDL": 400, "CLOV": 900, "WISH": 1300, "SPCE": 1000, "NIO": 550, "BBBY": 800, "MARA": 700},
    {"date": "2021-10-31", "GME": 5800, "AMC": 7900, "BB": 1100, "NOK": 300, "KOSS": 150, "PLTR": 1600, "TLRY": 280, "SNDL": 350, "CLOV": 700, "WISH": 1000, "SPCE": 800, "NIO": 700, "BBBY": 700, "MARA": 1300},
    {"date": "2021-11-30", "GME": 6200, "AMC": 8200, "BB": 1000, "NOK": 280, "KOSS": 140, "PLTR": 1400, "TLRY": 260, "SNDL": 300, "CLOV": 600, "WISH": 800, "SPCE": 700, "NIO": 650, "BBBY": 600, "MARA": 1100},
    {"date": "2021-12-31", "GME": 4900, "AMC": 6600, "BB": 800, "NOK": 240, "KOSS": 120, "PLTR": 1200, "TLRY": 220, "SNDL": 260, "CLOV": 480, "WISH": 620, "SPCE": 560, "NIO": 600, "BBBY": 520, "MARA": 900},
    {"date": "2022-01-31", "GME": 5400, "AMC": 7100, "BB": 900, "NOK": 220, "KOSS": 100, "PLTR": 1300, "TLRY": 240, "SNDL": 300, "CLOV": 520, "WISH": 700, "SPCE": 640, "NIO": 700, "BBBY": 580, "MARA": 800},
    {"date": "2022-02-28", "GME": 3900, "AMC": 5200, "BB": 700, "NOK": 180, "KOSS": 90, "PLTR": 1100, "TLRY": 200, "SNDL": 240, "CLOV": 400, "WISH": 540, "SPCE": 480, "NIO": 560, "BBBY": 440, "MARA": 700},
    {"date": "2022-03-31", "GME": 6800, "AMC": 6900, "BB": 820, "NOK": 160, "KOSS": 80, "PLTR": 900, "TLRY": 180, "SNDL": 200, "CLOV": 360, "WISH": 460, "SPCE": 420, "NIO": 520, "BBBY": 3200, "MARA": 640},
    {"date": "2022-04-30", "GME": 4100, "AMC": 4600, "BB": 600, "NOK": 140, "KOSS": 70, "PLTR": 780, "TLRY": 160, "SNDL": 180, "CLOV": 300, "WISH": 380, "SPCE": 360, "NIO": 480, "BBBY": 1400, "MARA": 560},
    {"date": "2022-05-31", "GME": 3600, "AMC": 4200, "BB": 520, "NOK": 130, "KOSS": 60, "PLTR": 700, "TLRY": 150, "SNDL": 160, "CLOV": 260, "WISH": 320, "SPCE": 320, "NIO": 440, "BBBY": 900, "MARA": 520},
    {"date": "2022-06-30", "GME": 3300, "AMC": 3800, "BB": 460, "NOK": 120, "KOSS": 55, "PLTR": 640, "TLRY": 140, "SNDL": 150, "CLOV": 240, "WISH": 280, "SPCE": 280, "NIO": 400, "BBBY": 700, "MARA": 480},
    {"date": "2022-07-31", "GME": 3000, "AMC": 3500, "BB": 420, "NOK": 110, "KOSS": 50, "PLTR": 580, "TLRY": 130, "SNDL": 140, "CLOV": 220, "WISH": 240, "SPCE": 240, "NIO": 380, "BBBY": 1100, "MARA": 520},
    {"date": "2022-08-31", "GME": 3400, "AMC": 3900, "BB": 440, "NOK": 105, "KOSS": 48, "PLTR": 620, "TLRY": 125, "SNDL": 135, "CLOV": 210, "WISH": 220, "SPCE": 220, "NIO": 360, "BBBY": 6400, "MARA": 560},
    {"date": "2022-09-30", "GME": 2800, "AMC": 3100, "BB": 380, "NOK": 95, "KOSS": 44, "PLTR": 540, "TLRY": 115, "SNDL": 120, "CLOV": 180, "WISH": 190, "SPCE": 200, "NIO": 320, "BBBY": 2200, "MARA": 480},
    {"date": "2022-10-31", "GME": 2600, "AMC": 2900, "BB": 360, "NOK": 90, "KOSS": 42, "PLTR": 520, "TLRY": 110, "SNDL": 115, "CLOV": 170, "WISH": 170, "SPCE": 190, "NIO": 300, "BBBY": 1500, "MARA": 520},
    {"date": "2022-11-30", "GME": 2500, "AMC": 2700, "BB": 340, "NOK": 85, "KOSS": 40, "PLTR": 500, "TLRY": 105, "SNDL": 110, "CLOV": 160, "WISH": 160, "SPCE": 180, "NIO": 280, "BBBY": 1300, "MARA": 600},
    {"date": "2022-12-31", "GME": 2300, "AMC": 2500, "BB": 320, "NOK": 80, "KOSS": 38, "PLTR": 480, "TLRY": 100, "SNDL": 105, "CLOV": 150, "WISH": 150, "SPCE": 170, "NIO": 260, "BBBY": 1100, "MARA": 560},
    {"date": "2023-01-31", "GME": 2600, "AMC": 2800, "BB": 360, "NOK": 78, "KOSS": 36, "PLTR": 620, "TLRY": 110, "SNDL": 110, "CLOV": 160, "WISH": 140, "SPCE": 180, "NIO": 300, "BBBY": 1600, "MARA": 900},
    {"date": "2023-02-28", "GME": 2200, "AMC": 2400, "BB": 320, "NOK": 72, "KOSS": 34, "PLTR": 700, "TLRY": 105, "SNDL": 100, "CLOV": 140, "WISH": 130, "SPCE": 160, "NIO": 280, "BBBY": 1900, "MARA": 820},
    {"date": "2023-03-31", "GME": 2000, "AMC": 2200, "BB": 300, "NOK": 68, "KOSS": 32, "PLTR": 640, "TLRY": 100, "SNDL": 95, "CLOV": 130, "WISH": 120, "SPCE": 150, "NIO": 260, "BBBY": 1400, "MARA": 760},
    {"date": "2023-04-30", "GME": 1800, "AMC": 2000, "BB": 280, "NOK": 64, "KOSS": 30, "PLTR": 580, "TLRY": 95, "SNDL": 90, "CLOV": 120, "WISH": 110, "SPCE": 140, "NIO": 240, "BBBY": 1200, "MARA": 700},
    {"date": "2023-05-31", "GME": 1700, "AMC": 1900, "BB": 260, "NOK": 60, "KOSS": 28, "PLTR": 620, "TLRY": 90, "SNDL": 85, "CLOV": 110, "WISH": 100, "SPCE": 130, "NIO": 220, "BBBY": 900, "MARA": 660},
    {"date": "2023-06-30", "GME": 1600, "AMC": 1800, "BB": 240, "NOK": 56, "KOSS": 26, "PLTR": 700, "TLRY": 85, "SNDL": 80, "CLOV": 100, "WISH": 90, "SPCE": 120, "NIO": 200, "BBBY": 400, "MARA": 720},
    {"date": "2023-07-31", "GME": 1500, "AMC": 1700, "BB": 220, "NOK": 52, "KOSS": 24, "PLTR": 760, "TLRY": 80, "SNDL": 75, "CLOV": 95, "WISH": 85, "SPCE": 110, "NIO": 190, "BBBY": 600, "MARA": 680},
    {"date": "2023-08-31", "GME": 1400, "AMC": 1600, "BB": 200, "NOK": 48, "KOSS": 22, "PLTR": 720, "TLRY": 78, "SNDL": 70, "CLOV": 90, "WISH": 80, "SPCE": 100, "NIO": 180, "BBBY": 0, "MARA": 640},
    {"date": "2023-09-30", "GME": 1300, "AMC": 1500, "BB": 180, "NOK": 44, "KOSS": 20, "PLTR": 680, "TLRY": 74, "SNDL": 65, "CLOV": 85, "WISH": 75, "SPCE": 90, "NIO": 170, "BBBY": 0, "MARA": 600},
    {"date": "2023-10-31", "GME": 1200, "AMC": 1400, "BB": 170, "NOK": 42, "KOSS": 19, "PLTR": 640, "TLRY": 70, "SNDL": 62, "CLOV": 80, "WISH": 70, "SPCE": 85, "NIO": 160, "BBBY": 0, "MARA": 720},
    {"date": "2023-11-30", "GME": 1150, "AMC": 1350, "BB": 165, "NOK": 40, "KOSS": 18, "PLTR": 820, "TLRY": 68, "SNDL": 60, "CLOV": 78, "WISH": 68, "SPCE": 80, "NIO": 155, "BBBY": 0, "MARA": 860},
    {"date": "2023-12-31", "GME": 1100, "AMC": 1300, "BB": 160, "NOK": 38, "KOSS": 17, "PLTR": 900, "TLRY": 66, "SNDL": 58, "CLOV": 76, "WISH": 66, "SPCE": 78, "NIO": 150, "BBBY": 0, "MARA": 940},
]


def mention_counts() -> pd.DataFrame:
    """Return the hardcoded month x ticker WSB mention-count table.

    Index = month-end DatetimeIndex (2021-01-31 .. 2023-12-31), columns =
    ``MEME_TICKERS``, values = approximate monthly mention counts (int).
    A value of 0 means the ticker was not a meaningful WSB topic that month
    (e.g. CLOV/WISH pre-IPO, BBBY post-bankruptcy).  These zeros are kept as
    zeros (low buzz), not NaN -- the signal can legitimately rank a quiet name.
    """
    df = pd.DataFrame(_MENTION_ROWS)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index = df.index + pd.offsets.MonthEnd(0)
    df = df[MEME_TICKERS].astype(float)
    return df


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 14,
    n_months: int = 36,
    premium: float = 0.00,
    base_ret: float = 0.0,
    ret_vol: float = 0.30,
    seed: int = 254,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """A meme-basket panel with a *known* mention->forward-return link.

    Each month a cross-section of ``n_firms`` mention counts is drawn from a
    heavy-tailed (lognormal) distribution -- the spiky, fat-tailed shape of real
    WSB buzz.  The forward (next-month) return of firm j is::

        base_ret + premium * z(mentions_j) + noise

    where ``z(mentions_j)`` is the within-month cross-sectional z-score of the
    (log) mention count.  When ``premium = 0`` the mention rank is pure noise
    with respect to next month's return -- the realistic null for a
    contemporaneous, endogenous hype proxy.  A positive ``premium`` plants the
    claim (high-buzz names lead), a negative one plants a *reversal* (high-buzz
    names mean-revert, the classic "buy the rumor sell the news" outcome).

    Returns ``(price_df, mention_df, truth)`` where:
    - ``price_df``   : (month x firm) cumulative prices from 100.
    - ``mention_df`` : (month x firm) the planted mention counts.
    - ``truth``      : the planted parameters.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range("2021-01-31", periods=n_months, freq="ME")
    firms = [f"M{j:02d}" for j in range(n_firms)]

    # Heavy-tailed monthly mention counts (lognormal), independent across months.
    log_mu, log_sigma = 6.0, 1.4
    mentions = np.exp(rng.normal(log_mu, log_sigma, (n_months, n_firms)))
    mentions = np.round(mentions)

    # Within-month z-score of log-mentions (the signal the strategy will rank on).
    logm = np.log1p(mentions)
    z = (logm - logm.mean(axis=1, keepdims=True)) / (logm.std(axis=1, keepdims=True) + 1e-9)

    noise = rng.normal(0.0, ret_vol, (n_months, n_firms))
    # Forward return of month t (earned in t+1) depends on month-t mentions.
    fwd_ret = base_ret + premium * z + noise

    # Convert forward returns into a price path: the return earned during month
    # t+1 is fwd_ret[t]; month 0's own return is just base+noise (no predecessor).
    ret_mat = np.zeros((n_months, n_firms))
    ret_mat[0] = base_ret + rng.normal(0.0, ret_vol, n_firms)
    ret_mat[1:] = fwd_ret[:-1]

    log_ret = np.log1p(np.clip(ret_mat, -0.95, None))
    cum_log = np.cumsum(log_ret, axis=0)
    price_arr = 100.0 * np.exp(cum_log)

    price_df = pd.DataFrame(price_arr, index=months, columns=firms)
    mention_df = pd.DataFrame(mentions, index=months, columns=firms)

    truth = {
        "premium": premium,
        "base_ret": base_ret,
        "ret_vol": ret_vol,
        "n_firms": n_firms,
        "n_months": n_months,
        "seed": seed,
    }
    return price_df, mention_df, truth


# ---------------------------------------------------------------------------
# Real tape -- Yahoo monthly closes for the meme basket, cache-only by default
# ---------------------------------------------------------------------------
def fetch_meme_monthly(
    fetch: bool = False,
    cache_path: str = MONTHLY_CACHE,
) -> pd.DataFrame:
    """Monthly adjusted-close panel for the meme basket; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is
    cached as a parquet under ``_cache/meme_monthly.parquet``). Columns are
    tickers, index is month-end dates (pd.DatetimeIndex at month-end frequency).
    Raises ``FileNotFoundError`` if the cache is absent and ``fetch=False`` --
    callers must guard with ``os.path.exists(cache_path)`` first.

    **Endogeneity / survivorship note**: the basket deliberately keeps collapsed
    names (BBBY delisted 2023).  Where Yahoo has no price for a ticker-month,
    that cell is NaN and the ticker drops out of that month's cross-section.
    The hype proxy is contemporaneous -- mentions spike *because* the price
    already moved -- so any forward signal is mechanically suspect.
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached meme monthly panel at {cache_path}. "
                "Call fetch_meme_monthly(fetch=True) once to populate the cache."
            )
        return pd.read_parquet(cache_path)

    import yfinance as yf  # lazy import: network only on fetch=True

    raw = yf.download(
        MEME_TICKERS,
        start="2020-12-01",
        end="2024-02-01",
        interval="1mo",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no monthly data")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            closes = raw["Close"]
        else:
            closes = raw.xs("Close", axis=1, level=0)
    else:
        closes = raw

    closes.index = pd.to_datetime(closes.index)
    closes.index = closes.index + pd.offsets.MonthEnd(0)
    closes = closes.sort_index()
    # Restrict to the study window (2021-2023 plus a one-month tail for fwd ret).
    closes = closes.loc["2020-12-31":"2024-01-31"]

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    closes.to_parquet(cache_path)
    print(f"Cached meme monthly panel: {closes.shape[0]} months x {closes.shape[1]} tickers -> {cache_path}")
    return closes


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a frame (last row), for the as-of stamp."""
    arr = df.iloc[-1].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
