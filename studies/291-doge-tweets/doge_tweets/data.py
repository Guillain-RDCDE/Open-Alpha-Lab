"""Data layer for Study 291 (Doge-Tweets).

Three components, the first two fully offline:

- ``MUSK_DOGE_TWEETS`` -- a hardcoded, curated table of the most widely cited
  Elon Musk Dogecoin-related tweets / public statements (2019-2025), each with a
  UTC date and a ``direction`` tag (+1 = bullish/promotional, -1 = dismissive or
  negative, 0 = ambiguous). The dates are well-documented and small in number, so
  they are hardcoded here to keep the study reproducible and offline. Sources:
  contemporaneous reporting (Reuters, CNBC, Bloomberg, The Verge), Musk's public
  X/Twitter timeline, and the SEC's references in the 2021-2024 DOGE litigation.
  Exact intraday timestamps are NOT claimed -- we work at daily resolution and
  apply an honest one-day execution lag, so a tweet on day t is only tradable from
  the day-t close onward (see ``strategy``).

- ``synthetic_panel`` -- a deterministic, offline event/return generator with an
  optional planted "tweet bump" knob. ``bump_bps = 0`` is the null (tweets carry
  no information); a positive ``bump_bps`` plants a known abnormal return on the
  tweet day so the machinery can be validated as a positive control.

- ``fetch_prices`` -- real DOGE-USD and BTC-USD daily closes via yfinance,
  CACHE-ONLY by default (``fetch=False``). Network is touched only on an explicit
  ``fetch=True`` (lazy yfinance import). The cache lives under
  ``_cache/doge_btc_daily.parquet`` (a date x {DOGE-USD, BTC-USD} frame of
  auto-adjusted daily closes).

Honesty notes baked in:
- Crypto trades 24/7, so "daily" returns are close-to-close UTC; there is no
  overnight gap convention to exploit. We label everything as price-only
  (DOGE/BTC pay no dividend) total return is identical here.
- Survivorship is not a concern for two single assets, but the *tweet table* is
  itself a curated, hindsight-selected list of the famous tweets -- which biases
  any event study UPWARD. This is named on the Signal axis.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
PRICE_CACHE = os.path.join(DEFAULT_CACHE, "doge_btc_daily.parquet")

# ---------------------------------------------------------------------------
# The hardcoded Musk/Doge tweet table -- the study's deterministic offline core
# ---------------------------------------------------------------------------
# Columns:
#   date      : UTC calendar date of the tweet / public statement (YYYY-MM-DD)
#   direction : +1 = bullish/promotional (e.g. "Dogecoin is the people's crypto"),
#               -1 = dismissive / negative,
#                0 = ambiguous / joke with no clear directional intent
#   note      : short description (for the curious notebook)
#
# These are the most widely reported Musk-Dogecoin events. Exact intraday times
# are intentionally NOT encoded -- we work at daily resolution and apply a one-day
# execution lag. The list is curated from contemporaneous reporting and is, by
# construction, the set of FAMOUS tweets -- a hindsight-selected sample that biases
# an event study upward (named on the Signal axis).
#
# As-of: 2026-06-17.
MUSK_DOGE_TWEETS: list[dict] = [
    {"date": "2019-04-02", "direction": +1, "note": "Musk: 'Dogecoin might be my fav cryptocurrency'"},
    {"date": "2019-04-04", "direction": +1, "note": "Musk briefly 'CEO of Dogecoin' Twitter poll"},
    {"date": "2020-07-18", "direction": +1, "note": "'It's inevitable' Doge meme tweet"},
    {"date": "2020-12-20", "direction": +1, "note": "'One word: Doge'"},
    {"date": "2021-01-28", "direction": +1, "note": "'Doge' single-word tweet during GME mania"},
    {"date": "2021-02-04", "direction": +1, "note": "'Dogecoin is the people's crypto' + 'No highs, no lows, only Doge'"},
    {"date": "2021-02-07", "direction": +1, "note": "'Who let the Doge out' / Doge-to-the-Moon tweets"},
    {"date": "2021-02-24", "direction": +1, "note": "'Bought some Dogecoin for lil X'"},
    {"date": "2021-04-15", "direction": +1, "note": "'Doge Barking at the Moon' (Joan Miro) image"},
    {"date": "2021-04-28", "direction": +1, "note": "'The Dogefather SNL May 8' announcement"},
    {"date": "2021-05-08", "direction": -1, "note": "SNL appearance: 'it's a hustle' -> DOGE sold off"},
    {"date": "2021-05-13", "direction": +1, "note": "Working with Doge devs to improve transaction efficiency"},
    {"date": "2021-05-20", "direction": +1, "note": "'How much is that Doge in the window' tweet"},
    {"date": "2021-06-03", "direction": +1, "note": "'Baby Doge' tweet"},
    {"date": "2021-07-01", "direction": +1, "note": "Reaffirms support for Doge as currency"},
    {"date": "2021-12-14", "direction": +1, "note": "Tesla to accept Doge for some merchandise"},
    {"date": "2022-01-14", "direction": +1, "note": "Tesla merch buyable with Doge goes live"},
    {"date": "2022-04-25", "direction": +1, "note": "Twitter acquisition agreed -> Doge speculation"},
    {"date": "2022-10-28", "direction": +1, "note": "Musk closes Twitter deal; Doge rallies"},
    {"date": "2023-04-03", "direction": +1, "note": "Twitter bluebird logo swapped for Doge logo"},
    {"date": "2024-03-15", "direction": +1, "note": "X/Doge payments speculation tweet"},
    {"date": "2024-12-04", "direction": +1, "note": "'DOGE' Dept. of Government Efficiency branding tweets"},
    {"date": "2025-01-20", "direction": +1, "note": "D.O.G.E. department launch references"},
]

MUSK_DOGE_DF: pd.DataFrame = pd.DataFrame(MUSK_DOGE_TWEETS)


def tweet_table() -> pd.DataFrame:
    """Return a clean copy of the hardcoded Musk/Doge tweet table.

    Columns: ``date`` (datetime64), ``direction`` (int in {+1, -1, 0}),
    ``note`` (str). Dates are UTC midnight timestamps.
    """
    df = MUSK_DOGE_DF.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic event/return generator -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 1000,
    n_events: int = 23,
    bump_bps: float = 0.0,
    base_ret: float = 0.0,
    vol_ann: float = 1.10,
    beta_btc: float = 0.8,
    seed: int = 291,
) -> tuple[pd.DataFrame, dict]:
    """A daily DOGE/BTC return panel with an optional planted tweet-day bump.

    Returns ``(df, truth)`` where ``df`` has a DatetimeIndex of business-ish daily
    dates and columns:
      - ``doge_ret`` : DOGE daily simple return
      - ``btc_ret``  : BTC daily simple return (the market proxy)
      - ``is_event`` : bool, True on a planted tweet day
      - ``direction``: +1 / -1 / 0 event direction (0 off-event)

    DOGE returns are ``base_ret + beta_btc * btc_ret + noise`` plus, on event
    days, ``direction * bump_bps`` bps of abnormal return. ``bump_bps = 0`` is the
    null: tweets carry no information beyond the market move. ``vol_ann`` is the
    DOGE idiosyncratic annualized vol (~110% reflects Dogecoin's real wildness).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    daily_vol = vol_ann / np.sqrt(365.0)

    btc_ret = rng.normal(0.0005, 0.55 / np.sqrt(365.0), n_days)

    # Plant events at evenly spaced positions (deterministic).
    event_pos = np.unique(np.linspace(50, n_days - 50, n_events).astype(int))
    is_event = np.zeros(n_days, dtype=bool)
    is_event[event_pos] = True
    direction = np.zeros(n_days, dtype=int)
    # Alternate-ish directions but mostly bullish (mirrors the real table).
    dir_choices = rng.choice([+1, +1, +1, -1], size=len(event_pos))
    direction[event_pos] = dir_choices

    noise = rng.normal(0.0, daily_vol, n_days)
    doge_ret = base_ret + beta_btc * btc_ret + noise
    doge_ret = doge_ret + direction * (bump_bps * 1e-4)

    df = pd.DataFrame(
        {
            "doge_ret": doge_ret,
            "btc_ret": btc_ret,
            "is_event": is_event,
            "direction": direction,
        },
        index=dates,
    )
    truth = {
        "n_days": n_days,
        "n_events": int(len(event_pos)),
        "bump_bps": bump_bps,
        "base_ret": base_ret,
        "vol_ann": vol_ann,
        "beta_btc": beta_btc,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape -- yfinance DOGE-USD / BTC-USD daily closes, cache-only by default
# ---------------------------------------------------------------------------
def fetch_prices(
    fetch: bool = False,
    cache_path: str = PRICE_CACHE,
    start: str = "2019-01-01",
) -> pd.DataFrame:
    """DOGE-USD and BTC-USD daily auto-adjusted closes; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is
    cached as a parquet). Returns a DataFrame indexed by date with columns
    ``DOGE-USD`` and ``BTC-USD`` (daily closes). Raises ``FileNotFoundError`` if
    the cache is absent and ``fetch=False``.

    Crypto trades 24/7; these are close-to-close (UTC) prices. Price-only and
    total-return are identical (no dividend).
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached price panel at {cache_path}. "
                "Call fetch_prices(fetch=True) once to populate the cache."
            )
        return pd.read_parquet(cache_path)

    import yfinance as yf  # lazy import: network only on fetch=True

    raw = yf.download(
        ["DOGE-USD", "BTC-USD"],
        start=start,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no daily data")

    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", axis=1, level=0)
    else:
        closes = raw

    closes = closes[["DOGE-USD", "BTC-USD"]].copy()
    closes.index = pd.to_datetime(closes.index)
    closes = closes.sort_index().dropna(how="all")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    closes.to_parquet(cache_path)
    print(f"Cached price panel: {closes.shape[0]} days x {closes.shape[1]} cols -> {cache_path}")
    return closes


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert a DOGE/BTC close panel into a daily-return frame.

    Returns a DataFrame indexed by date with columns ``doge_ret`` and ``btc_ret``
    (simple daily close-to-close returns), with the first NaN row dropped.
    """
    rets = prices.pct_change()
    rets = rets.rename(columns={"DOGE-USD": "doge_ret", "BTC-USD": "btc_ret"})
    return rets[["doge_ret", "btc_ret"]].dropna(how="any")


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of the doge return column, for the as-of stamp."""
    col = "doge_ret" if "doge_ret" in df.columns else df.columns[0]
    vals = df[col].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]
