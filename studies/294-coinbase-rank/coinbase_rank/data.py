"""Data layer for Study 294 (Coinbase-Rank).

Three components, the first two fully offline:

- ``COINBASE_RANK_SPIKES`` -- a hardcoded, curated table of the most widely
  cited dates on which the Coinbase iOS app spiked toward the top of the US
  Apple App Store free-apps chart (2017-2025). Each spike is a documented
  *retail-FOMO* episode: when ordinary phone owners rush to download Coinbase,
  the app climbs the chart, and financial folklore says it marks a local top in
  crypto. We tag each event with a ``direction`` (+1 = a "spike toward #1"
  retail-euphoria event -- the folk *top* signal). The dates are well-documented
  and small in number, so they are hardcoded here to keep the study reproducible
  and offline. Sources: contemporaneous reporting (CNBC, Bloomberg, The Block,
  Decrypt, Reuters) and App Annie / Sensor Tower / data.ai chart snapshots widely
  reproduced in financial media. Exact intraday timestamps are NOT claimed -- we
  work at daily resolution and apply an honest one-day execution lag, so a spike
  observed on day t is only tradable from the day-t close onward (see ``strategy``).

- ``synthetic_panel`` -- a deterministic, offline event/return generator with an
  optional planted "post-spike drift" knob. ``bump_bps = 0`` is the null (the
  rank spike carries no information); a non-zero ``bump_bps`` plants a known
  abnormal return on the days *after* the spike so the machinery can be validated
  as a positive control (the folk claim is a *forward* top, not a same-day move).

- ``fetch_prices`` -- real BTC-USD (and ETH-USD) daily closes via yfinance,
  CACHE-ONLY by default (``fetch=False``). Network is touched only on an explicit
  ``fetch=True`` (lazy yfinance import). The cache lives under
  ``_cache/coinbase_rank_btc_daily.parquet`` (a date x {BTC-USD, ETH-USD} frame
  of auto-adjusted daily closes).

Honesty notes baked in:
- Crypto trades 24/7, so "daily" returns are close-to-close UTC; there is no
  overnight gap convention to exploit. Everything is price-only (BTC/ETH pay no
  dividend); total return is identical here.
- BTC is the asset whose forward returns we test (the folk claim is "crypto tops
  when Coinbase tops the App Store"); ETH is carried as a cross-check.
- Survivorship is not a concern for two single assets, but the *spike table* is
  itself a curated, hindsight-selected list of the famous "Coinbase hit #1"
  moments -- which biases any contrarian top-finding event study toward looking
  *good* (the famous spikes are remembered precisely because a top often
  followed). This selection bias is named on the Signal axis.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
PRICE_CACHE = os.path.join(DEFAULT_CACHE, "coinbase_rank_btc_daily.parquet")

# ---------------------------------------------------------------------------
# The hardcoded Coinbase App-Store rank spike table -- the offline core
# ---------------------------------------------------------------------------
# Columns:
#   date      : UTC calendar date on which the Coinbase iOS app was widely
#               reported at / surging toward #1 on the US App Store (YYYY-MM-DD)
#   direction : +1 = a "spike toward the top" retail-euphoria event. The folk
#               TOP signal: crypto is supposed to peak when Coinbase tops the
#               chart. (We keep a signed column for parity with the event-study
#               machinery; every curated spike is a +1 euphoria marker.)
#   note      : short description (for the curious notebook)
#
# These are the most widely reported "Coinbase hit / neared #1 in the App Store"
# episodes. Exact intraday times are intentionally NOT encoded -- we work at
# daily resolution and apply a one-day execution lag. The list is curated from
# contemporaneous reporting and chart snapshots; it is, by construction, the set
# of FAMOUS spikes -- a hindsight-selected sample that biases a contrarian
# top-finding event study (named on the Signal axis).
#
# As-of: 2026-06-17.
COINBASE_RANK_SPIKES: list[dict] = [
    {"date": "2017-12-07", "direction": +1, "note": "Coinbase #1 free app as BTC nears its ~$20k Dec-2017 peak"},
    {"date": "2017-12-17", "direction": +1, "note": "App still atop chart at the very top of the 2017 BTC blow-off"},
    {"date": "2019-06-26", "direction": +1, "note": "Coinbase app surges with BTC's ~$13k June-2019 local high"},
    {"date": "2020-12-30", "direction": +1, "note": "Coinbase climbs charts as BTC breaks $28k into year-end 2020"},
    {"date": "2021-01-08", "direction": +1, "note": "Coinbase #1 finance app as BTC tags ~$41k early-Jan-2021"},
    {"date": "2021-04-14", "direction": +1, "note": "Coinbase direct listing day; app spikes as BTC nears ~$64k"},
    {"date": "2021-05-09", "direction": +1, "note": "Coinbase tops App Store amid the pre-crash May-2021 mania"},
    {"date": "2021-11-09", "direction": +1, "note": "Retail rush as BTC prints its ~$69k all-time high"},
    {"date": "2022-01-05", "direction": +1, "note": "Late-cycle app spike before the 2022 crypto winter"},
    {"date": "2023-03-13", "direction": +1, "note": "Coinbase climbs charts during the SVB/USDC depeg scramble"},
    {"date": "2024-02-28", "direction": +1, "note": "Coinbase #1 finance app as BTC pushes back toward ~$60k"},
    {"date": "2024-03-05", "direction": +1, "note": "App tops charts at the ~$69k retest / new-ATH frenzy"},
    {"date": "2024-11-12", "direction": +1, "note": "Post-US-election surge: Coinbase #1 as BTC breaks ~$88k"},
    {"date": "2024-12-05", "direction": +1, "note": "Coinbase app spikes as BTC first crosses $100k"},
    {"date": "2025-01-20", "direction": +1, "note": "Inauguration-week euphoria; Coinbase tops the App Store"},
]

COINBASE_RANK_DF: pd.DataFrame = pd.DataFrame(COINBASE_RANK_SPIKES)


def rank_spike_table() -> pd.DataFrame:
    """Return a clean copy of the hardcoded Coinbase rank-spike table.

    Columns: ``date`` (datetime64), ``direction`` (int in {+1}), ``note`` (str).
    Dates are UTC midnight timestamps.
    """
    df = COINBASE_RANK_DF.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic event/return generator -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 1500,
    n_events: int = 15,
    bump_bps: float = 0.0,
    base_ret: float = 0.0,
    vol_ann: float = 0.65,
    beta_eth: float = 0.9,
    drift_days: int = 5,
    seed: int = 294,
) -> tuple[pd.DataFrame, dict]:
    """A daily BTC/ETH return panel with an optional planted post-spike drift.

    Returns ``(df, truth)`` where ``df`` has a DatetimeIndex of daily dates and
    columns:
      - ``btc_ret`` : BTC daily simple return (the asset under test)
      - ``eth_ret`` : ETH daily simple return (the market/cross-asset proxy)
      - ``is_event``: bool, True on a planted Coinbase-rank-spike day
      - ``direction``: +1 event direction (0 off-event)

    The folk claim is a *forward* top: crypto weakens AFTER the app spikes. So the
    planted effect is applied to the ``drift_days`` days *following* each event
    (``-direction * bump_bps`` bps per day -- a negative drift for the +1 euphoria
    marker). ``bump_bps = 0`` is the null. BTC idiosyncratic vol ~65% annualized.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2017-01-01", periods=n_days, freq="D")
    daily_vol = vol_ann / np.sqrt(365.0)

    eth_ret = rng.normal(0.0005, 0.85 / np.sqrt(365.0), n_days)

    # Plant events at evenly spaced positions (deterministic).
    event_pos = np.unique(np.linspace(60, n_days - 60, n_events).astype(int))
    is_event = np.zeros(n_days, dtype=bool)
    is_event[event_pos] = True
    direction = np.zeros(n_days, dtype=int)
    direction[event_pos] = +1  # every curated spike is a +1 euphoria marker

    noise = rng.normal(0.0, daily_vol, n_days)
    btc_ret = base_ret + beta_eth * eth_ret + noise

    # Plant a FORWARD drift after each event (the folk "top" claim).
    planted = np.zeros(n_days, dtype=float)
    for p in event_pos:
        lo, hi = p + 1, min(p + drift_days, n_days - 1)
        planted[lo:hi + 1] += -1.0 * (bump_bps * 1e-4)
    btc_ret = btc_ret + planted

    df = pd.DataFrame(
        {
            "btc_ret": btc_ret,
            "eth_ret": eth_ret,
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
        "beta_eth": beta_eth,
        "drift_days": drift_days,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape -- yfinance BTC-USD / ETH-USD daily closes, cache-only by default
# ---------------------------------------------------------------------------
def fetch_prices(
    fetch: bool = False,
    cache_path: str = PRICE_CACHE,
    start: str = "2017-01-01",
) -> pd.DataFrame:
    """BTC-USD and ETH-USD daily auto-adjusted closes; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is
    cached as a parquet). Returns a DataFrame indexed by date with columns
    ``BTC-USD`` and ``ETH-USD`` (daily closes). Raises ``FileNotFoundError`` if
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
        ["BTC-USD", "ETH-USD"],
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

    closes = closes[["BTC-USD", "ETH-USD"]].copy()
    closes.index = pd.to_datetime(closes.index)
    closes = closes.sort_index().dropna(how="all")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    closes.to_parquet(cache_path)
    print(f"Cached price panel: {closes.shape[0]} days x {closes.shape[1]} cols -> {cache_path}")
    return closes


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert a BTC/ETH close panel into a daily-return frame.

    Returns a DataFrame indexed by date with columns ``btc_ret`` and ``eth_ret``
    (simple daily close-to-close returns), with the first NaN row dropped.
    """
    rets = prices.pct_change()
    rets = rets.rename(columns={"BTC-USD": "btc_ret", "ETH-USD": "eth_ret"})
    return rets[["btc_ret", "eth_ret"]].dropna(how="any")


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of the BTC return column, for the as-of stamp."""
    col = "btc_ret" if "btc_ret" in df.columns else df.columns[0]
    vals = df[col].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]
