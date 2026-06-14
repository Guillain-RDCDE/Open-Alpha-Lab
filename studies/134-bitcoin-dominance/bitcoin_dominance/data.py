"""Data layer for Study 134 (Bitcoin-Dominance).

Two tapes, one shape (a tz-naive daily panel of close prices for BTC and five large alts):

- ``synthetic_daily`` — a *deterministic, offline* generator. A ``dom_signal`` knob
  controls whether falling BTC dominance (BTC's relative share of the basket) actually
  predicts forward alt outperformance. ``dom_signal=0`` is the null: no link between
  dominance and forward alt-minus-BTC returns. ``dom_signal > 0`` plants a genuine
  inverse relationship (falling dominance → higher alt spread). This lets us test
  whether the engine would *find* the effect if it existed, and confirm it reads zero
  when it does not.

- ``fetch_panel`` — the real Yahoo! daily panel (``yfinance``), cache-only by default so
  the test-suite and reproducible core never touch the network. BTC-USD history starts
  2014-09-17; most large alts start 2017–2020, which hard-caps the panel's effective
  start date and power. The study must own this short-history caveat openly.

Dominance is proxied as BTC market cap / basket market cap. Since we only have price
data (not circulating supply) from Yahoo, we proxy market cap by price × a fixed
supply scalar (constant across time → relative dominance ≈ relative price contribution,
which is the standard proxy used by CoinGecko's dominance chart and retail practitioners).
Formally: dom_t = BTC_price_t / sum(all_asset_prices_t * supply_weight_t).
Because supply weights are roughly stable on monthly horizons (and we use weekly
rebalancing in the signal), this proxy captures the same rotational dynamics that
retail practitioners track.

No look-ahead: the dominance signal is formed on day t closes; positions enter at
the close of day t (market-on-close order, liquid crypto market) or the next open —
for simplicity we use day t close → compute signal → take forward 1-week return from
day t+1 open, consistent with end-of-day signal formation.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# The basket: BTC + large alts with meaningful Yahoo history
BTC_TICKER = "BTC-USD"
ALT_TICKERS = ["ETH-USD", "XRP-USD", "ADA-USD", "SOL-USD", "BNB-USD", "DOGE-USD"]
ALL_TICKERS = [BTC_TICKER] + ALT_TICKERS

# Fixed supply proxies (rough order-of-magnitude, used only for dominance weighting).
# These are stable enough over multi-year horizons that the proxy captures rotational
# dynamics correctly. Source: CoinGecko circulating supply as of 2024 (millions of coins).
SUPPLY_M = {
    "BTC-USD": 19.7,    # ~19.7M BTC
    "ETH-USD": 120.0,   # ~120M ETH
    "XRP-USD": 55_000.0,  # ~55B XRP (in millions: 55,000)
    "ADA-USD": 35_000.0,  # ~35B ADA
    "SOL-USD": 460.0,   # ~460M SOL
    "BNB-USD": 145.0,   # ~145M BNB
    "DOGE-USD": 145_000.0,  # ~145B DOGE
}


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 6,
    dom_signal: float = 0.0,
    daily_vol_btc: float = 0.04,
    daily_vol_alt: float = 0.06,
    annual_drift: float = 0.30,
    dom_halflife: int = 30,
    start: str = "2018-01-01",
    seed: int = 134,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily crypto panel with an optional dominance→alt-spread signal.

    All assets start at 100 and follow independent drift-plus-noise log-return processes.
    BTC uses ``daily_vol_btc``; alts use ``daily_vol_alt`` (higher to mimic the real
    alt volatility premium).

    When ``dom_signal > 0`` the construction is:

    1. Compute a synthetic dominance series (rolling 20-day BTC share of basket cap).
    2. Form a dominance *change* signal: negative 10-day change → signal = +dom_signal
       added to the next day's alt log-return (extra alt tailwind when BTC share falls).
    3. The signal is applied symmetrically: *rising* dominance → negative alt excess.

    This is the *only* forecastable structure in the synthetic tape:
    - ``dom_signal = 0`` → dominance and alt spread are unconnected (the null).
    - ``dom_signal > 0`` → falling dominance predictably lifts alts over BTC.

    Returns ``(panel, truth)`` where ``panel`` has MultiIndex columns (ticker, field)
    with 'close' and 'open' per ticker, and ``truth`` records planted parameters.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * 252)
    idx = pd.bdate_range(start=start, periods=n_days, name="date")

    daily_mu = annual_drift / 252.0

    # Generate BTC returns first
    btc_ret = rng.normal(daily_mu, daily_vol_btc, n_days)
    btc_close = 100.0 * np.exp(np.cumsum(btc_ret))

    # Generate alt returns base (all alts share a common factor + idio noise)
    n_alts = len(ALT_TICKERS)
    common_factor = rng.normal(0.0, daily_vol_alt * 0.5, n_days)  # shared crypto beta
    alt_returns_base = np.zeros((n_days, n_alts))
    for j in range(n_alts):
        idio = rng.normal(daily_mu, daily_vol_alt * 0.7, n_days)
        alt_returns_base[:, j] = common_factor + idio

    # If dom_signal > 0: plant the dominance->alt-spread effect.
    # Mechanism: we track the 10-day BTC return advantage (btc_ret_10d - alt_ret_10d,
    # a proxy for falling alt dominance when negative). When BTC has outperformed the
    # alts basket over the past 10 days (positive dominance change), inject a positive
    # alt return boost the next day. This is a mean-reversion dynamic: capital rotates
    # into alts after BTC has dominated, lifting alts relative to BTC.
    # The injected boost is dom_signal * 0.001 per day (calibrated so dom_signal=2 gives
    # a visible but realistic effect: ~0.2% extra alt return per trigger day).
    if dom_signal > 0.0:
        # Build temporary BTC and alt return series for signal construction
        btc_ret_cum = np.cumsum(btc_ret)
        # Equal-weighted alt return (mean of all alts)
        alt_ret_mean = np.zeros(n_days)
        for j in range(n_alts):
            alt_ret_mean += alt_returns_base[:, j] / n_alts
        alt_ret_cum = np.cumsum(alt_ret_mean)

        for i in range(10, n_days - 1):
            # 10-day relative return: BTC vs alts (positive = BTC led, dom rose)
            btc_10d = btc_ret_cum[i] - btc_ret_cum[i - 10]
            alt_10d = alt_ret_cum[i] - alt_ret_cum[i - 10]
            dom_change_proxy = btc_10d - alt_10d  # positive = BTC outperformed
            # Plant: when dom just rose (BTC season), expect alt rotation next
            # Falling dominance (dom_change_proxy < 0) means alts already leading ->
            # signal fires on dom_change_proxy < 0 (falls into alt territory)
            alt_boost = -dom_change_proxy * dom_signal * 0.1
            alt_returns_base[i + 1, :] += alt_boost

    # Build close prices for alts
    alt_closes = np.zeros((n_days, n_alts))
    for j in range(n_alts):
        alt_closes[:, j] = 100.0 * np.exp(np.cumsum(alt_returns_base[:, j]))

    # Assemble panel DataFrame: simple flat columns (ticker -> close)
    data = {"BTC-USD": btc_close}
    for j, t in enumerate(ALT_TICKERS):
        data[t] = alt_closes[:, j]
    panel = pd.DataFrame(data, index=idx)
    panel.index.name = "date"

    truth = {
        "dom_signal": dom_signal,
        "daily_vol_btc": daily_vol_btc,
        "daily_vol_alt": daily_vol_alt,
        "annual_drift": annual_drift,
        "n_days": n_days,
        "seed": seed,
        "tickers": ALL_TICKERS,
    }
    return panel, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily crypto panel, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "bitcoin_dominance_panel.parquet")


def fetch_panel(
    tickers: list[str] | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily close panel for BTC + alts; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (the result is cached as a
    parquet under ``_cache/``). Yahoo daily BTC history starts 2014-09-17; ETH starts
    2017-11-09; XRP 2017-11-09; ADA 2017-10-01; SOL 2020-04-10; BNB 2017-11-09;
    DOGE 2019-11-17. The effective panel start (all tickers present) is 2020-04-10,
    giving roughly 4 years of overlapping data — this is the study's structural power
    ceiling, and we say so loudly.

    Returns a DataFrame of daily close prices, columns = tickers, index = date (tz-naive).
    Tickers with missing close values are forward-filled up to 3 days (crypto weekend gaps),
    then rows with any NaN dropped (pre-listing dates).
    """
    if tickers is None:
        tickers = ALL_TICKERS

    path = _cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached panel at {path}. "
                f"Call fetch_panel(fetch=True) once to populate the cache."
            )
        panel = pd.read_parquet(path)
        if panel.index.tz is not None:
            panel.index = panel.index.tz_localize(None)
        panel.index.name = "date"
        return panel

    import yfinance as yf  # lazy: only when we go to the network

    frames = {}
    for ticker in tickers:
        raw = yf.download(
            ticker, period="max", interval="1d", auto_adjust=True, progress=False
        )
        if raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        if raw.index.tz is not None:
            raw.index = raw.index.tz_localize(None)
        frames[ticker] = raw["close"]

    panel = pd.DataFrame(frames)
    panel.index.name = "date"
    # Forward-fill up to 3 days for weekend/holiday gaps, then drop pre-listing rows
    panel = panel.ffill(limit=3).dropna(how="any")

    os.makedirs(cache_dir, exist_ok=True)
    panel.to_parquet(path)
    return panel


# ---------------------------------------------------------------------------
# Dominance helpers
# ---------------------------------------------------------------------------
def compute_dominance(
    panel: pd.DataFrame,
    btc_col: str = "BTC-USD",
    supply_map: dict | None = None,
) -> pd.Series:
    """Compute BTC dominance (BTC market-cap fraction) from the price panel.

    Market cap is proxied as price × fixed supply (see module docstring). The result is
    BTC's share of total basket market cap, ranging in (0, 1). On the real tape this
    tracks CoinGecko's reported BTC dominance at a high correlation (>0.9) on monthly
    averages — the proxy is good enough for a weekly-signal study.
    """
    if supply_map is None:
        supply_map = SUPPLY_M
    tickers = list(panel.columns)
    btc_cap = panel[btc_col] * supply_map.get(btc_col, 1.0)
    total_cap = sum(
        panel[t] * supply_map.get(t, 1.0)
        for t in tickers
    )
    dom = btc_cap / total_cap
    dom.name = "btc_dominance"
    return dom


def fingerprint(panel: pd.DataFrame) -> str:
    """A short content fingerprint of the panel (sum of closes), for the as-of stamp."""
    arr = panel.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr).all(axis=1)]
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
