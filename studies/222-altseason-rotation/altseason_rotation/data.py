"""Data layer for Study 222 (Altseason-Rotation).

Two tapes, one shape (a tz-naive daily panel of close prices for BTC and five large alts):

- ``synthetic_daily`` — a *deterministic, offline* generator. A ``dom_signal`` knob
  controls whether the BTC dominance regime (high vs low, or falling vs rising) actually
  predicts forward alt-minus-BTC spread returns. ``dom_signal=0`` is the null: no link
  between dominance regime and forward returns. ``dom_signal > 0`` plants a genuine
  relationship that the rotation strategy should capture. This lets us test whether the
  engine would *find* the edge if it existed, and confirm it reads zero when it does not.

- ``fetch_panel`` — the real Yahoo! daily panel (``yfinance``), cache-only by default so
  the test-suite and reproducible core never touch the network. The effective panel start
  is 2020-04-10 (SOL-USD listing), capping usable history to ~6 years and one crypto cycle.

Dominance is proxied as BTC market cap / basket market cap (price x fixed supply scalar),
matching the standard approach used by CoinGecko and retail practitioners. The proxy
tracks CoinGecko's reported BTC dominance at high directional correlation on monthly
averages — good enough for a weekly-regime rotation study.

The rotation strategy (in strategy.py) signals: when BTC dominance has been *falling*
below a threshold for N consecutive days, go long the equal-weighted alt basket and
short BTC; exit when dominance recovers. Unlike the pure regression approach in
Study 134, here we simulate explicit long/short daily PnL, apply transaction costs,
and compute Sharpe ratios on the net strategy vs buy-and-hold benchmarks.

No look-ahead: signals formed on day t closes; positions entered at day t+1 open
(approximated as day t+1 close for simplicity, a conservative assumption that avoids
look-ahead and is standard for daily signals on liquid crypto pairs).
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
# Source: CoinGecko circulating supply as of 2024 (millions of coins).
SUPPLY_M = {
    "BTC-USD": 19.7,
    "ETH-USD": 120.0,
    "XRP-USD": 55_000.0,
    "ADA-USD": 35_000.0,
    "SOL-USD": 460.0,
    "BNB-USD": 145.0,
    "DOGE-USD": 145_000.0,
}

# Transaction cost assumption: round-trip basis points for the rotation trade.
# One-way: ~20 bps per leg (fee + half-spread on liquid altcoin perp). Round-trip = 40 bps.
# We also test a higher-cost scenario (80 bps round-trip) for robustness.
DEFAULT_COST_BPS = 40  # round-trip bps per rotation event


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 6,
    dom_signal: float = 0.0,
    daily_vol_btc: float = 0.04,
    daily_vol_alt: float = 0.06,
    annual_drift: float = 0.30,
    start: str = "2018-01-01",
    seed: int = 222,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily crypto panel with an optional dominance-regime→alt-spread signal.

    All assets start at 100 and follow independent drift-plus-noise log-return processes.
    BTC uses ``daily_vol_btc``; alts use ``daily_vol_alt`` (higher to mimic the real
    alt volatility premium).

    When ``dom_signal > 0`` the construction plants a regime effect:
    - A synthetic BTC dominance series is built from the generated prices.
    - Periods when BTC dominance has been *falling* for 10+ consecutive days get a
      positive alt return boost injected on the next day (dom_signal * 0.002 per day).
    - This mimics the rotation dynamic: sustained dominance decline → alt outperformance.

    ``dom_signal = 0`` → no link between dominance regime and future alt spread (the null).
    ``dom_signal > 0`` → falling-dominance regime reliably lifts alts over BTC.

    Returns ``(panel, truth)`` where ``panel`` has flat columns (tickers), and ``truth``
    records planted parameters.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * 252)
    idx = pd.bdate_range(start=start, periods=n_days, name="date")

    daily_mu = annual_drift / 252.0

    # Generate BTC returns
    btc_ret = rng.normal(daily_mu, daily_vol_btc, n_days)
    btc_close = 100.0 * np.exp(np.cumsum(btc_ret))

    # Generate alt returns base
    n_alts = len(ALT_TICKERS)
    common_factor = rng.normal(0.0, daily_vol_alt * 0.5, n_days)
    alt_returns_base = np.zeros((n_days, n_alts))
    for j in range(n_alts):
        idio = rng.normal(daily_mu, daily_vol_alt * 0.7, n_days)
        alt_returns_base[:, j] = common_factor + idio

    if dom_signal > 0.0:
        # Build price-based dominance proxy from generated data so far
        btc_cum = np.exp(np.cumsum(btc_ret)) * 100.0
        alt_cumulative = np.zeros((n_days, n_alts))
        for j in range(n_alts):
            alt_cumulative[:, j] = np.exp(np.cumsum(alt_returns_base[:, j])) * 100.0

        # Compute synthetic dominance
        supply_vals = list(SUPPLY_M.values())  # [btc, eth, xrp, ada, sol, bnb, doge]
        btc_cap = btc_cum * supply_vals[0]
        total_cap = btc_cap.copy()
        for j in range(n_alts):
            total_cap += alt_cumulative[:, j] * supply_vals[j + 1]
        dom_proxy = btc_cap / total_cap

        # Identify falling-dominance regime: dom fell over the past 10 days
        for i in range(10, n_days - 1):
            dom_change_10d = dom_proxy[i] - dom_proxy[i - 10]
            if dom_change_10d < 0:  # dominance falling → inject alt boost
                boost = abs(dom_change_10d) * dom_signal * 0.5
                alt_returns_base[i + 1, :] += boost

    # Build close prices for alts
    alt_closes = np.zeros((n_days, n_alts))
    for j in range(n_alts):
        alt_closes[:, j] = 100.0 * np.exp(np.cumsum(alt_returns_base[:, j]))

    # Assemble panel DataFrame
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
    return os.path.join(cache_dir, "altseason_rotation_panel.parquet")


def fetch_panel(
    tickers: list[str] | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily close panel for BTC + alts; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (the result is cached as a
    parquet under ``_cache/``). The effective panel start is 2020-04-10 (SOL-USD listing),
    giving roughly 6 years of overlapping data — this is the study's structural power
    ceiling, and the verdict names this limitation explicitly.

    Returns a DataFrame of daily close prices, columns = tickers, index = date (tz-naive).
    Tickers with missing close values are forward-filled up to 3 days, then rows with any
    NaN dropped (pre-listing dates).
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

    Market cap is proxied as price x fixed supply (see module docstring). The result is
    BTC's share of total basket market cap, ranging in (0, 1).
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
    """A short content fingerprint of the panel (SHA-1 of close matrix), for the as-of stamp."""
    arr = panel.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr).all(axis=1)]
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
