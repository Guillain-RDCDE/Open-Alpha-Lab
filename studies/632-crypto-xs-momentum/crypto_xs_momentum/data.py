"""Data layer for Study 632 — Crypto Cross-Sectional Momentum.

Two tapes, one shape (a Monday-indexed wide frame of **weekly USD closes**, one column
per coin):

* **Real tape.** Weekly closes for a fixed ~44-coin top-cap universe, spliced from two
  public, keyless sources:

    - **Binance spot 1w klines** (``/api/v3/klines``, weeks open Monday 00:00 UTC) —
      the primary tape from each pair's Binance listing onward. Crucially, Binance keeps
      serving full historical klines for **delisted** pairs, so the panel includes names
      that *died or were kicked off the venue*: EOS (delisted 2025), XMR (2024), WAVES
      (2024), OMG (2024), NANO (2022), MATIC (migrated to POL, 2024), FTM (migrated to S,
      2025), ANT (2024), SRM (Serum — FTX collapse), FTT (FTX's token, −93% week captured)
      and **LUNA including its −100% death week** (the ``LUNAUSDT`` symbol was re-used for
      Terra 2.0 after the halt, so the series is TRUNCATED at 2022-05-15 — keeping the
      crash, dropping the unrelated successor token; see ``LUNA_TRUNCATE``).
    - **yfinance ``-USD`` daily closes**, resampled to the same Monday-open weeks, used
      ONLY to backfill a coin's history *before* its Binance listing (e.g. DOGE trades on
      Binance from 2019-07 but has a Yahoo tape from 2017-11). Both sources are USD prices
      of the same asset; the one splice-week return crosses sources and is noted.

  **Survivorship — named regardless.** The delisted pairs soften but do not remove the
  bias: the *live* two-thirds of the universe is chosen from coins that are still top-cap
  in 2026, and pre-2019 top-20 casualties that never had a fetchable Binance USDT pair
  (BCC/BSV forks, XEM, BTG, many 2017 ICO tokens) are absent. The panel is therefore still
  tilted toward eventual winners; this is said out loud on the Signal axis.

* **Synthetic.** ``synthetic_panel`` — a deterministic weekly price panel with a
  **plantable cross-sectional momentum** knob: each coin's idiosyncratic weekly return is
  an AR(1) with coefficient ``rho`` (``rho = 0`` → the null, no cross-sectional
  predictability; ``rho > 0`` → past idiosyncratic winners keep winning). The positive /
  null control for the quintile machinery. Spans ~9 years (far below the 250-year
  ns-Timestamp ceiling).

Cache-first: ``fetch_panel(fetch=True)`` touches the network ONCE and writes
``_cache/xs_weekly_panel.parquet``; everything else reads the cache. No look-ahead lives
here — signal lagging is in ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
PANEL_CACHE = os.path.join(CACHE_DIR, "xs_weekly_panel.parquet")

WEEKS_PER_YEAR = 52.18  # crypto trades every calendar day; 365.25/7

UA = {"User-Agent": "OpenAlphaLab research desk guillain@poulpe.us"}

# --------------------------------------------------------------------------- #
# Universe (fixed, transparent). Binance USDT spot pairs; coin key = base asset.
# --------------------------------------------------------------------------- #
# Live top-cap names (still listed as of the fetch date).
BINANCE_LIVE = [
    "BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "SOL", "TRX", "DOT", "LTC",
    "BCH", "LINK", "AVAX", "XLM", "ATOM", "ETC", "FIL", "NEAR", "ALGO", "VET",
    "XTZ", "THETA", "NEO", "IOTA", "DASH", "ZEC", "UNI", "AAVE", "SHIB",
    "SAND", "MANA", "AXS", "QTUM",
]
# Pairs DELISTED from Binance (or dead outright) whose full kline history is still
# served — the survivorship softeners. LUNA is truncated (see LUNA_TRUNCATE).
BINANCE_DELISTED = [
    "EOS", "XMR", "WAVES", "OMG", "NANO", "MATIC", "FTM", "FTT", "ANT", "SRM", "LUNA",
]
UNIVERSE = BINANCE_LIVE + BINANCE_DELISTED

# LUNAUSDT klines concatenate old LUNA (crash to $0.00005, halt 2022-05-13) and the
# unrelated Terra 2.0 token relisted under the SAME symbol on 2022-05-31. Keep only the
# old token — including its -100% death week — and drop everything after the halt.
LUNA_TRUNCATE = pd.Timestamp("2022-05-15")

# yfinance -USD tickers used ONLY to backfill weeks BEFORE the Binance listing.
YF_BACKFILL = {
    "BTC": "BTC-USD", "ETH": "ETH-USD", "XRP": "XRP-USD", "ADA": "ADA-USD",
    "DOGE": "DOGE-USD", "TRX": "TRX-USD", "LTC": "LTC-USD", "BCH": "BCH-USD",
    "LINK": "LINK-USD", "XLM": "XLM-USD", "ETC": "ETC-USD", "XTZ": "XTZ-USD",
    "NEO": "NEO-USD", "IOTA": "MIOTA-USD", "DASH": "DASH-USD", "ZEC": "ZEC-USD",
    "QTUM": "QTUM-USD", "EOS": "EOS-USD", "XMR": "XMR-USD", "WAVES": "WAVES-USD",
    "OMG": "OMG-USD",
}

START = "2017-01-01"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def _binance_weekly(symbol: str, retries: int = 3) -> pd.Series:
    """Full weekly close history for one Binance spot pair (weeks indexed by Monday open).

    One request per symbol (full history < 1000 weekly bars). Works for delisted pairs too.
    """
    url = (f"https://api.binance.com/api/v3/klines?symbol={symbol}"
           f"&interval=1w&startTime=0&limit=1000")
    last_err = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                k = json.loads(r.read())
            idx = pd.to_datetime([row[0] for row in k], unit="ms")
            close = pd.Series([float(row[4]) for row in k], index=idx, name=symbol)
            return close[close > 0]
        except Exception as e:                              # noqa: BLE001
            last_err = e
            time.sleep(2.0)
    raise RuntimeError(f"Binance klines failed for {symbol}: {last_err}")


def _yf_weekly(tickers: list[str], start: str = START) -> pd.DataFrame:
    """yfinance daily closes resampled to Binance's Monday-open weeks.

    A Binance week opening Monday M closes Sunday 23:59 UTC — i.e. Yahoo's daily close
    stamped Sunday M+6. We take the last daily close of each W-SUN bucket and re-label
    the bucket by its Monday open.
    """
    import yfinance as yf

    raw = yf.download(tickers, start=start, interval="1d", auto_adjust=True,
                      progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close.index = pd.to_datetime(close.index).tz_localize(None)
    wk = close.resample("W-SUN").last()
    wk.index = wk.index - pd.Timedelta(days=6)              # label by Monday open
    return wk


def fetch_panel(fetch: bool = False, cache_path: str = PANEL_CACHE) -> pd.DataFrame:
    """The spliced weekly close panel; cache-only unless ``fetch=True``.

    Splice rule per coin: Binance weekly closes from the pair's first Binance week
    onward; yfinance weekly closes strictly BEFORE that (backfill only, never forward
    extension past a delisting — once a pair leaves the venue you could not trade it
    there, so the name leaves the panel). The final in-progress week is dropped so the
    tape holds only complete Monday-to-Sunday weeks.
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached panel at {cache_path}. Run fetch_panel(fetch=True) once."
            )
        df = pd.read_parquet(cache_path)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    cols = {}
    for coin in UNIVERSE:
        s = _binance_weekly(coin + "USDT")
        if coin == "LUNA":
            s = s[s.index <= LUNA_TRUNCATE]
        cols[coin] = s
        time.sleep(0.15)

    yf_wk = _yf_weekly(sorted(set(YF_BACKFILL.values())))
    all_weeks = pd.date_range(
        start=START,
        end=max(s.index.max() for s in cols.values()),
        freq="W-MON",
    )
    panel = pd.DataFrame(index=all_weeks, columns=list(cols), dtype=float)
    for coin, s in cols.items():
        common = panel.index.intersection(s.index)
        panel.loc[common, coin] = s.reindex(common).to_numpy()
        yft = YF_BACKFILL.get(coin)
        if yft is not None and yft in yf_wk.columns:
            pre = yf_wk[yft].dropna()
            pre = pre[pre.index < s.index.min()]            # backfill ONLY pre-listing
            common = panel.index.intersection(pre.index)
            panel.loc[common, coin] = pre.reindex(common).to_numpy()

    # Keep only complete weeks: a week opening Monday W is complete once W+7d has passed.
    now = pd.Timestamp.utcnow().tz_localize(None).normalize()
    panel = panel[panel.index <= now - pd.Timedelta(days=7)]
    panel = panel.dropna(how="all")
    panel.index.name = "week_open"

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    panel.to_parquet(cache_path)
    return panel


def have_real(cache_path: str = PANEL_CACHE) -> bool:
    return os.path.exists(cache_path)


def load_real(cache_path: str = PANEL_CACHE) -> pd.DataFrame:
    return fetch_panel(fetch=False, cache_path=cache_path)


def fingerprint(panel: pd.DataFrame) -> str:
    """Short content hash of the whole close panel, for the as-of stamp."""
    arr = np.ascontiguousarray(np.nan_to_num(panel.to_numpy(dtype=float)))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic control — plantable cross-sectional momentum
# --------------------------------------------------------------------------- #
def synthetic_panel(n_coins: int = 40, n_weeks: int = 470, rho: float = 0.0,
                    mkt_vol_ann: float = 0.75, idio_vol_ann: float = 1.10,
                    seed: int = 632) -> pd.DataFrame:
    """Deterministic weekly close panel with a PLANTED 1-week cross-sectional momentum.

    Coin *i*'s weekly return is ``beta_i * mkt_t + eps_{i,t}`` with idiosyncratic AR(1)
    ``eps_{i,t} = rho * eps_{i,t-1} + noise``. With ``rho = 0`` there is NO
    cross-sectional predictability (the null: a winners-minus-losers quintile spread must
    not reach significance however the noise falls). With ``rho > 0`` last week's
    idiosyncratic winners keep winning and the same quintile machinery must light up.
    ~9 years of weeks — far below the 250-year pandas ns-Timestamp ceiling.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2017-01-02", periods=n_weeks, freq="W-MON")
    s_mkt = mkt_vol_ann / np.sqrt(WEEKS_PER_YEAR)
    s_idio = idio_vol_ann / np.sqrt(WEEKS_PER_YEAR)
    mkt = rng.normal(0.0, s_mkt, n_weeks)
    beta = rng.uniform(0.7, 1.4, n_coins)
    noise = rng.normal(0.0, s_idio * np.sqrt(max(1.0 - rho * rho, 1e-9)),
                       (n_weeks, n_coins))
    eps = np.zeros((n_weeks, n_coins))
    for t in range(1, n_weeks):
        eps[t] = rho * eps[t - 1] + noise[t]
    r = beta[None, :] * mkt[:, None] + eps
    close = 100.0 * np.exp(np.cumsum(r, axis=0))
    cols = [f"C{i:02d}" for i in range(n_coins)]
    return pd.DataFrame(close, index=idx, columns=cols)
