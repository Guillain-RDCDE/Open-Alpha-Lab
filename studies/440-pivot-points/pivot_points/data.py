"""Data layer for Study 440 — Floor-Trader Pivot Points.

Two tapes, one shape: an **intraday OHLC bar frame** (tz-aware, New-York time, one row per
intraday bar) carrying a ``date`` helper column so the strategy can split it into sessions and
compute each session's pivots from the *prior* session's high/low/close.

* **Synthetic.** ``synthetic_panel(...)`` is a deterministic, offline generator with a planted
  **bounce knob** (``edge``). Each session is a random-walk of intraday bars *plus*, when
  ``edge > 0``, a genuine **bounce-away impulse** fired in the level's respect direction
  whenever price first touches that session's pivot set (P, R1, S1, …): price is pushed *away*
  from the level over the next few bars — exactly what the touch/bounce detector looks for — so
  a "fade the touch" rule has a real edge to harvest. (An elastic *pull-to-level* was tried first
  but plants the wrong thing for this detector: it glues price to the line instead of bouncing it
  off, so the bounce-away impulse is the faithful positive control.) With ``edge = 0`` the pivots
  are arbitrary lines drawn on a pure random walk — price has no reason to respect them, and a
  touch-test must NOT manufacture significance. Returns ``(bars, truth)``.

* **Real tape.** ``load_real(cache_dir=DEFAULT_CACHE)`` reads cached **yfinance 5-minute** bars
  for a small basket of very liquid names (SPY, QQQ, AAPL, …), cache-first. Yahoo caps 5-minute
  history at ~60 calendar days, so the real tape is **short** — stated loudly on the Signal axis
  and in every doc. ``fetch_real(..., fetch=True)`` populates the cache once (network), then all
  re-runs are offline.

No look-ahead is baked in here: pivots for session *D* use only sessions strictly before *D*;
the touch/decision logic lives in :mod:`pivot_points.strategy`.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

NY_TZ = "America/New_York"

# A small basket of extremely liquid US names — the only place pivots could plausibly survive
# spread. ETFs first (tightest spreads), then mega-caps.
BASKET = ["SPY", "QQQ", "AAPL", "MSFT", "IWM"]

OHLC = ["open", "high", "low", "close"]


# --------------------------------------------------------------------------- #
# Synthetic tape — deterministic offline core
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_sessions: int = 55,
    bars_per_session: int = 78,            # ~6.5h of 5-minute bars
    edge: float = 0.0,
    annual_vol: float = 0.20,
    seed: int = 440,
    start: str = "2024-01-02",
) -> tuple[pd.DataFrame, dict]:
    """Deterministic intraday OHLC tape with a PLANTED pivot-**bounce** knob.

    Each session is a random walk of ``bars_per_session`` 5-minute bars. The session's pivot
    set is computed from the *previous* session's H/L/C (the real recipe). When ``edge > 0`` a
    genuine **bounce impulse** fires whenever price first comes within a tick of any pivot: over
    the following few bars price gets pushed *away* from that level (the exact thing the touch /
    bounce detector looks for — a real "respect the level" reaction the fade trade can harvest).
    With ``edge = 0`` the levels are arbitrary lines on a martingale: a touch carries no
    information and the bounce-rate must match a randomly placed control line.

    Returns ``(bars, truth)``. ``bars`` is a tz-aware (NY) intraday frame with columns
    ``open, high, low, close, volume`` and a ``date`` (session) column.
    """
    rng = np.random.default_rng(seed)
    bar_vol = annual_vol / np.sqrt(252.0 * bars_per_session)
    bounce_bars = 4            # how long a bounce impulse lasts

    sessions = pd.bdate_range(start=start, periods=n_sessions)
    rows = []
    times = []
    level = 100.0
    prev_hlc = None  # (high, low, close) of the previous session

    # supports push price UP on a touch, resistances push DOWN (the detector's bounce sign)
    SIGN = {"S3": +1.0, "S2": +1.0, "S1": +1.0, "P": +1.0,
            "R1": -1.0, "R2": -1.0, "R3": -1.0}

    for d in sessions:
        # this session's pivot set (None on the first session -> no bounce)
        if prev_hlc is not None:
            piv = _pivot_levels(*prev_hlc)
            piv_names = list(piv.keys())
            piv_arr = np.array([piv[n] for n in piv_names])
        else:
            piv_arr = None

        s_open = level
        prices = np.empty(bars_per_session)
        p = s_open
        impulse = 0.0          # remaining bounce drift per bar
        impulse_left = 0
        touched = set()        # pivots already reacted to this session
        tol = bar_vol * 0.8    # "within a tick" band for triggering a bounce
        for k in range(bars_per_session):
            eps = rng.normal(0.0, bar_vol)
            if piv_arr is not None and edge > 0.0 and impulse_left == 0:
                near_i = int(np.argmin(np.abs(piv_arr - p)))
                nearest = piv_arr[near_i]
                if near_i not in touched and abs(p - nearest) / p <= tol:
                    # fire a bounce in the level's RESPECT direction (support up, resist down)
                    direction = SIGN[piv_names[near_i]]
                    impulse = edge * bar_vol * direction
                    impulse_left = bounce_bars
                    touched.add(near_i)
            drift = 0.0
            if impulse_left > 0:
                drift = impulse
                impulse_left -= 1
            p = p * (1.0 + eps + drift)
            prices[k] = p

        # build OHLC bars: open=prev close, intrabar wick a fraction of bar vol
        opens = np.empty(bars_per_session)
        opens[0] = s_open
        opens[1:] = prices[:-1]
        wick = np.abs(rng.normal(0.0, bar_vol * 0.5, bars_per_session)) * prices
        highs = np.maximum(opens, prices) + wick
        lows = np.minimum(opens, prices) - wick
        vols = rng.integers(1_000, 50_000, bars_per_session).astype(float)

        # 5-minute timestamps 09:30 -> 16:00 NY
        ts = pd.date_range(
            pd.Timestamp(d.date()).tz_localize(NY_TZ) + pd.Timedelta(hours=9, minutes=30),
            periods=bars_per_session, freq="5min",
        )
        for k in range(bars_per_session):
            rows.append((opens[k], highs[k], lows[k], prices[k], vols[k]))
            times.append(ts[k])

        prev_hlc = (float(highs.max()), float(lows.min()), float(prices[-1]))
        level = float(prices[-1])

    bars = pd.DataFrame(rows, columns=OHLC + ["volume"],
                        index=pd.DatetimeIndex(times, name="ts"))
    bars["date"] = bars.index.tz_convert(NY_TZ).normalize()
    truth = {"edge": edge, "n_sessions": n_sessions, "bars_per_session": bars_per_session,
             "annual_vol": annual_vol, "seed": seed}
    return bars, truth


# --------------------------------------------------------------------------- #
# Pivot arithmetic (shared by synthetic generator + strategy)
# --------------------------------------------------------------------------- #
def _pivot_levels(prev_high: float, prev_low: float, prev_close: float) -> dict:
    """Classic floor-trader pivots from the prior session's H/L/C.

    P  = (H + L + C) / 3
    R1 = 2P - L      S1 = 2P - H
    R2 = P + (H - L) S2 = P - (H - L)
    R3 = H + 2(P - L) S3 = L - 2(H - P)
    """
    p = (prev_high + prev_low + prev_close) / 3.0
    rng_ = prev_high - prev_low
    return {
        "S3": prev_low - 2.0 * (prev_high - p),
        "S2": p - rng_,
        "S1": 2.0 * p - prev_high,
        "P": p,
        "R1": 2.0 * p - prev_low,
        "R2": p + rng_,
        "R3": prev_high + 2.0 * (p - prev_low),
    }


# --------------------------------------------------------------------------- #
# Real tape — yfinance 5-minute bars, cache-first
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_5m.parquet")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns, lowercase, tz-convert to NY, RTH only."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].copy()
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    df.index = idx.tz_convert(NY_TZ)
    df.index.name = "ts"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.between_time("09:30", "15:55")     # regular trading hours
    df["date"] = df.index.normalize()
    return df


def fetch_real(ticker: str, period: str = "60d", interval: str = "5m",
               fetch: bool = False, cache_dir: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Real intraday OHLC for ``ticker``; cache-first (``fetch=False`` raises on a miss).

    Yahoo caps ``5m`` history at ~60 calendar days and ``1m`` at ~7 — a deliberately SHORT
    real tape (the capacity/short-span caveat lives on the Signal axis). Pass ``fetch=True``
    once to populate the parquet cache; afterwards all calls are offline.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached {interval} tape for {ticker} at {path}. "
                f"Call fetch_real({ticker!r}, fetch=True) once to populate the cache."
            )
        return pd.read_parquet(path)

    import yfinance as yf  # lazy: only on a real fetch

    last_err = None
    for attempt in range(3):
        try:
            raw = yf.download(ticker, period=period, interval=interval,
                              auto_adjust=False, progress=False)
            if raw is not None and not raw.empty:
                bars = _normalize(raw)
                os.makedirs(cache_dir, exist_ok=True)
                bars.to_parquet(path)
                return bars
        except Exception as e:  # pragma: no cover - network
            last_err = e
        import time
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"yfinance returned no {interval} bars for {ticker}: {last_err}")


def have_real(cache_dir: str = DEFAULT_CACHE, basket: list[str] | None = None) -> bool:
    basket = basket or BASKET
    return any(os.path.exists(_cache_path(t, cache_dir)) for t in basket)


def load_real(cache_dir: str = DEFAULT_CACHE,
              basket: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Cached 5-minute bars for every basket name present in the cache → {ticker: bars}."""
    basket = basket or BASKET
    out = {}
    for t in basket:
        path = _cache_path(t, cache_dir)
        if os.path.exists(path):
            out[t] = pd.read_parquet(path)
    return out


def fingerprint(bars: pd.DataFrame) -> str:
    """Short content fingerprint of an intraday tape (close column)."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
