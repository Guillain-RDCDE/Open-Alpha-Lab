"""Data layer for Study 443 — Volume Profile POC ("does price return to the point of control?").

Two tapes, one shape — a per-**session** event table with one row per trading day, carrying
the *prior* day's volume-profile **point of control** (POC, the price level with the most
traded volume) and the things we need to ask "did price come back to it?".

* **Real tape.** yfinance **5-minute** intraday bars (~60 calendar days, Yahoo's cap on 5m
  US equities) for a short basket of very liquid names (SPY, QQQ, AAPL, …). For each session
  we build a volume profile by binning the session's traded volume into price levels, take the
  POC = the modal price level, and stamp it onto the *next* session. The cache is one wide-ish
  CSV per ticker under ``_cache/`` so re-runs are fully offline.

* **Synthetic.** A deterministic, fixed-seed generator that produces intraday-like sessions
  whose next-session path is pulled toward the prior POC with a controllable strength
  (``pull`` knob). ``pull = 0`` is the null — the POC is just a price level with no magnetism;
  a touch is whatever a random walk hands you. ``pull > 0`` plants a real reversion the harness
  must recover. It is the positive control, never market evidence.

The short-span (~60d) and capacity/cost caveats are LOUD: intraday levels die to the spread.

Pure numpy + pandas + stdlib for the offline path. ``fetch_panel`` (network) is used only to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

NY_TZ = "America/New_York"

# A short basket of the most liquid US equities/ETFs with clean 5m volume on yfinance.
# Survivors all — but this is a within-name, day-over-day level test, not a cross-sectional
# basket sort, so survivorship does not tilt the POC-reversion hit-rate (named on the Signal
# axis anyway for honesty).
BASKET = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA"]

N_BINS = 50          # price bins per session volume profile
EVENT_COLUMNS = [
    "ticker", "date",
    "poc",            # prior-session point-of-control price level
    "open",           # this session's open
    "high", "low",    # this session's range (did it touch the POC?)
    "close",          # this session's close
    "open_gap",       # (open - poc) / poc : signed distance of the open from the POC
    "atr",            # prior-session high-low range / close, a scale for the control level
]


# --------------------------------------------------------------------------- #
# Volume profile / POC
# --------------------------------------------------------------------------- #
def session_poc(bars: pd.DataFrame, n_bins: int = N_BINS) -> float:
    """Point of control of one session's 5m bars = the price bin with the most volume.

    Each 5m bar contributes its ``volume`` to the bin of its *typical price*
    (high+low+close)/3. The POC is the centre of the highest-volume bin — the classic
    volume-profile definition (the price the market "agreed on" most that day).
    """
    if bars.empty:
        return float("nan")
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    lo, hi = float(tp.min()), float(tp.max())
    if hi <= lo:
        return float(tp.iloc[0])
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(tp.values, edges) - 1, 0, n_bins - 1)
    vol = bars["volume"].values.astype(float)
    acc = np.zeros(n_bins)
    np.add.at(acc, idx, vol)
    k = int(np.argmax(acc))
    return float((edges[k] + edges[k + 1]) / 2.0)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str = DEFAULT_CACHE) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"vp_{safe}_5m.csv")


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
    return df.between_time("09:30", "15:55")


def fetch_panel(ticker: str, period: str = "60d", cache_dir: str = DEFAULT_CACHE,
                retries: int = 3) -> pd.DataFrame:
    """Download 5m bars for one ticker and cache the per-session event table.

    Network-only; used once to build the cache. Retries a couple of times with a small backoff
    on a rate-limit, then caches a CSV so every re-run is offline. Never imported by the
    notebooks' offline cells.
    """
    import time

    import yfinance as yf

    raw = None
    for k in range(retries):
        try:
            raw = yf.download(ticker, period=period, interval="5m",
                              auto_adjust=True, progress=False)
            if raw is not None and not raw.empty:
                break
        except Exception:
            raw = None
        time.sleep(1.5 * (k + 1))
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no 5m bars for {ticker}")

    bars = _normalize(raw)
    ev = build_event_table(bars, ticker)
    os.makedirs(cache_dir, exist_ok=True)
    ev.to_csv(_cache_path(ticker, cache_dir), index=False)
    return ev


def build_event_table(bars: pd.DataFrame, ticker: str, n_bins: int = N_BINS) -> pd.DataFrame:
    """Reduce 5m RTH bars to a per-session event table carrying the PRIOR-day POC.

    For each trading day we compute that day's POC (volume profile of its 5m bars). The event
    row for day ``D`` carries the POC of day ``D-1`` (the level a believer would watch coming
    into ``D``), plus day ``D``'s open / high / low / close — so we can ask whether ``D``'s
    range touched the prior POC. No look-ahead: the POC is known at the prior close.
    """
    if bars.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    dates = pd.Index(bars.index.date, name="date")
    per_day = []
    for d, sl in bars.groupby(dates):
        sl = sl.sort_index()
        if len(sl) < 20:                       # need a reasonably full session
            continue
        poc = session_poc(sl, n_bins)
        per_day.append({
            "date": pd.Timestamp(d), "poc": poc,
            "open": float(sl["open"].iloc[0]),
            "high": float(sl["high"].max()),
            "low": float(sl["low"].min()),
            "close": float(sl["close"].iloc[-1]),
        })
    day = pd.DataFrame(per_day).sort_values("date").reset_index(drop=True)
    if len(day) < 2:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    rows = []
    for i in range(1, len(day)):
        prev, cur = day.iloc[i - 1], day.iloc[i]
        poc_prev = float(prev["poc"])
        if not np.isfinite(poc_prev) or poc_prev <= 0:
            continue
        atr = (float(prev["high"]) - float(prev["low"])) / float(prev["close"])
        rows.append({
            "ticker": ticker, "date": cur["date"],
            "poc": poc_prev,
            "open": float(cur["open"]), "high": float(cur["high"]),
            "low": float(cur["low"]), "close": float(cur["close"]),
            "open_gap": float(cur["open"]) / poc_prev - 1.0,
            "atr": float(atr),
        })
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def have_real(cache_dir: str = DEFAULT_CACHE, basket=BASKET) -> bool:
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in basket)


def load_real(cache_dir: str = DEFAULT_CACHE, basket=BASKET) -> pd.DataFrame:
    """Concatenated per-session event table for the whole basket (cache-first)."""
    frames = []
    for tk in basket:
        p = _cache_path(tk, cache_dir)
        if os.path.exists(p):
            frames.append(pd.read_csv(p, parse_dates=["date"]))
    if not frames:
        raise FileNotFoundError(
            f"No cached 5m event tables under {cache_dir}. "
            f"Call fetch_panel(tk) once per ticker to populate the cache.")
    ev = pd.concat(frames, ignore_index=True)
    return ev.sort_values(["ticker", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_names: int = 6, n_sessions: int = 60, pull: float = 0.0,
                    seed: int = 443, sigma_sess: float = 0.006,
                    gap_sd: float = 0.010) -> tuple[pd.DataFrame, dict]:
    """Deterministic per-session event table with a PLANTED POC-reversion knob.

    Each name walks a daily price. The prior-session POC is set to the prior close (a clean
    proxy). The next session opens with a small gap away from the POC; its intraday path then
    drifts back toward the POC with strength ``pull`` (an Ornstein-Uhlenbeck-like reversion of
    the close to the POC) — so a larger ``pull`` makes the high/low straddle the POC more often
    and pulls the close toward it. With ``pull = 0`` the path ignores the POC entirely: any
    "touch" is whatever the random range hands you, and the reversion hit-rate must NOT beat a
    distance-matched random control.

    Returns ``(events, truth)`` in the same shape as :func:`load_real`.
    """
    rng = np.random.default_rng(seed)
    names = [f"N{i:02d}" for i in range(n_names)]
    idx = pd.bdate_range("2026-01-02", periods=n_sessions + 1)
    rows = []
    for nm in names:
        price = 100.0 * np.exp(np.cumsum(rng.normal(0.0, sigma_sess, n_sessions + 1)))
        for i in range(1, n_sessions + 1):
            poc = float(price[i - 1])                 # prior close as POC proxy
            gap = rng.normal(0.0, gap_sd)
            op = poc * (1.0 + gap)                     # open gaps off the POC
            # intraday excursion half-width — tight enough that touching is NOT automatic
            hw = (abs(rng.normal(0.0, sigma_sess)) + 0.001) * op
            # close drifts back toward the POC by fraction `pull` of the gap
            cl = op + pull * (poc - op) + rng.normal(0.0, sigma_sess * 0.4) * op
            # the high/low reach `hw` beyond the open in the direction of travel; when `pull`
            # is on, the close sits near the POC, so the range straddles the POC. With pull=0
            # the range is symmetric around the open and the POC is reached only by chance.
            hi = max(op, cl) + hw
            lo = min(op, cl) - hw
            atr = abs(rng.normal(0.0, sigma_sess)) + 0.004
            rows.append({
                "ticker": nm, "date": idx[i], "poc": poc,
                "open": op, "high": hi, "low": lo, "close": cl,
                "open_gap": op / poc - 1.0, "atr": atr,
            })
    ev = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    truth = {"pull": pull, "n_names": n_names, "n_sessions": n_sessions, "seed": seed}
    return ev, truth


def fingerprint(ev: pd.DataFrame) -> str:
    """Short content fingerprint of an event table (the open_gap column), for the as-of stamp."""
    arr = np.ascontiguousarray(ev["open_gap"].to_numpy(dtype=float))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
