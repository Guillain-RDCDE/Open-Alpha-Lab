"""Data layer for Study 653 — Dividend-Cut-Drift.

Two ingredients:

* **Real tape.** Per-ticker adjusted (total-return) daily closes and the raw dividend-per-share
  stream from yfinance (no key), for a hardcoded basket of ~100 large/mid-cap US "mature
  dividend payers" (a mix of names that never cut and names that visibly did — 2008-09 banks,
  2015-16 energy, 2020 pandemic travel/retail, 2022-24 industrials/staples). Cached as parquet
  under the study's own ``_cache/`` (cache-first: fetch once, read cache after). SPY (also
  auto-adjusted / total-return) is the benchmark for the event-study abnormal return.
  Adjusted closes are used throughout specifically so the *mechanical* ex-dividend price drop
  never contaminates the abnormal-return math (an auto-adjusted series has already re-invested
  the dividend into the price series, so a payment being cut or omitted shows up only in the
  ``Dividends`` stream, never as a phantom jump in price).

* **Synthetic world.** A deterministic, seeded panel of random-walk names against a random-walk
  benchmark, each optionally carrying a scheduled "cut" event with a TUNABLE planted post-event
  daily abnormal drift. ``drift = 0`` is the null world — the event-study detector must NOT fire;
  a nonzero drift is the positive control the detector must recover.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build the
cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

AS_OF = "2026-06-30"        # last complete calendar month at publication (2026-07-10)
START = "1996-01-01"        # earliest date we bother slicing to (most tickers start later)

# --------------------------------------------------------------------------- #
# Universe — ~100 large/mid-cap US "mature dividend payers", named.
#
# Deliberately a MIX, not a cherry-picked list of cutters: about a third are long-run stable
# payers that (as far as the public record goes) never materially cut, so the event detector has
# a real chance of finding *zero* events on them; the rest are payers that visibly cut or
# suspended at least once across 2008-09 (banks), 2015-16 (energy), 2020 (pandemic
# travel/retail/industrials) or 2022-24 (industrials, staples, apparel, discount retail).
# Survivorship note travels with this list: every name below is a CURRENT survivor (still listed
# as of 2026); names that cut a dividend and later went bankrupt or were delisted/acquired
# outright (Lehman, pre-2009 GM, WaMu, Kodak, RadioShack, Bed Bath & Beyond...) are, by
# construction, absent — the basket is biased toward cuts that were survivable. Named on the
# Signal axis, not just tradability.
# --------------------------------------------------------------------------- #
UNIVERSE = [
    # --- stable / rare-cutter control group (staples, healthcare, quality industrials) ---
    "JNJ", "PG", "KO", "PEP", "MCD", "WMT", "HD", "CL", "KMB", "GIS",
    "SYY", "ADM", "CLX", "CHD", "MDT", "ABT", "XOM", "CVX", "VZ", "MO",
    "PM", "GPC", "DOV", "EMR", "ITW", "SHW", "LOW", "TGT", "BEN", "NKE",
    "O", "IBM", "PFE", "TXN", "MCK",
    # --- financials — 2008-09 crisis cutters ---
    "C", "BAC", "WFC", "USB", "KEY", "RF", "CFG", "PNC", "TFC", "GS",
    "MS", "AXP",
    # --- energy / materials — 2015-16 and 2020 commodity-bust cutters ---
    "KMI", "WMB", "OKE", "NOV", "MRO", "APA", "DVN", "OXY", "COP", "HAL",
    "SLB", "BKR", "FCX", "NUE", "CLF", "AA", "NEM", "GOLD", "RIG",
    # --- pandemic-era suspenders (2020) — travel, leisure, apparel retail ---
    "DIS", "CCL", "RCL", "NCLH", "MAR", "HLT", "BA", "DAL", "UAL", "AAL",
    "LUV", "MGM", "WYNN", "LVS", "M", "KSS", "GPS", "HOG", "BBWI", "COTY",
    # --- 2016-24 industrials / staples / telecom / apparel cutters ---
    "GE", "F", "KHC", "T", "LEG", "VFC", "IP", "WHR", "MMM", "LUMN",
    # --- REITs — pandemic-era reductions ---
    "FRT", "VNO", "KIM", "SPG", "IRM",
]

SPY_TICKER = "SPY"


def _safe(t: str) -> str:
    return t.replace("=", "").replace("^", "").replace("/", "").replace(".", "_")


def _px_path(ticker: str, cache_dir: str = CACHE_DIR) -> str:
    return os.path.join(cache_dir, f"dcd_{_safe(ticker)}_px.parquet")


def _div_path(ticker: str, cache_dir: str = CACHE_DIR) -> str:
    return os.path.join(cache_dir, f"dcd_{_safe(ticker)}_div.parquet")


def _split_adjust(div: pd.Series, splits: pd.Series) -> pd.Series:
    """Express every raw historical dividend-per-share in TODAY's share-count terms.

    yfinance's raw ``Ticker.dividends`` stream is NOT split-adjusted: a 2-for-1 split makes the
    per-share payment mechanically halve even with zero change in payout policy. Without this
    correction a split reads as a ~50%+ "cut" (it hit PEP, MO, C, SLB, MS and others in this
    universe's history). Adjustment: divide each historical payment by the product of every
    split ratio that occurred AFTER it — the same convention ``auto_adjust`` uses for prices.
    """
    if len(splits) == 0 or len(div) == 0:
        return div
    s = splits.sort_index()
    s = s[s > 0]
    out = div.copy().astype(float)
    split_dates = s.index.to_numpy()
    split_ratios = s.to_numpy()
    # suffix cumulative product: factor[i] = product of split_ratios[i:]
    suffix = np.cumprod(split_ratios[::-1])[::-1]
    for d in out.index:
        pos = int(np.searchsorted(split_dates, np.datetime64(d), side="right"))
        factor = float(suffix[pos]) if pos < len(suffix) else 1.0
        out.loc[d] = out.loc[d] / factor
    return out


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(tickers: list[str] | None = None, cache_dir: str = CACHE_DIR
          ) -> tuple[list[str], list[str]]:
    """Download adjusted closes + raw dividend streams for ``tickers`` (default UNIVERSE) plus
    SPY; cache as parquet. Network; run once. Per-ticker failures (renamed/delisted symbols) are
    skipped, not fatal — returns (ok, skipped)."""
    import yfinance as yf

    tickers = tickers or UNIVERSE
    os.makedirs(cache_dir, exist_ok=True)
    ok, skipped = [], []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            raw = tk.history(period="max", auto_adjust=True)
            if raw is None or raw.empty:
                skipped.append(t)
                continue
            px = raw["Close"].rename("close")
            px.index = pd.to_datetime(px.index).tz_localize(None)
            px = px[~px.index.duplicated()].sort_index()

            div = tk.dividends
            if div is None or len(div) == 0:
                div = pd.Series(dtype=float, name="div", index=pd.DatetimeIndex([]))
            else:
                div = div.rename("div")
                div.index = pd.to_datetime(div.index).tz_localize(None)
                div = div[~div.index.duplicated()].sort_index()
                splits = tk.splits
                if splits is not None and len(splits) > 0:
                    splits.index = pd.to_datetime(splits.index).tz_localize(None)
                    div = _split_adjust(div, splits)

            px.to_frame().to_parquet(_px_path(t, cache_dir))
            div.to_frame().to_parquet(_div_path(t, cache_dir))
            ok.append(t)
        except Exception as e:  # noqa: BLE001 - a bad symbol must not kill the whole fetch
            print(f"  [skip] {t}: {e}")
            skipped.append(t)

    tk = yf.Ticker(SPY_TICKER)
    raw = tk.history(period="max", auto_adjust=True)
    spy = raw["Close"].rename("close")
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy = spy[~spy.index.duplicated()].sort_index()
    spy.to_frame().to_parquet(_px_path(SPY_TICKER, cache_dir))

    return ok, skipped


def have_real(tickers: list[str] | None = None, cache_dir: str = CACHE_DIR,
              min_names: int = 40) -> bool:
    tickers = tickers or UNIVERSE
    n = sum(
        os.path.exists(_px_path(t, cache_dir)) and os.path.exists(_div_path(t, cache_dir))
        for t in tickers
    )
    return n >= min_names and os.path.exists(_px_path(SPY_TICKER, cache_dir))


def load_real(tickers: list[str] | None = None, cache_dir: str = CACHE_DIR, asof: str = AS_OF,
              start: str = START) -> tuple[dict[str, pd.Series], dict[str, pd.Series], pd.Series]:
    """Cached (px_map, div_map, spy) sliced to [start, asof]. Tickers missing a cache are
    silently dropped (renamed/delisted symbols that failed ``fetch``)."""
    tickers = tickers or UNIVERSE
    lo, hi = pd.Timestamp(start), pd.Timestamp(asof)
    px_map: dict[str, pd.Series] = {}
    div_map: dict[str, pd.Series] = {}
    for t in tickers:
        pp, dp = _px_path(t, cache_dir), _div_path(t, cache_dir)
        if not (os.path.exists(pp) and os.path.exists(dp)):
            continue
        px = pd.read_parquet(pp)["close"]
        px = px[(px.index >= lo) & (px.index <= hi)]
        if len(px) < 250:
            continue
        div = pd.read_parquet(dp)["div"]
        if len(div) == 0:
            div = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        else:
            div.index = pd.to_datetime(div.index)
            div = div[(div.index >= lo) & (div.index <= hi)]
        px_map[t] = px
        div_map[t] = div
    spy = pd.read_parquet(_px_path(SPY_TICKER, cache_dir))["close"]
    spy = spy[(spy.index >= lo) & (spy.index <= hi)]
    return px_map, div_map, spy


# --------------------------------------------------------------------------- #
# Synthetic world — planted post-cut drift (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(n_names: int = 150, n_days: int = 2500, drift: float = 0.0, seed: int = 653,
                     base_vol: float = 0.020, mkt_vol: float = 0.010, event_prob: float = 0.6,
                     hold: int = 120,
                     ) -> tuple[dict[str, pd.Series], pd.Series, dict[str, pd.Timestamp]]:
    """Deterministic single-factor market-model panel with a scheduled planted "cut" event.

    Each name follows ``r_i = r_mkt + idio_i``; ``event_prob`` of the names carry a single
    scheduled cut event at a random (but fixed-by-seed) date with room for the full
    [-20, +``hold``] window. When ``drift != 0`` the event adds a constant EXTRA daily abnormal
    return for ``hold`` sessions starting the day after the event (an execution-lag-consistent
    plant — the same convention as the real backtests). ``drift = 0`` is the null world: the
    event-study detector must NOT fire. Business-day index, ~2500 rows (~10y) — far below the
    ns-timestamp span trap.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n_days)
    mkt_ret = rng.normal(0.0003, mkt_vol, n_days)
    bench = pd.Series(100.0 * np.exp(np.cumsum(mkt_ret)), index=idx, name="close")

    px_map: dict[str, pd.Series] = {}
    events: dict[str, pd.Timestamp] = {}
    lo_pos, hi_pos = 300, n_days - hold - 150
    for k in range(n_names):
        name = f"S{k:03d}"
        idio = rng.normal(0.0, base_vol, n_days)
        log_ret = mkt_ret + idio
        if rng.random() < event_prob:
            ev_pos = int(rng.integers(lo_pos, hi_pos))
            if drift != 0.0:
                log_ret[ev_pos + 1: ev_pos + 1 + hold] += drift
            events[name] = idx[ev_pos]
        price = 50.0 * np.exp(np.cumsum(log_ret))
        px_map[name] = pd.Series(price, index=idx, name="close")
    return px_map, bench, events
