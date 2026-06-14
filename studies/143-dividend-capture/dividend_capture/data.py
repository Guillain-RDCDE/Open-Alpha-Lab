"""Data layer for Study 143 (Dividend-Capture).

Two tapes — both built around the *ex-dividend date*:

- ``synthetic_events`` — a deterministic, offline generator. An ``efficiency``
  knob (0 = fully efficient: price drops by exactly the dividend on the ex-date;
  1 = fully inefficient: no price adjustment at all) lets a test confirm the
  strategy only earns anything when efficiency is below 1.0. This is the null in
  a bottle: at ``efficiency=0.0`` (markets impound the dividend 100%), the
  capture trade earns nothing gross and loses to costs.

  Convention matches the real-tape convention: ``efficiency`` is the fraction of
  the dividend that the market does NOT take back. So ``efficiency=0.0`` means
  the price drops by the full dividend (fully efficient / standard EMH); and
  ``efficiency=1.0`` means the price does not drop at all.

- ``fetch_events`` — the real tape via yfinance. Pulls **unadjusted** daily
  OHLCV for a basket of tickers plus their ``Dividends`` column (expressed on the
  ex-dividend date). Using unadjusted prices is critical: with auto-adjusted
  prices the backward dividend-adjustment is distributed across all historical
  bars, so the ex-date "drop" in the adjusted series does NOT correspond to the
  actual cash dividend received by a holder on that specific day.

  We cache the unadjusted daily OHLCV + dividends as parquet under ``_cache/``.
  Cache-only by default so tests and the reproducible core never hit the network.

No look-ahead: the signal is formed at the *close before the ex-date* (t-1);
positions entered at the open of the ex-date (t) and exited at the close of the
day after (t+1). For simplicity we use the daily close of t-1 as the entry price
and the daily close of t+1 as the exit price (conservative, using daily data).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Default basket: SPY + high-yield large-caps with long dividend histories.
DEFAULT_TICKERS = [
    "SPY",   # S&P 500 ETF — quarterly divs
    "VYM",   # Vanguard High Dividend Yield ETF
    "T",     # AT&T — high yield
    "MO",    # Altria — very high yield
    "VZ",    # Verizon
    "XOM",   # ExxonMobil
    "JNJ",   # Johnson & Johnson
    "KO",    # Coca-Cola
]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_events(
    n_events: int = 200,
    div_yield_annual: float = 0.04,
    price_level: float = 100.0,
    efficiency: float = 0.0,
    daily_vol: float = 0.012,
    cost_bps: float = 5.0,
    seed: int = 143,
) -> tuple[pd.DataFrame, dict]:
    """Generate a deterministic sequence of synthetic ex-dividend events.

    Each event captures a single dividend-capture cycle:
    - A dividend ``D`` is drawn from a yield-scaled Poisson.
    - On the ex-date the price drops by ``(1 - efficiency) * D`` (plus noise).
      Under full efficiency (``efficiency=0.0``) the price drops by exactly the
      dividend, so capture earns zero gross. Under partial efficiency
      (``efficiency=0.5``) the drop is only 50% of D, leaving a gross gain of
      ~0.5 * D / price. Under no adjustment (``efficiency=1.0``) the full
      dividend is profit.
    - ``cost_bps`` is the round-trip transaction cost charged on the trade.

    Returns a DataFrame with columns:
        ex_date, ticker, div, price_before, price_ex, price_after,
        drop, drop_ratio (= drop / div), ret_gross, ret_net
    and a ``truth`` dict recording the planted parameters.
    """
    rng = np.random.default_rng(seed)

    # Simulate quarterly divs: space events ~63 trading days apart.
    dates = pd.bdate_range("2000-01-03", periods=n_events * 63, freq="B")[::63][:n_events]

    prices_before = price_level * np.exp(
        rng.normal(0.0, daily_vol * np.sqrt(63), n_events).cumsum()
    )

    # Quarterly dividend ~ annual_yield / 4, with mild cross-sectional noise.
    quarterly_div = div_yield_annual / 4.0
    divs = prices_before * quarterly_div * (1.0 + rng.normal(0.0, 0.10, n_events))
    divs = np.maximum(divs, 0.01)  # dividends are non-negative

    # Ex-date price: drops by (1 - efficiency)*D plus daily noise.
    # efficiency=0.0 → fully efficient, drop = D; efficiency=1.0 → no drop.
    noise = rng.normal(0.0, daily_vol, n_events) * prices_before
    prices_ex = prices_before - (1.0 - efficiency) * divs + noise

    # Price the day after: just daily noise around price_ex.
    noise_after = rng.normal(0.0, daily_vol, n_events) * prices_ex
    prices_after = prices_ex + noise_after

    drop = prices_before - prices_ex          # raw price drop on ex-date
    drop_ratio = drop / divs                  # >1 means market over-adjusts

    # Capture trade: buy at close t-1, sell at close t+1, collect div at t.
    # Gross return = (price_after - price_before + div) / price_before
    ret_gross = (prices_after - prices_before + divs) / prices_before
    ret_net = ret_gross - cost_bps * 1e-4

    tickers = [f"SYN{i % 8:02d}" for i in range(n_events)]
    events = pd.DataFrame(
        {
            "ex_date": dates,
            "ticker": tickers,
            "div": divs,
            "price_before": prices_before,
            "price_ex": prices_ex,
            "price_after": prices_after,
            "drop": drop,
            "drop_ratio": drop_ratio,
            "ret_gross": ret_gross,
            "ret_net": ret_net,
        }
    )
    truth = {
        "n_events": n_events,
        "efficiency": efficiency,
        "daily_vol": daily_vol,
        "cost_bps": cost_bps,
        "seed": seed,
        "div_yield_annual": div_yield_annual,
    }
    return events, truth


# ---------------------------------------------------------------------------
# Real tape — yfinance unadjusted daily OHLCV + dividends, cache-first
# ---------------------------------------------------------------------------
def _price_cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "").replace(".", "")
    return os.path.join(cache_dir, f"div_cap2_{safe}_daily.parquet")


def fetch_events(
    tickers: list[str] | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    start: str = "2000-01-01",
    end: str = "2026-06-14",
) -> pd.DataFrame:
    """Real per-ticker ex-dividend events with surrounding **unadjusted** price context.

    We use ``auto_adjust=False`` because the ex-date analysis requires *unadjusted*
    prices: backward-adjusted prices distribute the dividend adjustment across all
    historical bars, so the adjusted close on the ex-date does NOT show the actual
    cash-dividend-sized drop. Only unadjusted prices reveal the raw ex-date mechanics.

    For each ticker we align the ``Dividends`` column (ex-dates) with daily OHLCV,
    then for every ex-event we extract:

        price_before — unadjusted close on the trading day *before* the ex-date
        price_ex     — unadjusted close *on* the ex-date
        price_after  — unadjusted close on the day *after* the ex-date

    The drop is ``price_before - price_ex``; ``drop_ratio = drop / div`` measures
    how much of the dividend the market takes back in the price. Efficient markets
    predict drop_ratio ≈ 1.0 (ignoring taxes); the capture trade profits only when
    it is systematically < 1.0. In practice the ratio tends to be ≈ 1.0–1.15 (market
    noise plus slight over-adjustment from institutional selling pressure at ex-dates).

    Cache-only (``fetch=False``) raises FileNotFoundError if no per-ticker parquet
    exists. With ``fetch=True`` the network is queried and results cached.

    Columns returned:
        ex_date (Date), ticker, div, price_before, price_ex, price_after,
        drop, drop_ratio, ret_gross, ret_net (cost = 5 bps round-trip)
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    frames = []
    for ticker in tickers:
        path = _price_cache_path(ticker, cache_dir)
        if not fetch:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"No cached unadjusted daily data for {ticker} at {path}. "
                    f"Call fetch_events(fetch=True) once to populate the cache."
                )
            raw = pd.read_parquet(path)
        else:
            import yfinance as yf  # lazy import — network only on explicit request

            tk = yf.Ticker(ticker)
            raw_px = tk.history(start=start, end=end, auto_adjust=False, actions=True)
            if raw_px.empty:
                continue
            if isinstance(raw_px.index, pd.DatetimeIndex) and raw_px.index.tz is not None:
                raw_px.index = raw_px.index.tz_localize(None)
            raw_px.index.name = "Date"
            os.makedirs(cache_dir, exist_ok=True)
            raw_px.to_parquet(path)
            raw = raw_px

        if isinstance(raw.index, pd.DatetimeIndex) and raw.index.tz is not None:
            raw.index = raw.index.tz_localize(None)
        raw.index.name = "Date"

        # Identify the close and dividend columns (yfinance uses title case).
        close_col = "Close" if "Close" in raw.columns else "close"
        div_col = "Dividends" if "Dividends" in raw.columns else "dividends"
        if close_col not in raw.columns or div_col not in raw.columns:
            continue

        close = raw[close_col]
        divs_series = raw[div_col]
        ex_dates = divs_series[divs_series > 0].index

        for ex_date in ex_dates:
            loc = close.index.get_loc(ex_date)
            if loc < 1 or loc + 1 >= len(close):
                continue
            price_before = float(close.iloc[loc - 1])
            price_ex = float(close.iloc[loc])
            price_after = float(close.iloc[loc + 1])
            div = float(divs_series.iloc[loc])
            if price_before <= 0 or div <= 0:
                continue
            drop = price_before - price_ex
            drop_ratio = drop / div
            ret_gross = (price_after - price_before + div) / price_before
            ret_net = ret_gross - 5e-4  # 5 bps round-trip cost

            frames.append(
                {
                    "ex_date": ex_date,
                    "ticker": ticker,
                    "div": div,
                    "price_before": price_before,
                    "price_ex": price_ex,
                    "price_after": price_after,
                    "drop": drop,
                    "drop_ratio": drop_ratio,
                    "ret_gross": ret_gross,
                    "ret_net": ret_net,
                }
            )

    if not frames:
        return pd.DataFrame(
            columns=[
                "ex_date", "ticker", "div", "price_before", "price_ex",
                "price_after", "drop", "drop_ratio", "ret_gross", "ret_net",
            ]
        )
    df = pd.DataFrame(frames).sort_values("ex_date").reset_index(drop=True)
    df["ex_date"] = pd.to_datetime(df["ex_date"])
    return df


def fingerprint(events: pd.DataFrame) -> str:
    """A short content fingerprint of the events table (drop_ratio column)."""
    arr = events["drop_ratio"].to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
