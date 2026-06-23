"""Data layer for Study 376 — MOC-Imbalance (closing-auction pressure proxy + overnight gap).

Two sources, both offline-friendly:

* **Real tape.** Daily OHLC for a liquid, MOC-heavy ETF basket (SPY/QQQ/IWM, yfinance, no
  key), cached under ``_cache/ohlc.csv`` (a tidy long CSV: date, ticker, open/high/low/close).
  From the daily bars we build the **end-of-day-pressure proxy** — the signed intraday
  *displacement* of the close inside the day's range — plus the **overnight gap**
  (next open / today close − 1). True closing-auction order-imbalance messages are a paid
  data product (NYSE/Nasdaq MOC imbalance feed), so the displacement is an explicit,
  transparent **proxy** for "how hard price was pushed into the close."

* **Synthetic.** A deterministic, fixed-seed generator producing a daily price path in which
  a *controllable fraction* of each day's into-the-close push is mechanically **reversed**
  overnight (the planted MOC-imbalance edge). It is the positive control: with the reversal
  knob at zero the inference must NOT find significance; with a large knob it must light up.

Pure numpy + pandas + stdlib for the offline path. ``fetch_ohlc`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "ohlc.csv")

# Liquid, MOC-heavy index ETFs: their official closes are the ones passive funds and
# index-rebalance flows trade, so they carry the largest real closing-auction imbalances.
TICKERS = ["SPY", "QQQ", "IWM"]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_ohlc(start: str = "2005-01-01", end: str | None = None,
               path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download daily OHLC for the ETF basket via yfinance and cache a tidy long CSV.

    Network-only; used once to build ``_cache/ohlc.csv``. Never imported by the offline
    notebook cells. We keep **raw** (not auto-adjusted) OHLC because the overnight gap and
    the intraday displacement are price-level objects; splits are rare in these ETFs and
    handled by dropping single-day >25% jumps in :func:`load_real`.
    """
    import yfinance as yf

    frames = []
    for tk in TICKERS:
        raw = yf.download(tk, start=start, end=end, auto_adjust=False,
                          progress=False)
        if raw is None or len(raw) == 0:
            continue
        df = raw[["Open", "High", "Low", "Close"]].copy()
        df.columns = ["open", "high", "low", "close"]
        df = df.dropna()
        df["ticker"] = tk
        df = df.reset_index().rename(columns={"Date": "date", "index": "date"})
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out[["date", "ticker", "open", "high", "low", "close"]]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    return out


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached tidy long OHLC frame (columns: date, ticker, open/high/low/close)."""
    df = pd.read_csv(path, parse_dates=["date"]).sort_values(["ticker", "date"])
    return df.reset_index(drop=True)


def is_rebalance_day(dates: pd.DatetimeIndex) -> np.ndarray:
    """Index-rebalance proxy: the **third Friday of a quarter-end month** (Mar/Jun/Sep/Dec).

    The big quarterly S&P/Russell reconstitutions and triple-witching expiries settle on the
    third-Friday close of quarter-end months — the days with the largest real MOC imbalances.
    """
    dates = pd.DatetimeIndex(dates)
    is_fri = np.asarray(dates.weekday == 4)
    is_qend = np.asarray(dates.month.isin([3, 6, 9, 12]))
    # third Friday => day-of-month between 15 and 21 inclusive
    third = np.asarray((dates.day >= 15) & (dates.day <= 21))
    return is_fri & is_qend & third


def build_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Build the per-(ticker,day) analysis panel from daily OHLC.

    Columns added (all per ticker, chronological):
      * ``disp``      — signed intraday **displacement** proxy for closing pressure:
                        ``(close - open) / (high - low)`` clipped to [-1, 1]. +1 ⇒ the day
                        opened at the low and closed at the high (max upward push into the
                        close); −1 ⇒ the mirror. This is the explicit PROXY for the
                        closing-auction order imbalance.
      * ``intraday``  — the intraday return ``close/open - 1`` (the size of the push).
      * ``overnight`` — the **next** session's gap ``next_open / close - 1`` (the close→open
                        return the reversal is supposed to live in).
      * ``rebal``     — True on index-rebalance proxy days (third Friday of quarter-end).

    Rows where ``high == low`` (zero range) or with no next session are dropped.
    """
    out = []
    for tk, g in prices.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        rng = (g["high"] - g["low"]).replace(0.0, np.nan)
        # drop suspected split bars (single-day open->close jump > 25%)
        jump = (g["close"] / g["open"] - 1.0).abs()
        disp = ((g["close"] - g["open"]) / rng).clip(-1.0, 1.0)
        intraday = g["close"] / g["open"] - 1.0
        next_open = g["open"].shift(-1)
        overnight = next_open / g["close"] - 1.0
        d = pd.DataFrame({
            "date": g["date"],
            "ticker": tk,
            "disp": disp,
            "intraday": intraday,
            "overnight": overnight,
            "rebal": is_rebalance_day(pd.DatetimeIndex(g["date"])),
        })
        d = d[jump.values <= 0.25]
        d = d.dropna(subset=["disp", "overnight"])
        out.append(d)
    panel = pd.concat(out, ignore_index=True)
    return panel


def load_real(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Convenience: cached OHLC -> analysis panel in one call."""
    return build_panel(load_prices(path))


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_days: int = 4_000, reversal: float = 0.0, seed: int = 376,
                    sig_intraday: float = 0.010, sig_overnight: float = 0.006,
                    ticker: str = "SYN") -> pd.DataFrame:
    """Deterministic OHLC-style panel with a **planted overnight-reversal edge**.

    Each day gets a random intraday push (``intraday``) and a random *base* overnight gap.
    The planted MOC-imbalance edge mechanically subtracts ``reversal`` × (the day's signed
    displacement, scaled to a return) from the next overnight gap — i.e. a fraction of the
    into-the-close push snaps back overnight. With ``reversal = 0`` no reversal is planted
    and the inference must NOT manufacture significance; with a large ``reversal`` it must.

    A consistent High/Low is fabricated around the open/close so the displacement proxy is
    well-defined; the synthetic ``disp`` therefore matches the real definition exactly.
    """
    rng = np.random.default_rng(seed)
    # intraday push and its sign-encoded displacement (open=base, close=open*(1+intraday))
    intraday = rng.normal(0.0, sig_intraday, size=n_days)
    # displacement proxy in [-1,1]: encode the push direction with a random fill ratio
    fill = rng.uniform(0.4, 1.0, size=n_days)           # how much of the range the close used
    disp = np.sign(intraday) * fill
    disp[intraday == 0.0] = 0.0

    base_overnight = rng.normal(0.0, sig_overnight, size=n_days)
    # planted reversal: a fraction of today's intraday push reverses next overnight.
    overnight = base_overnight.copy()
    overnight[:-1] += -reversal * intraday[:-1]         # tomorrow's gap fades today's push

    # fabricate a consistent OHLC so build_panel-style fields are exact
    open_ = 100.0 * np.exp(np.cumsum(np.r_[0.0, (intraday + overnight)[:-1]]))
    close = open_ * (1.0 + intraday)
    span = np.maximum(np.abs(intraday), 1e-4) / np.maximum(fill, 1e-3)
    hi = np.maximum(open_, close) + 0.5 * span * open_ * (1.0 - fill)
    lo = np.minimum(open_, close) - 0.5 * span * open_ * (1.0 - fill)

    idx = pd.bdate_range("1990-01-02", periods=n_days)
    rebal = is_rebalance_day(idx)
    panel = pd.DataFrame({
        "date": idx,
        "ticker": ticker,
        "disp": np.clip(disp, -1.0, 1.0),
        "intraday": intraday,
        "overnight": overnight,
        "rebal": rebal,
    })
    return panel
