"""Data layer for Study 516 — Dividend-Month-Premium.

Two sources, both offline-friendly once cached:

* **Real tape.** Daily adjusted closes for a fixed basket of long-listed US large-caps
  (yfinance, no key) plus, for each name, its **dividend payment dates** (yfinance
  ``Ticker.dividends`` / ``.history(actions=True)``). From the dividend history we learn each
  name's **payment-month pattern** (which calendar months it historically pays), then flag, for
  every month on the tape, whether the name is **predicted** to pay a dividend that month. The
  prediction uses only dividends *strictly before* the month in question — no look-ahead, the
  whole point of Hartzmark-Solomon's "predicted" framing. Prices cached wide
  (``dmp_prices.csv``); dividend dates cached long (``dmp_divs.csv``).

* **Synthetic.** A deterministic, fixed-seed generator producing a price panel whose names pay
  quarterly dividends on a fixed cadence, with a controllable **planted in-month premium**
  (knob ``edge``): an extra mean monthly return injected on predicted-dividend months. It is the
  positive control — with ``edge = 0`` the predicted-month-vs-rest gap must NOT manufacture
  significance; with a large planted ``edge`` it must light up.

Pure numpy + pandas + stdlib for the offline path. ``fetch_panel`` (network) is used only once
to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "dmp_prices.csv")
DIVS_CACHE = os.path.join(CACHE_DIR, "dmp_divs.csv")

# A transparent, fixed basket of large, long-listed US large-caps that have paid dividends for
# many years (deep, clean dividend histories on yfinance). Chosen for long price history + a
# regular dividend cadence + sector spread. This is a *survivors* basket (all still trading in
# 2026, all still paying) — survivorship is named on the Signal axis: a fixed surviving-and-still
# -paying-names basket cannot include firms that cut their dividend or blew up, a mild upward
# tilt on every leg, and it selects on *being a steady payer* (exactly the regime the premium
# is supposed to live in).
BASKET = [
    "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "IBM", "CVX",
    "PFE", "MRK", "INTC", "CSCO", "HD", "MCD", "DIS", "BA", "CAT", "MMM",
    "HON", "ORCL", "PEP", "ABT", "TXN", "COST", "LOW", "AMGN", "GS", "T",
]

# A name must have paid in a calendar month at least this many times in the past for that month
# to count as a *predicted* dividend month. Hartzmark-Solomon predict from the regular cadence.
MIN_HISTORY = 2


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = "2000-01-01", end: str | None = None,
                prices_path: str = PRICES_CACHE, divs_path: str = DIVS_CACHE,
                retries: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download basket prices + per-name dividend payment dates and cache them.

    Network-only; used once to build the cache. Never imported by the offline notebook cells.
    Guards yfinance flakiness with simple retries. Writes a wide adjusted-close CSV and a long
    dividend CSV with one row per (ticker, ex-dividend date, amount).
    """
    import yfinance as yf

    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(BASKET, start=start, end=end, auto_adjust=True,
                              progress=False)["Close"]
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            time.sleep(2.0)
    raw = raw.dropna(how="all")
    keep = [c for c in raw.columns if raw[c].notna().mean() >= 0.60]
    prices = raw[keep].copy()
    os.makedirs(os.path.dirname(prices_path), exist_ok=True)
    prices.to_csv(prices_path)

    rows = []
    for tk in keep:
        div = None
        for _ in range(retries):
            try:
                div = yf.Ticker(tk).dividends
                if div is not None and len(div) > 0:
                    break
            except Exception:
                time.sleep(1.5)
        if div is None or len(div) == 0:
            continue
        for ts, amt in div.items():
            d = pd.Timestamp(ts).tz_localize(None).normalize()
            rows.append({"ticker": tk, "date": d, "amount": float(amt)})
    divs = (pd.DataFrame(rows).dropna(subset=["date"])
            .drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"]))
    divs.to_csv(divs_path, index=False)
    return prices, divs


def have_real(prices_path: str = PRICES_CACHE, divs_path: str = DIVS_CACHE) -> bool:
    return os.path.exists(prices_path) and os.path.exists(divs_path)


def load_prices(path: str = PRICES_CACHE) -> pd.DataFrame:
    """Wide adjusted-close frame (index = date, columns = tickers)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def load_divs(path: str = DIVS_CACHE) -> pd.DataFrame:
    """Long dividend frame: columns ticker, date, amount."""
    dv = pd.read_csv(path, parse_dates=["date"])
    return dv.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_real() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: cached prices + dividend dates."""
    return load_prices(), load_divs()


# --------------------------------------------------------------------------- #
# Monthly returns + predicted-dividend-month flags (the core construction)
# --------------------------------------------------------------------------- #
def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Wide monthly simple returns (month-end to month-end), index = monthly period-end.

    Drops the final, *incomplete* calendar month so a stamped run never contains a partial
    month: if the price tape ends before its last calendar month is over, that month-end bucket
    is built from a truncated window and is removed.
    """
    m = prices.resample("ME").last()
    ret = m.pct_change()
    last_px = prices.index.max()
    last_bucket = ret.index.max()
    # if the tape stops before the last bucket's month genuinely ended, drop that partial month
    if last_px < (last_bucket - pd.offsets.MonthEnd(0)) or last_px.day < last_bucket.day:
        ret = ret.iloc[:-1]
    return ret


def build_predicted_flags(prices: pd.DataFrame, divs: pd.DataFrame,
                          min_history: int = MIN_HISTORY) -> dict:
    """Per-ticker monthly return series + a PAST-ONLY predicted-dividend-month flag.

    For each name and each month ``M`` on the tape we ask: among the dividends paid *strictly
    before* month ``M``, has this name paid in calendar-month ``M.month`` at least ``min_history``
    times? If so, ``M`` is a *predicted* dividend month. The flag therefore uses no information
    from month ``M`` or later — it is exactly the Hartzmark-Solomon "predicted to pay" signal,
    knowable in advance from the regular payment cadence.

    Returns ``{ticker: (period_index, monthly_ret, is_predicted)}`` where ``monthly_ret[i]`` is
    the simple return of month ``i`` and ``is_predicted[i]`` flags it as a predicted-dividend
    month for that name.
    """
    mret = monthly_returns(prices)
    # map each dividend to its (year, month) and the calendar month number
    dv = divs.copy()
    dv["ym"] = dv["date"].dt.to_period("M")
    out = {}
    for tk in mret.columns:
        s = mret[tk].dropna()
        if len(s) < 24:
            continue
        periods = s.index.to_period("M")
        vals = s.values
        flag = np.zeros(len(vals), dtype=bool)
        d_tk = dv[dv["ticker"] == tk]
        if len(d_tk) == 0:
            out[tk] = (periods, vals, flag)
            continue
        # for past-only prediction: list of (period, calendar_month) of each past dividend
        pay_periods = d_tk["ym"].values            # numpy PeriodArray-ish
        pay_months = d_tk["date"].dt.month.values
        pay_periods_int = np.array([p.ordinal for p in d_tk["ym"]])
        for i, per in enumerate(periods):
            cal_month = per.month
            cutoff = per.ordinal                    # strictly-before this month's ordinal
            # count past payments in this calendar month
            mask = (pay_months == cal_month) & (pay_periods_int < cutoff)
            if mask.sum() >= min_history:
                flag[i] = True
        out[tk] = (periods, vals, flag)
    return out


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_names: int = 30, n_months: int = 240, edge: float = 0.0,
                    seed: int = 516, sig_monthly: float = 0.055) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic monthly price panel + dividend dates with a PLANTED dividend-month premium.

    Each name pays quarterly dividends on a fixed offset (so 4 calendar months per year are
    "dividend months"). If ``edge`` != 0, EVERY predicted dividend month gets an *extra* mean
    monthly return of ``edge`` (the planted premium). With ``edge = 0`` dividend months are
    statistically identical to the rest: the predicted-month-vs-rest gap must NOT reach
    significance however the noise falls.

    Returns (daily-ish price wide frame, dividend-date frame) in the same shape as ``load_real``.
    We synthesise *monthly* prices on a decorative period index and expose them as a daily-style
    frame whose month-end resample reproduces the planted monthly returns exactly.
    """
    rng = np.random.default_rng(seed)
    # decorative monthly period index (period_range avoids the ns-overflow trap of date_range)
    pidx = pd.period_range("2000-01", periods=n_months, freq="M")
    month_end = pidx.to_timestamp(how="end").normalize()
    names = [f"N{i:02d}" for i in range(n_names)]

    price_cols = {}
    div_rows = []
    for j, name in enumerate(names):
        offset = j % 3                       # stagger the quarterly cadence across names
        pay_months = [(offset + k * 3) % 12 + 1 for k in range(4)]  # 4 calendar months/yr
        ret = rng.normal(0.006, sig_monthly, size=n_months)
        is_div = np.array([(p.month in pay_months) for p in pidx])
        if edge != 0.0:
            ret[is_div] += edge
        price = 100.0 * np.exp(np.cumsum(ret))
        price_cols[name] = price
        # record dividend payment dates (the actual pay months, from the first year on)
        for i, p in enumerate(pidx):
            if is_div[i] and p.year >= pidx[0].year + 1:   # leave a year of "history" to learn
                div_rows.append({"ticker": name, "date": month_end[i], "amount": 0.5})
        # also seed the early-history payments so the predictor has >= MIN_HISTORY
        for i, p in enumerate(pidx):
            if is_div[i] and p.year == pidx[0].year:
                div_rows.append({"ticker": name, "date": month_end[i], "amount": 0.5})

    prices = pd.DataFrame(price_cols, index=month_end)
    divs = (pd.DataFrame(div_rows).drop_duplicates(["ticker", "date"])
            .sort_values(["ticker", "date"]).reset_index(drop=True))
    return prices, divs
