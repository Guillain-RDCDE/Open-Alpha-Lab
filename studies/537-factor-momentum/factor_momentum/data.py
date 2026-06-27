"""Data layer for Study 537 — Factor-Momentum (Ehsani-Linnainmaa 2022).

The claim under test is one level up from stock momentum: it is the **factors themselves**
that trend. So the data layer's job is to manufacture a small panel of **monthly factor
return series**, on which the strategy layer then runs time-series momentum.

Two tapes, one schema — a (month × factor) frame of long-short factor returns:

* **Real tape.** Daily adjusted closes for a fixed large-cap **survivor** basket (yfinance,
  no key), cached under this study's own ``_cache/``. From prices alone we build a set of
  transparent, **price-derived** long-short factor portfolios, rebalanced monthly:

    - ``mom``   12-1 cross-sectional momentum (top-third minus bottom-third)
    - ``lowvol``  low trailing-vol minus high trailing-vol
    - ``lowbeta`` low rolling-beta minus high (a BAB-flavoured factor, cf. Study 238)
    - ``strev``  short-term (1-month) reversal: past-loser minus past-winner
    - ``size``   small minus big (rolling dollar-size proxy = price level on split-adjusted
                 series is unreliable, so we proxy size by trailing 60-day average $-range;
                 named explicitly as a proxy)

  We deliberately stay **price-only**: yfinance fundamentals (``.info`` P/B, P/E) are thin,
  point-in-time-unsafe and flaky on CI, so a value/quality factor cannot be built honestly
  here. We *name* that limitation rather than fake it (see ``docs/results.md``).

* **Synthetic.** A deterministic, fixed-seed generator that emits a set of factor return
  series with a **planted autocorrelation** knob (``ar``): ``ar = 0`` is the null (white-noise
  factors — time-series momentum must NOT manufacture a premium); ``ar > 0`` plants genuine
  factor-level persistence that factor-momentum must recover. The positive control.

The real basket is **survivorship-biased** (current large-caps projected back): the factor
long legs tilt up. Named on the Signal axis; results are upper bounds.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(HERE, ".."))
CACHE_DIR = os.path.join(STUDY_DIR, "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "fm_prices.parquet")

# Fixed large-cap survivor basket (still trading 2026). ~40 names, sector-spread, deep
# history. Survivorship is named on the Signal axis: a fixed surviving-names basket cannot
# include firms that blew up, tilting factor long legs up — results are upper bounds.
BASKET = [
    "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "IBM", "CVX",
    "PFE", "MRK", "INTC", "CSCO", "HD", "MCD", "DIS", "BA", "CAT", "MMM",
    "HON", "UNH", "ORCL", "PEP", "ABT", "TXN", "COST", "LOW", "AMGN", "GS",
    "AXP", "BAC", "C", "WFC", "T", "VZ", "GE", "F", "GM", "TGT",
]


# --------------------------------------------------------------------------- #
# Real tape — yfinance, study-local cache
# --------------------------------------------------------------------------- #
def have_real(prices_path: str = PRICES_CACHE) -> bool:
    return os.path.exists(prices_path)


def fetch_panel(start: str = "2004-01-01", end: str | None = None,
                prices_path: str = PRICES_CACHE, retries: int = 3) -> pd.DataFrame:
    """Download basket daily adjusted closes (+SPY) and cache to parquet.

    Network-only; used once to build the cache. Retries guard yfinance flakiness. Returns
    the wide adjusted-close frame (columns = tickers + SPY).
    """
    import time

    import yfinance as yf

    tickers = list(BASKET) + ["SPY"]
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                              progress=False, threads=True)["Close"]
            raw = raw.dropna(how="all")
            if raw.empty:
                raise RuntimeError("empty download")
            keep = [c for c in raw.columns if raw[c].notna().mean() >= 0.60]
            prices = raw[keep].copy()
            os.makedirs(os.path.dirname(prices_path), exist_ok=True)
            prices.to_parquet(prices_path)
            return prices
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"fetch_panel failed after {retries} retries: {last_err}")


def load_prices(path: str = PRICES_CACHE) -> pd.DataFrame:
    """Wide adjusted-close frame (index = date, columns = tickers incl. SPY)."""
    return pd.read_parquet(path).sort_index()


# --------------------------------------------------------------------------- #
# Building the monthly factor return panel from prices
# --------------------------------------------------------------------------- #
FACTORS = ("mom", "lowvol", "lowbeta", "strev", "size")


def _month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last trading day of each *complete* calendar month present in ``idx``.

    A trailing **partial** month (the data ends mid-month, e.g. an as-of of 2026-06-26 when
    June is not yet over) is dropped: its last available trading day sits well before the
    calendar month-end, so it would label a partial-month return as a full month. House rule:
    no partial periods in a stamped run. We keep a month only if its last observed trading day
    is within a few business days of the calendar month-end (markets close before the 1st/2nd
    etc., so a small slack is normal); the in-progress final month fails this and is removed.
    """
    s = pd.Series(idx, index=idx)
    me = pd.DatetimeIndex(s.groupby([idx.year, idx.month]).last().values).sort_values()
    if len(me) == 0:
        return me
    # Drop the trailing month unless the data extends *past* that month's calendar end — i.e.
    # the month is only "complete" if a later calendar day exists in the series. The in-progress
    # final month (as-of mid-month) is removed; an as-of that lands exactly on/after a month-end
    # keeps it. This is the run-date-independent way to refuse a partial-month observation.
    last = me[-1]
    last_cal_end = last + pd.offsets.MonthEnd(0)
    if idx.max() < last_cal_end:
        me = me[:-1]
    return me


def build_factor_panel(prices: pd.DataFrame, mkt_col: str = "SPY",
                       n_split: int = 3) -> pd.DataFrame:
    """Build a (month × factor) frame of long-short monthly factor returns from prices.

    For each month-end *t* we rank the basket on each characteristic using only data up to
    *t*, form a long-short (top-third vs bottom-third) portfolio, and hold it for the next
    calendar month (so every factor return is realised **out of sample** of its signal — a
    clean one-month execution lag at the factor level). Characteristics:

      * ``mom``     trailing 12-1 month return (skip the most recent month) — long winners.
      * ``lowvol``  trailing 60-day return vol — long the **low**-vol third.
      * ``lowbeta`` rolling 120-day beta vs the market — long the **low**-beta third.
      * ``strev``   prior-month return — long the **losers** (short-term reversal).
      * ``size``    trailing 60-day mean absolute daily $-move (a price-only liquidity/size
                    proxy) — long the **small** (low-activity) third. Named a *proxy*.

    Returns a DataFrame indexed by holding-month-end, columns = FACTORS, values = the
    long-short monthly return of each factor.
    """
    px = prices.copy()
    mkt = px[mkt_col] if mkt_col in px.columns else None
    stocks = [c for c in px.columns if c != mkt_col]
    sub = px[stocks]
    rets = sub.pct_change()
    mkt_ret = mkt.pct_change() if mkt is not None else None

    me = _month_ends(px.index)
    rows: list[dict] = []
    for i in range(len(me) - 1):
        t = me[i]
        t_next = me[i + 1]
        hist = sub.loc[:t]
        if len(hist) < 130:
            continue

        # --- characteristics measured at t (signal), using only data up to t ---
        # 12-1 momentum: return from t-252 to t-21
        def _ret_window(back_a: int, back_b: int) -> pd.Series:
            if len(hist) <= back_a:
                return pd.Series(dtype=float)
            a = hist.iloc[-back_a]
            b = hist.iloc[-back_b] if back_b > 0 else hist.iloc[-1]
            return (b / a - 1.0)

        mom = _ret_window(252, 21)
        strev = _ret_window(21, 0)
        vol = rets.loc[:t].iloc[-60:].std()
        # rolling beta vs market over last 120 days
        if mkt_ret is not None:
            r120 = rets.loc[:t].iloc[-120:]
            m120 = mkt_ret.loc[:t].iloc[-120:]
            cov = r120.apply(lambda col: col.cov(m120))
            beta = cov / m120.var()
        else:
            beta = pd.Series(np.nan, index=stocks)
        # size proxy: trailing 60-day mean absolute daily return (low = "small"/quiet)
        size_proxy = rets.loc[:t].iloc[-60:].abs().mean()

        # holding-period returns over (t, t_next]
        hold = sub.loc[(sub.index > t) & (sub.index <= t_next)]
        if hold.empty:
            continue
        fwd = (1.0 + hold.pct_change().fillna(0.0)).prod() - 1.0  # per-stock monthly return

        def _long_short(signal: pd.Series, long_low: bool) -> float:
            s = signal.dropna()
            common = s.index.intersection(fwd.dropna().index)
            s = s.loc[common]
            f = fwd.loc[common]
            if len(s) < 3 * n_split:
                return np.nan
            order = s.sort_values()
            k = len(order) // n_split
            low = order.index[:k]
            high = order.index[-k:]
            long_leg = low if long_low else high
            short_leg = high if long_low else low
            return float(f[long_leg].mean() - f[short_leg].mean())

        rows.append({
            "date": t_next,
            "mom": _long_short(mom, long_low=False),       # long winners
            "lowvol": _long_short(vol, long_low=True),     # long low-vol
            "lowbeta": _long_short(beta, long_low=True),   # long low-beta
            "strev": _long_short(strev, long_low=True),    # long losers (reversal)
            "size": _long_short(size_proxy, long_low=True),  # long quiet/small
        })

    fp = pd.DataFrame(rows).set_index("date").sort_index()
    return fp.dropna(how="all")


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_factors(n_factors: int = 5, n_months: int = 240, ar: float = 0.0,
                      mu: float = 0.0, sigma: float = 0.04,
                      seed: int = 537) -> tuple[pd.DataFrame, float]:
    """Deterministic monthly factor panel with a PLANTED AR(1) persistence knob.

    Each factor follows ``f_t = mu + ar * (f_{t-1} - mu) + eps_t`` with eps ~ N(0, sigma).
    ``ar = 0`` -> white-noise factors: time-series momentum on them must NOT manufacture a
    premium (the null). ``ar > 0`` -> genuine factor-level persistence that a
    momentum-on-factors strategy must recover. Returns ``(panel, ar)``.

    Uses ``pd.period_range`` for the decorative monthly index (avoids the ns-overflow trap
    on long synthetic spans).
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1990-01", periods=n_months, freq="M").to_timestamp("M")
    cols = [f"F{j}" for j in range(n_factors)]
    out = np.zeros((n_months, n_factors))
    eps = rng.normal(0.0, sigma, size=(n_months, n_factors))
    out[0] = mu + eps[0]
    for t in range(1, n_months):
        out[t] = mu + ar * (out[t - 1] - mu) + eps[t]
    return pd.DataFrame(out, index=idx, columns=cols), float(ar)


# --------------------------------------------------------------------------- #
# Fingerprint helper
# --------------------------------------------------------------------------- #
def fingerprint(df: pd.DataFrame | pd.Series) -> str:
    """Short content fingerprint for the as-of stamp in docs/results.md."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    arr = np.ascontiguousarray(df.fillna(0).to_numpy())
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
