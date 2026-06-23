"""Data layer for Study 387 — Economic-Surprise-Index (FRED macro + SPY).

Two sources, both offline-friendly once the caches exist:

* **Real tape.** A handful of public, monthly U.S. real-activity series from FRED
  (``PAYEMS`` nonfarm payrolls, ``INDPRO`` industrial production, ``RSAFS`` retail sales,
  ``UMCSENT`` consumer sentiment, ``HOUST`` housing starts, ``DGORDER`` durable-goods
  orders), cached under ``_cache/fred_macro.csv``; plus SPY daily adjusted closes under
  ``_cache/spy.csv``. From the macro panel we **construct a transparent surprise index**:
  Citi's CESI is proprietary, and there is no free history of *analyst expectations*, so we
  proxy the consensus with each series' own **trailing-12-month average growth** (an
  unbiased, clearly-labelled stand-in for the Street's forecast). The surprise is
  actual-minus-proxy, standardized by a trailing rolling std, and averaged across series.

* **Synthetic.** A deterministic, fixed-seed generator that produces a controllable surprise
  index and a SPY-like price with a PLANTED forward edge knob. ``edge=0`` must NOT manufacture
  significance; a large planted edge MUST light up. It is the positive control: the inference
  must recover the truth on data where we know it.

Pure numpy + pandas + stdlib for the offline path. ``fetch_macro`` / ``fetch_spy`` (network)
are only used once to build the caches and are never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
MACRO_CACHE = os.path.join(CACHE_DIR, "fred_macro.csv")
SPY_CACHE = os.path.join(CACHE_DIR, "spy.csv")

# The FRED panel that builds the surprise index. Each series is a public, monthly U.S.
# real-activity indicator with a long history. "diff" series are measured as the
# month-over-month change of the level (a sentiment index in points); "growth" series as
# month-over-month log-growth (a quantity that compounds). This is a transparent PROXY for
# Citi's proprietary CESI, named as such everywhere.
SERIES = {
    "PAYEMS":  "growth",   # nonfarm payrolls (employment)
    "INDPRO":  "growth",   # industrial production
    "RSAFS":   "growth",   # advance retail sales
    "UMCSENT": "diff",     # U. Michigan consumer sentiment (points)
    "HOUST":   "growth",   # housing starts
    "DGORDER": "growth",   # durable goods new orders
}


# --------------------------------------------------------------------------- #
# Real tape — fetchers (network; used once to build the caches)
# --------------------------------------------------------------------------- #
def fetch_macro(path: str = MACRO_CACHE) -> pd.DataFrame:
    """Download the FRED macro panel and cache a wide monthly CSV (index=date)."""
    import io
    import urllib.request

    def _fred(sid: str) -> pd.Series:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        raw = urllib.request.urlopen(url, timeout=40).read().decode()
        df = pd.read_csv(io.StringIO(raw))
        df.columns = ["date", "val"]
        df["date"] = pd.to_datetime(df["date"])
        df["val"] = pd.to_numeric(df["val"], errors="coerce")
        return df.dropna().set_index("date")["val"]

    panel = pd.DataFrame({sid: _fred(sid) for sid in SERIES}).sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    panel.to_csv(path)
    return panel


def fetch_spy(path: str = SPY_CACHE, start: str = "1992-01-01",
              end: str | None = None) -> pd.Series:
    """Download SPY daily adjusted closes and cache them (network; once)."""
    import yfinance as yf

    spy = yf.download("SPY", start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    spy.to_csv(path)
    return spy


# --------------------------------------------------------------------------- #
# Real tape — offline loaders
# --------------------------------------------------------------------------- #
def have_real(macro: str = MACRO_CACHE, spy: str = SPY_CACHE) -> bool:
    return os.path.exists(macro) and os.path.exists(spy)


def load_macro(path: str = MACRO_CACHE) -> pd.DataFrame:
    """Cached wide FRED panel (index=month, columns=series)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def load_spy(path: str = SPY_CACHE) -> pd.Series:
    """Cached SPY daily adjusted close as a Series named 'spy'."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    s = df.iloc[:, 0].astype(float)
    s.name = "spy"
    return s


def spy_month_end(spy: pd.Series | None = None) -> pd.Series:
    """SPY resampled to month-end closes (the monthly tape the ESI trades on)."""
    if spy is None:
        spy = load_spy()
    me = spy.resample("ME").last().dropna()
    me.name = "spy"
    return me


# --------------------------------------------------------------------------- #
# The surprise index — the transparent CESI proxy
# --------------------------------------------------------------------------- #
def _series_surprise(level: pd.Series, kind: str, consensus_window: int = 12,
                     z_window: int = 36) -> pd.Series:
    """One series' standardized surprise: actual-minus-trailing-consensus, z-scored.

    ``kind='growth'`` uses month-over-month log-growth; ``kind='diff'`` the level change.
    The **consensus proxy** is the trailing ``consensus_window``-month mean of that
    change (shifted so it uses only past data — no look-ahead). The raw surprise is
    actual-minus-proxy; it is standardized by a trailing ``z_window``-month rolling std.
    """
    s = level.astype(float).dropna()
    if kind == "growth":
        chg = np.log(s).diff()
    else:
        chg = s.diff()
    consensus = chg.shift(1).rolling(consensus_window, min_periods=6).mean()
    raw = chg - consensus
    sd = raw.shift(1).rolling(z_window, min_periods=12).std()
    z = raw / sd
    return z


def surprise_index(macro: pd.DataFrame | None = None, consensus_window: int = 12,
                   z_window: int = 36) -> pd.Series:
    """Composite Economic Surprise Index = mean of the per-series standardized surprises.

    A transparent, public-data PROXY for Citi's CESI. Positive ⇒ the data, on average, came
    in *above* its own trailing-average pace (a 'beat'); negative ⇒ a 'miss'. No look-ahead:
    every input to month ``t`` is known by the end of month ``t``.
    """
    if macro is None:
        macro = load_macro()
    cols = []
    for sid, kind in SERIES.items():
        if sid in macro.columns:
            cols.append(_series_surprise(macro[sid], kind, consensus_window, z_window))
    panel = pd.concat(cols, axis=1)
    esi = panel.mean(axis=1, skipna=True)
    esi.name = "esi"
    return esi.dropna()


def build_real(consensus_window: int = 12, z_window: int = 36) -> pd.DataFrame:
    """Convenience: monthly frame with columns ``esi`` and ``spy`` (month-end), aligned.

    The ESI for month ``t`` is known at month-end ``t``; ``spy`` is the month-end close. The
    strategy layer applies the execution lag, so this frame carries no look-ahead by itself.
    """
    esi = surprise_index(load_macro(), consensus_window, z_window)
    spy = spy_month_end()
    # align ESI month label to the month-end SPY stamp
    esi_me = esi.copy()
    esi_me.index = esi_me.index + pd.offsets.MonthEnd(0)
    out = pd.DataFrame({"esi": esi_me}).join(pd.DataFrame({"spy": spy}), how="inner")
    return out.dropna()


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic(n_months: int = 360, edge: float = 0.0, seed: int = 387,
              mu_monthly: float = 0.006, sig_monthly: float = 0.040,
              phi: float = 0.55) -> pd.DataFrame:
    """Deterministic monthly ESI + SPY-like price with a PLANTED forward edge.

    The surprise index is an AR(1) (persistence ``phi``) standardized to ~unit variance — the
    same mild autocorrelation a real surprise index carries. SPY-like monthly returns have
    drift ``mu_monthly`` and vol ``sig_monthly``; if ``edge`` != 0 an *extra* return of
    ``edge`` per unit of (lagged, sign-of) ESI is injected, i.e. next month's return gets
    ``edge * sign(esi_t)`` so a high-ESI month genuinely predicts a higher next-month return.

    ``edge = 0`` is the null: the ESI carries no forward information and the inference must
    NOT manufacture significance. A large ``edge`` must light the test up. The date index is a
    decorative monthly label built with ``period_range`` (no OutOfBounds for long spans).
    """
    rng = np.random.default_rng(seed)
    esi = np.empty(n_months)
    esi[0] = rng.normal()
    for t in range(1, n_months):
        esi[t] = phi * esi[t - 1] + rng.normal(0, np.sqrt(1 - phi ** 2))

    ret = rng.normal(mu_monthly, sig_monthly, size=n_months)
    if edge != 0.0:
        # high-surprise month t -> extra drift in month t+1 (proportional to the surprise)
        ret[1:] += edge * esi[:-1]

    price = 100.0 * np.exp(np.cumsum(ret))
    idx = pd.period_range("1990-01", periods=n_months, freq="M").to_timestamp(how="end")
    return pd.DataFrame({"esi": esi, "spy": price}, index=idx)
