"""Data layer for Study 569 — SBC-Dilution.

The claim (an accounting *hidden-cost* anomaly, a cousin of net-share-issuance): firms that pay
their people the most in **stock-based compensation** are quietly handing away ownership. That
comp is a real economic cost that barely dents GAAP earnings but shows up as **dilution** —
year-over-year growth in the share count. If the market under-weights that hidden cost, the
heaviest SBC / fastest-diluting names should go on to earn *lower* returns.

We build the signal a no-key retail stack can actually reach, from two yfinance sources:

* **Stock-based compensation intensity** — ``Stock Based Compensation`` (cash-flow statement)
  divided by ``Total Revenue`` (income statement). This is the "how much of the firm are you
  paying out in equity each year" leg. yfinance exposes only a **shallow annual statement
  history** (~4 fiscal years), so the SBC leg is a recent *snapshot* rather than a deep
  point-in-time panel — the limitation is named openly on the SIGNAL axis.
* **Dilution** — year-over-year growth in **split-adjusted** shares outstanding
  (``Ticker.get_shares_full`` corrected with the split history, so a 4-for-1 split does *not*
  masquerade as 300% dilution). The share count rises when a firm dilutes (SBC vesting, secondary
  raises) and falls when it buys back.

Two tapes, one schema:

* ``synthetic_panel`` — a deterministic, offline generator with a single knob ``edge`` that
  plants the anomaly: the higher a firm's true dilution/SBC type, the **lower** its forward
  return (``edge > 0`` reproduces the anomaly, the heavy-diluters underperform; ``edge = 0`` is
  the null). The study's positive control and null in one bottle. Tests never touch the network.
* ``fetch_real`` — the real yfinance tape (year-end adjusted prices, a per-name SBC/revenue
  snapshot, and split-adjusted year-end shares), cache-first into this study's own ``_cache/`` so
  the reproducible core never needs the network.

Survivorship: the basket is names *still trading in 2026*, so any firm whose over-dilution ran it
into the ground (delisting, wipe-out) is absent by construction. A point-in-time,
survivorship-free issuance/SBC universe is a Compustat product, not a free feed — stated on the
SIGNAL axis, and a hard cap: a synthetic-only or survivor-only study can never be `REAL`.

No look-ahead: the SBC intensity and the share count are public at fiscal year-end *t*; we sort
on them and hold the *next* year (*t → t+1*), one documented execution lag.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
PRICES_CSV = os.path.join(CACHE_DIR, "prices.csv")
SHARES_CSV = os.path.join(CACHE_DIR, "shares.csv")
SBC_CSV = os.path.join(CACHE_DIR, "sbc_intensity.csv")

START = "2014-06-01"

# A fixed, transparent large-cap SURVIVOR basket (~40 names) spanning SBC regimes: SBC-heavy
# mega-cap tech / software (META, NVDA, CRM, ADBE...), through to near-zero-SBC staples,
# industrials, banks and telecoms. The point is the *axis* (SBC/dilution vs return), not the
# exact roster; a deliberate survivorship choice, named on the Signal axis.
BASKET = [
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "ORCL", "CSCO", "INTC", "IBM",
    "QCOM", "TXN", "ADBE", "CRM", "AMD", "NFLX", "XOM", "CVX", "COP", "JPM",
    "BAC", "WFC", "GS", "JNJ", "PFE", "MRK", "ABBV", "UNH", "WMT", "HD",
    "MCD", "KO", "PEP", "PG", "COST", "NKE", "DIS", "VZ", "T", "CAT",
]


# --------------------------------------------------------------------------- #
# Real tape (network) — builds the cache; never imported by the offline core
# --------------------------------------------------------------------------- #
def _split_adjusted_shares(tk: str, start: str = START, retries: int = 3):
    """Split-adjusted shares-outstanding series for one ticker (None if unavailable).

    ``get_shares_full`` returns the **raw** count, which jumps on a split; we multiply every
    pre-split observation by the split ratio so the whole series is on the post-split basis,
    leaving only true issuance/buyback in the level.
    """
    import yfinance as yf

    last_err = None
    for attempt in range(retries):
        try:
            t = yf.Ticker(tk)
            s = t.get_shares_full(start=start)
            if s is None or len(s) == 0:
                return None
            s = s[~s.index.duplicated(keep="last")].sort_index()
            s.index = pd.DatetimeIndex(s.index).tz_localize(None)
            h = t.history(start=start, auto_adjust=True, actions=True)
            sp = h["Stock Splits"]
            sp = sp[sp != 0]
            if len(sp):
                sp.index = pd.DatetimeIndex(sp.index).tz_localize(None)
            factor = pd.Series(1.0, index=s.index)
            for d, r in sp.items():
                factor.loc[s.index < d] *= float(r)
            return (s * factor).astype(float)
        except Exception as exc:  # pragma: no cover - network flakiness
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    print(f"  ! shares fetch failed for {tk}: {last_err}")
    return None


def _adjusted_closes(tickers, start: str = START, retries: int = 3) -> pd.DataFrame:
    """Wide daily adjusted-close frame for ``tickers`` (yfinance, total-return adjusted)."""
    import yfinance as yf

    last_err = None
    for attempt in range(retries):
        try:
            raw = yf.download(list(tickers), start=start, auto_adjust=True,
                              progress=False)["Close"]
            raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
            return raw.dropna(how="all")
        except Exception as exc:  # pragma: no cover - network flakiness
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"price download failed: {last_err}")


def _sbc_intensity(tk: str, retries: int = 3):
    """Per-fiscal-year SBC / revenue for one ticker as a year-indexed Series (None if missing).

    ``Stock Based Compensation`` (cash-flow statement) / ``Total Revenue`` (income statement).
    yfinance exposes only a shallow annual history (~4 fiscal years), so this is a recent
    snapshot; indexed by the fiscal-year-end date so it can be resampled onto the year grid.
    """
    import yfinance as yf

    last_err = None
    for attempt in range(retries):
        try:
            t = yf.Ticker(tk)
            cf = t.cashflow
            fin = t.financials
            if cf is None or fin is None or cf.empty or fin.empty:
                return None
            if "Stock Based Compensation" not in cf.index or "Total Revenue" not in fin.index:
                return None
            sbc = cf.loc["Stock Based Compensation"].astype(float)
            rev = fin.loc["Total Revenue"].astype(float)
            common = sbc.index.intersection(rev.index)
            if len(common) == 0:
                return None
            s = (sbc.reindex(common) / rev.reindex(common)).dropna()
            s.index = pd.DatetimeIndex(s.index).tz_localize(None)
            return s.sort_index().astype(float)
        except Exception as exc:  # pragma: no cover - network flakiness
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    print(f"  ! SBC fetch failed for {tk}: {last_err}")
    return None


def fetch_real(tickers=None, start: str = START):
    """Download the basket, build year-end **adjusted close**, **split-adjusted shares** and a
    **SBC-intensity** frame, and cache all three under ``_cache/``. Network-only; run once.

    Returns ``(prices_ye, shares_ye, sbc_ye)`` — wide frames indexed by year-end date, one column
    per ticker. A ticker missing shares *or* SBC is dropped from all three so the sort sees an
    aligned panel.
    """
    if tickers is None:
        tickers = BASKET
    px = _adjusted_closes(tickers, start=start)
    px_ye = px.resample("YE").last()

    shares, sbc = {}, {}
    for tk in tickers:
        a = _split_adjusted_shares(tk, start=start)
        if a is not None:
            shares[tk] = a.resample("YE").last().ffill()
        s = _sbc_intensity(tk)
        if s is not None:
            # forward-fill the shallow SBC snapshot onto the year-end grid
            sbc[tk] = s.resample("YE").last()
    sh_ye = pd.DataFrame(shares)
    sbc_ye = pd.DataFrame(sbc)

    keep = [c for c in px_ye.columns if c in sh_ye.columns and c in sbc_ye.columns]
    idx = px_ye.index.union(sh_ye.index).union(sbc_ye.index)
    px_ye = px_ye[keep].reindex(idx).ffill()
    sh_ye = sh_ye[keep].reindex(idx).ffill()
    sbc_ye = sbc_ye[keep].reindex(idx).ffill()

    os.makedirs(CACHE_DIR, exist_ok=True)
    px_ye.to_csv(PRICES_CSV)
    sh_ye.to_csv(SHARES_CSV)
    sbc_ye.to_csv(SBC_CSV)
    return px_ye, sh_ye, sbc_ye


# --------------------------------------------------------------------------- #
# Offline core
# --------------------------------------------------------------------------- #
def have_real() -> bool:
    return (os.path.exists(PRICES_CSV) and os.path.exists(SHARES_CSV)
            and os.path.exists(SBC_CSV))


def load_real():
    """Load the cached year-end (prices, split-adjusted shares, SBC-intensity) frames."""
    px = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True).sort_index()
    sh = pd.read_csv(SHARES_CSV, index_col=0, parse_dates=True).sort_index()
    sbc = pd.read_csv(SBC_CSV, index_col=0, parse_dates=True).sort_index()
    return px, sh, sbc


def drop_partial_last_year(px, sh, sbc, asof: pd.Timestamp | None = None):
    """House rule: a stamped run carries no partial year. Drop the final year-end row if it is
    the in-progress current year (its year-end bar has not occurred yet)."""
    if asof is None:
        asof = pd.Timestamp.today().normalize()
    if len(px) == 0:
        return px, sh, sbc
    last = px.index[-1]
    if last.year >= asof.year:
        return px.iloc[:-1], sh.iloc[:-1], sbc.iloc[:-1]
    return px, sh, sbc


def fingerprint(*frames) -> str:
    """Short content fingerprint of the panel(s) for the as-of stamp."""
    h = hashlib.sha1()
    for f in frames:
        h.update(np.ascontiguousarray(f.fillna(0.0).to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_names: int = 40, n_years: int = 6, edge: float = 0.0,
                    seed: int = 569, idio_vol: float = 0.28,
                    mkt_vol: float = 0.16, mkt_drift: float = 0.09):
    """Deterministic cross-sectional panel with a **planted SBC/dilution anomaly**.

    Each name has a persistent SBC/dilution *type* (chronic heavy-diluters vs lean buyback-ers),
    which sets both its **SBC intensity** and its annual **share-count growth**. The annual
    *return* of a name = market + idiosyncratic noise **minus** ``edge × (its dilution rank,
    centred)`` — so with ``edge > 0`` the heavy-SBC / fast-diluting names earn *less* (the
    anomaly). With ``edge = 0`` SBC/dilution is unrelated to return: the long-short sort must NOT
    manufacture a t≥2 from ~40 names over a handful of years, however the draws fall.

    Returns ``(prices_ye, shares_ye, sbc_ye)`` in the same shape the strategy layer consumes
    (wide year-end frames, columns ``N00..``). A decorative year-end index is built with
    ``pd.period_range`` (never a huge ``date_range`` span) to stay overflow-safe.
    """
    rng = np.random.default_rng(seed)
    cols = [f"N{j:02d}" for j in range(n_names)]
    idx = pd.period_range("2018", periods=n_years + 1, freq="Y").to_timestamp(
        how="end").normalize()
    idx = pd.DatetimeIndex(idx, name="date")

    # Persistent dilution/SBC "type": high type = chronic heavy SBC and fast dilution.
    dil_type = rng.normal(0.0, 1.0, size=n_names)
    # SBC intensity: a positive level rising with the type (heavy-SBC names pay more equity).
    base_sbc = np.clip(0.02 + 0.03 * dil_type + 0.01 * rng.standard_normal(n_names), 0.0, 0.5)

    shares = np.empty((n_years + 1, n_names))
    shares[0] = 1.0e9 * np.exp(rng.normal(0.0, 0.2, size=n_names))
    sbc = np.empty((n_years + 1, n_names))
    rets = np.empty((n_years, n_names))
    for y in range(n_years):
        # this year's dilution: heavy types dilute more (SBC vesting), lean types buy back
        dilution = 0.005 * dil_type + rng.normal(0.0, 0.015, size=n_names)
        shares[y + 1] = shares[y] * (1.0 + dilution)
        sbc[y] = np.clip(base_sbc + 0.005 * rng.standard_normal(n_names), 0.0, 0.6)
        # rank of this year's dilution, centred, is the tilt the edge acts on
        z = (dilution - dilution.mean()) / (dilution.std() + 1e-12)
        mkt = mkt_drift + mkt_vol * rng.standard_normal()
        idio = idio_vol * rng.standard_normal(n_names)
        rets[y] = mkt + idio - edge * z          # heavy dilution (z>0) earns less
    sbc[n_years] = sbc[n_years - 1]

    px = np.empty((n_years + 1, n_names))
    px[0] = 100.0
    for y in range(n_years):
        px[y + 1] = px[y] * (1.0 + rets[y])

    prices = pd.DataFrame(px, index=idx, columns=cols)
    shares_df = pd.DataFrame(shares, index=idx, columns=cols)
    sbc_df = pd.DataFrame(sbc, index=idx, columns=cols)
    return prices, shares_df, sbc_df
