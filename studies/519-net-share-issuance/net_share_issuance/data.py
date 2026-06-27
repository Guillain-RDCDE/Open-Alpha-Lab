"""Data layer for Study 519 — Net-Share-Issuance.

Two sources, both offline-friendly once the cache exists:

* **Real tape.** A fixed ~40-name large-cap **survivor** basket (the same kind of basket
  238/330/363 use). For each name we pull, via yfinance (no key):

  - daily **adjusted** closes (split- & dividend-adjusted total-return prices), and
  - the **shares-outstanding** history (``Ticker.get_shares_full``), which we then
    **split-adjust** using the firm's split history (``history(actions=True)``) so a 4-for-1
    split does NOT masquerade as a 300% issuance. The split-adjusted share count is the clean
    signal: it falls when a firm buys back and rises when it dilutes.

  We resample both to a year-end grid and cache two wide CSVs under this study's own
  ``_cache/`` (``prices.csv`` = year-end adjusted close per ticker; ``shares.csv`` =
  year-end split-adjusted shares per ticker).

* **Synthetic.** A deterministic, fixed-seed cross-sectional panel with a **planted issuance
  edge** (low-issuance names earn ``edge`` more per year than high-issuance names). It is the
  positive control: ``edge=0`` must NOT manufacture a long-short t≥2 from ~40 names a handful
  of years; a large planted ``edge`` must light the sort up.

Network is touched ONLY by ``fetch_real`` (used once to build the cache). The offline core
(``load_real``, ``synthetic_panel``) never imports yfinance.
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

START = "2014-06-01"   # get_shares_full reaches back ~10y; first usable year-change is FY2016

# A fixed, transparent large-cap SURVIVOR basket (~40 names) spanning sectors. These are firms
# still trading in 2026 with a long shares-outstanding history on yfinance — a deliberate
# survivorship choice, named on the Signal axis. The point is the *axis* (issuance vs return),
# not the exact names; the verdict does not hinge on the roster.
BASKET = [
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "ORCL", "CSCO", "INTC", "IBM",
    "QCOM", "TXN", "ADBE", "CRM", "AMD", "XOM", "CVX", "COP", "JPM", "BAC",
    "WFC", "GS", "JNJ", "PFE", "MRK", "ABBV", "UNH", "WMT", "HD", "MCD",
    "KO", "PEP", "PG", "COST", "NKE", "DIS", "VZ", "T", "CAT", "BA",
]


# --------------------------------------------------------------------------- #
# Real tape (network) — builds the cache; never imported by the offline core
# --------------------------------------------------------------------------- #
def _split_adjusted_shares(tk: str, start: str = START, retries: int = 3):
    """Split-adjusted shares-outstanding series for one ticker (None if unavailable).

    yfinance's ``get_shares_full`` returns the **raw** share count, which jumps discontinuously
    on a split (a 4-for-1 split quadruples the count overnight with zero real issuance). We
    multiply every pre-split observation by the split ratio so the whole series is on the
    *post-split* basis — leaving only true issuance/buyback in the level.
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


def fetch_real(tickers=None, start: str = START) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download the basket, build year-end **adjusted close** and **split-adjusted shares**
    frames, and cache both under ``_cache/``. Network-only; run once.

    Returns ``(prices_ye, shares_ye)`` — both wide frames indexed by year-end date, one column
    per ticker. A ticker missing a shares history is dropped from BOTH frames so the sort sees
    an aligned panel.
    """
    if tickers is None:
        tickers = BASKET
    px = _adjusted_closes(tickers, start=start)
    px_ye = px.resample("YE").last()

    shares = {}
    for tk in tickers:
        a = _split_adjusted_shares(tk, start=start)
        if a is None:
            continue
        ye = a.resample("YE").last().ffill()
        shares[tk] = ye
    sh_ye = pd.DataFrame(shares)

    keep = [c for c in px_ye.columns if c in sh_ye.columns]
    px_ye, sh_ye = px_ye[keep], sh_ye[keep]
    # align on the common year-end grid
    idx = px_ye.index.union(sh_ye.index)
    px_ye = px_ye.reindex(idx).ffill()
    sh_ye = sh_ye.reindex(idx).ffill()

    os.makedirs(CACHE_DIR, exist_ok=True)
    px_ye.to_csv(PRICES_CSV)
    sh_ye.to_csv(SHARES_CSV)
    return px_ye, sh_ye


# --------------------------------------------------------------------------- #
# Offline core
# --------------------------------------------------------------------------- #
def have_real() -> bool:
    return os.path.exists(PRICES_CSV) and os.path.exists(SHARES_CSV)


def load_real() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the cached year-end (prices, split-adjusted shares) frames."""
    px = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True).sort_index()
    sh = pd.read_csv(SHARES_CSV, index_col=0, parse_dates=True).sort_index()
    return px, sh


def drop_partial_last_year(px: pd.DataFrame, sh: pd.DataFrame,
                           asof: pd.Timestamp | None = None
                           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """House rule: a stamped run carries no partial year. Drop the final year-end row if it is
    the in-progress current year (its year-end bar has not occurred yet)."""
    if asof is None:
        asof = pd.Timestamp.today().normalize()
    if len(px) == 0:
        return px, sh
    last = px.index[-1]
    if last.year >= asof.year:
        return px.iloc[:-1], sh.iloc[:-1]
    return px, sh


def fingerprint(px: pd.DataFrame, sh: pd.DataFrame) -> str:
    """Short content fingerprint of the (prices, shares) panel for the as-of stamp."""
    a = np.ascontiguousarray(px.fillna(0.0).to_numpy(dtype=float)).tobytes()
    b = np.ascontiguousarray(sh.fillna(0.0).to_numpy(dtype=float)).tobytes()
    return hashlib.sha1(a + b).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_names: int = 40, n_years: int = 9, edge: float = 0.0,
                    seed: int = 519, idio_vol: float = 0.28,
                    mkt_vol: float = 0.16, mkt_drift: float = 0.08
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic cross-sectional panel with a **planted net-issuance edge**.

    Each name draws an annual net-issuance value (its split-adjusted share-count change). The
    annual *return* of a name = market + idiosyncratic noise **minus** ``edge × (its issuance
    rank, centred)`` — so when ``edge>0``, high-issuance names earn less and low-issuance names
    earn more, exactly the factor the literature claims. With ``edge=0`` issuance is unrelated
    to return: the long-short sort must NOT manufacture a t≥2 from ~40 names over a handful of
    years, however the draws fall.

    Returns ``(prices_ye, shares_ye)`` in the same shape the strategy layer consumes (wide
    year-end frames, columns ``N00..``). A decorative year-end index is built with
    ``pd.period_range`` (never a huge ``date_range`` span) to stay overflow-safe.
    """
    rng = np.random.default_rng(seed)
    cols = [f"N{j:02d}" for j in range(n_names)]
    # decorative year-end index, overflow-safe
    idx = pd.period_range("2015", periods=n_years + 1, freq="Y").to_timestamp(how="end").normalize()
    idx = pd.DatetimeIndex(idx, name="date")

    # Each name has a persistent issuance "type": some are chronic diluters, some chronic
    # buybackers, plus year noise. Centre the per-year cross-section so the edge is a pure tilt.
    base_issue = rng.normal(0.0, 0.06, size=n_names)         # persistent dilution propensity
    shares = np.empty((n_years + 1, n_names))
    shares[0] = 1.0e9 * np.exp(rng.normal(0.0, 0.2, size=n_names))
    rets = np.empty((n_years, n_names))
    for y in range(n_years):
        issue = base_issue + rng.normal(0.0, 0.03, size=n_names)   # this year's net issuance
        shares[y + 1] = shares[y] * (1.0 + issue)
        # cross-sectionally centre this year's issuance to a pure tilt
        z = (issue - issue.mean()) / (issue.std() + 1e-12)
        mkt = mkt_drift + mkt_vol * rng.standard_normal()
        idio = idio_vol * rng.standard_normal(n_names)
        rets[y] = mkt + idio - edge * z          # high issuance (z>0) earns less
    # build a price level from the annual returns (start at 100)
    px = np.empty((n_years + 1, n_names))
    px[0] = 100.0
    for y in range(n_years):
        px[y + 1] = px[y] * (1.0 + rets[y])

    prices = pd.DataFrame(px, index=idx, columns=cols)
    shares_df = pd.DataFrame(shares, index=idx, columns=cols)
    return prices, shares_df
