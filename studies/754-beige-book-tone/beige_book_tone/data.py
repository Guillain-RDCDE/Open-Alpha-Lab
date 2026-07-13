"""Data layer for Study 754 — Beige-Book-Tone (release calendar + LM-tone proxy + SPY).

Three components, the first and third fully offline and deterministic:

* **Real release calendar (reconstructed) + LM-tone proxy (labelled).** The Federal
  Reserve publishes the **Beige Book** eight times a year, on a Wednesday roughly two
  weeks before each FOMC meeting. ``NOMINAL_CADENCE`` places the eight releases on that
  cadence and :func:`_nearest_wednesday` snaps each to a real Wednesday, so the release
  *dates* are genuine calendar Wednesdays on the Beige-Book schedule (2011..2024). The
  ``TONE_BY_YEAR`` table is a **labelled proxy** for the Loughran-McDonald net tone
  (``(#positive − #negative) / (#positive + #negative)``, z-scaled) that a full-text
  scrape of each release would produce — a small, narrative-anchored reconstruction
  (positive through the 2013–2019 expansion, deeply negative in the COVID spring of
  2020, negative through the 2022 inflation scare). It is **not** a live dictionary
  count of the full text; the full scrape is the beat-7 extension. Exactly like Study
  358 (watch-index) and Study 708 (eurovision-effect) use a small hardcoded, clearly
  labelled proxy series, the tone here is never dressed as a real tape.

* **Real SPY tape.** ``load_spy`` reads the cached daily SPY adjusted close
  (``_cache/spy.csv``, yfinance, no key). ``fetch_spy`` (network) rebuilds the cache and
  is never imported by the offline notebook cells. Price = total-return adjusted close
  (``auto_adjust=True``); labelled as such.

* **Synthetic positive control.** :func:`synthetic` is a deterministic, fixed-seed
  generator producing a release calendar with tones and a daily SPY-like price with a
  *planted* link: a positive-tone release nudges the following days' drift up by a
  controllable ``edge`` knob. ``edge = 0`` is the null (tone carries no forward
  information) and must NOT manufacture significance; a large planted ``edge`` must light
  the test up. The control runs anywhere with no network.

Pure numpy + pandas + stdlib for the offline path.
"""

from __future__ import annotations

import datetime as _dt
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPY_CACHE = os.path.join(HERE, "..", "_cache", "spy.csv")


# --------------------------------------------------------------------------- #
# Release calendar — the real Beige-Book Wednesday cadence (8x / year)
# --------------------------------------------------------------------------- #
# The Beige Book comes out ~2 weeks before each FOMC meeting, on a Wednesday. The eight
# nominal mid-cycle dates below are snapped to the nearest real Wednesday of each year, so
# every release date is a genuine calendar Wednesday on the Beige-Book schedule. Any that
# lands on a market holiday is resolved forward to the next SPY trading day at alignment.
NOMINAL_CADENCE: list[tuple[int, int]] = [
    (1, 16), (3, 6), (4, 17), (6, 5), (7, 17), (9, 4), (10, 16), (12, 4),
]

# LABELLED PROXY — Loughran-McDonald net-tone reconstruction, z-scaled, one value per
# release (eight per year, ordered by NOMINAL_CADENCE). Narrative-anchored to the actual
# tenor of each period; NOT a live full-text dictionary count. Positive => the anecdotes
# ran "better than typical"; negative => "worse than typical". As-of 2026-06-30.
TONE_BY_YEAR: dict[int, list[float]] = {
    2011: [0.30, 0.40, 0.50, 0.20, -0.10, -0.30, -0.50, 0.00],
    2012: [0.40, 0.50, 0.40, 0.20, 0.10, 0.30, 0.40, 0.50],
    2013: [0.50, 0.60, 0.40, 0.50, 0.60, 0.40, 0.20, 0.50],
    2014: [0.10, 0.20, 0.50, 0.60, 0.60, 0.70, 0.60, 0.60],
    2015: [0.30, 0.20, 0.00, 0.10, 0.20, -0.10, -0.20, 0.00],
    2016: [-0.20, -0.10, 0.10, 0.20, 0.30, 0.20, 0.30, 0.40],
    2017: [0.50, 0.60, 0.60, 0.70, 0.60, 0.70, 0.70, 0.80],
    2018: [0.70, 0.60, 0.60, 0.70, 0.60, 0.50, 0.30, -0.20],
    2019: [0.20, 0.30, 0.20, 0.10, 0.00, -0.10, 0.00, 0.10],
    2020: [0.20, 0.00, -1.60, -2.40, -1.60, -0.80, -0.30, 0.00],
    2021: [0.10, 0.40, 0.70, 0.80, 0.60, 0.50, 0.40, 0.30],
    2022: [0.30, 0.20, 0.00, -0.20, -0.40, -0.50, -0.60, -0.50],
    2023: [-0.30, -0.40, -0.20, -0.10, 0.00, 0.10, 0.00, 0.10],
    2024: [0.20, 0.30, 0.20, 0.10, 0.20, 0.30, 0.20, 0.30],
}


def _nearest_wednesday(year: int, month: int, day: int) -> pd.Timestamp:
    """Snap a nominal (year, month, day) to the nearest Wednesday (weekday() == 2)."""
    base = _dt.date(year, month, day)
    offset = (base.weekday() - 2) % 7          # days since the last Wednesday
    if offset > 3:                              # closer to the next Wednesday
        offset -= 7
    return pd.Timestamp(base - _dt.timedelta(days=offset))


def have_real(path: str = DEFAULT_SPY_CACHE) -> bool:
    """True iff the SPY cache exists (the release/tone table is always available)."""
    return os.path.exists(path)


def releases() -> pd.DataFrame:
    """Beige-Book releases: index = snapped release Wednesday, column ``tone`` (proxy).

    Eight rows per year (2011..2024), sorted by date. The tone is the labelled
    LM-net-tone proxy, never a live dictionary count of the full text.
    """
    rows = []
    for yr in sorted(TONE_BY_YEAR):
        tones = TONE_BY_YEAR[yr]
        for (m, d), tone in zip(NOMINAL_CADENCE, tones):
            rows.append((_nearest_wednesday(yr, m, d), float(tone)))
    idx = pd.DatetimeIndex([d for d, _ in rows])
    return pd.DataFrame({"tone": [t for _, t in rows]}, index=idx).sort_index()


def fetch_spy(start: str = "2010-10-01", end: str | None = None,
              path: str = DEFAULT_SPY_CACHE) -> pd.Series:
    """Download SPY daily adjusted close via yfinance and cache it (network-only).

    Used once to build ``_cache/spy.csv``. Never imported by offline cells. Total-return
    adjusted close (``auto_adjust=True``).
    """
    import yfinance as yf

    raw = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.DataFrame):          # recent yfinance returns a 1-col frame
        raw = raw.iloc[:, 0]
    out = pd.DataFrame({"SPY": raw.astype(float)}).dropna()
    out.index.name = "Date"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out["SPY"].astype(float)


def load_spy(path: str = DEFAULT_SPY_CACHE) -> pd.Series:
    """Load the cached daily SPY adjusted-close series (total-return adjusted)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df["SPY"].astype(float)


def load_real(path: str = DEFAULT_SPY_CACHE) -> tuple[pd.DataFrame, pd.Series]:
    """(`releases`, `spy_daily`) — the real-tape objects the strategy runs on.

    Releases whose forward window would overrun the SPY tape are kept; the strategy drops
    per-event returns that run past the last trading day. Only releases on/after the SPY
    start are retained.
    """
    spy = load_spy(path)
    rel = releases()
    rel = rel[(rel.index >= spy.index.min()) & (rel.index <= spy.index.max())]
    return rel, spy


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic(n_years: int = 14, edge: float = 0.0, seed: int = 754,
              mu_d: float = 0.0004, sig_d: float = 0.011) -> tuple[pd.DataFrame, pd.Series]:
    """Deterministic release calendar + daily SPY-like price with a PLANTED tone->drift link.

    Builds ``8 * n_years`` releases (tone ~ N(0,1), fixed seed) on a decorative business-day
    grid, and a daily SPY-like price. When ``edge != 0`` the ``h``-day window *after* each
    release gets an extra per-day drift of ``edge * tone`` — a positive-tone book genuinely
    pushes the following days up (the believers' story, injected with a knob).

    ``edge = 0`` => tone carries no forward information (the null); the inference must NOT
    manufacture significance. A large ``edge`` (e.g. 0.004 per day for 5 days) must drive
    the Welch t well past 2. The date index is a decorative business-day label built with
    ``bdate_range`` (no OutOfBounds risk).
    """
    rng = np.random.default_rng(seed)
    n_days = 260 * n_years + 40
    idx = pd.bdate_range("2000-01-05", periods=n_days)
    ret = rng.normal(mu_d, sig_d, size=n_days)

    # eight releases per year, evenly spaced across each 260-day block
    rel_pos = []
    for y in range(n_years):
        base = y * 260
        for k in range(8):
            rel_pos.append(base + 12 + k * 30)
    rel_pos = [p for p in rel_pos if p < n_days - 12]
    tones = rng.normal(0.0, 1.0, size=len(rel_pos))

    if edge != 0.0:
        for p, tone in zip(rel_pos, tones):
            ret[p + 1:p + 6] += edge * tone      # 5-day post-release window nudged by tone

    price = 100.0 * np.exp(np.cumsum(ret))
    spy = pd.Series(price, index=idx, name="SPY")
    rel = pd.DataFrame({"tone": tones}, index=idx[rel_pos]).sort_index()
    return rel, spy
