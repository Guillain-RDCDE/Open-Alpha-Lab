"""Data layer for Study 737 — Sunspot-Cycle.

The claim under test, in its oldest form: **the ~11-year solar/sunspot cycle drives
the economy and, through it, the stock market** — high solar activity → good times →
rising equities, and back again. William Stanley Jevons (1875-1878) tied the sunspot
cycle to commercial crises through harvests; a long tail of market folklore has kept a
"solar cycle → stock returns" curio alive ever since. We steelman it and test it.

Three ingredients, all offline-friendly once cached:

* **The solar calendar, hardcoded & cited.** ``SOLAR_CYCLES`` is the canonical table of
  solar cycles **16 → 25** with each cycle's minimum date, maximum date, and (SILSO v2,
  smoothed) peak sunspot number. No free daily-machine-readable feed of the raw SILSO
  file ships with this repo, so — exactly like the sibling studies that hand-build a
  cited calendar (``707-plane-crash-effect``'s ``DISASTERS``, ``313-geopolitical-shock``'s
  ``SHOCK_TABLE``, ``358-watch-index``'s hand-keyed proxy index) — the turning-point
  dates and amplitudes are keyed from SILSO / NOAA-SWPC published values. From them we
  reconstruct a **LABELLED PROXY** of the smoothed monthly sunspot number (a cosine
  pinned to the turning points and scaled by each cycle's peak amplitude) — a proxy for
  the phase of the cycle, *not* the raw daily SILSO tape, and named as such everywhere.

* **Real tape.** ``^GSPC`` — the S&P 500 **price index** (Yahoo! Finance, no key),
  daily close from 1927-12-30, cached under the study's own ``_cache/``. This is a
  **price-only** index: it carries no dividends, so it is labelled price-only throughout
  and never called "total return". We accept price-only deliberately here because the
  whole point is a **long** tape (≈ 98 years, ten solar cycles) — no total-return equity
  series reaches back to the Jevons-era length, and a dividend adjustment cannot
  manufacture or destroy an 11-year cyclicality anyway.

* **Synthetic world.** A deterministic, seeded monthly "equity-like" tape with a
  TUNABLE planted solar-cycle return component (``amp`` in monthly-return units, phased
  to the real cycle calendar). ``amp = 0`` is the null world — months statistically
  identical regardless of solar phase; the regime / phase machinery must NOT manufacture
  a cycle from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

GSPC_CACHE = os.path.join(CACHE_DIR, "sunspot_gspc.csv")

INDEX_TICKER = "^GSPC"          # S&P 500 PRICE index (price-only, labelled everywhere)
START = "1927-12-01"
AS_OF = "2026-06-30"            # last complete calendar month at publication (2026-07)

# --------------------------------------------------------------------------- #
# The solar-cycle calendar, hardcoded from SILSO (WDC-SILSO, Royal Observatory of
# Belgium) / NOAA-SWPC published turning points. Each row:
#   (cycle_number, minimum "YYYY-MM", maximum "YYYY-MM", peak_smoothed_ssn_v2)
# ``peak_smoothed_ssn_v2`` is the SILSO version-2 smoothed monthly sunspot number at
# the cycle maximum (approximate; only used to SCALE the labelled amplitude proxy, never
# in the phase tests). Cycle 25's maximum (2024-10) and amplitude are provisional at the
# 2026 as-of. Source: SILSO "Sunspot cycles data" and NOAA-SWPC "Solar Cycle Progression"
# — the same numbers reproduced across the solar-physics literature (Clette et al. 2016,
# Space Science Reviews, for the v2 recalibration).
# --------------------------------------------------------------------------- #
SOLAR_CYCLES: list[tuple[int, str, str, float]] = [
    (16, "1923-08", "1928-04", 130.2),
    (17, "1933-09", "1937-04", 198.6),
    (18, "1944-02", "1947-05", 218.7),
    (19, "1954-04", "1958-03", 285.0),   # the "space-age" giant
    (20, "1964-10", "1968-11", 156.6),
    (21, "1976-03", "1979-12", 232.9),
    (22, "1986-09", "1989-11", 212.5),
    (23, "1996-08", "2001-11", 180.3),
    (24, "2008-12", "2014-04", 116.4),   # the weakest in a century
    (25, "2019-12", "2024-10", 156.0),   # provisional at the 2026 as-of
]


def _ms(ym: str) -> pd.Timestamp:
    """Month-start Timestamp from a 'YYYY-MM' string."""
    return pd.Timestamp(ym + "-01")


def solar_cycles() -> pd.DataFrame:
    """The cycle table as a frame: cycle, min_date, max_date, peak_ssn."""
    df = pd.DataFrame(SOLAR_CYCLES, columns=["cycle", "min", "max", "peak_ssn"])
    df["min_date"] = df["min"].map(_ms)
    df["max_date"] = df["max"].map(_ms)
    return df[["cycle", "min_date", "max_date", "peak_ssn"]]


def turning_points() -> pd.DataFrame:
    """Every solar minimum and maximum as a dated event: ``date``, ``kind``, ``cycle``.

    ``kind`` is ``"min"`` (solar minimum, quiet Sun) or ``"max"`` (solar maximum, active
    Sun). These are the study's independent, non-overlapping "events" — each is ≈ 5–6
    years from its neighbours, so a forward-return window around one never overlaps the
    next. NOTE (honesty, stated once and loudly): a turning point is only *known to be*
    a turning point in hindsight, after ≈ 6–13 months of SILSO smoothing — so the event
    study using these exact dates is a **generous, retrospective** measurement (perfect
    knowledge of the cycle you'd never have live); the tradable timer pays the smoothing
    lag explicitly (see ``strategy.solar_timer``).
    """
    rows = []
    for cyc, mn, mx, _peak in SOLAR_CYCLES:
        rows.append({"date": _ms(mn), "kind": "min", "cycle": cyc})
        rows.append({"date": _ms(mx), "kind": "max", "cycle": cyc})
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Labelled monthly sunspot proxy — a cosine reconstruction pinned to the turning points
# --------------------------------------------------------------------------- #
def sunspot_proxy(index: pd.DatetimeIndex) -> pd.DataFrame:
    """A LABELLED PROXY of the smoothed monthly sunspot number, on ``index``.

    Reconstruction, stated plainly: between a minimum (phase 0) and the next maximum
    (phase π) and on to the following minimum (phase 2π), the solar *phase* advances
    smoothly; the proxy activity is ``(1 - cos φ)/2`` (0 at every minimum, 1 at every
    maximum), and the proxy sunspot number multiplies that by the cycle's peak amplitude.
    This is a stand-in for *where in the 11-year cycle we are*, keyed to the published
    SILSO turning points — it is **not** the raw SILSO monthly file, and is labelled a
    proxy everywhere it is used. Columns: ``phase`` (radians, 0..2π within a cycle),
    ``activity`` (0..1), ``ssn_proxy`` (proxy sunspot number), ``rising`` (bool: activity
    increasing, i.e. min→max half of the cycle).
    """
    tp = []
    for cyc, mn, mx, peak in SOLAR_CYCLES:
        tp.append((_ms(mn), "min", peak))
        tp.append((_ms(mx), "max", peak))
    tp.sort(key=lambda r: r[0])

    dates = pd.DatetimeIndex(index)
    phase = np.full(len(dates), np.nan)
    activity = np.full(len(dates), np.nan)
    ssn = np.full(len(dates), np.nan)
    rising = np.zeros(len(dates), dtype=bool)

    for (d0, k0, p0), (d1, k1, p1) in zip(tp[:-1], tp[1:]):
        mask = (dates >= d0) & (dates < d1)
        if not mask.any():
            continue
        span = (d1 - d0).days
        frac = np.array([(d - d0).days / span for d in dates[mask]])
        if k0 == "min":                     # rising half: phase 0 -> pi
            ph = frac * np.pi
            rise = True
            amp = p1                         # scale by the max we are climbing toward
        else:                               # falling half: phase pi -> 2pi
            ph = np.pi + frac * np.pi
            rise = False
            amp = p0                         # scale by the max we are falling from
        act = (1.0 - np.cos(ph)) / 2.0
        idx = np.where(mask)[0]
        phase[idx] = ph
        activity[idx] = act
        ssn[idx] = amp * act
        rising[idx] = rise

    return pd.DataFrame({"phase": phase, "activity": activity,
                         "ssn_proxy": ssn, "rising": rising}, index=dates)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "1927-01-01", end: str = "2026-07-01") -> None:
    """Download the ^GSPC price-index daily close and cache it. Network; run once.

    ``auto_adjust=False`` on purpose: ^GSPC is a **price index** with no dividend stream
    to reinvest, so there is nothing to "adjust" — the raw close *is* the series, and we
    label it price-only rather than pretend it is total return.
    """
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    d = yf.download(INDEX_TICKER, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d[["Close"]].dropna().to_csv(GSPC_CACHE)


def have_real() -> bool:
    return os.path.exists(GSPC_CACHE)


def load_real(asof: str = AS_OF) -> pd.Series:
    """Cached ^GSPC price-only daily close, sliced to [START, asof]."""
    s = pd.read_csv(GSPC_CACHE, index_col=0, parse_dates=True).sort_index()["Close"]
    return s.loc[(s.index >= START) & (s.index <= asof)]


def monthly_close(close: pd.Series, asof: str = AS_OF) -> pd.Series:
    """Month-end price-only close (last print of each complete calendar month).

    The in-progress final month is dropped by slicing to ``asof`` (the last complete
    month) *before* resampling, so no partial-month bar ever enters the statistics.
    """
    s = close.loc[close.index <= pd.Timestamp(asof)]
    m = s.resample("ME").last().dropna()
    return m


# --------------------------------------------------------------------------- #
# Synthetic world — a monthly tape with a TUNABLE planted solar-cycle component
# --------------------------------------------------------------------------- #
def synthetic_world(amp: float = 0.0, seed: int = 737,
                    start: str = "1928-01", periods: int = 1176,
                    monthly_vol: float = 0.04) -> tuple[pd.Series, pd.DataFrame]:
    """A reproducible monthly "equity-like" tape with a planted solar-cycle return term.

    Monthly log returns are i.i.d. normal (std ``monthly_vol``) PLUS ``amp`` times the
    cosine of the real solar phase — so when ``amp != 0`` the tape genuinely earns more
    in one half of the cycle than the other, and the detector must recover it; when
    ``amp = 0`` the solar phase carries no information and the regime/phase tests must
    NOT reach significance. Returns ``(monthly_close, proxy_frame)`` aligned on a
    month-end index (span ≈ 98 years, far below the pandas ns-timestamp trap).
    """
    idx = pd.date_range(_ms(start), periods=periods, freq="ME")
    prox = sunspot_proxy(idx)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.005, monthly_vol, len(idx))       # ~6%/yr drift + noise
    signal = amp * np.cos(prox["phase"].to_numpy())        # +amp at max, -amp at min
    signal = np.nan_to_num(signal, nan=0.0)
    log_ret = noise + signal
    close = pd.Series(100.0 * np.exp(np.cumsum(log_ret)), index=idx)
    return close, prox
