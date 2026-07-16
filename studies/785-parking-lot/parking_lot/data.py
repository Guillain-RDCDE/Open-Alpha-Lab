"""Data layer for Study 785 — Parking-Lot (shape C: labelled-proxy signal -> forward return).

The claim under test: **satellite parking-lot counts beat the earnings print.** The pitch,
sold hard by the alt-data industry (Orbital Insight, RS Metrics, Probe / Thinknum et al.) since
the mid-2010s, is that counting cars in a big-box retailer's parking lots from orbit gives you
a real-time read on foot traffic — so you can *nowcast* the quarter's sales and front-run the
earnings print. The canonical poster child is **Walmart (WMT)**: thousands of Supercenters, huge
lots, easy to image. The folklore: if satellites saw the lots getting *busier* year-over-year,
the quarter beat and the stock drifts UP after the print; if lots were *emptier*, it misses and
drifts DOWN. A "long busy-quarter, short slow-quarter" timing rule around each WMT earnings date
should therefore pay.

Four ingredients:

* **THE PARKING SIGNAL IS A LABELLED PROXY — NOT REAL SATELLITE DATA, NOT A LIVE FEED.**
  Real orbital car-count panels (Orbital Insight / RS Metrics / Prellis-style products) are
  **paywalled, license-restricted, and not redistributable** — there is no free, backfilled,
  look-back-consistent public parking-count series for WMT. So we do **not** fabricate one and
  pass it off as a tape. ``_parking_proxy_levels()`` below is a **hardcoded, DETERMINISTIC,
  LABELLED PROXY**: a stylised quarterly foot-traffic index (2009->2025) generated from a
  transparent, seeded rule that encodes the *publicly-known qualitative shape* of WMT store
  traffic — a mild secular uptrend, the 2020 COVID stock-up surge (WMT was an essential
  retailer), the 2021-2022 post-COVID / inflation normalisation, the 2015 traffic soft-patch —
  plus reproducible noise. It is used **ordinally only**: the SIGN of the year-over-year change
  ("were the lots busier or emptier than a year ago?"), never a number to trade or present as a
  real feed. See the loud banner on the generator below.

* **The tradable instrument (yfinance).** ``WMT`` (Walmart Inc.), benchmarked against ``SPY``
  (S&P 500, total return) so the test measures WMT's *abnormal* forward return, not the market's.

* **The earnings anchors are STYLISED, not claimed exact-to-the-day.** WMT reports on a
  publicly-known quarterly cadence — roughly mid/late February (fiscal Q4/full year), mid-May
  (Q1), mid-August (Q2), mid-November (Q3). We anchor each quarter on a fixed stylised reporting
  date and snap it to the last available session on/before it. These are **approximate reporting
  windows** (a calendar every trader knows years ahead — hence zero look-ahead), NOT a claim of
  the precise release timestamp. A print landing a few days off its stylised anchor only *smears*
  the reaction and *weakens* the test — which is the honest, conservative direction.

* **Synthetic world.** A deterministic, seeded paired (asset, benchmark) log-return world
  carrying a two-sided synthetic parking signal, with a TUNABLE planted "busy-quarter ->
  forward-up / slow-quarter -> forward-down" link. ``bump = 0`` is the null world (parking
  carries no forward information); the one-sample-t machinery must not manufacture significance.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build the
cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

AS_OF = "2026-06-30"     # last complete month at publication
INSTRUMENT = "WMT"       # Walmart Inc.
BENCHMARK = "SPY"        # S&P 500 total-return proxy

# The publicly-known WMT reporting cadence, as stylised anchors (month, day). WMT reports
# ~mid/late Feb (Q4/FY), ~mid May (Q1), ~mid Aug (Q2), ~mid Nov (Q3). These are APPROXIMATE
# reporting windows (known years ahead -> zero look-ahead), snapped on the tape to the last
# session on/before the stylised date. They are NOT a claim of the exact release date.
REPORT_TAGS = [("Feb", 2, 20), ("May", 5, 16), ("Aug", 8, 16), ("Nov", 11, 15)]
WARMUP_YEAR = 2009                       # year -1 for the first year's YoY (no events emitted)
EVENT_YEARS = list(range(2010, 2026))    # 2010..2025 inclusive -> 16 years x 4 = 64 quarters


def all_tickers() -> list[str]:
    return [INSTRUMENT, BENCHMARK]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"parking_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# LABELLED PROXY — stylised quarterly WMT foot-traffic ("parking count") index.
#
# ⚠️  THIS IS NOT REAL SATELLITE DATA, NOT A SCRAPED FEED, NOT A PRICE TAPE.  ⚠️
# It is a hardcoded, DETERMINISTIC, seeded stand-in that reproduces the publicly-known SHAPE of
# WMT store traffic. It is used ONLY as an ORDINAL signal: the SIGN of the year-over-year change
# (busier vs emptier lots), never a number to trade or to present as a real feed. Real orbital
# car-count panels (Orbital Insight, RS Metrics, ...) are paywalled and not redistributable, so
# no genuine series is fabricated here. The named shocks below (2020 COVID stock-up surge; the
# 2021-2022 normalisation/inflation soft-patch; the 2015 traffic dip) are widely reported; the
# exact per-quarter decimals are a STYLISED proxy, not audited data.
# --------------------------------------------------------------------------- #
# Named, publicly-reported traffic shocks, keyed (year, tag), added to the secular trend before
# reproducible noise. Positive = busier lots than trend; negative = emptier.
_TRAFFIC_SHOCKS = {
    (2015, "May"): -2.5, (2015, "Aug"): -2.0,                       # 2015 WMT traffic soft-patch
    (2020, "Feb"): +1.0, (2020, "May"): +9.0, (2020, "Aug"): +7.0,  # COVID essential-retailer surge
    (2020, "Nov"): +4.0,
    (2021, "Feb"): +3.0, (2021, "May"): -1.0,                       # elevated, then easing
    (2022, "May"): -3.5, (2022, "Aug"): -3.0, (2022, "Nov"): -1.5,  # inflation / normalisation dip
    (2023, "Feb"): +1.0, (2024, "Nov"): +1.0,
    (2017, "Nov"): +1.0, (2018, "Nov"): +1.5, (2019, "Nov"): +1.5,  # holiday-quarter strength
}


def _parking_proxy_levels(seed: int = 810) -> dict[tuple[int, str], float]:
    """LABELLED PROXY: {(year, tag): stylised foot-traffic level}, 2009->2025.

    Deterministic & reproducible: level = 100 + 0.4*(year-2009) [mild secular uptrend]
    + named public shock [COVID surge, 2015/2022 dips, holiday strength] + N(0, 1.0) noise.
    The seasonal component is intentionally omitted because the signal is used as a
    same-quarter YoY change (Feb vs prior Feb, ...), which differences seasonality out. Used
    ORDINALLY only — the sign of the YoY change, never the number itself.
    """
    rng = np.random.default_rng(seed)
    levels: dict[tuple[int, str], float] = {}
    for y in range(WARMUP_YEAR, EVENT_YEARS[-1] + 1):
        for tag, _m, _d in REPORT_TAGS:
            trend = 0.4 * (y - WARMUP_YEAR)
            shock = _TRAFFIC_SHOCKS.get((y, tag), 0.0)
            noise = rng.normal(0.0, 1.0)
            levels[(y, tag)] = 100.0 + trend + shock + noise
    return levels


def parking_events() -> pd.DataFrame:
    """One row per (year>=2010, tag): the LABELLED-PROXY parking level, its same-quarter
    year-over-year change, and the ordinal direction.

    ``yoy = level[(Y, tag)] - level[(Y-1, tag)]``; direction is 'busy' (yoy>0), 'slow' (yoy<0),
    or 'flat'. ``anchor`` is the stylised WMT reporting date for that quarter (snapped to the
    tape downstream). Used ordinally — only the SIGN of ``yoy`` matters.
    """
    levels = _parking_proxy_levels()
    rows = []
    for y in EVENT_YEARS:
        for tag, m, d in REPORT_TAGS:
            lvl = levels[(y, tag)]
            prev = levels[(y - 1, tag)]
            yoy = float(lvl - prev)
            direction = "busy" if yoy > 1e-9 else ("slow" if yoy < -1e-9 else "flat")
            rows.append(dict(year=y, tag=tag, anchor=f"{y:04d}-{m:02d}-{d:02d}",
                             level=round(lvl, 3), prev=round(prev, 3), yoy=round(yoy, 3),
                             direction=direction))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2008-01-01", end: str = "2026-07-01", retries: int = 4) -> None:
    """Download adjusted (total-return) daily closes for WMT + SPY; cache them.

    Retries with linear backoff — Yahoo rate-limits transient bursts, so a first empty
    frame is usually cured by a short wait rather than a real "no such ticker".
    """
    import time

    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in all_tickers():
        last_err = None
        for attempt in range(retries):
            try:
                d = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                d = d[["Close"]].dropna()
                if len(d) > 0:
                    d.to_csv(_cache_path(t))
                    break
                last_err = f"empty frame for {t}"
            except Exception as e:  # noqa: BLE001 -- transient network/rate-limit
                last_err = str(e)
            time.sleep(2.0 * (attempt + 1))
        else:
            raise RuntimeError(f"fetch failed for {t} after {retries} tries: {last_err}")


def have_real() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in all_tickers())


def load_real(asof: str = AS_OF) -> dict[str, pd.Series]:
    """Cached {ticker: adjusted-close Series}, each sliced to <= asof."""
    out = {}
    for t in all_tickers():
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True).sort_index()
        s = df["Close"]
        out[t] = s[s.index <= pd.Timestamp(asof)]
    return out


# --------------------------------------------------------------------------- #
# Synthetic world -- planted "busy-quarter -> forward-up, slow-quarter -> forward-down" link
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, seed: int = 810,
                    n_events: int = 64, n_days: int = 5000, spacing: int = 63,
                    ) -> tuple[pd.Series, pd.Series, pd.Series, list[int]]:
    """Deterministic paired (asset, benchmark) log-return world carrying a TWO-SIDED synthetic
    parking signal, with a planted "busy -> forward up / slow -> forward down" link.

    Both series are correlated (rho ~ 0.6, like a single large-cap name vs SPY) zero-mean noise.
    At each synthetic "print" position (every ``spacing``-th business day) a latent direction
    ``dir in {-1, +1}`` is drawn (the satellite verdict: busier / emptier lots). The asset then
    gets an EXTRA ``bump * dir`` spread across the FORWARD window (the next ``fwd_k`` sessions) —
    a planted "parking predicts the post-print drift." ``bump = 0`` is the null world (parking
    carries no forward information).

    Business-day integer index (positions 0..n_days). Returns
    (asset_logret, bench_logret, direction_signal, print_positions), where ``direction_signal``
    holds +1/-1 at each print position (0 elsewhere).
    """
    rng = np.random.default_rng(seed)
    rho = 0.6
    common = rng.normal(0.0, 0.011, n_days)
    idio_a = rng.normal(0.0, 0.013, n_days)
    idio_b = rng.normal(0.0, 0.008, n_days)
    a = rho * common + np.sqrt(1 - rho**2) * idio_a
    b = rho * common + np.sqrt(1 - rho**2) * idio_b

    signal = np.zeros(n_days)
    print_pos = list(range(spacing, n_days - 260, spacing))[:n_events]
    fwd_k = 21
    for p in print_pos:
        d = 1 if rng.random() < 0.5 else -1
        signal[p] = d
        for j in range(1, fwd_k + 1):     # spread the planted drift across the forward window
            if p + j < n_days:
                a[p + j] += bump * d / fwd_k

    idx = pd.RangeIndex(n_days)
    return (pd.Series(a, index=idx), pd.Series(b, index=idx),
            pd.Series(signal, index=idx), print_pos)
