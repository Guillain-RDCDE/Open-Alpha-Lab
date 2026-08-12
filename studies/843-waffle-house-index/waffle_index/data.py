"""Data layer for Study 843 — Waffle House Index.

FEMA's informal "Waffle House Index" reads disaster severity off whether the
always-open chain closes: green (full menu) → mild, yellow (limited menu) →
serious, red (closed) → catastrophic. We turn the folklore around and ask the
tradable question: when a **major US natural disaster** lands, does it move the
equities with the most obvious exposure — **property & casualty insurers**
(payout shock) and **home-improvement / rebuild** names (reconstruction demand)?

Three ingredients, all offline-friendly once cached:

* **The disaster table, hardcoded.** ``DISASTERS`` is a curated table of **16
  major US-landfalling hurricanes / natural-disaster events, 2005 → 2024** — the
  storms that dominated the national front page and drove multi-billion-dollar
  insured losses. Each row is a *public-record* landfall date (National Hurricane
  Center advisories / NOAA), the plain-English name, and a rough insured-loss
  tier. No free, machine-readable "major US disaster index" keyed to market days
  exists, so — exactly like the sibling event studies that hand-build a shock
  calendar (``707-plane-crash-effect``'s ``DISASTERS``, ``316-bank-failure``'s
  ``EVENTS``, ``313-geopolitical-shock``'s ``SHOCK_TABLE``) — this is a hand-built
  table of the disasters any reasonable person would call "the market's front page
  that week".

* **Real tape.** Daily total-return closes (``auto_adjust=True``) for **SPY** (the
  market benchmark), the three big listed P&C insurers — **ALL** (Allstate),
  **TRV** (Travelers) and **PGR** (Progressive) — and the two home-improvement
  rebuild names — **HD** (Home Depot) and **LOW** (Lowe's) — all from yfinance
  (no key), cached as CSV under ``_cache/``. Every ticker trades continuously
  across the full 2005 → 2026 window, so the basket has no survivorship holes on
  this calendar (named on the Signal axis: this is the *surviving* set of large-cap
  names, not the full P&C universe including any insurer that went under).

* **Synthetic world.** A deterministic, seeded generator that builds an SPY-like
  market tape plus five single-name tapes, and around planted "disaster" dates
  injects a KNOWN *market-adjusted* drift — insurers down, rebuilders up — that
  decays over a few weeks (``edge`` knob, in daily market-adjusted return units).
  ``edge = 0`` is the pure null: event windows are statistically identical to any
  other window, and the market-adjusted event-study detector must NOT manufacture
  significance from it. A positive ``edge`` is the positive control the harness
  must recover.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once
to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

START = "2004-06-01"          # a run-up before the first (2005) event
AS_OF = "2026-06-30"          # last complete calendar month at publication

BENCHMARK = "SPY"
INSURERS = ("ALL", "TRV", "PGR")          # listed P&C insurers (payout exposure)
REBUILDERS = ("HD", "LOW")                # home-improvement / rebuild demand
TICKERS = (BENCHMARK,) + INSURERS + REBUILDERS

CACHE = {t: os.path.join(CACHE_DIR, f"whi_{t.lower()}.csv") for t in TICKERS}


# --------------------------------------------------------------------------- #
# Hardcoded table of major US natural-disaster (hurricane) landfalls, 2005-2024.
# Each row: (landfall_date, label, loss_tier). ``landfall_date`` is the calendar
# date of US landfall as reported by the National Hurricane Center; the event-study
# code snaps it to the first NYSE session on/after that date via ``searchsorted``
# (a weekend landfall rolls forward to the next open). ``loss_tier`` is a coarse
# public-record insured-loss bucket (3 = >$25bn, 2 = $10-25bn, 1 = <$10bn).
#
# IMPORTANT anticipation caveat (documented, not hidden): a hurricane is FORECAST
# days before landfall, so the market can price it in ahead of day 0. The event
# window therefore spans [-10..+20] and the CAR is read from the pre-storm run-up,
# not only from the landfall session. See strategy.py / docs/results.md.
# --------------------------------------------------------------------------- #
DISASTERS: list[tuple[str, str, int]] = [
    ("2005-08-29", "Hurricane Katrina (LA/MS, New Orleans)", 3),
    ("2005-09-24", "Hurricane Rita (TX/LA)", 1),
    ("2005-10-24", "Hurricane Wilma (South Florida)", 2),
    ("2008-09-13", "Hurricane Ike (Galveston/Houston, TX)", 2),
    ("2012-10-29", "Hurricane Sandy (NJ/NY metro)", 3),
    ("2016-10-08", "Hurricane Matthew (SE US coast)", 1),
    ("2017-08-25", "Hurricane Harvey (Houston, TX)", 3),
    ("2017-09-10", "Hurricane Irma (Florida)", 3),
    ("2018-09-14", "Hurricane Florence (Carolinas)", 1),
    ("2018-10-10", "Hurricane Michael (FL panhandle)", 2),
    ("2020-08-27", "Hurricane Laura (LA/TX)", 1),
    ("2021-08-29", "Hurricane Ida (LA + NE flooding)", 3),
    ("2022-09-28", "Hurricane Ian (SW Florida)", 3),
    ("2023-08-30", "Hurricane Idalia (Florida Big Bend)", 1),
    ("2024-09-26", "Hurricane Helene (FL/SE Appalachia)", 3),
    ("2024-10-09", "Hurricane Milton (Florida)", 2),
]


def disaster_table(asof: str = AS_OF) -> pd.DataFrame:
    """The curated table as a frame: ``date`` (Timestamp), ``label``, ``loss_tier``.

    Optionally truncated to landfalls on/before ``asof`` (default: the publication
    as-of, which keeps all rows).
    """
    df = pd.DataFrame(DISASTERS, columns=["date", "label", "loss_tier"])
    df["date"] = pd.to_datetime(df["date"])
    if asof is not None:
        df = df[df["date"] <= pd.Timestamp(asof)]
    return df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str = "2026-07-01", retries: int = 4) -> None:
    """Download SPY + the insurer & rebuilder tickers (total-return closes); cache them.

    Network; run once. ``auto_adjust=True`` folds splits and dividends into the
    close (total-return, not price-only). Retries up to ``retries`` times per ticker
    against transient yfinance failures.
    """
    import time

    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in TICKERS:
        last = None
        for attempt in range(retries):
            try:
                df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                close = df[["Close"]].dropna()
                if close.empty:
                    raise RuntimeError(f"yfinance returned no bars for {t}")
                close.to_csv(CACHE[t])
                last = None
                break
            except Exception as e:  # transient network / rate-limit
                last = e
                time.sleep(1.5 * (attempt + 1))
        if last is not None:
            raise last


def have_real() -> bool:
    """True iff every ticker's cache CSV is present (offline-safe gate)."""
    return all(os.path.exists(p) for p in CACHE.values())


def load_real(asof: str = AS_OF) -> dict[str, pd.Series]:
    """Cached ``{ticker: close}`` series, each sliced to [START, asof], tz-naive.

    Raises ``FileNotFoundError`` if any ticker's cache is missing, so an offline
    caller can banner the run rather than silently drop a leg.
    """
    if not have_real():
        missing = [t for t, p in CACHE.items() if not os.path.exists(p)]
        raise FileNotFoundError(
            f"No cached tape for {missing}. Call data.fetch() once to populate _cache/."
        )
    out = {}
    for t, path in CACHE.items():
        s = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()["Close"]
        s.index = pd.DatetimeIndex(s.index)
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        s = s.loc[(s.index >= START) & (s.index <= asof)]
        out[t] = s.astype(float).rename(t)
    return out


# --------------------------------------------------------------------------- #
# Synthetic world — planted market-adjusted disaster drift (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(
    edge: float = 0.0,
    seed: int = 843,
    n_days: int = 5200,
    n_events: int = 16,
    daily_vol: float = 0.011,
    idio_vol: float = 0.009,
    decay_days: int = 15,
    start: str = "2005-01-03",
) -> tuple[dict[str, pd.Series], pd.DatetimeIndex]:
    """A reproducible daily world: an SPY-like market + 5 single names + planted drift.

    The market (SPY) is a geometric random walk (i.i.d. normal log returns, std
    ``daily_vol``). Each single name = the market return + its own idiosyncratic
    noise (std ``idio_vol``). Around ``n_events`` scheduled "disaster" dates we inject
    a planted **market-adjusted** drift that decays linearly over ``decay_days``
    sessions: insurers (ALL/TRV/PGR) take ``-edge`` per day (payout hit), rebuilders
    (HD/LOW) take ``+edge`` per day (reconstruction demand). Because the drift is added
    on top of the market component, it survives the market-adjustment
    (``r_i - r_spy``) that the event study applies — that is exactly the abnormal
    move the detector must recover.

    ``edge = 0`` is the null world: every name is just market + idiosyncratic noise,
    event windows statistically identical to any other window, and the detector must
    NOT reach significance. Business-day index, span ~20 years — far below the
    pandas ns-timestamp horizon. Returns (``{ticker: close}``, event-date index).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    mkt = rng.normal(0.0002, daily_vol, n_days)          # market log returns

    margin = max(decay_days + 20, 40)
    pool = np.arange(margin, n_days - margin)
    locs = np.sort(rng.choice(pool, size=min(n_events, pool.size), replace=False))

    sign = {t: -1.0 for t in INSURERS}
    sign.update({t: +1.0 for t in REBUILDERS})

    out: dict[str, pd.Series] = {}
    out[BENCHMARK] = pd.Series(100.0 * np.exp(np.cumsum(mkt)), index=idx, name=BENCHMARK)
    for t in INSURERS + REBUILDERS:
        r = mkt + rng.normal(0.0, idio_vol, n_days)      # market + idiosyncratic
        for loc in locs:                                  # planted decaying drift
            for k in range(0, decay_days):
                if loc + k < n_days:
                    w = 1.0 - k / decay_days              # linear decay
                    r[loc + k] += sign[t] * edge * w
        out[t] = pd.Series(100.0 * np.exp(np.cumsum(r)), index=idx, name=t)
    return out, idx[locs]


def fingerprint(s: pd.Series) -> str:
    """A short content fingerprint of a close series, for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(s.to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]
