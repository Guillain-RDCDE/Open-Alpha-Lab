"""Data layer for Study 1013 — It's Official.

Three ingredients, all offline:

1. **The NBER chronology, hard-coded.** Sixteen monthly peak → trough pairs since 1926 and the
   twelve official announcement dates the Business Cycle Dating Committee has issued since it
   began announcing in 1980 (six "a peak occurred", six "a trough occurred"). Source: NBER
   Business Cycle Dating Committee, https://www.nber.org/research/business-cycle-dating, and the
   committee's announcements index. These are facts, not estimates, so they live in code.

2. **Two real tapes from** :mod:`quantlab.bundled` — frozen inside Python packages and SHA-256
   pinned, so a rerun cannot quietly read a different history:

   * ``load_monthly()`` — Fama-French monthly ``mkt`` (**total return**, dividends included) and
     ``rf`` (one-month T-bill), 1926-07 → 2018-11, shipped in ``arch``. Carries the century-long
     lead/lag analysis of fifteen of the sixteen cycles and ten of the twelve announcements.
   * ``load_daily()`` — skfolio's daily S&P 500 **price index** (no dividends), 1990-01-02 →
     2022-12-28. The only tape that reaches the 2020 recession and its two announcements; also
     a daily-resolution re-run of every announcement since 1990. **Price-only** throughout: a
     number from this tape is never labelled total return.

   Both are pinned at ``AS_OF_*`` — the last *complete* month of each tape (the daily tape stops
   on 28 December 2022, so December 2022 is a partial month and is dropped).

3. **A deterministic synthetic world** (``synthetic_world``) with a ``signal_strength`` knob. At
   ``1.0`` it plants, at known sizes, (a) a bear market that *leads* every synthetic NBER peak
   and trough by ``lead_months`` and (b) an abnormal return after every announcement. At ``0.0``
   it is the matched null: i.i.d. returns, the same calendar of dates, no information in either.
   Every strategy test runs on it, and it prices the study's power — how big an effect twelve
   events can see at all.
"""

from __future__ import annotations

import functools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from quantlab import bundled  # noqa: E402

MONTHS_PER_YEAR = 12
TRADING_DAYS_PER_YEAR = 252

# Pinned as-of dates: the last COMPLETE month of each tape.
AS_OF_MONTHLY = "2018-11-30"    # Fama-French (arch) ends 2018-11
AS_OF_DAILY = "2022-11-30"      # skfolio S&P 500 ends 2022-12-28 -> December 2022 is partial
AS_OF = AS_OF_DAILY             # the study-wide as-of (latest pinned date of any tape used)

# --------------------------------------------------------------------------- #
# The NBER chronology — NBER Business Cycle Dating Committee
# https://www.nber.org/research/business-cycle-dating
# --------------------------------------------------------------------------- #
# Monthly reference dates, peak -> trough, every US recession since 1926.
NBER_CYCLES = (
    ("1926-10", "1927-11"),
    ("1929-08", "1933-03"),
    ("1937-05", "1938-06"),
    ("1945-02", "1945-10"),
    ("1948-11", "1949-10"),
    ("1953-07", "1954-05"),
    ("1957-08", "1958-04"),
    ("1960-04", "1961-02"),
    ("1969-12", "1970-11"),
    ("1973-11", "1975-03"),
    ("1980-01", "1980-07"),
    ("1981-07", "1982-11"),
    ("1990-07", "1991-03"),
    ("2001-03", "2001-11"),
    ("2007-12", "2009-06"),
    ("2020-02", "2020-04"),
)

# Official announcement dates (the committee began announcing turning points in 1980).
# (kind, reference month, announcement date)
ANNOUNCEMENTS = (
    ("peak", "1980-01", "1980-06-03"),
    ("trough", "1980-07", "1981-07-08"),
    ("peak", "1981-07", "1982-01-06"),
    ("trough", "1982-11", "1983-07-08"),
    ("peak", "1990-07", "1991-04-25"),
    ("trough", "1991-03", "1992-12-22"),
    ("peak", "2001-03", "2001-11-26"),
    ("trough", "2001-11", "2003-07-17"),
    ("peak", "2007-12", "2008-12-01"),
    ("trough", "2009-06", "2010-09-20"),
    ("peak", "2020-02", "2020-06-08"),
    ("trough", "2020-04", "2021-07-19"),
)


def cycles() -> pd.DataFrame:
    """The sixteen NBER cycles as month-end timestamps, with length in months."""
    rows = []
    for p, t in NBER_CYCLES:
        pe = pd.Timestamp(p) + pd.offsets.MonthEnd(0)
        te = pd.Timestamp(t) + pd.offsets.MonthEnd(0)
        rows.append({"peak": pe, "trough": te,
                     "months": (te.year - pe.year) * 12 + (te.month - pe.month)})
    return pd.DataFrame(rows)


def announcements() -> pd.DataFrame:
    """The twelve official announcements: kind, reference month-end, date, lag in months."""
    rows = []
    for kind, ref, ann in ANNOUNCEMENTS:
        r = pd.Timestamp(ref) + pd.offsets.MonthEnd(0)
        a = pd.Timestamp(ann)
        rows.append({"kind": kind, "reference": r, "date": a,
                     "lag_months": (a.year - r.year) * 12 + (a.month - r.month),
                     "lag_days": int((a - (r - pd.offsets.MonthBegin(1))).days)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Real tapes — quantlab.bundled, offline and pinned
# --------------------------------------------------------------------------- #
def have_monthly() -> bool:
    """True iff the arch Fama-French tape is installed (always true in CI)."""
    return bundled.have_arch("frenchdata")


def have_daily(cache_dir: str = bundled.DEFAULT_CACHE) -> bool:
    """True iff the skfolio S&P 500 tape is in the bundled cache."""
    return os.path.exists(bundled._skfolio_cache("sp500_index", cache_dir))


def have_real() -> bool:
    """True iff BOTH real tapes this study needs are available offline."""
    try:
        return bool(have_monthly() and have_daily())
    except Exception:
        return False


def load_monthly(asof: str = AS_OF_MONTHLY) -> pd.DataFrame:
    """Fama-French monthly, decimal: ``mkt`` (**total return**), ``rf`` (T-bill), ``mkt_rf``.

    Month-end index, 1926-07-31 → ``asof``. ``mkt`` includes dividends; this is the tape every
    total-return number in the study comes from.
    """
    ff = bundled.ff_monthly_total_return()[["mkt", "rf", "mkt_rf"]]
    return ff[ff.index <= pd.Timestamp(asof)].copy()


def load_daily(asof: str = AS_OF_DAILY, cache_dir: str = bundled.DEFAULT_CACHE) -> pd.Series:
    """Daily S&P 500 **price index** (no dividends), 1990-01-02 → ``asof``.

    Raises ``FileNotFoundError`` if the skfolio tape is not cached (it never fetches).
    """
    s = bundled.load_skfolio("sp500_index", cache_dir=cache_dir)["SP500"].dropna()
    s = s[s.index <= pd.Timestamp(asof)]
    s.name = "sp500_price"
    return s


def total_return_index(monthly: pd.DataFrame) -> pd.Series:
    """Cumulative total-return index of ``mkt`` (1.0 before the first month)."""
    return (1.0 + monthly["mkt"]).cumprod().rename("tr_index")


def daily_to_monthly(price: pd.Series) -> pd.Series:
    """Month-end closes of a daily price index (last trading day of each month)."""
    m = price.resample("ME").last().dropna()
    return m


def fingerprint(obj) -> str:
    """Short content fingerprint (12 hex chars) via ``quantlab.bundled.fingerprint``."""
    return bundled.fingerprint(obj)


def provenance() -> dict:
    """What the headline run read: package, file, SHA-256 pin and span of each tape."""
    out = {"monthly": {"package": "arch", "file": "frenchdata.csv.gz",
                       "sha256": bundled.ARCH_FILES["frenchdata"][2],
                       "label": "Fama-French Mkt-RF + RF = market TOTAL return, monthly"},
           "daily": {"package": "skfolio 1.8.1", "file": "sp500_index.csv.gz",
                     "sha256": bundled.SKFOLIO_FILES["sp500_index"][1],
                     "label": "S&P 500 PRICE index (no dividends), daily"}}
    return out


# --------------------------------------------------------------------------- #
# Synthetic world — the deterministic offline core
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=32)
def _month_ends(start: str, n: int) -> pd.DatetimeIndex:
    """Cached month-end calendar (building it is the slow part of a synthetic world)."""
    return pd.date_range(start=start, periods=n, freq="ME")


def synthetic_world(
    n_years: int = 92,
    n_cycles: int = 16,
    signal_strength: float = 1.0,
    lead_months: int = 5,
    bear_depth: float = 0.30,
    announce_premium: float = 0.15,
    premium_months: int = 12,
    recession_months=(6, 18),
    peak_lag=(5, 11),
    trough_lag=(9, 21),
    mu: float = 0.07,
    vol: float = 0.16,
    rf: float = 0.03,
    start: str = "1926-07-31",
    seed: int = 1013,
) -> dict:
    """A monthly market with an NBER-style calendar and (optionally) planted structure.

    The calendar is always built: ``n_cycles`` recessions placed one per equal segment of the
    sample, each ``recession_months`` long, with a peak announcement ``peak_lag`` months after
    the peak and a trough announcement ``trough_lag`` months after the trough (uniform integer
    draws). Announcement dates fall on the 15th of their month, so the date → bar mapping is
    exercised exactly as on the real tape.

    ``signal_strength`` scales the two planted effects:

    * **Lead.** A bear market whose own peak and trough sit ``round(lead_months × s)`` months
      before the NBER peak and trough; it falls by ``bear_depth × s`` (log-equivalent spread
      evenly) and recovers the same amount over an equal number of months afterwards.
    * **Announcement premium.** An abnormal ``announce_premium × s`` log return, spread evenly
      over the ``premium_months`` bars that follow every announcement bar — the "buy signal"
      the claim asserts. One-lag convention: announced during bar *t*, earned from *t+1*.

    At ``s = 0`` both vanish and the world is i.i.d. returns plus a calendar that carries no
    information — the matched null. Returns a dict with ``returns`` (``mkt``, ``rf`` monthly,
    decimal), ``cycles`` and ``announcements`` frames shaped like the real ones, and ``truth``.
    """
    s = float(signal_strength)
    rng = np.random.default_rng(seed)
    n = int(n_years * MONTHS_PER_YEAR)
    idx = _month_ends(start, n)
    mu_m = mu / MONTHS_PER_YEAR
    sd_m = vol / np.sqrt(MONTHS_PER_YEAR)
    rf_m = rf / MONTHS_PER_YEAR
    logr = rng.normal(mu_m - 0.5 * sd_m ** 2, sd_m, n)

    lead = int(round(lead_months * s))
    seg = n // n_cycles
    cyc_rows, ann_rows = [], []
    abn = np.zeros(n)
    for c in range(n_cycles):
        lo = c * seg
        length = int(rng.integers(recession_months[0], recession_months[1] + 1))
        pk_lag = int(rng.integers(peak_lag[0], peak_lag[1] + 1))
        tr_lag = int(rng.integers(trough_lag[0], trough_lag[1] + 1))
        room = seg - length - tr_lag - premium_months - 2
        first = lo + max(lead_months + 2, 2)
        p = int(rng.integers(first, max(first + 1, lo + room)))
        t = p + length
        pa, ta = p + pk_lag, t + tr_lag
        if ta + premium_months >= n:
            break
        cyc_rows.append({"peak": idx[p], "trough": idx[t], "months": length})
        ann_rows.append({"kind": "peak", "reference": idx[p],
                         "date": idx[pa] - pd.offsets.MonthBegin(1) + pd.Timedelta(days=14),
                         "lag_months": pk_lag})
        ann_rows.append({"kind": "trough", "reference": idx[t],
                         "date": idx[ta] - pd.offsets.MonthBegin(1) + pd.Timedelta(days=14),
                         "lag_months": tr_lag})
        if s > 0:
            mp, mt = p - lead, t - lead
            span = max(mt - mp, 1)
            fall = np.log(max(1.0 - bear_depth * s, 1e-6))
            abn[mp + 1: mp + 1 + span] += fall / span
            abn[mt + 1: mt + 1 + span] -= fall / span
            for a in (pa, ta):
                abn[a + 1: a + 1 + premium_months] += announce_premium * s / premium_months
    logr = logr + abn
    mkt = np.exp(logr) - 1.0
    returns = pd.DataFrame({"mkt": mkt, "rf": np.full(n, rf_m)},
                           index=pd.DatetimeIndex(idx, name="date"))
    returns["mkt_rf"] = returns["mkt"] - returns["rf"]
    cyc = pd.DataFrame(cyc_rows)
    ann = pd.DataFrame(ann_rows)
    truth = {"signal_strength": s, "lead_months": lead, "bear_depth": bear_depth * s,
             "announce_premium": announce_premium * s, "premium_months": premium_months,
             "n_cycles": len(cyc), "n_announcements": len(ann), "seed": seed,
             "vol": vol, "mu": mu, "rf": rf}
    return {"returns": returns, "cycles": cyc, "announcements": ann, "truth": truth}
