"""Data layer for Study 929 — Rights Offering (the deep subscription discount).

Three inputs, one shape (a date-indexed daily close frame plus an event list):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the twenty US-listed rights issuers this
  study follows, the market benchmark **SPY** and a cash proxy **BIL**. ``fetch``
  touches the network and writes parquet into the **shared** ``studies/_cache``
  (retry up to 4x); ``load_prices`` reads that cache **offline** and never imports
  yfinance. The whole test-suite runs with NO cache present (synthetic-only), so CI
  is green on a fresh checkout.

- ``RIGHTS_EVENTS`` — the hand-compiled event list. **This is the study's weakest
  input and is labelled a PROXY throughout.** Each row carries one *observed anchor*
  (the announcement date of a rights offering, mostly to month precision) plus two
  *assumed* fields: a coarse subscription-discount band and the standard US rights
  timetable (see ``TIMETABLE``). Both assumptions are swept in ``strategy.py``.

- ``synthetic_panel`` / ``synthetic_daily`` — *deterministic, offline* generators.
  A market factor plus idiosyncratic name returns, with a *planted* rights-offering
  effect (a drop on announcement, drift through the subscription period, a bounce
  after expiry) scaled by ``signal_strength``. At ``signal_strength=0`` there is no
  event effect at all (the null); at 1 the effect is large and the event study must
  recover it. Seeds are fixed → tests are deterministic.

**Adjustment caveat (important).** ``auto_adjust=True`` back-adjusts for splits and
cash dividends. It does **not** adjust for a rights issue: the theoretical ex-rights
price drop stays in the tape as a raw price fall, even though a subscribing holder
was made whole by the value of the right. So a negative abnormal return measured
across the ex-rights date is partly *mechanical dilution*, not news. Announcement and
post-expiry windows are free of that artefact; the ex-rights window is not, and is
labelled as such wherever it is reported.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (an event
announced through the close of day ``t`` is acted on at ``t+1``).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The SHARED desk cache (studies/_cache), not a per-study one.
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# Study-wide as-of: the last COMPLETE calendar month at build time.
AS_OF = "2026-06-30"

# Study-wide start. The shared cache is populated by many studies and some tapes there
# (SPY, BIL) reach back much further than this one needs; pinning the start keeps the
# loaded frame — and therefore the published fingerprint — identical no matter how much
# extra history a neighbouring study has cached.
START = "2005-01-01"

MARKET = "SPY"
CASH = "BIL"

# --------------------------------------------------------------------------- #
# The event list — a hand-compiled PROXY, not a vendor feed
# --------------------------------------------------------------------------- #
# Fields: (ticker, announce, precision, discount_band, kind)
#
#   announce       the anchor date. Rights offerings are announced by press release;
#                  this list was compiled by hand from public announcements and is
#                  accurate to the **month** for every row flagged "month" (the vast
#                  majority). Day-level event studies are therefore NOT credible on
#                  this list — which is why every window used here is wide (>= 10
#                  trading days) and why strategy.py sweeps a +/- 10 calendar-day
#                  jitter on the anchor.
#   precision      "month" — the anchor is the mid-month convention date, the true
#                  announcement lies within about two weeks of it.
#   discount_band  ASSUMPTION. A coarse hand-assigned band for how deep the
#                  subscription price sat below the pre-deal market price:
#                  "deep"     ~ >= 20% below (serial dilutive issuers, small caps),
#                  "moderate" ~ 10-20% below (the typical Gabelli-style CEF deal),
#                  "shallow"  ~ < 10% below.
#                  The band, not a number, is the honest resolution of this input;
#                  strategy.permutation_discount_test asks whether the deep/shallow
#                  split does anything a random relabelling would not.
#   kind           "cef" (closed-end fund), "bdc", "reit", "opco".
#
# Issuer clustering is severe and deliberate: rights offerings in the US are
# dominated by a handful of serial closed-end-fund issuers (Cornerstone, Gabelli,
# Oxford Lane). Any cross-sectional t on this list overstates its evidence, so the
# study reports a leave-one-**issuer**-out jackknife and a calendar-time portfolio
# with a HAC t alongside the naive cross-sectional number.
RIGHTS_EVENTS = (
    # Cornerstone — the archetypal serial, deeply discounted, dilutive CEF issuer.
    ("CLM", "2013-09-16", "month", "deep", "cef"),
    ("CLM", "2015-09-15", "month", "deep", "cef"),
    ("CLM", "2017-09-15", "month", "deep", "cef"),
    ("CLM", "2019-09-16", "month", "deep", "cef"),
    ("CLM", "2021-09-15", "month", "deep", "cef"),
    ("CLM", "2023-09-15", "month", "deep", "cef"),
    ("CRF", "2014-09-15", "month", "deep", "cef"),
    ("CRF", "2016-09-15", "month", "deep", "cef"),
    ("CRF", "2018-09-17", "month", "deep", "cef"),
    ("CRF", "2020-09-15", "month", "deep", "cef"),
    ("CRF", "2022-09-15", "month", "deep", "cef"),
    # Gabelli — the other serial issuer; deals typically priced nearer to market.
    ("GAB", "2017-04-17", "month", "moderate", "cef"),
    ("GAB", "2019-04-15", "month", "moderate", "cef"),
    ("GAB", "2021-04-15", "month", "moderate", "cef"),
    ("GGZ", "2018-04-16", "month", "moderate", "cef"),
    ("GGZ", "2019-10-15", "month", "moderate", "cef"),
    ("GGZ", "2021-05-17", "month", "moderate", "cef"),
    ("GGZ", "2023-04-17", "month", "moderate", "cef"),
    ("GUT", "2016-10-17", "month", "shallow", "cef"),
    ("GUT", "2019-10-15", "month", "shallow", "cef"),
    ("GRX", "2015-10-15", "month", "moderate", "cef"),
    ("GRX", "2021-10-15", "month", "moderate", "cef"),
    ("GDV", "2018-05-15", "month", "moderate", "cef"),
    ("GLU", "2016-03-15", "month", "moderate", "cef"),
    ("GGT", "2017-10-16", "month", "moderate", "cef"),
    # Credit / niche CEFs and BDCs.
    ("OXLC", "2013-06-17", "month", "deep", "cef"),
    ("OXLC", "2014-06-16", "month", "deep", "cef"),
    ("OXLC", "2015-06-15", "month", "deep", "cef"),
    ("OPP", "2020-11-16", "month", "moderate", "cef"),
    ("RIV", "2021-06-15", "month", "moderate", "cef"),
    ("SPE", "2017-09-15", "month", "moderate", "cef"),
    ("SPE", "2020-05-15", "month", "moderate", "cef"),
    ("CET", "2021-10-15", "month", "shallow", "cef"),
    ("SWZ", "2016-05-16", "month", "moderate", "cef"),
    ("JOF", "2013-05-15", "month", "moderate", "cef"),
    ("GECC", "2020-06-15", "month", "deep", "bdc"),
    ("EQS", "2014-05-15", "month", "deep", "bdc"),
    ("RAND", "2021-06-15", "month", "deep", "bdc"),
    ("WHLR", "2020-08-17", "month", "deep", "reit"),
)

# Candidate deals that had to be dropped because no retrievable price history exists
# on Yahoo! Finance (delisted, merged or renamed). Named here so the SURVIVORSHIP of
# the retained list is explicit rather than silent: rights offerings are most common
# among distressed issuers, and distressed issuers are exactly the ones that vanish.
DROPPED_NO_TAPE = ("CUBA", "NHF", "ENZN", "SPLP", "TUEM")

# ASSUMPTION — the standard US rights timetable, in calendar days from the anchor.
# A US rights offering runs roughly: announcement -> record date (about a week) ->
# ex-rights (record + 1) -> a subscription period of three to five weeks -> expiry.
# We do not have per-deal record/expiry dates, so we model them with these offsets
# and sweep them (strategy.timetable_sweep).
TIMETABLE = {"ex_rights_days": 10, "expiry_days": 40}

TICKERS = tuple(sorted({e[0] for e in RIGHTS_EVENTS})) + (MARKET, CASH)


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2005-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` so the
    ``close`` column is split- and dividend-adjusted — essential for closed-end funds,
    which distribute most of their return as (often return-of-capital) dividends. It
    does **not** adjust for the rights issue itself; see the module docstring.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    out = {}
    os.makedirs(cache_dir, exist_ok=True)
    for tk in tickers:
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(
                    tk, start=start, end=end, interval="1d",
                    auto_adjust=True, progress=False,
                )
                if raw is not None and len(raw) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"yfinance returned no data for {tk}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        df = raw[["close"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.dropna(subset=["close"])
        # NON-DESTRUCTIVE: the cache is shared with every other study on the desk and some
        # tapes there (SPY, BIL) start decades before this study needs. Merge rather than
        # overwrite, so a refresh here can never truncate a neighbour's history.
        path = _cache_path(tk, cache_dir)
        if os.path.exists(path):
            old = pd.read_parquet(path)
            old.index = pd.to_datetime(old.index)
            df = pd.concat([old[["close"]], df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df.index.name = "date"
        df.to_parquet(path)
        out[tk] = df
    return out


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every ticker's parquet is present in the shared cache (offline-testable)."""
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_prices(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
    start: str = START,
) -> pd.DataFrame:
    """Read cached daily total-return closes OFFLINE into one aligned close frame.

    Returns a frame indexed by date, one column per ticker, sliced to ``[start, asof]``
    so the sample never creeps — neither forward as new sessions arrive, nor backward
    when a neighbouring study extends a shared tape's history. Raises
    ``FileNotFoundError`` if any ticker is missing; the offline core and the whole
    test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call rights_offering.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(asof))]
    return df


def events_frame(events=RIGHTS_EVENTS, timetable=None) -> pd.DataFrame:
    """The event list as a frame, with the modelled ex-rights and expiry dates attached.

    ``announce`` is the observed (month-precision) anchor; ``ex_rights`` and ``expiry``
    are the anchor plus the ``TIMETABLE`` offsets — an ASSUMPTION, not a fact.
    """
    tt = dict(TIMETABLE if timetable is None else timetable)
    rows = []
    for tk, ann, prec, band, kind in events:
        a = pd.Timestamp(ann)
        rows.append({
            "ticker": tk,
            "announce": a,
            "ex_rights": a + pd.Timedelta(days=int(tt["ex_rights_days"])),
            "expiry": a + pd.Timedelta(days=int(tt["expiry_days"])),
            "precision": prec,
            "discount_band": band,
            "kind": kind,
        })
    return pd.DataFrame(rows)


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted rights-offering effect)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_names: int = 20,
    n_events_per_name: int = 2,
    n_years: int = 14,
    signal_strength: float = 1.0,
    ann_drop: float = -0.05,        # abnormal return on the announcement day
    sub_drift: float = -0.06,       # abnormal drift across the subscription period
    post_bounce: float = 0.05,      # abnormal drift over the 20 days after expiry
    beta: float = 0.9,
    idio_vol_ann: float = 0.28,
    mkt_vol_ann: float = 0.16,
    mkt_drift_ann: float = 0.08,
    start: str = "2010-01-04",
    seed: int = 929,
) -> tuple:
    """A synthetic panel of issuers plus a market index, with planted rights events.

    The planted effect is scaled by ``signal_strength``:

    - ``signal_strength = 1`` → a full-size announcement drop, a negative drift across
      the subscription window and a positive bounce after expiry. The event study must
      recover all three.
    - ``signal_strength = 0`` → **no event effect at all**; the announcement dates are
      arbitrary points in an ordinary return series (the null — the study must stay
      quiet).

    Deep-band names get 1.5x the effect and shallow-band names 0.5x, so the
    discount-band split has something real to find when the signal is on.

    Returns ``(prices, events, truth)``: ``prices`` is a close frame with one column
    per name plus ``SPY`` and ``BIL``; ``events`` is a tuple in ``RIGHTS_EVENTS``
    shape; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    n_days = n_years * TRADING_DAYS_PER_YEAR
    dates = pd.bdate_range(start=start, periods=n_days)  # OOB-safe: modest length

    mkt = rng.normal(mkt_drift_ann / TRADING_DAYS_PER_YEAR,
                     mkt_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR), n_days)
    idio_sd = idio_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)

    names = [f"N{i:02d}" for i in range(n_names)]
    bands = ["deep", "moderate", "shallow"]
    band_scale = {"deep": 1.5, "moderate": 1.0, "shallow": 0.5}

    tt = TIMETABLE
    ex_off = int(round(tt["ex_rights_days"] * 252 / 365))
    exp_off = int(round(tt["expiry_days"] * 252 / 365))

    # Event anchors: keep clear of the estimation warmup and the tail.
    lo, hi = 300, n_days - (exp_off + 90)
    ev_rows = []
    rets = {}
    for i, nm in enumerate(names):
        band = bands[i % 3]
        r = rng.normal(0.0, idio_sd, n_days) + beta * mkt
        anchors = np.sort(rng.choice(np.arange(lo, hi), size=n_events_per_name,
                                     replace=False))
        # keep events from overlapping inside a name
        anchors = [int(a) for a in anchors]
        keep = []
        for a in anchors:
            if not keep or a - keep[-1] > exp_off + 120:
                keep.append(a)
        for a in keep:
            s = signal_strength * band_scale[band]
            # announcement day drop, acted on from the next day by any strategy
            r[a] += s * ann_drop
            # subscription drift, spread evenly over announce+1 .. expiry
            span = slice(a + 1, a + exp_off + 1)
            r[span] += s * sub_drift / max(exp_off, 1)
            # post-expiry bounce over the following 20 sessions
            post = slice(a + exp_off + 1, a + exp_off + 21)
            r[post] += s * post_bounce / 20.0
            ev_rows.append((nm, str(dates[a].date()), "month", band, "cef"))
        rets[nm] = r

    px = {nm: 20.0 * np.exp(np.cumsum(rets[nm])) for nm in names}
    px[MARKET] = 100.0 * np.exp(np.cumsum(mkt))
    px[CASH] = np.cumprod(np.full(n_days, (1.0 + 0.02) ** (1.0 / TRADING_DAYS_PER_YEAR)))
    prices = pd.DataFrame(px, index=pd.DatetimeIndex(dates, name="date"))

    truth = {
        "signal_strength": signal_strength,
        "ann_drop": ann_drop, "sub_drift": sub_drift, "post_bounce": post_bounce,
        "n_names": n_names, "n_events": len(ev_rows), "n_days": n_days,
        "beta": beta, "seed": seed, "ex_off": ex_off, "exp_off": exp_off,
    }
    return prices, tuple(sorted(ev_rows, key=lambda e: (e[1], e[0]))), truth


def synthetic_daily(
    n_years: int = 14,
    signal_strength: float = 1.0,
    seed: int = 929,
) -> tuple:
    """One-name convenience wrapper on :func:`synthetic_panel` (single issuer, 3 deals)."""
    return synthetic_panel(n_names=1, n_events_per_name=3, n_years=n_years,
                           signal_strength=signal_strength, seed=seed)
