"""Data layer for Study 919 — Methodology Shock (index rule changes as events).

Three inputs, one shape (a date-indexed daily total-return close frame plus an event
calendar):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the four index wrappers this study races
  (QQQ, SPY, IWM, MDY) and a cash proxy (BIL, the 1-3 month T-bill ETF). ``fetch``
  touches the network and writes parquet into the **shared** cache at
  ``studies/_cache`` (``prices_<TICKER>_1d.parquet``, retry up to 4x); ``load_prices``
  reads that cache **offline** and never imports yfinance. The whole test-suite runs
  with NO cache present (synthetic only), so CI is green on a fresh checkout.

- ``EVENTS`` — a **hardcoded, hand-assembled event calendar** of index methodology
  changes: the announcement date, the effective date, the affected wrapper, and its
  unaffected sibling. This is the study's one non-tape input and is labelled a
  **PROXY/ASSUMPTION** everywhere: the dates come from public index-provider press
  releases, not from a machine-readable feed, and a few carry a ``date_confidence``
  of ``"approximate"``. That uncertainty is not hand-waved — every window in
  ``strategy`` is swept from +/-1 to +/-10 trading days and a drop-one jackknife is
  reported, so a two- or three-day date error cannot manufacture (or hide) a result.

- ``synthetic_daily`` / ``synthetic_panel`` — *deterministic, offline* generators. A
  treated/control price pair with a **planted abnormal drift** injected into the
  post-announcement window of each planted event. The ``signal_strength`` knob scales
  the planted effect: at ``signal_strength=0`` there is nothing to find (the null, on
  which the detector must stay quiet); at ``signal_strength=1`` there is a large,
  unmistakable methodology shock the event study must recover.

Nothing here peeks: the announcement is observed at the close of day 0 and the trade
is put on at day 1, so the day-0 return (which contains the news) is never earned.
That discipline lives in ``strategy.py``.
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

# The wrappers raced here. QQQ tracks the Nasdaq-100; SPY the S&P 500; IWM the Russell
# 2000; MDY the S&P MidCap 400. BIL is the cash leg for the deployed-capital race.
TICKERS = ("QQQ", "SPY", "IWM", "MDY", "BIL")

# Study-wide as-of: the last COMPLETE calendar month at build time.
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# The event calendar — a hardcoded PROXY, not a data feed
# --------------------------------------------------------------------------- #
# Each row: a publicly announced change to how an index is *constructed* (not a routine
# constituent add/drop), the wrapper it hits, and a sibling wrapper the same change does
# NOT touch. ``announce`` is the date the rule change became public; ``effective`` is the
# date the rebalanced/refiltered index went live.
#
# PROXY / ASSUMPTION. Dates were transcribed by hand from index-provider announcements.
# ``date_confidence`` flags how tightly each one is pinned:
#   "exact"       — a dated press release / rebalance date we are confident in;
#   "approximate" — the announcement fell within a few sessions of the date given.
# See docs/references.md for the provenance of each row, and note that the headline
# windows (+/-5, +/-10 trading days) are far wider than the residual date uncertainty.
EVENTS = (
    dict(
        event_id="ndx-2011-special-rebalance",
        label="Nasdaq-100 special rebalance (Apple weight cut ~20% -> ~12%)",
        treated="QQQ", control="SPY",
        # NASDAQ OMX's press release announcing the special rebalance is dated
        # 2011-04-05 (widely reported the same day); the rebalanced index went live
        # on 2011-05-02. An earlier draft of this calendar carried 2011-03-24, which
        # was wrong -- see docs/references.md for the provenance note.
        announce="2011-04-05", effective="2011-05-02",
        date_confidence="exact",
    ),
    dict(
        event_id="ndx-2023-special-rebalance",
        label="Nasdaq-100 special rebalance (the 'Magnificent Seven' concentration fix)",
        treated="QQQ", control="SPY",
        announce="2023-07-07", effective="2023-07-24",
        date_confidence="exact",
    ),
    dict(
        event_id="spx-float-adjustment-phase-1",
        label="S&P 500 float adjustment, phase 1 (half-float weights)",
        treated="SPY", control="IWM",
        announce="2004-03-01", effective="2005-03-18",
        date_confidence="approximate",
    ),
    # Phase 2 was covered by the SAME March-2004 announcement as phase 1, so its
    # announce leg is None: counting one press release twice would fake an extra
    # independent observation out of a single piece of news. It still contributes a
    # genuine, separate observation to the *effective* leg.
    dict(
        event_id="spx-float-adjustment-phase-2",
        label="S&P 500 float adjustment, phase 2 (full-float weights)",
        treated="SPY", control="IWM",
        announce=None, effective="2005-09-16",
        date_confidence="approximate",
    ),
    dict(
        event_id="spx-multiple-share-class-ban",
        label="S&P 500 eligibility: multiple-share-class companies excluded",
        treated="SPY", control="IWM",
        announce="2017-07-31", effective="2017-07-31",
        date_confidence="exact",
    ),
    dict(
        event_id="spx-multiple-share-class-reversal",
        label="S&P 500 eligibility: multiple-share-class exclusion reversed",
        treated="SPY", control="IWM",
        announce="2023-04-17", effective="2023-04-17",
        date_confidence="approximate",
    ),
    dict(
        event_id="russell-banding",
        label="Russell US indexes: banding introduced at reconstitution",
        treated="IWM", control="MDY",
        announce="2007-04-13", effective="2007-06-22",
        date_confidence="approximate",
    ),
    dict(
        event_id="russell-semiannual-reconstitution",
        label="Russell US indexes: move to semi-annual reconstitution announced",
        treated="IWM", control="MDY",
        announce="2023-09-14", effective="2026-11-20",
        date_confidence="approximate",
    ),
)


def leg_date(event: dict, leg: str = "announce"):
    """The event's date for ``leg``, or ``None`` when that leg does not apply.

    A ``None`` announce date means the change shared an earlier announcement (see the
    float-adjustment phases) and must not be counted as a second independent observation.
    """
    v = event.get(leg)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    ts = pd.Timestamp(v)
    return None if pd.isna(ts) else ts


def events_frame(events=EVENTS) -> pd.DataFrame:
    """The hardcoded event calendar as a frame (dates parsed, sorted by effective date)."""
    df = pd.DataFrame(list(events))
    df["announce"] = pd.to_datetime(df["announce"])
    df["effective"] = pd.to_datetime(df["effective"])
    return df.sort_values("effective").reset_index(drop=True)


def usable_events(events=EVENTS, asof: str = AS_OF, leg: str = "announce") -> list:
    """Events with a usable ``leg`` date on or before ``asof`` (never slice the future).

    Two rows are trimmed by this: the semi-annual-reconstitution row has an *effective*
    date in November 2026, beyond the study's as-of, so its effective leg is dropped while
    its announcement leg stays; and float-adjustment phase 2 has no announcement of its
    own, so it appears on the effective leg only.
    """
    cut = pd.Timestamp(asof)
    out = []
    for e in events:
        d = leg_date(e, leg)
        if d is not None and d <= cut:
            out.append(e)
    return out


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "1993-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` so the
    ``close`` column is split- and dividend-adjusted **total return** — the wrappers here
    all distribute, and a price-only comparison of QQQ (low yield) against SPY (higher
    yield) would bias every relative return by roughly a percent a year.
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
        df.to_parquet(_cache_path(tk, cache_dir))
        out[tk] = df
    return out


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every ticker's parquet is present in the shared cache (offline-testable)."""
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_prices(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
) -> pd.DataFrame:
    """Read cached daily total-return closes OFFLINE into one aligned close frame.

    Returns a frame indexed by date with one column per ticker, sliced to ``asof`` so the
    sample never creeps. Raises ``FileNotFoundError`` if any ticker is missing — the
    offline core and the test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call methodology_shock.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df = df[df.index <= pd.Timestamp(asof)]
    return df


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted methodology shock)
# --------------------------------------------------------------------------- #
def synthetic_daily(
    n_years: int = 20,
    n_events: int = 8,
    beta: float = 1.15,
    vol_control_ann: float = 0.17,
    vol_idio_ann: float = 0.11,
    drift_ann: float = 0.07,
    shock_bps_total: float = 250.0,   # planted abnormal move, in bps, over the shock window
    shock_days: int = 10,             # trading days after the announcement it bleeds in over
    signal_strength: float = 1.0,     # 0 = pure null, 1 = the full planted shock
    start: str = "2000-01-03",
    seed: int = 919,
    cash_rate_ann: float = 0.02,
) -> tuple:
    """One treated/control price pair with ``n_events`` planted methodology shocks.

    The control follows a plain geometric random walk. The treated wrapper is generated
    from the market model ``r_treated = beta * r_control + idio``, and — for the
    ``shock_days`` sessions **after** each planted announcement — carries an extra
    abnormal drift totalling ``signal_strength * shock_bps_total`` basis points.

    - ``signal_strength = 1`` → a large, detectable shock the event study must recover.
    - ``signal_strength = 0`` → nothing planted at all (the null); the same machinery must
      return a pooled CAR indistinguishable from its own placebo distribution.

    Returns ``(prices, events, truth)``: ``prices`` has columns ``treated``, ``control``
    and ``cash``; ``events`` is a list of event dicts in the same shape as ``EVENTS``;
    ``truth`` records the planted parameters. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: a few thousand business days stays far inside pandas' ns Timestamp range.
    dates = pd.bdate_range(start=start, periods=n_days)

    s_c = vol_control_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    s_i = vol_idio_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    mu = drift_ann / TRADING_DAYS_PER_YEAR

    r_control = rng.normal(mu, s_c, n_days)
    idio = rng.normal(0.0, s_i, n_days)
    r_treated = beta * r_control + idio

    # Plant the events on an even grid, with room for the 250-day estimation window and
    # the post-event window at both ends.
    lo, hi = 320, n_days - 40
    ev_idx = np.linspace(lo, hi, n_events).astype(int)
    per_day = (signal_strength * shock_bps_total * 1e-4) / max(shock_days, 1)
    for i in ev_idx:
        end = min(i + shock_days, n_days - 1)
        r_treated[i + 1:end + 1] += per_day

    control = 100.0 * np.exp(np.cumsum(np.log1p(r_control)))
    treated = 100.0 * np.exp(np.cumsum(np.log1p(r_treated)))
    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cash = np.cumprod(np.full(n_days, cash_daily))

    prices = pd.DataFrame(
        {"treated": treated, "control": control, "cash": cash},
        index=pd.DatetimeIndex(dates, name="date"),
    )

    events = []
    for k, i in enumerate(ev_idx):
        d = dates[i]
        events.append(dict(
            event_id=f"synthetic-{k:02d}",
            label=f"planted methodology shock #{k}",
            treated="treated", control="control",
            announce=d.strftime("%Y-%m-%d"),
            effective=dates[min(i + 15, n_days - 1)].strftime("%Y-%m-%d"),
            date_confidence="exact",
        ))

    truth = dict(
        n_years=n_years, n_days=n_days, n_events=int(n_events), beta=beta,
        vol_control_ann=vol_control_ann, vol_idio_ann=vol_idio_ann,
        drift_ann=drift_ann, shock_bps_total=shock_bps_total, shock_days=shock_days,
        signal_strength=signal_strength, seed=seed, cash_rate_ann=cash_rate_ann,
        planted_car_bps=float(signal_strength * shock_bps_total),
        event_positions=[int(i) for i in ev_idx],
    )
    return prices, events, truth


def synthetic_panel(
    n_pairs: int = 3,
    n_events_per_pair: int = 4,
    signal_strength: float = 1.0,
    seed: int = 919,
    **kwargs,
) -> tuple:
    """Several independent treated/control pairs, as the real study has (QQQ/SPY, SPY/IWM …).

    Returns ``(pairs, events, truth)`` where ``pairs`` maps a pair id to its price frame
    and ``events`` is the pooled event list (each event naming its pair). Determinstic
    given ``seed``; the ``signal_strength`` knob has the same meaning as in
    ``synthetic_daily``.
    """
    pairs = {}
    events = []
    truths = []
    for p in range(n_pairs):
        px, evs, truth = synthetic_daily(
            n_events=n_events_per_pair, signal_strength=signal_strength,
            seed=seed + 100 * p, **kwargs
        )
        pid = f"pair{p}"
        pairs[pid] = px
        for e in evs:
            e = dict(e)
            e["event_id"] = f"{pid}-{e['event_id']}"
            e["pair"] = pid
            events.append(e)
        truths.append(truth)
    truth = dict(
        n_pairs=int(n_pairs), n_events=len(events), seed=seed,
        signal_strength=signal_strength,
        planted_car_bps=truths[0]["planted_car_bps"],
        per_pair=truths,
    )
    return pairs, events, truth
