"""Data layer for Study 918 — Creation Halt.

Three things live here: the real tape, the hardcoded event list, and the synthetic
generators.

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for every capped fund and its uncapped twin,
  written to the **shared** cache at ``studies/_cache`` under the desk convention
  ``prices_<TICKER>_1d.parquet`` (``=`` and ``^`` stripped from the filename).
  ``fetch`` touches the network and retries; ``load_prices`` reads the cache
  **offline** and never imports yfinance. The test-suite runs with no cache at all.

- ``EVENTS`` — the hardcoded, publicly verifiable list of share-issuance suspensions
  (and, for GBTC, a *redemption* suspension, which flips the expected sign). Each
  event names the capped fund, the uncapped instrument that tracks the same thing,
  the announcement date, the resumption date and a confidence tag. The announcement
  dates are firm; several **resumption dates are approximate** and are labelled as
  ASSUMPTIONs in the README and swept in ``strategy.resume_date_sweep``.

- ``synthetic_panel`` / ``synthetic_daily`` — deterministic, offline generators. A
  fund and an uncapped twin driven by the same underlying, with a *planted* premium
  that accretes while issuance is halted and fades when it resumes. The
  ``signal_strength`` knob takes the world from that planted effect (1.0) to a pure
  null where nothing happens on the event dates (0.0).

**Price-only vs total-return.** Every ETF/ETN leg is total-return (``auto_adjust``).
The two futures legs (``NG=F``, ``CL=F``) are **price-only** continuous front-month
prints — they carry no distributions by construction, and they are used only as the
underlying ruler for a commodity fund, never raced for total return.

**Fees are baked into the spread, and are an ASSUMPTION.** A fund's total return is
already net of its management fee, so the fund-minus-ruler spread mechanically contains
``(fee_ruler − fee_fund)/252`` per day *before any halt effect exists*. Where the ruler
is an unfeed spot or futures print (``BTC-USD``, ``NG=F``) that term is one-sided and
does not cancel. ``FEE_ANNUAL`` below carries the published expense ratios so
``strategy.fee_drag_bps`` can net them out; the ratios are hardcoded prospectus figures,
not tape, and are therefore labelled ASSUMPTIONs.
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

# Capped funds, their uncapped twins, the underlying rulers and the cash leg.
TICKERS = (
    "UNG", "USO", "VXX", "OIL", "GBTC", "BITO",   # the capped instruments
    "NGF", "DBO", "BNO", "USL", "VIXY", "BTC-USD",  # the uncapped twins / rulers
    "BIL",                                          # cash leg (excess-of-cash reference)
)

# yfinance symbols for the tickers whose cache filename differs (the '=' is stripped).
YF_SYMBOL = {"NGF": "NG=F", "CLF": "CL=F"}

# Published annual expense ratios, in PERCENT PER YEAR. **ASSUMPTION** — these are
# prospectus/sponsor figures typed in by hand, not measured from the tape, and several
# have changed over the sample (GBTC charged 2.00%/yr for its whole pre-ETF life and
# 1.50%/yr after the conversion; the 2.00% figure is used because it covers the halt
# regime being measured). Spot bitcoin and a front-month futures print carry no fee at
# all, which is exactly why the drag is one-sided for the GBTC and BITO pairs.
FEE_ANNUAL = {
    "UNG": 1.35, "USO": 0.79, "VXX": 0.89, "OIL": 0.75, "GBTC": 2.00, "BITO": 0.95,
    "VIXY": 0.85, "DBO": 0.77, "BNO": 1.00, "USL": 0.88,
    "BTC-USD": 0.00, "NGF": 0.00,
    "BIL": 0.1356,
}


# --------------------------------------------------------------------------- #
# The hardcoded event list
# --------------------------------------------------------------------------- #
# Each row: (key, fund, proxy, halt announcement, resumption, direction, kind, note).
#
# ``direction`` is +1 when the suspension caps *creation* (supply of new shares is
# frozen, so the fund can only get richer than what it holds) and −1 when it caps
# *redemption* (holders cannot take the assets out, so the fund can only get cheaper).
# Signing by direction is what lets the six events be pooled.
#
# ``announcement`` is False for GBTC alone: its suspension is a *standing regime* that
# was in force from the first listed day, so there is no announcement day to study — only
# the regime drift and the day the restriction was lifted.
#
# ``ruler`` records how good the uncapped comparison instrument is. "exact" means it
# tracks the *same* object as the capped fund (VIXY holds the same VIX-futures index as
# VXX; BTC-USD is the very asset GBTC holds), so the spread really is a premium proxy.
# "curve-mismatched" means the ruler holds a different point of a futures curve (DBO and
# USL sit further out the WTI curve than USO; NG=F is the front month a fund rolls out
# of), so the spread mixes premium with roll yield — the dominant term in a dislocation.
# This split turns out to be the whole story.
#
# ``confidence`` is FIRM when both dates are publicly documented to the day, APPROX when the
# resumption date is our best public reading, and SOFT when the "halt" was a capacity
# constraint rather than a formal suspension. SOFT events are the honest weak link and
# are jackknifed out in ``strategy.jackknife``.
EVENTS = (
    dict(
        key="UNG-2009",
        fund="UNG", proxy="NGF",
        halt="2009-07-07", resume="2009-09-28",
        direction=+1, confidence="APPROX", announcement=True, ruler="curve-mismatched",
        kind="creation suspended (registered shares exhausted; CFTC position-limit review)",
        proxy_kind="front-month natural-gas futures print (PRICE-ONLY ruler)",
        note="The textbook case: the fund stopped issuing creation baskets and traded at a "
             "double-digit premium to the gas it held for most of a quarter. The resumption "
             "date is our public reading (ASSUMPTION) and is swept.",
    ),
    dict(
        key="USO-2020",
        fund="USO", proxy="DBO",
        halt="2020-04-21", resume="2020-04-29",
        direction=+1, confidence="APPROX", announcement=True, ruler="curve-mismatched",
        kind="creation suspended (all registered shares used; new registration pending)",
        proxy_kind="uncapped WTI futures ETF (total return)",
        note="Announced the day after WTI settled at −$37.63. The proxy is DBO rather than "
             "CL=F precisely because a negative futures print makes a return ratio "
             "meaningless. Resumption is dated to the new registration becoming effective "
             "(ASSUMPTION) and is swept.",
    ),
    dict(
        key="VXX-2022",
        fund="VXX", proxy="VIXY",
        halt="2022-03-14", resume="2022-08-08",
        direction=+1, confidence="FIRM", announcement=True, ruler="exact",
        kind="issuer suspended further sales and issuances (registered amount exceeded)",
        proxy_kind="uncapped ETF on the same short-dated VIX futures index (total return)",
        note="The cleanest event on the desk: two instruments tracking the *same* index, "
             "one capped for five months, one not.",
    ),
    dict(
        key="OIL-2022",
        fund="OIL", proxy="DBO",
        halt="2022-03-14", resume="2022-08-08",
        direction=+1, confidence="FIRM", announcement=True, ruler="curve-mismatched",
        kind="issuer suspended further sales and issuances (same announcement as VXX)",
        proxy_kind="uncapped WTI futures ETF (total return)",
        note="Same issuer, same announcement, different underlying — so it is NOT an "
             "independent draw. The jackknife shows what the pooled number does without it.",
    ),
    dict(
        key="GBTC-2024",
        fund="GBTC", proxy="BTC-USD",
        halt="2015-05-11", resume="2024-01-11",
        direction=-1, confidence="FIRM", announcement=False, ruler="exact",
        kind="redemption suspended for the trust's whole pre-ETF life (no redemption programme)",
        proxy_kind="spot bitcoin (24/7 print sampled on fund trading days)",
        note="The mirror image: the *redemption* side was frozen, so the price could only "
             "drift to a discount. Direction = −1. The 'resumption' is the ETF conversion, "
             "when creations and redemptions both switched on — that date is FIRM. The "
             "'halt' date is NOT an announcement and is not firm in the same sense: the "
             "redemption programme was already suspended before the shares were publicly "
             "quoted, so 2015-05-11 is simply where the free tape begins. Consequence: "
             "this pair has NO pre-halt control period — its 'outside' window is the "
             "post-conversion era alone (n_out is reported). Its 2.00%/yr fee against an "
             "unfeed spot ruler is also the study's largest fee drag (+0.79 bps/day of a "
             "measured +4.35), and is netted out in the results table.",
    ),
    dict(
        key="BITO-2021",
        fund="BITO", proxy="BTC-USD",
        halt="2021-11-01", resume="2022-01-03",
        direction=+1, confidence="SOFT", announcement=True, ruler="curve-mismatched",
        kind="capacity constraint: CME position limits forced later-dated contracts "
             "(NOT a formal creation suspension)",
        proxy_kind="spot bitcoin (24/7 print sampled on fund trading days)",
        note="Included because the brief asks for it, and flagged SOFT: no issuance was "
             "ever formally suspended. Both dates are ASSUMPTIONs. Treat this event as a "
             "placebo-grade observation, not evidence.",
    ),
)

EVENT_KEYS = tuple(e["key"] for e in EVENTS)


def event(key: str) -> dict:
    """Look up one event row by key (raises ``KeyError`` on an unknown key)."""
    for e in EVENTS:
        if e["key"] == key:
            return dict(e)
    raise KeyError(f"unknown event {key!r}; known: {EVENT_KEYS}")


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily closes, shared cache, offline by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2004-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily closes for ``tickers`` and cache each as parquet in the shared cache.

    ``auto_adjust=True`` so ETF/ETN closes are split- and distribution-adjusted total
    return. The two futures symbols are continuous front-month prints and are
    price-only by nature; they are labelled as such wherever they are used.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    out: dict[str, pd.DataFrame] = {}
    os.makedirs(cache_dir, exist_ok=True)
    for tk in tickers:
        sym = YF_SYMBOL.get(tk, tk)
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(
                    sym, start=start, end=end, interval="1d",
                    auto_adjust=True, progress=False,
                )
                if raw is not None and len(raw) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"yfinance returned no data for {sym}")
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
    """Read cached daily closes OFFLINE into one aligned close frame, sliced to ``asof``.

    Columns are the ticker keys of ``tickers``; the index is the union of trading days,
    with NaN where an instrument was not listed. Raises ``FileNotFoundError`` if any
    ticker is missing — the offline core and the test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call creation_halt.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        s = s[~s.index.duplicated(keep="last")]
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
# Synthetic tape — the deterministic offline core
# --------------------------------------------------------------------------- #
def synthetic_daily(
    n_days: int = 2520,
    halt_start: int = 800,
    halt_len: int = 100,
    premium_per_day_bps: float = 12.0,
    fade_days: int = 15,
    vol_underlying: float = 0.35,
    tracking_vol: float = 0.06,
    signal_strength: float = 1.0,
    start: str = "2010-01-04",
    seed: int = 918,
) -> tuple[pd.DataFrame, dict]:
    """One (fund, proxy) pair with a *planted* creation-halt premium.

    The underlying is a driftless GBM. The uncapped twin tracks it with idiosyncratic
    tracking noise; the capped fund tracks it too, but during the halt window
    ``[halt_start, halt_start + halt_len)`` its price accretes an extra
    ``premium_per_day_bps`` per day, which is then given back linearly over the
    ``fade_days`` sessions after issuance resumes. ``signal_strength`` scales the whole
    planted premium: at 0.0 the two legs differ only by tracking noise (the null).

    Returns ``(frame, truth)`` with columns ``fund``, ``proxy`` and ``premium``
    (the planted log premium level, for inspection only — the detector never sees it).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=int(n_days))

    s_u = vol_underlying / np.sqrt(TRADING_DAYS_PER_YEAR)
    s_t = tracking_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    r_u = rng.normal(0.0, s_u, n_days)

    prem_step = np.zeros(n_days)
    accrete = signal_strength * premium_per_day_bps * 1e-4
    hi = min(halt_start + halt_len, n_days)
    prem_step[halt_start:hi] = accrete
    total = accrete * max(hi - halt_start, 0)
    fade_hi = min(hi + fade_days, n_days)
    if fade_hi > hi and total > 0:
        prem_step[hi:fade_hi] = -total / (fade_hi - hi)

    r_fund = r_u + prem_step + rng.normal(0.0, s_t, n_days)
    r_proxy = r_u + rng.normal(0.0, s_t, n_days)

    frame = pd.DataFrame(
        {
            "fund": 100.0 * np.exp(np.cumsum(r_fund)),
            "proxy": 100.0 * np.exp(np.cumsum(r_proxy)),
            "premium": np.cumsum(prem_step),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "n_days": int(n_days),
        "halt_date": str(dates[halt_start].date()),
        "resume_date": str(dates[min(hi, n_days - 1)].date()),
        "halt_len": int(halt_len),
        "premium_per_day_bps": float(premium_per_day_bps),
        "planted_total_premium": float(total),
        "signal_strength": float(signal_strength),
        "fade_days": int(fade_days),
        "seed": int(seed),
    }
    return frame, truth


def synthetic_panel(
    n_events: int = 6,
    signal_strength: float = 1.0,
    n_days: int = 2520,
    halt_len: int = 100,
    premium_per_day_bps: float = 12.0,
    seed: int = 918,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """A panel of ``n_events`` independent (fund, proxy) pairs, one halt each.

    Mirrors the shape of the real study: several capped funds, each with its own
    uncapped twin and its own halt window. Halt starts are spread deterministically so
    the events do not line up in calendar time. Returns ``(frames, events)`` where
    ``events`` has the same schema as ``EVENTS`` (keys ``fund``/``proxy``/``halt``/
    ``resume``/``direction``) so the same estimator runs on synthetic and real alike.
    """
    frames: dict[str, pd.DataFrame] = {}
    evs: list[dict] = []
    for i in range(int(n_events)):
        halt_start = 500 + 200 * i
        frame, truth = synthetic_daily(
            n_days=n_days, halt_start=halt_start, halt_len=halt_len,
            premium_per_day_bps=premium_per_day_bps,
            signal_strength=signal_strength, seed=seed + 17 * i,
        )
        key = f"SYN{i}"
        frames[key] = frame
        evs.append(dict(
            key=key, fund=key, proxy=key,
            halt=truth["halt_date"], resume=truth["resume_date"],
            direction=+1, confidence="SYNTHETIC", announcement=True, ruler="exact",
            kind="planted creation halt", proxy_kind="planted uncapped twin", note="",
        ))
    return frames, evs
