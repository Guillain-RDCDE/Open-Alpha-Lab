"""Data layer for Study 634 — US-Leads-the-World ("America sneezes").

Two sources, both offline-friendly once cached:

* **Real tape.** Daily **Open + Close** bars (yfinance, no key) for **SPY** (the US leader,
  total-return adjusted close) and four overseas cash indices whose sessions START after the
  NYSE close — **^N225** (Tokyo), **^AXJO** (Sydney), **^GDAXI** (Frankfurt), **^FTSE**
  (London) — 1997 onward. The mechanism under test IS the time zone: by the time Tokyo rings
  its opening bell (~19:00 ET), today's full US session is public information. Cached long
  (``uslw_bars.csv``: date, ticker, open, close).

  *A data honesty note baked in here:* Yahoo's daily **opens** for some cash indices are
  **stale** in stretches of the history (the "open" is just the previous close carried
  forward, a known Yahoo artefact — most severe for ^AXJO/^N225 in the early years). The
  loader exposes a per-day ``live_open`` flag (open differs from the previous close by more
  than a tolerance) so every open-based leg (the overnight-gap vs open→close decomposition,
  the feasible open-entry trade) can be measured on the live-open subsample and the stale
  share is reported out loud.

* **Synthetic.** A deterministic, fixed-seed world generator with a **planted spillover
  knob** ``beta``: US day-t return feeds the foreign day-t+1 overnight gap with coefficient
  ``beta`` (and the foreign intraday leg stays clean noise). ``beta = 0`` is the null — the
  detector must NOT manufacture a slope; a large planted ``beta`` must light up. It is the
  positive control, never market evidence.

Pure numpy + pandas + stdlib for the offline path. ``fetch_panel`` (network) is used only
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
BARS_CACHE = os.path.join(CACHE_DIR, "uslw_bars.csv")

US = "SPY"
# Overseas cash indices whose next session STARTS after the NYSE close (all times ET, winter):
#   ^AXJO Sydney   opens ~18:00 ET  (2h after the US close)
#   ^N225 Tokyo    opens ~19:00 ET  (3h after the US close)
#   ^GDAXI Frankfurt opens ~03:00 ET next morning
#   ^FTSE London   opens ~03:00 ET next morning
FOREIGN = ["^N225", "^GDAXI", "^FTSE", "^AXJO"]
LABELS = {"^N225": "Nikkei 225 (Tokyo)", "^GDAXI": "DAX (Frankfurt)",
          "^FTSE": "FTSE 100 (London)", "^AXJO": "ASX 200 (Sydney)"}

START = "1997-01-01"
AS_OF = "2026-06-30"          # last complete month before the 2026-07-03 run date

# A daily open is considered LIVE (not a stale carry-forward of the previous close) if it
# differs from the previous close by more than this relative tolerance.
STALE_TOL = 1e-6


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = START, end: str | None = None,
                path: str = BARS_CACHE, retries: int = 3) -> pd.DataFrame:
    """Download Open+Close bars for SPY + the four overseas indices and cache them long.

    Network-only; used once to build the cache. SPY uses auto-adjusted (total-return)
    prices; cash indices are price-only levels (they carry no dividend adjustment on
    Yahoo — labeled price-only everywhere they are quoted).
    """
    import yfinance as yf

    rows = []
    for tk in [US] + FOREIGN:
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(tk, start=start, end=end, auto_adjust=True,
                                  progress=False)
                if raw is not None and len(raw) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"no data for {tk}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Open", "Close"]].dropna()
        for dt, r in df.iterrows():
            rows.append({"date": pd.Timestamp(dt).tz_localize(None).normalize(),
                         "ticker": tk, "open": float(r["Open"]), "close": float(r["Close"])})
    bars = (pd.DataFrame(rows).drop_duplicates(["ticker", "date"])
            .sort_values(["ticker", "date"]).reset_index(drop=True))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bars.to_csv(path, index=False)
    return bars


def have_real(path: str = BARS_CACHE) -> bool:
    return os.path.exists(path)


def load_real(path: str = BARS_CACHE, as_of: str = AS_OF) -> pd.DataFrame:
    """Cached long bar frame (date, ticker, open, close), pinned to the as-of date."""
    bars = pd.read_csv(path, parse_dates=["date"])
    return bars[bars["date"] <= pd.Timestamp(as_of)].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Alignment — the core construction
# --------------------------------------------------------------------------- #
def _per_ticker(bars: pd.DataFrame, tk: str) -> pd.DataFrame:
    df = (bars[bars["ticker"] == tk][["date", "open", "close"]]
          .sort_values("date").reset_index(drop=True))
    df["cc"] = df["close"].pct_change()                     # close -> close
    prev_close = df["close"].shift(1)
    df["gap"] = df["open"] / prev_close - 1.0               # prev close -> today's open
    df["oc"] = df["close"] / df["open"] - 1.0               # open -> close (intraday)
    df["live_open"] = (df["open"] - prev_close).abs() > STALE_TOL * prev_close.abs()
    return df.dropna(subset=["cc"]).reset_index(drop=True)


def build_pairs(bars: pd.DataFrame, foreign: str) -> pd.DataFrame:
    """Align each US day t with the FIRST foreign session strictly after date t.

    Returns one row per US day: ``us`` (SPY close-to-close return of day t, known at
    16:00 ET), then the next foreign session's ``cc`` (close-to-close), ``gap``
    (previous foreign close -> open: the overnight repricing that happens before anyone
    can trade), ``oc`` (open -> close: the only leg an open-entry trader can own) and
    ``live_open``. Skips pairs where the next foreign session is more than 7 calendar
    days out (long market closures).

    This is the study's single execution-lag convention: the predictor (US day-t close)
    is fully public BEFORE the target session begins — Tokyo/Sydney open 2-3 hours after
    the US close, Frankfurt/London the next morning. Exactly one lag, by construction.
    """
    us = _per_ticker(bars, US)
    fx = _per_ticker(bars, foreign)
    f_dates = fx["date"].values
    idx = np.searchsorted(f_dates, us["date"].values, side="right")
    ok = idx < len(fx)
    us, idx = us[ok].reset_index(drop=True), idx[ok]
    nxt = fx.iloc[idx].reset_index(drop=True)
    out = pd.DataFrame({
        "us_date": us["date"], "us": us["cc"],
        "f_date": nxt["date"], "cc": nxt["cc"], "gap": nxt["gap"],
        "oc": nxt["oc"], "live_open": nxt["live_open"],
    })
    lag_days = (out["f_date"] - out["us_date"]).dt.days
    out = out[(lag_days >= 1) & (lag_days <= 7)].dropna(subset=["us", "cc"])
    return out.reset_index(drop=True)


def build_all_pairs(bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """{ticker: aligned pair frame} for the four overseas markets."""
    return {tk: build_pairs(bars, tk) for tk in FOREIGN}


def basket_pairs(pairs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight GLOBAL-4 next-session basket, indexed by US day.

    For each US day t, the average of the four markets' next-session close-to-close
    returns (a day enters when at least 3 of the 4 markets have a next session).
    """
    frames = []
    for tk, p in pairs.items():
        frames.append(p.set_index("us_date")[["cc"]].rename(columns={"cc": tk}))
    wide = pd.concat(frames, axis=1)
    us = None
    for tk, p in pairs.items():
        s = p.set_index("us_date")["us"]
        us = s if us is None else us.combine_first(s)
    keep = wide.notna().sum(axis=1) >= 3
    out = pd.DataFrame({"us": us, "basket": wide.mean(axis=1)})[keep]
    return out.dropna().reset_index().rename(columns={"index": "us_date"})


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_world(n_days: int = 6000, beta: float = 0.0, seed: int = 634,
                    sig_us: float = 0.011, sig_gap: float = 0.004,
                    sig_oc: float = 0.011) -> dict[str, pd.DataFrame]:
    """Deterministic synthetic tape with a PLANTED overnight spillover ``beta``.

    US day-t return is i.i.d. noise. Each of two synthetic foreign markets opens the
    next day with an overnight gap = ``beta`` * US_t + noise, then trades an intraday
    leg of clean noise (no planted intraday edge — the spillover lives entirely in the
    gap, mirroring the claimed mechanism). ``beta = 0`` is the null: the same pipeline
    must find nothing. Returns the same ``{ticker: pair frame}`` shape as
    ``build_all_pairs`` so the entire real-tape machinery runs on it unchanged.

    Date span stays well under 250 years (n_days ~ 24y of business days).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2000-01-03", periods=n_days + 1)
    us = rng.normal(0.0003, sig_us, size=n_days)
    out = {}
    for j, tk in enumerate(["SYN_A", "SYN_B"]):
        gap = beta * us + rng.normal(0.0, sig_gap, size=n_days)
        oc = rng.normal(0.0002, sig_oc, size=n_days)
        cc = (1.0 + gap) * (1.0 + oc) - 1.0
        out[tk] = pd.DataFrame({
            "us_date": dates[:-1], "us": us,
            "f_date": dates[1:], "cc": cc, "gap": gap, "oc": oc,
            "live_open": np.ones(n_days, dtype=bool),
        })
    return out
