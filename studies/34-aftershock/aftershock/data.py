"""Data for the post-earnings-announcement-drift (PEAD) study — an offline synthetic stock panel where
earnings *surprises predict the following weeks of drift*, and the real hook that would pull genuine
earnings dates + surprises from the tape.

The desk's offline/cache split, in the "pending-fetch" flavour (cf. Study 27 Steamroller, whose real
tape also has no pre-populated cache):

  * :func:`synthetic_pead` — fully **offline, deterministic**. A daily panel of stocks; each name reports
    earnings on a fixed quarterly cadence, and at each event a **standardised surprise** ``s`` (a SUE,
    standardised-unexpected-earnings z-score) is drawn. For the ``drift_strength > 0`` *control*, the
    ~60 trading days after the event carry a small **drift proportional to ``s`` that decays** toward
    zero — surprise predicts post-event return, exactly the Ball-Brown / Bernard-Thomas effect.
    ``drift_strength = 0`` is the **null**: the same surprises are drawn, but they are pure noise — no
    post-event drift. Returns ``(panel, events, truth)``.
  * :func:`fetch_earnings_panel` — the **real hook**. It would pull genuine reported-earnings dates and
    surprises (e.g. ``yfinance``'s ``Ticker.get_earnings_dates`` / ``earnings_history``, which exposes
    *EPS estimate* vs *reported EPS* → a surprise) plus the matching daily prices, build the same
    ``(panel, events)`` shape, cache it, and return it. **Cache-only by default.** Because yfinance only
    surfaces ~6-8 reported quarters per name — far too short a history for a credible PEAD cross-section —
    this returns ``({} )`` in the current sandbox (the cache-miss path), and the synthetic control above
    is the validated offline proof, *exactly* as Steamroller's FRED path does.

Two data choices, stated up front. **The surprise is a standardised unexpected-earnings z-score (SUE)** —
PEAD is a statement about the *standardised* surprise, not the raw EPS miss (Bernard-Thomas 1989), so the
synthetic draws ``s`` as a standard normal and the real hook would standardise (reported − estimate) by
its trailing dispersion. **Total-return (split/dividend-adjusted) daily closes** — drift is a statement
about realised holding-period return.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
TRADING_DAYS = 252


@dataclass(frozen=True)
class PeadTruth:
    """What the synthetic generator baked in, so a test can check the book recovers it."""
    n_stocks: int
    n_bars: int
    drift_strength: float     # size of the post-event drift per unit surprise; 0 == the null
    drift_days: int           # window over which the drift plays out and decays
    quarter_days: int         # trading-day cadence between a name's earnings events

    @property
    def has_drift(self) -> bool:
        return self.drift_strength != 0.0


def synthetic_pead(n_stocks: int = 120, n_bars: int = 252 * 12, drift_strength: float = 0.0005,
                   drift_days: int = 60, quarter_days: int = 63, mkt_vol: float = 0.009,
                   idio: float = 0.013, seed: int = 34
                   ) -> tuple[pd.DataFrame, pd.DataFrame, PeadTruth]:
    """A daily cross-section where **earnings surprises predict the next weeks of drift**, by construction.

    Each stock ``i`` reports on a quarterly cadence (every ``quarter_days`` trading days, with a per-name
    phase offset so events are staggered across the calendar). At each event ``e`` a standardised surprise
    ``s_e ~ N(0, 1)`` is drawn. For the next ``drift_days`` trading days the stock earns an extra daily
    return

        drift_{i,t} = drift_strength * s_e * decay(τ),     τ = days since the event,

    where ``decay(τ) = (1 − τ/drift_days)`` ramps the *cumulative* abnormal return up roughly linearly and
    then flattens — the empirical PEAD shape (a sustained post-event drift that fades over ~a quarter).
    The full return is ``r_{i,t} = β_i · market_t + idio·ε + drift_{i,t}``. With ``drift_strength = 0`` the
    surprises are still drawn (so the event frame is identical) but carry **no** drift — the null.

    Returns ``(panel, events, truth)`` where ``panel`` is a ``dates × ticker`` daily-return frame and
    ``events`` is a long frame ``[date, ticker, surprise]`` (one row per earnings announcement, the signal
    a strategy is allowed to see *on* the event date). Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2012-01-02", periods=n_bars, name="date")
    cols = [f"STK{i:03d}" for i in range(n_stocks)]

    market = 0.0003 + mkt_vol * rng.standard_normal(n_bars)
    betas = np.clip(rng.normal(1.0, 0.25, n_stocks), 0.4, 1.8)
    rets = betas[None, :] * market[:, None] + idio * rng.standard_normal((n_bars, n_stocks))

    # decay weights for the post-event window: linear ramp-down so the *cumulative* drift rises then flattens
    tau = np.arange(drift_days)
    decay = 1.0 - tau / float(drift_days)                       # 1.0 at the event, →0 at the window end

    event_rows = []
    for j in range(n_stocks):
        phase = int(rng.integers(0, quarter_days))             # stagger each name's reporting calendar
        for t0 in range(phase, n_bars, quarter_days):
            s = float(rng.standard_normal())                   # the standardised surprise (SUE)
            event_rows.append((idx[t0], cols[j], s))
            if drift_strength != 0.0:
                hi = min(drift_days, n_bars - t0)
                rets[t0:t0 + hi, j] += drift_strength * s * decay[:hi]

    panel = pd.DataFrame(rets, index=idx, columns=cols)
    events = (pd.DataFrame(event_rows, columns=["date", "ticker", "surprise"])
              .sort_values(["date", "ticker"]).reset_index(drop=True))
    truth = PeadTruth(n_stocks=n_stocks, n_bars=n_bars, drift_strength=drift_strength,
                      drift_days=drift_days, quarter_days=quarter_days)
    return panel, events, truth


# --------------------------------------------------------------------------- #
# Real tape — earnings dates + surprises + prices. Cache-first; the network hook is a documented stub
# because no free source gives a long-enough reported-earnings history (see docs/results.md).
# --------------------------------------------------------------------------- #

def fetch_earnings_panel(symbols: list[str] | None = None, start: str = "2012-01-01",
                         cache_dir: str = DEFAULT_CACHE, fetch: bool = False
                         ) -> dict:
    """Return ``{'panel': DataFrame, 'events': DataFrame}`` of real daily returns + earnings surprises,
    cache-first — or ``{}`` when no earnings-history source is wired (the current sandbox).

    Reads a cached ``aftershock_pead.parquet`` pair if present. Otherwise, only with ``fetch=True``, it
    *would*:

      1. For each symbol, pull reported-earnings dates and the *EPS estimate vs reported EPS* via
         ``yfinance`` (``Ticker.get_earnings_dates`` / ``earnings_history``), forming a surprise and
         standardising it into a SUE.
      2. Download the matching split/dividend-adjusted daily closes and difference to returns.
      3. Assemble the ``(panel, events)`` shape :func:`synthetic_pead` returns, cache it, and return.

    **Why this is a stub in this environment.** A credible PEAD cross-section needs *years* of
    reported-earnings history per name; yfinance only surfaces ~6-8 recent quarters (and no reliable
    long surprise history is available free — the same wall the desk hit for options open-interest). So
    with no cache and no long-history source, this returns ``{}`` and the study's real run is **pending**;
    the synthetic control is the offline proof meanwhile. This mirrors Study 27 (Steamroller)'s cache-miss
    path exactly. The yfinance import stays lazy so the offline core never needs the network.
    """
    cache = os.path.join(cache_dir, "aftershock_pead.parquet")
    events_cache = os.path.join(cache_dir, "aftershock_events.parquet")
    if os.path.exists(cache) and os.path.exists(events_cache):
        panel = pd.read_parquet(cache)
        events = pd.read_parquet(events_cache)
        return {"panel": panel, "events": events}
    if not fetch:
        return {}

    # The real fetch path. In the current sandbox there is no long-enough reported-earnings source, so
    # this collects nothing and returns {} (the documented pending-fetch case). Kept here as the exact
    # hook a contributor would complete once a real earnings-history feed is wired.
    try:
        import yfinance as yf  # noqa: F401  (lazy: only on an explicit fetch)
    except Exception:
        return {}

    if symbols is None:
        return {}

    panels, event_rows = {}, []
    for sym in symbols:
        try:
            tk = yf.Ticker(sym)
            ed = tk.get_earnings_dates(limit=64)               # ~16 yrs IF the feed had it; it does not
            if ed is None or ed.empty or "Surprise(%)" not in ed.columns:
                continue
            hist = tk.history(start=start, auto_adjust=True)["Close"].dropna()
            if hist.empty:
                continue
            panels[sym] = hist.pct_change().dropna()
            sup = ed["Surprise(%)"].dropna()
            z = (sup - sup.mean()) / sup.std(ddof=0) if sup.std(ddof=0) > 0 else sup * 0.0
            for dt, s in z.items():
                event_rows.append((pd.Timestamp(dt).normalize().tz_localize(None), sym, float(s)))
        except Exception:
            continue

    if not panels or not event_rows:
        return {}

    panel = pd.DataFrame(panels).sort_index()
    panel.index.name = "date"
    events = (pd.DataFrame(event_rows, columns=["date", "ticker", "surprise"])
              .sort_values(["date", "ticker"]).reset_index(drop=True))
    os.makedirs(cache_dir, exist_ok=True)
    panel.to_parquet(cache)
    events.to_parquet(events_cache)
    return {"panel": panel, "events": events}
