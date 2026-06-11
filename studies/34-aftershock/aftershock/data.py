"""Data for the post-earnings-announcement-drift (PEAD) study — an offline synthetic stock panel where
earnings *surprises predict the following weeks of drift*, and the real hook that reads cached EDGAR
earnings + the S&P 500 price panel from the tape.

The desk's offline/cache split:

  * :func:`synthetic_pead` — fully **offline, deterministic**. A daily panel of stocks; each name reports
    earnings on a fixed quarterly cadence, and at each event a **standardised surprise** ``s`` (a SUE,
    standardised-unexpected-earnings z-score) is drawn. For the ``drift_strength > 0`` *control*, the
    ~60 trading days after the event carry a small **drift proportional to ``s`` that decays** toward
    zero — surprise predicts post-event return, exactly the Ball-Brown / Bernard-Thomas effect.
    ``drift_strength = 0`` is the **null**: the same surprises are drawn, but they are pure noise — no
    post-event drift. Returns ``(panel, events, truth)``. The tests and the offline demo run on this.
  * :func:`fetch_earnings_panel` — the **real hook**, now wired to the cache. It reads cached EDGAR
    quarterly EPS (``edgar_eps.parquet``) and the S&P 500 price panel (``panel_503_2010-01-01.parquet``),
    builds a **SUE event frame** (seasonal-random-walk surprise, standardised by its own trailing
    dispersion — see :func:`build_sue_events`) and a daily-returns panel from split/dividend-adjusted
    closes, caches the assembled pair, and returns ``{'panel': DataFrame, 'events': DataFrame}``.
    **Cache-first, offline**; the network is touched only on an explicit ``fetch=True`` cache miss (which
    does not happen here — the EDGAR/panel caches exist).

Three data choices, stated up front.

  * **The surprise is a standardised unexpected-earnings z-score (SUE), seasonal random walk** — PEAD is a
    statement about the *standardised* surprise, not the raw EPS miss (Bernard-Thomas 1989). With no
    analyst estimate available, the expectation is the *year-ago same-quarter* EPS (a seasonal random
    walk): ``surprise_q = eps_q − eps_{q−4}``, standardised by the **trailing** dispersion of that stock's
    own surprise series. The synthetic draws ``s`` as a standard normal directly.
  * **The announcement date is the EARLIEST `filed` date** for each ``(ticker, period_end)``. EDGAR
    filings repeat prior-period comparatives, so the same quarter appears in several filings; the *first*
    time a quarter is filed IS the announcement, so we keep the earliest-filed row per quarter.
  * **Total-return (split/dividend-adjusted) daily closes** — drift is a statement about realised
    holding-period return.

**Survivorship caveat.** The real price panel is *today's* S&P 500 membership, so it carries
**survivorship bias** (names that were delisted or dropped from the index are absent). The qualitative
PEAD result — a small, real drift whose break-even cost sits inside realistic trading costs — is robust to
this, but the precise magnitudes are biased upward by the missing losers, and we say so wherever a real
number appears.
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
# Real tape — cached EDGAR EPS + the S&P 500 price panel. Cache-first, offline; the SUE is a seasonal
# random walk (eps_q - eps_{q-4}) standardised by the stock's own trailing surprise dispersion.
# --------------------------------------------------------------------------- #

EDGAR_CACHE = "edgar_eps.parquet"
PANEL_CACHE = "panel_503_2010-01-01.parquet"
PEAD_PANEL_CACHE = "aftershock_pead.parquet"
PEAD_EVENTS_CACHE = "aftershock_events.parquet"


def _dedupe_announcements(eps: pd.DataFrame) -> pd.DataFrame:
    """Keep the EARLIEST ``filed`` row per ``(ticker, period_end)`` — that first filing IS the
    announcement. EDGAR repeats prior-period comparatives in later filings, so the same quarter shows up
    several times; the earliest filing is the real announcement date."""
    return (eps.sort_values(["ticker", "period_end", "filed"])
            .drop_duplicates(["ticker", "period_end"], keep="first")
            .reset_index(drop=True))


def build_sue_events(eps: pd.DataFrame, std_window: int = 8, min_periods: int = 4) -> pd.DataFrame:
    """Build the SUE event frame from a deduped EDGAR EPS table ``[ticker, period_end, filed, eps]``.

    For each stock, the **seasonal-random-walk surprise** is the year-over-year change in quarterly EPS,
    ``surprise_q = eps_q − eps_{q−4}`` (the classic Bernard-Thomas measure that needs no analyst estimate).
    It is standardised into a SUE by the **trailing** dispersion of that stock's own surprise series —
    ``SUE = surprise_q / rolling_std(surprise[:q])`` — strictly causal (the std uses only surprises
    strictly before ``q``, via a one-step lag). The announcement date is the earliest ``filed`` date.

    Returns a long frame ``[date, ticker, surprise]`` (``surprise`` is the SUE), one row per announcement,
    dropping the first ~year per name (no year-ago comparison) and any non-finite SUE.
    """
    eps = _dedupe_announcements(eps)
    out = []
    for tk, g in eps.groupby("ticker", sort=False):
        g = g.sort_values("period_end")
        surprise = g["eps"] - g["eps"].shift(4)                 # seasonal random walk: YoY EPS change
        trailing_std = surprise.shift(1).rolling(std_window, min_periods=min_periods).std()
        sue = surprise / trailing_std                           # standardise by own trailing dispersion
        out.append(pd.DataFrame({"date": g["filed"].to_numpy(), "ticker": tk, "surprise": sue.to_numpy()}))
    events = pd.concat(out, ignore_index=True)
    events = events[np.isfinite(events["surprise"])]
    return events.sort_values(["date", "ticker"]).reset_index(drop=True)


def _load_returns_panel(cache_dir: str) -> pd.DataFrame:
    """Read the cached S&P 500 price panel (MultiIndex ``(ticker, field)``) and return a ``dates × ticker``
    daily-returns frame from the split/dividend-adjusted Close."""
    p = pd.read_parquet(os.path.join(cache_dir, PANEL_CACHE))
    close = p.xs("Close", axis=1, level=1).sort_index()
    panel = close.pct_change().astype(float)
    panel.index.name = "date"
    return panel


def fetch_earnings_panel(symbols: list[str] | None = None, start: str = "2010-01-01",
                         cache_dir: str = DEFAULT_CACHE, fetch: bool = False,
                         rebuild: bool = False) -> dict:
    """Return ``{'panel': DataFrame, 'events': DataFrame}`` of real daily returns + SUE earnings events,
    **cache-first and offline**.

    Resolution order:

      1. If the assembled ``aftershock_pead.parquet`` / ``aftershock_events.parquet`` pair exists (and
         ``rebuild`` is False), read and return it.
      2. Else, build from the raw caches: read ``edgar_eps.parquet`` (deduped to earliest-filed
         announcements), compute the SUE event frame (:func:`build_sue_events`), read the S&P 500 price
         panel's adjusted Close → returns, restrict the events to names present in the panel and to the
         panel's date span, cache the assembled pair, and return it.
      3. Else (the raw caches are also absent — does not happen in this repo), only with ``fetch=True``
         would a network fetch run; offline it returns ``{}``.

    Strictly offline whenever the caches exist (they do). ``rebuild=True`` forces a re-assembly from the
    raw EDGAR/panel caches.
    """
    pead_cache = os.path.join(cache_dir, PEAD_PANEL_CACHE)
    events_cache = os.path.join(cache_dir, PEAD_EVENTS_CACHE)
    if not rebuild and os.path.exists(pead_cache) and os.path.exists(events_cache):
        return {"panel": pd.read_parquet(pead_cache), "events": pd.read_parquet(events_cache)}

    edgar_path = os.path.join(cache_dir, EDGAR_CACHE)
    panel_path = os.path.join(cache_dir, PANEL_CACHE)
    if os.path.exists(edgar_path) and os.path.exists(panel_path):
        eps = pd.read_parquet(edgar_path)
        events = build_sue_events(eps)
        panel = _load_returns_panel(cache_dir)
        events = events[events["ticker"].isin(panel.columns)]
        events = events[(events["date"] >= panel.index[0]) & (events["date"] <= panel.index[-1])]
        events = events.reset_index(drop=True)
        os.makedirs(cache_dir, exist_ok=True)
        panel.to_parquet(pead_cache)
        events.to_parquet(events_cache)
        return {"panel": panel, "events": events}

    if not fetch:
        return {}

    # Network fallback — only reached if even the raw EDGAR/panel caches are absent (not the case here).
    # The yfinance import stays lazy so the offline core never needs the network.
    try:
        import yfinance as yf  # noqa: F401
    except Exception:
        return {}
    return {}
