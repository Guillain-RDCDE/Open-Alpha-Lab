"""Data layer for Study 870 — Industry-Leader Lead-Lag.

The claim under test (Kewei **Hou 2007**, *"Industry Information Diffusion and the
Lead-Lag Effect in Stock Returns"*, RFS): information diffuses **within an industry**
from the *biggest* name outward. The **largest-cap** firm in a sector prices news
first; the smaller **followers** react with a delay, so the **leader's return this
week predicts the followers' return next week**. Long the followers whose leader
*rose*, short those whose leader *fell*, and you should collect a slow-diffusion
premium.

Two ingredients, both offline-friendly once cached.

* **Real tape — a liquid US cross-section.** Daily OHLC(V) for the same fixed list of
  ~50 liquid US large-caps (``UNIVERSE`` below), pulled with yfinance through the
  ``quantlab.universe`` **survivorship guard** (``download_panel(...,
  allow_survivorship_bias=True)``). ``auto_adjust=True`` (total-return prices). The
  panel parquet is cached under this study's OWN ``_cache/`` (we point
  ``quantlab.universe``'s cache there via ``OVERNIGHT_CACHE`` *before* importing it).

  **Sectors & leaders — a documented, static designation.** ``SECTORS`` assigns each
  name to a GICS-style sector (a public fact); ``LEADERS`` names the single
  **largest-cap** member of each sector across the sample (a public-record market-cap
  ranking, cited in ``docs/references.md``). The designation is **static** — the leader
  is a stable mega-cap throughout, exactly Hou's "big firm" within an industry — and is
  named on the Signal axis as a caveat (see below). The *signal itself* is strictly
  point-in-time: leader return through the close of week ``w`` predicts follower return
  in week ``w+1`` (a one-week execution lag, zero look-ahead into returns).

  **Survivorship — named on the Signal axis.** ``UNIVERSE`` is a *current* membership
  list of names that are liquid mega-caps *today*; feeding it to a backward-looking
  panel omits the delisted / de-rated names and biases any cross-sectional result. The
  guard forces the opt-in; the caveat travels with every published number. The
  *leader* map inherits the same caveat — it is today's largest-cap per sector.

* **Synthetic world — the positive control.** A deterministic, seeded panel
  (``synthetic_panel``) with a TUNABLE knob ``edge``: each sector has a designated
  leader carrying a weekly latent shock that — only when ``edge > 0`` — **diffuses into
  the sector's followers one week later**. ``edge = 0`` is the null world: leaders and
  followers each wander independently and the lead-lag sort must find nothing.
  ``edge > 0`` plants the Hou leader→follower diffusion.

The offline path is pure numpy + pandas + stdlib. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells; ``load_panel()``
reads the cached parquet directly (no yfinance import).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
# Point quantlab.universe's cache at THIS study's _cache/ before importing it.
os.environ.setdefault("OVERNIGHT_CACHE", CACHE_DIR)

from quantlab.universe import (  # noqa: E402  (after the env var is set)
    SurvivorshipBiasError,
    download_panel,
    panel_cache_path,
)

START = "2010-01-01"        # panel start (matches quantlab.universe default)
AS_OF = "2026-06-30"        # last complete calendar month at publication

# A fixed list of ~50 liquid US large-caps — *current* membership, a survivor set.
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "HD", "BAC", "XOM", "CVX", "KO", "PEP", "ABBV",
    "COST", "MRK", "PFE", "CSCO", "ORCL", "ADBE", "CRM", "NKE", "DIS", "MCD",
    "TXN", "INTC", "QCOM", "AMD", "IBM", "GE", "CAT", "BA", "MMM", "HON",
    "UNH", "T", "VZ", "WFC", "GS", "MS", "C", "AXP", "LMT", "UPS",
]

# GICS-style sector for each name (a public fact). Payments (V, MA) placed with
# Financials per current GICS; Alphabet/Meta/Disney/telecoms in Communication Services.
SECTORS: dict[str, list[str]] = {
    "InfoTech":       ["AAPL", "MSFT", "NVDA", "ORCL", "ADBE", "CRM", "CSCO",
                       "TXN", "INTC", "QCOM", "AMD", "IBM"],
    "Communications": ["GOOGL", "META", "DIS", "T", "VZ"],
    "ConsDiscr":      ["AMZN", "TSLA", "HD", "NKE", "MCD"],
    "ConsStaples":    ["WMT", "PG", "KO", "PEP", "COST"],
    "Financials":     ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "V", "MA"],
    "HealthCare":     ["JNJ", "ABBV", "MRK", "PFE", "UNH"],
    "Energy":         ["XOM", "CVX"],
    "Industrials":    ["GE", "CAT", "BA", "MMM", "HON", "UPS", "LMT"],
}

# The largest-cap member of each sector across the sample (public-record market-cap
# ranking, cited in docs/references.md). Static designation — Hou's "big firm".
LEADERS: dict[str, str] = {
    "InfoTech":       "AAPL",   # Apple — largest US tech-cap for most of 2010-2026
    "Communications": "GOOGL",  # Alphabet > Meta
    "ConsDiscr":      "AMZN",   # Amazon >> Tesla / Home Depot
    "ConsStaples":    "WMT",    # Walmart > Costco / P&G
    "Financials":     "JPM",    # JPMorgan > Visa / Mastercard
    "HealthCare":     "JNJ",    # Johnson & Johnson — largest health-cap most of sample
    "Energy":         "XOM",    # Exxon > Chevron
    "Industrials":    "GE",     # GE — the dominant mega-cap industrial (esp. 2010-2017)
}

__all__ = [
    "SurvivorshipBiasError",
    "UNIVERSE", "SECTORS", "LEADERS", "START", "AS_OF", "CACHE_DIR",
    "fetch", "have_real", "load_panel",
    "synthetic_panel", "synthetic_sectors", "synthetic_leaders",
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START) -> None:
    """Download the cross-section panel through the survivorship guard; cache it."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    download_panel(
        UNIVERSE, start=start, use_cache=True, allow_survivorship_bias=True,
    )


def have_real() -> bool:
    return os.path.exists(panel_cache_path(UNIVERSE, START))


def load_panel(start: str = START, asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    """Cached panel as ``{ticker: DataFrame[Open, High, Low, Close, Volume]}``, sliced
    to ``[start, asof]``. Reads the parquet directly — OFFLINE, no yfinance import.
    ``Volume`` is carried so the dollar-volume size proxy (a robustness re-designation
    of leaders) can be computed from the tape."""
    cache = panel_cache_path(UNIVERSE, start)
    raw = pd.read_parquet(cache)
    lo, hi = pd.Timestamp(start), pd.Timestamp(asof)
    panel: dict[str, pd.DataFrame] = {}
    cols = ["Open", "High", "Low", "Close", "Volume"]
    for s in UNIVERSE:
        if s not in raw.columns.get_level_values(0):
            continue
        avail = [c for c in cols if c in raw[s].columns]
        df = raw[s][avail].dropna()
        df = df[(df.index >= lo) & (df.index <= hi)]
        if not df.empty:
            panel[s] = df
    return panel


# --------------------------------------------------------------------------- #
# Synthetic world — planted leader->follower diffusion (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_sectors(n_sectors: int = 8, per_sector: int = 5) -> dict[str, list[str]]:
    """Deterministic sector map matching :func:`synthetic_panel`'s naming.

    Sector ``s`` holds one leader ``S{s}L`` and ``per_sector-1`` followers
    ``S{s}F{j}``.
    """
    out: dict[str, list[str]] = {}
    for s in range(n_sectors):
        names = [f"S{s}L"] + [f"S{s}F{j}" for j in range(per_sector - 1)]
        out[f"SEC{s}"] = names
    return out


def synthetic_leaders(n_sectors: int = 8) -> dict[str, str]:
    """The designated leader of each synthetic sector (``S{s}L``)."""
    return {f"SEC{s}": f"S{s}L" for s in range(n_sectors)}


def synthetic_panel(
    edge: float = 0.0,
    seed: int = 870,
    n_sectors: int = 8,
    per_sector: int = 5,
    n_weeks: int = 320,
    start: str = "2010-01-04",
    daily_vol: float = 0.012,
    drift: float = 0.05 / 252,
    wk_lead_vol: float = 0.02,
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded OHLC(V) panel with a TUNABLE planted lead-lag relation.

    Structure: ``n_sectors`` sectors, each a designated **leader** ``S{s}L`` and
    ``per_sector-1`` **followers** ``S{s}F{j}``. For sector ``s`` a weekly latent
    **leader shock** ``Lw[s, w]`` (sd ``wk_lead_vol``) is spread across the five
    business days of week ``w`` in the leader's returns. The followers receive that
    same shock — but **one week late** and scaled by ``edge``:

        leader daily return  (week w) = drift + Lw[s,w]/5 + noise
        follower daily return(week w) = drift + edge * Lw[s,w-1]/5 + noise

    So with ``edge > 0`` the leader's week-``w`` move predicts the followers' week-
    ``w+1`` move (slow within-industry diffusion). ``edge = 0`` is the null: leaders
    and followers wander independently and a lead-lag sort finds nothing. Leaders carry
    ~10x the followers' ``Volume`` so a dollar-volume size proxy re-designates the same
    leaders. Business-day index; span well below the pandas ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    n_days = n_weeks * 5
    idx = pd.bdate_range(start, periods=n_days)
    # week label per day (integer week number) so the leader shock is constant within a week
    week_of = np.arange(n_days) // 5

    panel: dict[str, pd.DataFrame] = {}
    for s in range(n_sectors):
        Lw = rng.normal(0.0, wk_lead_vol, n_weeks)           # weekly leader shock
        Lw_prev = np.concatenate([[0.0], Lw[:-1]])            # last week's shock
        lead_today = Lw[week_of]                              # per-day leader shock
        lead_lag = Lw_prev[week_of]                           # per-day lagged shock

        def _prices(r):
            close = 100.0 * np.cumprod(1.0 + r)
            prev = np.concatenate([[100.0], close[:-1]])
            open_ = prev * (1.0 + rng.normal(0.0, daily_vol / 3, n_days))
            hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
            lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
            return open_, hi, lo, close

        # leader
        r_lead = drift + lead_today / 5.0 + rng.normal(0.0, daily_vol, n_days)
        o, h, l, c = _prices(r_lead)
        vol_lead = rng.uniform(8e6, 1.2e7, n_days)
        panel[f"S{s}L"] = pd.DataFrame(
            {"Open": o, "High": h, "Low": l, "Close": c, "Volume": vol_lead}, index=idx
        )
        # followers
        for j in range(per_sector - 1):
            r_f = drift + edge * lead_lag / 5.0 + rng.normal(0.0, daily_vol, n_days)
            o, h, l, c = _prices(r_f)
            vol_f = rng.uniform(6e5, 1.2e6, n_days)
            panel[f"S{s}F{j}"] = pd.DataFrame(
                {"Open": o, "High": h, "Low": l, "Close": c, "Volume": vol_f}, index=idx
            )
    return panel
