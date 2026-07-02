"""Data layer for Study 550 (Box-Office-Momentum).

The claim (alt-data folklore): a strong *weekend box-office* — the domestic weekend gross of
theatrical releases — is a **consumer-sentiment leading indicator**. When people flood the
cinemas, the story goes, consumers are confident and spending, so media/studio stocks (and even
the broad market) should rise in the weeks that follow. The tradable version: build a
box-office *momentum* signal (this weekend's gross vs its trailing norm) and use it to predict
forward returns of a media/studio basket and SPY.

WHY THIS STUDY IS SYNTHETIC-ONLY
--------------------------------
A no-key retail stack cannot reach an honest historical box-office panel:

- Box Office Mojo / The Numbers publish weekend grosses on the web, but there is **no free,
  rate-limit-friendly, survivorship-honest API** — scraping is against terms, brittle, and the
  historical series is riddled with definitional breaks (3-day vs 4-day weekends, holiday
  shifts, the 2020-21 pandemic shutdown that zeroed the box office, the streaming-era decline).
- Worse, the *studios* whose stock you would trade have been **acquired or restructured** out of
  a clean panel (Fox → Disney 2019; Time Warner → AT&T → Warner Bros. Discovery; Viacom/CBS →
  Paramount; Lionsgate spin-offs). A pure-play "studio stock" basket that survives 15 years does
  not exist — survivorship bias on both legs.

So the study is built **synthetic-only** and states this limitation openly on the SIGNAL axis.
A synthetic-only study can never earn `REAL` (that needs a robust ``t >= 2`` on a real tape); it
is capped at `WEAK`/`NONE`. What the synthetic core *can* do honestly is (a) show the machinery
is truthful — a planted predictive edge is recovered, the null stays flat — and (b) illustrate
the twin confounds the folklore ignores: box office and stocks *both* load on the same aggregate
consumer/market factor, so a naive predictive regression finds a "signal" that is really just
the market predicting itself.

TWO OFFLINE COMPONENTS
----------------------
- ``BOX_OFFICE_INDEX`` — a curated, hardcoded weekly domestic-weekend-gross **index** (base 100),
  a stylised composite tracing the real arc: the 2015-19 plateau, the 2020 pandemic collapse to
  ~zero, the 2021-23 partial recovery, and the streaming-era structural decline. Deterministic,
  content-stable, treated as *illustrative* only (never as a fitted signal for a stamp).
- ``synthetic_panel`` — a deterministic, seeded generator of weekly (box-office-momentum,
  media-return, SPY-return) triples with a single knob, ``predictive_beta``, that plants how much
  *next week's* media return truly loads on *this week's* box-office momentum. ``predictive_beta =
  0`` is the null (box office has no predictive edge over the common market factor);
  ``predictive_beta > 0`` plants a genuine leading-indicator effect the engine must recover.

NO LOOK-AHEAD: the signal at week ``t`` uses only box-office data known by the close of week
``t`` (this weekend's gross and its trailing average). The position is entered at that close and
held over week ``t+1`` — one documented execution week of lag. No future data enters the signal.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

SEED = 550


# ---------------------------------------------------------------------------
# A curated weekly box-office index — illustrative only, hardcoded & offline
# ---------------------------------------------------------------------------
# Base 100 at the start. This is a STYLISED composite of the domestic weekend box
# office, not a tick-by-tick reconstruction. It traces the real qualitative arc so
# the notebooks can show the shape (plateau -> 2020 collapse -> partial recovery ->
# streaming decline) without pretending to be a fitted historical panel. It is
# NEVER used to earn a stamp — every published number comes from the synthetic
# engine or is labelled illustrative.
#
# Sources for the qualitative shape (not the exact levels, which are stylised):
#   - Box Office Mojo / The Numbers — annual domestic gross trend.
#   - MPAA/MPA Theatrical Home Entertainment Market reports (2015-2023).
# As-of: 2026-06-30.

def _curated_box_office_levels() -> list[float]:
    """A 12-year, monthly-cadence stylised box-office index (base 100)."""
    # 2014..2025, 12 points/yr = 144 months. Built from year-anchors with a light
    # deterministic seasonal ripple (summer + holiday peaks). Hardcoded shape.
    year_anchor = {
        2014: 100.0, 2015: 104.0, 2016: 107.0, 2017: 106.0, 2018: 110.0,
        2019: 112.0, 2020: 22.0, 2021: 55.0, 2022: 78.0, 2023: 86.0,
        2024: 82.0, 2025: 80.0,  # streaming-era structural decline
    }
    seasonal = np.array([0.82, 0.80, 0.92, 0.98, 1.05, 1.18,  # J F M A M J (summer ramp)
                         1.22, 1.15, 0.95, 0.90, 1.05, 1.20]) # J A S O N D (holiday peak)
    levels: list[float] = []
    years = sorted(year_anchor)
    for yi, y in enumerate(years):
        base = year_anchor[y]
        nxt = year_anchor[years[min(yi + 1, len(years) - 1)]]
        for m in range(12):
            # linear interpolation across the year, times the seasonal ripple
            w = m / 12.0
            lvl = (1 - w) * base + w * nxt
            levels.append(round(lvl * seasonal[m], 2))
    return levels


BOX_OFFICE_INDEX = _curated_box_office_levels()


def curated_index() -> pd.DataFrame:
    """The curated monthly box-office index as a tidy frame (illustrative only).

    Columns: ``t`` (0-based month), ``bo_index`` (level, base 100),
    ``bo_return`` (month-over-month change). Deterministic, offline, content-stable.
    """
    lvl = np.asarray(BOX_OFFICE_INDEX, dtype=float)
    df = pd.DataFrame({"t": np.arange(len(lvl)), "bo_index": lvl})
    df["bo_return"] = df["bo_index"].pct_change()
    return df


# ---------------------------------------------------------------------------
# The synthetic panel — the deterministic offline core (positive control + null)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WorldTruth:
    """The planted structure for the synthetic panel."""

    predictive_beta: float  # true loading of next-week media return on this-week BO momentum
    common_beta_media: float  # media loading on the contemporaneous common consumer/market factor
    common_beta_spy: float    # SPY loading on the same common factor

    @property
    def has_signal(self) -> bool:
        return self.predictive_beta != 0.0


def synthetic_panel(
    n_weeks: int = 520,
    predictive_beta: float = 0.0,
    common_beta_media: float = 1.15,
    common_beta_spy: float = 1.0,
    bo_common_load: float = 0.6,
    factor_ar1: float = 0.25,
    media_vol: float = 0.030,
    spy_vol: float = 0.018,
    weekly_drift: float = 0.003,
    seed: int = SEED,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible weekly panel with a tunable box-office predictive edge.

    The generative story, honest about the confound the folklore ignores:

    - A latent **common consumer/market factor** ``f_t`` (mildly *persistent*, AR(1)) drives box
      office, media stocks and SPY *contemporaneously* (confident consumers => busy cinemas AND
      rising stocks, in the SAME week). This is the confound: box office and returns co-move
      because both load on ``f``, and because ``f`` is persistent, this-week box office correlates
      with *next* week's factor too — so a naive predictive regression finds a SPURIOUS lead.
    - **Box-office momentum** ``m_t`` = this week's box-office surprise (its shock plus its load
      on ``f_t``), standardised. It is knowable at the close of week ``t``.
    - **Next-week media return** ``r^{media}_{t+1}`` loads on the *contemporaneous* factor
      ``f_{t+1}`` (the honest driver) PLUS ``predictive_beta`` times ``m_t`` (the planted genuine
      lead). With ``predictive_beta = 0`` the box office has NO predictive power beyond the common
      factor — the null. With ``predictive_beta > 0`` there is a real leading-indicator edge.
    - **SPY** next-week return loads only on the contemporaneous factor (no planted BO lead) — so
      the "does box office predict the *whole tape*" question is a clean null by construction
      unless you also raise its predictive loading.

    Columns returned (aligned so row ``t`` pairs signal ``bo_mom`` at ``t`` with the returns it is
    meant to predict, ``media_fwd`` / ``spy_fwd`` = the week-``t+1`` returns):

        ``week``, ``bo_mom`` (standardised box-office momentum, the signal),
        ``media_fwd`` (next-week media basket return), ``spy_fwd`` (next-week SPY return),
        ``media_now`` (contemporaneous media return, for the confound demo).

    Returns ``(panel, truth)``.
    """
    rng = np.random.default_rng(seed)
    T = n_weeks + 1  # one extra so every row has a forward return

    # latent common consumer/market factor, weekly, mildly PERSISTENT (AR(1)). The persistence
    # is the crux of the folklore's confound: because f_t and f_{t+1} are correlated, a naive
    # predictive regression of next-week return on this-week box-office momentum (both loading on
    # f) will pick up a SPURIOUS positive slope — which the contemporaneous-SPY control removes.
    inno = rng.standard_normal(T)
    f = np.empty(T)
    f[0] = inno[0]
    for i in range(1, T):
        f[i] = factor_ar1 * f[i - 1] + np.sqrt(max(0.0, 1.0 - factor_ar1**2)) * inno[i]
    f = f * 0.012

    # box-office momentum: loads on the common factor + its own idiosyncratic shock
    bo_shock = rng.standard_normal(T)
    bo_raw = bo_common_load * (f / 0.012) + np.sqrt(max(0.0, 1.0 - bo_common_load**2)) * bo_shock
    bo_mom = (bo_raw - bo_raw.mean()) / (bo_raw.std(ddof=1) + 1e-12)

    # contemporaneous returns loaded on f, plus idiosyncratic noise
    media_idio = media_vol * rng.standard_normal(T)
    spy_idio = spy_vol * rng.standard_normal(T)
    # a small positive weekly drift so buy-and-hold is a FAIR benchmark (an equity tape rises on
    # average; without drift a half-time-flat timer would "beat" a zero-drift market for free)
    media_ret = weekly_drift + common_beta_media * f + media_idio
    spy_ret = weekly_drift + common_beta_spy * f + spy_idio

    # PLANT the lead: next-week media return gets predictive_beta * this-week BO momentum
    media_ret_led = media_ret.copy()
    media_ret_led[1:] = media_ret[1:] + predictive_beta * bo_mom[:-1]

    # assemble: row t holds signal at t and the return realised at t+1
    week = np.arange(n_weeks)
    panel = pd.DataFrame(
        {
            "week": week,
            "bo_mom": bo_mom[:n_weeks],
            "media_fwd": media_ret_led[1:n_weeks + 1],
            "spy_fwd": spy_ret[1:n_weeks + 1],
            "media_now": media_ret_led[:n_weeks],
        }
    )
    truth = WorldTruth(
        predictive_beta=predictive_beta,
        common_beta_media=common_beta_media,
        common_beta_spy=common_beta_spy,
    )
    return panel, truth


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, (list, tuple)):
        arr = np.ascontiguousarray(np.asarray(obj, dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
