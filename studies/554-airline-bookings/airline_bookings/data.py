"""Data layer for Study 554 (Airline-Bookings) — travel-booking momentum as a lead.

The alt-data pitch: a proprietary **flight-bookings index** (aggregated ticket sales / search
volume from a card-panel or a GDS feed) turns up *before* the airline sector's earnings and
therefore *before* the sector's stock returns — so booking momentum should **lead** airline
equities and hand an early buyer a predictable edge. The vendors sell exactly this: "our panel
saw the travel rebound weeks before the tape did."

The catch this study is built to expose is the classic alt-data trap: even if bookings genuinely
lead *fundamentals*, the market is forward-looking, so the price may already embed the booking
signal by the time a public reader sees it — the lead is *contemporaneous*, not *predictive*, and
there is no tradable edge in the forward return.

**Data availability (named on the SIGNAL axis).** There is no free, retail-reachable, point-in-time
weekly flight-bookings index. The real vendors (card-panel aggregators, GDS/ARC feeds) are paywalled
and survivorship-/revision-prone. So this study is **synthetic-only**: a deterministic generator
that plants a *tunable* lead, plus a real-fetch stub that returns empty by design. A synthetic-only
study can never earn `REAL` (that needs a robust *t* >= 2 on a real tape) — it is capped at `WEAK`.

Two tapes, one schema (a monthly frame indexed by period-end, columns ``bookings_mom`` and
``fwd_ret`` plus helpers):

- ``synthetic_series`` — the deterministic offline core. A single knob, ``lead_beta``, plants the
  *predictive* lead: booking momentum this month explains the airline basket's *next*-month return
  (``lead_beta > 0`` = the pitch is true; ``= 0`` = the null, bookings carry no forward
  information). A second knob, ``contemp_beta``, sets how much of the booking signal is *already in
  the price this month* (the "efficient-market" leakage) — high ``contemp_beta`` with zero
  ``lead_beta`` is the realistic world: bookings and prices move together, but there is nothing
  left to predict. Tests never touch the network.
- ``fetch_series`` — the real-tape stub. No free booking index exists, so ``fetch=False`` (default)
  returns an empty frame and ``fetch=True`` raises a clear ``NotImplementedError`` naming the
  paywall. The reproducible core is therefore fully offline.

No look-ahead: the signal at month *t* (booking momentum computed from data through the close of
month *t*) is regressed on the airline basket's return over month *t+1*. The one-month execution
lag between signal and traded return is the study's single documented convention.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 554
N_MONTHS = 180  # 15 years of monthly observations


@dataclass(frozen=True)
class WorldTruth:
    """The planted structure for the synthetic series."""

    lead_beta: float      # forward-predictive loading (the pitch; > 0 = bookings lead returns)
    contemp_beta: float   # same-month loading (efficiency leakage; the signal already in price)

    @property
    def has_lead(self) -> bool:
        return self.lead_beta != 0.0


# ---------------------------------------------------------------------------
# Synthetic series — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_series(
    n_months: int = N_MONTHS,
    lead_beta: float = 0.0,
    contemp_beta: float = 0.9,
    ret_vol: float = 0.06,
    seed: int = SEED,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible monthly booking/return series with a tunable *predictive* lead.

    A latent monthly **travel-demand shock** ``u_t`` (AR(1), persistent, so bookings *trend*)
    drives an observed **bookings momentum** signal and the airline basket's return:

        bookings_mom_t = u_t + measurement noise
        fwd_ret_{t+1}  = drift + lead_beta * u_t                 (the predictive lead)
                                + idio_{t+1}
        ret_t          = drift + contemp_beta * u_t + idio_t     (the same-month co-move)

    - ``lead_beta > 0`` plants a genuine forward-predictive lead: *this* month's booking momentum
      forecasts *next* month's airline return (the pitch). ``lead_beta = 0`` is the null — bookings
      carry no forward information.
    - ``contemp_beta`` sets the *contemporaneous* co-movement: bookings and the same-month return
      share the demand shock. In an efficient market this is large (the price already reflects the
      demand) while ``lead_beta`` is ~0 — bookings *look* informative in a same-month plot but there
      is nothing left to trade on the *forward* return.

    Returns a frame indexed by month-end with columns ``bookings_mom`` (the tradable signal at *t*),
    ``ret`` (the airline basket's return *in* month *t*), ``fwd_ret`` (the return in month *t+1*, the
    thing we try to predict), and ``demand`` (the latent shock, for diagnostics). Plus ``truth``.
    """
    rng = np.random.default_rng(seed)

    # Latent AR(1) travel-demand shock (persistent -> bookings trend, momentum is meaningful).
    rho = 0.5
    demand = np.zeros(n_months)
    innov = rng.standard_normal(n_months)
    for t in range(1, n_months):
        demand[t] = rho * demand[t - 1] + np.sqrt(1 - rho**2) * innov[t]
    demand[0] = innov[0]

    meas_noise = 0.4 * rng.standard_normal(n_months)      # the panel is a noisy read of demand
    bookings_mom = demand + meas_noise

    drift = 0.006                                          # ~7%/yr baseline airline drift
    idio = ret_vol * rng.standard_normal(n_months)
    ret = drift + contemp_beta * ret_vol * demand + idio  # same-month co-move (efficiency leakage)

    # Forward return: month t+1's return, loaded on THIS month's demand shock (the lead).
    idio_fwd = ret_vol * rng.standard_normal(n_months)
    fwd_ret = drift + lead_beta * ret_vol * demand + idio_fwd

    idx = pd.period_range("2011-01", periods=n_months, freq="M").to_timestamp("M")
    frame = pd.DataFrame(
        {
            "bookings_mom": bookings_mom,
            "ret": ret,
            "fwd_ret": fwd_ret,
            "demand": demand,
        },
        index=idx,
    )
    frame.index.name = "month"
    return frame, WorldTruth(lead_beta=lead_beta, contemp_beta=contemp_beta)


# ---------------------------------------------------------------------------
# Real tape — a stub: no free flight-bookings index exists
# ---------------------------------------------------------------------------
def fetch_series(fetch: bool = False) -> pd.DataFrame:
    """Real-tape stub. No free, point-in-time flight-bookings index is retail-reachable.

    ``fetch=False`` (default) returns an empty frame so the reproducible core stays offline;
    ``fetch=True`` raises ``NotImplementedError`` naming the paywall, rather than silently
    fabricating a "real" tape. The data-availability limitation is stated on the SIGNAL axis and
    caps this study at `WEAK`.
    """
    if not fetch:
        return pd.DataFrame()
    raise NotImplementedError(
        "No free, point-in-time weekly/monthly flight-bookings index is retail-reachable. "
        "The real vendors (card-panel aggregators, GDS/ARC ticketing feeds) are paywalled and "
        "revision-prone. This study is synthetic-only and is capped at WEAK on the Signal axis."
    )


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, dict):
        blob = json.dumps(obj, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
