"""Data layer for Study 1023 — The Cushing Glut.

Two tapes, one shape.

**Real tape — ``load_crude``.** Monthly Brent and WTI spot prices in US$ per barrel,
1987-05 → 2020-01, as shipped inside the ``arch`` package (``arch/data/crude/crude.csv.gz``,
FRED-sourced) and read through :mod:`quantlab.bundled`, which refuses the file unless its bytes
match a pinned SHA-256. No network, no cache, no drift: the tape is frozen in the wheel.

Three labels travel with every number built on it:

* **Spot, not futures.** These are physical spot assessments. Nobody can buy "the spot
  spread": a real trade uses ICE Brent and NYMEX WTI futures and pays (or earns) the **roll**,
  and the roll yields of the two contracts diverged most violently exactly when Cushing was
  full (WTI's front spread sat in deep contango). Every P&L in this study is therefore an
  **upper bound** on what a futures trader could have booked.
* **Monthly averages, not month-end marks.** ``check_monthly_average`` verifies against
  ``arch``'s daily FRED WTI tape that the monthly WTI column is the *calendar-month mean* of
  daily prints. A trade cannot transact at a monthly average; averaging also smooths the
  series and puts an MA(1) into its changes (Working 1960), which flatters a mean-reversion
  rule a little.
* **Nominal US$.** No deflation; the spread is a difference of two nominal prices.

**Synthetic tape — ``synthetic_spread``.** A deterministic generator: WTI is a geometric
random walk, and the log Brent/WTI ratio is an Ornstein–Uhlenbeck process around a mean that
can jump once (a planted level break). ``signal_strength`` scales the mean-reversion *speed*:
``1.0`` plants a half-life of ``halflife_months``, ``0.0`` makes the ratio a pure random walk —
the matched null under which nothing in this study may find a tether.

``AS_OF`` pins the last full month of the tape; nothing past it is ever read.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from quantlab import bundled  # noqa: E402

TAPE = "crude"                 # arch dataset name
DAILY_TAPE = "wti"             # used only to verify the monthly-average construction
PACKAGE = "arch"
SHA256 = bundled.ARCH_FILES[TAPE][2]
DAILY_SHA256 = bundled.ARCH_FILES[DAILY_TAPE][2]

START = "1987-05-31"
AS_OF = "2020-01-31"           # last full month on the tape (monthly average of Jan-2020)
MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def have_real() -> bool:
    """True iff the ``arch`` crude tape (and the daily WTI tape) can be read offline."""
    return bool(bundled.have_arch(TAPE) and bundled.have_arch(DAILY_TAPE))


def load_crude(asof: str = AS_OF) -> pd.DataFrame:
    """Monthly Brent and WTI spot (US$/bbl, monthly averages), month-end indexed.

    Columns ``brent`` and ``wti``. Sliced to ``asof`` so a re-run cannot quietly grow the
    window; rows with a missing print are dropped, never filled. Raises
    :class:`quantlab.bundled.TapeMismatch` if the shipped bytes differ from the pin.
    """
    df = bundled.load_arch(TAPE).rename(columns={"Brent": "brent", "WTI": "wti"})
    df = df[["brent", "wti"]].dropna()
    df = df[(df.index >= pd.Timestamp(START)) & (df.index <= pd.Timestamp(asof))]
    df.index.name = "date"
    return df.astype(float)


def load_wti_daily() -> pd.Series:
    """Daily WTI spot (FRED ``DCOILWTICO``), 1986-01-02 → 2019-01-03. Context only."""
    return bundled.load_arch(DAILY_TAPE)["wti"].astype(float)


def check_monthly_average(monthly: pd.DataFrame | None = None,
                          daily: pd.Series | None = None) -> dict:
    """How closely does the monthly WTI column match the calendar-month mean of daily prints?

    Only full months covered by both tapes are compared (the daily tape ends 2019-01-03, so
    January 2019 is excluded as partial). The result decides a label, not a number: if the
    monthly column is an average, the backtest marks at prices nobody could trade.
    """
    if monthly is None:
        monthly = load_crude()
    if daily is None:
        daily = load_wti_daily()
    last_full = (daily.index[-1] + pd.offsets.MonthEnd(0)) - pd.offsets.MonthEnd(1)
    d = daily[daily.index <= last_full]
    avg = d.resample("ME").mean()
    end = d.resample("ME").last()
    j = pd.concat([monthly["wti"], avg.rename("avg"), end.rename("end")],
                  axis=1, sort=False).dropna()
    gap_avg = (j["wti"] - j["avg"]).abs()
    gap_end = (j["wti"] - j["end"]).abs()
    return {"n_months": int(len(j)),
            "median_abs_gap_avg": float(gap_avg.median()),
            "share_within_1c_avg": float((gap_avg <= 0.01).mean()),
            "median_abs_gap_end": float(gap_end.median()),
            "is_monthly_average": bool(gap_avg.median() < 0.01
                                       and gap_avg.median() < gap_end.median())}


def fingerprint(obj) -> str:
    """12-hex content fingerprint (``quantlab.bundled.fingerprint``) of a frame or series."""
    return bundled.fingerprint(obj)


# --------------------------------------------------------------------------- #
# Synthetic tape
# --------------------------------------------------------------------------- #
def synthetic_spread(n_months: int = 393,
                     signal_strength: float = 1.0,
                     halflife_months: float = 3.0,
                     sigma: float = 0.027,
                     mean_log_ratio: float = -0.06,
                     break_size: float = 0.0,
                     break_frac: float = 0.72,
                     wti_vol: float = 0.08,
                     start: str = "1987-05-31",
                     seed: int = 1023) -> tuple[pd.DataFrame, dict]:
    """A monthly Brent/WTI pair with a planted OU tether and an optional level break.

    Defaults are calibrated to the real tape's pre-break regime (log-ratio AR(1): half-life
    ≈ 2.7 months, residual s.d. ≈ 0.027, mean ≈ −0.06; WTI monthly log-vol ≈ 0.08), so the
    power numbers speak to the sample we actually have.

    ``log(wti)`` is a driftless random walk with monthly vol ``wti_vol`` from $20/bbl.
    ``x_t = log(brent/wti) = m_t + y_t`` with ``y_t = phi * y_{t-1} + sigma * e_t`` and
    ``phi = exp(-signal_strength * ln2 / halflife_months)``:

    * ``signal_strength = 1`` — the ratio reverts with the planted half-life;
    * ``signal_strength = 0`` — ``phi = 1``: a random walk, the matched null (same shocks);
    * ``break_size`` — ``m_t`` jumps from ``mean_log_ratio`` to ``mean_log_ratio + break_size``
      at observation ``int(break_frac * n_months)``. A break of ~0.15 in logs is the order of
      magnitude the real tape carries (a $15 discount on a $100 barrel).

    Returns ``(prices, truth)`` with columns ``brent``, ``wti`` on a month-end index that ends
    well inside pandas' nanosecond horizon. Deterministic given ``seed``.
    """
    s = float(signal_strength)
    rng = np.random.default_rng(seed)
    phi = float(np.exp(-s * np.log(2.0) / halflife_months))
    e = rng.normal(0.0, 1.0, n_months)
    w = rng.normal(0.0, wti_vol, n_months)
    y = np.zeros(n_months)
    # start the stationary leg from its stationary distribution (or at zero for the null)
    y[0] = sigma * e[0] if phi >= 1.0 else sigma * e[0] / np.sqrt(1.0 - phi ** 2)
    for t in range(1, n_months):
        y[t] = phi * y[t - 1] + sigma * e[t]
    k = int(break_frac * n_months)
    m = np.full(n_months, mean_log_ratio)
    m[k:] += break_size
    x = m + y
    log_wti = np.log(20.0) + np.cumsum(w)
    wti = np.exp(log_wti)
    brent = np.exp(log_wti + x)
    idx = pd.date_range(start=start, periods=n_months, freq="ME", name="date")
    prices = pd.DataFrame({"brent": brent, "wti": wti}, index=idx)
    truth = {"signal_strength": s, "phi": phi,
             "halflife_months": (float(np.log(0.5) / np.log(phi)) if phi < 1.0
                                 else float("inf")),
             "sigma": sigma, "mean_log_ratio": mean_log_ratio,
             "break_size": float(break_size), "break_index": k if break_size else None,
             "break_date": idx[k] if break_size else None,
             "n_months": n_months, "seed": seed}
    return prices, truth
