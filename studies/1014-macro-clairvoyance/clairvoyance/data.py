"""Data layer for Study 1014 — The Perfect Macro Forecaster.

Two real tapes, both frozen inside Python packages and SHA-256 pinned by
:mod:`quantlab.bundled`, so this study runs offline and a rerun cannot quietly move a number:

- **Macro** — ``statsmodels.datasets.macrodata``: US quarterly national accounts,
  1959Q1 → 2009Q3. These are **final-vintage** (revised) figures, *not* what anybody knew at
  the time. That is exactly what an oracle study wants for the "perfect foresight" legs and
  exactly what it must flag loudly for the "realistic" leg: the GDP number for 1974Q4 that sits
  in this file was published, revised and re-benchmarked years after 1975.
- **Equity** — Ken French's monthly market factor via ``arch`` (``Mkt-RF + RF``), a
  **total return** (dividends included) on the value-weighted CRSP universe, with the
  one-month T-bill ``RF`` as cash. Monthly returns are compounded to calendar quarters.

The four macro variables, all quarterly and aligned to the quarter-end date:

=========  ==============================================================================
``g``      real GDP growth, annualised percent: ``400 · Δlog(realgdp)``
``du``     change in the unemployment rate, percentage points: ``Δunemp``
``infl``   CPI inflation, annualised percent, as shipped in ``macrodata`` (``infl``)
``dtb``    change in the 3-month T-bill rate, percentage points: ``Δtbilrate``
=========  ==============================================================================

``AS_OF`` is the last full quarter common to both tapes (2009-09-30, the end of
``macrodata``); nothing after it is used.

``synthetic_panel`` is a deterministic, offline generator of the same shape with a
``signal_strength`` knob: at ``1.0`` equity excess returns are planted to load on *next*
quarter's GDP growth (stocks lead the economy, at a known size); at ``0.0`` returns are
independent of every macro series — the matched null under which no oracle may look useful.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from quantlab import bundled  # noqa: E402

QUARTERS_PER_YEAR = 4
MACRO_VARS = ("g", "du", "infl", "dtb")
# Which direction of each variable the folklore calls "good for stocks".
GOOD_SIGN = {"g": +1, "du": -1, "infl": -1, "dtb": -1}
LABELS = {"g": "real GDP growth", "du": "unemployment change",
          "infl": "CPI inflation", "dtb": "T-bill change"}

# The last full quarter of macrodata — the binding tape. Never in the future.
AS_OF = "2009-09-30"
START = "1959-01-01"

PROVENANCE = {
    "macro": ("statsmodels.datasets.macrodata — US quarterly, FINAL VINTAGE (revised), "
              "1959Q1-2009Q3"),
    "equity": ("arch frenchdata — Fama-French monthly Mkt-RF + RF (TOTAL return, CRSP "
               "value-weighted), compounded to quarters"),
    "equity_sha256": bundled.ARCH_FILES["frenchdata"][2],
    # statsmodels' macrodata.csv is not pinned by quantlab.bundled; this study pins it here.
    "macro_sha256": "d93c0d3a7a77ef83c3af14e46032bb1d02ae3a512b22ab94159a8ca226fcf708",
}


def macro_file_sha256() -> str:
    """SHA-256 of the ``macrodata.csv`` shipped in the installed statsmodels."""
    import hashlib

    from statsmodels.datasets import macrodata
    path = os.path.join(os.path.dirname(macrodata.__file__), "macrodata.csv")
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def package_versions() -> dict:
    """Versions of the packages whose wheels carry the tapes."""
    import arch
    import statsmodels
    return {"arch": arch.__version__, "statsmodels": statsmodels.__version__}


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def have_real() -> bool:
    """True iff both bundled tapes are available (arch + statsmodels; offline)."""
    if not bundled.have_arch("frenchdata"):
        return False
    try:
        import statsmodels.datasets.macrodata  # noqa: F401
    except ImportError:
        return False
    return True


def load_macro(asof: str = AS_OF) -> pd.DataFrame:
    """Quarterly macro variables ``g, du, infl, dtb`` (plus the raw levels), final vintage.

    Indexed by quarter-end date. The first quarter is lost to differencing. Raw levels
    (``realgdp``, ``unemp``, ``tbilrate``, ``cpi``) ride along for plotting.
    """
    m = bundled.load_macrodata()
    out = pd.DataFrame({
        "g": 400.0 * np.log(m["realgdp"]).diff(),
        "du": m["unemp"].diff(),
        "infl": m["infl"],
        "dtb": m["tbilrate"].diff(),
        "realgdp": m["realgdp"], "unemp": m["unemp"], "tbilrate": m["tbilrate"],
        "cpi": m["cpi"],
    }, index=m.index)
    out = out.iloc[1:]                      # first row: no difference, and infl is 0.00
    out.index = pd.DatetimeIndex(out.index).normalize()
    out.index.name = "date"
    return out[out.index <= pd.Timestamp(asof)]


def load_equity_quarterly(asof: str = AS_OF) -> pd.DataFrame:
    """Quarterly market **total return** ``mkt``, T-bill ``rf`` and excess ``ex``.

    Monthly Fama-French returns compounded within each calendar quarter; a quarter is kept
    only if all three of its months are present (no partial quarters).
    """
    ff = bundled.ff_monthly_total_return()[["mkt", "rf"]]
    grp = ff.groupby(ff.index.to_period("Q"))
    q = (1.0 + ff).groupby(ff.index.to_period("Q")).prod() - 1.0
    q = q[grp.size() == 3]
    q.index = q.index.to_timestamp(how="end").normalize()
    q.index.name = "date"
    q["ex"] = q["mkt"] - q["rf"]
    return q[(q.index <= pd.Timestamp(asof)) & (q.index >= pd.Timestamp(START))]


def load_panel(asof: str = AS_OF) -> pd.DataFrame:
    """The study panel: one row per quarter with ``mkt, rf, ex, g, du, infl, dtb``."""
    eq = load_equity_quarterly(asof)
    mac = load_macro(asof)
    df = eq.join(mac, how="inner").dropna(subset=["ex", *MACRO_VARS])
    return df


def fingerprint(obj) -> str:
    """Short content fingerprint (12 hex chars), via :func:`quantlab.bundled.fingerprint`."""
    return bundled.fingerprint(obj)


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core
# --------------------------------------------------------------------------- #
def synthetic_panel(n_quarters: int = 400, signal_strength: float = 1.0,
                    lead: int = 1, beta: float = 0.040, seed: int = 1014,
                    start: str = "1900-03-31") -> tuple[pd.DataFrame, dict]:
    """A quarterly panel with the *stocks-lead-the-economy* effect planted at a known size.

    Macro: ``g`` is an AR(1) around 3% (annualised, sd ≈ 3.5); ``du`` is Okun's law
    (−0.15·(g−3) plus noise); ``infl`` and ``dtb`` are independent AR(1)s.

    Equity excess return in quarter ``t``::

        ex_t = 0.015 + signal_strength · beta · z(g_{t+lead}) + e_t,   e_t ~ N(0, 0.08)

    so the market's return *now* is tied to the standardised GDP growth ``lead`` quarters
    *later*. At ``signal_strength=0`` the panel is the null: the same macro paths, returns
    independent of all of them. ``rf`` is a constant 1% a quarter. Deterministic in ``seed``;
    the date index stays well inside pandas' ns horizon.
    """
    rng = np.random.default_rng(seed)
    n_tot = n_quarters + max(lead, 0) + 5
    g = np.empty(n_tot)
    g[0] = 3.0
    eps = rng.normal(0.0, 3.0, n_tot)
    for t in range(1, n_tot):
        g[t] = 3.0 + 0.35 * (g[t - 1] - 3.0) + eps[t]
    du = -0.15 * (g - 3.0) + rng.normal(0.0, 0.25, n_tot)
    infl = np.empty(n_tot)
    infl[0] = 4.0
    e_i = rng.normal(0.0, 1.2, n_tot)
    for t in range(1, n_tot):
        infl[t] = 4.0 + 0.8 * (infl[t - 1] - 4.0) + e_i[t]
    dtb = 0.3 * rng.normal(0.0, 0.8, n_tot)
    z = (g - g.mean()) / g.std()
    lead_z = np.roll(z, -lead)            # z_{t+lead}
    noise = rng.normal(0.0, 0.08, n_tot)
    ex = 0.015 + float(signal_strength) * beta * lead_z + noise
    sl = slice(0, n_quarters)
    idx = pd.date_range(start=start, periods=n_quarters, freq="QE")
    rf = np.full(n_quarters, 0.01)
    df = pd.DataFrame({"ex": ex[sl], "rf": rf, "g": g[sl], "du": du[sl],
                       "infl": infl[sl], "dtb": dtb[sl]},
                      index=pd.DatetimeIndex(idx, name="date"))
    df["mkt"] = df["ex"] + df["rf"]
    truth = {"n_quarters": n_quarters, "signal_strength": float(signal_strength),
             "lead": lead, "beta": beta, "seed": seed,
             "beta_eff": float(signal_strength) * beta}
    return df[["mkt", "rf", "ex", *MACRO_VARS]], truth
