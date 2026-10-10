"""Data layer for Study 1015 — The 1929 Overlay.

Two real tapes, one synthetic world, all offline.

Real tapes (both from :mod:`quantlab.bundled`, frozen inside Python wheels and SHA-256 pinned,
so a rerun next year reads the same bytes):

- ``load_monthly`` — the Fama-French US market, **monthly total return** (``Mkt-RF + RF``,
  dividends reinvested), July 1926 to November 2018, shipped inside the ``arch`` package. It is
  the only tape on the desk that actually contains 1929, which is the whole point of an
  overlay chart. The index is rebuilt as the cumulative product of ``1 + mkt``; the one-month
  T-bill ``rf`` comes with it, so the timing rule's flat leg earns cash and every Sharpe ratio
  is excess-vs-excess.
- ``load_daily`` — the S&P 500 **price index** (no dividends), daily from January 1990 to
  December 2022, shipped inside the ``skfolio`` wheel and cached under ``studies/_cache/bundled``.
  The tape stops on 28 December 2022, so the partial month is dropped and ``AS_OF_DAILY`` is
  30 November 2022. No cash rate ships with it: the flat leg earns zero and that is labelled
  everywhere it matters (it flatters buy-and-hold slightly, never the timing rule).

The synthetic world (``synthetic_tape``) is a geometric random walk with the tape's drift and
volatility, plus — scaled by ``signal_strength`` — a **template path that genuinely recurs**:
a distinctive ``L``-period shape that is always followed by the same ``h``-period decline. At
``signal_strength=1`` an analogue forecaster has something real to find; at ``0`` the tape is a
pure random walk and anything it "finds" is the search talking. Every test runs on it.
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

MONTHS_PER_YEAR = 12
TRADING_DAYS_PER_YEAR = 252

# Last FULL period of each tape. Fama-French ends 2018-11 (a complete month); the skfolio
# S&P tape ends 2022-12-28, mid-month, so December 2022 is dropped.
AS_OF_MONTHLY = "2018-11-30"
AS_OF_DAILY = "2022-11-30"
AS_OF = AS_OF_DAILY  # the study-wide stamp: the later of the two, never in the future

PROVENANCE = {
    "monthly": {
        "loader": "quantlab.bundled.ff_monthly_total_return()",
        "package": "arch (data/frenchdata/frenchdata.csv.gz)",
        "sha256": bundled.ARCH_FILES["frenchdata"][2],
        "what": "Fama-French US market, monthly TOTAL return (Mkt-RF + RF), nominal, "
                "with one-month T-bill RF",
    },
    "daily": {
        "loader": "quantlab.bundled.load_skfolio('sp500_index')",
        "package": "skfolio 1.8.1 wheel (datasets/data/sp500_index.csv.gz)",
        "sha256": bundled.SKFOLIO_FILES["sp500_index"][1],
        "what": "S&P 500 daily PRICE index (no dividends), nominal; no cash rate shipped",
    },
}


# --------------------------------------------------------------------------- #
# Real tapes
# --------------------------------------------------------------------------- #
def have_monthly() -> bool:
    """True iff the arch Fama-French file is installed (always, in CI)."""
    return bundled.have_arch("frenchdata")


def have_daily() -> bool:
    """True iff the skfolio tapes are in the shared cache."""
    return bundled.have_skfolio()


def have_real() -> bool:
    """True iff BOTH tapes this study reads are available offline."""
    return have_monthly() and have_daily()


def load_monthly(asof: str = AS_OF_MONTHLY) -> pd.DataFrame:
    """US market monthly **total return**, 1926-07 → ``asof``.

    Columns: ``ret`` (simple total return), ``rf`` (one-month bill), ``lr`` (log total return),
    ``logp`` (log of the total-return index, 0 at the start of July 1926). Month-end index.
    """
    ff = bundled.ff_monthly_total_return()
    ff = ff[ff.index <= pd.Timestamp(asof)]
    out = pd.DataFrame({"ret": ff["mkt"], "rf": ff["rf"]})
    out["lr"] = np.log1p(out["ret"])
    out["logp"] = out["lr"].cumsum()
    out.index.name = "date"
    return out


def load_daily(asof: str = AS_OF_DAILY) -> pd.DataFrame:
    """S&P 500 daily **price index**, 1990-01-02 → ``asof``.

    Columns: ``px`` (index level), ``logp`` (log level), ``lr`` (log price return, NaN on the
    first day), ``ret`` (simple price return), ``rf`` (zero — no cash series ships with this
    tape; the flat leg of any rule earns nothing, which is labelled wherever it is used).
    """
    d = bundled.load_skfolio("sp500_index")
    d = d[d.index <= pd.Timestamp(asof)].dropna()
    out = pd.DataFrame({"px": d["SP500"].astype(float)})
    out["logp"] = np.log(out["px"])
    out["lr"] = out["logp"].diff()
    out["ret"] = out["px"].pct_change()
    out["rf"] = 0.0
    out.index.name = "date"
    return out


def fingerprint(obj) -> str:
    """Short content fingerprint of a frame or series (delegates to ``bundled.fingerprint``)."""
    return bundled.fingerprint(obj)


def tape_moments(lr: pd.Series, periods_per_year: int) -> dict:
    """Per-period drift and volatility of a log-return series, plus their annualised forms.

    These are what the random-walk null is calibrated to: the null tape has exactly the real
    tape's drift and volatility and nothing else.
    """
    x = np.asarray(lr.dropna(), dtype=float)
    mu, sd = float(x.mean()), float(x.std(ddof=1))
    return {"mu": mu, "sigma": sd, "mu_ann": mu * periods_per_year,
            "sigma_ann": sd * np.sqrt(periods_per_year), "n": int(x.size)}


# --------------------------------------------------------------------------- #
# Synthetic tape — a random walk, optionally with a template that truly recurs
# --------------------------------------------------------------------------- #
def template_path(L: int, h: int, amp: float = 0.35, drop: float = 0.22) -> np.ndarray:
    """The planted motif as a log-price path of length ``L + h`` (starting at 0).

    The first ``L`` points trace one and a half sine cycles of amplitude ``amp`` — a shape a
    plain random walk rarely draws — and the last ``h`` points fall linearly by ``drop``. The
    ``L``-period part is the "pre-crash" pattern; the ``h``-period part is the crash that, in
    this world and only in this world, really does follow it.
    """
    i = np.arange(1, L + 1)
    up = amp * np.sin(2.0 * np.pi * 1.5 * i / L)
    down = up[-1] - drop * np.arange(1, h + 1) / h
    return np.concatenate([up, down])


def synthetic_tape(n_periods: int = 1100, mu: float = 0.0075, sigma: float = 0.053,
                   signal_strength: float = 1.0, L: int = 24, h: int = 6,
                   mean_gap: int = 40, amp: float = 0.35, drop: float = 0.22,
                   rf: float = 0.003, freq: str = "M", start: str = "1926-07-31",
                   seed: int = 1015) -> tuple[pd.DataFrame, dict]:
    """A deterministic random-walk tape with (optionally) a genuinely recurring template.

    Log returns are ``mu + sigma * eps`` plus, at randomly spaced non-overlapping episodes,
    ``signal_strength`` times the increments of :func:`template_path`. Episodes are separated
    by ``L + h`` plus a geometric gap with mean ``mean_gap`` so they never overlap.

    ``signal_strength=0`` is the matched null: the same noise draws, the same calendar, no
    template — a pure random walk with the tape's drift and vol. Defaults are calibrated to the
    Fama-French monthly tape (drift 0.75%/month, vol 5.3%/month).

    Returns ``(frame, truth)``. The frame has ``ret``, ``rf``, ``lr``, ``logp`` like
    :func:`load_monthly`; ``truth`` records the episode end-points (the index of the last
    ``L``-part period of each template), so a test can check the detector fires where it should.
    """
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n_periods)
    lr = mu + sigma * eps
    motif = np.diff(np.concatenate([[0.0], template_path(L, h, amp, drop)]))
    starts = []
    pos = int(rng.integers(L, L + mean_gap))
    while pos + L + h < n_periods:
        starts.append(pos)
        pos += L + h + int(rng.geometric(1.0 / mean_gap))
    s = float(signal_strength)
    for p in starts:
        lr[p:p + L + h] += s * motif
    if freq == "M":
        idx = pd.date_range(start=start, periods=n_periods, freq="ME")
    else:
        idx = pd.bdate_range(start=start, periods=n_periods)
    df = pd.DataFrame({"lr": lr}, index=pd.DatetimeIndex(idx, name="date"))
    df["ret"] = np.expm1(df["lr"])
    df["rf"] = rf
    df["logp"] = df["lr"].cumsum()
    truth = {"signal_strength": s, "L": L, "h": h, "amp": amp * s, "drop": drop * s,
             "starts": starts, "pattern_ends": [p + L - 1 for p in starts],
             "n_episodes": len(starts), "mu": mu, "sigma": sigma, "seed": seed,
             "n_periods": n_periods}
    return df, truth
