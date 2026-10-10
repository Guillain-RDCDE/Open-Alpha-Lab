"""Data layer for Study 1021 — Stability Breeds Instability.

Three frozen real tapes and one synthetic world, all offline.

Real tapes — every one read through :mod:`quantlab.bundled`, SHA-256 pinned
--------------------------------------------------------------------------
- ``load_ff`` — Fama-French monthly, 1926-07 → 2018-11, from the ``arch`` package. The
  market column ``mkt = Mkt-RF + RF`` is a **total return** (dividends in), nominal, and
  ``rf`` is the one-month T-bill. This is the **primary** tape: 92 years is the only window
  on this desk long enough to hold more than a handful of independent multi-year crashes.
- ``load_spx_daily`` — S&P 500 daily close, 1990-01 → 2022-11, from the ``skfolio`` wheel.
  A **price index** — no dividends. Used for monthly realised volatility measured properly
  from daily returns (the short-horizon clustering leg) and for a post-1990 illustration.
  It is never used for a return-level claim.
- ``load_vix`` — daily VIX close, 2014-01 → 2018-12, from ``arch``. Five years: the 2017
  record calm and the 2018 "Volmageddon" sit inside it, which makes a lovely picture and
  **no inference at all** — one episode is an anecdote. The study says so wherever it
  appears.

Every loader slices to its tape's last **full** month: ``sp500_index`` ends 2022-12-28 and
``vix`` on 2019-01-03, so December 2022 and January 2019 are partial and dropped.

Synthetic world — the deterministic offline core
------------------------------------------------
``synthetic_monthly`` writes a monthly equity tape whose volatility is a GARCH(1,1) with
Student-t shocks — so it clusters and mean-reverts exactly like the null this study tests
against — plus a crash process. ``signal_strength`` is the planted Minsky channel: at
``1.0`` the monthly probability that a crash starts rises with the tape's own **prolonged
calm** score (computed online, past-only, with exactly the definition the strategy uses);
at ``0.0`` crashes still happen at the base rate but are blind to calm — the matched null,
where mean reversion is the only link between quiet and storm.
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

# Per-tape as-of: the last FULL month of each tape. Never in the future.
AS_OF_FF = "2018-11-30"
AS_OF_SPX = "2022-11-30"
AS_OF_VIX = "2018-12-31"
# The headline (primary-tape) as-of.
AS_OF = AS_OF_FF

# --- the pre-registered signal definition (shared by generator and strategy) -------- #
VOL_WINDOW = 12      # months of returns in one realised-vol reading
TREND_WINDOW = 120   # months in the slow trend that "low" is measured against (10 years)
CALM_WINDOW = 60     # months the below-trend shortfall is averaged over (5 years)


# --------------------------------------------------------------------------- #
# Real tapes
# --------------------------------------------------------------------------- #
def have_ff() -> bool:
    """True iff the arch Fama-French tape is installed (always, on the desk and in CI)."""
    return bundled.have_arch("frenchdata")


def have_vix() -> bool:
    return bundled.have_arch("vix")


def have_spx() -> bool:
    """True iff the skfolio S&P 500 tape is in the bundled cache."""
    return bundled.have_skfolio()


def have_real() -> bool:
    """True iff every tape this study reads is available offline."""
    return have_ff() and have_vix() and have_spx()


def load_ff(asof: str = AS_OF_FF) -> pd.DataFrame:
    """Fama-French monthly, decimal: ``mkt`` (**total return**), ``rf``, ``mkt_rf``.

    Month-end index, 1926-07-31 → ``asof``. Nominal. ``mkt`` is the CRSP value-weighted
    market including dividends — the right tape for drawdowns and terminal wealth.
    """
    ff = bundled.ff_monthly_total_return()[["mkt", "rf", "mkt_rf"]]
    return ff[ff.index <= pd.Timestamp(asof)].copy()


def load_spx_daily(asof: str = AS_OF_SPX) -> pd.Series:
    """S&P 500 daily close (**price index**, no dividends), 1990-01-02 → ``asof``."""
    s = bundled.load_skfolio("sp500_index")["SP500"].dropna()
    s.name = "spx"
    return s[s.index <= pd.Timestamp(asof)]


def load_vix(asof: str = AS_OF_VIX) -> pd.Series:
    """Daily VIX close, 2014-01-03 → ``asof``. Illustration only — five years, one episode."""
    s = bundled.load_arch("vix")["vix"].dropna()
    s.name = "vix"
    return s[s.index <= pd.Timestamp(asof)]


def monthly_rv_from_daily(px: pd.Series) -> pd.Series:
    """Annualised realised volatility of each calendar month from daily log returns.

    ``sqrt(252 · mean(r_d²))`` over the trading days of the month — zero-mean realised
    volatility, the standard estimator. Indexed by month-end.
    """
    r = np.log(px).diff().dropna()
    rv = (r ** 2).groupby(r.index.to_period("M")).mean()
    rv = np.sqrt(TRADING_DAYS_PER_YEAR * rv)
    rv.index = rv.index.to_timestamp(how="end").normalize()
    rv.name = "rv"
    return rv


def monthly_returns_from_daily(px: pd.Series) -> pd.Series:
    """Calendar-month simple **price** returns of a daily close series (month-end index)."""
    m = px.groupby(px.index.to_period("M")).last()
    m.index = m.index.to_timestamp(how="end").normalize()
    return m.pct_change().dropna().rename("ret")


def fingerprint(obj) -> str:
    """Short content fingerprint of a frame or series (``quantlab.bundled.fingerprint``)."""
    return bundled.fingerprint(obj)


def provenance() -> list[dict]:
    """One row per real tape: package, file, SHA-256 pin, label, span, fingerprint."""
    rows = []
    if have_ff():
        ff = load_ff()
        _, fname, sha = bundled.ARCH_FILES["frenchdata"]
        rows.append({"tape": "Fama-French monthly (mkt = Mkt-RF + RF, rf)",
                     "package": "arch", "file": fname, "sha256": sha,
                     "label": "monthly, nominal, **total return**",
                     "first": str(ff.index[0].date()), "last": str(ff.index[-1].date()),
                     "rows": int(len(ff)), "fingerprint": fingerprint(ff)})
    if have_spx():
        spx = load_spx_daily()
        fname, sha = bundled.SKFOLIO_FILES["sp500_index"]
        rows.append({"tape": "S&P 500 index (SP500)", "package": "skfolio 1.8.1",
                     "file": fname, "sha256": sha,
                     "label": "daily, nominal, **price only**",
                     "first": str(spx.index[0].date()), "last": str(spx.index[-1].date()),
                     "rows": int(len(spx)), "fingerprint": fingerprint(spx)})
    if have_vix():
        vix = load_vix()
        _, fname, sha = bundled.ARCH_FILES["vix"]
        rows.append({"tape": "CBOE VIX close", "package": "arch", "file": fname,
                     "sha256": sha, "label": "daily, index level (illustration only)",
                     "first": str(vix.index[0].date()), "last": str(vix.index[-1].date()),
                     "rows": int(len(vix)), "fingerprint": fingerprint(vix)})
    return rows


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core
# --------------------------------------------------------------------------- #
def synthetic_monthly(
    n_months: int = 2400,
    signal_strength: float = 1.0,
    seed: int = 1021,
    ann_vol: float = 0.16,          # unconditional annualised volatility of the GARCH
    alpha: float = 0.12,            # GARCH ARCH coefficient
    beta: float = 0.85,             # GARCH persistence term (alpha + beta = 0.97)
    nu: float = 8.0,                # Student-t degrees of freedom
    excess_mu: float = 0.005,       # monthly equity premium
    rf: float = 0.003,              # monthly bill rate
    base_crash_p: float = 0.003,    # monthly crash-start probability floor
    calm_crash_slope: float = 0.25,  # extra probability per unit of calm score (planted)
    calm_ref: float = 0.20,         # typical calm score; the null pays it out flat
    crash_months: int = 3,          # a crash unfolds over this many months
    crash_drift: float = -0.08,     # extra monthly return during a crash
    start: str = "1850-01-31",
    burn: int = 240,
) -> tuple[pd.DataFrame, dict]:
    """A monthly total-return tape with clustering vol and a **planted** Minsky channel.

    Each month the excess return is ``excess_mu + sqrt(h_t)·z_t + jump_t`` with
    ``h_t`` a GARCH(1,1) and ``z_t`` unit-variance Student-t. ``jump_t`` is
    ``crash_drift`` in each of the ``crash_months`` months of a crash (about −22% over a
    quarter at the defaults) and zero otherwise. Jumps feed the GARCH recursion, so a
    crash also leaves vol elevated afterwards — as real crashes do.

    A crash starts next month with probability
    ``base_crash_p + calm_crash_slope · (s · calm_t + (1 − s) · calm_ref)`` with
    ``s = signal_strength``, where ``calm_t`` is
    the **prolonged calm score** of this tape so far — the trailing 60-month average of
    the shortfall of 12-month realised vol below its trailing 10-year mean (exactly
    :func:`calm.strategy.calm_signal`). ``signal_strength = 0`` makes crashes calm-blind
    while paying out the typical calm score ``calm_ref`` flat, so the null has about the
    **same number** of crashes — the matched null, where only *when* they happen changes.
    Deterministic given ``seed``. Dates stay inside pandas' ns horizon.

    Returns ``(frame, truth)``: columns ``mkt`` (total return), ``rf``, ``mkt_rf``,
    ``calm_true`` (the online calm score the plant saw), ``crash_start``.
    """
    rng = np.random.default_rng(seed)
    n = int(n_months) + int(burn)
    var_m = (ann_vol ** 2) / MONTHS_PER_YEAR
    omega = var_m * (1.0 - alpha - beta)
    z = rng.standard_t(nu, n) * np.sqrt((nu - 2.0) / nu)
    u = rng.random(n)

    exc = np.zeros(n)
    tot = np.zeros(n)
    calm = np.full(n, np.nan)
    crash_start = np.zeros(n, dtype=bool)
    rv = np.full(n, np.nan)
    low = np.full(n, np.nan)
    h = var_m
    crash_left = 0
    s = float(signal_strength)
    for t in range(n):
        jump = crash_drift if crash_left > 0 else 0.0
        if crash_left > 0:
            crash_left -= 1
        e = np.sqrt(h) * z[t] + jump
        exc[t] = excess_mu + e
        tot[t] = exc[t] + rf
        h = omega + alpha * e ** 2 + beta * h
        # --- the online, past-only calm score (identical to strategy.calm_signal) ---
        if t >= VOL_WINDOW - 1:
            rv[t] = np.std(tot[t - VOL_WINDOW + 1:t + 1], ddof=1) * np.sqrt(MONTHS_PER_YEAR)
        if t >= VOL_WINDOW - 1 + TREND_WINDOW - 1:
            trend = np.mean(rv[t - TREND_WINDOW + 1:t + 1])
            low[t] = max(-np.log(rv[t] / trend), 0.0)
        if t >= VOL_WINDOW + TREND_WINDOW + CALM_WINDOW - 3:
            calm[t] = np.mean(low[t - CALM_WINDOW + 1:t + 1])
        # --- next month's crash draw ---
        if crash_left == 0:
            c = calm[t] if np.isfinite(calm[t]) else 0.0
            p = base_crash_p + calm_crash_slope * (s * c + (1.0 - s) * calm_ref)
            if u[t] < p:
                crash_start[t] = True       # decided at end of t, unfolds from t+1
                crash_left = crash_months

    sl = slice(burn, n)
    idx = pd.date_range(start=start, periods=int(n_months), freq="ME", name="date")
    frame = pd.DataFrame({"mkt": tot[sl], "rf": np.full(int(n_months), rf),
                          "mkt_rf": exc[sl], "calm_true": calm[sl],
                          "crash_start": crash_start[sl]}, index=idx)
    truth = {"n_months": int(n_months), "seed": seed, "signal_strength": s,
             "calm_crash_slope_eff": s * calm_crash_slope, "base_crash_p": base_crash_p,
             "calm_ref": calm_ref,
             "alpha": alpha, "beta": beta, "nu": nu, "ann_vol": ann_vol,
             "crash_months": crash_months, "crash_drift": crash_drift,
             "n_crashes": int(frame["crash_start"].sum()), "burn": burn}
    return frame, truth
