"""Data layer for Study 1026 — Gibson's Paradox.

Three real tapes, all frozen inside Python packages and SHA-256 pinned through
:mod:`quantlab.bundled` (no network, no Yahoo, nothing that drifts):

- ``load_us_monthly`` — Moody's seasoned **AAA** (and BAA) corporate bond yields, monthly,
  percent, from ``arch`` (``default`` tape, 1919-01 → 2018-12), joined to **US core CPI**
  (``CPILFESL``, seasonally adjusted, from ``arch``'s ``core_cpi`` tape, 1957-01 → 2018-11) and
  the one-month T-bill from the Fama-French tape (``frenchdata``, decimal per month) used as the
  cash leg. The join is what fixes the window: **1957-01 → 2018-11**.
- ``load_us_quarterly`` — the statsmodels ``macrodata`` tape (headline CPI, T-bill, 1959Q1 →
  2009Q3, **final-vintage, revised** figures) with the AAA yield sampled at each quarter-end.
  A robustness tape: a different price index (headline, not core) on a different clock.
- ``load_germany`` — the statsmodels ``interest_inflation`` tape: German quarterly ``Dp``
  (quarterly change in the log GDP deflator, **not seasonally adjusted** — Q1 averages −1.6%,
  Q4 +3.3%) and ``R`` (nominal long-term rate), 1972Q2 → 1998Q4. The out-of-country check.

**The limitation that frames the whole study.** Gibson's paradox lives, by every account, under
the classical gold standard (Keynes's Gibson data run 1791-1924; Barsky & Summers explain it as
a gold-standard phenomenon). The bundled tapes carry Moody's yields back to 1919 but **no price
index before 1957**. So this study cannot test the paradox where it is supposed to live. What it
tests is the investor's modern version — "bond yields follow the price level" — under **fiat
money**, which is still a sharp question: if the claim survives the regime change it is
interesting, and if it does not, it should stop being quoted as a rule for today.

Every series is labelled for what it is: yields are **nominal, percent, annual**; prices are
**100 × log** index levels; inflation is **trailing 12-month (or 4-quarter) change in that log**,
in percent; the price "gap" is **100 × log price minus a past-only linear trend** (see
:func:`price_gap`). The macro tapes are final-vintage; CPI publication lags are applied in
:mod:`gibson.strategy` before anything is used to forecast.

``synthetic_world`` is the deterministic offline generator with a ``signal_strength`` knob: at
``0.0`` yields are a pure **Fisher world** (real rate + inflation), at ``1.0`` a pure **Gibson
world** (yields cointegrated with the detrended price level), with the *same* price path in
both, so a test that cannot tell them apart is caught.
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

# Last full month common to the AAA, core-CPI and Fama-French tapes. Never in the future.
AS_OF = "2018-11-30"
US_START = "1957-01-31"
# The two regimes the claim is usually told about (fixed before the run, not snooped):
GREAT_INFLATION = ("1965-01-31", "1981-12-31")
DISINFLATION = ("1982-01-31", "2018-11-30")

# Past-only detrending of the price level.
GAP_MIN_OBS_MONTHLY = 60       # 5 years before the expanding trend is trusted
GAP_ROLL_MONTHLY = 120         # 10-year rolling trend
GAP_MIN_OBS_QUARTERLY = 20
GAP_ROLL_QUARTERLY = 40

def _month_end_alias() -> str:
    """'ME' on pandas >= 2.2, 'M' before it (both mean calendar month-end)."""
    try:
        pd.date_range("2000-01-31", periods=2, freq="ME")
        return "ME"
    except ValueError:
        return "M"


MONTH_END = _month_end_alias()

TAPES = {
    "default": "arch · Moody's AAA/BAA monthly yields (percent)",
    "core_cpi": "arch · US core CPI (CPILFESL, SA) monthly index",
    "frenchdata": "arch · Fama-French monthly RF (one-month T-bill), cash leg",
    "macrodata": "statsmodels · US quarterly macro, final vintage (headline CPI)",
    "interest_inflation": "statsmodels · German quarterly Dp (Δlog deflator, NSA) and R",
}


# --------------------------------------------------------------------------- #
# Availability and provenance
# --------------------------------------------------------------------------- #
def have_real() -> bool:
    """True iff every tape this study reads is installed (arch + statsmodels)."""
    if not (bundled.have_arch("default") and bundled.have_arch("core_cpi")
            and bundled.have_arch("frenchdata")):
        return False
    try:
        import statsmodels.datasets  # noqa: F401
    except ImportError:
        return False
    return True


def provenance() -> list[dict]:
    """Package, file and SHA-256 pin for each arch tape (statsmodels ships them unpinned)."""
    rows = []
    for name in ("default", "core_cpi", "frenchdata"):
        sub, fname, sha = bundled.ARCH_FILES[name]
        rows.append({"tape": name, "package": "arch", "file": f"{sub}/{fname}",
                     "sha256": sha, "what": TAPES[name]})
    for name in ("macrodata", "interest_inflation"):
        rows.append({"tape": name, "package": "statsmodels", "file": f"datasets/{name}",
                     "sha256": "unpinned — see fingerprint above", "what": TAPES[name]})
    return rows


def fingerprint(obj) -> str:
    """Short content fingerprint of a frame or series (12 hex chars)."""
    return bundled.fingerprint(obj)


# --------------------------------------------------------------------------- #
# The price gap — past-only detrending
# --------------------------------------------------------------------------- #
def price_gap(logp: pd.Series, method: str = "expanding", min_obs: int = 60,
              window: int = 120) -> pd.Series:
    """100 × log price level minus a linear trend fitted **only on data up to t**.

    Under the gold standard the price level wandered around a flat line, so "the price level"
    and "the price level relative to its norm" were the same thing. Under fiat money prices
    trend up for ever, and the raw level is a clock, not a signal. The gap is the closest
    fiat-era analogue of the quantity Gibson correlated with yields.

    - ``expanding``: OLS of log price on time over the whole history up to *t*, residual at *t*
      (needs ``min_obs`` points).
    - ``rolling``: the same over the trailing ``window`` points.

    Past-only by construction — the residual at *t* uses nothing after *t* — so it can be used
    as a forecasting variable once the CPI publication lag is applied. The price paid: the trend
    is re-estimated every period, so the gap carries a little trend-revision noise.
    """
    x = logp.to_numpy(dtype=float)
    n = len(x)
    out = np.full(n, np.nan)
    t = np.arange(n, dtype=float)
    if method == "expanding":
        ok = np.isfinite(x)
        # running sums for an O(n) expanding OLS
        s1 = np.cumsum(ok.astype(float))
        st = np.cumsum(np.where(ok, t, 0.0))
        stt = np.cumsum(np.where(ok, t * t, 0.0))
        sy = np.cumsum(np.where(ok, x, 0.0))
        sty = np.cumsum(np.where(ok, t * x, 0.0))
        for i in range(n):
            if s1[i] < min_obs or not ok[i]:
                continue
            m = s1[i]
            den = m * stt[i] - st[i] ** 2
            if den <= 0:
                continue
            b = (m * sty[i] - st[i] * sy[i]) / den
            a = (sy[i] - b * st[i]) / m
            out[i] = x[i] - (a + b * t[i])
    elif method == "rolling":
        for i in range(window - 1, n):
            seg = x[i - window + 1:i + 1]
            tt = t[i - window + 1:i + 1]
            ok = np.isfinite(seg)
            if ok.sum() < window * 0.9 or not np.isfinite(x[i]):
                continue
            b, a = np.polyfit(tt[ok], seg[ok], 1)
            out[i] = x[i] - (a + b * t[i])
    else:
        raise ValueError("method must be 'expanding' or 'rolling'")
    return pd.Series(out, index=logp.index, name=f"gap_{method[:3]}")


def _add_price_features(df: pd.DataFrame, logp: pd.Series, periods_per_year: int,
                        min_obs: int, window: int) -> pd.DataFrame:
    df = df.copy()
    df["logp"] = logp
    df["pi"] = logp - logp.shift(periods_per_year)          # trailing-year inflation, %
    df["gap_exp"] = price_gap(logp, "expanding", min_obs=min_obs)
    df["gap_rol"] = price_gap(logp, "rolling", window=window)
    return df


# --------------------------------------------------------------------------- #
# Real tapes
# --------------------------------------------------------------------------- #
def load_yields(asof: str = AS_OF) -> pd.DataFrame:
    """Moody's AAA and BAA, monthly, percent, 1919-01 onward, sliced to ``asof``."""
    d = bundled.load_arch("default").rename(columns={"AAA": "aaa", "BAA": "baa"})
    return d[d.index <= pd.Timestamp(asof)]


def load_us_monthly(asof: str = AS_OF) -> pd.DataFrame:
    """US monthly panel 1957-01 → ``asof``: yields, core CPI, inflation, gaps, cash.

    Columns: ``aaa``, ``baa`` (percent); ``cpi`` (core CPI index); ``logp`` (100 × log CPI);
    ``pi`` (12-month inflation, %); ``pi_1m`` (annualised one-month inflation, %);
    ``gap_exp`` / ``gap_rol`` (past-only detrended price level, %); ``rf`` (one-month T-bill,
    decimal per month). No CPI publication lag is applied here — that is the strategy's job.
    """
    y = load_yields(asof)
    cpi = bundled.load_arch("core_cpi")["CPILFESL"].rename("cpi")
    rf = bundled.ff_monthly_total_return()["rf"]
    df = pd.concat([y, cpi, rf.rename("rf")], axis=1, sort=False)
    df = df[(df.index >= pd.Timestamp(US_START)) & (df.index <= pd.Timestamp(asof))]
    df = df.dropna(subset=["aaa", "cpi"])
    logp = 100.0 * np.log(df["cpi"])
    df = _add_price_features(df, logp, 12, GAP_MIN_OBS_MONTHLY, GAP_ROLL_MONTHLY)
    df["pi_1m"] = 12.0 * logp.diff()
    df.index.name = "date"
    return df


def load_us_quarterly() -> pd.DataFrame:
    """US quarterly 1959Q1 → 2009Q3: headline CPI (final vintage) and quarter-end AAA.

    ``tbill`` is the macrodata 3-month T-bill (percent). Inflation is the 4-quarter log change.
    """
    m = bundled.load_macrodata()
    y = load_yields("2009-12-31")["aaa"]
    yq = y.groupby(y.index.to_period("Q")).last()
    yq.index = yq.index.to_timestamp(how="end").normalize()
    df = pd.DataFrame({"aaa": yq.reindex(m.index), "cpi": m["cpi"], "tbill": m["tbilrate"]})
    df = df.dropna(subset=["aaa", "cpi"])
    logp = 100.0 * np.log(df["cpi"])
    df = _add_price_features(df, logp, 4, GAP_MIN_OBS_QUARTERLY, GAP_ROLL_QUARTERLY)
    df.index.name = "date"
    return df


def load_germany() -> pd.DataFrame:
    """German quarterly 1972Q2 → 1998Q4: long rate ``R`` (percent) and the deflator level.

    ``Dp`` is not seasonally adjusted, so the log price level (cumulated ``Dp``) is smoothed
    with a **trailing** four-quarter average before detrending — past-only, and exact for a
    stable seasonal pattern. Inflation is the four-quarter sum of ``Dp`` (seasonality cancels).
    """
    g = bundled.load_interest_inflation()
    df = pd.DataFrame({"R": 100.0 * g["R"], "Dp": g["Dp"]})
    raw = 100.0 * g["Dp"].cumsum()
    logp = raw.rolling(4).mean()
    df = _add_price_features(df, logp, 4, GAP_MIN_OBS_QUARTERLY, GAP_ROLL_QUARTERLY)
    df["pi"] = 100.0 * g["Dp"].rolling(4).sum()             # override: seasonality-free
    df.index.name = "date"
    return df


# --------------------------------------------------------------------------- #
# Synthetic tape — two worlds, one price path
# --------------------------------------------------------------------------- #
def synthetic_world(n_months: int = 720, signal_strength: float = 1.0, seed: int = 1026,
                    burn: int = 120, pi_mean: float = 4.0, pi_phi: float = 0.99,
                    pi_sd: float = 3.0, r_mean: float = 2.0, r_phi: float = 0.95,
                    r_sd: float = 1.0, err_phi: float = 0.9, err_sd: float = 0.3,
                    start: str = "1900-01-31") -> tuple[pd.DataFrame, dict]:
    """A monthly yield / price world with a planted mix of Fisher and Gibson.

    The **price path is identical** whatever ``signal_strength`` is: annualised inflation is a
    persistent AR(1) (``pi_phi`` = 0.99 a month — fiat-like, near unit root) around ``pi_mean``;
    the log price level cumulates it. From that path the same past-only features as the real tape
    are built (12-month inflation ``pi``, the detrended level ``gap_exp``).

    Two candidate "fair yields":

    - **Fisher:** ``F = r + pi`` — a stationary real rate plus (trailing) inflation.
    - **Gibson:** ``G = c + kappa · gap + r`` — yields cointegrated with the detrended price level.
      ``kappa`` is set so ``kappa · gap`` has the same standard deviation as ``pi``, and ``c`` so
      the two worlds have the same mean yield: the worlds differ only in *which* price
      variable drives yields, not in how much yields move.

    The observed yield is ``(1 − s)·F + s·G + u`` with ``u`` a stationary AR(1) pricing error.
    ``signal_strength`` 0 is a pure Fisher world (the matched null for Gibson), 1 a pure Gibson
    world. Because ``u`` mean-reverts, the yield's distance from its fair value predicts its
    next moves — which gives the forecasting machinery something real to find in either world.

    Returns ``(frame, truth)``. The frame has the real tape's columns (``aaa``, ``logp``, ``pi``,
    ``gap_exp``, ``gap_rol``, ``rf``) on a month-end index starting ``start`` (inside pandas'
    ns horizon). Deterministic given ``seed``.
    """
    s = float(signal_strength)
    rng = np.random.default_rng(seed)
    n = int(n_months) + int(burn)
    e_pi = rng.normal(0.0, 1.0, n)
    e_r = rng.normal(0.0, 1.0, n)
    e_u = rng.normal(0.0, 1.0, n)
    e_p = rng.normal(0.0, 1.0, n)

    pi_sig = pi_sd * np.sqrt(1.0 - pi_phi ** 2)
    r_sig = r_sd * np.sqrt(1.0 - r_phi ** 2)
    u_sig = err_sd * np.sqrt(1.0 - err_phi ** 2)
    pi_ann = np.empty(n)
    r = np.empty(n)
    u = np.empty(n)
    pi_ann[0], r[0], u[0] = pi_mean, r_mean, 0.0
    for t in range(1, n):
        pi_ann[t] = pi_mean + pi_phi * (pi_ann[t - 1] - pi_mean) + pi_sig * e_pi[t]
        r[t] = r_mean + r_phi * (r[t - 1] - r_mean) + r_sig * e_r[t]
        u[t] = err_phi * u[t - 1] + u_sig * e_u[t]
    logp = np.cumsum(pi_ann / 12.0 + 0.05 * e_p) + 100.0 * np.log(100.0)

    idx = pd.date_range(start=start, periods=n, freq=MONTH_END)
    lp = pd.Series(logp, index=idx)
    df = _add_price_features(pd.DataFrame(index=idx), lp, 12, GAP_MIN_OBS_MONTHLY,
                             GAP_ROLL_MONTHLY)
    df = df.iloc[burn:].copy()
    r_s = pd.Series(r, index=idx).iloc[burn:]
    u_s = pd.Series(u, index=idx).iloc[burn:]

    gap, pi = df["gap_exp"], df["pi"]
    kappa = float(pi.std() / gap.std())
    c = float(pi.mean() - kappa * gap.mean())
    fisher = r_s + pi
    gibson = c + kappa * gap + r_s
    y = (1.0 - s) * fisher + s * gibson + u_s
    df["aaa"] = y
    df["rf"] = np.clip(r_s + pi - 1.5, 0.0, None) / 1200.0
    df.index.name = "date"
    truth = {"signal_strength": s, "kappa": kappa, "c": c, "seed": seed,
             "n_months": int(n_months), "pi_phi": pi_phi, "r_phi": r_phi,
             "err_phi": err_phi, "err_sd": err_sd,
             "gibson_weight": s, "fisher_weight": 1.0 - s}
    return df, truth
