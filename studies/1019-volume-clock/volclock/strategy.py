"""The volume clock, measured and then asked to earn its keep — Study 1019.

Clark (1973) wrote daily returns as a *subordinated* process: a Gaussian random walk observed
at a random number of "ticks" per day, with the ticks proxied by trading volume. If that is
right, dividing each day's return by the square root of how much the clock ran that day should
turn a fat-tailed series into a normal one. Ané & Geman (2000) pushed the same idea with the
number of trades as the clock and reported near-normality on individual stocks.

The study runs four standardisations side by side on the *same* sessions, so the comparison is
about the clock and not the sample:

=================  ==========================================================  ==============
clock              ``z_t = r_t / ...``                                           known before
=================  ==========================================================  ==============
raw                1                                                             —
volume clock       ``sqrt(V_t / median(V_{t-63..t-1}))`` — Clark's clock,        after close t
                   detrended past-only so the 2000s volume boom cannot dominate
range (Parkinson)  the day's own high-low sigma — an *upper* benchmark: it is   after close t
                   that day's volatility, measured
GARCH(1,1)         ``sqrt(h_t)`` from a past-only, expanding-window fit — the    close t-1
                   usual benchmark
GARCH × volume     GARCH sigma times the square root of *volume surprise*       after close t
=================  ==========================================================  ==============

The verdict is split along the column the elegant version of the theory skips. The first three
non-trivial clocks are read *at the close of the day they explain*; only GARCH is a forecast.
So the tradability half of the study asks a different, harder question: does **yesterday's**
volume improve tomorrow's volatility forecast beyond HAR and GARCH, out of sample and by a
Diebold–Mariano test — and if it does, does a vol-targeting overlay built on that better
forecast beat the standard one after costs, with one execution lag?

Everything is past-only: the volume detrend uses a rolling median that ends yesterday, GARCH is
refitted on an expanding window, the HAR regressions are refitted every month on data that was
available at the time.
"""

from __future__ import annotations

from math import erfc, sqrt

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import chi2

TRADING_DAYS = 252
LN2x4 = 4.0 * np.log(2.0)
CLOCKS = ("raw", "volume", "range", "garch", "garch_volume")
CLOCK_LABELS = {
    "raw": "raw daily return",
    "volume": "volume clock (Clark)",
    "range": "own-day range (Parkinson)",
    "garch": "GARCH(1,1), past-only",
    "garch_volume": "GARCH × volume surprise",
}
NORMAL_TAIL3 = 0.0026997960632601866     # P(|Z| > 3)
NORMAL_TAIL4 = 6.334248366623996e-05     # P(|Z| > 4)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def log_returns(close: pd.Series) -> pd.Series:
    """Close-to-close log returns (price index: dividends excluded)."""
    return np.log(close).diff().rename("r")


def parkinson_var(high: pd.Series, low: pd.Series) -> pd.Series:
    """Parkinson (1980) daily variance from the high-low range: ``ln(H/L)^2 / (4 ln 2)``."""
    return (np.log(high / low) ** 2 / LN2x4).rename("pk")


def relative_volume(volume: pd.Series, window: int = 63) -> pd.Series:
    """Volume divided by its own **past-only** rolling median (window ending yesterday).

    The index volume series roughly quadrupled over 1999-2009 for reasons that have little to
    do with how much information arrived — share counts, decimalisation, venue fragmentation.
    A full-sample detrend would leak the future; a past-only median does not, and the median
    rather than the mean keeps one crash day from deflating the next quarter.
    """
    base = volume.shift(1).rolling(window, min_periods=max(10, window // 2)).median()
    return (volume / base).rename("relvol")


def volume_surprise(volume: pd.Series, window: int = 21) -> pd.Series:
    """Volume relative to its own trailing-``window`` mean ending yesterday."""
    base = volume.shift(1).rolling(window, min_periods=window // 2).mean()
    return (volume / base).rename("vsurprise")


# --------------------------------------------------------------------------- #
# GARCH(1,1), vectorised and past-only
# --------------------------------------------------------------------------- #
def _logistic(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def garch_filter(r2: np.ndarray, omega: float, alpha: float, beta: float,
                 var0: float) -> np.ndarray:
    """``v_t = omega + alpha r_{t-1}^2 + beta v_{t-1}``, returned with length ``len(r2) + 1``.

    ``v[t]`` is the variance for day ``t`` formed with returns through ``t-1``; the extra last
    element is the one-step-ahead forecast after the final return. Implemented with
    ``scipy.signal.lfilter`` so a twenty-year path costs microseconds.
    """
    u = np.empty(r2.size + 1)
    u[0] = var0
    u[1:] = omega + alpha * r2
    return np.maximum(lfilter([1.0], [1.0, -beta], u), 1e-18)


def _garch_nll(params: np.ndarray, r2: np.ndarray, uncond: float) -> float:
    a, b = _logistic(params[0]), _logistic(params[1])
    if a + b >= 0.999:
        return 1e12
    v = garch_filter(r2, uncond * (1.0 - a - b), a, b, uncond)[:-1]
    return float(0.5 * np.sum(np.log(v) + r2 / v))


def fit_garch(r: np.ndarray) -> dict:
    """GARCH(1,1) by Gaussian QML with variance targeting (two free parameters)."""
    x = np.asarray(r, dtype=float)
    x = x[np.isfinite(x)]
    uncond = float(np.mean(x ** 2))
    r2 = x ** 2
    res = minimize(_garch_nll, np.array([np.log(0.08 / 0.92), np.log(0.90 / 0.10)]),
                   args=(r2, uncond), method="Nelder-Mead",
                   options={"maxiter": 400, "xatol": 1e-5, "fatol": 1e-7})
    a, b = _logistic(res.x[0]), _logistic(res.x[1])
    return {"omega": float(uncond * (1 - a - b)), "alpha": float(a), "beta": float(b),
            "uncond": uncond, "n": int(x.size)}


def garch_forecasts(r: pd.Series, burn: int = 504, refit: int = 252) -> pd.Series:
    """One-step-ahead GARCH variance forecasts, strictly past-only.

    Value at index ``t`` is the forecast **for** ``t+1``, made at the close of ``t`` with
    parameters fitted on returns through the most recent refit date (expanding window, refit
    every ``refit`` sessions). NaN for the first ``burn`` sessions.
    """
    x = r.to_numpy(dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    n = x.size
    out = np.full(n, np.nan)
    for t0 in range(burn, n, refit):
        p = fit_garch(x[1:t0 + 1])
        t1 = min(n, t0 + refit)
        v = garch_filter(x[:t1] ** 2, p["omega"], p["alpha"], p["beta"], p["uncond"])
        # forecast for t+1 made at t = v[t+1]
        out[t0:t1] = v[t0 + 1:t1 + 1]
    return pd.Series(out, index=r.index, name="garch")


# --------------------------------------------------------------------------- #
# Section 1 — the clocks and the kurtosis they remove
# --------------------------------------------------------------------------- #
def moments(x) -> dict:
    """Skew, excess kurtosis (moment estimator), Jarque–Bera and its chi-square(2) p-value."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 30:
        return {"n": n, "skew": np.nan, "excess_kurtosis": np.nan, "jb": np.nan,
                "jb_p": np.nan}
    d = x - x.mean()
    m2 = np.mean(d ** 2)
    s = float(np.mean(d ** 3) / m2 ** 1.5)
    k = float(np.mean(d ** 4) / m2 ** 2 - 3.0)
    jb = float(n / 6.0 * (s ** 2 + k ** 2 / 4.0))
    return {"n": int(n), "skew": s, "excess_kurtosis": k, "jb": jb,
            "jb_p": float(chi2.sf(jb, 2))}


def clock_panel(tape: pd.DataFrame, window: int = 63, burn: int = 504,
                refit: int = 252) -> pd.DataFrame:
    """Every standardisation of the same returns, on the common sample where all exist.

    Each column is rescaled to unit variance (kurtosis is scale-free, but tail quantiles and
    exceedance counts are read against N(0,1)).
    """
    r = log_returns(tape["close"])
    rv = relative_volume(tape["volume"], window)
    vs = volume_surprise(tape["volume"])
    pk = parkinson_var(tape["high"], tape["low"])
    h = garch_forecasts(r, burn=burn, refit=refit).shift(1)     # variance for day t, info t-1
    df = pd.DataFrame({
        "raw": r,
        "volume": r / np.sqrt(rv),
        "range": r / np.sqrt(pk.where(pk > 0)),
        "garch": r / np.sqrt(h),
        "garch_volume": r / np.sqrt(h * vs),
    })
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df / df.std(ddof=0)


def tail_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Tail quantiles and exceedance ratios of each unit-variance clock against N(0,1)."""
    rows = []
    for c in panel.columns:
        z = panel[c].to_numpy()
        rows.append({
            "clock": c,
            "q0.1%": float(np.quantile(z, 0.001)), "q1%": float(np.quantile(z, 0.01)),
            "q99%": float(np.quantile(z, 0.99)), "q99.9%": float(np.quantile(z, 0.999)),
            "beyond3_vs_normal": float(np.mean(np.abs(z) > 3) / NORMAL_TAIL3),
            "beyond4_vs_normal": float(np.mean(np.abs(z) > 4) / NORMAL_TAIL4),
        })
    return pd.DataFrame(rows).set_index("clock")


def _cbb_indices(n: int, n_boot: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap index matrix, ``(n_boot, n)``."""
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    return idx % n


def _excess_kurt_rows(X: np.ndarray) -> np.ndarray:
    d = X - X.mean(axis=1, keepdims=True)
    m2 = np.mean(d ** 2, axis=1)
    return np.mean(d ** 4, axis=1) / m2 ** 2 - 3.0


def kurtosis_removed(panel: pd.DataFrame, n_boot: int = 500, block: int = 21,
                     seed: int = 1019) -> pd.DataFrame:
    """Excess kurtosis of each clock and the share of the raw excess kurtosis it removes.

    ``share_removed = 1 - k_clock / k_raw``: 1 means the clock produced a normal (mesokurtic)
    series, 0 means it did nothing, negative means dividing by it *added* tail mass — which is
    what an unrelated clock does, since it is extra noise in the denominator. A share above 1
    means the clock over-corrects into thinner-than-normal tails.

    Intervals are a circular block bootstrap over days (block = a month), resampling the raw
    and clocked series **jointly** so the ratio's sampling error is paired. i.i.d. resampling
    would shred the volatility clustering that the kurtosis is made of.
    """
    rng = np.random.default_rng(seed)
    n = len(panel)
    idx = _cbb_indices(n, n_boot, block, rng)
    raw = panel["raw"].to_numpy()
    k_raw_b = _excess_kurt_rows(raw[idx])
    rows = []
    k_raw = moments(raw)["excess_kurtosis"]
    for c in panel.columns:
        z = panel[c].to_numpy()
        mo = moments(z)
        kb = _excess_kurt_rows(z[idx])
        share_b = 1.0 - kb / k_raw_b
        share = 1.0 - mo["excess_kurtosis"] / k_raw
        rows.append({
            "clock": c, "n": mo["n"], "skew": mo["skew"],
            "excess_kurtosis": mo["excess_kurtosis"],
            "kurt_ci_lo": float(np.quantile(kb, 0.025)),
            "kurt_ci_hi": float(np.quantile(kb, 0.975)),
            "share_removed": float(share),
            "share_ci_lo": float(np.quantile(share_b, 0.025)),
            "share_ci_hi": float(np.quantile(share_b, 0.975)),
            "jb": mo["jb"], "jb_p": mo["jb_p"],
        })
    return pd.DataFrame(rows).set_index("clock")


def incremental_removal(panel: pd.DataFrame, a: str = "garch_volume", b: str = "garch",
                        n_boot: int = 500, block: int = 21, seed: int = 1019) -> dict:
    """Does clock ``a`` thin the tails *beyond* clock ``b``? Paired block-bootstrap CI.

    Reports ``k_b - k_a`` (positive: ``a`` leaves less excess kurtosis than ``b``). The default
    asks the Lamoureux–Lastrapes question: once the GARCH state is accounted for, does same-day
    volume *surprise* still explain tail mass?
    """
    rng = np.random.default_rng(seed)
    idx = _cbb_indices(len(panel), n_boot, block, rng)
    A, B = panel[a].to_numpy(), panel[b].to_numpy()
    d_b = _excess_kurt_rows(B[idx]) - _excess_kurt_rows(A[idx])
    d = moments(B)["excess_kurtosis"] - moments(A)["excess_kurtosis"]
    return {"a": a, "b": b, "kurt_reduction": float(d),
            "ci_lo": float(np.quantile(d_b, 0.025)), "ci_hi": float(np.quantile(d_b, 0.975)),
            "share_boot_positive": float(np.mean(d_b > 0))}


def exponent_curve(tape: pd.DataFrame, exponents=None, window: int = 63) -> pd.Series:
    """Excess kurtosis of ``r / relvol^(b/2)`` across exponents ``b`` (b=1 is Clark's clock).

    The minimiser is chosen **in sample**, so it is an optimistic number and labelled as such:
    the point of the curve is its shape, not its minimum.
    """
    if exponents is None:
        exponents = np.round(np.linspace(0.0, 2.0, 21), 2)
    r = log_returns(tape["close"])
    rv = relative_volume(tape["volume"], window)
    df = pd.concat([r, rv], axis=1).dropna()
    out = {}
    for b in exponents:
        out[float(b)] = moments(df["r"] / df["relvol"] ** (b / 2.0))["excess_kurtosis"]
    return pd.Series(out, name="excess_kurtosis").rename_axis("exponent")


def window_sensitivity(tape: pd.DataFrame, windows=(21, 63, 126, 252)) -> pd.DataFrame:
    """Does the detrending window drive the answer? Kurtosis removed by the volume clock."""
    r = log_returns(tape["close"])
    rows = []
    for w in windows:
        rv = relative_volume(tape["volume"], w)
        df = pd.concat([r, rv], axis=1).dropna().iloc[252:]   # common burn for every window
        kr = moments(df["r"])["excess_kurtosis"]
        kv = moments(df["r"] / np.sqrt(df["relvol"]))["excess_kurtosis"]
        rows.append({"window": w, "k_raw": kr, "k_volume": kv,
                     "share_removed": 1.0 - kv / kr})
    return pd.DataFrame(rows).set_index("window")


def by_period(panel: pd.DataFrame, periods=None) -> pd.DataFrame:
    """Kurtosis removed by each clock within sub-periods (does the clock work in every era?)."""
    if periods is None:
        periods = [("2001-2007", "2001", "2007"), ("2008-2012", "2008", "2012"),
                   ("2013-2018", "2013", "2018")]
    rows = []
    for name, a, b in periods:
        sub = panel.loc[a:b]
        if len(sub) < 250:
            continue
        kr = moments(sub["raw"])["excess_kurtosis"]
        row = {"period": name, "n": len(sub), "k_raw": kr}
        for c in ("volume", "range", "garch", "garch_volume"):
            row[f"share_{c}"] = 1.0 - moments(sub[c])["excess_kurtosis"] / kr
        rows.append(row)
    return pd.DataFrame(rows).set_index("period")


# --------------------------------------------------------------------------- #
# Section 2 — the volume-volatility correlation, contemporaneous vs lagged
# --------------------------------------------------------------------------- #
def hac_slope(y: np.ndarray, x: np.ndarray, lags: int | None = None) -> dict:
    """OLS slope of ``y`` on ``x`` with Newey–West standard error (intercept included)."""
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = y.size
    if n < 60:
        return {"beta": np.nan, "t": np.nan, "n": n}
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    u = X * e[:, None]
    S = u.T @ u
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    return {"beta": float(beta[1]), "t": float(beta[1] / se) if se > 0 else np.nan,
            "n": int(n), "lags": lags}


def volume_vol_correlation(tape: pd.DataFrame, window: int = 63,
                           lags=(0, 1, 5)) -> pd.DataFrame:
    """Correlation of log relative volume with |return| and with log range-variance.

    ``lag = 0`` is the contemporaneous relation the clock theory lives on; ``lag = k`` pairs
    volume at ``t`` with volatility at ``t + k`` — the only version usable for a forecast.
    Correlations are of standardised series, so the HAC slope *is* the correlation and its
    Newey–West t-statistic is robust to the clustering both sides share.
    """
    r = log_returns(tape["close"])
    lv = np.log(relative_volume(tape["volume"], window))
    pk = parkinson_var(tape["high"], tape["low"])
    absr = r.abs()
    lpk = np.log(pk.where(pk > 0))
    rows = []
    for k in lags:
        for name, target in (("|r|", absr), ("log range var", lpk)):
            df = pd.concat([lv.rename("x"), target.shift(-k).rename("y")], axis=1).dropna()
            x = ((df["x"] - df["x"].mean()) / df["x"].std(ddof=0)).to_numpy()
            y = ((df["y"] - df["y"].mean()) / df["y"].std(ddof=0)).to_numpy()
            hs = hac_slope(y, x)
            rows.append({"lag": int(k), "target": name, "corr": hs["beta"],
                         "hac_t": hs["t"],
                         "spearman": float(df["x"].rank().corr(df["y"].rank())),
                         "n": hs["n"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Section 3 — does LAGGED volume improve the forecast?
# --------------------------------------------------------------------------- #
def har_design(tape: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Log-HAR regressors at close ``t`` (Corsi 2009), with Parkinson variance as the RV input.

    ``ld, lw, lm`` are logs of the day / week / month average range-variance; ``lv`` is log
    relative volume of day ``t`` — the volume clock as last read, available at the close.
    """
    pk = parkinson_var(tape["high"], tape["low"])
    pk = pk.where(pk > 0)
    return pd.DataFrame({
        "ld": np.log(pk),
        "lw": np.log(pk.rolling(5).mean()),
        "lm": np.log(pk.rolling(22).mean()),
        "lv": np.log(relative_volume(tape["volume"], window)),
    })


HAR_COLS = ("ld", "lw", "lm")
HARX_COLS = ("ld", "lw", "lm", "lv")


def forecast_race(tape: pd.DataFrame, burn: int = 756, refit: int = 21,
                  garch_refit: int = 252, window: int = 63) -> pd.DataFrame:
    """Out-of-sample next-day variance forecasts from GARCH, log-HAR and log-HAR + volume.

    Row ``t`` holds forecasts made at the close of ``t`` for day ``t+1`` and the realised
    targets for ``t+1``: the squared close-to-close return ``r2`` (the variance the overlay
    actually bears) and the Parkinson range variance ``pk`` (a less noisy proxy that misses the
    overnight gap). The HAR models are log regressions refitted every ``refit`` sessions on an
    expanding window of rows whose target was already observed; each prediction is mapped back
    to a variance level with a past-only calibration factor (``mean(target) / mean(exp(fit))``
    on the training rows), separately for each target, so HAR and HAR-X are put on exactly the
    same footing and the only difference between them is the volume regressor. GARCH forecasts
    the squared return directly; for the range target it is rescaled by the training-window
    ratio ``mean(pk) / mean(r2)``.
    """
    r = log_returns(tape["close"])
    r2 = (r ** 2).to_numpy()
    X = har_design(tape, window)
    pk = parkinson_var(tape["high"], tape["low"]).to_numpy()
    n = len(tape)
    y_pk_next = np.concatenate([np.log(np.where(pk > 0, pk, np.nan))[1:], [np.nan]])
    r2_next = np.concatenate([r2[1:], [np.nan]])
    pk_next = np.concatenate([pk[1:], [np.nan]])
    g = garch_forecasts(r, burn=min(burn, 504), refit=garch_refit).to_numpy()

    out = {k: np.full(n, np.nan) for k in
           ("har_r2", "harx_r2", "har_pk", "harx_pk", "garch_r2", "garch_pk")}
    for cols, tag in ((HAR_COLS, "har"), (HARX_COLS, "harx")):
        A = np.column_stack([np.ones(n), X[list(cols)].to_numpy()])
        okX = np.isfinite(A).all(axis=1)
        for t0 in range(burn, n, refit):
            t1 = min(n, t0 + refit)
            # training rows s < t0: their target (day s+1) is known by the close of t0
            tr = np.arange(t0)
            tr = tr[okX[tr] & np.isfinite(y_pk_next[tr])]
            if tr.size < 250:
                continue
            beta = np.linalg.lstsq(A[tr], y_pk_next[tr], rcond=None)[0]
            fit_tr = np.exp(A[tr] @ beta)
            c_r2 = np.nanmean(r2_next[tr]) / np.mean(fit_tr)
            c_pk = np.nanmean(pk_next[tr]) / np.mean(fit_tr)
            blk = np.arange(t0, t1)
            pred = np.where(okX[blk], np.exp(A[blk] @ np.where(np.isfinite(beta), beta, 0)),
                            np.nan)
            out[f"{tag}_r2"][blk] = c_r2 * pred
            out[f"{tag}_pk"][blk] = c_pk * pred
    for t0 in range(burn, n, garch_refit):
        t1 = min(n, t0 + garch_refit)
        tr = np.arange(1, t0)
        ratio = np.nanmean(pk[tr]) / np.nanmean(r2[tr])
        out["garch_r2"][t0:t1] = g[t0:t1]
        out["garch_pk"][t0:t1] = g[t0:t1] * ratio
    df = pd.DataFrame(out, index=tape.index)
    df["target_r2"] = r2_next
    df["target_pk"] = pk_next
    df.index.name = "date"
    return df.iloc[burn:]


def qlike(actual, forecast) -> np.ndarray:
    """QLIKE loss in Patton's (2011) form ``a/f + log f`` — robust to a noisy proxy and
    well-defined at ``a = 0`` (a flat day), unlike the ``a/f - log(a/f) - 1`` form. Loss
    *differences* are identical under both forms."""
    f = np.maximum(np.asarray(forecast, dtype=float), 1e-14)
    return np.asarray(actual, dtype=float) / f + np.log(f)


def diebold_mariano(loss_a, loss_b, lags: int | None = None) -> dict:
    """HAC Diebold–Mariano on ``loss_a - loss_b``. Negative t means model A is better."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 30:
        return {"dm": np.nan, "p": np.nan, "n": int(n), "mean_diff": np.nan}
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    e = d - d.mean()
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        lrv += 2.0 * (1.0 - k / (lags + 1.0)) * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    dm = float(d.mean() / se) if se > 0 else np.nan
    p = float(erfc(abs(dm) / sqrt(2.0))) if np.isfinite(dm) else np.nan
    return {"dm": dm, "p": p, "n": int(n), "mean_diff": float(d.mean()), "lags": lags}


def score_race(fc: pd.DataFrame) -> pd.DataFrame:
    """Mean QLIKE / MSE for each model and target, and DM tests of HAR-X vs HAR and vs GARCH.

    Every comparison is made on the rows where all three models have a forecast, so a model is
    never scored on an easier subset.
    """
    rows = []
    for tgt in ("r2", "pk"):
        cols = [f"har_{tgt}", f"harx_{tgt}", f"garch_{tgt}", f"target_{tgt}"]
        d = fc[cols].dropna()
        a = d[f"target_{tgt}"].to_numpy()
        L = {m: {"qlike": qlike(a, d[f"{m}_{tgt}"]),
                 "mse": (a - d[f"{m}_{tgt}"].to_numpy()) ** 2}
             for m in ("har", "harx", "garch")}
        for loss in ("qlike", "mse"):
            for other in ("har", "garch"):
                dm = diebold_mariano(L["harx"][loss], L[other][loss])
                base = float(np.mean(L[other][loss]))
                rows.append({
                    "target": tgt, "loss": loss, "vs": other, "n": dm["n"],
                    "loss_harx": float(np.mean(L["harx"][loss])), "loss_other": base,
                    "mean_diff": dm["mean_diff"], "dm_t": dm["dm"], "p": dm["p"],
                    # MSE: proportional change; QLIKE: raw difference (level is arbitrary)
                    "rel_change": (float(np.mean(L["harx"][loss])) / base - 1.0)
                    if loss == "mse" and base > 0 else np.nan,
                })
    return pd.DataFrame(rows)


def har_in_sample(tape: pd.DataFrame, window: int = 63) -> dict:
    """Full-sample log-HAR-X with Newey–West t on the volume coefficient (in-sample check)."""
    X = har_design(tape, window)
    pk = parkinson_var(tape["high"], tape["low"])
    y = np.log(pk.where(pk > 0)).shift(-1)
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    A = np.column_stack([np.ones(len(df)), df[list(HARX_COLS)].to_numpy()])
    yy = df["y"].to_numpy()
    beta = np.linalg.lstsq(A, yy, rcond=None)[0]
    e = yy - A @ beta
    n = len(df)
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    XtX_inv = np.linalg.inv(A.T @ A)
    u = A * e[:, None]
    S = u.T @ u
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    se = np.sqrt(np.diag(XtX_inv @ S @ XtX_inv))
    A0 = A[:, :4]
    e0 = yy - A0 @ np.linalg.lstsq(A0, yy, rcond=None)[0]
    tss = float(((yy - yy.mean()) ** 2).sum())
    return {"beta_lv": float(beta[4]), "t_lv": float(beta[4] / se[4]),
            "r2_har": 1.0 - float(e0 @ e0) / tss, "r2_harx": 1.0 - float(e @ e) / tss,
            "n": int(n)}


# --------------------------------------------------------------------------- #
# Section 4 — the vol-targeting overlay
# --------------------------------------------------------------------------- #
EXECUTION_LAG = 1   # sessions between the close the forecast is formed at and the trade


def overlay(r_simple: pd.Series, var_fc: pd.Series, rf: pd.Series,
            target_ann: float = 0.15, cap: float = 1.5, cost_bps: float = 2.0) -> pd.DataFrame:
    """Vol-targeting overlay on the index with **one execution lag**.

    ``var_fc`` at date ``t`` is the variance forecast for ``t+1`` formed at the close of ``t``.
    The weight ``min(cap, target / sigma_hat)`` is *traded at the close of t+1* — the day's
    composite volume is only published after the close of ``t``, so trading at that same close
    would use information nobody had — and therefore earns the return of ``t+2``. The rest of
    NAV earns the cash rate (borrowing at cash when the weight exceeds one — optimistic for a
    retail account, identical for every overlay compared). Costs are ``cost_bps`` one-way on
    traded NAV, ``|Δweight|``. Returns are **price-only** (the tapes exclude dividends), so
    levels understate total return by the dividend yield; comparisons between overlays do not.
    """
    tgt_d = target_ann / np.sqrt(TRADING_DAYS)
    w = np.minimum(cap, tgt_d / np.sqrt(var_fc.clip(lower=1e-12)))
    w = w.ffill()
    pos = w.shift(1 + EXECUTION_LAG).reindex(r_simple.index)
    ex = r_simple - rf.reindex(r_simple.index)
    turn = pos.diff().abs()
    gross = pos * ex
    net = gross - turn * cost_bps / 1e4
    return pd.DataFrame({"pos": pos, "turnover": turn, "gross": gross, "net": net,
                         "bh": ex}).dropna()


def sharpe(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan


def sharpe_diff_test(a: pd.Series, b: pd.Series, n_boot: int = 1000, block: int = 21,
                     seed: int = 1019) -> dict:
    """Sharpe(a) - Sharpe(b) on paired daily excess returns, circular block bootstrap.

    Pairs are resampled together so the strong correlation between two overlays on the same
    index is respected; the p-value is two-sided, from the share of resamples on the other
    side of zero.
    """
    d = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    A, B = d["a"].to_numpy(), d["b"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = _cbb_indices(len(d), n_boot, block, rng)

    def sr(M):
        return M.mean(axis=1) / M.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS)

    diff_b = sr(A[idx]) - sr(B[idx])
    diff = sharpe(A) - sharpe(B)
    p = float(min(1.0, 2.0 * min(np.mean(diff_b <= 0), np.mean(diff_b >= 0))))
    return {"diff": float(diff), "ci_lo": float(np.quantile(diff_b, 0.025)),
            "ci_hi": float(np.quantile(diff_b, 0.975)), "p": p, "n": int(len(d))}


def overlay_race(tape: pd.DataFrame, fc: pd.DataFrame, rf: pd.Series,
                 costs_bps=(0.0, 2.0, 5.0, 10.0), target_ann: float = 0.15,
                 cap: float = 1.5, n_boot: int = 1000, seed: int = 1019) -> dict:
    """HAR-X overlay vs HAR and GARCH overlays (and buy-and-hold), gross and across costs.

    All overlays are scored on the same sessions. The decisive line is HAR-X minus HAR net of
    costs: that difference *is* the value of the lagged volume clock.
    """
    r_s = tape["close"].pct_change().rename("r")
    ov = {m: overlay(r_s, fc[f"{m}_r2"], rf, target_ann, cap, 0.0)
          for m in ("har", "harx", "garch")}
    common = ov["har"].index.intersection(ov["harx"].index).intersection(ov["garch"].index)
    rows, tests = [], []
    for c in costs_bps:
        net = {m: (ov[m]["gross"] - ov[m]["turnover"] * c / 1e4).loc[common] for m in ov}
        for m in ov:
            rows.append({"cost_bps": c, "model": m, "sharpe": sharpe(net[m]),
                         "ann_ret": float(net[m].mean() * TRADING_DAYS),
                         "ann_vol": float(net[m].std() * np.sqrt(TRADING_DAYS)),
                         "turnover_ann": float(ov[m]["turnover"].loc[common].mean()
                                               * TRADING_DAYS),
                         "mean_pos": float(ov[m]["pos"].loc[common].mean())})
        for other in ("har", "garch"):
            t = sharpe_diff_test(net["harx"], net[other], n_boot=n_boot, seed=seed)
            tests.append({"cost_bps": c, "vs": other, **t})
    bh = ov["har"]["bh"].loc[common]
    rows.append({"cost_bps": 0.0, "model": "buy & hold", "sharpe": sharpe(bh),
                 "ann_ret": float(bh.mean() * TRADING_DAYS),
                 "ann_vol": float(bh.std() * np.sqrt(TRADING_DAYS)),
                 "turnover_ann": 0.0, "mean_pos": 1.0})
    return {"table": pd.DataFrame(rows), "tests": pd.DataFrame(tests),
            "n": int(len(common)), "start": str(common[0].date()),
            "end": str(common[-1].date())}


def lookahead_clock_sharpe(tape: pd.DataFrame, fc: pd.DataFrame, rf: pd.Series,
                           window: int = 63, target_ann: float = 0.15,
                           cap: float = 1.5) -> dict:
    """The impossible version: scale today's position by **today's** volume clock.

    Two zero-lag overlays on the same sessions: (a) yesterday's HAR forecast alone, and (b) the
    same forecast times today's relative volume (normalised by its past-only mean). Nobody can
    trade (b) — the volume is not known until after the close — and (a) is there so the
    difference isolates the clock rather than mixing it with the removal of the execution lag.
    Gross of costs; reported only to show where in time the clock's value lives.
    """
    r_s = tape["close"].pct_change()
    rv = relative_volume(tape["volume"], window)
    rv_n = rv / rv.shift(1).expanding(63).mean()
    base = fc["har_r2"].shift(1).reindex(tape.index)
    tgt_d = target_ann / np.sqrt(TRADING_DAYS)
    ex = (r_s - rf.reindex(r_s.index))
    pos_a = np.minimum(cap, tgt_d / np.sqrt(base.clip(lower=1e-12)))
    pos_b = np.minimum(cap, tgt_d / np.sqrt((base * rv_n).clip(lower=1e-12)))
    df = pd.DataFrame({"a": pos_a * ex, "b": pos_b * ex}).dropna()
    return {"sharpe_zero_lag": sharpe(df["a"]), "sharpe_gross": sharpe(df["b"]),
            "n": int(len(df))}


# --------------------------------------------------------------------------- #
# The verdict — thresholds fixed before the real tape was run
# --------------------------------------------------------------------------- #
MIN_SHARE_REMOVED = 0.25    # the clock must remove at least a quarter of the excess kurtosis
CORR_T = 2.0                # contemporaneous volume–|r| correlation, Newey–West t
DM_T = 2.0                  # HAR-X must beat HAR on QLIKE (r² target) with |DM| >= 2
SHARPE_P = 0.05             # overlay Sharpe gain, block-bootstrap p


def tape_flags(t: dict) -> dict:
    """Per-tape pass/fail flags read by :func:`verdict`."""
    clock_works = bool(t["share_removed"] >= MIN_SHARE_REMOVED and t["share_ci_lo"] > 0
                       and t["corr0_t"] >= CORR_T)
    forecast_helps = bool(t["dm_t"] <= -DM_T)
    overlay_wins = bool(t["sharpe_diff_net"] > 0 and t["sharpe_diff_p"] < SHARPE_P)
    return {"clock_works": clock_works, "clock_positive": bool(t["share_removed"] > 0),
            "forecast_helps": forecast_helps, "overlay_wins": overlay_wins,
            "overlay_positive": bool(t["sharpe_diff_net"] > 0)}


def verdict(h: dict) -> dict:
    """Stamps by a rule written down before the real run.

    ``h["tapes"]`` maps each tape to a dict with ``share_removed``, ``share_ci_lo``,
    ``corr0_t`` (contemporaneous), ``dm_t`` (HAR-X minus HAR, QLIKE on squared returns —
    negative favours volume), ``sharpe_diff_net`` / ``sharpe_diff_p`` (HAR-X overlay minus HAR
    overlay at the base cost), plus descriptive fields quoted in the prose.

    **Signal** — is the volume clock real?
      * **Real**: on *every* tape the clock removes ≥ 25% of the raw excess kurtosis with a
        block-bootstrap 95% CI above zero, and the contemporaneous volume–|r| correlation has
        a Newey–West t ≥ 2.
      * **Mixed**: that holds on some tapes while on another the clock removes nothing.
      * **Weak**: the clock removes something on every tape but misses the bar on at least one.
      * **None**: otherwise.
      The strong form — *Gaussian* in volume time — is reported (Jarque–Bera) but is not
      required for Real; the stamp certifies the clock, the prose says how far it gets.

    **Tradability** — is the clock usable before the close it is read at?
      * **Investable**: lagged volume improves next-day QLIKE over HAR with DM t ≤ −2 *and*
        the overlay built on it beats the HAR overlay net of costs with bootstrap p < 0.05, on
        every tape.
      * **Fragile**: the forecast gain is significant on at least one tape and the overlay
        gain is positive net of costs on at least one of those.
      * **Mirage**: otherwise — the information is spent by the time anyone can act on it.
    """
    tapes = h["tapes"]
    flags = {k: tape_flags(v) for k, v in tapes.items()}
    works = [f["clock_works"] for f in flags.values()]
    pos = [f["clock_positive"] for f in flags.values()]
    if all(works):
        signal = "Real"
    elif any(works) and not all(pos):
        signal = "Mixed"
    elif all(pos):
        signal = "Weak"
    else:
        signal = "None"

    if all(f["forecast_helps"] and f["overlay_wins"] for f in flags.values()):
        trad = "Investable"
    elif any(f["forecast_helps"] and f["overlay_positive"] for f in flags.values()):
        trad = "Fragile"
    else:
        trad = "Mirage"

    names = list(tapes)
    lab = h.get("labels", {k: k for k in names})

    def each(fmt):
        return "; ".join(f"{lab[k]} {fmt(tapes[k])}" for k in names)

    gauss = all(tapes[k]["jb_p_volume"] > 0.05 for k in names)
    if signal == "Real":
        clock_line = ". The clock is real, and it runs on the same day. "
    elif signal == "Weak":
        clock_line = (". The link is unmistakable but the clock is a weak one: on its own it "
                      f"misses the pre-registered bar (at least {MIN_SHARE_REMOVED:.0%} of the "
                      "excess kurtosis removed, bootstrap CI above zero) on "
                      + ("both tapes" if not any(flags[k]["clock_works"] for k in names)
                         else ", ".join(lab[k] for k in names
                                        if not flags[k]["clock_works"]))
                      + ". ")
    elif signal == "Mixed":
        clock_line = ". The clock works on one tape and does nothing on the other. "
    else:
        clock_line = ". Volume time does not thin the tails here. "
    signal_why = (
        "Divide each day's return by the square root of that day's volume (relative to its own "
        "past-only 63-day median) and the excess kurtosis moves — "
        + each(lambda t: f"from **{t['k_raw']:.1f} to {t['k_volume']:.1f}**, "
               f"{t['share_removed']:.0%} of it removed (block-bootstrap 95% CI "
               f"{t['share_ci_lo']:.0%} to {t['share_ci_hi']:.0%})")
        + ". The correlation of volume with the absolute return is "
        + each(lambda t: f"{t['corr0']:+.2f} on the same day (Newey–West t {t['corr0_t']:.1f}) "
               f"against {t['corr1']:+.2f} with the next day")
        + clock_line
        + ("Nor does it make returns Gaussian: Jarque–Bera still rejects normality in volume "
           "time on every tape" if not gauss else
           "In volume time Jarque–Bera no longer rejects normality")
        + ". The benchmarks bracket it: the day's own range (an upper bound, since it *is* that "
          "day's volatility) removes "
        + each(lambda t: f"{t['share_removed_range']:.0%}")
        + ", a past-only GARCH removes "
        + each(lambda t: f"{t['share_removed_garch']:.0%}")
        + ", and GARCH times same-day volume *surprise* removes "
        + each(lambda t: f"{t['share_removed_garch_volume']:.0%}")
        + (" — a further reduction over GARCH alone whose bootstrap CI excludes zero on "
           + ", ".join(lab[k] for k in names if tapes[k].get("inc_ci_lo", -1) > 0)
           if any(tapes[k].get("inc_ci_lo", -1) > 0 for k in names)
           else " — not distinguishable from GARCH alone")
        + ". Most of the fat tail is volatility clustering, which a past-only model already "
          "sees; the volume clock adds a same-day sliver on top.")
    trad_why = (
        "The clock is read at the close of the day it explains, so the usable question is "
        "whether **yesterday's** volume sharpens **tomorrow's** variance forecast beyond a "
        "range-based log-HAR. Out of sample (monthly refits, expanding window), adding lagged "
        "relative volume moves next-day QLIKE on squared returns with a Diebold–Mariano t of "
        + each(lambda t: f"**{t['dm_t']:+.2f}** (p {t['dm_p']:.2f})")
        + " — negative would favour volume. Against the range-variance target the DM t is "
        + each(lambda t: f"{t['dm_t_pk']:+.2f}")
        + ". Built into a 15%-target vol overlay traded one session after the forecast, at "
        f"{h.get('base_cost_bps', 2.0):.0f} bp one-way, the volume-aware overlay's Sharpe minus "
        "the plain HAR overlay's is "
        + each(lambda t: f"**{t['sharpe_diff_net']:+.3f}** (bootstrap p "
               f"{t['sharpe_diff_p']:.2f})")
        + ". (The range-based HAR is itself the better forecaster — against GARCH its QLIKE "
          "DM t is "
        + each(lambda t: f"{t['dm_t_vs_garch']:+.1f}")
        + " — so the benchmark being beaten is not a straw man.) The impossible version — "
          "sizing today's position by today's volume, which nobody has until after the close — "
          "moves the gross Sharpe of a zero-lag HAR overlay from "
        + each(lambda t: f"{t['lookahead_base_sharpe']:.2f} to {t['lookahead_sharpe']:.2f}")
        + ": whatever the clock is worth sits on the wrong side of the close.")
    lo = min(tapes[k]["share_removed"] for k in names)
    hi = max(tapes[k]["share_removed"] for k in names)
    head = {"Real": "Volume time is real",
            "Weak": "Volume and volatility move together, but volume time is a weak clock",
            "Mixed": "Volume time works on one tape and not the other",
            "None": "Volume time does not thin the tails"}[signal]
    tail = {"Mirage": "and by the time anyone can read the clock it is spent — lagged volume "
                      "adds nothing significant to a range-based HAR forecast and nothing "
                      "to a vol-targeting overlay.",
            "Fragile": "and once it is lagged a day it adds only a thin, unbankable edge to "
                       "a range-based HAR forecast.",
            "Investable": "and even lagged a day it pays its way in a vol-targeting overlay "
                          "after costs."}[trad]
    one = (f"{head}: dividing by same-day volume removes {lo:.0%}–{hi:.0%} of the excess "
           f"kurtosis of daily index returns{'' if gauss else ' and leaves them far from normal'}"
           f", {tail}")
    return {"signal": signal, "signal_why": signal_why, "trad": trad, "trad_why": trad_why,
            "one_sentence": one}
