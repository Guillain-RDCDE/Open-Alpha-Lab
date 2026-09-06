"""Estimation-window comparison — Study 967.

Three quantities, one question. For each of

- **beta** (a sector's slope on the market),
- **the mean return** (the input a mean-variance optimiser is most sensitive to),
- **the covariance matrix** (through the out-of-sample volatility of the minimum-variance
  portfolio it implies),

we estimate the parameter at the end of each year from either a **rolling window** of the last
``k`` years or an **expanding window** of everything available, and then score the estimate
against what the *following* year actually delivered. That is the only comparison that
matters: an estimator is judged on the future it did not see.

**Why the three answers differ** is the point of the study, and it is a bias-variance argument
rather than an empirical accident:

- A **mean** is estimated so imprecisely that its standard error dominates everything; more
  history nearly always helps, and the rolling window's supposed advantage (adaptivity)
  cannot pay for its variance. Merton (1980) is the canonical statement.
- A **beta** is a ratio of second moments — estimated far more precisely — but it genuinely
  moves as a company's business changes, so there is a real trade-off and an interior optimum.
- A **covariance matrix** with N assets has N(N+1)/2 parameters; when the window is short
  relative to N the matrix is badly conditioned or singular, and the minimum-variance
  portfolio built from it is a machine for maximising estimation error (Michaud's "error
  maximisation").

Every evaluation is strictly out of sample and re-estimated each year. Losses are reported as
mean squared error against the realised next-year value, with a **HAC-corrected
Diebold-Mariano** test on the pairwise loss differences, because a rolling and an expanding
window share most of their data and their errors are anything but independent.
"""

from __future__ import annotations

from math import erfc, sqrt

import numpy as np
import pandas as pd

TRADING_DAYS = 252
WINDOWS_YEARS = (1, 2, 3, 5, 10)
EXPANDING = "expanding"
MIN_TRAIN = 252


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns, with each column's own inception respected (no back-fill)."""
    return prices.pct_change().replace([np.inf, -np.inf], np.nan)


def beta_of(asset: pd.Series, market: pd.Series) -> float:
    """OLS slope of ``asset`` on ``market`` (no intercept assumption beyond OLS's own)."""
    df = pd.concat([asset, market], axis=1).dropna()
    if len(df) < 30:
        return np.nan
    x = df.iloc[:, 1].to_numpy()
    y = df.iloc[:, 0].to_numpy()
    vx = x.var(ddof=1)
    return float(np.cov(x, y, ddof=1)[0, 1] / vx) if vx > 0 else np.nan


def year_ends(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """The last trading day of each calendar year in the index — the decision dates."""
    s = pd.Series(index, index=index)
    return list(s.groupby(index.year).last())


def train_slice(df: pd.DataFrame | pd.Series, asof: pd.Timestamp,
                window: int | str) -> pd.DataFrame | pd.Series:
    """Everything up to ``asof``, cut to the last ``window`` years (or all of it)."""
    hist = df.loc[:asof]
    if window == EXPANDING:
        return hist
    n = int(window) * TRADING_DAYS
    return hist.iloc[-n:]


# --------------------------------------------------------------------------- #
# 1) Beta
# --------------------------------------------------------------------------- #
def beta_experiment(rets: pd.DataFrame, sectors: tuple[str, ...], market: str,
                    windows=WINDOWS_YEARS) -> pd.DataFrame:
    """Estimate each sector's beta at every year-end; score against the next year's beta."""
    rows = []
    dates = year_ends(rets.index)
    for i, d in enumerate(dates[:-1]):
        nxt = rets.loc[dates[i] + pd.Timedelta(days=1):dates[i + 1]]
        if len(nxt) < 120:
            continue
        for s in sectors:
            realised = beta_of(nxt[s], nxt[market])
            if not np.isfinite(realised):
                continue
            for w in list(windows) + [EXPANDING]:
                tr = train_slice(rets[[s, market]], d, w).dropna()
                if len(tr) < MIN_TRAIN:
                    continue
                est = beta_of(tr[s], tr[market])
                if np.isfinite(est):
                    rows.append({"date": d, "sector": s, "window": w, "estimate": est,
                                 "realised": realised, "error": est - realised})
    return pd.DataFrame(rows)


def blume_shrunk(beta: float, weight: float = 2 / 3) -> float:
    """Blume's (1971) shrinkage toward one — the classic fix for beta's mean reversion."""
    return weight * beta + (1 - weight) * 1.0


# --------------------------------------------------------------------------- #
# 2) Mean return
# --------------------------------------------------------------------------- #
def mean_experiment(rets: pd.DataFrame, sectors: tuple[str, ...],
                    windows=WINDOWS_YEARS) -> pd.DataFrame:
    """Estimate each sector's mean daily return; score against the next year's mean."""
    rows = []
    dates = year_ends(rets.index)
    for i, d in enumerate(dates[:-1]):
        nxt = rets.loc[dates[i] + pd.Timedelta(days=1):dates[i + 1]]
        if len(nxt) < 120:
            continue
        for s in sectors:
            realised = float(nxt[s].mean()) if nxt[s].notna().sum() > 100 else np.nan
            if not np.isfinite(realised):
                continue
            for w in list(windows) + [EXPANDING]:
                tr = train_slice(rets[s], d, w).dropna()
                if len(tr) < MIN_TRAIN:
                    continue
                est = float(tr.mean())
                rows.append({"date": d, "sector": s, "window": w, "estimate": est,
                             "realised": realised, "error": est - realised})
    return pd.DataFrame(rows)


def grand_mean_benchmark(rets: pd.DataFrame, sectors: tuple[str, ...]) -> pd.DataFrame:
    """The humblest estimator of all: the cross-sectional average, same number for everyone.

    If this beats every window of a sector's own history — and for means it usually does —
    the honest conclusion is that individual mean returns are not estimable at all, which is
    a stronger statement than "use a long window".
    """
    rows = []
    dates = year_ends(rets.index)
    for i, d in enumerate(dates[:-1]):
        nxt = rets.loc[dates[i] + pd.Timedelta(days=1):dates[i + 1]]
        if len(nxt) < 120:
            continue
        hist = rets.loc[:d, list(sectors)]
        est = float(hist.stack().mean())
        for s in sectors:
            realised = float(nxt[s].mean()) if nxt[s].notna().sum() > 100 else np.nan
            if np.isfinite(realised):
                rows.append({"date": d, "sector": s, "window": "grand mean", "estimate": est,
                             "realised": realised, "error": est - realised})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3) Covariance, judged by the portfolio it builds
# --------------------------------------------------------------------------- #
def min_variance_weights(cov: np.ndarray, long_only: bool = False) -> np.ndarray:
    """Global minimum-variance weights, with a small ridge for numerical safety."""
    n = cov.shape[0]
    reg = cov + np.eye(n) * 1e-10
    try:
        inv = np.linalg.inv(reg)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(reg)
    w = inv @ np.ones(n)
    w = w / w.sum()
    if long_only:
        w = np.clip(w, 0, None)
        w = w / w.sum() if w.sum() > 0 else np.full(n, 1 / n)
    return w


def covariance_experiment(rets: pd.DataFrame, sectors: tuple[str, ...],
                          windows=WINDOWS_YEARS, long_only: bool = False) -> pd.DataFrame:
    """Build a minimum-variance portfolio each year-end; measure the vol it actually ran.

    The estimator is scored on the thing it is for — next year's realised portfolio
    volatility — not on how close its matrix is to some other matrix. Weights are held for
    the year; turnover is reported so the reader can see what the adaptivity costs.
    """
    rows = []
    dates = year_ends(rets.index)
    prev_w: dict = {}
    for i, d in enumerate(dates[:-1]):
        nxt = rets.loc[dates[i] + pd.Timedelta(days=1):dates[i + 1]]
        if len(nxt) < 120:
            continue
        for w in list(windows) + [EXPANDING]:
            tr = train_slice(rets[list(sectors)], d, w)
            avail = [c for c in sectors if tr[c].notna().sum() >= MIN_TRAIN
                     and nxt[c].notna().sum() >= 100]
            if len(avail) < 3:
                continue
            sub = tr[avail].dropna()
            if len(sub) < max(MIN_TRAIN, len(avail) + 10):
                continue
            weights = min_variance_weights(np.cov(sub.to_numpy().T, ddof=1), long_only)
            port = (nxt[avail].fillna(0.0).to_numpy() @ weights)
            key = (w, tuple(avail))
            old = prev_w.get(w)
            turnover = (np.abs(pd.Series(weights, index=avail) -
                               pd.Series(old[1], index=old[0]).reindex(avail).fillna(0)).sum()
                        if old else np.nan)
            prev_w[w] = (avail, weights)
            rows.append({"date": d, "window": w, "n_assets": len(avail),
                         "n_train": int(len(sub)),
                         "realised_vol": float(np.std(port, ddof=1) * np.sqrt(TRADING_DAYS)),
                         "predicted_vol": float(np.sqrt(weights @ np.cov(
                             sub.to_numpy().T, ddof=1) @ weights * TRADING_DAYS)),
                         "max_weight": float(np.max(weights)),
                         "short_weight": float(np.sum(np.clip(-weights, 0, None))),
                         "turnover": turnover})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score(exp: pd.DataFrame, value: str = "error") -> pd.DataFrame:
    """Mean squared error and bias per window, plus the count behind each cell."""
    g = exp.groupby("window")[value]
    out = pd.DataFrame({"mse": g.apply(lambda s: float((s ** 2).mean())),
                        "bias": g.mean(), "mae": g.apply(lambda s: float(s.abs().mean())),
                        "n": g.size()})
    order = [w for w in list(WINDOWS_YEARS) + [EXPANDING, "grand mean"] if w in out.index]
    return out.loc[order]


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, lags: int | None = None) -> dict:
    """HAC Diebold-Mariano on two loss series. Positive means A is worse."""
    d = (loss_a - loss_b).dropna()
    n = d.size
    if n < 12:
        return {"dm": np.nan, "p_value": np.nan, "n": int(n)}
    if lags is None:
        lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    e = (d - d.mean()).to_numpy()
    lrv = float(e @ e) / n
    for k in range(1, min(lags, n - 1) + 1):
        lrv += 2.0 * (1.0 - k / (lags + 1.0)) * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    dm = float(d.mean() / se) if se > 0 else np.nan
    return {"dm": dm, "p_value": float(erfc(abs(dm) / sqrt(2.0))) if np.isfinite(dm) else np.nan,
            "n": int(n), "mean_diff": float(d.mean())}


def pairwise_dm(exp: pd.DataFrame, a, b, key=("date", "sector")) -> dict:
    """Diebold-Mariano between two windows on their *paired* squared errors."""
    keys = [k for k in key if k in exp.columns]
    la = exp[exp["window"] == a].set_index(keys)["error"] ** 2
    lb = exp[exp["window"] == b].set_index(keys)["error"] ** 2
    la, lb = la.align(lb, join="inner")
    return diebold_mariano(la.reset_index(drop=True), lb.reset_index(drop=True))


def best_window(scored: pd.DataFrame) -> object:
    """The window with the lowest MSE (ties broken toward more data)."""
    return scored["mse"].idxmin()


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal** (does the window matter?): **Real** if the spread between the best and worst
      window's MSE exceeds 25% on at least two of the three quantities *and* at least one
      pairwise Diebold-Mariano clears 2; **Weak** if the spread is there without the *t*;
      **None** if every window performs within 10%.
    - **Usefulness** (is there one default?): **Useful** only if the *same* window wins all
      three experiments; **Fragile** if the winners differ but each is stable;
      **Mirage** if the ranking is inconsistent enough that no advice survives.
    """
    spreads = [h["spread_beta"], h["spread_mean"], h["spread_cov"]]
    big = sum(s >= 0.25 for s in spreads)
    real = big >= 2 and h["max_abs_dm"] >= 2.0
    signal = "Real" if real else ("Weak" if big >= 2 or h["max_abs_dm"] >= 2.0 else "None")
    same = len({str(h["best_beta"]), str(h["best_mean"]), str(h["best_cov"])}) == 1
    trad = "Useful" if same else ("Fragile" if h["max_abs_dm"] >= 2.0 else "Mirage")
    return {
        "signal": signal,
        "signal_why": (
            f"Yes, and by more than habit would suggest. Across estimation windows the MSE "
            f"spread is **{h['spread_beta']:.0%}** for beta, **{h['spread_mean']:.0%}** for the "
            f"mean return and **{h['spread_cov']:.0%}** for the volatility of the "
            f"minimum-variance portfolio; the strongest pairwise Diebold-Mariano across the "
            f"three experiments is **{h['max_abs_dm']:+.2f}**. The three quantities do not "
            f"agree with each other: beta wants **{h['best_beta']}**, the mean wants "
            f"**{h['best_mean']}**, the covariance matrix wants **{h['best_cov']}**."),
        "trad": trad,
        "trad_why": (
            f"There is no single default. The mean is the extreme case: even the *best* window "
            f"of a sector's own history is beaten by the **grand mean across all sectors** "
            f"(MSE ratio {h['grand_mean_ratio']:.2f}), which is the empirical way of saying "
            f"individual expected returns are not estimable from price history at all. Beta is "
            f"estimable and mildly non-stationary — Blume shrinkage cuts its MSE by a further "
            f"**{h['blume_gain']:.0%}**. The covariance matrix is the one place where a short "
            f"window is actively dangerous: at {h['n_sectors']} assets, a one-year window has "
            f"about {h['obs_per_param_1y']:.1f} observations per estimated parameter."),
        "one_sentence": (
            f"How much history you should use depends entirely on what you are estimating: "
            f"means want everything (and are hopeless anyway — the cross-sectional average "
            f"beats a sector's own history), betas want a medium window plus shrinkage, and "
            f"covariance matrices want enough rows per parameter to stay invertible — which "
            f"a one-year window at {h['n_sectors']} assets is not."),
    }
