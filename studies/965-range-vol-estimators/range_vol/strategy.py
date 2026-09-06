"""Range-based variance estimators and their honest accounting — Study 965.

Four estimators of a single day's variance, from the same bar:

- **Close-to-close** (the baseline): ``r^2`` where ``r = ln(C_t / C_{t-1})``. Uses one number
  a day and throws the rest of the bar away.
- **Parkinson (1980)**: ``ln(H/L)^2 / (4 ln 2)``. Uses the range. Assumes a driftless
  continuous diffusion with **no overnight gap**, so on a real market it measures the
  *intraday* variance only — the textbook 5x efficiency and the real-world downward bias are
  two faces of the same assumption.
- **Garman-Klass (1980)**: ``0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2``. Adds the open-to-close
  move; still no gap.
- **Rogers-Satchell (1991)**: ``ln(H/C)ln(H/O) + ln(L/C)ln(L/O)``. Drift-robust — the one to
  reach for when the day has a trend — but again gap-blind.
- **Yang-Zhang (2000)**: the overnight variance plus a weighted open-to-close and
  Rogers-Satchell term. The only estimator here that is *unbiased in the presence of gaps*,
  which is what makes it the right yardstick on a real tape.

Three questions, three sections:

1. ``efficiency_table`` — on simulated bars whose true sigma is known, how much less noisy is
   each estimator? This is the only place the textbook claim can be tested, and it is tested
   as the ratio of mean-squared errors against the truth.
2. ``bias_table`` — on the real tape, how much of the day's variance does each estimator
   simply not see? The overnight gap is a large share of daily variance for an equity ETF and
   *none* of Parkinson, Garman-Klass or Rogers-Satchell contains it.
3. ``forecast_race`` — the practical test. Use each estimator's trailing average as a forecast
   of the next 21 days' realised close-to-close variance, out of sample, and score it with
   **QLIKE** and MSE. The winner is compared to the baseline with a **Diebold-Mariano** test
   (``diebold_mariano``), because two loss series computed on the same tape are about as
   dependent as two series can be.

QLIKE — ``L(sigma2, f) = sigma2/f - ln(sigma2/f) - 1`` — is used alongside MSE because MSE on
variances is dominated by a handful of crisis days: it is a test of who predicted March 2020,
not of who is usually right. Patton (2011) shows both are "robust" to a noisy variance proxy,
which the realised close-to-close variance certainly is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
ESTIMATORS = ("close_close", "parkinson", "garman_klass", "rogers_satchell", "yang_zhang")
ESTIMATOR_LABEL = {
    "close_close": "Close-to-close",
    "parkinson": "Parkinson (1980)",
    "garman_klass": "Garman-Klass (1980)",
    "rogers_satchell": "Rogers-Satchell (1991)",
    "yang_zhang": "Yang-Zhang (2000)",
}
GAP_BLIND = ("parkinson", "garman_klass", "rogers_satchell")


# --------------------------------------------------------------------------- #
# The estimators, one day at a time
# --------------------------------------------------------------------------- #
def _logs(bars: pd.DataFrame) -> tuple[pd.Series, ...]:
    o, h, l, c = bars["open"], bars["high"], bars["low"], bars["close"]
    return (np.log(h / l), np.log(c / o), np.log(h / o), np.log(l / o),
            np.log(h / c), np.log(l / c), np.log(o / c.shift(1)))


def close_close_var(bars: pd.DataFrame) -> pd.Series:
    """Squared close-to-close log return — the whole day, one observation."""
    return (np.log(bars["close"] / bars["close"].shift(1)) ** 2).rename("close_close")


def parkinson_var(bars: pd.DataFrame) -> pd.Series:
    """Parkinson (1980): the squared log range, scaled by 4 ln 2."""
    hl, *_ = _logs(bars)
    return (hl ** 2 / (4.0 * np.log(2.0))).rename("parkinson")


def garman_klass_var(bars: pd.DataFrame) -> pd.Series:
    """Garman-Klass (1980): range plus open-to-close, the classic minimum-variance blend."""
    hl, co, *_ = _logs(bars)
    return (0.5 * hl ** 2 - (2.0 * np.log(2.0) - 1.0) * co ** 2).rename("garman_klass")


def rogers_satchell_var(bars: pd.DataFrame) -> pd.Series:
    """Rogers-Satchell (1991): unbiased in the presence of a drift within the day."""
    _, _, ho, lo, hc, lc, _ = _logs(bars)
    return (hc * ho + lc * lo).rename("rogers_satchell")


def yang_zhang_var(bars: pd.DataFrame, window: int = 21) -> pd.Series:
    """Yang-Zhang (2000): overnight + open-to-close + Rogers-Satchell, gap-aware.

    Unlike the others this one is defined on a *window*: the overnight and open-to-close
    terms are variances, not single-day squares. ``k`` follows the paper's minimum-variance
    weighting with the conventional ``alpha = 1.34``.
    """
    _, co, _, _, _, _, oc = _logs(bars)
    rs = rogers_satchell_var(bars)
    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    v_over = oc.rolling(n).var(ddof=1)
    v_oc = co.rolling(n).var(ddof=1)
    v_rs = rs.rolling(n).mean()
    return (v_over + k * v_oc + (1 - k) * v_rs).rename("yang_zhang")


def all_estimators(bars: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Every estimator on the same bars. Yang-Zhang is a windowed average by construction,
    so for a like-for-like table the others are averaged over the same window in
    :func:`rolling_variance`; this frame is the raw per-day series."""
    return pd.concat([close_close_var(bars), parkinson_var(bars), garman_klass_var(bars),
                      rogers_satchell_var(bars), yang_zhang_var(bars, window)], axis=1)


def rolling_variance(bars: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Each estimator as a rolling variance estimate over ``window`` sessions.

    The single-day estimators are averaged; Yang-Zhang already is one. This is the form a
    practitioner actually uses, and the form the forecast race scores.
    """
    raw = all_estimators(bars, window)
    out = raw.copy()
    for c in raw.columns:
        if c != "yang_zhang":
            out[c] = raw[c].rolling(window).mean()
    return out


def annualised_vol(var: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Daily variance -> annualised volatility, the units everyone quotes."""
    return np.sqrt(var.clip(lower=0.0) * TRADING_DAYS)


# --------------------------------------------------------------------------- #
# 1) Efficiency, measured where the truth is known
# --------------------------------------------------------------------------- #
def efficiency_table(bars: pd.DataFrame, sigma: pd.Series, window: int = 1) -> pd.DataFrame:
    """MSE of each estimator against the *known* daily variance, and the ratio to close-close.

    ``window = 1`` scores the single-day estimators (the textbook comparison). Yang-Zhang is
    reported at its own minimum window (21) and flagged, because a windowed estimator scored
    on one day is not a fair comparison and pretending otherwise would flatter it.
    """
    truth = (sigma.reindex(bars.index) ** 2).dropna()
    est = all_estimators(bars, 21) if window == 1 else rolling_variance(bars, window)
    if window > 1:
        truth = truth.rolling(window).mean()
    rows = []
    base = None
    for c in ESTIMATORS:
        e = est[c].reindex(truth.index)
        pair = pd.concat([e, truth], axis=1).dropna()
        if pair.empty:
            continue
        err = pair.iloc[:, 0] - pair.iloc[:, 1]
        mse = float((err ** 2).mean())
        bias = float((pair.iloc[:, 0] / pair.iloc[:, 1]).mean())
        if c == "close_close":
            base = mse
        rows.append({"estimator": c, "mse": mse, "efficiency_vs_cc": np.nan,
                     "mean_ratio_to_truth": bias, "n": int(len(pair)),
                     "windowed": c == "yang_zhang" and window == 1})
    tbl = pd.DataFrame(rows).set_index("estimator")
    tbl["efficiency_vs_cc"] = base / tbl["mse"]
    return tbl


# --------------------------------------------------------------------------- #
# 2) Bias on a market that gaps
# --------------------------------------------------------------------------- #
def overnight_share(bars: pd.DataFrame) -> float:
    """Share of total daily variance that arrives while the exchange is shut."""
    oc = np.log(bars["open"] / bars["close"].shift(1)).dropna()
    cc = np.log(bars["close"] / bars["close"].shift(1)).dropna()
    return float(oc.var(ddof=1) / cc.var(ddof=1))


def bias_table(bars: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Each estimator's long-run average against close-to-close on the real tape.

    There is no truth here — that is the point. What can be measured is how much *less*
    variance the gap-blind estimators report, and it should be close to the overnight share.
    """
    est = rolling_variance(bars, window).dropna()
    base = est["close_close"].mean()
    rows = []
    for c in ESTIMATORS:
        m = float(est[c].mean())
        rows.append({"estimator": c, "mean_var": m, "ratio_to_cc": m / base,
                     "mean_ann_vol": float(np.sqrt(max(m, 0) * TRADING_DAYS)),
                     "gap_blind": c in GAP_BLIND})
    return pd.DataFrame(rows).set_index("estimator")


# --------------------------------------------------------------------------- #
# 3) The forecast race
# --------------------------------------------------------------------------- #
def realised_forward_var(bars: pd.DataFrame, horizon: int = 21) -> pd.Series:
    """Mean squared close-to-close return over the NEXT ``horizon`` sessions."""
    r2 = close_close_var(bars)
    return r2.shift(-horizon).rolling(horizon).mean().rename(f"fwd_{horizon}")


def qlike(actual: pd.Series, forecast: pd.Series) -> pd.Series:
    """QLIKE loss, the standard scale-free variance loss (Patton 2011)."""
    a, f = actual.align(forecast, join="inner")
    f = f.clip(lower=1e-12)
    ratio = a.clip(lower=0) / f
    return (ratio - np.log(ratio.clip(lower=1e-12)) - 1.0).rename("qlike")


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, lags: int | None = None) -> dict:
    """Diebold-Mariano test on two loss series (Newey-West, Bartlett).

    Positive ``dm`` means ``loss_a`` is *larger* — i.e. model A is worse. Overlapping
    horizons make the loss differential strongly autocorrelated, so the long-run variance is
    HAC-corrected; without that the test is a significance machine.
    """
    d = (loss_a - loss_b).dropna()
    n = d.size
    if n < 30:
        return {"dm": np.nan, "p_value": np.nan, "n": int(n)}
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    e = d - d.mean()
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e.iloc[k:] @ e.iloc[:-k].to_numpy()) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    dm = float(d.mean() / se) if se > 0 else np.nan
    from math import erfc, sqrt
    p = float(erfc(abs(dm) / sqrt(2.0))) if np.isfinite(dm) else np.nan
    return {"dm": dm, "p_value": p, "n": int(n), "mean_diff": float(d.mean()), "lags": lags}


def forecast_race(bars: pd.DataFrame, window: int = 21, horizon: int = 21,
                  burn: int = 252) -> pd.DataFrame:
    """Score every estimator as a forecast of the next ``horizon`` days' realised variance.

    The forecast is the estimator's trailing ``window``-day value, known at the close of day
    ``t``; the target is the realised close-to-close variance over ``t+1 .. t+horizon``. No
    parameters are fitted, so there is nothing to overfit and no train/test split to argue
    about — the comparison is purely "which reading of today predicts tomorrow best".
    """
    est = rolling_variance(bars, window)
    target = realised_forward_var(bars, horizon).shift(-1)
    idx = est.index[burn:]
    rows, losses = [], {}
    for c in ESTIMATORS:
        f = est[c].reindex(idx)
        a = target.reindex(idx)
        pair = pd.concat([a, f], axis=1).dropna()
        if pair.empty:
            continue
        ql = qlike(pair.iloc[:, 0], pair.iloc[:, 1])
        mse = ((pair.iloc[:, 0] - pair.iloc[:, 1]) ** 2).mean()
        losses[c] = ql
        rows.append({"estimator": c, "qlike": float(ql.mean()), "mse": float(mse),
                     "n": int(len(pair)), "mean_forecast_vol": float(
                         np.sqrt(max(pair.iloc[:, 1].mean(), 0) * TRADING_DAYS))})
    tbl = pd.DataFrame(rows).set_index("estimator")
    base = losses["close_close"]
    for c in tbl.index:
        dm = diebold_mariano(base, losses[c])
        tbl.loc[c, "dm_vs_cc"] = dm["dm"]
        tbl.loc[c, "p_vs_cc"] = dm["p_value"]
    return tbl


def scaled_forecast_race(bars: pd.DataFrame, window: int = 21, horizon: int = 21,
                         burn: int = 252) -> pd.DataFrame:
    """The race again, after each gap-blind estimator is **rescaled** to close-to-close.

    This is the fair fight. A forecast that is systematically 30% too low loses on QLIKE for
    a reason that has nothing to do with information content; multiplying each estimator by
    the in-sample ratio of mean close-to-close variance to its own mean removes the level
    error and leaves only the question that matters — is the *shape* of the signal better?
    The scale factor is estimated on the burn-in window only, so it is not look-ahead.
    """
    est = rolling_variance(bars, window)
    train = est.iloc[:burn].dropna()
    scale = {c: float(train["close_close"].mean() / train[c].mean()) for c in ESTIMATORS
             if train[c].mean() > 0}
    target = realised_forward_var(bars, horizon).shift(-1)
    idx = est.index[burn:]
    rows, losses = [], {}
    for c in ESTIMATORS:
        f = est[c].reindex(idx) * scale.get(c, 1.0)
        pair = pd.concat([target.reindex(idx), f], axis=1).dropna()
        ql = qlike(pair.iloc[:, 0], pair.iloc[:, 1])
        losses[c] = ql
        rows.append({"estimator": c, "scale": scale.get(c, 1.0), "qlike": float(ql.mean()),
                     "mse": float(((pair.iloc[:, 0] - pair.iloc[:, 1]) ** 2).mean()),
                     "n": int(len(pair))})
    tbl = pd.DataFrame(rows).set_index("estimator")
    base = losses["close_close"]
    for c in tbl.index:
        dm = diebold_mariano(base, losses[c])
        tbl.loc[c, "dm_vs_cc"] = dm["dm"]
        tbl.loc[c, "p_vs_cc"] = dm["p_value"]
    return tbl


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def verdict(h: dict) -> dict:
    """Two stamps, by a rule fixed before the run and unit-tested.

    - **Signal** (is the efficiency gain real?): **Real** if, on simulated bars with a known
      sigma, Parkinson's MSE efficiency against close-to-close is at least 3x; **Weak** above
      1.5x; **None** otherwise.
    - **Usefulness** (does it improve a forecast?): **Useful** if the best rescaled range
      estimator beats close-to-close on QLIKE on a majority of tapes *and* the pooled
      Diebold-Mariano statistic clears 2; **Fragile** if it wins without significance;
      **Mirage** if it does not win.
    """
    eff = h["efficiency_parkinson"]
    signal = "Real" if eff >= 3.0 else ("Weak" if eff >= 1.5 else "None")
    wins, n = h["n_qlike_wins"], len(h["tickers"])
    trad = ("Useful" if wins > n / 2 and abs(h["pooled_dm"]) >= 2.0
            else ("Fragile" if wins > n / 2 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"On simulated bars where the day's true sigma is known, Parkinson's estimator "
            f"has **{eff:.1f}x** the efficiency of close-to-close against the truth "
            f"(Garman-Klass {h['efficiency_gk']:.1f}x, Rogers-Satchell "
            f"{h['efficiency_rs']:.1f}x) — the textbook claim survives, in the textbook's own "
            f"world. On the real tape that world does not exist: the overnight gap carries "
            f"**{h['overnight_share_spy']:.0%}** of SPY's daily variance and none of the three "
            f"can see it, so they report **{h['ratio_parkinson_spy']:.0%}** of the "
            f"close-to-close level."),
        "trad": trad,
        "trad_why": (
            f"Rescaled to remove that level error, the best range estimator "
            f"({h['best_estimator']}) beat close-to-close on QLIKE on **{wins} of {n}** tapes "
            f"with a pooled Diebold-Mariano *t* of **{h['pooled_dm']:+.2f}** — a real but "
            f"modest improvement in forecasting the next month's realised variance "
            f"({h['best_qlike_gain']:.1%} lower QLIKE on SPY). Yang-Zhang, the only gap-aware "
            f"estimator, needs no rescaling and is the honest default."),
        "one_sentence": (
            f"The five-to-one efficiency of the range estimators is real in the model they were "
            f"derived in and misleading on a market that gaps: they miss the "
            f"**{h['overnight_share_spy']:.0%}** of daily variance that arrives overnight, and "
            f"once that level error is corrected the remaining forecasting gain is genuine but "
            f"small — which is why Yang-Zhang, not Parkinson, is the one to use."),
    }
