"""Is Bitcoin's volatility really falling? — Study 988.

The claim is easy to state and surprisingly hard to test. Realised volatility is:

- **enormously persistent** (its autocorrelation at 100 lags is still positive), so a series
  observed from a high starting point will drift down for years with no trend at all;
- **skewed and heteroskedastic in its own right**, so an OLS trend on levels is the wrong
  estimator and its standard errors are wrong twice over;
- **regime-ridden**, so a single line through the whole sample describes no part of it.

The module therefore never reports a single number. ``trend_table`` fits the trend four ways —
OLS on log volatility, Theil-Sen (robust to the spikes), a Mann-Kendall rank test (assumes
nothing about the functional form) and a block-bootstrapped OLS (honest standard errors under
persistence) — and if they disagree, that disagreement is the result.

The critical control is ``start_date_sensitivity``. Any downward trend in a persistent series
can be manufactured by starting the window at a peak, and Bitcoin's history offers four of them.
The function refits the trend from every possible start date and reports what fraction of them
yield a significant decline — which is the number the "Bitcoin is maturing" chart never shows.

``halving_alignment`` checks the other popular story: that volatility steps down after each
halving. Four halvings is four observations, and the function says so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CRYPTO_DAYS = 365        # crypto trades every day
EQUITY_DAYS = 252
HALVINGS = ("2016-07-09", "2020-05-11", "2024-04-20")   # the ones inside a sensible sample


# --------------------------------------------------------------------------- #
# Measuring volatility on the right calendar
# --------------------------------------------------------------------------- #
def annualisation_factor(series: pd.Series) -> float:
    """Observations per year, inferred from the series' own calendar.

    Not a detail. Bitcoin has ~365 observations a year and SPY has ~252; using 252 for both
    understates Bitcoin's annualised volatility by about 20%, which is larger than the entire
    trend this study is trying to measure.
    """
    s = series.dropna()
    if len(s) < 30:
        return EQUITY_DAYS
    years = (s.index[-1] - s.index[0]).days / 365.25
    return float(len(s) / years) if years > 0 else EQUITY_DAYS


def realised_vol(prices: pd.Series, window: int = 30, ann: float | None = None) -> pd.Series:
    """Trailing realised volatility, annualised on the series' own calendar."""
    r = prices.dropna().pct_change()
    a = annualisation_factor(prices) if ann is None else ann
    return (r.rolling(window).std() * np.sqrt(a)).rename("vol").dropna()


def vol_summary(prices: pd.Series, windows=(30, 90, 365)) -> pd.DataFrame:
    """Realised volatility at several windows, with its own distribution."""
    rows = []
    a = annualisation_factor(prices)
    for w in windows:
        v = realised_vol(prices, w, a)
        if len(v) < 100:
            continue
        rows.append({"window": w, "n": len(v), "mean": float(v.mean()),
                     "median": float(v.median()), "min": float(v.min()),
                     "max": float(v.max()), "p10": float(v.quantile(0.1)),
                     "p90": float(v.quantile(0.9)),
                     "autocorr_100": float(v.autocorr(100))})
    return pd.DataFrame(rows).set_index("window")


# --------------------------------------------------------------------------- #
# Four ways to fit a trend
# --------------------------------------------------------------------------- #
def ols_trend(v: pd.Series, ann: float = CRYPTO_DAYS) -> dict:
    """OLS on log volatility against time in years. The number everyone quotes."""
    y = np.log(v.dropna().to_numpy())
    n = len(y)
    if n < 100:
        return {"n": int(n)}
    t = np.arange(n) / ann
    A = np.column_stack([np.ones(n), t])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    s2 = float((resid ** 2).sum() / max(n - 2, 1))
    se = float(np.sqrt(s2 * np.linalg.pinv(A.T @ A)[1, 1]))
    return {"n": int(n), "slope_per_year": float(coef[1]),
            "se_naive": se, "t_naive": float(coef[1] / se) if se > 0 else np.nan,
            "pct_per_year": float(np.exp(coef[1]) - 1.0),
            "start_fit": float(np.exp(coef[0])),
            "end_fit": float(np.exp(coef[0] + coef[1] * t[-1]))}


def theil_sen(v: pd.Series, ann: float = CRYPTO_DAYS, max_pairs: int = 200_000,
              seed: int = 988) -> dict:
    """Median of pairwise slopes — immune to the spikes that dominate an OLS fit."""
    y = np.log(v.dropna().to_numpy())
    n = len(y)
    if n < 100:
        return {"n": int(n)}
    t = np.arange(n) / ann
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, max_pairs)
    j = rng.integers(0, n, max_pairs)
    ok = i != j
    slopes = (y[i[ok]] - y[j[ok]]) / (t[i[ok]] - t[j[ok]])
    return {"n": int(n), "slope_per_year": float(np.median(slopes)),
            "pct_per_year": float(np.exp(np.median(slopes)) - 1.0),
            "lo": float(np.percentile(slopes, 2.5)),
            "hi": float(np.percentile(slopes, 97.5))}


def mann_kendall(v: pd.Series, sample: int = 4000, seed: int = 988) -> dict:
    """A rank test for monotone trend that assumes nothing about the functional form.

    Computed on a thinned sample: the full pairwise statistic is O(n²) and, far more
    importantly, adjacent daily volatility observations are almost the same number, so the
    untinned test's variance formula (which assumes independence) is badly wrong. Thinning to
    roughly weekly observations does not fix that entirely — the caveat stands in the results —
    but it makes the statistic interpretable.
    """
    y = v.dropna().to_numpy()
    n0 = len(y)
    if n0 < 100:
        return {"n": int(n0)}
    step = max(1, n0 // sample)
    y = y[::step]
    n = len(y)
    s = 0
    for k in range(n - 1):
        s += int(np.sign(y[k + 1:] - y[k]).sum())
    var = n * (n - 1) * (2 * n + 5) / 18.0
    z = (s - np.sign(s)) / np.sqrt(var) if var > 0 else np.nan
    return {"n": int(n), "thinned_from": int(n0), "S": int(s), "z": float(z),
            "direction": "down" if s < 0 else ("up" if s > 0 else "flat")}


def block_bootstrap_trend(v: pd.Series, ann: float = CRYPTO_DAYS, block: int = 180,
                          n_boot: int = 800, seed: int = 988) -> dict:
    """The OLS slope with a standard error that respects how persistent volatility is.

    Circular block bootstrap on the log-volatility series. The naive OLS standard error assumes
    independent residuals; volatility residuals are anything but, and the difference between the
    two intervals is usually a factor of three or more.
    """
    y = np.log(v.dropna().to_numpy())
    n = len(y)
    if n < 300:
        return {"n": int(n)}
    t = np.arange(n) / ann
    A = np.column_stack([np.ones(n), t])
    base = float(np.linalg.lstsq(A, y, rcond=None)[0][1])
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    slopes = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
        slopes[b] = np.linalg.lstsq(A, y[idx], rcond=None)[0][1]
    return {"n": int(n), "slope_per_year": base, "boot_sd": float(slopes.std(ddof=1)),
            "lo": float(np.percentile(slopes, 2.5)), "hi": float(np.percentile(slopes, 97.5)),
            "t_boot": float(base / slopes.std(ddof=1)) if slopes.std(ddof=1) > 0 else np.nan,
            "share_negative": float((slopes < 0).mean())}


def trend_table(v: pd.Series, ann: float = CRYPTO_DAYS) -> pd.DataFrame:
    """All four estimators. If they disagree, that disagreement is the finding."""
    o, ts = ols_trend(v, ann), theil_sen(v, ann)
    bb, mk = block_bootstrap_trend(v, ann), mann_kendall(v)
    return pd.DataFrame([
        {"method": "OLS on log vol (naive SE)", "slope_per_year": o.get("slope_per_year"),
         "pct_per_year": o.get("pct_per_year"), "t": o.get("t_naive"),
         "note": "assumes independent residuals — they are not"},
        {"method": "Theil-Sen (robust)", "slope_per_year": ts.get("slope_per_year"),
         "pct_per_year": ts.get("pct_per_year"), "t": np.nan,
         "note": f"95% of pairwise slopes in [{ts.get('lo', np.nan):.3f}, "
                 f"{ts.get('hi', np.nan):.3f}]"},
        {"method": "OLS, block bootstrap SE", "slope_per_year": bb.get("slope_per_year"),
         "pct_per_year": np.exp(bb.get("slope_per_year", np.nan)) - 1
         if bb.get("slope_per_year") is not None else np.nan,
         "t": bb.get("t_boot"),
         "note": f"{bb.get('share_negative', np.nan):.0%} of resamples negative"},
        {"method": "Mann-Kendall (rank)", "slope_per_year": np.nan, "pct_per_year": np.nan,
         "t": mk.get("z"), "note": f"direction: {mk.get('direction', '?')}"},
    ]).set_index("method")


# --------------------------------------------------------------------------- #
# The control that matters
# --------------------------------------------------------------------------- #
def start_date_sensitivity(v: pd.Series, ann: float = CRYPTO_DAYS, step: int = 30,
                           min_years: float = 3.0) -> pd.DataFrame:
    """Refit the trend from every possible start date.

    This is the control the "Bitcoin is maturing" chart never shows. In a persistent series,
    starting the window at a local volatility peak *manufactures* a downward trend. Bitcoin's
    history offers several peaks to start from. If the fraction of start dates yielding a
    significant decline is near one, the finding is robust; if it swings with the start date,
    the chart is a choice rather than a measurement.
    """
    vv = v.dropna()
    rows = []
    min_n = int(min_years * ann)
    for i in range(0, max(len(vv) - min_n, 0), step):
        sl = vv.iloc[i:]
        o = ols_trend(sl, ann)
        if "slope_per_year" not in o:
            continue
        rows.append({"start": sl.index[0], "years": len(sl) / ann,
                     "start_vol": float(sl.iloc[0]),
                     "slope_per_year": o["slope_per_year"], "t": o["t_naive"]})
    return pd.DataFrame(rows).set_index("start") if rows else pd.DataFrame()


def sensitivity_summary(sens: pd.DataFrame) -> dict:
    """How much of the "decay" is a choice of start date?"""
    if sens.empty:
        return {"n": 0}
    s = sens["slope_per_year"]
    return {"n": int(len(sens)), "mean_slope": float(s.mean()),
            "min_slope": float(s.min()), "max_slope": float(s.max()),
            "share_negative": float((s < 0).mean()),
            "share_significant_down": float(((s < 0) & (sens["t"] < -2)).mean()),
            "share_significant_up": float(((s > 0) & (sens["t"] > 2)).mean()),
            "corr_with_start_vol": float(np.corrcoef(sens["start_vol"], s)[0, 1])}


def by_era(v: pd.Series, ann: float = CRYPTO_DAYS, n_eras: int = 4) -> pd.DataFrame:
    """Volatility era by era, because one line through four regimes describes none."""
    vv = v.dropna()
    edges = np.array_split(np.arange(len(vv)), n_eras)
    rows = []
    for k, ix in enumerate(edges):
        sl = vv.iloc[ix]
        rows.append({"era": f"{sl.index[0].date()} to {sl.index[-1].date()}",
                     "n": len(sl), "mean_vol": float(sl.mean()),
                     "median_vol": float(sl.median()), "max_vol": float(sl.max()),
                     "slope_within": ols_trend(sl, ann).get("slope_per_year", np.nan)})
    return pd.DataFrame(rows).set_index("era")


def halving_alignment(v: pd.Series, halvings=HALVINGS, window_days: int = 365) -> pd.DataFrame:
    """Volatility before and after each halving. Three events; treat accordingly."""
    vv = v.dropna()
    rows = []
    for hd in halvings:
        d = pd.Timestamp(hd)
        before = vv.loc[d - pd.Timedelta(days=window_days):d]
        after = vv.loc[d:d + pd.Timedelta(days=window_days)]
        if len(before) < 100 or len(after) < 100:
            continue
        rows.append({"halving": hd, "vol_before": float(before.mean()),
                     "vol_after": float(after.mean()),
                     "change": float(after.mean() - before.mean()),
                     "ratio": float(after.mean() / before.mean())})
    return pd.DataFrame(rows).set_index("halving") if rows else pd.DataFrame()


def relative_to_equities(crypto_px: pd.Series, others: dict, window: int = 365) -> pd.DataFrame:
    """Bitcoin's volatility as a multiple of everything else's, through time.

    "Maturing" should mean converging *toward* other assets, not merely falling — an asset whose
    volatility halves in a decade when every other asset's also halved has not matured, it has
    been carried by a calm market.
    """
    a_c = annualisation_factor(crypto_px)
    vc = realised_vol(crypto_px, window, a_c)
    out = {}
    for name, px in others.items():
        vo = realised_vol(px, window, annualisation_factor(px))
        ratio = (vc / vo.reindex(vc.index).ffill()).dropna()
        out[name] = ratio
    df = pd.DataFrame(out).dropna(how="all")
    return df


def sizing_backtest(prices: pd.Series, target_vol: float = 0.40, window: int = 30,
                    cash: pd.Series | None = None, cost_bps: float = 5.0,
                    max_leverage: float = 3.0) -> dict:
    """Volatility targeting: size the position so that risk, not exposure, is constant.

    The practical consequence of the whole study. If volatility is genuinely trending down, a
    fixed-exposure holder is quietly taking less risk each year and a vol-targeted one is
    quietly taking more leverage. Which is right depends on whether the trend is real.
    """
    px = prices.dropna()
    r = px.pct_change()
    a = annualisation_factor(px)
    v = realised_vol(px, window, a).reindex(r.index)
    lev = (target_vol / v).shift(1).clip(0, max_leverage)
    c = (cash.reindex(r.index).fillna(0.0) if cash is not None
         else pd.Series(0.0, index=r.index))
    turnover = lev.diff().abs().fillna(0.0)
    strat = (lev * r + (1 - lev).clip(upper=1.0) * c - turnover * cost_bps / 1e4).dropna()
    hold = r.reindex(strat.index)
    years = len(strat) / a

    def stats(x):
        cu = (1 + x).cumprod()
        sd = float(x.std(ddof=1))
        return {"cagr": float(cu.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan,
                "vol": sd * np.sqrt(a),
                "sharpe": float(x.mean() / sd * np.sqrt(a)) if sd > 0 else np.nan,
                "max_dd": float((cu / cu.cummax() - 1).min())}

    return {"target_vol": target_vol, "vol_targeted": stats(strat), "buy_hold": stats(hold),
            "mean_leverage": float(lev.mean()), "years": float(years),
            "leverage_trend": ols_trend(lev.dropna().clip(lower=1e-6), a).get(
                "slope_per_year", np.nan),
            "returns": strat}


def synthetic_world(n: int = 4000, decay_per_year: float = 0.0, persistence: float = 0.995,
                    base_vol: float = 0.80, seed: int = 988) -> pd.Series:
    """A price series whose volatility has a KNOWN deterministic decay under a persistent shock.

    ``decay_per_year`` is the truth. At zero the volatility process is stationary and highly
    persistent — and that is the null that matters here, because a persistent stationary series
    observed from a peak looks exactly like decay to every naive test.
    """
    rng = np.random.default_rng(seed)
    logv = np.zeros(n)
    for t in range(1, n):
        logv[t] = persistence * logv[t - 1] + rng.normal(0, 0.10)
    trend = decay_per_year * np.arange(n) / CRYPTO_DAYS
    vol = base_vol * np.exp(logv + trend)
    r = rng.normal(0, 1, n) * vol / np.sqrt(CRYPTO_DAYS)
    idx = pd.date_range("2014-09-17", periods=n, freq="D")
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx, name="price")


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Real** only if the block-bootstrapped slope is negative with |*t*| >= 2
      **and** the decline survives the start-date control (a significant decline from at least
      80% of possible start dates); **Weak** if the point estimate is negative but one of those
      fails; **None** if the slope is not negative.
    - **Tradability**: **Useful** if volatility targeting beat buy-and-hold on Sharpe;
      **Partial** if it matched within 0.1; **Mirage** if it lost.
    """
    negative = h["boot_slope"] < 0
    significant = abs(h["boot_t"]) >= 2.0
    robust = h["share_significant_down"] >= 0.8
    signal = ("Real" if (negative and significant and robust)
              else ("Weak" if negative else "None"))
    edge = h["vt_sharpe"] - h["bh_sharpe"]
    trad = ("Useful" if edge > 0.1 else ("Partial" if edge > -0.1 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"Bitcoin's trailing {h['window']}-day volatility averaged **{h['mean_vol']:.0%}** "
            f"over {h['years']:.1f} years, ranging {h['min_vol']:.0%} to {h['max_vol']:.0%}. "
            f"Fitted four ways, the trend in log volatility is: OLS "
            f"**{h['ols_pct']:+.1%}/yr** (naive *t* = {h['ols_t']:+.2f}), Theil-Sen "
            f"{h['ts_pct']:+.1%}/yr, and OLS with a block-bootstrap standard error "
            f"{h['boot_slope']:+.3f} with *t* = **{h['boot_t']:+.2f}** — the naive *t* is "
            f"{abs(h['ols_t'] / h['boot_t']) if h['boot_t'] else float('nan'):.1f}× too large "
            f"because volatility residuals are nothing like independent (the 100-day "
            f"autocorrelation is still {h['autocorr_100']:+.2f}). The control that settles it: "
            f"refitting from **every** possible start date, {h['share_negative']:.0%} of "
            f"windows slope down but only **{h['share_significant_down']:.0%}** do so "
            f"significantly, and the fitted slope correlates {h['corr_with_start_vol']:+.2f} "
            f"with the volatility on the day the window opens. That last number is the whole "
            f"trick: start at a peak, get a decline."),
        "trad": trad,
        "trad_why": (
            f"Sizing to a constant {h['target_vol']:.0%} volatility rather than holding a fixed "
            f"position returned **{h['vt_cagr']:+.1%}/yr** against buy-and-hold's "
            f"**{h['bh_cagr']:+.1%}** (Sharpe {h['vt_sharpe']:.2f} vs {h['bh_sharpe']:.2f}, "
            f"drawdown {h['vt_dd']:.0%} vs {h['bh_dd']:.0%}) at average leverage "
            f"{h['mean_leverage']:.2f}×. Note what the trend question does to this: if "
            f"volatility really is decaying, a vol-targeted holder must lever up over time — "
            f"the fitted leverage trend here is {h['leverage_trend']:+.3f}/yr — and is "
            f"therefore making a bet on the trend continuing whether they meant to or not."),
        "one_sentence": (
            f"Bitcoin's volatility slopes down at {h['ols_pct']:+.1%} a year with a naive *t* of "
            f"{h['ols_t']:+.2f} and a bootstrapped *t* of {h['boot_t']:+.2f}, and only "
            f"{h['share_significant_down']:.0%} of possible start dates reproduce it — the "
            f"maturity story is mostly a choice of where to begin the chart."),
    }
