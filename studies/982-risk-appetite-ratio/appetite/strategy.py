"""The high-beta / low-volatility ratio, and the confound it hides — Study 982.

The gauge is ``log(SPHB / SPLV)``: when it rises, the market is paying up for risk. Two facts
have to be held together:

1. The spread is **not** a neutral measurement. SPHB has a beta near 1.3 and SPLV near 0.7, so
   ``SPHB − SPLV`` is roughly a 0.6-beta position in the index. Its trend is therefore
   correlated with the index's trend nearly by construction, and any predictive regression that
   ignores this is measuring market momentum with extra steps.
2. Once the market's own return is projected out, what is left is a **genuine cross-sectional
   spread** — the low-volatility anomaly's own factor — and whether *that* predicts anything is
   an open question worth asking.

The module therefore builds three versions of the same signal and runs everything on all three:

- ``ratio_raw`` — the spread everybody quotes.
- ``ratio_beta_neutral`` — the spread with a rolling backward-looking beta on the market
  removed, so it is orthogonal to the index by construction.
- ``market_trend`` — the control: the market's own trailing return, which is what the raw
  spread is suspected of being a noisy copy of.

These three are not independent: by the definition of the neutralisation,
``raw = beta_neutral + beta * market_trend`` exactly. That identity is the study's spine rather
than an inconvenience. It means the raw gauge *decomposes without remainder* into a market
component and a residual component, so the question "is the gauge anything but the market?"
has a clean answer: run the two components against each other and see which one carries the
slope. ``horse_race`` runs each signal alone and then that two-way decomposition — and
deliberately never puts all three in one regression, which would be singular.

If the raw gauge predicts and the beta-neutral component does not, while the market component
does, the honest conclusion is that the gauge is momentum in disguise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
SIGNALS = ("raw", "beta_neutral", "market_trend")
SIGNAL_LABEL = {"raw": "High-beta minus low-vol (raw)",
                "beta_neutral": "The same spread, beta-neutralised",
                "market_trend": "The market's own trailing return (control)"}
HORIZONS = (5, 21, 63)
LOOKBACKS = (21, 63, 126)


# --------------------------------------------------------------------------- #
# The three signals
# --------------------------------------------------------------------------- #
def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns."""
    return prices.pct_change()


def ratio_raw(rets: pd.DataFrame, high: str, low: str) -> pd.Series:
    """The quoted gauge: the daily return difference between the two halves."""
    return (rets[high] - rets[low]).rename("appetite_raw")


def rolling_beta(y: pd.Series, x: pd.Series, window: int = 252) -> pd.Series:
    """Backward-looking beta of ``y`` on ``x``: the value on day *t* uses data through *t-1*."""
    cov = y.rolling(window).cov(x).shift(1)
    var = x.rolling(window).var().shift(1)
    return (cov / var).rename("beta")


def ratio_beta_neutral(rets: pd.DataFrame, high: str, low: str, market: str,
                       window: int = 252) -> pd.Series:
    """The spread with its market exposure removed, using a strictly trailing beta."""
    spread = ratio_raw(rets, high, low)
    b = rolling_beta(spread, rets[market], window)
    return (spread - b * rets[market]).rename("appetite_neutral")


def market_trend(rets: pd.DataFrame, market: str) -> pd.Series:
    """The control signal: the market's own daily return, to be averaged over a lookback."""
    return rets[market].rename("market")


def build_signals(rets: pd.DataFrame, high: str, low: str, market: str,
                  window: int = 252) -> pd.DataFrame:
    """All three raw daily series, aligned."""
    return pd.DataFrame({
        "raw": ratio_raw(rets, high, low),
        "beta_neutral": ratio_beta_neutral(rets, high, low, market, window),
        "market_trend": market_trend(rets, market),
    })


def trailing(series: pd.Series, lookback: int) -> pd.Series:
    """Mean over the trailing ``lookback`` sessions, known at that close."""
    return series.rolling(lookback).mean()


def spread_beta(rets: pd.DataFrame, high: str, low: str, market: str) -> dict:
    """How much of the raw spread is simply market exposure — the confound, measured."""
    spread = ratio_raw(rets, high, low).dropna()
    m = rets[market].reindex(spread.index)
    df = pd.concat([spread, m], axis=1).dropna()
    x = df.iloc[:, 1].to_numpy()
    beta = float(np.cov(x, df.iloc[:, 0].to_numpy(), ddof=1)[0, 1] / x.var(ddof=1))
    resid = df.iloc[:, 0].to_numpy() - beta * x
    return {"beta_of_spread": beta,
            "beta_high": float(np.cov(x, rets[high].reindex(df.index), ddof=1)[0, 1]
                               / x.var(ddof=1)),
            "beta_low": float(np.cov(x, rets[low].reindex(df.index), ddof=1)[0, 1]
                              / x.var(ddof=1)),
            "r2_on_market": float(1 - resid.var(ddof=1) / df.iloc[:, 0].var(ddof=1)),
            "corr_with_market": float(df.iloc[:, 0].corr(df.iloc[:, 1]))}


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def forward_return(r: pd.Series, horizon: int) -> pd.Series:
    """Compounded market return over the next ``horizon`` sessions."""
    return ((1 + r).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1.0)


def hac_regression(y: pd.Series, X: pd.DataFrame, lags: int) -> dict:
    """OLS with Newey-West standard errors at ``lags`` — the overlapping-window standard."""
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(df) < 200:
        return {"n": int(len(df))}
    names = list(X.columns)
    A = np.column_stack([np.ones(len(df)), df[names].to_numpy()])
    coef, *_ = np.linalg.lstsq(A, df["y"].to_numpy(), rcond=None)
    resid = df["y"].to_numpy() - A @ coef
    L = int(min(lags, len(df) // 4))
    u = A * resid[:, None]
    S = u.T @ u / len(df)
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        G = u[k:].T @ u[:-k] / len(df)
        S += w * (G + G.T)
    XtX_inv = np.linalg.pinv(A.T @ A / len(df))
    V = XtX_inv @ S @ XtX_inv / len(df)
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    ss_tot = float(((df["y"] - df["y"].mean()) ** 2).sum())
    out = {"n": int(len(df)), "lags": L, "alpha": float(coef[0]),
           "r2": float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan}
    for i, nm in enumerate(names, start=1):
        out[f"beta_{nm}"] = float(coef[i])
        out[f"t_{nm}"] = float(coef[i] / se[i]) if se[i] > 0 else np.nan
    return out


def univariate_grid(signals: pd.DataFrame, market_rets: pd.Series,
                    lookbacks=LOOKBACKS, horizons=HORIZONS) -> pd.DataFrame:
    """Each signal alone, at each lookback and horizon."""
    rows = []
    for name in signals.columns:
        for lb in lookbacks:
            x = trailing(signals[name], lb)
            for hz in horizons:
                y = forward_return(market_rets, hz)
                r = hac_regression(y, x.to_frame(name), lags=hz)
                rows.append({"signal": name, "lookback": lb, "horizon": hz,
                             "beta": r.get(f"beta_{name}", np.nan),
                             "t": r.get(f"t_{name}", np.nan), "r2": r.get("r2", np.nan),
                             "n": r.get("n", 0)})
    return pd.DataFrame(rows)


DECOMPOSITION = ("beta_neutral", "market_trend")


def horse_race(signals: pd.DataFrame, market_rets: pd.Series, lookback: int = 63,
               horizon: int = 21) -> pd.DataFrame:
    """Univariate, then the decomposition — the regression that settles the confound.

    A warning about a regression this study deliberately does **not** run. The three signals
    are exactly linearly dependent: ``raw = beta_neutral + beta * market_trend`` is the
    definition of the neutralisation, not an empirical finding. Putting all three on the
    right-hand side is a singular design matrix, and the pseudo-inverse's answer to it is
    meaningless — three enormous mutually-cancelling coefficients with *t*-statistics near zero.

    The informative regression is the **decomposition**: the raw gauge is split, without
    remainder, into the part that is the market and the part that is not, and those two compete.
    Whichever of them carries the slope is where the raw gauge's predictive content came from.
    """
    y = forward_return(market_rets, horizon)
    rows = []
    for name in signals.columns:
        r = hac_regression(y, trailing(signals[name], lookback).to_frame(name), lags=horizon)
        rows.append({"specification": f"{name} alone", "signal": name,
                     "beta": r.get(f"beta_{name}", np.nan), "t": r.get(f"t_{name}", np.nan),
                     "r2": r.get("r2", np.nan)})
    cols = [c for c in DECOMPOSITION if c in signals.columns]
    X = pd.DataFrame({c: trailing(signals[c], lookback) for c in cols})
    r = hac_regression(y, X, lags=horizon)
    for name in cols:
        rows.append({"specification": "the decomposition", "signal": name,
                     "beta": r.get(f"beta_{name}", np.nan), "t": r.get(f"t_{name}", np.nan),
                     "r2": r.get("r2", np.nan)})
    return pd.DataFrame(rows)


def decomposition_residual(signals: pd.DataFrame, rets: pd.DataFrame, high: str, low: str,
                           market: str, window: int = 252) -> float:
    """Max absolute gap in ``raw - (beta_neutral + beta * market)`` — should be ~0 by algebra."""
    b = rolling_beta(ratio_raw(rets, high, low), rets[market], window)
    gap = (signals["raw"] - signals["beta_neutral"] - b * rets[market]).dropna()
    return float(gap.abs().max()) if len(gap) else np.nan


def expected_false_positives(n_cells: int, size: float = 0.05) -> float:
    """How many of ``n_cells`` tests clear |t| = 2 by luck."""
    return float(n_cells * size)


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #
def timing_rule(rets: pd.DataFrame, signal: pd.Series, market: str, cash: str,
                lookback: int = 63, cost_bps: float = 2.0) -> dict:
    """Hold the market while the signal's trailing average is positive, else T-bills."""
    s = trailing(signal, lookback)
    invested = (s > 0).shift(1).fillna(False)
    r_mkt = rets[market].fillna(0.0)
    r_cash = rets[cash].fillna(0.0)
    switches = invested.astype(int).diff().abs().fillna(0.0)
    strat = pd.Series(np.where(invested, r_mkt, r_cash), index=rets.index) \
        - switches * cost_bps / 1e4
    valid = s.notna()
    strat, hold = strat[valid], r_mkt[valid]
    years = len(strat) / TRADING_DAYS
    def stats(x):
        c = (1 + x).cumprod()
        sd = float(x.std(ddof=1))
        return {"cagr": float(c.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan,
                "vol": sd * np.sqrt(TRADING_DAYS),
                "sharpe": float(x.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
                "max_dd": float((c / c.cummax() - 1).min())}
    a, b = stats(strat), stats(hold)
    d = (strat - hold).dropna()
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 30 else np.nan
    return {"strategy": a, "buy_hold": b, "cagr_gap": a["cagr"] - b["cagr"],
            "sharpe_gap": a["sharpe"] - b["sharpe"],
            "t_gap": float(d.mean() / se) if se and se > 0 else np.nan,
            "time_invested": float(invested[valid].mean()),
            "switches_per_year": float(switches[valid].sum() / years),
            "returns": strat}


def crisis_table(signal: pd.Series, market_rets: pd.Series, lookback: int = 63,
                 windows=None) -> pd.DataFrame:
    """Did the gauge fall *before* the market did, in the sample's actual drawdowns?

    A leading indicator's whole claim is about turning points, and turning points are exactly
    where a full-sample regression has no power. This looks at each named episode and reports
    how many sessions before the market's peak the signal turned negative — a small,
    anecdotal, and honestly-labelled piece of evidence.
    """
    windows = windows or {
        "2015 China devaluation": ("2015-06-01", "2015-09-30"),
        "2018 Q4": ("2018-09-01", "2018-12-31"),
        "COVID crash": ("2020-01-15", "2020-04-01"),
        "2022 bear market": ("2021-12-01", "2022-10-31"),
    }
    s = trailing(signal, lookback)
    rows = []
    for name, (a, b) in windows.items():
        sl_m = market_rets.loc[a:b]
        sl_s = s.loc[a:b]
        if len(sl_m) < 20 or sl_s.notna().sum() < 20:
            continue
        curve = (1 + sl_m).cumprod()
        peak = curve.idxmax()
        neg = sl_s[(sl_s < 0)]
        first_neg = neg.index[0] if len(neg) else None
        lead = (int(np.busday_count(first_neg.date(), peak.date()))
                if first_neg is not None else np.nan)
        rows.append({"episode": name, "market_peak": str(peak.date()),
                     "signal_turned_negative": str(first_neg.date()) if first_neg is not None
                     else "never",
                     "sessions_of_warning": lead,
                     "drawdown": float((curve / curve.cummax() - 1).min())})
    cols = ["episode", "market_peak", "signal_turned_negative", "sessions_of_warning",
            "drawdown"]
    return pd.DataFrame(rows, columns=cols).set_index("episode")


def synthetic_world(n: int = 4000, appetite_strength: float = 0.0, spread_beta: float = 0.6,
                    seed: int = 982) -> pd.DataFrame:
    """A market, a high-beta sleeve and a low-vol sleeve, with a controllable appetite factor.

    A latent state ``a_t`` (an AR(1)) drives the *residual* spread between the two sleeves and,
    with weight ``appetite_strength``, tomorrow's market return. At zero the spread is nothing
    but a leveraged market position — the null this study exists to survive.
    """
    rng = np.random.default_rng(seed)
    phi = 0.97
    a = np.zeros(n)
    e = rng.normal(0, 1, n)
    for t in range(1, n):
        a[t] = phi * a[t - 1] + np.sqrt(1 - phi ** 2) * e[t]
    mkt = rng.normal(0.0003, 0.01, n)
    mkt[1:] += appetite_strength * 0.004 * a[:-1]
    resid = 0.004 * a + rng.normal(0, 0.004, n)
    high = 1.3 * mkt + 0.5 * resid
    low = 0.7 * mkt - 0.5 * resid
    idx = pd.bdate_range("2005-01-03", periods=n)
    return pd.DataFrame({"SPY": mkt, "SPHB": high, "SPLV": low,
                         "BIL": np.full(n, 0.02 / TRADING_DAYS)}, index=idx)


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Real** only if the **beta-neutral** component keeps a |*t*| >= 2 in the
      decomposition regression that also contains the market component — that is, it survives
      its own confound; **Weak** if only the raw spread predicts; **None** if neither does.
    - **Tradability**: **Investable** if the rule beats buy-and-hold on Sharpe with |*t*| >= 2;
      **Fragile** if it wins without significance; **Mirage** if it loses.
    """
    survives = abs(h["t_neutral_multi"]) >= 2.0
    raw_only = abs(h["t_raw_uni"]) >= 2.0
    signal = "Real" if survives else ("Weak" if raw_only else "None")
    trad = ("Investable" if h["sharpe_gap"] > 0 and abs(h["t_gap"]) >= 2.0
            else ("Fragile" if h["sharpe_gap"] > 0 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"The gauge is **{h['r2_on_market']:.0%} explained by the market itself** — the "
            f"high-beta minus low-volatility spread carries a beta of "
            f"**{h['beta_of_spread']:+.2f}** to the index, so a rising gauge is very largely a "
            f"rising market. On its own the raw spread's trailing {h['lookback']}-day average "
            f"predicts the next {h['horizon']} days with *t* = **{h['t_raw_uni']:+.2f}**; once "
            f"it is beta-neutralised that falls to **{h['t_neutral_uni']:+.2f}**, and in the "
            f"decomposition where it competes with the market component it is "
            f"**{h['t_neutral_multi']:+.2f}** against the market's {h['t_trend_multi']:+.2f}. "
            f"Across the {h['n_cells']} lookback × horizon × signal cells, {h['n_hits']} clear "
            f"|*t*| = 2 against {h['expected_hits']:.1f} expected by luck — and the sample is "
            f"only {h['years']:.0f} years long, which caps what any of these numbers can mean."),
        "trad": trad,
        "trad_why": (
            f"The rule — own the index while the gauge is positive, hold bills otherwise — was "
            f"invested {h['time_invested']:.0%} of the time, switched "
            f"{h['switches_per_year']:.1f} times a year and returned "
            f"**{h['cagr_strategy']:+.2%}/yr** against **{h['cagr_hold']:+.2%}** for holding "
            f"the index ({h['cagr_gap']:+.2%}/yr; Sharpe {h['sharpe_strategy']:+.2f} vs "
            f"{h['sharpe_hold']:+.2f}, *t* = {h['t_gap']:+.2f}). Its worst drawdown was "
            f"{h['dd_strategy']:.1%} against {h['dd_hold']:.1%} — the familiar trade of return "
            f"for a shallower hole, available from any trend filter."),
        "one_sentence": (
            f"The high-beta / low-volatility ratio is **{h['r2_on_market']:.0%} the market**, "
            f"and once that is projected out the residual gauge predicts the index with "
            f"*t* = {h['t_neutral_multi']:+.2f} in the regression where the market component "
            f"sits beside it — so what the desks are watching is, mostly, the market."),
    }
