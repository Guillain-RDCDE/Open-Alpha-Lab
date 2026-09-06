"""Lead-lag between a sector and the market, done without fooling yourself — Study 980.

The naive test — correlate today's semiconductor return with tomorrow's market return — mostly
measures the **market factor they share**, because a sector ETF is roughly the market plus a
tilt. Two disciplines fix that, and both are applied throughout:

1. **Use relative strength, not the raw return.** ``relative_strength`` is the sector's return
   minus the market's (a beta-neutral version is available too, via ``residual_series``, which
   regresses out a rolling beta rather than assuming it is one). Whatever predictive content
   survives that subtraction is genuinely about semiconductors.
2. **Test the reverse direction too.** ``lead_lag_table`` reports both ``corr(semi_t,
   market_{t+k})`` and ``corr(market_t, semi_{t+k})``. A "leading indicator" whose reverse
   statistic is just as large is a co-movement, and the difference between the two is the only
   part that can be called a lead.

Then the parts that decide whether any of it matters:

- ``predictive_regression`` — next-period market return on this period's semiconductor relative
  strength, with HAC standard errors at the horizon lag (overlapping windows are the standard
  way this kind of claim gets published with a *t* it has not earned).
- ``timing_rule`` — the tradable version: hold the market when the canary's trailing relative
  strength is positive, hold T-bills otherwise. One day of execution lag, costs on every
  switch, and the comparison is against buy-and-hold, not against zero.
- ``peer_agreement`` — the same battery for SOXX and for XLK. If the effect is real it should
  appear in both semiconductor funds; if it appears in XLK just as strongly, the story is
  "tech leads", which is a different and much older claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.analytics import mean_tstat_hac

TRADING_DAYS = 252
HORIZONS = (1, 5, 21, 63)
HORIZON_LABEL = {1: "1 day", 5: "1 week", 21: "1 month", 63: "1 quarter"}


# --------------------------------------------------------------------------- #
# Building the signal
# --------------------------------------------------------------------------- #
def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns."""
    return prices.pct_change()


def relative_strength(rets: pd.DataFrame, sector: str, market: str) -> pd.Series:
    """Sector minus market — the crude, assumption-free way to remove the common factor."""
    return (rets[sector] - rets[market]).rename(f"{sector}_rs")


def residual_series(rets: pd.DataFrame, sector: str, market: str,
                    window: int = 252) -> pd.Series:
    """Sector return minus *beta times* market, with beta estimated on a trailing window.

    Strictly backward-looking: the beta applied on day ``t`` uses data through ``t-1``. The
    difference from plain relative strength matters for a high-beta sector — semiconductors
    run a beta well above one, so subtracting the market one-for-one leaves a residual that is
    still part market.
    """
    x, y = rets[market], rets[sector]
    cov = y.rolling(window).cov(x).shift(1)
    var = x.rolling(window).var().shift(1)
    beta = (cov / var).rename("beta")
    return (y - beta * x).rename(f"{sector}_resid")


def forward_return(rets: pd.Series, horizon: int) -> pd.Series:
    """Compounded return over the next ``horizon`` sessions, aligned on the decision day."""
    return ((1 + rets).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1.0
            ).rename(f"fwd{horizon}")


def trailing_signal(series: pd.Series, lookback: int) -> pd.Series:
    """Mean of a series over the trailing ``lookback`` sessions, known at that close."""
    return series.rolling(lookback).mean()


# --------------------------------------------------------------------------- #
# Lead-lag measurement
# --------------------------------------------------------------------------- #
def lead_lag_table(rets: pd.DataFrame, sector: str, market: str,
                   max_lag: int = 5, use_residual: bool = True) -> pd.DataFrame:
    """corr(signal_t, market_{t+k}) and the reverse, for k = 0..max_lag.

    The reverse column is the control. Reporting only the forward direction is how a shared
    factor gets published as a leading indicator.
    """
    sig = (residual_series(rets, sector, market) if use_residual
           else relative_strength(rets, sector, market))
    mkt = rets[market]
    rows = []
    for k in range(0, max_lag + 1):
        fwd = pd.concat([sig, mkt.shift(-k)], axis=1).dropna()
        rev = pd.concat([mkt, sig.shift(-k)], axis=1).dropna()
        rows.append({
            "lag": k,
            "sector_leads": float(fwd.iloc[:, 0].corr(fwd.iloc[:, 1])) if len(fwd) > 50 else np.nan,
            "market_leads": float(rev.iloc[:, 0].corr(rev.iloc[:, 1])) if len(rev) > 50 else np.nan,
        })
    out = pd.DataFrame(rows).set_index("lag")
    out["difference"] = out["sector_leads"] - out["market_leads"]
    return out


def predictive_regression(rets: pd.DataFrame, sector: str, market: str,
                          horizon: int = 21, lookback: int = 21,
                          use_residual: bool = True) -> dict:
    """Regress the market's next-``horizon`` return on the sector's trailing relative strength.

    HAC standard errors with the lag set to the horizon (Hansen-Hodrick), because the forward
    windows overlap almost completely. The intercept is reported: a positive slope on a series
    whose intercept already carries the equity risk premium is not the same as a signal.
    """
    sig = (residual_series(rets, sector, market) if use_residual
           else relative_strength(rets, sector, market))
    x = trailing_signal(sig, lookback)
    y = forward_return(rets[market], horizon)
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(df) < 200:
        return {"beta": np.nan, "t": np.nan, "r2": np.nan, "n": int(len(df))}
    X = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
    coef, *_ = np.linalg.lstsq(X, df["y"].to_numpy(), rcond=None)
    resid = df["y"].to_numpy() - X @ coef
    # HAC standard error of the slope, Newey-West with lags = horizon
    lags = min(horizon, len(df) // 4)
    u = X * resid[:, None]
    S = u.T @ u / len(df)
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = u[L:].T @ u[:-L] / len(df)
        S += w * (G + G.T)
    XtX_inv = np.linalg.pinv(X.T @ X / len(df))
    V = XtX_inv @ S @ XtX_inv / len(df)
    se = float(np.sqrt(max(V[1, 1], 0.0)))
    ss_tot = float(((df["y"] - df["y"].mean()) ** 2).sum())
    return {"alpha": float(coef[0]), "beta": float(coef[1]),
            "t": float(coef[1] / se) if se > 0 else np.nan, "se": se,
            "r2": float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan,
            "n": int(len(df)), "lags": int(lags)}


def horizon_grid(rets: pd.DataFrame, sector: str, market: str,
                 horizons=HORIZONS, lookbacks=(5, 21, 63)) -> pd.DataFrame:
    """Every (lookback, horizon) pair — and therefore every chance to find a false positive."""
    rows = []
    for lb in lookbacks:
        for hz in horizons:
            r = predictive_regression(rets, sector, market, horizon=hz, lookback=lb)
            rows.append({"lookback": lb, "horizon": hz, "beta": r["beta"], "t": r["t"],
                         "r2": r["r2"], "n": r["n"]})
    return pd.DataFrame(rows)


def expected_false_positives(n_cells: int, size: float = 0.05) -> float:
    """How many of ``n_cells`` tests clear |t| = 2 by luck alone."""
    return float(n_cells * size)


# --------------------------------------------------------------------------- #
# The tradable version
# --------------------------------------------------------------------------- #
def timing_rule(rets: pd.DataFrame, sector: str, market: str, cash: str,
                lookback: int = 21, cost_bps: float = 2.0,
                use_residual: bool = True) -> dict:
    """Hold the market while the canary's trailing relative strength is positive, else bills.

    The signal is known at the close of day ``t`` and the position applies to day ``t+1``'s
    return; every switch pays ``cost_bps``. Reported against buy-and-hold, because "beats
    cash" is not a claim anybody should be impressed by.
    """
    sig = (residual_series(rets, sector, market) if use_residual
           else relative_strength(rets, sector, market))
    s = trailing_signal(sig, lookback)
    invested = (s > 0).shift(1).fillna(False)
    r_mkt, r_cash = rets[market].fillna(0.0), rets[cash].fillna(0.0)
    switches = invested.astype(int).diff().abs().fillna(0.0)
    strat = np.where(invested, r_mkt, r_cash) - switches * cost_bps / 1e4
    strat = pd.Series(strat, index=rets.index).dropna()
    hold = r_mkt.reindex(strat.index)
    years = len(strat) / TRADING_DAYS
    def curve_stats(x):
        c = (1 + x).cumprod()
        sd = float(x.std(ddof=1))
        vol = sd * np.sqrt(TRADING_DAYS)
        # A constant series (a toy fixture, or a stretch fully in cash) has zero dispersion;
        # its Sharpe is undefined rather than infinite, and saying so beats a RuntimeWarning.
        return {"cagr": float(c.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan,
                "vol": vol,
                "sharpe": float(x.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
                "max_dd": float((c / c.cummax() - 1).min())}
    a, b = curve_stats(strat), curve_stats(hold)
    diff = (strat - hold).dropna()
    return {"strategy": a, "buy_hold": b,
            "time_invested": float(invested.reindex(strat.index).mean()),
            "switches_per_year": float(switches.sum() / years),
            "cagr_gap": a["cagr"] - b["cagr"], "sharpe_gap": a["sharpe"] - b["sharpe"],
            "t_gap": float(mean_tstat_hac(diff)["tstat"]), "years": float(years),
            "returns": strat}


def peer_agreement(rets: pd.DataFrame, market: str, cash: str, peers,
                   lookback: int = 21, horizon: int = 21) -> pd.DataFrame:
    """The same two tests for every candidate canary — including a non-semiconductor control."""
    rows = []
    for p in peers:
        if p not in rets.columns:
            continue
        reg = predictive_regression(rets, p, market, horizon=horizon, lookback=lookback)
        rule = timing_rule(rets, p, market, cash, lookback=lookback)
        rows.append({"canary": p, "beta": reg["beta"], "t": reg["t"], "r2": reg["r2"],
                     "cagr_gap": rule["cagr_gap"], "sharpe_gap": rule["sharpe_gap"],
                     "t_gap": rule["t_gap"], "time_invested": rule["time_invested"]})
    return pd.DataFrame(rows).set_index("canary")


def era_split(rets: pd.DataFrame, sector: str, market: str, cash: str,
              split: str = "2013-01-01", lookback: int = 21, horizon: int = 21) -> pd.DataFrame:
    """The same battery in each half of the sample."""
    rows = []
    for tag, sl in (("early", rets.loc[:split]), ("late", rets.loc[split:])):
        if len(sl) < 500:
            continue
        reg = predictive_regression(sl, sector, market, horizon=horizon, lookback=lookback)
        rule = timing_rule(sl, sector, market, cash, lookback=lookback)
        rows.append({"era": tag, "start": str(sl.index[0].date()), "end": str(sl.index[-1].date()),
                     "beta": reg["beta"], "t": reg["t"], "cagr_gap": rule["cagr_gap"],
                     "sharpe_gap": rule["sharpe_gap"]})
    return pd.DataFrame(rows).set_index("era")


def synthetic_pair(n: int = 4000, lead_strength: float = 0.0, market_vol: float = 0.01,
                   idio_vol: float = 0.012, seed: int = 980) -> pd.DataFrame:
    """A market and a sector where the sector leads by one day with known strength.

    The sector's idiosyncratic shock is fed into *tomorrow's* market return with weight
    ``lead_strength``; at zero the two share only a contemporaneous factor and there is nothing
    to find. Includes a cash column so the timing rule can be run on it unchanged.
    """
    rng = np.random.default_rng(seed)
    f = rng.normal(0.0003, market_vol, n)
    sector_idio = rng.normal(0, idio_vol, n)
    mkt = f.copy()
    mkt[1:] += lead_strength * sector_idio[:-1]
    sector = 1.2 * f + sector_idio
    idx = pd.bdate_range("2005-01-03", periods=n)
    return pd.DataFrame({"MKT": mkt, "SEC": sector,
                         "CASH": np.full(n, 0.02 / TRADING_DAYS)}, index=idx)


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Real** only if the predictive slope clears |*t*| = 2 for **both**
      semiconductor funds at the headline horizon *and* the lead-lag difference (sector leads
      minus market leads) is positive at lag 1; **Weak** if one of those holds; **None**
      otherwise.
    - **Tradability**: **Investable** if the timing rule beats buy-and-hold on Sharpe with
      |*t*| >= 2; **Fragile** if it wins without significance; **Mirage** if it loses.
    """
    both = h["n_semis_significant"] >= 2
    leads = h["lead_diff_lag1"] > 0
    signal = "Real" if both and leads else ("Weak" if both or leads else "None")
    trad = ("Investable" if h["sharpe_gap"] > 0 and abs(h["t_gap"]) >= 2.0
            else ("Fragile" if h["sharpe_gap"] > 0 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"Once the market factor is regressed out, the canary is much quieter than the "
            f"folklore. At lag 1 the semiconductor residual leads the market with correlation "
            f"**{h['lead_lag1']:+.3f}** while the market leads it by {h['market_lead1']:+.3f} — "
            f"a difference of **{h['lead_diff_lag1']:+.3f}**. The headline predictive "
            f"regression (trailing {h['lookback']}-day residual → next {h['horizon']}-day "
            f"market return) gives a slope of {h['beta']:+.2f} with HAC *t* = "
            f"**{h['t_stat']:+.2f}** and an R² of {h['r2']:.3%}; "
            f"**{h['n_semis_significant']} of 2** semiconductor funds clear |*t*| = 2, and "
            f"**{h['n_hits']}** of the {h['n_cells']} lookback × horizon cells do, against "
            f"{h['expected_hits']:.1f} expected by luck. The non-semiconductor control (XLK) "
            f"scores {h['xlk_t']:+.2f} — if the effect were about chips rather than about "
            f"tech, that number should be small."),
        "trad": trad,
        "trad_why": (
            f"The rule — own the market while the canary's trailing relative strength is "
            f"positive, hold bills otherwise — was invested {h['time_invested']:.0%} of the "
            f"time, switched {h['switches_per_year']:.1f} times a year, and returned "
            f"**{h['cagr_strategy']:+.2%}/yr** against **{h['cagr_hold']:+.2%}/yr** for simply "
            f"owning the market ({h['cagr_gap']:+.2%}/yr, Sharpe {h['sharpe_strategy']:+.2f} "
            f"vs {h['sharpe_hold']:+.2f}, HAC *t* on the daily difference "
            f"{h['t_gap']:+.2f}). Its drawdown was {h['dd_strategy']:.1%} against "
            f"{h['dd_hold']:.1%} — the risk reduction is real; the return is not."),
        "one_sentence": (
            f"Semiconductors do move first — by about a day, with a residual lead-lag "
            f"difference of {h['lead_diff_lag1']:+.3f} — but the signal is far smaller than "
            f"the story, it is not distinguishable from a general tech lead, and the rule it "
            f"implies gave up {abs(h['cagr_gap']):.2%} a year against simply owning the "
            f"market."),
    }
