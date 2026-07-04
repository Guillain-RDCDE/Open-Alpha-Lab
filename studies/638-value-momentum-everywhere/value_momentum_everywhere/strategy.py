"""Tested rules + statistics for Study 638 — Value-Momentum-Everywhere.

Asness-Moskowitz-Pedersen (2013): **value and momentum both pay in every asset class,
they are negatively correlated, and the 50/50 combo is the real free lunch.** We rebuild
it on free data with ONE uniform signal pair across all four sleeves:

* **VALUE** = the 5-year reversal proxy AMP themselves use outside single stocks:
  minus the cumulative return over months t-59..t-12 (the most recent year is skipped so
  value is not contaminated by momentum).
* **MOMENTUM** = the classic 12-1: cumulative return over months t-11..t-1 (the most
  recent month t is skipped).

Portfolio construction is identical everywhere: at each month-end t rank the sleeve's
assets on the signal (computed from data through t), go long the top third and short the
bottom third, equal-weighted 100%/100% (gross 2x NAV), and hold over month t+1 — the ONE
execution lag (signal known at the close of month t, position earns month t+1).
The 50/50 COMBO averages the VAL and MOM *weights* (so internal netting reduces
turnover), and the GLOBAL series is the equal-weight average of the sleeves live that
month (>= 2 sleeves required).

Statistics: HAC/Newey-West t's throughout (monthly L/S returns are serially correlated);
Sharpe races are excess-vs-excess (L/S portfolios are self-financing; futures returns are
excess by construction). The random-signal placebo is averaged over ``PLACEBO_SEEDS = 50``
seeds — single-seed baselines are banned by house rules. Costs: one-way cost x traded
notional (turnover x NAV), and the EQ sleeve's ETF short leg pays borrow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PLACEBO_SEEDS = 50
MIN_ASSETS = 3          # a sleeve-month needs at least this many ranked assets
MIN_SLEEVES = 2         # a global month needs at least this many live sleeves


# --------------------------------------------------------------------------- #
# Robust inference primitives
# --------------------------------------------------------------------------- #
def _nw_lags(n: int) -> int:
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def hac_mean(returns, lags: int | None = None) -> dict:
    """Newey-West (HAC) t for a sample mean (Bartlett kernel)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 12:
        return {"mean_bps": np.nan, "t": np.nan, "n": n}
    mu = r.mean()
    e = r - mu
    if lags is None:
        lags = _nw_lags(n)
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return {"mean_bps": mu * 1e4, "t": mu / se if se > 0 else np.nan, "n": n}


def sharpe(r: pd.Series) -> float:
    """Annualised Sharpe of a monthly excess-return series."""
    r = r.dropna()
    if len(r) < 12 or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1) * np.sqrt(12.0))


# --------------------------------------------------------------------------- #
# Signals — uniform across sleeves
# --------------------------------------------------------------------------- #
def signals(mret: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """VALUE (5y reversal, months t-59..t-12) and MOMENTUM (12-1, months t-11..t-1),
    both computed from returns through month t only. Complete windows required."""
    logr = np.log1p(mret)
    cum = logr.cumsum()
    # cumulative return over (t-a .. t-b] in log space, complete-window only
    def window(a: int, b: int, minlen: int) -> pd.DataFrame:
        w = cum.shift(b) - cum.shift(a)
        cnt = logr.notna().rolling(a - b).sum().shift(b)
        return w.where(cnt >= minlen)
    val = -window(60, 12, 44)     # months t-59..t-12 (48 months, allow a few gaps)
    mom = window(12, 1, 10)       # months t-11..t-1  (11 months, skip month t)
    return val, mom


def weights(sig: pd.DataFrame, min_assets: int = MIN_ASSETS) -> pd.DataFrame:
    """Long the top third, short the bottom third, equal weight (1 long / 1 short)."""
    w = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    for t in sig.index:
        row = sig.loc[t].dropna()
        n = len(row)
        if n < min_assets:
            w.loc[t] = np.nan
            continue
        k = max(1, int(round(n / 3.0)))
        order = row.sort_values()
        w.loc[t, order.index[-k:]] = 1.0 / k
        w.loc[t, order.index[:k]] = -1.0 / k
    return w


def portfolio(mret: pd.DataFrame, w: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """L/S return with the ONE execution lag (weights at month t earn month t+1),
    plus the one-way traded notional (sum |Δw|) per month."""
    wl = w.shift(1)
    live = wl.abs().sum(axis=1) > 0          # rows with an actual book
    ret = (wl * mret).sum(axis=1, min_count=1)[live].dropna()
    traded = (wl.fillna(0.0) - wl.shift(1).fillna(0.0)).abs().sum(axis=1)
    return ret, traded.reindex(ret.index).fillna(0.0)


# --------------------------------------------------------------------------- #
# Sleeve + global reports
# --------------------------------------------------------------------------- #
def sleeve_series(mret: pd.DataFrame) -> dict:
    """VAL, MOM and 50/50-combo monthly L/S series (+ combo turnover) for one sleeve."""
    val, mom = signals(mret)
    wv, wm = weights(val), weights(mom)
    rv, _ = portfolio(mret, wv)
    rm, _ = portfolio(mret, wm)
    wc = 0.5 * (wv.fillna(0.0) + wm.fillna(0.0))
    dead = ~(wv.notna().any(axis=1) | wm.notna().any(axis=1))
    wc.loc[dead] = np.nan
    rc, tc = portfolio(mret, wc)
    return {"val": rv, "mom": rm, "combo": rc, "combo_turnover": tc}


def sleeve_stats(ss: dict) -> dict:
    """Headline stats for one sleeve: Sharpes, HAC t's, VAL-MOM correlation."""
    both = pd.concat([ss["val"], ss["mom"]], axis=1, keys=["v", "m"]).dropna()
    rho = float(both["v"].corr(both["m"])) if len(both) > 24 else np.nan
    out = {"rho": rho, "n": len(ss["combo"]),
           "start": str(ss["combo"].index.min().date()) if len(ss["combo"]) else "",
           "end": str(ss["combo"].index.max().date()) if len(ss["combo"]) else ""}
    for leg in ("val", "mom", "combo"):
        h = hac_mean(ss[leg])
        out[leg] = {"sharpe": sharpe(ss[leg]), "t": h["t"],
                    "mean_bps": h["mean_bps"], "n": h["n"]}
    return out


def global_series(sleeves: dict[str, dict], leg: str,
                  min_sleeves: int = MIN_SLEEVES) -> pd.Series:
    """Equal-weight average of the sleeves' monthly L/S returns (>= 2 sleeves live)."""
    df = pd.concat({k: v[leg] for k, v in sleeves.items()}, axis=1)
    live = df.notna().sum(axis=1)
    return df.mean(axis=1)[live >= min_sleeves].dropna()


def diversification_check(rv: pd.Series, rm: pd.Series) -> dict:
    """Is the combo Sharpe just two-asset diversification arithmetic?

    Predicted Sharpe of a 50/50 mix of two series with Sharpes S_v, S_m, vols s_v, s_m
    and correlation rho — computed on the common sample — vs the realised combo Sharpe
    of the *same* 50/50 return mix. If realised ~= predicted, there is no magic beyond
    the arithmetic (that IS the finding; contrast study 401, where the ingredients were
    noise and the same arithmetic multiplied zero).
    """
    both = pd.concat([rv, rm], axis=1, keys=["v", "m"]).dropna()
    v, m = both["v"], both["m"]
    rho = float(v.corr(m))
    mu = 0.5 * (v.mean() + m.mean())
    var = 0.25 * (v.var(ddof=1) + m.var(ddof=1) + 2.0 * rho * v.std(ddof=1) * m.std(ddof=1))
    predicted = float(mu / np.sqrt(var) * np.sqrt(12.0))
    realized = sharpe(0.5 * (v + m))
    return {"rho": rho, "predicted": predicted, "realized": realized,
            "sr_val": sharpe(v), "sr_mom": sharpe(m), "n": len(both)}


# --------------------------------------------------------------------------- #
# Random-signal placebo (>= 20 seeds by house rule)
# --------------------------------------------------------------------------- #
def random_placebo(panels: dict[str, pd.DataFrame],
                   n_seeds: int = PLACEBO_SEEDS) -> dict:
    """Same construction, random signals: per seed, replace VAL and MOM with i.i.d.
    noise rank signals in every sleeve, build the same L/S + 50/50 combo + global
    average, and record the global-combo Sharpe and HAC t. Averaged over ``n_seeds``."""
    sh, ts = [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(1000 + seed)
        legs = {}
        for name, mret in panels.items():
            val, mom = signals(mret)                    # for live-mask alignment only
            noise_v = pd.DataFrame(rng.standard_normal(val.shape),
                                   index=val.index, columns=val.columns).where(val.notna())
            noise_m = pd.DataFrame(rng.standard_normal(mom.shape),
                                   index=mom.index, columns=mom.columns).where(mom.notna())
            wv, wm = weights(noise_v), weights(noise_m)
            wc = 0.5 * (wv.fillna(0.0) + wm.fillna(0.0))
            dead = ~(wv.notna().any(axis=1) | wm.notna().any(axis=1))
            wc.loc[dead] = np.nan
            rc, _ = portfolio(mret, wc)
            legs[name] = {"combo": rc}
        g = global_series(legs, "combo")
        sh.append(sharpe(g))
        ts.append(hac_mean(g)["t"])
    sh, ts = np.array(sh), np.array(ts)
    return {"mean_sharpe": float(np.nanmean(sh)), "sd_sharpe": float(np.nanstd(sh)),
            "mean_t": float(np.nanmean(ts)),
            "frac_t_ge_2": float(np.nanmean(np.abs(ts) >= 2.0)), "n_seeds": n_seeds}


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def net_global_combo(sleeves: dict[str, dict], cost_bps: float,
                     borrow_bps_yr: float = 50.0) -> dict:
    """Global 50/50 combo net of trading costs (one-way cost x traded notional each
    month, per sleeve) and, for the EQ sleeve only, ETF short-leg borrow at
    ``borrow_bps_yr`` on the short notional (1x NAV). FX spot and futures sleeves pay
    no borrow. Returns gross/net annualised mean, net Sharpe and net HAC t."""
    nets = {}
    for name, ss in sleeves.items():
        drag = ss["combo_turnover"] * cost_bps * 1e-4
        if name == "EQ":
            drag = drag + borrow_bps_yr * 1e-4 / 12.0
        nets[name] = {"combo": ss["combo"] - drag}
    g_gross = global_series(sleeves, "combo")
    g_net = global_series(nets, "combo")
    h = hac_mean(g_net)
    return {"cost_bps": cost_bps,
            "gross_ann_pct": float(g_gross.mean() * 12 * 100),
            "net_ann_pct": float(g_net.mean() * 12 * 100),
            "net_sharpe": sharpe(g_net), "net_t": h["t"]}


def avg_turnover(sleeves: dict[str, dict]) -> float:
    """Average monthly one-way traded notional of the combo, across sleeves."""
    tos = [ss["combo_turnover"].mean() for ss in sleeves.values()]
    return float(np.mean(tos))
