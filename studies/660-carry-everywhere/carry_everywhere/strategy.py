"""Strategy + inference for Study 660 — Carry-Everywhere.

The claim (Koijen-Moskowitz-Pedersen-Vrugt 2018, "Carry"): **a carry signal — the
return you'd earn if prices never moved — predicts returns in every major asset
class**, and because the four carry sleeves are only loosely correlated, a
diversified cross-asset carry basket is a more *robust* premium than any single one
of FX/bond/equity/commodity carry alone.

Construction. Four **static** long/short sleeves (composition fixed *ex ante* from
each asset class's textbook carry classification, not fit to the sample — like a
public calendar, this involves **zero look-ahead**; see ``data.py`` for the ticker
choice and reasoning):

* **FX** — long AUD, NZD / short JPY, CHF (equal-weighted, dollar-neutral).
* **BOND** — long IEF (7-10y UST) / short SHY (1-3y UST) — the term-spread trade.
* **EQ** — long VYM (high dividend yield) / short VUG (growth, low yield).
* **CMD** — long DBC (carry-optimised roll) / short GSG (naive front-month roll).

The one documented execution convention: **monthly rebalancing to par weights at
each month-end close** (returns are month-end-to-month-end; there is no signal
formed from the data to look ahead on — the basket itself never changes).

Measurements: HAC (Newey-West) *t* on each sleeve's and the combo's mean monthly
return (one-sample, since these are single non-cross-sectional time series, not
paired groups); Sharpe (excess-vs-cash by construction — every sleeve is
dollar-neutral, hence self-financing); a circular block-bootstrap CI on the combo
Sharpe; skew and max drawdown; and the two hardcoded crisis windows (2008 GFC,
2020 COVID) for the "does carry crash when you need it not to" tail check. Costs:
one-way cost x monthly rebalancing turnover (first-order: weight x |leg return|),
plus a modest short-leg borrow spread on the two ETF sleeves (EQ, CMD) — FX/Treasury
legs carry no separate stock-loan fee (the differential the carry proxy captures
already **is** the financing cost).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

MONTHS_YR = 12

# --------------------------------------------------------------------------- #
# Sleeve construction — a fixed long/short leg list per asset class:
# (weight, ticker, sign) — sign=+1 means "ticker's own return is the carry payoff",
# used identically for the two directly-long-short pairs and for the FX basket
# (whose 4 legs, per data.py's ticker choice, are ALL already correctly signed).
# --------------------------------------------------------------------------- #
SLEEVE_LEGS = {
    "FX": [(0.5, "AUDUSD=X"), (0.5, "NZDUSD=X"), (0.5, "JPY=X"), (0.5, "CHF=X")],
    "BOND": [(1.0, dt.BOND_LONG), (-1.0, dt.BOND_SHORT)],
    "EQ": [(1.0, dt.EQ_LONG), (-1.0, dt.EQ_SHORT)],
    "CMD": [(1.0, dt.CMD_LONG), (-1.0, dt.CMD_SHORT)],
}
# note: FX legs use +0.5 for all four tickers because AUDUSD=X/NZDUSD=X are
# already "long AUD/NZD" and JPY=X/CHF=X are already "long USD vs JPY/CHF"
# (i.e. short JPY/CHF) in their raw quoted direction — see data.py's docstring.

# short-leg annual borrow spread (bps/yr on 1x short notional); FX/Treasury legs
# pay none (the carry proxy itself already IS the financing differential)
BORROW_BPS_YR = {"FX": 0.0, "BOND": 0.0, "EQ": 40.0, "CMD": 25.0}


def sleeve_gross(mret: pd.DataFrame, sleeve: str) -> pd.Series:
    """Monthly gross return of one sleeve: sum of weight x leg return."""
    legs = SLEEVE_LEGS[sleeve]
    cols = [mret[t] for _, t in legs]
    w = np.array([w for w, _ in legs])
    df = pd.concat(cols, axis=1).dropna()
    return pd.Series(df.values @ w, index=df.index, name=sleeve)


def sleeve_turnover(mret: pd.DataFrame, sleeve: str) -> pd.Series:
    """First-order monthly rebalance-to-par turnover: sum_legs |weight * leg_return|
    (the trade needed to reset each leg back to its target weight)."""
    legs = SLEEVE_LEGS[sleeve]
    df = pd.concat([mret[t] for _, t in legs], axis=1).dropna()
    w = np.array([w for w, _ in legs])
    return pd.Series((df.values * w).__abs__().sum(axis=1), index=df.index)


def all_sleeves(mret: pd.DataFrame) -> pd.DataFrame:
    """Gross monthly returns for the four sleeves, common index."""
    out = {s: sleeve_gross(mret, s) for s in SLEEVE_LEGS}
    return pd.concat(out, axis=1).dropna()


def all_turnover(mret: pd.DataFrame) -> pd.DataFrame:
    return pd.concat({s: sleeve_turnover(mret, s) for s in SLEEVE_LEGS}, axis=1).dropna()


def sleeve_net(mret: pd.DataFrame, sleeve: str, cost_bps: float) -> pd.Series:
    """Sleeve return net of one-way rebalance cost x turnover, plus the sleeve's
    annual short-leg borrow spread (prorated monthly on 1x short notional)."""
    gross = sleeve_gross(mret, sleeve)
    to = sleeve_turnover(mret, sleeve).reindex(gross.index)
    drag = to * cost_bps * 1e-4 + BORROW_BPS_YR[sleeve] * 1e-4 / MONTHS_YR
    return (gross - drag).dropna()


def combo(sleeves: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    """Cross-asset combo — equal-weight (1/4 each) unless ``weights`` given."""
    cols = list(sleeves.columns)
    if weights is None:
        w = np.full(len(cols), 1.0 / len(cols))
    else:
        w = np.array([weights[c] for c in cols])
    return pd.Series(sleeves.values @ w, index=sleeves.index, name="COMBO")


def inv_vol_weights(sleeves: pd.DataFrame) -> dict[str, float]:
    """Inverse full-sample volatility weights, renormalised to sum to 1 (a
    robustness-check combo weighting, secondary to the primary equal-weight one)."""
    vol = sleeves.std(ddof=1)
    inv = 1.0 / vol
    w = inv / inv.sum()
    return w.to_dict()


def combo_net(mret: pd.DataFrame, cost_bps: float,
              weights: dict[str, float] | None = None) -> pd.Series:
    nets = {s: sleeve_net(mret, s, cost_bps) for s in SLEEVE_LEGS}
    df = pd.concat(nets, axis=1).dropna()
    return combo(df, weights)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def _nw_lags(n: int) -> int:
    return max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def hac_mean(returns, lags: int | None = None) -> dict:
    """Newey-West (HAC, Bartlett kernel) t of a one-sample mean."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 12:
        return {"mean_bps": float("nan"), "t": float("nan"), "n": n}
    mu = r.mean()
    e = r - mu
    if lags is None:
        lags = _nw_lags(n)
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return {"mean_bps": float(mu * 1e4), "t": float(mu / se) if se > 0 else float("nan"), "n": n}


def sharpe(r: pd.Series) -> float:
    r = pd.Series(r).dropna()
    if len(r) < 12 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(MONTHS_YR))


def skewness(r: pd.Series) -> float:
    r = pd.Series(r).dropna()
    if len(r) < 3:
        return float("nan")
    return float(r.skew())


def max_drawdown(r: pd.Series) -> float:
    r = pd.Series(r).dropna()
    if len(r) == 0:
        return float("nan")
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def block_bootstrap_sharpe_ci(r: pd.Series, block: int = 6, n_boot: int = 2000,
                              seed: int = 660) -> tuple[float, float]:
    """Circular block-bootstrap 95% CI on the annualised Sharpe (i.i.d. resampling
    destroys the serial correlation the inference is supposed to respect)."""
    x = np.asarray(pd.Series(r).dropna(), dtype=float)
    n = len(x)
    if n < 24:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    sh = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        idx = np.concatenate([np.arange(s, s + block) % n for s in starts])[:n]
        xb = x[idx]
        sh[b] = xb.mean() / xb.std(ddof=1) * np.sqrt(MONTHS_YR) if xb.std(ddof=1) > 0 else np.nan
    lo, hi = np.nanpercentile(sh, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Crisis-window tail check
# --------------------------------------------------------------------------- #
def crisis_stats(r: pd.Series, window: tuple[str, str]) -> dict:
    """Cumulative return of ``r`` (PeriodIndex-monthly) inside a crisis window vs
    the mean of all other months."""
    months = dt.crisis_months(window)
    inside = r[r.index.isin(months)]
    outside = r[~r.index.isin(months)]
    cum_in = float((1.0 + inside).prod() - 1.0) if len(inside) else float("nan")
    return {"n_in": len(inside), "cum_return_pct": cum_in * 100,
            "mean_other_mo_pct": float(outside.mean() * 100) if len(outside) else float("nan")}


# --------------------------------------------------------------------------- #
# Synthetic-control detector — the machinery proof (faithful-engine / power check)
# --------------------------------------------------------------------------- #
def synthetic_detect(carry_bps_mo: float, crash_beta: float, seed: int) -> dict:
    """Run the headline HAC-mean test on a synthetic equal-weight combo."""
    sl = dt.synthetic_sleeves(carry_bps_mo=carry_bps_mo, crash_beta=crash_beta, seed=seed)
    cb = combo(sl)
    h = hac_mean(cb)
    return {"sharpe": sharpe(cb), "t": h["t"], "mean_bps": h["mean_bps"],
            "skew": skewness(cb), "n": h["n"]}
