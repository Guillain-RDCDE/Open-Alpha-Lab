"""Strategy + inference for Study 381 — TIPS-Breakeven.

The claim under test: the bond market's **breakeven inflation rate** (here a transparent
proxy, ``be = log(TIP / IEF)``) is a tradable macro timing signal — its *level* or *change*
predicts forward returns of inflation hedges (TIPS, gold) and/or equities.

We build two signals from the month-end breakeven proxy:

  * **level-z** — the expanding z-score of ``be`` (how high is breakeven vs its own history);
  * **3m-change** — the trailing 3-month change in ``be`` (is breakeven rising or falling).

and confront each against forward ``H``-month returns of an asset with:

  * a **predictive regression** ``r_{t->t+H} = a + b * signal_t + e`` and its **HAC
    (Newey-West) t-stat** of the slope ``b`` — the honest test when forward windows overlap
    (overlap induces serial correlation that a naive OLS t ignores);
  * a **placebo / randomization null** — shuffle the signal-to-return pairing many times and
    ask how often a random alignment beats the observed ``|t|`` (the small-sample workhorse);
  * a **directional win-rate** vs the unconditional base rate;
  * a **traded** sign rule with a **1-month execution lag** and **one-way costs × turnover**.

The decisive object is not any single point estimate but the **multiple-testing** picture:
across signal × asset × horizon, an honest ``|t| >= 2`` survives on a genuinely independent
asset only by luck — and the one that does (TIPS itself) is a mechanical self-prediction
artifact, since the signal is built from TIP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import breakeven_proxy


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
def breakeven_series(monthly: pd.DataFrame) -> pd.Series:
    """Month-end breakeven proxy ``log(TIP/IEF)`` (real frame) or the ``be`` column (synthetic)."""
    if "be" in monthly.columns:
        return monthly["be"].rename("be")
    return breakeven_proxy(monthly)


def level_z(be: pd.Series, min_obs: int = 24) -> pd.Series:
    """Expanding z-score of the breakeven level (uses only past data — no look-ahead)."""
    mu = be.expanding(min_obs).mean()
    sd = be.expanding(min_obs).std()
    return ((be - mu) / sd).rename("level_z")


def change_3m(be: pd.Series, k: int = 3) -> pd.Series:
    """Trailing ``k``-month change in the breakeven proxy."""
    return be.diff(k).rename("change_3m")


def forward_return(price: pd.Series, h: int) -> pd.Series:
    """Forward ``h``-month return for every month (NaN near the tail)."""
    return price.shift(-h) / price - 1.0


# --------------------------------------------------------------------------- #
# Inference — HAC predictive regression
# --------------------------------------------------------------------------- #
def hac_regression(signal: pd.Series, fwd: pd.Series, lags: int) -> dict:
    """OLS of ``fwd ~ a + b*signal`` with a **Newey-West (HAC)** t-stat on the slope.

    Overlapping forward windows make residuals serially correlated; the HAC correction
    (Bartlett kernel, ``lags`` bandwidth — set to the horizon) gives an honest standard error.
    Returns slope, HAC t, R², and n.
    """
    x = np.asarray(signal, float)
    y = np.asarray(fwd, float)
    msk = np.isfinite(x) & np.isfinite(y)
    x, y = x[msk], y[msk]
    n = len(x)
    if n < 10:
        return {"slope": float("nan"), "t": float("nan"), "r2": float("nan"), "n": n}
    X = np.column_stack([np.ones_like(x), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    XtX_inv = np.linalg.inv(X.T @ X)
    # Newey-West HAC sandwich
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = (X[L:] * resid[L:, None]).T @ (X[:-L] * resid[:-L, None])
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else float("nan")
    t = float(b[1] / se[1]) if se[1] > 0 else float("nan")
    return {"slope": float(b[1]), "t": t, "r2": float(r2), "n": n}


def placebo_pvalue(signal: pd.Series, fwd: pd.Series, lags: int,
                   n_draws: int = 5_000, seed: int = 381) -> float:
    """Randomization null: how often does a *shuffled* signal beat the observed ``|t|``?

    Holds the forward returns fixed and circularly shuffles the signal (breaking any genuine
    alignment while preserving each series' marginal distribution), recomputing the HAC t.
    Returns ``P[|t_shuffled| >= |t_obs|]`` — the small-sample answer to "could random
    alignment have looked this predictive?"
    """
    x = np.asarray(signal, float)
    y = np.asarray(fwd, float)
    msk = np.isfinite(x) & np.isfinite(y)
    x, y = x[msk], y[msk]
    if len(x) < 10:
        return float("nan")
    t_obs = abs(_hac_t(x, y, lags))
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_draws):
        shift = rng.integers(1, len(x))
        xs = np.roll(x, shift)             # circular shuffle preserves the marginal
        if abs(_hac_t(xs, y, lags)) >= t_obs:
            hits += 1
    return hits / n_draws


def _hac_t(x: np.ndarray, y: np.ndarray, lags: int) -> float:
    X = np.column_stack([np.ones_like(x), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = (X[L:] * resid[L:, None]).T @ (X[:-L] * resid[:-L, None])
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return float(b[1] / se[1]) if se[1] > 0 else 0.0


# --------------------------------------------------------------------------- #
# Win-rate vs base rate
# --------------------------------------------------------------------------- #
def directional_winrate(signal: pd.Series, fwd: pd.Series) -> dict:
    """Win-rate of a sign rule (go with the slope's sign) vs the unconditional base rate.

    Returns the conditional win-rate (fraction of months the *signed* trade is positive) and
    the unconditional fraction of positive forward returns (the honest base rate).
    """
    x = np.asarray(signal, float)
    y = np.asarray(fwd, float)
    msk = np.isfinite(x) & np.isfinite(y)
    x, y = x[msk], y[msk]
    if len(x) < 10:
        return {"cond_win": float("nan"), "base_win": float("nan"), "n": len(x)}
    # OLS sign of the slope tells us whether to go long or short on a positive signal
    b = np.polyfit(x, y, 1)[0]
    traded = np.sign(b) * np.sign(x) * y
    return {"cond_win": float((traded > 0).mean()),
            "base_win": float((y > 0).mean()), "n": len(x)}


# --------------------------------------------------------------------------- #
# Tradable backtest — 1-month execution lag + one-way costs × turnover
# --------------------------------------------------------------------------- #
def sign_timer(price: pd.Series, signal: pd.Series, contrarian: bool = False,
               cost_bps: float = 10.0) -> dict:
    """Trade a monthly sign rule on the breakeven signal with a 1-month execution lag.

    Position next month = ``sign(signal_t)`` (or its negative if ``contrarian``); held one
    month; ``cost_bps`` one-way cost applied to monthly turnover (|Δposition|). Returns
    annualised gross/net mean, net Sharpe, turnover, and the buy-and-hold comparison.
    """
    px = price.dropna()
    r = px.pct_change().shift(-1)                       # next-month return (entry at t+1 close)
    pos = np.sign(signal.reindex(px.index)).fillna(0.0)
    if contrarian:
        pos = -pos
    aligned = pd.DataFrame({"r": r, "pos": pos}).dropna()
    strat = aligned["pos"] * aligned["r"]
    turn = aligned["pos"].diff().abs().fillna(0.0)
    net = strat - turn * (cost_bps / 1e4)
    bh = aligned["r"]
    ann = lambda s: float(s.mean() * 12)
    shp = lambda s: float(s.mean() / s.std() * np.sqrt(12)) if s.std() > 0 else float("nan")
    return {
        "n": int(len(strat)),
        "gross_ann": ann(strat), "net_ann": ann(net),
        "net_sharpe": shp(net), "turnover_ann": float(turn.mean() * 12),
        "bh_ann": ann(bh), "bh_sharpe": shp(bh), "cost_bps": cost_bps,
    }


# --------------------------------------------------------------------------- #
# One-shot summary across signal × asset × horizon
# --------------------------------------------------------------------------- #
def scan(monthly: pd.DataFrame, asset: str, signals: dict | None = None,
         horizons=(3, 6, 12)) -> list[dict]:
    """HAC predictive-regression scan for one asset over the two signals and the horizons."""
    be = breakeven_series(monthly)
    if signals is None:
        signals = {"level-z": level_z(be), "3m-change": change_3m(be)}
    price = monthly[asset]
    rows = []
    for sname, sig in signals.items():
        for h in horizons:
            fwd = forward_return(price, h)
            reg = hac_regression(sig, fwd, lags=h)
            rows.append({"signal": sname, "asset": asset, "h": h, **reg})
    return rows
