"""Strategy + inference for Study 721 — Most-Admired (Fortune's list as a return signal).

The claim, steelmanned two ways: (a) the **admiration premium** — the best-run firms on
earth compound faster than the market, so owning the list beats owning the index; (b) the
**reversal** — a firm becomes admired only once it is already loved and richly priced, so the
label marks a stock due to mean-revert (buy the spurned). We make both falsifiable with a
monthly **characteristic sort**:

  * **Excess-of-market returns.** An equal-weight admired book's month-end return in excess of
    SPY — the "premium" is supposed to be abnormal, not just being long equities.
  * **Market-model alpha.** Regress the book's return on SPY's return (CAPM-style): the
    intercept is the abnormal monthly return net of beta; the slope is how much of the "edge"
    is simply beta you were always paid for.
  * **HAC inference.** A **Newey-West** *t* on the mean monthly excess return (and on the
    market-model alpha) — monthly equity returns are mildly autocorrelated and fat-tailed, so
    an OLS/iid *t* overstates significance. REAL needs a HAC |t| >= 2 on the real tape.
  * **Publication lag / no look-ahead.** The honest variant owns a name only *after* Fortune
    first crowns it (Feb of its first_year); the naive variant owns the *current* list from
    day one and is therefore look-ahead selection. We report both and name the gap.
  * **Placebo.** Random equal-weight books of the same size from a broad large-cap pool — does
    a *random* basket of famous large caps beat SPY as much as the admired one?
  * **Costs.** Low turnover (annual rebalance to the new list); one-way bps x turnover x NAV.

The decisive object is the HAC *t* on the admiration premium, read against the look-ahead
caveat and the market-model beta: an equal-weight book of mega-cap winners can beat cap-
weighted SPY, but that is a size-within-large-cap + tech/quality tilt and a hindsight-selected
roster before it is an "admiration" effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Portfolio construction
# --------------------------------------------------------------------------- #
def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-over-month simple returns from a month-end price frame."""
    return prices.sort_index().pct_change()


def admired_book(prices: pd.DataFrame, admired: list, entry: dict | None = None,
                 lagged: bool = True) -> pd.Series:
    """Equal-weight monthly return of the admired book.

    If ``lagged`` (and ``entry`` given), a name contributes to a month's return only once the
    date is on/after its publication-lagged entry (Feb of first_year) — no look-ahead into
    which firms would later be crowned. If ``lagged`` is False, the *current* list is held for
    the whole sample (the naive, look-ahead book). Names not yet active are simply excluded
    from that month's equal-weight average (weights renormalise across the active set).
    """
    rets = monthly_returns(prices)
    tickers = [t for t, *_ in admired if t in rets.columns]
    R = rets[tickers].copy()
    if lagged and entry is not None:
        for t in tickers:
            e = entry.get(t)
            if e is not None:
                R.loc[R.index < e, t] = np.nan
    # equal weight across the names ACTIVE (non-NaN) that month
    return R.mean(axis=1, skipna=True)


def excess_over_market(book: pd.Series, prices: pd.DataFrame,
                       mkt: str = "SPY") -> pd.Series:
    """Book return minus the market (SPY) return, aligned on month-ends."""
    m = monthly_returns(prices)[mkt]
    df = pd.concat([book, m], axis=1).dropna()
    return (df.iloc[:, 0] - df.iloc[:, 1]).rename("excess")


def long_short(prices: pd.DataFrame, admired: list, spurned: list,
               entry: dict | None = None, lagged: bool = True) -> pd.Series:
    """Admired-minus-spurned equal-weight long/short monthly return (survivor-biased spurned)."""
    long_leg = admired_book(prices, admired, entry=entry, lagged=lagged)
    short_leg = admired_book(prices, [(t, 2004, lbl) for t, lbl in spurned],
                             entry=None, lagged=False)
    df = pd.concat([long_leg, short_leg], axis=1).dropna()
    return (df.iloc[:, 0] - df.iloc[:, 1]).rename("long_short")


# --------------------------------------------------------------------------- #
# Inference — Newey-West (HAC) t, market-model alpha
# --------------------------------------------------------------------------- #
def _nw_lags(n: int) -> int:
    """Newey-West automatic lag ~ floor(4*(n/100)^(2/9)) (Newey-West 1994 rule of thumb)."""
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west_t(x: np.ndarray, lags: int | None = None) -> dict:
    """HAC (Newey-West) t of ``mean(x) != 0`` for a mean-only regression.

    The Newey-West long-run variance of the sample mean:
        S = gamma_0 + 2 * sum_{l=1..L} (1 - l/(L+1)) * gamma_l
    with gamma_l the sample autocovariance at lag l (Bartlett kernel). The HAC SE of the mean
    is sqrt(S/n). Returns mean, HAC SE, HAC t, the annualised mean, and the lag used.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return {"n": n, "mean": float("nan"), "se": float("nan"), "t": float("nan"),
                "ann": float("nan"), "lags": 0}
    if lags is None:
        lags = _nw_lags(n)
    xc = x - x.mean()
    g0 = np.dot(xc, xc) / n
    S = g0
    for l in range(1, lags + 1):
        gl = np.dot(xc[l:], xc[:-l]) / n
        S += 2.0 * (1.0 - l / (lags + 1.0)) * gl
    S = max(S, 1e-18)
    se = np.sqrt(S / n)
    return {"n": n, "mean": float(x.mean()), "se": float(se),
            "t": float(x.mean() / se), "ann": float(x.mean() * 12), "lags": int(lags)}


def market_model_alpha(book: pd.Series, prices: pd.DataFrame, mkt: str = "SPY",
                       lags: int | None = None) -> dict:
    """CAPM-style regression book_ret = alpha + beta * mkt_ret; HAC t on alpha.

    Returns monthly alpha (%), its Newey-West t, beta, R^2, and the annualised alpha. The
    intercept is the book's abnormal monthly return *net of its market beta* — the honest
    "admiration alpha", separating edge from the beta you were always paid for.
    """
    m = monthly_returns(prices)[mkt]
    df = pd.concat([book.rename("y"), m.rename("x")], axis=1).dropna()
    if len(df) < 6:
        return {"alpha": float("nan"), "alpha_t": float("nan"), "beta": float("nan"),
                "r2": float("nan"), "alpha_ann": float("nan"), "n": int(len(df))}
    y = df["y"].to_numpy(); x = df["x"].to_numpy()
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = coef[0], coef[1]
    resid = y - X @ coef
    n = len(y)
    if lags is None:
        lags = _nw_lags(n)
    # HAC covariance of OLS coefficients (Newey-West, Bartlett kernel)
    XtX_inv = np.linalg.inv(X.T @ X)
    S = np.zeros((2, 2))
    u = X * resid[:, None]
    S += u.T @ u
    for l in range(1, lags + 1):
        G = u[l:].T @ u[:-l]
        w = 1.0 - l / (lags + 1.0)
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se_alpha = np.sqrt(max(cov[0, 0], 1e-24))
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else float("nan")
    return {"alpha": float(alpha), "alpha_t": float(alpha / se_alpha),
            "beta": float(beta), "r2": float(r2), "alpha_ann": float(alpha * 12),
            "n": int(n), "lags": int(lags)}


# --------------------------------------------------------------------------- #
# Placebo — random equal-weight books from a broad large-cap pool
# --------------------------------------------------------------------------- #
def placebo_pvalue(prices: pd.DataFrame, pool: list, k: int, observed_ann: float,
                   start=None, n_draws: int = 5000, seed: int = 721,
                   mkt: str = "SPY") -> dict:
    """How often does a RANDOM k-name large-cap book beat SPY by >= the admired book's premium?

    Draws ``n_draws`` random equal-weight k-name books from ``pool`` (survivors with full
    history over the common window), computes each one's annualised mean excess-over-SPY
    return, and returns the share whose |excess| >= |observed_ann|. The honest answer to
    "is a basket of famous large caps out-performing the cap-weighted index special, or just
    the equal-weight large-cap tilt?".
    """
    rets = monthly_returns(prices)
    avail = [t for t in pool if t in rets.columns and t != mkt]
    sub = rets[avail]
    if start is not None:
        sub = sub[sub.index >= pd.Timestamp(start)]
    m = rets[mkt].reindex(sub.index)
    # keep names with full history on the window (a fair random draw)
    full = [t for t in avail if sub[t].notna().all()]
    if len(full) < k:
        return {"k": k, "obs_ann": observed_ann, "p_value": float("nan"),
                "placebo_mean_ann": float("nan"), "n_full": len(full)}
    A = sub[full].to_numpy()
    mv = m.to_numpy()
    rng = np.random.default_rng(seed)
    excess = np.empty(n_draws)
    for i in range(n_draws):
        pick = rng.choice(len(full), size=k, replace=False)
        book = A[:, pick].mean(axis=1)
        excess[i] = np.nanmean(book - mv) * 12
    p = float((np.abs(excess) >= abs(observed_ann)).mean())
    return {"k": k, "obs_ann": float(observed_ann), "p_value": p,
            "placebo_mean_ann": float(excess.mean()), "n_full": len(full)}


# --------------------------------------------------------------------------- #
# Tradability — costs on the admired book
# --------------------------------------------------------------------------- #
def net_of_costs(excess: pd.Series, cost_bps: float = 10.0,
                 annual_rebalance_turnover: float = 0.20) -> dict:
    """Annualised gross vs net excess return of the admired book.

    The list changes slowly; owning it is close to buy-and-hold with an annual rebalance.
    ``annual_rebalance_turnover`` is the one-way fraction of the book that changes each year;
    the annual cost drag is ``turnover x (cost_bps/1e4)``. Returns annualised gross and net
    mean excess-of-SPY return. (As with the rest of the study, costs are not the binding
    constraint — the look-ahead selection is.)
    """
    x = np.asarray(excess, dtype=float)
    x = x[np.isfinite(x)]
    gross_ann = float(np.mean(x) * 12)
    drag = annual_rebalance_turnover * (cost_bps / 1e4)
    return {"gross_ann": gross_ann, "net_ann": float(gross_ann - drag),
            "cost_bps": cost_bps, "turnover": annual_rebalance_turnover, "drag": float(drag)}


# --------------------------------------------------------------------------- #
# Headline summary
# --------------------------------------------------------------------------- #
def summarize(bundle: dict, lagged: bool = True) -> dict:
    """Headline stats for the admired book: excess mean, HAC t, market-model alpha, sharpe."""
    prices = bundle["prices"]
    book = admired_book(prices, bundle["admired"], entry=bundle.get("entry"), lagged=lagged)
    ex = excess_over_market(book, prices)
    nw = newey_west_t(ex.to_numpy())
    mm = market_model_alpha(book, prices)
    x = ex.to_numpy(); x = x[np.isfinite(x)]
    sharpe = float(np.mean(x) / np.std(x, ddof=1) * np.sqrt(12)) if len(x) > 2 else float("nan")
    return {
        "n_months": int(len(ex)),
        "excess_mean_m": float(np.mean(x)),
        "excess_ann": float(np.mean(x) * 12),
        "hac_t": nw["t"], "hac_se": nw["se"], "hac_lags": nw["lags"],
        "alpha_m": mm["alpha"], "alpha_ann": mm["alpha_ann"], "alpha_t": mm["alpha_t"],
        "beta": mm["beta"], "r2": mm["r2"],
        "sharpe_excess": sharpe,
        "start": str(ex.index.min().date()), "end": str(ex.index.max().date()),
    }
