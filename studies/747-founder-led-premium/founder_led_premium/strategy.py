"""Strategy + inference for Study 747 — Founder-Led-Premium.

The claim: *founder-led firms outperform* (Fahlenbrach 2009; the Bain "founder's
mentality" thesis). We pin it down as a textbook **long/short characteristic sort**:

    Each month, equal-weight the **founder-led** basket (long) and the **professional-CEO**
    basket (short). The long/short return is ``r_LS = r_founder − r_pro``. Its **abnormal
    return** is the intercept of a market-model (CAPM) regression
    ``r_LS = alpha + beta·r_mkt + eps`` — the part of the spread the market factor does not
    explain — and we judge alpha with a **Newey-West (HAC) t-stat**.

We then interrogate whether any "premium" is a founder effect or an artefact:

  * **Long-only founder alpha** vs SPY (the rawest form of the believers' claim).
  * **Leave-one-out (jackknife)** — does the whole spread live in one or two names
    (NVDA, TSLA)? A premium that evaporates when you drop the top name is concentration,
    not a characteristic.
  * **Placebo** — random equal-size baskets drawn from the *pooled* universe: where does
    the real founder basket sit in the luck cloud? (The founder set is hindsight-selected
    from today's winners, so the honest yardstick is "could a random winner-heavy basket
    have done as well?")
  * **Costs / borrow** — one-way turnover cost on both legs plus a short-borrow charge on
    the professional leg; gross AND net alpha.

The decisive tension is not the sign of the spread (a basket of the founder firms we
remember in 2024 will, of course, look spectacular) but whether that spread is (a) robust
to dropping a name or two and (b) anything other than survivorship + tech-sector beta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# One-month execution convention: membership is frozen at formation and rebalanced to
# equal weight monthly; a position formed on information known at month t earns month t's
# return (calendar-known weights, no forward peek). No extra shift is applied.

TRADING_MONTHS = 12


# --------------------------------------------------------------------------- #
# Basket construction
# --------------------------------------------------------------------------- #
def basket_returns(rets: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """Equal-weighted monthly return of a basket, using whichever names have data.

    A name that is missing / delisted in a given month simply drops out of that month's
    equal weight (so an acquired founder firm like FIT contributes only while it trades).
    """
    cols = [t for t in tickers if t in rets.columns]
    if not cols:
        return pd.Series(dtype=float)
    return rets[cols].mean(axis=1, skipna=True)


def long_short(rets: pd.DataFrame, founder_tickers: list[str],
               pro_tickers: list[str]) -> pd.DataFrame:
    """Founder (long), professional (short) and their difference, monthly."""
    f = basket_returns(rets, founder_tickers)
    p = basket_returns(rets, pro_tickers)
    idx = f.index.intersection(p.index)
    out = pd.DataFrame({"founder": f.loc[idx], "pro": p.loc[idx]})
    out["ls"] = out["founder"] - out["pro"]
    return out.dropna()


# --------------------------------------------------------------------------- #
# Newey-West (HAC) inference
# --------------------------------------------------------------------------- #
def _nw_lags(n: int) -> int:
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def hac_mean_t(x: np.ndarray, lags: int | None = None) -> dict:
    """Newey-West HAC t-stat for the sample mean of ``x`` (Bartlett kernel)."""
    r = np.asarray(x, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 3:
        return {"mean": float("nan"), "t": float("nan"), "n": n, "lags": 0}
    mu = r.mean()
    e = r - mu
    if lags is None:
        lags = _nw_lags(n)
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return {"mean": float(mu), "t": float(mu / se) if se > 0 else float("nan"),
            "n": n, "lags": lags}


def capm_alpha(y: np.ndarray, mkt: np.ndarray, rf: float = 0.0,
               lags: int | None = None) -> dict:
    """Market-model regression ``y = alpha + beta·mkt + eps`` with a **Newey-West HAC**
    standard error on the intercept (the abnormal return / Jensen alpha).

    ``y`` and ``mkt`` are monthly returns; ``rf`` is a scalar monthly risk-free (default 0,
    labelled — near-zero for much of 2016-21). Returns monthly alpha (both raw and in bps),
    beta, HAC t on alpha, R², and n.
    """
    y = np.asarray(y, dtype=float) - rf
    m = np.asarray(mkt, dtype=float) - rf
    ok = np.isfinite(y) & np.isfinite(m)
    y, m = y[ok], m[ok]
    n = y.size
    if n < 5:
        return {"alpha": float("nan"), "alpha_bps": float("nan"), "beta": float("nan"),
                "t_alpha": float("nan"), "r2": float("nan"), "n": n, "lags": 0}
    X = np.column_stack([np.ones(n), m])
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ (X.T @ y)
    resid = y - X @ coef
    if lags is None:
        lags = _nw_lags(n)
    # Newey-West HAC covariance of the OLS coefficients (Bartlett kernel)
    S = (X * resid[:, None]).T @ (X * resid[:, None])            # gamma_0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        Xu = X * resid[:, None]
        Gk = Xu[k:].T @ Xu[:-k]
        S += w * (Gk + Gk.T)
    cov = XtX_inv @ S @ XtX_inv
    se_alpha = float(np.sqrt(max(cov[0, 0], 0.0)))
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"alpha": float(coef[0]), "alpha_bps": float(coef[0] * 1e4),
            "beta": float(coef[1]),
            "t_alpha": float(coef[0] / se_alpha) if se_alpha > 0 else float("nan"),
            "r2": r2, "n": n, "lags": lags}


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def annualize(mean_monthly: float) -> float:
    return (1.0 + mean_monthly) ** TRADING_MONTHS - 1.0


def sharpe_monthly(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(TRADING_MONTHS)) if sd > 0 else float("nan")


def summarize(rets: pd.DataFrame, founder_tickers: list[str], pro_tickers: list[str],
              mkt_col: str = "SPY", rf: float = 0.0) -> dict:
    """Headline stats: founder / pro / LS mean, LS CAPM alpha + HAC t, founder long-only
    alpha + HAC t, Sharpes."""
    ls = long_short(rets, founder_tickers, pro_tickers)
    mkt = rets[mkt_col].reindex(ls.index)
    out = {
        "n_months": int(len(ls)),
        "founder_ann": float(annualize(ls["founder"].mean())),
        "pro_ann": float(annualize(ls["pro"].mean())),
        "ls_mean_bps": float(ls["ls"].mean() * 1e4),
        "ls_ann": float(annualize(ls["ls"].mean())),
        "ls_sharpe": sharpe_monthly(ls["ls"].to_numpy()),
        "ls_hac": hac_mean_t(ls["ls"].to_numpy()),
        "ls_capm": capm_alpha(ls["ls"].to_numpy(), mkt.to_numpy(), rf=rf),
        "founder_capm": capm_alpha(ls["founder"].to_numpy(), mkt.to_numpy(), rf=rf),
        "pro_capm": capm_alpha(ls["pro"].to_numpy(), mkt.to_numpy(), rf=rf),
    }
    return out


# --------------------------------------------------------------------------- #
# Leave-one-out (jackknife) — is the spread one or two names?
# --------------------------------------------------------------------------- #
def jackknife_alpha(rets: pd.DataFrame, founder_tickers: list[str],
                    pro_tickers: list[str], mkt_col: str = "SPY",
                    rf: float = 0.0) -> pd.DataFrame:
    """Drop each founder name in turn; recompute the LS CAPM alpha + HAC t.

    A robust characteristic premium barely moves; a concentration artefact collapses when
    its top name is removed. Returns a frame sorted by resulting alpha (most-load-bearing
    name first)."""
    mkt = rets[mkt_col]
    rows = []
    avail = [t for t in founder_tickers if t in rets.columns]
    for drop in avail:
        keep = [t for t in avail if t != drop]
        ls = long_short(rets, keep, pro_tickers)
        m = mkt.reindex(ls.index)
        c = capm_alpha(ls["ls"].to_numpy(), m.to_numpy(), rf=rf)
        rows.append({"dropped": drop, "alpha_bps": c["alpha_bps"],
                     "t_alpha": c["t_alpha"], "beta": c["beta"]})
    df = pd.DataFrame(rows).sort_values("alpha_bps").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Placebo — random equal-size baskets from the pooled universe
# --------------------------------------------------------------------------- #
def placebo_alpha_dist(rets: pd.DataFrame, pool: list[str], k_long: int, k_short: int,
                       mkt_col: str = "SPY", n_draws: int = 5000, rf: float = 0.0,
                       seed: int = 747) -> np.ndarray:
    """Null distribution of the LS CAPM alpha when the two baskets are drawn at random.

    Repeatedly split the pooled universe into a random ``k_long`` long / ``k_short`` short
    and record the market-model alpha. This asks: given these very names, how often does a
    *random* long/short label reproduce the founder basket's alpha? (It does NOT undo the
    hindsight selection of the pool itself — that bias is named separately on the Signal
    axis; this null isolates the founder *label* from basket-membership luck.)"""
    rng = np.random.default_rng(seed)
    pool = [t for t in pool if t in rets.columns]
    mkt = rets[mkt_col].to_numpy()
    out = np.empty(n_draws)
    for i in range(n_draws):
        perm = rng.permutation(pool)
        longs = list(perm[:k_long])
        shorts = list(perm[k_long:k_long + k_short])
        ls = long_short(rets, longs, shorts)
        m = pd.Series(mkt, index=rets.index).reindex(ls.index)
        out[i] = capm_alpha(ls["ls"].to_numpy(), m.to_numpy(), rf=rf)["alpha_bps"]
    return out[np.isfinite(out)]


def placebo_pvalue(obs: float, null: np.ndarray, two_sided: bool = True) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0:
        return float("nan")
    if two_sided:
        c = null.mean()
        return float((np.abs(null - c) >= abs(obs - c)).mean())
    return float((null >= obs).mean())


# --------------------------------------------------------------------------- #
# Costs + borrow
# --------------------------------------------------------------------------- #
def net_of_costs(ls_mean_monthly: float, n_names_long: int, n_names_short: int,
                 cost_bps: float = 10.0, borrow_ann_bps: float = 300.0,
                 rebalance_turnover: float = 0.15) -> dict:
    """Net monthly LS return after trading costs + short borrow.

    - **Trading cost**: ``cost_bps`` one-way × the fraction of the book turned over each
      month to restore equal weight (``rebalance_turnover``, both legs).
    - **Borrow**: shorting quality large-caps is cheap but not free; ``borrow_ann_bps``
      annual on the short leg, charged monthly. The founder shorts (professional peers)
      are easy to borrow, so this is deliberately light.

    Returns gross vs net monthly bps. The point, as usual, is that costs are *not* the
    binding constraint here — the hindsight selection is."""
    c = (cost_bps / 1e4) * rebalance_turnover * 2.0        # both legs, one-way
    b = (borrow_ann_bps / 1e4) / TRADING_MONTHS            # monthly borrow on the short
    net = ls_mean_monthly - c - b
    return {"gross_bps": float(ls_mean_monthly * 1e4), "net_bps": float(net * 1e4),
            "cost_bps": cost_bps, "borrow_ann_bps": borrow_ann_bps,
            "monthly_drag_bps": float((c + b) * 1e4)}
