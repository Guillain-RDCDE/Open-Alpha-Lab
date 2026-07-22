"""Strategy + inference for Study 790 — Composite Equity Issuance (Daniel-Titman 2006).

The claim: a firm's **5-year composite equity issuance** — the part of its 5-year log
market-cap growth NOT explained by its own cumulative stock return — predicts the cross-section
of returns. High composite issuers (net dilution: SEOs, share-count drift) **underperform**;
low/negative issuers (net buyback-ers) **outperform**. We test it as a clean annual
cross-sectional sort:

For each formation year-end *t* (the composite window [t-5, t] is fully public — the share
count is point-in-time filed), we compute every name's composite issuance

    iota_{i,t} = log( ME_{i,t} / ME_{i,t-5} ) - r_{i}(t-5, t),

rank the cross-section, go **long the low/negative-issuance** quantile (buyback-ers) and
**short the high-issuance** quantile (diluters), and hold for the **next** year (one execution
lag — we trade *after* the formation year closes, never the same bar). We record the long-short
annual return, test its mean against zero with a one-sample t, cross-check with a Newey-West
HAC t, stress it with a **label-shuffle placebo**, charge **costs x turnover + short borrow**,
and cut it by sub-period and quantile width.

The decisive object is the long-short mean against its standard error: on a ~40-name survivor
basket over ~7 annual rebalances, a few-percent spread is easily inside the sampling noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# The signal: 5-year composite equity issuance
# --------------------------------------------------------------------------- #
def composite_issuance(raw_px: pd.DataFrame, adj_px: pd.DataFrame, shares: pd.DataFrame,
                       lookback: int = 5) -> pd.DataFrame:
    """Daniel-Titman 5-year composite issuance per formation year-end.

        iota_{i,t} = log( ME_{i,t} / ME_{i,t-5} ) - log( adj_{i,t} / adj_{i,t-5} )

    where ``ME = shares x raw_price`` is market cap and the second term is the 5-year cumulative
    **total** (adjusted-close) log return. Positive iota = the firm grew its market cap by more
    than price appreciation alone, i.e. it **issued** net equity; negative iota = it shrank the
    share base faster than dividends, i.e. net **buyback**. Splits cancel between the two legs.
    Rows with an undefined 5-year window are NaN.
    """
    me = shares * raw_px
    with np.errstate(invalid="ignore", divide="ignore"):
        log_me_growth = np.log(me / me.shift(lookback))
        cum_logret = np.log(adj_px / adj_px.shift(lookback))
    return log_me_growth - cum_logret


def forward_returns(adj_px: pd.DataFrame) -> pd.DataFrame:
    """Year-ahead total return per name: ``adj_{t+1}/adj_t - 1`` (adjusted closes). The value on
    formation row *t* is the return earned over the holding year *t -> t+1* (one execution lag)."""
    return adj_px.pct_change().shift(-1)


# --------------------------------------------------------------------------- #
# Cross-sectional long-short
# --------------------------------------------------------------------------- #
def _quantile_legs(sig_row: pd.Series, frac: float = 0.3) -> tuple[list, list]:
    """Names in the bottom (low/negative-issuance, LONG) and top (high-issuance, SHORT) ``frac``
    of a formation-year cross-section of composite issuance. Returns ``(long_names, short_names)``."""
    s = sig_row.dropna()
    if len(s) < 4:
        return [], []
    k = max(1, int(round(len(s) * frac)))
    order = s.sort_values()
    longs = list(order.index[:k])            # lowest composite issuance (buybacks) -> long
    shorts = list(order.index[-k:])          # highest composite issuance (dilution) -> short
    return longs, shorts


def long_short_series(raw_px: pd.DataFrame, adj_px: pd.DataFrame, shares: pd.DataFrame,
                      frac: float = 0.3, lookback: int = 5) -> pd.DataFrame:
    """Build the annual long-short panel.

    For each formation year-end *t* with a defined composite cross-section, sort names, form an
    equal-weight LONG (low issuance) and SHORT (high issuance) basket, and realise their
    **next-year** total returns (the one execution lag — the composite window is public at *t*,
    we hold *t->t+1*). Returns a frame indexed by formation year-end with columns
    ``long_ret, short_ret, ls_ret`` and the basket sizes.
    """
    sig = composite_issuance(raw_px, adj_px, shares, lookback=lookback)
    fwd = forward_returns(adj_px)
    rows = []
    for t in sig.index:
        if t not in fwd.index:
            continue
        longs, shorts = _quantile_legs(sig.loc[t], frac=frac)
        if not longs or not shorts:
            continue
        fr = fwd.loc[t]
        lr = fr.reindex(longs).dropna()
        sr = fr.reindex(shorts).dropna()
        if len(lr) == 0 or len(sr) == 0:
            continue
        long_ret, short_ret = float(lr.mean()), float(sr.mean())
        rows.append({"date": t, "long_ret": long_ret, "short_ret": short_ret,
                     "ls_ret": long_ret - short_ret, "n_long": len(lr), "n_short": len(sr)})
    if not rows:
        return pd.DataFrame(columns=["long_ret", "short_ret", "ls_ret", "n_long", "n_short"])
    return pd.DataFrame(rows).set_index("date")


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray, mu0: float = 0.0) -> float:
    """One-sample t of ``mean(x) - mu0``. NaN if fewer than 2 finite points."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float((x.mean() - mu0) / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances). NaN if either < 2 obs."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 1) -> float:
    """HAC (Newey-West, Bartlett kernel) t of the mean of ``x`` against zero — the
    autocorrelation-robust cross-check on the annual long-short series (overlapping 5-year
    formation windows induce serial dependence, so a plain one-sample t understates the SE)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    u = x - x.mean()
    S = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        g = float(u[l:] @ u[:-l]) / n
        S += 2.0 * w * g
    if S <= 0:
        return float("nan")
    se = np.sqrt(S / n)
    return float(x.mean() / se) if se > 0 else float("nan")


def ann_sharpe(x: np.ndarray) -> float:
    """Sharpe of an annual return series (already annual; mean/std, no scaling)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share k/n (the long-short win-rate)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Label-shuffle placebo
# --------------------------------------------------------------------------- #
def placebo_pvalue(raw_px, adj_px, shares, frac: float = 0.3, lookback: int = 5,
                   n_draws: int = 20_000, seed: int = 790) -> dict:
    """Label-shuffle placebo null: within each formation year, **permute** which name carries
    which composite-issuance value (destroying the issuance->name link but preserving the
    marginal issuance distribution and the realised cross-section of forward returns), rebuild
    the long-short, and ask how often a shuffled book's mean LS return **beats** the real sort.

    Returns the observed mean, the placebo mean, and ``p = P[placebo mean >= observed mean]``.
    """
    real = long_short_series(raw_px, adj_px, shares, frac=frac, lookback=lookback)
    obs = float(real["ls_ret"].mean()) if len(real) else float("nan")
    if not np.isfinite(obs):
        return {"n_years": 0, "obs_mean": float("nan"),
                "placebo_mean": float("nan"), "p_value": float("nan")}

    sig = composite_issuance(raw_px, adj_px, shares, lookback=lookback)
    fwd = forward_returns(adj_px)
    years = []
    for t in sig.index:
        if t not in fwd.index:
            continue
        srow = sig.loc[t].dropna()
        frow = fwd.loc[t].reindex(srow.index).dropna()
        common = srow.index.intersection(frow.index)
        if len(common) < 4:
            continue
        years.append((srow.loc[common].to_numpy(), frow.loc[common].to_numpy()))
    if not years:
        return {"n_years": 0, "obs_mean": obs,
                "placebo_mean": float("nan"), "p_value": float("nan")}

    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for d in range(n_draws):
        yr_ls = []
        for iss, ret in years:
            n = len(iss)
            k = max(1, int(round(n * frac)))
            perm = rng.permutation(n)
            order = np.argsort(iss[perm])
            yr_ls.append(ret[order[:k]].mean() - ret[order[-k:]].mean())
        means[d] = float(np.mean(yr_ls))
    return {"n_years": int(len(years)), "obs_mean": obs,
            "placebo_mean": float(means.mean()), "p_value": float((means >= obs).mean())}


# --------------------------------------------------------------------------- #
# Headline summary
# --------------------------------------------------------------------------- #
def summarize(raw_px, adj_px, shares, frac: float = 0.3, lookback: int = 5,
              placebo: bool = True) -> dict:
    """Headline long-short stats: n years, long/short/LS annual means, win-rate + Wilson CI,
    annual Sharpe, one-sample t (vs 0), Newey-West t, and (optionally) the label-shuffle p."""
    ls = long_short_series(raw_px, adj_px, shares, frac=frac, lookback=lookback)
    x = ls["ls_ret"].to_numpy() if len(ls) else np.array([])
    k_win = int((x > 0).sum()) if len(x) else 0
    lo, hi = wilson_interval(k_win, len(x)) if len(x) else (float("nan"), float("nan"))
    out = {
        "n_years": int(len(ls)),
        "long_mean": float(ls["long_ret"].mean()) if len(ls) else float("nan"),
        "short_mean": float(ls["short_ret"].mean()) if len(ls) else float("nan"),
        "ls_mean": float(ls["ls_ret"].mean()) if len(ls) else float("nan"),
        "ls_median": float(ls["ls_ret"].median()) if len(ls) else float("nan"),
        "win": float((x > 0).mean()) if len(x) else float("nan"),
        "win_lo": lo, "win_hi": hi,
        "sharpe": ann_sharpe(x),
        "t": one_sample_t(x, 0.0),
        "nw_t": newey_west_t(x, lags=1),
    }
    if placebo:
        out["p_placebo"] = placebo_pvalue(raw_px, adj_px, shares,
                                          frac=frac, lookback=lookback)["p_value"]
    return out


# --------------------------------------------------------------------------- #
# Costs + borrow (the timer)
# --------------------------------------------------------------------------- #
def net_of_costs(raw_px, adj_px, shares, frac: float = 0.3, lookback: int = 5,
                 cost_bps: float = 10.0, borrow_ann_bps: float = 50.0,
                 turnover: float = 1.0) -> dict:
    """Charge the long-short book one-way costs x turnover + a short-leg borrow.

    The sort is re-struck every year and equal-weight quantile baskets turn over almost
    completely year to year, so we charge ``turnover`` (default 1.0 = full) x one-way
    ``cost_bps`` on **both** legs per annual rebalance, plus a short-leg annual borrow
    ``borrow_ann_bps``. Returns gross and net annualised LS mean and the net t.
    """
    ls = long_short_series(raw_px, adj_px, shares, frac=frac, lookback=lookback)
    if len(ls) == 0:
        return {"n_years": 0, "gross_mean": float("nan"), "net_mean": float("nan"),
                "net_t": float("nan"), "cost_bps": cost_bps, "borrow_bps": borrow_ann_bps}
    gross = ls["ls_ret"].to_numpy()
    net = gross - 2.0 * cost_bps * 1e-4 * turnover - borrow_ann_bps * 1e-4
    return {"n_years": int(len(ls)), "gross_mean": float(gross.mean()),
            "net_mean": float(net.mean()), "net_t": one_sample_t(net, 0.0),
            "cost_bps": cost_bps, "borrow_bps": borrow_ann_bps}


# --------------------------------------------------------------------------- #
# Robustness — quantile-width sweep & era split
# --------------------------------------------------------------------------- #
def width_sweep(raw_px, adj_px, shares, fracs=(0.2, 0.3, 0.4), lookback: int = 5) -> list[dict]:
    """Re-run the sort at several quantile widths; a real factor should be stable across them."""
    rows = []
    for f in fracs:
        s = summarize(raw_px, adj_px, shares, frac=f, lookback=lookback, placebo=False)
        rows.append({"frac": f, "n_years": s["n_years"], "ls_mean": s["ls_mean"],
                     "t": s["t"], "sharpe": s["sharpe"]})
    return rows


def era_contrast(raw_px, adj_px, shares, split_year: int = 2020, frac: float = 0.3,
                 lookback: int = 5) -> dict:
    """Long-short mean before vs since ``split_year`` (formation-year era split)."""
    ls = long_short_series(raw_px, adj_px, shares, frac=frac, lookback=lookback)
    early = ls[ls.index.year < split_year]["ls_ret"].to_numpy()
    late = ls[ls.index.year >= split_year]["ls_ret"].to_numpy()
    return {"n_early": len(early), "n_late": len(late),
            "early_mean": float(early.mean()) if len(early) else float("nan"),
            "late_mean": float(late.mean()) if len(late) else float("nan"),
            "t_early": one_sample_t(early), "t_late": one_sample_t(late),
            "t_diff": welch_t(late, early)}


# --------------------------------------------------------------------------- #
# Synthetic positive control (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_control(synthetic_panel, edges=(0.0, 0.10), n_names: int = 40,
                      n_years: int = 12, frac: float = 0.3, lookback: int = 5,
                      n_seeds: int = 25) -> list[dict]:
    """Faithful-engine / power control averaged over ``n_seeds`` synthetic worlds.

    For each planted ``edge``, build ``n_seeds`` independent panels (seed = 790+k), run the
    long-short sort, and average the one-sample t and LS mean across seeds. With ``edge=0`` the
    averaged t must stay well under 2 (no false positive); with a large planted edge it must
    clear 2. Never a single lucky draw.
    """
    rows = []
    for edge in edges:
        ts, means = [], []
        for k in range(n_seeds):
            raw, adj, sh = synthetic_panel(n_names=n_names, n_years=n_years, edge=edge,
                                           seed=790 + k, lookback=lookback)
            s = summarize(raw, adj, sh, frac=frac, lookback=lookback, placebo=False)
            ts.append(s["t"]); means.append(s["ls_mean"])
        rows.append({"edge": edge, "n_seeds": n_seeds,
                     "mean_t": float(np.nanmean(ts)),
                     "ls_mean": float(np.nanmean(means))})
    return rows
