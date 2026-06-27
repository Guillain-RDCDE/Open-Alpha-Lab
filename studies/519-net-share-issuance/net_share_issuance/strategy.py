"""Strategy + inference for Study 519 — Net-Share-Issuance.

The claim (Pontiff-Woodgate 2008; Daniel-Titman 2006 composite issuance): a firm's **net
change in shares outstanding** predicts its return — issuers (diluters) underperform,
repurchasers (share-count shrinkers) outperform. We test it as a clean **annual cross-sectional
sort**:

For each formation year-end *t* (signal public once the share count is on file), we compute
every name's **net issuance** = the split-adjusted share-count change over the trailing year,
rank the cross-section, go **long the low-issuance** quantile and **short the high-issuance**
quantile, and hold for the **next** year (one execution lag — we trade *after* the signal year
closes, never the same bar). We record the long-short annual return, test its mean against zero
with a one-sample t, stress it with a **label-shuffle placebo** (permute which name gets which
issuance value), charge **costs × turnover + short borrow**, and cut it by sub-period and by
quantile width.

The decisive object is the long-short mean against its standard error: on a ~40-name survivor
basket over a handful of years, a few-percent annual spread is easily inside the sampling noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Signal + per-name forward returns
# --------------------------------------------------------------------------- #
def net_issuance(shares: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year **net issuance** = fractional change in split-adjusted shares.

    ``issuance_{i,t} = shares_{i,t}/shares_{i,t-1} - 1``. Positive = the firm *diluted* (issued
    net new shares); negative = it *bought back* (shrank the count). One row per formation
    year-end *t*; the first year is NaN (no prior count).
    """
    return shares.pct_change()


def forward_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Year-ahead total return per name: ``prices_{t+1}/prices_t - 1`` (adjusted closes)."""
    return prices.pct_change().shift(-1)


# --------------------------------------------------------------------------- #
# Cross-sectional long-short
# --------------------------------------------------------------------------- #
def _quantile_legs(sig_row: pd.Series, frac: float = 0.3) -> tuple[list, list]:
    """Names in the bottom (low-issuance, LONG) and top (high-issuance, SHORT) ``frac`` of a
    one-year cross-section of issuance. Returns ``(long_names, short_names)``."""
    s = sig_row.dropna()
    if len(s) < 4:
        return [], []
    k = max(1, int(round(len(s) * frac)))
    order = s.sort_values()
    longs = list(order.index[:k])           # lowest issuance (buybacks) -> long
    shorts = list(order.index[-k:])          # highest issuance (dilution) -> short
    return longs, shorts


def long_short_series(prices: pd.DataFrame, shares: pd.DataFrame,
                      frac: float = 0.3) -> pd.DataFrame:
    """Build the annual long-short panel.

    For each formation year-end *t* with a defined issuance cross-section, sort names, form an
    equal-weight LONG (low issuance) and SHORT (high issuance) basket, and realise their
    **next-year** total returns (the one execution lag — the share count is public at *t*, we
    hold *t→t+1*). Returns a frame indexed by formation year-end with columns
    ``long_ret, short_ret, ls_ret`` (``ls_ret = long_ret - short_ret``) and the basket sizes.
    """
    sig = net_issuance(shares)
    fwd = forward_returns(prices)
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
        long_ret = float(lr.mean())
        short_ret = float(sr.mean())
        rows.append({
            "date": t, "long_ret": long_ret, "short_ret": short_ret,
            "ls_ret": long_ret - short_ret, "n_long": len(lr), "n_short": len(sr),
        })
    out = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(
        columns=["long_ret", "short_ret", "ls_ret", "n_long", "n_short"])
    return out


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray, mu0: float = 0.0) -> float:
    """One-sample t of ``mean(x) - mu0``. NaN if fewer than 2 finite points."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float((x.mean() - mu0) / se) if se > 0 else float("nan")


def ann_sharpe(x: np.ndarray) -> float:
    """Sharpe of an annual return series (already annual; mean/std, no scaling)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1))


def placebo_pvalue(prices: pd.DataFrame, shares: pd.DataFrame, frac: float = 0.3,
                   n_draws: int = 20_000, seed: int = 519) -> dict:
    """Label-shuffle placebo null: within each formation year, **permute** which name carries
    which issuance value (destroying the issuance→name link but preserving the marginal
    distribution of issuance and the realised cross-section of forward returns), rebuild the
    long-short, and ask how often a shuffled book's mean LS return **beats** the real sort.

    This is the strict null for a cross-sectional factor: it asks "could a random relabelling
    of the issuance signal across the same names, same years, same returns, have produced this
    spread by luck?" Returns the observed mean, the placebo mean, and
    ``p = P[placebo mean >= observed mean]``.
    """
    real = long_short_series(prices, shares, frac=frac)
    obs = float(real["ls_ret"].mean()) if len(real) else float("nan")
    if not np.isfinite(obs):
        return {"n_years": 0, "obs_mean": float("nan"),
                "placebo_mean": float("nan"), "p_value": float("nan")}

    sig = net_issuance(shares)
    fwd = forward_returns(prices)
    # Precompute, per formation year, the (issuance values, forward returns) over names present.
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
            perm = rng.permutation(n)               # shuffle the issuance labels
            order = np.argsort(iss[perm])           # rank shuffled issuance
            longs = ret[order[:k]]
            shorts = ret[order[-k:]]
            yr_ls.append(longs.mean() - shorts.mean())
        means[d] = np.mean(yr_ls)
    p = float((means >= obs).mean())
    return {"n_years": int(len(years)), "obs_mean": obs,
            "placebo_mean": float(means.mean()), "p_value": p}


def summarize(prices: pd.DataFrame, shares: pd.DataFrame, frac: float = 0.3,
              placebo: bool = True) -> dict:
    """Headline long-short stats: n years, long/short/LS annual means, win-rate of the LS book,
    annual Sharpe, one-sample t (vs 0), and (optionally) the label-shuffle placebo p."""
    ls = long_short_series(prices, shares, frac=frac)
    x = ls["ls_ret"].to_numpy() if len(ls) else np.array([])
    out = {
        "n_years": int(len(ls)),
        "long_mean": float(ls["long_ret"].mean()) if len(ls) else float("nan"),
        "short_mean": float(ls["short_ret"].mean()) if len(ls) else float("nan"),
        "ls_mean": float(ls["ls_ret"].mean()) if len(ls) else float("nan"),
        "ls_median": float(ls["ls_ret"].median()) if len(ls) else float("nan"),
        "win": float((x > 0).mean()) if len(x) else float("nan"),
        "sharpe": ann_sharpe(x),
        "t": one_sample_t(x, 0.0),
    }
    if placebo:
        out["p_placebo"] = placebo_pvalue(prices, shares, frac=frac)["p_value"]
    return out


# --------------------------------------------------------------------------- #
# Costs + borrow
# --------------------------------------------------------------------------- #
def net_of_costs(prices: pd.DataFrame, shares: pd.DataFrame, frac: float = 0.3,
                 cost_bps: float = 10.0, borrow_ann_bps: float = 50.0,
                 turnover: float = 1.0) -> dict:
    """Charge the long-short book one-way costs × turnover + a short-leg borrow.

    The sort is **re-struck every year**, and equal-weight quantile baskets turn over almost
    completely year to year, so we charge ``turnover`` (default 1.0 = full) × one-way
    ``cost_bps`` on **both** legs (long + short trade) per annual rebalance, plus a short-leg
    annual borrow ``borrow_ann_bps``. Returns gross and net annualised LS mean and the net t.
    """
    ls = long_short_series(prices, shares, frac=frac)
    if len(ls) == 0:
        return {"n_years": 0, "gross_mean": float("nan"), "net_mean": float("nan"),
                "net_t": float("nan"), "cost_bps": cost_bps, "borrow_bps": borrow_ann_bps}
    gross = ls["ls_ret"].to_numpy()
    annual_cost = 2.0 * cost_bps * 1e-4 * turnover       # both legs trade each rebalance
    annual_borrow = borrow_ann_bps * 1e-4                # short leg financing, annual
    net = gross - annual_cost - annual_borrow
    return {
        "n_years": int(len(ls)),
        "gross_mean": float(gross.mean()),
        "net_mean": float(net.mean()),
        "net_t": one_sample_t(net, 0.0),
        "cost_bps": cost_bps, "borrow_bps": borrow_ann_bps,
    }


# --------------------------------------------------------------------------- #
# Robustness — quantile width sensitivity
# --------------------------------------------------------------------------- #
def width_sweep(prices: pd.DataFrame, shares: pd.DataFrame,
                fracs=(0.2, 0.3, 0.4)) -> list[dict]:
    """Re-run the sort at several quantile widths; a real factor should be stable across them."""
    rows = []
    for f in fracs:
        s = summarize(prices, shares, frac=f, placebo=False)
        rows.append({"frac": f, "n_years": s["n_years"], "ls_mean": s["ls_mean"],
                     "t": s["t"], "sharpe": s["sharpe"]})
    return rows


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_control(synthetic_panel, edges=(0.0, 0.10), n_names: int = 40,
                      n_years: int = 9, frac: float = 0.3, n_seeds: int = 25) -> list[dict]:
    """Faithful-engine / power control averaged over ``n_seeds`` synthetic worlds.

    For each planted ``edge``, build ``n_seeds`` independent synthetic panels (seed = 519+k),
    run the long-short sort, and average the one-sample t and LS mean across seeds (the
    house rule: any synthetic-dependent claim is averaged over many seeds, never a single lucky
    draw). With ``edge=0`` the averaged t must stay well under 2 (no false positive); with a
    large planted edge it must clear 2.
    """
    rows = []
    for edge in edges:
        ts, means = [], []
        for k in range(n_seeds):
            px, sh = synthetic_panel(n_names=n_names, n_years=n_years, edge=edge, seed=519 + k)
            s = summarize(px, sh, frac=frac, placebo=False)
            ts.append(s["t"])
            means.append(s["ls_mean"])
        ts = np.asarray(ts, dtype=float)
        means = np.asarray(means, dtype=float)
        rows.append({"edge": edge, "n_seeds": n_seeds,
                     "mean_t": float(np.nanmean(ts)),
                     "ls_mean": float(np.nanmean(means))})
    return rows
