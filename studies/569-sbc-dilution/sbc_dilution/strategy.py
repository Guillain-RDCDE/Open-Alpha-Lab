"""Strategy + inference for Study 569 — SBC-Dilution.

The claim (an accounting hidden-cost anomaly, cousin of net-share-issuance): stock-based
compensation is a real economic cost that dilutes shareholders while barely touching GAAP
earnings. Firms that lean hardest on SBC dilute the most; if the market under-prices that hidden
cost, the heaviest SBC / fastest-diluting names should earn *lower* forward returns.

We test it as a clean **annual cross-sectional sort**:

For each formation year-end *t* (SBC intensity and the share count are public once on file), we
build a **dilution score** — ``z(SBC/revenue) + z(share-count growth)``, each z-scored across the
basket — rank the cross-section, go **long the low-dilution** quantile and **short the
high-dilution** quantile, and hold the **next** year (*t → t+1*, one execution lag). We record
the long-short annual return, test its mean against zero with a one-sample *t*, stress it with a
**label-shuffle placebo** (permute which name carries which dilution score), charge **costs ×
turnover + short borrow**, sweep the quantile width, and confirm the engine on a deterministic
synthetic positive control.

The decisive object is the long-short mean against its standard error: on a ~40-name survivor
basket over a handful of years, a few-percent annual spread is easily inside the sampling noise —
so the honest test is *whether the t clears 2 at all*, not just its sign.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Signal — the dilution score
# --------------------------------------------------------------------------- #
def share_growth(shares: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year **share-count growth** = fractional change in split-adjusted shares.

    Positive = the firm *diluted* (net new shares, incl. SBC vesting); negative = it bought back.
    One row per formation year-end *t*; the first year is NaN (no prior count).
    """
    return shares.pct_change()


def _z_row(row: pd.Series) -> pd.Series:
    s = row.dropna()
    sd = s.std(ddof=1)
    if not sd or sd <= 0 or len(s) < 2:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / sd


def dilution_score(shares: pd.DataFrame, sbc: pd.DataFrame) -> pd.DataFrame:
    """Composite **dilution score** per formation year-end: ``z(SBC/rev) + z(share growth)``.

    Both legs are z-scored *across the basket within each year* so the score is a pure
    cross-sectional tilt (higher = more dilutive: heavy SBC and/or fast share-count growth).
    Returns a wide frame indexed by formation year-end; rows with too few names are all-zero.
    """
    grow = share_growth(shares)
    rows = {}
    for t in grow.index:
        g = grow.loc[t]
        b = sbc.loc[t] if t in sbc.index else pd.Series(dtype=float)
        zg = _z_row(g)
        zb = _z_row(b)
        combined = zg.add(zb, fill_value=0.0)
        # keep only names with a defined growth (the binding leg); SBC contributes where present
        combined = combined.reindex(zg.index)
        rows[t] = combined
    return pd.DataFrame(rows).T.reindex(grow.index)


def forward_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Year-ahead total return per name: ``prices_{t+1}/prices_t - 1`` (adjusted closes)."""
    return prices.pct_change().shift(-1)


# --------------------------------------------------------------------------- #
# Cross-sectional long-short
# --------------------------------------------------------------------------- #
def _quantile_legs(sig_row: pd.Series, frac: float = 0.3) -> tuple[list, list]:
    """Names in the bottom (low-dilution, LONG) and top (high-dilution, SHORT) ``frac`` of a
    one-year cross-section of the dilution score. Returns ``(long_names, short_names)``."""
    s = sig_row.dropna()
    if len(s) < 4:
        return [], []
    k = max(1, int(round(len(s) * frac)))
    order = s.sort_values()
    longs = list(order.index[:k])            # lowest dilution (buybacks / lean SBC) -> long
    shorts = list(order.index[-k:])          # highest dilution (heavy SBC) -> short
    return longs, shorts


def long_short_series(prices: pd.DataFrame, shares: pd.DataFrame, sbc: pd.DataFrame,
                      frac: float = 0.3) -> pd.DataFrame:
    """Build the annual long-short panel.

    For each formation year-end *t* with a defined dilution cross-section, sort names, form an
    equal-weight LONG (low dilution) and SHORT (high dilution) basket, and realise their
    **next-year** total returns (the one execution lag — the score is public at *t*, we hold
    *t→t+1*). Returns a frame indexed by formation year-end with columns
    ``long_ret, short_ret, ls_ret`` and the basket sizes.
    """
    sig = dilution_score(shares, sbc)
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


def placebo_pvalue(prices: pd.DataFrame, shares: pd.DataFrame, sbc: pd.DataFrame,
                   frac: float = 0.3, n_draws: int = 20_000, seed: int = 569) -> dict:
    """Label-shuffle placebo null: within each formation year, **permute** which name carries
    which dilution score (destroying the score→name link but preserving the marginal
    distribution of the score and the realised cross-section of forward returns), rebuild the
    long-short, and ask how often a shuffled book's mean LS return **beats** the real sort.

    The strict null for a cross-sectional factor. Returns the observed mean, the placebo mean,
    and ``p = P[placebo mean >= observed mean]``.
    """
    real = long_short_series(prices, shares, sbc, frac=frac)
    obs = float(real["ls_ret"].mean()) if len(real) else float("nan")
    if not np.isfinite(obs):
        return {"n_years": 0, "obs_mean": float("nan"),
                "placebo_mean": float("nan"), "p_value": float("nan")}

    sig = dilution_score(shares, sbc)
    fwd = forward_returns(prices)
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
        for sco, ret in years:
            n = len(sco)
            k = max(1, int(round(n * frac)))
            perm = rng.permutation(n)               # shuffle the dilution-score labels
            order = np.argsort(sco[perm])
            longs = ret[order[:k]]
            shorts = ret[order[-k:]]
            yr_ls.append(longs.mean() - shorts.mean())
        means[d] = np.mean(yr_ls)
    p = float((means >= obs).mean())
    return {"n_years": int(len(years)), "obs_mean": obs,
            "placebo_mean": float(means.mean()), "p_value": p}


def summarize(prices: pd.DataFrame, shares: pd.DataFrame, sbc: pd.DataFrame,
              frac: float = 0.3, placebo: bool = True) -> dict:
    """Headline long-short stats: n years, long/short/LS annual means, win-rate of the LS book,
    annual Sharpe, one-sample t (vs 0), and (optionally) the label-shuffle placebo p."""
    ls = long_short_series(prices, shares, sbc, frac=frac)
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
        out["p_placebo"] = placebo_pvalue(prices, shares, sbc, frac=frac)["p_value"]
    return out


# --------------------------------------------------------------------------- #
# Costs + borrow
# --------------------------------------------------------------------------- #
def net_of_costs(prices: pd.DataFrame, shares: pd.DataFrame, sbc: pd.DataFrame,
                 frac: float = 0.3, cost_bps: float = 10.0, borrow_ann_bps: float = 50.0,
                 turnover: float = 1.0) -> dict:
    """Charge the long-short book one-way costs × turnover + a short-leg borrow.

    The sort is **re-struck every year** and equal-weight quantile baskets turn over almost
    completely year to year, so we charge ``turnover`` (default 1.0 = full) × one-way
    ``cost_bps`` on **both** legs per annual rebalance, plus a short-leg annual borrow. Returns
    gross and net annualised LS mean and the net t.
    """
    ls = long_short_series(prices, shares, sbc, frac=frac)
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
# Robustness — quantile width + SBC-only vs dilution-only legs
# --------------------------------------------------------------------------- #
def width_sweep(prices: pd.DataFrame, shares: pd.DataFrame, sbc: pd.DataFrame,
                fracs=(0.2, 0.3, 0.4)) -> list[dict]:
    """Re-run the sort at several quantile widths; a real factor should be stable across them."""
    rows = []
    for f in fracs:
        s = summarize(prices, shares, sbc, frac=f, placebo=False)
        rows.append({"frac": f, "n_years": s["n_years"], "ls_mean": s["ls_mean"],
                     "t": s["t"], "sharpe": s["sharpe"]})
    return rows


def leg_sweep(prices: pd.DataFrame, shares: pd.DataFrame, sbc: pd.DataFrame,
              frac: float = 0.3) -> list[dict]:
    """Compare the composite score against each leg alone (SBC-only, dilution-only).

    Sorts on: the composite ``z(SBC)+z(growth)``; just SBC intensity; just share growth. A real
    hidden-cost anomaly should show a consistent sign across all three; noise won't.
    """
    grow = share_growth(shares)
    fwd = forward_returns(prices)
    variants = {
        "composite": dilution_score(shares, sbc),
        "sbc_only": sbc.reindex(grow.index),
        "dilution_only": grow,
    }
    rows = []
    for name, sig in variants.items():
        xs = []
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
            xs.append(float(lr.mean()) - float(sr.mean()))
        xs = np.asarray(xs, dtype=float)
        rows.append({"signal": name, "n_years": int(len(xs)),
                     "ls_mean": float(np.nanmean(xs)) if len(xs) else float("nan"),
                     "t": one_sample_t(xs, 0.0)})
    return rows


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_control(synthetic_panel, edges=(0.0, 0.10), n_names: int = 40,
                      n_years: int = 6, frac: float = 0.3, n_seeds: int = 25) -> list[dict]:
    """Faithful-engine / power control averaged over ``n_seeds`` synthetic worlds.

    For each planted ``edge``, build ``n_seeds`` independent panels (seed = 569+k), run the sort,
    and average the one-sample t and LS mean across seeds (the house rule: any synthetic-dependent
    claim is averaged over many seeds, never a single lucky draw). With ``edge=0`` the averaged t
    must stay well under 2 (no false positive); with a large planted edge it must clear 2.
    """
    rows = []
    for edge in edges:
        ts, means = [], []
        for k in range(n_seeds):
            px, sh, sbc = synthetic_panel(n_names=n_names, n_years=n_years, edge=edge,
                                          seed=569 + k)
            s = summarize(px, sh, sbc, frac=frac, placebo=False)
            ts.append(s["t"])
            means.append(s["ls_mean"])
        ts = np.asarray(ts, dtype=float)
        means = np.asarray(means, dtype=float)
        rows.append({"edge": edge, "n_seeds": n_seeds,
                     "mean_t": float(np.nanmean(ts)),
                     "ls_mean": float(np.nanmean(means))})
    return rows
