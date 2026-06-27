"""The engine and its honest controls — Study 540 (Distress-Risk-Anomaly).

The claim, at full strength (Campbell, Hilscher & Szilagyi 2008, *In Search of Distress Risk*,
JF): a firm's *failure probability* — a logit of leverage, profitability, equity volatility,
size, cash and the price level — sorts the cross-section, and the most-distressed quintile/decile
goes on to earn the **lowest** subsequent returns, with a large *negative* alpha. That is the
**distress puzzle**: bearing distress risk is *not* compensated; if anything it is punished.

This module measures that, honestly, on what a retail stack can reach:

1. **The distress score.** A CHS-style standardised composite of three observable legs — high
   leverage, low profitability (ROA), high equity volatility — each z-scored across the basket and
   summed so that *higher = more distressed*. (The full CHS logit adds size/cash/price/excess
   return; we keep the three legs a no-key stack can build cleanly and name the simplification.)

2. **The decile/tercile sort.** Rank the cross-section by distress, form a *safe* bucket (lowest
   scores) and a *distressed* bucket (highest scores), and compare their forward returns.

3. **The long-short.** The puzzle's tradable expression is **long safe, short distressed** (the
   *opposite* of a risk-premium trade). We report its spread, a one-sample *t* on the
   cross-section of bucket-return differences, a **label-shuffle placebo** null, and costs +
   short borrow.

4. **The firm-level relation.** An OLS of forward return on the distress score across all names —
   the sign of the slope *is* the puzzle (negative = distressed firms underperform).

5. **A synthetic positive control.** A deterministic world where the anomaly is planted
   (``distress_alpha < 0``); the engine must recover a negative slope / positive long-short and
   stay flat at the null.

Execution convention: distress is scored from *trailing* data only (the realised-vol window
ending at the split close and the last reported statements), so the signal is fully public at the
split close; the position is then entered at that same close and held over the *forward* window.
No future data enters the signal (no look-ahead). The entry is a single documented convention —
a same-close fill on the bar the signal becomes public; because the hold is ~2 years (504 bars),
shifting entry one bar later moves the headline spread by ~1pt and changes no stamp (see
METHODOLOGY). The robustness sweep and synthetic control use the identical convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Building the panel from real prices + a fundamentals snapshot
# ---------------------------------------------------------------------------
def equity_vol(prices: pd.DataFrame, window_end: str, lookback: int = 252) -> pd.Series:
    """Trailing annualised realised volatility per name, ending at ``window_end``.

    Uses only data up to and including ``window_end`` (no look-ahead).
    """
    px = prices.loc[:window_end]
    ret = px.pct_change().tail(lookback)
    return ret.std(ddof=1) * np.sqrt(TRADING_DAYS)


def forward_return(prices: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Simple total return per name over the forward window (start, end]."""
    px = prices.loc[start:end]
    if len(px) < 2:
        return pd.Series(dtype=float)
    return px.iloc[-1] / px.iloc[0] - 1.0


def build_panel(
    prices: pd.DataFrame,
    fundamentals: dict,
    split_date: str,
    end_date: str | None = None,
    vol_lookback: int = 252,
) -> pd.DataFrame:
    """Assemble the cross-sectional panel the sort consumes, from real inputs.

    - *Leverage* and *ROA* come from the fundamentals snapshot (the accounting legs).
    - *Equity vol* is trailing realised vol ending at ``split_date`` (the price leg).
    - *forward_ret* is the return from ``split_date`` to ``end_date`` (the holding period).

    Everything left of ``split_date`` builds the score; everything right of it is the test
    window — the single execution lag, made explicit.
    """
    names = [t for t in fundamentals.keys() if t in prices.columns]
    if not names:
        return pd.DataFrame()
    end_date = end_date or str(prices.index[-1].date())
    ev = equity_vol(prices[names], split_date, vol_lookback)
    fr = forward_return(prices[names], split_date, end_date)
    rows = {}
    for t in names:
        f = fundamentals[t]
        if t not in ev.index or t not in fr.index:
            continue
        rows[t] = {
            "leverage": f["leverage"],
            "roa": f["roa"],
            "equity_vol": float(ev[t]),
            "forward_ret": float(fr[t]),
        }
    panel = pd.DataFrame.from_dict(rows, orient="index")
    panel.index.name = "ticker"
    return panel.dropna()


# ---------------------------------------------------------------------------
# The CHS-style distress score
# ---------------------------------------------------------------------------
def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=1)
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0


def distress_score(panel: pd.DataFrame) -> pd.Series:
    """A standardised CHS-style distress composite; higher = more distressed.

    distress = z(leverage) - z(roa) + z(equity_vol)

    High leverage, *low* profitability (so we subtract z(roa)) and high equity volatility all
    raise the failure probability — the three observable legs of the CHS measure.
    """
    if panel.empty:
        return pd.Series(dtype=float)
    return (_z(panel["leverage"]) - _z(panel["roa"]) + _z(panel["equity_vol"])).rename("distress")


# ---------------------------------------------------------------------------
# The sort and the long-short
# ---------------------------------------------------------------------------
def bucket_returns(panel: pd.DataFrame, frac: float = 0.3) -> dict:
    """Sort by distress, return safe / distressed bucket forward-return means and the spread.

    ``frac`` is the tail fraction in each bucket (0.3 ≈ terciles, appropriate for ~40 names; the
    paper uses deciles on thousands of stocks). The puzzle predicts safe > distressed.

    Returns a dict with bucket means, the spread (safe − distressed), bucket sizes, and the
    per-name forward returns of each bucket (for the one-sample t).
    """
    if panel.empty:
        return {}
    d = distress_score(panel)
    df = panel.assign(distress=d).sort_values("distress")
    k = max(1, int(round(len(df) * frac)))
    safe = df.head(k)            # lowest distress
    distressed = df.tail(k)      # highest distress
    safe_ret = safe["forward_ret"]
    dist_ret = distressed["forward_ret"]
    return {
        "n": int(len(df)),
        "k": int(k),
        "safe_mean": float(safe_ret.mean()),
        "distressed_mean": float(dist_ret.mean()),
        "spread": float(safe_ret.mean() - dist_ret.mean()),
        "safe_ret": safe_ret,
        "distressed_ret": dist_ret,
    }


def long_short_tstat(buckets: dict) -> dict:
    """One-sample (Welch two-sample) t on safe − distressed forward returns.

    The puzzle's tradable spread is a *cross-sectional* difference in means, so the honest test is
    a two-sample t between the safe and distressed buckets' forward returns. Returns the spread,
    the t-stat and n.
    """
    if not buckets:
        return {"spread": float("nan"), "t": float("nan"), "n": 0}
    a = buckets["safe_ret"].to_numpy(dtype=float)
    b = buckets["distressed_ret"].to_numpy(dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"spread": float("nan"), "t": float("nan"), "n": na + nb}
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    t = (a.mean() - b.mean()) / se if se > 0 else float("nan")
    return {"spread": float(a.mean() - b.mean()), "t": float(t), "n": na + nb}


# ---------------------------------------------------------------------------
# The firm-level relation — the sign IS the puzzle
# ---------------------------------------------------------------------------
def cross_section_regression(panel: pd.DataFrame) -> dict:
    """OLS of forward return on the distress score across all names.

    Slope < 0 = the distress puzzle (more-distressed firms earn less). Returns the slope, its
    t-stat, and the Pearson correlation between distress and forward return.
    """
    if panel.empty or len(panel) < 4:
        return {"slope": float("nan"), "slope_t": float("nan"), "corr": float("nan"), "n": 0}
    d = distress_score(panel).to_numpy(dtype=float)
    y = panel["forward_ret"].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(d), d])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, kk = X.shape
    dof = max(n - kk, 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se_slope = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    corr = float(np.corrcoef(d, y)[0, 1]) if d.std() > 0 and y.std() > 0 else float("nan")
    return {
        "slope": float(coef[1]),
        "slope_t": float(coef[1] / se_slope) if se_slope > 0 else float("nan"),
        "corr": corr,
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(panel: pd.DataFrame, n_perm: int = 2000, seed: int = 540) -> float:
    """Label-shuffle placebo p-value for the long-short spread.

    Shuffle the distress *labels* against forward returns ``n_perm`` times; the p-value is the
    fraction of shuffles whose |spread| ≥ the observed |spread|. A real cross-sectional signal
    sits in the tail; noise does not.
    """
    if panel.empty or len(panel) < 6:
        return float("nan")
    obs = abs(bucket_returns(panel)["spread"])
    rng = np.random.default_rng(seed)
    d = distress_score(panel).to_numpy(dtype=float)
    y = panel["forward_ret"].to_numpy(dtype=float)
    n = len(y)
    k = max(1, int(round(n * 0.3)))
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        order = np.argsort(d[perm])           # sort returns by shuffled distress
        ys = y[order]
        spread = ys[:k].mean() - ys[-k:].mean()
        if abs(spread) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Costs + borrow
# ---------------------------------------------------------------------------
def net_spread(buckets: dict, cost_bps: float = 5.0, borrow_ann_bps: float = 100.0,
               holding_years: float = 1.0) -> float:
    """Long-safe / short-distressed spread net of costs and short borrow.

    The book is built once and held ``holding_years``: charge a round-trip one-way cost on each
    leg (entry + exit, both legs) plus a short borrow on the distressed leg (annual, pro-rated).
    Distressed names are exactly the ones that are *expensive and hard to borrow*, so the borrow
    is deliberately punitive.
    """
    if not buckets:
        return float("nan")
    gross = buckets["spread"]
    # entry + exit, two legs -> 4 one-way crossings of cost_bps
    cost = 4.0 * cost_bps * 1e-4
    borrow = borrow_ann_bps * 1e-4 * holding_years
    return float(gross - cost - borrow)


# ---------------------------------------------------------------------------
# Robustness — seed-robust synthetic Welch (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, distress_alpha: float, n_seeds: int = 25,
                     base_seed: int = 540) -> float:
    """Average the cross-section regression slope t over ``n_seeds`` synthetic worlds.

    The house rule: any synthetic-dependent claim averages a Welch-style stat over >= 20 seeds so
    no single lucky RNG seed can manufacture significance. Returns the mean slope-t across seeds
    for a planted ``distress_alpha``.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_panel(distress_alpha=distress_alpha, seed=s)
        ts.append(cross_section_regression(panel)["slope_t"])
    return float(np.nanmean(ts))
