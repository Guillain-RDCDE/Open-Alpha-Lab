"""The engine and its honest controls — Study 565 (Filing-Readability).

The claim, at full strength (Loughran & McDonald 2014, *Measuring Readability in Financial
Disclosures*, JF): a 10-K's **readability** predicts the cross-section of returns. Less-readable
filings — higher Gunning **fog**, longer documents, bigger **file size** (LM's preferred robust
proxy) — precede **lower** future returns and **higher** post-filing volatility. The mechanism is
obfuscation: managers bury bad news in murk, and the market underreacts.

This module measures that, honestly, on a synthetic panel (no free real tape exists — see data.py):

1. **The readability score.** A standardised composite of the three observable legs — fog, length,
   file size — each z-scored across the cross-section and summed so *higher = less readable / more
   obfuscated*. (LM prefer file size alone; we also expose each leg on its own.)

2. **The information coefficient (IC).** The Spearman rank correlation between the readability
   score and forward return. The anomaly predicts a **negative** IC (murkier filings, lower
   returns). We report the IC, its t-stat (Fisher-z), and the OLS firm-level slope.

3. **The long-short.** The tradable expression is **long readable, short murky**: sort into terciles,
   long the readable tail, short the murky tail. We report the spread, a two-sample (Welch) t, a
   **label-shuffle placebo** null, and costs + a short borrow (the murky short leg pays borrow).

4. **The risk leg.** Post-filing volatility regressed on readability — the anomaly predicts a
   *positive* slope (murkier filings, higher risk).

5. **A synthetic positive control.** A deterministic world where the anomaly is planted
   (``obf_alpha < 0``); the engine must recover a negative IC / positive long-short and stay flat at
   the null, averaged over >= 20 seeds (house rule).

Execution convention: readability is scored from the *filing* (fully public the moment the 10-K is
filed), and the forward return is measured strictly *after* the filing date — a documented
execution lag (enter no earlier than the next close after the filing is public). No future data
enters the score (no look-ahead). The synthetic panel bakes this in: ``forward_ret`` is the
post-filing return by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# The readability (obfuscation) score
# ---------------------------------------------------------------------------
def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=1)
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0


def readability_score(panel: pd.DataFrame) -> pd.Series:
    """A standardised readability composite; higher = LESS readable (more obfuscated).

    obfuscation = z(fog) + z(length_kw) + z(filesize_mb)

    High fog, long documents and big files all make a filing harder to process — the three
    observable legs LM discuss. (LM prefer file size alone as the robust proxy; ``leg_score``
    exposes each leg individually.)
    """
    if panel.empty:
        return pd.Series(dtype=float)
    return (_z(panel["fog"]) + _z(panel["length_kw"]) + _z(panel["filesize_mb"])).rename("obf")


def leg_score(panel: pd.DataFrame, leg: str) -> pd.Series:
    """A single-leg readability score (z of one column), for the per-leg IC breakdown."""
    if panel.empty or leg not in panel.columns:
        return pd.Series(dtype=float)
    return _z(panel[leg]).rename(leg)


# ---------------------------------------------------------------------------
# The information coefficient — the sign IS the anomaly
# ---------------------------------------------------------------------------
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def information_coefficient(panel: pd.DataFrame, score: pd.Series | None = None) -> dict:
    """Spearman IC between the readability score and forward return, with a Fisher-z t-stat.

    The anomaly predicts IC < 0 (less-readable filings earn lower returns). Returns the IC, its
    Fisher-z t-stat, and n. The t is ``atanh(IC) * sqrt(n - 3)`` — the standard rank-corr test.
    """
    if panel.empty or len(panel) < 6:
        return {"ic": float("nan"), "ic_t": float("nan"), "n": int(len(panel))}
    s = readability_score(panel) if score is None else score
    ic = _spearman(s.to_numpy(dtype=float), panel["forward_ret"].to_numpy(dtype=float))
    n = len(panel)
    if not np.isfinite(ic) or abs(ic) >= 1.0:
        return {"ic": ic, "ic_t": float("nan"), "n": n}
    ic_t = float(np.arctanh(ic) * np.sqrt(n - 3))
    return {"ic": float(ic), "ic_t": ic_t, "n": int(n)}


def cross_section_regression(panel: pd.DataFrame) -> dict:
    """OLS of forward return on the readability score across all names.

    Slope < 0 = the anomaly (less-readable filings earn less). Returns the slope, its t-stat, and
    the Pearson correlation.
    """
    if panel.empty or len(panel) < 4:
        return {"slope": float("nan"), "slope_t": float("nan"), "corr": float("nan"), "n": 0}
    d = readability_score(panel).to_numpy(dtype=float)
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
# The sort and the long-short
# ---------------------------------------------------------------------------
def bucket_returns(panel: pd.DataFrame, frac: float = 0.3) -> dict:
    """Sort by readability, return readable / murky bucket forward-return means and the spread.

    ``frac`` is the tail fraction in each bucket (0.3 ≈ terciles). The anomaly predicts
    readable > murky (a positive long-readable/short-murky spread). Returns bucket means, the
    spread, sizes, and the per-name forward returns of each bucket (for the two-sample t).
    """
    if panel.empty:
        return {}
    d = readability_score(panel)
    df = panel.assign(obf=d).sort_values("obf")
    k = max(1, int(round(len(df) * frac)))
    readable = df.head(k)         # lowest obfuscation
    murky = df.tail(k)            # highest obfuscation
    r_ret = readable["forward_ret"]
    m_ret = murky["forward_ret"]
    return {
        "n": int(len(df)),
        "k": int(k),
        "readable_mean": float(r_ret.mean()),
        "murky_mean": float(m_ret.mean()),
        "spread": float(r_ret.mean() - m_ret.mean()),
        "readable_ret": r_ret,
        "murky_ret": m_ret,
    }


def long_short_tstat(buckets: dict) -> dict:
    """Welch two-sample t on readable − murky forward returns.

    The anomaly's tradable spread is a cross-sectional difference in means, so the honest test is a
    two-sample t between the readable and murky buckets' forward returns. Returns the spread, the
    t-stat and n.
    """
    if not buckets:
        return {"spread": float("nan"), "t": float("nan"), "n": 0}
    a = buckets["readable_ret"].to_numpy(dtype=float)
    b = buckets["murky_ret"].to_numpy(dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"spread": float("nan"), "t": float("nan"), "n": na + nb}
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    t = (a.mean() - b.mean()) / se if se > 0 else float("nan")
    return {"spread": float(a.mean() - b.mean()), "t": float(t), "n": na + nb}


# ---------------------------------------------------------------------------
# The risk leg — post-filing volatility rises with obfuscation
# ---------------------------------------------------------------------------
def risk_regression(panel: pd.DataFrame) -> dict:
    """OLS of post-filing volatility on the readability score.

    The LM "higher risk" leg predicts slope > 0 (murkier filings, higher post-filing vol). Returns
    the slope, its t-stat and n. Silently degrades to NaN if ``post_vol`` is absent.
    """
    if panel.empty or "post_vol" not in panel.columns or len(panel) < 4:
        return {"slope": float("nan"), "slope_t": float("nan"), "n": 0}
    d = readability_score(panel).to_numpy(dtype=float)
    y = panel["post_vol"].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(d), d])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, kk = X.shape
    dof = max(n - kk, 1)
    sigma2 = float(resid @ resid) / dof
    se_slope = float(np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1]))
    return {
        "slope": float(coef[1]),
        "slope_t": float(coef[1] / se_slope) if se_slope > 0 else float("nan"),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(panel: pd.DataFrame, n_perm: int = 2000, seed: int = 565) -> float:
    """Label-shuffle placebo p-value for the long-short spread.

    Shuffle the readability *labels* against forward returns ``n_perm`` times; the p-value is the
    fraction of shuffles whose |spread| ≥ the observed |spread|. A real cross-sectional signal sits
    in the tail; noise does not.
    """
    if panel.empty or len(panel) < 6:
        return float("nan")
    obs = abs(bucket_returns(panel)["spread"])
    rng = np.random.default_rng(seed)
    d = readability_score(panel).to_numpy(dtype=float)
    y = panel["forward_ret"].to_numpy(dtype=float)
    n = len(y)
    k = max(1, int(round(n * 0.3)))
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        order = np.argsort(d[perm])           # sort returns by shuffled readability
        ys = y[order]
        spread = ys[:k].mean() - ys[-k:].mean()
        if abs(spread) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Costs + borrow
# ---------------------------------------------------------------------------
def net_spread(buckets: dict, cost_bps: float = 5.0, borrow_ann_bps: float = 150.0,
               holding_years: float = 1.0) -> float:
    """Long-readable / short-murky spread net of costs and short borrow.

    The book is built once and held ``holding_years``: charge a round-trip one-way cost on each leg
    (entry + exit, both legs) plus a short borrow on the murky leg (annual, pro-rated). The murky,
    hard-to-analyse names skew small/illiquid, so the borrow is deliberately punitive.
    """
    if not buckets:
        return float("nan")
    gross = buckets["spread"]
    cost = 4.0 * cost_bps * 1e-4          # entry + exit, two legs -> 4 one-way crossings
    borrow = borrow_ann_bps * 1e-4 * holding_years
    return float(gross - cost - borrow)


# ---------------------------------------------------------------------------
# Robustness — per-leg IC sweep (fog vs length vs file size vs composite)
# ---------------------------------------------------------------------------
def leg_ic_sweep(panel: pd.DataFrame) -> dict:
    """IC of each individual readability leg + the composite against forward return.

    LM's point is that fog is noisy while file size is a cleaner proxy — this sweep reads whether
    the anomaly's sign is stable across legs. Returns ``{leg: (ic, ic_t)}``.
    """
    out: dict[str, tuple[float, float]] = {}
    if panel.empty:
        return out
    for leg in ("fog", "length_kw", "filesize_mb"):
        if leg in panel.columns:
            r = information_coefficient(panel, score=leg_score(panel, leg))
            out[leg] = (r["ic"], r["ic_t"])
    r = information_coefficient(panel)
    out["composite"] = (r["ic"], r["ic_t"])
    return out


# ---------------------------------------------------------------------------
# Robustness — seed-robust synthetic IC (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_ic(data_mod, obf_alpha: float, n_seeds: int = 25,
                      base_seed: int = 565, n_stocks: int = 400) -> dict:
    """Average the IC and its t over ``n_seeds`` synthetic worlds for a planted ``obf_alpha``.

    The house rule: any synthetic-dependent claim averages a stat over >= 20 seeds so no single
    lucky RNG seed can manufacture significance. Returns ``{"ic": mean IC, "ic_t": mean IC-t}``.
    """
    ics, tts = [], []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_panel(n_stocks=n_stocks, obf_alpha=obf_alpha, seed=s)
        r = information_coefficient(panel)
        ics.append(r["ic"])
        tts.append(r["ic_t"])
    return {"ic": float(np.nanmean(ics)), "ic_t": float(np.nanmean(tts))}
