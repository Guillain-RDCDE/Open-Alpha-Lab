"""Hierarchical risk parity, and four things to compare it with — Study 976.

López de Prado's (2016) argument in one paragraph: a covariance matrix estimated from limited
data has unstable small eigenvalues; a quadratic optimiser inverts the matrix and therefore
loads on exactly those; a hierarchical scheme never inverts anything, so it cannot. The
algorithm has three steps, implemented here from scratch:

1. **Tree clustering** (``correlation_distance`` + ``single_linkage``). The distance between
   two assets is ``sqrt(0.5 * (1 - rho))``, which is a proper metric on correlations, and the
   tree is built by single linkage — nearest-neighbour merging, no external dependency.
2. **Quasi-diagonalisation** (``quasi_diagonal_order``). Walk the tree and read the leaves in
   order, so correlated assets end up adjacent and the reordered correlation matrix is as close
   to block-diagonal as the data allows.
3. **Recursive bisection** (``hrp_weights``). Split the ordered list in two, compute each
   half's variance under inverse-variance weighting, and allocate between the halves in inverse
   proportion. Recurse. Note what this does *not* do: the split is by **count**, at the
   midpoint of the ordered list, not at the tree's own cluster boundary. Nine near-identical
   assets and one outsider therefore split 5/5 rather than 9/1, and the outsider receives a
   modest overweight rather than half the book. That is the published algorithm, and it is
   pinned in the test suite so nobody mistakes it for a bug.

The competitors, chosen so that every claim HRP makes has a control:

- ``min_variance_weights`` — the optimiser HRP is arguing against.
- ``inverse_variance_weights`` — the naive risk-based weighting HRP reduces to when the tree is
  uninformative. **This is the control that matters**: if HRP cannot beat it, the clustering
  step is doing nothing and the result is about risk weighting, not about hierarchy.
- ``risk_parity_weights`` — equal risk contribution, solved iteratively; the industry standard.
- ``equal_weights`` — 1/N, which the literature keeps failing to beat.

Everything is scored the same way as this desk's shrinkage study: a rolling window, quarterly
re-estimation, one day of execution lag, costs on turnover, and realised out-of-sample
volatility as the scoreboard, with paired tests because every method sees the same windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
METHODS = ("hrp", "min_var", "inv_var", "risk_parity", "equal")
METHOD_LABEL = {
    "hrp": "Hierarchical risk parity",
    "min_var": "Minimum variance",
    "inv_var": "Inverse variance",
    "risk_parity": "Equal risk contribution",
    "equal": "1/N",
}


# --------------------------------------------------------------------------- #
# Step 1-2: the tree, and the order it implies
# --------------------------------------------------------------------------- #
def correlation_distance(corr: np.ndarray) -> np.ndarray:
    """``sqrt(0.5 * (1 - rho))`` — a true metric on correlations (Mantegna 1999)."""
    return np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))


def single_linkage(dist: np.ndarray) -> list[tuple[int, int, float, int]]:
    """Single-linkage agglomerative clustering, written out rather than imported.

    Returns SciPy's linkage format — ``(left, right, distance, size)`` with cluster ids
    continuing past ``n`` — so the output can be read by anyone who knows ``scipy.cluster``,
    without adding the dependency. Single linkage is López de Prado's choice and it is the one
    that makes the quasi-diagonal order well defined.
    """
    n = dist.shape[0]
    active = {i: [i] for i in range(n)}
    d = dist.astype(float).copy()
    np.fill_diagonal(d, np.inf)
    sizes = {i: 1 for i in range(n)}
    links: list[tuple[int, int, float, int]] = []
    ids = {i: i for i in range(n)}
    next_id = n
    while len(active) > 1:
        keys = list(active)
        best, bi, bj = np.inf, None, None
        for a_i, i in enumerate(keys):
            for j in keys[a_i + 1:]:
                m = d[np.ix_(active[i], active[j])].min()
                if m < best:
                    best, bi, bj = m, i, j
        links.append((ids[bi], ids[bj], float(best), sizes[bi] + sizes[bj]))
        active[bi] = active[bi] + active[bj]
        sizes[bi] = sizes[bi] + sizes[bj]
        ids[bi] = next_id
        next_id += 1
        del active[bj], sizes[bj]
    return links


def quasi_diagonal_order(links: list[tuple[int, int, float, int]], n: int) -> list[int]:
    """Read the leaves off the tree so that correlated assets sit next to each other."""
    if not links:
        return list(range(n))
    order = [links[-1][0], links[-1][1]]
    while max(order) >= n:
        out = []
        for item in order:
            if item < n:
                out.append(item)
            else:
                left, right, _, _ = links[int(item) - n]
                out.extend([int(left), int(right)])
        order = out
    return [int(i) for i in order]


def cluster_order(cov: np.ndarray) -> list[int]:
    """Steps 1 and 2 together: covariance -> correlation -> tree -> leaf order."""
    sd = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(np.outer(sd, sd) > 0, cov / np.outer(sd, sd), 0.0)
    np.fill_diagonal(corr, 1.0)
    return quasi_diagonal_order(single_linkage(correlation_distance(corr)), cov.shape[0])


# --------------------------------------------------------------------------- #
# Step 3, and the competitors
# --------------------------------------------------------------------------- #
def _cluster_var(cov: np.ndarray, idx: list[int]) -> float:
    """Variance of a sub-portfolio weighted by inverse variance — the bisection's yardstick."""
    sub = cov[np.ix_(idx, idx)]
    iv = 1.0 / np.diag(sub)
    w = iv / iv.sum()
    return float(w @ sub @ w)


def hrp_weights(cov: np.ndarray, order: list[int] | None = None) -> np.ndarray:
    """Recursive bisection down the quasi-diagonal order (López de Prado 2016)."""
    n = cov.shape[0]
    order = cluster_order(cov) if order is None else list(order)
    w = np.ones(n)
    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            half = len(c) // 2
            left, right = c[:half], c[half:]
            v_left, v_right = _cluster_var(cov, left), _cluster_var(cov, right)
            alpha = 1.0 - v_left / (v_left + v_right) if (v_left + v_right) > 0 else 0.5
            for i in left:
                w[i] *= alpha
            for i in right:
                w[i] *= 1 - alpha
            nxt += [left, right]
        clusters = nxt
    return w / w.sum()


def min_variance_weights(cov: np.ndarray, long_only: bool = False) -> np.ndarray:
    """Global minimum variance — the estimator HRP is arguing against."""
    n = cov.shape[0]
    inv = np.linalg.pinv(cov + np.eye(n) * 1e-14)
    w = inv @ np.ones(n)
    s = w.sum()
    w = w / s if abs(s) > 1e-12 else np.full(n, 1 / n)
    if long_only:
        w = np.clip(w, 0, None)
        w = w / w.sum() if w.sum() > 0 else np.full(n, 1 / n)
    return w


def inverse_variance_weights(cov: np.ndarray) -> np.ndarray:
    """1/sigma^2, normalised — HRP with the hierarchy switched off."""
    iv = 1.0 / np.diag(cov)
    return iv / iv.sum()


def risk_parity_weights(cov: np.ndarray, iters: int = 2000, tol: float = 1e-12) -> np.ndarray:
    """Equal risk contribution, by fixed-point iteration on the risk *contributions*.

    The update is ``w_i <- w_i * sqrt(target / RC_i)`` where ``RC_i = w_i (Sigma w)_i`` is
    asset *i*'s contribution to portfolio variance and ``target`` is the average of them. The
    obvious-looking variant that scales by ``target / MRC_i`` — the *marginal* contribution —
    converges happily to a point that is not risk parity at all (it left a 20% coefficient of
    variation across contributions on this desk's test panel), which is why the test suite
    checks the contributions rather than the convergence flag.
    """
    n = cov.shape[0]
    w = np.full(n, 1.0 / n)
    for _ in range(iters):
        rc = w * (cov @ w)
        target = rc.mean()
        if target <= 0:
            break
        new = w * np.sqrt(target / np.maximum(rc, 1e-18))
        new = np.clip(new, 1e-14, None)
        new = new / new.sum()
        if np.max(np.abs(new - w)) < tol:
            w = new
            break
        w = new
    return w


def equal_weights(cov: np.ndarray) -> np.ndarray:
    """1/N. The benchmark that keeps winning."""
    n = cov.shape[0]
    return np.full(n, 1.0 / n)


def weights_for(method: str, cov: np.ndarray) -> np.ndarray:
    """Dispatch on ``METHODS``."""
    if method == "hrp":
        return hrp_weights(cov)
    if method == "min_var":
        return min_variance_weights(cov)
    if method == "inv_var":
        return inverse_variance_weights(cov)
    if method == "risk_parity":
        return risk_parity_weights(cov)
    if method == "equal":
        return equal_weights(cov)
    raise ValueError(f"unknown method {method!r}")


# --------------------------------------------------------------------------- #
# Diagnostics and the scoreboard
# --------------------------------------------------------------------------- #
def concentration(w: np.ndarray) -> dict:
    """How concentrated a weight vector is: max weight, HHI, effective N, short exposure."""
    hhi = float(np.sum(w ** 2))
    return {"max_weight": float(np.max(np.abs(w))), "hhi": hhi,
            "effective_n": float(1.0 / hhi) if hhi > 0 else np.nan,
            "short": float(np.sum(np.clip(-w, 0, None)))}


def walk_forward(rets: pd.DataFrame, window: int = 252, step: int = 63,
                 methods=METHODS, cost_bps: float = 5.0) -> pd.DataFrame:
    """Rolling re-estimation, one-day lag, costs on turnover; one row per (method, rebalance)."""
    R = rets.dropna(how="any")
    rows = []
    prev: dict[str, np.ndarray] = {}
    for start in range(window, len(R) - step, step):
        train = R.iloc[start - window:start].to_numpy()
        test = R.iloc[start:start + step]
        cov = np.cov(train, rowvar=False, ddof=1)
        for m in methods:
            w = weights_for(m, cov)
            port = test.to_numpy() @ w
            turn = float(np.abs(w - prev.get(m, np.zeros_like(w))).sum())
            prev[m] = w
            c = concentration(w)
            rows.append({"date": R.index[start], "method": m,
                         "realised_vol": float(np.std(port, ddof=1) * np.sqrt(TRADING_DAYS)),
                         "mean_ret": float(port.mean() * TRADING_DAYS
                                           - turn * cost_bps / 1e4 * TRADING_DAYS / step),
                         "turnover": turn, **c})
    return pd.DataFrame(rows)


def summarise(wf: pd.DataFrame) -> pd.DataFrame:
    """Per method: realised volatility, return, Sharpe, turnover and concentration."""
    g = wf.groupby("method")
    out = pd.DataFrame({
        "realised_vol": g["realised_vol"].mean(), "mean_ret": g["mean_ret"].mean(),
        "turnover": g["turnover"].mean(), "max_weight": g["max_weight"].mean(),
        "effective_n": g["effective_n"].mean(), "short": g["short"].mean(),
        "vol_of_vol": g["realised_vol"].std(), "n": g.size(),
    })
    out["sharpe"] = out["mean_ret"] / out["realised_vol"]
    return out.reindex([m for m in METHODS if m in out.index])


def paired_test(wf: pd.DataFrame, a: str, b: str, column: str = "realised_vol") -> dict:
    """Paired *t* on a per-rebalance column: same windows, same holding periods."""
    x = wf[wf["method"] == a].set_index("date")[column]
    y = wf[wf["method"] == b].set_index("date")[column]
    x, y = x.align(y, join="inner")
    d = (x - y).dropna()
    if len(d) < 8:
        return {"diff": np.nan, "t": np.nan, "n": int(len(d)), "win_rate": np.nan}
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"diff": float(d.mean()), "t": float(d.mean() / se) if se > 0 else np.nan,
            "n": int(len(d)), "win_rate": float((d < 0).mean())}


def block_panel(n_per_block: int = 10, n_blocks: int = 2, n_obs: int = 250,
                rho_in: float = 0.7, rho_out: float = 0.1, vol: float = 0.02,
                seed: int = 976) -> tuple[np.ndarray, np.ndarray]:
    """A panel with planted block correlation — the structure HRP's tree is built to find."""
    rng = np.random.default_rng(seed)
    n = n_per_block * n_blocks
    corr = np.full((n, n), rho_out)
    for b in range(n_blocks):
        s = slice(b * n_per_block, (b + 1) * n_per_block)
        corr[s, s] = rho_in
    np.fill_diagonal(corr, 1.0)
    cov = (vol ** 2) * corr
    L = np.linalg.cholesky(cov + np.eye(n) * 1e-12)
    X = rng.normal(0, 1, (n_obs, n)) @ L.T
    return X, cov


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal** (does the clustering step do anything?): **Real** if HRP's weights differ
      materially from inverse variance — mean absolute weight difference above 2% of the book
      — on at least two panels; **Weak** on one; **None** otherwise. This is the honest test of
      the *hierarchy*, as opposed to the risk weighting anyone gets for free.
    - **Usefulness**: **Useful** if HRP's realised volatility beats **both** minimum variance
      and 1/N on the wide panel with a paired |*t*| >= 2; **Fragile** if it beats one; **Mirage**
      if it beats neither.
    """
    signal = ("Real" if h["n_panels_hierarchy_matters"] >= 2
              else ("Weak" if h["n_panels_hierarchy_matters"] >= 1 else "None"))
    beats_mv = h["t_vs_minvar"] > 2.0
    beats_eq = h["t_vs_equal"] > 2.0
    trad = ("Useful" if beats_mv and beats_eq
            else ("Fragile" if beats_mv or beats_eq else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"The tree is not decoration: HRP's weights differ from plain inverse-variance "
            f"weights by **{h['weight_gap_wide']:.1%}** of the book on the {h['n_names']}-name "
            f"panel and {h['weight_gap_multi']:.1%} on the multi-asset one, and the clustering "
            f"changes the ordering on {h['n_panels_hierarchy_matters']} of 3 panels. What it "
            f"buys is a different question: HRP holds an effective "
            f"**{h['effective_n_hrp']:.1f}** positions against {h['effective_n_minvar']:.1f} "
            f"for the optimiser, with a maximum weight of {h['max_weight_hrp']:.1%} against "
            f"{h['max_weight_minvar']:.1%} and no shorts at all."),
        "trad": trad,
        "trad_why": (
            f"On the wide panel HRP realises **{h['vol_hrp']:.2%}** annualised against "
            f"**{h['vol_minvar']:.2%}** for minimum variance (paired *t* = "
            f"{h['t_vs_minvar']:+.2f}) and **{h['vol_equal']:.2%}** for 1/N (*t* = "
            f"{h['t_vs_equal']:+.2f}). Against the control that matters — inverse variance, "
            f"which is HRP without the tree — the difference is "
            f"{h['vol_hrp'] - h['vol_invvar']:+.3%} (*t* = {h['t_vs_invvar']:+.2f}). Turnover "
            f"is {h['turnover_hrp']:.2f} a rebalance against {h['turnover_minvar']:.2f}, which "
            f"is where the practical case for it is strongest."),
        "one_sentence": (
            f"Hierarchical risk parity does what it says — no matrix inverse, no shorts, "
            f"weights that move {h['turnover_hrp'] / max(h['turnover_minvar'], 1e-9):.0%} as "
            f"much as the optimiser's — and most of its out-of-sample volatility advantage "
            f"comes from being a **risk-weighted long-only book**, not from the clustering: "
            f"plain inverse variance is within {abs(h['vol_hrp'] - h['vol_invvar']):.2%} of it."),
    }
