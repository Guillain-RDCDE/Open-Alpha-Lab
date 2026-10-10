"""Myopic loss aversion, measured — Study 1025 (Don't Look).

The machinery has four parts, matching the four questions of the study.

**1 · How often do you see a loss?** ``loss_probability_curve`` compounds monthly (or daily)
returns into every overlapping window of ``h`` periods and counts the share that end below zero.
This is the purely mechanical half of the story: a positive drift is swamped by noise over a day
and dominates it over a decade, so the loss rate falls with horizon for any asset with a premium.

**2 · How does a loss-averse investor feel about it?** ``cpt_value`` is cumulative prospect
theory (Tversky & Kahneman 1992) applied to an empirical distribution of horizon returns: a
power value function ``x^α`` for gains and ``−λ(−x)^α`` for losses, with rank-dependent
probability weights ``w(p) = p^γ / (p^γ + (1−p)^γ)^{1/γ}`` (γ = 0.61 for gains, δ = 0.69 for
losses). Benartzi & Thaler (1995) used exactly these parameters with gains and losses measured
against a zero nominal return, and so does the headline here; ``weighting=False`` switches the
probability weights off for the robustness sweep. Because the value function is homogeneous of
degree α, the *sign* of a comparison between two assets does not depend on the unit returns are
measured in — only the sign matters for the break-even.

``pt_curve`` evaluates every asset at every horizon; ``break_even`` reads off the
"equilibrium" horizon — the shortest horizon beyond which stocks are preferred at *every* longer
horizon on the grid (log-linear interpolation at the last crossing). A curve that never turns
positive returns ``inf``: no evaluation period on the grid makes stocks attractive. A curve that
is positive from the first horizon is censored at the grid's first point.

Three estimators of the horizon distribution, because ninety years hold only nine independent
decades:

- **overlapping** — every window, equally weighted. The point estimate; inference never treats
  its windows as independent (see the bootstrap).
- **non-overlapping** — back-to-back windows from the first month, the textbook estimator that
  throws data away at long horizons.
- **i.i.d. convolution** — monthly returns drawn independently with replacement and compounded:
  the horizon distribution if returns had no serial dependence at all (no mean reversion).

``bootstrap_curves`` is the correction for overlap: a **circular block bootstrap** of the joint
monthly return vector (stocks, bonds and bills resampled together, so their correlation and
short-run dependence survive), with the whole curve and its break-even recomputed on every
resample. Confidence intervals and p-values are percentiles of that distribution.

**3 · Robustness** — ``param_sweep`` over λ and α (weights on and off); the callers run nominal vs
real and pre/post-1970 through the same functions.

**4 · The discipline question** — ``myopic_switcher``. Looking less changes what you *see*, not
what you *earn*: the only money in "don't look" is the cost of the behaviour it is meant to
prevent. The rule, fixed before the real run: at the end of every evaluation period (day, week,
month, quarter, year) the investor looks at the stock market's return over that period; after a
**loss** they hold **bills** for the next period, after a **gain** they hold **stocks**. One
execution lag: the sign known at the close of the period's last day sets the position from the
next return onward — a single shift, applied once. Switching stocks↔bills turns over 1× NAV
one-way and pays ``cost_bps`` on it. ``discipline_table`` compares it with buy-and-hold, gross
and net, excess-of-bills Sharpe against excess-of-bills Sharpe, with Newey-West and paired
block-bootstrap inference on the return gap.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from quantlab.analytics import mean_tstat_hac  # noqa: E402

# Tversky & Kahneman (1992) median estimates — Benartzi & Thaler's (1995) calibration.
ALPHA = 0.88
LAMBDA = 2.25
GAMMA_GAIN = 0.61
GAMMA_LOSS = 0.69

# Evaluation horizons in months (1 month … 10 years).
HORIZONS_M = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 24, 30, 36, 42, 48, 60, 72, 84,
              96, 108, 120)
# Daily-tape horizons in trading days (1 day … 1 year).
HORIZONS_D = (1, 2, 5, 10, 21, 63, 126, 252)
BLOCK_M = 24          # circular-block length, months
BLOCK_D = 21          # circular-block length, trading days
BTH_BREAK_EVEN = 12.0  # Benartzi-Thaler: indifference at roughly one year

FREQ_PERIODS = {"D": 252, "W": 52, "M": 12, "Q": 4, "Y": 1}
FREQ_LABEL = {"D": "daily", "W": "weekly", "M": "monthly", "Q": "quarterly", "Y": "yearly"}


# --------------------------------------------------------------------------- #
# Prospect theory
# --------------------------------------------------------------------------- #
def tk_weight(p, gamma: float) -> np.ndarray:
    """Tversky-Kahneman (1992) probability weighting ``p^γ / (p^γ + (1−p)^γ)^{1/γ}``."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    if gamma == 1.0:
        return p
    num = p ** gamma
    return num / (num + (1.0 - p) ** gamma) ** (1.0 / gamma)


def decision_weights(n: int, gamma_gain: float = GAMMA_GAIN, gamma_loss: float = GAMMA_LOSS,
                     weighting: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Rank-dependent decision weights for ``n`` equally likely outcomes sorted ascending.

    Returns ``(w_loss, w_gain)``, each of length ``n``: the weight position ``k`` gets *if* its
    outcome is a loss (cumulated from the worst outcome up) or a gain (decumulated from the best
    outcome down). With ``weighting=False`` both are ``1/n``.
    """
    k = np.arange(n, dtype=float)
    if not weighting:
        w = np.full(n, 1.0 / n)
        return w, w
    i = k + 1.0                      # rank from the bottom
    w_loss = tk_weight(i / n, gamma_loss) - tk_weight((i - 1.0) / n, gamma_loss)
    j = n - k                        # rank from the top
    w_gain = tk_weight(j / n, gamma_gain) - tk_weight((j - 1.0) / n, gamma_gain)
    return w_loss, w_gain


def value_fn(x, alpha: float = ALPHA, lam: float = LAMBDA) -> np.ndarray:
    """Piecewise power value: ``x^α`` on gains, ``−λ(−x)^α`` on losses (reference point 0)."""
    x = np.asarray(x, dtype=float)
    ax = np.abs(x) ** alpha
    return np.where(x >= 0.0, ax, -lam * ax)


def cpt_value(outcomes, alpha: float = ALPHA, lam: float = LAMBDA,
              gamma_gain: float = GAMMA_GAIN, gamma_loss: float = GAMMA_LOSS,
              weighting: bool = True, _weights=None) -> np.ndarray:
    """Cumulative-prospect-theory value of equally likely ``outcomes`` (last axis).

    Works on a 1-D sample or a 2-D (replications × outcomes) array; returns a scalar or one
    value per row. Outcomes are simple returns relative to the reference point (0).
    """
    x = np.sort(np.asarray(outcomes, dtype=float), axis=-1)
    if x.ndim == 1:
        x = x[np.isfinite(x)]
    n = x.shape[-1]
    if n == 0:
        return np.nan
    wl, wg = _weights if _weights is not None else decision_weights(
        n, gamma_gain, gamma_loss, weighting)
    pi = np.where(x < 0.0, wl, wg)
    return (pi * value_fn(x, alpha, lam)).sum(axis=-1)


# --------------------------------------------------------------------------- #
# Horizon returns
# --------------------------------------------------------------------------- #
def horizon_returns(r, h: int, overlapping: bool = True) -> np.ndarray:
    """Compound ``h``-period simple returns from per-period simple returns ``r``.

    ``overlapping=True`` gives every window (``n − h + 1`` of them); ``False`` gives the
    back-to-back windows starting at the first observation. Works along the last axis of a 1-D
    or 2-D array.
    """
    r = np.asarray(r, dtype=float)
    n = r.shape[-1]
    if h > n:
        return np.empty(r.shape[:-1] + (0,))
    if overlapping:
        L = np.cumsum(np.log1p(r), axis=-1)
        pad = np.zeros(r.shape[:-1] + (1,))
        L = np.concatenate([pad, L], axis=-1)
        return np.expm1(L[..., h:] - L[..., :-h])
    m = (n // h) * h
    blocks = np.log1p(r[..., :m]).reshape(r.shape[:-1] + (n // h, h))
    return np.expm1(blocks.sum(axis=-1))


def loss_probability_curve(df: pd.DataFrame, horizons=HORIZONS_M,
                           cols=None, overlapping: bool = True) -> pd.DataFrame:
    """Share of ``h``-period windows with a negative compounded return, per asset."""
    cols = list(cols or df.columns)
    rows = []
    for h in horizons:
        row = {"h": h}
        for c in cols:
            R = horizon_returns(df[c].dropna().to_numpy(), h, overlapping)
            row[c] = float((R < 0).mean()) if R.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("h")


def pt_curve(df: pd.DataFrame, horizons=HORIZONS_M, cols=None, method: str = "overlapping",
             alpha: float = ALPHA, lam: float = LAMBDA, weighting: bool = True,
             n_iid: int = 20000, seed: int = 1025) -> pd.DataFrame:
    """Prospect-theory value of holding each asset, per evaluation horizon.

    ``method`` ∈ {``overlapping``, ``nonoverlapping``, ``iid``}. The ``iid`` estimator draws
    ``n_iid`` paths of joint monthly returns independently with replacement (fixed ``seed``).
    """
    cols = list(cols or df.columns)
    X = df[cols].dropna().to_numpy()
    rows = []
    if method == "iid":
        rng = np.random.default_rng(seed)
        hmax = max(horizons)
        idx = rng.integers(0, len(X), size=(n_iid, hmax))
        L = np.cumsum(np.log1p(X[idx]), axis=1)            # n_iid × hmax × assets
        wts = decision_weights(n_iid, weighting=weighting)
        for h in horizons:
            row = {"h": h}
            for j, c in enumerate(cols):
                row[c] = float(cpt_value(np.expm1(L[:, h - 1, j]), alpha, lam,
                                         weighting=weighting, _weights=wts))
            rows.append(row)
        return pd.DataFrame(rows).set_index("h")
    ov = method == "overlapping"
    if method not in ("overlapping", "nonoverlapping"):
        raise ValueError("method must be overlapping, nonoverlapping or iid")
    for h in horizons:
        row = {"h": h}
        for j, c in enumerate(cols):
            R = horizon_returns(X[:, j], h, ov)
            row[c] = float(cpt_value(R, alpha, lam, weighting=weighting)) if R.size >= 2 \
                else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("h")


def break_even(horizons, gap) -> float:
    """Shortest horizon beyond which ``gap`` (stocks − alternative) stays positive.

    Log-linear interpolation between the last non-positive grid point and the next one.
    ``inf`` if the gap is not positive at the longest horizon; the first grid point if it is
    positive everywhere (censored). NaNs are skipped.
    """
    h = np.asarray(horizons, dtype=float)
    g = np.asarray(gap, dtype=float)
    ok = np.isfinite(g)
    h, g = h[ok], g[ok]
    if g.size == 0 or g[-1] <= 0.0:
        return float("inf")
    neg = np.where(g <= 0.0)[0]
    if neg.size == 0:
        return float(h[0])
    i = int(neg[-1])
    t = -g[i] / (g[i + 1] - g[i])
    return float(np.exp(np.log(h[i]) + t * (np.log(h[i + 1]) - np.log(h[i]))))


# --------------------------------------------------------------------------- #
# Block bootstrap
# --------------------------------------------------------------------------- #
def cbb_indices(n: int, block: int, n_boot: int, rng) -> np.ndarray:
    """Circular-block-bootstrap index matrix (``n_boot × n``)."""
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(n_boot, nb * block)[:, :n]


def bootstrap_curves(df: pd.DataFrame, horizons=HORIZONS_M, cols=None, n_boot: int = 500,
                     block: int = BLOCK_M, alpha: float = ALPHA, lam: float = LAMBDA,
                     weighting: bool = True, seed: int = 1025) -> dict:
    """Joint circular block bootstrap of the loss-probability and PT curves.

    Returns ``{"cols", "horizons", "loss": B×H×A, "pt": B×H×A}`` computed on overlapping
    windows of each resampled monthly path. The whole path is resampled jointly across assets,
    so cross-asset correlation and within-block serial dependence are preserved.
    """
    cols = list(cols or df.columns)
    X = df[cols].dropna().to_numpy()
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = cbb_indices(n, block, n_boot, rng)
    H, A = len(horizons), len(cols)
    loss = np.full((n_boot, H, A), np.nan)
    pt = np.full((n_boot, H, A), np.nan)
    for j in range(A):
        R = X[idx, j]                                       # B × n
        L = np.concatenate([np.zeros((n_boot, 1)), np.cumsum(np.log1p(R), axis=1)], axis=1)
        for k, h in enumerate(horizons):
            if h >= n:
                continue
            Rh = np.expm1(L[:, h:] - L[:, :-h])
            loss[:, k, j] = (Rh < 0).mean(axis=1)
            wts = decision_weights(Rh.shape[1], weighting=weighting)
            pt[:, k, j] = cpt_value(Rh, alpha, lam, weighting=weighting, _weights=wts)
    return {"cols": cols, "horizons": tuple(horizons), "loss": loss, "pt": pt}


def two_sided_p(draws) -> float:
    """Percentile-bootstrap two-sided p-value for "the statistic is zero"."""
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan
    return float(min(1.0, 2.0 * min((d <= 0).mean(), (d >= 0).mean())))


def summarise_gap(point: pd.DataFrame, boot: dict, a: str = "stock", b: str = "bond",
                  ci: float = 0.95) -> dict:
    """Point gap curve, its bootstrap band and p-values, and the break-even with its CI."""
    hz = np.asarray(boot["horizons"], dtype=float)
    ia, ib = boot["cols"].index(a), boot["cols"].index(b)
    G = boot["pt"][:, :, ia] - boot["pt"][:, :, ib]
    gap = (point[a] - point[b]).to_numpy()
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    be_draws = np.array([break_even(hz, g) for g in G])
    be_point = break_even(hz, gap)
    finite = np.isfinite(be_draws)
    table = pd.DataFrame({
        "gap": gap,
        "gap_lo": np.nanquantile(G, lo_q, axis=0),
        "gap_hi": np.nanquantile(G, hi_q, axis=0),
        "p": [two_sided_p(G[:, k]) for k in range(G.shape[1])],
        "share_stocks_preferred": (G > 0).mean(axis=0),
    }, index=pd.Index(boot["horizons"], name="h"))
    return {
        "table": table, "break_even": be_point,
        # inverted-CDF quantiles: no interpolation, so an infinite draw stays infinite
        "be_lo": float(np.quantile(be_draws, lo_q, method="inverted_cdf")),
        "be_hi": float(np.quantile(be_draws, hi_q, method="inverted_cdf")),
        "be_median": float(np.quantile(be_draws, 0.5, method="inverted_cdf")),
        "share_finite": float(finite.mean()),
        "share_contains_12": float(np.mean((be_draws >= 6) & (be_draws <= 24))),
        "be_draws": be_draws,
    }


def loss_band(boot: dict, ci: float = 0.95) -> dict:
    """Bootstrap 95% band of each asset's loss-probability curve."""
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    out = {}
    for j, c in enumerate(boot["cols"]):
        out[c] = pd.DataFrame({
            "lo": np.nanquantile(boot["loss"][:, :, j], lo_q, axis=0),
            "hi": np.nanquantile(boot["loss"][:, :, j], hi_q, axis=0)},
            index=pd.Index(boot["horizons"], name="h"))
    return out


def param_sweep(df: pd.DataFrame, a: str = "stock", b: str = "bond", horizons=HORIZONS_M,
                alphas=(0.70, 0.80, 0.88, 0.95, 1.0),
                lambdas=(1.0, 1.5, 2.0, 2.25, 2.5, 3.0), weighting: bool = True
                ) -> pd.DataFrame:
    """Break-even horizon (months) over a λ × α grid, overlapping estimator."""
    rows = []
    for al in alphas:
        for la in lambdas:
            P = pt_curve(df, horizons, cols=[a, b], alpha=al, lam=la, weighting=weighting)
            rows.append({"alpha": al, "lambda": la,
                         "break_even": break_even(horizons, (P[a] - P[b]).to_numpy())})
    return pd.DataFrame(rows).pivot(index="lambda", columns="alpha", values="break_even")


def break_even_lambda(df: pd.DataFrame, horizon: int = 12, a: str = "stock", b: str = "bond",
                      alpha: float = ALPHA, weighting: bool = True,
                      lo: float = 0.5, hi: float = 10.0) -> float:
    """The λ that makes an investor exactly indifferent at ``horizon`` (bisection).

    The "implied loss aversion" reading of the equity premium: how loss-averse must an annual
    evaluator be for the historical premium to be a fair price? ``nan`` if no root in range.
    """
    X = df[[a, b]].dropna()
    Ra = horizon_returns(X[a].to_numpy(), horizon)
    Rb = horizon_returns(X[b].to_numpy(), horizon)

    def f(lam):
        return float(cpt_value(Ra, alpha, lam, weighting=weighting)
                     - cpt_value(Rb, alpha, lam, weighting=weighting))
    flo, fhi = f(lo), f(hi)
    if np.sign(flo) == np.sign(fhi):
        return float("nan")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if np.sign(fm) == np.sign(flo):
            lo, flo = mid, fm
        else:
            hi = mid
    return float(0.5 * (lo + hi))


# --------------------------------------------------------------------------- #
# The discipline question — the myopic switcher
# --------------------------------------------------------------------------- #
def period_codes(index: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """Integer evaluation-period id per observation.

    ``D`` every observation; ``W`` Saturday-to-Friday weeks; ``M`` calendar months;
    ``Q`` calendar quarters; ``Y`` calendar years.
    """
    idx = pd.DatetimeIndex(index)
    if freq == "D":
        return np.arange(len(idx))
    if freq == "W":
        return ((idx.normalize() - pd.Timestamp("1970-01-03")).days // 7).to_numpy()
    if freq == "M":
        return (idx.year * 12 + idx.month - 1).to_numpy()
    if freq == "Q":
        return (idx.year * 4 + (idx.month - 1) // 3).to_numpy()
    if freq == "Y":
        return idx.year.to_numpy()
    raise ValueError(f"unknown frequency {freq!r}")


def myopic_switcher(stock: pd.Series, bill: pd.Series, freq: str, cost_bps: float = 5.0
                    ) -> pd.DataFrame:
    """The behaviour "don't look" is meant to prevent, run on native-frequency returns.

    At the end of each evaluation period the investor sees the stock market's compounded
    return over that period. Loss (< 0) → hold bills for the whole next period; gain → hold
    stocks. The first period is held in stocks. **Effective lag: one** — the period's sign is
    known at the close of its last observation and sets the weight from the next observation
    on. A switch turns over 1× NAV one-way and pays ``cost_bps`` on it, charged on the first
    observation of the new period.

    Returns a frame with ``w`` (stock weight), ``turnover``, ``gross``, ``net``, ``bh`` (buy and
    hold stocks), ``bill``.
    """
    df = pd.concat([stock.rename("s"), bill.rename("b")], axis=1).dropna()
    codes = period_codes(df.index, freq)
    uniq, inv = np.unique(codes, return_inverse=True)
    lr = np.log1p(df["s"].to_numpy())
    per_ret = np.expm1(np.bincount(inv, weights=lr, minlength=len(uniq)))
    sig = (per_ret > 0.0).astype(float)
    pos = np.empty(len(uniq))
    pos[0] = 1.0
    pos[1:] = sig[:-1]                       # one lag: period p's sign sets period p+1
    w = pos[inv]
    first = np.r_[True, inv[1:] != inv[:-1]]
    prev = np.r_[1.0, w[:-1]]
    turnover = np.where(first, np.abs(w - prev), 0.0)
    s, b = df["s"].to_numpy(), df["b"].to_numpy()
    gross = w * s + (1.0 - w) * b
    net = gross - turnover * cost_bps / 1e4
    return pd.DataFrame({"w": w, "turnover": turnover, "gross": gross, "net": net,
                         "bh": s, "bill": b}, index=df.index)


def perf_stats(r: np.ndarray, bill: np.ndarray, ppy: int) -> dict:
    """Annualised geometric return, volatility, excess-of-bills Sharpe, max drawdown."""
    r = np.asarray(r, dtype=float)
    ex = r - np.asarray(bill, dtype=float)
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    sd = ex.std(ddof=1)
    return {"cagr": float(wealth[-1] ** (ppy / len(r)) - 1.0),
            "vol": float(r.std(ddof=1) * np.sqrt(ppy)),
            "sharpe": float(ex.mean() / sd * np.sqrt(ppy)) if sd > 0 else np.nan,
            "maxdd": float((wealth / peak - 1.0).min())}


def _sharpe_rows(ex: np.ndarray, ppy: int) -> np.ndarray:
    sd = ex.std(axis=-1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ex.mean(axis=-1) / sd * np.sqrt(ppy)


def discipline_test(sw: pd.DataFrame, ppy: int, block: int, n_boot: int = 1000,
                    seed: int = 1025) -> dict:
    """Buy-and-hold vs the myopic switcher (net): return gap and Sharpe gap, with inference.

    The return gap ``d_t = bh_t − net_t`` gets a Newey-West t and a paired circular-block
    bootstrap p; the Sharpe gap (both in excess of bills) gets a paired bootstrap p from the
    same resamples.
    """
    bh, net, bill = sw["bh"].to_numpy(), sw["net"].to_numpy(), sw["bill"].to_numpy()
    d = bh - net
    hac = mean_tstat_hac(pd.Series(d))
    rng = np.random.default_rng(seed)
    idx = cbb_indices(len(d), block, n_boot, rng)
    dm = d[idx].mean(axis=1)
    sh_bh = _sharpe_rows((bh - bill)[idx], ppy)
    sh_my = _sharpe_rows((net - bill)[idx], ppy)
    s_bh = perf_stats(bh, bill, ppy)
    s_net = perf_stats(net, bill, ppy)
    s_gross = perf_stats(sw["gross"].to_numpy(), bill, ppy)
    return {
        "bh_cagr": s_bh["cagr"], "my_cagr_gross": s_gross["cagr"],
        "my_cagr_net": s_net["cagr"], "d_cagr_net": s_bh["cagr"] - s_net["cagr"],
        "d_cagr_gross": s_bh["cagr"] - s_gross["cagr"],
        "d_mean_ann": float(d.mean() * ppy), "t_hac": float(hac["tstat"]),
        "p_ret": two_sided_p(dm),
        "bh_sharpe": s_bh["sharpe"], "my_sharpe_net": s_net["sharpe"],
        "my_sharpe_gross": s_gross["sharpe"],
        "d_sharpe": s_bh["sharpe"] - s_net["sharpe"], "p_sharpe": two_sided_p(sh_bh - sh_my),
        "bh_vol": s_bh["vol"], "my_vol": s_net["vol"],
        "bh_maxdd": s_bh["maxdd"], "my_maxdd": s_net["maxdd"],
        "time_invested": float(sw["w"].mean()),
        "switches_per_year": float(sw["turnover"].sum() / len(sw) * ppy),
        "n": int(len(sw)),
    }


def discipline_table(native: pd.DataFrame, freqs, native_ppy: int, cost_bps: float = 5.0,
                     n_boot: int = 1000, base_block: int = 12, seed: int = 1025
                     ) -> pd.DataFrame:
    """One row per evaluation frequency: buy-and-hold vs the myopic switcher, net and gross.

    ``native`` holds ``stock`` and ``bill`` at the native frequency (daily or monthly). The
    bootstrap block is at least twice the evaluation period in native units, so a resample never
    cuts through the dependence the rule itself creates.
    """
    per_len = {"D": 1, "W": 5, "M": 21 if native_ppy > 12 else 1,
               "Q": 63 if native_ppy > 12 else 3, "Y": 252 if native_ppy > 12 else 12}
    rows = []
    for f in freqs:
        sw = myopic_switcher(native["stock"], native["bill"], f, cost_bps)
        block = max(base_block, 2 * per_len[f])
        rows.append({"freq": f, **discipline_test(sw, native_ppy, block, n_boot, seed)})
    return pd.DataFrame(rows).set_index("freq")


def cost_sweep(native: pd.DataFrame, freq: str, native_ppy: int,
               costs=(0.0, 2.0, 5.0, 10.0, 25.0, 50.0)) -> pd.DataFrame:
    """Annualised gap (buy-and-hold minus myopic, CAGR) as the one-way cost rises."""
    rows = []
    for c in costs:
        sw = myopic_switcher(native["stock"], native["bill"], freq, c)
        bh = perf_stats(sw["bh"].to_numpy(), sw["bill"].to_numpy(), native_ppy)
        my = perf_stats(sw["net"].to_numpy(), sw["bill"].to_numpy(), native_ppy)
        rows.append({"cost_bps": c, "my_cagr": my["cagr"], "d_cagr": bh["cagr"] - my["cagr"],
                     "my_sharpe": my["sharpe"], "bh_sharpe": bh["sharpe"]})
    return pd.DataFrame(rows).set_index("cost_bps")


# --------------------------------------------------------------------------- #
# Verdict — thresholds fixed before the real tape was run
# --------------------------------------------------------------------------- #
def _join(names) -> str:
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _fmt_months(m: float) -> str:
    if not np.isfinite(m):
        return "never (beyond 10 years)"
    if m < 12:
        return f"{m:.1f} months"
    return f"{m:.0f} months ({m / 12:.1f} years)"


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    **Signal** (the Benartzi-Thaler calculation, stocks vs bonds, TK92 parameters, nominal,
    overlapping windows, block-bootstrap inference):

    - the preference must genuinely *flip* with horizon: the PT gap (stocks − bonds) is
      negative at one month and positive at ten years, **each with bootstrap p < 0.05**;
    - the break-even must be *pinned near a year*: its 95% bootstrap CI is finite, lies
      inside [3, 36] months and contains 12;
    - and the mechanism must exist in both regimes: a finite break-even point estimate both
      pre- and post-1970.

    No significant flip → **None**. Flip + pinned + both regimes → **Real**. Flip + pinned but
    one regime without a finite break-even → **Mixed** (a genuine regime split). Flip without
    a pinned horizon → **Weak** (the horizon matters; the one-year calibration is not
    certified by this tape).

    **Tradability** (the cost of the myopic behaviour, buy-and-hold vs the switcher, net of
    5 bp one-way, evaluation daily, weekly, monthly, quarterly, yearly):

    - **Investable** if buy-and-hold beats the switcher on excess-of-bills Sharpe with
      bootstrap p < 0.05 at *every* frequency — looking is unambiguously costly,
      risk-adjusted;
    - **Fragile** if buy-and-hold's net return advantage is significant (p < 0.05) at one
      frequency or more but not risk-adjusted everywhere — the behaviour costs money, but
      mostly forgone premium, and not at every clock;
    - **Mirage** if no frequency shows a significant net cost — looking less buys comfort,
      not return.
    """
    flip = (h["gap_short"] < 0 and h["p_short"] < 0.05
            and h["gap_long"] > 0 and h["p_long"] < 0.05)
    pinned = (np.isfinite(h["be_hi"]) and h["be_lo"] >= 3.0 and h["be_hi"] <= 36.0
              and h["be_lo"] <= BTH_BREAK_EVEN <= h["be_hi"])
    regimes = np.isfinite(h["be_pre"]) and np.isfinite(h["be_post"])
    if not flip:
        signal = "None"
    elif pinned and regimes:
        signal = "Real"
    elif pinned:
        signal = "Mixed"
    else:
        signal = "Weak"

    disc = h["discipline"]
    risk_costly = [d["d_sharpe"] > 0 and d["p_sharpe"] < 0.05 for d in disc]
    costly = [d["d_cagr_net"] > 0 and d["p_ret"] < 0.05 for d in disc]
    if all(risk_costly):
        trad = "Investable"
    elif any(costly):
        trad = "Fragile"
    else:
        trad = "Mirage"

    costly_names = [FREQ_LABEL[d["freq"]] for d, c in zip(disc, costly) if c]
    sharpe_better = [FREQ_LABEL[d["freq"]] for d in disc if d["d_sharpe"] < 0]
    worst = max(disc, key=lambda d: d["d_cagr_net"])
    best = min(disc, key=lambda d: d["d_cagr_net"])

    if flip:
        opener = (
            f"The horizon really does flip a loss-averse investor's preference, and the flip "
            f"clears the bar{' — only just, at ten years' if h['p_long'] > 0.01 else ''}: "
            f"with Tversky-Kahneman (1992) preferences a one-month evaluator "
            f"prefers bonds to stocks (prospect-theory gap {h['gap_short']:+.4f}, "
            f"block-bootstrap p {h['p_short']:.3f}) and a ten-year evaluator prefers stocks "
            f"(gap {h['gap_long']:+.3f}, p {h['p_long']:.3f}). ")
    else:
        opener = (
            f"The horizon does not flip the preference with any statistical confidence: the "
            f"prospect-theory gap (stocks − bonds) is {h['gap_short']:+.4f} at one month "
            f"(p {h['p_short']:.3f}) and {h['gap_long']:+.3f} at ten years "
            f"(p {h['p_long']:.3f}). ")
    if pinned:
        calib = "That interval is tight enough to pin the one-year story. "
    else:
        calib = (
            f"That interval is far too wide to certify the one-year calibration — and the "
            f"one-year figure itself turns out to be the *i.i.d.* answer: drawing the same "
            f"months independently gives {_fmt_months(h['be_iid'])}, while the real sequence "
            f"of returns (overlapping windows) gives {_fmt_months(h['be'])}, and "
            f"{_fmt_months(h['be_bt'])} on Benartzi and Thaler's own 1926-1990 window. ")
    signal_why = (
        opener
        + f"The chance of *seeing* a loss on stocks falls from {h['loss_1d']:.0%} of days to "
        f"{h['loss_1m']:.0%} of months, {h['loss_12m']:.0%} of years and "
        f"{h['loss_120m']:.0%} of decades. The break-even horizon against bonds on 1926-2018 "
        f"is **{_fmt_months(h['be'])}**, against Benartzi and Thaler's roughly one year, with a "
        f"95% block-bootstrap interval of {_fmt_months(h['be_lo'])} to "
        f"{_fmt_months(h['be_hi'])}. "
        + calib
        + f"Against bills the break-even is {_fmt_months(h['be_bills'])} (95% CI "
        f"{_fmt_months(h['be_bills_lo'])} to {_fmt_months(h['be_bills_hi'])}). Pre-1970 it is "
        f"{_fmt_months(h['be_pre'])}, post-1970 {_fmt_months(h['be_post'])}; in real terms "
        f"(core CPI, 1957+) {_fmt_months(h['be_real'])}. The loss aversion that would make an "
        f"annual evaluator exactly indifferent is λ = {h['lambda_12']:.2f} "
        f"(Tversky-Kahneman: 2.25).")
    if costly_names:
        cost_s = (f"The behaviour is measurably costly, net, only at the "
                  f"{_join(costly_names)} clock{'s' if len(costly_names) > 1 else ''}. ")
    else:
        cost_s = "No evaluation clock shows a statistically significant net cost. "
    if sharpe_better:
        sharpe_s = (f"Risk-adjusted, the switcher's excess Sharpe is at least as high as "
                    f"buy-and-hold's at the {_join(sharpe_better)} "
                    f"clock{'s' if len(sharpe_better) > 1 else ''} — what the "
                    f"slow checker gives up is mostly premium it had stopped carrying risk "
                    f"for. ")
    else:
        sharpe_s = "Buy-and-hold has the higher excess Sharpe at every clock. "
    trad_why = (
        "Looking less changes what you see, not what you earn, so the only money in the "
        "advice is the cost of the behaviour it prevents. The myopic investor here flees to "
        "bills after seeing a loss and returns after a gain — one lag, 5 bp one-way. "
        + cost_s
        + f"{FREQ_LABEL[worst['freq']].capitalize()} checking gives up "
        f"{worst['d_cagr_net']:.2%} a year of compound return (HAC t {worst['t_hac']:.2f}, "
        f"{worst['switches_per_year']:.0f} switches a year), while "
        f"{FREQ_LABEL[best['freq']]} checking gives up {best['d_cagr_net']:.2%} "
        f"(HAC t {best['t_hac']:.2f}). "
        + sharpe_s
        + "So 'don't look' is a seatbelt against one specific, expensive habit — reacting "
        "to daily and weekly noise — not an edge: it earns the equity premium you were always "
        "paid for, and nothing on top.")
    iid_clause = (" — the famous one year is the i.i.d. answer —"
                  if (not pinned and 6.0 <= h["be_iid"] <= 18.0) else "")
    if costly_names:
        pay = f"it only pays against {_join(costly_names)} checking"
    else:
        pay = "no checking clock shows a significant cost"
    one = (
        f"A loss-averse investor does come to prefer stocks once they look rarely enough, but "
        f"the break-even is {_fmt_months(h['be'])} with a 95% interval of "
        f"{_fmt_months(h['be_lo'])} to {_fmt_months(h['be_hi'])}{iid_clause} and looking "
        f"less is a seatbelt, not an edge: {pay}.")
    return {"signal": signal, "signal_why": signal_why, "trad": trad, "trad_why": trad_why,
            "one_sentence": one}
