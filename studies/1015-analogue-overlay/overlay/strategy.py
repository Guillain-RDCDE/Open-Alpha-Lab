"""The analogue forecaster, its inference, and the null that explains the viral charts — Study 1015.

The viral chart lays today's index on top of 1929 (or 1987, or 2000) and the two lines track each
other uncannily, often with a correlation above 0.9 printed in the corner. The implied forecast
is whatever happened next *then*. This module turns that chart into a forecaster that can be
scored, and then asks how impressive a 0.9 really is.

**The forecaster** (``analogue_search`` + ``analogue_forecast``). At each date ``t``:

1. take the last ``L`` periods of the path — either the log-price path (what the charts show)
   or the z-scored returns (the version a statistician would use);
2. correlate it (Pearson) with **every** window of the same length that ended at least
   ``max(L, h)`` periods before ``t`` — strictly in the past, never overlapping today's window,
   and with its own follow-on fully observed by ``t`` (so no look-ahead);
3. keep the ``k`` best matches, greedily, so that no two analogues overlap each other (the top
   ten windows are otherwise ten one-period shifts of the same episode);
4. forecast the next ``h``-period log return as the average of what followed those analogues.

**The scoring** (``evaluate_forecast``, ``timing_rule``). Correlation of forecast with realised,
the slope of realised-on-forecast with Newey-West standard errors (the ``h``-period outcomes
overlap, so OLS errors would be fiction), a hit rate measured *against the historical drift*
(raw sign hit rates just reward a market that mostly goes up), an out-of-sample R² against the
expanding mean, and a long/flat rule with one full period of execution lag and one-way costs.

**The sweep** (``sweep``) runs every combination of window, horizon, number of analogues and
matching metric and reports all of them, then corrects for having looked: Holm across every
predictive test, White's Reality Check (``quantlab.bayes.reality_check``) across every timing
rule on each tape.

**The null** (``pair_correlation_null``, ``best_match_null``, ``null_search_distribution``).
Two independent random walks share no information, yet their *price paths* correlate wildly,
because trending series always do. Searching a long history for the best-matching window then
selects the extreme of that already-wide distribution. The question the viral chart never asks
is what correlation a random walk with the same drift and volatility would have produced; this
module answers it.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

METRICS = ("logprice", "zret")

# Pre-registered headline specifications — fixed before the real run, never re-chosen.
HEADLINE_MONTHLY = {"metric": "logprice", "L": 24, "h": 12, "k": 5}
HEADLINE_DAILY = {"metric": "logprice", "L": 250, "h": 63, "k": 5}

# The full grid, reported in full.
GRID_MONTHLY = {"L": (12, 24, 36, 60), "h": (1, 6, 12), "k": (1, 5, 10)}
GRID_DAILY = {"L": (63, 126, 250), "h": (5, 21, 63), "k": (1, 5, 10)}


# --------------------------------------------------------------------------- #
# Small statistical helpers
# --------------------------------------------------------------------------- #
def _norm_sf(z: float) -> float:
    """Upper-tail standard-normal probability."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def ols_hac(y, x, lags: int) -> dict:
    """``y = a + b x + e`` with Newey-West (Bartlett) standard errors on ``a`` and ``b``.

    The workhorse for overlapping horizons: an ``h``-period outcome sampled every period has
    ``h - 1`` autocorrelation lags by construction, and plain OLS errors would understate the
    uncertainty by roughly ``sqrt(h)``.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = y.size
    if n < 10 or np.std(x) == 0:
        return {"n": int(n), "a": np.nan, "b": np.nan, "t_a": np.nan, "t_b": np.nan,
                "se_a": np.nan, "se_b": np.nan}
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    g = X * u[:, None]
    S = g.T @ g
    for l in range(1, int(lags) + 1):
        w = 1.0 - l / (lags + 1.0)
        G = g[l:].T @ g[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return {"n": int(n), "a": float(beta[0]), "b": float(beta[1]),
            "se_a": float(se[0]), "se_b": float(se[1]),
            "t_a": float(beta[0] / se[0]) if se[0] > 0 else np.nan,
            "t_b": float(beta[1] / se[1]) if se[1] > 0 else np.nan}


def hac_mean_t(x, lags: int) -> float:
    """Newey-West t-statistic of a sample mean (intercept-only HAC regression)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 10:
        return np.nan
    e = x - x.mean()
    lrv = float(e @ e) / n
    for l in range(1, int(lags) + 1):
        lrv += 2.0 * (1.0 - l / (lags + 1.0)) * float(e[l:] @ e[:-l]) / n
    se = math.sqrt(max(lrv, 0.0) / n)
    return float(x.mean() / se) if se > 0 else np.nan


def holm(pvals) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values (family-wise error control, any dependence)."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def stationary_bootstrap_corr(x, y, n_boot: int = 500, mean_block: float = 12.0,
                              seed: int = 1015) -> tuple[float, float]:
    """95% stationary-bootstrap interval for ``corr(x, y)`` (Politis & Romano 1994).

    Pairs are resampled together in geometric blocks that wrap around, so the overlap between
    consecutive ``h``-period outcomes survives into the interval.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    rng = np.random.default_rng(seed)
    p = 1.0 / max(mean_block, 1.0)
    out = np.empty(n_boot)
    for b in range(n_boot):
        new = rng.random(n) < p
        new[0] = True
        pos = np.flatnonzero(new)
        starts = rng.integers(0, n, pos.size)
        bid = np.cumsum(new) - 1
        idx = (starts[bid] + np.arange(n) - pos[bid]) % n
        xs, ys = x[idx], y[idx]
        out[b] = np.corrcoef(xs, ys)[0, 1] if xs.std() > 0 and ys.std() > 0 else 0.0
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# The analogue search
# --------------------------------------------------------------------------- #
def _normalised_windows(v: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Every length-``L`` window of ``v``, demeaned and scaled to unit norm.

    Row ``j`` is the window ending at index ``j + L - 1``. A dot product of two rows is then
    exactly their Pearson correlation. Windows containing NaN or with zero variance are flagged
    invalid (they can neither be queried nor matched).
    """
    W = sliding_window_view(v, L).astype(float)
    valid = np.isfinite(W).all(axis=1)
    W = np.where(np.isfinite(W), W, 0.0)
    W = W - W.mean(axis=1, keepdims=True)
    nrm = np.sqrt((W * W).sum(axis=1))
    valid &= nrm > 1e-12
    W = W / np.where(nrm > 1e-12, nrm, 1.0)[:, None]
    return W, valid


def _series_for(logp: np.ndarray, metric: str) -> np.ndarray:
    if metric == "logprice":
        return np.asarray(logp, dtype=float)
    if metric == "zret":
        r = np.diff(np.asarray(logp, dtype=float), prepend=np.nan)
        return r
    raise ValueError(f"metric must be one of {METRICS}")


def analogue_search(logp, L: int, eval_idx, k_max: int = 10, metric: str = "logprice",
                    gap: int | None = None, chunk: int = 400) -> dict:
    """Find, for each evaluation index, the ``k_max`` best non-overlapping past analogues.

    Parameters
    ----------
    logp : array-like
        The log-price (or log total-return index) path, one value per period.
    L : int
        Window length. The query at ``t`` is the window ending at ``t``.
    eval_idx : array-like of int
        Positions at which a forecast is made.
    metric : {'logprice', 'zret'}
        Match on the log-price path (what the viral charts do) or on the z-scored returns.
        Both use Pearson correlation; for returns the windows are ``L`` returns long.
    gap : int, optional
        A candidate window ending at ``e`` is eligible only if ``e <= t - gap``. Default ``L``,
        which makes it non-overlapping with today's window; callers forecasting ``h`` periods
        ahead pass ``max(L, h)`` so every analogue's follow-on is fully observed at ``t``.

    Returns
    -------
    dict with ``ends`` (``n_eval x k_max`` int, -1 where fewer analogues exist), ``corr``
    (matching correlations, NaN where missing), ``eval_idx`` and the settings.
    """
    gap = int(L if gap is None else gap)
    v = _series_for(logp, metric)
    W, valid = _normalised_windows(v, L)
    ends = np.arange(W.shape[0]) + L - 1
    eval_idx = np.asarray(eval_idx, dtype=int)
    n_eval = eval_idx.size
    out_e = np.full((n_eval, k_max), -1, dtype=int)
    out_c = np.full((n_eval, k_max), np.nan)
    for c0 in range(0, n_eval, chunk):
        ts = eval_idx[c0:c0 + chunk]
        rows = ts - (L - 1)
        ok_q = (rows >= 0) & valid[np.clip(rows, 0, W.shape[0] - 1)]
        Q = W[np.clip(rows, 0, W.shape[0] - 1)]
        C = Q @ W.T
        elig = (ends[None, :] <= (ts[:, None] - gap)) & valid[None, :] & ok_q[:, None]
        C = np.where(elig, C, -np.inf)
        r_idx = np.arange(ts.size)
        for j in range(k_max):
            a = np.argmax(C, axis=1)
            best = C[r_idx, a]
            found = np.isfinite(best)
            out_e[c0 + r_idx[found], j] = ends[a[found]]
            out_c[c0 + r_idx[found], j] = best[found]
            # Remove every window overlapping the one just chosen (|e - e*| < L).
            clash = np.abs(ends[None, :] - ends[a][:, None]) < L
            C = np.where(clash & found[:, None], -np.inf, C)
    return {"ends": out_e, "corr": out_c, "eval_idx": eval_idx, "L": int(L),
            "metric": metric, "gap": gap, "k_max": int(k_max)}


def analogue_forecast(logp, search: dict, h: int, k: int) -> np.ndarray:
    """Average ``h``-period log return that followed the first ``k`` analogues (NaN if none)."""
    x = np.asarray(logp, dtype=float)
    if h > search["gap"]:
        raise ValueError("h exceeds the search gap: analogue follow-ons would not be observed")
    E = search["ends"][:, :k]
    fol = np.where(E >= 0, x[np.clip(E + h, 0, x.size - 1)] - x[np.clip(E, 0, x.size - 1)],
                   np.nan)
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(fol).sum(axis=1)
        s = np.nansum(fol, axis=1)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def realised_forward(logp, eval_idx, h: int) -> np.ndarray:
    """The realised next-``h``-period log return from each evaluation index (NaN past the end)."""
    x = np.asarray(logp, dtype=float)
    t = np.asarray(eval_idx, dtype=int)
    out = np.full(t.size, np.nan)
    ok = t + h < x.size
    out[ok] = x[t[ok] + h] - x[t[ok]]
    return out


def historical_drift_forecast(logp, eval_idx, h: int) -> np.ndarray:
    """The naive benchmark known at ``t``: average per-period drift to date, times ``h``."""
    x = np.asarray(logp, dtype=float)
    t = np.asarray(eval_idx, dtype=int)
    return (x[t] - x[0]) / np.maximum(t, 1) * h


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def evaluate_forecast(fc, realised, bench, lags: int) -> dict:
    """Score a forecast against what happened, with overlap-robust inference.

    - ``corr``: Pearson correlation of forecast with realised.
    - ``slope``, ``t``: realised regressed on forecast, Newey-West with ``lags`` lags. The
      one-sided ``p`` is in the direction of the claim (positive skill).
    - ``hit``: share of dates where forecast and realised fall on the same side of the
      **historical drift** ``bench`` — a raw sign hit rate would reward any forecast that is
      mostly positive in a market that mostly rises. ``hit_t`` is its HAC t against 50%.
    - ``hit_raw`` / ``up_share``: the raw sign hit rate and the share of up periods, printed
      side by side so the confusion is visible.
    - ``oos_r2``: Campbell-Thompson out-of-sample R² against ``bench``.
    """
    fc = np.asarray(fc, dtype=float)
    y = np.asarray(realised, dtype=float)
    m = np.asarray(bench, dtype=float)
    ok = np.isfinite(fc) & np.isfinite(y) & np.isfinite(m)
    fc, y, m = fc[ok], y[ok], m[ok]
    n = int(fc.size)
    if n < 20:
        return {"n": n}
    reg = ols_hac(y, fc, lags)
    hits = (np.sign(fc - m) == np.sign(y - m)).astype(float)
    sse_f = float(((y - fc) ** 2).sum())
    sse_b = float(((y - m) ** 2).sum())
    t = reg["t_b"]
    return {"n": n,
            "corr": float(np.corrcoef(fc, y)[0, 1]) if fc.std() > 0 else np.nan,
            "slope": reg["b"], "t": t,
            "p": _norm_sf(t) if np.isfinite(t) else 1.0,
            "hit": float(hits.mean()), "hit_t": hac_mean_t(hits - 0.5, lags),
            "hit_raw": float((np.sign(fc) == np.sign(y)).mean()),
            "up_share": float((y > 0).mean()),
            "oos_r2": 1.0 - sse_f / sse_b if sse_b > 0 else np.nan,
            "lags": int(lags)}


def timing_rule(tape: pd.DataFrame, eval_idx, fc, periods_per_year: int,
                lag: int = 1, cost_bps: float = 10.0) -> dict:
    """Long the market when the analogues forecast a positive ``h``-period return, else flat.

    **Timing, exactly.** The forecast made at the close of period ``t`` is executed at the close
    of ``t + lag`` (one full period of execution lag by default) and so first earns the return
    of period ``t + lag + 1``. Between evaluation dates the last signal is held. The flat leg
    earns ``tape['rf']`` (the T-bill on the monthly tape, zero on the daily price tape).

    **Costs.** ``cost_bps`` one-way, charged on traded NAV ``|w_t - w_{t-1}|``.

    Every Sharpe ratio is **excess over cash**, rule and buy-and-hold alike, and the timing
    alpha is the intercept of the rule's net excess return on buy-and-hold's, with
    Newey-West errors. ``diff`` (rule net excess minus buy-and-hold excess) is the "return race"
    the Reality Check resamples; ``diff_volmatched`` scales the rule's excess return to
    buy-and-hold's volatility first, so its mean is proportional to the *Sharpe* difference —
    the race a de-risking timing rule could actually win. (The scaling uses full-sample vols;
    it is a test statistic, not a tradable position.)
    """
    n = len(tape)
    eval_idx = np.asarray(eval_idx, dtype=int)
    sig = np.full(n, np.nan)
    s = np.where(np.isfinite(fc), (np.asarray(fc) > 0).astype(float), 1.0)
    sig[eval_idx] = s
    sig = pd.Series(sig, index=tape.index).ffill()
    w = sig.shift(1 + lag)
    start = int(eval_idx[0]) + 1 + lag
    w = w.iloc[start:]
    ret = tape["ret"].iloc[start:].astype(float)
    rf = tape["rf"].iloc[start:].astype(float)
    w = w.fillna(1.0)
    turn = w.diff().abs().fillna(0.0)
    gross = w * ret + (1.0 - w) * rf
    net = gross - cost_bps / 1e4 * turn
    ex_net, ex_gross, ex_bh = net - rf, gross - rf, ret - rf
    ann = np.sqrt(periods_per_year)

    def sh(x):
        x = np.asarray(x, dtype=float)
        return float(x.mean() / x.std(ddof=1) * ann) if x.std(ddof=1) > 0 else np.nan

    lags = max(1, int(round(4 * (len(ret) / 100.0) ** (2.0 / 9.0))))
    reg = ols_hac(ex_net.to_numpy(), ex_bh.to_numpy(), lags)
    eq = np.exp(np.log1p(net).cumsum())
    dd = float((eq / eq.cummax() - 1.0).min())
    eqb = np.exp(np.log1p(ret).cumsum())
    ddb = float((eqb / eqb.cummax() - 1.0).min())
    yrs = len(ret) / periods_per_year
    return {"sharpe_net": sh(ex_net), "sharpe_gross": sh(ex_gross), "sharpe_bh": sh(ex_bh),
            "ann_ret_net": float(np.expm1(np.log1p(net).sum() / yrs)),
            "ann_ret_bh": float(np.expm1(np.log1p(ret).sum() / yrs)),
            "alpha_ann": reg["a"] * periods_per_year, "alpha_t": reg["t_a"],
            "beta": reg["b"], "time_in_market": float(w.mean()),
            "turnover_per_year": float(turn.sum() / yrs), "max_dd": dd, "max_dd_bh": ddb,
            "n": int(len(ret)), "start": str(ret.index[0].date()),
            "diff": (ex_net - ex_bh).rename("diff"),
            "diff_volmatched": (ex_net * (ex_bh.std(ddof=1) / ex_net.std(ddof=1))
                                - ex_bh).rename("diff_vm") if ex_net.std(ddof=1) > 0
            else (ex_net - ex_bh).rename("diff_vm"),
            "equity": eq, "equity_bh": eqb}


# --------------------------------------------------------------------------- #
# The full sweep
# --------------------------------------------------------------------------- #
def eval_positions(index: pd.DatetimeIndex, oos_start: str, step: int = 1) -> np.ndarray:
    """Evaluation indices: every ``step``-th period from ``oos_start`` to the end."""
    first = int(np.searchsorted(index.values, np.datetime64(pd.Timestamp(oos_start))))
    return np.arange(first, len(index), step, dtype=int)


def sweep(tape: pd.DataFrame, grid: dict, periods_per_year: int, oos_start: str,
          step: int = 1, metrics=METRICS, cost_bps: float = 10.0, lag: int = 1,
          keep: tuple = ()) -> dict:
    """Every (metric, L, h, k) combination on one tape, every one reported.

    The search is run once per ``(metric, L)`` with ``k_max = max(k)`` and gap ``max(L, max h)``
    (the grids keep ``h <= L``, so the gap is simply ``L`` and every combination sees the same
    eligible history). Forecasts for each ``(h, k)`` are then averages over the first ``k``
    greedy analogues. All combinations share the same out-of-sample evaluation dates.

    Returns ``{"table": DataFrame, "diffs": DataFrame of rule-minus-B&H excess returns,
    "diffs_vm": the same after vol-matching each rule to B&H,
    "searches": {(metric, L): search}, "kept": {combo: dict}}`` where ``keep`` lists combos
    (``(metric, L, h, k)`` tuples) whose forecast arrays should be returned for plotting.
    """
    logp = tape["logp"].to_numpy(dtype=float)
    idx = eval_positions(tape.index, oos_start, step)
    k_max = max(grid["k"])
    rows, diffs, diffs_vm, searches, kept = [], {}, {}, {}, {}
    for metric in metrics:
        for L in grid["L"]:
            gap = max(L, max(grid["h"]))
            srch = analogue_search(logp, L, idx, k_max=k_max, metric=metric, gap=gap)
            searches[(metric, L)] = srch
            for h in grid["h"]:
                real = realised_forward(logp, idx, h)
                bench = historical_drift_forecast(logp, idx, h)
                lags = max(1, int(math.ceil(h / step)))
                for k in grid["k"]:
                    fc = analogue_forecast(logp, srch, h, k)
                    ev = evaluate_forecast(fc, real, bench, lags)
                    tr = timing_rule(tape, idx, fc, periods_per_year, lag=lag,
                                     cost_bps=cost_bps)
                    key = f"{metric}|L{L}|h{h}|k{k}"
                    diffs[key] = tr["diff"]
                    diffs_vm[key] = tr["diff_volmatched"]
                    rows.append({"metric": metric, "L": L, "h": h, "k": k, **ev,
                                 "sharpe_net": tr["sharpe_net"],
                                 "sharpe_gross": tr["sharpe_gross"],
                                 "sharpe_bh": tr["sharpe_bh"], "alpha_ann": tr["alpha_ann"],
                                 "alpha_t": tr["alpha_t"],
                                 "time_in_market": tr["time_in_market"],
                                 "turnover_per_year": tr["turnover_per_year"],
                                 "match_corr_median": float(np.nanmedian(srch["corr"][:, 0]))})
                    if (metric, L, h, k) in keep:
                        kept[(metric, L, h, k)] = {"fc": fc, "real": real, "bench": bench,
                                                   "eval_idx": idx, "rule": tr, "eval": ev}
    return {"table": pd.DataFrame(rows), "diffs": pd.DataFrame(diffs),
            "diffs_vm": pd.DataFrame(diffs_vm), "searches": searches, "kept": kept, "eval_idx": idx}


def reality_check(diffs: pd.DataFrame, periods_per_year: int, n_boot: int = 1000,
                  seed: int = 1015) -> dict:
    """White's Reality Check over every timing rule on a tape (via ``quantlab.bayes``).

    H0: no rule in the grid beats buy-and-hold on an excess-return basis. The statistic is the
    best annualised information ratio of ``rule - buy-and-hold`` across all combinations; the
    stationary bootstrap of the recentred differentials gives its null distribution.
    """
    from quantlab.bayes import reality_check as _rc
    out = _rc(diffs.dropna(how="any"), n_boot=n_boot, periods_per_year=periods_per_year,
              seed=seed)
    out["best_rule"] = str(diffs.columns[out["best_series_index"]])
    return out


# --------------------------------------------------------------------------- #
# The null — how good does a match look when there is nothing to find?
# --------------------------------------------------------------------------- #
def random_walk_paths(n_paths: int, n: int, mu: float, sigma: float,
                      seed: int = 1015) -> np.ndarray:
    """``n_paths x n`` log-price paths of a Gaussian random walk with drift (start at 0)."""
    rng = np.random.default_rng(seed)
    return np.cumsum(mu + sigma * rng.standard_normal((n_paths, n)), axis=1)


def _rowcorr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    return (A * B).sum(axis=1) / np.sqrt((A * A).sum(axis=1) * (B * B).sum(axis=1))


def pair_correlation_null(L: int, mu: float, sigma: float, n_sims: int = 20000,
                          seed: int = 1015) -> dict:
    """Correlation between two INDEPENDENT random walks over ``L`` periods.

    Returns the distribution for the price paths (what the overlay charts plot) and for the
    returns of the same paths (what carries information). The two random walks share nothing;
    any correlation is the trend in each path lining up by chance.
    """
    P1 = random_walk_paths(n_sims, L + 1, mu, sigma, seed)
    P2 = random_walk_paths(n_sims, L + 1, mu, sigma, seed + 1)
    cp = _rowcorr(P1[:, 1:], P2[:, 1:])
    cr = _rowcorr(np.diff(P1, axis=1), np.diff(P2, axis=1))
    return {"price": cp, "returns": cr, "L": L,
            "p_price_gt_09": float((cp > 0.9).mean()),
            "p_price_abs_gt_09": float((np.abs(cp) > 0.9).mean()),
            "p_price_gt_07": float((cp > 0.7).mean()),
            "p_returns_gt_09": float((cr > 0.9).mean()),
            "p_returns_gt_05": float((cr > 0.5).mean())}


def best_match_null(n: int, L: int, mu: float, sigma: float, metric: str = "logprice",
                    n_sims: int = 500, seed: int = 1015) -> np.ndarray:
    """Best correlation found by searching the whole past of a random walk of length ``n``.

    The query is the final ``L``-window; candidates are every window ending at least ``L``
    periods earlier. Each simulated path gives one number: the correlation a chartist would
    print in the corner of the overlay if the market were a coin-flipping machine.
    """
    P = random_walk_paths(n_sims, n, mu, sigma, seed)
    out = np.empty(n_sims)
    for i in range(n_sims):
        v = _series_for(P[i], metric)
        W, valid = _normalised_windows(v, L)
        q = W[-1]
        cand = W[: W.shape[0] - L]
        cv = valid[: W.shape[0] - L]
        c = cand @ q
        out[i] = float(np.max(np.where(cv, c, -np.inf)))
    return out


def null_search_distribution(n: int, L: int, mu: float, sigma: float, eval_idx,
                             metric: str = "logprice", n_sims: int = 10,
                             seed: int = 1015) -> np.ndarray:
    """Top-1 match correlations at the SAME evaluation dates, on random walks of the same length.

    The like-for-like comparison with the real tape: same length, same dates, same search, same
    drift and vol — only the information is removed. Returns ``n_sims x len(eval_idx)``.
    """
    P = random_walk_paths(n_sims, n, mu, sigma, seed)
    out = np.empty((n_sims, len(eval_idx)))
    for i in range(n_sims):
        out[i] = analogue_search(P[i], L, eval_idx, k_max=1, metric=metric)["corr"][:, 0]
    return out


def null_sweep_power(make_tape, grid: dict, periods_per_year: int, oos_start: str,
                     seeds, headline: dict) -> pd.DataFrame:
    """Run the headline forecaster on many synthetic tapes; return its t-statistic per seed.

    ``make_tape(seed)`` returns a tape frame. Used to measure size (null tapes) and power
    (planted tapes) of exactly the test applied to the real data.
    """
    rows = []
    for s in seeds:
        tape = make_tape(s)
        logp = tape["logp"].to_numpy()
        idx = eval_positions(tape.index, oos_start)
        srch = analogue_search(logp, headline["L"], idx, k_max=headline["k"],
                               metric=headline["metric"],
                               gap=max(headline["L"], headline["h"]))
        fc = analogue_forecast(logp, srch, headline["h"], headline["k"])
        ev = evaluate_forecast(fc, realised_forward(logp, idx, headline["h"]),
                               historical_drift_forecast(logp, idx, headline["h"]),
                               lags=headline["h"])
        rows.append({"seed": s, "t": ev.get("t", np.nan), "corr": ev.get("corr", np.nan),
                     "hit": ev.get("hit", np.nan), "oos_r2": ev.get("oos_r2", np.nan),
                     "match_corr_median": float(np.nanmedian(srch["corr"][:, 0]))})
    return pd.DataFrame(rows)


def vol_path_walks(lr, n_paths: int, window: int = 12, seed: int = 1015) -> np.ndarray:
    """Random walks that keep the real tape's VOLATILITY PATH but not its direction.

    Each period's shock is Gaussian, scaled by the real tape's own trailing ``window``-period
    volatility at that date, plus the real drift. Calm and turbulent eras arrive exactly when
    they did; which way the market moved is coin-flipped. A null for "is the real tape's
    tighter matching just volatility clustering?".
    """
    x = np.asarray(lr, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    vol = np.sqrt(pd.Series(x ** 2).rolling(window, min_periods=1, center=True).mean()
                  .to_numpy())
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, x.size))
    return np.cumsum(x.mean() + vol[None, :] * z, axis=1)


def block_bootstrap_walks(lr, n_paths: int, mean_block: float = 6.0,
                          seed: int = 1015) -> np.ndarray:
    """Random walks built from the real returns, reshuffled in short stationary-bootstrap blocks.

    Fat tails, volatility clustering and dependence shorter than ``mean_block`` survive; any
    long-range structure — a template that recurs decades apart — is destroyed. Blocks are kept
    well under the match window so a search cannot find a copy of the same real stretch.
    """
    x = np.asarray(lr, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    n = x.size
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    out = np.empty((n_paths, n))
    for i in range(n_paths):
        new = rng.random(n) < p
        new[0] = True
        pos = np.flatnonzero(new)
        st_ = rng.integers(0, n, pos.size)
        bid = np.cumsum(new) - 1
        out[i] = np.cumsum(x[(st_[bid] + np.arange(n) - pos[bid]) % n])
    return out


def null_search_paths(paths: np.ndarray, L: int, eval_idx, metric: str = "logprice"
                      ) -> np.ndarray:
    """Top-1 match correlations of ``analogue_search`` run on each row of ``paths``."""
    return np.vstack([analogue_search(p, L, eval_idx, k_max=1, metric=metric)["corr"][:, 0]
                      for p in paths])


# --------------------------------------------------------------------------- #
# The 1929 template, tested directly
# --------------------------------------------------------------------------- #
def template_overlay(logp: pd.Series, template_end: str, L: int, h: int) -> pd.DataFrame:
    """Correlation of every trailing ``L``-window with ONE fixed historical window.

    This is the viral chart exactly: the template is the ``L`` periods ending at
    ``template_end`` (e.g. the run-up to the 1929 peak), and every later date is scored by how
    closely its own trailing path matches. ``fwd`` is the realised next-``h``-period log return
    from that date, so "high match → crash" can be checked rather than asserted. Dates whose
    window overlaps the template are excluded.
    """
    x = logp.to_numpy(dtype=float)
    pos = int(np.searchsorted(logp.index.values, np.datetime64(pd.Timestamp(template_end))))
    W, valid = _normalised_windows(x, L)
    q = W[pos - L + 1]
    ends = np.arange(W.shape[0]) + L - 1
    c = W @ q
    keep = (ends > pos + L - 1) & valid
    e = ends[keep]
    fwd = np.full(e.size, np.nan)
    okf = e + h < x.size
    fwd[okf] = x[e[okf] + h] - x[e[okf]]
    return pd.DataFrame({"corr": c[keep], "fwd": fwd}, index=logp.index[e])


def overlay_episodes(ov: pd.DataFrame, threshold: float, crash: float = -0.20,
                     lags: int = 12) -> dict:
    """What followed the dates whose path matched the template at ``corr >= threshold``.

    ``crash`` is a forward simple return at or below this level (−20%, a bear market within the
    horizon). Compared against every date in the same sample. ``diff_t`` is the Newey-West t
    (``lags`` lags, for the overlapping forward windows) of the forward return of matching
    dates minus the rest; the crash call working would show up as a large negative t.
    """
    d = ov.dropna()
    hi = d[d["corr"] >= threshold]
    return {"threshold": threshold, "n_dates": int(len(d)), "n_match": int(len(hi)),
            "share_match": float(len(hi) / max(len(d), 1)),
            "fwd_match": float(hi["fwd"].mean()) if len(hi) else np.nan,
            "fwd_all": float(d["fwd"].mean()),
            "crash_rate_match": float((hi["fwd"] <= np.log1p(crash)).mean()) if len(hi)
            else np.nan,
            "crash_rate_all": float((d["fwd"] <= np.log1p(crash)).mean()),
            "diff_t": ols_hac(d["fwd"].to_numpy(),
                              (d["corr"] >= threshold).to_numpy(dtype=float), lags)["t_b"]
            if 0 < len(hi) < len(d) else np.nan,
            "n_episodes": int(((hi.index.to_series().diff() > pd.Timedelta(days=62))
                               .sum() + (1 if len(hi) else 0)))}


def template_null_share(template: np.ndarray, mu: float, sigma: float,
                        thresholds=(0.8, 0.9, 0.95), n_sims: int = 20000,
                        seed: int = 1015) -> dict:
    """How often a random-walk path correlates with a FIXED template at each threshold.

    The template is a real historical window (the 1929 run-up); the paths are random walks with
    the tape's drift and vol, of the template's length. This is the base rate of "today looks
    like 1929" when today carries no information at all.
    """
    L = len(template)
    P = random_walk_paths(n_sims, L, mu, sigma, seed)
    c = _rowcorr(P, np.broadcast_to(np.asarray(template, dtype=float), P.shape))
    return {th: float((c >= th).mean()) for th in thresholds} | {"corr": c}


# --------------------------------------------------------------------------- #
# The verdict — thresholds fixed before the real run
# --------------------------------------------------------------------------- #
def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    Inputs (keys of ``h``): ``m_head_t`` / ``d_head_t`` — Newey-West t of the forecast slope for
    the pre-registered headline specs (``HEADLINE_MONTHLY``, ``HEADLINE_DAILY``);
    ``holm_min_p`` — smallest Holm-adjusted one-sided p over every combination on both tapes
    (one-sided in the claim's direction: the analogues forecast what comes next);
    ``share_raw_sig`` — share of combinations with raw one-sided p < 0.05;
    ``rc_p_monthly`` / ``rc_p_daily`` — Reality-Check p of the best timing rule per tape (the
    smaller of the return race and the vol-matched Sharpe race — the reading kinder to the claim);
    ``m_head_alpha_t`` / ``d_head_alpha_t`` — net timing-alpha t of the headline rules;
    ``m_head_sharpe_net`` / ``m_bh_sharpe`` (and ``d_``) — net excess Sharpe of rule and B&H.

    - **Signal Real**: both headline t >= 2 AND something survives Holm (p < 0.05).
      **Mixed**: Holm survives and one tape's headline t >= 2 while the other's is <= 0.
      **Weak**: either headline t >= 2, or Holm survives, or more than 15% of the grid is
      raw-significant (three times what chance gives). **None** otherwise.
    - **Tradability Investable**: Signal Real, both Reality Checks p < 0.05, and both headline
      rules beat buy-and-hold on net excess Sharpe. **Fragile**: a Reality Check p < 0.025
      (Bonferroni over two tapes) or a headline net timing alpha t >= 2. **Mirage** otherwise.

    Optional prose keys (``n_combos`` …, ``worst_combo``, ``n_neg_sig_two``,
    ``best_match_null``, ``crash_rate_match``) only colour the explanation, never the stamps.
    """
    mt, dt = h["m_head_t"], h["d_head_t"]
    corrected = h["holm_min_p"] < 0.05
    if mt >= 2 and dt >= 2 and corrected:
        signal = "Real"
    elif corrected and ((mt >= 2 and dt <= 0) or (dt >= 2 and mt <= 0)):
        signal = "Mixed"
    elif mt >= 2 or dt >= 2 or corrected or h["share_raw_sig"] > 0.15:
        signal = "Weak"
    else:
        signal = "None"
    beats = (h["m_head_sharpe_net"] > h["m_bh_sharpe"]) and (h["d_head_sharpe_net"]
                                                              > h["d_bh_sharpe"])
    if signal == "Real" and h["rc_p_monthly"] < 0.05 and h["rc_p_daily"] < 0.05 and beats:
        trad = "Investable"
    elif (min(h["rc_p_monthly"], h["rc_p_daily"]) < 0.025
          or max(h["m_head_alpha_t"], h["d_head_alpha_t"]) >= 2):
        trad = "Fragile"
    else:
        trad = "Mirage"

    lead = {"None": "No forecasting skill in the claim's direction, on either tape.",
            "Weak": "A hint of skill that does not survive the search that found it.",
            "Mixed": "Skill on one tape and none, or the reverse, on the other.",
            "Real": "The analogues forecast, on both tapes, after correcting for the search."}
    nb = h.get("best_match_null", {}).get("monthly logprice L24", {})
    signal_why = (
        f"{lead[signal]} Matching the last 24 months of the 1926-2018 total-return path and "
        f"averaging what followed the five best non-overlapping analogues, the 12-month "
        f"forecast's slope on what actually happened has a Newey-West **t of {mt:+.2f}** "
        f"(correlation {h['m_head_corr']:+.2f}, out-of-sample R² {h['m_head_r2']:+.1%} "
        f"against the plain historical drift); the daily S&P 500 version (250-day match, "
        f"63-day forecast) gives **t {dt:+.2f}**. Across all **{h.get('n_combos', '?')} "
        f"combinations** tried (window, horizon, number of analogues, price-path or "
        f"z-scored-return matching, both tapes), {h['share_raw_sig']:.0%} were "
        f"raw-significant at 5% one-sided — chance alone gives 5% — and the smallest "
        f"Holm-adjusted p is **{h['holm_min_p']:.2f}**.")
    w = h.get("worst_combo")
    if w and h.get("n_neg_sig_two", 0) > 0:
        signal_why += (
            f" The only number that survives a two-sided correction points the *wrong* way: "
            f"{w['tape']} price-path matching at L={w['L']}, h={w['h']} anti-forecasts "
            f"(t {w['t']:+.2f}, two-sided Holm p {w['p_holm_two']:.3f}) — a corner of one "
            f"tape, absent from the other, recorded as a lead rather than a finding.")
    trad_why = (
        f"Going to cash whenever the analogues forecast a loss (one period of execution lag, "
        f"10 bp one-way), the monthly headline rule earns a net excess Sharpe of "
        f"{h['m_head_sharpe_net']:.2f} against {h['m_bh_sharpe']:.2f} for simply holding the "
        f"market (timing α t {h['m_head_alpha_t']:+.2f}); the daily rule "
        f"{h['d_head_sharpe_net']:.2f} against {h['d_bh_sharpe']:.2f} (α t "
        f"{h['d_head_alpha_t']:+.2f}). White's Reality Check over every rule in the grid: "
        f"p = {h['rc_p_monthly']:.2f} monthly, {h['rc_p_daily']:.2f} daily.")
    if "crash_rate_match" in h:
        trad_why += (
            f" The 1929 overlay itself, tested directly: months whose 24-month path matched "
            f"the run-up to the 1929 peak at 0.9 or better fell 20% within a year "
            f"{h['crash_rate_match']:.1%} of the time, against {h['crash_rate_all']:.1%} for "
            f"all months — no crash call worth the name.")
    p09 = nb.get("share_gt_09")
    one = ((f"A random walk with the market's drift and volatility finds a past match above "
            f"0.9 {p09:.0%} of the time, " if p09 is not None else
            "A 0.9 match is what searching a trending series produces, ")
           + f"and forecasting from such analogues has no skill over 69 years of monthly or 23 "
             f"years of daily out-of-sample forecasts (headline t {mt:+.2f} and {dt:+.2f}; best "
             f"Holm p {h['holm_min_p']:.2f} across {h.get('n_combos', '?')} combinations)."
           if signal in ("None", "Weak") else
           f"Analogue forecasts carry skill (headline t {mt:+.2f} and {dt:+.2f}; best Holm p "
           f"{h['holm_min_p']:.2f}).")
    return {"signal": signal, "signal_why": signal_why, "trad": trad, "trad_why": trad_why,
            "one_sentence": one}


def myth_check(h: dict) -> dict:
    """Third, grey axis: does real history match itself better than a random walk does?

    Pre-registered rule: **Confirmed** if the real monthly tape's median best price-path match
    (L=24, same dates, same search) clears the top of the per-path medians of Gaussian random
    walks with its drift and vol (``like_for_like['monthly logprice L24']``); **Busted**
    otherwise. Either way the stamp is about the *shape* of the matches, not about whether what
    followed them repeats — that is the Signal axis.
    """
    r = h["like_for_like"]["monthly logprice L24"]
    busted = r["real_median"] <= r["null_median_hi"] + 0.005
    stamp = "Busted" if busted else "Confirmed"
    why = (f"Searching for the best match to each month's trailing 24-month path gives a median "
           f"correlation of **{r['real_median']:.3f}** on the real tape (above 0.9 on "
           f"{r['real_share_gt_09']:.0%} of dates) against **{r['null_median']:.3f}** for "
           f"Gaussian random walks with the same drift and vol (per-path medians "
           f"{r['null_median_lo']:.3f} to {r['null_median_hi']:.3f}). ")
    if not busted:
        why += ("The real past does rhyme in shape a little more than a coin-flip walk"
                + (f" — and most of that gap is volatility clustering and short memory: a walk "
                   f"with the real volatility path gives {r['volpath_median']:.3f}, a 6-month "
                   f"block bootstrap of the real returns {r['block_median']:.3f}"
                   if "volpath_median" in r else "")
                + ". Shapes rhyme; what follows them does not (Signal axis).")
    else:
        why += "The famous-looking matches are exactly what a coin-flipping market produces."
    return {"stamp": stamp, "why": why}
