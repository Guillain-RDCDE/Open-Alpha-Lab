"""Strategy + inference for Study 890 — Sector Risk-Parity.

The recipe: on a quarterly grid, weight the GICS sector ETFs by **inverse volatility**
(or full **equal-risk-contribution**, ERC), using a trailing ``lookback``-day estimate
known at the close of the day *before* the rebalance (one documented lag, zero look-ahead),
hold for the quarter with realistic weight drift, and charge a one-way cost on the turnover
at each rebalance. Then race the resulting return stream against **cap-weight SPY**, both
sides measured **excess of cash** (minus BIL), on:

* the annualised **excess-of-cash Sharpe** (the risk-adjusted headline — an unlevered
  risk-parity book is expected to earn *less* than a tech-heavy cap-weight index, so the
  fair question is Sharpe and drawdown, not raw return);
* the **max drawdown** and a **calendar-year** table;
* a **paired block-bootstrap** CI on the *Sharpe difference* (RP − SPY) and a Newey-West
  (HAC) *t* on the mean daily excess-return difference;
* an **era cut**;
* a **costed** version (quarterly one-way cost × turnover) and a **levered-to-SPY-vol**
  version that pays financing on the borrowed exposure — the honest test of whether a
  better-Sharpe-but-lower-return book can be turned into a bankable edge.

Everything is vectorised (weights are recomputed only on the ~quarterly rebalance grid, then
drift daily via a segment cumulative-product; no per-date ``.loc`` loop). numpy / pandas /
scipy / statsmodels only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Weighting schemes (long-only, sum to 1)
# --------------------------------------------------------------------------- #
def inverse_vol_weights(vol: np.ndarray) -> np.ndarray:
    """Risk-parity (inverse-volatility) weights from a vector of asset vols → sum to 1."""
    vol = np.asarray(vol, dtype=float)
    iv = np.where(vol > 0, 1.0 / vol, 0.0)
    s = iv.sum()
    return iv / s if s > 0 else np.full_like(iv, 1.0 / len(iv))


def equal_weights(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def erc_weights(cov: np.ndarray, iters: int = 500, tol: float = 1e-10) -> np.ndarray:
    """Equal-Risk-Contribution (full risk-parity) weights via cyclical coordinate descent.

    Griveau-Billion, Richard & Roncalli (2013): with equal risk budgets ``b_i = 1/n`` the
    solution is the fixed point of the per-coordinate update

        w_i ← ( −α_i + sqrt(α_i² + 4 σ_ii · b_i) ) / (2 σ_ii),   α_i = Σ_{j≠i} w_j σ_ij,

    iterated over coordinates and renormalised (scale-invariant → the Lagrange multiplier
    cancels). Long-only, sums to 1. Falls back to inverse-vol if the covariance is
    degenerate.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    d = np.diag(cov).copy()
    if not np.all(np.isfinite(cov)) or np.any(d <= 0):
        return inverse_vol_weights(np.sqrt(np.abs(d)))
    b = np.full(n, 1.0 / n)
    w = inverse_vol_weights(np.sqrt(d))
    for _ in range(iters):
        w_prev = w.copy()
        for i in range(n):
            alpha = float(cov[i] @ w) - cov[i, i] * w[i]     # Σ_{j≠i} w_j σ_ij
            w[i] = (-alpha + np.sqrt(alpha * alpha + 4.0 * cov[i, i] * b[i])) / (2.0 * cov[i, i])
        s = w.sum()
        if s > 0:
            w = w / s
        if np.max(np.abs(w - w_prev)) < tol:
            break
    return w


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Per-asset share of total portfolio variance — equal across assets for a true ERC book."""
    w = np.asarray(w, dtype=float)
    cov = np.asarray(cov, dtype=float)
    mrc = cov @ w
    rc = w * mrc
    tot = rc.sum()
    return rc / tot if tot != 0 else rc


# --------------------------------------------------------------------------- #
# The rebalanced allocator (vectorised, with weight drift + turnover)
# --------------------------------------------------------------------------- #
def _rebalance_flags(index: pd.DatetimeIndex, freq: str = "Q") -> np.ndarray:
    """Boolean array: True on the first trading day of each ``freq`` period (Q/M/A)."""
    s = pd.Series(index, index=index)
    if freq == "Q":
        key = s.dt.year.astype(str) + "Q" + s.dt.quarter.astype(str)
    elif freq == "M":
        key = s.dt.year.astype(str) + "-" + s.dt.month.astype(str)
    elif freq == "A":
        key = s.dt.year.astype(str)
    else:
        raise ValueError("freq must be 'Q', 'M' or 'A'")
    return key.ne(key.shift()).to_numpy()


def allocate(
    sector_ret: pd.DataFrame,
    scheme: str = "invvol",
    lookback: int = 63,
    freq: str = "Q",
    lag: int = 1,
    cost_bps: float = 0.0,
) -> dict:
    """Daily return of a ``scheme``-weighted, ``freq``-rebalanced sector book.

    Weights are set on each rebalance date from the trailing ``lookback``-day covariance
    **known at the close of ``lag`` day(s) earlier** (zero look-ahead), then held with
    realistic daily drift until the next rebalance. ``cost_bps`` charges a one-way cost on
    the turnover (Σ|Δweight|) at each rebalance, debited from that day's return.

    ``scheme`` ∈ {``'invvol'`` (inverse-vol risk parity), ``'erc'`` (full equal-risk-
    contribution), ``'equal'`` (1/N)}. Returns a dict with the gross and net daily return
    series, the daily weight matrix, and the per-rebalance turnover.
    """
    R = sector_ret.to_numpy(dtype=float)
    idx = sector_ret.index
    n_days, n_assets = R.shape
    flags = _rebalance_flags(idx, freq)

    # Trailing covariance / vol estimator, shifted by `lag` so a rebalance on day t uses
    # only information known at the close of day t-lag (no look-ahead).
    cov_daily = sector_ret.rolling(lookback, min_periods=lookback).cov()
    vol_daily = sector_ret.rolling(lookback, min_periods=lookback).std(ddof=1)

    seg_id = np.cumsum(flags) - 1                      # segment index per day
    target = np.full((n_days, n_assets), np.nan)      # target weights on rebalance rows
    rebal_rows = np.where(flags)[0]
    for r in rebal_rows:
        src = r - lag
        if src < 0:
            continue
        d = idx[src]
        if scheme == "equal":
            w = equal_weights(n_assets)
        elif scheme == "invvol":
            v = vol_daily.loc[d].to_numpy()
            if not np.all(np.isfinite(v)):
                continue
            w = inverse_vol_weights(v)
        elif scheme == "erc":
            c = cov_daily.loc[d].to_numpy().reshape(n_assets, n_assets)
            if not np.all(np.isfinite(c)):
                continue
            w = erc_weights(c)
        else:
            raise ValueError("scheme must be 'invvol', 'erc' or 'equal'")
        target[r] = w

    # First live segment starts at the first rebalance row that got a target.
    live = ~np.isnan(target[:, 0])
    first = np.argmax(live) if live.any() else n_days
    if first >= n_days:
        empty = pd.Series(dtype=float, index=idx)
        return {"gross": empty, "net": empty, "weights": pd.DataFrame(index=idx, columns=sector_ret.columns),
                "turnover": pd.Series(dtype=float), "n_rebalances": 0}

    # Segment cumulative gross growth of each asset SINCE its segment start (for drift).
    # cg[t] = product of (1+R) from the segment's start up to (but excluding) day t.
    seg = seg_id.copy()
    onep = 1.0 + R
    W = np.zeros((n_days, n_assets))
    port = np.zeros(n_days)
    turnover = {}
    cur_target = None
    cg = np.ones(n_assets)                             # cumulative growth since segment start
    prev_seg = -1
    prev_end_w = None                                  # drifted weights at end of prior segment
    for t in range(first, n_days):
        if seg[t] != prev_seg:                         # entering a new segment
            new_target = target[t]
            if np.isnan(new_target[0]):
                new_target = cur_target if cur_target is not None else equal_weights(n_assets)
            # turnover vs the drifted weights carried in from the prior segment
            if prev_end_w is not None:
                turnover[idx[t]] = float(np.abs(new_target - prev_end_w).sum())
            else:
                turnover[idx[t]] = float(np.abs(new_target).sum())     # initial build
            cur_target = new_target
            cg = np.ones(n_assets)
            prev_seg = seg[t]
        w_t = cur_target * cg
        s = w_t.sum()
        w_t = w_t / s if s > 0 else cur_target
        W[t] = w_t
        port[t] = float(w_t @ R[t])
        cg = cg * onep[t]                              # grow into next day
        # drifted weights at the *end* of this day (start of next) — used for next turnover
        end_w = cur_target * cg
        es = end_w.sum()
        prev_end_w = end_w / es if es > 0 else cur_target

    gross = pd.Series(port, index=idx, name=scheme)
    gross.iloc[:first] = np.nan
    to = pd.Series(turnover).sort_index()
    # Debit one-way cost × turnover on each rebalance day.
    cost = pd.Series(0.0, index=idx)
    if cost_bps and len(to):
        cost.loc[to.index] = to.to_numpy() * (cost_bps / 1e4)
    net = gross - cost
    return {
        "gross": gross,
        "net": net,
        "weights": pd.DataFrame(W, index=idx, columns=sector_ret.columns).iloc[first:],
        "turnover": to,
        "n_rebalances": int(len(to)),
    }


# --------------------------------------------------------------------------- #
# Excess-of-cash performance stats
# --------------------------------------------------------------------------- #
def perf_stats(r: pd.Series, cash: pd.Series | float = 0.0) -> dict:
    """Annualised excess-of-cash performance: ann return, vol, Sharpe, max drawdown.

    ``r`` and ``cash`` are daily simple returns; the Sharpe is on the **excess** series
    ``r − cash``. The drawdown is on the *total* (not excess) equity curve.
    """
    r = pd.Series(r).astype(float).dropna()
    if len(r) < 3:
        return {k: np.nan for k in ("ann", "vol", "sharpe", "max_drawdown", "n")}
    if isinstance(cash, pd.Series):
        c = cash.reindex(r.index).fillna(0.0)
    else:
        c = pd.Series(cash, index=r.index)
    ex = r - c
    ann = float(r.mean() * TRADING_DAYS)
    vol = float(ex.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float(ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS)) if ex.std(ddof=1) > 0 else np.nan
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    return {"ann": ann, "vol": vol, "sharpe": sharpe, "max_drawdown": dd, "n": int(len(r))}


def calendar_year_table(strat: pd.Series, bench: pd.Series) -> pd.DataFrame:
    """Per-calendar-year total return of the strategy vs the benchmark (%)."""
    def by_year(s):
        s = pd.Series(s).dropna()
        return s.groupby(s.index.year).apply(lambda x: (1.0 + x).prod() - 1.0)
    tbl = pd.DataFrame({"strategy": by_year(strat) * 100, "SPY": by_year(bench) * 100})
    tbl["diff"] = tbl["strategy"] - tbl["SPY"]
    return tbl


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def annualized_sharpe(ex: np.ndarray) -> float:
    ex = np.asarray(ex, dtype=float)
    ex = ex[np.isfinite(ex)]
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def sharpe_diff_bootstrap(
    ex_strat: pd.Series,
    ex_bench: pd.Series,
    n_boot: int = 3000,
    alpha: float = 0.05,
    block: int | None = None,
    seed: int = 890,
) -> dict:
    """Paired circular-block-bootstrap CI for the **Sharpe difference** SR(strat) − SR(bench).

    Both inputs are daily **excess-of-cash** return series on a common index. Resampling
    keeps the two legs *paired* (same block indices for both) so cross-correlation is
    preserved; blocks preserve the serial dependence. Returns the point difference, the
    (1−alpha) percentile interval, and the share of resamples with a *negative* difference
    (a blunt one-sided p-value on "could the RP Sharpe advantage be ≤ 0?").
    """
    a = pd.Series(ex_strat).astype(float)
    b = pd.Series(ex_bench).astype(float)
    df = pd.concat([a, b], axis=1).dropna()
    A = df.iloc[:, 0].to_numpy()
    B = df.iloc[:, 1].to_numpy()
    n = len(A)
    if n < 10:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": n, "sr_strat": float("nan"),
                "sr_bench": float("nan")}
    blk = int(block) if block is not None else max(1, round(n ** (1.0 / 3.0)))
    blk = max(1, min(blk, n))
    rng = np.random.default_rng(seed)
    point = annualized_sharpe(A) - annualized_sharpe(B)
    n_blocks = int(np.ceil(n / blk))
    offsets = np.arange(blk)
    diffs = np.full(n_boot, np.nan)
    for i in range(n_boot):
        if blk == 1:
            idx = rng.integers(0, n, n)
        else:
            starts = rng.integers(0, n, n_blocks)
            idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        sa, sb = A[idx], B[idx]
        da, db = sa.std(ddof=1), sb.std(ddof=1)
        if da > 0 and db > 0:
            diffs[i] = sa.mean() / da * np.sqrt(TRADING_DAYS) - sb.mean() / db * np.sqrt(TRADING_DAYS)
    valid = diffs[np.isfinite(diffs)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "diff": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "frac_negative": float((valid < 0).mean()),
        "n_obs": int(n),
        "sr_strat": annualized_sharpe(A),
        "sr_bench": annualized_sharpe(B),
        "n_boot_valid": int(valid.size),
        "block": blk,
    }


# --------------------------------------------------------------------------- #
# The race + the costed / levered timer
# --------------------------------------------------------------------------- #
def race(
    sector_ret: pd.DataFrame,
    bench_ret: pd.Series,
    cash_ret: pd.Series,
    scheme: str = "invvol",
    lookback: int = 63,
    freq: str = "Q",
    cost_bps: float = 3.0,
    nw_lags: int = 10,
) -> dict:
    """Full excess-vs-excess race: the ``scheme`` sector book vs cap-weight SPY.

    Both legs are put on an excess-of-cash footing (minus ``cash_ret``). Returns the gross
    and net Sharpes / drawdowns, the Sharpe-difference bootstrap, the HAC *t* on the mean
    daily excess-return difference, and the per-rebalance turnover / cost drag.
    """
    alloc = allocate(sector_ret, scheme=scheme, lookback=lookback, freq=freq, cost_bps=cost_bps)
    gross, net = alloc["gross"], alloc["net"]
    idx = gross.dropna().index
    bench = bench_ret.reindex(idx)
    cash = cash_ret.reindex(idx).fillna(0.0)

    ex_gross = (gross.reindex(idx) - cash).dropna()
    ex_net = (net.reindex(idx) - cash).dropna()
    ex_bench = (bench - cash).dropna()
    common = ex_gross.index.intersection(ex_bench.index)

    sg = perf_stats(gross.reindex(common), cash)
    sn = perf_stats(net.reindex(common), cash)
    sb = perf_stats(bench.reindex(common), cash)

    bs_gross = sharpe_diff_bootstrap(ex_gross.reindex(common), ex_bench.reindex(common))
    bs_net = sharpe_diff_bootstrap(ex_net.reindex(common), ex_bench.reindex(common))
    diff_daily = (ex_net.reindex(common) - ex_bench.reindex(common)).to_numpy()

    ann_turnover = float(alloc["turnover"].sum() / (len(common) / TRADING_DAYS)) if len(common) else np.nan
    cost_drag_bps_yr = ann_turnover * cost_bps    # one-way cost × annual turnover, in bps/yr

    return {
        "scheme": scheme,
        "n_days": int(len(common)),
        "start": str(common.min().date()) if len(common) else None,
        "end": str(common.max().date()) if len(common) else None,
        "strat_gross": sg, "strat_net": sn, "bench": sb,
        "sharpe_diff_gross": bs_gross["diff"], "sharpe_diff_net": bs_net["diff"],
        "sr_strat_gross": sg["sharpe"], "sr_strat_net": sn["sharpe"], "sr_bench": sb["sharpe"],
        "ci_gross": (bs_gross["ci_low"], bs_gross["ci_high"]),
        "ci_net": (bs_net["ci_low"], bs_net["ci_high"]),
        "frac_neg_gross": bs_gross["frac_negative"], "frac_neg_net": bs_net["frac_negative"],
        "nw_t_diff": newey_west_t(diff_daily, nw_lags),
        "n_rebalances": alloc["n_rebalances"],
        "ann_turnover": ann_turnover,
        "cost_drag_bps_yr": cost_drag_bps_yr,
        "dd_strat": sn["max_drawdown"], "dd_bench": sb["max_drawdown"],
        "_ex_net": ex_net.reindex(common), "_ex_bench": ex_bench.reindex(common),
        "_gross": gross.reindex(common), "_net": net.reindex(common), "_bench": bench.reindex(common),
    }


def levered_to_bench_vol(
    ex_net: pd.Series,
    ex_bench: pd.Series,
    cash_ret: pd.Series,
    finance_spread_bps_yr: float = 60.0,
) -> dict:
    """Lever the (lower-vol) RP book up to SPY's vol and pay financing on the borrowed part.

    An unlevered risk-parity book earns *less* than cap-weight SPY even when its Sharpe is
    higher, because its vol is lower. To turn a Sharpe edge into a *return* edge you must
    lever it to the benchmark's volatility — which means borrowing ``(L−1)×NAV`` at roughly
    cash + ``finance_spread_bps_yr``. This charges that financing and re-measures.
    """
    ex_net = pd.Series(ex_net).dropna()
    ex_bench = pd.Series(ex_bench).reindex(ex_net.index)
    cash = pd.Series(cash_ret).reindex(ex_net.index).fillna(0.0)
    vol_s = ex_net.std(ddof=1)
    vol_b = ex_bench.std(ddof=1)
    L = float(vol_b / vol_s) if vol_s > 0 else 1.0
    fin_daily = (finance_spread_bps_yr / 1e4) / TRADING_DAYS
    # Levered excess = L × excess return of the book, minus financing on the (L−1) borrow.
    ex_lev = L * ex_net - (L - 1.0) * fin_daily
    total_lev = ex_lev + cash
    return {
        "leverage": L,
        "sharpe_lev": annualized_sharpe(ex_lev.to_numpy()),
        "sharpe_bench": annualized_sharpe(ex_bench.to_numpy()),
        "ann_lev_pct": float(ex_lev.mean() * TRADING_DAYS * 100),
        "ann_bench_pct": float(ex_bench.mean() * TRADING_DAYS * 100),
        "finance_drag_bps_yr": float((L - 1.0) * finance_spread_bps_yr),
        "max_dd_lev": float(((1 + total_lev).cumprod() / (1 + total_lev).cumprod().cummax() - 1).min()),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(world: dict, lookback: int = 63) -> dict:
    """Machinery check: does inverse-vol risk-parity out-Sharpe the concentrated cap-weight
    benchmark when (and only when) the assets' vols are dispersed?  Returns the excess-of-cash
    Sharpe advantage of the inverse-vol book over the cap-weight benchmark on a synthetic
    world (null ≈ 0 at ``vol_spread = 0``; positive when vols are dispersed)."""
    sr = world["sector_ret"]
    cash = world["cash_ret"]
    bench = world["bench_ret"]
    rp = allocate(sr, scheme="invvol", lookback=lookback, freq="Q", cost_bps=0.0)["gross"]
    idx = rp.dropna().index
    c = cash.reindex(idx).fillna(0.0)
    sr_rp = annualized_sharpe((rp.reindex(idx) - c).to_numpy())
    sr_bench = annualized_sharpe((bench.reindex(idx) - c).to_numpy())
    return {"sharpe_advantage": sr_rp - sr_bench, "sr_rp": sr_rp, "sr_bench": sr_bench,
            "n_days": int(len(idx))}
