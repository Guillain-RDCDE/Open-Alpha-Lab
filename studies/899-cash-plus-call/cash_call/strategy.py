"""The 90/10 "cash + call" overlay + inference for Study 899.

Bill **Gross's "90/10"** (and Bodie's "T-bills + calls"): keep ~90% in bills so capital is
roughly preserved and spend ~10% on **convex upside** via call options. Since listed-option
history is not free, the ~10% sleeve is a **documented proxy**: a rolling **1-year at-the-money
SPY call marked daily with Black–Scholes**. Each annual roll strikes a fresh ATM call
(``K = spot``), prices it off SPY's trailing realized vol and the ^IRX bill rate, and spends the
``prem_frac`` (10%) premium budget on as much call **notional** as that fair price affords — so
protection costs *more* in high-vol regimes (exactly when you want it). The other ``cash_w``
(90%) sits in BIL. At expiry the call pays ``notional · max(S_T/K − 1, 0)`` and the book rolls.

Three books race on the same tape, all measured **excess-of-cash** (minus BIL):

* **90/10** — the cash + call book above (the convex proxy).
* **buy-and-hold** — 100% SPY, the upside the premium is paid out of.
* **matched static mix** — a *constant-mix* book rebalanced daily to 90/10's realized **average
  delta-weight** (its effective average equity exposure), so it runs the same average risk and
  isolates the honest question: does the **convexity** of the call payoff beat statically holding
  the same average linear exposure? A spanning alpha (90/10-excess on static-excess, HAC *t* on
  the intercept) answers it — leverage/level-clean.

We put the daily excess-return *difference* through a Newey–West (HAC) *t*, interval the excess
Sharpe difference with a circular block bootstrap, cut the sample into eras, sweep costs and the
premium budget, tabulate crash-year drawdowns, and measure the **up/down capture asymmetry** (the
convexity signature). The Black–Scholes mark is a proxy for a real listed call — model risk and
the forgone dividend are named, never hidden. The synthetic control only proves the machinery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr   # standard-normal CDF ufunc (fast on scalars & arrays)

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Return helpers
# --------------------------------------------------------------------------- #
def to_returns(prices: pd.DataFrame, cols=("SPY", "BIL")) -> pd.DataFrame:
    """Simple daily total returns for the price columns (first row dropped).

    Only SPY / BIL are *prices*; the IRX column is a rate, carried through separately.
    """
    return prices[list(cols)].pct_change().dropna()


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def _ann_sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def _sortino(r: np.ndarray) -> float:
    """Annualised Sortino (downside-deviation) ratio of a return series."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    down = r[r < 0.0]
    dd = np.sqrt(np.mean(down * down)) if down.size else 0.0
    return float(r.mean() / dd * np.sqrt(TRADING_DAYS)) if dd > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Black–Scholes call price (the option-free proxy's marking model)
# --------------------------------------------------------------------------- #
def bs_call(S, K, tau, sigma, r):
    """Black–Scholes European **call** value (vectorized). ``tau`` in years, ``sigma`` / ``r``
    annualised. At ``tau <= 0`` (expiry) returns the intrinsic value ``max(S − K, 0)``."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    tau = np.asarray(tau, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    r = np.asarray(r, dtype=float)
    intrinsic = np.maximum(S - K, 0.0)
    live = tau > 1e-9
    vt = np.maximum(sigma, 1e-6) * np.sqrt(np.where(live, tau, 1.0))
    d1 = (np.log(np.where(S > 0, S, 1.0) / K) + (r + 0.5 * sigma * sigma) * tau) / vt
    d2 = d1 - vt
    val = S * ndtr(d1) - K * np.exp(-r * tau) * ndtr(d2)
    return np.where(live, val, intrinsic)


def bs_call_delta(S, K, tau, sigma, r):
    """Black–Scholes call **delta** ``Φ(d1)`` (vectorized); the intrinsic step at expiry."""
    S = np.asarray(S, dtype=float); K = np.asarray(K, dtype=float)
    tau = np.asarray(tau, dtype=float); sigma = np.asarray(sigma, dtype=float)
    r = np.asarray(r, dtype=float)
    live = tau > 1e-9
    vt = np.maximum(sigma, 1e-6) * np.sqrt(np.where(live, tau, 1.0))
    d1 = (np.log(np.where(S > 0, S, 1.0) / K) + (r + 0.5 * sigma * sigma) * tau) / vt
    return np.where(live, ndtr(d1), (S >= K).astype(float))


def realized_vol(spy_ret: np.ndarray, window: int = 63, floor: float = 0.07,
                 cap: float = 0.80) -> np.ndarray:
    """Point-in-time trailing annualised realized vol of SPY (past-only, expanding until full).

    Used both to price the option at each roll and to mark it daily. Clipped to a sane
    ``[floor, cap]`` band so an early tiny window or a vol spike can't produce a degenerate price.
    """
    s = pd.Series(spy_ret, dtype=float)
    rv = s.rolling(window, min_periods=10).std(ddof=1)
    rv = rv.fillna(s.expanding(min_periods=2).std(ddof=1)).fillna(0.01)
    rv = rv.to_numpy() * np.sqrt(TRADING_DAYS)
    return np.clip(rv, floor, cap)


# --------------------------------------------------------------------------- #
# The 90/10 engine — a single O(n) state recursion (path-dependent by nature)
# --------------------------------------------------------------------------- #
def _size_call(V, S, sig, r, tau0, prem_frac, notional_cap, prem_mult):
    """Size a fresh ATM call at a roll: insure ``notional_cap × NAV`` of underlying if it costs
    ``<= prem_frac`` of NAV, else spend the full ``prem_frac`` budget on as much notional as it
    buys. Returns ``(notional, premium)`` — notional in calls-on-one-share (equity $ = notional·S).
    """
    c1 = float(bs_call(S, S, tau0, sig, r)) * prem_mult     # premium of one ATM call (on 1 share)
    if c1 <= 0:
        return 0.0, 0.0
    notional_full = notional_cap * V / S                    # calls to insure notional_cap·V dollars
    prem_full = notional_full * c1
    if prem_full <= prem_frac * V:                          # cheap vol: full cover, spend < budget
        return notional_full, prem_full
    prem = prem_frac * V                                    # dear vol: budget-capped partial cover
    return prem / c1, prem


def ninety_ten_book(
    spy: np.ndarray,
    spy_ret: np.ndarray,
    bill_ret: np.ndarray,
    rate_ann: np.ndarray,
    prem_frac: float = 0.10,
    cash_w: float = 0.90,      # kept for signature symmetry; cash = NAV − premium spent
    roll_days: int = 252,
    vol_window: int = 63,
    cost_bps: float = 0.0,
    prem_mult: float = 1.0,
    notional_cap: float = 1.0,
) -> dict:
    """Simulate the daily-marked 90/10 cash + call book on aligned daily arrays.

    ``V`` starts at 1.0. On each **roll** (every ``roll_days``) the book buys a fresh **1-year ATM
    call** (``K = spot``) to insure up to ``notional_cap`` (default **100%**) of NAV in SPY
    upside, priced by Black–Scholes off the trailing realized vol and the ^IRX rate. If that costs
    at most ``prem_frac`` (**10%**) of NAV it takes full cover and banks the change in bills; if
    vol makes it dearer it spends the whole 10% budget on the notional that buys (partial cover).
    ``prem_mult`` scales the fair premium (a vol/skew-cost stress — ``>1`` makes protection
    dearer). The call is **marked daily** as spot, vol, rate and time-to-expiry move; the rest of
    NAV sits in BIL. At expiry the call pays intrinsic and the book rolls; ``cost_bps`` is charged
    on the notional turned over at each roll. Everything is numpy scalars — no pandas ``.loc``.

    Because cover is capped at ``notional_cap`` the effective equity exposure never *leverages*
    past full capital in calm markets — the faithful "insure your capital, rent the upside" profile
    (not a vol-target that levers up). Returns a dict of daily net returns, the effective
    **delta-weight** (option Δ-exposure / NAV), the equity / cash / option paths and per-roll
    turnover.
    """
    S = np.asarray(spy, dtype=float)
    rr = np.asarray(spy_ret, dtype=float)     # aligned to S, i.e. rr[t] is S[t]/S[t-1]-1
    cr = np.asarray(bill_ret, dtype=float)
    ra = np.asarray(rate_ann, dtype=float) / 100.0     # percent -> decimal
    ra = np.clip(ra, 0.0, 0.25)
    sig = realized_vol(rr, window=vol_window)
    n = S.shape[0]
    cost = cost_bps * 1e-4
    tau0 = roll_days / TRADING_DAYS

    net = np.zeros(n)
    dweight = np.zeros(n)
    equity = np.empty(n)
    cashpath = np.empty(n)
    optpath = np.empty(n)
    turnover = np.zeros(n)

    # --- establish the position at t = 0 (state known at inception; no look-ahead) ---
    V = 1.0
    notional, prem0 = _size_call(V, S[0], sig[0], ra[0], tau0, prem_frac, notional_cap, prem_mult)
    cash_val = V - prem0
    K = S[0]
    ttl = roll_days                     # days remaining to expiry
    opt_val = notional * float(bs_call(S[0], K, tau0, sig[0], ra[0]))
    equity[0] = V
    cashpath[0] = cash_val
    optpath[0] = opt_val
    dweight[0] = notional * float(bs_call_delta(S[0], K, tau0, sig[0], ra[0])) * S[0] / V

    for t in range(1, n):
        cash_val = cash_val * (1.0 + cr[t])
        ttl -= 1
        tau = ttl / TRADING_DAYS
        if ttl <= 0:
            opt_val = notional * max(S[t] - K, 0.0)          # expiry: intrinsic
            delta = 1.0 if S[t] >= K else 0.0
        else:
            opt_val = notional * float(bs_call(S[t], K, tau, sig[t], ra[t]))
            delta = float(bs_call_delta(S[t], K, tau, sig[t], ra[t]))
        V_new = cash_val + opt_val
        net[t] = V_new / V - 1.0 if V > 0 else 0.0
        dweight[t] = notional * delta * S[t] / V_new if V_new > 0 else 0.0
        equity[t] = V_new
        cashpath[t] = cash_val
        optpath[t] = opt_val
        V = V_new

        if ttl <= 0:                                         # roll into a fresh 1-yr ATM call
            # size the new cover on the pre-cost NAV to price the turnover, then charge cost.
            notional, prem = _size_call(V, S[t], sig[t], ra[t], tau0, prem_frac,
                                        notional_cap, prem_mult)
            turn = (opt_val + prem) / V if V > 0 else 0.0    # sell the expiring stub + buy new cover
            V = V * (1.0 - cost * turn)
            notional, prem = _size_call(V, S[t], sig[t], ra[t], tau0, prem_frac,
                                        notional_cap, prem_mult)
            cash_val = V - prem
            K = S[t]
            ttl = roll_days
            opt_val = notional * float(bs_call(S[t], K, tau0, sig[t], ra[t]))
            equity[t] = V
            cashpath[t] = cash_val
            optpath[t] = opt_val
            turnover[t] = turn

    return {
        "net": net,
        "dweight": dweight,
        "equity": equity,
        "cash": cashpath,
        "option": optpath,
        "turnover": turnover,
        "avg_weight": float(np.mean(dweight)),
        "n_rolls": int((turnover > 0).sum()),
        "turnover_ann": float(turnover.sum() / (n / TRADING_DAYS)) if n > 0 else 0.0,
    }


def constant_mix(risky_ret: np.ndarray, cash_ret: np.ndarray, weight: float) -> np.ndarray:
    """Daily net return of a constant-mix book rebalanced each day to a fixed ``weight``."""
    rr = np.asarray(risky_ret, dtype=float)
    cr = np.asarray(cash_ret, dtype=float)
    return weight * rr + (1.0 - weight) * cr


# --------------------------------------------------------------------------- #
# Capture asymmetry — the convexity signature
# --------------------------------------------------------------------------- #
def capture(book_ret: np.ndarray, spy_ret: np.ndarray) -> dict:
    """Up-capture / down-capture of a book vs SPY (the convexity fingerprint of a call sleeve).

    ``up`` = Σ book_ret over SPY-up days ÷ Σ SPY_ret over those days; ``down`` likewise on SPY-down
    days. A *convex* long-call sleeve captures more upside than downside ⇒ ``up − down > 0``; a
    *linear* book captures symmetrically ⇒ ``up ≈ down``.
    """
    b = np.asarray(book_ret, dtype=float)
    s = np.asarray(spy_ret, dtype=float)
    up = s > 0
    dn = s < 0
    up_cap = float(b[up].sum() / s[up].sum()) if s[up].sum() != 0 else float("nan")
    dn_cap = float(b[dn].sum() / s[dn].sum()) if s[dn].sum() != 0 else float("nan")
    return {"up_capture": up_cap, "down_capture": dn_cap, "asymmetry": up_cap - dn_cap}


# --------------------------------------------------------------------------- #
# Headline stats — excess-of-cash Sharpe, CAGR, vol, drawdown, Sortino
# --------------------------------------------------------------------------- #
def stats(net: pd.Series, cash: pd.Series | None = None) -> dict:
    """Headline stats. Sharpe/Sortino are **excess-of-cash** when ``cash`` is given; CAGR/vol/DD raw."""
    net = net.astype(float).dropna()
    equity = (1.0 + net).cumprod()
    years = len(net) / TRADING_DAYS
    cagr = (float(equity.iloc[-1] ** (1.0 / years) - 1.0)
            if years > 0 and equity.iloc[-1] > 0 else float("nan"))
    vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    ex = (net - cash.reindex(net.index).fillna(0.0)) if cash is not None else net
    return {
        "sharpe": _ann_sharpe(ex.to_numpy()),
        "sortino": _sortino(ex.to_numpy()),
        "cagr": cagr,
        "vol": vol,
        "max_dd": _max_drawdown(equity.to_numpy()),
        "final": float(equity.iloc[-1]),
        "n_days": int(len(net)),
    }


def calendar_year_returns(net: pd.Series) -> pd.Series:
    """Compounded calendar-year total return of a daily return series."""
    g = (1.0 + net.astype(float)).groupby(net.index.year).prod() - 1.0
    g.index.name = "year"
    return g.rename("year_return")


def calendar_year_drawdowns(net: pd.Series) -> pd.Series:
    """Worst intra-year drawdown of a daily return series, per calendar year."""
    net = net.astype(float)
    out = {}
    for yr, r in net.groupby(net.index.year):
        eq = (1.0 + r).cumprod().to_numpy()
        out[yr] = _max_drawdown(eq)
    s = pd.Series(out).rename("max_dd")
    s.index.name = "year"
    return s


# --------------------------------------------------------------------------- #
# Inference primitives (self-contained; mirror the desk's shared helpers)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int | None = None) -> float:
    """HAC (Newey–West, Bartlett kernel) *t* of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def spanning_alpha(
    managed_excess: np.ndarray,
    static_excess: np.ndarray,
    lags: int | None = None,
) -> dict:
    """Spanning test: regress the managed (90/10) excess return on the matched static-mix excess
    return; the intercept ``alpha`` is the risk-adjusted pickup that *survives* matching the static
    book's average beta — the **leverage/level-clean** read on whether the call sleeve's
    **convexity** (its dynamic, spot-dependent delta) earns anything beyond statically holding the
    same average linear exposure. HAC (Newey–West) *t* on the intercept.
    """
    y = np.asarray(managed_excess, dtype=float)
    x = np.asarray(static_excess, dtype=float)
    m = np.isfinite(y) & np.isfinite(x)
    y, x = y[m], x[m]
    n = y.size
    if n < 20:
        return {"alpha_daily": float("nan"), "alpha_ann": float("nan"),
                "beta": float("nan"), "t_alpha": float("nan"), "n": n}
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ (X.T @ y)
    resid = y - X @ coef
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    S = (X * resid[:, None]).T @ (X * resid[:, None]) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        Xu = X * resid[:, None]
        G = Xu[k:].T @ Xu[:-k] / n
        S += w * (G + G.T)
    cov = XtX_inv @ (n * S) @ XtX_inv
    se_alpha = float(np.sqrt(max(cov[0, 0], 0.0)))
    alpha = float(coef[0])
    return {
        "alpha_daily": alpha,
        "alpha_ann": alpha * TRADING_DAYS,
        "beta": float(coef[1]),
        "t_alpha": float(alpha / se_alpha) if se_alpha > 0 else float("nan"),
        "n": n,
    }


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def bootstrap_sharpe_diff(
    a: pd.Series,
    b: pd.Series,
    cash: pd.Series | None = None,
    block: int = 21,
    n_boot: int = 2000,
    seed: int = 899,
) -> dict:
    """Circular block bootstrap CI for the excess-of-cash Sharpe **difference** (``a`` − ``b``).

    Resamples the two aligned excess-return series **jointly** in circular blocks (so the
    cross-correlation and the volatility clustering survive), recomputes each arm's annualised
    Sharpe per resample, and returns the point difference, a 95% percentile CI, and the bootstrap
    fraction of resamples in which ``a`` beats ``b``.
    """
    idx = a.dropna().index.intersection(b.dropna().index)
    ra = a.reindex(idx).to_numpy(dtype=float)
    rb = b.reindex(idx).to_numpy(dtype=float)
    if cash is not None:
        f = cash.reindex(idx).fillna(0.0).to_numpy(dtype=float)
        ra, rb = ra - f, rb - f
    n = len(idx)
    point = _ann_sharpe(ra) - _ann_sharpe(rb)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    a_wins = 0
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        offsets = (starts[:, None] + np.arange(block)[None, :]) % n
        sel = offsets.reshape(-1)[:n]
        sa, sb = _ann_sharpe(ra[sel]), _ann_sharpe(rb[sel])
        diffs[i] = sa - sb
        if sa > sb:
            a_wins += 1
    fin = diffs[np.isfinite(diffs)]
    lo, hi = np.percentile(fin, [2.5, 97.5])
    return {
        "point": float(point),
        "ci95": (float(lo), float(hi)),
        "frac_a_wins": a_wins / n_boot,
        "n": n, "block": block, "n_boot": n_boot,
    }


# --------------------------------------------------------------------------- #
# The full race — 90/10 vs buy-and-hold vs matched static mix
# --------------------------------------------------------------------------- #
def race(
    prices: pd.DataFrame,
    prem_frac: float = 0.10,
    cash_w: float = 0.90,
    roll_days: int = 252,
    vol_window: int = 63,
    cost_bps: float = 0.0,
    prem_mult: float = 1.0,
) -> dict:
    """90/10 vs buy-and-hold vs a matched-average-exposure static mix, all excess-of-cash.

    Takes a price frame with columns ``SPY`` / ``BIL`` (total-return levels) and ``IRX`` (rate in
    percent). Runs the 90/10 engine, builds the 100%-SPY buy-and-hold book and a constant-mix book
    at 90/10's realized average delta-weight, and reports each arm's stats (excess Sharpe / Sortino
    / CAGR / vol / maxDD), the excess-Sharpe advantages, the convexity spanning alpha (90/10 on the
    matched static), the HAC *t* on the daily 90/10−static excess-return difference, the up/down
    capture asymmetry, and the average delta-weight & turnover.
    """
    spy = prices["SPY"].to_numpy(dtype=float)
    ret = to_returns(prices)
    idx = ret.index
    rr = ret["SPY"].to_numpy(dtype=float)
    cr = ret["BIL"].to_numpy(dtype=float)
    irx = prices["IRX"].reindex(ret.index).to_numpy(dtype=float)
    spy_al = prices["SPY"].reindex(ret.index).to_numpy(dtype=float)

    book = ninety_ten_book(spy_al, rr, cr, irx, prem_frac=prem_frac, cash_w=cash_w,
                           roll_days=roll_days, vol_window=vol_window, cost_bps=cost_bps,
                           prem_mult=prem_mult)
    tt_net = pd.Series(book["net"], index=idx, name="ninety_ten")
    bh_net = pd.Series(rr, index=idx, name="buy_hold")
    avg_w = book["avg_weight"]
    static_net = pd.Series(constant_mix(rr, cr, avg_w), index=idx, name="static_mix")
    cash = ret["BIL"]

    s_tt = stats(tt_net, cash)
    s_bh = stats(bh_net, cash)
    s_static = stats(static_net, cash)

    ex_tt = (tt_net - cash).to_numpy(dtype=float)
    ex_static = (static_net - cash).to_numpy(dtype=float)
    span = spanning_alpha(ex_tt, ex_static)
    cap = capture(book["net"], rr)

    return {
        "prem_frac": float(prem_frac),
        "ninety_ten": s_tt,
        "buy_hold": s_bh,
        "static": s_static,
        "sharpe_vs_bh": float(s_tt["sharpe"] - s_bh["sharpe"]),
        "sharpe_vs_static": float(s_tt["sharpe"] - s_static["sharpe"]),
        "sortino_vs_bh": float(s_tt["sortino"] - s_bh["sortino"]),
        "dd_tt": s_tt["max_dd"],
        "dd_bh": s_bh["max_dd"],
        "dd_static": s_static["max_dd"],
        "alpha_ann": span["alpha_ann"],
        "alpha_beta": span["beta"],
        "t_alpha": span["t_alpha"],
        "diff_t_nw": newey_west_t(ex_tt - ex_static),
        "diff_t_1s": one_sample_t(ex_tt - ex_static),
        "up_capture": cap["up_capture"],
        "down_capture": cap["down_capture"],
        "capture_asym": cap["asymmetry"],
        "avg_weight": avg_w,
        "min_weight": float(book["dweight"].min()),
        "max_weight": float(book["dweight"].max()),
        "turnover_ann": book["turnover_ann"],
        "n_rolls": book["n_rolls"],
        "n_days": int(len(idx)),
        "tt_net": tt_net, "bh_net": bh_net, "static_net": static_net,
        "cash_net": cash, "dweight": pd.Series(book["dweight"], index=idx),
        "equity": pd.Series(book["equity"], index=idx),
    }


def cost_sweep(
    prices: pd.DataFrame,
    one_way_bps=(0.0, 5.0, 10.0, 25.0, 50.0),
    prem_frac: float = 0.10,
    cash_w: float = 0.90,
    roll_days: int = 252,
    vol_window: int = 63,
) -> pd.DataFrame:
    """Excess Sharpe of 90/10 (vs buy-and-hold and vs the matched static) per one-way roll cost (bps).

    Each row charges ``one_way`` bps on the notional turned over at each annual roll (options carry
    real bid/ask). The drawdown column shows the capital-protection function surviving cost.
    """
    rows = {}
    for c in one_way_bps:
        r = race(prices, prem_frac=prem_frac, cash_w=cash_w, roll_days=roll_days,
                 vol_window=vol_window, cost_bps=c)
        rows[c] = {
            "tt_sharpe": r["ninety_ten"]["sharpe"],
            "bh_sharpe": r["buy_hold"]["sharpe"],
            "static_sharpe": r["static"]["sharpe"],
            "sharpe_vs_bh": r["sharpe_vs_bh"],
            "sharpe_vs_static": r["sharpe_vs_static"],
            "dd_tt": r["dd_tt"],
        }
    out = pd.DataFrame(rows).T
    out.index.name = "one_way_bps"
    return out


def premium_sweep(
    prices: pd.DataFrame,
    prem_mults=(0.8, 1.0, 1.25, 1.5, 2.0),
    prem_frac: float = 0.10,
    cash_w: float = 0.90,
    roll_days: int = 252,
    vol_window: int = 63,
) -> pd.DataFrame:
    """90/10 Sharpe / drawdown / capture vs the option-cost multiplier ``prem_mult`` (the vol/skew
    stress). The Black–Scholes ATM price is a *floor* on what a real listed call costs (skew, bid/ask
    and demand push it up), so ``prem_mult > 1`` is the honest range; each extra unit of premium buys
    less notional and thins the convexity."""
    rows = {}
    for pm in prem_mults:
        r = race(prices, prem_frac=prem_frac, cash_w=cash_w, roll_days=roll_days,
                 vol_window=vol_window, prem_mult=pm)
        rows[pm] = {
            "tt_sharpe": r["ninety_ten"]["sharpe"],
            "bh_sharpe": r["buy_hold"]["sharpe"],
            "sharpe_vs_bh": r["sharpe_vs_bh"],
            "avg_weight": r["avg_weight"],
            "dd_tt": r["dd_tt"],
            "capture_asym": r["capture_asym"],
        }
    out = pd.DataFrame(rows).T
    out.index.name = "prem_mult"
    return out


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof — never cited for the stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(
    frame: pd.DataFrame,
    prem_frac: float = 0.10,
    cash_w: float = 0.90,
    roll_days: int = 252,
    vol_window: int = 63,
) -> dict:
    """Run the headline race on a synthetic price frame (offline)."""
    r = race(frame, prem_frac=prem_frac, cash_w=cash_w, roll_days=roll_days, vol_window=vol_window)
    return {
        "sharpe_tt": r["ninety_ten"]["sharpe"],
        "sharpe_bh": r["buy_hold"]["sharpe"],
        "sharpe_static": r["static"]["sharpe"],
        "sharpe_vs_static": r["sharpe_vs_static"],
        "sharpe_vs_bh": r["sharpe_vs_bh"],
        "t_alpha": r["t_alpha"],
        "alpha_ann": r["alpha_ann"],
        "up_capture": r["up_capture"],
        "down_capture": r["down_capture"],
        "capture_asym": r["capture_asym"],
        "dd_tt": r["dd_tt"],
        "dd_bh": r["dd_bh"],
        "dd_protection": float(r["dd_tt"] - r["dd_bh"]),    # >0 ⇒ 90/10 drawdown is shallower
        "avg_weight": r["avg_weight"],
        "n_days": r["n_days"],
    }
