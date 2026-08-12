"""The CPPI overlay + inference for Study 897 — CPPI Floor.

**Constant-Proportion Portfolio Insurance** (Black & Jones 1987; Perold & Sharpe 1988).
Fix a **floor** ``F`` (a fraction of NAV that accretes at the cash rate) and a **multiplier**
``m``. Each day hold

    exposure_t = clip(m · (V_{t−1} − F_{t−1}), 0, lev_cap · V_{t−1})

in the risky asset (SPY) and the remainder in cash (BIL). The cushion ``C = V − F`` drives
everything: as the market falls the cushion shrinks and the rule mechanically de-risks; if
the cushion hits zero the book is **cash-locked** on the floor. The exposure is set from the
state known at the **previous close** (one explicit lag — zero look-ahead), and the daily
turnover ``|Δw|`` is charged at ``cost_bps``. A ``band`` tolerance suppresses trades until the
drifted weight strays past it (the realistic whipsaw brake).

Three books race on the same tape, all measured **excess-of-cash** (minus BIL):

* **CPPI** — the insured book above.
* **buy-and-hold** — 100% SPY, the upside benchmark the insurance is paid out of.
* **matched static mix** — a *constant-mix* book rebalanced daily to CPPI's own realized
  **average** equity weight, so it runs the same average risk and isolates the honest
  question: does CPPI's dynamic **re-timing** beat just holding that average weight? A
  Perold–Sharpe spanning alpha (CPPI-excess on static-excess, HAC *t* on the intercept)
  answers it — leverage/level-clean.

We put the daily excess-return *difference* through a Newey–West (HAC) *t*, interval the
excess-Sharpe difference with a circular block bootstrap, cut the sample into eras, sweep
costs, tabulate crash-year drawdowns and floor breaches, and stress the **gap risk** (a
one-day crash breaches the floor iff it exceeds ``1/m``). The synthetic control only proves
the machinery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Return helpers
# --------------------------------------------------------------------------- #
def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily total returns from a price frame (first row dropped)."""
    return prices.pct_change().dropna()


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def _ann_sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


# --------------------------------------------------------------------------- #
# The CPPI engine — a single O(n) numpy state recursion (path-dependent by nature)
# --------------------------------------------------------------------------- #
def cppi_book(
    risky_ret: np.ndarray,
    cash_ret: np.ndarray,
    multiplier: float = 5.0,
    floor_frac: float = 0.80,
    lev_cap: float = 1.0,
    cost_bps: float = 0.0,
    band: float = 0.0,
) -> dict:
    """Simulate the CPPI book on aligned daily return arrays.

    ``V`` starts at 1.0; the floor ``F`` starts at ``floor_frac`` and accretes at the cash
    rate. On each day the target weight is ``w = clip(m·(V−F)/V, 0, lev_cap)`` computed from
    the state at the **previous close** (no look-ahead). A trade fires only when the drifted
    weight strays past ``band`` (``band=0`` ⇒ rebalance daily); ``cost_bps`` is charged on
    the traded ``|Δw|``. Returns a dict of daily net returns, applied weights, the equity /
    floor paths, the breach mask (``V < F``) and turnover — all numpy, no pandas ``.loc``.

    Gap risk is explicit: within a day the weight is fixed, so a single-day risky drop ``d``
    takes the cushion to ``V(1 − wd) − F`` and **breaches the floor iff ``d > 1/m``** (at
    full ``w = m·C/V``). Daily rebalancing cannot pre-empt an overnight gap.
    """
    rr = np.asarray(risky_ret, dtype=float)
    cr = np.asarray(cash_ret, dtype=float)
    n = rr.shape[0]
    cost = cost_bps * 1e-4

    net = np.empty(n)
    weight = np.empty(n)
    equity = np.empty(n)
    floorpath = np.empty(n)

    V = 1.0
    F = float(floor_frac)
    w_hold = 0.0            # weight carried into the day, drifted from the prior close
    for t in range(n):
        cushion = V - F
        if V > 0.0 and cushion > 0.0:
            w_tgt = min(multiplier * cushion / V, lev_cap)
        else:
            w_tgt = 0.0     # cushion exhausted -> cash-locked on the floor
        if t == 0 or abs(w_tgt - w_hold) > band:
            w_new = w_tgt
        else:
            w_new = w_hold
        turn = abs(w_new - w_hold)

        port_ret = w_new * rr[t] + (1.0 - w_new) * cr[t]
        V_next = V * (1.0 + port_ret - cost * turn)
        F_next = F * (1.0 + cr[t])

        gross = 1.0 + port_ret
        w_hold = (w_new * (1.0 + rr[t]) / gross) if gross > 0.0 else w_new

        net[t] = V_next / V - 1.0 if V > 0.0 else 0.0
        weight[t] = w_new
        equity[t] = V_next
        floorpath[t] = F_next
        V, F = V_next, F_next

    breach = equity < floorpath
    # A breach ONSET is a fresh crossing below the floor. Once the cushion is exhausted the
    # book is cash-locked and stays just below the (still-accreting) floor, so the raw count of
    # sub-floor days would balloon; the honest metric is the number of distinct breach events.
    onsets = int((breach & ~np.r_[False, breach[:-1]]).sum())
    return {
        "net": net,
        "weight": weight,
        "equity": equity,
        "floor": floorpath,
        "breach": breach,
        "turnover": np.abs(np.diff(weight, prepend=0.0)),
        "n_breach": onsets,
        "ever_breached": bool(breach.any()),
        "min_cushion": float((equity - floorpath).min()),
        "avg_weight": float(weight.mean()),
    }


def constant_mix(risky_ret: np.ndarray, cash_ret: np.ndarray, weight: float) -> np.ndarray:
    """Daily net return of a constant-mix book rebalanced each day to a fixed ``weight``."""
    rr = np.asarray(risky_ret, dtype=float)
    cr = np.asarray(cash_ret, dtype=float)
    return weight * rr + (1.0 - weight) * cr


# --------------------------------------------------------------------------- #
# Headline stats — excess-of-cash Sharpe, CAGR, vol, drawdown
# --------------------------------------------------------------------------- #
def stats(net: pd.Series, cash: pd.Series | None = None) -> dict:
    """Headline stats. Sharpe is **excess-of-cash** when ``cash`` is given; CAGR/vol/DD raw."""
    net = net.astype(float).dropna()
    equity = (1.0 + net).cumprod()
    years = len(net) / TRADING_DAYS
    cagr = (float(equity.iloc[-1] ** (1.0 / years) - 1.0)
            if years > 0 and equity.iloc[-1] > 0 else float("nan"))
    vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    ex = (net - cash.reindex(net.index).fillna(0.0)) if cash is not None else net
    return {
        "sharpe": _ann_sharpe(ex.to_numpy()),
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
    """Perold–Sharpe / Moreira–Muir spanning test: regress the managed (CPPI) excess return
    on the matched static-mix excess return; the intercept ``alpha`` is the risk-adjusted
    pickup that *survives* matching the static book's beta — the **leverage/level-clean**
    read on whether the dynamic **re-timing** earns anything.

    A raw mean-return difference is contaminated: CPPI runs a time-varying exposure, so a
    plain *t* on the return difference confounds the average-weight level with the re-timing
    skill. The spanning alpha nets out the static beta and asks the honest question — does
    re-timing risk beat statically holding the same average risk? HAC (Newey–West) *t* on
    the intercept.
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
    seed: int = 897,
) -> dict:
    """Circular block bootstrap CI for the excess-of-cash Sharpe **difference** (``a`` − ``b``).

    Resamples the two aligned excess-return series **jointly** in circular blocks (so the
    cross-correlation and the volatility clustering survive), recomputes each arm's annualised
    Sharpe per resample, and returns the point difference, a 95% percentile CI, and the
    bootstrap fraction of resamples in which ``a`` beats ``b``.
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
# The full race — CPPI vs buy-and-hold vs matched static mix
# --------------------------------------------------------------------------- #
def race(
    returns: pd.DataFrame,
    risky_col: str = "SPY",
    cash_col: str = "BIL",
    multiplier: float = 5.0,
    floor_frac: float = 0.80,
    lev_cap: float = 1.0,
    cost_bps: float = 0.0,
    band: float = 0.0,
) -> dict:
    """CPPI vs buy-and-hold vs a matched-average-exposure static mix, all excess-of-cash.

    Runs the CPPI engine, builds the 100%-SPY buy-and-hold book and a constant-mix book at
    CPPI's realized average weight, and reports each arm's stats (excess Sharpe / CAGR / vol /
    maxDD), the excess-Sharpe advantages, the leverage-clean re-timing spanning alpha (CPPI on
    the matched static), the HAC *t* on the daily CPPI−static excess-return difference, the
    floor-breach count / minimum cushion, and the average weight & turnover.
    """
    idx = returns.index
    rr = returns[risky_col].to_numpy(dtype=float)
    cr = returns[cash_col].to_numpy(dtype=float)

    book = cppi_book(rr, cr, multiplier, floor_frac, lev_cap, cost_bps=cost_bps, band=band)
    cppi_net = pd.Series(book["net"], index=idx, name="cppi")
    bh_net = pd.Series(rr, index=idx, name="buy_hold")
    avg_w = book["avg_weight"]
    static_net = pd.Series(constant_mix(rr, cr, avg_w), index=idx, name="static_mix")
    cash = returns[cash_col]

    s_cppi = stats(cppi_net, cash)
    s_bh = stats(bh_net, cash)
    s_static = stats(static_net, cash)

    ex_cppi = (cppi_net - cash).to_numpy(dtype=float)
    ex_bh = (bh_net - cash).to_numpy(dtype=float)
    ex_static = (static_net - cash).to_numpy(dtype=float)
    span = spanning_alpha(ex_cppi, ex_static)

    return {
        "multiplier": float(multiplier),
        "floor_frac": float(floor_frac),
        "cppi": s_cppi,
        "buy_hold": s_bh,
        "static": s_static,
        "sharpe_vs_bh": float(s_cppi["sharpe"] - s_bh["sharpe"]),
        "sharpe_vs_static": float(s_cppi["sharpe"] - s_static["sharpe"]),
        "dd_cppi": s_cppi["max_dd"],
        "dd_bh": s_bh["max_dd"],
        "dd_static": s_static["max_dd"],
        "alpha_ann": span["alpha_ann"],
        "alpha_beta": span["beta"],
        "t_alpha": span["t_alpha"],
        "diff_t_nw": newey_west_t(ex_cppi - ex_static),
        "diff_t_1s": one_sample_t(ex_cppi - ex_static),
        "avg_weight": avg_w,
        "min_weight": float(book["weight"].min()),
        "max_weight": float(book["weight"].max()),
        "turnover_ann": float(np.abs(np.diff(book["weight"], prepend=book["weight"][0])).mean()
                              * TRADING_DAYS),
        "n_breach": book["n_breach"],
        "min_cushion": book["min_cushion"],
        "breach_frac": float(book["breach"].mean()),
        "n_days": int(len(idx)),
        "cppi_net": cppi_net, "bh_net": bh_net, "static_net": static_net,
        "cash_net": cash, "weight": pd.Series(book["weight"], index=idx),
        "equity": pd.Series(book["equity"], index=idx),
        "floor": pd.Series(book["floor"], index=idx),
    }


def cost_sweep(
    returns: pd.DataFrame,
    one_way_bps=(0.0, 1.0, 2.0, 5.0, 10.0),
    risky_col: str = "SPY",
    cash_col: str = "BIL",
    multiplier: float = 5.0,
    floor_frac: float = 0.80,
    lev_cap: float = 1.0,
    band: float = 0.0,
) -> pd.DataFrame:
    """Excess Sharpe of CPPI (vs buy-and-hold and vs the matched static) per one-way cost (bps).

    Each row charges ``one_way`` bps on the CPPI overlay's daily ``|Δw|`` turnover — the honest
    tradability test for the *return* claim (the buy-and-hold benchmark has no overlay turnover
    to charge). The drawdown columns show the insurance function surviving cost untouched.
    """
    rows = {}
    for c in one_way_bps:
        r = race(returns, risky_col=risky_col, cash_col=cash_col, multiplier=multiplier,
                 floor_frac=floor_frac, lev_cap=lev_cap, cost_bps=c, band=band)
        rows[c] = {
            "cppi_sharpe": r["cppi"]["sharpe"],
            "bh_sharpe": r["buy_hold"]["sharpe"],
            "static_sharpe": r["static"]["sharpe"],
            "sharpe_vs_bh": r["sharpe_vs_bh"],
            "sharpe_vs_static": r["sharpe_vs_static"],
            "dd_cppi": r["dd_cppi"],
        }
    out = pd.DataFrame(rows).T
    out.index.name = "one_way_bps"
    return out


def multiplier_sweep(
    returns: pd.DataFrame,
    multipliers=(3.0, 4.0, 5.0, 6.0, 8.0),
    risky_col: str = "SPY",
    cash_col: str = "BIL",
    floor_frac: float = 0.80,
    lev_cap: float = 1.0,
) -> pd.DataFrame:
    """CPPI drawdown / breach / average-weight / Sharpe vs the multiplier ``m`` (gross).

    Larger ``m`` chases the floor more aggressively (more equity per unit of cushion) — a
    higher average exposure and a *lower* breach threshold: a one-day drop past ``1/m`` breaks
    the floor. This is the gap-risk knob, shown alongside the return trade-off.
    """
    rows = {}
    for m in multipliers:
        r = race(returns, risky_col=risky_col, cash_col=cash_col, multiplier=m,
                 floor_frac=floor_frac, lev_cap=lev_cap)
        rows[m] = {
            "breach_thresh": 1.0 / m,
            "avg_weight": r["avg_weight"],
            "cppi_sharpe": r["cppi"]["sharpe"],
            "dd_cppi": r["dd_cppi"],
            "n_breach": r["n_breach"],
            "turnover_ann": r["turnover_ann"],
        }
    out = pd.DataFrame(rows).T
    out.index.name = "multiplier"
    return out


def gap_stress(
    returns: pd.DataFrame,
    gap_sizes=(0.10, 0.15, 0.20, 0.25, 0.30),
    risky_col: str = "SPY",
    cash_col: str = "BIL",
    multiplier: float = 5.0,
    floor_frac: float = 0.80,
    lev_cap: float = 1.0,
) -> pd.DataFrame:
    """Gap-risk stress: inject a single overnight ``−gap`` crash at the worst-cushion day and
    measure whether the floor breaks.

    Daily rebalancing cannot pre-empt an overnight jump; a one-day drop ``d`` at full exposure
    ``w = m·C/V`` takes the book below the floor iff ``d > 1/m``. We run the real CPPI book,
    find the day of thinnest cushion, splice a ``−gap`` shock into the risky return on that
    day, and report the resulting floor breach — the honest read on the "guarantee".
    """
    rr = returns[risky_col].to_numpy(dtype=float)
    cr = returns[cash_col].to_numpy(dtype=float)
    base = cppi_book(rr, cr, multiplier, floor_frac, lev_cap)
    # thinnest-cushion day (most vulnerable to a gap)
    hit = int(np.argmin(base["equity"] - base["floor"]))
    rows = {}
    for g in gap_sizes:
        rr2 = rr.copy()
        rr2[hit] = (1.0 + rr2[hit]) * (1.0 - g) - 1.0   # compound the crash into that day
        b = cppi_book(rr2, cr, multiplier, floor_frac, lev_cap)
        rows[g] = {
            "breaches_floor": b["equity"][hit] < b["floor"][hit],
            "cushion_after": float(b["equity"][hit] - b["floor"][hit]),
            "equity_after": float(b["equity"][hit]),
            "floor_after": float(b["floor"][hit]),
        }
    out = pd.DataFrame(rows).T
    out.index.name = "gap_size"
    return out


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof — never cited for the stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(
    frame: pd.DataFrame,
    multiplier: float = 5.0,
    floor_frac: float = 0.80,
    lev_cap: float = 1.0,
) -> dict:
    """Run the headline race on a synthetic price frame (offline)."""
    ret = to_returns(frame)
    r = race(ret, multiplier=multiplier, floor_frac=floor_frac, lev_cap=lev_cap)
    return {
        "sharpe_cppi": r["cppi"]["sharpe"],
        "sharpe_bh": r["buy_hold"]["sharpe"],
        "sharpe_static": r["static"]["sharpe"],
        "sharpe_vs_static": r["sharpe_vs_static"],
        "sharpe_vs_bh": r["sharpe_vs_bh"],
        "t_alpha": r["t_alpha"],
        "alpha_ann": r["alpha_ann"],
        "dd_cppi": r["dd_cppi"],
        "dd_bh": r["dd_bh"],
        "dd_protection": float(r["dd_cppi"] - r["dd_bh"]),   # >0 ⇒ CPPI drawdown is shallower
        "n_breach": r["n_breach"],
        "avg_weight": r["avg_weight"],
        "n_days": r["n_days"],
    }
