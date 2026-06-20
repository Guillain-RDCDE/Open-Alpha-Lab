"""The "Don't fight the Fed" engine and its honest controls — Study 317.

The claim: stocks track the Fed's balance sheet — long in QE, get out (or short) in QT.
We turn that into a literal daily timing rule and pin it against buy-and-hold.

Three deliverables, all honest:

1. **Regime means.** Average daily SPY return on QE days vs QT days vs FLAT days, each
   with a HAC (Newey–West) *t*-stat and a circular-block-bootstrap CI on the
   **difference** QE − QT (the contrast the slogan rests on).
2. **The timing rule.** ``regime_timing_returns`` is long SPY on QE days, in cash on QT
   days (and, in the aggressive variant, *short* on QT days). The regime label is
   calendar-known (announced programme direction), but to be conservative-by-default we
   still apply **one** execution lag: the regime read at the close of *t* sizes the
   position that earns the return of *t+1* (one ``shift``, applied once, documented here).
3. **Costs & the race.** Costs are charged one-way × NAV on every change in position
   (turnover). Because the timing rule sits in cash part of the time, it is raced against
   buy-and-hold on an **excess-of-cash vs excess-of-cash** Sharpe — comparing a
   part-time-in-cash book to a fully-invested one on raw Sharpe would flatter it.

A synthetic positive control (``data.synthetic_daily(qe_edge>0)``) proves the engine can
bank a planted regime edge; the null (``qe_edge=0``) proves it does not invent one. A
synthetic number may never back a *real* stamp — that is the inference bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Returns & regime conditioning
# ---------------------------------------------------------------------------
def daily_returns(frame: pd.DataFrame) -> pd.Series:
    """Simple daily returns from the ``close`` column (price-only / split-only)."""
    return frame["close"].pct_change().rename("ret")


def regime_means(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean daily return, vol and count by regime, with a HAC *t* on each mean.

    Columns: ``n, mean_bps, vol_bps, sharpe_ann, tstat`` indexed by regime. The
    ``tstat`` is the Newey–West HAC *t* on the regime's mean return (H0: mean = 0).
    """
    ret = daily_returns(frame)
    reg = frame["regime"]
    df = pd.DataFrame({"ret": ret, "regime": reg}).dropna()
    rows = {}
    for r, grp in df.groupby("regime"):
        x = grp["ret"].to_numpy(dtype=float)
        n = x.size
        mu = x.mean() if n else np.nan
        sd = x.std(ddof=1) if n > 1 else np.nan
        rows[r] = {
            "n": int(n),
            "mean_bps": float(mu * 1e4) if n else np.nan,
            "vol_bps": float(sd * 1e4) if n > 1 else np.nan,
            "sharpe_ann": (float(mu / sd * np.sqrt(TRADING_DAYS_PER_YEAR))
                           if n > 1 and sd > 0 else np.nan),
            "tstat": hac_tstat(x),
        }
    out = pd.DataFrame(rows).T
    return out.reindex([r for r in ("QE", "FLAT", "QT") if r in out.index])


def hac_tstat(x: np.ndarray) -> float:
    """Newey–West HAC *t*-stat on the mean of ``x`` (H0: mean = 0).

    Self-contained (no hard dependency on quantlab being importable). Lag chosen by
    the standard ``floor(4 (n/100)^{2/9})`` rule.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def diff_block_bootstrap_ci(
    frame: pd.DataFrame,
    a: str = "QE",
    b: str = "QT",
    block: int = 21,
    n_boot: int = 2000,
    seed: int = 317,
    alpha: float = 0.05,
) -> dict:
    """Circular-block-bootstrap CI on the mean-return *difference* (regime ``a`` − ``b``).

    Resamples the full daily return series in circular blocks (preserving the volatility
    clustering an i.i.d. bootstrap would destroy), recomputing the QE-minus-QT mean
    contrast each time. Returns the point estimate, the (1-alpha) CI, and a bootstrap
    two-sided *p*-value for H0: difference = 0.
    """
    ret = daily_returns(frame)
    reg = frame["regime"]
    df = pd.DataFrame({"ret": ret, "regime": reg}).dropna().reset_index(drop=True)
    r = df["ret"].to_numpy(dtype=float)
    g = df["regime"].to_numpy()
    n = r.size

    def contrast(rr, gg):
        ma = rr[gg == a]
        mb = rr[gg == b]
        if ma.size == 0 or mb.size == 0:
            return np.nan
        return (ma.mean() - mb.mean()) * 1e4  # bps/day

    point = contrast(r, g)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    boots = np.empty(n_boot)
    for j in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        idx = idx[:n]
        boots[j] = contrast(r[idx], g[idx])
    boots = boots[np.isfinite(boots)]
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    p = float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return {"diff_bps": float(point), "ci_lo": lo, "ci_hi": hi,
            "p_boot": p, "n_boot": int(boots.size)}


# ---------------------------------------------------------------------------
# The timing rule
# ---------------------------------------------------------------------------
def regime_timing_returns(
    frame: pd.DataFrame,
    short_qt: bool = False,
    cost_bps: float = 0.0,
    lag: int = 1,
) -> pd.DataFrame:
    """The "don't fight the Fed" timing book vs buy-and-hold, net of costs.

    Position by regime: **+1 (long) in QE**, **0 (cash) in QT** — or **−1 (short) in QT**
    when ``short_qt=True`` — and **+1 (long) in FLAT** (the slogan only fights the Fed when
    it is *tightening*; FLAT is treated as benign, fully invested).

    Execution lag: the regime is calendar-known, but we still apply **one** ``shift(lag)``
    so the position held on day *t* is the one decided at the close of *t-1* — the desk's
    one-lag convention, applied exactly once. Costs are one-way × NAV charged on the
    absolute change in position (turnover), so a 0→1 entry or a 1→−1 flip pays accordingly.

    Returns a frame with columns ``ret`` (buy-and-hold daily), ``pos`` (the lagged
    position), ``timing_gross``, ``timing_net``.
    """
    ret = daily_returns(frame)
    reg = frame["regime"]
    target = reg.map({"QE": 1.0, "FLAT": 1.0, "QT": (-1.0 if short_qt else 0.0)})
    pos = target.shift(lag).fillna(0.0)  # one execution lag, applied once
    turn = pos.diff().abs().fillna(pos.abs())  # one-way turnover × NAV
    gross = pos * ret
    net = gross - turn * cost_bps * 1e-4
    out = pd.DataFrame({"ret": ret, "pos": pos,
                        "timing_gross": gross, "timing_net": net})
    return out.dropna(subset=["ret"])


def ann_stats(r: pd.Series) -> dict:
    """Annualised return, vol, Sharpe and HAC *t* for a daily return series."""
    x = r.dropna().to_numpy(dtype=float)
    n = x.size
    if n < 2:
        return {"ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan,
                "tstat": np.nan, "n": int(n)}
    mu, sd = x.mean(), x.std(ddof=1)
    return {
        "ann_ret": float((1 + mu) ** TRADING_DAYS_PER_YEAR - 1),
        "ann_vol": float(sd * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "sharpe": float(mu / sd * np.sqrt(TRADING_DAYS_PER_YEAR)) if sd > 0 else np.nan,
        "tstat": hac_tstat(x),
        "n": int(n),
    }


def race(
    frame: pd.DataFrame,
    short_qt: bool = False,
    cost_bps: float = 0.0,
    rf_annual: float = 0.0,
    lag: int = 1,
) -> dict:
    """Race the timing rule against buy-and-hold on **excess-of-cash** Sharpe.

    Because the timing book sits in cash part of the time and buy-and-hold never does,
    comparing raw Sharpes would flatter the part-time book. We subtract the per-day
    risk-free rate from *both* legs before computing Sharpe (excess-vs-excess). ``rf_annual``
    is the cash rate (default 0 keeps it simple and conservative — set it for a real race).

    Returns a dict of both legs' annualised stats plus the net-of-cost timing Sharpe and
    the difference in excess Sharpe (timing − buy&hold).
    """
    book = regime_timing_returns(frame, short_qt=short_qt, cost_bps=cost_bps, lag=lag)
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    bh_x = book["ret"] - rf_daily
    tg_x = book["timing_gross"] - rf_daily * book["pos"].abs().clip(upper=1.0)
    tn_x = book["timing_net"] - rf_daily * book["pos"].abs().clip(upper=1.0)
    bh = ann_stats(bh_x)
    tg = ann_stats(tg_x)
    tn = ann_stats(tn_x)
    return {
        "buy_hold": bh,
        "timing_gross": tg,
        "timing_net": tn,
        "excess_sharpe_diff": (tn["sharpe"] - bh["sharpe"]
                               if np.isfinite(tn["sharpe"]) and np.isfinite(bh["sharpe"])
                               else np.nan),
        "pct_in_market": float((book["pos"].abs() > 0).mean()),
        "turnover_per_yr": float(book["pos"].diff().abs().fillna(0).sum()
                                 / max(len(book) / TRADING_DAYS_PER_YEAR, 1e-9)),
    }
