"""The Special K indicator, its crossover claim, and its honest controls — Study 671.

Martin Pring introduced **Special K** (StockCharts ChartSchool; Pring, *Momentum Explained*
Vol. 2) as a "reduced-whipsaw" cousin of his own Know Sure Thing (KST): instead of summing
four smoothed rate-of-change (ROC) series, Special K sums **twelve**, spanning ROC lookbacks
from 10 to 530 trading days (roughly two weeks to two years), each smoothed by an SMA of the
same length and weighted 1-2-3-4 within four "bands" (short/short-medium/medium/long):

    ROC_i      = close / close.shift(p_i) - 1                       (i = 1..12)
    smoothed_i = SMA(ROC_i, p_i)                                     (same window as the ROC)
    SpecialK   = sum_i  w_i * smoothed_i                             (w cycles 1,2,3,4)
    signal     = SMA(SpecialK, 100)

Canonical daily periods (StockCharts ChartSchool "Special K"):
``p = (10, 15, 20, 30, 50, 65, 75, 100, 195, 265, 390, 530)``,
``w = (1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4)``, signal SMA = 100.

**The claim, steelmanned:** by blending short, medium and long cycles into one line, a
SpecialK/signal-line crossover is supposed to mark **major cyclic turns** in the market —
not just "trend is up/down" (that's what plain KST already claims) but specifically the
*big* turns, filtered of the whipsaw a shorter-window crossover would suffer.

We test that literally, three ways:

* **Event study.** Do trading days *after* a bullish (bearish) SpecialK/signal crossover earn
  abnormally high (low) returns versus baseline — Newey-West HAC *t* on a post-crossover
  daily-return dummy (lags = the event horizon, absorbing the overlap autocorrelation) — and
  does the mean forward return at real crossover dates beat a random-timing placebo of the
  same size (Coppock-style)?
* **Long/flat timer.** The crossover as a trading rule: NET excess-of-cash Sharpe / HAC *t*
  raced against buy-and-hold and a one-line 200-day SMA, with one execution lag and costs
  one-way x NAV.
* **Parameter robustness.** Scale every ROC/SMA period by a common factor (0.7x/1.0x/1.3x)
  and vary the signal-line window — does any of this survive outside Pring's exact numbers?

No look-ahead: every indicator value at bar *t* uses only closes up to *t*; the acted-on
position/window is shifted by one bar (the single documented execution lag).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# --------------------------------------------------------------------------- #
# Pring's Special K — canonical daily parameters (StockCharts ChartSchool)
# --------------------------------------------------------------------------- #
ROC_PERIODS = (10, 15, 20, 30, 50, 65, 75, 100, 195, 265, 390, 530)
SMA_PERIODS = ROC_PERIODS               # each ROC smoothed by an SMA of its own window
WEIGHTS = (1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4)
SIGNAL_N = 100

# Weekly-bar variant: same calendar lookback, periods scaled by /5 (rounded, min 1)
ROC_PERIODS_WEEKLY = tuple(max(1, round(p / 5)) for p in ROC_PERIODS)
SMA_PERIODS_WEEKLY = ROC_PERIODS_WEEKLY
SIGNAL_N_WEEKLY = max(1, round(SIGNAL_N / 5))


# --------------------------------------------------------------------------- #
# The indicator
# --------------------------------------------------------------------------- #
def special_k(close: pd.Series, roc_periods=ROC_PERIODS, sma_periods=SMA_PERIODS,
             weights=WEIGHTS, signal_n: int = SIGNAL_N) -> pd.DataFrame:
    """Pring's Special K and its signal line.

    Twelve SMA-smoothed ROC series, weighted 1..4 within four bands, summed; the signal
    line is a ``signal_n``-period SMA of Special K. Leading NaNs cover the long warm-up
    (max ROC + its own SMA + the signal SMA, ~1,160 bars for the daily defaults).
    """
    sk = pd.Series(0.0, index=close.index)
    for w, rp, sp in zip(weights, roc_periods, sma_periods):
        roc = close.pct_change(rp)
        smoothed = roc.rolling(sp, min_periods=sp).mean()
        sk = sk + w * smoothed
    sig = sk.rolling(signal_n, min_periods=signal_n).mean()
    return pd.DataFrame({"sk": sk, "signal": sig}, index=close.index)


def sk_position(close: pd.Series, allow_short: bool = False, **kw) -> pd.Series:
    """SpecialK/signal-line crossover position: long while SK > signal, else flat (or short).

    Raw signal at close *t*; the execution lag is applied in :func:`book_returns`.
    """
    k = special_k(close, **kw)
    long_ = (k["sk"] > k["signal"]).astype(float)
    pos = long_ if not allow_short else (2.0 * long_ - 1.0)
    pos[k["signal"].isna()] = 0.0
    return pos.rename("position")


def crossover_dates(close: pd.Series, **kw) -> dict[str, pd.DatetimeIndex]:
    """Dates of bullish (SK crosses above signal) and bearish (below) crossovers."""
    k = special_k(close, **kw)
    state = (k["sk"] > k["signal"]).astype(float)
    state[k["signal"].isna()] = np.nan
    chg = state.diff()
    bull = close.index[chg == 1.0]
    bear = close.index[chg == -1.0]
    return {"bull": bull, "bear": bear}


# --------------------------------------------------------------------------- #
# Event study — post-crossover forward returns vs unconditional
# --------------------------------------------------------------------------- #
def forward_log_return(close: pd.Series, horizon: int, lag: int = 1) -> pd.Series:
    """Forward log return earned entering ``lag`` bars after the signal date, held ``horizon``
    bars: ``log(close[t+lag+horizon]) - log(close[t+lag])``, indexed on the signal date *t*.
    """
    lc = np.log(close)
    fwd = lc.shift(-lag - horizon) - lc.shift(-lag)
    return fwd.rename(f"fwd_{horizon}")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances). NaN if either < 2 obs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def one_sample_t(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) < 2:
        return float("nan")
    se = a.std(ddof=1) / np.sqrt(len(a))
    return float(a.mean() / se) if se > 0 else float("nan")


def newey_west_t(y: np.ndarray, d: np.ndarray, lags: int) -> float:
    """HAC (Newey-West, Bartlett kernel) t of the slope in y = a + b*d.

    ``b`` is exactly the flagged-minus-baseline mean difference; ``lags`` should cover the
    autocorrelation induced by an overlapping event window (set it to the window length).
    """
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=float)
    keep = ~(np.isnan(y) | np.isnan(d))
    y, d = y[keep], d[keep]
    n = len(y)
    if n < 10:
        return float("nan")
    X = np.column_stack([np.ones(n), d])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    s = X * u[:, None]
    S = s.T @ s
    lags = max(1, min(lags, n - 2))
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = s[l:].T @ s[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(V[1, 1])
    return float(beta[1] / se) if se > 0 else float("nan")


def post_signal_flag(index: pd.DatetimeIndex, signal_dates: pd.DatetimeIndex,
                     horizon: int, lag: int = 1) -> pd.Series:
    """Boolean flag: True on bars in [t+lag, t+lag+horizon-1] sessions after any signal date
    ``t`` in ``signal_dates`` (offsets computed on ``index`` positions; OR'd across events)."""
    flag = pd.Series(False, index=index)
    pos_of = {d: i for i, d in enumerate(index)}
    n = len(index)
    for d in signal_dates:
        p = pos_of.get(d)
        if p is None:
            continue
        lo, hi = p + lag, p + lag + horizon - 1
        lo, hi = max(lo, 0), min(hi, n - 1)
        if lo <= hi:
            flag.iloc[lo:hi + 1] = True
    return flag


def regime_return_stats(close: pd.Series, signal_dates: pd.DatetimeIndex,
                        horizon: int, lag: int = 1) -> dict:
    """Mean daily log return in the post-signal window vs baseline, Newey-West t (lags=horizon).

    The primary Signal-axis statistic: uses daily granularity (many observations) and HAC
    lags equal to the window length, which absorbs the autocorrelation that overlapping
    horizon windows induce (the honest alternative to a naive event-level Welch t on
    overlapping forward returns).
    """
    ret = np.log(close).diff()
    flag = post_signal_flag(close.index, signal_dates, horizon, lag)
    y = ret.to_numpy()
    d = flag.to_numpy(dtype=float)
    keep = ~np.isnan(y)
    a, b = y[keep & flag.to_numpy()], y[keep & ~flag.to_numpy()]
    return {
        "n_events": len(signal_dates), "n_flag_days": int(flag.sum()),
        "mean_flag_pct": float(np.nanmean(a) * 100) if len(a) else float("nan"),
        "mean_base_pct": float(np.nanmean(b) * 100) if len(b) else float("nan"),
        "nw_t": newey_west_t(y, d, lags=horizon),
    }


def random_timing_test(close: pd.Series, signal_dates: pd.DatetimeIndex, horizon: int,
                       lag: int = 1, n_draws: int = 2000, seed: int = 671,
                       tail: str = "high") -> dict:
    """Coppock-style random-timing placebo: draw N random dates (same n as real signals),
    same horizon/lag forward return, compare the observed mean to the placebo distribution.

    ``tail="high"``: p = share of random draws whose mean forward return >= observed (tests
    whether real signal dates precede *unusually high* returns — the bullish claim).
    ``tail="low"``: p = share of random draws whose mean <= observed (the bearish claim).
    """
    fwd = forward_log_return(close, horizon, lag)
    universe = fwd.dropna()
    obs_vals = fwd.reindex(signal_dates).dropna()
    n = len(obs_vals)
    if n == 0 or len(universe) < n + 5:
        return {"obs_mean": float("nan"), "placebo_mean": float("nan"),
                "placebo_sd": float("nan"), "p_value": float("nan"), "n_events": n}
    obs_mean = float(obs_vals.mean())
    rng = np.random.default_rng(seed)
    pool = universe.to_numpy()
    draws = np.array([pool[rng.choice(len(pool), size=n, replace=False)].mean()
                      for _ in range(n_draws)])
    p = float((draws >= obs_mean).mean()) if tail == "high" else float((draws <= obs_mean).mean())
    return {"obs_mean": obs_mean, "placebo_mean": float(draws.mean()),
            "placebo_sd": float(draws.std(ddof=1)), "p_value": p, "n_events": n}


# --------------------------------------------------------------------------- #
# Long/flat timer — one execution lag, costs one-way x NAV, borrow on shorts
# --------------------------------------------------------------------------- #
def book_returns(close: pd.Series, position: pd.Series, cost_bps: float = 1.0,
                 rf_annual: float = 0.0, borrow_bps_ann: float = 50.0,
                 periods_per_year: int = TRADING_DAYS) -> pd.DataFrame:
    """Bar-by-bar strategy returns from a position series, with one execution lag and costs.

    Convention (the desk's one-shift rule): position known at the close of bar *t* earns the
    return of bar *t+1* — exactly one ``shift(1)``. Costs: ``cost_bps`` one-way x NAV on every
    change in |position| (turnover = one-way x NAV); shorts pay ``borrow_bps_ann`` annualised;
    ``rf_annual`` is earned on the flat portion (fair excess-vs-excess Sharpe races).
    ``periods_per_year`` lets the same engine price a weekly bar series (52) as well as daily
    (252, the default).
    """
    mkt = close.pct_change().fillna(0.0)
    pos = position.shift(1).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs())
    cost = turn * (cost_bps * 1e-4)
    rf_period = rf_annual / periods_per_year
    borrow = (pos < 0).astype(float) * pos.abs() * (borrow_bps_ann * 1e-4 / periods_per_year)
    gross = pos * mkt
    cash_leg = np.clip(1.0 - pos.abs(), 0.0, 1.0) * rf_period
    net = gross + cash_leg - cost - borrow
    net_excess = net - rf_period
    return pd.DataFrame({"mkt": mkt, "pos": pos, "gross": gross, "cost": cost,
                         "borrow": borrow, "net": net, "net_excess": net_excess},
                        index=close.index)


def ann_sharpe(daily_excess: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    r = np.asarray(daily_excess, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def cagr(daily_net: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    r = np.asarray(daily_net, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    growth = np.prod(1.0 + r)
    yrs = r.size / periods_per_year
    return float(growth ** (1.0 / yrs) - 1.0) if yrs > 0 and growth > 0 else float("nan")


def max_drawdown(daily_net: np.ndarray) -> float:
    r = np.asarray(daily_net, dtype=float)
    r = r[np.isfinite(r)]
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min()) if eq.size else float("nan")


def hac_t(daily_excess: np.ndarray) -> float:
    """Newey-West HAC t-stat of the mean daily excess return against zero (Andrews 1991
    plug-in lag). The inference-bar number for the timer: REAL needs |t| >= 2 here."""
    r = np.asarray(daily_excess, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n <= 5:
        return float("nan")
    mu = r.mean()
    e = r - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def permutation_pvalue(book: pd.DataFrame, n_draws: int = 5000, seed: int = 671) -> dict:
    """Sign-flip placebo on the timer's daily position (same spirit as sibling KST study)."""
    mkt = book["mkt"].to_numpy(dtype=float)
    pos = book["pos"].to_numpy(dtype=float)
    obs = ann_sharpe(pos * mkt)
    rng = np.random.default_rng(seed)
    n = pos.size
    sharpes = np.empty(n_draws)
    for i in range(n_draws):
        flips = rng.choice([-1.0, 1.0], size=n)
        sharpes[i] = ann_sharpe(flips * pos * mkt)
    return {"obs_sharpe": float(obs), "p_value": float((sharpes >= obs).mean()), "draws": sharpes}


def summarize_arm(book: pd.DataFrame, periods_per_year: int = TRADING_DAYS) -> dict:
    exc = book["net_excess"].to_numpy(dtype=float)
    net = book["net"].to_numpy(dtype=float)
    pos = book["pos"].to_numpy(dtype=float)
    pos_changes = np.abs(np.diff(np.r_[0.0, pos]))
    n_trades = int((pos_changes > 1e-9).sum())
    return {"n_days": int(np.isfinite(exc).sum()),
            "sharpe_excess": ann_sharpe(exc, periods_per_year),
            "cagr_net": cagr(net, periods_per_year), "maxdd_net": max_drawdown(net),
            "hac_t": hac_t(exc),
            "exposure": float(np.nanmean(np.abs(pos))), "n_trades": n_trades,
            "ann_turnover": float(pos_changes.sum()
                                  / max(np.isfinite(exc).sum() / periods_per_year, 1e-9))}


def sma_position(close: pd.Series, window: int = 200, allow_short: bool = False) -> pd.Series:
    sma = close.rolling(window, min_periods=window).mean()
    long_ = (close > sma).astype(float)
    pos = long_ if not allow_short else (2.0 * long_ - 1.0)
    pos[sma.isna()] = 0.0
    return pos.rename("position")


def run_experiment(close: pd.Series, cost_bps: float = 1.0, rf_annual: float = 0.0,
                   allow_short: bool = False, n_perm: int = 5000, seed: int = 671,
                   sk_kwargs: dict | None = None, periods_per_year: int = TRADING_DAYS
                   ) -> dict:
    """Full Special K timer teardown: SK arm vs buy-and-hold vs SMA-200, HAC t + permutation.

    ``periods_per_year`` lets the identical engine race the weekly-bar variant (52) as well
    as the daily default (252) — pass matching ``sk_kwargs`` (weekly-scaled periods) for that.
    """
    sk_kwargs = sk_kwargs or {}
    sk_pos = sk_position(close, allow_short=allow_short, **sk_kwargs)
    sk_book = book_returns(close, sk_pos, cost_bps=cost_bps, rf_annual=rf_annual,
                           periods_per_year=periods_per_year)
    sk_stats = summarize_arm(sk_book, periods_per_year)
    perm = permutation_pvalue(sk_book, n_draws=n_perm, seed=seed)

    bah_pos = pd.Series(1.0, index=close.index)
    bah_pos.iloc[0] = 0.0
    bah_book = book_returns(close, bah_pos, cost_bps=cost_bps, rf_annual=rf_annual,
                            periods_per_year=periods_per_year)
    bah_stats = summarize_arm(bah_book, periods_per_year)

    sma_pos = sma_position(close, window=200, allow_short=allow_short)
    sma_book = book_returns(close, sma_pos, cost_bps=cost_bps, rf_annual=rf_annual,
                            periods_per_year=periods_per_year)
    sma_stats = summarize_arm(sma_book, periods_per_year)

    return {"sk": sk_stats, "buy_and_hold": bah_stats, "sma200": sma_stats,
            "perm": {"obs_sharpe": perm["obs_sharpe"], "p_value": perm["p_value"]},
            "cost_bps": cost_bps, "rf_annual": rf_annual, "allow_short": allow_short,
            "books": {"sk": sk_book, "buy_and_hold": bah_book, "sma200": sma_book}}


def cost_sweep(close: pd.Series, costs=(0.0, 1.0, 2.0, 5.0), rf_annual: float = 0.0) -> list[dict]:
    out = []
    for c in costs:
        kb = book_returns(close, sk_position(close), cost_bps=c, rf_annual=rf_annual)
        bah_pos = pd.Series(1.0, index=close.index)
        bah_pos.iloc[0] = 0.0
        bb = book_returns(close, bah_pos, cost_bps=c, rf_annual=rf_annual)
        out.append({"cost_bps": c, "sk_sharpe": ann_sharpe(kb["net_excess"].to_numpy()),
                    "bah_sharpe": ann_sharpe(bb["net_excess"].to_numpy())})
    return out


# --------------------------------------------------------------------------- #
# Parameter robustness — scale every period by a common factor, and the signal window
# --------------------------------------------------------------------------- #
def param_robustness(close: pd.Series, scales=(0.7, 1.0, 1.3), cost_bps: float = 1.0) -> list[dict]:
    """SK-timer Sharpe/HAC t as every ROC/SMA period is scaled by a common factor.

    Scaling keeps the 1-2-3-4 weight structure and the four-band shape intact — this asks
    whether Pring's *exact* numbers are special, or any similarly-shaped multi-scale ROC
    blend works about as well (or as poorly).
    """
    out = []
    for s in scales:
        rp = tuple(max(2, round(p * s)) for p in ROC_PERIODS)
        sig_n = max(2, round(SIGNAL_N * s))
        pos = sk_position(close, roc_periods=rp, sma_periods=rp, signal_n=sig_n)
        book = book_returns(close, pos, cost_bps=cost_bps)
        stats = summarize_arm(book)
        out.append({"scale": s, "sharpe": stats["sharpe_excess"], "hac_t": stats["hac_t"],
                    "cagr": stats["cagr_net"], "n_trades": stats["n_trades"]})
    return out


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(close: pd.Series, horizon: int = 126, lag: int = 1) -> dict:
    """Run the headline regime-flag NW-t and the timer's value-add spread on a synthetic tape."""
    cross = crossover_dates(close)
    reg_bull = regime_return_stats(close, cross["bull"], horizon=horizon, lag=lag)
    res = run_experiment(close, cost_bps=1.0, n_perm=1)
    sk_net = res["books"]["sk"]["net"].to_numpy()
    bah_net = res["books"]["buy_and_hold"]["net"].to_numpy()
    spread = sk_net - bah_net
    return {"n_bull": len(cross["bull"]), "n_bear": len(cross["bear"]),
            "reg_nw_t": reg_bull["nw_t"], "sk_sharpe": res["sk"]["sharpe_excess"],
            "bah_sharpe": res["buy_and_hold"]["sharpe_excess"],
            "spread_ann_pct": float(np.nanmean(spread) * TRADING_DAYS * 100),
            "spread_hac_t": hac_t(spread)}
