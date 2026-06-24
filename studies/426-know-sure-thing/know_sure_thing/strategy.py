"""The KST timing rule and its honest controls — Study 426 (Know Sure Thing).

Martin Pring's **Know Sure Thing** (KST, 1992) is a long-term momentum oscillator that
sums four smoothed rate-of-change (ROC) series at different lookbacks, weighted 1-2-3-4:

    ROC_i      = close / close.shift(roc_i) - 1            (i = 1..4)
    smoothed_i = SMA(ROC_i, sma_i)
    KST        = 1*smoothed_1 + 2*smoothed_2 + 3*smoothed_3 + 4*smoothed_4
    signal     = SMA(KST, sig_n)

Pring's daily parameters: ROC = (10, 15, 20, 30), SMA = (10, 10, 10, 15), signal = 9.

The folk rule, steelmanned: **go long when KST crosses above its signal line, go flat
(or short) when it crosses below** — a smoother, "less whipsaw-prone" trend filter than a
plain moving average.  We test it as a daily timing overlay and race it honestly:

- **NET Sharpe vs buy-and-hold**, excess-vs-excess (both sides excess of cash), so we are
  not paid for a part-time-in-cash artefact.
- **HAC (Newey-West) t-stat** on the strategy's *excess* daily returns (the inference bar).
- **A permutation / sign-flip placebo** — randomly flip the daily position signs and ask
  how often a random long/flat schedule with the same exposure beats the real one.
- **Costs one-way × NAV** charged on every change of position, with shorts paying borrow.
- **One documented execution lag** — signal known at close *t*, position earns return *t+1*.
- **Head-to-head vs the obvious simpler benchmarks** — SMA-200 long/flat and MACD(12,26,9)
  crossover — so the "KST is better" claim is actually tested, not assumed.

No look-ahead: every indicator is computed on closes up to bar *t*; the resulting position
is shifted one bar so it earns the return of *t+1* (one ``shift``, applied once).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Pring's daily KST defaults
ROC_PERIODS = (10, 15, 20, 30)
SMA_PERIODS = (10, 10, 10, 15)
SIGNAL_N = 9
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# KST indicator
# ---------------------------------------------------------------------------
def kst(
    close: pd.Series,
    roc_periods: tuple[int, ...] = ROC_PERIODS,
    sma_periods: tuple[int, ...] = SMA_PERIODS,
    signal_n: int = SIGNAL_N,
) -> pd.DataFrame:
    """Pring's Know Sure Thing and its signal line.

    Returns a DataFrame with columns ``kst`` and ``signal`` indexed like ``close``.  The
    KST sums four SMA-smoothed ROC series weighted 1..4 (longer lookbacks weighted more);
    the signal line is an ``signal_n``-period SMA of the KST.  Leading NaNs cover the
    warm-up (max ROC + max SMA + signal ≈ 50+ bars).
    """
    weights = (1, 2, 3, 4)
    kst_sum = pd.Series(0.0, index=close.index)
    for w, rp, sp in zip(weights, roc_periods, sma_periods):
        roc = close.pct_change(rp)
        smoothed = roc.rolling(sp, min_periods=sp).mean()
        kst_sum = kst_sum + w * smoothed
    sig = kst_sum.rolling(signal_n, min_periods=signal_n).mean()
    return pd.DataFrame({"kst": kst_sum, "signal": sig}, index=close.index)


# ---------------------------------------------------------------------------
# Position generators (one row per bar; +1 long, 0 flat, -1 short)
# ---------------------------------------------------------------------------
def kst_position(close: pd.Series, allow_short: bool = False, **kw) -> pd.Series:
    """KST/signal-line crossover position: long when KST > signal, else flat (or short).

    The state is the *level* relation (KST above/below its signal line), which is the
    standard implementation of the crossover rule — long while above, flat/short while
    below.  Position is the raw signal at close *t*; the one-bar execution lag is applied
    in :func:`book_returns`, not here.
    """
    k = kst(close, **kw)
    long_ = (k["kst"] > k["signal"]).astype(float)
    pos = long_ if not allow_short else (2.0 * long_ - 1.0)
    pos[k["signal"].isna()] = 0.0  # flat during warm-up
    return pos.rename("position")


def sma_position(close: pd.Series, window: int = 200, allow_short: bool = False) -> pd.Series:
    """SMA-N long/flat (Faber-style) benchmark: long when close > SMA(window)."""
    sma = close.rolling(window, min_periods=window).mean()
    long_ = (close > sma).astype(float)
    pos = long_ if not allow_short else (2.0 * long_ - 1.0)
    pos[sma.isna()] = 0.0
    return pos.rename("position")


def macd_position(
    close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9,
    allow_short: bool = False,
) -> pd.Series:
    """MACD(12,26,9) crossover benchmark: long when MACD line > signal line."""
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    signal = macd.ewm(span=sig, adjust=False).mean()
    long_ = (macd > signal).astype(float)
    pos = long_ if not allow_short else (2.0 * long_ - 1.0)
    pos.iloc[:slow] = 0.0  # warm-up
    return pos.rename("position")


# ---------------------------------------------------------------------------
# Backtest engine — one execution lag, costs one-way x NAV, borrow on shorts
# ---------------------------------------------------------------------------
def book_returns(
    close: pd.Series,
    position: pd.Series,
    cost_bps: float = 1.0,
    rf_annual: float = 0.0,
    borrow_bps_ann: float = 50.0,
) -> pd.DataFrame:
    """Daily strategy returns from a position series, with one execution lag and costs.

    Convention (the desk's one-shift rule): the position known at the **close of bar t**
    earns the **return of bar t+1**.  We apply exactly one ``shift(1)`` to the position.

    Costs: ``cost_bps`` one-way × NAV is charged on every change in |position| (turnover =
    one-way × NAV).  Shorts pay ``borrow_bps_ann`` annualised on the short notional.  ``rf_annual``
    is the cash rate earned on the *flat* portion (so excess-vs-excess races are fair).

    Returns a frame with columns: ``mkt`` (raw market return), ``pos`` (lagged position),
    ``gross`` (pos*mkt), ``cost``, ``borrow``, ``net`` (strategy net return), and
    ``net_excess`` (net minus the cash earned on un-invested capital at ``rf_annual``).
    """
    mkt = close.pct_change().fillna(0.0)
    pos = position.shift(1).fillna(0.0)        # one execution lag
    turn = pos.diff().abs().fillna(pos.abs())  # |position| change = one-way turnover
    cost = turn * (cost_bps * 1e-4)
    rf_daily = rf_annual / TRADING_DAYS
    borrow = (pos < 0).astype(float) * pos.abs() * (borrow_bps_ann * 1e-4 / TRADING_DAYS)

    gross = pos * mkt
    # cash earned on the un-invested fraction (1 - |pos|, floored at 0) at the rf rate
    cash_leg = np.clip(1.0 - pos.abs(), 0.0, 1.0) * rf_daily
    net = gross + cash_leg - cost - borrow
    net_excess = net - rf_daily                 # strategy excess-of-cash

    return pd.DataFrame(
        {"mkt": mkt, "pos": pos, "gross": gross, "cost": cost, "borrow": borrow,
         "net": net, "net_excess": net_excess},
        index=close.index,
    )


# ---------------------------------------------------------------------------
# Performance + inference
# ---------------------------------------------------------------------------
def ann_sharpe(daily_excess: np.ndarray) -> float:
    """Annualised Sharpe of a daily *excess* return series."""
    r = np.asarray(daily_excess, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def cagr(daily_net: np.ndarray) -> float:
    """Compound annual growth rate from a daily net (total) return series."""
    r = np.asarray(daily_net, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    growth = np.prod(1.0 + r)
    yrs = r.size / TRADING_DAYS
    return float(growth ** (1.0 / yrs) - 1.0) if yrs > 0 and growth > 0 else float("nan")


def max_drawdown(daily_net: np.ndarray) -> float:
    """Worst peak-to-trough drawdown of the equity curve (as a negative fraction)."""
    r = np.asarray(daily_net, dtype=float)
    r = r[np.isfinite(r)]
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min()) if eq.size else float("nan")


def hac_t(daily_excess: np.ndarray) -> float:
    """Newey-West HAC t-stat of the mean daily *excess* return against zero.

    This is the inference-bar number: the autocorrelation-robust t on the strategy's
    own excess returns.  A REAL Signal stamp needs |t| >= 2 here, on the real tape.
    """
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


def permutation_pvalue(
    book: pd.DataFrame, n_draws: int = 5000, seed: int = 426,
) -> dict:
    """Sign-flip placebo on the strategy's daily position.

    We keep the realised market returns and the realised *exposure schedule* fixed, but
    randomly **flip the sign** of each day's position (a fair coin per bar).  Each draw
    re-prices the book and records its annualised excess Sharpe.  ``p`` = share of random
    sign-schedules whose Sharpe matches or beats the real one — the honest answer to "could
    a coin with the same exposure profile have looked this good?".  Costs are ignored here
    (we test the *directional* information, like the engine's gross signal).
    """
    mkt = book["mkt"].to_numpy(dtype=float)
    pos = book["pos"].to_numpy(dtype=float)
    obs = ann_sharpe(pos * mkt)
    rng = np.random.default_rng(seed)
    sharpes = np.empty(n_draws)
    n = pos.size
    for i in range(n_draws):
        flips = rng.choice([-1.0, 1.0], size=n)
        sharpes[i] = ann_sharpe(flips * pos * mkt)
    p = float((sharpes >= obs).mean())
    return {"obs_sharpe": float(obs), "p_value": p, "draws": sharpes}


# ---------------------------------------------------------------------------
# One arm's full summary
# ---------------------------------------------------------------------------
def summarize_arm(book: pd.DataFrame) -> dict:
    """Headline stats for one timing arm's book."""
    exc = book["net_excess"].to_numpy(dtype=float)
    net = book["net"].to_numpy(dtype=float)
    pos = book["pos"].to_numpy(dtype=float)
    pos_changes = np.abs(np.diff(np.r_[0.0, pos]))
    n_trades = int((pos_changes > 1e-9).sum())
    return {
        "n_days": int(np.isfinite(exc).sum()),
        "sharpe_excess": ann_sharpe(exc),
        "cagr_net": cagr(net),
        "maxdd_net": max_drawdown(net),
        "hac_t": hac_t(exc),
        "exposure": float(np.nanmean(np.abs(pos))),
        "n_trades": n_trades,
        "ann_turnover": float(pos_changes.sum()
                              / max(np.isfinite(exc).sum() / TRADING_DAYS, 1e-9)),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_experiment(
    bars: pd.DataFrame,
    cost_bps: float = 1.0,
    rf_annual: float = 0.0,
    allow_short: bool = False,
    n_perm: int = 5000,
    seed: int = 426,
) -> dict:
    """Run the full KST teardown on a daily OHLCV frame.

    Builds the KST long/flat (or long/short) book and the two benchmark books (buy-and-hold,
    SMA-200, MACD), races their NET excess Sharpes, runs the HAC t and the sign-flip
    permutation placebo on the KST arm, and returns one dict of everything the notebooks
    and results.md quote.  Deterministic given ``seed``.
    """
    close = bars["close"]

    # KST arm
    kst_pos = kst_position(close, allow_short=allow_short)
    kst_book = book_returns(close, kst_pos, cost_bps=cost_bps, rf_annual=rf_annual)
    kst_stats = summarize_arm(kst_book)
    perm = permutation_pvalue(kst_book, n_draws=n_perm, seed=seed)

    # Buy-and-hold (always long, no timing cost beyond one entry)
    bah_pos = pd.Series(1.0, index=close.index)
    bah_pos.iloc[0] = 0.0
    bah_book = book_returns(close, bah_pos, cost_bps=cost_bps, rf_annual=rf_annual)
    bah_stats = summarize_arm(bah_book)

    # SMA-200 benchmark
    sma_pos = sma_position(close, window=200, allow_short=allow_short)
    sma_book = book_returns(close, sma_pos, cost_bps=cost_bps, rf_annual=rf_annual)
    sma_stats = summarize_arm(sma_book)

    # MACD benchmark
    macd_pos = macd_position(close, allow_short=allow_short)
    macd_book = book_returns(close, macd_pos, cost_bps=cost_bps, rf_annual=rf_annual)
    macd_stats = summarize_arm(macd_book)

    return {
        "kst": kst_stats,
        "buy_and_hold": bah_stats,
        "sma200": sma_stats,
        "macd": macd_stats,
        "perm": {"obs_sharpe": perm["obs_sharpe"], "p_value": perm["p_value"]},
        "cost_bps": cost_bps,
        "rf_annual": rf_annual,
        "allow_short": allow_short,
        "books": {"kst": kst_book, "buy_and_hold": bah_book,
                  "sma200": sma_book, "macd": macd_book},
    }


def cost_sweep(bars: pd.DataFrame, costs=(0.0, 1.0, 2.0, 5.0),
               rf_annual: float = 0.0) -> list[dict]:
    """KST net excess Sharpe vs buy-and-hold across a one-way cost ladder."""
    close = bars["close"]
    out = []
    for c in costs:
        kb = book_returns(close, kst_position(close), cost_bps=c, rf_annual=rf_annual)
        bah_pos = pd.Series(1.0, index=close.index); bah_pos.iloc[0] = 0.0
        bb = book_returns(close, bah_pos, cost_bps=c, rf_annual=rf_annual)
        out.append({
            "cost_bps": c,
            "kst_sharpe": ann_sharpe(kb["net_excess"].to_numpy()),
            "bah_sharpe": ann_sharpe(bb["net_excess"].to_numpy()),
        })
    return out
