"""The Detrended Price Oscillator and its honest arbiters — Study 425.

The **Detrended Price Oscillator** (DPO; standard charting definition, e.g. Achelis,
*Technical Analysis from A to Z*) strips the trend out of price so that whatever *cycle*
remains stands out::

    DPO(t) = Close(t - (n//2 + 1)) - SMA_n(Close)(t)

The displacement back by ``n//2 + 1`` bars lines the detrended series up under the *centre*
of the averaging window — which is exactly why the textbook DPO is **centered and peeks**:
``SMA_n(Close)(t)`` is the average of bars ``t-n+1 .. t``, so the value plotted "at"
``t - n//2 - 1`` uses bars *in its own future*.  Folklore: when the DPO turns up from a
trough it is an oversold cycle low → buy; when it turns down from a peak it is an overbought
cycle high → sell.  The claim is that detrending **isolates a tradable cycle** that a plain
moving average buries.

We test that claim two ways, with the look-ahead discipline made explicit:

- ``dpo_centered`` is the **textbook** indicator (peeks; for plotting / illustration only).
- ``dpo_tradable`` is the look-ahead-free version: ``Close(t) - SMA_n(Close)(t)``, i.e. the
  current detrended deviation, *no* forward displacement.  A timing rule may only use this.

The timing rule turns the DPO into a daily **long/flat** (and **long/short**) position:
go long when the DPO is below its lower band (oversold cycle low, expect a bounce) and
flat/short when above the upper band (overbought cycle high).  The position formed on the
close of *t* earns the return of *t+1* — one documented execution lag.  We then race the
strategy's **net** Sharpe against buy-and-hold, both **excess of cash**, with a HAC *t* on
the daily edge, a sign-permutation placebo, a cost sweep, and a head-to-head against the
obvious simpler benchmarks (SMA-cross and MACD).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# DPO indicator — centered (textbook, peeks) and tradable (no look-ahead)
# ---------------------------------------------------------------------------
def dpo_centered(close: pd.Series, period: int = 20) -> pd.Series:
    """Textbook DPO: ``Close(t - (period//2 + 1)) - SMA_period(Close)(t)``.

    The forward displacement makes this **peek**: do not use it to trade.  Provided for
    faithful illustration of how the indicator is drawn on a chart.
    """
    sma = close.rolling(period).mean()
    disp = period // 2 + 1
    return (close.shift(disp) - sma).rename("dpo_centered")


def dpo_tradable(close: pd.Series, period: int = 20) -> pd.Series:
    """Look-ahead-free DPO: ``Close(t) - SMA_period(Close)(t)``.

    The current detrended deviation — the oscillator value an honest trader could actually
    know at the close of bar ``t``.  This is what every timing rule below uses.
    """
    sma = close.rolling(period).mean()
    return (close - sma).rename("dpo")


# ---------------------------------------------------------------------------
# Timing rules — DPO band rule, and the simpler benchmarks
# ---------------------------------------------------------------------------
def dpo_position(
    close: pd.Series,
    period: int = 20,
    band_k: float = 1.0,
    long_short: bool = False,
) -> pd.Series:
    """Daily position in {0,1} (long/flat) or {-1,0,+1} (long/short) from the tradable DPO.

    Bands are ``+/- band_k * rolling_std(DPO, period)``.  The cycle-trading folk rule:

    - DPO below the lower band (deep oversold cycle low) → **long** (expect the cycle to
      turn up).
    - DPO above the upper band (overbought cycle high) → **flat** (long/flat) or **short**
      (long/short): expect the cycle to roll over.
    - In between → hold the previous position (a hysteresis band, so we are not whipsawed
      every bar).

    The position is formed from information known at the close of ``t``; the caller applies
    the one-bar execution lag (``book_returns``) so it earns ``r_{t+1}``.
    """
    dpo = dpo_tradable(close, period)
    band = dpo.rolling(period).std()
    lower = -band_k * band
    upper = band_k * band

    n = len(close)
    raw = np.zeros(n)
    state = 0
    dpo_v = dpo.to_numpy()
    lo_v = lower.to_numpy()
    hi_v = upper.to_numpy()
    for i in range(n):
        d, lo, hi = dpo_v[i], lo_v[i], hi_v[i]
        if np.isnan(d) or np.isnan(lo):
            raw[i] = 0
            continue
        if d < lo:
            state = 1
        elif d > hi:
            state = -1 if long_short else 0
        # else keep state (hysteresis)
        raw[i] = state
    return pd.Series(raw, index=close.index, name="pos")


def sma_cross_position(close: pd.Series, fast: int = 20, slow: int = 50,
                       long_short: bool = False) -> pd.Series:
    """Benchmark #1: classic SMA fast/slow crossover (trend-following), long/flat or long/short."""
    f = close.rolling(fast).mean()
    s = close.rolling(slow).mean()
    up = (f > s)
    pos = up.astype(float)
    if long_short:
        pos = up.astype(float) - (~up).astype(float)
    pos[s.isna()] = 0.0
    return pos.rename("pos")


def macd_position(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9,
                  long_short: bool = False) -> pd.Series:
    """Benchmark #2: MACD line vs signal line (the other detrending oscillator)."""
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    up = (macd > sig)
    pos = up.astype(float)
    if long_short:
        pos = up.astype(float) - (~up).astype(float)
    # warm-up: hold flat until both EMAs are meaningfully seeded
    pos.iloc[: slow] = 0.0
    return pos.rename("pos")


# ---------------------------------------------------------------------------
# Book the strategy — one execution lag, costs one-way x turnover, excess of cash
# ---------------------------------------------------------------------------
def book_returns(
    close: pd.Series,
    position: pd.Series,
    cost_bps: float = 1.0,
    rf_annual: float = 0.0,
    borrow_bps_annual: float = 50.0,
) -> pd.DataFrame:
    """Daily strategy P&L with one execution lag, costs, and excess-of-cash accounting.

    The position known at the close of ``t`` earns the simple return of ``t+1`` — applied as
    a single ``shift(1)``.  Costs are charged **one-way x |turnover|** (``|pos_t - pos_{t-1}|``
    each entry/exit leg counted), in bps of NAV.  When flat (long/flat book) the capital earns
    the daily risk-free rate; the short leg of a long/short book pays ``borrow_bps_annual``.

    Returns a frame with: ``mkt`` (asset return), ``pos`` (lagged position), ``gross``,
    ``cost``, ``net`` (strategy daily return), ``rf`` (daily cash), and ``edge`` =
    strategy-excess minus market-excess (the excess-vs-excess daily race we test).
    """
    mkt = close.pct_change().fillna(0.0)
    pos_lag = position.shift(1).fillna(0.0)
    rf_daily = rf_annual / TRADING_DAYS

    gross = pos_lag * mkt
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover.shift(1).fillna(0.0) * (cost_bps * 1e-4)

    # cash earned when not fully long; borrow paid on short exposure
    cash_weight = (1.0 - pos_lag).clip(lower=0.0)  # idle capital earns rf
    short_weight = (-pos_lag).clip(lower=0.0)      # short exposure pays borrow
    cash_pnl = cash_weight * rf_daily
    # borrow_bps_annual is an annual rate; convert to a daily charge on short exposure
    borrow_pnl = -short_weight * (borrow_bps_annual * 1e-4) / TRADING_DAYS

    net = gross - cost + cash_pnl + borrow_pnl

    strat_excess = net - rf_daily
    mkt_excess = mkt - rf_daily
    edge = strat_excess - mkt_excess

    return pd.DataFrame(
        {"mkt": mkt, "pos": pos_lag, "gross": gross, "cost": cost,
         "rf": rf_daily, "net": net, "strat_excess": strat_excess,
         "mkt_excess": mkt_excess, "edge": edge},
        index=close.index,
    )


# ---------------------------------------------------------------------------
# Honest arbiters
# ---------------------------------------------------------------------------
def sharpe(daily: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised Sharpe of a daily return series (excess returns in, Sharpe out)."""
    r = np.asarray(daily.dropna(), dtype=float)
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def hac_tstat(daily: pd.Series) -> float:
    """Newey-West HAC t-stat that the mean daily return differs from zero.

    Lag truncation follows the standard ``floor(4*(n/100)^(2/9))`` rule.  Returned on the
    series passed in — call it on ``edge`` (strategy-excess minus market-excess) for the
    excess-vs-excess race that decides the Signal stamp.
    """
    r = np.asarray(daily.dropna(), dtype=float)
    n = r.size
    if n < 6:
        return float("nan")
    mu = r.mean()
    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def permutation_pvalue(book: pd.DataFrame, n_perm: int = 2000, seed: int = 425) -> float:
    """Sign/shuffle placebo on the position vector — does the *timing* carry information?

    We hold the realised market returns fixed and randomly **circularly shift** the position
    series (preserving its autocorrelation / turnover) ``n_perm`` times, recomputing the mean
    daily edge each time.  The p-value is the fraction of shuffled timings whose mean edge is
    >= the real one.  A real cycle-timing edge sits in the right tail; noise sits in the bulk.
    """
    rng = np.random.default_rng(seed)
    mkt = book["mkt"].to_numpy()
    rf = book["rf"].to_numpy()
    pos = book["pos"].to_numpy()  # already lagged
    n = len(pos)
    real = float(np.nanmean(book["edge"].to_numpy()))
    cnt = 0
    for _ in range(n_perm):
        shift = int(rng.integers(1, n))
        p = np.roll(pos, shift)
        strat = p * mkt + (1.0 - np.clip(p, 0, None)) * rf - np.clip(-p, 0, None) * 0.0
        edge = (strat - rf) - (mkt - rf)
        if float(np.nanmean(edge)) >= real:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def summarize_book(book: pd.DataFrame) -> dict:
    """Headline stats for one booked run: Sharpes (strat vs market, excess), edge t, etc."""
    strat_sh = sharpe(book["strat_excess"])
    mkt_sh = sharpe(book["mkt_excess"])
    edge = book["edge"].dropna()
    ann_edge = float(edge.mean() * TRADING_DAYS)
    t = hac_tstat(edge)
    # time-in-market and turnover diagnostics
    expo = float(book["pos"].mean())
    n_trades = float((book["pos"].diff().abs() > 1e-9).sum())
    yrs = max(len(book) / TRADING_DAYS, 1e-9)
    return {
        "strat_sharpe": strat_sh,
        "mkt_sharpe": mkt_sh,
        "sharpe_delta": strat_sh - mkt_sh,
        "ann_edge": ann_edge,
        "edge_t": t,
        "exposure": expo,
        "trades_per_yr": n_trades / yrs,
        "n_days": int(len(book)),
    }


def run_experiment(
    close: pd.Series,
    period: int = 20,
    band_k: float = 1.0,
    cost_bps: float = 1.0,
    long_short: bool = False,
    rf_annual: float = 0.0,
    n_perm: int = 2000,
    seed: int = 425,
) -> dict:
    """Orchestrate the full DPO timing teardown on one close series.

    Builds the DPO long/flat (or long/short) book, races net excess Sharpe vs buy-and-hold,
    computes the HAC edge *t* and a permutation placebo, and books the two simpler benchmarks
    (SMA-cross, MACD) on the same tape for the "is DPO actually better?" comparison.
    """
    pos = dpo_position(close, period=period, band_k=band_k, long_short=long_short)
    book = book_returns(close, pos, cost_bps=cost_bps, rf_annual=rf_annual)
    s = summarize_book(book)
    s["perm_p"] = permutation_pvalue(book, n_perm=n_perm, seed=seed)

    # benchmarks on identical tape / cost / lag
    bench = {}
    for name, bpos in (
        ("sma", sma_cross_position(close, long_short=long_short)),
        ("macd", macd_position(close, long_short=long_short)),
    ):
        bb = book_returns(close, bpos, cost_bps=cost_bps, rf_annual=rf_annual)
        bs = summarize_book(bb)
        bench[name] = {"sharpe_delta": bs["sharpe_delta"], "edge_t": bs["edge_t"],
                       "ann_edge": bs["ann_edge"], "strat_sharpe": bs["strat_sharpe"]}
    s["bench"] = bench
    return s
