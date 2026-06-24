"""The strategy and its honest controls — Study 424 (Ease of Movement).

Richard Arms' Ease of Movement (EMV/EOM, 1977) is a volume-weighted momentum
oscillator. Each bar it asks "how far did the mid-price travel per unit of volume?":

    midpoint_move = (H_t + L_t)/2 - (H_{t-1} + L_{t-1})/2
    box_ratio     = (volume / scale) / (H_t - L_t)
    EMV_1(t)      = midpoint_move / box_ratio
    EOM(t)        = SMA_n( EMV_1 )            # n = 14 by default

A large positive EOM means price rose a lot on little volume — an "effortless" advance.
The folk rule sold on charting sites:

- EOM crosses above 0 → effortless up move → *go long*.
- EOM crosses below 0 → effortless down move → *go flat (or short)*.

We turn this into a **daily position series** and race its NET Sharpe against
buy-and-hold, on an **excess-of-cash** basis (both legs measured vs the risk-free
rate so a part-time-in-cash timing rule is compared fairly). Three honest arbiters:

1. **HAC (Newey-West) t-stat** on the timing strategy's daily excess returns and on the
   strategy-minus-benchmark difference — the inference-bar number.
2. **A sign-shuffle permutation placebo** — keep the realised position *durations* but
   randomise their signs, so the null is "the timing dates carry no directional info".
3. **Costs** charged one-way x NAV on every position change (shorts also pay borrow),
   with a break-even sweep.

We also race EOM against the two obvious simpler benchmarks an EOM advocate must beat —
an SMA(50/200)-style fast/slow cross and MACD(12/26/9) — so the "it's better" claim is
actually tested, not assumed.

No look-ahead: the indicator is computed on closes/HLV up to bar *t*; the position is
held over the return of bar *t+1*. Exactly one ``shift`` is applied (in
:func:`position_returns`), and the docstring states that effective lag explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


# ---------------------------------------------------------------------------
# Ease of Movement indicator
# ---------------------------------------------------------------------------
def ease_of_movement(
    bars: pd.DataFrame, period: int = 14, scale: float = 1.0e8
) -> pd.Series:
    """Richard Arms' Ease of Movement oscillator over a rolling ``period`` window.

    EMV_1(t) = [ (H_t+L_t)/2 - (H_{t-1}+L_{t-1})/2 ] / [ (volume/scale) / (H_t - L_t) ]
    EOM(t)   = SMA_period( EMV_1 )

    ``scale`` (Arms used 1e8 for stocks) only rescales the magnitude; the *sign* of EOM
    — all the timing rule uses — is invariant to it. Returns a Series indexed identically
    to ``bars``; the first ``period`` bars are NaN (one for the midpoint diff, the rest
    for the rolling mean warm-up). Zero-range bars (H==L) are treated as no information
    (EMV_1 = 0) to avoid division blow-ups.
    """
    hl_mid = (bars["high"] + bars["low"]) / 2.0
    midpoint_move = hl_mid.diff()
    rng = (bars["high"] - bars["low"]).replace(0.0, np.nan)
    box_ratio = (bars["volume"] / scale) / rng
    emv_1 = midpoint_move / box_ratio
    emv_1 = emv_1.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    eom = emv_1.rolling(period, min_periods=period).mean()
    # NaN out the warm-up explicitly (first `period` bars carry no valid signal)
    eom.iloc[:period] = np.nan
    return pd.Series(eom, index=bars.index, name="eom")


# ---------------------------------------------------------------------------
# Benchmark indicators (the simpler rules EOM must beat)
# ---------------------------------------------------------------------------
def sma_cross_signal(close: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """+1 when fast SMA > slow SMA, else 0 (the classic long/flat trend rule)."""
    f = close.rolling(fast, min_periods=fast).mean()
    s = close.rolling(slow, min_periods=slow).mean()
    sig = (f > s).astype(float)
    sig[f.isna() | s.isna()] = np.nan
    return sig.rename("sma_cross")


def macd_signal(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.Series:
    """+1 when MACD line > signal line, else 0 (long/flat MACD trend rule)."""
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig_line = macd.ewm(span=signal, adjust=False).mean()
    out = (macd > sig_line).astype(float)
    out.iloc[:slow] = np.nan
    return out.rename("macd")


# ---------------------------------------------------------------------------
# EOM timing rule → daily position series
# ---------------------------------------------------------------------------
def eom_position(
    eom: pd.Series, long_short: bool = False
) -> pd.Series:
    """Map the EOM oscillator to a daily target position in {0,1} or {-1,0,+1}.

    - ``long_short=False`` (default, long/flat): position = +1 when EOM > 0 else 0.
    - ``long_short=True``  (long/short):          position = sign(EOM)  (+1 / -1).

    The position is the *target* for bar *t* formed from EOM known at the close of *t*;
    :func:`position_returns` applies the single ``t -> t+1`` execution lag. NaN EOM
    (warm-up) yields 0 (flat) so no position is taken before the indicator is valid.
    """
    if long_short:
        pos = np.sign(eom)
    else:
        pos = (eom > 0).astype(float)
    pos = pd.Series(pos, index=eom.index, name="position")
    pos[eom.isna()] = 0.0
    return pos


# ---------------------------------------------------------------------------
# Returns engine — daily, excess-of-cash, one execution lag, costs
# ---------------------------------------------------------------------------
def position_returns(
    bars: pd.DataFrame,
    position: pd.Series,
    rf_annual: float = 0.0,
    cost_bps: float = 1.0,
    borrow_bps_annual: float = 50.0,
) -> pd.DataFrame:
    """Daily NET, excess-of-cash returns for a target ``position`` series.

    Execution lag (stated exactly): the target position known at the close of *t* earns
    the asset return of *t+1*. Implemented as ONE ``shift(1)`` on the position before it
    multiplies the asset return — no second hidden shift.

    - **Excess of cash.** The asset's daily excess return is ``r_asset - rf_daily``; the
      strategy holds the asset only when in-position, and earns the cash rate otherwise.
      Net of cash, an in-position day earns ``pos*(r_asset - rf_daily)`` and a flat day
      earns 0 — so the race against buy-and-hold is excess-vs-excess by construction.
    - **Costs.** A one-way cost ``cost_bps`` x |position change| x NAV is charged on every
      change in target (turnover is |Δpos|, counted once per leg per change).
    - **Borrow.** A short day (pos < 0) additionally pays ``borrow_bps_annual`` financing.

    Columns: ``r_asset_excess`` (buy-and-hold leg), ``r_strat_gross``, ``r_strat_net``.
    """
    close = bars["close"]
    r_asset = close.pct_change().fillna(0.0)
    rf_daily = rf_annual / TRADING_DAYS
    r_excess = r_asset - rf_daily

    pos_lag = position.shift(1).fillna(0.0)  # the single, documented execution lag
    gross = pos_lag * r_excess

    turnover = pos_lag.diff().abs().fillna(pos_lag.abs())
    cost = turnover * (cost_bps * 1e-4)
    borrow = (pos_lag < 0).astype(float) * (borrow_bps_annual * 1e-4 / TRADING_DAYS)

    net = gross - cost - borrow
    out = pd.DataFrame(
        {
            "r_asset_excess": r_excess,
            "r_strat_gross": gross,
            "r_strat_net": net,
            "position": pos_lag,
            "turnover": turnover,
        },
        index=bars.index,
    )
    return out


# ---------------------------------------------------------------------------
# Performance summary + HAC inference
# ---------------------------------------------------------------------------
def _hac_tstat(x: np.ndarray) -> float:
    """Newey-West HAC t-stat for the mean of a daily return series (vs zero)."""
    x = x[np.isfinite(x)]
    n = x.size
    if n < 10:
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


def annualised_sharpe(r: np.ndarray) -> float:
    """Annualised Sharpe of a daily (already-excess) return series."""
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def summarize(book: pd.DataFrame, col: str = "r_strat_net") -> dict:
    """Headline stats for one daily return column: annualised return, vol, Sharpe,
    HAC *t* (vs zero), max drawdown, exposure, and annualised turnover."""
    r = book[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 2:
        return {k: float("nan") for k in
                ["n_days", "ann_ret", "ann_vol", "sharpe", "tstat", "max_dd",
                 "exposure", "ann_turnover"]}
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak - 1.0).min()
    expo = float((book["position"].abs() > 0).mean()) if "position" in book else float("nan")
    turn = (float(book["turnover"].sum()) / (n / TRADING_DAYS)
            if "turnover" in book else float("nan"))
    return {
        "n_days": int(n),
        "ann_ret": float(r.mean() * TRADING_DAYS),
        "ann_vol": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "sharpe": annualised_sharpe(r),
        "tstat": _hac_tstat(r),
        "max_dd": float(dd),
        "exposure": expo,
        "ann_turnover": turn,
    }


# ---------------------------------------------------------------------------
# Sign-shuffle permutation placebo
# ---------------------------------------------------------------------------
def permutation_pvalue(
    bars: pd.DataFrame,
    position: pd.Series,
    n_perm: int = 2000,
    rf_annual: float = 0.0,
    cost_bps: float = 0.0,
    seed: int = 424,
    metric: str = "sharpe",
) -> dict:
    """Sign-shuffle placebo: is the timing rule's Sharpe better than chance?

    We keep the *episodes* (consecutive runs of constant position) exactly as the rule
    produced them but flip each episode's sign by a fair coin. This preserves the holding
    durations and the number of trades while destroying the claim that the timing dates
    carry directional information. The p-value is the fraction of shuffles whose Sharpe
    (gross, to isolate the signal from costs) meets or beats the real rule's.

    Returns ``{observed, p_value, null_mean, null_std}``.
    """
    rng = np.random.default_rng(seed)
    pos = position.fillna(0.0).to_numpy(dtype=float)

    # identify episodes (runs of identical position)
    change = np.r_[True, pos[1:] != pos[:-1]]
    episode_id = np.cumsum(change) - 1
    n_ep = int(episode_id.max()) + 1

    real_book = position_returns(bars, position, rf_annual=rf_annual,
                                 cost_bps=cost_bps, borrow_bps_annual=0.0)
    observed = summarize(real_book, "r_strat_gross")[metric]

    null = np.empty(n_perm)
    for j in range(n_perm):
        flips = rng.choice([-1.0, 1.0], size=n_ep)
        shuffled = pd.Series(pos * flips[episode_id], index=position.index)
        b = position_returns(bars, shuffled, rf_annual=rf_annual,
                             cost_bps=cost_bps, borrow_bps_annual=0.0)
        null[j] = summarize(b, "r_strat_gross")[metric]

    null = null[np.isfinite(null)]
    p = float((null >= observed).mean()) if null.size else float("nan")
    return {
        "observed": float(observed),
        "p_value": p,
        "null_mean": float(null.mean()) if null.size else float("nan"),
        "null_std": float(null.std(ddof=1)) if null.size > 1 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Strategy-vs-benchmark difference test
# ---------------------------------------------------------------------------
def diff_tstat(book_a: pd.DataFrame, book_b: pd.DataFrame,
               col: str = "r_strat_net") -> dict:
    """HAC t-stat of (strategy A - strategy B) daily returns; positive = A wins."""
    a = book_a[col].reindex(book_b.index).fillna(0.0).to_numpy(dtype=float)
    b = book_b[col].to_numpy(dtype=float)
    d = a - b
    sh = annualised_sharpe(d)
    return {"mean_ann": float(np.nanmean(d) * TRADING_DAYS),
            "tstat": _hac_tstat(d), "sharpe": sh}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_experiment(
    bars: pd.DataFrame,
    period: int = 14,
    long_short: bool = False,
    rf_annual: float = 0.0,
    cost_bps: float = 1.0,
    borrow_bps_annual: float = 50.0,
    n_perm: int = 2000,
    seed: int = 424,
) -> dict:
    """Run the full EOM timing teardown on one tape and return a results dict.

    Builds the EOM(period) signal, the long/flat (or long/short) position, the NET
    excess-of-cash book; races it against buy-and-hold and against the SMA-cross and
    MACD benchmarks; computes HAC t-stats, the difference tests, and the sign-shuffle
    permutation p-value. All on one documented execution lag.
    """
    eom = ease_of_movement(bars, period=period)
    pos = eom_position(eom, long_short=long_short)
    book = position_returns(bars, pos, rf_annual=rf_annual,
                            cost_bps=cost_bps, borrow_bps_annual=borrow_bps_annual)

    # buy-and-hold leg (always long the excess return)
    bh = book.copy()
    bh["r_strat_gross"] = bh["r_asset_excess"]
    bh["r_strat_net"] = bh["r_asset_excess"]
    bh["position"] = 1.0
    bh["turnover"] = 0.0

    # benchmarks
    sma = eom_position_from_signal(sma_cross_signal(bars["close"]), long_short)
    macd = eom_position_from_signal(macd_signal(bars["close"]), long_short)
    book_sma = position_returns(bars, sma, rf_annual=rf_annual, cost_bps=cost_bps,
                               borrow_bps_annual=borrow_bps_annual)
    book_macd = position_returns(bars, macd, rf_annual=rf_annual, cost_bps=cost_bps,
                                borrow_bps_annual=borrow_bps_annual)

    perm = permutation_pvalue(bars, pos, n_perm=n_perm, rf_annual=rf_annual,
                              cost_bps=0.0, seed=seed)

    return {
        "eom": summarize(book, "r_strat_net"),
        "eom_gross": summarize(book, "r_strat_gross"),
        "bh": summarize(bh, "r_strat_net"),
        "sma": summarize(book_sma, "r_strat_net"),
        "macd": summarize(book_macd, "r_strat_net"),
        "vs_bh": diff_tstat(book, bh),
        "vs_sma": diff_tstat(book, book_sma),
        "vs_macd": diff_tstat(book, book_macd),
        "perm": perm,
        "books": {"eom": book, "bh": bh, "sma": book_sma, "macd": book_macd},
    }


def eom_position_from_signal(sig: pd.Series, long_short: bool = False) -> pd.Series:
    """Turn a {0,1} (or NaN) benchmark signal into a position aligned with EOM's framing.

    Long/flat: pass the {0,1} signal through. Long/short: map 0 → -1, 1 → +1 (so the
    benchmark is on the same exposure footing as the EOM long/short rule).
    """
    s = sig.copy()
    if long_short:
        out = s.where(s.isna(), s * 2.0 - 1.0)
    else:
        out = s
    out = out.fillna(0.0)
    return pd.Series(out, index=sig.index, name="position")
