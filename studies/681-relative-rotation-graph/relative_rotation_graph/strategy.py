"""Strategy + inference for Study 681 — Relative-Rotation-Graph (RRG).

The claim (Julius de Kempenaer, StockCharts / RRG Research, 2000s): plot each sector's
**RS-Ratio** (its relative-strength *level* vs the benchmark) against its **RS-Momentum**
(the *rate of change* of that relative strength) and you get four quadrants — a sector
rotates **clockwise** through them across a cycle:

* **Leading**  (RS-Ratio > 100, RS-Momentum > 100) — outperforming *and* accelerating.
* **Weakening** (RS-Ratio > 100, RS-Momentum < 100) — still outperforming, decelerating.
* **Lagging**  (RS-Ratio < 100, RS-Momentum < 100) — underperforming *and* decelerating.
* **Improving** (RS-Ratio < 100, RS-Momentum > 100) — underperforming, turning up.

The claimed cycle order (clockwise on the RS-Ratio x / RS-Momentum y plane) is
**Leading → Weakening → Lagging → Improving → Leading**.

The trading claim: buy sectors in the **Leading** quadrant, sell/avoid **Lagging** ones,
and the two-dimensional (level + momentum) read beats a plain one-dimensional momentum
rank because it separates *strong-and-still-rising* from *strong-but-rolling-over*.

Both axes are computed as rolling z-scores centred on 100 (our own explicit, documented
construction — RRG vendors don't publish their exact smoothing constants):

    RS_t          = price_sector_t / price_benchmark_t
    RS-Ratio_t    = 100 + (RS_t − rolling_mean(RS, W)) / rolling_std(RS, W)
    ROC_t         = RS-Ratio_t − RS-Ratio_{t-M}
    RS-Momentum_t = 100 + (ROC_t − rolling_mean(ROC, W)) / rolling_std(ROC, W)

with W = 63 trading days (~one quarter, the classic RRG "tail" scaled from weekly to
daily bars) and M = 21 trading days (~one month). One documented execution lag
throughout: the quadrant is read off the **month-end close**, the position is held over
the **following** month — no same-bar fill, no look-ahead.

Measurements:

* **The headline** — RRG "long the Leading quadrant" monthly rotation vs three controls:
  SPY buy-and-hold, an equal-weight all-sector basket, and a plain 6-1 top-3 trailing-
  momentum sort (the one-dimensional signal RRG claims to beat). Newey-West HAC *t* on
  each active-return series (RRG minus the control).
* **A matched-size random control** — each month, pick as many random sectors as RRG
  actually held that month: does the *quadrant selection* beat picking the same number of
  sectors blindly (rules out "RRG just happens to be levered/concentrated")?
* **Turnover & costs** — one-way cost x NAV per leg, charged on realised turnover.
* **Synthetic positive control** — proves the quadrant machinery lights up on a planted
  persistent relative-drift and stays null (|t| < 2 across seeds) with none planted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
TRADING_MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# The RRG axes
# --------------------------------------------------------------------------- #
def _rolling_zscore(x: pd.Series, window: int, center: float = 100.0) -> pd.Series:
    """100 + rolling z-score of ``x`` over ``window``; zero-variance windows -> NaN."""
    m = x.rolling(window, min_periods=window).mean()
    s = x.rolling(window, min_periods=window).std(ddof=1)
    s = s.where(s > 1e-12)
    return center + (x - m) / s


def rs_ratio(price_sector: pd.Series, price_bench: pd.Series, window: int) -> pd.Series:
    """JdK-style RS-Ratio: rolling z-score of the relative-strength line, centred on 100."""
    rs = price_sector / price_bench
    return _rolling_zscore(rs, window)


def rs_momentum(rs_ratio_series: pd.Series, mom_window: int, window: int) -> pd.Series:
    """JdK-style RS-Momentum: rolling z-score of the RS-Ratio's ``mom_window``-day change."""
    roc = rs_ratio_series.diff(mom_window)
    return _rolling_zscore(roc, window)


def quadrant_label(rs_ratio_s: pd.Series, rs_mom_s: pd.Series) -> pd.Series:
    """Vectorised quadrant classification; NaN where either axis is undefined."""
    r, m = rs_ratio_s.to_numpy(), rs_mom_s.to_numpy()
    out = np.full(len(r), None, dtype=object)
    valid = ~np.isnan(r) & ~np.isnan(m)
    lead = valid & (r >= 100) & (m >= 100)
    weak = valid & (r >= 100) & (m < 100)
    lag = valid & (r < 100) & (m < 100)
    impr = valid & (r < 100) & (m >= 100)
    out[lead], out[weak], out[lag], out[impr] = "Leading", "Weakening", "Lagging", "Improving"
    return pd.Series(out, index=rs_ratio_s.index)


def rrg_frame(prices: pd.DataFrame, tickers: list[str], bench_col: str,
              window: int, mom_window: int) -> dict[str, pd.DataFrame]:
    """Daily RS-Ratio / RS-Momentum / quadrant for every ticker. One frame per ticker."""
    out = {}
    bench = prices[bench_col]
    for tk in tickers:
        if tk not in prices.columns:
            continue
        rr = rs_ratio(prices[tk], bench, window)
        rm = rs_momentum(rr, mom_window, window)
        q = quadrant_label(rr, rm)
        out[tk] = pd.DataFrame({"rs_ratio": rr, "rs_momentum": rm, "quadrant": q})
    return out


# --------------------------------------------------------------------------- #
# Monthly panels — quadrant snapshot at month-end, monthly simple returns
# --------------------------------------------------------------------------- #
def monthly_quadrants(daily_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Month-end quadrant label per ticker (last valid daily reading each month)."""
    cols = {}
    for tk, df in daily_frames.items():
        q = df["quadrant"].dropna()
        if q.empty:
            continue
        monthly = q.resample("ME").last()
        monthly.index = pd.PeriodIndex(monthly.index, freq="M")
        cols[tk] = monthly
    return pd.DataFrame(cols)


def monthly_returns(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Monthly simple returns (month-end close to month-end close) for each ticker."""
    monthly_px = prices[tickers].resample("ME").last()
    rets = monthly_px.pct_change()
    rets.index = pd.PeriodIndex(rets.index, freq="M")
    return rets


def benchmark_monthly_returns(prices: pd.DataFrame, bench_col: str) -> pd.Series:
    """Monthly simple returns of the benchmark (SPY), same PeriodIndex convention."""
    m = prices[bench_col].resample("ME").last().pct_change()
    m.index = pd.PeriodIndex(m.index, freq="M")
    return m


# --------------------------------------------------------------------------- #
# Portfolio construction
# --------------------------------------------------------------------------- #
def leading_weights(quad_row: pd.Series) -> pd.Series:
    """Equal-weight every ticker labelled 'Leading' this month; all-cash if none."""
    leading = quad_row[quad_row == "Leading"].index
    w = pd.Series(0.0, index=quad_row.index)
    if len(leading) == 0:
        return w
    w[leading] = 1.0 / len(leading)
    return w


def top_k_momentum_weights(score_row: pd.Series, k: int) -> pd.Series:
    """Equal-weight top-K sectors by plain trailing momentum score (the 1-D control)."""
    valid = score_row.dropna()
    w = pd.Series(0.0, index=score_row.index)
    if len(valid) < k:
        return w
    ranked = valid.nlargest(k)
    w[ranked.index] = 1.0 / k
    return w


def _turnover(w_prev: pd.Series, w_cur: pd.Series) -> float:
    """One-way turnover = 0.5 * sum(|w_cur - w_prev|)."""
    return 0.5 * float((w_cur - w_prev).abs().sum())


def run_rrg_strategy(quad_panel: pd.DataFrame, ret_panel: pd.DataFrame,
                      cost_bps: float = 5.0) -> pd.DataFrame:
    """Backtest 'long the Leading quadrant, equal-weight, monthly rebalance, cash if empty'.

    One documented execution lag: the quadrant is read at month-end close ``t``; the
    position formed from it is held over month ``t+1`` (entered at the ``t+1`` open, i.e.
    the previous month's *weights* are applied to month ``t+1``'s *return*) — no look-
    ahead. Cost = one-way ``cost_bps`` x NAV per leg, charged twice per rebalance
    (round trip) on realised turnover.
    """
    idx = ret_panel.index
    cols = ret_panel.columns
    w_prev = pd.Series(0.0, index=cols)
    c = cost_bps * 1e-4
    rows = []
    for t in idx:
        q_row = quad_panel.loc[t] if t in quad_panel.index else pd.Series(index=cols, dtype=object)
        q_row = q_row.reindex(cols)
        w_cur = leading_weights(q_row)
        r_gross = float((w_prev * ret_panel.loc[t].fillna(0.0)).sum())
        to = _turnover(w_prev, w_cur)
        r_net = r_gross - c * to * 2.0
        rows.append({"date": t, "r_gross": r_gross, "r_net": r_net,
                     "n_leading": int((w_cur > 0).sum()), "turnover": to})
        w_prev = w_cur
    return pd.DataFrame(rows).set_index("date")


def run_momentum_strategy(ret_panel: pd.DataFrame, lookback: int = 6, skip: int = 1,
                           k: int = 3, cost_bps: float = 5.0) -> pd.DataFrame:
    """The 1-D control: classic 6-1 top-K trailing-momentum sort (cf. sibling 225)."""
    cum = (1.0 + ret_panel.fillna(0.0)).cumprod()
    score = cum.shift(skip) / cum.shift(skip + lookback) - 1.0
    w_prev = pd.Series(0.0, index=ret_panel.columns)
    c = cost_bps * 1e-4
    rows = []
    for t in ret_panel.index:
        w_cur = top_k_momentum_weights(score.loc[t], k)
        r_gross = float((w_prev * ret_panel.loc[t].fillna(0.0)).sum())
        to = _turnover(w_prev, w_cur)
        r_net = r_gross - c * to * 2.0
        rows.append({"date": t, "r_gross": r_gross, "r_net": r_net, "turnover": to})
        w_prev = w_cur
    return pd.DataFrame(rows).set_index("date")


def _quad_arrays(quad_panel: pd.DataFrame, ret_panel: pd.DataFrame
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align quad_panel/ret_panel to ret_panel's (index, columns) and return plain numpy:
    ret_arr (months x tickers, NaN->0), is_leading (bool), is_valid (bool, has a quadrant)."""
    cols = list(ret_panel.columns)
    idx = ret_panel.index
    q = quad_panel.reindex(index=idx, columns=cols)
    ret_arr = ret_panel[cols].fillna(0.0).to_numpy()
    is_leading = (q == "Leading").to_numpy()
    is_valid = q.notna().to_numpy()
    return ret_arr, is_leading, is_valid


def run_random_matched(quad_panel: pd.DataFrame, ret_panel: pd.DataFrame,
                        seed: int = 0, cost_bps: float = 5.0) -> pd.DataFrame:
    """Each month pick a RANDOM set of sectors, sized to match RRG's actual holding
    count that month (0 sectors -> cash, same as RRG). Tests whether the quadrant
    *selection* beats picking the same number of names blindly. Pure-numpy inner loop
    (no per-row pandas ``.loc``) so it is cheap to Monte-Carlo over many seeds."""
    ret_arr, is_leading, is_valid = _quad_arrays(quad_panel, ret_panel)
    n_months, n_tickers = ret_arr.shape
    n_leading = is_leading.sum(axis=1)
    rng = np.random.default_rng(seed)
    c = cost_bps * 1e-4
    w_prev = np.zeros(n_tickers)
    r_net = np.zeros(n_months)
    for t in range(n_months):
        nl = int(n_leading[t])
        valid_idx = np.nonzero(is_valid[t])[0]
        w_cur = np.zeros(n_tickers)
        if nl > 0 and len(valid_idx) >= nl:
            chosen = rng.choice(valid_idx, size=nl, replace=False)
            w_cur[chosen] = 1.0 / nl
        r_gross = float(w_prev @ ret_arr[t])
        to = 0.5 * float(np.abs(w_cur - w_prev).sum())
        r_net[t] = r_gross - c * to * 2.0
        w_prev = w_cur
    return pd.DataFrame({"r_net": r_net}, index=ret_panel.index)


def equal_weight_returns(ret_panel: pd.DataFrame) -> pd.Series:
    """Equal-weight all tickers with a valid return each month (no rebalancing cost)."""
    return ret_panel.mean(axis=1, skipna=True)


def matched_random_control(quad_panel: pd.DataFrame, ret_panel: pd.DataFrame,
                            n_seeds: int = 200, base_seed: int = 0,
                            cost_bps: float = 5.0) -> pd.Series:
    """Monte-Carlo-averaged 'pick N random sectors, same N as RRG held that month'
    control (N=0 -> cash, exactly matching RRG's cash months). Averaging many seeds'
    monthly returns isolates **selection skill** (which sectors) from the mechanical
    **count/cash-timing** effect (how many, including going to cash) that a single
    EW-basket or SPY comparison would otherwise conflate with skill. Vectorised inner
    loop (see :func:`run_random_matched`) so 200+ seeds run in well under a second.
    """
    ret_arr, is_leading, is_valid = _quad_arrays(quad_panel, ret_panel)
    n_months, n_tickers = ret_arr.shape
    n_leading = is_leading.sum(axis=1)
    c = cost_bps * 1e-4
    acc = np.zeros(n_months)
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        w_prev = np.zeros(n_tickers)
        for t in range(n_months):
            nl = int(n_leading[t])
            valid_idx = np.nonzero(is_valid[t])[0]
            w_cur = np.zeros(n_tickers)
            if nl > 0 and len(valid_idx) >= nl:
                chosen = rng.choice(valid_idx, size=nl, replace=False)
                w_cur[chosen] = 1.0 / nl
            r_gross = float(w_prev @ ret_arr[t])
            to = 0.5 * float(np.abs(w_cur - w_prev).sum())
            acc[t] += r_gross - c * to * 2.0
            w_prev = w_cur
    return pd.Series(acc / n_seeds, index=ret_panel.index)


# --------------------------------------------------------------------------- #
# Inference primitives (Newey-West HAC t on a mean; matches sibling 225/637 style)
# --------------------------------------------------------------------------- #
def newey_west_mean_t(x: np.ndarray, lags: int | None = None) -> float:
    """HAC (Newey-West, Bartlett kernel) t-stat that the mean of ``x`` is zero."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mean = x.mean()
    e = x - mean
    lrv = float(e @ e) / n
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        lrv += 2.0 * w * float(e[lag:] @ e[:-lag]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mean / se) if se > 0 else float("nan")


def summarize(returns: pd.Series, periods_per_year: int = TRADING_MONTHS_PER_YEAR) -> dict:
    """Annualised mean/vol/Sharpe/max-drawdown + Newey-West t-stat on the mean."""
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: float("nan") for k in
                ("n", "ann_ret", "ann_vol", "sharpe", "max_drawdown", "tstat")}
    mean_m, std_m = float(r.mean()), float(r.std(ddof=1))
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    sharpe = (mean_m / std_m * np.sqrt(periods_per_year)) if std_m > 0 else float("nan")
    return {
        "n": int(n), "ann_ret": float(mean_m * periods_per_year),
        "ann_vol": float(std_m * np.sqrt(periods_per_year)), "sharpe": float(sharpe),
        "max_drawdown": dd, "tstat": newey_west_mean_t(r.to_numpy()),
    }


def active_stats(strategy_ret: pd.Series, control_ret: pd.Series) -> dict:
    """Active-return (strategy - control) mean/annualised/HAC-t, aligned on shared dates."""
    a, b = strategy_ret.align(control_ret, join="inner")
    diff = (a - b).dropna()
    s = summarize(diff)
    return {"active_ann": s["ann_ret"], "active_tstat": s["tstat"], "n": s["n"]}


# --------------------------------------------------------------------------- #
# The myth-check — does the chart actually rotate clockwise?
# --------------------------------------------------------------------------- #
CYCLE_ORDER = ["Leading", "Weakening", "Lagging", "Improving"]  # clockwise


def quadrant_transition_matrix(quad_panel: pd.DataFrame) -> pd.DataFrame:
    """Pooled month-to-month quadrant transition COUNTS across every ticker.

    Row = quadrant at month *t*, column = quadrant at month *t+1*; pooled across all
    tickers (each ticker contributes its own consecutive-month pairs). Purely
    descriptive — this is the third-axis myth-check, never used for the Signal or
    Tradability stamps.
    """
    trans = pd.DataFrame(0, index=CYCLE_ORDER, columns=CYCLE_ORDER)
    for tk in quad_panel.columns:
        s = quad_panel[tk].dropna()
        for a, b in zip(s.values[:-1], s.values[1:]):
            if a in CYCLE_ORDER and b in CYCLE_ORDER:
                trans.loc[a, b] += 1
    return trans


def clockwise_share(trans: pd.DataFrame) -> dict:
    """Given quadrant changes only (excludes staying put), what share moved to the
    NEXT quadrant in the claimed clockwise cycle vs the two other possibilities
    (one-step reversal, or the diagonal opposite)? A uniform random mover would score
    33.3%; a de Kempenaer-faithful world scores well above it.
    """
    n = len(CYCLE_ORDER)
    fwd = 0
    non_stay = 0
    per_quadrant = {}
    for i, q in enumerate(CYCLE_ORDER):
        nxt = CYCLE_ORDER[(i + 1) % n]
        row = trans.loc[q]
        stay = int(row[q])
        total = int(row.sum())
        moved = total - stay
        f = int(row[nxt])
        fwd += f
        non_stay += moved
        per_quadrant[q] = (f / moved * 100.0) if moved > 0 else float("nan")
    pooled = (fwd / non_stay * 100.0) if non_stay > 0 else float("nan")
    return {"pooled_clockwise_pct": pooled, "per_quadrant_pct": per_quadrant,
            "n_moves": non_stay, "random_baseline_pct": 100.0 / (n - 1)}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, sector_cols: list[str], bench_col: str,
                      window: int, mom_window: int, cost_bps: float = 5.0,
                      n_control_seeds: int = 50) -> dict:
    """Run the RRG-vs-matched-random-control active-return test on a synthetic panel.

    Uses the matched-random control (same headline test as the real tape) rather than
    a plain equal-weight basket, so the detector isolates *quadrant-selection* skill
    from the mechanical cash-timing drag of "go to cash when nothing is Leading" —
    a drag any positive-drift world creates whether or not real rotation exists.
    """
    frames = rrg_frame(prices, sector_cols, bench_col, window, mom_window)
    quad = monthly_quadrants(frames)
    rets = monthly_returns(prices, sector_cols)
    rrg = run_rrg_strategy(quad, rets, cost_bps=cost_bps)
    ctrl = matched_random_control(quad, rets, n_seeds=n_control_seeds, cost_bps=cost_bps)
    return active_stats(rrg["r_net"], ctrl)
