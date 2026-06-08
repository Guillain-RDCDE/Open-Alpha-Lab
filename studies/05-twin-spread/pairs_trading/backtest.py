"""The trading rule, and the honest P&L — GGR (1999), with the costs charged back.

Formation hands the trader a set of twins and, per pair, the spread ``sigma`` measured
over the formation window. Here we *trade* them in the following window:

    * the **spread** is the live difference of the two normalized prices, kept
      compounding continuously from the formation start (so it is comparable to the
      formation ``sigma``);
    * **open** when ``|spread| > k·sigma`` — short the high leg, long the low leg, one
      dollar each (dollar-neutral);
    * **close** when the spread crosses back through zero (the prices cross), or at the
      end of the trading window — whichever comes first.

Two execution truths the headline GGR numbers gloss over, both first-class here:

    * **You can't trade the close that defined the signal.** The spread crossing
      ``2·sigma`` is computed from today's close; you act *next* session. ``wait`` days
      of execution lag (default **1**) is the difference between the famous number and
      the one you could book — and historically most of the edge lives in that one day
      (the bid-ask bounce). ``wait=0`` reproduces the optimistic in-sample figure.
    * **Every entry and exit crosses the spread on both legs.** A round trip pays four
      half-spreads. On the liquid names this is survivable; the cost sweep shows where
      it stops being.

P&L convention is GGR's conservative **return on committed capital**: each of the
``top_n`` pairs is allotted an equal slice of the book whether or not it ever opens, so
the portfolio daily return is the *mean* pair P&L across all pairs (a flat pair
contributes 0). The looser **return on employed capital** — P&L over only the
pair-days actually in the market — is reported alongside, because that is the number
the folklore quotes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pairs import Pair, normalized_prices

TRADING_DAYS = 252
MONTH_DAYS = 21


@dataclass(frozen=True)
class CostModel:
    """Per-leg, one-way trading cost in basis points (1 bp = 0.01%).

    Defaults are a *liquid large-cap* scenario — the universe this study actually runs
    on — not the micro-cap spreads of Study 04. A round trip on a pair pays
    ``4 × (half_spread + commission + slippage)`` (two legs, in and out).
    """
    half_spread_bps: float = 5.0
    commission_bps: float = 0.5
    slippage_bps: float = 1.0

    def leg_cost_frac(self) -> float:
        return (self.half_spread_bps + self.commission_bps + self.slippage_bps) / 1e4


@dataclass
class PairTrade:
    a: str
    b: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    days_held: int
    direction: int          # +1 = long A / short B, -1 = short A / long B
    gross_ret: float        # long-short P&L over the hold, before costs
    net_ret: float          # after 4 half-spreads (open + close, both legs)


@dataclass
class WindowResult:
    daily_net: pd.Series        # committed-capital portfolio return, trading-window dates
    daily_gross: pd.Series
    trades: list[PairTrade]
    deployed_frac: float        # mean fraction of pairs in the market on a given day


@dataclass
class BacktestResult:
    daily: pd.Series            # full committed-capital daily return series
    equity: pd.Series
    trades: list[PairTrade]
    stats: dict
    params: dict


# --------------------------------------------------------------------------- #
# Single-pair simulation
# --------------------------------------------------------------------------- #

def _signal_positions(spread: np.ndarray, sigma: float, k: float) -> np.ndarray:
    """Desired position each day from a close-to-close state machine (no exec lag yet).

    ``sig[t]`` is the holding implied by information through close ``t``: 0 flat,
    +1 long-A/short-B (spread below −k·sigma, A too cheap), −1 the mirror. Opens on a
    ``k·sigma`` divergence, holds until the spread crosses zero, then waits flat for the
    next divergence. Execution lag and costs are applied by the caller.
    """
    n = len(spread)
    sig = np.zeros(n, dtype=float)
    thr = k * sigma
    pos = 0
    for t in range(n):
        if pos == 0:
            if spread[t] > thr:
                pos = -1                      # A rich -> short A, long B
            elif spread[t] < -thr:
                pos = +1                      # A cheap -> long A, short B
        else:
            # close when the spread reverts through zero (sign flip relative to entry)
            if pos == +1 and spread[t] >= 0.0:
                pos = 0
            elif pos == -1 and spread[t] <= 0.0:
                pos = 0
        sig[t] = pos
    return sig


def _executed_positions(
    sig: np.ndarray,
    r_a: np.ndarray,
    r_b: np.ndarray,
    wait: int,
    stop_loss: float | None,
) -> np.ndarray:
    """Turn signal positions into *executed* ones: lag by ``wait``, optionally stop out.

    The position held on day t is the signal decided at close ``t-wait`` (no look-ahead).
    With ``stop_loss`` set, an open episode is force-flattened for the rest of that
    episode once its running long-short P&L since entry breaches ``-stop_loss`` — the
    fix for the naive rule's habit of holding a broken pair to the window edge. With
    ``stop_loss=None`` this reproduces the plain lagged signal exactly.
    """
    n = len(sig)
    exec_pos = np.zeros(n)
    cur_dir = 0.0
    pnl_acc = 0.0
    stopped = False
    for t in range(n):
        desired = sig[t - wait] if t - wait >= 0 else 0.0
        if desired != cur_dir:           # episode boundary (open / close / sign flip)
            cur_dir = desired
            pnl_acc = 0.0
            stopped = False
        if cur_dir != 0.0 and not stopped:
            exec_pos[t] = cur_dir
            pnl_acc += cur_dir * (r_a[t] - r_b[t])
            if stop_loss is not None and pnl_acc <= -stop_loss:
                stopped = True            # close at today's end; flat for rest of episode
    return exec_pos


def simulate_pair(
    norm_a: np.ndarray,
    norm_b: np.ndarray,
    r_a: np.ndarray,
    r_b: np.ndarray,
    dates: pd.DatetimeIndex,
    sigma: float,
    pair: Pair,
    k: float = 2.0,
    wait: int = 1,
    stop_loss: float | None = None,
    costs: CostModel = CostModel(),
) -> tuple[np.ndarray, np.ndarray, list[PairTrade]]:
    """Trade one pair across a window. Returns ``(daily_net, daily_gross, trades)``.

    ``daily_*`` are length-``T`` arrays of the pair's long-short return on each trading
    day (0 when flat). ``norm_*`` are the continuous normalized prices over the trading
    window; ``r_*`` the matching daily returns. The signal is lagged ``wait`` days, so a
    divergence seen at close ``t`` is executed from ``t+wait`` — the trade you could
    actually place. ``stop_loss`` (a fraction, e.g. 0.10) caps the per-episode loss.
    """
    if sigma <= 0 or len(dates) == 0:
        z = np.zeros(len(dates))
        return z, z.copy(), []

    spread = norm_a - norm_b
    sig = _signal_positions(spread, sigma, k)
    exec_pos = _executed_positions(sig, r_a, r_b, wait, stop_loss)

    leg = costs.leg_cost_frac()
    daily_gross = exec_pos * (r_a - r_b)

    # Cost on any day the held position changes: each unit change turns over both legs.
    dpos = np.abs(np.diff(np.concatenate([[0.0], exec_pos, [0.0]])))
    turn = dpos[:-1]                      # change entering each day (len T)
    # the final unwind cost (exec_pos[-1] -> 0) lands on the last day
    turn[-1] += abs(exec_pos[-1])
    daily_net = daily_gross - turn * 2.0 * leg     # 2 legs per unit turnover

    trades = _extract_trades(exec_pos, r_a, r_b, dates, pair, leg)
    return daily_net, daily_gross, trades


def _extract_trades(
    exec_pos: np.ndarray,
    r_a: np.ndarray,
    r_b: np.ndarray,
    dates: pd.DatetimeIndex,
    pair: Pair,
    leg: float,
) -> list[PairTrade]:
    """Collapse the executed-position path into discrete round-trip trades."""
    trades: list[PairTrade] = []
    n = len(exec_pos)
    t = 0
    while t < n:
        if exec_pos[t] == 0:
            t += 1
            continue
        direction = int(exec_pos[t])
        start = t
        while t < n and exec_pos[t] == direction:
            t += 1
        end = t - 1                       # last day held (inclusive)
        gross = float(np.sum(direction * (r_a[start:end + 1] - r_b[start:end + 1])))
        net = gross - 4.0 * leg           # open both legs + close both legs
        trades.append(PairTrade(
            a=pair.a, b=pair.b, entry_date=dates[start], exit_date=dates[end],
            days_held=end - start + 1, direction=direction,
            gross_ret=gross, net_ret=net,
        ))
    return trades


# --------------------------------------------------------------------------- #
# Window-level: form, trade, aggregate over committed capital
# --------------------------------------------------------------------------- #

def trade_window(
    combined_close: pd.DataFrame,
    pairs: list[Pair],
    form_len: int,
    k: float = 2.0,
    wait: int = 1,
    stop_loss: float | None = None,
    costs: CostModel = CostModel(),
) -> WindowResult:
    """Trade ``pairs`` over the post-formation portion of ``combined_close``.

    ``combined_close`` is the close panel sliced to *formation + trading* (the formation
    block first, ``form_len`` rows, then the trading block). Normalized prices are built
    over the whole slice so the trading spread compounds continuously from the formation
    base; only the trading block is actually traded. Returns the committed-capital
    portfolio daily series and the trades.
    """
    trade_dates = combined_close.index[form_len:]
    T = len(trade_dates)
    if T == 0 or not pairs:
        empty = pd.Series(dtype=float, index=trade_dates)
        return WindowResult(empty, empty.copy(), [], 0.0)

    net_mat = np.zeros((T, len(pairs)))
    gross_mat = np.zeros((T, len(pairs)))
    deployed = np.zeros((T, len(pairs)))
    all_trades: list[PairTrade] = []

    for j, p in enumerate(pairs):
        sub = combined_close[[p.a, p.b]]
        norm = normalized_prices(sub)                 # continuous index, base=1 at form start
        rets = sub.pct_change().fillna(0.0)
        na = norm[p.a].to_numpy()[form_len:]
        nb = norm[p.b].to_numpy()[form_len:]
        ra = rets[p.a].to_numpy()[form_len:]
        rb = rets[p.b].to_numpy()[form_len:]
        net, gross, trades = simulate_pair(
            na, nb, ra, rb, trade_dates, p.sigma, p,
            k=k, wait=wait, stop_loss=stop_loss, costs=costs,
        )
        net_mat[:, j] = net
        gross_mat[:, j] = gross
        deployed[:, j] = (gross != 0.0).astype(float)
        all_trades.extend(trades)

    daily_net = pd.Series(net_mat.mean(axis=1), index=trade_dates, name="pairs_ret")
    daily_gross = pd.Series(gross_mat.mean(axis=1), index=trade_dates, name="pairs_gross")
    return WindowResult(daily_net, daily_gross, all_trades, float(deployed.mean()))


# --------------------------------------------------------------------------- #
# Full rolling backtest
# --------------------------------------------------------------------------- #

def _windows(n_rows: int, form_len: int, trade_len: int):
    """Yield ``(form_start, trade_start, trade_end)`` row indices for non-overlapping
    trading windows rolled forward by ``trade_len`` (the simple GGR cadence)."""
    start = 0
    while start + form_len + trade_len <= n_rows:
        yield start, start + form_len, start + form_len + trade_len
        start += trade_len


def run(
    panel: pd.DataFrame,
    top_n: int = 20,
    form_len: int = 252,
    trade_len: int = 126,
    k: float = 2.0,
    wait: int = 1,
    stop_loss: float | None = None,
    cointegration: bool = False,
    df_crit: float = -2.86,
    costs: CostModel = CostModel(),
) -> BacktestResult:
    """Roll the full GGR pipeline across ``panel`` and stitch the daily P&L together.

    For each non-overlapping trading window: select the ``top_n`` minimum-SSD pairs on
    the prior ``form_len`` sessions, trade them ``trade_len`` sessions, collect the
    committed-capital daily return. Concatenes every window into one series and reports
    headline stats. Deterministic.

    Beat-7 extension knobs (each tests whether one modern fix rescues the naive rule):
      * ``stop_loss`` — cap the per-episode loss (the −77%-drawdown short-gamma tail).
      * ``cointegration`` — keep only pairs whose formation spread passes a Dickey–Fuller
        mean-reversion gate (``df_crit``), so a pair needs an *economic* reason to revert,
        not a lucky formation year.
    """
    from .pairs import select_pairs        # local import keeps module graph flat

    pieces: list[pd.Series] = []
    gross_pieces: list[pd.Series] = []
    trades: list[PairTrade] = []
    deployed_fracs: list[float] = []

    closes = panel
    for fs, ts, te in _windows(len(closes), form_len, trade_len):
        formation = closes.iloc[fs:ts]
        pairs = select_pairs(formation, top_n=top_n,
                             cointegration=cointegration, df_crit=df_crit)
        if not pairs:
            continue
        combined = closes.iloc[fs:te]
        wr = trade_window(combined, pairs, form_len=form_len, k=k, wait=wait,
                          stop_loss=stop_loss, costs=costs)
        pieces.append(wr.daily_net)
        gross_pieces.append(wr.daily_gross)
        trades.extend(wr.trades)
        deployed_fracs.append(wr.deployed_frac)

    daily = pd.concat(pieces) if pieces else pd.Series(dtype=float)
    gross = pd.concat(gross_pieces) if gross_pieces else pd.Series(dtype=float)
    equity = (1.0 + daily).cumprod().rename("equity")
    stats = _performance(daily, gross, equity, trades, deployed_fracs)
    params = {"top_n": top_n, "form_len": form_len, "trade_len": trade_len,
              "k": k, "wait": wait, "stop_loss": stop_loss,
              "cointegration": cointegration, "costs": costs}
    return BacktestResult(daily, equity, trades, stats, params)


def _performance(daily, gross, equity, trades, deployed_fracs) -> dict:
    if len(daily) == 0:
        return {"n_days": 0, "n_trades": 0}
    sd = daily.std(ddof=1)
    sharpe = float(daily.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan
    running_max = equity.cummax()
    max_dd = float((equity / running_max - 1.0).min())
    deployed = float(np.mean(deployed_fracs)) if deployed_fracs else np.nan

    # employed-capital monthly: P&L per pair-day actually in the market
    gross_arr = np.asarray([t.gross_ret for t in trades]) if trades else np.array([])
    net_arr = np.asarray([t.net_ret for t in trades]) if trades else np.array([])
    mean_daily_net = float(daily.mean())
    emp_monthly = mean_daily_net / deployed * MONTH_DAYS if deployed and deployed > 0 else np.nan

    return {
        "n_days": int(len(daily)),
        "n_trades": int(len(trades)),
        "committed_monthly_net": mean_daily_net * MONTH_DAYS,
        "committed_monthly_gross": float(gross.mean()) * MONTH_DAYS,
        "employed_monthly_net": emp_monthly,
        "sharpe_net": sharpe,
        "ann_return_net": float((1.0 + mean_daily_net) ** TRADING_DAYS - 1.0),
        "max_drawdown": max_dd,
        "deployed_frac": deployed,
        "mean_trade_gross": float(gross_arr.mean()) if gross_arr.size else np.nan,
        "mean_trade_net": float(net_arr.mean()) if net_arr.size else np.nan,
        "win_rate_net": float((net_arr > 0).mean()) if net_arr.size else np.nan,
        "median_days_held": float(np.median([t.days_held for t in trades])) if trades else np.nan,
    }


def cost_sweep(
    panel: pd.DataFrame,
    half_spread_grid=(0, 2, 5, 10, 20, 40),
    top_n: int = 20,
    form_len: int = 252,
    trade_len: int = 126,
    k: float = 2.0,
    wait: int = 1,
    base: CostModel = CostModel(),
) -> pd.DataFrame:
    """Committed-capital monthly net return as the half-spread rises — where it dies."""
    rows = []
    for hs in half_spread_grid:
        c = CostModel(float(hs), base.commission_bps, base.slippage_bps)
        res = run(panel, top_n=top_n, form_len=form_len, trade_len=trade_len,
                  k=k, wait=wait, costs=c)
        s = res.stats
        rows.append({
            "half_spread_bps": hs,
            "roundtrip_bps": 4.0 * c.leg_cost_frac() * 1e4,
            "committed_monthly_net": s.get("committed_monthly_net", np.nan),
            "sharpe_net": s.get("sharpe_net", np.nan),
            "mean_trade_net": s.get("mean_trade_net", np.nan),
            "n_trades": s.get("n_trades", 0),
        })
    return pd.DataFrame(rows).set_index("half_spread_bps")
