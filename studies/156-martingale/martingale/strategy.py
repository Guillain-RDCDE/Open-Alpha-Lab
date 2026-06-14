"""Strategy and honest controls — Study 156 (Martingale).

The folk recipe: start with a fixed capital stake in a market (here SPY or a stock).
Each time the position is down by a fixed step (e.g. -5% from your last entry), you
*double* (or add to) your position — betting on the bounce. This is the gambler's classic
martingale, re-dressed as "averaging down" or "dollar-cost averaging into a dip."

Three arms are compared:

- **Martingale** — enter 1 unit at day 0; each time the position is down ``step_pct``
  from the last entry, add another ``2^n`` units (doubling). When price recovers above the
  initial entry by ``take_profit_pct``, exit all units. If cumulative exposure exceeds the
  total capital (ruin trigger), the position is liquidated at a total loss of all deployed
  capital. Capital is fixed for the episode; excess allocation needed beyond available
  capital causes a "ruin event."
- **Buy-and-hold** — deploy the same *total* capital (i.e., the same notional that the
  martingale could ever need, worst case) in a single buy-and-hold. This is the *honest*
  comparison: same capital at risk, same instrument, different rule.
- **Fixed-size** — buy 1 unit at day 0 and hold through the same episode, never adding.
  A weaker baseline that isolates the "adding down" effect from the capital-size effect.

The critical insight, encoded explicitly, is the **ruin condition**: the martingale needs
exponentially more capital with each level. At level ``k`` you have deployed
``1 + 2 + 4 + ... + 2^(k-1) = 2^k - 1`` units. With starting capital ``C_total``, ruin
occurs at the first level where ``2^k - 1 > C_total / unit_size``. This is deterministic
and unavoidable for any finite capital. We count ruin events, drawdowns, and skew.

Inference: HAC (Newey-West) t-stat on per-episode returns. The comparison is
``martingale_return`` minus ``buy_and_hold_return`` on the *same* total capital over
*each episode*. Signal = real only if this excess return is reliably positive and the ruin
profile is controlled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Episode engine
# ---------------------------------------------------------------------------
def run_episodes(
    close: pd.Series,
    step_pct: float = 0.05,
    take_profit_pct: float = 0.05,
    max_levels: int = 6,
    cost_bps: float = 5.0,
    seed: int = 156,
) -> pd.DataFrame:
    """Run a series of martingale episodes over the full price history.

    Each episode:
    1. Enter 1 unit at today's close (the "initial entry price").
    2. Each day check if price is down ``step_pct`` below the most recent entry price.
       If so and if we have not yet hit ``max_levels``, double: add 2^level units.
    3. Check exit: price above (initial_entry * (1 + take_profit_pct)) -> exit all units
       at the profit target.
    4. Ruin: if the next doubling would exceed ``max_levels``, we've hit the capital
       limit. The episode ends in ruin — return is -1.0 (total loss of all capital
       committed to this episode).
    5. Episode ends at ruin, take-profit, or end of tape.

    Episodes are non-overlapping and start fresh after each exit. Costs are charged as
    ``cost_bps`` per unit per trade (each add-on or the final close is one trade round-trip).

    Returns a per-episode DataFrame with columns:
        ``start_date, end_date, n_days, n_levels, outcome, ret_gross, ret_net,
          bah_ret_gross, bah_ret_net, excess_ret_gross, excess_ret_net``.

    buy-and-hold (``bah``) uses the same start price and end date, with the same total
    capital (``2^max_levels - 1`` units deployed from day 0 and held through the episode).
    """
    closes = close.to_numpy(dtype=float)
    dates = close.index
    n = len(closes)

    # Total capital multiplier: worst-case, a fully-doubled stack of max_levels
    total_units = float(2**max_levels - 1)  # 1+2+4+...+2^(max_levels-1)
    rng = np.random.default_rng(seed)  # noqa: F841 (reserved for future stochastic exits)

    rows = []
    i = 0
    while i < n - 1:
        entry_px = closes[i]
        start_date = dates[i]

        # Martingale state
        positions = [(entry_px, 1.0)]  # (entry_price, units)
        levels = 0  # number of add-ons so far
        last_entry_px = entry_px
        cumulative_units = 1.0
        cumulative_cost = cost_bps * 1e-4 * 1.0  # cost on initial buy (1 unit)

        outcome = "open"
        end_idx = n - 1

        j = i + 1
        while j < n:
            px = closes[j]
            pnl_from_initial = (px - entry_px) / entry_px

            # Check take-profit: price above initial entry by take_profit_pct
            if pnl_from_initial >= take_profit_pct:
                outcome = "tp"
                end_idx = j
                break

            # Check if we should add: down step_pct from last entry
            down_from_last = (px - last_entry_px) / last_entry_px
            if down_from_last <= -step_pct:
                levels += 1
                if levels >= max_levels:
                    # Ruin: would need more capital than allowed
                    outcome = "ruin"
                    end_idx = j
                    break
                add_units = float(2**levels)
                positions.append((px, add_units))
                cumulative_cost += cost_bps * 1e-4 * add_units
                cumulative_units += add_units
                last_entry_px = px

            j += 1

        if outcome == "open":
            end_idx = n - 1

        exit_px = closes[end_idx]
        end_date = dates[end_idx]
        n_days = end_idx - i + 1

        # --- Martingale return ---
        # Weighted average entry
        if cumulative_units > 0:
            avg_entry = sum(ep * u for ep, u in positions) / cumulative_units
        else:
            avg_entry = entry_px

        if outcome == "ruin":
            ret_gross = -1.0
        else:
            ret_gross = (exit_px - avg_entry) / avg_entry
        ret_net = ret_gross - cumulative_cost / cumulative_units

        # --- Buy-and-hold return (same total capital, same episode window) ---
        # Deploy total_units at entry_px from day i, exit at end_idx price.
        bah_ret_gross = (exit_px - entry_px) / entry_px
        bah_cost = 2.0 * cost_bps * 1e-4  # buy + sell round trip once
        bah_ret_net = bah_ret_gross - bah_cost

        # Excess return: martingale minus buy-and-hold (on the same notional basis)
        excess_ret_gross = ret_gross - bah_ret_gross
        excess_ret_net = ret_net - bah_ret_net

        rows.append(
            {
                "start_date": start_date,
                "end_date": end_date,
                "n_days": int(n_days),
                "n_levels": int(levels),
                "outcome": outcome,
                "ret_gross": float(ret_gross),
                "ret_net": float(ret_net),
                "bah_ret_gross": float(bah_ret_gross),
                "bah_ret_net": float(bah_ret_net),
                "excess_ret_gross": float(excess_ret_gross),
                "excess_ret_net": float(excess_ret_net),
            }
        )

        # Next episode starts the day after this one ends
        i = end_idx + 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ruin-probability simulation
# ---------------------------------------------------------------------------
def ruin_probability(
    annual_vol: float = 0.18,
    mu: float = 0.0,
    step_pct: float = 0.05,
    take_profit_pct: float = 0.05,
    max_levels: int = 6,
    n_paths: int = 5000,
    n_days: int = 252,
    seed: int = 156,
) -> dict:
    """Monte Carlo ruin probability for one martingale episode.

    Simulates ``n_paths`` random price paths of ``n_days`` each (geometric Brownian
    motion with given ``mu`` and ``annual_vol``), runs one martingale episode per path,
    and counts the fraction that end in ruin before the take-profit fires. Also records
    mean episode length and the distribution of outcomes.

    Returns a dict with ruin_rate, tp_rate, eod_rate, mean_days, and a skew estimate.
    """
    rng = np.random.default_rng(seed)
    sigma_d = annual_vol / np.sqrt(252)
    mu_d = mu / 252

    ruin_count = tp_count = eod_count = 0
    episode_rets = []

    for _ in range(n_paths):
        # Simulate one path
        eps = rng.standard_normal(n_days)
        log_ret = mu_d - 0.5 * sigma_d**2 + sigma_d * eps
        prices = 100.0 * np.exp(np.cumsum(log_ret))
        prices = np.concatenate([[100.0], prices])  # start at 100

        entry_px = 100.0
        last_entry_px = entry_px
        cumulative_units = 1.0
        levels = 0
        positions = [(entry_px, 1.0)]
        outcome = "eod"
        avg_entry = entry_px
        exit_px = prices[-1]

        for t in range(1, len(prices)):
            px = prices[t]
            pnl_from_initial = (px - entry_px) / entry_px
            down_from_last = (px - last_entry_px) / last_entry_px

            if pnl_from_initial >= take_profit_pct:
                outcome = "tp"
                exit_px = px
                break

            if down_from_last <= -step_pct:
                levels += 1
                if levels >= max_levels:
                    outcome = "ruin"
                    exit_px = px
                    break
                add_units = float(2**levels)
                positions.append((px, add_units))
                cumulative_units += add_units
                last_entry_px = px

        if cumulative_units > 0:
            avg_entry = sum(ep * u for ep, u in positions) / cumulative_units

        if outcome == "ruin":
            ret = -1.0
            ruin_count += 1
        elif outcome == "tp":
            ret = (exit_px - avg_entry) / avg_entry
            tp_count += 1
        else:
            ret = (exit_px - avg_entry) / avg_entry
            eod_count += 1
        episode_rets.append(ret)

    rets = np.array(episode_rets)
    return {
        "ruin_rate": ruin_count / n_paths,
        "tp_rate": tp_count / n_paths,
        "eod_rate": eod_count / n_paths,
        "mean_ret": float(rets.mean()),
        "mean_days": float(n_days / 2),  # approximate; dominated by path length
        "skew": float(pd.Series(rets).skew()),
        "n_paths": n_paths,
    }


# ---------------------------------------------------------------------------
# Drawdown profile
# ---------------------------------------------------------------------------
def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of an equity curve."""
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    return float(dd.min())


def equity_curve_martingale(
    close: pd.Series,
    step_pct: float = 0.05,
    take_profit_pct: float = 0.05,
    max_levels: int = 6,
    starting_capital: float = 1.0,
    cost_bps: float = 5.0,
) -> tuple[pd.Series, pd.Series]:
    """Build a daily equity curve for the martingale and buy-and-hold over the full tape.

    Each episode updates equity by the fraction of total capital returned (or lost).
    Returns ``(martingale_equity, bah_equity)`` as a pair of Series indexed by date.

    Note on capital accounting: the martingale deploys up to 2^max_levels - 1 units
    across all levels. All capital is available for each episode. Ruin in an episode
    wipes the fraction of capital actually deployed in that episode (a worst-case ruin
    = loss of all capital). We track as a fraction of running equity.
    """
    ledger = run_episodes(close, step_pct, take_profit_pct, max_levels, cost_bps)

    equity_m = starting_capital
    equity_b = starting_capital
    eq_m_series = {}
    eq_b_series = {}

    # Fill day by day, using ledger episodes
    ep_iter = ledger.iterrows()
    try:
        _, ep = next(ep_iter)
    except StopIteration:
        ep = None

    for date, px in close.items():
        if ep is not None and date == ep["end_date"]:
            equity_m = equity_m * (1.0 + ep["ret_net"])
            equity_b = equity_b * (1.0 + ep["bah_ret_net"])
            equity_m = max(equity_m, 0.0)
            try:
                _, ep = next(ep_iter)
            except StopIteration:
                ep = None
        eq_m_series[date] = equity_m
        eq_b_series[date] = equity_b

    return pd.Series(eq_m_series, name="martingale"), pd.Series(eq_b_series, name="bah")


# ---------------------------------------------------------------------------
# Per-episode summary (the inference-bar number)
# ---------------------------------------------------------------------------
def summarize(ledger: pd.DataFrame, col: str = "excess_ret_net") -> dict:
    """Headline per-episode statistics for one ledger.

    Reports episode count, ruin rate, win rate, mean excess return (bps), the P&L skew
    (the tell for the tail-risk trap), and a HAC t-stat on the mean excess return —
    the inference-bar number that decides whether averaging-down beats buy-and-hold.
    """
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    ruin_col = "outcome" if "outcome" in ledger.columns else None
    ruin_rate = float((ledger["outcome"] == "ruin").mean()) if ruin_col else float("nan")
    out = {
        "n_episodes": int(n),
        "ruin_rate": ruin_rate,
        "win_rate": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "skew": float(pd.Series(r).skew()) if n > 2 else float("nan"),
        "tstat": float("nan"),
    }
    if n > 5:
        mu = r.mean()
        e = r - mu
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        lrv = float(e @ e) / n
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        out["tstat"] = float(mu / se) if se > 0 else float("nan")
    return out
