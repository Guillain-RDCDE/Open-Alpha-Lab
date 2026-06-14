"""Strategy and honest controls — Study 147 (FX-Momentum).

The recipe (Menkhoff, Sarno, Schmeling & Schrimpf 2012): each month, rank G10
currencies by their trailing 12-month log-return vs USD (skipping the most recent
month to sidestep short-term reversal). Go long the top 3, short the bottom 3, equal
weight within each leg, hold for one month. The portfolio is long-short so the
USD exposure cancels (roughly), and the bet is purely on cross-sectional momentum.

The honest control: the same portfolio construction but with currencies assigned
**random ranks** (i.i.d. uniform draw) instead of the momentum ranks. If the
momentum signal carries information, the momentum portfolio should beat the random
control. If it doesn't, both earn zero and the signal is noise.

No look-ahead: signals are computed on returns through month t; the portfolio enters
at the start of month t+1 (equivalently: we use ``shift(1)`` on the rank signal so
that month-t+1 return is matched to month-t signal).

Cost model: FX spot is highly liquid; a one-way round-trip cost of 2–5 bps is
typical for institutional G10 trades, rising to 10 bps for a fully inclusive retail
estimate. We sweep costs from 0 to 20 bps per leg (40 bps round-trip per pair).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import LOOKBACK_MONTHS, SKIP_MONTHS, TOP_N, BOT_N


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def momentum_ranks(
    monthly: pd.DataFrame,
    lookback: int = LOOKBACK_MONTHS,
    skip: int = SKIP_MONTHS,
) -> pd.DataFrame:
    """Trailing (lookback - skip) cumulative log-return, ranked cross-sectionally.

    At each month-end t, the signal is the sum of log-returns from t-lookback to
    t-skip (inclusive). Returns a DataFrame of the same shape as ``monthly`` with
    the rank (1 = weakest, n = strongest) of each currency at each month-end.
    Rank is computed on the SIGNAL date t; the portfolio enters at t+1.

    Exactly mirrors the Menkhoff et al. (2012) 12-1 sort.
    """
    # Rolling sum from t-lookback to t-skip: sum over `window` months ending at t-skip.
    window = lookback - skip
    cum_ret = monthly.shift(skip).rolling(window, min_periods=window).sum()
    # Cross-sectional rank (1 = lowest cum return = short leg).
    ranks = cum_ret.rank(axis=1, method="first")
    return ranks


def random_ranks(
    monthly: pd.DataFrame,
    seed: int = 0,
) -> pd.DataFrame:
    """Reproducible i.i.d. random cross-sectional ranks — the control arm.

    Each month independently draws a uniformly random permutation of 1..n_ccy.
    Passing this to :func:`build_portfolio` in place of real ranks gives a
    portfolio that has identical turnover and legs as the momentum portfolio but
    no information — a fair coin.
    """
    rng = np.random.default_rng(seed)
    n_rows, n_cols = monthly.shape
    out = np.zeros_like(monthly.values, dtype=float)
    for i in range(n_rows):
        out[i] = rng.permutation(n_cols) + 1  # ranks 1..n_cols
    return pd.DataFrame(out, index=monthly.index, columns=monthly.columns)


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------
def build_portfolio(
    monthly: pd.DataFrame,
    ranks: pd.DataFrame,
    top_n: int = TOP_N,
    bot_n: int = BOT_N,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Monthly long-short portfolio from cross-sectional ranks.

    Long the top ``top_n`` currencies (equal-weight, +1/top_n each), short the
    bottom ``bot_n`` (equal-weight, -1/bot_n each). The portfolio return for month
    t+1 uses the signal formed at month t (no look-ahead via ``shift(1)``).

    ``cost_bps`` is charged per **leg** per **month** (i.e. on the gross notional
    of each side), applied whenever that currency's side changes.  At full monthly
    turnover and two legs that is 2 * cost_bps bps gross per rebalance.

    Returns a DataFrame with columns:
        ``ret_gross``   — portfolio log-return before costs
        ``ret_net``     — portfolio log-return after costs
        ``n_long``      — number of long currencies that month
        ``n_short``     — number of short currencies that month
        ``turnover``    — fraction of gross notional that changed side this month
    """
    n_ccy = monthly.shape[1]

    # Weights derived from ranks (same-date ranks → next-month return).
    rows = []
    prev_weight = pd.Series(0.0, index=monthly.columns)
    for date, rank_row in ranks.iterrows():
        # Only compute if we have at least top_n + bot_n currencies ranked.
        valid = rank_row.dropna()
        if len(valid) < top_n + bot_n:
            continue
        # Long the highest-ranked, short the lowest-ranked.
        sorted_ccy = valid.sort_values()
        longs = sorted_ccy.tail(top_n).index.tolist()
        shorts = sorted_ccy.head(bot_n).index.tolist()
        weight = pd.Series(0.0, index=monthly.columns)
        weight[longs] = 1.0 / top_n
        weight[shorts] = -1.0 / bot_n

        # Portfolio return is the NEXT month's return (shift applied to monthly).
        next_month = monthly.shift(-1).loc[date]  # month t+1 return
        if next_month.isna().all():
            continue

        ret_gross = float((weight * next_month).sum())

        # Turnover: fraction of gross notional that changed side.
        delta = (weight - prev_weight).abs().sum() / 2.0
        cost = delta * cost_bps * 1e-4
        ret_net = ret_gross - cost

        rows.append(
            {
                "date": date,
                "ret_gross": ret_gross,
                "ret_net": ret_net,
                "n_long": len(longs),
                "n_short": len(shorts),
                "turnover": float(delta),
            }
        )
        prev_weight = weight

    df = pd.DataFrame(rows).set_index("date")
    return df


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summarize(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline monthly statistics for one ledger.

    Returns observation count, annualised mean (%), annualised Sharpe, P&L skew,
    and a Newey-West HAC t-stat on the monthly mean — the inference-bar number
    that decides whether the edge clears zero.
    """
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 2:
        return {
            "n_months": int(n),
            "mean_ann_pct": float("nan"),
            "sharpe_ann": float("nan"),
            "skew": float("nan"),
            "tstat": float("nan"),
        }

    mu = r.mean()
    sd = r.std(ddof=1)
    out = {
        "n_months": int(n),
        "mean_ann_pct": float(mu * 12 * 100),
        "sharpe_ann": float(mu / sd * np.sqrt(12)) if sd > 0 else float("nan"),
        "skew": float(pd.Series(r).skew()),
        "tstat": float("nan"),
    }

    # Newey-West HAC t-stat on the monthly mean (Bartlett kernel).
    if n > 5:
        e = r - mu
        lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
        lrv = float(e @ e) / n
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        out["tstat"] = float(mu / se) if se > 0 else float("nan")

    return out


def cost_sweep(
    monthly: pd.DataFrame,
    ranks: pd.DataFrame,
    costs_bps: list[float] | None = None,
    top_n: int = TOP_N,
    bot_n: int = BOT_N,
) -> pd.DataFrame:
    """Net monthly return statistics across a range of per-leg costs.

    Returns a tidy DataFrame indexed by cost_bps with columns from :func:`summarize`.
    """
    if costs_bps is None:
        costs_bps = [0, 2, 5, 10, 20]
    rows = {}
    for c in costs_bps:
        led = build_portfolio(monthly, ranks, top_n=top_n, bot_n=bot_n, cost_bps=c)
        rows[c] = summarize(led, "ret_net")
    return pd.DataFrame(rows).T
