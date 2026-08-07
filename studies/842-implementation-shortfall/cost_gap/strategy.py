"""Strategy + cost model + inference for Study 842 — Implementation Shortfall.

The claim, at full strength (Perold 1988, *The Implementation Shortfall*): the return in
the backtest ("paper portfolio") and the return you actually capture ("live portfolio")
differ by the cost of *trading into* the positions — commissions, spread, and market
impact — and that gap scales with how much you **trade**. A moderate-turnover signal with
a gorgeous 0-cost Sharpe can be a money-loser once the friction of rotating the book is
charged, and the faster the signal decays (the higher the turnover), the deadlier the gap.

The engine:

1. **The book.** A dollar-neutral cross-sectional long-short: on each day rank the names by
   the signal known at the close of ``t-1`` (one ``shift``), go long the top ``frac``,
   short the bottom ``frac``, equal-weight each leg. Fully vectorised (rank via
   ``argsort``); no per-day Python loop over dates.

2. **Turnover.** The one-way traded fraction of the book as the weights rotate,
   ``turnover_t = 0.5 * sum_i |w_{i,t} - w_{i,t-1}|``. A signal that barely moves rotates
   little (low turnover); a fast-decaying one reshuffles the ranks daily (high turnover).

3. **The cost model (the crux).** Two pieces, both scaled by turnover:
   * a **linear** cost — commissions + half-spread — at ``cost_bps`` per unit turnover;
   * a **market-impact** term ~ *participation*. The bigger the fraction of the book you
     rotate in a day, the larger your footprint per share, so impact per unit turnover
     rises with turnover: ``impact_bps_per_unit = impact_coef_bps * turnover``. The daily
     impact drag is therefore ``impact_coef_bps * turnover^2`` — **super-linear** in
     turnover, which is why high-turnover strategies are punished hardest.

   Net daily return: ``net_t = gross_t - turnover_t * cost_bps*1e-4
   - impact_coef_bps*1e-4 * turnover_t^2``.

4. **Inference & diagnostics.** Newey-West (HAC) *t* on the daily spread (gross and net); a
   one-sample and Welch cross-check; the annualised Sharpe; the **cost ladder** (0 /
   realistic / stressed); the **break-even one-way cost** (where net alpha hits zero); the
   **turnover curve** (sweep the persistence knob and watch net Sharpe fall as turnover
   rises while gross stays put); and a seed-robust synthetic control (fires on a planted
   edge, silent on the null — a machinery proof, never market evidence).

Honesty discipline: one execution lag (signal at close ``t-1`` -> hold day ``t``, a single
``shift``); costs are one-way x NAV per rebalance leg; the dollar-neutral book earns
excess-of-zero by construction on the synthetic tape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Inference primitives (shared house engine)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    gamma0 = float(u @ u) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        cov = float(u[l:] @ u[:-l]) / n
        var += 2.0 * w * cov
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def sharpe(returns: np.ndarray, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe (excess-of-zero; the dollar-neutral book funds itself)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else float("nan")


# --------------------------------------------------------------------------- #
# The cross-sectional long-short book — vectorised
# --------------------------------------------------------------------------- #
def book_weights(signal: pd.DataFrame, frac: float = 0.2) -> pd.DataFrame:
    """Dollar-neutral equal-weight long-short weights from a *point-in-time* signal.

    The signal is shifted one row (known at close ``t-1``, held day ``t``), then each row is
    ranked; the top ``frac`` get ``+1/k`` (long), the bottom ``frac`` get ``-1/k`` (short),
    the rest 0. Vectorised via a double ``argsort`` (rank of each entry within its row) — no
    per-day loop. Long and short legs each sum to +/-1, so the gross book is 2x NAV,
    dollar-neutral.
    """
    S = signal.shift(1)                     # the one and only execution lag
    A = S.to_numpy(dtype=float)
    T, N = A.shape
    k = max(1, int(np.floor(N * frac)))
    W = np.zeros((T, N))
    # rank[t, i] = position of A[t, i] in ascending order within row t (0..N-1)
    order = np.argsort(A, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(T)[:, None]
    rank[rows, order] = np.arange(N)[None, :]
    long_mask = rank >= (N - k)
    short_mask = rank < k
    W[long_mask] = 1.0 / k
    W[short_mask] = -1.0 / k
    # Row 0 has an all-NaN shifted signal -> argsort is arbitrary; force flat (no position).
    valid = ~np.isnan(A).all(axis=1)
    W[~valid] = 0.0
    return pd.DataFrame(W, index=signal.index, columns=signal.columns)


def book_returns(returns: pd.DataFrame, signal: pd.DataFrame, frac: float = 0.2) -> pd.DataFrame:
    """The daily gross spread and one-way turnover of the long-short book.

    ``gross_t = sum_i w_{i,t} * r_{i,t}`` (the paper return, no cost). ``turnover_t =
    0.5 * sum_i |w_{i,t} - w_{i,t-1}|`` — the one-way fraction of the book that rotates
    (a full name flip from +1/k to -1/k contributes 2/k; summed over the k longs and k
    shorts and halved, a complete reshuffle is turnover 1.0). Returns a DataFrame with
    columns ``gross`` and ``turnover``.
    """
    W = book_weights(signal, frac)
    Wv = W.to_numpy(dtype=float)
    Rv = returns.to_numpy(dtype=float)
    gross = np.nansum(Wv * Rv, axis=1)
    dW = np.diff(Wv, axis=0, prepend=np.zeros((1, Wv.shape[1])))
    turnover = 0.5 * np.abs(dW).sum(axis=1)
    out = pd.DataFrame({"gross": gross, "turnover": turnover}, index=returns.index)
    # Drop the warm-up first row (initial build, no prior return earned yet).
    return out.iloc[1:]


# --------------------------------------------------------------------------- #
# The cost model — linear + turnover-scaled market impact (~ participation)
# --------------------------------------------------------------------------- #
def apply_costs(
    book: pd.DataFrame, cost_bps: float = 0.0, impact_coef_bps: float = 0.0
) -> pd.Series:
    """Net daily return = gross - linear turnover cost - super-linear market impact.

    ``cost_bps`` is the one-way linear cost (commission + half-spread) per unit turnover.
    ``impact_coef_bps`` scales the market-impact term: impact per unit turnover rises with
    participation (proxied by turnover itself), so the daily impact drag is
    ``impact_coef_bps * turnover^2`` — super-linear, punishing high turnover hardest.
    """
    gross = book["gross"].to_numpy(dtype=float)
    turn = book["turnover"].to_numpy(dtype=float)
    linear = turn * cost_bps * 1e-4
    impact = impact_coef_bps * 1e-4 * turn * turn
    return pd.Series(gross - linear - impact, index=book.index, name="net")


def book_stats(book: pd.DataFrame, cost_bps: float = 0.0, impact_coef_bps: float = 0.0,
               nw_lags: int = 10) -> dict:
    """Headline stats for one cost setting: gross/net Sharpe, spread, turnover, HAC t."""
    gross = book["gross"].to_numpy(dtype=float)
    net = apply_costs(book, cost_bps, impact_coef_bps).to_numpy(dtype=float)
    turn = book["turnover"].to_numpy(dtype=float)
    return {
        "n_days": int(len(book)),
        "cost_bps": float(cost_bps),
        "impact_coef_bps": float(impact_coef_bps),
        "mean_turnover": float(np.nanmean(turn)),
        "gross_bps": float(np.nanmean(gross) * 1e4),
        "net_bps": float(np.nanmean(net) * 1e4),
        "cost_bps_per_day": float(np.nanmean(gross - net) * 1e4),
        "gross_sharpe": sharpe(gross),
        "net_sharpe": sharpe(net),
        "gross_t": newey_west_t(gross, nw_lags),
        "net_t": newey_west_t(net, nw_lags),
        "ann_gross_pct": float(np.nanmean(gross) * TRADING_DAYS * 100),
        "ann_net_pct": float(np.nanmean(net) * TRADING_DAYS * 100),
    }


# --------------------------------------------------------------------------- #
# The cost ladder — 0 / realistic / stressed
# --------------------------------------------------------------------------- #
COST_LADDER = (
    ("paper (0 cost)", 0.0, 0.0),
    ("optimistic", 5.0, 20.0),
    ("realistic", 10.0, 50.0),
    ("stressed", 20.0, 100.0),
)


def cost_ladder(
    book: pd.DataFrame,
    rungs=COST_LADDER,
    nw_lags: int = 10,
) -> pd.DataFrame:
    """Gross vs net across a labelled ladder of ``(label, cost_bps, impact_coef_bps)``.

    The first rung is the paper portfolio (0 cost); each subsequent rung adds a linear cost
    (commission + half-spread) and a turnover-scaled market-impact coefficient. The
    ``realistic`` rung reflects a book that rotates a real fraction of NAV daily; the
    ``stressed`` rung reflects scaled AUM / thinner names (bigger footprint). Returns a
    DataFrame indexed by the human label.
    """
    rows = []
    for label, cb, ic in rungs:
        st = book_stats(book, cost_bps=cb, impact_coef_bps=ic, nw_lags=nw_lags)
        rows.append({
            "scenario": label,
            "cost_bps": cb,
            "impact_coef_bps": ic,
            "gross_sharpe": st["gross_sharpe"],
            "net_sharpe": st["net_sharpe"],
            "net_bps": st["net_bps"],
            "cost_bps_per_day": st["cost_bps_per_day"],
            "net_t": st["net_t"],
            "ann_net_pct": st["ann_net_pct"],
        })
    return pd.DataFrame(rows).set_index("scenario")


def breakeven_cost_bps(book: pd.DataFrame) -> float:
    """The one-way *linear* cost (bps) at which the net alpha hits zero, impact aside.

    Solving ``mean(gross) - mean(turnover) * c*1e-4 = 0`` for the one-way cost ``c``:
    ``c = mean(gross) / mean(turnover) * 1e4``. This is the pedagogical head-room number —
    the cost you can bear before the paper alpha is fully eaten; any realistic cost *above*
    it means the strategy is a Mirage. (Market impact lowers the true break-even further.)
    """
    gross = book["gross"].to_numpy(dtype=float)
    turn = book["turnover"].to_numpy(dtype=float)
    mt = np.nanmean(turn)
    if mt <= 0:
        return float("inf")
    return float(np.nanmean(gross) / mt * 1e4)


# --------------------------------------------------------------------------- #
# The turnover curve — the money chart: alpha dies as a FUNCTION of turnover
# --------------------------------------------------------------------------- #
def turnover_curve(
    data_mod,
    persistences=(0.995, 0.98, 0.96, 0.9, 0.7, 0.3),
    edge: float = 0.0005,
    n_assets: int = 30,
    n_days: int = 2520,
    frac: float = 0.2,
    cost_bps: float = 10.0,
    impact_coef_bps: float = 50.0,
    seed: int = 842,
) -> pd.DataFrame:
    """Sweep the persistence knob; watch net Sharpe collapse as turnover rises.

    The gross edge is held fixed (``edge`` constant, unit-variance signal), so gross Sharpe
    is roughly flat across the sweep — but turnover rises as persistence falls, and the net
    Sharpe (at a fixed realistic cost) falls with it. This is the study's central picture:
    the paper alpha is intact; the *tradable* alpha is a function of how fast you must trade.
    Returns a DataFrame indexed by persistence.
    """
    rows = []
    for phi in persistences:
        rets, sig, _ = data_mod.synthetic_panel(
            edge=edge, persistence=phi, n_assets=n_assets, n_days=n_days, seed=seed
        )
        book = book_returns(rets, sig, frac)
        st = book_stats(book, cost_bps=cost_bps, impact_coef_bps=impact_coef_bps)
        rows.append({
            "persistence": phi,
            "mean_turnover": st["mean_turnover"],
            "gross_sharpe": st["gross_sharpe"],
            "net_sharpe": st["net_sharpe"],
            "breakeven_bps": breakeven_cost_bps(book),
            "net_t": st["net_t"],
        })
    return pd.DataFrame(rows).set_index("persistence")


# --------------------------------------------------------------------------- #
# Synthetic control — the machinery is unbiased (fires on planted edge, silent on null)
# --------------------------------------------------------------------------- #
def synthetic_detect(data_mod, edge: float, persistence: float = 0.96,
                     n_assets: int = 30, n_days: int = 2520, frac: float = 0.2,
                     seed: int = 842) -> dict:
    """Run the gross-book stats on one synthetic world (a machinery proof)."""
    rets, sig, _ = data_mod.synthetic_panel(
        edge=edge, persistence=persistence, n_assets=n_assets, n_days=n_days, seed=seed
    )
    book = book_returns(rets, sig, frac)
    st = book_stats(book, cost_bps=0.0, impact_coef_bps=0.0)
    return {"gross_sharpe": st["gross_sharpe"], "gross_t": st["gross_t"],
            "gross_bps": st["gross_bps"], "mean_turnover": st["mean_turnover"],
            "n_days": st["n_days"]}


def seed_robust_control(data_mod, edge: float, persistence: float = 0.96,
                        n_seeds: int = 20, n_assets: int = 30, n_days: int = 1500,
                        frac: float = 0.2, base_seed: int = 842) -> dict:
    """Average the gross-book HAC t over ``n_seeds`` worlds (the >=20-seed house rule).

    On the null (``edge = 0``) the gross book must not fire (|t| >= 2 in ~0 of the seeds);
    on a planted edge it must light up. A faithful-engine / power check only — never cited
    in support of a stamp.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        d = synthetic_detect(data_mod, edge=edge, persistence=persistence,
                             n_assets=n_assets, n_days=n_days, frac=frac, seed=s)
        ts.append(d["gross_t"])
    ts = np.asarray(ts, dtype=float)
    return {
        "edge": edge,
        "mean_t": float(np.nanmean(ts)),
        "sd_t": float(np.nanstd(ts, ddof=1)) if len(ts) > 1 else float("nan"),
        "fire_count": int(np.sum(np.abs(ts) >= 2)),
        "n_seeds": n_seeds,
    }
