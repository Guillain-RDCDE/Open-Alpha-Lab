"""Strategy + inference for Study 382 — Treasury cash-futures Basis Trade.

The trade (on our transparent carry model, since true cash-vs-futures bond prices and repo
are not free): hold the duration-hedged basis and earn the daily **net carry**
``(yield − funding)/252`` (1-day lag, no look-ahead) plus a small residual mark-to-market.
Lever it N× via repo, as real basis desks do (50–100×).

We test the two halves of "free carry" separately:

  * **Signal (is the carry real?)** — a **HAC / Newey-West** *t* of the mean unlevered
    daily return, and the **annualised Sharpe** with its standard error. Leverage-invariant,
    so this is the honest "is there an edge" question.
  * **Tradability (does it survive leverage?)** — the same return series levered N×: the
    Sharpe does **not** move (mechanical), but the **max drawdown**, the worst day, and the
    **funding-shock** tail (CVaR, a March-2020-style stress unwind) explode. A high Sharpe
    that turns into a –75% drawdown the moment you apply the trade's actual leverage is not
    free carry — it is a short-volatility / funding-tail position.

A **placebo null** (sign-randomised carry) and a **synthetic positive control** (planted
unlevered Sharpe) keep the inference honest: a high Sharpe must come from a real positive
mean, not from autocorrelation or from levering noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Leverage
# --------------------------------------------------------------------------- #
def levered_returns(frame: pd.DataFrame, leverage: float = 1.0) -> np.ndarray:
    """Daily levered basis return = ``leverage × ret`` (the unlevered model P&L)."""
    return leverage * frame["ret"].values


def equity_curve(daily: np.ndarray) -> np.ndarray:
    """Cumulative equity (compounded) of a daily return array, starting at 1.0."""
    return np.cumprod(1.0 + daily)


def max_drawdown(daily: np.ndarray) -> float:
    """Worst peak-to-trough drawdown of the compounded equity curve (negative)."""
    eq = equity_curve(daily)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


# --------------------------------------------------------------------------- #
# Inference — is the (leverage-invariant) carry real?
# --------------------------------------------------------------------------- #
def nw_se(x: np.ndarray, lag: int) -> float:
    """Newey-West (HAC) standard error of the sample mean of ``x``."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n < 2:
        return float("nan")
    s = (x @ x) / n
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)
        s += 2.0 * w * (x[k:] @ x[:-k]) / n
    return float(np.sqrt(s / n))


def hac_t(daily: np.ndarray, lag: int = 21) -> float:
    """HAC (Newey-West) *t* of the mean daily return — the Signal-axis statistic.

    Leverage-invariant: scaling ``daily`` by any positive constant leaves the *t* unchanged
    (mean and SE scale together), which is exactly why this — not the levered P&L — is the
    honest test of whether the carry is real.
    """
    d = np.asarray(daily, dtype=float)
    se = nw_se(d, lag)
    if not se or np.isnan(se):
        return float("nan")
    return float(d.mean() / se)


def sharpe(daily: np.ndarray, ann: int = TRADING_DAYS) -> float:
    """Annualised Sharpe of a daily return array (leverage-invariant)."""
    d = np.asarray(daily, dtype=float)
    sd = d.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(d.mean() / sd * np.sqrt(ann))


def sharpe_t(daily: np.ndarray, ann: int = TRADING_DAYS) -> float:
    """Approximate *t* of the annualised Sharpe (≈ Sharpe × √years)."""
    d = np.asarray(daily, dtype=float)
    years = len(d) / ann
    return float(sharpe(d, ann) * np.sqrt(years))


def placebo_pvalue(frame: pd.DataFrame, n_draws: int = 5_000, seed: int = 382) -> dict:
    """Sign-randomisation placebo: flip the carry's sign at random and ask how often a
    randomised series produces a Sharpe at least as high as the real one.

    The honest "could autocorrelated noise fake this?" test. Returns the observed Sharpe,
    the placebo mean Sharpe, and ``p = P[placebo Sharpe >= observed]``.
    """
    r = frame["ret"].values
    obs = sharpe(r)
    rng = np.random.default_rng(seed)
    n = len(r)
    out = np.empty(n_draws)
    for i in range(n_draws):
        sign = rng.choice((-1.0, 1.0), size=n)
        out[i] = sharpe(sign * r)
    return {"obs_sharpe": float(obs), "placebo_sharpe": float(out.mean()),
            "p_value": float((out >= obs).mean())}


# --------------------------------------------------------------------------- #
# Tail / funding-shock — where leverage bites
# --------------------------------------------------------------------------- #
def tail_stats(daily: np.ndarray, q: float = 0.01) -> dict:
    """Tail diagnostics of a daily levered return: skew, excess kurtosis, VaR & CVaR at ``q``,
    worst single day, and worst rolling 5-day sum (a funding-shock window)."""
    d = np.asarray(daily, dtype=float)
    n = len(d)
    mu, sd = d.mean(), d.std(ddof=1)
    z = (d - mu) / sd
    skew = float((z ** 3).mean())
    exkurt = float((z ** 4).mean() - 3.0)
    var = float(np.quantile(d, q))
    cvar = float(d[d <= var].mean()) if (d <= var).any() else float("nan")
    worst1 = float(d.min())
    worst5 = float(pd.Series(d).rolling(5).sum().min())
    return {"skew": skew, "exkurt": exkurt, "var": var, "cvar": cvar,
            "worst_day": worst1, "worst_5day": worst5, "n": n}


def leverage_table(frame: pd.DataFrame, leverages=(1, 50, 100), q: float = 0.01) -> list:
    """For each leverage: annualised return, Sharpe (constant!), max drawdown, worst day,
    and CVaR — the table that shows Sharpe frozen while the tail explodes."""
    rows = []
    for L in leverages:
        d = levered_returns(frame, float(L))
        t = tail_stats(d, q=q)
        rows.append({
            "leverage": int(L),
            "ann_return": float(d.mean() * TRADING_DAYS),
            "sharpe": sharpe(d),
            "max_dd": max_drawdown(d),
            "worst_day": t["worst_day"],
            "cvar": t["cvar"],
        })
    return rows


def funding_shock(frame: pd.DataFrame, start: str, end: str,
                  leverages=(1, 50, 100)) -> dict:
    """Cumulative levered P&L of the basis over a named stress window (e.g. March 2020).

    Returns the unlevered cumulative return over ``[start, end]`` and the levered outcomes —
    the concrete "a repo spike force-unwinds the book" number behind the Tradability mirage.
    """
    seg = frame.loc[start:end, "ret"].values
    unlev = float(np.prod(1.0 + seg) - 1.0)
    out = {"start": start, "end": end, "n_days": len(seg), "unlev": unlev}
    for L in leverages:
        out[f"L{int(L)}"] = float(np.prod(1.0 + L * seg) - 1.0)
    return out


# --------------------------------------------------------------------------- #
# Costs — turnover is tiny, so costs are NOT the binding constraint
# --------------------------------------------------------------------------- #
def net_of_costs(frame: pd.DataFrame, leverage: float = 50.0,
                 roll_per_year: int = 4, cost_bps: float = 1.0) -> dict:
    """Charge a one-way cost on the quarterly futures roll (the basis trade's main turnover).

    A basis position is rolled ~4×/yr; each roll pays a one-way cost on the levered notional.
    Returns the gross and net annualised levered return — the point being that costs barely
    move the number (the trade dies on the *tail*, not on costs)."""
    d = levered_returns(frame, leverage)
    gross_ann = float(d.mean() * TRADING_DAYS)
    cost_ann = roll_per_year * (cost_bps / 1e4) * leverage   # one-way × levered notional
    return {"leverage": leverage, "gross_ann": gross_ann,
            "net_ann": gross_ann - cost_ann, "cost_ann": cost_ann}


def summarize(frame: pd.DataFrame, lag: int = 21) -> dict:
    """Headline Signal-axis stats (all leverage-invariant): n, years, mean daily, HAC *t*,
    Sharpe and its *t*, and the unlevered annualised return."""
    r = frame["ret"].values
    n = len(r)
    return {
        "n": n,
        "years": n / TRADING_DAYS,
        "mean_daily": float(r.mean()),
        "hac_t": hac_t(r, lag),
        "sharpe": sharpe(r),
        "sharpe_t": sharpe_t(r),
        "ann_return": float(r.mean() * TRADING_DAYS),
    }
