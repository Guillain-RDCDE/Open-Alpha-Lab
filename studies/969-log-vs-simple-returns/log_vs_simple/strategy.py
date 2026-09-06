"""Return conventions and the errors they cause — Study 969.

Two definitions of "the return", both correct, neither interchangeable:

    simple:  R_t = P_t / P_{t-1} - 1
    log:     r_t = ln(P_t / P_{t-1}) = ln(1 + R_t)

Everything in this module follows from one Taylor expansion,
``ln(1+R) = R - R^2/2 + R^3/3 - ...``, and from two aggregation facts:

- **Across time, logs add.** ``sum(r) = ln(P_T / P_0)`` exactly, for any path. Simple returns
  do not add; they compound.
- **Across assets, simple returns are linear.** A portfolio's simple return is
  ``sum_i w_i R_i`` exactly. The log return of a portfolio is **not** the weighted average of
  the log returns — and the gap is Jensen's inequality, not a rounding error.

The four places this actually bites, each with a function that measures it:

1. ``drag_table`` — the mean log return sits below the mean simple return by roughly
   ``sigma^2 / 2`` per period. That difference is *volatility drag*, and it is the whole
   distance between an arithmetic and a geometric average return.
2. ``portfolio_error`` — building a portfolio return by weighting **log** returns. Always
   understates. On a two-asset rebalanced book the error is a rebalancing bonus that has been
   thrown away.
3. ``sharpe_gap`` — a Sharpe computed on log returns versus on simple returns. Both are
   defensible; they are not the same number, and the difference grows with volatility.
4. ``annualisation_table`` — the four ways people annualise a mean, and which of them is a
   claim about the median rather than the mean.

Nothing here is an anomaly and none of it is arguable — it is arithmetic. What is genuinely
open, and what this study measures, is **how large the error is on real tapes**, which is the
only thing that decides whether a convention mismatch is pedantry or a bug.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# The two conventions
# --------------------------------------------------------------------------- #
def simple_returns(prices: pd.Series | pd.DataFrame):
    """``P_t / P_{t-1} - 1`` — what actually lands in the account."""
    return prices.pct_change()


def log_returns(prices: pd.Series | pd.DataFrame):
    """``ln(P_t / P_{t-1})`` — what adds up across time."""
    return np.log(prices / prices.shift(1))


def convert_simple_to_log(r):
    """Exact conversion (not the approximation): ``ln(1 + R)``."""
    return np.log1p(r)


def convert_log_to_simple(r):
    """Exact conversion: ``exp(r) - 1``."""
    return np.expm1(r)


# --------------------------------------------------------------------------- #
# 1) Volatility drag: the gap between the two means
# --------------------------------------------------------------------------- #
def drag_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Per tape: mean simple, mean log, the gap, and the ``sigma^2/2`` prediction.

    The classic second-order result is ``E[ln(1+R)] ~= E[R] - Var(R)/2``. This table checks it
    on real data, where returns are neither small nor Gaussian, so the approximation's own
    error is visible: the residual column is the part that ``sigma^2/2`` does not explain, and
    it grows with skewness and kurtosis.
    """
    rows = []
    for c in prices.columns:
        p = prices[c].dropna()
        if len(p) < 500:
            continue
        R = p.pct_change().dropna()
        r = np.log(p / p.shift(1)).dropna()
        var = float(R.var(ddof=1))
        gap = float(R.mean() - r.mean())
        rows.append({
            "ticker": c, "n": int(len(R)),
            "vol_ann": float(R.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            "mean_simple_ann": float(R.mean() * TRADING_DAYS),
            "mean_log_ann": float(r.mean() * TRADING_DAYS),
            "gap_ann": float(gap * TRADING_DAYS),
            "half_var_ann": float(var / 2 * TRADING_DAYS),
            "residual_ann": float((gap - var / 2) * TRADING_DAYS),
            "cagr": float((p.iloc[-1] / p.iloc[0]) ** (TRADING_DAYS / len(R)) - 1),
            "skew": float(R.skew()), "excess_kurtosis": float(R.kurtosis()),
        })
    return pd.DataFrame(rows).set_index("ticker")


def cagr_from_means(mean_simple: float, variance: float, periods: int = TRADING_DAYS) -> float:
    """The textbook geometric-return approximation, ``(mu - var/2)`` annualised and exponentiated."""
    return float(np.exp((mean_simple - variance / 2) * periods) - 1)


# --------------------------------------------------------------------------- #
# 2) The portfolio mistake
# --------------------------------------------------------------------------- #
def portfolio_simple(rets: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """The correct daily portfolio return: weights applied to **simple** returns."""
    return pd.Series(rets.to_numpy() @ weights, index=rets.index, name="portfolio")


def portfolio_from_logs(log_rets: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """The common mistake: weighting **log** returns as if they were linear in the assets."""
    return pd.Series(log_rets.to_numpy() @ weights, index=log_rets.index, name="portfolio_wrong")


def portfolio_error(prices: pd.DataFrame, tickers: tuple[str, ...],
                    weights: np.ndarray | None = None) -> dict:
    """How much a daily-rebalanced portfolio loses to the log-weighting mistake.

    The two series are compared as terminal wealth over the whole sample, which is where the
    error compounds. The gap is bounded below by Jensen: the log of a weighted average is at
    least the weighted average of the logs, so the mistake **always** understates — this is a
    directional error, not noise, which is what makes it dangerous.
    """
    px = prices[list(tickers)].dropna()
    R = px.pct_change().dropna()
    L = np.log(px / px.shift(1)).dropna()
    n = len(tickers)
    w = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, dtype=float)
    right = portfolio_simple(R, w)
    wrong_log = portfolio_from_logs(L, w)
    wrong = np.expm1(wrong_log)   # convert back so both are simple returns
    years = len(R) / TRADING_DAYS
    grow_right = float((1 + right).prod())
    grow_wrong = float((1 + wrong).prod())
    return {
        "n": int(len(R)), "years": float(years), "weights": w.tolist(),
        "cagr_correct": float(grow_right ** (1 / years) - 1),
        "cagr_wrong": float(grow_wrong ** (1 / years) - 1),
        "cagr_gap": float(grow_right ** (1 / years) - grow_wrong ** (1 / years)),
        "terminal_ratio": float(grow_right / grow_wrong),
        "mean_daily_gap_bps": float((right - wrong).mean() * 1e4),
        "always_understates": bool((right >= wrong - 1e-15).mean() > 0.99),
    }


def rebalancing_bonus(prices: pd.DataFrame, tickers: tuple[str, ...]) -> dict:
    """The related, real effect: a rebalanced book grows faster than the weighted *geometric*
    average of its holdings — Fernholz's excess growth rate, also called the diversification
    return (Booth & Fama 1992).

    The benchmark has to be the **weighted geometric** average, not the arithmetic one, and
    the distinction is the whole reason this function exists next to
    :func:`portfolio_error`. Weighting log returns and exponentiating gives exactly
    ``prod_i (1 + g_i)^{w_i} - 1`` where ``g_i`` is asset *i*'s own CAGR — so the
    "log-weighting mistake" *is* the diversification return, discarded. Against the
    arithmetic average the comparison is a different (and usually smaller, sometimes
    negative) number, which is reported too so the two are never confused.
    """
    px = prices[list(tickers)].dropna()
    R = px.pct_change().dropna()
    n = len(tickers)
    w = np.full(n, 1.0 / n)
    years = len(R) / TRADING_DAYS
    reb = float(((1 + portfolio_simple(R, w)).prod()) ** (1 / years) - 1)
    holds = [(float(px[c].iloc[-1] / px[c].iloc[0]) ** (1 / years) - 1) for c in tickers]
    geo = float(np.prod([(1 + g) ** wi for g, wi in zip(holds, w)]) - 1)
    return {"rebalanced_cagr": reb,
            "buy_hold_geometric_cagr": geo,
            "buy_hold_avg_cagr": float(np.mean(holds)),
            "bonus": float(reb - geo),
            "bonus_vs_arithmetic": float(reb - np.mean(holds)),
            "per_asset_cagr": dict(zip(tickers, holds))}


# --------------------------------------------------------------------------- #
# 3) Statistics that move
# --------------------------------------------------------------------------- #
def sharpe_gap(prices: pd.DataFrame) -> pd.DataFrame:
    """Annualised Sharpe on simple returns versus on log returns, per tape."""
    rows = []
    for c in prices.columns:
        p = prices[c].dropna()
        if len(p) < 500:
            continue
        R, L = p.pct_change().dropna(), np.log(p / p.shift(1)).dropna()
        s_simple = float(R.mean() / R.std(ddof=1) * np.sqrt(TRADING_DAYS))
        s_log = float(L.mean() / L.std(ddof=1) * np.sqrt(TRADING_DAYS))
        rows.append({"ticker": c, "sharpe_simple": s_simple, "sharpe_log": s_log,
                     "gap": s_simple - s_log,
                     "relative_gap": (s_simple - s_log) / abs(s_simple) if s_simple else np.nan,
                     "vol_ann": float(R.std(ddof=1) * np.sqrt(TRADING_DAYS))})
    return pd.DataFrame(rows).set_index("ticker")


def beta_gap(prices: pd.DataFrame, asset: str, market: str) -> dict:
    """A regression slope computed both ways — the case where the difference is tiny."""
    px = prices[[asset, market]].dropna()
    R = px.pct_change().dropna()
    L = np.log(px / px.shift(1)).dropna()
    def slope(df):
        x, y = df[market].to_numpy(), df[asset].to_numpy()
        return float(np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1))
    b_s, b_l = slope(R), slope(L)
    return {"beta_simple": b_s, "beta_log": b_l, "gap": b_s - b_l,
            "relative_gap": (b_s - b_l) / b_s if b_s else np.nan, "n": int(len(R))}


def annualisation_table(r_daily_simple: pd.Series) -> pd.DataFrame:
    """The four annualisations people write down, and what each one actually claims."""
    R = r_daily_simple.dropna()
    L = np.log1p(R)
    mu, var = float(R.mean()), float(R.var(ddof=1))
    rows = [
        {"method": "arithmetic x 252", "value": mu * TRADING_DAYS,
         "claims": "the expected return of one year, if you never compound"},
        {"method": "(1 + mean)^252 - 1", "value": (1 + mu) ** TRADING_DAYS - 1,
         "claims": "the return of a hypothetical path with no volatility at all"},
        {"method": "exp(mean log x 252) - 1", "value": float(np.expm1(L.mean() * TRADING_DAYS)),
         "claims": "the geometric mean: the CAGR the path actually delivered"},
        {"method": "exp((mu - var/2) x 252) - 1", "value": cagr_from_means(mu, var),
         "claims": "the same thing, second-order approximation"},
    ]
    return pd.DataFrame(rows).set_index("method")


def drag_curve(vols=np.linspace(0.0, 1.2, 25), mu_ann: float = 0.08) -> pd.DataFrame:
    """The textbook curve: geometric return falls away from arithmetic as ``sigma^2/2``."""
    rows = []
    for v in vols:
        rows.append({"vol_ann": float(v), "arithmetic": mu_ann,
                     "geometric_approx": float(mu_ann - v ** 2 / 2)})
    return pd.DataFrame(rows).set_index("vol_ann")


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal** (does the convention move the numbers?): **Real** if the largest annualised
      mean-return gap across the universe exceeds 5 percentage points; **Weak** above 1 point;
      **None** below.
    - **Usefulness** (is there a rule?): **Useful** if the log-weighting mistake is
      *directional* (it understates on essentially every day, so the fix is unambiguous);
      **Fragile** if the sign is mixed; **Mirage** if the effect is unmeasurable.
    """
    gap = h["max_gap_ann"]
    signal = "Real" if gap >= 0.05 else ("Weak" if gap >= 0.01 else "None")
    trad = "Useful" if h["understates_always"] else ("Fragile" if gap >= 0.01 else "Mirage")
    return {
        "signal": signal,
        "signal_why": (
            f"It scales with the square of volatility, so it is invisible at one end of the "
            f"tape and enormous at the other. On **{h['calm_ticker']}** "
            f"({h['calm_vol']:.0%} annualised vol) the mean simple and mean log returns differ "
            f"by **{h['calm_gap']:.2%}/yr**; on **{h['wild_ticker']}** ({h['wild_vol']:.0%} vol) "
            f"they differ by **{h['max_gap_ann']:.1%}/yr**, and the sigma-squared-over-two "
            f"prediction accounts for {h['half_var_explains']:.0%} of it. The Sharpe ratio moves "
            f"too: up to **{h['max_sharpe_gap']:.2f}** between the two conventions."),
        "trad": trad,
        "trad_why": (
            f"Yes, and it is two lines. **Across time, use logs** (they add, and their mean "
            f"exponentiates to the CAGR the path delivered). **Across assets, use simple "
            f"returns** (a portfolio's return is a weighted average of them, exactly). The "
            f"failure mode is weighting log returns: on an equal-weight book of "
            f"{h['portfolio_n']} tapes it understated the CAGR by "
            f"**{h['portfolio_cagr_gap']:.2%}/yr** — {h['portfolio_terminal_ratio']:.2f}x of "
            f"terminal wealth over {h['portfolio_years']:.0f} years — and it understated on "
            f"**{'essentially every day' if h['understates_always'] else 'most days'}**, "
            f"because Jensen's inequality only points one way."),
        "one_sentence": (
            f"The two conventions differ by about half the variance, which is nothing on a "
            f"bond fund ({h['calm_gap']:.2%}/yr) and **{h['max_gap_ann']:.0%}/yr** on bitcoin — "
            f"so the rule is not 'pick one', it is 'logs across time, simple across assets', "
            f"and the mistake worth hunting in a codebase is a portfolio built by weighting "
            f"log returns."),
    }
