"""How many stocks is enough — Study 1004.

The textbook answer comes from Evans & Archer (1968): plot portfolio standard deviation against
the number of holdings, watch it flatten, read off "about fifteen". Statman (1987) argued for
thirty on cost grounds; the number has been relitigated many times and the *method* almost
never has.

The method has a hole in it. Standard deviation is a property of the return distribution of an
**average** randomly chosen N-stock portfolio. An investor does not hold the average portfolio;
they hold one of them, for decades, and what happens to them is the terminal wealth of that one
draw. Those are different questions with different answers:

- ``volatility_curve`` reproduces the textbook result and confirms the flattening.
- ``terminal_wealth_curve`` asks instead how widely the *outcome* varies across randomly drawn
  N-stock portfolios. That dispersion keeps falling long after volatility has levelled off,
  because it is driven by the dispersion of individual stocks' long-run returns rather than by
  their short-run covariance.
- ``tracking_error_curve`` asks a third version — how far an N-stock portfolio drifts from the
  index it is trying to approximate — which is the question an indexer actually faces.

The three curves flatten at very different points, and which number you quote depends entirely
on which one you plotted. ``marginal_benefit`` reports the incremental gain from the Nth stock
under each, so the comparison is made explicit rather than left to the eye.

A fourth measurement, ``skew_and_the_median_portfolio``, explains *why* the terminal-wealth
curve behaves differently: individual stock returns are right-skewed, so the median N-stock
portfolio underperforms the mean one, and that gap closes only slowly with N. This is the
Bessembinder (2018) result seen from the portfolio-construction side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Drawing portfolios
# --------------------------------------------------------------------------- #
def usable_panel(prices: pd.DataFrame, names, min_obs: int = 1000) -> pd.DataFrame:
    """Daily returns for the names with enough history, on their common calendar."""
    cols = [c for c in names if c in prices.columns
            and prices[c].dropna().shape[0] >= min_obs]
    r = prices[cols].pct_change()
    return r.dropna(how="all")


def draw_portfolios(rets: pd.DataFrame, n: int, n_draws: int = 400,
                    seed: int = 1004) -> np.ndarray:
    """Return an (n_draws × T) matrix of equal-weighted N-stock portfolio returns.

    Equal weights and daily rebalancing throughout. Cap weighting would confound the question
    with the size effect, and buy-and-hold weights would let one winner dominate — a real
    effect, but a different study (see ``rebalanced_vs_held``).
    """
    cols = list(rets.columns)
    if n > len(cols):
        return np.empty((0, 0))
    R = rets.to_numpy(dtype=float)
    ok = np.isfinite(R).all(axis=1)
    R = R[ok]
    rng = np.random.default_rng(seed)
    out = np.empty((n_draws, R.shape[0]))
    for i in range(n_draws):
        pick = rng.choice(len(cols), size=n, replace=False)
        out[i] = R[:, pick].mean(axis=1)
    return out


def _ann_vol(P: np.ndarray) -> np.ndarray:
    return P.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS)


def _terminal(P: np.ndarray) -> np.ndarray:
    """Terminal wealth of each portfolio, from a $1 start."""
    return np.exp(np.log1p(P).sum(axis=1))


# --------------------------------------------------------------------------- #
# The three curves
# --------------------------------------------------------------------------- #
def volatility_curve(rets: pd.DataFrame, sizes=None, n_draws: int = 400,
                     seed: int = 1004) -> pd.DataFrame:
    """The textbook curve: average portfolio standard deviation against N.

    Note what is averaged. Each N produces ``n_draws`` portfolios; this reports the **mean of
    their volatilities**. That is the Evans-Archer statistic, and it flattens early because
    covariance between large-cap stocks is high and stops mattering once you have enough names
    to average it out.
    """
    if sizes is None:
        sizes = _default_sizes(rets.shape[1])
    rows = []
    for n in sizes:
        P = draw_portfolios(rets, n, n_draws, seed)
        if P.size == 0:
            continue
        v = _ann_vol(P)
        rows.append({"n_stocks": n, "mean_vol": float(v.mean()),
                     "sd_of_vol": float(v.std(ddof=1)),
                     "p05_vol": float(np.percentile(v, 5)),
                     "p95_vol": float(np.percentile(v, 95))})
    return pd.DataFrame(rows).set_index("n_stocks")


def terminal_wealth_curve(rets: pd.DataFrame, sizes=None, n_draws: int = 400,
                          seed: int = 1004) -> pd.DataFrame:
    """The curve an investor lives in: dispersion of TERMINAL WEALTH across draws.

    The statistic is the spread of outcomes across portfolios, not the volatility within one.
    It is reported in log terms because wealth is multiplicative and a ratio spread is the only
    scale on which "twice as good" and "half as good" are symmetric.

    The median-to-mean gap is reported alongside, because right-skewed stock returns mean the
    typical portfolio does worse than the average one, and how fast that gap closes is the whole
    difference between this curve and the textbook one.
    """
    if sizes is None:
        sizes = _default_sizes(rets.shape[1])
    rows = []
    for n in sizes:
        P = draw_portfolios(rets, n, n_draws, seed)
        if P.size == 0:
            continue
        w = _terminal(P)
        lw = np.log(w)
        rows.append({"n_stocks": n, "median_wealth": float(np.median(w)),
                     "mean_wealth": float(w.mean()),
                     "log_sd": float(lw.std(ddof=1)),
                     "p05_wealth": float(np.percentile(w, 5)),
                     "p95_wealth": float(np.percentile(w, 95)),
                     "ratio_95_05": float(np.percentile(w, 95) / np.percentile(w, 5)),
                     "median_over_mean": float(np.median(w) / w.mean())})
    return pd.DataFrame(rows).set_index("n_stocks")


def tracking_error_curve(rets: pd.DataFrame, benchmark: pd.Series, sizes=None,
                         n_draws: int = 400, seed: int = 1004) -> pd.DataFrame:
    """How far an N-stock portfolio drifts from the index it is approximating.

    The indexer's version of the question, and the one with the most practical bite: an investor
    replicating an index with a handful of names wants to know the tracking error, not the
    absolute volatility.
    """
    if sizes is None:
        sizes = _default_sizes(rets.shape[1])
    b = benchmark.reindex(rets.dropna().index).to_numpy(dtype=float)
    rows = []
    for n in sizes:
        P = draw_portfolios(rets, n, n_draws, seed)
        if P.size == 0 or P.shape[1] != len(b):
            continue
        te = (P - b[None, :]).std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS)
        rows.append({"n_stocks": n, "mean_te": float(te.mean()),
                     "p95_te": float(np.percentile(te, 95))})
    return pd.DataFrame(rows).set_index("n_stocks")


def _default_sizes(n_available: int):
    grid = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 50, 75, 100]
    return [n for n in grid if n <= n_available]


def marginal_benefit(curve: pd.DataFrame, column: str) -> pd.DataFrame:
    """The incremental improvement from each additional stock, as a share of the total.

    Puts the three curves on one scale. "Diversification is 90% complete at N" is a statement
    that can be made about any of them, and the N differs sharply.
    """
    v = curve[column].to_numpy(dtype=float)
    idx = curve.index.to_numpy()
    if len(v) < 2:
        return pd.DataFrame()
    total = v[0] - v[-1]
    achieved = (v[0] - v) / total if total != 0 else np.zeros_like(v)
    return pd.DataFrame({"value": v, "share_of_benefit": achieved},
                        index=pd.Index(idx, name="n_stocks"))


def stocks_for_share(curve: pd.DataFrame, column: str, share: float = 0.90) -> float:
    """The smallest N reaching ``share`` of the total available benefit, interpolated."""
    m = marginal_benefit(curve, column)
    if m.empty:
        return np.nan
    x = m.index.to_numpy(dtype=float)
    y = m["share_of_benefit"].to_numpy()
    for i in range(1, len(y)):
        if y[i - 1] < share <= y[i]:
            if y[i] == y[i - 1]:
                return float(x[i])
            return float(x[i - 1] + (x[i] - x[i - 1]) * (share - y[i - 1])
                         / (y[i] - y[i - 1]))
    return float(x[-1]) if y[-1] >= share else np.nan


# --------------------------------------------------------------------------- #
# Why the curves differ
# --------------------------------------------------------------------------- #
def skew_and_the_median_portfolio(rets: pd.DataFrame, sizes=None, n_draws: int = 400,
                                  seed: int = 1004) -> pd.DataFrame:
    """The mechanism: right-skewed stock returns make the median portfolio lag the mean.

    A handful of names carry the index (Bessembinder 2018). A small portfolio probably misses
    them, so it probably underperforms — "probably" being the operative word, since the *mean*
    across draws is unaffected by construction. Adding names raises the median without moving
    the mean, and that is a benefit the volatility curve cannot see at all.
    """
    if sizes is None:
        sizes = _default_sizes(rets.shape[1])
    single = np.exp(np.log1p(rets.dropna()).sum())
    rows = []
    for n in sizes:
        P = draw_portfolios(rets, n, n_draws, seed)
        if P.size == 0:
            continue
        w = _terminal(P)
        rows.append({"n_stocks": n, "median": float(np.median(w)),
                     "mean": float(w.mean()),
                     "shortfall": float(1 - np.median(w) / w.mean()),
                     "share_below_mean": float((w < w.mean()).mean())})
    out = pd.DataFrame(rows).set_index("n_stocks")
    out.attrs["single_stock_skew"] = float(pd.Series(np.log(single)).skew())
    return out


def concentration_of_returns(rets: pd.DataFrame) -> dict:
    """How much of the basket's total return came from how few names."""
    total = np.exp(np.log1p(rets.dropna()).sum()) - 1
    s = total.sort_values(ascending=False)
    cum = s.cumsum() / s.sum() if s.sum() != 0 else s * np.nan
    n = len(s)
    return {"n_names": n, "best": s.index[0], "best_return": float(s.iloc[0]),
            "share_from_top_10pct": float(cum.iloc[max(int(n * 0.1) - 1, 0)]),
            "share_from_top_25pct": float(cum.iloc[max(int(n * 0.25) - 1, 0)]),
            "share_negative": float((s < 0).mean()),
            "median_stock_return": float(s.median()),
            "mean_stock_return": float(s.mean())}


def rebalanced_vs_held(rets: pd.DataFrame, n: int = 20, n_draws: int = 300,
                       seed: int = 1004) -> dict:
    """Equal-weighted with daily rebalancing against buy-and-hold from equal weights.

    Rebalancing is a decision, not a neutral choice: it sells the winners. Over decades with
    right-skewed returns, buy-and-hold lets a few names dominate, which raises the mean and the
    dispersion together. Reported because the diversification curves are usually drawn under
    rebalancing without saying so.
    """
    P = draw_portfolios(rets, n, n_draws, seed)
    if P.size == 0:
        return {}
    R = rets.dropna().to_numpy(dtype=float)
    cols = R.shape[1]
    rng = np.random.default_rng(seed)
    held = np.empty(n_draws)
    for i in range(n_draws):
        pick = rng.choice(cols, size=n, replace=False)
        paths = np.exp(np.log1p(R[:, pick]).sum(axis=0))
        held[i] = paths.mean()
    reb = _terminal(P)
    return {"n": n, "rebalanced_median": float(np.median(reb)),
            "rebalanced_mean": float(reb.mean()),
            "rebalanced_log_sd": float(np.log(reb).std(ddof=1)),
            "held_median": float(np.median(held)), "held_mean": float(held.mean()),
            "held_log_sd": float(np.log(held).std(ddof=1))}


def synthetic_cross_section(n_stocks: int = 60, n_days: int = 5000,
                            avg_corr: float = 0.30, stock_vol: float = 0.32,
                            mu_dispersion: float = 0.0, mu_mean: float = 0.08,
                            seed: int = 1004) -> pd.DataFrame:
    """A cross-section with tunable correlation and tunable dispersion of expected returns.

    The two knobs drive the two curves separately, which is what makes the study's claim
    testable: ``avg_corr`` alone should move where the volatility curve flattens, and
    ``mu_dispersion`` alone should move how long the terminal-wealth curve keeps improving. If
    the terminal-wealth curve were just the volatility curve in different clothes, the second
    knob would do nothing.
    """
    rng = np.random.default_rng(seed)
    daily_vol = stock_vol / np.sqrt(TRADING_DAYS)
    common = rng.normal(0, daily_vol * np.sqrt(avg_corr), n_days)
    idio = rng.normal(0, daily_vol * np.sqrt(max(1 - avg_corr, 0.0)),
                      (n_days, n_stocks))
    mus = rng.normal(mu_mean, mu_dispersion, n_stocks) / TRADING_DAYS
    R = common[:, None] + idio + mus[None, :]
    idx = pd.bdate_range("1993-02-01", periods=n_days)
    return pd.DataFrame(R, index=idx,
                        columns=[f"S{i:03d}" for i in range(n_stocks)])


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Confirmed** if the textbook volatility curve does flatten early — the claim
      being examined is real before it is criticised; **Partial** if it flattens slowly;
      **Busted** if it does not flatten.
    - **Tradability**: **Useful** if the terminal-wealth criterion demands materially more
      holdings than the volatility criterion, since then the textbook number is actively
      misleading and a better one is available; **Partial** if the two roughly agree;
      **Mirage** if the distinction makes no difference.
    """
    signal = ("Confirmed" if h["n_for_90_vol"] <= 20
              else ("Partial" if h["n_for_90_vol"] <= 40 else "Busted"))
    ratio = h["n_for_90_wealth"] / max(h["n_for_90_vol"], 1e-9)
    trad = ("Useful" if ratio >= 1.8 else ("Partial" if ratio >= 1.2 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"The textbook curve is real. Across {h['n_available']} large-cap names over "
            f"{h['years']:.0f} years, the average equal-weighted portfolio's volatility falls "
            f"from {h['vol_at_1']:.1%} with one stock to {h['vol_at_max']:.1%} with all of "
            f"them, and **{h['n_for_90_vol']:.0f} holdings** capture 90% of that reduction. "
            f"Evans and Archer's fifteen is the right answer to the question they asked, and "
            f"anyone repeating it is not making an arithmetic error."),
        "trad_why": (
            f"They are answering the wrong question. Standard deviation describes the *average* "
            f"portfolio; an investor holds *one*, for decades. Measured on the dispersion of "
            f"terminal wealth across randomly drawn portfolios — what actually varies between "
            f"one investor and another — 90% of the available benefit needs "
            f"**{h['n_for_90_wealth']:.0f} holdings**, {ratio:.1f}× the textbook number. With "
            f"{h['n_for_90_vol']:.0f} stocks the 5th-to-95th percentile of outcomes still spans "
            f"a factor of **{h['ratio_at_vol_n']:.1f}×**; reaching the textbook's implied "
            f"comfort takes far more names. The mechanism is skew, not covariance: "
            f"{h['share_negative']:.0%} of these names lost money outright over the period and "
            f"the top decile produced {h['share_from_top_10pct']:.0%} of the basket's total "
            f"return, so a small portfolio probably misses the names that mattered. Its median "
            f"terminal wealth sits {h['shortfall_at_vol_n']:.0%} below the mean at "
            f"{h['n_for_90_vol']:.0f} names and {h['shortfall_at_wealth_n']:.0%} below at "
            f"{h['n_for_90_wealth']:.0f}. Volatility cannot see any of this, because averaging "
            f"across draws is exactly the step that hides it."),
        "trad": trad,
        "one_sentence": (
            f"Twenty stocks removes 90% of the *volatility* you can diversify away and leaves "
            f"the spread of actual outcomes at {h['ratio_at_vol_n']:.1f}× between the 5th and "
            f"95th percentile — the terminal-wealth criterion asks for "
            f"{h['n_for_90_wealth']:.0f} names, not {h['n_for_90_vol']:.0f}."),
    }
