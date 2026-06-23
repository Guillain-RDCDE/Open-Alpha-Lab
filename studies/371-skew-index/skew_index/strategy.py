"""Strategy + inference for Study 371 — CBOE SKEW index vs forward SPY.

The claim: a high **SKEW** (CBOE's "black-swan" gauge — the relative price of out-of-the-money
S&P 500 puts) **warns you before crashes** the way the symmetric, at-the-money **VIX** can't.
Two testable readings:

  * **Directional.** Does a high SKEW predict *weaker* (or more negative) forward SPY returns?
    We bucket days by SKEW level, compute forward 1/3-month returns with a 1-day execution lag,
    and run a Welch *t* of the top-decile bucket against the rest, plus a placebo null sized to
    the bucket count.
  * **Tail-warning.** Does a high SKEW raise the probability of a forward *tail event* (a large
    drawdown over the next month)? We compare the tail-event rate after high-SKEW days to the
    unconditional base rate.

The decisive question is **incremental information beyond VIX**: we regress forward returns on
*both* SKEW and VIX and read the HAC (Newey-West) *t* on the SKEW coefficient — if VIX already
prices everything, SKEW's own coefficient is indistinguishable from zero. A simple long/short
"fade high SKEW" sleeve is costed (1-day lag + one-way costs × turnover) for the Tradability axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = {1: 21, 3: 63}                # months -> trading days


# --------------------------------------------------------------------------- #
# Forward returns & tail events
# --------------------------------------------------------------------------- #
def forward_return(spy: pd.Series, horizon_days: int, lag: int = 1) -> pd.Series:
    """Forward ``horizon_days`` SPY return entered ``lag`` days after the signal date.

    Aligned to the *signal* date: value at date ``d`` is the return from the close ``lag`` days
    after ``d`` to ``lag + horizon`` days after ``d`` (no look-ahead). NaN near the tail.
    """
    entry = spy.shift(-lag)
    exit_ = spy.shift(-(lag + horizon_days))
    return exit_ / entry - 1.0


def forward_min_return(spy: pd.Series, horizon_days: int, lag: int = 1) -> pd.Series:
    """Worst close-to-close drawdown over the forward window (min path return).

    Value at date ``d`` is ``min_{1<=h<=horizon} spy[entry+h]/spy[entry] - 1`` with entry one
    ``lag`` after ``d``. Used for the tail-warning test.
    """
    n = len(spy)
    v = spy.values
    out = np.full(n, np.nan)
    for i in range(n):
        e = i + lag
        if e + horizon_days >= n:
            continue
        window = v[e + 1:e + 1 + horizon_days] / v[e] - 1.0
        out[i] = window.min()
    return pd.Series(out, index=spy.index)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def welch_t(sample: np.ndarray, base: np.ndarray) -> float:
    """Welch t of ``mean(sample) - mean(base)`` (unequal variance). NaN if sample < 2."""
    sample = np.asarray(sample, float)
    base = np.asarray(base, float)
    if len(sample) < 2 or len(base) < 2:
        return float("nan")
    m1, m0 = sample.mean(), base.mean()
    v1 = sample.var(ddof=1) / len(sample)
    v0 = base.var(ddof=1) / len(base)
    se = np.sqrt(v1 + v0)
    if se == 0:
        return float("nan")
    return float((m1 - m0) / se)


def placebo_pvalue(values: np.ndarray, k: int, obs_mean: float,
                   n_draws: int = 20_000, seed: int = 371,
                   tail: str = "low") -> float:
    """Small-sample placebo null: draw ``k`` random values many times and ask how often a random
    draw's mean is **at least as extreme** as ``obs_mean``.

    ``tail='low'`` (the believers' direction: high SKEW => *worse* forward returns) returns
    P[random-draw mean <= obs_mean]; ``tail='high'`` returns P[random-draw mean >= obs_mean].
    """
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    if len(values) == 0 or k == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for i in range(n_draws):
        means[i] = values[rng.integers(0, len(values), size=k)].mean()
    if tail == "low":
        return float((means <= obs_mean).mean())
    return float((means >= obs_mean).mean())


def _hac_se(x: np.ndarray, resid: np.ndarray, xtx_inv: np.ndarray,
            lags: int) -> np.ndarray:
    """Newey-West HAC standard errors for an OLS with design ``x`` (n x p) and residuals.

    Returns the vector of HAC standard errors of the coefficients. ``xtx_inv`` is (X'X)^-1.
    """
    n, p = x.shape
    # meat: sum of weighted autocovariances of the score x_i * e_i
    s = x * resid[:, None]
    S = s.T @ s
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = s[L:].T @ s[:-L]
        S += w * (G + G.T)
    cov = xtx_inv @ S @ xtx_inv
    return np.sqrt(np.diag(cov))


def regress_forward(frame: pd.DataFrame, months: int, lag: int = 1,
                    hac_lags: int | None = None) -> dict:
    """Regress forward ``months``-month SPY return on standardised SKEW and VIX (both lagged via
    the forward-return alignment). Returns coefficients and **HAC (Newey-West)** t-stats.

    The decisive object is ``t_skew``: the SKEW coefficient *controlling for VIX*. If VIX already
    prices the forward distribution, SKEW's own t is indistinguishable from zero — SKEW is not
    informative beyond VIX. Overlapping windows induce strong autocorrelation, so HAC is
    mandatory (lags default to the horizon length).
    """
    h = TRADING_DAYS[months]
    y = forward_return(frame["spy"], h, lag=lag)
    skew = frame["skew"]
    vix = frame["vix"]
    d = pd.DataFrame({"y": y, "skew": skew, "vix": vix}).dropna()
    if len(d) < 50:
        return {"n": len(d), "t_skew": float("nan"), "t_vix": float("nan"),
                "beta_skew": float("nan"), "beta_vix": float("nan")}
    # standardise predictors so coefficients are per-1-sigma and comparable
    sk = (d["skew"] - d["skew"].mean()) / d["skew"].std()
    vx = (d["vix"] - d["vix"].mean()) / d["vix"].std()
    X = np.column_stack([np.ones(len(d)), sk.values, vx.values])
    yv = d["y"].values
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ yv
    resid = yv - X @ beta
    lags = hac_lags if hac_lags is not None else h
    se = _hac_se(X, resid, xtx_inv, lags)
    return {
        "n": int(len(d)),
        "beta_skew": float(beta[1]), "beta_vix": float(beta[2]),
        "t_skew": float(beta[1] / se[1]), "t_vix": float(beta[2] / se[2]),
    }


# --------------------------------------------------------------------------- #
# Bucketed forward returns (the headline directional table)
# --------------------------------------------------------------------------- #
def decile_summary(frame: pd.DataFrame, months: int, lag: int = 1,
                   top_q: float = 0.90, seed: int = 371) -> dict:
    """Forward ``months``-month return after **high-SKEW** days (top ``1-top_q`` quantile) vs the
    rest, with a Welch *t* and a placebo *p* (believers' tail: high SKEW => worse returns)."""
    h = TRADING_DAYS[months]
    y = forward_return(frame["spy"], h, lag=lag)
    d = pd.DataFrame({"y": y, "skew": frame["skew"]}).dropna()
    thr = d["skew"].quantile(top_q)
    hi = d.loc[d["skew"] >= thr, "y"].values
    rest = d.loc[d["skew"] < thr, "y"].values
    allv = d["y"].values
    p = placebo_pvalue(allv, len(hi), hi.mean(), seed=seed, tail="low")
    return {
        "months": months, "thr": float(thr),
        "n_hi": int(len(hi)), "n_rest": int(len(rest)),
        "hi_mean": float(hi.mean()), "rest_mean": float(rest.mean()),
        "all_mean": float(allv.mean()),
        "hi_win": float((hi > 0).mean()), "rest_win": float((rest > 0).mean()),
        "t": welch_t(hi, rest), "p_placebo": p,
    }


def block_bootstrap_decile(frame: pd.DataFrame, months: int, lag: int = 1,
                           top_q: float = 0.90, block: int | None = None,
                           n_boot: int = 5_000, seed: int = 371) -> dict:
    """Honest small-sample test for the high-SKEW vs rest forward-return gap.

    The naive Welch *t* treats every overlapping forward-return day as independent, but high-SKEW
    days cluster into a handful of *episodes* and the overlapping windows are strongly
    autocorrelated — so the naive *t* is wildly overstated. We (a) count the **distinct
    episodes** (top-``q`` days separated by more than one horizon) and (b) run a **circular block
    bootstrap** (block = horizon) of the mean difference to get an autocorrelation-respecting
    standard error and *t*. This is the number that decides the directional test, not the naive
    Welch *t*.
    """
    h = TRADING_DAYS[months]
    bl = block if block is not None else h
    y = forward_return(frame["spy"], h, lag=lag)
    d = pd.DataFrame({"y": y, "skew": frame["skew"]}).dropna()
    thr = d["skew"].quantile(top_q)
    mask = (d["skew"] >= thr).values
    yv = d["y"].values
    n = len(yv)
    obs = yv[mask].mean() - yv[~mask].mean()
    # distinct high-SKEW episodes
    hi_idx = np.where(mask)[0]
    n_episodes = 1 + int((np.diff(hi_idx) > h).sum()) if len(hi_idx) else 0
    rng = np.random.default_rng(seed)
    nb = max(1, n // bl)
    diffs = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=nb)
        idx = np.concatenate([(np.arange(s, s + bl) % n) for s in starts])[:n]
        ys, ms = yv[idx], mask[idx]
        if ms.sum() > 1 and (~ms).sum() > 1:
            diffs.append(ys[ms].mean() - ys[~ms].mean())
    diffs = np.asarray(diffs)
    se = float(diffs.std()) if len(diffs) else float("nan")
    return {
        "months": months, "obs_diff": float(obs), "block_se": se,
        "block_t": float(obs / se) if se else float("nan"),
        "n_hi_days": int(mask.sum()), "n_episodes": n_episodes,
    }


def tail_warning(frame: pd.DataFrame, months: int = 1, lag: int = 1,
                 top_q: float = 0.90, dd: float = -0.10, seed: int = 371) -> dict:
    """Does a high SKEW raise the probability of a forward **tail event** (a drawdown worse than
    ``dd`` over the next ``months``-month window)? Compares the tail rate after high-SKEW days to
    the unconditional base rate, with a placebo *p* on the conditional tail rate."""
    h = TRADING_DAYS[months]
    md = forward_min_return(frame["spy"], h, lag=lag)
    d = pd.DataFrame({"md": md, "skew": frame["skew"]}).dropna()
    thr = d["skew"].quantile(top_q)
    is_tail = (d["md"] <= dd).astype(float)
    hi = is_tail.loc[d["skew"] >= thr].values
    base_rate = float(is_tail.mean())
    hi_rate = float(hi.mean()) if len(hi) else float("nan")
    p = placebo_pvalue(is_tail.values, len(hi), hi_rate, seed=seed, tail="high")
    return {
        "months": months, "dd": dd, "thr": float(thr),
        "n_hi": int(len(hi)), "hi_tail_rate": hi_rate, "base_tail_rate": base_rate,
        "p_placebo": p,
    }


# --------------------------------------------------------------------------- #
# Tradability — a costed "fade high SKEW" sleeve
# --------------------------------------------------------------------------- #
def fade_sleeve(frame: pd.DataFrame, top_q: float = 0.90, lag: int = 1,
                cost_bps: float = 2.0) -> dict:
    """A long/flat sleeve that **goes to cash** while SKEW is in its top ``1-top_q`` quantile
    (the believers' "warning => de-risk" trade) and holds SPY otherwise, vs buy-and-hold SPY.

    One-day execution lag on the signal; one-way cost in bps applied per position change.
    Returns gross/net annualised return, Sharpe (excess of cash≈0 here, so total), and turnover.
    Reports both the sleeve and plain buy-and-hold so the race is excess-vs-excess on the *same*
    tape. (Price-only SPY proxy; no dividends — labelled in docs.)
    """
    spy = frame["spy"]
    ret = spy.pct_change().fillna(0.0)
    thr = frame["skew"].quantile(top_q)
    # signal known at close of day d, acted on at next open/close (lag): position for day t is
    # "in market" unless SKEW was high `lag` days ago.
    in_mkt = (frame["skew"] < thr).shift(lag).fillna(True).astype(float)
    pos_change = in_mkt.diff().abs().fillna(0.0)
    c = cost_bps / 1e4
    sleeve_ret = in_mkt * ret - pos_change * c
    bh_ret = ret

    def _ann(r):
        n = len(r)
        if n == 0:
            return float("nan"), float("nan")
        cagr = (1.0 + r).prod() ** (252.0 / n) - 1.0
        sharpe = (r.mean() / r.std() * np.sqrt(252.0)) if r.std() > 0 else float("nan")
        return float(cagr), float(sharpe)

    s_cagr, s_sharpe = _ann(sleeve_ret)
    b_cagr, b_sharpe = _ann(bh_ret)
    turnover = float(pos_change.sum())
    return {
        "thr": float(thr),
        "sleeve_cagr": s_cagr, "sleeve_sharpe": s_sharpe,
        "bh_cagr": b_cagr, "bh_sharpe": b_sharpe,
        "turnover": turnover, "cost_bps": cost_bps,
        "pct_in_cash": float((in_mkt == 0).mean()),
    }
