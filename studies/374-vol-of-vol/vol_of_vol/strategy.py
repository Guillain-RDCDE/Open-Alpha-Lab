"""Strategy + inference for Study 374 — Vol-of-Vol (VVIX vs VIX as a risk timer).

The claim: **VVIX** — the volatility of the VIX — times equity risk *better than the VIX
itself*. The believers' picture: when the vol-of-vol spikes, option markets are pricing a
fat tail the level of vol hasn't shown yet, so high VVIX should precede weak / drawdown-y
forward equity returns, and do so **incrementally to VIX**.

We test it three ways, all with a 1-day execution lag (signal known at the close of *t*,
acted on the return of *t+1*):

  * **Conditional forward returns.** Forward SPY return after a high-VVIX day vs the
    unconditional base rate, with a Welch *t* and a block-style placebo null.
  * **The decisive test — incremental over VIX.** A bivariate regression of forward
    returns on standardized VIX **and** standardized VVIX, with **Newey-West (HAC)**
    standard errors (overlapping windows induce serial correlation). The vol-of-vol claim
    lives or dies on the VVIX coefficient's HAC *t* *once VIX is already in the model*.
  * **Win-rate** of a high-VVIX state vs the base rate, and a costed long/flat timing rule.

The honest finding is about *incrementality*: the VIX and VVIX are ~0.7 correlated, so a
raw high-VVIX signal looks predictive simply because high VVIX co-occurs with high VIX. The
question is whether vol-of-vol adds anything once VIX is controlled — and on the real tape
the incremental HAC *t* does not clear the desk's bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = {1: 21, 3: 63}            # months -> trading days (forward horizons)


# --------------------------------------------------------------------------- #
# Forward returns / drawdowns
# --------------------------------------------------------------------------- #
def forward_return(spy: pd.Series, horizon_days: int) -> pd.Series:
    """Simple forward return over ``horizon_days`` for every date (NaN near the tail)."""
    return spy.shift(-horizon_days) / spy - 1.0


def forward_drawdown(spy: pd.Series, horizon_days: int) -> pd.Series:
    """Worst peak-to-trough drawdown over the next ``horizon_days`` (<= 0), per date."""
    vals = spy.values
    n = len(vals)
    out = np.full(n, np.nan)
    for t in range(n - horizon_days):
        win = vals[t + 1: t + 1 + horizon_days]
        if len(win) == 0:
            continue
        run_max = np.maximum.accumulate(np.concatenate([[vals[t]], win]))
        dd = win / run_max[1:] - 1.0
        out[t] = dd.min()
    return pd.Series(out, index=spy.index)


# --------------------------------------------------------------------------- #
# Signal — a high-VVIX state
# --------------------------------------------------------------------------- #
def vvix_state(frame: pd.DataFrame, q: float = 0.80, lookback: int = 252) -> pd.Series:
    """Boolean 'high vol-of-vol' state: VVIX above its trailing ``lookback``-day ``q``
    quantile (a causal, point-in-time threshold — no look-ahead)."""
    v = frame["vvix"]
    thr = v.rolling(lookback, min_periods=lookback // 2).quantile(q)
    return (v > thr).fillna(False)


# --------------------------------------------------------------------------- #
# Inference — Welch t, placebo null
# --------------------------------------------------------------------------- #
def welch_t(sample: np.ndarray, base: np.ndarray) -> float:
    """Welch t of ``mean(sample) - mean(base)`` (unequal variance). NaN if sample < 2."""
    sample = np.asarray(sample, float); base = np.asarray(base, float)
    if len(sample) < 2:
        return float("nan")
    m1, m0 = sample.mean(), base.mean()
    se = np.sqrt(sample.var(ddof=1) / len(sample) + base.var(ddof=1) / len(base))
    if se == 0:
        return float("nan")
    return float((m1 - m0) / se)


def placebo_pvalue(frame: pd.DataFrame, state: pd.Series, months: int,
                   n_draws: int = 20_000, lag: int = 1, seed: int = 374) -> dict:
    """Block-style placebo null: the high-VVIX days form clusters, so we draw the SAME
    number of state-days at random (with the observed cluster size preserved via a simple
    contiguous-block resample) and ask how often a random draw's mean forward return is as
    *low* as the high-VVIX set (we are testing for under-performance).

    Returns the conditional mean, the placebo mean, and the empirical p-value
    P[random-draw mean <= conditional mean]."""
    h = TRADING_DAYS[months]
    fwd = forward_return(frame["spy"], h).shift(-lag)
    valid = fwd.dropna()
    fv = valid.values
    n = len(fv)
    mask = state.reindex(valid.index).fillna(False).values
    obs = fv[mask]
    k = int(mask.sum())
    if k == 0 or n == 0:
        return {"k": 0, "cond_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    # block size ~ mean run length of the state (preserve autocorrelation in the draw)
    runs, cur = [], 0
    for m in mask:
        if m:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    block = max(1, int(np.mean(runs))) if runs else 1
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(k / block))
    means = np.empty(n_draws)
    for i in range(n_draws):
        idx = []
        for _ in range(n_blocks):
            s = rng.integers(0, n)
            idx.extend(range(s, min(s + block, n)))
        idx = np.array(idx[:k])
        means[i] = fv[idx].mean()
    obs_mean = float(obs.mean())
    p = float((means <= obs_mean).mean())
    return {"k": k, "cond_mean": obs_mean, "placebo_mean": float(means.mean()),
            "p_value": p}


def summarize(frame: pd.DataFrame, state: pd.Series, months: int, lag: int = 1) -> dict:
    """Headline stats for one horizon: n high-VVIX days, conditional vs base mean/win-rate,
    Welch t, placebo p, and conditional vs base mean forward drawdown."""
    h = TRADING_DAYS[months]
    fwd = forward_return(frame["spy"], h).shift(-lag)
    dd = forward_drawdown(frame["spy"], h).shift(-lag)
    valid = fwd.dropna().index
    m = state.reindex(valid).fillna(False).values
    fv = fwd.loc[valid].values
    ddv = dd.reindex(valid).values
    cond, base = fv[m], fv
    pl = placebo_pvalue(frame, state, months, lag=lag)
    return {
        "months": months,
        "n": int(m.sum()),
        "n_base": int(len(fv)),
        "cond_mean": float(cond.mean()) if len(cond) else float("nan"),
        "cond_win": float((cond > 0).mean()) if len(cond) else float("nan"),
        "base_mean": float(base.mean()),
        "base_win": float((base > 0).mean()),
        "cond_dd": float(np.nanmean(ddv[m])) if m.any() else float("nan"),
        "base_dd": float(np.nanmean(ddv)),
        "t": welch_t(cond, base),
        "p_placebo": pl["p_value"],
    }


# --------------------------------------------------------------------------- #
# THE decisive test — incremental signal over VIX (HAC / Newey-West regression)
# --------------------------------------------------------------------------- #
def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def newey_west_se(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West (HAC) covariance for OLS coefficients. ``X`` includes the intercept."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * resid[:, None]).T @ (X * resid[:, None])      # lag 0
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        u = (X[L:] * resid[L:, None])
        v = (X[:-L] * resid[:-L, None])
        G = u.T @ v
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    return cov


def incremental_regression(frame: pd.DataFrame, months: int, lag: int = 1) -> dict:
    """Regress forward ``months``-month SPY return on standardized VIX and VVIX (with an
    intercept), HAC/Newey-West *t* at the overlap-length lag.

    Three fits, all on the same aligned sample:
      * VIX-only   (does the level of vol predict?)
      * VVIX-only  (does vol-of-vol predict on its own — the *raw* claim?)
      * VIX + VVIX (does VVIX add anything **once VIX is in**? — the decisive claim)

    Returns each fit's coefficient(s) and HAC *t*. The vol-of-vol thesis requires the VVIX
    HAC *t* in the **bivariate** model to clear 2 (and to keep its sign)."""
    h = TRADING_DAYS[months]
    fwd = forward_return(frame["spy"], h).shift(-lag)
    df = pd.DataFrame({
        "y": fwd,
        "vix": _zscore(frame["vix"]),
        "vvix": _zscore(frame["vvix"]),
    }).dropna()
    y = df["y"].values
    n = len(y)
    L = h                                     # overlap length -> NW lag

    def fit(cols):
        X = np.column_stack([np.ones(n)] + [df[c].values for c in cols])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        cov = newey_west_se(X, resid, L)
        se = np.sqrt(np.diag(cov))
        tt = beta / se
        return beta, tt

    b_vix, t_vix = fit(["vix"])
    b_vvix, t_vvix = fit(["vvix"])
    b_both, t_both = fit(["vix", "vvix"])
    return {
        "months": months, "n": n, "nw_lag": L,
        # univariate
        "vix_only_beta": float(b_vix[1]), "vix_only_t": float(t_vix[1]),
        "vvix_only_beta": float(b_vvix[1]), "vvix_only_t": float(t_vvix[1]),
        # bivariate (the decisive coefficients)
        "vix_beta": float(b_both[1]), "vix_t": float(t_both[1]),
        "vvix_beta": float(b_both[2]), "vvix_t": float(t_both[2]),
        "corr_vix_vvix": float(np.corrcoef(df["vix"], df["vvix"])[0, 1]),
    }


# --------------------------------------------------------------------------- #
# Tradability — a costed long/flat timing rule
# --------------------------------------------------------------------------- #
def timing_backtest(frame: pd.DataFrame, state: pd.Series, lag: int = 1,
                    cost_bps: float = 2.0) -> dict:
    """Long SPY when NOT in the high-VVIX state, flat (in cash) when in it; 1-day lag,
    one-way cost ``cost_bps`` charged on every position change. Compared to buy-and-hold.

    Returns gross/net annualised return, vol, Sharpe (excess of zero cash for simplicity,
    both legs price-on-price), turnover, and the buy-and-hold benchmark — all gross/net
    labelled. This is a *timing* read of the signal, not an investable claim by itself.
    """
    spy = frame["spy"]
    ret = spy.pct_change().fillna(0.0)
    # position: 0 (flat) when high-VVIX, else 1 (long); acted with a 1-day lag
    pos = (~state.astype(bool)).astype(float).shift(lag).fillna(1.0)
    turn = pos.diff().abs().fillna(0.0)
    c = cost_bps / 1e4
    strat_gross = pos * ret
    strat_net = strat_gross - turn * c
    ann = 252.0

    def stats(r):
        mu = r.mean() * ann
        sd = r.std(ddof=1) * np.sqrt(ann)
        return mu, sd, (mu / sd if sd > 0 else float("nan"))

    g_mu, g_sd, g_sh = stats(strat_gross)
    n_mu, n_sd, n_sh = stats(strat_net)
    b_mu, b_sd, b_sh = stats(ret)
    return {
        "gross_ann": g_mu, "gross_vol": g_sd, "gross_sharpe": g_sh,
        "net_ann": n_mu, "net_vol": n_sd, "net_sharpe": n_sh,
        "bh_ann": b_mu, "bh_vol": b_sd, "bh_sharpe": b_sh,
        "turnover": float(turn.sum()), "n_trades": int((turn > 0).sum()),
        "cost_bps": cost_bps, "pct_flat": float((pos == 0).mean()),
    }
