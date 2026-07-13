"""Strategy + inference for Study 763 — Puell-Multiple.

The claim (David Puell, 2019, "The Puell Multiple"): the ratio of daily Bitcoin miner
*issuance value* to its trailing 365-day average is a contrarian cycle timer. When the multiple
is **high** (issuance running hot vs its own year-average — miners flush, the market euphoric)
BTC is near a top; when it is **low** (issuance depressed vs its year-average — miners squeezed,
some capitulating) BTC is near a bottom. The canonical bands: Puell **> 4** = overheated top
(sell / stand aside), Puell **< 0.5** = undervalued bottom (buy). We formalise the folklore
three honest ways and pit each against the only benchmark that matters for a single trending
asset — **buy-and-hold BTC**:

1. **Predictive regression.** Regress the forward ``horizon``-day BTC log-return on the log
   Puell *stretch* ``log(puell)``, HAC (Newey-West) *t*, lag >= horizon to respect the overlap.
   A genuine contrarian gauge needs a *negative* slope whose |HAC t| clears 2. We also run a
   horse race that adds price's own trailing momentum, to see if Puell adds anything the price
   trend didn't already say.

2. **Band-state forward returns.** Map each day to {top, neutral, bottom} and report the mean
   forward return earned from each band, with a Welch *t* against the unconditional distribution
   and a random-date placebo — the cleanest read on whether the bands actually mark tops/bottoms.

3. **Buy-low / sell-high timer.** Stay long BTC by default, step to cash whenever Puell > sell
   threshold (overheated). Net of one-way costs on every flip, compared to buy-and-hold.

**No look-ahead:** the Puell value known at the close of day t is applied to the return earned
on day t+1 (one-day execution lag). **Costs:** one-way ``cost_bps`` on |Δposition| × NAV; the
timer is long/flat (no borrow); returns are price-only (BTC pays no yield), labelled as such.
**Single-survivor (named on the Signal axis):** BTC is the one crypto that survived and 100x+'d,
and the Puell Multiple is *mechanically derived from BTC's own price* (within a halving epoch it
is essentially price / trailing-365d-mean(price)); the bands are read off a handful of cycles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365   # BTC trades every calendar day


# --------------------------------------------------------------------------- #
# Signal transforms
# --------------------------------------------------------------------------- #
def log_puell(puell: pd.Series) -> pd.Series:
    """Log Puell stretch: log(puell). >0 = above its own year-average, <0 = below."""
    z = np.log(puell.astype(float))
    z.name = "log_puell"
    return z


def price_momentum(btc: pd.Series, lookback: int = 180) -> pd.Series:
    """Trailing log price growth over ``lookback`` days — the horse-race control regressor."""
    mom = np.log(btc.astype(float)).diff(lookback)
    mom.name = "price_mom"
    return mom


def forward_return(btc: pd.Series, horizon: int) -> pd.Series:
    """Simple forward BTC return from day t to day t+horizon (aligned to the signal at t)."""
    p = btc.astype(float)
    fwd = p.shift(-horizon) / p - 1.0
    fwd.name = f"fwd_{horizon}"
    return fwd


# --------------------------------------------------------------------------- #
# Predictive regression — HAC, overlap-aware
# --------------------------------------------------------------------------- #
def _hac_ols(y: np.ndarray, X: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS with Newey-West (HAC) standard errors. Returns (beta, se, R^2)."""
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    u = X * resid[:, None]
    S = u.T @ u
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        g = u[L:].T @ u[:-L]
        S += w * (g + g.T)
    cov = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return beta, se, r2


def predictive_regression(puell: pd.Series, btc: pd.Series, horizon: int = 30,
                          add_price_control: bool = False) -> dict:
    """Regress forward ``horizon``-day BTC log-return on log Puell (HAC t, lag = horizon).

    Model:  r_{t->t+h} = a + b·log(puell_t) [+ c·price_mom_t] + e.
    A contrarian gauge predicts ``b < 0``. Because forward windows overlap heavily on daily
    data, the HAC lag is set to at least the horizon (Newey-West), so the *t* is honest about
    the induced autocorrelation. Returns slope(s), HAC *t*(s), R^2 and n.
    """
    z = log_puell(puell)
    fwd = forward_return(btc, horizon)
    fwd_log = np.log1p(fwd)
    cols = {"log_puell": z}
    if add_price_control:
        cols["price_mom"] = price_momentum(btc, lookback=180)
    X = pd.DataFrame(cols)
    d = pd.concat([fwd_log.rename("y"), X], axis=1).dropna()
    if len(d) < 60:
        return {"n": int(len(d)), "slope_puell": np.nan, "t_puell": np.nan,
                "slope_price": np.nan, "t_price": np.nan, "r2": np.nan}

    y = d["y"].to_numpy()
    reg_cols = list(X.columns)
    Xmat = np.column_stack([np.ones(len(d))] + [d[c].to_numpy() for c in reg_cols])
    lags = max(horizon, int(np.floor(4.0 * (len(d) / 100.0) ** (2.0 / 9.0))))
    beta, se, r2 = _hac_ols(y, Xmat, lags)
    out = {"n": int(len(d)), "r2": float(r2),
           "slope_puell": float(beta[1]),
           "t_puell": float(beta[1] / se[1]) if se[1] > 0 else np.nan,
           "slope_price": np.nan, "t_price": np.nan, "lags": lags}
    if add_price_control:
        out["slope_price"] = float(beta[2])
        out["t_price"] = float(beta[2] / se[2]) if se[2] > 0 else np.nan
    return out


# --------------------------------------------------------------------------- #
# Band classification + forward-return event study
# --------------------------------------------------------------------------- #
def band_state(puell: pd.Series, high: float = 4.0, low: float = 0.5) -> pd.Series:
    """Map Puell to a 3-state band: +1 bottom (buy), 0 neutral, -1 top (sell).

    ``puell >= high`` -> top (-1, contrarian "sell"); ``puell <= low`` -> bottom
    (+1, contrarian "buy"); else neutral (0). Known at the close of day t.
    """
    m = puell.astype(float)
    s = pd.Series(0.0, index=puell.index)
    s[m >= high] = -1.0
    s[m <= low] = 1.0
    s.name = "band"
    return s


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances). NaN if either < 2 obs."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def state_forward_stats(puell: pd.Series, btc: pd.Series, horizon: int = 90,
                        high: float = 4.0, low: float = 0.5,
                        n_seeds: int = 20, n_draws: int = 1000, seed0: int = 763) -> pd.DataFrame:
    """Mean forward ``horizon``-day BTC return per Puell band vs the unconditional distribution.

    Per band: mean/median forward return, hit rate, count of *days*, a Welch *t* vs the
    unconditional forward-return distribution, and a random-date placebo *p* (draw the same
    number of random days, ``n_seeds`` × ``n_draws`` times; share of draws whose mean matches or
    beats the band mean — right-tailed for the buy band, left-tailed for the sell band, since the
    contrarian claim predicts *out*performance after bottoms and *under*performance after tops).
    Day counts are large but heavily autocorrelated — the placebo, not the raw n, is the guard.
    """
    fwd = forward_return(btc, horizon)
    st = band_state(puell, high=high, low=low)
    d = pd.concat([st.rename("band"), fwd.rename("fwd")], axis=1).dropna()
    all_fwd = d["fwd"].to_numpy()
    labels = {1.0: "bottom (buy)", 0.0: "neutral", -1.0: "top (sell)"}
    rows = []
    for val, lab in labels.items():
        sub = d.loc[d["band"] == val, "fwd"].to_numpy()
        n = len(sub)
        row = {"band": lab, "n": int(n),
               "mean_pct": float(np.mean(sub)) * 100 if n else np.nan,
               "median_pct": float(np.median(sub)) * 100 if n else np.nan,
               "hit": float((sub > 0).mean()) if n else np.nan,
               "welch_t": welch_t(sub, all_fwd) if n else np.nan,
               "placebo_p": np.nan}
        if n and val != 0.0:
            means = []
            for s in range(n_seeds):
                rng = np.random.default_rng(seed0 + s)
                for _ in range(n_draws):
                    pick = rng.choice(len(all_fwd), size=n, replace=False)
                    means.append(all_fwd[pick].mean())
            means = np.asarray(means)
            obs = float(np.mean(sub))
            if val == 1.0:      # buy band — claim is outperformance (right tail)
                row["placebo_p"] = float((means >= obs).mean())
            else:               # sell band — claim is underperformance (left tail)
                row["placebo_p"] = float((means <= obs).mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("band")


# --------------------------------------------------------------------------- #
# Buy-low / sell-high timer vs buy-and-hold
# --------------------------------------------------------------------------- #
def timing_signal(puell: pd.Series, high: float = 4.0) -> pd.Series:
    """Long/flat exposure for day t+1, known at the close of day t.

    Default long (1); step to flat (0) whenever ``puell >= high`` (overheated top). The buy
    band is already long, so the rule never shorts — the charitable "sell the top, stay long
    otherwise" reading without a borrow leg.
    """
    pos = pd.Series(1.0, index=puell.index)
    pos[puell.astype(float) >= high] = 0.0
    pos.name = "position"
    return pos


def backtest_timing(puell: pd.Series, btc: pd.Series, high: float = 4.0,
                    cost_bps: float = 10.0) -> dict:
    """Net-of-cost buy-low/sell-high timer vs continuous buy-and-hold over the same window.

    ``position.loc[t]`` (0/1) is the exposure held during day t+1 (one-day execution lag). A flip
    costs ``cost_bps`` one-way on |Δposition| × NAV. Long/flat, no borrow, price-only. Both legs
    are compounded over the same [first valid Puell day → tape end] window.
    """
    pos = timing_signal(puell, high=high).reindex(btc.index).dropna()
    ret = btc.astype(float).pct_change().dropna()
    idx = pos.index.intersection(ret.index)
    pos = pos.loc[idx]
    ret = ret.loc[idx]

    exposed = pos.shift(1).fillna(0.0)          # yesterday's signal governs today's return
    gross = exposed * ret
    dpos = exposed.diff().abs().fillna(exposed.abs())
    cost = dpos * (cost_bps / 1e4)
    net = gross - cost

    strat_w = (1.0 + net).cumprod()
    bh_w = (1.0 + ret).cumprod()
    strat_w, bh_w = strat_w / strat_w.iloc[0], bh_w / bh_w.iloc[0]
    n_years = (idx[-1] - idx[0]).days / 365.25
    n_flips = int((dpos > 0).sum())

    def _sharpe(r):
        r = r.dropna()
        return float(r.mean() / r.std() * np.sqrt(DAYS_PER_YEAR)) if r.std() > 0 else float("nan")

    excess = net - ret                          # timer minus buy-and-hold, daily
    return {
        "years": n_years, "n_flips": n_flips,
        "exposure_pct": float(exposed.mean()) * 100,
        "strat_total_pct": float(strat_w.iloc[-1] - 1) * 100,
        "strat_cagr_pct": float(strat_w.iloc[-1] ** (1 / n_years) - 1) * 100,
        "strat_sharpe": _sharpe(net),
        "bh_total_pct": float(bh_w.iloc[-1] - 1) * 100,
        "bh_cagr_pct": float(bh_w.iloc[-1] ** (1 / n_years) - 1) * 100,
        "bh_sharpe": _sharpe(ret),
        "excess_cagr_pct": float(excess.mean()) * DAYS_PER_YEAR * 100,
        "excess_t": _hac_t_mean(excess.to_numpy()),
    }


def _hac_t_mean(x: np.ndarray, lags: int | None = None) -> float:
    """HAC (Newey-West) t-stat of the mean of a daily series (for the timer-minus-BH excess)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return float("nan")
    mu = x.mean()
    e = x - mu
    if lags is None:
        lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(puell: pd.Series, price: pd.Series, horizon: int = 30) -> dict:
    """Point the real predictive regression at a synthetic (puell, price) world."""
    return predictive_regression(puell, price, horizon=horizon)
