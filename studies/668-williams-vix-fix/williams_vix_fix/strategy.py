"""Strategy + inference for Study 668 — Williams VIX Fix.

The claim: Larry Williams' **VIX Fix** turns *any* price series into a synthetic fear
gauge, computed from OHLC alone (no options data, no VIX):

    WVF(t) = (highest_close(22) - low(t)) / highest_close(22) * 100

where ``highest_close(22)`` is the rolling 22-session maximum close. WVF is near zero
when the low sits at/above the recent high closes, and spikes when the low undercuts the
recent close-highs by a large margin — a proxy for the intraday panic a real VIX spike
represents. The folk rule: when WVF pokes above a Bollinger band on itself (mean + 2
standard deviations, 20-session lookback) it marks a **capitulation bottom** — buy.

Three questions, all with one documented execution lag (signal known at the close of bar
*t*; every trade enters at bar *t+1*'s **open**):

* **Forward returns.** Do WVF-spike days lead to better [5, 10, 20]-day forward returns
  than an unconditional day? Welch *t* (pooled across the basket) is the primary test; a
  per-ticker Newey-West (HAC) dummy regression — lags = the forward horizon, since
  overlapping forward windows are mechanically autocorrelated — is the serial-correlation
  cross-check.
* **More than a drawdown proxy?** WVF is driven almost entirely by "how far below the
  recent high is price" — the third axis asks whether the *intrabar low* (the wick) adds
  anything a plain **close-only** drawdown signal, calibrated with the identical Bollinger
  rule, doesn't already capture. A two-dummy HAC regression (WVF spike + drawdown-proxy
  spike, same day) isolates the marginal contribution of the wick.
* **The tradable timer.** WVF-spike-onset entries, held a fixed horizon, one round trip =
  2 × one-way cost × NAV, HAC *t* on the trade ledger.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# The indicator and its plain-drawdown sibling
# --------------------------------------------------------------------------- #
def wvf(df: pd.DataFrame, lookback: int = 22) -> pd.Series:
    """Williams VIX Fix: (highest_close(lookback) - low) / highest_close(lookback) * 100."""
    hc = df["Close"].rolling(lookback, min_periods=lookback).max()
    return ((hc - df["Low"]) / hc * 100.0).rename("wvf")


def drawdown_proxy(df: pd.DataFrame, lookback: int = 22) -> pd.Series:
    """Close-only sibling of WVF: (highest_close(lookback) - close) / highest_close(lookback) * 100.

    Same normalisation, same lookback, but built from the **close** alone — no intrabar
    low. This is the "just a drawdown" control the WVF claim must beat.
    """
    hc = df["Close"].rolling(lookback, min_periods=lookback).max()
    return ((hc - df["Close"]) / hc * 100.0).rename("dd")


def bollinger_spike(x: pd.Series, bb_len: int = 20, bb_mult: float = 2.0) -> pd.Series:
    """Boolean flag: x >= rolling mean(bb_len) + bb_mult * rolling std(bb_len).

    Uses only values through bar *t* (rolling, no centering) — no look-ahead. The
    Bollinger band is the canonical WVF trigger (CM Williams Vix Fix indicator).
    """
    mid = x.rolling(bb_len, min_periods=bb_len).mean()
    sd = x.rolling(bb_len, min_periods=bb_len).std()
    return (x >= mid + bb_mult * sd).fillna(False).rename("spike")


def onset(flag: pd.Series) -> pd.Series:
    """Keep only the *first* True of each run (one entry per capitulation episode)."""
    prev = flag.shift(1).fillna(False)
    return (flag & ~prev).rename("onset")


# --------------------------------------------------------------------------- #
# Forward returns — one documented execution lag: enter next bar's open
# --------------------------------------------------------------------------- #
def forward_returns(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Log return from bar *t+1*'s open to bar *t+horizon*'s close, indexed by signal day *t*.

    Signal known at close of *t*; enter at *t+1* open (the study's single documented
    execution lag); exit at the close *horizon* sessions after entry (i.e. bar
    *t+horizon*'s close — a *horizon*-session hold). NaN for the last *horizon* bars.
    """
    op = df["Open"].to_numpy(dtype=float)
    cl = df["Close"].to_numpy(dtype=float)
    n = len(df)
    out = np.full(n, np.nan)
    for t in range(n - horizon):
        e = t + 1
        x = t + horizon
        if x < n and op[e] > 0:
            out[t] = np.log(cl[x] / op[e])
    return pd.Series(out, index=df.index, name=f"fwd_{horizon}")


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances). NaN if either < 2 obs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def ols_hac(X: np.ndarray, y: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients + Newey-West (Bartlett kernel) HAC t-stats. X includes the intercept."""
    keep = ~np.isnan(y) & np.all(~np.isnan(X), axis=1)
    X, y = X[keep], y[keep]
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    s = X * u[:, None]
    S = s.T @ s
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = s[l:].T @ s[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    t = np.divide(beta, se, out=np.full(k, np.nan), where=se > 0)
    return beta, t


def newey_west_dummy_t(y: np.ndarray, d: np.ndarray, lags: int) -> float:
    """Newey-West t of the slope in y = a + b*d (single-dummy convenience wrapper)."""
    X = np.column_stack([np.ones(len(d)), d.astype(float)])
    _, t = ols_hac(X, y, lags)
    return float(t[1])


def placebo_pvalue(pool: np.ndarray, obs: float, k: int, n_draws_per_seed: int = 1_000,
                   n_seeds: int = 20, base_seed: int = 668) -> dict:
    """Random-calendar placebo: draw k random days from the unconditional pool.

    p = share of draws whose mean is >= the observed spike-day mean (a RIGHT-tail test —
    the claim is a boost). Averaged over ``n_seeds`` seeds x ``n_draws_per_seed`` draws.
    """
    pool = pool[~np.isnan(pool)]
    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            means.append(pool[rng.choice(len(pool), size=k, replace=False)].mean())
    means = np.asarray(means)
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)),
            "p_value": float((means >= obs).mean()), "n_draws": len(means)}


# --------------------------------------------------------------------------- #
# Per-ticker frame builder
# --------------------------------------------------------------------------- #
def ticker_frame(df: pd.DataFrame, wvf_lookback: int = 22, bb_len: int = 20,
                 bb_mult: float = 2.0, horizons: tuple[int, ...] = (5, 10, 20)) -> pd.DataFrame:
    """One row per trading day: WVF, drawdown proxy, both spike flags (+ onset), forward returns."""
    w = wvf(df, wvf_lookback)
    d = drawdown_proxy(df, wvf_lookback)
    out = pd.DataFrame(index=df.index)
    out["wvf"] = w
    out["dd"] = d
    out["spike_wvf"] = bollinger_spike(w, bb_len, bb_mult)
    out["spike_dd"] = bollinger_spike(d, bb_len, bb_mult)
    out["onset_wvf"] = onset(out["spike_wvf"])
    out["onset_dd"] = onset(out["spike_dd"])
    for h in horizons:
        out[f"fwd_{h}"] = forward_returns(df, h)
    return out.iloc[max(wvf_lookback, bb_len):]   # drop the warm-up window


# --------------------------------------------------------------------------- #
# Headline: spike-day forward return vs unconditional
# --------------------------------------------------------------------------- #
def spike_stats(frame: pd.DataFrame, horizon: int, spike_col: str = "onset_wvf") -> dict:
    f = frame[spike_col].to_numpy(dtype=bool)
    y = frame[f"fwd_{horizon}"].to_numpy(dtype=float)
    a, b = y[f], y[~f]
    k_up = int((a > 0).sum())
    n_valid = int((~np.isnan(a)).sum())
    lo, hi = wilson_interval(k_up, n_valid)
    return {
        "n_spike": int(f.sum()), "n_rest": int((~f).sum()),
        "spike_mean": float(np.nanmean(a)) if len(a) else float("nan"),
        "rest_mean": float(np.nanmean(b)) if len(b) else float("nan"),
        "gap": float(np.nanmean(a) - np.nanmean(b)) if len(a) and len(b) else float("nan"),
        "welch_t": welch_t(a, b),
        "hit_up": k_up, "n_valid": n_valid,
        "hit_rate": k_up / n_valid if n_valid else float("nan"),
        "hit_lo": lo, "hit_hi": hi,
    }


def nw_dummy_stats(frame: pd.DataFrame, horizon: int, spike_col: str = "onset_wvf") -> float:
    """NW(lags=horizon) t of the spike dummy in a full-panel forward-return regression."""
    y = frame[f"fwd_{horizon}"].to_numpy(dtype=float)
    d = frame[spike_col].to_numpy(dtype=float)
    return newey_west_dummy_t(y, d, lags=horizon)


# --------------------------------------------------------------------------- #
# "More than a drawdown proxy?" — the third-axis myth-check
# --------------------------------------------------------------------------- #
def wick_marginal_stats(frame: pd.DataFrame, horizon: int) -> dict:
    """Two-dummy HAC regression: fwd_h ~ a + b*onset_wvf + c*onset_dd.

    b is the marginal contribution of the WVF wick *controlling for* the plain
    close-only drawdown spike on the same day; c is the drawdown spike's own
    contribution controlling for WVF. Also reports the day-level overlap between the
    two flags and the difference-in-means between "WVF-only" and "drawdown-only" days.
    """
    y = frame[f"fwd_{horizon}"].to_numpy(dtype=float)
    dw = frame["onset_wvf"].to_numpy(dtype=bool)
    dd = frame["onset_dd"].to_numpy(dtype=bool)
    X = np.column_stack([np.ones(len(y)), dw.astype(float), dd.astype(float)])
    beta, t = ols_hac(X, y, lags=horizon)

    both = dw & dd
    wvf_only = dw & ~dd
    dd_only = dd & ~dw
    y_wvf_only, y_dd_only = y[wvf_only], y[dd_only]
    overlap = float(both.sum() / dw.sum()) if dw.sum() else float("nan")
    return {
        "n_wvf": int(dw.sum()), "n_dd": int(dd.sum()), "n_both": int(both.sum()),
        "overlap_of_wvf": overlap,
        "b_wvf_marginal": float(beta[1]), "t_wvf_marginal": float(t[1]),
        "b_dd_marginal": float(beta[2]), "t_dd_marginal": float(t[2]),
        "n_wvf_only": int(wvf_only.sum()), "mean_wvf_only": float(np.nanmean(y_wvf_only)) if wvf_only.sum() else float("nan"),
        "n_dd_only": int(dd_only.sum()), "mean_dd_only": float(np.nanmean(y_dd_only)) if dd_only.sum() else float("nan"),
        "welch_t_wvf_vs_dd_only": welch_t(y_wvf_only, y_dd_only),
    }


# --------------------------------------------------------------------------- #
# The tradable timer — onset entries, fixed horizon, costs
# --------------------------------------------------------------------------- #
def timer_ledger(frame: pd.DataFrame, horizon: int, cost_bps: float = 5.0,
                 spike_col: str = "onset_wvf") -> pd.DataFrame:
    """One row per spike-onset trade: gross/net return (round trip = 2 x cost_bps x NAV)."""
    sig = frame.index[frame[spike_col].astype(bool)]
    gross = frame.loc[sig, f"fwd_{horizon}"].dropna()
    net = gross - 2.0 * cost_bps / 1e4
    return pd.DataFrame({"entry_ts": gross.index, "ret_gross": gross.to_numpy(),
                         "ret_net": net.to_numpy()})


def summarize_ledger(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Trade count, win rate, mean bps/trade, per-trade Sharpe, HAC (auto-lag) t-stat."""
    if len(ledger) == 0:
        return {k: float("nan") for k in
                ("n_trades", "win_rate", "mean_bps", "sharpe", "tstat")}
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    out = {
        "n_trades": int(n),
        "win_rate": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "sharpe": float(r.mean() / r.std(ddof=1)) if n > 1 and r.std() > 0 else float("nan"),
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


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(df: pd.DataFrame, horizon: int = 10) -> dict:
    """Run the headline Welch + NW split on a synthetic OHLC world."""
    frame = ticker_frame(df, horizons=(horizon,))
    s = spike_stats(frame, horizon, "onset_wvf")
    s["nw_t"] = nw_dummy_stats(frame, horizon, "onset_wvf")
    return s
