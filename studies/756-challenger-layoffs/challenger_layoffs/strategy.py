"""Strategy + inference for Study 756 — Challenger-Layoffs.

The believers' rule (macro nowcasting): **a spike in announced job cuts warns you the
labour market — and the stock market — is about to weaken.** Operationalised on the
monthly Challenger job-cut tape:

    Let ``x`` = monthly announced job cuts (thousands) and
    ``s_t = x_t / mean(x_{t-w..t-1}) - 1`` its excess over the trailing ``w``-month
    average (default w = 12). Job cuts are "SPIKING" at month t when ``s_t > thresh``
    (default 0 => running above the trailing-year norm); the believers say the forward
    equity return is then DEPRESSED relative to the unconditional mean.

We test it by splitting forward H-month SPY returns into a SPIKE set and a CALM set and
comparing each to the unconditional mean, with:

  * a **Welch two-sample t** of the SPIKE-set forward mean against the unconditional
    forward mean (a real early-warning => SPIKE mean is *significantly negative-excess*);
  * a **Newey-West (HAC) t** on the spike-dummy coefficient in a forward-return
    regression, with the lag truncation set to the horizon so the *overlapping* forward
    windows don't inflate significance (the honest inference for H > 1);
  * a **placebo / randomization null** — draw the same number of random months and ask
    how often a random draw's mean forward return is at least as *low* as the SPIKE set;
  * a **win-rate** (share of forward returns < 0) vs the unconditional base rate;
  * **one execution lag** — the Challenger report for month t is released early in month
    t+1, so the signal is acted on at month t+1's close; forward returns start from t+1
    (strictly no look-ahead);
  * **one-way costs** x turnover for the simple "go to cash when cuts spike" overlay.

The decisive question is not whether equities are weak *during* recessions (they are —
that's mechanical) but whether a job-cut spike **leads** the market enough, and cleanly
enough, to be a tradable early-warning rather than a coincident or lagging tell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 6, 12)            # forward horizons in months


# --------------------------------------------------------------------------- #
# Signal construction
# --------------------------------------------------------------------------- #
def cut_spike(frame: pd.DataFrame, window: int = 12, smooth: int = 1) -> pd.Series:
    """Excess of monthly cuts over their trailing ``window``-month average (ex-current).

    ``s_t = x_t / mean(x_{t-window..t-1}) - 1``. Uses only past months for the baseline
    (``.shift(1)``), so the ratio is strictly non-look-ahead. Optional light smoothing
    of the level first.
    """
    x = frame["cuts"]
    if smooth > 1:
        x = x.rolling(smooth, min_periods=1).mean()
    trail = x.rolling(window, min_periods=max(3, window // 2)).mean().shift(1)
    return x / trail - 1.0


def spike_mask(frame: pd.DataFrame, window: int = 12, thresh: float = 0.0,
               smooth: int = 1) -> pd.Series:
    """Boolean: job cuts running strictly above ``thresh`` excess vs trailing avg."""
    return cut_spike(frame, window=window, smooth=smooth) > thresh


# --------------------------------------------------------------------------- #
# Forward returns (one-month execution lag)
# --------------------------------------------------------------------------- #
def forward_returns(frame: pd.DataFrame, months: int, lag: int = 1) -> pd.Series:
    """Forward ``months``-month SPY return entered ``lag`` months after the signal.

    Signal known at month-end t (after the report is released) is acted on at
    month-end t+lag (no look-ahead); the return runs t+lag -> t+lag+months. NaN where
    the horizon overruns the tape.
    """
    spy = frame["spy"]
    entry = spy.shift(-lag)
    exit_ = spy.shift(-lag - months)
    return (exit_ / entry - 1.0)


def split_returns(frame: pd.DataFrame, months: int, window: int = 12, thresh: float = 0.0,
                  smooth: int = 1, lag: int = 1):
    """(spike_fwd, calm_fwd, all_fwd) arrays of forward returns, NaNs dropped."""
    fwd = forward_returns(frame, months, lag=lag)
    spike = spike_mask(frame, window=window, thresh=thresh, smooth=smooth)
    ok = fwd.notna() & spike.notna()
    fwd, spike = fwd[ok], spike[ok].astype(bool)
    r = fwd[spike].values.astype(float)
    c = fwd[~spike].values.astype(float)
    a = fwd.values.astype(float)
    return r, c, a


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def welch_t(sample: np.ndarray, base: np.ndarray) -> float:
    """Welch t of mean(sample) - mean(base) (unequal variance). NaN if sample < 2."""
    if len(sample) < 2 or len(base) < 2:
        return float("nan")
    m1, m0 = sample.mean(), base.mean()
    se = np.sqrt(sample.var(ddof=1) / len(sample) + base.var(ddof=1) / len(base))
    if se == 0:
        return float("nan")
    return float((m1 - m0) / se)


def _newey_west_se(resid: np.ndarray, X: np.ndarray, maxlags: int) -> np.ndarray:
    """Newey-West HAC covariance of OLS coefficients for design ``X`` and residuals."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    # S = sum of weighted autocovariances of the score u_t = x_t * e_t
    u = X * resid[:, None]
    S = u.T @ u
    for L in range(1, maxlags + 1):
        w = 1.0 - L / (maxlags + 1.0)          # Bartlett kernel
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    return cov


def hac_spike_t(frame: pd.DataFrame, months: int, window: int = 12, thresh: float = 0.0,
                smooth: int = 1, lag: int = 1) -> dict:
    """Newey-West HAC t of the SPIKE dummy in a forward-return regression.

    Regress forward H-month return on [const, spike_dummy]; the slope is the spike-vs-calm
    excess. Overlapping H-month windows induce MA(H-1) autocorrelation, so the NW lag
    truncation is set to ``months`` (H). Returns the slope, HAC SE and HAC t.
    """
    fwd = forward_returns(frame, months, lag=lag)
    spike = spike_mask(frame, window=window, thresh=thresh, smooth=smooth)
    ok = fwd.notna() & spike.notna()
    y = fwd[ok].values.astype(float)
    d = spike[ok].astype(float).values
    if len(y) < 5 or d.sum() < 2 or (len(y) - d.sum()) < 2:
        return {"slope": float("nan"), "se": float("nan"), "t": float("nan"), "n": int(len(y))}
    X = np.column_stack([np.ones_like(d), d])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    cov = _newey_west_se(resid, X, maxlags=max(1, months))
    se = float(np.sqrt(cov[1, 1]))
    t = float(beta[1] / se) if se > 0 else float("nan")
    return {"slope": float(beta[1]), "se": se, "t": t, "n": int(len(y))}


def placebo_pvalue(frame: pd.DataFrame, months: int, window: int = 12, thresh: float = 0.0,
                   smooth: int = 1, lag: int = 1, n_draws: int = 20_000,
                   seed: int = 756) -> dict:
    """Small-sample placebo null for the early-warning claim.

    Draw ``n_spike`` random months ``n_draws`` times and ask how often a random draw's
    mean forward return is **at most** the SPIKE set's mean (as *bearish* or more).
    p = P[random-draw mean <= spike mean]. A real early-warning => small p.
    """
    r, _c, a = split_returns(frame, months, window=window, thresh=thresh, smooth=smooth, lag=lag)
    kk = len(r)
    if kk == 0 or len(a) == 0:
        return {"k": 0, "spike_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    n = len(a)
    for i in range(n_draws):
        means[i] = a[rng.integers(0, n, size=kk)].mean()
    obs = float(r.mean())
    p = float((means <= obs).mean())
    return {"k": kk, "spike_mean": obs, "placebo_mean": float(means.mean()), "p_value": p}


def summarize(frame: pd.DataFrame, months: int, window: int = 12, thresh: float = 0.0,
              smooth: int = 1, lag: int = 1) -> dict:
    """Headline stats for one horizon: n, SPIKE vs CALM vs base forward means and
    down-rates, the Welch t (SPIKE vs base), the HAC t (spike dummy), placebo p."""
    r, c, a = split_returns(frame, months, window=window, thresh=thresh, smooth=smooth, lag=lag)
    pl = placebo_pvalue(frame, months, window=window, thresh=thresh, smooth=smooth, lag=lag)
    hac = hac_spike_t(frame, months, window=window, thresh=thresh, smooth=smooth, lag=lag)
    return {
        "months": months,
        "n_spike": int(len(r)),
        "n_calm": int(len(c)),
        "spike_mean": float(r.mean()) if len(r) else float("nan"),
        "calm_mean": float(c.mean()) if len(c) else float("nan"),
        "base_mean": float(a.mean()) if len(a) else float("nan"),
        "spike_downrate": float((r < 0).mean()) if len(r) else float("nan"),
        "base_downrate": float((a < 0).mean()) if len(a) else float("nan"),
        "t": welch_t(r, a),
        "hac_t": hac["t"],
        "p_placebo": pl["p_value"],
    }


def lead_lag(frame: pd.DataFrame, window: int = 12, smooth: int = 1,
             leads=range(-6, 7)) -> pd.Series:
    """Correlation of the cut-spike signal at t with the SPY return over [t+L, t+L+1].

    L < 0 => the cut spike *lags* the market (coincident/lagging tell); L > 0 => the cut
    spike *leads* it (a genuine early-warning would peak at L > 0 with a negative
    correlation). Returns a Series indexed by lead L (months)."""
    sig = cut_spike(frame, window=window, smooth=smooth)
    spy = frame["spy"]
    out = {}
    for L in leads:
        fwd = spy.shift(-L - 1) / spy.shift(-L) - 1.0
        s = pd.concat([sig, fwd], axis=1).dropna()
        out[L] = float(np.corrcoef(s.iloc[:, 0], s.iloc[:, 1])[0, 1]) if len(s) > 3 else float("nan")
    return pd.Series(out, name="corr")


# --------------------------------------------------------------------------- #
# Tradability — the "go to cash when cuts spike" timing overlay
# --------------------------------------------------------------------------- #
def timing_overlay(frame: pd.DataFrame, window: int = 12, thresh: float = 0.0,
                   smooth: int = 1, lag: int = 1, cost_bps: float = 10.0) -> dict:
    """Hold SPY when cuts are CALM, sit in cash when they SPIKE (the believers' overlay).

    One-month execution lag; one-way cost ``cost_bps`` charged on each switch (turnover
    counted one-way x NAV). Compares the overlay's monthly return stream to buy-and-hold
    on the same months, reporting annualised mean, vol, Sharpe (excess-of-zero, since the
    cash leg earns 0 here — a conservative, clearly-labelled simplification), and the
    number of switches. Gross and net both reported.
    """
    spy = frame["spy"]
    spike = spike_mask(frame, window=window, thresh=thresh, smooth=smooth)
    # signal known at t acts from t+lag: position for month t+1 = (not spiking at t)
    pos = (~spike).astype(float).shift(lag)
    df = pd.DataFrame({"r": spy.pct_change(), "pos": pos}).dropna()
    bh = df["r"]
    switches = df["pos"].diff().abs().fillna(0.0)
    cst = cost_bps / 1e4
    gross = df["pos"] * bh
    net = gross - switches * cst
    n_sw = int((switches > 0).sum())

    def _ann(x):
        mu = x.mean() * 12.0
        vol = x.std(ddof=1) * np.sqrt(12.0)
        sharpe = mu / vol if vol > 0 else float("nan")
        return mu, vol, sharpe

    bh_mu, bh_vol, bh_sh = _ann(bh)
    g_mu, g_vol, g_sh = _ann(gross)
    n_mu, n_vol, n_sh = _ann(net)
    return {
        "n_months": int(len(df)), "n_switches": n_sw,
        "bh_mean": bh_mu, "bh_vol": bh_vol, "bh_sharpe": bh_sh,
        "overlay_gross_mean": g_mu, "overlay_gross_sharpe": g_sh,
        "overlay_net_mean": n_mu, "overlay_net_vol": n_vol, "overlay_net_sharpe": n_sh,
        "cost_bps": cost_bps,
    }
