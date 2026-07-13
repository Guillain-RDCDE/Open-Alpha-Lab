"""Strategy + inference for Study 752 — TGA-Drawdown.

The believers' rule (the macro-liquidity thesis): **a falling Treasury General
Account is hidden stimulus.** When the Treasury draws down its cash balance at the
Fed, that cash flows out into the banking system as bank reserves — a "liquidity
injection" that supposedly lifts risk assets over the following weeks. When the TGA
*builds* (Treasury borrows and parks the cash), reserves drain and equities are
supposedly pressured. Operationalised on the monthly TGA proxy tape:

    Let ``g`` = the TGA balance ($B) and ``d_t = g_t - g_{t-k}`` its k-month change
    (default k = 1). The TGA is "DRAWING DOWN" at month t when ``d_t < thresh``
    (default 0) — liquidity injected; the believers say the forward equity return is
    then ELEVATED relative to the unconditional mean. We also treat the *injection*
    ``x_t = -d_t`` as a continuous regressor.

We test it three ways:

  * a **Welch two-sample t** of the DRAWDOWN-set forward mean against the
    unconditional forward mean (a real liquidity lever => DRAWDOWN mean is
    *significantly positive-excess*);
  * a **Newey-West (HAC) predictive regression** of the forward H-month SPY return on
    the trailing injection ``x_t`` — the headline inference, with HAC lags set to the
    return horizon to handle the overlap (a real lever => *significantly positive*
    slope, HAC ``|t| >= 2``);
  * a **placebo / randomization null** on the DRAWDOWN-set forward mean;
  * plus a **lead/lag** cross-correlation (does the injection actually *lead* the
    market?) and a tradable **timing overlay** (hold SPY when the TGA is drawing down).

Lag convention: the month-t TGA level is public (the weekly WTREGEN / Daily Treasury
Statement is released with only a few days' delay), but to be strictly non-look-ahead
we enter at the **next** month-end close (a one-month execution lag) — conservative,
and documented.

The decisive question is not whether reserves and asset prices co-move over long
sweeps (they can, mechanically, through the same macro backdrop) but whether a TGA
*drawdown* leads equities cleanly enough, at monthly resolution, to be a tradable
lever rather than a coincident narrative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 2, 3, 6)            # forward horizons in months


# --------------------------------------------------------------------------- #
# Signal construction
# --------------------------------------------------------------------------- #
def tga_change(frame: pd.DataFrame, k: int = 1) -> pd.Series:
    """k-month change in the TGA balance ($B): g_t - g_{t-k}. Negative = drawdown."""
    g = frame["tga"]
    return g - g.shift(k)


def injection(frame: pd.DataFrame, k: int = 1) -> pd.Series:
    """Liquidity injection = negative of the TGA change ($B). Positive = drawdown."""
    return -tga_change(frame, k=k)


def drawdown_mask(frame: pd.DataFrame, k: int = 1, thresh: float = 0.0) -> pd.Series:
    """Boolean: TGA change strictly below ``thresh`` (the TGA is DRAWING DOWN)."""
    return tga_change(frame, k=k) < thresh


# --------------------------------------------------------------------------- #
# Forward returns (one-month execution lag)
# --------------------------------------------------------------------------- #
def forward_returns(frame: pd.DataFrame, months: int, lag: int = 1) -> pd.Series:
    """Forward ``months``-month SPY return entered ``lag`` months after the signal.

    Signal known at month-end t is acted on at month-end t+lag (no look-ahead); the
    return runs t+lag -> t+lag+months. NaN where the horizon overruns the tape.
    """
    spy = frame["spy"]
    entry = spy.shift(-lag)
    exit_ = spy.shift(-lag - months)
    return (exit_ / entry - 1.0)


def split_returns(frame: pd.DataFrame, months: int, k: int = 1, thresh: float = 0.0,
                  lag: int = 1):
    """(drawdown_fwd, build_fwd, all_fwd) arrays of forward returns, NaNs dropped."""
    fwd = forward_returns(frame, months, lag=lag)
    draw = drawdown_mask(frame, k=k, thresh=thresh)
    ok = fwd.notna() & draw.notna()
    fwd, draw = fwd[ok], draw[ok].astype(bool)
    d = fwd[draw].values.astype(float)
    b = fwd[~draw].values.astype(float)
    a = fwd.values.astype(float)
    return d, b, a


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


def hac_regression(frame: pd.DataFrame, months: int, k: int = 1, lag: int = 1) -> dict:
    """Newey-West (HAC) predictive regression: forward H-month SPY return on injection.

    Regresses the forward return (entered ``lag`` months after the signal) on the
    trailing k-month liquidity injection ``x_t = -(g_t - g_{t-k})`` ($100B units, so
    the slope reads per +$100B drawn down). HAC lags = ``months`` (Newey-West) to
    absorb the overlap in overlapping H-month returns. A real liquidity lever =>
    *positive* slope with HAC ``|t| >= 2``.

    Falls back to a plain-OLS closed form with a hand-rolled Newey-West variance if
    statsmodels is unavailable, so the offline core never depends on it.
    """
    x = injection(frame, k=k) / 100.0          # per +$100B drawn down
    y = forward_returns(frame, months, lag=lag)
    s = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    n = len(s)
    if n < 5:
        return {"months": months, "n": n, "beta": float("nan"),
                "t_hac": float("nan"), "r2": float("nan")}
    xv = s["x"].values.astype(float)
    yv = s["y"].values.astype(float)
    try:
        import statsmodels.api as sm
        X = sm.add_constant(xv)
        res = sm.OLS(yv, X).fit(cov_type="HAC", cov_kwds={"maxlags": months})
        return {"months": months, "n": n, "beta": float(res.params[1]),
                "t_hac": float(res.tvalues[1]), "r2": float(res.rsquared)}
    except Exception:
        # hand-rolled OLS + Newey-West (Bartlett kernel) fallback
        X = np.column_stack([np.ones(n), xv])
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ (X.T @ yv)
        resid = yv - X @ beta
        L = months
        S = (X * resid[:, None]).T @ (X * resid[:, None])
        for lg in range(1, L + 1):
            w = 1.0 - lg / (L + 1.0)
            g = np.zeros((2, 2))
            for t in range(lg, n):
                u_t = (X[t] * resid[t])[:, None]
                u_tl = (X[t - lg] * resid[t - lg])[:, None]
                g += u_t @ u_tl.T
            S += w * (g + g.T)
        cov = XtX_inv @ S @ XtX_inv
        se1 = float(np.sqrt(cov[1, 1]))
        ss_tot = float(((yv - yv.mean()) ** 2).sum())
        r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
        return {"months": months, "n": n, "beta": float(beta[1]),
                "t_hac": float(beta[1] / se1) if se1 > 0 else float("nan"), "r2": r2}


def placebo_pvalue(frame: pd.DataFrame, months: int, k: int = 1, thresh: float = 0.0,
                   lag: int = 1, n_draws: int = 20_000, seed: int = 752) -> dict:
    """Small-sample placebo null for the liquidity-lever claim.

    Draw ``n_draw`` random months ``n_draws`` times and ask how often a random draw's
    mean forward return is **at least** the DRAWDOWN set's mean (i.e. as *bullish* or
    more). p = P[random-draw mean >= drawdown mean]. A real lever => small p.
    """
    d, _b, a = split_returns(frame, months, k=k, thresh=thresh, lag=lag)
    kk = len(d)
    if kk == 0 or len(a) == 0:
        return {"k": 0, "drawdown_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    n = len(a)
    for i in range(n_draws):
        means[i] = a[rng.integers(0, n, size=kk)].mean()
    obs = float(d.mean())
    p = float((means >= obs).mean())
    return {"k": kk, "drawdown_mean": obs, "placebo_mean": float(means.mean()),
            "p_value": p}


def summarize(frame: pd.DataFrame, months: int, k: int = 1, thresh: float = 0.0,
              lag: int = 1) -> dict:
    """Headline stats for one horizon: n, DRAWDOWN vs BUILD vs base forward means and
    up-rates, the Welch t (DRAWDOWN vs base), the HAC regression t, the placebo p."""
    d, b, a = split_returns(frame, months, k=k, thresh=thresh, lag=lag)
    pl = placebo_pvalue(frame, months, k=k, thresh=thresh, lag=lag)
    reg = hac_regression(frame, months, k=k, lag=lag)
    return {
        "months": months,
        "n_draw": int(len(d)),
        "n_build": int(len(b)),
        "draw_mean": float(d.mean()) if len(d) else float("nan"),
        "build_mean": float(b.mean()) if len(b) else float("nan"),
        "base_mean": float(a.mean()) if len(a) else float("nan"),
        "draw_uprate": float((d > 0).mean()) if len(d) else float("nan"),
        "base_uprate": float((a > 0).mean()) if len(a) else float("nan"),
        "t": welch_t(d, a),
        "beta": reg["beta"],
        "t_hac": reg["t_hac"],
        "r2": reg["r2"],
        "p_placebo": pl["p_value"],
    }


def lead_lag(frame: pd.DataFrame, k: int = 1, leads=range(-6, 7)) -> pd.Series:
    """Correlation of the injection at t with the SPY return over [t+L, t+L+1].

    L > 0 => the TGA injection *leads* the market (a genuine liquidity lever would peak
    at L > 0 with a *positive* correlation); L < 0 => the injection *lags* the market.
    Returns a Series indexed by lead L (months)."""
    inj = injection(frame, k=k)
    spy = frame["spy"]
    out = {}
    for L in leads:
        fwd = spy.shift(-L - 1) / spy.shift(-L) - 1.0
        s = pd.concat([inj, fwd], axis=1).dropna()
        out[L] = float(np.corrcoef(s.iloc[:, 0], s.iloc[:, 1])[0, 1]) if len(s) > 3 else float("nan")
    return pd.Series(out, name="corr")


# --------------------------------------------------------------------------- #
# Tradability — the "hold when TGA drawing down, cash when building" overlay
# --------------------------------------------------------------------------- #
def timing_overlay(frame: pd.DataFrame, k: int = 1, thresh: float = 0.0,
                   lag: int = 1, cost_bps: float = 10.0) -> dict:
    """Hold SPY when the TGA is DRAWING DOWN, sit in cash when it is BUILDING.

    One-month execution lag; one-way cost ``cost_bps`` charged on each switch (turnover
    counted one-way × NAV). Compares the overlay's monthly return stream to buy-and-hold
    on the same months, reporting annualised mean, vol, Sharpe (excess-of-zero, since the
    cash leg earns 0 here — a conservative, clearly-labelled simplification), and the
    number of switches. Gross and net both reported.
    """
    spy = frame["spy"]
    draw = drawdown_mask(frame, k=k, thresh=thresh)
    # signal known at t acts from t+lag: position for month t+1 = (drawing down at t)
    pos = draw.astype(float).shift(lag)
    df = pd.DataFrame({"r": spy.pct_change(), "pos": pos}).dropna()
    bh = df["r"]
    switches = df["pos"].diff().abs().fillna(0.0)
    c = cost_bps / 1e4
    gross = df["pos"] * bh
    net = gross - switches * c
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
