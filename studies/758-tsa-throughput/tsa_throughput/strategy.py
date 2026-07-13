"""Strategy + inference for Study 758 — TSA-Throughput.

The believers' rule (alt-data nowcasting): **accelerating TSA checkpoint throughput calls a
travel-stock tailwind you can trade early.** Operationalised on the monthly TSA tape:

    Let ``v`` = average daily TSA throughput (millions) and
    ``m_t = v_t / v_{t-k} - 1`` its k-month momentum (default k = 12, i.e. year-over-year,
    the natural choice for a strongly seasonal travel series). TSA is "ACCELERATING" at
    month t when ``m_t > thresh`` (default 0); the believers say the forward travel-basket
    return is then LIFTED relative to the unconditional mean.

We test it by splitting forward H-month **travel-basket** returns into an ACCELERATING set
and a DECELERATING set and comparing each to the unconditional mean, with:

  * a **Welch two-sample t** of the ACCELERATING-set forward mean against the unconditional
    forward mean (a real nowcast => ACCELERATING mean is *significantly positive-excess*);
  * a **placebo / randomization null** — draw the same number of random months and ask how
    often a random draw's mean forward return is at least as *high* as the ACCELERATING set;
  * a **win-rate** (share of forward returns > 0) vs the unconditional base rate;
  * **one execution lag** — the month-t TSA average (published next-day, fully known early in
    month t+1) is acted on at month t+1's close, so forward returns start from t+1 (no
    look-ahead);
  * a **market-beta control** — regress forward basket returns on forward SPY returns and ask
    whether the ACCELERATING dummy still carries a coefficient once market beta is removed
    (i.e. is the "nowcast" anything beyond the reopening beta the whole tape shares?);
  * a **lead/lag scan** — does the TSA uptick actually come *first*, or do travel equities
    price the recovery before throughput recovers?
  * **one-way costs** x turnover for the "long the travel basket when TSA is accelerating"
    timing overlay (shorts, when enabled, pay borrow).

Macro convention on the lag: TSA daily numbers for month t are fully published within days,
so the monthly average is known at the start of month t+1; to be strictly non-look-ahead we
enter at the **next** month-end close (a one-month execution lag) — conservative, documented.

The decisive question is not whether travel stocks are strong *while* travel recovers (they
are — that's mechanical) but whether a TSA uptick **leads** the travel trade, cleanly and
outside the one COVID regime, enough to be a tradable nowcast rather than a coincident echo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 6, 12)            # forward horizons in months


# --------------------------------------------------------------------------- #
# Signal construction
# --------------------------------------------------------------------------- #
def tsa_momentum(frame: pd.DataFrame, k: int = 12, smooth: int = 1) -> pd.Series:
    """k-month momentum of the (optionally smoothed) TSA level: v_t / v_{t-k} - 1.

    Default k = 12 (year-over-year) neutralises the strong annual travel season; a real
    'acceleration' is throughput running above where it was a full year ago.
    """
    v = frame["tsa"]
    if smooth > 1:
        v = v.rolling(smooth, min_periods=1).mean()
    return v / v.shift(k) - 1.0


def accel_mask(frame: pd.DataFrame, k: int = 12, thresh: float = 0.0,
               smooth: int = 1) -> pd.Series:
    """Boolean: TSA momentum strictly above ``thresh`` (throughput ACCELERATING)."""
    return tsa_momentum(frame, k=k, smooth=smooth) > thresh


# --------------------------------------------------------------------------- #
# Forward returns (one-month execution lag)
# --------------------------------------------------------------------------- #
def forward_returns(frame: pd.DataFrame, months: int, col: str = "basket",
                    lag: int = 1) -> pd.Series:
    """Forward ``months``-month return of ``col`` entered ``lag`` months after the signal.

    Signal known at month-end t is acted on at month-end t+lag (no look-ahead); the return
    runs t+lag -> t+lag+months. NaN where the horizon overruns the tape.
    """
    p = frame[col]
    entry = p.shift(-lag)
    exit_ = p.shift(-lag - months)
    return (exit_ / entry - 1.0)


def split_returns(frame: pd.DataFrame, months: int, k: int = 12, thresh: float = 0.0,
                  smooth: int = 1, lag: int = 1, col: str = "basket"):
    """(accel_fwd, decel_fwd, all_fwd) arrays of forward returns, NaNs dropped."""
    fwd = forward_returns(frame, months, col=col, lag=lag)
    accel = accel_mask(frame, k=k, thresh=thresh, smooth=smooth)
    ok = fwd.notna() & accel.notna()
    fwd, accel = fwd[ok], accel[ok].astype(bool)
    a = fwd[accel].values.astype(float)
    d = fwd[~accel].values.astype(float)
    allv = fwd.values.astype(float)
    return a, d, allv


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


def placebo_pvalue(frame: pd.DataFrame, months: int, k: int = 12, thresh: float = 0.0,
                   smooth: int = 1, lag: int = 1, col: str = "basket",
                   n_draws: int = 20_000, seed: int = 758) -> dict:
    """Small-sample placebo null for the nowcast claim.

    Draw ``n_accel`` random months ``n_draws`` times and ask how often a random draw's mean
    forward return is **at least** the ACCELERATING set's mean (i.e. as *bullish* or more).
    p = P[random-draw mean >= accel mean]. A real nowcast => small p.
    """
    a, _d, allv = split_returns(frame, months, k=k, thresh=thresh, smooth=smooth,
                                lag=lag, col=col)
    kk = len(a)
    if kk == 0 or len(allv) == 0:
        return {"k": 0, "accel_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    n = len(allv)
    for i in range(n_draws):
        means[i] = allv[rng.integers(0, n, size=kk)].mean()
    obs = float(a.mean())
    p = float((means >= obs).mean())
    return {"k": kk, "accel_mean": obs, "placebo_mean": float(means.mean()),
            "p_value": p}


def summarize(frame: pd.DataFrame, months: int, k: int = 12, thresh: float = 0.0,
              smooth: int = 1, lag: int = 1, col: str = "basket") -> dict:
    """Headline stats for one horizon: n, ACCEL vs DECEL vs base forward means and up-rates,
    the Welch t (ACCEL vs base), and the placebo p (nowcast)."""
    a, d, allv = split_returns(frame, months, k=k, thresh=thresh, smooth=smooth,
                               lag=lag, col=col)
    pl = placebo_pvalue(frame, months, k=k, thresh=thresh, smooth=smooth, lag=lag, col=col)
    return {
        "months": months,
        "n_accel": int(len(a)),
        "n_decel": int(len(d)),
        "accel_mean": float(a.mean()) if len(a) else float("nan"),
        "decel_mean": float(d.mean()) if len(d) else float("nan"),
        "base_mean": float(allv.mean()) if len(allv) else float("nan"),
        "accel_uprate": float((a > 0).mean()) if len(a) else float("nan"),
        "base_uprate": float((allv > 0).mean()) if len(allv) else float("nan"),
        "t": welch_t(a, allv),
        "p_placebo": pl["p_value"],
    }


def beta_control(frame: pd.DataFrame, months: int, k: int = 12, thresh: float = 0.0,
                 smooth: int = 1, lag: int = 1) -> dict:
    """Is the ACCELERATING tilt anything beyond market beta?

    Regress forward basket returns on (1) a constant, (2) the contemporaneous forward SPY
    return (the market/reopening beta the whole travel tape shares), and (3) the ACCELERATING
    dummy. Report the dummy's coefficient and its OLS t. If market beta explains the tilt, the
    dummy coefficient collapses toward zero once SPY is in the regression.

    Returns the raw ACCEL-vs-base gap, the beta-adjusted dummy coefficient, its t, and the
    estimated basket beta to SPY.
    """
    fwd_b = forward_returns(frame, months, col="basket", lag=lag)
    fwd_m = forward_returns(frame, months, col="spy", lag=lag)
    accel = accel_mask(frame, k=k, thresh=thresh, smooth=smooth).astype(float)
    df = pd.DataFrame({"b": fwd_b, "m": fwd_m, "a": accel}).dropna()
    if len(df) < 5:
        return {"raw_gap": float("nan"), "adj_coef": float("nan"), "adj_t": float("nan"),
                "beta": float("nan"), "n": int(len(df))}
    y = df["b"].values
    X = np.column_stack([np.ones(len(df)), df["m"].values, df["a"].values])
    # OLS with (X'X)^-1 X'y and classical SEs (a beta control, not the headline inference)
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ (X.T @ y)
    resid = y - X @ beta_hat
    dof = max(len(df) - X.shape[1], 1)
    sigma2 = (resid @ resid) / dof
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    raw_gap = float(df.loc[df["a"] > 0, "b"].mean() - df["b"].mean())
    return {
        "raw_gap": raw_gap,
        "adj_coef": float(beta_hat[2]),
        "adj_t": float(beta_hat[2] / se[2]) if se[2] > 0 else float("nan"),
        "beta": float(beta_hat[1]),
        "n": int(len(df)),
    }


def lead_lag(frame: pd.DataFrame, k: int = 12, smooth: int = 1,
             leads=range(-6, 7)) -> pd.Series:
    """Correlation of TSA momentum at t with the basket return over [t+L, t+L+1].

    L < 0 => TSA momentum *lags* the travel trade (coincident/lagging echo); L > 0 => TSA
    momentum *leads* it (a genuine nowcast would peak at L > 0 with a positive correlation).
    Returns a Series indexed by lead L (months)."""
    mom = tsa_momentum(frame, k=k, smooth=smooth)
    p = frame["basket"]
    out = {}
    for L in leads:
        fwd = p.shift(-L - 1) / p.shift(-L) - 1.0
        s = pd.concat([mom, fwd], axis=1).dropna()
        out[L] = float(np.corrcoef(s.iloc[:, 0], s.iloc[:, 1])[0, 1]) if len(s) > 3 else float("nan")
    return pd.Series(out, name="corr")


# --------------------------------------------------------------------------- #
# Tradability — the "long the travel basket when TSA is accelerating" overlay
# --------------------------------------------------------------------------- #
def timing_overlay(frame: pd.DataFrame, k: int = 12, thresh: float = 0.0, smooth: int = 1,
                   lag: int = 1, cost_bps: float = 10.0, borrow_bps: float = 0.0,
                   allow_short: bool = False) -> dict:
    """Hold the travel basket when TSA is ACCELERATING, else cash (or short if enabled).

    One-month execution lag; one-way cost ``cost_bps`` charged on each position change
    (turnover counted one-way x NAV). When ``allow_short`` the DECELERATING leg is short the
    basket and pays ``borrow_bps``/yr borrow. Compares the overlay's monthly return stream to
    buy-and-hold the basket on the same months, reporting annualised mean, vol, Sharpe
    (excess-of-zero — the cash leg earns 0 here, a conservative, clearly-labelled
    simplification), and the number of switches. Gross and net both reported.
    """
    p = frame["basket"]
    accel = accel_mask(frame, k=k, thresh=thresh, smooth=smooth)
    short_leg = -1.0 if allow_short else 0.0
    # signal known at t acts from t+lag: position for month t+1 depends on accel at t
    pos = accel.map({True: 1.0, False: short_leg}).astype(float).shift(lag)
    df = pd.DataFrame({"r": p.pct_change(), "pos": pos}).dropna()
    bh = df["r"]
    switches = df["pos"].diff().abs().fillna(0.0)
    c = cost_bps / 1e4
    borrow_m = borrow_bps / 1e4 / 12.0
    gross = df["pos"] * bh
    short_days = (df["pos"] < 0).astype(float)
    net = gross - switches * c - short_days * borrow_m
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
        "exposure": float((df["pos"] != 0).mean()),
        "bh_mean": bh_mu, "bh_vol": bh_vol, "bh_sharpe": bh_sh,
        "overlay_gross_mean": g_mu, "overlay_gross_sharpe": g_sh,
        "overlay_net_mean": n_mu, "overlay_net_vol": n_vol, "overlay_net_sharpe": n_sh,
        "cost_bps": cost_bps, "allow_short": allow_short,
    }
