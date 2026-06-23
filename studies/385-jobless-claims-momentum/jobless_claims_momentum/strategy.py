"""Strategy + inference for Study 385 — Jobless-Claims-Momentum.

The believers' rule (macro nowcasting): **rising initial jobless claims call equity
downturns early.** Operationalised on the monthly 4-week-MA claims tape:

    Let ``c`` = the 4-week-MA initial-claims level (thousands) and
    ``m_t = c_t / c_{t-k} - 1`` its k-month momentum (default k = 3).
    Claims are "RISING" at month t when ``m_t > thresh`` (default 0); the believers
    say the forward equity return is then DEPRESSED relative to the unconditional mean.

We test it by splitting forward H-month SPY returns into a RISING-claims set and a
FALLING-claims set and comparing each to the unconditional mean, with:

  * a **Welch two-sample t** of the RISING-set forward mean against the unconditional
    forward mean (a real early-warning => RISING mean is *significantly negative-excess*);
  * a **placebo / randomization null** — draw the same number of random months and ask
    how often a random draw's mean forward return is at least as *low* as the RISING set
    (the honest small-sample early-warning test);
  * a **win-rate** (share of forward returns < 0) vs the unconditional base rate;
  * **one execution lag** — the month-t claims print (released early in month t+1) is
    acted on at month t+1's close, so forward returns start from t+1 (no look-ahead);
  * **one-way costs** × turnover for the simple "go to cash when claims are rising"
    timing overlay.

Macro convention on the lag: claims for month t are reported within month t, but to be
strictly non-look-ahead we enter at the **next** month-end close (a one-month execution
lag) — conservative, and documented.

The decisive question is not whether equities are weak *during* recessions (they are —
that's mechanical) but whether a claims uptick **leads** the market enough, and cleanly
enough, to be a tradable early-warning rather than a coincident or lagging tell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 6, 12)            # forward horizons in months


# --------------------------------------------------------------------------- #
# Signal construction
# --------------------------------------------------------------------------- #
def claims_momentum(frame: pd.DataFrame, k: int = 3, smooth: int = 1) -> pd.Series:
    """k-month momentum of the (optionally smoothed) claims level: c_t / c_{t-k} - 1."""
    c = frame["claims"]
    if smooth > 1:
        c = c.rolling(smooth, min_periods=1).mean()
    return c / c.shift(k) - 1.0


def rising_mask(frame: pd.DataFrame, k: int = 3, thresh: float = 0.0,
                smooth: int = 1) -> pd.Series:
    """Boolean: claims momentum strictly above ``thresh`` (claims RISING)."""
    return claims_momentum(frame, k=k, smooth=smooth) > thresh


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


def split_returns(frame: pd.DataFrame, months: int, k: int = 3, thresh: float = 0.0,
                  smooth: int = 1, lag: int = 1):
    """(rising_fwd, falling_fwd, all_fwd) arrays of forward returns, NaNs dropped."""
    fwd = forward_returns(frame, months, lag=lag)
    rising = rising_mask(frame, k=k, thresh=thresh, smooth=smooth)
    ok = fwd.notna() & rising.notna()
    fwd, rising = fwd[ok], rising[ok].astype(bool)
    r = fwd[rising].values.astype(float)
    f = fwd[~rising].values.astype(float)
    a = fwd.values.astype(float)
    return r, f, a


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


def placebo_pvalue(frame: pd.DataFrame, months: int, k: int = 3, thresh: float = 0.0,
                   smooth: int = 1, lag: int = 1, n_draws: int = 20_000,
                   seed: int = 385) -> dict:
    """Small-sample placebo null for the early-warning claim.

    Draw ``n_rising`` random months ``n_draws`` times and ask how often a random draw's
    mean forward return is **at most** the RISING set's mean (i.e. as *bearish* or more).
    p = P[random-draw mean <= rising mean]. A real early-warning => small p.
    """
    r, _f, a = split_returns(frame, months, k=k, thresh=thresh, smooth=smooth, lag=lag)
    kk = len(r)
    if kk == 0 or len(a) == 0:
        return {"k": 0, "rising_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    n = len(a)
    for i in range(n_draws):
        means[i] = a[rng.integers(0, n, size=kk)].mean()
    obs = float(r.mean())
    p = float((means <= obs).mean())
    return {"k": kk, "rising_mean": obs, "placebo_mean": float(means.mean()),
            "p_value": p}


def summarize(frame: pd.DataFrame, months: int, k: int = 3, thresh: float = 0.0,
              smooth: int = 1, lag: int = 1) -> dict:
    """Headline stats for one horizon: n, RISING vs FALLING vs base forward means and
    down-rates, the Welch t (RISING vs base), and the placebo p (early-warning)."""
    r, f, a = split_returns(frame, months, k=k, thresh=thresh, smooth=smooth, lag=lag)
    pl = placebo_pvalue(frame, months, k=k, thresh=thresh, smooth=smooth, lag=lag)
    return {
        "months": months,
        "n_rising": int(len(r)),
        "n_falling": int(len(f)),
        "rising_mean": float(r.mean()) if len(r) else float("nan"),
        "falling_mean": float(f.mean()) if len(f) else float("nan"),
        "base_mean": float(a.mean()) if len(a) else float("nan"),
        "rising_downrate": float((r < 0).mean()) if len(r) else float("nan"),
        "base_downrate": float((a < 0).mean()) if len(a) else float("nan"),
        "t": welch_t(r, a),
        "p_placebo": pl["p_value"],
    }


def lead_lag(frame: pd.DataFrame, k: int = 3, smooth: int = 1,
             leads=range(-6, 7)) -> pd.Series:
    """Correlation of claims momentum at t with the SPY return over [t+L, t+L+1].

    L < 0 => claims momentum *lags* the market (coincident/lagging tell); L > 0 =>
    claims momentum *leads* it (a genuine early-warning would peak at L > 0 with a
    negative correlation). Returns a Series indexed by lead L (months)."""
    mom = claims_momentum(frame, k=k, smooth=smooth)
    spy = frame["spy"]
    out = {}
    for L in leads:
        fwd = spy.shift(-L - 1) / spy.shift(-L) - 1.0
        s = pd.concat([mom, fwd], axis=1).dropna()
        out[L] = float(np.corrcoef(s.iloc[:, 0], s.iloc[:, 1])[0, 1]) if len(s) > 3 else float("nan")
    return pd.Series(out, name="corr")


# --------------------------------------------------------------------------- #
# Tradability — the "go to cash when claims are rising" timing overlay
# --------------------------------------------------------------------------- #
def timing_overlay(frame: pd.DataFrame, k: int = 3, thresh: float = 0.0,
                   smooth: int = 1, lag: int = 1, cost_bps: float = 10.0) -> dict:
    """Hold SPY when claims are FALLING, sit in cash when RISING (the believers' overlay).

    One-month execution lag; one-way cost ``cost_bps`` charged on each switch (turnover
    counted one-way × NAV). Compares the overlay's monthly return stream to buy-and-hold
    on the same months, reporting annualised mean, vol, Sharpe (excess-of-zero, since the
    cash leg earns 0 here — a conservative, clearly-labelled simplification), and the
    number of switches. Gross and net both reported.
    """
    spy = frame["spy"]
    mret = spy.pct_change().shift(-0)          # month t return
    rising = rising_mask(frame, k=k, thresh=thresh, smooth=smooth)
    # signal known at t acts from t+lag: position for month t+1 = (not rising at t)
    pos = (~rising).astype(float).shift(lag)
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
