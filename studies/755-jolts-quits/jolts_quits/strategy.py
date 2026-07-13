"""Strategy + inference for Study 755 — JOLTS-Quits.

The believers' rule (labour nowcasting): **the JOLTS quits rate is worker confidence —
people quit when they're sure they can find something better — so when the quits rate
turns DOWN, confidence is fading and equities (especially cyclicals) are about to
soften.** Operationalised on the monthly quits-rate tape:

    Let ``q`` = the JOLTS quits rate (percent) and ``m_t = q_t - q_{t-k}`` its k-month
    change (default k = 3; a *rate* difference, in percentage points). Quits are
    "FALLING" at month t when ``m_t < -thresh`` (default thresh = 0); the believers say
    the forward equity return is then DEPRESSED relative to the unconditional mean.

We test it by splitting forward H-month returns into a FALLING-quits set and a
RISING-quits set and comparing each to the unconditional mean, with:

  * a **Welch two-sample t** of the FALLING-set forward mean against the unconditional
    forward mean (a real confidence gauge => FALLING mean is *significantly negative-excess*);
  * a **placebo / randomization null** — draw the same number of random months and ask
    how often a random draw's mean forward return is at least as *low* as the FALLING set
    (the honest small-sample test);
  * a **win-rate** (share of forward returns < 0) vs the unconditional base rate;
  * a **release lag** — JOLTS for reference month t is published in the first week of
    month **t+2** (a ~5–6-week reporting delay), so the earliest a trader can act on the
    month-t print is the close of month **t+2**. We use a **2-month execution lag** by
    default (no look-ahead), which is the honest, distinguishing detail of this series;
  * **one-way costs** × turnover for the "go to cash when quits are falling" overlay.

The decisive question is not whether equities are weak *during* recessions (they are —
that's mechanical) but whether a quits *downturn* **leads** the market enough, and
cleanly enough, to be a tradable gauge rather than a coincident or lagging tell — and
whether the ~6-week JOLTS publication delay leaves any timing edge at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 6, 12)            # forward horizons in months
RELEASE_LAG = 2                     # JOLTS reference month t is public in month t+2


# --------------------------------------------------------------------------- #
# Signal construction
# --------------------------------------------------------------------------- #
def quits_momentum(frame: pd.DataFrame, k: int = 3, smooth: int = 1) -> pd.Series:
    """k-month change of the (optionally smoothed) quits rate: q_t - q_{t-k} (pp)."""
    q = frame["quits"]
    if smooth > 1:
        q = q.rolling(smooth, min_periods=1).mean()
    return q - q.shift(k)


def falling_mask(frame: pd.DataFrame, k: int = 3, thresh: float = 0.0,
                 smooth: int = 1) -> pd.Series:
    """Boolean: quits momentum strictly below ``-thresh`` (quits FALLING)."""
    return quits_momentum(frame, k=k, smooth=smooth) < -thresh


# --------------------------------------------------------------------------- #
# Forward returns (release-aware execution lag)
# --------------------------------------------------------------------------- #
def forward_returns(frame: pd.DataFrame, months: int, lag: int = RELEASE_LAG,
                    price: str = "spy") -> pd.Series:
    """Forward ``months``-month return on ``price`` entered ``lag`` months after the signal.

    Signal (reference month t) becomes public and is acted on at month-end t+lag (no
    look-ahead); the return runs t+lag -> t+lag+months. NaN where the horizon overruns
    the tape or the price column is absent.
    """
    if price not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    p = frame[price]
    entry = p.shift(-lag)
    exit_ = p.shift(-lag - months)
    return (exit_ / entry - 1.0)


def split_returns(frame: pd.DataFrame, months: int, k: int = 3, thresh: float = 0.0,
                  smooth: int = 1, lag: int = RELEASE_LAG, price: str = "spy"):
    """(falling_fwd, rising_fwd, all_fwd) arrays of forward returns, NaNs dropped."""
    fwd = forward_returns(frame, months, lag=lag, price=price)
    falling = falling_mask(frame, k=k, thresh=thresh, smooth=smooth)
    ok = fwd.notna() & falling.notna()
    fwd, falling = fwd[ok], falling[ok].astype(bool)
    d = fwd[falling].values.astype(float)
    u = fwd[~falling].values.astype(float)
    a = fwd.values.astype(float)
    return d, u, a


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
                   smooth: int = 1, lag: int = RELEASE_LAG, price: str = "spy",
                   n_draws: int = 20_000, seed: int = 755) -> dict:
    """Small-sample placebo null for the worker-confidence claim.

    Draw ``n_falling`` random months ``n_draws`` times and ask how often a random draw's
    mean forward return is **at most** the FALLING set's mean (i.e. as *bearish* or more).
    p = P[random-draw mean <= falling mean]. A real gauge => small p.
    """
    d, _u, a = split_returns(frame, months, k=k, thresh=thresh, smooth=smooth,
                             lag=lag, price=price)
    kk = len(d)
    if kk == 0 or len(a) == 0:
        return {"k": 0, "falling_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    n = len(a)
    for i in range(n_draws):
        means[i] = a[rng.integers(0, n, size=kk)].mean()
    obs = float(d.mean())
    p = float((means <= obs).mean())
    return {"k": kk, "falling_mean": obs, "placebo_mean": float(means.mean()),
            "p_value": p}


def summarize(frame: pd.DataFrame, months: int, k: int = 3, thresh: float = 0.0,
              smooth: int = 1, lag: int = RELEASE_LAG, price: str = "spy") -> dict:
    """Headline stats for one horizon: n, FALLING vs RISING vs base forward means and
    down-rates, the Welch t (FALLING vs base), and the placebo p (confidence gauge)."""
    d, u, a = split_returns(frame, months, k=k, thresh=thresh, smooth=smooth,
                            lag=lag, price=price)
    pl = placebo_pvalue(frame, months, k=k, thresh=thresh, smooth=smooth, lag=lag, price=price)
    return {
        "months": months,
        "n_falling": int(len(d)),
        "n_rising": int(len(u)),
        "falling_mean": float(d.mean()) if len(d) else float("nan"),
        "rising_mean": float(u.mean()) if len(u) else float("nan"),
        "base_mean": float(a.mean()) if len(a) else float("nan"),
        "falling_downrate": float((d < 0).mean()) if len(d) else float("nan"),
        "base_downrate": float((a < 0).mean()) if len(a) else float("nan"),
        "t": welch_t(d, a),
        "p_placebo": pl["p_value"],
    }


def lead_lag(frame: pd.DataFrame, k: int = 3, smooth: int = 1, price: str = "spy",
             leads=range(-6, 7)) -> pd.Series:
    """Correlation of quits momentum at t with the ``price`` return over [t+L, t+L+1].

    L < 0 => quits momentum *lags* the market (coincident/lagging tell); L > 0 => quits
    momentum *leads* it (a genuine leading gauge would peak at L > 0 with a *positive*
    correlation, since falling quits should precede *lower* returns => rising quits
    precede higher). Returns a Series indexed by lead L (months)."""
    mom = quits_momentum(frame, k=k, smooth=smooth)
    p = frame[price]
    out = {}
    for L in leads:
        fwd = p.shift(-L - 1) / p.shift(-L) - 1.0
        s = pd.concat([mom, fwd], axis=1).dropna()
        out[L] = float(np.corrcoef(s.iloc[:, 0], s.iloc[:, 1])[0, 1]) if len(s) > 3 else float("nan")
    return pd.Series(out, name="corr")


# --------------------------------------------------------------------------- #
# Tradability — the "go to cash when quits are falling" timing overlay
# --------------------------------------------------------------------------- #
def timing_overlay(frame: pd.DataFrame, k: int = 3, thresh: float = 0.0, smooth: int = 1,
                   lag: int = RELEASE_LAG, cost_bps: float = 10.0, price: str = "spy") -> dict:
    """Hold ``price`` when quits are RISING, sit in cash when FALLING (the believers' overlay).

    Release-aware execution lag (default 2 months, the JOLTS publication delay); one-way
    cost ``cost_bps`` charged on each switch (turnover counted one-way × NAV). Compares
    the overlay's monthly return stream to buy-and-hold on the same months, reporting
    annualised mean, vol, Sharpe (excess-of-zero, since the cash leg earns 0 here — a
    conservative, clearly-labelled simplification), and the number of switches. Gross and
    net both reported.
    """
    p = frame[price]
    falling = falling_mask(frame, k=k, thresh=thresh, smooth=smooth)
    # signal for reference month t is public at t+lag; position for the next month = (not falling)
    pos = (~falling).astype(float).shift(lag)
    df = pd.DataFrame({"r": p.pct_change(), "pos": pos}).dropna()
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
