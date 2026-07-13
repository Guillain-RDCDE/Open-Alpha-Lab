"""Strategy + inference for Study 759 — Redbook-Retail.

The believers' rule (consumer nowcasting): **accelerating same-store retail sales lead
retail stocks up.** The weekly Johnson Redbook Index is pitched as a real-time read on the
consumer, so when its year-over-year same-store growth **turns up**, the folklore says the
retail sector (XRT) is about to follow — a nowcast you can trade. Operationalised on the
monthly Redbook YoY tape:

    Let ``y`` = the Redbook same-store-sales YoY level (percent) and
    ``m_t = y_t - y_{t-k}`` its k-month momentum / acceleration (default k = 3).
    Redbook is "ACCELERATING" at month t when ``m_t > thresh`` (default 0); the believers
    say the forward retail-stock return is then ELEVATED relative to the unconditional mean.

We test it by splitting forward H-month XRT returns into an ACCELERATING set and a
DECELERATING set and comparing each to the unconditional mean, with:

  * a **Welch two-sample t** of the ACCEL-set forward mean against the unconditional forward
    mean (a real nowcast => ACCEL mean is *significantly positive-excess*);
  * a **placebo / randomization null** — draw the same number of random months and ask how
    often a random draw's mean forward return is at least as *high* as the ACCEL set (the
    honest small-sample nowcast test);
  * a **win-rate** (share of forward returns > 0) vs the unconditional base rate;
  * a **lead/lag cross-correlation** — *where* does Redbook momentum line up with XRT? A real
    leading nowcast peaks (positively) at a **positive** lead (Redbook first, retail later);
  * a **regime split on the YoY level** (strong vs weak same-store growth) as a robustness;
  * a **retail-vs-market relative test** — does Redbook predict XRT *outperformance* over SPY,
    the sharper "leads retail *stocks*" question?
  * **one execution lag** — the month-t Redbook print (weekly, released within the month) is
    acted on at month t+1's close, so forward returns start from t+1 (no look-ahead);
  * **one-way costs** × turnover for the "own XRT when Redbook is accelerating" timing overlay.

The decisive question is not whether retail sales and retail stocks co-move (they must, at
low frequency) but whether a Redbook uptick **leads** XRT cleanly enough to be a tradable
nowcast rather than a coincident (and inflation-contaminated) echo of a sector the market
already reprices in real time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 6, 12)            # forward horizons in months


# --------------------------------------------------------------------------- #
# Signal construction
# --------------------------------------------------------------------------- #
def redbook_momentum(frame: pd.DataFrame, k: int = 3, smooth: int = 1) -> pd.Series:
    """k-month momentum (acceleration) of the (optionally smoothed) Redbook YoY level:
    y_t - y_{t-k}. Redbook is already a YoY growth rate, so this is its *acceleration*."""
    y = frame["redbook"]
    if smooth > 1:
        y = y.rolling(smooth, min_periods=1).mean()
    return y - y.shift(k)


def accel_mask(frame: pd.DataFrame, k: int = 3, thresh: float = 0.0,
               smooth: int = 1) -> pd.Series:
    """Boolean: Redbook momentum strictly above ``thresh`` (same-store sales ACCELERATING)."""
    return redbook_momentum(frame, k=k, smooth=smooth) > thresh


def strong_mask(frame: pd.DataFrame, thresh: float | None = None) -> pd.Series:
    """Boolean level-regime: Redbook YoY above ``thresh`` (default = full-sample median).

    The 'strong consumer' regime the folklore says you should own retail in — distinct from
    the *acceleration* signal above. Used for the regime robustness."""
    y = frame["redbook"]
    if thresh is None:
        thresh = float(y.median())
    return y > thresh


# --------------------------------------------------------------------------- #
# Forward returns (one-month execution lag)
# --------------------------------------------------------------------------- #
def forward_returns(frame: pd.DataFrame, months: int, lag: int = 1,
                    relative: bool = False) -> pd.Series:
    """Forward ``months``-month XRT return entered ``lag`` months after the signal.

    Signal known at month-end t is acted on at month-end t+lag (no look-ahead); the return
    runs t+lag -> t+lag+months. NaN where the horizon overruns the tape. When
    ``relative=True`` the return is XRT-minus-SPY (retail *outperformance* over the broad
    market) — the sharper 'leads retail stocks' object.
    """
    xrt = frame["xrt"]
    entry = xrt.shift(-lag)
    exit_ = xrt.shift(-lag - months)
    r = exit_ / entry - 1.0
    if relative:
        spy = frame["spy"]
        rs = spy.shift(-lag - months) / spy.shift(-lag) - 1.0
        r = r - rs
    return r


def split_returns(frame: pd.DataFrame, months: int, k: int = 3, thresh: float = 0.0,
                  smooth: int = 1, lag: int = 1, relative: bool = False):
    """(accel_fwd, decel_fwd, all_fwd) arrays of forward returns, NaNs dropped."""
    fwd = forward_returns(frame, months, lag=lag, relative=relative)
    accel = accel_mask(frame, k=k, thresh=thresh, smooth=smooth)
    ok = fwd.notna() & accel.notna()
    fwd, accel = fwd[ok], accel[ok].astype(bool)
    a = fwd[accel].values.astype(float)
    d = fwd[~accel].values.astype(float)
    al = fwd.values.astype(float)
    return a, d, al


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
                   smooth: int = 1, lag: int = 1, relative: bool = False,
                   n_draws: int = 20_000, seed: int = 759) -> dict:
    """Small-sample placebo null for the nowcast claim.

    Draw ``n_accel`` random months ``n_draws`` times and ask how often a random draw's mean
    forward return is **at least** the ACCEL set's mean (i.e. as *bullish* or more).
    p = P[random-draw mean >= accel mean]. A real nowcast => small p.
    """
    a, _d, al = split_returns(frame, months, k=k, thresh=thresh, smooth=smooth, lag=lag,
                              relative=relative)
    kk = len(a)
    if kk == 0 or len(al) == 0:
        return {"k": 0, "accel_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    n = len(al)
    for i in range(n_draws):
        means[i] = al[rng.integers(0, n, size=kk)].mean()
    obs = float(a.mean())
    p = float((means >= obs).mean())
    return {"k": kk, "accel_mean": obs, "placebo_mean": float(means.mean()), "p_value": p}


def summarize(frame: pd.DataFrame, months: int, k: int = 3, thresh: float = 0.0,
              smooth: int = 1, lag: int = 1, relative: bool = False) -> dict:
    """Headline stats for one horizon: n, ACCEL vs DECEL vs base forward means and up-rates,
    the Welch t (ACCEL vs base), and the placebo p (nowcast)."""
    a, d, al = split_returns(frame, months, k=k, thresh=thresh, smooth=smooth, lag=lag,
                             relative=relative)
    pl = placebo_pvalue(frame, months, k=k, thresh=thresh, smooth=smooth, lag=lag,
                        relative=relative)
    return {
        "months": months,
        "n_accel": int(len(a)),
        "n_decel": int(len(d)),
        "accel_mean": float(a.mean()) if len(a) else float("nan"),
        "decel_mean": float(d.mean()) if len(d) else float("nan"),
        "base_mean": float(al.mean()) if len(al) else float("nan"),
        "accel_uprate": float((a > 0).mean()) if len(a) else float("nan"),
        "base_uprate": float((al > 0).mean()) if len(al) else float("nan"),
        "t": welch_t(a, al),
        "p_placebo": pl["p_value"],
    }


def regime_summary(frame: pd.DataFrame, months: int, thresh: float | None = None,
                   lag: int = 1, relative: bool = False) -> dict:
    """Level-regime split: forward XRT returns in STRONG (YoY high) vs WEAK same-store months.

    A robustness on the *level* rather than the *acceleration*: does simply owning retail when
    same-store growth is strong beat owning it when growth is weak? Welch t of strong vs weak.
    """
    fwd = forward_returns(frame, months, lag=lag, relative=relative)
    strong = strong_mask(frame, thresh=thresh)
    ok = fwd.notna() & strong.notna()
    fwd, strong = fwd[ok], strong[ok].astype(bool)
    s = fwd[strong].values.astype(float)
    w = fwd[~strong].values.astype(float)
    return {
        "months": months,
        "n_strong": int(len(s)), "n_weak": int(len(w)),
        "strong_mean": float(s.mean()) if len(s) else float("nan"),
        "weak_mean": float(w.mean()) if len(w) else float("nan"),
        "spread": float(s.mean() - w.mean()) if len(s) and len(w) else float("nan"),
        "t": welch_t(s, w),
    }


def lead_lag(frame: pd.DataFrame, k: int = 3, smooth: int = 1, relative: bool = False,
             leads=range(-6, 7)) -> pd.Series:
    """Correlation of Redbook momentum at t with the XRT return over [t+L, t+L+1].

    L < 0 => Redbook momentum *lags* the retail move (coincident/lagging echo); L > 0 =>
    Redbook momentum *leads* it (a genuine nowcast would peak at L > 0 with a *positive*
    correlation). ``relative=True`` uses XRT-minus-SPY. Returns a Series indexed by lead L
    (months)."""
    mom = redbook_momentum(frame, k=k, smooth=smooth)
    xrt = frame["xrt"]
    spy = frame["spy"]
    out = {}
    for L in leads:
        fwd = xrt.shift(-L - 1) / xrt.shift(-L) - 1.0
        if relative:
            fwd = fwd - (spy.shift(-L - 1) / spy.shift(-L) - 1.0)
        s = pd.concat([mom, fwd], axis=1).dropna()
        out[L] = float(np.corrcoef(s.iloc[:, 0], s.iloc[:, 1])[0, 1]) if len(s) > 3 else float("nan")
    return pd.Series(out, name="corr")


# --------------------------------------------------------------------------- #
# Tradability — the "own XRT when Redbook is accelerating" timing overlay
# --------------------------------------------------------------------------- #
def timing_overlay(frame: pd.DataFrame, k: int = 3, thresh: float = 0.0,
                   smooth: int = 1, lag: int = 1, cost_bps: float = 10.0) -> dict:
    """Hold XRT when Redbook is ACCELERATING, sit in cash when DECELERATING (believers' rule).

    One-month execution lag; one-way cost ``cost_bps`` charged on each switch (turnover
    counted one-way × NAV). Compares the overlay's monthly return stream to buy-and-hold XRT
    on the same months, reporting annualised mean, vol, Sharpe (excess-of-zero, since the cash
    leg earns 0 here — a conservative, clearly-labelled simplification), and the number of
    switches. Gross and net both reported.
    """
    xrt = frame["xrt"]
    accel = accel_mask(frame, k=k, thresh=thresh, smooth=smooth)
    # signal known at t acts from t+lag: position for month t+1 = (accel at t)
    pos = accel.astype(float).shift(lag)
    df = pd.DataFrame({"r": xrt.pct_change(), "pos": pos}).dropna()
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
